//! [`StreamingExecutor`] — the concrete [`Executor`] and the plan-tree dispatch it
//! runs on. `build_cursor` turns a read `PlanNode` into a streaming
//! [`RowCursor`](crate::RowCursor); `build_dml` is the write entry point DML
//! operators fill in.
//!
//! Every arm is present: an implemented operator builds its cursor, and an
//! unimplemented one returns a loud `Error` (never a silent empty result or a
//! panic), so a plan node the executor cannot yet run fails visibly rather than
//! quietly producing the wrong answer. A DML operator adds its ops file and
//! fills one arm of `build_dml`; it does not touch this dispatch shape.

use minisqlite_catalog::Catalog;
use minisqlite_plan::{Plan, PlanNode};
use minisqlite_types::{Error, Result, Value};

use crate::env::{Env, PagerSet};
use crate::executor::{Executor, RowCursor};
use crate::ops::{
    aggregate, cte, delete, distinct, filter, indexscan, insert, instead_of, join, limit,
    minmax_seek, pragma_function, project, scan, setop, sort, table_function, update, values,
    window,
};
use crate::statement::StatementRoot;

/// The top-level `outer` row: empty, with `'static` lifetime so it coerces to any
/// cursor lifetime. A `&[]` temporary would not live long enough for the returned
/// cursor; a `'static` const does.
const EMPTY_OUTER: &[Value] = &[];

/// The concrete executor. Unit struct — all per-run state (RNG, counters, params)
/// lives in the [`Runtime`] threaded through `next_row`, and all read context lives
/// in the borrowed [`Env`]; the executor itself owns nothing.
pub struct StreamingExecutor;

impl Executor for StreamingExecutor {
    fn execute<'a>(
        &'a mut self,
        plan: &'a Plan,
        catalog: &'a dyn Catalog,
        pagers: PagerSet<'a>,
    ) -> Result<Box<dyn RowCursor + 'a>> {
        // Build the statement's inner root cursor, then wrap it in `StatementRoot`.
        // That wrapper clears the connection `Runtime`'s per-statement uncorrelated-
        // subquery cache on the first pull (see [`crate::statement`]); since `execute`
        // is the single place a root cursor is produced, wrapping here is the one choke
        // point that keeps a cached subquery result from bleeding into the next
        // statement that reuses the same `Runtime`.
        let inner: Box<dyn RowCursor + 'a> = match &plan.root {
            // DML holds the whole pager SET: phase 1 reads the source through a shared
            // view of the set (the source may live in another namespace), phase 2
            // reborrows its target store mutably for the write. `InsteadOf` (view DML
            // redirected through triggers) fires actions that mutate base tables.
            PlanNode::Insert(_)
            | PlanNode::Update(_)
            | PlanNode::Delete(_)
            | PlanNode::InsteadOf(_) => build_dml(&plan.root, catalog, pagers, plan, EMPTY_OUTER)?,
            // Reads consume the set into a shared view for the whole cursor lifetime (many
            // cursors can read at once, across namespaces) and run through the shared `Env`.
            _ => {
                let env = Env { catalog, pagers: pagers.into_source(), plan };
                build_cursor(&plan.root, env, EMPTY_OUTER)?
            }
        };
        Ok(Box::new(StatementRoot::new(inner)))
    }
}

/// Build a streaming cursor for a read `PlanNode`. Interior operators build their
/// input(s) with the same `env` and `outer`; every FROM leaf prepends `outer` (a `CteScan`
/// does so at its output boundary — its standalone base-0 body is drained with an empty
/// outer; `RecursiveScan` is not a correlated leaf, it reads its step's working-table
/// frame). Recursion depth mirrors the plan tree depth, which the parser/planner bound
/// upstream.
pub(crate) fn build_cursor<'e>(
    node: &'e PlanNode,
    env: Env<'e>,
    outer: &'e [Value],
) -> Result<Box<dyn RowCursor + 'e>> {
    match node {
        // --- access leaves ---
        PlanNode::SeqScan(s) => scan::seq_scan(s, env, outer),
        PlanNode::RowidScan(r) => scan::rowid_scan(r, env, outer),

        // --- interior read operators (build input, wrap) ---
        PlanNode::Filter { input, predicate } => {
            let input = build_cursor(input, env, outer)?;
            Ok(filter::filter(env, predicate, input))
        }
        PlanNode::Project { input, exprs } => {
            let input = build_cursor(input, env, outer)?;
            Ok(project::project(env, exprs, input))
        }
        PlanNode::Sort { input, keys, limit } => {
            let input = build_cursor(input, env, outer)?;
            Ok(sort::sort(env, keys, limit.as_ref(), input))
        }
        PlanNode::Limit { input, limit: limit_expr, offset } => {
            let input = build_cursor(input, env, outer)?;
            Ok(limit::limit(env, limit_expr.as_ref(), offset.as_ref(), input))
        }
        PlanNode::Distinct { input, column_collations } => {
            let input = build_cursor(input, env, outer)?;
            Ok(distinct::distinct(input, column_collations))
        }

        // --- literal / no-table leaves ---
        PlanNode::Values { rows } => Ok(values::values(env, rows, outer)),
        PlanNode::SingleRow => Ok(values::single_row(outer)),

        // --- access leaf: index seek / scan ---
        PlanNode::IndexScan(s) => indexscan::index_scan(s, env, outer),

        // --- MIN/MAX index/rowid seek fast path (one b-tree seek, not a full-scan agg) ---
        PlanNode::MinMaxSeek(m) => minmax_seek::minmax_seek(m, env, outer),

        // --- read operators (build input, wrap) ---
        PlanNode::Aggregate(a) => aggregate::aggregate(a, env, outer),
        PlanNode::Join(j) => join::join(j, env, outer),
        PlanNode::SetOp(s) => setop::set_op(s, env, outer),
        PlanNode::CteScan { id, column_count } => cte::cte_scan(*id, *column_count, env, outer),
        PlanNode::RecursiveScan { column_count } => Ok(cte::recursive_scan(*column_count)),
        PlanNode::TableFunctionScan { kind, arg, path, column_count, emit_json } => {
            table_function::table_function_scan(
                *kind, arg, path.as_ref(), *column_count, *emit_json, env, outer,
            )
        }
        PlanNode::PragmaFunctionScan { kind, name_arg, schema_arg, column_count } => {
            pragma_function::pragma_function_scan(
                *kind,
                name_arg.as_ref(),
                schema_arg.as_ref(),
                *column_count,
                env,
                outer,
            )
        }
        PlanNode::Window(w) => window::window(w, env, outer),

        // --- nodes that never belong on the read path ---
        PlanNode::Insert(_)
        | PlanNode::Update(_)
        | PlanNode::Delete(_)
        | PlanNode::InsteadOf(_) => Err(Error::sql("DML node reached read dispatch")),
        PlanNode::CreateTable(_) | PlanNode::CreateIndex(_) => Err(Error::sql(
            "CREATE TABLE/INDEX is executed by the engine via the catalog, not the executor",
        )),
    }
}

/// Build a cursor for a DML `PlanNode` (`INSERT`/`UPDATE`/`DELETE`), holding the whole
/// pager SET (`pagers`) the write needs. `INSERT` (in [`insert`]), `UPDATE` (in
/// [`update`]), and `DELETE` (in [`delete`]) are implemented. The implementation
/// pattern they follow: read the source/scan rows via a scoped shared view of the SET
/// (`pagers.source()`) through [`build_cursor`] — the source may live in a different
/// namespace than the target — buffer what the write needs, then reborrow the TARGET store
/// (`pagers.target(node.db)`) mutably and apply the mutation, recording the change on the
/// [`Runtime`].
///
/// The two-phase shape is what makes this compile with ordinary borrows: the shared
/// SET borrow is released before the mutable target reborrow begins. It is sound because
/// a DML write phase touches only its target namespace — SQLite forbids cross-database
/// foreign keys, and CHECK/generated/DEFAULT/RETURNING carry no table subqueries — so the
/// only cross-namespace access is the buffered phase-1 source read.
///
/// `outer` is the correlated frame this DML runs under: empty ([`EMPTY_OUTER`]) for a
/// top-level statement, or the firing trigger's `OLD ++ NEW` frame when this DML is a
/// trigger action ([`crate::ops::trigger`]). The source/scan is built with `outer`, and
/// each operator strips it back off to recover the table row (see the ops files); an
/// empty `outer` makes that a no-op, so a non-trigger DML is byte-for-byte unchanged.
pub(crate) fn build_dml<'e>(
    node: &'e PlanNode,
    catalog: &'e dyn Catalog,
    pagers: PagerSet<'e>,
    plan: &'e Plan,
    outer: &'e [Value],
) -> Result<Box<dyn RowCursor + 'e>> {
    match node {
        PlanNode::Insert(i) => insert::insert(i, catalog, pagers, plan, outer),
        PlanNode::Update(u) => update::update(u, catalog, pagers, plan, outer),
        PlanNode::Delete(d) => delete::delete(d, catalog, pagers, plan, outer),
        PlanNode::InsteadOf(io) => instead_of::instead_of(io, catalog, pagers, plan, outer),
        _ => Err(Error::sql("build_dml received a non-DML node")),
    }
}
