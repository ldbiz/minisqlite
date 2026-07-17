//! [`QueryPlanner`] — the concrete implementation of the [`Planner`] seam. It owns
//! the built-in [`FunctionRegistry`] (one per planner, shared read-only across
//! statements) and dispatches each parsed statement to its compiler in the
//! [`compile`](crate::compile) module.
//!
//! Read queries (`SELECT` / `VALUES`) and DML (`INSERT` / `UPDATE` / `DELETE`) are
//! each routed to their own compiler; a compiler not yet filled
//! in returns a clear "not yet implemented" error rather than a bogus plan. DDL /
//! transaction / utility statements are executed by the engine directly rather than
//! planned, so they too return a loud, specific error here.

use minisqlite_catalog::Catalog;
use minisqlite_expr::EvalExpr;
use minisqlite_functions::FunctionRegistry;
use minisqlite_sql::{CreateIndex, Statement};
use minisqlite_types::{DbIndex, Error, Result};

use crate::compile::{compile_delete, compile_insert, compile_select, compile_update};
use crate::plan::{GeneratedProgram, Plan};
use crate::plan_ctx::PlanCtx;
use crate::planner::Planner;

/// The engine's query planner: the live [`Planner`] the facade routes every
/// statement through.
pub struct QueryPlanner {
    registry: FunctionRegistry,
}

impl QueryPlanner {
    /// A planner with the full set of built-in functions registered.
    pub fn new() -> Self {
        QueryPlanner { registry: FunctionRegistry::builtins() }
    }

    /// Bind the GENERATED-column programs (STORED + VIRTUAL, in `CREATE TABLE` column
    /// order) of base table `table` IN NAMESPACE `db`, or an empty vec when that table is
    /// absent or has no generated column. Same binding a compiled statement's
    /// `Plan::generated` carries, exposed as a standalone entry for the `CREATE INDEX`
    /// backfill: an index over a VIRTUAL generated column must compute that column to key
    /// its entries, and the backfill runs outside the normal statement-planning path.
    /// Binding lives on the planner because it owns the [`FunctionRegistry`] and the
    /// executor never binds.
    ///
    /// `db` is resolved by the caller to the index's own target namespace
    /// (`index_target_db`) and MUST be looked up with `table_in(db, ..)`, not a bare
    /// search-order `table(..)`: a schema-qualified `CREATE INDEX main.i ON t` builds the
    /// index in `main`, but a bare lookup of `t` while a temp `t` shadows it would bind the
    /// TEMP table's generation programs into a MAIN index's backfill (wrong keyed values).
    /// Over a concrete single-namespace catalog (VACUUM) the `_in` default is the bare
    /// lookup, so passing `DbIndex::MAIN` there is byte-identical.
    pub fn table_generated_programs(
        &self,
        catalog: &dyn Catalog,
        db: DbIndex,
        table: &str,
    ) -> Result<Vec<GeneratedProgram>> {
        let td = match catalog.table_in(db, table)? {
            Some(td) => td,
            // Not a base table (or dropped): no programs. `create_index` raises the real
            // "no such table" error for a genuinely missing table; this just yields none.
            None => return Ok(Vec::new()),
        };
        let mut ctx = PlanCtx::new(&self.registry, catalog);
        crate::compile::generated::bind_generated_programs(&mut ctx, td)
    }

    /// Bind the per-key-column key EXPRESSIONS of a `CREATE INDEX` (`Some(EvalExpr)` for a
    /// genuine expression key like `t(a+b)` / `t(lower(a))`, `None` for an ordinary named
    /// column) against the target table's `[c0..c_{N-1}, rowid]` row layout, or an empty
    /// vec when the table is absent. Same binding [`compile::index_expr`] applies for DML
    /// index maintenance, exposed standalone for the `CREATE INDEX` backfill: the backfill
    /// keys the new index's entries by evaluating these against each existing row, and it
    /// runs outside the normal statement-planning path (mirrors [`table_generated_programs`]
    /// for VIRTUAL generated columns). Binding lives on the planner because it owns the
    /// [`FunctionRegistry`] and the executor never binds.
    ///
    /// `db` is the index's own target namespace (`index_target_db`); the target table is
    /// resolved with `table_in(db, ..)` for the same shadow-safety reason as
    /// [`table_generated_programs`] — a schema-qualified index over a shadowed table must
    /// key its backfill against the table in the index's namespace, not whatever the bare
    /// search order finds first.
    pub fn index_key_programs(
        &self,
        catalog: &dyn Catalog,
        db: DbIndex,
        ci: &CreateIndex,
    ) -> Result<Vec<Option<EvalExpr>>> {
        let td = match catalog.table_in(db, &ci.table)? {
            Some(td) => td,
            // Not a base table: no key programs. `create_index` raises the real
            // "no such table" error; this just yields none (an empty backfill key set).
            None => return Ok(Vec::new()),
        };
        // Classify the AST key columns the SAME way the catalog builder does (the shared
        // `index_ast_key_exprs`), so the backfill's compiled key programs line up exactly
        // with the stored index's `key_exprs` the DML paths bind.
        let key_exprs = minisqlite_catalog::index_ast_key_exprs(ci);
        let mut ctx = PlanCtx::new(&self.registry, catalog);
        crate::compile::index_expr::compile_index_key_exprs(&mut ctx, td, &key_exprs)
    }

    /// Bind the WHERE predicate of a PARTIAL `CREATE INDEX` (`Some(EvalExpr)`), or `None` for
    /// an ordinary full index, against the target table's `[c0..c_{N-1}, rowid]` row layout.
    /// The sibling of [`index_key_programs`] for the same `CREATE INDEX` backfill: the backfill
    /// inserts an entry only for existing rows whose predicate is TRUE (`partialindex.html`
    /// §2), so a `CREATE UNIQUE INDEX … WHERE p` on a populated table neither back-indexes nor
    /// duplicate-checks rows outside `p`. Binds from the `ci` AST's `where_clause` (as
    /// [`index_key_programs`] binds from the AST key columns), so it needs no stored
    /// [`IndexDef`](minisqlite_catalog::IndexDef) — the index may not be in the catalog yet.
    ///
    /// `db` is the index's own target namespace; the target table is resolved with
    /// `table_in(db, ..)` for the same shadow-safety reason as [`index_key_programs`].
    pub fn index_partial_predicate(
        &self,
        catalog: &dyn Catalog,
        db: DbIndex,
        ci: &CreateIndex,
    ) -> Result<Option<EvalExpr>> {
        let predicate = match &ci.where_clause {
            Some(p) => p,
            // A full index: every existing row is backfilled, no predicate to gate on.
            None => return Ok(None),
        };
        let td = match catalog.table_in(db, &ci.table)? {
            Some(td) => td,
            // Not a base table: `create_index` raises the real "no such table"; no backfill,
            // so no predicate to bind.
            None => return Ok(None),
        };
        let mut ctx = PlanCtx::new(&self.registry, catalog);
        let compiled =
            crate::compile::index_expr::compile_index_partial_predicate(&mut ctx, td, predicate)?;
        Ok(Some(compiled))
    }
}

impl Default for QueryPlanner {
    fn default() -> Self {
        Self::new()
    }
}

impl Planner for QueryPlanner {
    fn plan(&self, statement: &Statement, catalog: &dyn Catalog) -> Result<Plan> {
        let mut ctx = PlanCtx::new(&self.registry, catalog);
        let (root, result_columns, mutates) = match statement {
            Statement::Select(sel) => {
                let (root, result_columns) = compile_select(&mut ctx, sel)?;
                (root, result_columns, false)
            }
            Statement::Insert(ins) => {
                let (root, result_columns) = compile_insert(&mut ctx, ins)?;
                (root, result_columns, true)
            }
            Statement::Update(upd) => {
                let (root, result_columns) = compile_update(&mut ctx, upd)?;
                (root, result_columns, true)
            }
            Statement::Delete(del) => {
                let (root, result_columns) = compile_delete(&mut ctx, del)?;
                (root, result_columns, true)
            }
            other => {
                return Err(Error::sql(format!(
                    "{} is executed by the engine, not the query planner",
                    stmt_kind(other)
                )))
            }
        };
        // Move the side tables out of the context, releasing its catalog/registry borrow
        // so the generated-column pass can re-borrow them.
        let ctes = std::mem::take(&mut ctx.ctes);
        let subqueries = std::mem::take(&mut ctx.subqueries);
        drop(ctx);
        let mut plan =
            Plan { root, result_columns, ctes, subqueries, mutates, generated: Vec::new() };
        // Bind the GENERATED-column expressions of every base table this plan touches and
        // hang them on `plan.generated` (recursing into nested trigger action plans). The
        // scan leaves compute VIRTUAL columns on read; the INSERT/UPDATE executors compute
        // every generated column on write. See `compile::generated`.
        crate::compile::generated::populate_generated(&mut plan, &self.registry, catalog)?;
        // Covering-index post-pass: mark an `IndexScan` leaf so the executor reads covered
        // columns straight from the index entry and skips the by-rowid table fetch. Runs
        // AFTER `populate_generated` so it can see (and decline) any generated column. Purely
        // an optimization — it only flips a flag on a leaf, never changing the plan's shape.
        crate::compile::covering::mark_covering(catalog, &mut plan);
        Ok(plan)
    }
}

/// A human-readable name for a statement kind, for error messages.
fn stmt_kind(s: &Statement) -> &'static str {
    match s {
        Statement::Select(_) => "SELECT",
        Statement::Insert(_) => "INSERT",
        Statement::Update(_) => "UPDATE",
        Statement::Delete(_) => "DELETE",
        Statement::CreateTable(_) => "CREATE TABLE",
        Statement::CreateIndex(_) => "CREATE INDEX",
        Statement::CreateView(_) => "CREATE VIEW",
        Statement::CreateTrigger(_) => "CREATE TRIGGER",
        Statement::Drop(_) => "DROP",
        Statement::AlterTable(_) => "ALTER TABLE",
        Statement::Begin { .. } => "BEGIN",
        Statement::Commit => "COMMIT",
        Statement::Rollback { .. } => "ROLLBACK",
        Statement::Savepoint(_) => "SAVEPOINT",
        Statement::Release(_) => "RELEASE",
        Statement::Pragma { .. } => "PRAGMA",
        Statement::Vacuum { .. } => "VACUUM",
        Statement::Analyze { .. } => "ANALYZE",
        Statement::Reindex { .. } => "REINDEX",
        Statement::Attach { .. } => "ATTACH",
        Statement::Detach { .. } => "DETACH",
        Statement::Explain(_) => "EXPLAIN",
        Statement::ExplainQueryPlan(_) => "EXPLAIN QUERY PLAN",
    }
}
