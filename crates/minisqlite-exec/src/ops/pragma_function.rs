//! The `pragma_*` schema-introspection table-valued function operator
//! ([`PlanNode::PragmaFunctionScan`]): a FROM-clause row source for `pragma_table_info()`,
//! `pragma_index_list()`, and the other introspection PRAGMAs exposed as functions
//! (pragma.html §2 "PRAGMA functions").
//!
//! Like the other leaf sources ([`crate::ops::values`], [`crate::ops::table_function`]) this
//! emits `outer ++ local`: the local row is the pragma row (the fixed visible columns, no
//! trailing rowid) and the `outer` prefix is the correlated frame (empty at the top level).
//!
//! ## One row-builder, shared with the statement form
//! The rows come from [`minisqlite_catalog::pragma_rows`] over the live [`Catalog`] — the
//! SAME builder the engine's `PRAGMA table_info(t)` STATEMENT path calls. So
//! `SELECT * FROM pragma_table_info('t')` and `PRAGMA table_info(t)` are byte-identical by
//! construction; there is no second copy of the resolution rule or the row shape to drift.
//!
//! ## Correlation (implicit LATERAL)
//! Pragma table-valued functions are implicitly LATERAL (pragma.html's example joins
//! `pragma_index_list('t')` against `pragma_index_info(il.name)`, the inner argument reading
//! the outer row). The planner makes the correlated TVF the right operand of a
//! [`JoinStrategy::IndexNestedLoop`](minisqlite_plan::JoinStrategy) join, whose executor
//! rebuilds the right cursor with the current left row as `outer` for every left row (see
//! [`crate::ops::join`]). This operator therefore evaluates its `name`/`schema` argument
//! expressions against `outer`, so a rebuild per left row re-resolves the object for that
//! row — exactly the per-outer-row re-evaluation LATERAL requires.
//!
//! ## Laziness and bounds
//! Setup is deferred to the first pull because argument evaluation needs the [`Runtime`]
//! (parameters, correlated subqueries), only available in [`RowCursor::next_row`]. The rows
//! are built once (bounded by the object's column/index/FK count — a pure O(schema) read, no
//! data-page access) and then streamed one per pull.

use minisqlite_catalog::{pragma_rows, PragmaFunction};
use minisqlite_expr::{eval, EvalExpr};
use minisqlite_types::{cast_to, Affinity, Result, Row, Value};

use crate::context::EvalCtx;
use crate::env::Env;
use crate::ops::with_outer;
use crate::runtime::Runtime;
use crate::RowCursor;

/// Build a `pragma_*` introspection table-valued-function scan. `name_arg` is the
/// object-name argument (`pragma_table_info('t')` → `'t'`) and `schema_arg` the optional
/// trailing schema argument (`pragma_table_info('t','main')` → `'main'`); both are evaluated
/// against `outer`. Emits `outer ++ <the pragma's fixed columns>` (`column_count` wide, no
/// trailing rowid), the same rows as the corresponding `PRAGMA` statement.
pub(crate) fn pragma_function_scan<'e>(
    kind: PragmaFunction,
    name_arg: Option<&'e EvalExpr>,
    schema_arg: Option<&'e EvalExpr>,
    column_count: usize,
    env: Env<'e>,
    outer: &'e [Value],
) -> Result<Box<dyn RowCursor + 'e>> {
    Ok(Box::new(PragmaFunctionScanCursor {
        kind,
        name_arg,
        schema_arg,
        column_count,
        env,
        outer,
        built: None,
    }))
}

struct PragmaFunctionScanCursor<'e> {
    kind: PragmaFunction,
    name_arg: Option<&'e EvalExpr>,
    schema_arg: Option<&'e EvalExpr>,
    column_count: usize,
    env: Env<'e>,
    outer: &'e [Value],
    /// The introspection rows, built on the first pull; `None` until then.
    built: Option<std::vec::IntoIter<Row>>,
}

impl RowCursor for PragmaFunctionScanCursor<'_> {
    fn next_row(&mut self, rt: &mut Runtime) -> Result<Option<Row>> {
        if self.built.is_none() {
            self.build(rt)?;
        }
        let rows = self.built.as_mut().expect("built on first pull");
        let Some(local) = rows.next() else { return Ok(None) };
        debug_assert_eq!(
            local.len(),
            self.column_count,
            "PragmaFunctionScan row is the pragma's fixed columns (no trailing rowid)"
        );
        Ok(Some(with_outer(self.outer, local)))
    }
}

impl PragmaFunctionScanCursor<'_> {
    /// Evaluate the name/schema arguments against `outer` and build the introspection rows on
    /// the first pull. A NULL (or absent) object name yields zero rows — the statement form's
    /// convention for an unknown/absent object — as does an unknown schema qualifier.
    fn build(&mut self, rt: &mut Runtime) -> Result<()> {
        let mut ctx = EvalCtx { rt, env: self.env, outer: self.outer };
        let name = match self.name_arg {
            Some(e) => value_to_name(eval(e, self.outer, &mut ctx)?),
            None => None,
        };
        let schema = match self.schema_arg {
            Some(e) => value_to_name(eval(e, self.outer, &mut ctx)?),
            None => None,
        };
        let rows = pragma_rows(self.env.catalog, self.kind, schema.as_deref(), name.as_deref())?;
        debug_assert!(
            rows.iter().all(|r| r.len() == self.column_count),
            "pragma_rows returns the pragma's fixed-width columns"
        );
        self.built = Some(rows.into_iter());
        Ok(())
    }
}

/// Coerce a pragma table-valued-function argument to the object / schema NAME it denotes.
/// SQLite passes the argument through text coercion (`sqlite3_value_text`), so a text value
/// is the name verbatim, a number renders to its text form, and a BLOB's bytes are read as
/// text (this reuses the engine's one `CAST … AS TEXT` path, [`cast_to`]). A NULL argument
/// denotes "no name" → `None`, which [`pragma_rows`] renders as the empty result, matching
/// the statement form for an absent/NULL object.
fn value_to_name(v: Value) -> Option<String> {
    match cast_to(v, Affinity::Text) {
        Value::Text(s) => Some(s),
        Value::Null => None,
        other => unreachable!("cast_to(_, Text) yields Text or Null, got {other:?}"),
    }
}
