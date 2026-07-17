//! The JSON table-valued function operator ([`PlanNode::TableFunctionScan`]): a
//! FROM-clause row source for `json_each()` / `json_tree()` (json1.html §4.24).
//!
//! Like the other leaf sources ([`crate::ops::values`], [`crate::ops::scan`]) this emits
//! `outer ++ local`: the local row is the TVF row — the 8 visible columns plus the two
//! hidden input columns `json`/`root` (json1.html §4.24) appended last — and the `outer`
//! prefix is the correlated frame (empty at the top level).
//!
//! ## Lazy `json` materialization (avoiding an `O(n·|document|)` copy)
//! The hidden `json` column is the WHOLE document, identical for every row of a walk.
//! Appending a clone of it to each of the `n` emitted rows would cost `O(n·|document|)` —
//! quadratic for a flat array/object whose text grows with its element count — even for
//! the common queries that never name `json` (`SELECT value`, `SELECT *`, `count(*)`).
//! SQLite computes a hidden column only when it is selected; so does this operator. The
//! planner sets `emit_json` iff the statement actually references the `json` column, and
//! this operator clones the document into a row's `json` slot ONLY then — otherwise it
//! writes SQL NULL there (the slot is never read, so the value is immaterial). The small
//! `root` path string is always appended. `json_table_rows` never copies the document; the
//! evaluated first-argument `Value` this operator already holds is the source of the clone.
//!
//! ## Correlation (implicit LATERAL)
//! SQLite table-valued functions are implicitly LATERAL, so `FROM t, json_each(t.col)`
//! is legal and the argument references the LEFT table's columns. The planner makes this
//! TVF the right operand of a [`JoinStrategy::IndexNestedLoop`] join, whose executor
//! rebuilds the right cursor with the current left row as `outer` for every left row
//! (see [`crate::ops::join`]). This operator therefore evaluates its `arg`/`path`
//! expressions against `outer` — so a rebuild per left row re-generates the rows for
//! that row's JSON — exactly the per-outer-row re-evaluation LATERAL requires.
//!
//! ## Laziness and bounds
//! Setup is deferred to the first pull because argument evaluation needs the [`Runtime`]
//! (parameters, correlated subqueries), which is only available in
//! [`RowCursor::next_row`]. The generated rows are buffered once (bounded by the JSON
//! document's element count — never a copy of a base table) and then streamed one per
//! pull. Row generation itself lives in `minisqlite-functions` ([`json_table_rows`]);
//! this operator only evaluates the arguments and threads the `outer` prefix.

use minisqlite_expr::{eval, EvalExpr};
use minisqlite_functions::{json_table_rows, JsonTableKind, JSON_TABLE_COLUMN_COUNT};
use minisqlite_types::{Result, Row, Value};

use crate::context::EvalCtx;
use crate::env::Env;
use crate::ops::with_outer;
use crate::runtime::Runtime;
use crate::RowCursor;

/// Build a `json_each` / `json_tree` table-valued-function scan. `arg` is the JSON
/// document expression, `path` the optional start-path expression; both are evaluated
/// against `outer`. Emits `outer ++ [key, value, type, atom, id, parent, fullkey, path,
/// json, root]` (the two hidden input columns last). `emit_json` (set by the planner when
/// the statement references the `json` column) selects whether the `json` slot carries the
/// document or SQL NULL — see the lazy-materialization note in the module docs.
pub(crate) fn table_function_scan<'e>(
    kind: JsonTableKind,
    arg: &'e EvalExpr,
    path: Option<&'e EvalExpr>,
    column_count: usize,
    emit_json: bool,
    env: Env<'e>,
    outer: &'e [Value],
) -> Result<Box<dyn RowCursor + 'e>> {
    Ok(Box::new(TableFunctionScanCursor {
        kind,
        arg,
        path,
        column_count,
        emit_json,
        env,
        outer,
        built: None,
    }))
}

/// The generated rows plus the constant hidden values, materialized together on the first
/// pull. Collapsing the row iterator and the hidden values into one `Option` makes the
/// "both set or neither" invariant unrepresentable (there is no half-built state).
struct Built {
    /// The content rows (the eight visible columns), streamed one per pull.
    rows: std::vec::IntoIter<Row>,
    /// The hidden `json` document, present only when the statement references it (else the
    /// `json` slot is SQL NULL). Cloned per emitted row when present.
    json: Option<Value>,
    /// The hidden `root` start-path string, appended to every row (cheap).
    root: Value,
}

struct TableFunctionScanCursor<'e> {
    kind: JsonTableKind,
    arg: &'e EvalExpr,
    path: Option<&'e EvalExpr>,
    column_count: usize,
    /// Whether to carry the document in each row's hidden `json` slot (true iff the
    /// statement references the `json` column). See the module docs.
    emit_json: bool,
    env: Env<'e>,
    outer: &'e [Value],
    /// The generated rows and hidden values, built on the first pull; `None` until then.
    built: Option<Built>,
}

impl RowCursor for TableFunctionScanCursor<'_> {
    fn next_row(&mut self, rt: &mut Runtime) -> Result<Option<Row>> {
        if self.built.is_none() {
            self.build(rt)?;
        }
        let built = self.built.as_mut().expect("built on first pull");
        let Some(mut local) = built.rows.next() else { return Ok(None) };
        // Append the two hidden input columns (`json`, `root`) after the visible ones, so
        // the local row matches the `Source::Derived` schema width the binder resolved
        // `json`/`root` against. The document is cloned into the `json` slot only when the
        // query references it; otherwise the slot is SQL NULL (never read).
        local.push(built.json.clone().unwrap_or(Value::Null));
        local.push(built.root.clone());
        debug_assert_eq!(
            local.len(),
            self.column_count,
            "TableFunctionScan row is the visible columns plus json/root (no trailing rowid)"
        );
        Ok(Some(with_outer(self.outer, local)))
    }
}

impl TableFunctionScanCursor<'_> {
    /// Evaluate the argument expressions against `outer` and generate the TVF rows on the
    /// first pull. A malformed-JSON argument or a bad path is a returned error (surfacing
    /// at query time, as SQLite does); a NULL document / NULL path / a path selecting
    /// nothing yields zero rows. Stores the content-row iterator plus the hidden `root`
    /// (and the `json` document only when `emit_json`) for [`RowCursor::next_row`].
    fn build(&mut self, rt: &mut Runtime) -> Result<()> {
        let mut ctx = EvalCtx { rt, env: self.env, outer: self.outer };
        let arg_val = eval(self.arg, self.outer, &mut ctx)?;
        let path_val = match self.path {
            Some(p) => Some(eval(p, self.outer, &mut ctx)?),
            None => None,
        };
        let out = json_table_rows(self.kind, &arg_val, path_val.as_ref())?;
        debug_assert!(
            out.rows.iter().all(|r| r.len() == JSON_TABLE_COLUMN_COUNT),
            "json_table_rows content rows are the visible columns wide"
        );
        // Keep the document only when it will be read: the hidden `json` column echoes the
        // raw first argument (json1.html §4.24), and `arg_val` is exactly that. Dropping it
        // otherwise is what removes the per-row document copy on the common path.
        let json = if self.emit_json { Some(arg_val) } else { None };
        self.built = Some(Built { rows: out.rows.into_iter(), json, root: out.root });
        Ok(())
    }
}
