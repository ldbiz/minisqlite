//! The evaluation context: the narrow seams the evaluator reaches out through for
//! everything it cannot compute from the row alone.
//!
//! Two traits, split by who needs what. [`FnContext`] is the smaller surface a
//! *scalar/aggregate function* is handed — wall clock, RNG, and the connection
//! counters. [`EvalContext`] is the superset the *expression evaluator* needs,
//! adding bind parameters and the subquery callbacks. Keeping them separate means
//! a function implementation cannot reach the subquery machinery, and the executor
//! can hand a function the smaller capability by upcasting.

use minisqlite_types::{Result, Value};

use crate::ir::{CompareMeta, SubqueryId};

/// The capabilities a built-in function may use while running: the wall clock, the
/// connection RNG, and the connection's mutation counters. Deliberately small so a
/// function cannot see the subquery/parameter machinery.
///
/// This trait is expected to grow as the date/time and other function families
/// need more from the connection — it is owned here, so adding a method is a
/// forward change, not a fork.
pub trait FnContext {
    /// Current wall-clock time as Unix milliseconds (UTC), for `CURRENT_*` and the
    /// date/time functions.
    fn now_unix_millis(&self) -> i64;
    /// A pseudo-random `i64`, for `random()`.
    fn random_i64(&mut self) -> i64;
    /// Fill `buf` with pseudo-random bytes, for `randomblob(n)`.
    fn fill_random(&mut self, buf: &mut [u8]);
    /// The rowid of the most recent successful `INSERT` on this connection.
    fn last_insert_rowid(&self) -> i64;
    /// Rows changed by the most recent `INSERT`/`UPDATE`/`DELETE` statement.
    fn changes(&self) -> i64;
    /// Total rows changed since the connection was opened.
    fn total_changes(&self) -> i64;

    // --- Ephemeral JSON value-subtype channel (json1.html §3.4) ---------------
    //
    // SQLite tags the RESULT of a JSON function with a "subtype" so that when that
    // result is passed *directly* as an argument to another JSON function, the
    // inner function embeds it as literal JSON rather than re-quoting it as a
    // string: `json_array(json('[1,2]'))` is `[[1,2]]`, not `["[1,2]"]`. The
    // subtype is intentionally EPHEMERAL — it rides a value only within a single
    // expression evaluation and is lost the moment the value is stored to a
    // row/column, read from one, or crosses a subquery — so it lives here on the
    // evaluation context, never on `Value`.
    //
    // The evaluator owns the wiring: before calling a function it publishes the
    // per-argument subtypes (`set_arg_subtypes`) so the function can read arg `i`'s
    // subtype (`arg_subtype`); the function marks its own result's subtype
    // (`set_result_subtype`); and the evaluator reads-and-clears it afterward
    // (`take_result_subtype`) to carry up to an enclosing call. A subtype of `0`
    // means "no subtype" (an ordinary SQL value); the JSON functions use `74`
    // (ASCII `'J'`), SQLite's JSON subtype tag.
    //
    // The default bodies are no-ops so a context that does not care about subtypes
    // (every test mock, and any non-executor `FnContext`) keeps compiling with
    // subtype tracking silently disabled — which quotes nested JSON, the correct
    // behavior for a context that never produces a subtype.

    /// Publish the per-argument subtypes for the call about to run. `s[i]` is the
    /// subtype of argument `i` (`0` = none). The evaluator sets this immediately
    /// before invoking a function.
    fn set_arg_subtypes(&mut self, _s: &[u8]) {}

    /// The subtype of the current call's argument `i` (`0` = none, or out of range).
    /// A JSON function reads this to decide whether to embed a `value` argument as
    /// JSON (subtype `74`) or quote it as a string.
    fn arg_subtype(&self, _i: usize) -> u8 {
        0
    }

    /// Mark the subtype of the value this function is about to return. A JSON
    /// function whose result is itself JSON calls this with `74` so an enclosing
    /// JSON function will embed the result.
    fn set_result_subtype(&mut self, _st: u8) {}

    /// Read and clear the result subtype the just-finished function set (`0` if it
    /// set none). The evaluator calls this right after the function returns and
    /// carries the value up as the call's subtype.
    fn take_result_subtype(&mut self) -> u8 {
        0
    }
}

/// Everything the expression evaluator needs beyond the current row: bind
/// parameters and the subquery callbacks, on top of [`FnContext`].
///
/// `FnContext` is a supertrait, so an `&mut dyn EvalContext` upcasts to
/// `&mut dyn FnContext` (Rust trait upcasting) when the evaluator hands a scalar
/// function its context.
///
/// The `regs` argument threaded through the subquery callbacks is the current
/// *outer* row, so a correlated subquery can read outer columns; the executor
/// implements these by running the referenced subplan.
pub trait EvalContext: FnContext {
    /// The value bound to parameter `index` (1-based `?N` / named params are mapped
    /// to indices by the binder). An out-of-range index is an error.
    fn param(&self, index: usize) -> Result<Value>;

    /// Signal that a trigger body evaluated `RAISE(IGNORE)`
    /// (lang_createtrigger.html §RAISE): a non-error request to abandon the current
    /// row's operation and continue the statement. The evaluator calls this from the
    /// `RAISE(IGNORE)` arm and then returns a sentinel `Err` to unwind out of the
    /// trigger body; the executor's context records the request on its runtime so the
    /// enclosing trigger-fire loop can turn it into a row-skip rather than an error.
    ///
    /// The default body is a NO-OP so a non-executor [`EvalContext`] (every test mock,
    /// and any context outside a trigger) keeps `RAISE(IGNORE)` failing as an ordinary
    /// error — the correct behavior where there is no row to skip.
    fn signal_raise_ignore(&mut self) {}

    /// Evaluate a scalar subquery: the first column of its first row, or NULL if it
    /// returns no rows. More than one column/row is the binder's concern, not this
    /// callback's.
    fn eval_scalar_subquery(&mut self, id: SubqueryId, regs: &[Value]) -> Result<Value>;

    /// Evaluate one column of a row-value subquery: column `col` of the FIRST row of
    /// subplan `id`, or NULL if it returns no row (or has no such column). The
    /// generalization of [`eval_scalar_subquery`](EvalContext::eval_scalar_subquery)
    /// (which is `col == 0`) that a column-list `UPDATE` source
    /// (`SET (a, b, …) = (SELECT …)`, rowvalue.html §2.3) uses: it emits one
    /// [`ScalarSubqueryColumn`](crate::EvalExpr::ScalarSubqueryColumn) per target
    /// column, all sharing the one subplan `id`.
    ///
    /// The default body errors so an existing non-executor [`EvalContext`] impl (a test
    /// mock) needs no change; only the real executor context overrides it — exactly as
    /// [`eval_in_subquery_row`](EvalContext::eval_in_subquery_row) does.
    fn eval_scalar_subquery_column(
        &mut self,
        id: SubqueryId,
        col: usize,
        regs: &[Value],
    ) -> Result<Value> {
        let _ = (id, col, regs);
        Err(minisqlite_types::Error::sql(
            "row-value subquery column not supported by this context",
        ))
    }

    /// Evaluate `EXISTS (subquery)`: whether the subquery returns at least one row.
    fn eval_exists(&mut self, id: SubqueryId, regs: &[Value]) -> Result<bool>;

    /// Evaluate `probe IN (subquery)` with three-valued membership:
    /// `Some(true)` = present, `Some(false)` = absent and the subquery had no
    /// NULLs that could hide a match, `None` = unknown (a NULL was involved). The
    /// executor applies `meta` (affinity + collation) to the comparison.
    fn eval_in_subquery(
        &mut self,
        id: SubqueryId,
        probe: &Value,
        meta: &CompareMeta,
        regs: &[Value],
    ) -> Result<Option<bool>>;

    /// Evaluate `(probe…) IN (subquery)` for a row-value (tuple) probe with
    /// three-valued membership (rowvalue.html §2.2) — the tuple generalization of
    /// [`eval_in_subquery`](EvalContext::eval_in_subquery). `probe` and `metas` are the
    /// same width as the subquery's output; element `i` uses `metas[i]` (affinity +
    /// collation). Returns `Some(true)` if some candidate row equals the probe
    /// element-wise, `Some(false)` if there were candidate rows (or none — an empty
    /// subquery) and none can match even accounting for NULLs, and `None` (unknown)
    /// when no candidate fully matches but some was equal on every non-NULL element and
    /// blocked only by a NULL. The evaluator applies `negated`/3VL wrapping around this.
    ///
    /// The default body errors so an existing non-executor [`EvalContext`] impl (a test
    /// mock) needs no change; only the real executor context overrides it.
    fn eval_in_subquery_row(
        &mut self,
        id: SubqueryId,
        probe: &[Value],
        metas: &[CompareMeta],
        regs: &[Value],
    ) -> Result<Option<bool>> {
        let _ = (id, probe, metas, regs);
        Err(minisqlite_types::Error::sql(
            "row-value IN subquery not supported by this context",
        ))
    }
}
