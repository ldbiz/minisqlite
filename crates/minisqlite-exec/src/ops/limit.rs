//! `Limit` — `LIMIT`/`OFFSET`. Both are scalar expressions evaluated once before
//! iteration; skip `offset` rows then emit up to `limit`. Passes rows through
//! unchanged (input layout).

use minisqlite_expr::{eval, EvalExpr};
use minisqlite_types::{Result, Row};

use crate::context::EvalCtx;
use crate::env::Env;
use crate::ops::must_be_int::must_be_int;
use crate::runtime::Runtime;
use crate::RowCursor;

/// Build a limit/offset over `input`.
pub(crate) fn limit<'a>(
    env: Env<'a>,
    limit: Option<&'a EvalExpr>,
    offset: Option<&'a EvalExpr>,
    input: Box<dyn RowCursor + 'a>,
) -> Box<dyn RowCursor + 'a> {
    Box::new(LimitCursor { env, limit, offset, input, state: None })
}

struct LimitCursor<'a> {
    env: Env<'a>,
    limit: Option<&'a EvalExpr>,
    offset: Option<&'a EvalExpr>,
    input: Box<dyn RowCursor + 'a>,
    /// Resolved iteration state, computed once on the first pull. `None` until then.
    state: Option<State>,
}

/// The resolved limit state after evaluating the expressions and skipping `offset`.
struct State {
    /// Remaining rows to emit; `None` means unbounded (no `LIMIT` / a negative one).
    remaining: Option<i64>,
}

impl RowCursor for LimitCursor<'_> {
    fn next_row(&mut self, rt: &mut Runtime) -> Result<Option<Row>> {
        if self.state.is_none() {
            self.state = Some(self.init(rt)?);
        }
        // An exhausted cap stops the scan without pulling another input row.
        if matches!(self.state.as_ref().expect("state set on first pull").remaining, Some(0)) {
            return Ok(None);
        }
        match self.input.next_row(rt)? {
            Some(row) => {
                if let Some(rem) =
                    self.state.as_mut().expect("state set on first pull").remaining.as_mut()
                {
                    *rem -= 1;
                }
                Ok(Some(row))
            }
            None => Ok(None),
        }
    }
}

impl LimitCursor<'_> {
    /// Evaluate `offset`/`limit` once (on the first pull) and skip the offset rows.
    ///
    /// Per SQLite (`lang_select.html` §5): an ABSENT `OFFSET` is 0 and a NEGATIVE
    /// `OFFSET` is treated as 0; an ABSENT `LIMIT` is unbounded and a NEGATIVE
    /// `LIMIT` is likewise unbounded. But a PRESENT clause whose expression does
    /// not losslessly convert to an integer — NULL, a fractional or out-of-range
    /// REAL, non-numeric TEXT, a BLOB — is an ERROR (see [`Self::eval_count`]), not
    /// "no clause". The expressions are constant per query, so they are evaluated
    /// with an empty row. (A correlated `LIMIT`/`OFFSET` referencing outer columns
    /// is a documented follow-up; it would thread the outer row here.)
    ///
    /// WARNING for that follow-up: the `Sort` below this `Limit` re-evaluates the SAME
    /// bound INDEPENDENTLY (`SortCursor::retain_bound`, ops/sort.rs) to size its bounded
    /// top-k. Threading the outer row into ONLY this node would let the two evaluations
    /// diverge and the sort could drop rows this node still needs (silent row loss). A
    /// correlated bound MUST be threaded into BOTH evaluations, or the sort's retention
    /// bound disabled for it (evaluating the bound ONCE and sharing the scalar is the
    /// SQLite-faithful fix). See `select::limit_expr_is_deterministic`, which keeps the
    /// sort bound off any non-deterministic LIMIT today.
    fn init(&mut self, rt: &mut Runtime) -> Result<State> {
        // Absent or negative OFFSET => 0 (a present NULL/non-integral already errored).
        let offset = self.eval_count(rt, self.offset)?.filter(|&n| n > 0).unwrap_or(0);
        // Absent or negative LIMIT => unbounded (`None`); >= 0 caps at that many rows.
        let remaining = self.eval_count(rt, self.limit)?.filter(|&n| n >= 0);

        // Skip `offset` input rows up front.
        for _ in 0..offset {
            if self.input.next_row(rt)?.is_none() {
                break;
            }
        }
        Ok(State { remaining })
    }

    /// Evaluate an optional `LIMIT`/`OFFSET` count expression and coerce it with the
    /// shared [`must_be_int`] (SQLite's `OP_MustBeInt`).
    ///
    /// - Clause ABSENT (`None`) -> `Ok(None)` (the caller reads this as unbounded /
    ///   zero).
    /// - Clause PRESENT and losslessly integral -> `Ok(Some(i))`. `i` may be
    ///   negative; the caller maps a negative `LIMIT` to unbounded and a negative
    ///   `OFFSET` to zero.
    /// - Clause PRESENT but NOT losslessly convertible to an integer — including a
    ///   present NULL, which is an error here, NOT "no clause" — -> `Err`, via
    ///   `require_int` (the caller for which a NULL bound is a mismatch).
    ///
    /// The evaluation (`eval`) is the effectful shell; the decision is the pure
    /// [`must_be_int`] core, so the coercion rule is testable without a cursor.
    fn eval_count(&mut self, rt: &mut Runtime, expr: Option<&EvalExpr>) -> Result<Option<i64>> {
        match expr {
            None => Ok(None),
            Some(e) => {
                let mut ctx = EvalCtx { rt, env: self.env, outer: &[] };
                let v = eval(e, &[], &mut ctx)?;
                must_be_int(&v).require_int().map(Some)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;

    use minisqlite_catalog::SchemaCatalog;
    use minisqlite_pager::MemPager;
    use minisqlite_plan::{Plan, PlanNode};
    // `Value` is used only by the tests (the operator code no longer names it), so it is
    // imported here rather than at module scope. `Row`, `Result`, `EvalExpr`, `limit`,
    // `Env`, `Runtime`, `RowCursor` all come through `super::*`.
    use minisqlite_types::Value;

    use super::*;

    /// A source that yields up to `remaining` one-column rows, counting how many times
    /// it is pulled. Lets a test observe whether `Limit` over-pulls its input.
    struct CountingSource {
        pulls: Rc<Cell<usize>>,
        remaining: usize,
    }

    impl RowCursor for CountingSource {
        fn next_row(&mut self, _rt: &mut Runtime) -> Result<Option<Row>> {
            self.pulls.set(self.pulls.get() + 1);
            if self.remaining == 0 {
                return Ok(None);
            }
            self.remaining -= 1;
            Ok(Some(vec![Value::Integer(1)]))
        }
    }

    /// The streaming behavior, shown directly: `LIMIT 2` over a 1000-row
    /// source pulls the input only ~twice, not 1000 times. A materializing rewrite
    /// (collect all 1000, then take 2) would pull ~1000 times and fail this — the
    /// result-count-only integration test cannot catch that (a materializing impl
    /// passes it too), so this counting source is the real non-materialization proof.
    #[test]
    fn limit_stops_pulling_input_after_cap() {
        let pager = MemPager::new(4096);
        let cat = SchemaCatalog::new();
        // A minimal plan just to satisfy `Env`; the integer LIMIT literal never reads it.
        let plan = Plan {
            root: PlanNode::SingleRow,
            result_columns: Vec::new(),
            ctes: Vec::new(),
            subqueries: Vec::new(),
            mutates: false,
            generated: Vec::new(),
        };
        let env = Env {
            catalog: &cat,
            pagers: crate::env::Pagers::One { db: minisqlite_types::DbIndex::MAIN, pager: &pager },
            plan: &plan,
        };
        let pulls = Rc::new(Cell::new(0usize));
        let source = CountingSource { pulls: Rc::clone(&pulls), remaining: 1000 };
        let cap = EvalExpr::Literal(Value::Integer(2));

        let mut cur = limit(env, Some(&cap), None, Box::new(source));
        let mut rt = Runtime::new();
        let mut emitted = 0;
        while cur.next_row(&mut rt).unwrap().is_some() {
            emitted += 1;
        }

        assert_eq!(emitted, 2, "LIMIT 2 emits exactly 2 rows");
        // Offset 0 => exactly 2 pulls to emit 2 rows; the terminating None comes from
        // the exhausted cap WITHOUT another pull. Bound at 3 to stay robust while still
        // failing hard on materialization (which would pull ~1000).
        assert!(
            pulls.get() <= 3,
            "streaming: input pulled {} times; a materializing LIMIT would pull ~1000",
            pulls.get()
        );
    }

    /// A plain `n`-row one-column source (unlike `CountingSource`, the coercion
    /// tests below assert the emitted row COUNT, not pull counts).
    struct RowSource {
        remaining: usize,
    }

    impl RowCursor for RowSource {
        fn next_row(&mut self, _rt: &mut Runtime) -> Result<Option<Row>> {
            if self.remaining == 0 {
                return Ok(None);
            }
            self.remaining -= 1;
            Ok(Some(vec![Value::Integer(1)]))
        }
    }

    /// Drive `limit(..)` with the given `LIMIT`/`OFFSET` literals over an `n`-row
    /// source and drain it, returning the emitted rows or the first error. Owns all
    /// the borrowed `Env` pieces so each test below is a single call; the drain
    /// happens before return so nothing outlives the local borrows. Because
    /// `init` evaluates the bound expressions on the FIRST pull, a coercion error
    /// surfaces on that first `next_row`, so `run_limit(..).is_err()` witnesses it.
    fn run_limit(
        limit_expr: Option<EvalExpr>,
        offset_expr: Option<EvalExpr>,
        n: usize,
    ) -> Result<Vec<Row>> {
        let pager = MemPager::new(4096);
        let cat = SchemaCatalog::new();
        let plan = Plan {
            root: PlanNode::SingleRow,
            result_columns: Vec::new(),
            ctes: Vec::new(),
            subqueries: Vec::new(),
            mutates: false,
            generated: Vec::new(),
        };
        let env = Env {
            catalog: &cat,
            pagers: crate::env::Pagers::One { db: minisqlite_types::DbIndex::MAIN, pager: &pager },
            plan: &plan,
        };
        let source = RowSource { remaining: n };
        let mut cur = limit(env, limit_expr.as_ref(), offset_expr.as_ref(), Box::new(source));
        let mut rt = Runtime::new();
        let mut rows = Vec::new();
        while let Some(row) = cur.next_row(&mut rt)? {
            rows.push(row);
        }
        Ok(rows)
    }

    // ---- Error path: a PRESENT non-integral LIMIT/OFFSET must error (OP_MustBeInt) ----
    // These mirror `lang_select.html` §5: NULL / a fractional REAL / non-numeric TEXT
    // that cannot be losslessly converted to an integer is an ERROR, NOT "no clause".

    #[test]
    fn limit_fractional_real_errors() {
        // 2.5 is not losslessly integral -> error (was truncated to 2 before the fix).
        assert!(run_limit(Some(EvalExpr::Literal(Value::Real(2.5))), None, 5).is_err());
    }

    #[test]
    fn limit_null_errors() {
        // A present NULL LIMIT errors (was treated as unbounded before the fix).
        assert!(run_limit(Some(EvalExpr::Literal(Value::Null)), None, 5).is_err());
    }

    #[test]
    fn offset_fractional_real_errors() {
        // OFFSET is evaluated first in `init`; a fractional OFFSET errors on its own.
        assert!(run_limit(None, Some(EvalExpr::Literal(Value::Real(2.5))), 5).is_err());
    }

    #[test]
    fn offset_null_errors() {
        // A present NULL OFFSET errors (was treated as 0 before the fix).
        assert!(run_limit(None, Some(EvalExpr::Literal(Value::Null)), 5).is_err());
    }

    #[test]
    fn limit_fractional_text_errors() {
        // '2.5' IS numeric text, but NUMERIC affinity converts it to Real(2.5) (not an
        // integer), so it errors — the required `LIMIT '2.5'` case. (The error is because
        // it is fractional, NOT because it is non-numeric; see `Text("x")` below for that.)
        assert!(run_limit(Some(EvalExpr::Literal(Value::Text("2.5".into()))), None, 5).is_err());
    }

    // ---- Success path: a losslessly-integral bound is accepted ----

    #[test]
    fn limit_integral_real_is_accepted() {
        // 2.0 demotes to Integer(2) under NUMERIC affinity -> exactly 2 rows.
        let rows = run_limit(Some(EvalExpr::Literal(Value::Real(2.0))), None, 5).unwrap();
        assert_eq!(rows.len(), 2, "LIMIT 2.0 (lossless) caps at 2 rows");
    }

    #[test]
    fn limit_numeric_text_is_accepted() {
        // '3' converts to Integer(3) under NUMERIC affinity -> exactly 3 rows.
        let rows = run_limit(Some(EvalExpr::Literal(Value::Text("3".into()))), None, 5).unwrap();
        assert_eq!(rows.len(), 3, "LIMIT '3' (numeric text) caps at 3 rows");
    }

    #[test]
    fn negative_limit_is_unbounded() {
        // A negative LIMIT is unbounded, not an error -> all 5 rows.
        let rows = run_limit(Some(EvalExpr::Literal(Value::Integer(-1))), None, 5).unwrap();
        assert_eq!(rows.len(), 5, "LIMIT -1 is unbounded, returns all rows");
    }

    #[test]
    fn negative_offset_is_zero() {
        // A negative OFFSET is treated as 0 (not an error); LIMIT 2 then takes 2 rows.
        let rows = run_limit(
            Some(EvalExpr::Literal(Value::Integer(2))),
            Some(EvalExpr::Literal(Value::Integer(-5))),
            5,
        )
        .unwrap();
        assert_eq!(rows.len(), 2, "OFFSET -5 acts as 0; LIMIT 2 takes the first 2 rows");
    }

    #[test]
    fn limit_zero_emits_no_rows() {
        // Regression guard: `LIMIT 0` is `Some(0)` (a valid, non-negative bound), NOT
        // unbounded. The cursor must stop at the exhausted cap and emit zero rows over a
        // non-empty source — distinguishing it from an absent/negative LIMIT.
        let rows = run_limit(Some(EvalExpr::Literal(Value::Integer(0))), None, 5).unwrap();
        assert!(rows.is_empty(), "LIMIT 0 emits zero rows, got {}", rows.len());
    }

    // The pure `OP_MustBeInt` coercion core is exercised exhaustively in
    // `ops::must_be_int` (its own home); the cursor tests above are the thin per-caller
    // coverage that `Limit` wires it in and applies the negative/absent bound meanings.
}
