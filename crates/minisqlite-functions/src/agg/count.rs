//! `count(*)` and `count(X)` (`lang_aggfunc.html`).
//!
//! One registration under `count` at `Arity::Range(0, 1)` covers both forms: the
//! binder lowers `count(*)` to a zero-argument `count` call, and `count(X)` is the
//! one-argument form. The accumulator distinguishes them per row by whether an
//! argument is present — `count(*)` counts every row, `count(X)` counts only rows
//! where `X` is non-NULL. The result is always INTEGER and is `0` (never NULL) for
//! a group with no rows.

use std::sync::Arc;

use minisqlite_expr::{AggregateAccumulator, AggregateFunction, FnContext};
use minisqlite_types::{Collation, Result, Value};

use crate::registry::{Arity, FunctionRegistry};

/// Register `count` (the single entry covering `count(*)` and `count(X)`).
pub(crate) fn register(reg: &mut FunctionRegistry) {
    reg.add_aggregate("count", Arity::Range(0, 1), Arc::new(Count));
}

/// The `count` aggregate factory.
#[derive(Debug)]
struct Count;

impl AggregateFunction for Count {
    fn new_accumulator(&self, _collation: Collation) -> Box<dyn AggregateAccumulator> {
        Box::new(CountAcc { n: 0 })
    }

    /// `count`'s finalized value is exactly its step count, so it is the one aggregate
    /// eligible for the operator's `count(*)` row-count fast path (see
    /// [`AggregateFunction::is_row_count`]). The caller additionally restricts that path
    /// to the bare zero-argument `count(*)` form.
    fn is_row_count(&self) -> bool {
        true
    }
}

/// Running count for one group.
struct CountAcc {
    n: i64,
}

impl AggregateAccumulator for CountAcc {
    fn step(&mut self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<()> {
        // `count(*)` (no argument) counts every row; `count(X)` counts rows whose
        // argument is non-NULL. `saturating_add` keeps a hostile row count from
        // panicking on i64 overflow (unreachable in practice at ~9.2e18 rows).
        let counts = match args.first() {
            None => true,
            Some(v) => !v.is_null(),
        };
        if counts {
            self.n = self.n.saturating_add(1);
        }
        Ok(())
    }

    fn step_many(&mut self, n: usize, args: &[Value], _ctx: &mut dyn FnContext) -> Result<()> {
        // Repeating the SAME row `n` times adds `n` — for `count(*)` always, for
        // `count(X)` only when `X` is non-NULL — the closed form of `n` `step` calls. This
        // lets the `count(*)` fast path apply a whole hash-join bucket in O(1) instead of
        // `n` virtual `step` calls. `saturating_add` matches `step`'s overflow guard, and
        // `n` (a row count) is clamped to `i64::MAX` on the unreachable >9.2e18 overflow.
        let counts = match args.first() {
            None => true,
            Some(v) => !v.is_null(),
        };
        if counts {
            self.n = self.n.saturating_add(i64::try_from(n).unwrap_or(i64::MAX));
        }
        Ok(())
    }

    fn finalize(&mut self, _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(Value::Integer(self.n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agg::testutil::{drive1, drive_rows, NullCtx};

    fn int(v: Result<Value>) -> i64 {
        match v.expect("count never errors") {
            Value::Integer(i) => i,
            other => panic!("expected Integer, got {other:?}"),
        }
    }

    #[test]
    fn count_x_skips_nulls() {
        // count(X): only non-NULL values are counted.
        let vals = [Value::Integer(1), Value::Null, Value::Text("a".into()), Value::Null];
        assert_eq!(int(drive1(&Count, &vals)), 2);
    }

    #[test]
    fn count_star_counts_every_row() {
        // count(*): the zero-argument form counts every row, NULLs included.
        let rows = vec![vec![], vec![], vec![]];
        assert_eq!(int(drive_rows(&Count, &rows)), 3);
    }

    #[test]
    fn count_of_no_rows_is_zero_not_null() {
        assert_eq!(int(drive1(&Count, &[])), 0);
        assert_eq!(int(drive_rows(&Count, &[])), 0);
    }

    #[test]
    fn count_x_of_all_nulls_is_zero() {
        let vals = [Value::Null, Value::Null];
        assert_eq!(int(drive1(&Count, &vals)), 0);
    }

    #[test]
    fn count_x_includes_every_non_null_storage_class() {
        // Zero, empty string, and empty blob are all non-NULL, so all are counted.
        let vals = [
            Value::Integer(0),
            Value::Real(0.0),
            Value::Text(String::new()),
            Value::Blob(Vec::new()),
        ];
        assert_eq!(int(drive1(&Count, &vals)), 4);
    }

    #[test]
    fn step_many_is_row_count_marker() {
        // Only `count` opts into the operator's row-count fast path.
        assert!(Count.is_row_count());
    }

    #[test]
    fn step_many_count_star_adds_n() {
        // count(*): step_many(n, &[]) adds exactly n — the closed form the fast path uses.
        let mut ctx = NullCtx;
        let mut acc = Count.new_accumulator(Collation::Binary);
        acc.step_many(1000, &[], &mut ctx).unwrap();
        assert_eq!(int(acc.finalize(&mut ctx)), 1000);
    }

    #[test]
    fn step_many_count_x_respects_null_rule() {
        // count(X): repeating a non-NULL arg adds n; repeating a NULL arg adds 0.
        let mut ctx = NullCtx;
        let mut acc = Count.new_accumulator(Collation::Binary);
        acc.step_many(5, std::slice::from_ref(&Value::Integer(7)), &mut ctx).unwrap();
        acc.step_many(3, std::slice::from_ref(&Value::Null), &mut ctx).unwrap();
        assert_eq!(int(acc.finalize(&mut ctx)), 5);
    }

    #[test]
    fn step_many_zero_is_noop() {
        let mut ctx = NullCtx;
        let mut acc = Count.new_accumulator(Collation::Binary);
        acc.step_many(0, &[], &mut ctx).unwrap();
        assert_eq!(int(acc.finalize(&mut ctx)), 0);
    }

    #[test]
    fn step_many_equals_stepwise_loop() {
        // The O(1) override must leave the accumulator in the same state as N `step`s.
        let mut ctx = NullCtx;
        let mut fast = Count.new_accumulator(Collation::Binary);
        fast.step_many(250, &[], &mut ctx).unwrap();
        let mut slow = Count.new_accumulator(Collation::Binary);
        for _ in 0..250 {
            slow.step(&[], &mut ctx).unwrap();
        }
        assert_eq!(int(fast.finalize(&mut ctx)), int(slow.finalize(&mut ctx)));
    }
}
