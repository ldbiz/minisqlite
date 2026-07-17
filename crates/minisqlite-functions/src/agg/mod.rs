//! Built-in aggregate functions (`spec/sqlite-doc/lang_aggfunc.html`):
//! `count`, `sum`/`total`/`avg`, aggregate `min`/`max`, and
//! `group_concat`/`string_agg`.
//!
//! Each family lives in its own submodule (declared here, which is the only file
//! the crate root wires into `register`) so a family lands in its own cell. This
//! hub only declares the submodules and fans out registration.
//!
//! An accumulator sees only the values it must fold: the aggregation operator (in
//! `minisqlite-exec`) applies `DISTINCT`, `FILTER (WHERE …)`, and aggregate
//! `ORDER BY` *before* the values reach [`step`](minisqlite_expr::AggregateAccumulator::step),
//! so none of that logic belongs here (see the doc comment on `AggregateCall` in
//! `minisqlite-expr`). `finalize` takes `&mut self` and must not consume state, so
//! a window frame can read a running value repeatedly; every accumulator here
//! clones its output rather than moving out of the group state.

mod concat;
mod count;
mod minmax;
mod sum;

use crate::FunctionRegistry;

/// Register every aggregate family into `reg`. Adding a family is a one-line change
/// here plus its own submodule.
pub(crate) fn register(reg: &mut FunctionRegistry) {
    count::register(reg);
    sum::register(reg);
    minmax::register(reg);
    concat::register(reg);
}

#[cfg(test)]
pub(crate) mod testutil {
    //! Shared test scaffolding for the aggregate submodules: a do-nothing
    //! [`FnContext`] (aggregates here never touch the clock/RNG/counters) and two
    //! drivers that feed a sequence of rows through a fresh accumulator and
    //! finalize it, mirroring how the executor drives one group.

    use minisqlite_expr::{AggregateFunction, FnContext};
    use minisqlite_types::{Collation, Result, Value};

    /// A [`FnContext`] whose every capability is a fixed constant. The aggregate
    /// accumulators never call these, so the values are arbitrary; the stub only
    /// exists to satisfy the `&mut dyn FnContext` parameter.
    pub(crate) struct NullCtx;
    impl FnContext for NullCtx {
        fn now_unix_millis(&self) -> i64 {
            0
        }
        fn random_i64(&mut self) -> i64 {
            0
        }
        fn fill_random(&mut self, _buf: &mut [u8]) {}
        fn last_insert_rowid(&self) -> i64 {
            0
        }
        fn changes(&self) -> i64 {
            0
        }
        fn total_changes(&self) -> i64 {
            0
        }
    }

    /// Drive `func` over a group of single-argument rows and finalize, under BINARY.
    /// Convenience for the many aggregates that take exactly one argument and are not
    /// collation-sensitive; use [`drive1_collated`] to exercise min/max under a collation.
    pub(crate) fn drive1(func: &dyn AggregateFunction, vals: &[Value]) -> Result<Value> {
        drive1_collated(func, Collation::Binary, vals)
    }

    /// Like [`drive1`] but with an explicit argument collation, so an order-sensitive
    /// aggregate (min/max) can be driven under NOCASE/RTRIM.
    pub(crate) fn drive1_collated(
        func: &dyn AggregateFunction,
        collation: Collation,
        vals: &[Value],
    ) -> Result<Value> {
        let mut ctx = NullCtx;
        let mut acc = func.new_accumulator(collation);
        for v in vals {
            acc.step(std::slice::from_ref(v), &mut ctx)?;
        }
        acc.finalize(&mut ctx)
    }

    /// Drive `func` over a group of multi-argument rows and finalize under BINARY (e.g.
    /// `group_concat(X, SEP)`, where each row carries its own separator).
    pub(crate) fn drive_rows(func: &dyn AggregateFunction, rows: &[Vec<Value>]) -> Result<Value> {
        let mut ctx = NullCtx;
        let mut acc = func.new_accumulator(Collation::Binary);
        for row in rows {
            acc.step(row, &mut ctx)?;
        }
        acc.finalize(&mut ctx)
    }
}

#[cfg(test)]
mod tests {
    use crate::FunctionRegistry;

    /// The families register under the aggregate namespace and are classified as
    /// aggregates, and the scalar/aggregate namespaces stay separate (an aggregate
    /// name does not resolve as a scalar).
    #[test]
    fn aggregates_register_and_classify() {
        let reg = FunctionRegistry::builtins();

        // Every implemented aggregate resolves at a representative arity.
        for (name, argc) in [
            ("count", 0usize),
            ("count", 1),
            ("sum", 1),
            ("total", 1),
            ("avg", 1),
            ("min", 1),
            ("max", 1),
            ("group_concat", 1),
            ("group_concat", 2),
            ("string_agg", 2),
        ] {
            assert!(reg.resolve_aggregate(name, argc).is_ok(), "{name}/{argc} should resolve");
            assert!(reg.is_aggregate(name), "{name} should classify as an aggregate");
        }

        // Case-insensitive, like every other function name.
        assert!(reg.resolve_aggregate("COUNT", 0).is_ok());
        assert!(reg.is_aggregate("SUM"));

        // Namespaces are separate: an aggregate name is unknown as a scalar.
        assert!(reg.resolve_scalar("sum", 1).is_err());
        assert!(reg.resolve_scalar("group_concat", 1).is_err());
    }

    /// The single-argument `min(X)`/`max(X)` are aggregates and classify as such.
    /// (The multi-argument scalar `min`/`max` live in the separate scalar namespace,
    /// `scalar/math.rs`, so aggregate resolution here is independent of them.)
    #[test]
    fn min_max_are_aggregates_in_this_namespace() {
        let reg = FunctionRegistry::builtins();
        assert!(reg.resolve_aggregate("min", 1).is_ok());
        assert!(reg.resolve_aggregate("max", 1).is_ok());
        assert!(reg.is_aggregate("min"));
        assert!(reg.is_aggregate("max"));
    }

    /// Wrong argument counts are rejected with the registry's wrong-arity error,
    /// pinning the registered arities (`count` 0..=1, single-arg aggregates, and
    /// the 1..=2 / exactly-2 concat forms).
    #[test]
    fn registered_arities_reject_bad_argc() {
        let reg = FunctionRegistry::builtins();
        assert!(reg.resolve_aggregate("count", 2).is_err()); // Range(0,1)
        assert!(reg.resolve_aggregate("sum", 0).is_err()); // Exact(1)
        assert!(reg.resolve_aggregate("sum", 2).is_err());
        assert!(reg.resolve_aggregate("group_concat", 0).is_err()); // Range(1,2)
        assert!(reg.resolve_aggregate("group_concat", 3).is_err());
        assert!(reg.resolve_aggregate("string_agg", 1).is_err()); // Exact(2)
        assert!(reg.resolve_aggregate("string_agg", 3).is_err());
    }

    /// Each aggregate name resolves through the registry to the CORRECT
    /// implementation: driving the resolved `Arc<dyn AggregateFunction>` yields the
    /// expected result. The per-family tests drive the concrete structs directly and
    /// the arity checks only assert `is_ok`, so neither would catch a mis-wire in
    /// [`register`] (e.g. `"sum"` bound to `Total`). This pins name→impl over the
    /// exact path the planner's binder takes: `resolve_aggregate` → `new_accumulator`
    /// → `step` → `finalize`.
    #[test]
    fn resolved_aggregates_produce_correct_results() {
        use crate::agg::testutil::{drive1, drive_rows};
        use minisqlite_types::Value;

        fn as_int(v: Value) -> i64 {
            match v {
                Value::Integer(i) => i,
                other => panic!("expected Integer, got {other:?}"),
            }
        }
        fn as_real(v: Value) -> f64 {
            match v {
                Value::Real(r) => r,
                other => panic!("expected Real, got {other:?}"),
            }
        }
        fn as_text(v: Value) -> String {
            match v {
                Value::Text(s) => s,
                other => panic!("expected Text, got {other:?}"),
            }
        }

        let reg = FunctionRegistry::builtins();

        // Resolve a single-argument aggregate by name and drive it end to end.
        let drive1_by = |name: &str, vals: &[Value]| -> Value {
            let f = reg.resolve_aggregate(name, 1).unwrap_or_else(|_| panic!("{name}/1 should resolve"));
            drive1(f.as_ref(), vals).unwrap_or_else(|e| panic!("{name} finalize: {e:?}"))
        };

        // count(X) counts non-NULLs (the 0-arg count(*) form is driven below).
        assert_eq!(as_int(drive1_by("count", &[Value::Integer(1), Value::Null])), 1);
        // sum of integers stays INTEGER — pins "sum" → Sum (not Total/Avg).
        assert_eq!(as_int(drive1_by("sum", &[Value::Integer(1), Value::Integer(2), Value::Integer(3)])), 6);
        // total is always REAL and 0.0 over an empty group — pins "total" → Total.
        assert_eq!(as_real(drive1_by("total", &[])), 0.0);
        // avg is REAL — pins "avg" → Avg.
        let avg_in = [Value::Integer(1), Value::Integer(2), Value::Integer(3), Value::Integer(4)];
        assert_eq!(as_real(drive1_by("avg", &avg_in)), 2.5);
        // min/max pick the extremes — pins "min" → Min and "max" → Max (not swapped).
        let nums = [Value::Integer(3), Value::Integer(1), Value::Integer(2)];
        assert_eq!(as_int(drive1_by("min", &nums)), 1);
        assert_eq!(as_int(drive1_by("max", &nums)), 3);
        // group_concat defaults to a comma separator — pins "group_concat".
        let cc = [Value::Text("a".into()), Value::Text("b".into())];
        assert_eq!(as_text(drive1_by("group_concat", &cc)), "a,b");

        // 0-argument count(*) counts every row.
        let count_star = reg.resolve_aggregate("count", 0).expect("count/0 should resolve");
        let star_rows: [Vec<Value>; 2] = [Vec::new(), Vec::new()];
        assert_eq!(as_int(drive_rows(count_star.as_ref(), &star_rows).expect("count(*) finalize")), 2);

        // string_agg(X, SEP): the 2-arg form, pinned to the concat impl.
        let string_agg = reg.resolve_aggregate("string_agg", 2).expect("string_agg/2 should resolve");
        let sa_rows = [
            vec![Value::Text("a".into()), Value::Text("; ".into())],
            vec![Value::Text("b".into()), Value::Text("; ".into())],
        ];
        assert_eq!(as_text(drive_rows(string_agg.as_ref(), &sa_rows).expect("string_agg finalize")), "a; b");
    }
}
