//! Aggregate `min(X)` and `max(X)` (`lang_aggfunc.html`).
//!
//! These are the single-argument AGGREGATEs: `min(X)`/`max(X)` return the
//! minimum/maximum non-NULL value of a group, where "minimum"/"maximum" is the
//! first/last value under an `ORDER BY` of the column — i.e. SQLite's storage-class
//! total order. Extremum selection goes through the shared `extremum_wins` predicate
//! (`minisqlite-types`), the one source of truth the single-min/max bare-column capture
//! also uses. NULLs are skipped, and the result is NULL only when the group has no
//! non-NULL value. The winning `Value` is
//! returned verbatim, preserving its storage class (so `max(2, 2.0)` keeps whichever
//! it first saw, exactly as sqlite does).
//!
//! The multi-argument `min(...)`/`max(...)` are the *scalar* functions and live in
//! `scalar/math.rs`, a separate namespace — this file owns only the aggregates.

use std::sync::Arc;

use minisqlite_expr::{AggregateAccumulator, AggregateFunction, FnContext};
use minisqlite_types::{extremum_wins, Collation, Result, Value};

use crate::registry::{Arity, FunctionRegistry};

/// Register the aggregate `min` and `max` (single-argument).
pub(crate) fn register(reg: &mut FunctionRegistry) {
    reg.add_aggregate("min", Arity::Exact(1), Arc::new(Min));
    reg.add_aggregate("max", Arity::Exact(1), Arc::new(Max));
}

/// `min(X)` aggregate factory.
#[derive(Debug)]
struct Min;
impl AggregateFunction for Min {
    fn new_accumulator(&self, collation: Collation) -> Box<dyn AggregateAccumulator> {
        Box::new(MinMaxAcc { best: None, is_max: false, collation })
    }
}

/// `max(X)` aggregate factory.
#[derive(Debug)]
struct Max;
impl AggregateFunction for Max {
    fn new_accumulator(&self, collation: Collation) -> Box<dyn AggregateAccumulator> {
        Box::new(MinMaxAcc { best: None, is_max: true, collation })
    }
}

/// Running best value for one group. `best` is `None` until the first non-NULL
/// input; `is_max` selects the direction; `collation` is the argument's §7.1 collating
/// sequence, applied to TEXT comparisons (numeric/BLOB/cross-class order is unaffected).
struct MinMaxAcc {
    best: Option<Value>,
    is_max: bool,
    collation: Collation,
}

impl AggregateAccumulator for MinMaxAcc {
    fn step(&mut self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<()> {
        let v = &args[0];
        if v.is_null() {
            return Ok(()); // NULLs never win and never seed `best`.
        }
        // Replace only on a strict win (via the shared `extremum_wins`), so ties keep
        // the first-seen value — matching sqlite, which preserves the earlier value's
        // storage class on an equal compare.
        //
        // Compares under `self.collation`: min/max return the value that sorts
        // first/last in an ORDER BY of the argument, and ORDER BY uses the §7.1 collation
        // (datatype3.html §7.1, lang_aggfunc.html), so a NOCASE/RTRIM column folds text
        // rather than comparing bytewise. Extremum selection is SHARED with the
        // single-min/max bare-column capture (`minisqlite-exec` `ops/aggregate.rs`)
        // through `extremum_wins`; that site keys by the SAME `arg_collations[0]`, so the
        // captured bare column always comes from the row this accumulator reports.
        let replace = match &self.best {
            None => true,
            Some(cur) => extremum_wins(cur, v, self.is_max, self.collation),
        };
        if replace {
            self.best = Some(v.clone());
        }
        Ok(())
    }

    fn finalize(&mut self, _ctx: &mut dyn FnContext) -> Result<Value> {
        // Clone rather than take: `finalize` must be re-callable for window frames.
        Ok(self.best.clone().unwrap_or(Value::Null))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agg::testutil::{drive1, drive1_collated};

    fn ok(v: Result<Value>) -> Value {
        v.expect("min/max never error")
    }

    #[test]
    fn min_and_max_of_integers() {
        let vals = [Value::Integer(3), Value::Integer(1), Value::Integer(2)];
        assert!(matches!(ok(drive1(&Min, &vals)), Value::Integer(1)));
        assert!(matches!(ok(drive1(&Max, &vals)), Value::Integer(3)));
    }

    #[test]
    fn nulls_are_skipped() {
        let vals = [Value::Null, Value::Integer(5), Value::Null, Value::Integer(2)];
        assert!(matches!(ok(drive1(&Min, &vals)), Value::Integer(2)));
        assert!(matches!(ok(drive1(&Max, &vals)), Value::Integer(5)));
    }

    #[test]
    fn empty_and_all_null_groups_are_null() {
        assert!(matches!(ok(drive1(&Min, &[])), Value::Null));
        assert!(matches!(ok(drive1(&Max, &[])), Value::Null));
        assert!(matches!(ok(drive1(&Min, &[Value::Null, Value::Null])), Value::Null));
        assert!(matches!(ok(drive1(&Max, &[Value::Null, Value::Null])), Value::Null));
    }

    #[test]
    fn storage_class_order_null_numeric_text_blob() {
        // Total order across classes: numeric < text < blob (NULL excluded).
        let vals = [
            Value::Blob(vec![0]),
            Value::Text("a".into()),
            Value::Integer(9999),
            Value::Null,
        ];
        // Minimum is the number (lowest non-NULL class), maximum is the blob.
        assert!(matches!(ok(drive1(&Min, &vals)), Value::Integer(9999)));
        assert!(matches!(ok(drive1(&Max, &vals)), Value::Blob(b) if b == vec![0]));
    }

    #[test]
    fn text_compares_bytewise_under_binary() {
        let vals = [Value::Text("banana".into()), Value::Text("Apple".into())];
        // Uppercase 'A' (0x41) sorts before lowercase 'b' (0x62) under BINARY.
        assert!(matches!(ok(drive1(&Min, &vals)), Value::Text(s) if s == "Apple"));
        assert!(matches!(ok(drive1(&Max, &vals)), Value::Text(s) if s == "banana"));
    }

    #[test]
    fn text_folds_case_under_nocase() {
        // Under NOCASE 'a'=='A' and 'b'>'a', so 'Apple' (folded 'apple') is the MIN and
        // 'banana' the MAX — but the WINNING VALUE is returned verbatim (original case).
        // Contrast `text_compares_bytewise_under_binary`: here the order is by folded
        // text, which for this pair happens to agree; the discriminating case is below.
        let vals = [Value::Text("banana".into()), Value::Text("Apple".into())];
        assert!(matches!(ok(drive1_collated(&Min, Collation::NoCase, &vals)), Value::Text(s) if s == "Apple"));
        assert!(matches!(ok(drive1_collated(&Max, Collation::NoCase, &vals)), Value::Text(s) if s == "banana"));

        // Discriminator: 'B' vs 'a'. BINARY: 'B'(0x42) < 'a'(0x61) so min='B'. NOCASE:
        // 'a'<'b' so min='a', max='B'. This is the witness case.
        let ba = [Value::Text("B".into()), Value::Text("a".into())];
        assert!(matches!(ok(drive1_collated(&Min, Collation::NoCase, &ba)), Value::Text(s) if s == "a"));
        assert!(matches!(ok(drive1_collated(&Max, Collation::NoCase, &ba)), Value::Text(s) if s == "B"));
        // BINARY control on the same pair: min='B', max='a' (the buggy-before answer).
        assert!(matches!(ok(drive1(&Min, &ba)), Value::Text(s) if s == "B"));
        assert!(matches!(ok(drive1(&Max, &ba)), Value::Text(s) if s == "a"));
    }

    #[test]
    fn text_folds_trailing_space_under_rtrim() {
        // Under RTRIM 'a  ' == 'a', a tie, so the FIRST-seen value wins and is returned
        // verbatim (trailing spaces preserved). min and max both keep 'a  ' (seen first).
        let vals = [Value::Text("a  ".into()), Value::Text("a".into())];
        assert!(matches!(ok(drive1_collated(&Min, Collation::Rtrim, &vals)), Value::Text(s) if s == "a  "));
        assert!(matches!(ok(drive1_collated(&Max, Collation::Rtrim, &vals)), Value::Text(s) if s == "a  "));
    }

    #[test]
    fn numeric_mix_compares_by_value_and_preserves_first_on_ties() {
        // Integer/Real compare by mathematical value.
        let vals = [Value::Integer(3), Value::Real(2.5), Value::Integer(1)];
        assert!(matches!(ok(drive1(&Min, &vals)), Value::Integer(1)));
        assert!(matches!(ok(drive1(&Max, &vals)), Value::Integer(3)));
        // A tie (2 == 2.0) keeps the first-seen storage class.
        let tie = [Value::Integer(2), Value::Real(2.0)];
        assert!(matches!(ok(drive1(&Max, &tie)), Value::Integer(2)));
        assert!(matches!(ok(drive1(&Min, &tie)), Value::Integer(2)));
        let tie_rev = [Value::Real(2.0), Value::Integer(2)];
        assert!(matches!(ok(drive1(&Max, &tie_rev)), Value::Real(r) if r == 2.0));
    }

    #[test]
    fn single_value_is_returned_verbatim() {
        assert!(matches!(ok(drive1(&Min, &[Value::Real(1.25)])), Value::Real(r) if r == 1.25));
        assert!(matches!(ok(drive1(&Max, &[Value::Text("x".into())])), Value::Text(s) if s == "x"));
    }
}
