//! `sum(X)`, `total(X)`, and `avg(X)` (`lang_aggfunc.html`).
//!
//! All three fold the non-NULL inputs of a group into a running numeric total, so
//! they share one accumulator ([`NumericAcc`]) and differ only in how they
//! finalize. The subtle parts are `sum`'s result *type*, its
//! overflow rule, and the floating-point summation semantics:
//!
//! * `sum` — INTEGER when every non-NULL input is an integer and the exact i64 sum
//!   never overflows; REAL as soon as any input is non-integer; NULL over zero
//!   non-NULL rows; an `"integer overflow"` error only on the all-integer overflow
//!   path (a REAL input anywhere suppresses it — see rule 1).
//! * `total` — always REAL, `0.0` over zero rows, and never overflows.
//! * `avg` — `sum / count()` as REAL, NULL over zero non-NULL rows.
//!
//! Three rules the spec pins that a naive `rSum += x` would get wrong:
//!
//! 1. **A REAL input anywhere suppresses the overflow error.** The spec throws
//!    "integer overflow" only "if all inputs are integers or NULL and an integer
//!    overflow occurs"; a single REAL input makes the group non-all-integer, so `sum`
//!    returns the (compensated) floating-point sum with no error — real sqlite returns
//!    REAL for e.g. `[i64::MAX, i64::MAX, 1.0]` in *any* order. `finalize` therefore
//!    checks `approx` *before* `ovrfl`, and this is order-independent even though scan
//!    order is arbitrary: `step` latches `ovrfl` only while `!approx`, so a float
//!    before the overflow prevents the latch and a float after it is caught by the
//!    `approx`-first check — both orderings of the same group agree. The error fires
//!    only on the all-integer path (`[i64::MAX, i64::MAX]`, or a transient overflow
//!    like `[i64::MAX, 1, -i64::MAX]`). (The spec's "no error if any *prior* input was
//!    a floating point value" is imprecise wording for this "any REAL suppresses"
//!    rule, confirmed against the reference: a *trailing* float suppresses too.)
//! 2. **NaN → NULL** (`lang_aggfunc.html`: when `±Inf` of differing signs cancel so
//!    the result is indeterminate, e.g. `sum` of `(1.0),(-9e+999),(2.0),(+9e+999),(3.0)`,
//!    the answer is NULL). SQLite cannot store NaN, so any NaN float result from
//!    sum/total/avg becomes NULL, never `Real(NaN)`. See [`real_or_null`].
//! 3. **Precision** (`lang_aggfunc.html`: the float sum uses compensated summation).
//!    The float running sum carries a Kahan-Babushka-Neumaier error term (see
//!    [`neumaier_add`]) so small addends are not lost under a large magnitude, and
//!    `total`/`avg` over an all-integer group use the *exact* i64 sum cast once
//!    rather than the lossy float sum (an integer beyond 2^53 is exact in i64 but not
//!    as the running `f64`). See [`NumericAcc::real_result`].
//!
//! Input typing mirrors `sqlite3_value_numeric_type` + `sqlite3_value_double`: an
//! input counts toward the integer path only when its *storage class after numeric
//! affinity* is INTEGER — i.e. an actual INTEGER, or TEXT that is a whole integer
//! literal fitting i64. A REAL, an over-wide integer literal, non-/partly-numeric
//! text, or *any* BLOB routes through the floating-point path (its `double` value
//! is the longest numeric prefix), which forces a REAL result for `sum`. See
//! [`classify`] for the exact rule.

use std::sync::Arc;

use minisqlite_expr::{AggregateAccumulator, AggregateFunction, FnContext};
use minisqlite_types::{
    looks_like_integer, parse_real_prefix, text_to_numeric, Collation, Error, Result, Value,
};

use crate::registry::{Arity, FunctionRegistry};

/// Register `sum`, `total`, and `avg`.
pub(crate) fn register(reg: &mut FunctionRegistry) {
    reg.add_aggregate("sum", Arity::Exact(1), Arc::new(Sum));
    reg.add_aggregate("total", Arity::Exact(1), Arc::new(Total));
    reg.add_aggregate("avg", Arity::Exact(1), Arc::new(Avg));
}

/// Which of the three functions a shared [`NumericAcc`] is finalizing as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Sum,
    Total,
    Avg,
}

/// `sum(X)` factory.
#[derive(Debug)]
struct Sum;
impl AggregateFunction for Sum {
    fn new_accumulator(&self, _collation: Collation) -> Box<dyn AggregateAccumulator> {
        Box::new(NumericAcc::new(Mode::Sum))
    }
}

/// `total(X)` factory.
#[derive(Debug)]
struct Total;
impl AggregateFunction for Total {
    fn new_accumulator(&self, _collation: Collation) -> Box<dyn AggregateAccumulator> {
        Box::new(NumericAcc::new(Mode::Total))
    }
}

/// `avg(X)` factory.
#[derive(Debug)]
struct Avg;
impl AggregateFunction for Avg {
    fn new_accumulator(&self, _collation: Collation) -> Box<dyn AggregateAccumulator> {
        Box::new(NumericAcc::new(Mode::Avg))
    }
}

/// A non-NULL input after numeric classification: either it participates in the
/// exact integer sum, or it only contributes to the floating-point sum.
enum Numeric {
    Int(i64),
    Real(f64),
}

/// Classify a non-NULL value the way `sum`/`total`/`avg` see it. Returns `None` for
/// NULL (which is skipped entirely — it does not even count toward `avg`'s divisor).
///
/// The integer path is taken only for a value whose storage class after numeric
/// affinity is INTEGER: an actual INTEGER, or TEXT that is a whole integer literal
/// fitting i64. Everything else (REAL, an integer literal too wide for i64, text
/// that is only partly/nonnumeric, and *every* BLOB) takes the floating-point path
/// with the value's `double` reading — the longest numeric prefix, `0.0` when there
/// is none. A BLOB is never affinity-converted (numeric affinity applies to TEXT
/// only), so `sum` of a blob is always REAL even for a blob like `x'35'` ("5").
fn classify(v: &Value) -> Option<Numeric> {
    match v {
        Value::Null => None,
        Value::Integer(i) => Some(Numeric::Int(*i)),
        Value::Real(r) => Some(Numeric::Real(*r)),
        Value::Text(s) => Some(classify_text(s)),
        Value::Blob(b) => Some(Numeric::Real(parse_real_prefix(&String::from_utf8_lossy(b)))),
    }
}

/// Numeric classification of a TEXT value (see [`classify`]). A whole integer
/// literal that fits i64 is INTEGER; anything else is REAL via the longest-prefix
/// `double` reading. `text_to_numeric` returns `Integer` exactly for an in-range
/// whole integer literal here (we only reach it when `looks_like_integer` holds, so
/// there is no `.`/`e` for it to demote), and `Real` for an over-wide one.
fn classify_text(s: &str) -> Numeric {
    if looks_like_integer(s)
        && let Value::Integer(i) = text_to_numeric(s)
    {
        return Numeric::Int(i);
    }
    Numeric::Real(parse_real_prefix(s))
}

/// One Kahan-Babushka-Neumaier compensated addition: fold `x` into the running
/// `(sum, err)` pair and return the updated pair; the corrected total is `sum + err`.
/// Naive `sum += x` drops the low-order bits of a small `x` under a large `sum`
/// (the divergence the spec's floating-point-summation note warns about); tracking
/// the rounding error in `err` recovers them. `+inf + -inf` still yields NaN, which
/// [`real_or_null`] maps to NULL, matching the spec's Inf-cancellation rule.
fn neumaier_add(sum: f64, err: f64, x: f64) -> (f64, f64) {
    let t = sum + x;
    let err = if sum.abs() >= x.abs() {
        err + ((sum - t) + x)
    } else {
        err + ((x - t) + sum)
    };
    (t, err)
}

/// Wrap a finalized aggregate double as a `Value`, mapping NaN to NULL. SQLite
/// cannot store NaN, so an indeterminate result (e.g. `+inf + -inf`) is reported as
/// NULL, never `Real(NaN)` (`lang_aggfunc.html`).
fn real_or_null(v: f64) -> Value {
    if v.is_nan() {
        Value::Null
    } else {
        Value::Real(v)
    }
}

/// The shared running state for `sum`/`total`/`avg` over one group.
///
/// It keeps *both* an exact integer sum and a compensated floating-point sum: the
/// integer sum is reported by `sum` (and used by `total`/`avg`) while every input has
/// been an integer with no overflow, and the float sum carries every input's `double`
/// value for the mixed/overflow/`total`/`avg` cases. `ovrfl` records an i64 overflow
/// that happened while still on the pure-integer path.
struct NumericAcc {
    mode: Mode,
    /// Exact i64 running sum; meaningful only while `!approx && !ovrfl`.
    i_sum: i64,
    /// Compensated (Neumaier) running float sum: the total is `r_sum + r_err`. Holds
    /// every input's `double` value; what `total` returns and `avg` divides, and what
    /// `sum` returns once `approx`.
    r_sum: f64,
    /// Neumaier error term for `r_sum` (see [`neumaier_add`]).
    r_err: f64,
    /// Count of non-NULL inputs (`avg`'s divisor; distinguishes empty from summed).
    count: i64,
    /// A non-integer input has been seen, so `sum` must return REAL.
    approx: bool,
    /// An i64 overflow occurred on the pure-integer path.
    ovrfl: bool,
}

impl NumericAcc {
    fn new(mode: Mode) -> Self {
        NumericAcc { mode, i_sum: 0, r_sum: 0.0, r_err: 0.0, count: 0, approx: false, ovrfl: false }
    }

    /// The running total as a float, for `total`/`avg` (and `sum`'s REAL branch).
    /// On the pure-integer path (`!approx && !ovrfl`) this is the *exact* i64 sum cast
    /// once — an all-integer `total`/`avg` keeps full precision even past 2^53, which
    /// the running `f64` sum would have lost. Otherwise it is the compensated float
    /// sum `r_sum + r_err`. May be NaN (Inf cancellation); callers map NaN to NULL.
    fn real_result(&self) -> f64 {
        if !self.approx && !self.ovrfl {
            self.i_sum as f64
        } else {
            self.r_sum + self.r_err
        }
    }
}

impl AggregateAccumulator for NumericAcc {
    fn step(&mut self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<()> {
        let Some(num) = classify(&args[0]) else {
            return Ok(()); // NULL: skipped, does not count.
        };
        self.count = self.count.saturating_add(1);
        match num {
            Numeric::Int(iv) => {
                // The compensated float sum always accumulates (it is `total`/`avg`'s
                // answer and `sum`'s answer once approximate). Attempt the exact i64
                // add only while still on the pure-integer path; a first overflow there
                // sets `ovrfl` and stops further exact adds. Crucially the exact add is
                // skipped once `approx` holds, so a float BEFORE any overflow prevents
                // an overflow from ever latching (the spec's "prior" rule).
                (self.r_sum, self.r_err) = neumaier_add(self.r_sum, self.r_err, iv as f64);
                if !self.approx && !self.ovrfl {
                    match self.i_sum.checked_add(iv) {
                        Some(s) => self.i_sum = s,
                        None => self.ovrfl = true,
                    }
                }
            }
            Numeric::Real(rv) => {
                (self.r_sum, self.r_err) = neumaier_add(self.r_sum, self.r_err, rv);
                self.approx = true;
            }
        }
        Ok(())
    }

    fn finalize(&mut self, _ctx: &mut dyn FnContext) -> Result<Value> {
        match self.mode {
            // `total` is always REAL, 0.0 over zero rows (i_sum starts at 0), never
            // raises overflow; a NaN (Inf cancellation) becomes NULL.
            Mode::Total => Ok(real_or_null(self.real_result())),
            // `avg` = total/count as REAL, NULL over zero non-NULL rows; NaN -> NULL.
            Mode::Avg => {
                if self.count == 0 {
                    Ok(Value::Null)
                } else {
                    Ok(real_or_null(self.real_result() / self.count as f64))
                }
            }
            // `sum`: NULL over zero non-NULL rows; then `approx` is checked BEFORE
            // `ovrfl`, because a REAL input anywhere makes the group non-all-integer and
            // the spec throws "integer overflow" only for an all-integer group — so any
            // float yields the compensated REAL (NaN -> NULL) and the error fires only
            // on the pure-integer overflow path. This is order-independent: `step`
            // latches `ovrfl` only while `!approx`, so a float before the overflow
            // prevents the latch and a float after it is caught here — real sqlite
            // returns REAL for e.g. `[i64::MAX, i64::MAX, 1.0]` in either order.
            // Otherwise the exact INTEGER sum.
            Mode::Sum => {
                if self.count == 0 {
                    Ok(Value::Null)
                } else if self.approx {
                    Ok(real_or_null(self.real_result()))
                } else if self.ovrfl {
                    Err(Error::sql("integer overflow"))
                } else {
                    Ok(Value::Integer(self.i_sum))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agg::testutil::drive1;

    fn ok(v: Result<Value>) -> Value {
        v.expect("aggregate should succeed")
    }
    fn int(v: Result<Value>) -> i64 {
        match ok(v) {
            Value::Integer(i) => i,
            other => panic!("expected Integer, got {other:?}"),
        }
    }
    fn real(v: Result<Value>) -> f64 {
        match ok(v) {
            Value::Real(r) => r,
            other => panic!("expected Real, got {other:?}"),
        }
    }

    // ----- sum: result typing -------------------------------------------------

    #[test]
    fn sum_all_integers_is_integer() {
        let vals = [Value::Integer(1), Value::Integer(2), Value::Integer(3)];
        assert_eq!(int(drive1(&Sum, &vals)), 6);
    }

    #[test]
    fn sum_any_real_is_real() {
        let vals = [Value::Integer(1), Value::Real(2.5)];
        assert_eq!(real(drive1(&Sum, &vals)), 3.5);
        // Pure reals summing to a whole number still stay REAL.
        let whole = [Value::Real(1.5), Value::Real(1.5)];
        assert_eq!(real(drive1(&Sum, &whole)), 3.0);
    }

    #[test]
    fn sum_text_integer_literals_stay_integer() {
        // TEXT that is a whole integer literal is numeric-affinity INTEGER.
        let vals = [Value::Text("10".into()), Value::Text("20".into())];
        assert_eq!(int(drive1(&Sum, &vals)), 30);
    }

    #[test]
    fn sum_text_real_literal_is_real() {
        assert_eq!(real(drive1(&Sum, &[Value::Text("1.5".into())])), 1.5);
    }

    #[test]
    fn sum_non_numeric_text_contributes_zero_but_forces_real() {
        // Non-numeric text has numeric type TEXT (not INTEGER), so it takes the
        // float path (double value 0.0) and forces a REAL result.
        assert_eq!(real(drive1(&Sum, &[Value::Text("abc".into())])), 0.0);
        // Its 0.0 contribution plus an integer: REAL result, integer value added.
        let mixed = [Value::Text("abc".into()), Value::Integer(4)];
        assert_eq!(real(drive1(&Sum, &mixed)), 4.0);
        // A partly-numeric text uses its longest numeric prefix as the double.
        assert_eq!(real(drive1(&Sum, &[Value::Text("12abc".into())])), 12.0);
    }

    #[test]
    fn sum_blob_is_always_real() {
        // A blob is never affinity-converted, so sum() of even a "5" blob is REAL.
        assert_eq!(real(drive1(&Sum, &[Value::Blob(b"5".to_vec())])), 5.0);
        assert_eq!(real(drive1(&Sum, &[Value::Blob(b"1.5".to_vec())])), 1.5);
        assert_eq!(real(drive1(&Sum, &[Value::Blob(b"zzz".to_vec())])), 0.0);
    }

    #[test]
    fn sum_over_wide_integer_text_is_real() {
        // A whole integer literal too wide for i64 demotes to REAL under affinity.
        let r = real(drive1(&Sum, &[Value::Text("99999999999999999999".into())]));
        assert!(r > 9e19, "expected ~1e20, got {r}");
    }

    // ----- sum: NULL / empty / overflow --------------------------------------

    #[test]
    fn sum_of_no_rows_is_null() {
        assert!(matches!(ok(drive1(&Sum, &[])), Value::Null));
    }

    #[test]
    fn sum_of_all_nulls_is_null() {
        assert!(matches!(ok(drive1(&Sum, &[Value::Null, Value::Null])), Value::Null));
    }

    #[test]
    fn sum_skips_nulls_between_values() {
        let vals = [Value::Integer(5), Value::Null, Value::Integer(7)];
        assert_eq!(int(drive1(&Sum, &vals)), 12);
    }

    #[test]
    fn sum_integer_overflow_errors() {
        let vals = [Value::Integer(i64::MAX), Value::Integer(1)];
        match drive1(&Sum, &vals) {
            Err(Error::Sql(m)) => assert_eq!(m, "integer overflow"),
            other => panic!("expected integer overflow error, got {other:?}"),
        }
        // Negative overflow too.
        let neg = [Value::Integer(i64::MIN), Value::Integer(-1)];
        assert!(matches!(drive1(&Sum, &neg), Err(Error::Sql(_))));
    }

    #[test]
    fn sum_float_anywhere_suppresses_overflow_in_any_order() {
        // A REAL input makes the group non-all-integer, so no "integer overflow" is
        // thrown regardless of where the float falls. Aggregation order is arbitrary
        // (no ORDER BY), so the result must be order-independent: real sqlite3 3.46.1
        // returns REAL 1.844674407370955161e+19 for this multiset in BOTH orderings.
        // Both give the same compensated sum (the +1.0 is lost below the ULP at ~2^64).
        // NB: this is the *reference-observed* behavior; an earlier revision read the
        // spec's "prior" wording literally and errored on the trailing-float case,
        // which diverged from sqlite and made the result order-dependent.
        let expect = (i64::MAX as f64) * 2.0;
        let after = [Value::Integer(i64::MAX), Value::Integer(i64::MAX), Value::Real(1.0)];
        let before = [Value::Real(1.0), Value::Integer(i64::MAX), Value::Integer(i64::MAX)];
        assert_eq!(real(drive1(&Sum, &after)), expect);
        assert_eq!(real(drive1(&Sum, &before)), expect);
    }

    #[test]
    fn sum_at_i64_bounds_does_not_falsely_overflow() {
        // Summing exactly to i64::MAX is fine; it is one more that overflows.
        let vals = [Value::Integer(i64::MAX - 1), Value::Integer(1)];
        assert_eq!(int(drive1(&Sum, &vals)), i64::MAX);
    }

    // ----- sum/total/avg: Inf cancellation -> NULL ---------------------------

    #[test]
    fn sum_total_avg_inf_cancellation_is_null() {
        // lang_aggfunc.html: (1.0),(-9e+999),(2.0),(+9e+999),(3.0) -> NULL. The two
        // infinities of opposite sign cancel to NaN, which SQLite reports as NULL
        // (it cannot store NaN) for sum, total, AND avg.
        let vals = [
            Value::Real(1.0),
            Value::Real(f64::NEG_INFINITY),
            Value::Real(2.0),
            Value::Real(f64::INFINITY),
            Value::Real(3.0),
        ];
        assert!(matches!(ok(drive1(&Sum, &vals)), Value::Null));
        assert!(matches!(ok(drive1(&Total, &vals)), Value::Null));
        assert!(matches!(ok(drive1(&Avg, &vals)), Value::Null));
    }

    // ----- sum/total/avg: precision ------------------------------------------

    #[test]
    fn total_and_avg_keep_full_integer_precision() {
        // 2^53+1 is not representable as f64 (rounds to 2^53), so a naive running
        // float sum of [2^53+1, 1] loses the low bit. total/avg over an all-integer
        // group use the EXACT i64 sum cast once: total = 2^53+2, avg = (2^53+2)/2.
        let v = [Value::Integer(9_007_199_254_740_993), Value::Integer(1)];
        assert_eq!(real(drive1(&Total, &v)), 9_007_199_254_740_994.0);
        assert_eq!(real(drive1(&Avg, &v)), 4_503_599_627_370_497.0);
    }

    #[test]
    fn float_sum_is_compensated() {
        // With reals, the exact-integer path does not apply, so the compensated
        // (Neumaier) float sum must recover the small addends a naive sum drops:
        // 2^53 + 1.0 + 1.0 = 2^53 + 2 under compensation (naive gives 2^53, each +1
        // rounded away). Exercises sum's REAL branch and total together.
        let big = 9_007_199_254_740_992.0_f64; // 2^53
        let vals = [Value::Real(big), Value::Real(1.0), Value::Real(1.0)];
        assert_eq!(real(drive1(&Sum, &vals)), 9_007_199_254_740_994.0);
        assert_eq!(real(drive1(&Total, &vals)), 9_007_199_254_740_994.0);
    }

    // ----- total --------------------------------------------------------------

    #[test]
    fn total_is_always_real() {
        assert_eq!(real(drive1(&Total, &[Value::Integer(1), Value::Integer(2)])), 3.0);
    }

    #[test]
    fn total_of_no_rows_is_zero_not_null() {
        assert_eq!(real(drive1(&Total, &[])), 0.0);
        assert_eq!(real(drive1(&Total, &[Value::Null])), 0.0);
    }

    #[test]
    fn total_never_overflows() {
        // The same inputs that make sum() raise overflow give total() a finite real.
        let vals = [Value::Integer(i64::MAX), Value::Integer(i64::MAX)];
        let r = real(drive1(&Total, &vals));
        assert!(r.is_finite() && r > 0.0, "expected a large finite real, got {r}");
    }

    // ----- avg ----------------------------------------------------------------

    #[test]
    fn avg_is_real_mean_of_non_nulls() {
        let vals = [Value::Integer(1), Value::Integer(2), Value::Integer(3), Value::Integer(4)];
        assert_eq!(real(drive1(&Avg, &vals)), 2.5);
    }

    #[test]
    fn avg_of_no_or_all_null_rows_is_null() {
        assert!(matches!(ok(drive1(&Avg, &[])), Value::Null));
        assert!(matches!(ok(drive1(&Avg, &[Value::Null])), Value::Null));
    }

    #[test]
    fn avg_counts_non_numeric_text_as_zero() {
        // Non-numeric text counts as a row contributing 0, so it drags the mean.
        let vals = [Value::Text("abc".into()), Value::Integer(4)];
        assert_eq!(real(drive1(&Avg, &vals)), 2.0);
    }

    #[test]
    fn avg_single_integer_is_that_value_as_real() {
        assert_eq!(real(drive1(&Avg, &[Value::Integer(7)])), 7.0);
    }
}
