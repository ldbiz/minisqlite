//! Value comparison — the storage-class ordering that ORDER BY, indexes, and the
//! comparison operators are built on (datatype3.html §4 and §6).
//!
//! The ordering across classes is fixed: NULL < numeric (INTEGER/REAL as one
//! class) < TEXT < BLOB. Within the numeric class, INTEGER and REAL compare by
//! mathematical value with *no* precision loss — the subtle part, and the reason
//! this is its own module.

use std::cmp::Ordering;

use crate::collation::{compare_text, Collation};
use crate::numeric::TWO_POW_63;
use crate::value::Value;

/// The storage-class sort rank: NULL(0) < numeric(1) < TEXT(2) < BLOB(3). INTEGER
/// and REAL share rank 1 (they are one comparison class, datatype3.html §4.1).
pub fn storage_class_rank(v: &Value) -> u8 {
    match v {
        Value::Null => 0,
        Value::Integer(_) | Value::Real(_) => 1,
        Value::Text(_) => 2,
        Value::Blob(_) => 3,
    }
}

/// The total order used by ORDER BY and index keys (datatype3.html §6). NULLs sort
/// first and compare equal to each other; different storage classes order by rank;
/// within a class, numerics compare by exact value, text by the collation, blobs by
/// `memcmp`. No storage-class conversion happens here (affinity, if any, is applied
/// by the caller *before* comparing).
///
/// `#[inline]`: this is the leaf every ORDER BY / index / operator / aggregate
/// comparison bottoms out in, called once (or more) per row on the hottest paths;
/// inlining lets the cross-crate callers avoid a call frame under the default
/// (non-LTO) release profile.
#[inline]
pub fn compare_values(a: &Value, b: &Value, c: Collation) -> Ordering {
    let (ra, rb) = (storage_class_rank(a), storage_class_rank(b));
    if ra != rb {
        return ra.cmp(&rb);
    }
    match (a, b) {
        (Value::Null, Value::Null) => Ordering::Equal,
        (Value::Text(x), Value::Text(y)) => compare_text(x, y, c),
        (Value::Blob(x), Value::Blob(y)) => x.cmp(y),
        // Both numeric (any Integer/Real mix).
        _ => compare_numeric(a, b),
    }
}

/// Operator comparison semantics (`=`, `<`, `>`, ...) where NULL is *unknown*:
/// returns `None` if either operand is NULL (so three-valued logic upstream yields
/// NULL), otherwise `Some` of the same class/numeric/text/blob ordering as
/// [`compare_values`]. The engine applies operand affinity *before* calling this.
pub fn compare_for_eq(a: &Value, b: &Value, c: Collation) -> Option<Ordering> {
    if a.is_null() || b.is_null() {
        return None;
    }
    Some(compare_values(a, b, c))
}

/// Whether candidate `cand` STRICTLY beats the current extremum `cur` for a running
/// `min`/`max` under `collation`: for `max`, `cand` must sort strictly after `cur`; for
/// `min`, strictly before. A tie (`Equal`) is NOT a win, so the first-seen extremum is
/// kept — matching SQLite's aggregate `min`/`max`, which preserves the earlier value's
/// storage class on an equal compare.
///
/// This is the ONE source of truth for min/max extremum selection: the aggregate
/// accumulator (`minisqlite-functions` `agg/minmax.rs`) and the single-min/max
/// bare-column capture (`minisqlite-exec` `ops/aggregate.rs`) both decide "does this
/// candidate advance the extremum?" through here, so they can never disagree on which
/// row is the extremum. If the comparison is later made collation-aware, thread the
/// column's collation through `collation` at every call site TOGETHER — that is the
/// whole point of sharing this predicate. NULL handling (skip) and first-value seeding
/// stay at the call sites, which own the accumulator state.
///
/// `#[inline]`: runs once per non-NULL row inside the min/max accumulator's `step` (for
/// EVERY min/max query, not only the bare-column case), so inlining collapses the call
/// frame the consolidation added on that hot path.
#[inline]
pub fn extremum_wins(cur: &Value, cand: &Value, is_max: bool, collation: Collation) -> bool {
    match compare_values(cur, cand, collation) {
        Ordering::Less => is_max,
        Ordering::Greater => !is_max,
        Ordering::Equal => false,
    }
}

/// Compare two numeric values (each Integer or Real) by mathematical value. The
/// Integer-vs-Real case is done exactly, without ever casting the `i64` to `f64`
/// (which would lose precision for magnitudes above 2^53).
fn compare_numeric(a: &Value, b: &Value) -> Ordering {
    match (a, b) {
        (Value::Integer(x), Value::Integer(y)) => x.cmp(y),
        (Value::Real(x), Value::Real(y)) => cmp_real(*x, *y),
        (Value::Integer(x), Value::Real(y)) => cmp_int_real(*x, *y),
        (Value::Real(x), Value::Integer(y)) => cmp_int_real(*y, *x).reverse(),
        // `compare_values` only ever routes numeric pairs here. Fail loudly on a
        // mis-route or a new Value variant rather than corrupting sort/index order
        // with a silent Equal.
        _ => unreachable!("compare_numeric received a non-numeric pair"),
    }
}

/// Real-vs-real. `partial_cmp` gives numeric equality including `-0.0 == 0.0`; a
/// NaN (which a stored REAL never is) degrades to `Equal` rather than panicking.
fn cmp_real(x: f64, y: f64) -> Ordering {
    x.partial_cmp(&y).unwrap_or(Ordering::Equal)
}

/// Exact comparison of an `i64` against an `f64`. Returns the ordering of `i`
/// relative to `r`.
///
/// The trick: reduce to comparing `i` against `floor(r)`, an integer that (once we
/// know `r` is in `[-2^63, 2^63)`) is exactly representable and fits an `i64`. If
/// `i` differs from `floor(r)` the answer is immediate; if they are equal, any
/// fractional part of `r` breaks the tie. This never rounds `i`, so it is exact for
/// the full `i64` range — e.g. `i64::MAX` vs a nearby large REAL is decided
/// correctly where an `i as f64` cast would collapse both to `2^63`.
fn cmp_int_real(i: i64, r: f64) -> Ordering {
    if r.is_nan() {
        return Ordering::Equal; // not reachable for a stored REAL
    }
    // Out-of-i64-range reals settle it directly.
    if r >= TWO_POW_63 {
        return Ordering::Less; // i <= i64::MAX < 2^63 <= r
    }
    if r < -TWO_POW_63 {
        return Ordering::Greater; // i >= i64::MIN >= -2^63 > r
    }
    // r is in [-2^63, 2^63): floor(r) is an integer in i64 range, hence exact.
    let floor = r.floor();
    let fi = floor as i64;
    match i.cmp(&fi) {
        Ordering::Equal => {
            // i == floor(r); r has a positive fractional part iff r > floor(r).
            if r > floor {
                Ordering::Less
            } else {
                Ordering::Equal
            }
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering::*;

    fn cmp(a: Value, b: Value) -> Ordering {
        compare_values(&a, &b, Collation::Binary)
    }

    #[test]
    fn class_ordering_null_numeric_text_blob() {
        // NULL < numeric < text < blob.
        assert_eq!(cmp(Value::Null, Value::Integer(-9999)), Less);
        assert_eq!(cmp(Value::Integer(9999), Value::Text("".into())), Less);
        assert_eq!(cmp(Value::Real(1e300), Value::Text("a".into())), Less);
        assert_eq!(cmp(Value::Text("zzz".into()), Value::Blob(vec![0])), Less);
        assert_eq!(cmp(Value::Blob(vec![1]), Value::Real(0.0)), Greater);
    }

    #[test]
    fn nulls_are_equal_and_first() {
        assert_eq!(cmp(Value::Null, Value::Null), Equal);
        assert_eq!(storage_class_rank(&Value::Null), 0);
    }

    #[test]
    fn integers_and_reals_compare_by_value() {
        assert_eq!(cmp(Value::Integer(2), Value::Real(2.0)), Equal);
        assert_eq!(cmp(Value::Integer(2), Value::Real(2.5)), Less);
        assert_eq!(cmp(Value::Real(2.5), Value::Integer(2)), Greater);
        assert_eq!(cmp(Value::Integer(3), Value::Real(2.5)), Greater);
        assert_eq!(cmp(Value::Real(-3.5), Value::Integer(-3)), Less);
        assert_eq!(cmp(Value::Integer(-3), Value::Real(-3.5)), Greater);
    }

    #[test]
    fn int_real_exactness_at_large_magnitude() {
        // i64::MAX = 9223372036854775807. As f64 it would round to 2^63, colliding
        // with many nearby reals; the exact comparison must not.
        let max = Value::Integer(i64::MAX);
        // (double)9223372036854775807 == 2^63 == 9223372036854775808.0 > i64::MAX.
        assert_eq!(cmp(max.clone(), Value::Real(9_223_372_036_854_775_808.0)), Less);
        // 2^63 - 1024 is exactly representable and less than i64::MAX.
        assert_eq!(cmp(max.clone(), Value::Real(9_223_372_036_854_774_784.0)), Greater);
        // i64::MIN vs a real just below it.
        let min = Value::Integer(i64::MIN);
        assert_eq!(cmp(min.clone(), Value::Real(-9_223_372_036_854_777_856.0)), Greater);
        assert_eq!(cmp(min, Value::Real(f64::NEG_INFINITY)), Greater);
    }

    #[test]
    fn infinities_and_ranges() {
        assert_eq!(cmp(Value::Integer(0), Value::Real(f64::INFINITY)), Less);
        assert_eq!(cmp(Value::Integer(0), Value::Real(f64::NEG_INFINITY)), Greater);
    }

    #[test]
    fn text_uses_collation() {
        assert_eq!(
            compare_values(&Value::Text("abc".into()), &Value::Text("ABC".into()), Collation::NoCase),
            Equal
        );
        assert_eq!(
            compare_values(&Value::Text("abc".into()), &Value::Text("ABC".into()), Collation::Binary),
            Greater
        );
    }

    #[test]
    fn blobs_compare_bytewise() {
        assert_eq!(cmp(Value::Blob(vec![1, 2]), Value::Blob(vec![1, 3])), Less);
        assert_eq!(cmp(Value::Blob(vec![1, 2]), Value::Blob(vec![1, 2])), Equal);
        assert_eq!(cmp(Value::Blob(vec![1, 2, 3]), Value::Blob(vec![1, 2])), Greater);
    }

    #[test]
    fn extremum_wins_directions_ties_and_cross_class() {
        let (one, two) = (Value::Integer(1), Value::Integer(2));
        // max: a strictly greater candidate wins; a strictly smaller one does not.
        assert!(extremum_wins(&one, &two, true, Collation::Binary));
        assert!(!extremum_wins(&two, &one, true, Collation::Binary));
        // min: a strictly smaller candidate wins; a strictly greater one does not.
        assert!(extremum_wins(&two, &one, false, Collation::Binary));
        assert!(!extremum_wins(&one, &two, false, Collation::Binary));
        // A tie is never a win in either direction (first-seen extremum is kept).
        assert!(!extremum_wins(&Value::Integer(2), &Value::Real(2.0), true, Collation::Binary));
        assert!(!extremum_wins(&Value::Integer(2), &Value::Real(2.0), false, Collation::Binary));
        // Cross-class uses the storage-class order (numeric < text): text wins a max,
        // loses a min.
        let (num, txt) = (Value::Integer(9), Value::Text("a".into()));
        assert!(extremum_wins(&num, &txt, true, Collation::Binary));
        assert!(!extremum_wins(&num, &txt, false, Collation::Binary));
        // Collation is honored: under NoCase "ABC" ties "abc" (no win), under Binary
        // "ABC" < "abc" so it wins a min.
        let abc = Value::Text("abc".into());
        assert!(!extremum_wins(&abc, &Value::Text("ABC".into()), false, Collation::NoCase));
        assert!(extremum_wins(&abc, &Value::Text("ABC".into()), false, Collation::Binary));
    }

    #[test]
    fn compare_for_eq_is_null_aware() {
        assert_eq!(compare_for_eq(&Value::Null, &Value::Integer(1), Collation::Binary), None);
        assert_eq!(compare_for_eq(&Value::Integer(1), &Value::Null, Collation::Binary), None);
        assert_eq!(compare_for_eq(&Value::Null, &Value::Null, Collation::Binary), None);
        assert_eq!(
            compare_for_eq(&Value::Integer(1), &Value::Integer(1), Collation::Binary),
            Some(Equal)
        );
        // Cross-class: a number is less than text, and it is a definite answer, not NULL.
        assert_eq!(
            compare_for_eq(&Value::Integer(5), &Value::Text("5".into()), Collation::Binary),
            Some(Less)
        );
    }

    // A small model-based check: for a spread of int/real pairs, the exact
    // comparison agrees with an f64-based comparison whenever the f64 cast is
    // lossless, and stays a valid total order everywhere.
    #[test]
    fn int_real_matches_lossless_reference() {
        let ints = [0i64, 1, -1, 2, 3, 100, -100, 1 << 40, -(1 << 40)];
        let reals = [-0.5f64, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, -1.5, 100.25, 1e12];
        for &i in &ints {
            for &r in &reals {
                let got = cmp(Value::Integer(i), Value::Real(r));
                // For these small magnitudes i as f64 is lossless, so it's a valid oracle.
                let want = (i as f64).partial_cmp(&r).unwrap();
                assert_eq!(got, want, "cmp({i}, {r})");
            }
        }
    }
}
