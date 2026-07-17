//! The one `ORDER BY` comparator, shared by every operator that orders rows by a
//! list of [`SortKey`]s: [`Sort`](crate::ops::sort) (`ORDER BY`),
//! [`Aggregate`](crate::ops::aggregate) (aggregate `ORDER BY`), and — when it lands —
//! the window operator's `PARTITION BY` / `ORDER BY`.
//!
//! SQLite's ordering rules live here, in ONE place, so they cannot silently drift
//! between operators (a `NULLS FIRST` default or collation-edge fix to one copy that
//! never reaches the other is exactly the conformance trap this consolidates away):
//! the first key that is not `Equal` decides; a NULL defaults to the low end (first
//! for `ASC`, last for `DESC`) unless the query pinned `NULLS FIRST`/`LAST`; two
//! non-NULLs compare by storage class then value/collation via
//! [`compare_values`], with `DESC` reversing that order. The unit tests below pin
//! each rule and are the standing gate against re-divergence.

use std::cmp::Ordering;

use minisqlite_expr::SortKey;
use minisqlite_types::{compare_values, Value};

/// Compare two rows by their precomputed key values under the sort modifiers. The
/// first key that is not `Equal` decides; all-equal is `Equal`.
///
/// A key-value vector shorter than `keys` (which a correct caller never produces —
/// each caller sizes its vector to `keys.len()`) stops at the missing index with
/// `Equal` rather than panicking.
pub(crate) fn compare_sort_keys(a: &[Value], b: &[Value], keys: &[SortKey]) -> Ordering {
    for (i, key) in keys.iter().enumerate() {
        let (av, bv) = match (a.get(i), b.get(i)) {
            (Some(av), Some(bv)) => (av, bv),
            _ => return Ordering::Equal,
        };
        let ord = compare_one(av, bv, key);
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

/// Compare one key column with SQLite's NULL placement and direction rules.
///
/// NULLs default to the low end (first for `ASC`, last for `DESC`) unless the query
/// pinned `NULLS FIRST/LAST`; here `nulls_first` defaults to `!desc`. For two
/// non-NULLs, [`compare_values`] orders by storage class then value/collation, and
/// `desc` reverses it.
fn compare_one(a: &Value, b: &Value, key: &SortKey) -> Ordering {
    let nulls_first = key.nulls_first.unwrap_or(!key.desc);
    match (a.is_null(), b.is_null()) {
        (true, true) => Ordering::Equal,
        (true, false) => {
            if nulls_first {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }
        (false, true) => {
            if nulls_first {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        }
        (false, false) => {
            let ord = compare_values(a, b, key.collation);
            if key.desc {
                ord.reverse()
            } else {
                ord
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::compare_sort_keys;
    use std::cmp::Ordering;

    use minisqlite_expr::{EvalExpr, SortKey};
    use minisqlite_types::{Collation, Value};

    /// A `SortKey` with a dummy expr (the comparator reads precomputed values, not the
    /// expr) and the given modifiers.
    fn key(desc: bool, nulls_first: Option<bool>, collation: Collation) -> SortKey {
        SortKey { expr: EvalExpr::Column(0), desc, nulls_first, collation }
    }

    fn asc() -> SortKey {
        key(false, None, Collation::Binary)
    }

    fn desc() -> SortKey {
        key(true, None, Collation::Binary)
    }

    /// Compare two single-column rows under one key.
    fn cmp1(a: Value, b: Value, k: SortKey) -> Ordering {
        compare_sort_keys(&[a], &[b], std::slice::from_ref(&k))
    }

    #[test]
    fn asc_null_sorts_first_by_default() {
        assert_eq!(cmp1(Value::Null, Value::Integer(1), asc()), Ordering::Less);
        assert_eq!(cmp1(Value::Integer(1), Value::Null, asc()), Ordering::Greater);
    }

    #[test]
    fn desc_null_sorts_last_by_default() {
        // DESC default is NULLS LAST: NULL compares "greater" so it lands at the end.
        assert_eq!(cmp1(Value::Null, Value::Integer(1), desc()), Ordering::Greater);
        assert_eq!(cmp1(Value::Integer(1), Value::Null, desc()), Ordering::Less);
    }

    #[test]
    fn explicit_nulls_first_overrides_desc_default() {
        let k = key(true, Some(true), Collation::Binary);
        assert_eq!(cmp1(Value::Null, Value::Integer(1), k), Ordering::Less);
    }

    #[test]
    fn explicit_nulls_last_overrides_asc_default() {
        let k = key(false, Some(false), Collation::Binary);
        assert_eq!(cmp1(Value::Null, Value::Integer(1), k), Ordering::Greater);
    }

    #[test]
    fn both_null_is_equal() {
        assert_eq!(cmp1(Value::Null, Value::Null, asc()), Ordering::Equal);
        assert_eq!(cmp1(Value::Null, Value::Null, desc()), Ordering::Equal);
    }

    #[test]
    fn non_nulls_compare_by_value_and_desc_reverses() {
        assert_eq!(cmp1(Value::Integer(1), Value::Integer(2), asc()), Ordering::Less);
        assert_eq!(cmp1(Value::Integer(1), Value::Integer(2), desc()), Ordering::Greater);
        assert_eq!(cmp1(Value::Integer(2), Value::Integer(2), asc()), Ordering::Equal);
    }

    #[test]
    fn collation_folds_text_equality() {
        // NoCase folds case (so "abc" == "ABC"); Binary does not.
        assert_eq!(
            cmp1(
                Value::Text("abc".into()),
                Value::Text("ABC".into()),
                key(false, None, Collation::NoCase),
            ),
            Ordering::Equal,
        );
        assert_ne!(
            cmp1(
                Value::Text("abc".into()),
                Value::Text("ABC".into()),
                key(false, None, Collation::Binary),
            ),
            Ordering::Equal,
        );
    }

    #[test]
    fn first_unequal_key_decides_then_ties_fall_through() {
        let keys = [asc(), asc()];
        // Tie on key 0; key 1 (ASC) breaks it.
        assert_eq!(
            compare_sort_keys(
                &[Value::Integer(1), Value::Integer(2)],
                &[Value::Integer(1), Value::Integer(3)],
                &keys,
            ),
            Ordering::Less,
        );
        // Key 0 already decides (Greater); key 1 is never consulted.
        assert_eq!(
            compare_sort_keys(
                &[Value::Integer(5), Value::Integer(2)],
                &[Value::Integer(1), Value::Integer(9)],
                &keys,
            ),
            Ordering::Greater,
        );
    }

    #[test]
    fn mixed_direction_keys() {
        // key0 ASC, key1 DESC: tie on key0 -> key1 DESC decides (2 sorts after 3).
        let keys = [asc(), desc()];
        assert_eq!(
            compare_sort_keys(
                &[Value::Integer(1), Value::Integer(2)],
                &[Value::Integer(1), Value::Integer(3)],
                &keys,
            ),
            Ordering::Greater,
        );
    }

    #[test]
    fn short_key_vector_stops_without_panicking() {
        // A row shorter than `keys` returns Equal at the missing index (no panic).
        let keys = [asc(), asc()];
        assert_eq!(
            compare_sort_keys(&[Value::Integer(1)], &[Value::Integer(1)], &keys),
            Ordering::Equal,
        );
    }
}
