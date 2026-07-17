//! A canonical, hashable row key for the dedup operators (`DISTINCT`, and later
//! `GROUP BY` / `UNION` / `INTERSECT` / `EXCEPT`).
//!
//! Two values that SQLite treats as the *same* group must produce the SAME key:
//! * numeric value-equality across storage classes — `Integer(2)` and `Real(2.0)`
//!   dedup together (SQLite's `SELECT DISTINCT` folds `2` and `2.0`), so an integral
//!   real is canonicalized to its integer key;
//! * text under the column's collation — `NOCASE` folds case, `RTRIM` ignores
//!   trailing spaces, so `'abc'` and `'ABC'` (NOCASE) share a key.
//!
//! `f64` is not `Hash`/`Eq`, so a non-integral real is keyed by its raw bit pattern
//! ([`f64::to_bits`]): equal reals hash equal, and two distinct reals never collide.
//! (`NaN` never occurs in a stored REAL; if one did, `to_bits` still gives it a
//! stable self-equal key, which is the desired "distinct row" behavior for dedup.)

use minisqlite_types::{real_to_int_if_exact, Collation, Value};

/// One column's canonical dedup key. Derived from a [`Value`] under a [`Collation`]
/// so equal-under-SQLite values compare and hash equal.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum CellKey {
    Null,
    /// An integer, OR an integral real folded to its exact integer value.
    Int(i64),
    /// A non-integral (or out-of-i64-range) real, keyed by its bit pattern.
    Real(u64),
    /// Text, already folded under its collation (so the key comparison is plain `Eq`).
    Text(Vec<u8>),
    Blob(Vec<u8>),
}

/// Canonicalize one value to its dedup key under collation `c`.
pub(crate) fn cell_key(v: &Value, c: Collation) -> CellKey {
    match v {
        Value::Null => CellKey::Null,
        Value::Integer(i) => CellKey::Int(*i),
        Value::Real(r) => {
            // An integral real in i64 range folds to the same key as the integer, so
            // 2 and 2.0 dedup together; everything else keys by its exact bits. The
            // "is this real an exact i64?" rule lives once in `minisqlite-types` (the
            // column-affinity rule), so the dedup-key path here and the rowid-lookup
            // path (`ops::scan`) cannot drift from each other or from affinity.
            match real_to_int_if_exact(*r) {
                Some(i) => CellKey::Int(i),
                None => CellKey::Real(r.to_bits()),
            }
        }
        Value::Text(s) => CellKey::Text(fold_text(s, c)),
        Value::Blob(b) => CellKey::Blob(b.clone()),
    }
}

/// The whole-row key: one [`CellKey`] per column under its collation. If `collations`
/// is shorter than the row (or empty, as for the still-collation-less callers — the
/// recursive-CTE `UNION` dedup in `ops::cte` and the window-DISTINCT defensive path),
/// the remaining columns default to [`Collation::Binary`] — the core default. The
/// collation-aware callers (`DISTINCT`, compound set-ops, and window `PARTITION BY`)
/// pass a full per-column `collations` slice resolved at plan time.
pub(crate) fn row_key(row: &[Value], collations: &[Collation]) -> Vec<CellKey> {
    row.iter()
        .enumerate()
        .map(|(i, v)| cell_key(v, collations.get(i).copied().unwrap_or(Collation::Binary)))
        .collect()
}

/// Fold text bytes for its collation so keys compare with plain byte equality:
/// `Binary` keeps the raw UTF-8, `NoCase` lowercases ASCII, `Rtrim` strips trailing
/// spaces. Mirrors `minisqlite_types::compare_text`'s notion of equality.
fn fold_text(s: &str, c: Collation) -> Vec<u8> {
    match c {
        Collation::Binary => s.as_bytes().to_vec(),
        Collation::NoCase => s.bytes().map(|b| b.to_ascii_lowercase()).collect(),
        Collation::Rtrim => {
            let bytes = s.as_bytes();
            let end = bytes.iter().rposition(|&b| b != b' ').map_or(0, |i| i + 1);
            bytes[..end].to_vec()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_and_integral_real_share_a_key() {
        assert_eq!(cell_key(&Value::Integer(2), Collation::Binary), CellKey::Int(2));
        assert_eq!(cell_key(&Value::Real(2.0), Collation::Binary), CellKey::Int(2));
        assert_eq!(
            cell_key(&Value::Integer(2), Collation::Binary),
            cell_key(&Value::Real(2.0), Collation::Binary),
            "2 and 2.0 dedup together"
        );
    }

    #[test]
    fn non_integral_real_keyed_by_bits() {
        let k = cell_key(&Value::Real(2.5), Collation::Binary);
        assert_eq!(k, CellKey::Real((2.5f64).to_bits()));
        assert_ne!(k, cell_key(&Value::Integer(2), Collation::Binary));
    }

    #[test]
    fn out_of_range_real_is_not_folded_to_int() {
        // 2^63 is exactly representable but one past i64::MAX, so it must NOT fold to
        // an Int key — it keys by its bits. (Pins the upper boundary of the fold.)
        let big = 9_223_372_036_854_775_808.0_f64; // 2^63
        assert_eq!(cell_key(&Value::Real(big), Collation::Binary), CellKey::Real(big.to_bits()));
    }

    #[test]
    fn text_collation_folding() {
        assert_eq!(
            cell_key(&Value::Text("ABC".into()), Collation::NoCase),
            cell_key(&Value::Text("abc".into()), Collation::NoCase),
        );
        assert_ne!(
            cell_key(&Value::Text("ABC".into()), Collation::Binary),
            cell_key(&Value::Text("abc".into()), Collation::Binary),
        );
        assert_eq!(
            cell_key(&Value::Text("abc  ".into()), Collation::Rtrim),
            cell_key(&Value::Text("abc".into()), Collation::Rtrim),
        );
    }

    #[test]
    fn distinct_storage_classes_do_not_collide() {
        // Null / numeric / text / blob keys are all distinct even when they "look"
        // alike (e.g. text "2" vs integer 2).
        let two_int = cell_key(&Value::Integer(2), Collation::Binary);
        let two_txt = cell_key(&Value::Text("2".into()), Collation::Binary);
        let two_blob = cell_key(&Value::Blob(vec![b'2']), Collation::Binary);
        assert_ne!(two_int, two_txt);
        assert_ne!(two_txt, two_blob);
        assert_ne!(cell_key(&Value::Null, Collation::Binary), two_int);
    }

    #[test]
    fn row_key_defaults_extra_columns_to_binary() {
        // Only one collation supplied; the second column falls back to Binary (so
        // "ABC" != "abc" there).
        let row = [Value::Text("ABC".into()), Value::Text("ABC".into())];
        let other = [Value::Text("abc".into()), Value::Text("abc".into())];
        let k1 = row_key(&row, &[Collation::NoCase]);
        let k2 = row_key(&other, &[Collation::NoCase]);
        // col 0 folds equal (NoCase), col 1 does not (Binary) -> whole keys differ.
        assert_ne!(k1, k2);
        assert_eq!(k1[0], k2[0]);
        assert_ne!(k1[1], k2[1]);
    }

    #[test]
    fn cell_key_equality_matches_compare_for_eq_for_non_null() {
        // The load-bearing invariant TWO consumers now rely on — the hash join
        // (`ops::join`) and the `IN`-subquery cache probe (`context::probe_in_set`): for
        // NON-NULL values, `a` and `b` share a `cell_key` under collation `c` IFF
        // `compare_for_eq` reports them Equal under `c`. That equivalence is what lets a
        // `HashSet<CellKey>` membership test stand in for the streaming `compare_for_eq`
        // 3VL scan. `cell_key` (here) and `compare_for_eq` (`minisqlite-types`) live in
        // different crates with no compile-time link, so this pins the biconditional over
        // representative pairs — including the numeric-fold boundaries — as a standing
        // gate: a future change to EITHER function that breaks the agreement fails here.
        // (NULL is excluded: `compare_for_eq(NULL, NULL)` is `None`, yet `CellKey::Null`
        //  equals itself — dedup keys NULLs together while comparison leaves them unknown.)
        use minisqlite_types::compare_for_eq;
        use std::cmp::Ordering;

        let two_53 = 9_007_199_254_740_992_i64; // 2^53: the exact integer/real boundary
        let cases: &[(Value, Value, Collation)] = &[
            // numeric fold: an integer and an integral real are equal AND key-equal
            (Value::Integer(2), Value::Real(2.0), Collation::Binary),
            (Value::Integer(0), Value::Real(-0.0), Collation::Binary), // +0.0/-0.0 fold to 0
            // numeric non-equal
            (Value::Integer(2), Value::Real(2.5), Collation::Binary),
            (Value::Integer(2), Value::Integer(3), Collation::Binary),
            // precision boundary: 2^53+1 (exact i64) vs 2^53 (real) must NOT be equal
            (Value::Integer(two_53 + 1), Value::Real(two_53 as f64), Collation::Binary),
            // text under collation: folds equal only under the matching collation
            (Value::Text("a".into()), Value::Text("A".into()), Collation::NoCase),
            (Value::Text("a".into()), Value::Text("A".into()), Collation::Binary),
            (Value::Text("ab ".into()), Value::Text("ab".into()), Collation::Rtrim),
            (Value::Text("ab ".into()), Value::Text("ab".into()), Collation::Binary),
            // cross storage-class: never equal (text "2" is not integer 2)
            (Value::Integer(2), Value::Text("2".into()), Collation::Binary),
            (Value::Blob(vec![b'2']), Value::Text("2".into()), Collation::Binary),
            // blobs: bytewise
            (Value::Blob(vec![1, 2]), Value::Blob(vec![1, 2]), Collation::Binary),
            (Value::Blob(vec![1, 2]), Value::Blob(vec![1, 3]), Collation::Binary),
            // identity
            (Value::Integer(7), Value::Integer(7), Collation::Binary),
        ];
        for (a, b, c) in cases {
            let key_eq = cell_key(a, *c) == cell_key(b, *c);
            let cmp_eq = compare_for_eq(a, b, *c) == Some(Ordering::Equal);
            assert_eq!(
                key_eq, cmp_eq,
                "cell_key equality must match compare_for_eq Equal: {a:?} vs {b:?} under {c:?}"
            );
        }
    }
}
