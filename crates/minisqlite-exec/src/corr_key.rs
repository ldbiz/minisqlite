//! The correlation key for the correlated-subquery memo ([`Runtime::correlated_subquery_cache`]).
//!
//! A correlated subquery is re-evaluated per outer row today; memoizing its result
//! keyed by the VALUES of the outer registers it depends on
//! ([`SubPlan::correlated_cols`](minisqlite_plan::SubPlan)) collapses that to one run per
//! DISTINCT combination of those values. For that to be CORRECT the key must distinguish
//! exactly the values SQLite would treat as different inputs.
//!
//! This is why [`CorrCell`] is a SEPARATE type from [`CellKey`](crate::keys::CellKey) and
//! must NOT reuse it: `CellKey` deliberately FOLDS values that dedup together —
//! `Integer(2)` and `Real(2.0)` collapse to one key, and text folds under a collation
//! (`'abc'`/`'ABC'` under `NOCASE`). Those folds are correct for `DISTINCT`/`GROUP BY` but
//! WRONG here: a correlated `(SELECT typeof(t.x))` returns `'integer'` for `2` and `'real'`
//! for `2.0`, and `(SELECT t.x)` returns the exact stored text — so folding the outer key
//! would hand two genuinely different outer rows the SAME cached result. `CorrCell` is
//! therefore storage-class-EXACT and fold-free:
//! * `Integer` and `Real` never share a key — a real is kept by its raw bits
//!   ([`f64::to_bits`]), with NO integral fold, so `2` and `2.0` differ;
//! * text is kept as its EXACT bytes with NO collation fold;
//! * blob is exact bytes; NULL is its own key.
//!
//! Over-distinguishing is safe here (at worst it runs the subquery an extra time for two
//! outer rows that would in fact yield the same result); under-distinguishing is a WRONG
//! answer. So the rule is: when in doubt, keep them apart. NULL keys to NULL and so
//! `NULL == NULL` for cache purposes — correct, because the same NULL outer input drives
//! the same deterministic subquery result (SQL's `NULL != NULL` governs comparison, not
//! "was the input the same value").

use minisqlite_types::Value;

/// One outer column's EXACT, fold-free correlation-cache key (see the module docs).
/// `Hash + Eq + Clone` so a `Vec<CorrCell>` is a `HashMap` key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum CorrCell {
    Null,
    Int(i64),
    /// A real, keyed by its raw bit pattern — NO integral fold, so `2.0` never shares a key
    /// with the integer `2`.
    Real(u64),
    /// Text, kept as its EXACT bytes — NO collation fold.
    Text(Vec<u8>),
    Blob(Vec<u8>),
}

/// Map one [`Value`] to its exact correlation key, distinguishing every storage class and
/// applying no folding (the inverse of [`cell_key`](crate::keys::cell_key)'s deliberate
/// folds — see the module docs for why the two must differ).
pub(crate) fn corr_cell(v: &Value) -> CorrCell {
    match v {
        Value::Null => CorrCell::Null,
        Value::Integer(i) => CorrCell::Int(*i),
        // Raw bits, with NO `real_to_int_if_exact` fold: `2.0` and `2` MUST key apart.
        Value::Real(r) => CorrCell::Real(r.to_bits()),
        // Exact bytes, with NO collation folding.
        Value::Text(s) => CorrCell::Text(s.as_bytes().to_vec()),
        Value::Blob(b) => CorrCell::Blob(b.clone()),
    }
}

/// Build the correlation key for one outer row: one [`CorrCell`] per entry of `cols`, in
/// `cols` order (the caller passes `SubPlan::correlated_cols`, which is sorted + deduped).
///
/// INVARIANT: every `c` in `cols` is `< row.len()` — it is an outer register `< outer_width`
/// and `row` is that outer row (width `outer_width`). The completeness of `correlated_cols`
/// (built at the binder's resolve site) is what makes this key sufficient: two outer rows
/// with equal `CorrCell`s at exactly these registers drive the same deterministic result.
pub(crate) fn corr_key(row: &[Value], cols: &[usize]) -> Vec<CorrCell> {
    cols.iter().map(|&c| corr_cell(&row[c])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_classes_key_apart_no_folding() {
        // The whole reason this type exists: `2` (int), `2.0` (real), and `'2'` (text) are
        // THREE distinct keys — none folds into another. A `CellKey` would collapse the
        // first two (and could fold text under a collation), which here is a wrong answer.
        assert_ne!(corr_cell(&Value::Integer(2)), corr_cell(&Value::Real(2.0)));
        assert_ne!(corr_cell(&Value::Integer(2)), corr_cell(&Value::Text("2".into())));
        assert_ne!(corr_cell(&Value::Real(2.0)), corr_cell(&Value::Text("2".into())));
        assert_ne!(corr_cell(&Value::Text("2".into())), corr_cell(&Value::Blob(vec![b'2'])));
    }

    #[test]
    fn text_is_case_sensitive_and_space_exact() {
        // No collation fold: case and trailing spaces both matter (unlike `CellKey`).
        assert_ne!(corr_cell(&Value::Text("abc".into())), corr_cell(&Value::Text("ABC".into())));
        assert_ne!(corr_cell(&Value::Text("ab".into())), corr_cell(&Value::Text("ab ".into())));
        // Identical text keys equal.
        assert_eq!(corr_cell(&Value::Text("abc".into())), corr_cell(&Value::Text("abc".into())));
    }

    #[test]
    fn equal_values_key_equal_including_null() {
        assert_eq!(corr_cell(&Value::Null), corr_cell(&Value::Null), "NULL == NULL for the cache");
        assert_eq!(corr_cell(&Value::Integer(7)), corr_cell(&Value::Integer(7)));
        assert_eq!(corr_cell(&Value::Real(2.5)), corr_cell(&Value::Real(2.5)));
        assert_eq!(corr_cell(&Value::Blob(vec![1, 2])), corr_cell(&Value::Blob(vec![1, 2])));
        // Distinct reals never collide.
        assert_ne!(corr_cell(&Value::Real(2.5)), corr_cell(&Value::Real(2.6)));
    }

    #[test]
    fn corr_key_selects_columns_in_order() {
        // Picks exactly the named registers, in the given order, so the key mirrors
        // `correlated_cols`. Register 1 is skipped here (not a correlated col).
        let row = [Value::Integer(10), Value::Text("skip".into()), Value::Real(2.0)];
        let key = corr_key(&row, &[0, 2]);
        assert_eq!(key, vec![CorrCell::Int(10), CorrCell::Real((2.0f64).to_bits())]);
        // A different value at a keyed register changes the key; a change at the SKIPPED
        // register does not.
        let row2 = [Value::Integer(10), Value::Text("other".into()), Value::Real(2.0)];
        assert_eq!(corr_key(&row2, &[0, 2]), key, "the unkeyed column does not affect the key");
        let row3 = [Value::Integer(11), Value::Text("skip".into()), Value::Real(2.0)];
        assert_ne!(corr_key(&row3, &[0, 2]), key, "a keyed column change changes the key");
    }
}
