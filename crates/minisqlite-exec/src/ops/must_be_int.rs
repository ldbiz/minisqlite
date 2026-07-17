//! Shared `OP_MustBeInt` integer coercion for every operator that requires a value be a
//! signed 64-bit integer: the `LIMIT`/`OFFSET` bounds ([`super::limit`]) and the
//! `INTEGER PRIMARY KEY` (rowid-alias) column that `INSERT` and `UPDATE` write
//! ([`super::insert`], [`super::update`]). A recursive-CTE body `LIMIT`/`OFFSET` bound
//! reaches this rule INDIRECTLY — the planner lowers it to a `PlanNode::Limit` over the
//! fixpoint scan, so [`super::cte`] coerces its bound through [`super::limit`] rather than
//! calling here itself.
//!
//! They share ONE rule — SQLite's `OP_MustBeInt`: apply NUMERIC affinity, then require the
//! result to be an integer, else raise a "datatype mismatch". This rule was previously
//! copied into each operator, every copy re-deriving the same edge cases (the 2^63 range
//! guard, the fractional-text rejection) and re-spelling the error string. It lives here
//! ONCE so the callers cannot drift — the same convention the `integral real -> exact i64`
//! primitive it builds on already follows, living once in
//! [`minisqlite_types::real_to_int_if_exact`] with scan / keys / affinity all calling it.
//!
//! The rule over the whole input space (the five storage classes):
//!
//! - `Integer(i)` -> [`MustBeInt::Int`] verbatim.
//! - an exactly-integral, in-range `Real` (`2.0` -> `2`) -> `Int`, via
//!   `real_to_int_if_exact` (the LOSSLESS check — never a truncation, so `2.5` is
//!   rejected, not folded to `2`); a fractional / out-of-range / non-finite real (`2.5`,
//!   `1e30`, `2^63`, ±inf, NaN) -> [`MustBeInt::NotInt`].
//! - numeric integer `Text` (`'123'`, `'2.0'`) -> `Int` (NUMERIC affinity converts it);
//!   fractional (`'2.5'`), non-numeric (`'abc'`, `'0x10'`), or empty text -> `NotInt`.
//! - a `Blob` (affinity never coerces it) -> `NotInt`.
//! - `Null` -> [`MustBeInt::Null`], reported SEPARATELY from `NotInt`: the callers decide
//!   it differently — an `INSERT` rowid-alias NULL auto-assigns the rowid, whereas
//!   `LIMIT`/`OFFSET` (and a recursive-CTE bound, which lowers to the same `Limit`) and an
//!   `UPDATE` rowid-alias all reject a NULL. Folding NULL into `NotInt` here would break
//!   `INSERT`'s NULL -> auto-assign.
//!
//! The NULL-is-also-an-error callers collapse the outcome with [`MustBeInt::require_int`];
//! `INSERT` matches the three-way [`MustBeInt`] directly (its `Null` arm auto-assigns).

use minisqlite_types::{apply_affinity, real_to_int_if_exact, Affinity, Error, Result, Value};

/// The outcome of coercing a value to a 64-bit integer under SQLite's `OP_MustBeInt`
/// ([`must_be_int`]). `Null` is kept distinct from `NotInt` so a caller whose NULL
/// semantics differ (INSERT's rowid auto-assign vs. everyone else's reject) can branch on
/// it; callers that treat NULL as an error collapse the two via [`Self::require_int`].
pub(crate) enum MustBeInt {
    /// The value is, or losslessly converts to, a signed 64-bit integer.
    Int(i64),
    /// The value is NULL — a storage class the callers decide for themselves.
    Null,
    /// A blob, a fractional / out-of-range / non-finite real, or a non-numeric or
    /// fractional string — none can be losslessly converted, so it is a datatype mismatch.
    NotInt,
}

impl MustBeInt {
    /// Collapse to `Result<i64>` for the callers where NULL is ALSO an error — the
    /// `LIMIT`/`OFFSET` bounds ([`super::limit`], which a recursive-CTE body bound also
    /// reaches, being lowered to a `Limit`) and the `UPDATE` rowid alias (an UPDATE cannot
    /// null the rowid): `Int(i)` -> `Ok(i)`, both `Null` and `NotInt` ->
    /// `Err(datatype_mismatch())`. `INSERT` does NOT use this — its NULL alias
    /// auto-assigns, so it matches the enum directly.
    pub(crate) fn require_int(self) -> Result<i64> {
        match self {
            MustBeInt::Int(i) => Ok(i),
            MustBeInt::Null | MustBeInt::NotInt => Err(datatype_mismatch()),
        }
    }
}

/// Coerce a value under SQLite's `OP_MustBeInt`: apply NUMERIC affinity, then require an
/// integer, with NULL reported separately (see [`MustBeInt`]).
///
/// PURE — inspects the value only, no I/O or globals, so the rule is unit-testable without
/// a cursor / `Env` / `Runtime`. Allocation-free on the common `Integer` / `Null` / `Real`
/// path (matched by reference, and the real fold reuses [`real_to_int_if_exact`] — the
/// same primitive `apply_affinity`'s NUMERIC arm uses, so a real classifies identically
/// whether or not affinity ran first); it clones only to run affinity over a `Text` /
/// `Blob` slot, the rare (numeric-string) or reject (junk / blob) case. So a per-row
/// caller — the rowid alias on every `INSERT`/`UPDATE` row — pays no allocation for an
/// ordinary integer rowid.
pub(crate) fn must_be_int(v: &Value) -> MustBeInt {
    match v {
        Value::Null => MustBeInt::Null,
        Value::Integer(i) => MustBeInt::Int(*i),
        Value::Real(f) => match real_to_int_if_exact(*f) {
            Some(i) => MustBeInt::Int(i),
            None => MustBeInt::NotInt,
        },
        // NUMERIC affinity converts a well-formed numeric string ('123' -> 123, '2.0' ->
        // 2) and demotes an integral real, but leaves a fractional string (-> Real), a
        // non-numeric string (stays Text), and a blob (never coerced) as non-integers.
        // Requiring an Integer afterward is exactly the "lossless to integer" test.
        Value::Text(_) | Value::Blob(_) => match apply_affinity(v.clone(), Affinity::Numeric) {
            Value::Integer(i) => MustBeInt::Int(i),
            _ => MustBeInt::NotInt,
        },
    }
}

/// The "datatype mismatch" error every `OP_MustBeInt` caller raises when a value cannot be
/// losslessly converted to an integer. Centralized so the wording is spelled once.
///
/// Real sqlite raises `SQLITE_MISMATCH` here; the facade `Error` re-exports a 4-variant
/// type (`Sql`/`Format`/`Io`/`Constraint`) with no mismatch kind — an engine-wide
/// limitation out of scope to change here — so this is a plain `Sql` error carrying the
/// text. The behavior that matters is that the statement *errors*; the exact code is the
/// residual gap.
pub(crate) fn datatype_mismatch() -> Error {
    Error::sql("datatype mismatch")
}

#[cfg(test)]
mod tests {
    use super::*;

    // Exhaustive over the five storage classes and the convert/reject boundary each
    // admits — the whole space `must_be_int` classifies. `MustBeInt` has no `PartialEq`,
    // so outcomes are matched with `matches!`.

    #[test]
    fn integer_is_returned_verbatim() {
        for i in [0_i64, 5, -5, i64::MAX, i64::MIN] {
            assert!(matches!(must_be_int(&Value::Integer(i)), MustBeInt::Int(j) if j == i));
        }
    }

    #[test]
    fn null_is_its_own_outcome_not_notint() {
        // The caller decides NULL (INSERT auto-assigns, everyone else rejects); the core
        // must not pre-fold it into NotInt.
        assert!(matches!(must_be_int(&Value::Null), MustBeInt::Null));
    }

    #[test]
    fn integral_real_converts_but_fractional_and_out_of_range_do_not() {
        assert!(matches!(must_be_int(&Value::Real(2.0)), MustBeInt::Int(2)));
        assert!(matches!(must_be_int(&Value::Real(-7.0)), MustBeInt::Int(-7)));
        assert!(matches!(must_be_int(&Value::Real(0.0)), MustBeInt::Int(0)));
        for f in [
            2.5_f64,
            1e30,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            // 2^63 is one past i64::MAX: integral but out of range -> NotInt, not a clamp.
            9_223_372_036_854_775_808.0,
        ] {
            assert!(matches!(must_be_int(&Value::Real(f)), MustBeInt::NotInt), "real {f} -> NotInt");
        }
    }

    #[test]
    fn numeric_integer_text_converts_via_affinity() {
        // A string that losslessly converts to an integer IS an integer (real sqlite:
        // `LIMIT '3'` and `INSERT INTO k VALUES('123', ..)` both accept it). An integral
        // real in string form ('2.0', '3.0e+2') demotes; affinity trims whitespace.
        assert!(matches!(must_be_int(&Value::Text("123".into())), MustBeInt::Int(123)));
        assert!(matches!(must_be_int(&Value::Text("-9".into())), MustBeInt::Int(-9)));
        assert!(matches!(must_be_int(&Value::Text("2.0".into())), MustBeInt::Int(2)));
        assert!(matches!(must_be_int(&Value::Text("3.0e+2".into())), MustBeInt::Int(300)));
        assert!(matches!(must_be_int(&Value::Text("  7 ".into())), MustBeInt::Int(7)));
    }

    #[test]
    fn fractional_or_non_numeric_text_is_notint() {
        // Junk, a fractional string (-> Real under affinity), hex (not a runtime literal),
        // trailing garbage, empty, and an i64-overflowing literal all fail to be integers.
        for s in ["abc", "2.5", "0x10", "12a", "", "99999999999999999999"] {
            assert!(matches!(must_be_int(&Value::Text(s.into())), MustBeInt::NotInt), "text {s:?} -> NotInt");
        }
    }

    #[test]
    fn blob_is_never_an_integer() {
        // A blob is never coerced by affinity — even one whose bytes spell digits.
        for b in [vec![1, 2, 3], b"123".to_vec(), Vec::new()] {
            assert!(matches!(must_be_int(&Value::Blob(b)), MustBeInt::NotInt));
        }
    }

    #[test]
    fn require_int_accepts_int_and_maps_null_and_notint_to_mismatch() {
        assert_eq!(MustBeInt::Int(42).require_int().unwrap(), 42);
        // Both non-Int outcomes collapse to the same "datatype mismatch" for the
        // NULL-is-error callers (LIMIT/OFFSET, CTE bounds, UPDATE rowid).
        for bad in [MustBeInt::Null, MustBeInt::NotInt] {
            match bad.require_int() {
                Err(Error::Sql(m)) => assert_eq!(m, "datatype mismatch"),
                other => panic!("expected Err(Sql(\"datatype mismatch\")), got {other:?}"),
            }
        }
    }

    #[test]
    fn datatype_mismatch_is_a_sql_error_with_sqlite_wording() {
        match datatype_mismatch() {
            Error::Sql(m) => assert_eq!(m, "datatype mismatch"),
            other => panic!("expected Error::Sql(\"datatype mismatch\"), got {other:?}"),
        }
    }
}
