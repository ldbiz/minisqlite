//! Value -> number coercions used by the operators (arithmetic, bitwise, boolean
//! context). These are distinct from CAST and from column affinity: they are the
//! implicit coercions SQLite applies at the operators themselves (lang_expr.html
//! §8, datatype3.html §4/§8).
//!
//! Three public entry points, plus a private [`Num`] the arithmetic core uses so
//! the "coerce then compute" path has no unrepresentable state (a coerced operand
//! is always a concrete `i64` or `f64`, never a stray text/blob).

use minisqlite_types::{
    numeric_prefix, parse_int_prefix, parse_real_prefix, real_to_int_trunc, NumericPrefix, Value,
};

/// A numeric operand after implicit coercion: exactly an integer or a real, never
/// anything else. Keeping the coerced form in this two-variant type (rather than a
/// `Value`) makes the arithmetic core total — there is no "what if it's text" arm
/// to get wrong.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Num {
    Int(i64),
    Real(f64),
}

impl Num {
    /// The `f64` view of this number, for the floating-point arithmetic path.
    pub(crate) fn to_f64(self) -> f64 {
        match self {
            Num::Int(i) => i as f64,
            Num::Real(r) => r,
        }
    }
}

/// Interpret a text/blob value as a number the way the operators do: take the
/// longest *leading* numeric prefix (lang_expr.html §13, which the boolean rule §14
/// defers to), not a whole-string literal. A prefix that carries a `.` or an
/// exponent is a REAL (`'2.5abc'` -> 2.5, `'2e3x'` -> 2000.0); a plain sign+digits
/// prefix is an INTEGER (`'2abc'` -> 2, `'1english'` -> 1, `'-3x'` -> -3); a string
/// with no leading numeric run is 0 (`'abc'`, `''`, `'-'`, `'.'`).
///
/// This takes the same leading-prefix extraction the engine's `CAST(x AS NUMERIC)`
/// performs, and the INT-vs-REAL split agrees on the common cases, but two
/// deliberate differences remain from CAST-to-NUMERIC (`text_to_numeric`):
/// * A real-form prefix stays REAL here (`'2.0'` -> `2.0`, `'2e3abc'` -> `2000.0`),
///   whereas CAST-to-NUMERIC demotes an integral real-form to INTEGER. This mirrors
///   real sqlite's arithmetic coercion, which — unlike CAST/affinity storage —
///   keeps `'2.0' + 0` as `2.0`.
/// * An integer-form prefix that overflows `i64` clamps to `i64::MAX`/`MIN` here
///   (via [`parse_int_prefix`]), whereas CAST-to-NUMERIC promotes an overflowing
///   integer-form to REAL. This matches the pre-existing clamp the bitwise/`%`
///   path already uses ([`to_integer`]); it only bites on 19-20+ digit
///   numeric-prefix text, a case not observed in practice.
fn text_to_num(s: &str) -> Num {
    match numeric_prefix(s) {
        NumericPrefix::Real => Num::Real(parse_real_prefix(s)),
        NumericPrefix::Integer => Num::Int(parse_int_prefix(s)),
        NumericPrefix::None => Num::Int(0),
    }
}

/// Coerce a value to a [`Num`] for arithmetic. NULL is handled by callers before
/// this point (arithmetic/`truth`/unary `-` all return NULL up front); the `Null`
/// arm only keeps the match total and is never reached on the live path. The
/// `debug_assert` turns a future caller that forgot that NULL-check into a loud test
/// failure rather than a silent `0`.
pub(crate) fn as_num(v: &Value) -> Num {
    debug_assert!(!v.is_null(), "as_num on NULL: callers must NULL-check first");
    match v {
        Value::Integer(i) => Num::Int(*i),
        Value::Real(r) => Num::Real(*r),
        Value::Text(s) => text_to_num(s),
        // A blob's bytes are read as text, then the same rule applies.
        Value::Blob(b) => text_to_num(&String::from_utf8_lossy(b)),
        Value::Null => Num::Int(0),
    }
}

/// Implicit numeric coercion for arithmetic operands (`+ - * /` and unary `-`).
/// NULL stays NULL; INTEGER/REAL are unchanged; text/blob follow [`text_to_num`]
/// (longest leading numeric prefix, else 0). A real-form prefix like `'2.0'` or
/// `'2e3abc'` yields REAL, unlike CAST-to-NUMERIC.
pub fn to_number_for_arith(v: &Value) -> Value {
    if v.is_null() {
        return Value::Null;
    }
    match as_num(v) {
        Num::Int(i) => Value::Integer(i),
        Num::Real(r) => Value::Real(r),
    }
}

/// Coerce a value to an `i64` for the operators that are integer-only: bitwise
/// `& | << >>` and modulo `%`. A REAL is truncated toward zero; text/blob take the
/// longest leading integer prefix (0 if none). NULL is handled by callers before
/// this point; the `Null` arm returns 0 only to keep the function total.
pub fn to_integer(v: &Value) -> i64 {
    match v {
        Value::Integer(i) => *i,
        Value::Real(r) => real_to_int_trunc(*r),
        Value::Text(s) => parse_int_prefix(s),
        Value::Blob(b) => parse_int_prefix(&String::from_utf8_lossy(b)),
        Value::Null => 0,
    }
}

/// The truth value of a value in a boolean context (`WHERE`, `AND`/`OR`/`NOT`,
/// `CASE WHEN`, `CHECK`): `None` for NULL (unknown), else `Some(true)` iff the
/// numeric coercion is non-zero. This is SQLite's rule that any non-zero number —
/// and any text/blob that coerces to a non-zero number — is true.
pub fn truth(v: &Value) -> Option<bool> {
    if v.is_null() {
        return None;
    }
    match as_num(v) {
        Num::Int(i) => Some(i != 0),
        Num::Real(r) => Some(r != 0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_number_for_arith_rules() {
        assert!(matches!(to_number_for_arith(&Value::Null), Value::Null));
        assert!(matches!(to_number_for_arith(&Value::Integer(5)), Value::Integer(5)));
        assert!(matches!(to_number_for_arith(&Value::Real(2.5)), Value::Real(r) if r == 2.5));
        // Whole-string integer -> Integer.
        assert!(matches!(to_number_for_arith(&Value::Text("42".into())), Value::Integer(42)));
        // Whole-string real form -> Real (NOT demoted, unlike CAST-to-NUMERIC).
        assert!(matches!(to_number_for_arith(&Value::Text("2.0".into())), Value::Real(r) if r == 2.0));
        // Leading integer prefix -> INTEGER of that prefix (the bug was returning 0).
        assert!(matches!(to_number_for_arith(&Value::Text("2abc".into())), Value::Integer(2)));
        assert!(matches!(to_number_for_arith(&Value::Text("1english".into())), Value::Integer(1)));
        assert!(matches!(to_number_for_arith(&Value::Text("-3x".into())), Value::Integer(-3)));
        // Leading real-form prefix -> REAL (a '.' or exponent in the consumed run).
        assert!(matches!(to_number_for_arith(&Value::Text("2.5abc".into())), Value::Real(r) if r == 2.5));
        assert!(matches!(to_number_for_arith(&Value::Text("2e3abc".into())), Value::Real(r) if r == 2000.0));
        // No leading numeric prefix -> 0.
        assert!(matches!(to_number_for_arith(&Value::Text("abc".into())), Value::Integer(0)));
        assert!(matches!(to_number_for_arith(&Value::Text("".into())), Value::Integer(0)));
        // Blob read as text, same leading-prefix rule.
        assert!(matches!(to_number_for_arith(&Value::Blob(b"7".to_vec())), Value::Integer(7)));
        assert!(matches!(to_number_for_arith(&Value::Blob(b"1.5".to_vec())), Value::Real(r) if r == 1.5));
        assert!(matches!(to_number_for_arith(&Value::Blob(b"9zz".to_vec())), Value::Integer(9)));
    }

    #[test]
    fn to_integer_rules() {
        assert_eq!(to_integer(&Value::Integer(9)), 9);
        assert_eq!(to_integer(&Value::Real(9.9)), 9); // truncates toward zero
        assert_eq!(to_integer(&Value::Real(-9.9)), -9);
        assert_eq!(to_integer(&Value::Text("12x".into())), 12); // longest int prefix
        assert_eq!(to_integer(&Value::Text("abc".into())), 0);
        assert_eq!(to_integer(&Value::Blob(b"34zz".to_vec())), 34);
    }

    #[test]
    fn truth_three_valued() {
        assert_eq!(truth(&Value::Null), None);
        assert_eq!(truth(&Value::Integer(0)), Some(false));
        assert_eq!(truth(&Value::Integer(5)), Some(true));
        assert_eq!(truth(&Value::Integer(-1)), Some(true));
        assert_eq!(truth(&Value::Real(0.0)), Some(false));
        assert_eq!(truth(&Value::Real(-0.0)), Some(false));
        assert_eq!(truth(&Value::Real(0.5)), Some(true));
        // Text coerces first: "0"/"abc"/"" are false, "1"/"2.5" true.
        assert_eq!(truth(&Value::Text("0".into())), Some(false));
        assert_eq!(truth(&Value::Text("abc".into())), Some(false));
        assert_eq!(truth(&Value::Text("".into())), Some(false));
        assert_eq!(truth(&Value::Text("1".into())), Some(true));
        assert_eq!(truth(&Value::Text("2.5".into())), Some(true));
        // A leading numeric prefix drives the truth value (§14 casts to NUMERIC by
        // the leading prefix): '1english' -> 1 -> true, '0abc' -> 0 -> false, and a
        // string with no numeric prefix ('abc') is 0 -> false.
        assert_eq!(truth(&Value::Text("1english".into())), Some(true));
        assert_eq!(truth(&Value::Text("0abc".into())), Some(false));
        assert_eq!(truth(&Value::Text("2.5xyz".into())), Some(true));
        assert_eq!(truth(&Value::Text("-3x".into())), Some(true));
    }
}
