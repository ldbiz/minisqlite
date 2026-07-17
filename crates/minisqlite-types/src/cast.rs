//! `CAST(expr AS type)` conversions (lang_expr.html §13).
//!
//! A CAST is like applying column affinity except that it **always** converts,
//! even when the conversion is lossy and irreversible — the opposite of affinity's
//! lossless-only rule. The caller maps the target type-name to an [`Affinity`] with
//! `affinity_of_declared_type` (so "INT" -> Integer, "NUMERIC" -> Numeric, etc.),
//! then calls [`cast_to`]. NULL casts to NULL for every target.

use crate::affinity::Affinity;
use crate::numeric::{
    integer_to_text, parse_int_prefix, parse_real_prefix, real_to_int_trunc, real_to_text,
    text_to_numeric,
};
use crate::value::Value;

/// Convert `v` to the storage class implied by a CAST target affinity
/// (lang_expr.html §13). This is distinct from `apply_affinity`: it forces the
/// conversion.
pub fn cast_to(v: Value, target: Affinity) -> Value {
    if v.is_null() {
        return Value::Null;
    }
    match target {
        Affinity::Text => cast_to_text(v),
        Affinity::Blob => cast_to_blob(v),
        Affinity::Real => Value::Real(cast_to_real(v)),
        Affinity::Integer => Value::Integer(cast_to_integer(v)),
        Affinity::Numeric => cast_to_numeric(v),
    }
}

/// CAST AS TEXT: numbers render to their text form; a BLOB's bytes are reinterpreted
/// as text.
fn cast_to_text(v: Value) -> Value {
    match v {
        Value::Text(s) => Value::Text(s),
        Value::Integer(i) => Value::Text(integer_to_text(i)),
        Value::Real(r) => Value::Text(real_to_text(r)),
        Value::Blob(b) => Value::Text(bytes_as_text(&b)),
        Value::Null => unreachable!("cast_to handles NULL before dispatch"),
    }
}

/// CAST AS BLOB: "first cast to TEXT, then interpret the bytes as a BLOB". For a
/// value that is already a BLOB this is a no-op; for text the bytes are the text's
/// UTF-8 bytes; for numbers, the bytes of their text form.
fn cast_to_blob(v: Value) -> Value {
    match v {
        Value::Blob(b) => Value::Blob(b),
        Value::Text(s) => Value::Blob(s.into_bytes()),
        Value::Integer(i) => Value::Blob(integer_to_text(i).into_bytes()),
        Value::Real(r) => Value::Blob(real_to_text(r).into_bytes()),
        Value::Null => unreachable!("cast_to handles NULL before dispatch"),
    }
}

/// CAST AS REAL: exact for a number; longest real prefix for text; a BLOB is first
/// read as text.
fn cast_to_real(v: Value) -> f64 {
    match v {
        Value::Real(r) => r,
        Value::Integer(i) => i as f64,
        Value::Text(s) => parse_real_prefix(&s),
        Value::Blob(b) => parse_real_prefix(&bytes_as_text(&b)),
        Value::Null => unreachable!("cast_to handles NULL before dispatch"),
    }
}

/// CAST AS INTEGER: truncate a real toward zero (with i64 clamping); longest integer
/// prefix for text; a BLOB is first read as text.
fn cast_to_integer(v: Value) -> i64 {
    match v {
        Value::Integer(i) => i,
        Value::Real(r) => real_to_int_trunc(r),
        Value::Text(s) => parse_int_prefix(&s),
        Value::Blob(b) => parse_int_prefix(&bytes_as_text(&b)),
        Value::Null => unreachable!("cast_to handles NULL before dispatch"),
    }
}

/// CAST AS NUMERIC: a REAL or INTEGER is unchanged (a no-op, even when a real is
/// exactly integral — this is the one place NUMERIC differs from INTEGER affinity).
/// TEXT/BLOB yield INTEGER or REAL per [`text_to_numeric`].
fn cast_to_numeric(v: Value) -> Value {
    match v {
        Value::Integer(i) => Value::Integer(i),
        Value::Real(r) => Value::Real(r),
        Value::Text(s) => text_to_numeric(&s),
        Value::Blob(b) => text_to_numeric(&bytes_as_text(&b)),
        Value::Null => unreachable!("cast_to handles NULL before dispatch"),
    }
}

/// Interpret raw bytes as text in the database encoding (UTF-8).
///
/// NOTE: `Value::Text` is a Rust `String` and cannot hold invalid UTF-8, so bytes
/// that are not valid UTF-8 are replaced with U+FFFD here (lossy). Real sqlite
/// would keep the raw bytes as a text value; a fully faithful engine would need
/// `Text` to carry bytes. Flagged for the type-representation owner; numeric/ASCII
/// blobs (the common CAST inputs) are unaffected.
fn bytes_as_text(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_casts_to_null_everywhere() {
        for t in [Affinity::Text, Affinity::Blob, Affinity::Real, Affinity::Integer, Affinity::Numeric] {
            assert!(matches!(cast_to(Value::Null, t), Value::Null));
        }
    }

    #[test]
    fn cast_as_text() {
        assert!(matches!(cast_to(Value::Integer(42), Affinity::Text), Value::Text(s) if s == "42"));
        assert!(matches!(cast_to(Value::Real(1.5), Affinity::Text), Value::Text(s) if s == "1.5"));
        assert!(matches!(cast_to(Value::Blob(b"abc".to_vec()), Affinity::Text), Value::Text(s) if s == "abc"));
    }

    #[test]
    fn cast_as_integer_truncates_and_prefixes() {
        // Real truncates toward zero.
        assert!(matches!(cast_to(Value::Real(3.9), Affinity::Integer), Value::Integer(3)));
        assert!(matches!(cast_to(Value::Real(-3.9), Affinity::Integer), Value::Integer(-3)));
        // Text takes the longest integer prefix; the exponent is not part of it.
        assert!(matches!(cast_to(Value::Text("123abc".into()), Affinity::Integer), Value::Integer(123)));
        assert!(matches!(cast_to(Value::Text("123e5".into()), Affinity::Integer), Value::Integer(123)));
        // Hex conversion stops at the 'x' -> 0 (lang_expr.html §13).
        assert!(matches!(cast_to(Value::Text("0x10".into()), Affinity::Integer), Value::Integer(0)));
        assert!(matches!(cast_to(Value::Text("abc".into()), Affinity::Integer), Value::Integer(0)));
        // Overflow clamps.
        assert!(matches!(
            cast_to(Value::Real(1e30), Affinity::Integer),
            Value::Integer(i64::MAX)
        ));
    }

    #[test]
    fn cast_as_real() {
        assert!(matches!(cast_to(Value::Integer(7), Affinity::Real), Value::Real(r) if r == 7.0));
        assert!(matches!(cast_to(Value::Text("1.5abc".into()), Affinity::Real), Value::Real(r) if r == 1.5));
        assert!(matches!(cast_to(Value::Text("abc".into()), Affinity::Real), Value::Real(r) if r == 0.0));
    }

    #[test]
    fn cast_as_numeric_vs_integer_difference() {
        // datatype3.html §3: CAST(4.0 AS INT) == 4, CAST(4.0 AS NUMERIC) == 4.0.
        assert!(matches!(cast_to(Value::Real(4.0), Affinity::Integer), Value::Integer(4)));
        assert!(matches!(cast_to(Value::Real(4.0), Affinity::Numeric), Value::Real(r) if r == 4.0));
        // Text into NUMERIC picks INTEGER or REAL.
        assert!(matches!(cast_to(Value::Text("42".into()), Affinity::Numeric), Value::Integer(42)));
        assert!(matches!(cast_to(Value::Text("4.5".into()), Affinity::Numeric), Value::Real(r) if r == 4.5));
    }

    #[test]
    fn cast_as_blob_roundtrips_text_bytes() {
        assert!(matches!(cast_to(Value::Text("hi".into()), Affinity::Blob), Value::Blob(b) if b == b"hi"));
        assert!(matches!(cast_to(Value::Integer(255), Affinity::Blob), Value::Blob(b) if b == b"255"));
        // Casting a blob to blob is a no-op.
        assert!(matches!(cast_to(Value::Blob(vec![1, 2, 3]), Affinity::Blob), Value::Blob(b) if b == vec![1, 2, 3]));
    }
}
