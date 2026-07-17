//! Conformance battery for `CAST(expr AS type-name)` expressions.
//!
//! Every expected value here is transcribed from the SQLite documentation in
//! `spec/sqlite-doc/`, NEVER from what the engine happens to return:
//!   * `lang_expr.html` §13 "CAST expressions" — the per-affinity conversion table
//!     (REAL truncation toward zero, the text prefix rules, hex stopping at `x`,
//!     BLOB<->TEXT byte reinterpretation, the NUMERIC integer/real decision).
//!   * `datatype3.html` §3 "Type Affinity" — the INTEGER-vs-NUMERIC distinction
//!     (`CAST(4.0 AS INT)` = 4 but `CAST(4.0 AS NUMERIC)` = 4.0) and the worked
//!     example `'3.0e+5'` -> integer 300000; §3.1 the type-name -> affinity rules;
//!     §4.2 text<->numeric conversion.
//!
//! `minisqlite::Value` deliberately has no `PartialEq`, so every comparison goes
//! through the shared harness (`eval_eq` / `value_eq`). If the engine disagrees with
//! a case, the spec-correct assertion STAYS (and fails) — a check is never weakened
//! to match the engine.

mod conformance;

use conformance::*;

// ---- CAST( <expr> AS INTEGER ) ----------------------------------------------

/// §13 INTEGER: "A cast of a REAL value into an INTEGER results in the integer
/// between the REAL value and zero that is closest to the REAL value." That is
/// truncation *toward zero* — 4.9 -> 4 and -4.9 -> -4 (not floor, not round).
#[test]
fn cast_real_to_integer_truncates_toward_zero() {
    eval_eq("CAST(4.9 AS INTEGER)", int(4));
    eval_eq("CAST(-4.9 AS INTEGER)", int(-4));
}

/// §13 INTEGER: "the longest possible prefix of the value that can be interpreted
/// as an integer number is extracted from the TEXT value and the remainder
/// ignored." So `'123'` -> 123 and `'12abc'` -> 12.
#[test]
fn cast_text_to_integer_takes_longest_prefix() {
    eval_eq("CAST('123' AS INTEGER)", int(123));
    eval_eq("CAST('12abc' AS INTEGER)", int(12));
}

/// §13 INTEGER: "If there is no prefix that can be interpreted as an integer
/// number, the result of the conversion is 0." Covers non-numeric and empty text.
#[test]
fn cast_text_to_integer_without_prefix_is_zero() {
    eval_eq("CAST('abc' AS INTEGER)", int(0));
    eval_eq("CAST('' AS INTEGER)", int(0));
}

/// §13 INTEGER: "conversion of hexadecimal integers stops at the 'x' in the '0x'
/// prefix of the hexadecimal integer string and thus result of the CAST is always
/// zero." The only integer prefix of `'0x10'` is the leading `0`, so it casts to 0.
#[test]
fn cast_hex_text_to_integer_stops_at_x() {
    eval_eq("CAST('0x10' AS INTEGER)", int(0));
}

/// §13 INTEGER: "if the text looks like a floating point value with an exponent,
/// the exponent will be ignored because it is no part of the integer prefix. For
/// example, CAST('123e+5' AS INTEGER) results in 123, not in 12300000."
#[test]
fn cast_text_to_integer_ignores_exponent() {
    eval_eq("CAST('123e+5' AS INTEGER)", int(123));
}

/// §13 INTEGER: "Any leading spaces in the TEXT value when converting from TEXT to
/// INTEGER are ignored."
#[test]
fn cast_text_to_integer_ignores_leading_spaces() {
    eval_eq("CAST('   42' AS INTEGER)", int(42));
}

/// §13 INTEGER: "If the prefix integer is greater than +9223372036854775807 then
/// the result of the cast is exactly +9223372036854775807. Similarly, if the prefix
/// integer is less than -9223372036854775808 then the result of the cast is exactly
/// -9223372036854775808." (i64::MAX / i64::MIN.)
#[test]
fn cast_text_to_integer_overflow_clamps_to_i64_bounds() {
    eval_eq("CAST('99999999999999999999999' AS INTEGER)", int(i64::MAX));
    eval_eq("CAST('-99999999999999999999999' AS INTEGER)", int(i64::MIN));
}

/// §13 INTEGER: "If a REAL is greater than the greatest possible signed integer
/// (+9223372036854775807) then the result is the greatest possible signed integer
/// and if the REAL is less than the least possible signed integer
/// (-9223372036854775808) then the result is the least possible signed integer."
/// (1e19 > i64::MAX, so it clamps rather than wrapping.)
#[test]
fn cast_real_to_integer_overflow_clamps_to_i64_bounds() {
    eval_eq("CAST(1e19 AS INTEGER)", int(i64::MAX));
    eval_eq("CAST(-1e19 AS INTEGER)", int(i64::MIN));
}

/// The storage class of an INTEGER cast is `integer` (datatype3 §2 storage classes;
/// `typeof` is defined in lang_corefunc.html). Task-required probe.
#[test]
fn cast_as_integer_typeof_is_integer() {
    eval_eq("typeof(CAST(1.9 AS INTEGER))", text("integer"));
}

// ---- CAST( <expr> AS REAL ) -------------------------------------------------

/// §13 REAL: "the longest possible prefix of the value that can be interpreted as a
/// real number is extracted from the TEXT value and the remainder ignored." A clean
/// literal converts exactly; `'2e2'` is 200.0.
#[test]
fn cast_text_to_real() {
    eval_eq("CAST('1.5' AS REAL)", real(1.5));
    eval_eq("CAST('2e2' AS REAL)", real(200.0));
}

/// §13 REAL: an INTEGER value converts to the corresponding REAL.
#[test]
fn cast_integer_to_real() {
    eval_eq("CAST(3 AS REAL)", real(3.0));
}

/// §13 REAL: "If there is no prefix that can be interpreted as a real number, the
/// result of the conversion is 0.0."
#[test]
fn cast_text_to_real_without_prefix_is_zero() {
    eval_eq("CAST('abc' AS REAL)", real(0.0));
}

/// §13 REAL: "Any leading spaces in the TEXT value are ignored when converting from
/// TEXT to REAL."
#[test]
fn cast_text_to_real_ignores_leading_spaces() {
    eval_eq("CAST('   3.5' AS REAL)", real(3.5));
}

/// The storage class of a REAL cast is `real`. Task-required probe.
#[test]
fn cast_as_real_typeof_is_real() {
    eval_eq("typeof(CAST(3 AS REAL))", text("real"));
}

// ---- CAST( <expr> AS TEXT ) -------------------------------------------------

/// §13 TEXT: "Casting an INTEGER or REAL value into TEXT renders the value as if via
/// sqlite3_snprintf()." An integer renders as plain decimal; a real keeps a decimal
/// point (`1.5` -> "1.5").
#[test]
fn cast_numbers_to_text() {
    eval_eq("CAST(123 AS TEXT)", text("123"));
    eval_eq("CAST(1.5 AS TEXT)", text("1.5"));
}

/// §13 TEXT: a REAL with no fractional part still renders WITH a decimal point, so
/// the text reads back as a real and not an integer: 2.0 -> "2.0", not "2".
#[test]
fn cast_whole_real_to_text_keeps_decimal_point() {
    eval_eq("CAST(2.0 AS TEXT)", text("2.0"));
}

/// The storage class of a TEXT cast is `text`. Task-required probe.
#[test]
fn cast_as_text_typeof_is_text() {
    eval_eq("typeof(CAST(1 AS TEXT))", text("text"));
}

// ---- CAST( <expr> AS BLOB ) -------------------------------------------------

/// §13 NONE affinity (a `BLOB` type-name has no affinity): "Casting to a BLOB
/// consists of first casting the value to TEXT ... then interpreting the resulting
/// byte sequence as a BLOB instead of as TEXT." So `'abc'` and the integer `123`
/// become the UTF-8 bytes of their text form.
#[test]
fn cast_text_and_number_to_blob_reinterprets_utf8_bytes() {
    eval_eq("CAST('abc' AS BLOB)", blob(&[0x61, 0x62, 0x63]));
    eval_eq("CAST(123 AS BLOB)", blob(&[0x31, 0x32, 0x33]));
}

/// The storage class of a BLOB cast is `blob`. Task-required probe.
#[test]
fn cast_as_blob_typeof_is_blob() {
    eval_eq("typeof(CAST('a' AS BLOB))", text("blob"));
}

// ---- CAST( <expr> AS NUMERIC ) vs INTEGER -----------------------------------

/// datatype3 §3: "The difference between INTEGER and NUMERIC affinity is only
/// evident in a CAST expression: The expression CAST(4.0 AS INT) returns an integer
/// 4, whereas CAST(4.0 AS NUMERIC) leaves the value as a floating-point 4.0."
/// §13 NUMERIC: "Casting a REAL or INTEGER value to NUMERIC is a no-op, even if a
/// real value could be losslessly converted to an integer." typeof pins the class,
/// which is the whole point of this distinction.
#[test]
fn cast_real_to_numeric_is_noop_while_integer_truncates() {
    eval_eq("CAST(4.0 AS NUMERIC)", real(4.0));
    eval_eq("CAST(4.0 AS INT)", int(4));
    eval_eq("typeof(CAST(4.0 AS NUMERIC))", text("real"));
    eval_eq("typeof(CAST(4.0 AS INT))", text("integer"));
}

/// datatype3 §3 / §13 NUMERIC: float-form text whose value is exactly an integer
/// (within the lossless margin) becomes INTEGER. The documented worked example is
/// "the string '3.0e+5' is stored in a column with NUMERIC affinity as the integer
/// 300000, not as the floating point value 300000.0."
#[test]
fn cast_numeric_float_text_that_is_integral_yields_integer() {
    eval_eq("CAST('3.0e+5' AS NUMERIC)", int(300000));
    eval_eq("typeof(CAST('3.0e+5' AS NUMERIC))", text("integer"));
}

/// §13 NUMERIC: float-form text with a genuine fractional value stays REAL.
#[test]
fn cast_numeric_float_text_with_fraction_yields_real() {
    eval_eq("CAST('4.5' AS NUMERIC)", real(4.5));
    eval_eq("typeof(CAST('4.5' AS NUMERIC))", text("real"));
}

/// §13 NUMERIC: integer-form text that "is small enough to fit in a 64-bit signed
/// integer" yields INTEGER.
#[test]
fn cast_numeric_integer_text_yields_integer() {
    eval_eq("CAST('123' AS NUMERIC)", int(123));
    eval_eq("typeof(CAST('123' AS NUMERIC))", text("integer"));
}

/// §13 NUMERIC: "Any text input that describes a value outside the range of a 64-bit
/// signed integer yields a REAL result." The class is pinned with typeof (the part
/// the spec guarantees exactly); the magnitude is checked only approximately, with a
/// wide epsilon, so the case does not hinge on the IEEE-rounded f64 of a 20-digit
/// value yet still rejects a wrong-but-real result such as 0.0.
#[test]
fn cast_numeric_out_of_range_integer_text_yields_real() {
    eval_eq("typeof(CAST('99999999999999999999' AS NUMERIC))", text("real"));
    let mut db = mem();
    assert_scalar_approx(
        &mut db,
        "SELECT CAST('99999999999999999999' AS NUMERIC)",
        1e20,
        1e6,
    );
}

// ---- CAST( NULL AS <type> ) -------------------------------------------------

/// §13: "If the value of expr is NULL, then the result of the CAST expression is
/// also NULL." True for every target type-name.
#[test]
fn cast_null_is_null_for_every_target() {
    eval_eq("CAST(NULL AS INTEGER)", null());
    eval_eq("CAST(NULL AS TEXT)", null());
    eval_eq("CAST(NULL AS REAL)", null());
    eval_eq("CAST(NULL AS BLOB)", null());
    eval_eq("CAST(NULL AS NUMERIC)", null());
}

/// typeof of a NULL cast is `null` regardless of the target type-name — all five.
#[test]
fn cast_null_typeof_is_null() {
    eval_eq("typeof(CAST(NULL AS INTEGER))", text("null"));
    eval_eq("typeof(CAST(NULL AS REAL))", text("null"));
    eval_eq("typeof(CAST(NULL AS TEXT))", text("null"));
    eval_eq("typeof(CAST(NULL AS BLOB))", text("null"));
    eval_eq("typeof(CAST(NULL AS NUMERIC))", text("null"));
}

// ---- CAST( <blob literal> AS <type> ) ---------------------------------------

/// §13 TEXT: "To cast a BLOB value to TEXT, the sequence of bytes that make up the
/// BLOB is interpreted as text encoded using the database encoding." The blob
/// x'616263' is the UTF-8 byte sequence of "abc".
#[test]
fn cast_blob_to_text_interprets_bytes_as_text() {
    eval_eq("CAST(x'616263' AS TEXT)", text("abc"));
}

/// §13 INTEGER: "When casting a BLOB value to INTEGER, the value is first converted
/// to TEXT" and the text prefix rule then applies. The blob x'3132' is the bytes of
/// "12", so it casts to the integer 12.
#[test]
fn cast_blob_to_integer_via_text() {
    eval_eq("CAST(x'3132' AS INTEGER)", int(12));
}

/// §13 REAL: "When casting a BLOB value to a REAL, the value is first converted to
/// TEXT." The blob x'312e35' is the bytes of "1.5".
#[test]
fn cast_blob_to_real_via_text() {
    eval_eq("CAST(x'312e35' AS REAL)", real(1.5));
}

/// §13 NUMERIC: a BLOB is first converted to TEXT, then the NUMERIC integer/real
/// decision applies. x'3132' is the bytes of "12" (integer-form) -> INTEGER 12;
/// x'342e35' is "4.5" (fractional float-form) -> REAL 4.5.
#[test]
fn cast_blob_to_numeric_via_text() {
    eval_eq("CAST(x'3132' AS NUMERIC)", int(12));
    eval_eq("CAST(x'342e35' AS NUMERIC)", real(4.5));
}

/// §13 NONE affinity: casting a BLOB to BLOB is a no-op — the raw bytes are returned
/// unchanged, including bytes that are not valid UTF-8 text (there is no text
/// round-trip that could alter them).
#[test]
fn cast_blob_to_blob_is_noop() {
    eval_eq("CAST(x'00ff10' AS BLOB)", blob(&[0x00, 0xff, 0x10]));
}

// ---- CAST over a non-constant (column) operand ------------------------------

/// The §13 rules apply identically when the operand is a value read from a row, not
/// a compile-time constant. This exercises the runtime (row-evaluation) CAST path
/// rather than any constant folding at bind/plan time: a typeless column stores the
/// inserted values as-is (TEXT / NULL), and `CAST(x AS INTEGER)` then takes the
/// longest integer prefix per row ('12abc' -> 12, '3.5' -> 3) and maps NULL -> NULL.
#[test]
fn cast_of_column_value_uses_runtime_path() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x)");
    exec(&mut db, "INSERT INTO t VALUES ('12abc'), ('3.5'), (NULL)");
    assert_rows_unordered(
        &mut db,
        "SELECT CAST(x AS INTEGER) FROM t",
        &[vec![int(12)], vec![int(3)], vec![null()]],
    );
}
