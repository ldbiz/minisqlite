//! Conformance battery: SQLite **literal values** (constants).
//!
//! Every assertion here is transcribed from the SQLite documentation, never from
//! whatever the engine currently returns — a failing case is the intended signal
//! that the engine diverges from the spec.
//!
//! Spec sources:
//! - `spec/sqlite-doc/lang_expr.html` §3 "Literal Values (Constants)": the numeric
//!   float-vs-integer rule (a numeric literal is floating point if it has a decimal
//!   point or an exponent, or if it is `< -9223372036854775808` or
//!   `> 9223372036854775807`; otherwise an integer), hexadecimal `0x`/`0X` integer
//!   literals (64-bit two's complement, added 3.8.6), single-quoted strings with
//!   `''` escaping a quote, `x'..'`/`X'..'` BLOB literals, and the `NULL` token.
//! - `spec/sqlite-doc/datatype3.html` §2 (the five storage classes reported by
//!   `typeof()`: null/integer/real/text/blob) and §2.1 (Boolean: `TRUE`/`FALSE`
//!   are alternate spellings for the integer literals `1` and `0`).
//! - `spec/sqlite-doc/lang_createtable.html`: `CURRENT_TIME`/`CURRENT_DATE`/
//!   `CURRENT_TIMESTAMP` evaluate to a *text* representation of the current UTC
//!   time, so their storage class is `text`.
//!
//! Each case is its own `#[test]` (one assertion) so an unsupported literal fails
//! exactly that case rather than masking the rest.

mod conformance;
use conformance::*;

// `Value` is needed for the one direct storage-class match (numeric overflow ->
// REAL); every other case goes through the harness `eval_eq` value comparison.
use minisqlite::Value;

// ---------------------------------------------------------------------------
// Integer literals — lang_expr.html §3; datatype3.html §2.
// A numeric literal with no decimal point, no exponent, and in i64 range is an
// integer literal of storage class INTEGER.
// ---------------------------------------------------------------------------

#[test]
fn integer_literal_one() {
    eval_eq("1", int(1));
}

#[test]
fn integer_literal_zero() {
    eval_eq("0", int(0));
}

#[test]
fn integer_literal_multidigit() {
    eval_eq("123", int(123));
}

#[test]
fn integer_literal_max_i64() {
    // 9223372036854775807 == i64::MAX is still an integer literal (not > MAX).
    eval_eq("9223372036854775807", int(9223372036854775807));
}

#[test]
fn integer_literal_typeof_is_integer() {
    eval_eq("typeof(1)", text("integer"));
}

#[test]
fn integer_literal_underscore_separator() {
    // lang_expr.html §3 (SQLite 3.46.0): a single "_" between two digits is
    // allowed purely for readability and ignored, so 1_000 is the integer 1000.
    eval_eq("1_000", int(1000));
}

// ---------------------------------------------------------------------------
// Real (floating point) literals — lang_expr.html §3; datatype3.html §2.
// A numeric literal with a decimal point or an exponent is a floating point
// literal of storage class REAL. The "E" of the exponent is case-insensitive.
// All expected values below are exactly representable in IEEE-754 f64.
// ---------------------------------------------------------------------------

#[test]
fn real_literal_fraction() {
    eval_eq("1.5", real(1.5));
}

#[test]
fn real_literal_trailing_zero_value() {
    // 1.0 has a decimal point, so it is REAL even though it is integral.
    eval_eq("1.0", real(1.0));
}

#[test]
fn real_literal_trailing_zero_typeof_is_real() {
    eval_eq("typeof(1.0)", text("real"));
}

#[test]
fn real_literal_leading_dot() {
    eval_eq(".5", real(0.5));
}

#[test]
fn real_literal_exponent_lowercase_e() {
    eval_eq("1e3", real(1000.0));
}

#[test]
fn real_literal_exponent_uppercase_e() {
    // lang_expr.html §3: the "E" that begins the exponent may be upper or lower case.
    eval_eq("1E3", real(1000.0));
}

#[test]
fn real_literal_fraction_with_exponent() {
    eval_eq("2.0e2", real(200.0));
}

#[test]
fn real_literal_typeof_is_real() {
    eval_eq("typeof(2.5)", text("real"));
}

// ---------------------------------------------------------------------------
// Hexadecimal integer literals — lang_expr.html §3 (added SQLite 3.8.6).
// "0x"/"0X" followed by hex digits; interpreted as 64-bit two's complement.
// The spec gives 0x1234 == 4660 as an example. Storage class is INTEGER.
// ---------------------------------------------------------------------------

#[test]
fn hex_literal_0x10_is_16() {
    eval_eq("0x10", int(16));
}

#[test]
fn hex_literal_0xff_is_255() {
    eval_eq("0xff", int(255));
}

#[test]
fn hex_literal_uppercase_prefix() {
    eval_eq("0X1", int(1));
}

#[test]
fn hex_literal_spec_example_0x1234() {
    eval_eq("0x1234", int(4660));
}

#[test]
fn hex_literal_uppercase_digits() {
    eval_eq("0xFF", int(255));
}

#[test]
fn hex_literal_high_bit_is_min_i64() {
    // lang_expr.html §3 example: 0x8000000000000000 == -9223372036854775808,
    // i.e. the sign bit makes it i64::MIN under two's-complement interpretation.
    eval_eq("0x8000000000000000", int(i64::MIN));
}

#[test]
fn hex_literal_all_ones_is_minus_one() {
    // Sixteen hex digits of all-ones is -1 in 64-bit two's complement.
    eval_eq("0xffffffffffffffff", int(-1));
}

#[test]
fn hex_literal_typeof_is_integer() {
    eval_eq("typeof(0x1)", text("integer"));
}

// ---------------------------------------------------------------------------
// String literals — lang_expr.html §3; datatype3.html §2.
// Single quotes; a single quote inside the string is written as two single
// quotes. C-style backslash escapes are NOT supported. Storage class is TEXT.
// ---------------------------------------------------------------------------

#[test]
fn string_literal_plain() {
    eval_eq("'abc'", text("abc"));
}

#[test]
fn string_literal_escaped_quote() {
    // '' inside the quotes encodes one literal single quote.
    eval_eq("'it''s'", text("it's"));
}

#[test]
fn string_literal_empty() {
    eval_eq("''", text(""));
}

#[test]
fn string_literal_only_escaped_quote() {
    // '''' is: open-quote, '' (one escaped quote), close-quote -> the text `'`.
    eval_eq("''''", text("'"));
}

#[test]
fn string_literal_typeof_is_text() {
    eval_eq("typeof('x')", text("text"));
}

// ---------------------------------------------------------------------------
// BLOB literals — lang_expr.html §3; datatype3.html §2.
// x'..' / X'..' with an even number of hex digits; each byte is one hex pair.
// Storage class is BLOB.
// ---------------------------------------------------------------------------

#[test]
fn blob_literal_single_byte() {
    eval_eq("x'41'", blob(&[0x41]));
}

#[test]
fn blob_literal_two_bytes_with_zero() {
    eval_eq("x'0500'", blob(&[0x05, 0x00]));
}

#[test]
fn blob_literal_uppercase_prefix() {
    eval_eq("X'4a'", blob(&[0x4a]));
}

#[test]
fn blob_literal_uppercase_hex_digits() {
    eval_eq("X'FF'", blob(&[0xff]));
}

#[test]
fn blob_literal_empty() {
    // Zero hex digits is a valid, empty BLOB.
    eval_eq("x''", blob(&[]));
}

#[test]
fn blob_literal_typeof_is_blob() {
    eval_eq("typeof(x'41')", text("blob"));
}

// ---------------------------------------------------------------------------
// NULL literal — lang_expr.html §3; datatype3.html §2.
// The bare token NULL is the NULL storage class. Keywords are case-insensitive.
// ---------------------------------------------------------------------------

#[test]
fn null_literal_value() {
    eval_eq("NULL", null());
}

#[test]
fn null_literal_lowercase() {
    eval_eq("null", null());
}

#[test]
fn null_literal_typeof_is_null() {
    eval_eq("typeof(NULL)", text("null"));
}

// ---------------------------------------------------------------------------
// Boolean keywords — datatype3.html §2.1.
// SQLite has no Boolean storage class: TRUE and FALSE are alternate spellings
// for the integer literals 1 and 0, so their storage class is INTEGER.
// ---------------------------------------------------------------------------

#[test]
fn boolean_true_is_integer_one() {
    eval_eq("TRUE", int(1));
}

#[test]
fn boolean_false_is_integer_zero() {
    eval_eq("FALSE", int(0));
}

#[test]
fn boolean_true_lowercase() {
    eval_eq("true", int(1));
}

#[test]
fn boolean_false_lowercase() {
    eval_eq("false", int(0));
}

#[test]
fn boolean_true_typeof_is_integer() {
    eval_eq("typeof(TRUE)", text("integer"));
}

#[test]
fn boolean_false_typeof_is_integer() {
    eval_eq("typeof(FALSE)", text("integer"));
}

// ---------------------------------------------------------------------------
// Numeric literal too large for i64 -> REAL — lang_expr.html §3.
// A numeric literal greater than 9223372036854775807 is a floating point literal,
// so it is stored as REAL rather than overflowing or erroring.
// ---------------------------------------------------------------------------

#[test]
fn numeric_overflow_typeof_is_real() {
    eval_eq("typeof(99999999999999999999)", text("real"));
}

#[test]
fn numeric_overflow_value_is_real() {
    // Direct storage-class check: the value is REAL, not INTEGER.
    let v = eval("99999999999999999999");
    assert!(matches!(v, Value::Real(_)), "expected REAL storage class, got {v:?}");
}

// ---------------------------------------------------------------------------
// CURRENT_DATE / CURRENT_TIME / CURRENT_TIMESTAMP — lang_expr.html §3 literal
// tokens; lang_createtable.html: each yields a text representation of the
// current UTC time, so the storage class is TEXT. The value is time-dependent,
// so only the storage class is asserted (not the exact string).
// ---------------------------------------------------------------------------

#[test]
fn current_date_typeof_is_text() {
    eval_eq("typeof(CURRENT_DATE)", text("text"));
}

#[test]
fn current_time_typeof_is_text() {
    eval_eq("typeof(CURRENT_TIME)", text("text"));
}

#[test]
fn current_timestamp_typeof_is_text() {
    eval_eq("typeof(CURRENT_TIMESTAMP)", text("text"));
}

// ---------------------------------------------------------------------------
// Malformed literals must be rejected — lang_expr.html §3.
// A BLOB literal is hexadecimal data, so it needs an even count of hex digits
// (`x'4'` is odd) made only of hex characters (`x'zz'` is not). A hex integer
// literal is "0x"/"0X" *followed by* hexadecimal digits, so a bare "0x" with no
// digits is not a valid literal. The `'` in a BLOB literal forces blob-literal
// tokenization, so these cannot be reinterpreted as an identifier/alias.
// ---------------------------------------------------------------------------

#[test]
fn blob_literal_odd_hex_digit_count_errors() {
    let mut db = mem();
    assert_query_error(&mut db, "SELECT x'4'");
}

#[test]
fn blob_literal_non_hex_digit_errors() {
    let mut db = mem();
    assert_query_error(&mut db, "SELECT x'zz'");
}

#[test]
fn hex_literal_bare_prefix_errors() {
    let mut db = mem();
    assert_query_error(&mut db, "SELECT 0x");
}
