//! Conformance battery: COLUMN TYPE AFFINITY and how it converts values on
//! INSERT, observed through `typeof(col)`.
//!
//! Spec (binding): `spec/sqlite-doc/datatype3.html`
//!   - §3.1   Determination Of Column Affinity (the five ordered rules)
//!   - §3.1.1 Affinity Name Examples (the typename -> affinity table + gotchas)
//!   - §3.4   Column Affinity Behavior Example (the verbatim typeof example)
//!   - §3     prose on how each affinity coerces an inserted value's storage class
//!
//! Every expected value here is TRANSCRIBED FROM THE SPEC, never from what this
//! engine happens to return. A case that fails is therefore a real affinity
//! discrepancy in the engine to fix — it is left as a genuine failing assertion
//! rather than weakened to pass. Assertions run through the `minisqlite` facade.
//!
//! Probe design: the declared column type fixes the column's affinity (§3.1),
//! and the affinity decides how an inserted literal's storage class is converted
//! (§3.4 / §3 prose). So each case declares a column with a chosen type, inserts
//! a literal, and reads back `typeof(col)` (and, where the spec pins the exact
//! converted value, the value itself).

mod conformance;

use conformance::*;
use minisqlite::{Connection, Value};

// ---- local probes ------------------------------------------------------------

/// A fresh in-memory database with `t(c <decl>)` created and the raw SQL literal
/// `lit` inserted into `c`. An empty `decl` yields a column with no declared type
/// (BLOB affinity by §3.1 rule 3). Each call runs in its own database, so the
/// cases are independent and a failure in one does not mask the others.
fn fresh_insert(decl: &str, lit: &str) -> Connection {
    let mut db = mem();
    let coldef = if decl.is_empty() {
        "c".to_string()
    } else {
        format!("c {decl}")
    };
    exec(&mut db, &format!("CREATE TABLE t({coldef})"));
    exec(&mut db, &format!("INSERT INTO t(c) VALUES({lit})"));
    db
}

/// Insert `lit` into a fresh `t(c <decl>)`, then assert `typeof(c)` equals
/// `expected`.
fn typeof_after_insert(decl: &str, lit: &str, expected: &str) {
    let mut db = fresh_insert(decl, lit);
    assert_scalar(&mut db, "SELECT typeof(c) FROM t", text(expected));
}

/// Like [`typeof_after_insert`], but also asserts the stored value equals
/// `expected_value`. Use only where the spec pins the exact converted value
/// (e.g. '3.0e+5' -> integer 300000), not merely the resulting storage class.
fn value_after_insert(decl: &str, lit: &str, expected_typeof: &str, expected_value: Value) {
    let mut db = fresh_insert(decl, lit);
    assert_scalar(&mut db, "SELECT typeof(c) FROM t", text(expected_typeof));
    assert_scalar(&mut db, "SELECT c FROM t", expected_value);
}

// =============================================================================
// §3.4  Column Affinity Behavior Example (verbatim)
// =============================================================================

/// The exact five-column schema from datatype3 §3.4.
const T1_SCHEMA: &str = "CREATE TABLE t1(t TEXT, nu NUMERIC, i INTEGER, r REAL, no BLOB)";

/// The typeof projection used throughout §3.4.
const T1_TYPEOF: &str = "SELECT typeof(t), typeof(nu), typeof(i), typeof(r), typeof(no) FROM t1";

/// The five §3.4 cases as (INSERT statement, expected `typeof` row), transcribed
/// verbatim from datatype3 §3.4. This is the SINGLE SOURCE OF TRUTH for the
/// example: both the sequential test and the isolated per-row tests read their
/// expectations here, so a spec correction changes exactly one place and the two
/// forms can never silently diverge.
const T1_CASES: [(&str, [&str; 5]); 5] = [
    (
        "INSERT INTO t1 VALUES('500.0', '500.0', '500.0', '500.0', '500.0')",
        ["text", "integer", "integer", "real", "text"],
    ),
    (
        "INSERT INTO t1 VALUES(500.0, 500.0, 500.0, 500.0, 500.0)",
        ["text", "integer", "integer", "real", "real"],
    ),
    (
        "INSERT INTO t1 VALUES(500, 500, 500, 500, 500)",
        ["text", "integer", "integer", "real", "integer"],
    ),
    (
        "INSERT INTO t1 VALUES(x'0500', x'0500', x'0500', x'0500', x'0500')",
        ["blob", "blob", "blob", "blob", "blob"],
    ),
    (
        "INSERT INTO t1 VALUES(NULL, NULL, NULL, NULL, NULL)",
        ["null", "null", "null", "null", "null"],
    ),
];

/// Assert the §3.4 `typeof` projection over `t1` equals `expected` (the five
/// storage-class names), reading through the pinned facade.
fn assert_t1_typeof_row(db: &mut Connection, expected: [&str; 5]) {
    let want: Vec<Value> = expected.iter().map(|s| text(s)).collect();
    assert_rows(db, T1_TYPEOF, &[want]);
}

/// Reproduce the §3.4 example exactly: one table, DELETE + re-INSERT before each
/// probe, asserting the documented `typeof` row each time. This uniquely
/// exercises the documented delete-then-reinsert-into-the-same-columns flow.
#[test]
fn section_3_4_verbatim_sequence() {
    let mut db = mem();
    exec(&mut db, T1_SCHEMA);
    for (insert_sql, expected) in T1_CASES {
        exec(&mut db, "DELETE FROM t1");
        exec(&mut db, insert_sql);
        assert_t1_typeof_row(&mut db, expected);
    }
}

// Each §3.4 row again, but in its own isolated table and #[test], so that under
// the parallel test runner every row is verified independently and a single
// failing row reports on its own instead of aborting the sequence above. All
// expectations come from the shared T1_CASES, so the isolated and sequential
// forms cannot silently diverge.

/// Create a fresh `t1`, run the `T1_CASES[idx]` insert, and assert its row.
fn run_t1_case(idx: usize) {
    let mut db = mem();
    exec(&mut db, T1_SCHEMA);
    let (insert_sql, expected) = T1_CASES[idx];
    exec(&mut db, insert_sql);
    assert_t1_typeof_row(&mut db, expected);
}

#[test]
fn section_3_4_row_text_literals() {
    run_t1_case(0); // '500.0' text literals
}

#[test]
fn section_3_4_row_real_literals() {
    run_t1_case(1); // 500.0 real literals
}

#[test]
fn section_3_4_row_integer_literals() {
    run_t1_case(2); // 500 integer literals
}

#[test]
fn section_3_4_row_blob_literals() {
    run_t1_case(3); // x'0500' blob literals
}

#[test]
fn section_3_4_row_null_literals() {
    run_t1_case(4); // NULL literals
}

// =============================================================================
// §3.1 rule 1 + §3.1.1: declared type contains "INT" -> INTEGER affinity.
// Probe: a well-formed integer *text* in an INTEGER-affinity column is
// converted to INTEGER (a TEXT-affinity column would leave it as text).
// =============================================================================

#[test]
fn rule1_int() {
    typeof_after_insert("INT", "'123'", "integer");
}

#[test]
fn rule1_integer() {
    typeof_after_insert("INTEGER", "'123'", "integer");
}

#[test]
fn rule1_tinyint() {
    typeof_after_insert("TINYINT", "'123'", "integer");
}

#[test]
fn rule1_smallint() {
    typeof_after_insert("SMALLINT", "'123'", "integer");
}

#[test]
fn rule1_mediumint() {
    typeof_after_insert("MEDIUMINT", "'123'", "integer");
}

#[test]
fn rule1_bigint() {
    typeof_after_insert("BIGINT", "'123'", "integer");
}

#[test]
fn rule1_unsigned_big_int() {
    typeof_after_insert("UNSIGNED BIG INT", "'123'", "integer");
}

#[test]
fn rule1_int2() {
    typeof_after_insert("INT2", "'123'", "integer");
}

#[test]
fn rule1_int8() {
    typeof_after_insert("INT8", "'123'", "integer");
}

// =============================================================================
// §3.1 rule 2 + §3.1.1: declared type contains "CHAR"/"CLOB"/"TEXT" -> TEXT.
// Probe: an integer literal in a TEXT-affinity column is converted to text
// (a numeric-affinity column would keep it numeric). VARCHAR matches via "CHAR".
// =============================================================================

#[test]
fn rule2_character_20() {
    typeof_after_insert("CHARACTER(20)", "123", "text");
}

#[test]
fn rule2_varchar_255() {
    typeof_after_insert("VARCHAR(255)", "123", "text");
}

#[test]
fn rule2_varying_character_255() {
    typeof_after_insert("VARYING CHARACTER(255)", "123", "text");
}

#[test]
fn rule2_nchar_55() {
    typeof_after_insert("NCHAR(55)", "123", "text");
}

#[test]
fn rule2_native_character_70() {
    typeof_after_insert("NATIVE CHARACTER(70)", "123", "text");
}

#[test]
fn rule2_nvarchar_100() {
    typeof_after_insert("NVARCHAR(100)", "123", "text");
}

#[test]
fn rule2_text() {
    typeof_after_insert("TEXT", "123", "text");
}

#[test]
fn rule2_clob() {
    typeof_after_insert("CLOB", "123", "text");
}

// =============================================================================
// §3.1 rule 3 + §3.1.1: declared type contains "BLOB" or no type -> BLOB
// affinity ("NONE"). A BLOB-affinity column performs NO coercion, so an
// inserted value keeps its original storage class exactly.
// =============================================================================

#[test]
fn rule3_blob_text_unchanged() {
    typeof_after_insert("BLOB", "'123'", "text");
}

#[test]
fn rule3_blob_real_unchanged() {
    typeof_after_insert("BLOB", "1.0", "real");
}

#[test]
fn rule3_blob_blob_passthrough() {
    typeof_after_insert("BLOB", "x'0500'", "blob");
}

#[test]
fn rule3_notype_text_unchanged() {
    typeof_after_insert("", "'123'", "text");
}

#[test]
fn rule3_notype_real_unchanged() {
    typeof_after_insert("", "1.0", "real");
}

#[test]
fn rule3_notype_blob_passthrough() {
    typeof_after_insert("", "x'0500'", "blob");
}

// =============================================================================
// §3.1 rule 4 + §3.1.1: declared type contains "REAL"/"FLOA"/"DOUB" -> REAL.
// Probe: an integer literal in a REAL-affinity column is forced to floating
// point, so typeof reports "real".
// =============================================================================

#[test]
fn rule4_real() {
    typeof_after_insert("REAL", "5", "real");
}

#[test]
fn rule4_double() {
    typeof_after_insert("DOUBLE", "5", "real");
}

#[test]
fn rule4_double_precision() {
    typeof_after_insert("DOUBLE PRECISION", "5", "real");
}

#[test]
fn rule4_float() {
    typeof_after_insert("FLOAT", "5", "real");
}

// =============================================================================
// §3.1 rule 5 + §3.1.1: any other declared type -> NUMERIC affinity.
// Probe: the well-formed real text '500.0' in a NUMERIC-affinity column is
// converted to INTEGER (500.0 is exactly an integer), so typeof reports
// "integer" (a TEXT-affinity column would leave it as text).
// =============================================================================

#[test]
fn rule5_numeric() {
    typeof_after_insert("NUMERIC", "'500.0'", "integer");
}

#[test]
fn rule5_decimal_10_5() {
    typeof_after_insert("DECIMAL(10,5)", "'500.0'", "integer");
}

#[test]
fn rule5_boolean() {
    typeof_after_insert("BOOLEAN", "'500.0'", "integer");
}

#[test]
fn rule5_date() {
    typeof_after_insert("DATE", "'500.0'", "integer");
}

#[test]
fn rule5_datetime() {
    typeof_after_insert("DATETIME", "'500.0'", "integer");
}

// =============================================================================
// §3.1 / §3.1.1 gotchas: rule ORDER and substring matching produce surprising
// affinities. These pin the exact notes from the spec.
// =============================================================================

/// "FLOATING POINT" contains "INT" (in "POINT"), so rule 1 fires before rule 4:
/// INTEGER affinity, NOT REAL. A float exactly representable as an integer is
/// then converted to INTEGER. (datatype3 §3.1.1 note.)
#[test]
fn gotcha_floating_point_is_integer_affinity() {
    typeof_after_insert("FLOATING POINT", "5.0", "integer");
}

/// "STRING" matches none of INT/CHAR/CLOB/TEXT/BLOB/REAL/FLOA/DOUB, so it falls
/// through to rule 5: NUMERIC affinity, NOT TEXT. A well-formed integer text is
/// therefore converted to INTEGER. (datatype3 §3.1.1 note.)
#[test]
fn gotcha_string_is_numeric_affinity() {
    typeof_after_insert("STRING", "'123'", "integer");
}

/// "CHARINT" matches both rule 1 ("INT") and rule 2 ("CHAR"); rule 1 takes
/// precedence, giving INTEGER affinity. (datatype3 §3.1 ordering note.)
#[test]
fn gotcha_charint_is_integer_affinity() {
    typeof_after_insert("CHARINT", "'123'", "integer");
}

// =============================================================================
// §3 conversion prose: exactly how NUMERIC affinity coerces an inserted value.
// =============================================================================

/// "If the TEXT value is a well-formed integer literal that is too large to fit
/// in a 64-bit signed integer, it is converted to REAL." The literal here is
/// i64::MAX + 1 (9223372036854775808 = 2^63), which is exactly representable as
/// an f64, so the correctly-rounded value is exact.
#[test]
fn numeric_wellformed_integer_text_too_large_becomes_real() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(c NUMERIC)");
    exec(&mut db, "INSERT INTO t(c) VALUES('9223372036854775808')");
    assert_scalar(&mut db, "SELECT typeof(c) FROM t", text("real"));
    // 9223372036854775808 = 2^63 = i64::MAX + 1 is exactly representable as an
    // f64, so the correctly-rounded conversion is exact — assert it exactly
    // rather than within a tolerance that would also admit the wrong neighbor
    // 2^63 - 1024.
    assert_scalar(&mut db, "SELECT c FROM t", real(9223372036854775808.0));
}

/// "hexadecimal integer literals are not considered well-formed and are stored
/// as TEXT." The original text is kept verbatim (no conversion).
#[test]
fn numeric_hex_integer_text_stored_as_text() {
    value_after_insert("NUMERIC", "'0x10'", "text", text("0x10"));
}

/// "the string '3.0e+5' is stored in a column with NUMERIC affinity as the
/// integer 300000, not as the floating point value 300000.0."
#[test]
fn numeric_exponent_text_becomes_integer() {
    value_after_insert("NUMERIC", "'3.0e+5'", "integer", int(300000));
}

/// A well-formed real literal that is NOT exactly an integer stays REAL.
#[test]
fn numeric_wellformed_real_text_with_fraction_stays_real() {
    value_after_insert("NUMERIC", "'2.5'", "real", real(2.5));
}

/// "If the TEXT value is not a well-formed integer or real literal, then the
/// value is stored as TEXT."
#[test]
fn numeric_non_numeric_text_stored_as_text() {
    value_after_insert("NUMERIC", "'hello'", "text", text("hello"));
}

/// "If a floating point value that can be represented exactly as an integer is
/// inserted into a column with NUMERIC affinity, the value is converted into an
/// integer." (Mirrors §3.4: 500.0 -> the NUMERIC column nu = integer.)
#[test]
fn numeric_float_exact_integer_becomes_integer() {
    value_after_insert("NUMERIC", "500.0", "integer", int(500));
}

/// An integer literal in a NUMERIC column stays an integer.
#[test]
fn numeric_integer_literal_stays_integer() {
    value_after_insert("NUMERIC", "500", "integer", int(500));
}

// =============================================================================
// §3 prose: INTEGER affinity "behaves the same as NUMERIC affinity" for storage
// (the two differ only inside a CAST expression, which is out of scope here).
// =============================================================================

#[test]
fn integer_affinity_exponent_text_becomes_integer() {
    value_after_insert("INTEGER", "'3.0e+5'", "integer", int(300000));
}

#[test]
fn integer_affinity_too_large_text_becomes_real() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(c INTEGER)");
    exec(&mut db, "INSERT INTO t(c) VALUES('9223372036854775808')");
    assert_scalar(&mut db, "SELECT typeof(c) FROM t", text("real"));
}

// =============================================================================
// §3 prose: REAL affinity behaves like NUMERIC "except that it forces integer
// values into floating point representation." So values a NUMERIC column would
// store as INTEGER become REAL here.
// =============================================================================

#[test]
fn real_affinity_forces_integer_literal_to_real() {
    value_after_insert("REAL", "500", "real", real(500.0));
}

#[test]
fn real_affinity_forces_integer_text_to_real() {
    // NUMERIC would store '500.0' as integer 500; REAL forces it to 500.0.
    value_after_insert("REAL", "'500.0'", "real", real(500.0));
}

#[test]
fn real_affinity_forces_exponent_text_to_real() {
    // NUMERIC would store '3.0e+5' as integer 300000; REAL forces it to 300000.0.
    value_after_insert("REAL", "'3.0e+5'", "real", real(300000.0));
}

#[test]
fn real_affinity_wellformed_real_text_stays_real() {
    value_after_insert("REAL", "'2.5'", "real", real(2.5));
}

// =============================================================================
// §3 prose: TEXT affinity stores data using only NULL, TEXT, or BLOB. Numerical
// data is converted to text form; a BLOB stays a BLOB; NULL stays NULL.
// =============================================================================

/// Integer -> text uses the canonical decimal form, which is unambiguous.
#[test]
fn text_affinity_integer_becomes_text() {
    value_after_insert("TEXT", "123", "text", text("123"));
}

/// A real is converted to text form, but §3.4 pins only the storage class (the
/// exact rendered string is a formatting concern beyond affinity), so assert the
/// class only.
#[test]
fn text_affinity_real_becomes_text() {
    typeof_after_insert("TEXT", "1.5", "text");
}

/// TEXT affinity leaves a BLOB as a BLOB (mirrors §3.4: the TEXT column t with
/// x'0500' -> blob).
#[test]
fn text_affinity_blob_stays_blob() {
    typeof_after_insert("TEXT", "x'0500'", "blob");
}

#[test]
fn text_affinity_null_stays_null() {
    typeof_after_insert("TEXT", "NULL", "null");
}

#[test]
fn text_affinity_text_stays_text() {
    value_after_insert("TEXT", "'abc'", "text", text("abc"));
}
