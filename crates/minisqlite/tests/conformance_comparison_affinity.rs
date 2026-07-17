//! Conformance: type affinity applied to operands BEFORE a comparison.
//!
//! Source of truth: `spec/sqlite-doc/datatype3.html`
//!   §4.2 "Type Conversions Prior To Comparison"
//!   §4.3 "Comparison Example" (transcribed verbatim below)
//!
//! A comparison applies the affinity of one operand to the OTHER operand, so
//! these cases need real table columns whose declared type fixes the affinity
//! (a bare literal or expression has no affinity). The whole point is that the
//! column on the left drives the conversion of the literal on the right.
//!
//! Every expected value here is transcribed from the spec, never from
//! what the engine returns. If the engine disagrees the assertion is kept
//! spec-correct and left to FAIL rather than weakened to pass.

mod conformance;

use conformance::*;
// `Connection` is not re-exported by the harness (it re-exports only its `pub fn`
// helpers), so name it from the facade for the shared table-setup helper.
use minisqlite::Connection;

/// The exact schema and row from datatype3.html §4.3, on a fresh in-memory db.
///
/// ```sql
/// CREATE TABLE t1(
///     a TEXT,      -- text affinity
///     b NUMERIC,   -- numeric affinity
///     c BLOB,      -- no affinity
///     d            -- no affinity
/// );
/// -- Values will be stored as TEXT, INTEGER, TEXT, and INTEGER respectively
/// INSERT INTO t1 VALUES('500', '500', '500', 500);
/// ```
fn t1() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t1(a TEXT, b NUMERIC, c BLOB, d)");
    exec(&mut db, "INSERT INTO t1 VALUES('500', '500', '500', 500)");
    db
}

// ---- §4.3 verbatim: storage classes actually stored -------------------------

/// The declared affinities coerce the inserted values on the way in:
/// `'500'` into a TEXT column stays TEXT; into a NUMERIC column becomes INTEGER;
/// into a BLOB/no-affinity column stays TEXT; the bare `500` literal into the
/// no-affinity `d` stays INTEGER.
///
/// ```sql
/// SELECT typeof(a), typeof(b), typeof(c), typeof(d) FROM t1;
/// text|integer|text|integer
/// ```
#[test]
fn stored_storage_classes_are_text_integer_text_integer() {
    let mut db = t1();
    assert_rows(
        &mut db,
        "SELECT typeof(a), typeof(b), typeof(c), typeof(d) FROM t1",
        &[vec![
            text("text"),
            text("integer"),
            text("text"),
            text("integer"),
        ]],
    );
}

// ---- §4.3 verbatim: column `a`, TEXT affinity -------------------------------

/// "Because column "a" has text affinity, numeric values on the right-hand side
/// of the comparisons are converted to text before the comparison occurs."
///
/// ```sql
/// SELECT a < 40, a < 60, a < 600 FROM t1;
/// 0|1|1
/// ```
/// (`'500'` vs `'40'`/`'60'`/`'600'` under the default BINARY collation.)
#[test]
fn a_text_affinity_numeric_rhs_converted_to_text() {
    let mut db = t1();
    assert_rows(
        &mut db,
        "SELECT a < 40, a < 60, a < 600 FROM t1",
        &[vec![int(0), int(1), int(1)]],
    );
}

/// "Text affinity is applied to the right-hand operands but since they are
/// already TEXT this is a no-op; no conversions occur."
///
/// ```sql
/// SELECT a < '40', a < '60', a < '600' FROM t1;
/// 0|1|1
/// ```
#[test]
fn a_text_affinity_text_rhs_is_a_noop() {
    let mut db = t1();
    assert_rows(
        &mut db,
        "SELECT a < '40', a < '60', a < '600' FROM t1",
        &[vec![int(0), int(1), int(1)]],
    );
}

// ---- §4.3 verbatim: column `b`, NUMERIC affinity ----------------------------

/// "Column "b" has numeric affinity and so numeric affinity is applied to the
/// operands on the right. Since the operands are already numeric, the
/// application of affinity is a no-op; no conversions occur. All values are
/// compared numerically."
///
/// ```sql
/// SELECT b < 40, b < 60, b < 600 FROM t1;
/// 0|0|1
/// ```
#[test]
fn b_numeric_affinity_numeric_rhs_compared_numerically() {
    let mut db = t1();
    assert_rows(
        &mut db,
        "SELECT b < 40, b < 60, b < 600 FROM t1",
        &[vec![int(0), int(0), int(1)]],
    );
}

/// "Numeric affinity is applied to operands on the right, converting them from
/// text to integers. Then a numeric comparison occurs."
///
/// ```sql
/// SELECT b < '40', b < '60', b < '600' FROM t1;
/// 0|0|1
/// ```
#[test]
fn b_numeric_affinity_text_rhs_converted_to_numeric() {
    let mut db = t1();
    assert_rows(
        &mut db,
        "SELECT b < '40', b < '60', b < '600' FROM t1",
        &[vec![int(0), int(0), int(1)]],
    );
}

// ---- §4.3 verbatim: column `c`, BLOB / no affinity --------------------------

/// "No affinity conversions occur. Right-hand side values all have storage
/// class INTEGER which are always less than the TEXT values on the left."
///
/// ```sql
/// SELECT c < 40, c < 60, c < 600 FROM t1;
/// 0|0|0
/// ```
/// (`c` stored the TEXT `'500'`; an INTEGER is always < any TEXT per §4.1, so
/// `c` is never less than an integer literal.)
#[test]
fn c_no_affinity_integer_rhs_always_less_than_text() {
    let mut db = t1();
    assert_rows(
        &mut db,
        "SELECT c < 40, c < 60, c < 600 FROM t1",
        &[vec![int(0), int(0), int(0)]],
    );
}

/// "No affinity conversions occur. Values are compared as TEXT."
///
/// ```sql
/// SELECT c < '40', c < '60', c < '600' FROM t1;
/// 0|1|1
/// ```
#[test]
fn c_no_affinity_text_rhs_compared_as_text() {
    let mut db = t1();
    assert_rows(
        &mut db,
        "SELECT c < '40', c < '60', c < '600' FROM t1",
        &[vec![int(0), int(1), int(1)]],
    );
}

// ---- §4.3 verbatim: column `d`, no affinity (stored INTEGER) ----------------

/// "No affinity conversions occur. Right-hand side values all have storage
/// class INTEGER which compare numerically with the INTEGER values on the
/// left."
///
/// ```sql
/// SELECT d < 40, d < 60, d < 600 FROM t1;
/// 0|0|1
/// ```
#[test]
fn d_no_affinity_integer_stored_numeric_rhs_compared_numerically() {
    let mut db = t1();
    assert_rows(
        &mut db,
        "SELECT d < 40, d < 60, d < 600 FROM t1",
        &[vec![int(0), int(0), int(1)]],
    );
}

/// "No affinity conversions occur. INTEGER values on the left are always less
/// than TEXT values on the right."
///
/// ```sql
/// SELECT d < '40', d < '60', d < '600' FROM t1;
/// 1|1|1
/// ```
#[test]
fn d_no_affinity_integer_stored_text_rhs_always_less() {
    let mut db = t1();
    assert_rows(
        &mut db,
        "SELECT d < '40', d < '60', d < '600' FROM t1",
        &[vec![int(1), int(1), int(1)]],
    );
}

// ---- §4.3 verbatim: commuted forms -----------------------------------------
//
// "All of the results in the example are the same if the comparisons are
// commuted - if expressions of the form "a<40" are rewritten as "40>a"."
//
// So `N > col` must reproduce the `col < N` grid for every column. The affinity
// rule is symmetric in the operands, so the literal on either side is converted
// toward the column's affinity all the same.

#[test]
fn commuted_a_grid_matches_a_lt_grid() {
    let mut db = t1();
    // mirrors `a < 40, a < 60, a < 600` -> 0|1|1
    assert_rows(
        &mut db,
        "SELECT 40 > a, 60 > a, 600 > a FROM t1",
        &[vec![int(0), int(1), int(1)]],
    );
}

#[test]
fn commuted_b_grid_matches_b_lt_grid() {
    let mut db = t1();
    // mirrors `b < 40, b < 60, b < 600` -> 0|0|1
    assert_rows(
        &mut db,
        "SELECT 40 > b, 60 > b, 600 > b FROM t1",
        &[vec![int(0), int(0), int(1)]],
    );
}

#[test]
fn commuted_c_grid_matches_c_lt_grid() {
    let mut db = t1();
    // mirrors `c < 40, c < 60, c < 600` -> 0|0|0
    assert_rows(
        &mut db,
        "SELECT 40 > c, 60 > c, 600 > c FROM t1",
        &[vec![int(0), int(0), int(0)]],
    );
}

#[test]
fn commuted_d_grid_matches_d_lt_grid() {
    let mut db = t1();
    // mirrors `d < 40, d < 60, d < 600` -> 0|0|1
    assert_rows(
        &mut db,
        "SELECT 40 > d, 60 > d, 600 > d FROM t1",
        &[vec![int(0), int(0), int(1)]],
    );
}

// The commute note covers the TEXT-literal grids just as well: `col < 'N'` must
// equal `'N' > col`. These mirror the four `col < '40'/'60'/'600'` grids above.

#[test]
fn commuted_a_text_grid_matches_a_lt_text_grid() {
    let mut db = t1();
    // mirrors `a < '40', a < '60', a < '600'` -> 0|1|1
    assert_rows(
        &mut db,
        "SELECT '40' > a, '60' > a, '600' > a FROM t1",
        &[vec![int(0), int(1), int(1)]],
    );
}

#[test]
fn commuted_b_text_grid_matches_b_lt_text_grid() {
    let mut db = t1();
    // mirrors `b < '40', b < '60', b < '600'` -> 0|0|1
    assert_rows(
        &mut db,
        "SELECT '40' > b, '60' > b, '600' > b FROM t1",
        &[vec![int(0), int(0), int(1)]],
    );
}

#[test]
fn commuted_c_text_grid_matches_c_lt_text_grid() {
    let mut db = t1();
    // mirrors `c < '40', c < '60', c < '600'` -> 0|1|1
    assert_rows(
        &mut db,
        "SELECT '40' > c, '60' > c, '600' > c FROM t1",
        &[vec![int(0), int(1), int(1)]],
    );
}

#[test]
fn commuted_d_text_grid_matches_d_lt_text_grid() {
    let mut db = t1();
    // mirrors `d < '40', d < '60', d < '600'` -> 1|1|1
    assert_rows(
        &mut db,
        "SELECT '40' > d, '60' > d, '600' > d FROM t1",
        &[vec![int(1), int(1), int(1)]],
    );
}

// ---- §4.2 rule coverage: two columns of differing affinity ------------------
//
// §4.2 lists three ordered rules. When both operands are columns, the affinity
// of each still selects which (if any) conversion applies to the other. These
// cases pin one rule each, comparing the columns of `t1` directly (stored:
// a=TEXT'500', b=INTEGER 500, c=TEXT'500', d=INTEGER 500).
//
// On "no affinity": columns `c` (BLOB) and `d` (no declared type) BOTH have BLOB
// affinity (§3.1 determination rule 3). §4.3's own comments label BLOB columns
// "no affinity", and §3.1's historical note records that BLOB affinity was
// literally named "NONE" before being renamed. So for these comparison rules a
// BLOB-affinity operand IS the "no affinity" operand: rule (ii) below applies
// TEXT affinity to the BLOB-affinity column `d`, exactly as it would to a bare
// literal with no affinity — it does not fall through to rule (iii).

/// §4.2 rule (i): "If one operand has INTEGER, REAL or NUMERIC affinity and the
/// other operand has TEXT or BLOB or no affinity then NUMERIC affinity is
/// applied to the other operand."
///
/// `b` has NUMERIC affinity, `c` has no affinity, so NUMERIC affinity is applied
/// to `c`: its stored TEXT `'500'` converts to the integer 500 and the compare
/// is numeric 500 vs 500 -> equal.
///
/// ```sql
/// SELECT b < c, b = c, b > c FROM t1;  -- 0|1|0
/// ```
#[test]
fn rule_i_numeric_affinity_applies_numeric_to_the_other() {
    let mut db = t1();
    assert_rows(
        &mut db,
        "SELECT b < c, b = c, b > c FROM t1",
        &[vec![int(0), int(1), int(0)]],
    );
}

/// §4.2 rule (i), the "or the other operand has TEXT ... affinity" branch:
/// `b` has NUMERIC affinity and `a` has TEXT affinity, so NUMERIC affinity is
/// applied to `a` (not TEXT applied to `b` — rule (i) precedes rule (ii)). `a`'s
/// stored TEXT `'500'` converts to the integer 500 and the compare is numeric
/// 500 vs 500 -> equal. (Contrast `rule_i_..._to_the_other`, which exercises the
/// "or BLOB or no affinity" branch of the same rule.)
///
/// ```sql
/// SELECT b < a, b = a, b > a FROM t1;  -- 0|1|0
/// ```
#[test]
fn rule_i_numeric_affinity_applies_numeric_to_a_text_operand() {
    let mut db = t1();
    assert_rows(
        &mut db,
        "SELECT b < a, b = a, b > a FROM t1",
        &[vec![int(0), int(1), int(0)]],
    );
}

/// §4.2, the "if and only if the conversion does not lose essential
/// information" clause: "TEXT values can be converted into numeric values if the
/// text content is a well-formed integer or real literal". `'abc'` is NOT a
/// well-formed number, so applying `b`'s NUMERIC affinity to it is a no-op — it
/// stays TEXT. The compare is then INTEGER `b` (500) vs TEXT `'abc'`, and per
/// §4.1 an INTEGER is always less than any TEXT, so `b < 'abc'` is true and the
/// two are never equal. (An over-eager converter that coerced `'abc'` to 0 would
/// wrongly yield `b < 'abc'` = 0; this pins that it must not.)
///
/// ```sql
/// SELECT b < 'abc', b = 'abc', b > 'abc' FROM t1;  -- 1|0|0
/// ```
#[test]
fn numeric_affinity_leaves_non_wellformed_text_unconverted() {
    let mut db = t1();
    assert_rows(
        &mut db,
        "SELECT b < 'abc', b = 'abc', b > 'abc' FROM t1",
        &[vec![int(1), int(0), int(0)]],
    );
}

/// §4.2 rule (ii): "If one operand has TEXT affinity and the other has no
/// affinity, then TEXT affinity is applied to the other operand."
///
/// `a` has TEXT affinity, `d` has no affinity (and neither has numeric affinity,
/// so rule (i) does not fire). TEXT affinity is applied to `d`: its stored
/// integer 500 converts to `'500'` and the compare is TEXT `'500'` vs `'500'`
/// -> equal.
///
/// ```sql
/// SELECT a < d, a = d, a > d FROM t1;  -- 0|1|0
/// ```
#[test]
fn rule_ii_text_affinity_applies_text_to_the_other() {
    let mut db = t1();
    assert_rows(
        &mut db,
        "SELECT a < d, a = d, a > d FROM t1",
        &[vec![int(0), int(1), int(0)]],
    );
}

/// §4.2 rule (iii): "Otherwise, no affinity is applied and both operands are
/// compared as is."
///
/// `c` and `d` both have no affinity, so no conversion happens. `c` is stored
/// TEXT `'500'` and `d` is stored INTEGER 500; per §4.1 an INTEGER is always
/// less than any TEXT, so `d < c` (equivalently `c > d`) and the two are never
/// equal (different storage classes).
///
/// ```sql
/// SELECT c < d, c = d, c > d FROM t1;  -- 0|0|1
/// ```
#[test]
fn rule_iii_no_affinity_compared_as_is() {
    let mut db = t1();
    assert_rows(
        &mut db,
        "SELECT c < d, c = d, c > d FROM t1",
        &[vec![int(0), int(0), int(1)]],
    );
}
