//! Conformance battery for SQL subquery expressions.
//!
//! Every expected value here is transcribed from the SQLite documentation, NOT
//! from what this engine happens to return. The binding sources are:
//!
//!   * `spec/sqlite-doc/lang_expr.html` §8 "The IN and NOT IN operators" — the
//!     right operand may be a subquery, and the result (including the NULL cases)
//!     is fixed by the documented matrix (transcribed in full below).
//!   * `spec/sqlite-doc/lang_expr.html` §10 "The EXISTS operator" — "always
//!     evaluates to one of the integer values 0 and 1"; 1 if the SELECT would
//!     return one or more rows, else 0. "rows containing NULL values are not
//!     handled any differently from rows without NULL values."
//!   * `spec/sqlite-doc/lang_expr.html` §11 "Subquery Expressions" — "The value
//!     of a subquery expression is the first row of the result from the enclosed
//!     SELECT statement. The value of a subquery expression is NULL if the
//!     enclosed SELECT statement returns no rows."
//!   * `spec/sqlite-doc/lang_expr.html` §12 "Correlated Subqueries" — a subquery
//!     may reference outer columns and "is reevaluated each time its result is
//!     required."
//!   * `spec/sqlite-doc/lang_select.html` — a `table-or-subquery` in FROM may be
//!     a parenthesized SELECT (a derived table), optionally aliased; its result
//!     columns take their names from the inner SELECT (an `AS` alias names them).
//!
//! The §8 IN / NOT IN result matrix (lang_expr.html §8), adapted from the spec
//! table (column headers abbreviated; the spec's "does not matter" shown as
//! "(any)"), with every row and result value preserved:
//!
//!   left NULL | right has NULL | right empty | left found | IN    | NOT IN
//!   ----------+---------------+-------------+------------+-------+-------
//!    no       | no            | no          | no         | false | true
//!    (any)    | no            | yes         | no         | false | true
//!    no       | (any)         | no          | yes        | true  | false
//!    no       | yes           | no          | no         | NULL  | NULL
//!    yes      | (any)         | no          | (any)      | NULL  | NULL
//!
//! Plus (§8): "When the right operand is an empty set, the result of IN is false
//! and the result of NOT IN is true, regardless of the left operand and even if
//! the left operand is NULL."
//!
//! In SQLite booleans ARE integers: TRUE = 1, FALSE = 0, UNKNOWN = NULL. A WHERE
//! clause keeps a row only when the predicate is TRUE; both FALSE and NULL drop
//! it. `Value` has no `PartialEq`, so every assertion goes through the shared
//! harness; never compare a `Value` with `==`.
//!
//! If a case fails because the engine disagrees, the assertion STAYS spec-correct
//! (it fails) rather than being weakened to pass.

mod conformance;

use conformance::*;
use minisqlite::{Connection, Value};

// ---------------------------------------------------------------------------
// Fixture. A fresh in-memory database per test keeps them independent and
// deterministic. Multi-row VALUES is exercised by the smoke set, so it is a safe
// setup primitive here.
//
//   t(id, v): (1,10), (2,20), (3,30), (4,40)
//   s(id):    2, 4                       (no NULLs)
//   sn(id):   2, NULL                    (one NULL — for the §8 NULL matrix)
// ---------------------------------------------------------------------------

fn fixture() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER, v INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30), (4, 40)");
    exec(&mut db, "CREATE TABLE s(id INTEGER)");
    exec(&mut db, "INSERT INTO s VALUES (2), (4)");
    exec(&mut db, "CREATE TABLE sn(id INTEGER)");
    exec(&mut db, "INSERT INTO sn VALUES (2), (NULL)");
    db
}

/// A single-column result set of integers, e.g. `ids(&[2, 4])` for `[[2], [4]]`.
/// `ids(&[])` is the empty result set.
fn ids(xs: &[i64]) -> Vec<Vec<Value>> {
    xs.iter().map(|&x| vec![int(x)]).collect()
}

// ===========================================================================
// §11 — Scalar subquery in the SELECT list (uncorrelated).
// "The value of a subquery expression is the first row of the result."
// ===========================================================================

#[test]
fn scalar_subquery_in_select_aggregate_max() {
    let mut db = fixture();
    // max(v) over t is 40; the scalar subquery yields that single value.
    assert_scalar(&mut db, "SELECT (SELECT max(v) FROM t)", int(40));
}

#[test]
fn scalar_subquery_in_select_count_all() {
    let mut db = fixture();
    assert_scalar(&mut db, "SELECT (SELECT count(*) FROM t)", int(4));
}

#[test]
fn scalar_subquery_uncorrelated_repeats_per_outer_row() {
    let mut db = fixture();
    // `(SELECT count(*) FROM s)` does not reference the outer row, so it is
    // evaluated once (§12) and the same value (2) appears on every t row.
    assert_rows(
        &mut db,
        "SELECT id, (SELECT count(*) FROM s) AS c FROM t ORDER BY id",
        &[
            vec![int(1), int(2)],
            vec![int(2), int(2)],
            vec![int(3), int(2)],
            vec![int(4), int(2)],
        ],
    );
}

#[test]
fn scalar_subquery_alias_names_the_column() {
    let mut db = fixture();
    // lang_select.html: a bare column reference keeps its name ("id"); an `AS`
    // alias names the subquery column ("c").
    assert_columns(
        &mut db,
        "SELECT id, (SELECT count(*) FROM s) AS c FROM t",
        &["id", "c"],
    );
}

// §11 — "the FIRST row of the result": pin it with a deterministic ordering.

#[test]
fn scalar_subquery_single_matching_row() {
    let mut db = fixture();
    assert_scalar(&mut db, "SELECT (SELECT v FROM t WHERE v = 30)", int(30));
}

#[test]
fn scalar_subquery_first_row_of_ordered_result() {
    let mut db = fixture();
    // ORDER BY v ASC LIMIT 1 -> the smallest v (10) is the "first row".
    assert_scalar(&mut db, "SELECT (SELECT v FROM t ORDER BY v LIMIT 1)", int(10));
    // ORDER BY v DESC LIMIT 1 -> the largest (40).
    assert_scalar(&mut db, "SELECT (SELECT v FROM t ORDER BY v DESC LIMIT 1)", int(40));
}

// §11 — "NULL if the enclosed SELECT statement returns no rows."

#[test]
fn scalar_subquery_no_rows_is_null() {
    let mut db = fixture();
    // s has no id = 99, so the subquery returns no rows -> the value is NULL.
    assert_scalar(&mut db, "SELECT (SELECT id FROM s WHERE id = 99)", null());
}

#[test]
fn scalar_subquery_null_propagates_through_arithmetic() {
    let mut db = fixture();
    // The empty subquery is NULL; NULL + 1 is NULL (lang_expr.html §2: an
    // arithmetic operator with a NULL operand yields NULL).
    assert_scalar(&mut db, "SELECT (SELECT id FROM s WHERE id = 99) + 1", null());
}

// ===========================================================================
// §11 / §12 — Scalar subquery in WHERE.
// ===========================================================================

#[test]
fn scalar_subquery_in_where_equals_max() {
    let mut db = fixture();
    // v = max(v) = 40 -> only id 4.
    assert_rows(&mut db, "SELECT id FROM t WHERE v = (SELECT max(v) FROM t)", &ids(&[4]));
}

#[test]
fn scalar_subquery_in_where_equals_min() {
    let mut db = fixture();
    assert_rows(&mut db, "SELECT id FROM t WHERE v = (SELECT min(v) FROM t)", &ids(&[1]));
}

#[test]
fn scalar_subquery_in_where_greater_than_avg() {
    let mut db = fixture();
    // avg(v) over {10,20,30,40} is the REAL 25.0 (SQLite's avg always yields a
    // float); v > 25.0 keeps {30,40} -> ids 3,4, an integer-vs-REAL numeric
    // comparison.
    assert_rows(
        &mut db,
        "SELECT id FROM t WHERE v > (SELECT avg(v) FROM t) ORDER BY id",
        &ids(&[3, 4]),
    );
    // Pin the REAL storage class the comment relies on: from the row result
    // alone, integer 25 and real 25.0 are indistinguishable (both give {30,40}).
    assert_scalar_approx(&mut db, "SELECT avg(v) FROM t", 25.0, 1e-9);
}

// ===========================================================================
// §8 — IN / NOT IN with a subquery (no NULLs involved: plain set membership).
// ===========================================================================

#[test]
fn in_subquery_selects_matches() {
    let mut db = fixture();
    // t.id in {2,4} -> [2,4].
    assert_rows(
        &mut db,
        "SELECT id FROM t WHERE id IN (SELECT id FROM s) ORDER BY id",
        &ids(&[2, 4]),
    );
}

#[test]
fn not_in_subquery_selects_non_matches() {
    let mut db = fixture();
    // t.id not in {2,4} -> [1,3] (no NULLs anywhere, so NOT IN is well-defined).
    assert_rows(
        &mut db,
        "SELECT id FROM t WHERE id NOT IN (SELECT id FROM s) ORDER BY id",
        &ids(&[1, 3]),
    );
}

// ===========================================================================
// §8 — the IN / NOT IN result matrix, as bare 0/1/NULL scalars.
// ===========================================================================

// Row 3: left non-NULL and found -> IN true, NOT IN false.
#[test]
fn in_found_is_true() {
    let mut db = fixture();
    assert_scalar(&mut db, "SELECT 2 IN (SELECT id FROM s)", int(1));
}

#[test]
fn not_in_found_is_false() {
    let mut db = fixture();
    assert_scalar(&mut db, "SELECT 2 NOT IN (SELECT id FROM s)", int(0));
}

// Row 1: left non-NULL, right has no NULL, not empty, not found -> IN false,
// NOT IN true.
#[test]
fn in_not_found_no_null_is_false() {
    let mut db = fixture();
    assert_scalar(&mut db, "SELECT 5 IN (SELECT id FROM s)", int(0));
}

#[test]
fn not_in_not_found_no_null_is_true() {
    let mut db = fixture();
    assert_scalar(&mut db, "SELECT 5 NOT IN (SELECT id FROM s)", int(1));
}

// Row 2 / the "empty set" rule: IN is false and NOT IN is true regardless of the
// left operand — including when the left operand is NULL.
#[test]
fn in_empty_set_is_false() {
    let mut db = fixture();
    assert_scalar(&mut db, "SELECT 5 IN (SELECT id FROM s WHERE id = 99)", int(0));
}

#[test]
fn not_in_empty_set_is_true() {
    let mut db = fixture();
    assert_scalar(&mut db, "SELECT 5 NOT IN (SELECT id FROM s WHERE id = 99)", int(1));
}

#[test]
fn in_empty_set_null_left_is_false() {
    let mut db = fixture();
    // Empty set: even a NULL left operand gives IN false (not NULL).
    assert_scalar(&mut db, "SELECT NULL IN (SELECT id FROM s WHERE id = 99)", int(0));
}

#[test]
fn not_in_empty_set_null_left_is_true() {
    let mut db = fixture();
    assert_scalar(&mut db, "SELECT NULL NOT IN (SELECT id FROM s WHERE id = 99)", int(1));
}

// Row 5: left operand NULL, right non-empty -> IN NULL, NOT IN NULL.
#[test]
fn in_null_left_nonempty_is_null() {
    let mut db = fixture();
    assert_scalar(&mut db, "SELECT NULL IN (SELECT id FROM s)", null());
}

#[test]
fn not_in_null_left_nonempty_is_null() {
    let mut db = fixture();
    assert_scalar(&mut db, "SELECT NULL NOT IN (SELECT id FROM s)", null());
}

// Row 4: left non-NULL, right contains NULL, left not found -> IN NULL, NOT IN
// NULL. (sn = {2, NULL}.)
#[test]
fn in_right_has_null_not_found_is_null() {
    let mut db = fixture();
    assert_scalar(&mut db, "SELECT 1 IN (SELECT id FROM sn)", null());
}

#[test]
fn not_in_right_has_null_not_found_is_null() {
    let mut db = fixture();
    assert_scalar(&mut db, "SELECT 1 NOT IN (SELECT id FROM sn)", null());
}

// Row 3 again, but with a NULL present in the right set: a definite match wins
// over the NULL ("right has NULL" does not matter once the value is found).
#[test]
fn in_right_has_null_found_is_true() {
    let mut db = fixture();
    assert_scalar(&mut db, "SELECT 2 IN (SELECT id FROM sn)", int(1));
}

#[test]
fn not_in_right_has_null_found_is_false() {
    let mut db = fixture();
    assert_scalar(&mut db, "SELECT 2 NOT IN (SELECT id FROM sn)", int(0));
}

// ===========================================================================
// §8 — NOT IN against a subquery that contains a NULL, applied in WHERE.
// The headline case: because the set contains NULL, NOT IN is never TRUE for any
// row (it is FALSE for the matching row and NULL for the rest), so the result is
// empty. Its IN complement keeps only the rows that DEFINITELY match.
// ===========================================================================

#[test]
fn not_in_subquery_with_null_yields_empty_result() {
    let mut db = fixture();
    // id=2 -> NOT IN false (found); id in {1,3,4} -> NOT IN NULL. None is TRUE.
    assert_rows(&mut db, "SELECT id FROM t WHERE id NOT IN (SELECT id FROM sn)", &ids(&[]));
}

#[test]
fn in_subquery_with_null_keeps_only_definite_match() {
    let mut db = fixture();
    // id=2 -> IN true; id in {1,3,4} -> IN NULL (unknown, dropped by WHERE).
    assert_rows(
        &mut db,
        "SELECT id FROM t WHERE id IN (SELECT id FROM sn) ORDER BY id",
        &ids(&[2]),
    );
}

// ===========================================================================
// §10 / §12 — EXISTS and NOT EXISTS, correlated to the outer row.
// ===========================================================================

#[test]
fn exists_correlated_selects_rows_with_a_match() {
    let mut db = fixture();
    // EXISTS is true for the t rows whose id appears in s ({2,4}).
    assert_rows(
        &mut db,
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM s WHERE s.id = t.id) ORDER BY id",
        &ids(&[2, 4]),
    );
}

#[test]
fn not_exists_correlated_selects_rows_without_a_match() {
    let mut db = fixture();
    assert_rows(
        &mut db,
        "SELECT id FROM t WHERE NOT EXISTS (SELECT 1 FROM s WHERE s.id = t.id) ORDER BY id",
        &ids(&[1, 3]),
    );
}

// ===========================================================================
// §10 — EXISTS as a bare scalar: always 0 or 1, and NULL rows are not special.
// ===========================================================================

#[test]
fn exists_over_nonempty_is_one() {
    let mut db = fixture();
    assert_scalar(&mut db, "SELECT EXISTS (SELECT 1 FROM s)", int(1));
}

#[test]
fn exists_over_empty_is_zero() {
    let mut db = fixture();
    assert_scalar(&mut db, "SELECT EXISTS (SELECT 1 FROM s WHERE id = 99)", int(0));
}

#[test]
fn not_exists_over_empty_is_one() {
    let mut db = fixture();
    assert_scalar(&mut db, "SELECT NOT EXISTS (SELECT 1 FROM s WHERE id = 99)", int(1));
}

#[test]
fn exists_counts_a_row_that_is_null() {
    let mut db = fixture();
    // The subquery returns one row, whose only value is NULL. §10: "rows
    // containing NULL values are not handled any differently" -> EXISTS is 1.
    assert_scalar(&mut db, "SELECT EXISTS (SELECT id FROM sn WHERE id IS NULL)", int(1));
}

#[test]
fn exists_over_single_null_row_is_one() {
    // `SELECT NULL` produces exactly one row (containing NULL); EXISTS -> 1. This
    // needs no table, so it uses the no-FROM `eval` helper.
    eval_eq("EXISTS (SELECT NULL)", int(1));
}

// ===========================================================================
// §12 — Correlated scalar subquery: reevaluated per outer row, reads outer cols.
// ===========================================================================

#[test]
fn correlated_scalar_count_per_outer_row() {
    let mut db = fixture();
    // For each t row, count the s rows with s.id = t.id: 0,1,0,1.
    assert_rows(
        &mut db,
        "SELECT id, (SELECT count(*) FROM s WHERE s.id = t.id) FROM t ORDER BY id",
        &[
            vec![int(1), int(0)],
            vec![int(2), int(1)],
            vec![int(3), int(0)],
            vec![int(4), int(1)],
        ],
    );
}

#[test]
fn correlated_scalar_used_as_where_predicate() {
    let mut db = fixture();
    // Keep the t rows for which the correlated count is positive -> {2,4}.
    assert_rows(
        &mut db,
        "SELECT id FROM t WHERE (SELECT count(*) FROM s WHERE s.id = t.id) > 0 ORDER BY id",
        &ids(&[2, 4]),
    );
}

#[test]
fn correlated_scalar_reads_outer_column() {
    let mut db = fixture();
    // `(SELECT t.v)` is a correlated scalar subquery (no FROM) that reads the
    // outer column, so it reproduces each row's v.
    assert_rows(
        &mut db,
        "SELECT (SELECT t.v) FROM t ORDER BY t.v",
        &ids(&[10, 20, 30, 40]),
    );
}

// ===========================================================================
// lang_select.html — Subquery in FROM (derived table).
// ===========================================================================

#[test]
fn derived_table_projects_aliased_column() {
    let mut db = fixture();
    // The derived table exposes column x = v for the rows with v >= 30.
    assert_rows(
        &mut db,
        "SELECT x FROM (SELECT v AS x FROM t WHERE v >= 30) ORDER BY x",
        &ids(&[30, 40]),
    );
}

#[test]
fn derived_table_count_of_distinct() {
    let mut db = fixture();
    // The inner DISTINCT over {10,20,30,40} yields four rows; the outer counts them.
    assert_scalar(&mut db, "SELECT count(*) FROM (SELECT DISTINCT v FROM t)", int(4));
}

#[test]
fn derived_table_empty_counts_zero() {
    let mut db = fixture();
    // No v > 100, so the derived table is empty and count(*) is 0.
    assert_scalar(&mut db, "SELECT count(*) FROM (SELECT v FROM t WHERE v > 100)", int(0));
}

#[test]
fn derived_table_outer_where_filters_alias() {
    let mut db = fixture();
    // The outer WHERE references the derived column x -> {30,40}.
    assert_rows(
        &mut db,
        "SELECT x FROM (SELECT v AS x FROM t) WHERE x > 20 ORDER BY x",
        &ids(&[30, 40]),
    );
}

#[test]
fn derived_table_of_aggregate_single_row() {
    let mut db = fixture();
    // A one-row derived table from an aggregate; the outer selects its column.
    assert_scalar(&mut db, "SELECT m FROM (SELECT max(v) AS m FROM t)", int(40));
}
