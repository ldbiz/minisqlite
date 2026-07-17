//! Conformance battery for compound SELECT operators: UNION, UNION ALL,
//! INTERSECT, and EXCEPT.
//!
//! Every expected result below is transcribed from the SQLite documentation,
//! NEVER from what this engine returns. The binding sources are:
//!
//!   * `spec/sqlite-doc/lang_select.html` §3 "Compound Select Statements":
//!     - "A compound SELECT created using the UNION ALL operator returns all the
//!       rows from the SELECT to the left ... and all the rows from the SELECT to
//!       the right of it. The UNION operator works the same way as UNION ALL,
//!       except that duplicate rows are removed from the final result set. The
//!       INTERSECT operator returns the intersection of the results of the left
//!       and right SELECTs. The EXCEPT operator returns the subset of rows
//!       returned by the left SELECT that are not also returned by the right-hand
//!       SELECT. Duplicate rows are removed from the results of INTERSECT and
//!       EXCEPT operators before the result set is returned."
//!     - "In a compound SELECT, all the constituent SELECTs must return the same
//!       number of result columns." (a mismatch is an error).
//!     - "For the purposes of determining duplicate rows ..., NULL values are
//!       considered equal to other NULL values and distinct from all non-NULL
//!       values. ... No affinity transformations are applied to any values when
//!       comparing rows as part of a compound SELECT."
//!     - "When three or more simple SELECTs are connected into a compound SELECT,
//!       they group from left to right. In other words, if 'A', 'B' and 'C' are
//!       all simple SELECT statements, (A op B op C) is processed as
//!       ((A op B) op C)." — all four operators share one precedence level and
//!       are LEFT-associative (SQLite does not give INTERSECT higher precedence).
//!   * `spec/sqlite-doc/lang_select.html` §4 "The ORDER BY clause": only the
//!     right-most SELECT may carry an ORDER BY, and it "will apply across all
//!     elements of the compound." ORDER BY terms are aliases for the compound's
//!     result columns (matched, left-to-right, against the constituent SELECTs).
//!   * `spec/sqlite-doc/datatype3.html` §6 "Sorting, Grouping and Compound
//!     SELECTs": "The compound SELECT operators UNION, INTERSECT and EXCEPT
//!     perform implicit comparisons between values. No affinity is applied to
//!     comparison operands ... the values are compared as is." Grouping considers
//!     "values with different storage classes ... distinct, except for INTEGER
//!     and REAL values which are considered equal if they are numerically equal."
//!   * `spec/sqlite-doc/datatype3.html` §4.1 "Sort Order": NULL sorts first; "An
//!     INTEGER or REAL value is less than any TEXT or BLOB value. When an INTEGER
//!     or REAL is compared to another INTEGER or REAL, a numerical comparison is
//!     performed."
//!   * `spec/sqlite-doc/c3ref/column_name.html`: "The name of a result column is
//!     the value of the 'AS' clause for that column, if there is an AS clause. If
//!     there is no AS clause then the name of the column is unspecified" — so a
//!     compound's column names come from the AS clauses of the FIRST (left-most)
//!     SELECT, and we only assert names that carry an explicit AS.
//!
//! Assertions encode the DOCUMENTED behavior. If the engine disagrees,
//! the assertion STAYS spec-correct (it fails) — it is never weakened to pass.
//! `Value` has no `PartialEq`, so
//! comparisons go through the shared harness (`assert_rows`, `value_eq`, ...);
//! never compare a `Value` with `==`. A trailing ORDER BY makes row order
//! deterministic so the ordered `assert_rows` is used wherever order is defined.

mod conformance;

use conformance::*;
// `Connection`/`Value` are named in this file's helper signatures. The harness
// imports them privately (so `conformance::*` does not re-export them); this file
// depends only on the pinned `minisqlite` facade surface, exactly like the harness.
use minisqlite::{Connection, Value};

/// A fresh in-memory database seeded with the two tables the compound-select
/// scenarios operate on:
///
///   * `a(x INTEGER)` = {1, 2, 3, 3}  (distinct set {1, 2, 3}; a duplicate 3)
///   * `b(x INTEGER)` = {3, 4, 4, 5}  (distinct set {3, 4, 5}; a duplicate 4)
///
/// The duplicate rows in each table are deliberate: they let a single query show
/// both that UNION/INTERSECT/EXCEPT return DISTINCT rows and that UNION ALL does
/// not.
fn ab() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE a(x INTEGER)");
    exec(&mut db, "INSERT INTO a(x) VALUES (1),(2),(3),(3)");
    exec(&mut db, "CREATE TABLE b(x INTEGER)");
    exec(&mut db, "INSERT INTO b(x) VALUES (3),(4),(4),(5)");
    db
}

/// Shorthand for a one-column expected result set: `col(&[1, 2, 3])`.
fn col(xs: &[i64]) -> Vec<Vec<Value>> {
    xs.iter().map(|&x| vec![int(x)]).collect()
}

/// Assert `sql` returns exactly one 1x1 row whose value is numerically 1.
///
/// This is the spec-faithful check for the cross-class cases where an INTEGER and
/// a REAL are numerically equal (datatype3 §6): the compound folds `1` and `1.0`
/// to a single row, so the row count (the documented invariant) is asserted
/// STRICTLY, but the spec prose does NOT pin the survivor's storage class, so
/// either `Integer(1)` or `Real(1.0)` is accepted (anything else — a different
/// number, a text `"1"`, an extra row — still fails). The observed survivor is
/// `Integer(1)`.
fn assert_single_row_numeric_one(db: &mut Connection, sql: &str) {
    let qr = query(db, sql);
    assert_eq!(qr.rows.len(), 1, "expected exactly one row: {sql}");
    assert_eq!(qr.rows[0].len(), 1, "expected one output column: {sql}");
    let got = &qr.rows[0][0];
    assert!(
        value_eq(got, &int(1)) || value_eq(got, &real(1.0)),
        "survivor must be numeric 1 (class unspecified by spec), got {got:?}: {sql}"
    );
}

// ---------------------------------------------------------------------------
// Core operator semantics (lang_select.html §3), one table pair.
// ---------------------------------------------------------------------------

#[test]
fn union_removes_duplicates_across_whole_result() {
    // UNION = UNION ALL then remove duplicates from the FINAL result set:
    // {1,2,3,3} ∪ {3,4,4,5} deduped = {1,2,3,4,5}.
    let mut db = ab();
    assert_rows(
        &mut db,
        "SELECT x FROM a UNION SELECT x FROM b ORDER BY x",
        &col(&[1, 2, 3, 4, 5]),
    );
}

#[test]
fn union_all_keeps_all_rows() {
    // UNION ALL returns every row from the left then every row from the right,
    // with no deduplication: a's {1,2,3,3} and b's {3,4,4,5} → three 3s, two 4s.
    let mut db = ab();
    assert_rows(
        &mut db,
        "SELECT x FROM a UNION ALL SELECT x FROM b ORDER BY x",
        &col(&[1, 2, 3, 3, 3, 4, 4, 5]),
    );
}

#[test]
fn intersect_distinct_rows_in_both() {
    // INTERSECT = distinct rows present in BOTH: distinct(a)={1,2,3} ∩
    // distinct(b)={3,4,5} = {3}.
    let mut db = ab();
    assert_rows(
        &mut db,
        "SELECT x FROM a INTERSECT SELECT x FROM b ORDER BY x",
        &col(&[3]),
    );
}

#[test]
fn except_distinct_rows_in_left_not_right() {
    // EXCEPT = distinct rows in the left that are NOT in the right:
    // distinct(a)={1,2,3} minus distinct(b)={3,4,5} = {1,2}.
    let mut db = ab();
    assert_rows(
        &mut db,
        "SELECT x FROM a EXCEPT SELECT x FROM b ORDER BY x",
        &col(&[1, 2]),
    );
}

#[test]
fn except_reversed_operands() {
    // EXCEPT is asymmetric: b EXCEPT a = distinct(b)={3,4,5} minus
    // distinct(a)={1,2,3} = {4,5}.
    let mut db = ab();
    assert_rows(
        &mut db,
        "SELECT x FROM b EXCEPT SELECT x FROM a ORDER BY x",
        &col(&[4, 5]),
    );
}

// ---------------------------------------------------------------------------
// Duplicate removal in the operators that dedup.
// ---------------------------------------------------------------------------

#[test]
fn union_identical_selects_dedups_to_distinct() {
    // a UNION a returns the distinct rows of a (the duplicate 3 collapses).
    let mut db = ab();
    assert_rows(
        &mut db,
        "SELECT x FROM a UNION SELECT x FROM a ORDER BY x",
        &col(&[1, 2, 3]),
    );
}

#[test]
fn intersect_removes_duplicates() {
    // "Duplicate rows are removed from the results of INTERSECT": a INTERSECT a is
    // distinct(a) = {1,2,3}, not the four raw rows of a.
    let mut db = ab();
    assert_rows(
        &mut db,
        "SELECT x FROM a INTERSECT SELECT x FROM a ORDER BY x",
        &col(&[1, 2, 3]),
    );
}

#[test]
fn except_self_is_empty() {
    // Every distinct row of a is also in a, so a EXCEPT a is empty.
    let mut db = ab();
    assert_rows(&mut db, "SELECT x FROM a EXCEPT SELECT x FROM a ORDER BY x", &[]);
}

#[test]
fn intersect_disjoint_is_empty() {
    // No common row → empty intersection.
    let mut db = mem();
    assert_rows(&mut db, "SELECT 1 INTERSECT SELECT 2 ORDER BY 1", &[]);
}

// ---------------------------------------------------------------------------
// Multi-way compounds group LEFT TO RIGHT (lang_select.html §3): all four
// operators share one precedence and are left-associative.
// ---------------------------------------------------------------------------

#[test]
fn union_multi_way_dedups() {
    // ((1 UNION 2) UNION 2) = {1,2} UNION {2} = {1,2}.
    let mut db = mem();
    assert_rows(
        &mut db,
        "SELECT 1 UNION SELECT 2 UNION SELECT 2 ORDER BY 1",
        &col(&[1, 2]),
    );
}

#[test]
fn compound_groups_left_to_right_union_then_except() {
    // (a UNION b) EXCEPT {5} = {1,2,3,4,5} EXCEPT {5} = {1,2,3,4}. Were it grouped
    // right-to-left it would be a UNION (b EXCEPT {5}) = {1,2,3,4} too here, so the
    // next test uses operands that actually distinguish the two groupings.
    let mut db = ab();
    assert_rows(
        &mut db,
        "SELECT x FROM a UNION SELECT x FROM b EXCEPT SELECT 5 ORDER BY x",
        &col(&[1, 2, 3, 4]),
    );
}

#[test]
fn compound_left_associativity_is_observable() {
    // (3 EXCEPT 2) UNION 4 = {3} UNION {4} = {3,4}  (LEFT-to-right, per the spec).
    // Right-to-left would be 3 EXCEPT (2 UNION 4) = {3} EXCEPT {2,4} = {3}. The two
    // groupings give different results, so this pins the documented associativity.
    let mut db = mem();
    assert_rows(
        &mut db,
        "SELECT 3 EXCEPT SELECT 2 UNION SELECT 4 ORDER BY 1",
        &col(&[3, 4]),
    );
}

#[test]
fn union_all_then_union_dedups_final() {
    // ((1 UNION ALL 2) UNION 2): the inner bag is [1,2]; the outer UNION dedups the
    // whole concatenation with {2} → {1,2}.
    let mut db = mem();
    assert_rows(
        &mut db,
        "SELECT 1 UNION ALL SELECT 2 UNION SELECT 2 ORDER BY 1",
        &col(&[1, 2]),
    );
}

#[test]
fn union_then_union_all_keeps_duplicate() {
    // ((2 UNION 2) UNION ALL 2): the inner UNION yields [2]; the outer UNION ALL
    // concatenates without dedup → [2,2]. Shows an outer UNION ALL preserves the
    // duplicate a left-nested UNION had removed.
    let mut db = mem();
    assert_rows(
        &mut db,
        "SELECT 2 UNION SELECT 2 UNION ALL SELECT 2 ORDER BY 1",
        &col(&[2, 2]),
    );
}

// ---------------------------------------------------------------------------
// The trailing ORDER BY applies to the ENTIRE compound (lang_select.html §4).
// ---------------------------------------------------------------------------

#[test]
fn trailing_order_by_desc_orders_whole_compound() {
    let mut db = ab();
    assert_rows(
        &mut db,
        "SELECT x FROM a UNION SELECT x FROM b ORDER BY x DESC",
        &col(&[5, 4, 3, 2, 1]),
    );
}

#[test]
fn trailing_order_by_desc_on_union_all() {
    // The ORDER BY sorts the full UNION ALL multiset (three 3s, two 4s), descending.
    let mut db = ab();
    assert_rows(
        &mut db,
        "SELECT x FROM a UNION ALL SELECT x FROM b ORDER BY x DESC",
        &col(&[5, 4, 4, 3, 3, 3, 2, 1]),
    );
}

#[test]
fn order_by_column_position_on_compound() {
    // "ORDER BY 1" is an alias for the first result column of the compound.
    let mut db = ab();
    assert_rows(
        &mut db,
        "SELECT x FROM a UNION SELECT x FROM b ORDER BY 1",
        &col(&[1, 2, 3, 4, 5]),
    );
}

#[test]
fn order_by_output_alias_name_on_compound() {
    // §4: an ORDER BY term is matched, as an alias, against a result column of the
    // compound (found in the left-most SELECT). "c" is the alias from the first
    // SELECT and orders the entire compound.
    let mut db = ab();
    assert_rows(
        &mut db,
        "SELECT x AS c FROM a UNION SELECT x FROM b ORDER BY c",
        &col(&[1, 2, 3, 4, 5]),
    );
}

// ---------------------------------------------------------------------------
// Duplicate determination is over the WHOLE row (all columns), multi-column.
// ---------------------------------------------------------------------------

#[test]
fn union_dedups_on_whole_row() {
    // Two identical two-column rows collapse to one.
    let mut db = mem();
    assert_rows(
        &mut db,
        "SELECT 1, 'a' UNION SELECT 1, 'a' ORDER BY 1, 2",
        &[vec![int(1), text("a")]],
    );
}

#[test]
fn union_distinct_when_any_column_differs() {
    // Rows equal in column 1 but differing in column 2 are DISTINCT rows.
    let mut db = mem();
    assert_rows(
        &mut db,
        "SELECT 1, 'a' UNION SELECT 1, 'b' ORDER BY 1, 2",
        &[vec![int(1), text("a")], vec![int(1), text("b")]],
    );
}

#[test]
fn intersect_multicolumn_match() {
    let mut db = mem();
    assert_rows(
        &mut db,
        "SELECT 1, 2 INTERSECT SELECT 1, 2 ORDER BY 1, 2",
        &[vec![int(1), int(2)]],
    );
}

#[test]
fn intersect_multicolumn_no_match_is_empty() {
    // Rows differ in column 2 → no whole-row match → empty.
    let mut db = mem();
    assert_rows(&mut db, "SELECT 1, 2 INTERSECT SELECT 1, 3 ORDER BY 1, 2", &[]);
}

// ---------------------------------------------------------------------------
// NULL handling in compound dedup (lang_select.html §3): "NULL values are
// considered equal to other NULL values and distinct from all non-NULL values."
// ---------------------------------------------------------------------------

#[test]
fn union_folds_equal_nulls_to_one_row() {
    // Two NULL rows are equal for dedup → a single NULL row (unlike SQL `=`, where
    // NULL = NULL is not true).
    let mut db = mem();
    assert_rows(&mut db, "SELECT NULL UNION SELECT NULL ORDER BY 1", &[vec![null()]]);
}

#[test]
fn union_all_keeps_both_nulls() {
    // UNION ALL does not dedup: two NULL rows stay two rows.
    let mut db = mem();
    assert_rows(
        &mut db,
        "SELECT NULL UNION ALL SELECT NULL ORDER BY 1",
        &[vec![null()], vec![null()]],
    );
}

#[test]
fn intersect_null_equals_null() {
    // NULL is present in both sides (NULL considered equal to NULL) → one NULL row.
    let mut db = mem();
    assert_rows(&mut db, "SELECT NULL INTERSECT SELECT NULL ORDER BY 1", &[vec![null()]]);
}

#[test]
fn except_null_removed_by_null() {
    // The left NULL is also on the right (NULL == NULL for dedup) → removed → empty.
    let mut db = mem();
    assert_rows(&mut db, "SELECT NULL EXCEPT SELECT NULL ORDER BY 1", &[]);
}

#[test]
fn except_null_distinct_from_value() {
    // NULL is distinct from the non-NULL 1, so EXCEPT does not remove it.
    let mut db = mem();
    assert_rows(&mut db, "SELECT NULL EXCEPT SELECT 1 ORDER BY 1", &[vec![null()]]);
}

#[test]
fn intersect_null_distinct_from_value_is_empty() {
    // NULL is distinct from 1 → no common row → empty.
    let mut db = mem();
    assert_rows(&mut db, "SELECT NULL INTERSECT SELECT 1 ORDER BY 1", &[]);
}

#[test]
fn union_null_and_value_are_two_rows() {
    // Distinct rows; NULL sorts before the integer (datatype3 §4.1).
    let mut db = mem();
    assert_rows(
        &mut db,
        "SELECT 1 UNION SELECT NULL ORDER BY 1",
        &[vec![null()], vec![int(1)]],
    );
}

// ---------------------------------------------------------------------------
// Cross-storage-class comparison (datatype3 §6): NO affinity, but INTEGER and
// REAL still compare NUMERICALLY, so 1 and 1.0 are equal — while 1 and '1' are
// NOT (text is never coerced here).
// ---------------------------------------------------------------------------

#[test]
fn union_int_and_real_numeric_equal_dedup_to_one_row() {
    // datatype3 §6 + §4.1: compared "as is" (no affinity), but an INTEGER and a
    // REAL compare numerically, so 1 and 1.0 are duplicates and UNION returns a
    // SINGLE row (numerically 1; the survivor's storage class is unspecified —
    // see `assert_single_row_numeric_one`).
    let mut db = mem();
    assert_single_row_numeric_one(&mut db, "SELECT 1 UNION SELECT 1.0 ORDER BY 1");
}

#[test]
fn intersect_int_and_real_numeric_equal_match() {
    // 1 and 1.0 are numerically equal, so the intersection is non-empty: one row,
    // numerically 1 (class unspecified by the prose).
    let mut db = mem();
    assert_single_row_numeric_one(&mut db, "SELECT 1 INTERSECT SELECT 1.0 ORDER BY 1");
}

#[test]
fn except_int_and_real_numeric_equal_removes_row() {
    // The left 1 is numerically equal to the right 1.0, so EXCEPT removes it → empty.
    let mut db = mem();
    assert_rows(&mut db, "SELECT 1 EXCEPT SELECT 1.0 ORDER BY 1", &[]);
}

#[test]
fn union_all_keeps_int_and_real_as_distinct_classes() {
    // UNION ALL neither dedups nor applies affinity: 2 (INTEGER) and 2.0 (REAL) are
    // both returned, each keeping its own storage class. They sort equal, so the
    // multiset (not the order) is the guaranteed contract.
    let mut db = mem();
    assert_rows_unordered(
        &mut db,
        "SELECT 2 UNION ALL SELECT 2.0",
        &[vec![int(2)], vec![real(2.0)]],
    );
}

#[test]
fn union_does_not_coerce_int_and_text() {
    // "No affinity transformations are applied": 1 (INTEGER) and '1' (TEXT) are
    // DISTINCT rows (text is not converted to a number). An INTEGER sorts before a
    // TEXT value (datatype3 §4.1), so ORDER BY gives [1, '1'].
    let mut db = mem();
    assert_rows(
        &mut db,
        "SELECT 1 UNION SELECT '1' ORDER BY 1",
        &[vec![int(1)], vec![text("1")]],
    );
}

// ---------------------------------------------------------------------------
// Column-count mismatch is an error (lang_select.html §3), for every operator.
// ---------------------------------------------------------------------------

#[test]
fn union_column_count_mismatch_is_error() {
    let mut db = mem();
    assert_query_error(&mut db, "SELECT 1 UNION SELECT 1, 2");
}

#[test]
fn union_all_column_count_mismatch_is_error() {
    let mut db = mem();
    assert_query_error(&mut db, "SELECT 1 UNION ALL SELECT 1, 2");
}

#[test]
fn intersect_column_count_mismatch_is_error() {
    let mut db = mem();
    assert_query_error(&mut db, "SELECT 1 INTERSECT SELECT 1, 2");
}

#[test]
fn except_column_count_mismatch_is_error() {
    let mut db = mem();
    assert_query_error(&mut db, "SELECT 1, 2 EXCEPT SELECT 1");
}

// ---------------------------------------------------------------------------
// Result column names come from the FIRST (left-most) SELECT
// (c3ref/column_name.html: a column's name is its AS clause). We only assert
// names that carry an explicit AS; without AS the name is "unspecified".
// ---------------------------------------------------------------------------

#[test]
fn compound_column_name_from_first_select() {
    let mut db = ab();
    assert_columns(
        &mut db,
        "SELECT x AS foo FROM a UNION SELECT x FROM b",
        &["foo"],
    );
}

#[test]
fn compound_column_name_ignores_right_select_alias() {
    // The right SELECT's alias "bar" is irrelevant; the name is "foo" from the left.
    let mut db = ab();
    assert_columns(
        &mut db,
        "SELECT x AS foo FROM a UNION SELECT x AS bar FROM b",
        &["foo"],
    );
}

#[test]
fn compound_multicolumn_names_from_first_select() {
    let mut db = ab();
    assert_columns(
        &mut db,
        "SELECT x AS p, x AS q FROM a UNION ALL SELECT x, x FROM b",
        &["p", "q"],
    );
}

#[test]
fn intersect_column_name_from_first_select() {
    // The "name from the left" rule holds for every compound operator, not just UNION.
    let mut db = ab();
    assert_columns(
        &mut db,
        "SELECT x AS foo FROM a INTERSECT SELECT x FROM b",
        &["foo"],
    );
}
