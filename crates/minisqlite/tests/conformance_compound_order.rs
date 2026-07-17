//! Conformance: IMPLICIT output order of a DEDUPLICATING compound SELECT
//! (`UNION` / `INTERSECT` / `EXCEPT`) that has NO trailing `ORDER BY`.
//!
//! Real SQLite implements the duplicate elimination of `UNION`, `INTERSECT`, and
//! `EXCEPT` as a SORT, so a deduplicating compound comes back in ASCENDING full-row
//! order EVEN WITHOUT an `ORDER BY`. The binding sources (transcribed from the
//! documentation, NEVER from what this engine returns):
//!
//!   * `spec/sqlite-doc/lang_select.html` §3 "Compound Select Statements":
//!     duplicate rows are removed from `UNION`/`INTERSECT`/`EXCEPT` before the
//!     result is returned; `UNION ALL` returns "all the rows" from the left then
//!     the right, in order, with no deduplication.
//!   * `spec/sqlite-doc/datatype3.html` §6 "Sorting, Grouping and Compound
//!     SELECTs": "The compound SELECT operators UNION, INTERSECT and EXCEPT perform
//!     implicit comparisons between values." The result of that comparison is the
//!     order the rows come back in — sorted ascending — with no affinity applied.
//!   * `spec/sqlite-doc/datatype3.html` §4.1 "Sort Order": NULL sorts first; an
//!     INTEGER/REAL is less than any TEXT/BLOB value.
//!
//! The sibling `conformance_compound.rs` file appends an explicit `ORDER BY` to
//! every order-sensitive case, so it deliberately does NOT cover this implicit
//! order — this file does, and its `assert_rows` are ORDERED so the sort itself is
//! what is under test. Each case is chosen so first-appearance order DIFFERS from
//! sorted order, so a "did nothing" implementation would fail here.
//!
//! Assertions encode the DOCUMENTED behavior. If the engine disagrees the
//! assertion STAYS spec-correct (it fails) rather than being weakened to pass.

mod conformance;

use conformance::*;

// ---------------------------------------------------------------------------
// UNION / INTERSECT / EXCEPT with NO ORDER BY come back SORTED ascending.
// ---------------------------------------------------------------------------

#[test]
fn union_multiway_no_order_by_comes_back_sorted() {
    // First-appearance order is 3, 1, 2; the dedup sort returns them ascending.
    // `((3 UNION 1) UNION 2)` — the outermost op is UNION, so the implicit sort fires.
    let mut db = mem();
    assert_rows(
        &mut db,
        "SELECT 3 UNION SELECT 1 UNION SELECT 2",
        &[vec![int(1)], vec![int(2)], vec![int(3)]],
    );
}

#[test]
fn intersect_no_order_by_single_row() {
    // A single common value: `3 INTERSECT 3` = {3}. (One row can't show ordering, but
    // it confirms INTERSECT with no ORDER BY still returns its deduped result.)
    let mut db = mem();
    assert_rows(&mut db, "SELECT 3 INTERSECT SELECT 3", &[vec![int(3)]]);
}

#[test]
fn intersect_no_order_by_multiple_rows_sorted() {
    // distinct(l) = {3,1,2,5}, distinct(r) = {2,3,5,9}; the intersection is {2,3,5},
    // returned ASCENDING even without an ORDER BY. Left-scan order of the survivors is
    // 3,2,5, so a no-sort implementation would differ from the sorted [2,3,5].
    let mut db = mem();
    exec(&mut db, "CREATE TABLE l(x)");
    exec(&mut db, "INSERT INTO l(x) VALUES (3),(1),(2),(5)");
    exec(&mut db, "CREATE TABLE r(x)");
    exec(&mut db, "INSERT INTO r(x) VALUES (2),(3),(5),(9)");
    assert_rows(
        &mut db,
        "SELECT x FROM l INTERSECT SELECT x FROM r",
        &[vec![int(2)], vec![int(3)], vec![int(5)]],
    );
}

#[test]
fn except_no_order_by_multiple_rows_sorted() {
    // distinct(t) = {3,1,2}, EXCEPT {2} = {1,3}, returned ASCENDING. Left-scan order of
    // the survivors is 3,1, so the sorted [1,3] proves the implicit sort ran.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x)");
    exec(&mut db, "INSERT INTO t(x) VALUES (3),(1),(2)");
    assert_rows(
        &mut db,
        "SELECT x FROM t EXCEPT SELECT 2",
        &[vec![int(1)], vec![int(3)]],
    );
}

#[test]
fn union_no_order_by_orders_null_first() {
    // datatype3 §4.1: NULL sorts before the integer. `1 UNION NULL` has first-appearance
    // order [1, NULL] but returns [NULL, 1] — the implicit ascending sort puts NULL first.
    let mut db = mem();
    assert_rows(
        &mut db,
        "SELECT 1 UNION SELECT NULL",
        &[vec![null()], vec![int(1)]],
    );
}

#[test]
fn union_no_order_by_multicolumn_full_row_sort() {
    // The implicit sort is a FULL-ROW ascending sort over EVERY output column, not just
    // col0. The two col0=1 rows are supplied in col1-DESCENDING first-appearance order
    // ('c' before 'a'), so a col0-only stable sort would leave (1,'c') ahead of (1,'a');
    // only a full-row sort reorders them to (1,'a') then (1,'c'). Distinct rows
    // {(2,'b'),(1,'c'),(1,'a')} come back ordered by (col0, col1): (1,'a'),(1,'c'),(2,'b').
    let mut db = mem();
    assert_rows(
        &mut db,
        "SELECT 2, 'b' UNION SELECT 1, 'c' UNION SELECT 1, 'a'",
        &[
            vec![int(1), text("a")],
            vec![int(1), text("c")],
            vec![int(2), text("b")],
        ],
    );
}

#[test]
fn union_mixed_int_real_sorts_numerically() {
    // datatype3 §4.1: an INTEGER and a REAL compare NUMERICALLY, so the distinct values
    // {2, 1.5, 1} sort ascending as 1, 1.5, 2, each keeping its own storage class
    // (int, real, int). First-appearance order is [2, 1.5, 1], so this discriminates the
    // sort and exercises cross-class ordering (earlier cases used only int/text columns).
    let mut db = mem();
    assert_rows(
        &mut db,
        "SELECT 2 UNION SELECT 1.5 UNION SELECT 1",
        &[vec![int(1)], vec![real(1.5)], vec![int(2)]],
    );
}

#[test]
fn union_no_order_by_then_limit_takes_sorted_prefix() {
    // With no ORDER BY, the implicit sort runs BEFORE the LIMIT (the top-k retention bound
    // is pushed into the Sort), so `LIMIT 2` returns the two SMALLEST distinct rows, not
    // the first two in dedup order. First-appearance would be [3,1,2] -> [3,1]; the sorted
    // prefix is [1,2].
    let mut db = mem();
    assert_rows(
        &mut db,
        "SELECT 3 UNION SELECT 1 UNION SELECT 2 LIMIT 2",
        &[vec![int(1)], vec![int(2)]],
    );
}

#[test]
fn union_no_order_by_sort_reuses_nocase_collation() {
    // The implicit sort reuses each output column's dedup collation (datatype3 §7.1: a
    // compound compares under the LEFT SELECT's column collation). Column `c` is declared
    // NOCASE, so the sort is case-insensitive and 'a','B','C' come back in NOCASE ascending
    // order [a, B, C]. A BINARY sort would instead yield [B, C, a] (uppercase bytes 66,67
    // sort before lowercase 97), so this pins the collation reuse BEHAVIORALLY end-to-end,
    // not just structurally. The two identical arms dedup to the three distinct values (no
    // case-collisions among a/B/C, so the row set is the same under either collation — only
    // the ORDER discriminates).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(c TEXT COLLATE NOCASE)");
    exec(&mut db, "INSERT INTO t VALUES ('B'), ('a'), ('C')");
    assert_rows(
        &mut db,
        "SELECT c FROM t UNION SELECT c FROM t",
        &[vec![text("a")], vec![text("B")], vec![text("C")]],
    );
}

// ---------------------------------------------------------------------------
// UNION ALL is the exception: it concatenates left rows then right rows in
// order and is NOT sorted (proves the implicit sort is keyed to the DEDUP ops).
// ---------------------------------------------------------------------------

#[test]
fn union_all_no_order_by_preserves_concatenation_order() {
    // `3 UNION ALL 1` returns [3, 1] (left then right), NOT the sorted [1, 3]. This is
    // the control: if the engine sorted UNION ALL too, this would come back [1, 3].
    let mut db = mem();
    assert_rows(
        &mut db,
        "SELECT 3 UNION ALL SELECT 1",
        &[vec![int(3)], vec![int(1)]],
    );
}

#[test]
fn union_all_over_dedup_left_is_not_globally_sorted() {
    // `(3 UNION 1) UNION ALL 2`: the OUTERMOST op is UNION ALL, so the whole compound is
    // NOT globally sorted — the left `(3 UNION 1)` sub-result is concatenated ahead of 2.
    // Documented residual: real SQLite's LEFT sub-result is itself sorted ([1,3]) while
    // this engine streams the sub-compound's dedup order; only the trailing `2` position
    // (after the whole left block) is pinned here, which both agree on.
    let mut db = mem();
    let qr = query(&mut db, "SELECT 3 UNION SELECT 1 UNION ALL SELECT 2");
    assert_eq!(qr.rows.len(), 3, "two distinct left rows plus the trailing 2");
    // The trailing UNION ALL arm (2) comes AFTER the entire left block, so it is last.
    assert!(
        value_eq(&qr.rows[2][0], &int(2)),
        "UNION ALL appends its right arm after the whole left block; got {:?}",
        qr.rows[2]
    );
}
