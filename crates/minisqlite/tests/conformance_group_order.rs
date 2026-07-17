//! Conformance: IMPLICIT output order of an unordered `GROUP BY` (one with NO
//! trailing `ORDER BY`).
//!
//! Real SQLite implements `GROUP BY` grouping via a sort (or an ordered index), so an
//! unordered `GROUP BY` returns its groups ASCENDING by the grouping keys EVEN WITHOUT
//! an `ORDER BY`. The binding sources (transcribed from the documentation, NEVER from
//! what this engine returns):
//!
//!   * `spec/sqlite-doc/datatype3.html` §6 "Sorting, Grouping and Compound SELECTs":
//!     GROUP BY grouping uses the SAME value comparison as sorting, so the groups come
//!     back ordered by that comparison.
//!   * `spec/sqlite-doc/datatype3.html` §4.1 "Sort Order": NULL sorts first; an
//!     INTEGER/REAL is less than any TEXT/BLOB, and numbers compare numerically.
//!   * `spec/sqlite-doc/datatype3.html` §7 / lang_select.html §7.2: grouping compares
//!     text under the grouping column's collating sequence (e.g. NOCASE), and the same
//!     collation orders the groups.
//!
//! This is the GROUP BY analogue of `conformance_compound_order.rs` (the compound
//! UNION/INTERSECT/EXCEPT implicit sort). The sibling `conformance_select_group_having.rs`
//! appends an explicit `ORDER BY` to every case, so it deliberately does NOT cover this
//! implicit order — this file does, and its `assert_rows` are ORDERED so the sort itself
//! is under test. Every fixture is chosen so first-appearance (scan) order DIFFERS from
//! sorted order, so a "did nothing" implementation would fail here.
//!
//! Assertions encode the DOCUMENTED behavior. If the engine disagrees the
//! assertion STAYS spec-correct (it fails) rather than being weakened to pass.

mod conformance;

use conformance::*;

// ---------------------------------------------------------------------------
// The witness: an unordered GROUP BY comes back ASCENDING by group key.
// ---------------------------------------------------------------------------

#[test]
fn unordered_group_by_returns_groups_sorted_ascending() {
    // First-appearance order is 3, 1, 2; the implicit group sort returns them ascending.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x)");
    exec(&mut db, "INSERT INTO t VALUES (3),(1),(2)");
    assert_rows(
        &mut db,
        "SELECT x, count(*) FROM t GROUP BY x",
        &[vec![int(1), int(1)], vec![int(2), int(1)], vec![int(3), int(1)]],
    );
}

// ---------------------------------------------------------------------------
// Multi-column GROUP BY: a FULL-ROW sort over every group key, in order — a
// key0-only sort would leave the equal-key0 groups in first-appearance order.
// ---------------------------------------------------------------------------

#[test]
fn multi_column_group_by_sorts_by_every_key() {
    // Groups by (x,y): (1,'c') count 2, (1,'a') count 1, (2,'b') count 1. First-appearance
    // is (1,'c'),(1,'a'),(2,'b'); a key0-ONLY stable sort would keep (1,'c') ahead of
    // (1,'a'). Only a full-row sort reorders them to (1,'a'),(1,'c'),(2,'b').
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x, y)");
    exec(&mut db, "INSERT INTO t VALUES (1,'c'),(1,'a'),(2,'b'),(1,'c')");
    assert_rows(
        &mut db,
        "SELECT x, y, count(*) FROM t GROUP BY x, y",
        &[
            vec![int(1), text("a"), int(1)],
            vec![int(1), text("c"), int(2)],
            vec![int(2), text("b"), int(1)],
        ],
    );
}

// ---------------------------------------------------------------------------
// NULL group key sorts FIRST (datatype3 §4.1), and all NULLs form one group.
// ---------------------------------------------------------------------------

#[test]
fn null_group_key_sorts_first() {
    // Groups: 1 (count 1), NULL (count 2), 2 (count 1). First-appearance is 1, NULL, 2;
    // the ascending sort puts the NULL group FIRST.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x)");
    exec(&mut db, "INSERT INTO t VALUES (1),(NULL),(2),(NULL)");
    assert_rows(
        &mut db,
        "SELECT x, count(*) FROM t GROUP BY x",
        &[vec![null(), int(2)], vec![int(1), int(1)], vec![int(2), int(1)]],
    );
}

// ---------------------------------------------------------------------------
// Mixed INTEGER/REAL keys sort NUMERICALLY, each keeping its storage class.
// ---------------------------------------------------------------------------

#[test]
fn mixed_int_real_group_keys_sort_numerically() {
    // Distinct keys {2, 1.5, 1} sort ascending as 1, 1.5, 2, each returned in its own
    // storage class (int, real, int). First-appearance is [2, 1.5, 1], so this exercises
    // cross-class numeric ordering (not just same-class integers).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x)");
    exec(&mut db, "INSERT INTO t VALUES (2),(1.5),(1)");
    assert_rows(
        &mut db,
        "SELECT x, count(*) FROM t GROUP BY x",
        &[vec![int(1), int(1)], vec![real(1.5), int(1)], vec![int(2), int(1)]],
    );
}

// ---------------------------------------------------------------------------
// A NOCASE grouping column groups AND orders case-insensitively — proving the
// sort reuses the group column's collation, not a hardcoded BINARY.
// ---------------------------------------------------------------------------

#[test]
fn nocase_group_column_orders_case_insensitively() {
    // Under NOCASE the values {'B','a','C','a'} group as 'B'(1), 'a'(2), 'C'(1), each group
    // keeping its FIRST-seen spelling. NOCASE ascending order is 'a','B','C'. A BINARY sort
    // (the bug this guards) would instead order the raw bytes 'B'(66),'C'(67),'a'(97) ->
    // 'B','C','a' — a DIFFERENT order — so this discriminates the reused collation.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(c TEXT COLLATE NOCASE)");
    exec(&mut db, "INSERT INTO t VALUES ('B'),('a'),('C'),('a')");
    assert_rows(
        &mut db,
        "SELECT c, count(*) FROM t GROUP BY c",
        &[vec![text("a"), int(2)], vec![text("B"), int(1)], vec![text("C"), int(1)]],
    );
}

// ---------------------------------------------------------------------------
// GROUP BY + HAVING, no ORDER BY: the sort runs on the SURVIVING groups (HAVING
// is applied inside the aggregate, before the sort).
// ---------------------------------------------------------------------------

#[test]
fn group_by_having_no_order_by_sorts_survivors() {
    // Groups: 3 (count 2), 1 (count 2), 2 (count 1). `HAVING count(*) > 1` keeps 3 and 1;
    // first-appearance of the survivors is 3, 1, and the implicit sort returns them 1, 3.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x)");
    exec(&mut db, "INSERT INTO t VALUES (3),(3),(1),(1),(2)");
    assert_rows(
        &mut db,
        "SELECT x, count(*) FROM t GROUP BY x HAVING count(*) > 1",
        &[vec![int(1), int(2)], vec![int(3), int(2)]],
    );
}

// ---------------------------------------------------------------------------
// GROUP BY + LIMIT, no ORDER BY: the sort runs BEFORE the LIMIT, so LIMIT takes
// the sorted PREFIX (the smallest keys), not the first groups in scan order.
// ---------------------------------------------------------------------------

#[test]
fn group_by_limit_no_order_by_takes_sorted_prefix() {
    // First-appearance is [3,1,2]; `LIMIT 2` on the sorted order [1,2,3] returns the two
    // SMALLEST groups [1,2], not the first two in scan order ([3,1]).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x)");
    exec(&mut db, "INSERT INTO t VALUES (3),(1),(2)");
    assert_rows(
        &mut db,
        "SELECT x, count(*) FROM t GROUP BY x LIMIT 2",
        &[vec![int(1), int(1)], vec![int(2), int(1)]],
    );
}

// ---------------------------------------------------------------------------
// A projection that DROPS the group key still sorts by the key VALUE — the sort
// keys the aggregate-output group column, BEFORE the projection drops it.
// ---------------------------------------------------------------------------

#[test]
fn projection_dropping_the_group_key_still_sorts_by_it() {
    // `SELECT count(*) ... GROUP BY x` projects `x` away, but the groups must still come
    // back in ascending-x order. Per-group counts differ (x=1 ->1, x=2 ->2, x=3 ->3), so
    // the count column REVEALS the key order: sorted-by-x gives [1],[2],[3], while the
    // first-appearance order (3,1,2) would give [3],[1],[2].
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x)");
    exec(&mut db, "INSERT INTO t VALUES (3),(3),(3),(1),(2),(2)");
    assert_rows(
        &mut db,
        "SELECT count(*) FROM t GROUP BY x",
        &[vec![int(1)], vec![int(2)], vec![int(3)]],
    );
}

// ---------------------------------------------------------------------------
// An explicit ORDER BY still wins over the implicit ascending order.
// ---------------------------------------------------------------------------

#[test]
fn explicit_order_by_overrides_the_implicit_ascending_order() {
    // `ORDER BY x DESC` must yield descending groups, not the implicit ascending order.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x)");
    exec(&mut db, "INSERT INTO t VALUES (3),(1),(2)");
    assert_rows(
        &mut db,
        "SELECT x, count(*) FROM t GROUP BY x ORDER BY x DESC",
        &[vec![int(3), int(1)], vec![int(2), int(1)], vec![int(1), int(1)]],
    );
}
