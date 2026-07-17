//! Conformance battery: the MIN/MAX index/rowid seek optimization (optoverview.html §13,
//! "The MIN/MAX Optimization").
//!
//! The optimization changes only HOW a single `MIN(col)`/`MAX(col)` over a whole table is
//! computed — one O(log n) b-tree seek instead of an O(n) full-scan aggregate — so its
//! result MUST be byte-identical to the scan it replaces for every input. These tests prove
//! that RESULT-EQUIVALENCE end-to-end through `minisqlite::Connection`: for an eligible query
//! we run it BOTH with an index present (the seek fires) and absent (the ordinary aggregate),
//! and assert both equal each other AND the spec-derived known value. The plan-SHAPE (that
//! the seek actually fires vs falls back) is pinned separately in
//! `minisqlite-plan`'s `compile::minmax_index` unit tests; here we pin CORRECTNESS.
//!
//! NULL / collation semantics under test (lang_aggfunc.html #min_agg/#max_agg,
//! datatype3.html §3/§4/§7): MIN/MAX ignore NULLs and return NULL iff there is no non-NULL
//! value; on ties the column's collation decides. The discriminating cases — all-NULL,
//! some-NULL, a NOCASE vs BINARY text column — are exactly the ones a mis-handled seek would
//! get wrong, so an index-present pass on them is a real test of the seek's NULL-skip and
//! collation fall back, not just of the scan.

mod conformance;

use conformance::*;

use minisqlite::{Connection, Value};

// ---- helpers -----------------------------------------------------------------

/// The single 1x1 cell of a query result (panics on any other shape). `Value` has no
/// `PartialEq`, so callers compare with [`value_eq`].
fn scalar(db: &mut Connection, sql: &str) -> Value {
    let qr = query(db, sql);
    assert_eq!(qr.rows.len(), 1, "expected 1 row for `{sql}`, got {:?}", qr.rows);
    assert_eq!(qr.rows[0].len(), 1, "expected 1 column for `{sql}`, got {:?}", qr.rows[0]);
    qr.rows[0][0].clone()
}

fn insert_rows(db: &mut Connection, values: &[&str]) {
    for v in values {
        exec(db, &format!("INSERT INTO t VALUES ({v})"));
    }
}

/// Prove an eligible `MIN`/`MAX` query is byte-identical on the seek and scan paths.
///
/// Builds two fresh databases over the SAME `t(x <decl>)` data: one WITHOUT an index (the
/// full-scan aggregate) and one WITH `CREATE INDEX ix ON t(<index_col>)` (which makes the
/// planner choose the seek — or, for a DESC / NOCASE index, deliberately decline back to the
/// scan). Asserts the seek result equals the scan result AND both equal `expected`.
///
/// This is the intended differential: the scan path is the known-correct oracle the
/// engine already had, and the seek must not diverge from it on any input.
fn assert_equiv(decl: &str, index_col: &str, values: &[&str], query_sql: &str, expected: Value) {
    let mut scan = mem();
    exec(&mut scan, &format!("CREATE TABLE t(x {decl})"));
    insert_rows(&mut scan, values);
    let scan_v = scalar(&mut scan, query_sql);

    let mut seek = mem();
    exec(&mut seek, &format!("CREATE TABLE t(x {decl})"));
    exec(&mut seek, &format!("CREATE INDEX ix ON t({index_col})"));
    insert_rows(&mut seek, values);
    let seek_v = scalar(&mut seek, query_sql);

    let joined = values.join(", ");
    assert!(
        value_eq(&scan_v, &expected),
        "SCAN `{query_sql}` over [{joined}] = {scan_v:?}, want {expected:?}"
    );
    assert!(
        value_eq(&seek_v, &expected),
        "SEEK `{query_sql}` over [{joined}] = {seek_v:?}, want {expected:?}"
    );
    assert!(
        value_eq(&scan_v, &seek_v),
        "seek diverged from scan for `{query_sql}` over [{joined}]: \
         seek {seek_v:?} vs scan {scan_v:?}"
    );
}

// ---- INTEGER column, index on x: the NULL edge matrix -------------------------

#[test]
fn integer_empty_table_min_max_is_null() {
    assert_equiv("INTEGER", "x", &[], "SELECT MIN(x) FROM t", null());
    assert_equiv("INTEGER", "x", &[], "SELECT MAX(x) FROM t", null());
}

#[test]
fn integer_all_null_min_max_is_null() {
    // Every value is NULL, so there is no non-NULL value: MIN/MAX are NULL. For MAX this
    // exercises the "last index entry is a NULL key" path; for MIN the "skip every leading
    // NULL and run off the end" path.
    let vals = ["NULL", "NULL", "NULL"];
    assert_equiv("INTEGER", "x", &vals, "SELECT MIN(x) FROM t", null());
    assert_equiv("INTEGER", "x", &vals, "SELECT MAX(x) FROM t", null());
}

#[test]
fn integer_some_null_ignores_nulls() {
    // NULLs sort FIRST in the ascending index, so a correct MIN must SKIP the leading NULL
    // run to reach 1 — a naive "first index entry" seek would wrongly return NULL here.
    let vals = ["3", "NULL", "1", "NULL", "2"];
    assert_equiv("INTEGER", "x", &vals, "SELECT MIN(x) FROM t", int(1));
    assert_equiv("INTEGER", "x", &vals, "SELECT MAX(x) FROM t", int(3));
}

#[test]
fn integer_no_null() {
    let vals = ["3", "1", "2"];
    assert_equiv("INTEGER", "x", &vals, "SELECT MIN(x) FROM t", int(1));
    assert_equiv("INTEGER", "x", &vals, "SELECT MAX(x) FROM t", int(3));
}

#[test]
fn integer_single_row() {
    assert_equiv("INTEGER", "x", &["5"], "SELECT MIN(x) FROM t", int(5));
    assert_equiv("INTEGER", "x", &["5"], "SELECT MAX(x) FROM t", int(5));
}

#[test]
fn integer_single_null_row() {
    assert_equiv("INTEGER", "x", &["NULL"], "SELECT MIN(x) FROM t", null());
    assert_equiv("INTEGER", "x", &["NULL"], "SELECT MAX(x) FROM t", null());
}

#[test]
fn integer_negative_values() {
    let vals = ["-5", "3", "-1", "0"];
    assert_equiv("INTEGER", "x", &vals, "SELECT MIN(x) FROM t", int(-5));
    assert_equiv("INTEGER", "x", &vals, "SELECT MAX(x) FROM t", int(3));
}

// ---- NUMERIC / REAL columns (non-BLOB affinity) ------------------------------

#[test]
fn numeric_mixed_int_and_real() {
    // NUMERIC affinity keeps 2.5 REAL and stores 1 as INTEGER (no lossless-real ties), so
    // every equal value has ONE representation — the seek's tie-break row choice is
    // irrelevant, and its result matches the scan's.
    let vals = ["1", "2.5", "-3"];
    assert_equiv("NUMERIC", "x", &vals, "SELECT MIN(x) FROM t", int(-3));
    assert_equiv("NUMERIC", "x", &vals, "SELECT MAX(x) FROM t", real(2.5));
}

#[test]
fn real_column() {
    let vals = ["1.5", "0.5", "2.25", "NULL"];
    assert_equiv("REAL", "x", &vals, "SELECT MIN(x) FROM t", real(0.5));
    assert_equiv("REAL", "x", &vals, "SELECT MAX(x) FROM t", real(2.25));
}

// ---- TEXT column: BINARY (seek fires) vs NOCASE (declines to scan) ------------

#[test]
fn text_binary_index_orders_by_binary() {
    // A plain TEXT column compares BINARY: 'B' (0x42) < 'a' (0x61) < 'c' (0x63). The BINARY
    // index presents exactly that order, so the seek gives MIN='B', MAX='c'.
    let vals = ["'a'", "'B'", "'c'"];
    assert_equiv("TEXT", "x", &vals, "SELECT MIN(x) FROM t", text("B"));
    assert_equiv("TEXT", "x", &vals, "SELECT MAX(x) FROM t", text("c"));
}

#[test]
fn text_nocase_column_honors_nocase_via_scan() {
    // A `TEXT COLLATE NOCASE` column's effective comparison collation is NOCASE, which the
    // physically-BINARY index cannot present in order — so the planner DECLINES the seek and
    // the (correct) NOCASE scan runs, giving MIN='a', MAX='c' ('a'/'B'/'c' fold to a<B<c).
    // `assert_equiv` still passes because with a NOCASE column BOTH databases run the scan.
    let vals = ["'a'", "'B'", "'c'"];
    assert_equiv("TEXT COLLATE NOCASE", "x", &vals, "SELECT MIN(x) FROM t", text("a"));
    assert_equiv("TEXT COLLATE NOCASE", "x", &vals, "SELECT MAX(x) FROM t", text("c"));
}

#[test]
fn text_with_nulls() {
    let vals = ["'m'", "NULL", "'a'", "'z'"];
    assert_equiv("TEXT", "x", &vals, "SELECT MIN(x) FROM t", text("a"));
    assert_equiv("TEXT", "x", &vals, "SELECT MAX(x) FROM t", text("z"));
}

// ---- DESC index: declines to the scan, still correct -------------------------

#[test]
fn desc_index_declines_but_stays_correct() {
    // A DESC-declared index is metadata over a b-tree still physically ascending, so the
    // planner declines the seek (it cannot prove the walk order); the scan gives the right
    // answer either way.
    let vals = ["3", "1", "NULL", "2"];
    assert_equiv("INTEGER", "x DESC", &vals, "SELECT MIN(x) FROM t", int(1));
    assert_equiv("INTEGER", "x DESC", &vals, "SELECT MAX(x) FROM t", int(3));
}

// ---- WITHOUT ROWID table: declines to the scan, still correct -----------------

#[test]
fn without_rowid_table_declines_but_stays_correct() {
    // A WITHOUT ROWID table has no integer rowid and its rows live in the PRIMARY KEY
    // b-tree, which the seek executor does not model, so the planner declines even on the PK
    // column and the (correct) scan runs. Exercises the decline boundary end-to-end.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE w(k INTEGER PRIMARY KEY, v TEXT) WITHOUT ROWID");
    exec(&mut db, "INSERT INTO w VALUES (10,'a'), (3,'b'), (27,'c'), (8,'d')");
    assert!(value_eq(&scalar(&mut db, "SELECT MIN(k) FROM w"), &int(3)));
    assert!(value_eq(&scalar(&mut db, "SELECT MAX(k) FROM w"), &int(27)));
}

// ---- Index built by BACKFILL (created after the rows) -------------------------

#[test]
fn backfilled_index_seek_matches_scan() {
    // Create the index AFTER inserting, so the seek reads a b-tree the backfill populated
    // (the other tests create it first, exercising insert-time maintenance). Both must give
    // the same extremum as the scan.
    let mut scan = mem();
    exec(&mut scan, "CREATE TABLE t(x INTEGER)");
    insert_rows(&mut scan, &["7", "NULL", "2", "9", "4"]);
    let scan_min = scalar(&mut scan, "SELECT MIN(x) FROM t");
    let scan_max = scalar(&mut scan, "SELECT MAX(x) FROM t");

    let mut seek = mem();
    exec(&mut seek, "CREATE TABLE t(x INTEGER)");
    insert_rows(&mut seek, &["7", "NULL", "2", "9", "4"]);
    exec(&mut seek, "CREATE INDEX ix ON t(x)");
    let seek_min = scalar(&mut seek, "SELECT MIN(x) FROM t");
    let seek_max = scalar(&mut seek, "SELECT MAX(x) FROM t");

    assert!(value_eq(&seek_min, &int(2)) && value_eq(&scan_min, &int(2)), "min: seek {seek_min:?} scan {scan_min:?}");
    assert!(value_eq(&seek_max, &int(9)) && value_eq(&scan_max, &int(9)), "max: seek {seek_max:?} scan {scan_max:?}");
}

// ---- Composite index: the seek reads the LEFT-MOST key column ------------------

#[test]
fn composite_index_leftmost_min_max_matches_scan() {
    // §13 "the left-most column of an index": a composite index (x, y) serves MIN(x)/MAX(x).
    // Each index record is `[x, y, rowid]`, so the seek must decode COLUMN 0 (x) — not y, not
    // the rowid — and MIN must skip the leading NULL-x run. The data makes a wrong-column
    // decode diverge (MIN/MAX of y are 1/200, of rowid are 1/5, of x are 3/8), so the
    // seek-vs-scan differential is a real test of the multi-column key decode + NULL skip.
    // (The plan-shape/real-catalog tests separately prove the seek actually FIRES here.)
    let rows = ["NULL, 100", "5, 1", "NULL, 2", "3, 200", "8, 3"];
    let build = |with_index: bool| {
        let mut db = mem();
        exec(&mut db, "CREATE TABLE t(x INTEGER, y INTEGER)");
        if with_index {
            exec(&mut db, "CREATE INDEX ixy ON t(x, y)");
        }
        for v in rows {
            exec(&mut db, &format!("INSERT INTO t VALUES ({v})"));
        }
        db
    };
    let mut scan = build(false);
    let mut seek = build(true);
    let scan_min = scalar(&mut scan, "SELECT MIN(x) FROM t");
    let seek_min = scalar(&mut seek, "SELECT MIN(x) FROM t");
    let scan_max = scalar(&mut scan, "SELECT MAX(x) FROM t");
    let seek_max = scalar(&mut seek, "SELECT MAX(x) FROM t");
    assert!(value_eq(&seek_min, &int(3)) && value_eq(&scan_min, &int(3)), "MIN(x): seek {seek_min:?} scan {scan_min:?}");
    assert!(value_eq(&seek_max, &int(8)) && value_eq(&scan_max, &int(8)), "MAX(x): seek {seek_max:?} scan {scan_max:?}");
}

// ---- rowid / INTEGER PRIMARY KEY: the table b-tree seek ----------------------

#[test]
fn rowid_alias_min_max_matches_a_parallel_unindexed_column() {
    // `id INTEGER PRIMARY KEY` aliases the rowid, so MIN/MAX(id) seek the table b-tree. The
    // parallel column `y` (= id, NO index) has no seek path, so `MAX(y)` is the full-scan
    // oracle: the two must agree, and equal the known extremum.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, y INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (10, 10), (3, 3), (27, 27), (8, 8)");
    let seek_min = scalar(&mut db, "SELECT MIN(id) FROM t");
    let seek_max = scalar(&mut db, "SELECT MAX(id) FROM t");
    let scan_min = scalar(&mut db, "SELECT MIN(y) FROM t");
    let scan_max = scalar(&mut db, "SELECT MAX(y) FROM t");
    assert!(value_eq(&seek_min, &int(3)), "MIN(id) = {seek_min:?}, want 3");
    assert!(value_eq(&seek_max, &int(27)), "MAX(id) = {seek_max:?}, want 27");
    assert!(value_eq(&seek_min, &scan_min), "MIN(id) seek {seek_min:?} vs MIN(y) scan {scan_min:?}");
    assert!(value_eq(&seek_max, &scan_max), "MAX(id) seek {seek_max:?} vs MAX(y) scan {scan_max:?}");
}

#[test]
fn rowid_keyword_min_max() {
    // The bare `rowid` keyword names the table b-tree even without an INTEGER PK alias. Rows
    // get rowids 1,2,3, so MIN(rowid)=1, MAX(rowid)=3.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "INSERT INTO t VALUES ('p'), ('q'), ('r')");
    assert!(value_eq(&scalar(&mut db, "SELECT MIN(rowid) FROM t"), &int(1)));
    assert!(value_eq(&scalar(&mut db, "SELECT MAX(rowid) FROM t"), &int(3)));
    assert!(value_eq(&scalar(&mut db, "SELECT MAX(_rowid_) FROM t"), &int(3)));
    assert!(value_eq(&scalar(&mut db, "SELECT MAX(oid) FROM t"), &int(3)));
}

#[test]
fn rowid_alias_empty_table_is_null() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY)");
    assert!(value_eq(&scalar(&mut db, "SELECT MIN(id) FROM t"), &null()));
    assert!(value_eq(&scalar(&mut db, "SELECT MAX(id) FROM t"), &null()));
}

// ---- Outer expression over the extremum (`MAX(x)+1`) --------------------------

#[test]
fn outer_expression_is_applied_after_the_seek() {
    // optoverview.html §13's own example `SELECT MAX(x)+1`: the seek computes MAX(x); the
    // `+1` is applied afterward, exactly as on the scan path.
    let vals = ["3", "1", "NULL", "2"];
    assert_equiv("INTEGER", "x", &vals, "SELECT MAX(x)+1 FROM t", int(4));
    assert_equiv("INTEGER", "x", &vals, "SELECT MIN(x)+1 FROM t", int(2));
    // The output COLUMN NAME is unchanged by the optimization.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x INTEGER)");
    exec(&mut db, "CREATE INDEX ix ON t(x)");
    insert_rows(&mut db, &vals);
    assert_columns(&mut db, "SELECT MAX(x)+1 FROM t", &["MAX(x)+1"]);
}

// ---- LIMIT interacts with the seek exactly as with the aggregate ---------------

#[test]
fn limit_wraps_the_single_seek_row() {
    // A bare MIN/MAX returns one row; the `Limit` wraps ABOVE the seek (same as above the
    // aggregate it replaces). `LIMIT 0` therefore yields ZERO rows — the seek's one row is
    // discarded by the Limit, not mis-optimized away — and `LIMIT 1` keeps the extremum.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x INTEGER)");
    exec(&mut db, "CREATE INDEX ix ON t(x)");
    insert_rows(&mut db, &["3", "1", "2"]);
    assert_rows(&mut db, "SELECT MIN(x) FROM t LIMIT 0", &[]);
    assert_rows(&mut db, "SELECT MAX(x) FROM t LIMIT 0", &[]);
    assert_rows(&mut db, "SELECT MIN(x) FROM t LIMIT 1", &[vec![int(1)]]);
    assert_rows(&mut db, "SELECT MAX(x) FROM t LIMIT 1", &[vec![int(3)]]);
}

// ---- Ineligible shapes: the optimization must NOT change the result -----------
// Each runs against a table that DOES have an index on the column, to prove the planner
// declines the seek and the ordinary aggregate still produces the correct answer.

fn indexed_xy(values: &[&str]) -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x INTEGER, y INTEGER)");
    exec(&mut db, "CREATE INDEX ix ON t(x)");
    for v in values {
        exec(&mut db, &format!("INSERT INTO t VALUES ({v})"));
    }
    db
}

#[test]
fn two_aggregates_still_scan() {
    // `MAX(x), COUNT(*)` is two aggregates → ineligible; both columns must be right.
    let mut db = indexed_xy(&["1, 10", "5, 20", "3, 30"]);
    assert_rows(&mut db, "SELECT MAX(x), COUNT(*) FROM t", &[vec![int(5), int(3)]]);
    assert_rows(&mut db, "SELECT MAX(x), MIN(x) FROM t", &[vec![int(5), int(1)]]);
}

#[test]
fn expression_argument_still_scans() {
    let mut db = indexed_xy(&["1, 0", "5, 0", "3, 0"]);
    assert!(value_eq(&scalar(&mut db, "SELECT MAX(x+1) FROM t"), &int(6)));
    assert!(value_eq(&scalar(&mut db, "SELECT MAX(x*1) FROM t"), &int(5)));
}

#[test]
fn where_clause_still_scans() {
    let mut db = indexed_xy(&["1, 0", "5, 0", "3, 0", "9, 0"]);
    assert!(value_eq(&scalar(&mut db, "SELECT MAX(x) FROM t WHERE x < 5"), &int(3)));
    assert!(value_eq(&scalar(&mut db, "SELECT MIN(x) FROM t WHERE x > 1"), &int(3)));
}

#[test]
fn group_by_still_scans() {
    let mut db = indexed_xy(&["1, 100", "5, 100", "3, 200", "9, 200"]);
    // MAX(x) per y group: y=100 → 5, y=200 → 9.
    assert_rows_unordered(
        &mut db,
        "SELECT y, MAX(x) FROM t GROUP BY y",
        &[vec![int(100), int(5)], vec![int(200), int(9)]],
    );
}

#[test]
fn distinct_argument_still_scans() {
    // MAX(DISTINCT x) is ineligible (conservatively skipped); DISTINCT does not change a
    // min/max, so the result equals plain MAX(x).
    let mut db = indexed_xy(&["1, 0", "5, 0", "5, 0", "3, 0"]);
    assert!(value_eq(&scalar(&mut db, "SELECT MAX(DISTINCT x) FROM t"), &int(5)));
    assert!(value_eq(&scalar(&mut db, "SELECT MIN(DISTINCT x) FROM t"), &int(1)));
}

#[test]
fn blob_affinity_column_still_scans() {
    // A BLOB-affinity (typeless) column is ineligible for the index seek (cross-storage-class
    // ties are not representation-stable), so the scan runs. With homogeneous integers there
    // is no tie and the answer is the obvious one.
    let mut scan = mem();
    exec(&mut scan, "CREATE TABLE t(x)");
    insert_rows(&mut scan, &["3", "1", "2"]);
    let mut seek = mem();
    exec(&mut seek, "CREATE TABLE t(x)");
    exec(&mut seek, "CREATE INDEX ix ON t(x)");
    insert_rows(&mut seek, &["3", "1", "2"]);
    assert!(value_eq(&scalar(&mut scan, "SELECT MAX(x) FROM t"), &int(3)));
    assert!(value_eq(&scalar(&mut seek, "SELECT MAX(x) FROM t"), &int(3)));
    assert!(value_eq(&scalar(&mut scan, "SELECT MIN(x) FROM t"), &int(1)));
    assert!(value_eq(&scalar(&mut seek, "SELECT MIN(x) FROM t"), &int(1)));
}
