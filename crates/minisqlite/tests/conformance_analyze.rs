//! Conformance battery: the **`ANALYZE` command** and the `sqlite_stat1` table it
//! creates and populates.
//!
//! Every expected value here is TRANSCRIBED FROM THE SQLITE DOCS
//! (`spec/sqlite-doc/`), never from what the engine happens to return — a failing
//! case is the intended signal of a spec divergence, and assertions are never
//! weakened to pass.
//!
//! Spec sources:
//!   * `lang_analyze.html` — statement syntax and scope: `ANALYZE` (whole schema),
//!     `ANALYZE <schema>`, `ANALYZE <table-or-index>`; all return no rows.
//!   * `fileformat2.html` §2.6.4 "The sqlite_stat1 table" — schema
//!     `CREATE TABLE sqlite_stat1(tbl,idx,stat)`; one row per index whose `stat` is
//!     `"N a1 ... aK"` (N = rows in the index; the m-th following integer is the
//!     average number of rows sharing the left-most m columns; a unique index's last
//!     integer is 1, unconditionally — so each NULL counts as its own distinct value);
//!     a table with no index instead gets one `idx = NULL` row whose `stat` is the row
//!     count; a WITHOUT ROWID table gets a row with `idx == tbl` (the stat over its PK
//!     b-tree).
//!   * `atomiccommit.html` — a committed `ANALYZE` is durable across close+reopen.
//!
//! The stat NUMBERS assert round-to-nearest of `N / nDistinct` (`(N + nDistinct/2) /
//! nDistinct`, a spec-correct approximation of §2.6.4's "estimated average"); each
//! case's comment shows the sorted key multiset it is derived from, so the expectation
//! is checkable by hand, not read off the engine.
//!
//! Each case is its own `#[test]`, so an unsupported behavior fails exactly that
//! case rather than masking the rest.

mod conformance;
use conformance::*;

use minisqlite::Connection;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

// ===========================================================================
// sqlite_stat1 shape and existence (fileformat2 §2.6.4).
// ===========================================================================

#[test]
fn analyze_empty_database_creates_empty_stat1() {
    // lang_analyze / §2.6.4: `ANALYZE` on an empty database still CREATES the
    // (internal) sqlite_stat1 table; it just has no rows to hold.
    let mut db = mem();
    exec(&mut db, "ANALYZE");
    assert_scalar(&mut db, "SELECT count(*) FROM sqlite_stat1", int(0));
    assert_scalar(
        &mut db,
        "SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name = 'sqlite_stat1'",
        int(1),
    );
}

#[test]
fn analyze_returns_no_rows() {
    // lang_analyze: `ANALYZE`, `ANALYZE <schema>`, `ANALYZE <table>` each produce NO
    // result set. Run through `query()` (which surfaces any rows) and assert emptiness —
    // `execute()` discards result rows, so it would pass even if the dispatch arm
    // wrongly returned a result set.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    for sql in ["ANALYZE", "ANALYZE main", "ANALYZE t"] {
        let qr = query(&mut db, sql);
        assert!(qr.rows.is_empty(), "{sql} must return no rows, got {:?}", qr.rows);
    }
}

#[test]
fn stat1_schema_sql_is_the_documented_text() {
    // §2.6.4: the schema is exactly `CREATE TABLE sqlite_stat1(tbl,idx,stat)` — three
    // columns, no type names — so the stored `sql` byte-matches what real SQLite writes.
    let mut db = mem();
    exec(&mut db, "ANALYZE");
    assert_scalar(
        &mut db,
        "SELECT sql FROM sqlite_master WHERE name = 'sqlite_stat1'",
        text("CREATE TABLE sqlite_stat1(tbl,idx,stat)"),
    );
}

#[test]
fn stat1_select_star_has_three_columns_in_order() {
    // §2.6.4: the columns are (tbl, idx, stat), in that order.
    let mut db = mem();
    exec(&mut db, "ANALYZE");
    assert_columns(&mut db, "SELECT * FROM sqlite_stat1", &["tbl", "idx", "stat"]);
}

// ===========================================================================
// A table with NO index → the idx=NULL, stat=row-count row (§2.6.4).
// ===========================================================================

#[test]
fn table_without_index_emits_null_idx_row_with_row_count() {
    // §2.6.4: "If the sqlite_stat1.idx column is NULL, then the sqlite_stat1.stat
    // column contains a single integer which is the approximate number of rows in the
    // table." A table with no index gets exactly that row.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "INSERT INTO t VALUES (1, 'x'), (2, 'y'), (3, 'z')");
    exec(&mut db, "ANALYZE");
    assert_rows(
        &mut db,
        "SELECT tbl, idx, stat FROM sqlite_stat1",
        &[vec![text("t"), null(), text("3")]],
    );
}

#[test]
fn empty_table_without_index_emits_no_row() {
    // An empty table describes nothing, so ANALYZE writes no stat row for it (the
    // table still exists; sqlite_stat1 is simply empty). Documents the skip-empty rule.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "ANALYZE");
    assert_scalar(&mut db, "SELECT count(*) FROM sqlite_stat1", int(0));
    // sqlite_stat1 itself is still created.
    assert_scalar(
        &mut db,
        "SELECT count(*) FROM sqlite_master WHERE name = 'sqlite_stat1'",
        int(1),
    );
}

// ===========================================================================
// Index rows: single-column, multi-column, unique (§2.6.4 stat format).
// ===========================================================================

#[test]
fn single_column_index_stat() {
    // Index keys (sorted): 1, 1, 2, 3, 3, 3 → N = 6, 3 distinct values in column 0.
    // stat = "6 2": 6 rows, and 6/3 = 2 rows on average per distinct value.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "CREATE INDEX ix ON t(a)");
    exec(&mut db, "INSERT INTO t VALUES (3), (1), (3), (2), (1), (3)");
    exec(&mut db, "ANALYZE");
    assert_rows(
        &mut db,
        "SELECT tbl, idx, stat FROM sqlite_stat1",
        &[vec![text("t"), text("ix"), text("6 2")]],
    );
}

#[test]
fn multi_column_index_stat() {
    // Keys (sorted (a,b)): (1,1),(1,2),(2,5),(2,5),(2,6) → N = 5.
    // distinct(a) = 2 → 5/2 = 2.5 → 3; distinct(a,b) = 4 → 5/4 = 1.25 → 1.
    // stat = "5 3 1". Rows inserted out of order to prove the scan sorts them, and a
    // duplicate (2,5) exercises a non-unique multi-column key.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "CREATE INDEX ix ON t(a, b)");
    exec(&mut db, "INSERT INTO t VALUES (2,6), (1,1), (2,5), (1,2), (2,5)");
    exec(&mut db, "ANALYZE");
    assert_rows(
        &mut db,
        "SELECT tbl, idx, stat FROM sqlite_stat1",
        &[vec![text("t"), text("ix"), text("5 3 1")]],
    );
}

#[test]
fn unique_index_last_integer_is_one() {
    // §2.6.4: "If the index is unique, then the last integer will be 1." Keys 10,20,30
    // → N = 3, all distinct → stat = "3 1".
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "CREATE UNIQUE INDEX ux ON t(a)");
    exec(&mut db, "INSERT INTO t VALUES (10), (20), (30)");
    exec(&mut db, "ANALYZE");
    assert_rows(
        &mut db,
        "SELECT tbl, idx, stat FROM sqlite_stat1",
        &[vec![text("t"), text("ux"), text("3 1")]],
    );
}

// ===========================================================================
// NULL keys: each NULL is its own distinct value, so a UNIQUE index's last integer
// stays 1 unconditionally (fileformat2 §2.6.4). SQLite allows multiple NULLs in a
// UNIQUE index, so this invariant must hold across duplicate NULLs.
// ===========================================================================

#[test]
fn unique_index_with_duplicate_nulls_last_integer_is_one() {
    // Index keys sorted: NULL, NULL, 1 → N = 3. Each NULL is a DISTINCT value, so
    // distinct(a) = 3 = N and the unique index's last integer is 1 (not 2). This pins
    // §2.6.4's unconditional "unique → last integer 1" against duplicate NULLs.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "CREATE UNIQUE INDEX u ON t(a)");
    exec(&mut db, "INSERT INTO t VALUES (NULL), (NULL), (1)");
    exec(&mut db, "ANALYZE");
    assert_rows(
        &mut db,
        "SELECT tbl, idx, stat FROM sqlite_stat1",
        &[vec![text("t"), text("u"), text("3 1")]],
    );
}

#[test]
fn non_unique_index_counts_each_null_as_distinct() {
    // A NON-unique index over NULL, NULL, NULL, 5 → N = 4. Each NULL is distinct, so
    // distinct(a) = 4 → 4/4 = 1: stat "4 1" (not "4 2", which collapsing the NULLs to
    // one value would wrongly produce).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "CREATE INDEX i ON t(a)");
    exec(&mut db, "INSERT INTO t VALUES (NULL), (NULL), (NULL), (5)");
    exec(&mut db, "ANALYZE");
    assert_rows(
        &mut db,
        "SELECT tbl, idx, stat FROM sqlite_stat1",
        &[vec![text("t"), text("i"), text("4 1")]],
    );
}

#[test]
fn index_mixes_distinct_nulls_with_repeated_non_nulls() {
    // Keys sorted: NULL, NULL, 5, 5, 5 → N = 5. Distinct values: two distinct NULLs
    // plus the single value 5 = 3 distinct. 5/3 = 1.67 → 2: stat "5 2". Proves NULLs
    // and repeated non-NULLs are counted together correctly (NULLs split, 5s merge).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "CREATE INDEX i ON t(a)");
    exec(&mut db, "INSERT INTO t VALUES (NULL), (NULL), (5), (5), (5)");
    exec(&mut db, "ANALYZE");
    assert_rows(
        &mut db,
        "SELECT tbl, idx, stat FROM sqlite_stat1",
        &[vec![text("t"), text("i"), text("5 2")]],
    );
}

#[test]
fn table_with_index_has_no_null_idx_row() {
    // §2.6.4: the idx=NULL row exists only for a table with NO index — a table that
    // has one gets the index row (whose first integer already carries the row count),
    // never an additional NULL row.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "CREATE INDEX ix ON t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1), (2)");
    exec(&mut db, "ANALYZE");
    assert_scalar(
        &mut db,
        "SELECT count(*) FROM sqlite_stat1 WHERE idx IS NULL",
        int(0),
    );
}

#[test]
fn empty_index_emits_no_row() {
    // An index over an empty table has nothing to describe → no stat row (skip-empty).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "CREATE INDEX ix ON t(a)");
    exec(&mut db, "ANALYZE");
    assert_scalar(&mut db, "SELECT count(*) FROM sqlite_stat1", int(0));
}

// ===========================================================================
// Re-running ANALYZE refreshes rows (no duplicates).
// ===========================================================================

#[test]
fn reanalyze_refreshes_row_without_duplicating() {
    // Running ANALYZE again REPLACES a table's rows rather than appending. After a
    // second ANALYZE the no-index table still has exactly ONE row, with the NEW count.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1), (2)");
    exec(&mut db, "ANALYZE");
    assert_rows(
        &mut db,
        "SELECT tbl, idx, stat FROM sqlite_stat1",
        &[vec![text("t"), null(), text("2")]],
    );
    exec(&mut db, "INSERT INTO t VALUES (3)");
    exec(&mut db, "ANALYZE");
    assert_rows(
        &mut db,
        "SELECT tbl, idx, stat FROM sqlite_stat1",
        &[vec![text("t"), null(), text("3")]],
    );
    // No-duplicate contract: the second ANALYZE must REUSE the existing sqlite_stat1,
    // not create a second internal table. The row assertions above cannot catch a
    // duplicate (a query reads only the newest root), so pin the table count directly —
    // this fails if the `ensure_sqlite_stat1` idempotency guard is ever dropped.
    assert_scalar(
        &mut db,
        "SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name = 'sqlite_stat1'",
        int(1),
    );
}

// ===========================================================================
// Scope: ANALYZE <table> and ANALYZE <index> restrict correctly (lang_analyze).
// ===========================================================================

#[test]
fn analyze_one_table_does_not_touch_others() {
    // `ANALYZE t1` analyzes only t1. t2 (never analyzed) has no stat row yet; a later
    // `ANALYZE t2` adds t2's row and LEAVES t1's in place (table scope deletes only
    // the analyzed table's rows).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t1(a)");
    exec(&mut db, "CREATE INDEX i1 ON t1(a)");
    exec(&mut db, "INSERT INTO t1 VALUES (1), (2)"); // 2 rows, 2 distinct → "2 1"
    exec(&mut db, "CREATE TABLE t2(a)");
    exec(&mut db, "CREATE INDEX i2 ON t2(a)");
    exec(&mut db, "INSERT INTO t2 VALUES (5), (5), (5)"); // 3 rows, 1 distinct → "3 3"

    exec(&mut db, "ANALYZE t1");
    assert_rows(
        &mut db,
        "SELECT tbl, idx, stat FROM sqlite_stat1 ORDER BY tbl",
        &[vec![text("t1"), text("i1"), text("2 1")]],
    );

    exec(&mut db, "ANALYZE t2");
    assert_rows(
        &mut db,
        "SELECT tbl, idx, stat FROM sqlite_stat1 ORDER BY tbl",
        &[
            vec![text("t1"), text("i1"), text("2 1")],
            vec![text("t2"), text("i2"), text("3 3")],
        ],
    );
}

#[test]
fn analyze_one_table_analyzes_all_of_its_indexes() {
    // lang_analyze: `ANALYZE <table>` analyzes THAT table "and all of its indexes". A
    // table with two indexes must get BOTH index rows in the single call (the
    // table-scope path iterates every index, not just the first).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "CREATE INDEX ia ON t(a)");
    exec(&mut db, "CREATE INDEX ib ON t(b)");
    exec(&mut db, "INSERT INTO t VALUES (1, 5), (2, 5), (3, 6)"); // a:1,2,3 ; b:5,5,6
    exec(&mut db, "ANALYZE t");
    assert_rows(
        &mut db,
        "SELECT tbl, idx, stat FROM sqlite_stat1 ORDER BY idx",
        &[
            vec![text("t"), text("ia"), text("3 1")], // 3 rows, 3 distinct
            vec![text("t"), text("ib"), text("3 2")], // 3 rows, 2 distinct (5,5,6)
        ],
    );
}

#[test]
fn analyze_one_index_refreshes_only_that_index() {
    // `ANALYZE <index>` analyzes ONLY that index and clears only its own row, so a
    // sibling index's row on the same table is left untouched (index scope, not table
    // scope). Here `ANALYZE ib` refreshes ib to the new row count while ia keeps its
    // earlier (now-stale) value — proof the delete was keyed on idx, not tbl.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "CREATE INDEX ia ON t(a)");
    exec(&mut db, "CREATE INDEX ib ON t(b)");
    exec(&mut db, "INSERT INTO t VALUES (1,5), (2,5), (3,6)"); // a:1,2,3 ; b:5,5,6

    exec(&mut db, "ANALYZE ia");
    // ia: 3 rows, 3 distinct → "3 1"; ib not analyzed yet.
    assert_rows(
        &mut db,
        "SELECT tbl, idx, stat FROM sqlite_stat1 ORDER BY idx",
        &[vec![text("t"), text("ia"), text("3 1")]],
    );

    exec(&mut db, "ANALYZE ib");
    // ib added (b sorted 5,5,6 → 3 rows, 2 distinct → 3/2 → "3 2"); ia still present.
    assert_rows(
        &mut db,
        "SELECT tbl, idx, stat FROM sqlite_stat1 ORDER BY idx",
        &[
            vec![text("t"), text("ia"), text("3 1")],
            vec![text("t"), text("ib"), text("3 2")],
        ],
    );

    // Grow the table, then ANALYZE only ib: ib refreshes to N=4, ia keeps its stale
    // "3 1" — a table-scoped delete would have removed ia, so its survival pins the
    // index scope.
    exec(&mut db, "INSERT INTO t VALUES (4,6)"); // a:1..4 ; b:5,5,6,6
    exec(&mut db, "ANALYZE ib");
    assert_rows(
        &mut db,
        "SELECT tbl, idx, stat FROM sqlite_stat1 ORDER BY idx",
        &[
            vec![text("t"), text("ia"), text("3 1")], // stale, untouched
            vec![text("t"), text("ib"), text("4 2")], // 4 rows, 2 distinct → 4/2 = 2
        ],
    );
}

#[test]
fn analyze_main_analyzes_every_table() {
    // lang_analyze: a lone database name analyzes that whole schema. `ANALYZE main`
    // covers every user table.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t1(a)");
    exec(&mut db, "CREATE INDEX i1 ON t1(a)");
    exec(&mut db, "INSERT INTO t1 VALUES (1), (2)");
    exec(&mut db, "CREATE TABLE t2(a)");
    exec(&mut db, "CREATE INDEX i2 ON t2(a)");
    exec(&mut db, "INSERT INTO t2 VALUES (7), (7), (8)");
    exec(&mut db, "ANALYZE main");
    assert_rows(
        &mut db,
        "SELECT tbl, idx, stat FROM sqlite_stat1 ORDER BY tbl",
        &[
            vec![text("t1"), text("i1"), text("2 1")],
            vec![text("t2"), text("i2"), text("3 2")], // 7,7,8 → 3 rows, 2 distinct
        ],
    );
}

#[test]
fn analyze_temp_schema_is_a_noop_and_creates_no_stat1() {
    // This single-file engine has no temp schema, so `ANALYZE temp` analyzes nothing.
    // Unlike whole-db `ANALYZE`, it must NOT even create main's sqlite_stat1: real SQLite
    // targets the temp schema and leaves main's sqlite_master untouched (a bare `temp`
    // name selects the temp database). So after `ANALYZE temp` on a populated main db,
    // main has no sqlite_stat1 at all. (Contrast `analyze_empty_database_creates_empty_stat1`,
    // where whole-db ANALYZE DOES create it.)
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    let qr = query(&mut db, "ANALYZE temp");
    assert!(qr.rows.is_empty(), "ANALYZE temp must return no rows, got {:?}", qr.rows);
    assert_scalar(
        &mut db,
        "SELECT count(*) FROM sqlite_master WHERE name = 'sqlite_stat1'",
        int(0),
    );
}

// ===========================================================================
// Unknown target errors like SQLite, and creates nothing (lang_analyze).
// ===========================================================================

#[test]
fn analyze_unknown_target_errors_and_creates_no_stat1() {
    // `ANALYZE <name>` with no such table/index errors ("no such table: ..."), and —
    // because the name is resolved before anything is written — leaves no sqlite_stat1.
    let mut db = mem();
    let e = assert_exec_error(&mut db, "ANALYZE nope");
    let msg = format!("{e:?}");
    assert!(msg.contains("no such table"), "expected a 'no such table' error, got: {msg}");
    assert_scalar(
        &mut db,
        "SELECT count(*) FROM sqlite_master WHERE name = 'sqlite_stat1'",
        int(0),
    );
}

// ===========================================================================
// ANALYZE is write-only: query results are identical with and without it
// (the planner must not start consuming sqlite_stat1).
// ===========================================================================

#[test]
fn analyze_does_not_change_query_results() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "CREATE INDEX ix ON t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1), (2), (2), (3)");
    let before = "SELECT a FROM t WHERE a = 2 ORDER BY a";
    let expected = &[vec![int(2)], vec![int(2)]];
    assert_rows(&mut db, before, expected);
    exec(&mut db, "ANALYZE");
    // Same query, same rows — the stats did not alter the answer.
    assert_rows(&mut db, before, expected);
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(4));
}

// ===========================================================================
// Atomicity: a rolled-back ANALYZE leaves no trace (atomiccommit.html).
// ===========================================================================

#[test]
fn rolled_back_analyze_leaves_no_trace() {
    // ANALYZE runs inside the surrounding transaction. When it (as the first ANALYZE)
    // creates and populates sqlite_stat1 and the transaction is rolled back, BOTH the
    // table and its rows must vanish — the schema reverts to its pre-transaction image.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1), (2)");

    exec(&mut db, "BEGIN");
    exec(&mut db, "ANALYZE");
    // Visible within the open transaction.
    assert_scalar(&mut db, "SELECT count(*) FROM sqlite_stat1", int(1));
    exec(&mut db, "ROLLBACK");

    // Gone after rollback: sqlite_stat1 was created by the rolled-back ANALYZE, so it
    // no longer exists at all.
    assert_scalar(
        &mut db,
        "SELECT count(*) FROM sqlite_master WHERE name = 'sqlite_stat1'",
        int(0),
    );
    // The user table (created and committed before the transaction) is untouched.
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2));
}

// ===========================================================================
// WITHOUT ROWID: the PK b-tree's stat row has idx == tbl (fileformat2 §2.6.4).
// ===========================================================================

#[test]
fn without_rowid_table_pk_row_has_idx_equal_tbl() {
    // §2.6.4: "If the sqlite_stat1.idx column is the same as the sqlite_stat1.tbl
    // column, then the table is a WITHOUT ROWID table and the sqlite_stat1.stat field
    // contains information about the index btree that implements the WITHOUT ROWID
    // table." A WR table stores its rows in its PRIMARY KEY b-tree, so its stat row is
    // the index stat over that b-tree with idx == tbl. Here PK `k` has 3 distinct
    // values over 3 rows and the PK is unique, so the last integer is 1 → "3 1". A
    // rowid table alongside it is analyzed independently (idx = NULL, count "2").
    let mut db = mem();
    exec(&mut db, "CREATE TABLE w(k, v, PRIMARY KEY(k)) WITHOUT ROWID");
    exec(&mut db, "INSERT INTO w VALUES ('a', 1), ('b', 2), ('c', 2)");
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "INSERT INTO t VALUES (10), (20)"); // no index → NULL-idx row "2"
    exec(&mut db, "ANALYZE");
    assert_rows(
        &mut db,
        "SELECT tbl, idx, stat FROM sqlite_stat1 ORDER BY tbl",
        &[
            vec![text("t"), null(), text("2")],
            vec![text("w"), text("w"), text("3 1")],
        ],
    );
}

#[test]
fn without_rowid_bare_table_emits_only_its_pk_row() {
    // A WITHOUT ROWID table with no secondary index gets EXACTLY its idx == tbl PK row
    // — never an idx=NULL row (that form is only for a ROWID table with no index).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE w(k, v, PRIMARY KEY(k)) WITHOUT ROWID");
    exec(&mut db, "INSERT INTO w VALUES ('a', 1), ('b', 2), ('c', 3)");
    exec(&mut db, "ANALYZE");
    assert_rows(
        &mut db,
        "SELECT tbl, idx, stat FROM sqlite_stat1",
        &[vec![text("w"), text("w"), text("3 1")]],
    );
    assert_scalar(
        &mut db,
        "SELECT count(*) FROM sqlite_stat1 WHERE idx IS NULL",
        int(0),
    );
}

#[test]
fn without_rowid_composite_pk_stat_spans_all_key_columns() {
    // A composite PRIMARY KEY (a, b) is a K=2 index over the PK b-tree → K+1 = 3
    // integers. Keys sorted (a,b): (1,1),(1,2),(2,1) → N=3; distinct(a)=2 → 3/2 → 2;
    // distinct(a,b)=3 (unique PK) → 1. stat = "3 2 1".
    let mut db = mem();
    exec(&mut db, "CREATE TABLE w(a, b, c, PRIMARY KEY(a, b)) WITHOUT ROWID");
    exec(&mut db, "INSERT INTO w VALUES (1, 1, 'x'), (1, 2, 'y'), (2, 1, 'z')");
    exec(&mut db, "ANALYZE");
    assert_rows(
        &mut db,
        "SELECT tbl, idx, stat FROM sqlite_stat1",
        &[vec![text("w"), text("w"), text("3 2 1")]],
    );
}

#[test]
fn analyze_without_rowid_table_by_name_refreshes_its_pk_row() {
    // `ANALYZE <wr-table>` scopes to that table; because the PK row's idx == tbl, the
    // table-scoped delete clears and re-inserts it (no duplication on re-run).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE w(k, v, PRIMARY KEY(k)) WITHOUT ROWID");
    exec(&mut db, "INSERT INTO w VALUES ('a', 1), ('b', 2)");
    exec(&mut db, "ANALYZE w");
    assert_rows(
        &mut db,
        "SELECT tbl, idx, stat FROM sqlite_stat1",
        &[vec![text("w"), text("w"), text("2 1")]],
    );
    exec(&mut db, "INSERT INTO w VALUES ('c', 3)");
    exec(&mut db, "ANALYZE w");
    assert_rows(
        &mut db,
        "SELECT tbl, idx, stat FROM sqlite_stat1",
        &[vec![text("w"), text("w"), text("3 1")]],
    );
}

#[test]
fn without_rowid_column_level_integer_pk_emits_pk_row() {
    // A column-level INTEGER PRIMARY KEY WITHOUT ROWID has no auto-index spec (an integer
    // PK is excluded); `wr_pk_key_col_count` recovers its PK column from `TableDef::primary_key`
    // (the single authority, populated for every PK form), not from the per-column flag. The
    // idx == tbl PK row must still be emitted (K = 1). Keys sorted: 1,2,3 → N = 3, unique PK
    // → "3 1". Pins the integer-PK ANALYZE path end-to-end.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, b) WITHOUT ROWID");
    exec(&mut db, "INSERT INTO t VALUES (3, 'x'), (1, 'y'), (2, 'z')");
    exec(&mut db, "ANALYZE");
    assert_rows(
        &mut db,
        "SELECT tbl, idx, stat FROM sqlite_stat1",
        &[vec![text("t"), text("t"), text("3 1")]],
    );
}

#[test]
fn without_rowid_table_level_integer_pk_emits_pk_row() {
    // A TABLE-LEVEL single-column INTEGER PRIMARY KEY (`PRIMARY KEY(a)` with `a INTEGER`)
    // on a WITHOUT ROWID table reserves no auto-index ordinal (an integer PK is excluded)
    // AND sets no per-column flag (a table-level constraint), yet `TableDef::primary_key`
    // records it as `[a]` — the single authority ANALYZE now reads. So K = 1 is recovered
    // and the idx == tbl PK row is emitted like any other WR table. Keys sorted: 1,2,3 →
    // N = 3, unique PK → "3 1". (Once an honest residual that SKIPPED this row — now CLOSED,
    // the same `TableDef::primary_key` fix `wr_layout` uses to let this shape hold rows.)
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b, PRIMARY KEY(a)) WITHOUT ROWID");
    exec(&mut db, "INSERT INTO t VALUES (3, 'x'), (1, 'y'), (2, 'z')");
    exec(&mut db, "ANALYZE");
    assert_rows(
        &mut db,
        "SELECT tbl, idx, stat FROM sqlite_stat1",
        &[vec![text("t"), text("t"), text("3 1")]],
    );
}

// ===========================================================================
// On-disk durability: sqlite_stat1 and its rows survive close + reopen.
// ===========================================================================

/// An RAII guard over a unique temporary database path. Each test gets its own file,
/// and `Drop` removes the database plus its `-journal` / `-wal` / `-shm` sidecars so
/// nothing is left behind even on a panic. No `.db` file is ever committed.
struct TempDb {
    path: PathBuf,
}

impl TempDb {
    fn new() -> TempDb {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
        let mut path = std::env::temp_dir();
        path.push(format!("minisqlite_analyze_{pid}_{n}_{nanos}.db"));
        let db = TempDb { path };
        db.remove_all();
        db
    }

    fn open(&self) -> Connection {
        let path: &Path = &self.path;
        Connection::open(path)
            .unwrap_or_else(|e| panic!("Connection::open failed for {path:?}: {e:?}"))
    }

    fn sidecar(&self, suffix: &str) -> PathBuf {
        let mut s = self.path.as_os_str().to_os_string();
        s.push(suffix);
        PathBuf::from(s)
    }

    fn remove_all(&self) {
        let _ = std::fs::remove_file(&self.path);
        for suffix in ["-journal", "-wal", "-shm"] {
            let _ = std::fs::remove_file(self.sidecar(suffix));
        }
    }
}

impl Drop for TempDb {
    fn drop(&mut self) {
        self.remove_all();
    }
}

#[test]
fn stat1_persists_across_close_and_reopen() {
    // atomiccommit.html: a committed ANALYZE is durable. Write and
    // analyze through a file-backed connection, drop it, reopen the same path, and the
    // stat table (schema row + data row) is still there.
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(a)");
        exec(&mut db, "CREATE INDEX ix ON t(a)");
        exec(&mut db, "INSERT INTO t VALUES (3), (1), (3), (2), (1), (3)"); // → "6 2"
        exec(&mut db, "ANALYZE");
    } // connection dropped: file closed.

    let mut db = tmp.open(); // reopen the SAME path.
    assert_scalar(
        &mut db,
        "SELECT count(*) FROM sqlite_master WHERE name = 'sqlite_stat1'",
        int(1),
    );
    assert_rows(
        &mut db,
        "SELECT tbl, idx, stat FROM sqlite_stat1",
        &[vec![text("t"), text("ix"), text("6 2")]],
    );
}

#[test]
fn without_rowid_table_level_integer_pk_reopens_and_analyzes() {
    // Durability for the fixed shape: a TABLE-LEVEL single INTEGER PRIMARY KEY on
    // a WITHOUT ROWID table must survive close+reopen — its rows read back AND ANALYZE emits
    // its idx == tbl PK row. The reopen exercises the LOAD path (`table_def_from_sql` ->
    // `table_def_from_ast`), which populates `TableDef::primary_key` the same way CREATE
    // does, so `wr_pk_columns` / `wr_pk_key_col_count` recover K = 1 for the loaded def too
    // (the no-fallback count is safe because primary_key is populated on load, not only on
    // create). Keys sorted 1,2,3 → N = 3, unique PK → "3 1".
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT, PRIMARY KEY(a)) WITHOUT ROWID");
        exec(&mut db, "INSERT INTO t VALUES (3, 'x'), (1, 'y'), (2, 'z')");
        exec(&mut db, "ANALYZE");
    } // connection dropped: file closed.

    let mut db = tmp.open(); // reopen the SAME path.
    // Rows survive and read back in PRIMARY KEY (a) order through the loaded WR layout.
    assert_rows(
        &mut db,
        "SELECT a, b FROM t",
        &[vec![int(1), text("y")], vec![int(2), text("z")], vec![int(3), text("x")]],
    );
    // The WR PK stat row (idx == tbl, K = 1) persists across the reopen.
    assert_rows(
        &mut db,
        "SELECT tbl, idx, stat FROM sqlite_stat1",
        &[vec![text("t"), text("t"), text("3 1")]],
    );
}

#[test]
fn reanalyze_after_reopen_still_single_row() {
    // Durability plus idempotence: after reopening, a fresh ANALYZE refreshes rather
    // than duplicating the persisted rows.
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(a)");
        exec(&mut db, "INSERT INTO t VALUES (1), (2)");
        exec(&mut db, "ANALYZE"); // → t | NULL | 2
    }
    let mut db = tmp.open();
    exec(&mut db, "INSERT INTO t VALUES (3), (4)");
    exec(&mut db, "ANALYZE"); // refresh
    assert_rows(
        &mut db,
        "SELECT tbl, idx, stat FROM sqlite_stat1",
        &[vec![text("t"), null(), text("4")]],
    );
    // After reopen the catalog reloads sqlite_stat1 from page 1, so the post-reopen
    // ANALYZE must take the cache branch and NOT persist a second internal table.
    assert_scalar(
        &mut db,
        "SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name = 'sqlite_stat1'",
        int(1),
    );
}
