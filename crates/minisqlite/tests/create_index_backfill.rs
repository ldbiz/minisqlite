//! `CREATE INDEX` backfill conformance: building an index on an ALREADY-POPULATED
//! table copies every existing row into the new index, atomically.
//!
//! An index is a redundant copy of ALL of a table's rows
//! (`spec/sqlite-doc/lang_createindex.html`), so building one after the rows exist
//! must make every pre-existing row reachable through it — identical to the same query
//! with no index. The sibling `conformance_index_queries.rs` pins the read results of
//! the backfill (`backfill_*`); this file pins the behaviors that file does not:
//!
//!   * a `UNIQUE` index whose existing data holds a duplicate key FAILS the create with
//!     `UNIQUE constraint failed` and leaves NO trace — no index b-tree, no
//!     `sqlite_schema` row, no cached definition (so a later query and a later
//!     re-`CREATE` behave as if the failed one never ran). §1.1: "Any attempt to insert
//!     a duplicate entry will result in an error", but "all NULL values are considered
//!     different ... and are thus unique" (a UNIQUE index admits many NULLs).
//!   * the maintained path keeps working after a backfill (a later `INSERT` is indexed
//!     too), and `IF NOT EXISTS` over an already-built index is a clean no-op.
//!
//! Every expectation is derived from the DATA and the documented semantics, never from
//! engine output; `Value` has no `PartialEq`, so comparisons go through the harness.

mod conformance;
use conformance::*;

use minisqlite::Connection;

// ---------------------------------------------------------------------------
// Fixtures. `_populated` builds a table with rows, THEN (caller) creates the index —
// the backfill path — in contrast to the maintained path (index before rows).
// ---------------------------------------------------------------------------

/// Table `t(a INTEGER, b TEXT)` with distinct integer keys 1,2,3 (no index yet).
fn distinct_keys() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
    exec(&mut db, "INSERT INTO t VALUES (1,'x'),(2,'y'),(3,'z')");
    db
}

/// Table `t(a INTEGER, b TEXT)` with a DUPLICATE key (2 appears twice) plus a NULL.
fn duplicate_keys() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
    exec(&mut db, "INSERT INTO t VALUES (2,'two'),(5,'five'),(2,'twob'),(8,'eight'),(NULL,'nil')");
    db
}

/// How many indexes named `i` the schema reports (0 or 1). Uses `sqlite_master`,
/// which reads page 1 directly, so a rolled-back CREATE leaves nothing for it to see.
fn index_count(db: &mut Connection) -> i64 {
    let qr = query(db, "SELECT count(*) FROM sqlite_master WHERE type = 'index' AND name = 'i'");
    match &qr.rows[0][0] {
        minisqlite::Value::Integer(n) => *n,
        other => panic!("count(*) should be an Integer, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// UNIQUE backfill: duplicate existing data aborts the whole create atomically.
// ---------------------------------------------------------------------------

#[test]
fn unique_backfill_rejects_existing_duplicate() {
    // Two pre-existing rows share key 2, so building a UNIQUE index over them must
    // raise the SAME error INSERTing a duplicate would (§1.1), reusing the shared
    // UNIQUE probe. Wording is not asserted, but the SQLite-shaped prefix is asserted
    // so a regression to a generic error is caught.
    let mut db = duplicate_keys();
    let e = assert_exec_error(&mut db, "CREATE UNIQUE INDEX i ON t(a)");
    // Pin the SQLite-shaped message including the `: <table>.<col>` suffix real
    // sqlite emits (detail comes from the shared `build_index_plan`, same as the INSERT
    // path), not just the generic prefix.
    assert!(
        e.to_string().contains("UNIQUE constraint failed: t.a"),
        "a duplicate key in the backfilled data must fail with `UNIQUE constraint failed: t.a`; got {e:?}"
    );
}

#[test]
fn unique_backfill_failure_leaves_no_index_row() {
    // ATOMICITY: the failed create rolls back entirely — `sqlite_master` shows no
    // index `i` (the schema row was reverted with the index b-tree).
    let mut db = duplicate_keys();
    let _ = try_exec(&mut db, "CREATE UNIQUE INDEX i ON t(a)");
    assert_eq!(index_count(&mut db), 0, "a failed UNIQUE backfill must leave no sqlite_master index row");
}

#[test]
fn unique_backfill_failure_keeps_queries_correct() {
    // ATOMICITY (cache side): after the failed create the schema CACHE must not hold an
    // orphaned index def pointing at the discarded b-tree root. If it did, the planner
    // could pick that index and read reverted pages. A full-scan lookup after the
    // failure must still return every matching row, exactly as before the create.
    let mut db = duplicate_keys();
    let _ = try_exec(&mut db, "CREATE UNIQUE INDEX i ON t(a)");
    assert_rows_unordered(
        &mut db,
        "SELECT b FROM t WHERE a = 2",
        &[vec![text("two")], vec![text("twob")]],
    );
    // The whole table is intact and unchanged by the failed DDL.
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(5));
}

#[test]
fn unique_backfill_failure_then_recreate_succeeds() {
    // The strongest atomicity check: the index NAME is free again after the failed
    // create (cache resynced), so a NON-unique index of the same name builds cleanly
    // over the same (duplicate-keyed) data and then serves the lookup.
    let mut db = duplicate_keys();
    let _ = try_exec(&mut db, "CREATE UNIQUE INDEX i ON t(a)");
    exec(&mut db, "CREATE INDEX i ON t(a)"); // non-unique: duplicates are allowed
    assert_eq!(index_count(&mut db), 1, "the re-created non-unique index must be registered");
    assert_rows_unordered(
        &mut db,
        "SELECT b FROM t WHERE a = 2",
        &[vec![text("two")], vec![text("twob")]],
    );
    assert_rows_unordered(&mut db, "SELECT b FROM t WHERE a >= 5", &[vec![text("five")], vec![text("eight")]]);
}

// ---------------------------------------------------------------------------
// UNIQUE backfill: clean data (incl. many NULLs) succeeds.
// ---------------------------------------------------------------------------

#[test]
fn unique_backfill_distinct_keys_succeeds() {
    let mut db = distinct_keys();
    exec(&mut db, "CREATE UNIQUE INDEX i ON t(a)");
    assert_rows_unordered(&mut db, "SELECT b FROM t WHERE a = 2", &[vec![text("y")]]);
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(3));
    // A duplicate INSERT is now rejected by the maintained UNIQUE index.
    assert_exec_error(&mut db, "INSERT INTO t VALUES (2,'dup')");
}

#[test]
fn unique_backfill_allows_multiple_existing_nulls() {
    // §1.1: NULLs are all distinct, so any number of pre-existing NULL keys backfill
    // into a UNIQUE index without conflict; the non-NULL keys stay unique.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
    exec(&mut db, "INSERT INTO t VALUES (NULL,'a'),(NULL,'b'),(1,'x'),(2,'y')");
    exec(&mut db, "CREATE UNIQUE INDEX i ON t(a)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(4));
    assert_rows_unordered(
        &mut db,
        "SELECT b FROM t WHERE a IS NULL",
        &[vec![text("a")], vec![text("b")]],
    );
    assert_rows_unordered(&mut db, "SELECT b FROM t WHERE a = 1", &[vec![text("x")]]);
}

// ---------------------------------------------------------------------------
// Non-unique backfill reachability (equality AND range) + maintained path after.
// ---------------------------------------------------------------------------

#[test]
fn backfill_equality_and_range_reachable() {
    // Non-unique index over already-present rows: both a duplicate-key equality and a
    // range must find every pre-existing matching row (NULL never satisfies a range).
    let mut db = duplicate_keys();
    exec(&mut db, "CREATE INDEX i ON t(a)");
    assert_rows_unordered(
        &mut db,
        "SELECT b FROM t WHERE a = 2",
        &[vec![text("two")], vec![text("twob")]],
    );
    assert_rows_unordered(
        &mut db,
        "SELECT b FROM t WHERE a >= 5",
        &[vec![text("five")], vec![text("eight")]],
    );
}

#[test]
fn backfill_then_insert_is_maintained() {
    // After a backfill, the index is a live maintained structure: a later INSERT is
    // indexed too, and the backfilled rows remain reachable.
    let mut db = distinct_keys();
    exec(&mut db, "CREATE INDEX i ON t(a)");
    exec(&mut db, "INSERT INTO t VALUES (4,'w')");
    assert_rows_unordered(&mut db, "SELECT b FROM t WHERE a = 4", &[vec![text("w")]]); // new row
    assert_rows_unordered(&mut db, "SELECT b FROM t WHERE a = 1", &[vec![text("x")]]); // backfilled row
    assert_rows_unordered(
        &mut db,
        "SELECT b FROM t WHERE a >= 2",
        &[vec![text("y")], vec![text("z")], vec![text("w")]],
    );
}

#[test]
fn backfill_multicolumn_index() {
    // A composite index over existing rows serves both a full-key and a leftmost-prefix
    // lookup, with a duplicate composite key ((1,10) twice) returning both rows.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b INTEGER, c TEXT)");
    exec(&mut db, "INSERT INTO t VALUES (1,10,'p'),(1,20,'q'),(2,10,'r'),(1,10,'t')");
    exec(&mut db, "CREATE INDEX i ON t(a,b)");
    assert_rows_unordered(
        &mut db,
        "SELECT c FROM t WHERE a = 1 AND b = 10",
        &[vec![text("p")], vec![text("t")]],
    );
    assert_rows_unordered(
        &mut db,
        "SELECT c FROM t WHERE a = 1",
        &[vec![text("p")], vec![text("q")], vec![text("t")]],
    );
}

#[test]
fn backfill_text_key_binary_order() {
    // A TEXT index over existing rows: equality and BINARY-order range both reach the
    // backfilled rows ('a'/'b' sort before 'm'; 'mango'/'pear' at or after it).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(s TEXT, n INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES ('apple',1),('mango',2),('pear',3),('banana',4)");
    exec(&mut db, "CREATE INDEX i ON t(s)");
    assert_rows_unordered(&mut db, "SELECT n FROM t WHERE s = 'mango'", &[vec![int(2)]]);
    assert_rows_unordered(
        &mut db,
        "SELECT n FROM t WHERE s >= 'm'",
        &[vec![int(2)], vec![int(3)]],
    );
}

#[test]
fn backfill_if_not_exists_reindex_is_noop() {
    // Building the index once backfills it; `IF NOT EXISTS` over the now-existing index
    // is a clean no-op (not an error, and results stay correct — never doubled).
    let mut db = duplicate_keys();
    exec(&mut db, "CREATE INDEX i ON t(a)");
    exec(&mut db, "CREATE INDEX IF NOT EXISTS i ON t(a)"); // no-op over the built index
    assert_eq!(index_count(&mut db), 1, "IF NOT EXISTS must not add a second index row");
    assert_rows_unordered(
        &mut db,
        "SELECT b FROM t WHERE a = 2",
        &[vec![text("two")], vec![text("twob")]],
    );
}

#[test]
fn backfill_on_empty_table_is_a_noop_then_maintained() {
    // Building an index on an EMPTY table backfills nothing (the scan finds no rows);
    // a later INSERT is then indexed by the maintained path.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
    exec(&mut db, "CREATE INDEX i ON t(a)"); // empty-table backfill: no rows to copy
    exec(&mut db, "INSERT INTO t VALUES (7,'g')");
    assert_rows_unordered(&mut db, "SELECT b FROM t WHERE a = 7", &[vec![text("g")]]);
    assert_rows_unordered(&mut db, "SELECT b FROM t WHERE a = 3", &[]);
}

#[test]
fn backfill_streams_across_multiple_leaf_pages() {
    // The re-seek-per-row streaming loop (`seek_ge(last_rowid + 1)`) is the novel code,
    // and a page-boundary bug (dropping the first/last row of a leaf, or stopping after
    // the first page) only surfaces once the table spans MORE THAN ONE b-tree leaf. The
    // other fixtures here are a handful of rows (one leaf); this one inserts 1000 rows
    // BEFORE the index — several leaves at the default 4096-byte page — then pins that
    // the backfill indexed EVERY row across page boundaries, not just the first leaf's.
    const N: i64 = 1000;
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b INTEGER)");
    // Single-row inserts inside one transaction: fast, and independent of how large a
    // multi-row VALUES list the parser accepts. rowid auto-increments 1..=N alongside a.
    exec(&mut db, "BEGIN");
    for a in 1..=N {
        exec(&mut db, &format!("INSERT INTO t VALUES ({a},{})", a * 10));
    }
    exec(&mut db, "COMMIT");
    exec(&mut db, "CREATE INDEX i ON t(a)"); // backfill over a multi-leaf table

    // A range over the whole key space is an index range scan; it counts every
    // backfilled entry, so a row dropped in ANY leaf makes this less than N.
    assert_scalar(&mut db, "SELECT count(*) FROM t WHERE a >= 1", int(N));
    // Boundary equality lookups route through the index; first / middle / last keys land
    // in different leaves, so a lost page or an off-by-one at a page edge is caught.
    assert_rows_unordered(&mut db, "SELECT b FROM t WHERE a = 1", &[vec![int(10)]]);
    assert_rows_unordered(&mut db, "SELECT b FROM t WHERE a = 500", &[vec![int(5000)]]);
    assert_rows_unordered(&mut db, "SELECT b FROM t WHERE a = 1000", &[vec![int(10000)]]);
    // The exact extremes of the index scan: the lowest two and highest two keys.
    assert_rows_unordered(&mut db, "SELECT a FROM t WHERE a <= 2", &[vec![int(1)], vec![int(2)]]);
    assert_rows_unordered(
        &mut db,
        "SELECT a FROM t WHERE a >= 999",
        &[vec![int(999)], vec![int(1000)]],
    );
    // A closed mid-range returns exactly its 501 keys (250..=750 inclusive).
    assert_scalar(&mut db, "SELECT count(*) FROM t WHERE a >= 250 AND a <= 750", int(501));
}

#[test]
fn backfill_nocase_index_is_accepted_and_correct() {
    // A `COLLATE NOCASE` index built over existing rows runs `build_index_plan` on the
    // NOCASE `IndexDef` through the SAME shared key path as the maintained side, so the
    // backfilled index b-tree is byte-identical to the maintained one for the same rows.
    // Building it must be accepted and must not change results (datatype3 §7: NOCASE
    // folds ASCII case, so 'apple' COLLATE NOCASE matches the stored 'Apple').
    // Correctness holds whether or not the planner routes the lookup through the index.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(s TEXT, n INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES ('Apple',1),('banana',2),('CHERRY',3)");
    let r = try_exec(&mut db, "CREATE INDEX i ON t(s COLLATE NOCASE)");
    assert!(r.is_ok(), "backfilling a COLLATE NOCASE index should be accepted; got {r:?}");
    assert_rows_unordered(
        &mut db,
        "SELECT n FROM t WHERE s = 'apple' COLLATE NOCASE",
        &[vec![int(1)]],
    );
    // A plain BINARY equality still finds the exact-case row, index present or not.
    assert_rows_unordered(&mut db, "SELECT n FROM t WHERE s = 'banana'", &[vec![int(2)]]);
}
