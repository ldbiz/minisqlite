//! Differential proof of the decode-free `count(*)` SCAN fast path — the b-tree
//! entry-count override on `SeqScanCursor` / `WrSeqScanCursor` / `RowidScanCursor`
//! (`minisqlite-exec` `ops/scan.rs`), backed by `table_entry_count` / `index_entry_count`
//! (`minisqlite-btree`). A bare `SELECT count(*) FROM t` counts b-tree ENTRIES without
//! decoding any row payload, matching real sqlite's count optimization.
//!
//! The optimization must be a PURE SPEED change: the fast count has to equal EXACTLY the
//! number of rows the row-materializing scan yields. So every full-scan test here pins the
//! fast `count(*)` against BOTH an independent materialized drain (`SELECT <col> FROM t`,
//! which pulls and DECODES every row through the scan's `next_row`) AND the known
//! inserted/surviving count, across:
//!   * empty / 1 / a few / enough-rows-for-multiple-leaves-and-an-interior-level;
//!   * a ROWID table, an INTEGER PRIMARY KEY table, and a WITHOUT ROWID table
//!     (WrSeqScanCursor walks the PK index b-tree — a classic B-tree whose interior
//!     dividers are real entries, so `index_entry_count` must count them too);
//!   * after DELETEs, which leave page gaps and free pages (free pages are not in the
//!     tree, so the decode-free walk never visits them).
//!
//! The `count(*) ... WHERE …` tests are the GATE GUARD: a filter (SeqScan + Filter) or a
//! bounded/point rowid scan selects a SUBSET, so `count_rows` there MUST drain (running the
//! predicate / range), never return the whole-table entry count. `FilterCursor` and a
//! bounded `RowidScanCursor` are therefore deliberately NOT on the fast path; a broken gate
//! that returned the entry count would report the full table size and fail these.
//!
//! Sharpness (mutation check): perturbing the b-tree entry count (e.g. `+1`) turns every
//! full-scan assertion here RED while the filtered ones (which drain) stay GREEN — that
//! split is the proof the fast path is actually taken on full scans and NOT on filters.

use minisqlite::{Connection, Value};

/// The single integer a scalar `count(*)` query returns.
fn count_of(db: &mut Connection, sql: &str) -> i64 {
    let rows = db.query(sql).expect("count query runs").rows;
    assert_eq!(rows.len(), 1, "a scalar count(*) yields exactly one row: {sql}");
    match rows[0][0] {
        Value::Integer(n) => n,
        ref other => panic!("count(*) must be INTEGER, got {other:?} for {sql}"),
    }
}

/// How many rows a query actually yields — the row-materializing drain (each row is pulled
/// and DECODED through the scan's `next_row`). The independent oracle for the fast count.
fn rowcount(db: &mut Connection, sql: &str) -> i64 {
    db.query(sql).expect("row query runs").rows.len() as i64
}

/// The three table shapes whose scans have a `count(*)` fast path.
#[derive(Clone, Copy)]
enum Kind {
    /// Implicit rowid, `id` a PLAIN column — a `SeqScanCursor` over the table b-tree.
    Rowid,
    /// `id INTEGER PRIMARY KEY` aliases the rowid — also a `SeqScanCursor` (and `WHERE id`
    /// can compile to a `RowidScanCursor`).
    IntPk,
    /// `WITHOUT ROWID` — a `WrSeqScanCursor` over the PRIMARY KEY index b-tree.
    WithoutRowid,
}

/// A fresh in-memory database with an empty table `t(id, pad)` of the given shape.
fn create(kind: Kind) -> Connection {
    let mut db = Connection::open_in_memory().unwrap();
    let ddl = match kind {
        Kind::Rowid => "CREATE TABLE t(id INTEGER, pad TEXT)",
        Kind::IntPk => "CREATE TABLE t(id INTEGER PRIMARY KEY, pad TEXT)",
        Kind::WithoutRowid => "CREATE TABLE t(id INTEGER PRIMARY KEY, pad TEXT) WITHOUT ROWID",
    };
    db.execute(ddl).unwrap();
    db
}

/// Insert rows `id = 1..=n` (each with the same `pad` text) into `t(id, pad)` via chunked
/// multi-row INSERTs, so no single SQL string grows unbounded with `n`.
fn fill(db: &mut Connection, n: i64, pad: &str) {
    const CHUNK: i64 = 400;
    let mut lo = 1;
    while lo <= n {
        let hi = (lo + CHUNK - 1).min(n);
        let mut sql = String::from("INSERT INTO t(id,pad) VALUES ");
        for id in lo..=hi {
            if id > lo {
                sql.push(',');
            }
            sql.push_str(&format!("({id},'{pad}')"));
        }
        db.execute(&sql).expect("bulk insert runs");
        lo = hi + 1;
    }
}

/// The fast `count(*) FROM t` must equal BOTH the materialized drain of `SELECT id FROM t`
/// AND the `expected` known count — the core differential invariant of the fast path.
fn assert_full_count(db: &mut Connection, expected: i64) {
    let fast = count_of(db, "SELECT count(*) FROM t");
    let materialized = rowcount(db, "SELECT id FROM t");
    assert_eq!(fast, expected, "fast count(*) must equal the known row count");
    assert_eq!(fast, materialized, "fast count(*) must equal the materialized drain");
}

/// An ~80-byte column value: wide enough that `2000` rows split well past a single leaf and
/// build an interior level at the default 4096-byte page size (~43 rows/leaf → ~47 leaves
/// under one interior root). The `minisqlite-btree` unit tests pin the interior-traversal
/// guarantee directly; here it makes the SQL path exercise a multi-level tree.
fn wide_pad() -> String {
    "x".repeat(80)
}

// --- Full-scan differential: fast count == materialized drain == known n. -------------
// The size sweep spans the boundaries: 0 (empty root leaf), 1 (single cell), a few, and
// 2000 (multiple leaves + an interior level). Run for each table shape.

#[test]
fn rowid_full_scan_count_matches_across_sizes() {
    let pad = wide_pad();
    for n in [0i64, 1, 5, 37, 2000] {
        let mut db = create(Kind::Rowid);
        fill(&mut db, n, &pad);
        assert_full_count(&mut db, n);
    }
}

#[test]
fn intpk_full_scan_count_matches_across_sizes() {
    let pad = wide_pad();
    for n in [0i64, 1, 5, 37, 2000] {
        let mut db = create(Kind::IntPk);
        fill(&mut db, n, &pad);
        assert_full_count(&mut db, n);
    }
}

#[test]
fn without_rowid_full_scan_count_matches_across_sizes() {
    let pad = wide_pad();
    for n in [0i64, 1, 5, 37, 2000] {
        let mut db = create(Kind::WithoutRowid);
        fill(&mut db, n, &pad);
        assert_full_count(&mut db, n);
    }
}

// --- After DELETEs: gaps and freed pages must not perturb the entry count. -------------
// Deleting a contiguous block frees whole leaf pages; those pages leave the tree and the
// decode-free walk must never revisit them. The survivor count is checked both exactly and
// against the materialized drain.

/// Delete a 100-row block (`100..=199`) and a 50-row tail (`>450`) from a 500-row table,
/// then require the fast count to equal both the exact 350 survivors and the drain.
fn assert_deletes_leave_350(mut db: Connection) {
    fill(&mut db, 500, &wide_pad());
    db.execute("DELETE FROM t WHERE id BETWEEN 100 AND 199").unwrap(); // 100 rows
    db.execute("DELETE FROM t WHERE id > 450").unwrap(); // ids 451..=500 = 50 rows
    let survivors = 500 - 100 - 50;
    let fast = count_of(&mut db, "SELECT count(*) FROM t");
    let materialized = rowcount(&mut db, "SELECT id FROM t");
    assert_eq!(fast, survivors, "fast count must equal the exact survivor count");
    assert_eq!(fast, materialized, "fast count must equal the materialized drain");
}

#[test]
fn rowid_count_after_deletes_matches_survivors() {
    assert_deletes_leave_350(create(Kind::Rowid));
}

#[test]
fn intpk_count_after_deletes_matches_survivors() {
    assert_deletes_leave_350(create(Kind::IntPk));
}

#[test]
fn without_rowid_count_after_deletes_matches_survivors() {
    assert_deletes_leave_350(create(Kind::WithoutRowid));
}

// --- Gate guard: a filtered count MUST still run the filter / range (drain), never take
// the whole-table entry-count shortcut. ------------------------------------------------

#[test]
fn count_star_where_seqscan_filter_still_runs() {
    // ROWID table: `id` is a PLAIN column (the rowid is implicit), so `WHERE id > 250` is a
    // SeqScan + Filter. `count_rows` is NOT overridden on the filter, so the predicate runs
    // and the count is the true subset — not the whole-table entry count.
    let mut db = create(Kind::Rowid);
    fill(&mut db, 500, &"x".repeat(40));
    let fast = count_of(&mut db, "SELECT count(*) FROM t WHERE id > 250");
    assert_eq!(fast, 250, "ids 251..=500");
    assert_eq!(fast, rowcount(&mut db, "SELECT id FROM t WHERE id > 250"));
}

#[test]
fn count_star_where_rowid_range_still_drains() {
    // INTEGER PRIMARY KEY: `id` IS the rowid, so `WHERE id > 250` / BETWEEN can compile to a
    // BOUNDED RowidScan. The RowidScanCursor fast path is GATED OFF for a bounded range
    // (only an UNBOUNDED full scan may use the entry count), so it drains and counts the
    // true subset. A broken gate that returned the whole entry count would report 500 here.
    let mut db = create(Kind::IntPk);
    fill(&mut db, 500, &"x".repeat(40));

    let gt = count_of(&mut db, "SELECT count(*) FROM t WHERE id > 250");
    assert_eq!(gt, 250, "ids 251..=500");
    assert_eq!(gt, rowcount(&mut db, "SELECT id FROM t WHERE id > 250"));

    let between = count_of(&mut db, "SELECT count(*) FROM t WHERE id BETWEEN 100 AND 199");
    assert_eq!(between, 100, "ids 100..=199");
    assert_eq!(between, rowcount(&mut db, "SELECT id FROM t WHERE id BETWEEN 100 AND 199"));
}

#[test]
fn count_star_where_rowid_eq_still_drains() {
    // A point lookup (`RowidOp::Eq`) selects at most one row; the gate keeps it off the
    // entry-count path so it seeks and counts exactly the matching row(s).
    let mut db = create(Kind::IntPk);
    fill(&mut db, 500, &"x".repeat(40));
    assert_eq!(count_of(&mut db, "SELECT count(*) FROM t WHERE id = 300"), 1);
    assert_eq!(count_of(&mut db, "SELECT count(*) FROM t WHERE id = 99999"), 0);
}

#[test]
fn count_star_where_without_rowid_filter_still_runs() {
    let mut db = create(Kind::WithoutRowid);
    fill(&mut db, 500, &"x".repeat(40));
    let fast = count_of(&mut db, "SELECT count(*) FROM t WHERE id > 250");
    assert_eq!(fast, 250, "ids 251..=500");
    assert_eq!(fast, rowcount(&mut db, "SELECT id FROM t WHERE id > 250"));
}

// --- A count over the empty implicit group is still exactly zero on every shape. -------

#[test]
fn empty_table_count_is_zero_all_shapes() {
    for kind in [Kind::Rowid, Kind::IntPk, Kind::WithoutRowid] {
        let mut db = create(kind);
        assert_eq!(count_of(&mut db, "SELECT count(*) FROM t"), 0);
    }
}

// --- Overflow rows: a spilled cell is still ONE entry; the count must not follow chains. -
// A value far larger than a page forces every row's cell to spill onto overflow pages. The
// decode-free walk reads only the on-page cell count and never reassembles a chain, so
// count(*) must equal the row count and the materialized drain (which DOES reassemble each
// chain). This pins "a cell with overflow is still ONE entry" at the SQL level, for all
// three shapes (the rowid/IPK table b-tree and the WITHOUT ROWID index b-tree key record).

#[test]
fn count_over_overflow_spilled_rows_counts_each_once_all_shapes() {
    let big = "z".repeat(9000); // >> one 4096-byte page → each row's cell overflows
    let n = 20i64;
    for kind in [Kind::Rowid, Kind::IntPk, Kind::WithoutRowid] {
        let mut db = create(kind);
        for id in 1..=n {
            db.execute(&format!("INSERT INTO t(id, pad) VALUES ({id}, '{big}')"))
                .expect("wide (overflow-forcing) insert runs");
        }
        let fast = count_of(&mut db, "SELECT count(*) FROM t");
        assert_eq!(fast, n, "count over overflow-spilled rows equals the row count");
        assert_eq!(fast, rowcount(&mut db, "SELECT id FROM t"), "equals the materialized drain");
    }
}

// --- On disk: the fast path is pager-agnostic — it holds against the FILE-backed pager and
// across a reopen, not just the in-memory pager. `entry_count` reads only page headers +
// child pointers through the same `read_page`/`page_count` the on-disk pager backs, so a
// committed table counted after reopen equals its row count and drain. This is the durable
// analogue of the bench's `durability_roundtrip` bare count(*).

#[test]
fn on_disk_full_scan_count_matches_after_reopen() {
    let path = std::env::temp_dir()
        .join(format!("minisqlite-count-scan-fastpath-ondisk-{}.db", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let n = 2000i64; // wide rows at this size build multiple leaves + an interior level
    {
        let mut db = Connection::open(&path).expect("open file-backed db");
        db.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, pad TEXT)").expect("create table");
        fill(&mut db, n, &wide_pad());
    }
    // Reopen a fresh connection so the count reads committed pages back from the file.
    let mut db = Connection::open(&path).expect("reopen file-backed db");
    let fast = count_of(&mut db, "SELECT count(*) FROM t");
    let materialized = rowcount(&mut db, "SELECT id FROM t");
    let _ = std::fs::remove_file(&path); // clean up before asserting so a failure still tidies
    assert_eq!(fast, n, "on-disk fast count equals the row count after reopen");
    assert_eq!(fast, materialized, "on-disk fast count equals the materialized drain");
}
