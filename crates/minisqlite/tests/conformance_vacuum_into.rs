//! Conformance battery: `VACUUM ... INTO <filename>` at the pinned facade.
//!
//! `VACUUM INTO 'target'` writes a fresh, defragmented copy of the current database
//! into a NEW file, leaving the source untouched (spec: `spec/sqlite-doc/lang_vacuum.html`
//! §3). The copy must be a valid SQLite database whose LOGICAL content is identical to
//! the source: same schema (tables, indexes, views, triggers), same rows with identical
//! rowids and byte-exact values of every storage class, and the same `sqlite_sequence` /
//! AUTOINCREMENT state. It need not be byte-identical to sqlite's own VACUUM output (the
//! physical page layout — in particular each object's `rootpage` — may differ), so the
//! schema comparisons below deliberately exclude `rootpage`.
//!
//! Every case reopens the copied file with a fresh `Connection` (the real read-back path)
//! and asserts against the SOURCE and against transcribed literals — never against
//! whatever the engine happens to emit. A failing case is a real fidelity discrepancy,
//! not something to weaken.
//!
//! Two behaviors are DEFERRED and asserted as loud, honest gaps rather than silently
//! wrong output: a WITHOUT ROWID table, and (not exercised here — no UTF-16 writer
//! exists) a UTF-16 source. The gap test pins the precise failure so the deferral
//! stays visible.

mod conformance;
use conformance::*;

use minisqlite::{Connection, Error, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Temp-file harness: a unique on-disk path per test, cleaned up even on panic.
// Mirrors the RAII guard in `conformance_ondisk_roundtrip.rs` — no `.db` is ever
// committed, and stale files from a crashed run are removed on construction.
// ---------------------------------------------------------------------------

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
        path.push(format!("minisqlite_vacuum_{pid}_{n}_{nanos}.db"));
        let db = TempDb { path };
        db.remove_all();
        db
    }

    /// Open (or reopen) a file-backed connection to this path.
    fn open(&self) -> Connection {
        let path: &Path = &self.path;
        Connection::open(path)
            .unwrap_or_else(|e| panic!("Connection::open failed for {path:?}: {e:?}"))
    }

    /// This path as a single-quoted SQL string literal (doubling any embedded quote),
    /// so it can be spliced into a `VACUUM INTO <literal>` statement.
    fn sql_literal(&self) -> String {
        let s = self.path.to_str().expect("temp path is valid UTF-8");
        format!("'{}'", s.replace('\'', "''"))
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

// ---------------------------------------------------------------------------
// Cross-connection helpers: compare the SAME query run on two connections.
// ---------------------------------------------------------------------------

/// All rows of `sql` on `db` (order as returned; callers pin it with `ORDER BY`).
fn rows_of(db: &mut Connection, sql: &str) -> Vec<Vec<Value>> {
    query(db, sql).rows
}

/// Structural multiset-free (ordered) equality of two row sets via `value_eq`
/// (`Value` has no `PartialEq`), matching shape and every cell.
fn rows_value_eq(a: &[Vec<Value>], b: &[Vec<Value>]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b.iter()).all(|(ra, rb)| {
            ra.len() == rb.len() && ra.iter().zip(rb.iter()).all(|(x, y)| value_eq(x, y))
        })
}

/// Assert `sql` returns identical rows on `src` and `tgt` (both pinned by `ORDER BY`).
fn assert_same(src: &mut Connection, tgt: &mut Connection, sql: &str) {
    let a = rows_of(src, sql);
    let b = rows_of(tgt, sql);
    assert!(
        rows_value_eq(&a, &b),
        "row mismatch between source and vacuumed copy\n  sql: {sql}\n  source: {a:?}\n  copy:   {b:?}"
    );
}

/// The full user schema of a database, EXCLUDING `rootpage` (which legitimately
/// differs after a defragmenting copy). Ordered deterministically so source and copy
/// line up row-for-row.
const SCHEMA_QUERY: &str =
    "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name";

/// Build the sample database used by the fidelity tests: an INTEGER PRIMARY KEY table
/// mixing every storage class (TEXT/REAL/BLOB/NULL and a NONE-affinity column), a
/// UNIQUE column (⇒ an auto-index to backfill), an explicit secondary index, a second
/// table, a view, and a trigger. Rows include NULLs, an empty blob, a negative-zero
/// real, and a byte blob spanning 0x00..0xFF so the copy's fidelity is exercised across
/// the whole record codec.
fn build_sample(db: &mut Connection) {
    exec(
        db,
        "CREATE TABLE items(\
           id INTEGER PRIMARY KEY, \
           label TEXT, \
           code TEXT UNIQUE, \
           score REAL, \
           data BLOB, \
           extra)",
    );
    exec(db, "CREATE INDEX idx_label ON items(label)");
    exec(db, "CREATE TABLE audit(item_id)");
    exec(db, "CREATE VIEW v_items AS SELECT id, label FROM items");
    exec(
        db,
        "CREATE TRIGGER trg_audit AFTER INSERT ON items \
         BEGIN INSERT INTO audit VALUES (new.id); END",
    );
    // Explicit ids so the rowids are known and stable regardless of insert mechanics.
    exec(
        db,
        "INSERT INTO items(id, label, code, score, data, extra) VALUES \
           (1, 'apple',  'A1', 1.5,  x'00ff10',       'x'), \
           (2, 'banana', 'B2', 2.25, x'01027f80feff', NULL), \
           (3, NULL,     'C3', NULL, NULL,            42), \
           (4, 'apple',  NULL, -0.0, x'',             3.14)",
    );
}

// ===========================================================================
// 1) Content round-trip: schema + every row copied byte-exact.
// ===========================================================================

#[test]
fn round_trip_content() {
    // A file-backed SOURCE (closest to a real on-disk `.db` reference) copied to a new
    // file, then reopened and diffed against the source.
    let src_db = TempDb::new();
    let dst = TempDb::new();
    let mut src = src_db.open();
    build_sample(&mut src);

    exec(&mut src, &format!("VACUUM INTO {}", dst.sql_literal()));

    let mut copy = dst.open();

    // Schema (excluding rootpage) is identical: same tables, the UNIQUE auto-index, the
    // explicit index, the view, and the trigger, with byte-identical `sql` text.
    assert_same(&mut src, &mut copy, SCHEMA_QUERY);

    // Every row, with its exact rowid, is identical — this is the fidelity bar.
    assert_same(&mut src, &mut copy, "SELECT rowid, * FROM items ORDER BY rowid");
    assert_same(&mut src, &mut copy, "SELECT rowid, * FROM audit ORDER BY rowid");

    // And the copied values are byte-exact against transcribed literals (not merely
    // equal to a possibly-buggy source): blob bytes incl. 0x00, an empty blob, a REAL,
    // negative-zero REAL, NULLs, and a NONE-affinity integer/real all survive.
    assert_rows(
        &mut copy,
        "SELECT id, label, code, score, data, extra FROM items ORDER BY id",
        &[
            vec![int(1), text("apple"), text("A1"), real(1.5), blob(&[0x00, 0xff, 0x10]), text("x")],
            vec![
                int(2),
                text("banana"),
                text("B2"),
                real(2.25),
                blob(&[0x01, 0x02, 0x7f, 0x80, 0xfe, 0xff]),
                null(),
            ],
            vec![int(3), null(), text("C3"), null(), null(), int(42)],
            vec![int(4), text("apple"), null(), real(-0.0), blob(&[]), real(3.14)],
        ],
    );
    // Storage classes are preserved exactly (typeof distinguishes integer/real/text/blob/null).
    assert_rows(
        &mut copy,
        "SELECT typeof(score), typeof(data), typeof(extra) FROM items ORDER BY id",
        &[
            vec![text("real"), text("blob"), text("text")],
            vec![text("real"), text("blob"), text("null")],
            vec![text("null"), text("null"), text("integer")],
            vec![text("real"), text("blob"), text("real")],
        ],
    );
    // The view resolves against the copied table.
    assert_rows(
        &mut copy,
        "SELECT id, label FROM v_items ORDER BY id",
        &[
            vec![int(1), text("apple")],
            vec![int(2), text("banana")],
            vec![int(3), null()],
            vec![int(4), text("apple")],
        ],
    );
}

// ===========================================================================
// 2) Indexes are present AND populated in the copy (both backfill paths).
// ===========================================================================

#[test]
fn index_usable_after_vacuum() {
    let dst = TempDb::new();
    let mut src = mem();
    build_sample(&mut src);
    exec(&mut src, &format!("VACUUM INTO {}", dst.sql_literal()));

    let mut copy = dst.open();

    // The EXPLICIT secondary index (idx_label) is present and a lookup on it returns the
    // right rows on the copy: both 'apple' rows (1 and 4). (This confirms the index exists
    // and queries resolve correctly; it does NOT by itself prove the index b-tree is
    // POPULATED — a full-table-scan plan would return the same rows over an empty index.
    // Population is proven for the UNIQUE auto-index below via the constraint check.)
    assert_rows(&mut copy, "SELECT id FROM items WHERE label = 'apple' ORDER BY id", &[
        vec![int(1)],
        vec![int(4)],
    ]);
    // The UNIQUE AUTO-index (on `code`) resolves a lookup on the copy.
    assert_scalar(&mut copy, "SELECT id FROM items WHERE code = 'B2'", int(2));

    // The auto-index is not just present but POPULATED: re-inserting an existing code
    // must raise a UNIQUE violation. If the backfill had left the auto-index empty, the
    // pre-existing 'A1' would be invisible and this INSERT would wrongly succeed.
    let e = assert_exec_error(&mut copy, "INSERT INTO items(id, code) VALUES (99, 'A1')");
    assert!(
        matches!(e, Error::Constraint(_)),
        "duplicate code in the copied UNIQUE index must be a constraint error; got {e:?}"
    );
    // A genuinely new code still inserts (the index accepts distinct keys).
    exec(&mut copy, "INSERT INTO items(id, code) VALUES (5, 'E5')");
    assert_scalar(&mut copy, "SELECT id FROM items WHERE code = 'E5'", int(5));

    // The copied AFTER INSERT trigger (trg_audit) FIRES on the copy: the insert above
    // appended item 5's id to `audit`. Ids 1..4 came from the source's inserts and were
    // copied over (the low-level row copy did NOT re-fire the trigger, so audit is not
    // double-populated); firing it here is what adds 5. This exercises a copied trigger's
    // effect, not just its presence in the schema.
    assert_rows(&mut copy, "SELECT item_id FROM audit ORDER BY item_id", &[
        vec![int(1)],
        vec![int(2)],
        vec![int(3)],
        vec![int(4)],
        vec![int(5)],
    ]);
}

// ===========================================================================
// 3) AUTOINCREMENT / sqlite_sequence state carries over.
// ===========================================================================

#[test]
fn rowid_and_autoincrement_preserved() {
    let dst = TempDb::new();
    let mut src = mem();
    exec(&mut src, "CREATE TABLE t(a INTEGER PRIMARY KEY AUTOINCREMENT, v)");
    exec(&mut src, "INSERT INTO t(v) VALUES ('a'), ('b'), ('c')"); // rowids 1,2,3; seq=3
    // Delete the LARGEST rowid so max(rowid)=2 but the remembered high-water (seq) is 3.
    // This is what distinguishes "sequence preserved" (next = 4) from "recomputed from
    // max(rowid)" (next = 3): only a carried-over sqlite_sequence yields 4.
    exec(&mut src, "DELETE FROM t WHERE a = 3");

    exec(&mut src, &format!("VACUUM INTO {}", dst.sql_literal()));

    let mut copy = dst.open();
    // The surviving rows copied with their exact rowids.
    assert_rows(&mut copy, "SELECT a, v FROM t ORDER BY a", &[
        vec![int(1), text("a")],
        vec![int(2), text("b")],
    ]);
    // sqlite_sequence carried the high-water mark 3.
    assert_scalar(&mut copy, "SELECT seq FROM sqlite_sequence WHERE name = 't'", int(3));
    // Therefore the next AUTOINCREMENT rowid is 4, not the reused 3.
    exec(&mut copy, "INSERT INTO t(v) VALUES ('d')");
    assert_scalar(&mut copy, "SELECT a FROM t WHERE v = 'd'", int(4));
}

#[test]
fn plain_rowid_table_gap_is_preserved_not_renumbered() {
    // The central fidelity contract is that rowids copy EXACTLY, never reassigned. Every
    // other table here uses a contiguous 1.. sequence or an INTEGER PRIMARY KEY alias, so
    // a copy that sequentially RENUMBERED rowids (what real sqlite's VACUUM does for a
    // plain table, and what THIS copy must NOT do) would still pass them. A plain rowid
    // table (no INTEGER PK) with a GAP is the discriminator: renumbering would collapse
    // rowids [1, 3] to [1, 2].
    let dst = TempDb::new();
    let mut src = mem();
    exec(&mut src, "CREATE TABLE r(x)");
    exec(&mut src, "INSERT INTO r VALUES ('a'), ('b'), ('c')"); // implicit rowids 1, 2, 3
    exec(&mut src, "DELETE FROM r WHERE x = 'b'"); // leaves rowids 1 and 3 — a gap at 2

    exec(&mut src, &format!("VACUUM INTO {}", dst.sql_literal()));

    let mut copy = dst.open();
    // The exact rowids (1 and 3) survive — NOT renumbered to (1, 2).
    assert_rows(&mut copy, "SELECT rowid, x FROM r ORDER BY rowid", &[
        vec![int(1), text("a")],
        vec![int(3), text("c")],
    ]);
}

// ===========================================================================
// 4) An existing NON-EMPTY target errors and is NOT clobbered.
// ===========================================================================

#[test]
fn error_target_exists() {
    let dst = TempDb::new();
    // Pre-create the target as a non-empty file (its contents are not a database).
    let sentinel: &[u8] = b"do not overwrite me";
    std::fs::write(&dst.path, sentinel).expect("write sentinel target file");

    let mut src = mem();
    exec(&mut src, "CREATE TABLE t(a)");
    exec(&mut src, "INSERT INTO t VALUES (1)");

    let e = assert_exec_error(&mut src, &format!("VACUUM INTO {}", dst.sql_literal()));
    assert!(
        matches!(&e, Error::Sql(m) if m.contains("already exists")),
        "VACUUM INTO an existing non-empty file must error; got {e:?}"
    );

    // The pre-existing file is untouched (byte-for-byte the sentinel), not clobbered.
    let after = std::fs::read(&dst.path).expect("target file still readable");
    assert_eq!(after, sentinel, "existing target file must not be overwritten on error");
}

// ===========================================================================
// 5) VACUUM cannot run inside a transaction.
// ===========================================================================

#[test]
fn error_in_transaction() {
    let dst = TempDb::new();
    let mut src = mem();
    exec(&mut src, "CREATE TABLE t(a)");
    exec(&mut src, "INSERT INTO t VALUES (1)");
    exec(&mut src, "BEGIN");

    let e = assert_exec_error(&mut src, &format!("VACUUM INTO {}", dst.sql_literal()));
    assert!(
        matches!(&e, Error::Sql(m) if m.to_lowercase().contains("transaction")),
        "VACUUM INTO inside a transaction must error; got {e:?}"
    );
    // No output file was created (the precondition fires before the target is opened).
    assert!(!dst.path.exists(), "a rejected VACUUM INTO must not create the target file");

    // The transaction is still usable and can be committed normally afterward.
    exec(&mut src, "INSERT INTO t VALUES (2)");
    exec(&mut src, "COMMIT");
    assert_rows_unordered(&mut src, "SELECT a FROM t", &[vec![int(1)], vec![int(2)]]);
}

// ===========================================================================
// 6) The source is unchanged after VACUUM INTO (it only reads the source).
// ===========================================================================

#[test]
fn source_unchanged() {
    let dst = TempDb::new();
    let mut src = mem();
    build_sample(&mut src);

    let schema_before = rows_of(&mut src, SCHEMA_QUERY);
    let rows_before = rows_of(&mut src, "SELECT rowid, * FROM items ORDER BY rowid");

    exec(&mut src, &format!("VACUUM INTO {}", dst.sql_literal()));

    let schema_after = rows_of(&mut src, SCHEMA_QUERY);
    let rows_after = rows_of(&mut src, "SELECT rowid, * FROM items ORDER BY rowid");
    assert!(rows_value_eq(&schema_before, &schema_after), "source schema changed by VACUUM INTO");
    assert!(rows_value_eq(&rows_before, &rows_after), "source rows changed by VACUUM INTO");

    // The source connection is still fully writable afterward.
    exec(&mut src, "INSERT INTO items(id, code) VALUES (10, 'Z9')");
    assert_scalar(&mut src, "SELECT code FROM items WHERE id = 10", text("Z9"));
}

// ===========================================================================
// 7) WITHOUT ROWID is a deferred, LOUD gap (tested so it stays honest).
// ===========================================================================

#[test]
fn without_rowid_loud_gap() {
    let dst = TempDb::new();
    let mut src = mem();
    // WITHOUT ROWID tables are supported for normal DML, so the CREATE/INSERT succeed;
    // only the VACUUM INTO copy of one is (deliberately) unimplemented.
    exec(&mut src, "CREATE TABLE t(a PRIMARY KEY, b) WITHOUT ROWID");
    exec(&mut src, "INSERT INTO t VALUES ('k', 'v'), ('a', 'w')");

    let e = assert_exec_error(&mut src, &format!("VACUUM INTO {}", dst.sql_literal()));
    assert!(
        matches!(&e, Error::Sql(m) if m.contains("WITHOUT ROWID")),
        "VACUUM INTO of a WITHOUT ROWID table must fail loudly naming the gap; got {e:?}"
    );
    // The gap is detected BEFORE the target is created, so no partial file is left.
    assert!(!dst.path.exists(), "a rejected VACUUM INTO must not create the target file");

    // The source WITHOUT ROWID table is untouched and still queryable.
    assert_rows(&mut src, "SELECT a, b FROM t ORDER BY a", &[
        vec![text("a"), text("w")],
        vec![text("k"), text("v")],
    ]);
}

// ===========================================================================
// 8) The copy adopts the SOURCE page size (spec item 2), content intact.
// ===========================================================================

#[test]
fn page_size_matched_in_copy() {
    let dst = TempDb::new();
    let mut src = mem();
    // `page_size` only takes effect while the database is still empty (before any table).
    exec(&mut src, "PRAGMA page_size = 8192");
    exec(&mut src, "CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT)");
    exec(&mut src, "INSERT INTO t VALUES (1, 'one'), (2, 'two')");
    assert_scalar(&mut src, "PRAGMA page_size", int(8192));

    exec(&mut src, &format!("VACUUM INTO {}", dst.sql_literal()));

    let mut copy = dst.open();
    // The copy is physically laid out in the SOURCE's page size (8192), not the 4096
    // default a fresh target would otherwise use.
    assert_scalar(&mut copy, "PRAGMA page_size", int(8192));
    assert_rows(&mut copy, "SELECT a, b FROM t ORDER BY a", &[
        vec![int(1), text("one")],
        vec![int(2), text("two")],
    ]);
}

// ===========================================================================
// 9) Large / overflowing values copy byte-exact (spec item 4).
// ===========================================================================

#[test]
fn overflow_values_round_trip() {
    use std::fmt::Write as _;
    let dst = TempDb::new();
    let mut src = mem();
    exec(&mut src, "CREATE TABLE big(id INTEGER PRIMARY KEY, blob_col BLOB, text_col TEXT)");

    // A blob and a text far past the inline threshold at the default page size, so each
    // spills onto a chain of overflow pages. Non-trivial, non-repeating-mod patterns so a
    // truncation, reordering, byte-swap, or dropped overflow page is caught — not masked
    // by a uniform fill.
    let big_blob: Vec<u8> = (0..10_000u32).map(|i| (i % 251) as u8).collect();
    let big_text: String = (0..10_000u32).map(|i| char::from(b'A' + (i % 26) as u8)).collect();
    let mut hex = String::from("x'");
    for b in &big_blob {
        let _ = write!(hex, "{b:02x}");
    }
    hex.push('\'');
    exec(&mut src, &format!("INSERT INTO big VALUES (1, {hex}, '{big_text}')"));

    exec(&mut src, &format!("VACUUM INTO {}", dst.sql_literal()));

    let mut copy = dst.open();
    // Every byte of the overflowing blob and every char of the overflowing text survive.
    assert_rows(&mut copy, "SELECT id, blob_col, text_col FROM big", &[vec![
        int(1),
        blob(&big_blob),
        text(&big_text),
    ]]);
    // Independent length cross-check that the overflow chains are complete, not truncated.
    assert_scalar(&mut copy, "SELECT length(blob_col) FROM big", int(10_000));
    assert_scalar(&mut copy, "SELECT length(text_col) FROM big", int(10_000));
}

// ===========================================================================
// 10) A Unicode object name with a multibyte char at the reserved-prefix boundary
//     copies cleanly (regression guard for the `name[..7]` string-slice PANIC).
// ===========================================================================

#[test]
fn unicode_table_name_copies_without_panic() {
    // The reserved-`sqlite_` check runs on EVERY source object name. A name that is
    // >= 7 bytes with a multibyte UTF-8 char straddling byte 7 — here "aaaa" + U+1F600
    // 😀 (8 bytes; the emoji occupies bytes 4..8, so byte 7 is mid-char) — used to drive
    // that check through `name[..7]` STRING slicing, which PANICS ("byte index 7 is not a
    // char boundary"). It is an ordinary user table and must round-trip cleanly.
    let dst = TempDb::new();
    let mut src = mem();
    exec(&mut src, "CREATE TABLE \"aaaa\u{1F600}\"(x)");
    exec(&mut src, "INSERT INTO \"aaaa\u{1F600}\" VALUES (1), (2), (3)");

    exec(&mut src, &format!("VACUUM INTO {}", dst.sql_literal()));

    let mut copy = dst.open();
    // The rows copied, and the object's name survived verbatim in the copied schema.
    assert_rows(&mut copy, "SELECT x FROM \"aaaa\u{1F600}\" ORDER BY x", &[
        vec![int(1)],
        vec![int(2)],
        vec![int(3)],
    ]);
    assert_scalar(
        &mut copy,
        "SELECT name FROM sqlite_master WHERE type = 'table'",
        text("aaaa\u{1F600}"),
    );
}

// ===========================================================================
// 11) `sqlite_stat1` (ANALYZE statistics) carries over — the copy_sqlite_stat1 path.
// ===========================================================================

#[test]
fn analyze_stats_carry_over() {
    // `sqlite_stat1` is real delivered content copied by its own path (`copy_sqlite_stat1`,
    // which ensures the target table then byte-copies the rows). Pin the carry-over so a
    // silent no-op or a mis-mapped root would fail here.
    let dst = TempDb::new();
    let mut src = mem();
    exec(&mut src, "CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT)");
    exec(&mut src, "CREATE INDEX idx_b ON t(b)");
    exec(&mut src, "INSERT INTO t VALUES (1,'x'), (2,'y'), (3,'y'), (4,'z')");
    exec(&mut src, "ANALYZE");

    // Guard against a vacuous test: the source must actually hold stat rows to carry.
    let src_stats = rows_of(&mut src, "SELECT tbl, idx, stat FROM sqlite_stat1 ORDER BY tbl, idx");
    assert!(!src_stats.is_empty(), "ANALYZE must populate sqlite_stat1 in the source");

    exec(&mut src, &format!("VACUUM INTO {}", dst.sql_literal()));

    // The copy's sqlite_stat1 holds the identical (tbl, idx, stat) rows, byte-exact.
    let mut copy = dst.open();
    assert_same(&mut src, &mut copy, "SELECT tbl, idx, stat FROM sqlite_stat1 ORDER BY tbl, idx");
}

// ===========================================================================
// 12) INTO-expression handling: parenthesized literal accepted; non-constant errors.
// ===========================================================================

#[test]
fn into_target_parenthesized_literal_is_accepted() {
    // `VACUUM INTO ('file')` — the parenthesized-literal branch of the INTO evaluator.
    let dst = TempDb::new();
    let mut src = mem();
    exec(&mut src, "CREATE TABLE t(a)");
    exec(&mut src, "INSERT INTO t VALUES (7), (8)");

    exec(&mut src, &format!("VACUUM INTO ({})", dst.sql_literal()));

    let mut copy = dst.open();
    assert_rows(&mut copy, "SELECT a FROM t ORDER BY a", &[vec![int(7)], vec![int(8)]]);
}

#[test]
fn into_target_non_constant_expression_errors() {
    // A non-constant INTO target (a text CONCAT real sqlite would evaluate) is rejected
    // loudly: the facade binds no parameters here, so there is no runtime value to
    // resolve — an honest documented gap, not a silent wrong path.
    let dst = TempDb::new();
    let mut src = mem();
    exec(&mut src, "CREATE TABLE t(a)");
    exec(&mut src, "INSERT INTO t VALUES (1)");

    let e = assert_exec_error(&mut src, &format!("VACUUM INTO {} || '.bak'", dst.sql_literal()));
    assert!(
        matches!(&e, Error::Sql(m) if m.to_lowercase().contains("constant string")),
        "a non-constant VACUUM INTO target must error naming the gap; got {e:?}"
    );
    // Nothing was written: the concat's would-be target (`<path>.bak`) was never created,
    // because the expression is rejected before any file is opened.
    assert!(!dst.sidecar(".bak").exists(), "a rejected VACUUM INTO must not create any file");
}

// ===========================================================================
// 13) A pre-existing ZERO-LENGTH target is accepted (only non-empty is rejected).
// ===========================================================================

#[test]
fn empty_target_file_is_accepted() {
    let dst = TempDb::new();
    // Pre-create the target as an empty (zero-length) file: acceptable, the copy fills it.
    std::fs::write(&dst.path, b"").expect("create an empty target file");
    assert_eq!(std::fs::metadata(&dst.path).unwrap().len(), 0, "target starts empty");

    let mut src = mem();
    exec(&mut src, "CREATE TABLE t(a INTEGER PRIMARY KEY, b)");
    exec(&mut src, "INSERT INTO t VALUES (1, 'x'), (2, 'y')");

    exec(&mut src, &format!("VACUUM INTO {}", dst.sql_literal()));

    let mut copy = dst.open();
    assert_rows(&mut copy, "SELECT a, b FROM t ORDER BY a", &[
        vec![int(1), text("x")],
        vec![int(2), text("y")],
    ]);
}

// ===========================================================================
// 14) The data copy COMMITS IN CHUNKS: a table larger than one chunk exercises the
//     commit + re-begin boundary, and every row must survive it intact.
// ===========================================================================

#[test]
fn large_copy_spans_multiple_commit_chunks() {
    // To bound memory the copy commits every `COMMIT_CHUNK_BYTES` (4 MiB) and re-begins,
    // rather than staging the whole database in one transaction. Every other test copies
    // < 1 chunk, so none crosses that boundary; this one copies ~10 MiB (a few chunks) so
    // a re-begin bug — a dropped or duplicated row, a broken rowid continuation, a
    // truncated value at the boundary — surfaces as a count / spot-check mismatch. The
    // point here is correctness across the boundary, not a memory measurement.
    let dst = TempDb::new();
    let mut src = mem();
    exec(&mut src, "CREATE TABLE t(id INTEGER PRIMARY KEY, s TEXT)");
    // ~5000 rows * ~2000 bytes ≈ 10 MiB of payload → the 4 MiB chunk is crossed twice.
    let n: i64 = 5000;
    let width = 2000;
    for i in 1..=n {
        // Zero-padded id as the body: distinct per row, so a mis-placed or duplicated row
        // at a chunk boundary is detectable, and fixed-length so a truncation shows up.
        let s = format!("{i:0>width$}");
        exec(&mut src, &format!("INSERT INTO t(id, s) VALUES ({i}, '{s}')"));
    }

    exec(&mut src, &format!("VACUUM INTO {}", dst.sql_literal()));

    let mut copy = dst.open();
    // No row dropped or duplicated across the chunk commits.
    assert_scalar(&mut copy, "SELECT count(*) FROM t", int(n));
    // Rowids stayed contiguous 1..=n (chunk re-begin continued the b-tree correctly).
    assert_scalar(&mut copy, "SELECT min(id) FROM t", int(1));
    assert_scalar(&mut copy, "SELECT max(id) FROM t", int(n));
    // No value was truncated at a boundary.
    assert_scalar(
        &mut copy,
        &format!("SELECT count(*) FROM t WHERE length(s) <> {width}"),
        int(0),
    );
    // Exact bytes of a first, a middle (well past the first commit), and the last row.
    assert_scalar(&mut copy, "SELECT s FROM t WHERE id = 1", text(&format!("{:0>width$}", 1)));
    assert_scalar(&mut copy, "SELECT s FROM t WHERE id = 2500", text(&format!("{:0>width$}", 2500)));
    assert_scalar(&mut copy, "SELECT s FROM t WHERE id = 5000", text(&format!("{:0>width$}", 5000)));
}
