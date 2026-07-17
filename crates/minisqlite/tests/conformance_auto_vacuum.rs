//! Conformance battery: the **auto_vacuum / incremental_vacuum WRITE side**, the
//! on-disk structure real `sqlite3` must be able to read back (fileformat2 §1.8
//! pointer-map pages + §1.3.12 vacuum header fields).
//!
//! `conformance_fileformat.rs` already pins the two page-1 HEADER bytes a vacuum
//! database sets (offset 52 non-zero, offset 64 = the incremental flag). This file
//! goes past the header to the STRUCTURE those bytes promise: that the file actually
//! carries valid ptrmap pages at the positions §1.8 computes, that each 5-byte entry
//! holds the correct `[type][big-endian parent]` back-pointer for the page it covers,
//! and that a file written this way reopens and round-trips all rows + indexes
//! (including overflowing values). Without this, offset 52 would be a lie — a header
//! that claims ptrmap pages that are not there, which real `sqlite3` cannot read.
//!
//! Every expected value is transcribed from the SQLite documentation, NEVER from the
//! engine's own `ptrmap.rs` (that would be a closed loop that shares its bugs):
//!   - `spec/sqlite-doc/fileformat2.html` §1.8 ("Pointer Map or Ptrmap Pages": the
//!     first ptrmap page is page 2; each covers `J = usable/5` following pages; ptrmap
//!     pages sit every `J+1` pages; each entry is `[1-byte type][4-byte big-endian
//!     parent page]`; types 1=rootpage, 2=freepage, 3=overflow-first, 4=overflow-rest,
//!     5=non-root b-tree).
//!   - `spec/sqlite-doc/fileformat2.html` §1.3.12 (offset 52 "largest root b-tree
//!     page", offset 64 the incremental-vacuum flag).
//!   - `spec/sqlite-doc/pragma.html` "auto_vacuum": mode 1 = FULL, 2 = INCREMENTAL,
//!     and "auto-vacuuming must be turned on before any tables are created".
//!
//! A failing case here is a WRITE-side format divergence a real `sqlite3` would
//! reject — left as a genuine failing assertion, never weakened to pass.

mod conformance;
use conformance::*;

use minisqlite::{Connection, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Temp-file harness: a unique on-disk path per test, cleaned up even on panic.
// (The same shape as the `TempDb` in the sibling on-disk test files; kept local
// rather than editing the shared `conformance` harness seam — see the note in
// `conformance_fileformat.rs`.)
// ---------------------------------------------------------------------------

/// An RAII guard over a unique temporary database path. `Drop` deletes the database
/// AND its `-journal` / `-wal` / `-shm` sidecars, so nothing is left behind even when
/// an assertion panics. No `.db` file is ever committed.
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
        path.push(format!("minisqlite_av_{pid}_{n}_{nanos}.db"));
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

    /// Byte length of the `-wal` sidecar (0 if absent). A committed WAL frame grows it
    /// past its 32-byte header, so this is a filesystem-level witness that writes went
    /// through the WAL backing rather than the rollback `-journal` — the check that keeps
    /// the WAL test below from silently degrading into a second rollback-mode case.
    fn wal_len(&self) -> u64 {
        std::fs::metadata(self.sidecar("-wal")).map(|m| m.len()).unwrap_or(0)
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
// Byte + §1.8 layout helpers (transcribed from the spec, not imported).
// Every multibyte header/ptrmap field is BIG-ENDIAN.
// ---------------------------------------------------------------------------

const HEADER_SIZE: usize = 100;

/// One ptrmap entry is 5 bytes: `[1-byte type][4-byte big-endian parent]` (§1.8).
const PTRMAP_ENTRY_SIZE: usize = 5;

/// The `-wal` header is 32 bytes; any committed frame grows the file past it. Used as
/// the "WAL actually engaged" witness (matches `minisqlite-pager`'s `WAL_HEADER_SIZE`).
const WAL_HEADER_SIZE: u64 = 32;

fn be16(b: &[u8], off: usize) -> u16 {
    u16::from_be_bytes([b[off], b[off + 1]])
}

fn be32(b: &[u8], off: usize) -> u32 {
    u32::from_be_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

/// Effective page size: the offset-16 u16, with the sentinel `1` meaning 65536 (§1.3.2).
fn effective_page_size(raw: &[u8]) -> u32 {
    let field = be16(raw, 16);
    if field == 1 {
        65_536
    } else {
        field as u32
    }
}

/// The usable page size: page size minus the reserved-bytes-per-page at offset 20 (§1.3.4).
fn usable_size(raw: &[u8]) -> usize {
    effective_page_size(raw) as usize - raw[20] as usize
}

/// `J = usable / 5`: the number of 5-byte entries a ptrmap page holds (§1.8).
fn ptrmap_entries_per_page(usable: usize) -> u64 {
    (usable / PTRMAP_ENTRY_SIZE) as u64
}

/// The spacing between consecutive ptrmap pages: `J + 1` (§1.8).
fn ptrmap_stride(usable: usize) -> u64 {
    ptrmap_entries_per_page(usable) + 1
}

/// Whether page `p` sits at a ptrmap POSITION (§1.8): page 2, then every `J+1` pages.
fn is_ptrmap_page(usable: usize, p: u64) -> bool {
    p >= 2 && (p - 2) % ptrmap_stride(usable) == 0
}

/// The ptrmap page carrying data page `p`'s entry: round `p` down to its stride window
/// (`B = 2 + ((p-2)/stride)*stride`). `p` must be a real data page (>= 3, not a ptrmap
/// page).
fn ptrmap_page_of(usable: usize, p: u64) -> u64 {
    let stride = ptrmap_stride(usable);
    2 + ((p - 2) / stride) * stride
}

/// Read data page `p`'s 5-byte ptrmap entry from the raw file as `(type, parent)` (§1.8).
fn read_ptrmap_entry(raw: &[u8], page_size: usize, usable: usize, p: u64) -> (u8, u32) {
    let b = ptrmap_page_of(usable, p);
    let within = (p - b - 1) as usize * PTRMAP_ENTRY_SIZE;
    let base = (b as usize - 1) * page_size + within;
    (raw[base], be32(raw, base + 1))
}

/// Read the whole on-disk file, asserting a full 100-byte page-1 header is present.
fn read_db(tmp: &TempDb) -> Vec<u8> {
    let raw = std::fs::read(&tmp.path)
        .unwrap_or_else(|e| panic!("failed to read db file {:?}: {e}", tmp.path));
    assert!(
        raw.len() >= HEADER_SIZE,
        "db file {:?} is only {} byte(s); a committed write must flush a full header",
        tmp.path,
        raw.len(),
    );
    raw
}

/// Best-effort request for rollback-journal mode so a just-committed page-1 value is read
/// from the MAIN db file rather than a lagging `-wal` sidecar (mirrors the sibling
/// on-disk tests). The result is ignored on purpose: setup must not hide the check.
fn request_rollback_journal(db: &mut Connection) {
    let _ = db.query("PRAGMA journal_mode=DELETE");
}

fn as_int(v: &Value) -> i64 {
    match v {
        Value::Integer(i) => *i,
        other => panic!("expected an Integer value, got {other:?}"),
    }
}

fn as_text(v: &Value) -> String {
    match v {
        Value::Text(s) => s.clone(),
        other => panic!("expected a Text value, got {other:?}"),
    }
}

/// Run a single-row, single-column integer PRAGMA (e.g. `freelist_count`, `page_count`,
/// `auto_vacuum`) and return its value.
fn pragma_int(db: &mut Connection, sql: &str) -> i64 {
    let qr = query(db, sql);
    assert_eq!(qr.rows.len(), 1, "{sql} must return exactly one row");
    as_int(&qr.rows[0][0])
}

/// The on-disk size of the main database file, in bytes (0 if it does not exist yet).
fn file_len(tmp: &TempDb) -> u64 {
    std::fs::metadata(&tmp.path).map(|m| m.len()).unwrap_or(0)
}

/// Run a `PRAGMA journal_mode[=…]` and return the single mode string it reports
/// (`"wal"`, `"delete"`, …), pinning the one-row / one-column shape (pragma.html
/// #pragma_journal_mode).
fn journal_mode(db: &mut Connection, sql: &str) -> String {
    let qr = query(db, sql);
    assert_eq!(qr.rows.len(), 1, "{sql} must return exactly one row");
    as_text(&qr.rows[0][0])
}

// ===========================================================================
// Ptrmap pages exist at the §1.8 positions with the correct 5-byte entries.
// ===========================================================================

#[test]
fn auto_vacuum_ptrmap_page_holds_root_entries_at_computed_position() {
    // §1.8 + §1.3.12. Enable FULL auto_vacuum on an empty db, then create two empty
    // tables. Allocation is sequential and reserves the ptrmap slot: page 1 = the
    // schema, page 2 = the first ptrmap page (§1.8 puts it immediately after page 1),
    // page 3 = table t's root, page 4 = table u's root — the same layout real sqlite3
    // produces for an auto_vacuum database's first two tables. Both roots are empty leaf
    // b-trees.
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        request_rollback_journal(&mut db);
        // Fix the page size so the single-ptrmap-page assumption below is explicit and
        // independent of the compiled-in default.
        let _ = db.query("PRAGMA page_size = 4096");
        let _ = db.query("PRAGMA auto_vacuum = 1");
        exec(&mut db, "CREATE TABLE t(a)");
        exec(&mut db, "CREATE TABLE u(b)");
    }
    let raw = read_db(&tmp);
    let ps = effective_page_size(&raw) as usize;
    let usable = usable_size(&raw);
    let page_count = (raw.len() / ps) as u64;

    // Offset 52 = largest root b-tree page = 4 (u's root, the larger of the two roots).
    assert_eq!(
        be32(&raw, 52),
        4,
        "off 52 largest-root-btree must be page 4 (the larger of the two table roots)"
    );
    // The incremental flag is 0 for FULL (§1.3.12).
    assert_eq!(be32(&raw, 64), 0, "off 64 must be 0 for FULL auto_vacuum");

    // Page 2 is the first (and here only) ptrmap page.
    assert!(is_ptrmap_page(usable, 2), "page 2 must be the first ptrmap page (§1.8)");
    assert!(
        page_count <= 2 + ptrmap_stride(usable),
        "this test assumes a single ptrmap page; page_count={page_count}, stride={}",
        ptrmap_stride(usable),
    );

    // Both table roots carry a ROOTPAGE (type 1) entry with parent 0 (§1.8), located in
    // ptrmap page 2 at the offset the §1.8 formula computes.
    for root in [3u64, 4] {
        assert_eq!(ptrmap_page_of(usable, root), 2, "page {root} is covered by ptrmap page 2");
        let (kind, parent) = read_ptrmap_entry(&raw, ps, usable, root);
        assert_eq!(kind, 1, "page {root} is a table root: ptrmap type must be 1 (ROOTPAGE)");
        assert_eq!(parent, 0, "a root page's ptrmap parent must be 0");
    }

    // The first 5 bytes of page 2 ARE page 3's entry: `01 00 00 00 00`. This also proves
    // page 2 is not a b-tree page (whose first byte would be 0x02/0x05/0x0a/0x0d).
    let page2 = &raw[ps..2 * ps];
    assert_eq!(
        &page2[0..PTRMAP_ENTRY_SIZE],
        &[1u8, 0, 0, 0, 0],
        "ptrmap page 2's first entry must be page 3's ROOTPAGE back-pointer (01 00 00 00 00)"
    );
}

// ===========================================================================
// An auto_vacuum-written file reopens and round-trips all rows + indexes, and
// every data page it contains has a valid ptrmap back-pointer (no leaked page).
// ===========================================================================

#[test]
fn auto_vacuum_file_reopens_and_round_trips_rows_and_index() {
    let tmp = TempDb::new();
    let n: i64 = 200;
    // A wide `pad` column makes each row ~300 bytes, so 200 rows overflow a single
    // 4096-byte leaf and the table b-tree grows an interior root over multiple leaf
    // pages — exercising the non-root b-tree (type-5) ptrmap path, not just roots.
    let pad = "p".repeat(300);
    {
        let mut db = tmp.open();
        request_rollback_journal(&mut db);
        let _ = db.query("PRAGMA page_size = 4096");
        let _ = db.query("PRAGMA auto_vacuum = 1");
        exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT, pad TEXT)");
        exec(&mut db, "CREATE INDEX i ON t(b)");
        for k in 0..n {
            exec(&mut db, &format!("INSERT INTO t VALUES ({k}, 'v{k:04}', '{pad}')"));
        }
    }

    // Still an auto_vacuum database after all the writes: offset 52 stays non-zero.
    let raw = read_db(&tmp);
    assert!(be32(&raw, 52) != 0, "off 52 must stay non-zero for an auto_vacuum db");

    // Every ptrmap-covered data page must carry a valid entry type (1..=5). A zero type
    // would mean a page reachable in the file that no ptrmap entry accounts for — a
    // structural leak that would make the file unreadable to real sqlite. (No deletes
    // ran, so there are no freelist pages; every page is a root/b-tree/overflow page.)
    let ps = effective_page_size(&raw) as usize;
    let usable = usable_size(&raw);
    let page_count = (raw.len() / ps) as u64;
    let mut saw_btree = false;
    for p in 3..=page_count {
        if is_ptrmap_page(usable, p) {
            continue;
        }
        let (kind, _) = read_ptrmap_entry(&raw, ps, usable, p);
        assert!(
            (1..=5).contains(&kind),
            "auto_vacuum: data page {p} has ptrmap type {kind}, expected 1..=5 (leaked page?)"
        );
        if kind == 5 {
            saw_btree = true;
        }
    }
    assert!(
        saw_btree,
        "200 rows + an index must produce at least one non-root b-tree page (type 5)"
    );

    // Reopen through the same header and verify the data survived byte-for-byte.
    {
        let mut db = tmp.open();
        assert_eq!(
            as_int(&query(&mut db, "SELECT count(*) FROM t").rows[0][0]),
            n,
            "all rows survive the reopen"
        );
        let qr = query(&mut db, "SELECT a FROM t WHERE b = 'v0137'");
        assert_eq!(qr.rows.len(), 1, "index lookup finds the row after reopen");
        assert_eq!(as_int(&qr.rows[0][0]), 137, "index lookup returns the right row");
        let got: Vec<i64> =
            query(&mut db, "SELECT a FROM t ORDER BY a").rows.iter().map(|r| as_int(&r[0])).collect();
        assert_eq!(got, (0..n).collect::<Vec<_>>(), "ordered scan matches inserted rows");
    }
}

// ===========================================================================
// DROP TABLE in auto_vacuum mode must not crash and must leave a structurally
// valid file: the dropped table's pages become FREEPAGE (type 2) entries, never
// an unrepresentable type-0. (Regression: `DROP TABLE` frees no pages in the
// catalog, so its former root/leaves are allocated-but-unreferenced; the ptrmap
// writer used to `debug_assert!` (panic) / emit an invalid type-0 for them. The
// finalize leak sweep now puts them on the freelist instead.)
// ===========================================================================

#[test]
fn auto_vacuum_drop_table_no_crash_and_structurally_valid() {
    // A minimal reproducer first: create then immediately drop, so the
    // former root (page 3) is the leaked page. In a debug build this used to panic
    // inside the DROP's autocommit; it must now complete.
    {
        let tmp = TempDb::new();
        {
            let mut db = tmp.open();
            request_rollback_journal(&mut db);
            let _ = db.query("PRAGMA page_size = 4096");
            let _ = db.query("PRAGMA auto_vacuum = 1");
            exec(&mut db, "CREATE TABLE t(a)");
            exec(&mut db, "DROP TABLE t"); // must not panic / corrupt
        }
        assert_no_type0_entries(&tmp);
    }

    // A larger table so the drop leaks many pages (root + interior + leaves), then a
    // second live table so page 1's schema still has a row and the forest is non-empty
    // after the drop.
    let tmp = TempDb::new();
    let pad = "p".repeat(300);
    {
        let mut db = tmp.open();
        request_rollback_journal(&mut db);
        let _ = db.query("PRAGMA page_size = 4096");
        let _ = db.query("PRAGMA auto_vacuum = 1");
        exec(&mut db, "CREATE TABLE keep(x)");
        exec(&mut db, "CREATE TABLE t(a INTEGER, pad TEXT)");
        for k in 0..200 {
            exec(&mut db, &format!("INSERT INTO t VALUES ({k}, '{pad}')"));
        }
        exec(&mut db, "INSERT INTO keep VALUES (42)");
        exec(&mut db, "DROP TABLE t"); // frees many pages; must not panic / corrupt
    }
    assert_no_type0_entries(&tmp);

    // The surviving table still round-trips after the drop + reopen.
    {
        let mut db = tmp.open();
        assert_eq!(as_int(&query(&mut db, "SELECT x FROM keep").rows[0][0]), 42);
        // The dropped table is gone from the schema.
        let names: Vec<String> = query(&mut db, "SELECT name FROM sqlite_master WHERE type='table'")
            .rows
            .iter()
            .map(|r| as_text(&r[0]))
            .collect();
        assert!(names.contains(&"keep".to_string()), "keep survives");
        assert!(!names.contains(&"t".to_string()), "t is dropped");
    }
}

/// Assert every ptrmap-covered data page in the file carries a valid §1.8 entry type
/// (1..=5) — never a type-0, which is not a real entry type and would make the file
/// unreadable to real sqlite. This is the structural-validity check the DROP path must
/// satisfy whether or not FULL compaction later truncates the freed pages.
fn assert_no_type0_entries(tmp: &TempDb) {
    let raw = read_db(tmp);
    let ps = effective_page_size(&raw) as usize;
    let usable = usable_size(&raw);
    let page_count = (raw.len() / ps) as u64;
    for p in 3..=page_count {
        if is_ptrmap_page(usable, p) {
            continue;
        }
        let (kind, _) = read_ptrmap_entry(&raw, ps, usable, p);
        assert!(
            (1..=5).contains(&kind),
            "auto_vacuum: page {p} has ptrmap type {kind}, expected 1..=5 (a leaked page \
             written as an invalid type-0 entry?)"
        );
    }
}

// ===========================================================================
// An overflowing value produces a valid overflow chain in the ptrmap and
// round-trips verbatim.
// ===========================================================================

#[test]
fn auto_vacuum_overflow_chain_recorded_and_round_trips() {
    let tmp = TempDb::new();
    // 20000 bytes far exceeds a 4096-byte page's usable payload, so the value spills onto
    // a multi-page overflow chain (§1.6): the first overflow page is OVERFLOW1 (type 3),
    // each later page OVERFLOW2 (type 4) (§1.8).
    let big = "x".repeat(20_000);
    {
        let mut db = tmp.open();
        request_rollback_journal(&mut db);
        let _ = db.query("PRAGMA page_size = 4096");
        let _ = db.query("PRAGMA auto_vacuum = 1");
        exec(&mut db, "CREATE TABLE t(a TEXT)");
        exec(&mut db, &format!("INSERT INTO t VALUES ('{big}')"));
    }
    let raw = read_db(&tmp);
    let ps = effective_page_size(&raw) as usize;
    let usable = usable_size(&raw);
    let page_count = (raw.len() / ps) as u64;

    let mut ovf1 = 0;
    let mut ovf2 = 0;
    for p in 3..=page_count {
        if is_ptrmap_page(usable, p) {
            continue;
        }
        let (kind, parent) = read_ptrmap_entry(&raw, ps, usable, p);
        if kind == 3 {
            ovf1 += 1;
            // The first overflow page's parent is the b-tree page holding the cell — a
            // real, non-ptrmap page in range, never 0.
            assert!(
                parent >= 3 && (parent as u64) <= page_count && !is_ptrmap_page(usable, parent as u64),
                "OVERFLOW1 page {p} parent {parent} must be a real b-tree page"
            );
        }
        if kind == 4 {
            ovf2 += 1;
        }
    }
    assert_eq!(ovf1, 1, "exactly one PTRMAP_OVERFLOW1 (type 3) entry — the chain's first page");
    assert!(ovf2 >= 1, "a 20000-byte value spans multiple overflow pages, so type-4 entries exist");

    // The overflowing value round-trips verbatim after a reopen.
    {
        let mut db = tmp.open();
        let qr = query(&mut db, "SELECT a, length(a) FROM t");
        assert_eq!(as_int(&qr.rows[0][1]), 20_000, "length survives the reopen");
        assert_eq!(as_text(&qr.rows[0][0]), big, "the overflowing value round-trips verbatim");
    }
}

// ===========================================================================
// FULL auto_vacuum truncates the freelist off the file at every commit.
// pragma.html "auto_vacuum": "the freelist pages are moved to the end of the
// database file and the database file is truncated to remove the freelist pages
// at every transaction commit" — so after any commit a FULL db has
// freelist_count == 0 and the file has shrunk.
// ===========================================================================

#[test]
fn full_auto_vacuum_truncates_freelist_to_zero_on_commit() {
    let tmp = TempDb::new();
    let pad = "p".repeat(60);
    {
        let mut db = tmp.open();
        request_rollback_journal(&mut db);
        // Small pages so a couple hundred rows span many pages (and cross ptrmap 105).
        let _ = db.query("PRAGMA page_size = 512");
        let _ = db.query("PRAGMA auto_vacuum = 1");
        exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
        exec(&mut db, "BEGIN");
        for k in 0..200 {
            exec(&mut db, &format!("INSERT INTO t VALUES ({k}, 'v{k:04}-{pad}')"));
        }
        exec(&mut db, "COMMIT");

        let pages_full = pragma_int(&mut db, "PRAGMA page_count");
        let len_full = file_len(&tmp);
        assert!(pages_full > 20, "200 wide rows must span many pages to compact; got {pages_full}");
        // A freshly loaded (insert-only) FULL db has no free pages: FULL compaction ran at
        // each commit but there was nothing to reclaim.
        assert_eq!(pragma_int(&mut db, "PRAGMA freelist_count"), 0, "insert-only FULL: no free pages");

        // Deleting every row frees the table's data pages. Under FULL, the DELETE's commit
        // must relocate + truncate them off, leaving ZERO free pages and a smaller file.
        exec(&mut db, "DELETE FROM t");
        assert_eq!(
            pragma_int(&mut db, "PRAGMA freelist_count"),
            0,
            "FULL auto_vacuum must truncate freed pages at commit (freelist_count == 0)"
        );
        let pages_after = pragma_int(&mut db, "PRAGMA page_count");
        assert!(
            pages_after < pages_full,
            "FULL must shrink the page count after a delete: {pages_after} !< {pages_full}"
        );
        assert!(
            file_len(&tmp) < len_full,
            "FULL must shrink the file on disk after a delete+commit"
        );

        // The emptied table still answers a scan, and new inserts still work post-compaction.
        assert_eq!(pragma_int(&mut db, "SELECT count(*) FROM t"), 0, "t is empty after DELETE");
        exec(&mut db, "INSERT INTO t VALUES (7, 'after')");
        assert_eq!(pragma_int(&mut db, "SELECT count(*) FROM t"), 1, "insert works after compaction");
    }

    // The compacted file is structurally valid (no leaked type-0 ptrmap entries) and
    // reopens with the surviving row intact.
    assert_no_type0_entries(&tmp);
    {
        let mut db = tmp.open();
        assert_eq!(pragma_int(&mut db, "PRAGMA freelist_count"), 0, "still zero free pages after reopen");
        let qr = query(&mut db, "SELECT a, b FROM t");
        assert_eq!(qr.rows.len(), 1, "the row inserted after compaction survives reopen");
        assert_eq!(as_int(&qr.rows[0][0]), 7);
        assert_eq!(as_text(&qr.rows[0][1]), "after");
    }
}

// ===========================================================================
// FULL auto_vacuum compaction that RELOCATES a table root (its rootpage moves to
// a lower page) rewrites the sqlite_schema.rootpage, reloads the catalog, and the
// database still reopens and round-trips. This exercises the type-1 (ROOTPAGE)
// relocation path + catalog reload, the riskiest part of compaction.
// ===========================================================================

#[test]
fn full_auto_vacuum_root_relocation_reopens_and_round_trips() {
    let tmp = TempDb::new();
    let pad = "p".repeat(60);
    {
        let mut db = tmp.open();
        request_rollback_journal(&mut db);
        let _ = db.query("PRAGMA page_size = 512");
        let _ = db.query("PRAGMA auto_vacuum = 1");

        // `big` is created and fully populated FIRST, so its data pages occupy the low/
        // middle of the file. `tail` is created LAST, so its root is the highest live page.
        exec(&mut db, "CREATE TABLE big(a INTEGER, b TEXT)");
        exec(&mut db, "BEGIN");
        for k in 0..200 {
            exec(&mut db, &format!("INSERT INTO big VALUES ({k}, 'v{k:04}-{pad}')"));
        }
        exec(&mut db, "COMMIT");
        exec(&mut db, "CREATE TABLE tail(x INTEGER, y TEXT)");
        exec(&mut db, "INSERT INTO tail VALUES (99, 'kept')");

        // Emptying `big` frees its many data pages (low/middle of the file), leaving
        // `tail`'s root stranded at the high end above a sea of free slots. FULL compaction
        // at this commit must relocate `tail`'s root DOWN into a freed slot — a root move —
        // rewrite sqlite_schema.rootpage, reload the catalog, and truncate to zero free.
        exec(&mut db, "DELETE FROM big");
        assert_eq!(
            pragma_int(&mut db, "PRAGMA freelist_count"),
            0,
            "FULL compaction (with a root relocation) must reach zero free pages"
        );

        // Immediately (same connection, catalog just reloaded) `tail` must still resolve to
        // its NEW root and return its row — proving the schema rewrite + reload are correct.
        let qr = query(&mut db, "SELECT x, y FROM tail");
        assert_eq!(qr.rows.len(), 1, "tail's row survives the root relocation (same connection)");
        assert_eq!(as_int(&qr.rows[0][0]), 99);
        assert_eq!(as_text(&qr.rows[0][1]), "kept");
        assert_eq!(pragma_int(&mut db, "SELECT count(*) FROM big"), 0, "big is empty");
    }

    // A FRESH connection re-reads the schema from the compacted page 1: `tail`'s moved
    // rootpage on disk must point at the relocated page, or this reopen finds no row.
    assert_no_type0_entries(&tmp);
    {
        let mut db = tmp.open();
        let qr = query(&mut db, "SELECT x, y FROM tail");
        assert_eq!(qr.rows.len(), 1, "tail's row survives a full reopen after root relocation");
        assert_eq!(as_int(&qr.rows[0][0]), 99);
        assert_eq!(as_text(&qr.rows[0][1]), "kept");
        exec(&mut db, "INSERT INTO tail VALUES (100, 'more')");
        assert_eq!(pragma_int(&mut db, "SELECT count(*) FROM tail"), 2, "insert into relocated table works");
    }
}

/// Assert table `keep` round-trips exactly: the full row count (`short_rows` short rows
/// plus the one overflow row), the multi-page overflow row's payload byte-for-byte (which
/// only reads back correctly if the type-3 head + type-4 continuation back-pointers were
/// rewritten right during relocation), and EVERY short row in order (a mis-rewired type-5
/// child pointer that drops or corrupts a leaf's rows shows here — not only at the ends,
/// and even if it happened to preserve `count(*)`).
fn assert_keep_intact(db: &mut Connection, short_rows: i64, overflow_val: &str) {
    assert_eq!(
        pragma_int(db, "SELECT count(*) FROM keep"),
        short_rows + 1,
        "keep must retain every row through the relocation"
    );
    let qr = query(db, "SELECT v FROM keep WHERE id = 9999");
    assert_eq!(qr.rows.len(), 1, "the overflow row is present after relocation");
    assert_eq!(
        as_text(&qr.rows[0][0]),
        overflow_val,
        "the multi-page overflow payload must round-trip byte-for-byte (types 3/4 rewrite)"
    );
    // Every short row, in id order, value-exact — the overflow row (id 9999) is excluded by
    // the predicate and checked above.
    let qr = query(db, "SELECT id, v FROM keep WHERE id < 9999 ORDER BY id");
    assert_eq!(qr.rows.len() as i64, short_rows, "all short rows survive the relocation");
    for (k, row) in qr.rows.iter().enumerate() {
        assert_eq!(as_int(&row[0]), k as i64, "short row order intact at index {k}");
        assert_eq!(as_text(&row[1]), format!("k{k:04}"), "short row {k} value intact");
    }
}

// ===========================================================================
// FULL auto_vacuum must relocate LIVE b-tree and overflow pages, not just a single
// ROOTPAGE. The sibling `..._root_relocation_...` case moves one type-1 root; this
// pins the harder ptrmap types that otherwise run only via reclaim's decline-swallow
// path: non-root B-TREE pages (type 5, via rewrite_child_pointer), an overflow head
// (type 3, via rewrite_overflow_head) and overflow continuation pages (type 4, via
// rewrite_overflow_next).
//
// Shape: `low` is created + filled FIRST so its many data pages occupy the low/middle
// of the file; `keep` is created LAST as a MULTI-LEVEL b-tree (enough rows to force an
// interior page over leaf children = type 5) carrying one row whose payload far exceeds
// the local threshold (a multi-page overflow chain = type 3 then type 4), so keep's live
// pages are the file's tail. DELETE FROM low frees the low pages; FULL must relocate
// keep's live tail pages DOWN into those slots to reach zero free. If any of the three
// pointer rewrites is wrong, reclaim either DECLINES (freelist_count != 0) or keep's rows
// come back corrupted — so this fails on a real break instead of passing inertly (the
// swallowed-decline path is exactly what left this untested).
// ===========================================================================

#[test]
fn full_auto_vacuum_relocates_live_btree_and_overflow_pages() {
    let tmp = TempDb::new();
    let low_pad = "L".repeat(48);
    // >> the 512-byte-page local payload threshold (U-35 == 477), so this spans several
    // overflow pages: a type-3 head followed by type-4 continuations.
    let overflow_val = "O".repeat(3000);
    const SHORT_ROWS: i64 = 200;
    {
        let mut db = tmp.open();
        request_rollback_journal(&mut db);
        // page_size FIRST (a later reset reformats page 1, clearing off52), then auto_vacuum.
        let _ = db.query("PRAGMA page_size = 512");
        let _ = db.query("PRAGMA auto_vacuum = 1");

        exec(&mut db, "CREATE TABLE low(a INTEGER, b TEXT)");
        exec(&mut db, "BEGIN");
        for k in 0..300 {
            exec(&mut db, &format!("INSERT INTO low VALUES ({k}, 'l{k:04}-{low_pad}')"));
        }
        exec(&mut db, "COMMIT");

        // `keep` created LAST: its pages are the highest live pages in the file.
        exec(&mut db, "CREATE TABLE keep(id INTEGER, v TEXT)");
        exec(&mut db, "BEGIN");
        for k in 0..SHORT_ROWS {
            exec(&mut db, &format!("INSERT INTO keep VALUES ({k}, 'k{k:04}')"));
        }
        exec(&mut db, &format!("INSERT INTO keep VALUES (9999, '{overflow_val}')"));
        exec(&mut db, "COMMIT");

        let pages_before = pragma_int(&mut db, "PRAGMA page_count");
        assert!(pages_before > 40, "low + multi-level keep + overflow must span many pages; got {pages_before}");
        assert_eq!(pragma_int(&mut db, "PRAGMA freelist_count"), 0, "insert-only FULL keeps no free pages");

        // Free the low/middle pages; FULL must relocate keep's live tail pages (types 5/3/4)
        // down into them to reach zero free.
        exec(&mut db, "DELETE FROM low");
        assert_eq!(
            pragma_int(&mut db, "PRAGMA freelist_count"),
            0,
            "FULL must relocate keep's live b-tree + overflow pages to reach zero free"
        );
        assert!(
            pragma_int(&mut db, "PRAGMA page_count") < pages_before,
            "FULL must shrink the page count after freeing `low`"
        );

        // Right after the relocation (same connection, catalog reloaded) keep is intact.
        assert_keep_intact(&mut db, SHORT_ROWS, &overflow_val);
        assert_eq!(pragma_int(&mut db, "SELECT count(*) FROM low"), 0, "low is empty");
    }

    // The relocated file is structurally valid (no leaked type-0 ptrmap entries) and a
    // fresh connection resolves keep's b-tree + overflow chain through the rewritten
    // on-disk pointers.
    assert_no_type0_entries(&tmp);
    {
        let mut db = tmp.open();
        assert_eq!(pragma_int(&mut db, "PRAGMA freelist_count"), 0, "still zero free after reopen");
        assert_keep_intact(&mut db, SHORT_ROWS, &overflow_val);
    }
}

// ===========================================================================
// FULL auto_vacuum compaction is journal-mode independent: it must also truncate
// the freelist at commit in WAL mode, where a commit appends frames to the -wal
// file (recording the new, smaller page count in the commit frame's db-size,
// walformat §4) instead of writing the database file directly. The reclaim path
// truncates through the same `truncate_to` seam; this pins that the WAL commit
// carries the shrink so the LOGICAL size (page_count / freelist_count) drops and
// survives a WAL reopen — the one path the rollback-journal cases above do not
// exercise.
//
// The WAL "reopen model" (see conformance_wal_durability.rs): the storage backing
// is fixed at Connection::open from the page-1 header version bytes, so
// `PRAGMA journal_mode=WAL` only takes effect on the NEXT open — a mid-connection
// switch keeps the rollback backing (its writes would go to a `-journal`, NOT the
// WAL, making a same-connection "WAL" test inert). So scope A fixes page_size then
// auto_vacuum (both must precede any table) then flips to WAL and closes; scope B
// reopens WAL-backed and does the CREATE/INSERT/DELETE so the FULL truncation
// genuinely runs through `WalStore::apply_commit`. ORDERING trap: page_size must be
// FIRST — a later page_size reset reformats page 1 to a fresh rollback (version-1)
// header, which would both undo the WAL flip AND clear off52 (see the engine's
// `reinit_pager_with_page_size`). The test PROVES WAL engaged (journal_mode=='wal'
// AND the `-wal` grows past its 32-byte header during the writes) so it can never
// silently degrade back into a second rollback-journal case. (Logical PRAGMAs for
// the size checks: in WAL mode the main db FILE lags the WAL until a checkpoint, so
// raw db-file bytes are not a valid post-commit oracle here.)
// ===========================================================================

#[test]
fn full_auto_vacuum_under_wal_truncates_and_round_trips() {
    let tmp = TempDb::new();
    let pad = "p".repeat(60);

    // Scope A: fix the creation-time header on the empty db — page_size FIRST (see the
    // ordering trap above), then auto_vacuum=FULL — then flip to WAL and close. WAL takes
    // effect only on the next open.
    {
        let mut db = tmp.open();
        let _ = db.query("PRAGMA page_size = 512");
        let _ = db.query("PRAGMA auto_vacuum = 1"); // FULL, before any table
        assert_eq!(
            journal_mode(&mut db, "PRAGMA journal_mode=WAL"),
            "wal",
            "PRAGMA journal_mode=WAL must report the resulting mode as wal"
        );
    }

    // Scope B: now WAL-backed. Every write here appends frames to the `-wal`, so the
    // DELETE's FULL truncation runs through WalStore, not the rollback DiskStore.
    {
        let mut db = tmp.open();
        assert_eq!(
            journal_mode(&mut db, "PRAGMA journal_mode"),
            "wal",
            "scope B must reopen WAL-backed (the reopen model engaged)"
        );
        assert_eq!(
            pragma_int(&mut db, "PRAGMA auto_vacuum"),
            1,
            "FULL auto_vacuum must survive into the WAL reopen (off52 preserved by the WAL flip)"
        );

        exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
        exec(&mut db, "BEGIN");
        for k in 0..200 {
            exec(&mut db, &format!("INSERT INTO t VALUES ({k}, 'v{k:04}-{pad}')"));
        }
        exec(&mut db, "COMMIT");

        // Witness that the writes genuinely went through the WAL backing (not a rollback
        // `-journal`): a committed frame grows the `-wal` past its 32-byte header. If this
        // fails the test is inert — the exact false-coverage trap this rewrite fixes.
        let wal_len = tmp.wal_len();
        assert!(
            wal_len > WAL_HEADER_SIZE,
            "WAL writes must grow the -wal past its {WAL_HEADER_SIZE}-byte header; got {wal_len}"
        );

        let pages_full = pragma_int(&mut db, "PRAGMA page_count");
        assert!(pages_full > 20, "200 wide rows must span many pages; got {pages_full}");
        assert_eq!(pragma_int(&mut db, "PRAGMA freelist_count"), 0, "insert-only FULL: no free pages");

        // The DELETE's WAL commit (and the FULL truncation that follows it) must relocate
        // + truncate the freed pages off, leaving zero free pages and a smaller logical db
        // — the shrink recorded in the WAL commit frame's db-size.
        exec(&mut db, "DELETE FROM t");
        assert_eq!(
            pragma_int(&mut db, "PRAGMA freelist_count"),
            0,
            "FULL under WAL must truncate freed pages at commit (freelist_count == 0)"
        );
        assert!(
            pragma_int(&mut db, "PRAGMA page_count") < pages_full,
            "FULL under WAL must shrink the logical page count after a delete"
        );

        exec(&mut db, "INSERT INTO t VALUES (7, 'after')");
        assert_eq!(pragma_int(&mut db, "SELECT count(*) FROM t"), 1, "insert works after WAL compaction");
    }

    // Scope C: reopen (WAL) — the surviving row is intact and the db is still compacted,
    // proving the shrink rode the WAL commit durably.
    {
        let mut db = tmp.open();
        assert_eq!(journal_mode(&mut db, "PRAGMA journal_mode"), "wal", "scope C still WAL-backed");
        assert_eq!(pragma_int(&mut db, "PRAGMA freelist_count"), 0, "still zero free pages after WAL reopen");
        let qr = query(&mut db, "SELECT a, b FROM t");
        assert_eq!(qr.rows.len(), 1, "the row inserted after compaction survives the WAL reopen");
        assert_eq!(as_int(&qr.rows[0][0]), 7);
        assert_eq!(as_text(&qr.rows[0][1]), "after");
    }
}

// ===========================================================================
// INCREMENTAL auto_vacuum KEEPS free pages at commit, and `PRAGMA
// incremental_vacuum(N)` reclaims up to N of them on demand; a no-arg
// `PRAGMA incremental_vacuum` clears the rest. pragma.html "incremental_vacuum".
// ===========================================================================

#[test]
fn incremental_auto_vacuum_keeps_then_reclaims_on_demand() {
    let tmp = TempDb::new();
    let pad = "p".repeat(60);
    {
        let mut db = tmp.open();
        request_rollback_journal(&mut db);
        let _ = db.query("PRAGMA page_size = 512");
        let _ = db.query("PRAGMA auto_vacuum = 2"); // INCREMENTAL
        // The GET form and the header flag both report incremental.
        assert_eq!(pragma_int(&mut db, "PRAGMA auto_vacuum"), 2, "auto_vacuum GET reports INCREMENTAL");

        exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
        exec(&mut db, "BEGIN");
        for k in 0..200 {
            exec(&mut db, &format!("INSERT INTO t VALUES ({k}, 'v{k:04}-{pad}')"));
        }
        exec(&mut db, "COMMIT");

        exec(&mut db, "DELETE FROM t");
        // INCREMENTAL: the DELETE's commit does NOT truncate — free pages stay.
        let fc0 = pragma_int(&mut db, "PRAGMA freelist_count");
        assert!(fc0 >= 5, "INCREMENTAL keeps freed pages on the freelist; got {fc0}");
        let pc0 = pragma_int(&mut db, "PRAGMA page_count");

        // `incremental_vacuum(3)` reclaims exactly 3 free pages (they sit at the tail after
        // deleting the whole table, so this is a pure truncation of 3 pages).
        exec(&mut db, "PRAGMA incremental_vacuum(3)");
        let fc1 = pragma_int(&mut db, "PRAGMA freelist_count");
        assert_eq!(fc1, fc0 - 3, "incremental_vacuum(3) reclaims 3 pages: {fc0} -> {fc1}");
        assert!(pragma_int(&mut db, "PRAGMA page_count") < pc0, "page count drops after reclaim");

        // A no-argument `incremental_vacuum` clears the ENTIRE remaining freelist.
        exec(&mut db, "PRAGMA incremental_vacuum");
        assert_eq!(
            pragma_int(&mut db, "PRAGMA freelist_count"),
            0,
            "no-arg incremental_vacuum clears the whole freelist"
        );

        // Still INCREMENTAL (a reclaim never changes the mode) and the table still works.
        assert_eq!(pragma_int(&mut db, "PRAGMA auto_vacuum"), 2, "mode unchanged by incremental_vacuum");
        assert_eq!(pragma_int(&mut db, "SELECT count(*) FROM t"), 0, "t empty after delete");
        exec(&mut db, "INSERT INTO t VALUES (1, 'x')");
        assert_eq!(pragma_int(&mut db, "SELECT count(*) FROM t"), 1, "insert works after reclaim");
    }
    assert_no_type0_entries(&tmp);
    {
        let mut db = tmp.open();
        assert_eq!(pragma_int(&mut db, "PRAGMA auto_vacuum"), 2, "INCREMENTAL persists across reopen");
        assert_eq!(pragma_int(&mut db, "SELECT count(*) FROM t"), 1, "row survives reopen after reclaim");
    }
}

// ===========================================================================
// PRAGMA auto_vacuum surface: keyword spellings, the GET form, and the
// "only before any table" enable gate (pragma.html "auto_vacuum").
// ===========================================================================

#[test]
fn auto_vacuum_pragma_keywords_get_form_and_enable_gate() {
    // Keyword `FULL` sets mode 1 (off52 != 0, off64 == 0) and GET reports 1.
    {
        let tmp = TempDb::new();
        let mut db = tmp.open();
        request_rollback_journal(&mut db);
        let _ = db.query("PRAGMA page_size = 4096");
        let _ = db.query("PRAGMA auto_vacuum = FULL");
        exec(&mut db, "CREATE TABLE t(a)");
        assert_eq!(pragma_int(&mut db, "PRAGMA auto_vacuum"), 1, "keyword FULL -> mode 1");
        drop(db);
        let raw = read_db(&tmp);
        assert!(be32(&raw, 52) != 0, "FULL: off 52 non-zero");
        assert_eq!(be32(&raw, 64), 0, "FULL: off 64 == 0");
    }

    // Keyword `INCREMENTAL` sets mode 2 (off64 == 1) and GET reports 2.
    {
        let tmp = TempDb::new();
        let mut db = tmp.open();
        request_rollback_journal(&mut db);
        let _ = db.query("PRAGMA page_size = 4096");
        let _ = db.query("PRAGMA auto_vacuum = INCREMENTAL");
        exec(&mut db, "CREATE TABLE t(a)");
        assert_eq!(pragma_int(&mut db, "PRAGMA auto_vacuum"), 2, "keyword INCREMENTAL -> mode 2");
        drop(db);
        let raw = read_db(&tmp);
        assert!(be32(&raw, 52) != 0, "INCREMENTAL: off 52 non-zero");
        assert_eq!(be32(&raw, 64), 1, "INCREMENTAL: off 64 == 1");
    }

    // The enable gate: auto_vacuum can only be turned on before any table exists. Enabling
    // it AFTER a table is created is a silent no-op (the db stays NONE, off52 == 0).
    {
        let tmp = TempDb::new();
        let mut db = tmp.open();
        request_rollback_journal(&mut db);
        let _ = db.query("PRAGMA page_size = 4096");
        exec(&mut db, "CREATE TABLE t(a)"); // a table exists first
        let _ = db.query("PRAGMA auto_vacuum = 1"); // too late
        assert_eq!(pragma_int(&mut db, "PRAGMA auto_vacuum"), 0, "enabling after a table is a no-op");
        drop(db);
        let raw = read_db(&tmp);
        assert_eq!(be32(&raw, 52), 0, "off 52 stays 0 when enable is gated out");
    }
}
