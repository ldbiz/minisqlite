//! Conformance battery: the **auto_vacuum / incremental_vacuum RECLAMATION side** —
//! relocating live pages into freed slots and truncating the file, the riskiest half of
//! the vacuum machinery (fileformat2 §1.8 pointer-map + pragma.html "auto_vacuum" /
//! "incremental_vacuum").
//!
//! `conformance_auto_vacuum.rs` already pins the WRITE side (ptrmap positions/entries)
//! and the *simple* reclaim outcomes: a delete-ALL FULL compaction, a table-ROOT
//! relocation, an INCREMENTAL keep-then-`incremental_vacuum(3)` where the freed pages sit
//! at the TAIL (a pure truncation), the overflow-chain recording, and the pragma
//! keyword surface. This file drives the paths those leave UNTESTED — the ones where a
//! relocation bug corrupts data rather than merely declining:
//!
//!   1. FULL compaction that relocates OVERFLOW pages (rewrite_overflow_head + _next).
//!   2. FULL compaction that relocates SECONDARY-INDEX b-tree pages.
//!   3. Interior (mid-file) relocation of non-root pages (tail live pages move DOWN).
//!   4. `incremental_vacuum(N)` exactness when RELOCATION (not pure truncation) is needed.
//!   5. NONE-mode byte stability: the default keeps freed pages, never shrinks.
//!   6. WAL-mode compaction + checkpoint: on-disk shrink, accepting a clean DECLINE.
//!   7. Crash mid-compacting-commit: recovery is old-larger-or-new-smaller, never torn.
//!
//! ## Discipline
//! Every expected value is derived from the SQLite documentation or from row data this
//! test itself wrote through real SQL — NEVER read back from the engine and compared to
//! itself. Row values are DETERMINISTIC per id, and the assertions are VALUE-exact
//! (never a bare "no error" or a bare count), so a single relocated-and-corrupted page
//! is caught and reproducible. A failing case is the intended signal of a real
//! relocation/recovery bug and must not be weakened to pass. Everything is validated
//! through `minisqlite::Connection` plus filesystem inspection.
//!
//! Spec sources:
//! - `spec/sqlite-doc/fileformat2.html` §1.8 (ptrmap page positions + 5-byte entries:
//!   types 1=rootpage, 2=freepage, 3=overflow-first, 4=overflow-rest, 5=non-root btree),
//!   §1.5 (freelist), §1.6 (overflow chains), §1.3.12 (offset 52 / offset 64 flags).
//! - `spec/sqlite-doc/pragma.html` "auto_vacuum" (FULL truncates the freelist off the
//!   file at every commit; INCREMENTAL keeps it), "incremental_vacuum" (reclaim up to N),
//!   #pragma_journal_mode / #pragma_wal_checkpoint.
//! - `spec/sqlite-doc/atomiccommit.html` + §3 of fileformat2 (hot-journal playback
//!   restores the pre-transaction image; a commit is the journal's invalidation).

mod conformance;
use conformance::*;

use minisqlite::{Connection, Value};
use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Temp-file harness: a unique on-disk path per test, cleaned up even on panic.
// (Same shape as the `TempDb` in the sibling on-disk test files; kept local
// rather than editing the shared `conformance` harness seam.)
// ---------------------------------------------------------------------------

/// An RAII guard over a unique temporary database path. `Drop` deletes the database AND
/// its `-journal` / `-wal` / `-shm` sidecars, so nothing is left behind even when an
/// assertion panics. No `.db` file is ever committed.
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
        path.push(format!("minisqlite_avr_{pid}_{n}_{nanos}.db"));
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

    fn journal(&self) -> PathBuf {
        self.sidecar("-journal")
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
// Byte + §1.8 layout helpers (transcribed from the spec, not imported from the
// engine — so the assertions cannot form a closed loop with the code under test).
// Every multibyte header/ptrmap field is BIG-ENDIAN.
// ---------------------------------------------------------------------------

const HEADER_SIZE: usize = 100;

/// One ptrmap entry is 5 bytes: `[1-byte type][4-byte big-endian parent]` (§1.8).
const PTRMAP_ENTRY_SIZE: usize = 5;

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

/// The ptrmap page carrying data page `p`'s entry (`B = 2 + ((p-2)/stride)*stride`).
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
/// from the MAIN db file rather than a lagging `-wal` sidecar. The result is ignored on
/// purpose: setup must not hide the check.
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

/// Run a single-row, single-column integer PRAGMA / scalar query and return its value.
fn pragma_int(db: &mut Connection, sql: &str) -> i64 {
    let qr = query(db, sql);
    assert_eq!(qr.rows.len(), 1, "{sql} must return exactly one row");
    as_int(&qr.rows[0][0])
}

/// The on-disk size of the main database file, in bytes (0 if it does not exist yet).
fn file_len(tmp: &TempDb) -> u64 {
    std::fs::metadata(&tmp.path).map(|m| m.len()).unwrap_or(0)
}

/// Two result-row sets are equal (`Value` has no `PartialEq`, so compare via the shared
/// `value_eq`). Used to cross-check an index-driven query against the table (rowid) path.
fn same_rows(a: &[Vec<Value>], b: &[Vec<Value>]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(r1, r2)| {
            r1.len() == r2.len() && r1.iter().zip(r2).all(|(x, y)| value_eq(x, y))
        })
}

/// Run `PRAGMA wal_checkpoint` and assert the documented one-row `(busy, log,
/// checkpointed)` shape with `busy == 0` (pragma.html #pragma_wal_checkpoint) — so a
/// file-length measured after it reflects a COMPLETED drain, not one skipped because the
/// checkpoint was busy.
fn checkpoint(db: &mut Connection) {
    let qr = query(db, "PRAGMA wal_checkpoint");
    assert_eq!(qr.rows.len(), 1, "wal_checkpoint returns exactly one row");
    assert_eq!(qr.rows[0].len(), 3, "wal_checkpoint returns three columns (busy, log, checkpointed)");
    assert_eq!(as_int(&qr.rows[0][0]), 0, "wal_checkpoint must report busy == 0");
}

/// Assert every ptrmap-covered data page in the file carries a valid §1.8 entry type
/// (1..=5) — never a type-0, which is not a real entry type and would make the file
/// unreadable to real sqlite. The structural-validity check any compacted image must pass.
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

// ---------------------------------------------------------------------------
// Deterministic, literal-safe row values ([A-Za-z0-9-] only, so each embeds
// directly in a single-quoted SQL literal). The hashed suffix varies per id, so a
// swapped / duplicated / corrupted row fails the value-exact assertion.
// ---------------------------------------------------------------------------

fn v_for(id: i64) -> String {
    let mut s = (id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let out = format!("v{id:05}-{:012x}", s >> 16);
    debug_assert!(
        out.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-'),
        "v_for must stay literal-safe: {out:?}"
    );
    out
}

/// A wide (~68 byte) deterministic value: several rows per 512-byte page, so a few
/// hundred rows span many pages and a delete leaves reclaimable free pages.
fn wide_for(id: i64) -> String {
    format!("{}-{}", v_for(id), "p".repeat(48))
}

/// A large (~2 KB) deterministic value that overflows a 512-byte page onto a multi-page
/// overflow chain (§1.6): one OVERFLOW1 page + several OVERFLOW2 pages per row.
fn big_for(id: i64) -> String {
    format!("{}-{}", v_for(id), "y".repeat(2000))
}

// Interior-hole workload (scenarios 3/4/5): insert ids 1..=N, then delete the CONTIGUOUS
// MIDDLE block `MID_DEL_LO..=MID_DEL_HI`. A contiguous run empties whole middle leaf
// pages (they are freed), leaving live pages BOTH below (the low block) and above (the
// high block) the holes — genuine interior free pages, unlike deleting every other row
// which leaves each leaf ~half-full (below the merge threshold, so nothing is freed).
const MID_DEL_LO: i64 = 51;
const MID_DEL_HI: i64 = 150;

/// The surviving `(id, v)` rows after the middle-block delete on ids 1..=N, value-exact
/// and id-ordered: the low block (`< MID_DEL_LO`) then the high block (`> MID_DEL_HI`).
fn middle_delete_survivors(n: i64) -> Vec<Vec<Value>> {
    (1..=n)
        .filter(|id| *id < MID_DEL_LO || *id > MID_DEL_HI)
        .map(|id| vec![int(id), text(&wide_for(id))])
        .collect()
}

/// Insert `id in lo..=hi` into a two-column table via `value_of`, batched so no single
/// INSERT is unreasonably large. Wrapped in one transaction for speed.
fn insert_ids(
    db: &mut Connection,
    table_and_cols: &str,
    lo: i64,
    hi: i64,
    batch: i64,
    value_of: impl Fn(i64) -> String,
) {
    exec(db, "BEGIN");
    let mut start = lo;
    while start <= hi {
        let end = (start + batch - 1).min(hi);
        let tuples: Vec<String> =
            (start..=end).map(|id| format!("({id}, '{}')", value_of(id))).collect();
        exec(db, &format!("INSERT INTO {table_and_cols} VALUES {}", tuples.join(", ")));
        start = end + 1;
    }
    exec(db, "COMMIT");
}

// ===========================================================================
// 1 — FULL compaction that RELOCATES OVERFLOW pages.
//
// A ~2 KB value on a 512-byte page spills onto an overflow chain (OVERFLOW1 +
// several OVERFLOW2). Inserting a run of rows then deleting the LOW half leaves
// every surviving row's overflow pages ABOVE a sea of freed slots, so the DELETE's
// FULL-compaction commit must relocate them DOWN — exercising `rewrite_overflow_head`
// (the OVERFLOW1 pointer in the b-tree leaf cell) and `rewrite_overflow_next` (the
// next-page pointer inside a chain). A bug in either loses or scrambles the value.
// ===========================================================================

#[test]
fn full_compaction_relocates_overflow_pages_verbatim() {
    const N: i64 = 20;
    const KEEP_FROM: i64 = 11; // delete ids 1..=10, keep 11..=20
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        request_rollback_journal(&mut db);
        let _ = db.query("PRAGMA page_size = 512");
        let _ = db.query("PRAGMA auto_vacuum = 1"); // FULL
        exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, big TEXT)");
        insert_ids(&mut db, "t(id, big)", 1, N, 4, big_for);

        // Insert-only FULL has nothing to reclaim yet.
        assert_eq!(pragma_int(&mut db, "PRAGMA freelist_count"), 0, "insert-only FULL: no free pages");
        let pages_before = pragma_int(&mut db, "PRAGMA page_count");
        assert!(pages_before > 40, "20 overflowing rows must span many 512B pages; got {pages_before}");

        // Delete the low half: their (low) overflow pages are freed, so every surviving
        // row's (higher) overflow pages sit above the holes and MUST relocate at commit.
        exec(&mut db, &format!("DELETE FROM t WHERE id < {KEEP_FROM}"));
        assert_eq!(
            pragma_int(&mut db, "PRAGMA freelist_count"),
            0,
            "FULL compaction relocates the surviving overflow pages and truncates to zero free"
        );
        let pages_after = pragma_int(&mut db, "PRAGMA page_count");
        assert!(
            pages_after < pages_before,
            "FULL must shrink after freeing half the overflow pages: {pages_after} !< {pages_before}"
        );

        // Every surviving overflowing value round-trips VERBATIM (exact length + bytes),
        // proving the relocated OVERFLOW1/OVERFLOW2 pointers still walk the whole chain.
        let expected: Vec<Vec<Value>> = (KEEP_FROM..=N)
            .map(|id| {
                let s = big_for(id);
                vec![int(id), int(s.len() as i64), text(&s)]
            })
            .collect();
        assert_rows(&mut db, "SELECT id, length(big), big FROM t ORDER BY id", &expected);
        assert_scalar(&mut db, "SELECT count(*) FROM t", int(N - KEEP_FROM + 1));
    }

    // The compacted file is structurally valid and reopens with every survivor verbatim.
    assert_no_type0_entries(&tmp);
    {
        let mut db = tmp.open();
        assert_eq!(pragma_int(&mut db, "PRAGMA freelist_count"), 0, "still zero free pages after reopen");
        let expected: Vec<Vec<Value>> = (KEEP_FROM..=N)
            .map(|id| {
                let s = big_for(id);
                vec![int(id), int(s.len() as i64), text(&s)]
            })
            .collect();
        assert_rows(&mut db, "SELECT id, length(big), big FROM t ORDER BY id", &expected);
    }
}

// ===========================================================================
// 2 — FULL compaction that RELOCATES SECONDARY-INDEX pages.
//
// A separate INDEX b-tree spans several pages once enough rows exist. Deleting a
// large low range frees index + table pages; FULL compaction at that commit must
// relocate the surviving INDEX pages (type-5 non-root b-tree) into freed slots and
// rewrite their parent pointers. If a relocated index page's pointer is wrong, an
// index-driven lookup returns the wrong rows — so we compare the index path
// (`WHERE b = ?` / `BETWEEN`) against the table path (`WHERE a = ?`), both of which
// must equal the spec-correct answer, before AND after a reopen.
// ===========================================================================

#[test]
fn full_compaction_relocates_secondary_index_pages() {
    const N: i64 = 300; // a = 0..=299, b = 'v0000'..'v0299'
    const KEEP_FROM: i64 = 200; // delete a < 200, keep 200..=299
    let tmp = TempDb::new();

    let index_eq = |db: &mut Connection, key: &str| query(db, &format!("SELECT a FROM t WHERE b = '{key}' ORDER BY a"));
    let table_eq = |db: &mut Connection, a: i64| query(db, &format!("SELECT a FROM t WHERE a = {a} ORDER BY a"));

    {
        let mut db = tmp.open();
        request_rollback_journal(&mut db);
        let _ = db.query("PRAGMA page_size = 512");
        let _ = db.query("PRAGMA auto_vacuum = 1"); // FULL
        exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
        exec(&mut db, "CREATE INDEX ix ON t(b)");
        insert_ids(&mut db, "t(a, b)", 0, N - 1, 50, |a| format!("v{a:04}"));

        let pages_before = pragma_int(&mut db, "PRAGMA page_count");
        assert!(pages_before > 20, "300 rows + index must span many 512B pages; got {pages_before}");

        exec(&mut db, &format!("DELETE FROM t WHERE a < {KEEP_FROM}"));
        assert_eq!(
            pragma_int(&mut db, "PRAGMA freelist_count"),
            0,
            "FULL compaction relocates surviving index pages and truncates to zero free"
        );
        assert!(
            pragma_int(&mut db, "PRAGMA page_count") < pages_before,
            "FULL must shrink the page count after deleting two thirds of the rows"
        );

        // Same connection (the catalog was reloaded if a root moved): the index still
        // resolves an equality AND a range, and agrees with the table (rowid) path.
        assert_rows(&mut db, "SELECT a FROM t WHERE b = 'v0250' ORDER BY a", &[vec![int(250)]]);
        assert!(
            same_rows(&index_eq(&mut db, "v0250").rows, &table_eq(&mut db, 250).rows),
            "index path must equal the table (rowid) path for the eq lookup"
        );
        let range: Vec<Vec<Value>> = (250..=259).map(|a| vec![int(a)]).collect();
        assert_rows(&mut db, "SELECT a FROM t WHERE b BETWEEN 'v0250' AND 'v0259' ORDER BY a", &range);
        assert_rows(&mut db, "SELECT a FROM t WHERE a BETWEEN 250 AND 259 ORDER BY a", &range);
        // A deleted key is gone from the index (not a stale pointer into a relocated page).
        assert!(index_eq(&mut db, "v0100").rows.is_empty(), "a deleted key is absent from the index");
        assert_scalar(&mut db, "SELECT count(*) FROM t", int(N - KEEP_FROM));
    }

    // A fresh connection re-reads the schema and index roots from the compacted page 1.
    assert_no_type0_entries(&tmp);
    {
        let mut db = tmp.open();
        assert_rows(&mut db, "SELECT a FROM t WHERE b = 'v0299' ORDER BY a", &[vec![int(299)]]);
        assert!(
            same_rows(&index_eq(&mut db, "v0299").rows, &table_eq(&mut db, 299).rows),
            "index path must equal the table (rowid) path after reopen"
        );
        let range: Vec<Vec<Value>> = (295..=299).map(|a| vec![int(a)]).collect();
        assert_rows(&mut db, "SELECT a FROM t WHERE b BETWEEN 'v0295' AND 'v0299' ORDER BY a", &range);
        assert_scalar(&mut db, "SELECT count(*) FROM t", int(N - KEEP_FROM));
    }
}

// ===========================================================================
// 3 — Interior (mid-file) relocation of NON-root pages.
//
// Deleting a CONTIGUOUS MIDDLE block empties whole middle leaf pages, freeing them
// while live pages remain in the low AND high blocks — so the freed pages sit in the
// MIDDLE of the file (not just at the tail), and FULL compaction must relocate the
// high live TAIL pages DOWN into those interior holes (the general relocation path,
// not a tail truncation). All surviving rows must return value-exact after the shuffle.
// ===========================================================================

#[test]
fn full_compaction_relocates_interior_free_pages() {
    const N: i64 = 200; // id 1..=200; delete the middle block (51..=150), keep the 100 low+high rows
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        request_rollback_journal(&mut db);
        let _ = db.query("PRAGMA page_size = 512");
        let _ = db.query("PRAGMA auto_vacuum = 1"); // FULL
        exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)");
        insert_ids(&mut db, "t(id, v)", 1, N, 40, wide_for);
        let pages_before = pragma_int(&mut db, "PRAGMA page_count");
        assert!(pages_before > 20, "200 wide rows must span many pages; got {pages_before}");

        // Empty the contiguous MIDDLE block: those whole leaf pages are freed, leaving
        // interior holes with live pages above them. FULL compaction must relocate the
        // high live pages DOWN into the interior holes at the commit.
        exec(&mut db, &format!("DELETE FROM t WHERE id BETWEEN {MID_DEL_LO} AND {MID_DEL_HI}"));
        assert_eq!(
            pragma_int(&mut db, "PRAGMA freelist_count"),
            0,
            "FULL compaction with interior holes must reach zero free pages"
        );
        assert!(
            pragma_int(&mut db, "PRAGMA page_count") < pages_before,
            "FULL must shrink after freeing the middle block of pages"
        );

        let expected = middle_delete_survivors(N);
        let kept = expected.len() as i64;
        assert_rows(&mut db, "SELECT id, v FROM t ORDER BY id", &expected);
        assert_scalar(&mut db, "SELECT count(*) FROM t", int(kept));
    }

    assert_no_type0_entries(&tmp);
    {
        let mut db = tmp.open();
        assert_rows(&mut db, "SELECT id, v FROM t ORDER BY id", &middle_delete_survivors(N));
        assert_eq!(pragma_int(&mut db, "PRAGMA freelist_count"), 0, "still zero free pages after reopen");
    }
}

// ===========================================================================
// 4 — `incremental_vacuum(N)` exactness WHEN RELOCATION is required.
//
// The existing incremental test deletes the WHOLE table, so its freed pages sit at
// the tail and `incremental_vacuum(N)` is a pure truncation. Here we delete a
// CONTIGUOUS MIDDLE block (interior holes with live low+high blocks above them), so
// honoring the budget forces the vacuum to RELOCATE live tail pages into holes. Each
// reclaim step removes exactly one page from the file and drops the rebuilt freelist by
// exactly one (a dropped free tail page, or a hole consumed by a relocated live page),
// so a budget of N drops `freelist_count` by EXACTLY N and shrinks `page_count`; a final
// no-arg call clears the rest to zero. Data stays intact throughout.
// ===========================================================================

#[test]
fn incremental_vacuum_budget_is_exact_with_relocation() {
    const N: i64 = 200;
    const BUDGET: i64 = 3;
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        request_rollback_journal(&mut db);
        let _ = db.query("PRAGMA page_size = 512");
        let _ = db.query("PRAGMA auto_vacuum = 2"); // INCREMENTAL
        assert_eq!(pragma_int(&mut db, "PRAGMA auto_vacuum"), 2, "auto_vacuum GET reports INCREMENTAL");
        exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)");
        insert_ids(&mut db, "t(id, v)", 1, N, 40, wide_for);

        // INCREMENTAL keeps freed pages on the freelist at the DELETE's commit. Emptying
        // the contiguous MIDDLE block frees interior pages while leaving live pages above
        // them, so honoring the budget forces the vacuum to RELOCATE, not just truncate.
        exec(&mut db, &format!("DELETE FROM t WHERE id BETWEEN {MID_DEL_LO} AND {MID_DEL_HI}"));
        let fc0 = pragma_int(&mut db, "PRAGMA freelist_count");
        assert!(fc0 > BUDGET, "INCREMENTAL keeps freed pages (need > {BUDGET} to test a budget); got {fc0}");
        let pc0 = pragma_int(&mut db, "PRAGMA page_count");

        // A budgeted reclaim drops the freelist by EXACTLY the budget and shrinks the file.
        exec(&mut db, &format!("PRAGMA incremental_vacuum({BUDGET})"));
        let fc1 = pragma_int(&mut db, "PRAGMA freelist_count");
        assert_eq!(fc1, fc0 - BUDGET, "incremental_vacuum({BUDGET}) drops freelist by exactly {BUDGET}: {fc0} -> {fc1}");
        let pc1 = pragma_int(&mut db, "PRAGMA page_count");
        assert_eq!(pc1, pc0 - BUDGET, "page_count drops by exactly the reclaimed count: {pc0} -> {pc1}");

        // Data is untouched by the partial reclaim.
        let expected = middle_delete_survivors(N);
        assert_rows(&mut db, "SELECT id, v FROM t ORDER BY id", &expected);

        // A no-arg call clears the ENTIRE remaining freelist.
        exec(&mut db, "PRAGMA incremental_vacuum");
        assert_eq!(pragma_int(&mut db, "PRAGMA freelist_count"), 0, "no-arg incremental_vacuum clears the rest");
        assert!(pragma_int(&mut db, "PRAGMA page_count") < pc1, "page_count drops again clearing the rest");
        assert_rows(&mut db, "SELECT id, v FROM t ORDER BY id", &expected);
    }

    assert_no_type0_entries(&tmp);
    {
        let mut db = tmp.open();
        assert_eq!(pragma_int(&mut db, "PRAGMA auto_vacuum"), 2, "INCREMENTAL persists across reopen");
        assert_eq!(pragma_int(&mut db, "PRAGMA freelist_count"), 0, "freelist still empty after reopen");
        assert_rows(&mut db, "SELECT id, v FROM t ORDER BY id", &middle_delete_survivors(N));
    }
}

// ===========================================================================
// 5 — NONE-mode (the DEFAULT) byte stability: freed pages STAY, nothing shrinks.
//
// The exact same delete-half workload as #3/#4, but with auto_vacuum=0 (the default).
// SQLite's NONE mode never relocates or truncates: freed pages go to the freelist and
// the file keeps its size (pragma.html "auto_vacuum" = NONE). So `freelist_count` is
// non-zero, `page_count` and the on-disk file length are UNCHANGED by the delete, and
// `PRAGMA incremental_vacuum` is a NO-OP (it does nothing outside incremental mode).
// This proves the reclaim machinery activates ONLY for auto_vacuum databases.
// ===========================================================================

#[test]
fn none_mode_keeps_freed_pages_and_incremental_vacuum_is_a_noop() {
    const N: i64 = 200;
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        request_rollback_journal(&mut db);
        let _ = db.query("PRAGMA page_size = 512");
        // auto_vacuum defaults to 0 (NONE); assert that rather than assume it.
        assert_eq!(pragma_int(&mut db, "PRAGMA auto_vacuum"), 0, "default auto_vacuum is NONE (0)");
        exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)");
        insert_ids(&mut db, "t(id, v)", 1, N, 40, wide_for);

        assert_eq!(pragma_int(&mut db, "PRAGMA freelist_count"), 0, "insert-only: no free pages yet");
        let pc_before = pragma_int(&mut db, "PRAGMA page_count");
        let len_before = file_len(&tmp);

        // The same middle-block delete that frees interior pages in the vacuum scenarios.
        exec(&mut db, &format!("DELETE FROM t WHERE id BETWEEN {MID_DEL_LO} AND {MID_DEL_HI}"));
        // NONE keeps the freed pages on the freelist and does not shrink the file.
        let fc = pragma_int(&mut db, "PRAGMA freelist_count");
        assert!(fc > 0, "NONE mode must keep freed pages on the freelist; got {fc}");
        assert_eq!(
            pragma_int(&mut db, "PRAGMA page_count"),
            pc_before,
            "NONE mode never truncates: page_count is unchanged by a delete"
        );
        assert_eq!(file_len(&tmp), len_before, "NONE mode never shrinks the on-disk file");

        // `incremental_vacuum` is a documented no-op outside incremental mode: freelist
        // and page_count are untouched (not an error, not a compaction).
        let _ = db.query("PRAGMA incremental_vacuum");
        assert_eq!(pragma_int(&mut db, "PRAGMA freelist_count"), fc, "incremental_vacuum is a no-op in NONE mode");
        assert_eq!(pragma_int(&mut db, "PRAGMA page_count"), pc_before, "page_count unchanged by the no-op vacuum");

        // The survivors are intact after the delete + the no-op vacuum.
        assert_rows(&mut db, "SELECT id, v FROM t ORDER BY id", &middle_delete_survivors(N));
    }
    {
        let mut db = tmp.open();
        assert_eq!(pragma_int(&mut db, "PRAGMA auto_vacuum"), 0, "still NONE after reopen");
        assert_rows(&mut db, "SELECT id, v FROM t ORDER BY id", &middle_delete_survivors(N));
    }
}

// ===========================================================================
// 6 — WAL-mode compaction + checkpoint: on-disk shrink, DECLINE accepted.
//
// A FULL delete-then-compact workload in WAL journal mode. A WAL commit records the
// new (smaller) db size in its commit frame; a checkpoint folds the frames into the
// database file and truncates it to that logical size. So after the compacting commit
// + `PRAGMA wal_checkpoint` the on-disk file should shrink AND the data round-trips.
//
// WAL-mode compaction is subtle; by design the reclaim path DECLINES (leaves the valid
// uncompacted file) on any inconsistency rather than risk corruption. Both outcomes
// are a PASS — (a) COMPACTED: freelist_count == 0 and the file shrank; (b) DECLINED:
// freelist_count > 0 and the file did not shrink — but the data must ALWAYS be intact,
// the file valid, and a reopen must round-trip. What is NEVER acceptable is a lost or
// wrong row or a torn file. (See the assertion at the branch for which outcome held.)
// ===========================================================================

#[test]
fn wal_mode_compaction_shrinks_on_checkpoint_or_declines_cleanly() {
    const N: i64 = 200;
    let tmp = TempDb::new();

    // Enter WAL mode + FULL auto_vacuum on the fresh db from a throwaway connection.
    {
        let mut db = tmp.open();
        let _ = db.query("PRAGMA page_size = 512");
        let _ = db.query("PRAGMA auto_vacuum = 1"); // FULL, before any table
        let qr = query(&mut db, "PRAGMA journal_mode = WAL");
        assert_eq!(as_text(&qr.rows[0][0]), "wal", "journal_mode=WAL must report wal");
    }

    let (len_full, fc_after, pc_after);
    {
        let mut db = tmp.open();
        assert_eq!(pragma_int(&mut db, "PRAGMA auto_vacuum"), 1, "FULL survives the WAL switch");
        exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)");
        insert_ids(&mut db, "t(id, v)", 1, N, 40, wide_for);

        // Checkpoint the insert-only state so the MAIN db file holds the full image; that
        // is the size a later compaction must shrink below.
        checkpoint(&mut db);
        len_full = file_len(&tmp);
        let pc_full = pragma_int(&mut db, "PRAGMA page_count");
        assert!(pc_full > 20, "200 wide rows must span many pages; got {pc_full}");
        assert!(len_full >= (pc_full as u64) * 512, "checkpoint drained the full image to the db file");

        // Delete everything (FULL compaction attempts at the WAL commit), then keep one
        // sentinel so "data intact" is a concrete value, not just an empty table.
        exec(&mut db, "DELETE FROM t");
        exec(&mut db, "INSERT INTO t VALUES (777, 'sentinel')");

        fc_after = pragma_int(&mut db, "PRAGMA freelist_count");
        pc_after = pragma_int(&mut db, "PRAGMA page_count");

        // Fold the WAL back into the db file (drains + truncates to the logical size).
        checkpoint(&mut db);
        assert_scalar(&mut db, "SELECT count(*) FROM t", int(1));
        assert_rows(&mut db, "SELECT id, v FROM t", &[vec![int(777), text("sentinel")]]);
    }

    let len_after = file_len(&tmp);
    // Structural validity and no-growth hold for BOTH outcomes: the on-disk image after
    // the checkpoint carries only valid §1.8 ptrmap entries (freed pages are type-2, live
    // pages 1/3/4/5 — never type-0), and neither a compaction nor a decline may grow the
    // file past the full image.
    assert!(len_after <= len_full, "the compacting/declining commit must not grow the file: {len_after} !<= {len_full}");
    assert_no_type0_entries(&tmp);

    // This scenario accepts BOTH compaction and a clean decline, so its `freelist_count == 0`
    // branch is not a firm requirement. The FIRM "freelist_count == 0 under WAL" guard lives in
    // `conformance_auto_vacuum.rs::full_auto_vacuum_under_wal_truncates_and_round_trips` — keep
    // that one, so this permissive test never silently becomes the sole WAL-compaction check.
    if fc_after == 0 {
        // (a) COMPACTED: the logical db shrank, so the checkpoint truncated the db file
        // strictly below the full image.
        assert!(pc_after > 0, "a compacted db still has its live pages");
        assert!(
            len_after < len_full,
            "WAL FULL compaction: after checkpoint the db file must shrink ({len_after} !< {len_full})"
        );
    } else {
        // (b) DECLINED (an accepted outcome): the reclaim kept a valid, larger file rather
        // than risk corruption. The logical page count never dropped, so the checkpoint
        // left the db file at the FULL image size — it did not shrink.
        assert_eq!(len_after, len_full, "declined: page count unchanged, so the file did not shrink");
    }

    // Whichever outcome held, a fresh reopen round-trips exactly the sentinel row.
    {
        let mut db = tmp.open();
        assert_scalar(&mut db, "SELECT count(*) FROM t", int(1));
        assert_rows(&mut db, "SELECT id, v FROM t", &[vec![int(777), text("sentinel")]]);
    }
}

// ===========================================================================
// 7 — Hot-journal REPLAY restores an auto_vacuum image (the "old-larger" consistent
//     outcome of an interrupted compacting DELETE).
//
// A crash mid-commit of a FULL-compacting DELETE (rollback-journal mode) leaves a hot
// `-journal` — the pre-images of the pages the commit was about to relocate/free — plus
// a partially rewritten, smaller database. Recovery on reopen must replay that journal
// and restore the OLD, larger, pre-delete image EXACTLY: a consistent state, never a
// torn mix of old and new pages.
//
// HONEST SCOPE: a test can neither kill a process nor make the facade leave a real hot
// journal (a clean commit removes it, and PERSIST/TRUNCATE alias to DELETE-style
// journaling here), so this FABRICATES the on-disk crash state directly — it captures
// the committed pre-delete image, synthesizes a hot journal from fileformat2 §3 wrapping
// the REAL pre-delete pages, and scribbles + truncates the db to stand in for the
// partially written image. It therefore pins the generic journal-REPLAY recovery path
// over an auto_vacuum image (a structurally-valid ptrmap + every row restored), NOT the
// compaction commit's OWN write-side journaling in `diskstore::apply_commit` (the code
// that journals the removed tail pages' pre-images before `set_len` down). That
// write-side path is exercised indirectly by the clean-reopen compaction tests #1/#3
// (the "new-smaller" consistent outcome). Both outcomes are self-consistent — which is
// the property under test — and a dedicated pager-internal shrink-commit crash test
// would pin the write-side journaling directly.
// ===========================================================================

/// The 8-byte rollback-journal magic (fileformat2 §3, "Header string").
const JOURNAL_MAGIC: [u8; 8] = [0xd9, 0xd5, 0x05, 0xf9, 0x20, 0xa1, 0x63, 0xd7];
/// The historical default sector size a journal header is padded to (records begin here).
const JOURNAL_SECTOR: u32 = 512;

/// The rollback-journal page-record checksum (fileformat2 §3): seed with the header
/// nonce, then add the byte at index N-200, N-400, … down to the first non-negative
/// index, as a wrapping u32 add.
fn journal_checksum(nonce: u32, page: &[u8]) -> u32 {
    let mut cksum = nonce;
    let mut x: isize = page.len() as isize - 200;
    while x >= 0 {
        cksum = cksum.wrapping_add(page[x as usize] as u32);
        x -= 200;
    }
    cksum
}

/// Encode a rollback journal: a sector-padded header (fileformat2 §3) then `records`
/// (`page_no(4) ++ content(page_size) ++ checksum(4)`). `initial_db_pages` is the
/// pre-transaction page count recovery truncates/extends the file back to.
fn encode_journal(page_size: u32, nonce: u32, initial_db_pages: u32, records: &[(u32, Vec<u8>)]) -> Vec<u8> {
    let mut buf = vec![0u8; JOURNAL_SECTOR as usize];
    buf[0..8].copy_from_slice(&JOURNAL_MAGIC);
    buf[8..12].copy_from_slice(&(records.len() as u32).to_be_bytes());
    buf[12..16].copy_from_slice(&nonce.to_be_bytes());
    buf[16..20].copy_from_slice(&initial_db_pages.to_be_bytes());
    buf[20..24].copy_from_slice(&JOURNAL_SECTOR.to_be_bytes());
    buf[24..28].copy_from_slice(&page_size.to_be_bytes());
    for (page_no, content) in records {
        assert_eq!(content.len(), page_size as usize, "record content must be one page");
        buf.extend_from_slice(&page_no.to_be_bytes());
        buf.extend_from_slice(content);
        buf.extend_from_slice(&journal_checksum(nonce, content).to_be_bytes());
    }
    buf
}

/// Overwrite `bytes` at absolute `offset` in an existing file.
fn write_at(path: &Path, offset: u64, bytes: &[u8]) {
    let mut f = OpenOptions::new().read(true).write(true).open(path).unwrap();
    f.seek(SeekFrom::Start(offset)).unwrap();
    f.write_all(bytes).unwrap();
    f.sync_all().unwrap();
}

#[test]
fn hot_journal_replay_restores_an_auto_vacuum_image() {
    const N: i64 = 200;
    let tmp = TempDb::new();

    // A committed, insert-only FULL auto_vacuum baseline (DELETE mode: no journal left).
    {
        let mut db = tmp.open();
        request_rollback_journal(&mut db);
        let _ = db.query("PRAGMA page_size = 512");
        let _ = db.query("PRAGMA auto_vacuum = 1"); // FULL
        exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)");
        insert_ids(&mut db, "t(id, v)", 1, N, 40, wide_for);
        assert_scalar(&mut db, "SELECT count(*) FROM t", int(N));
        assert_eq!(pragma_int(&mut db, "PRAGMA freelist_count"), 0, "insert-only FULL: no free pages");
    }
    assert!(!tmp.journal().exists(), "a clean DELETE-mode commit leaves no journal");

    // The committed pre-delete image is the larger state a rollback must reproduce.
    let committed = read_db(&tmp);
    let page_size = effective_page_size(&committed);
    assert_eq!(committed.len() % page_size as usize, 0, "db length is a whole number of pages");
    let count = (committed.len() / page_size as usize) as u32;
    assert!(count >= 3 && be32(&committed, 52) != 0, "several pages and an auto_vacuum header (off52 != 0)");

    // Fabricate the on-disk crash state (see the HONEST SCOPE note above): a hot journal
    // holding the pre-image of every page (standing in for the pages a real compacting
    // commit would journal before rewriting them), then a database scribbled and
    // truncated to a SMALLER size (a synthetic stand-in for the partially written image
    // a crash before the commit point would leave).
    let records: Vec<(u32, Vec<u8>)> = (1..=count)
        .map(|p| (p, committed[((p - 1) as usize) * page_size as usize..(p as usize) * page_size as usize].to_vec()))
        .collect();
    let journal = encode_journal(page_size, 0x5EED_AC17, count, &records);
    std::fs::write(tmp.journal(), &journal).unwrap();

    let scribble = vec![0xFFu8; page_size as usize];
    let shrunk_to = (count / 2).max(1);
    for p in 1..=shrunk_to {
        write_at(&tmp.path, (p as u64 - 1) * page_size as u64, &scribble);
    }
    {
        let f = OpenOptions::new().write(true).open(&tmp.path).unwrap();
        f.set_len(shrunk_to as u64 * page_size as u64).unwrap(); // synthetic partial shrink
        f.sync_all().unwrap();
    }
    assert!(file_len(&tmp) < committed.len() as u64, "the fabricated crash-state db is smaller than the committed image");

    // Reopen: recovery replays the journal, restoring the exact pre-delete image.
    {
        let mut db = tmp.open();
        let expected: Vec<Vec<Value>> = (1..=N).map(|id| vec![int(id), text(&wide_for(id))]).collect();
        assert_rows(&mut db, "SELECT id, v FROM t ORDER BY id", &expected);
        assert_scalar(&mut db, "SELECT count(*) FROM t", int(N));
    }
    assert_eq!(read_db(&tmp), committed, "recovery restored the exact pre-delete image (never a torn mix)");
    assert!(!tmp.journal().exists(), "the hot journal is removed once recovery completes");

    // The recovered auto_vacuum file is structurally valid and still writable.
    assert!(be32(&read_db(&tmp), 52) != 0, "recovered file is still an auto_vacuum db (off52 != 0)");
    assert_no_type0_entries(&tmp);
    {
        let mut db = tmp.open();
        exec(&mut db, "INSERT INTO t VALUES (9999, 'after-recovery')");
        assert_scalar(&mut db, "SELECT count(*) FROM t", int(N + 1));
    }
}
