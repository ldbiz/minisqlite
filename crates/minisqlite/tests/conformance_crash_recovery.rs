//! Conformance battery: **single-connection CRASH recovery** at the pinned facade
//! (`minisqlite::Connection`).
//!
//! The existing durability batteries (`conformance_ondisk_roundtrip.rs`,
//! `conformance_wal_durability.rs`) only exercise a *clean* close -> reopen. Real
//! crash recovery must additionally survive a CRASH mid-transaction and a reopen. Since a
//! test cannot kill a process, each case here instead:
//!
//!   1. forces committed state to disk through the public `Connection` API,
//!   2. fabricates the on-disk *crash state* by manipulating the `.db` / `-journal` /
//!      `-wal` / `-shm` files at the byte level (a hot rollback journal, a torn WAL
//!      frame, a half-done checkpoint, ...),
//!   3. reopens a fresh `Connection` and asserts the recovered database matches the
//!      spec-mandated outcome.
//!
//! ## Discipline
//! Every expected outcome is derived from the SQLite documentation, NEVER from
//! "what the engine currently returns". The committed row set — the ground truth a
//! recovery must reproduce — is established through real SQL, independent of the
//! engine's storage internals, so the assertions cannot form a closed loop with the
//! code under test. A case that reveals a recovery bug is left as a genuine failing
//! assertion rather than weakened to pass. Row values are DETERMINISTIC (an LCG per id)
//! and the assertions are VALUE-exact, so a single dropped / duplicated / corrupted row
//! is caught and reproducible — a bare row count is never sufficient.
//!
//! Real `sqlite3` is not used here: everything is validated through
//! `minisqlite::Connection` plus filesystem inspection.
//!
//! ## What is hand-crafted vs real
//! - Rollback journal (`-journal`): a clean commit is DELETE-mode, so it removes the
//!   journal — there is no way to leave a *hot* one through the public API. The journal
//!   FRAMING (header + per-page records + checksum) is therefore synthesized here
//!   directly from the byte format in `spec/sqlite-doc/fileformat2.html` §3, wrapping
//!   REAL committed page bytes read off the `.db` file. The b-tree content is genuine;
//!   only the journal metadata is synthesized, and the row assertions stay independent.
//! - WAL (`-wal`): commits are driven through the real engine (genuine frames land in
//!   the `-wal`), and only the *corruption* (a torn tail, a flipped byte, a half-done
//!   checkpoint) is applied at the byte level — the crash, not the data.
//!
//! Spec sources:
//! - `spec/sqlite-doc/fileformat2.html` §3 (rollback journal header + page record +
//!   checksum) and §4 (WAL header/frame + cumulative checksum + validity + reset).
//! - `spec/sqlite-doc/atomiccommit.html` (hot-journal playback restores the
//!   pre-transaction image; a commit is the journal's invalidation).
//! - `spec/sqlite-doc/walformat.html` (per-frame checksum + commit-marker semantics;
//!   frames after the last valid commit are ignored).
//! - `spec/sqlite-doc/pragma.html` #pragma_journal_mode / #pragma_wal_checkpoint.
//!
//! Each behavior is its own `#[test]`, so one discrepancy fails exactly that case.

mod conformance;
use conformance::*;

use minisqlite::{Connection, Value};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

// ===========================================================================
// Temp-file harness (a unique on-disk path per test, cleaned up even on panic).
// ===========================================================================

/// An RAII guard over a unique temporary database path. `Drop` removes the database
/// and every sidecar (`-journal`/`-wal`/`-shm`), so a panicking assertion never leaks
/// files. No `.db` is ever committed to the repo.
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
        path.push(format!("minisqlite_crash_{pid}_{n}_{nanos}.db"));
        let db = TempDb { path };
        db.remove_all();
        db
    }

    /// Open (or reopen) a file-backed connection, panicking with context on failure so
    /// the test body can use it unwrapped. A recovery bug that makes the reopen fail
    /// (e.g. a torn journal wrongly treated as a fatal error) surfaces loudly here.
    fn open(&self) -> Connection {
        let path: &Path = &self.path;
        Connection::open(path)
            .unwrap_or_else(|e| panic!("Connection::open failed for {path:?}: {e:?}"))
    }

    fn journal(&self) -> PathBuf {
        self.sidecar("-journal")
    }
    fn wal(&self) -> PathBuf {
        self.sidecar("-wal")
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

// ===========================================================================
// Deterministic row data.
// ===========================================================================

/// A deterministic, literal-safe TEXT value for row `id` (letters/digits/`-` only, so
/// it embeds directly in a single-quoted SQL literal). The hashed suffix varies per
/// id, so a swapped / duplicated / corrupted row fails the value-exact assertion.
fn val_for(id: i64) -> String {
    let mut state = (id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let v = format!("row-{id:05}-{:012x}", state >> 16);
    debug_assert!(
        v.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-'),
        "val_for must stay literal-safe ([A-Za-z0-9-] only): {v:?}"
    );
    v
}

/// The expected `SELECT id, v FROM t ORDER BY id` rows for ids `lo..=hi`.
fn expected_range(lo: i64, hi: i64) -> Vec<Vec<Value>> {
    (lo..=hi).map(|id| vec![int(id), text(&val_for(id))]).collect()
}

/// Insert ids `lo..=hi` into `t(id, v)` with deterministic values, batched so several
/// pages/frames are produced without any single statement being unreasonably large.
fn insert_range(db: &mut Connection, lo: i64, hi: i64) {
    const BATCH: i64 = 50;
    let mut start = lo;
    while start <= hi {
        let end = (start + BATCH - 1).min(hi);
        let tuples: Vec<String> =
            (start..=end).map(|id| format!("({id}, '{}')", val_for(id))).collect();
        exec(db, &format!("INSERT INTO t(id, v) VALUES {}", tuples.join(", ")));
        start = end + 1;
    }
}

// ===========================================================================
// Raw file helpers (byte-level crash simulation).
// ===========================================================================

fn read_file(path: &Path) -> Vec<u8> {
    let mut v = Vec::new();
    File::open(path)
        .unwrap_or_else(|e| panic!("open {path:?}: {e}"))
        .read_to_end(&mut v)
        .unwrap();
    v
}

fn file_len(path: &Path) -> u64 {
    match std::fs::metadata(path) {
        Ok(m) => m.len(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => 0,
        Err(e) => panic!("metadata({path:?}) failed unexpectedly: {e}"),
    }
}

/// Overwrite `bytes` at absolute `offset` in an existing file, extending it if needed.
fn write_at(path: &Path, offset: u64, bytes: &[u8]) {
    let mut f = OpenOptions::new().read(true).write(true).open(path).unwrap();
    f.seek(SeekFrom::Start(offset)).unwrap();
    f.write_all(bytes).unwrap();
    f.sync_all().unwrap();
}

/// Append `bytes` to the end of an existing file (used to simulate a torn WAL tail).
fn append(path: &Path, bytes: &[u8]) {
    let mut f = OpenOptions::new().append(true).open(path).unwrap();
    f.write_all(bytes).unwrap();
    f.sync_all().unwrap();
}

/// The database page size stored at header offset 16 (fileformat2 §1.3): a 2-byte
/// big-endian value, with the sentinel `1` meaning 65536. Read straight from the raw
/// `.db` bytes so the crafted journal uses the file's true page size, not an assumption.
fn db_page_size(db_bytes: &[u8]) -> u32 {
    let raw = u16::from_be_bytes([db_bytes[16], db_bytes[17]]);
    if raw == 1 { 65536 } else { raw as u32 }
}

// ===========================================================================
// Rollback-journal encoder — synthesized directly from fileformat2 §3.
// ===========================================================================

/// The 8-byte rollback-journal magic (fileformat2 §3, "Header string").
const JOURNAL_MAGIC: [u8; 8] = [0xd9, 0xd5, 0x05, 0xf9, 0x20, 0xa1, 0x63, 0xd7];

/// The historical default sector size a journal header is padded to. Records begin at
/// this offset (the header lives in its own sector, fileformat2 §3).
const JOURNAL_SECTOR: u32 = 512;

/// The rollback-journal page-record checksum, transcribed from fileformat2 §3:
/// initialize to the header nonce, then add the byte at index `X` for `X = N-200,
/// N-400, ...` down to the first non-negative index, as an unsigned 32-bit *wrapping*
/// add. Deliberately a sparse sample; the per-transaction nonce is what lets a torn or
/// stale record be detected.
fn journal_checksum(nonce: u32, page: &[u8]) -> u32 {
    let mut cksum = nonce;
    let mut x: isize = page.len() as isize - 200;
    while x >= 0 {
        cksum = cksum.wrapping_add(page[x as usize] as u32);
        x -= 200;
    }
    cksum
}

/// Encode a rollback journal: a sector-padded header (fileformat2 §3 header format)
/// followed by `records` (each `page_no(4) ++ content(page_size) ++ checksum(4)`).
///
/// `page_size_field_override` lets a test write a DELIBERATELY invalid page-size field
/// (e.g. a non-power-of-two) to exercise the "header not valid => journal ignored"
/// rule without disturbing the real content bytes.
fn encode_journal(
    page_size: u32,
    page_count_field: u32,
    nonce: u32,
    initial_db_pages: u32,
    page_size_field_override: Option<u32>,
    records: &[(u32, Vec<u8>)],
) -> Vec<u8> {
    let mut buf = vec![0u8; JOURNAL_SECTOR as usize];
    buf[0..8].copy_from_slice(&JOURNAL_MAGIC);
    buf[8..12].copy_from_slice(&page_count_field.to_be_bytes());
    buf[12..16].copy_from_slice(&nonce.to_be_bytes());
    buf[16..20].copy_from_slice(&initial_db_pages.to_be_bytes());
    buf[20..24].copy_from_slice(&JOURNAL_SECTOR.to_be_bytes());
    buf[24..28].copy_from_slice(&page_size_field_override.unwrap_or(page_size).to_be_bytes());
    for (page_no, content) in records {
        assert_eq!(content.len(), page_size as usize, "record content must be one page");
        buf.extend_from_slice(&page_no.to_be_bytes());
        buf.extend_from_slice(content);
        buf.extend_from_slice(&journal_checksum(nonce, content).to_be_bytes());
    }
    buf
}

/// Split a raw `.db` image into its 1-based pages (each `page_size` bytes).
fn split_pages(db_bytes: &[u8], page_size: u32) -> Vec<Vec<u8>> {
    let ps = page_size as usize;
    assert_eq!(db_bytes.len() % ps, 0, "db length must be a whole number of pages");
    db_bytes.chunks(ps).map(|c| c.to_vec()).collect()
}

/// Zero-pad `buf` up to the next `JOURNAL_SECTOR` boundary. A rollback journal may hold
/// several appended segments, each header aligned to a sector boundary; recovery finds
/// the next segment by rounding the previous segment's end up to the sector size
/// (fileformat2 §3, "a journal header ... always occurs at the beginning of a sector").
fn pad_to_sector(buf: &mut Vec<u8>) {
    let rem = buf.len() % JOURNAL_SECTOR as usize;
    if rem != 0 {
        buf.resize(buf.len() + (JOURNAL_SECTOR as usize - rem), 0);
    }
}

// ===========================================================================
// (a) A hot, COMPLETE rollback journal -> the interrupted txn is fully rolled back.
// ===========================================================================

/// Build a committed database, then fabricate the exact on-disk state a crash leaves
/// mid-commit (atomiccommit §3.5-§3.11): a hot `-journal` holding every touched page's
/// pre-image, the database pages overwritten with garbage, and the file grown by the
/// aborted transaction. Reopening must roll every page back to its pre-image and
/// truncate the file to the pre-transaction size, so the committed rows return exactly
/// and the aborted growth vanishes (atomiccommit §4).
#[test]
fn hot_journal_rolls_back_interrupted_transaction() {
    const N: i64 = 400;
    let tmp = TempDb::new();

    // A committed baseline through the real engine (rollback/DELETE mode: no journal).
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)");
        insert_range(&mut db, 1, N);
        assert_scalar(&mut db, "SELECT count(*) FROM t", int(N));
    }
    assert!(!tmp.journal().exists(), "a clean DELETE-mode commit leaves no journal");

    // The committed image is the ground truth a rollback must reproduce.
    let committed = read_file(&tmp.path);
    let page_size = db_page_size(&committed);
    let pages = split_pages(&committed, page_size);
    let count = pages.len() as u32;
    assert!(count >= 3, "expected several pages so a lost page is observable (got {count})");

    // Fabricate the crash: a hot journal with the pre-image of EVERY existing page
    // (initial_db_pages = the pre-transaction size), then overwrite every page with
    // garbage and grow the file by two pages — exactly what an aborted transaction
    // that had begun writing the database would leave behind.
    let records: Vec<(u32, Vec<u8>)> =
        (1..=count).map(|p| (p, pages[(p - 1) as usize].clone())).collect();
    let journal = encode_journal(page_size, count, 0x5EED_1234, count, None, &records);
    std::fs::write(tmp.journal(), &journal).unwrap();
    let garbage = vec![0xFFu8; page_size as usize];
    for p in 1..=count {
        write_at(&tmp.path, (p as u64 - 1) * page_size as u64, &garbage);
    }
    append(&tmp.path, &garbage); // aborted growth: page count+1
    append(&tmp.path, &garbage); // aborted growth: page count+2
    assert_eq!(
        file_len(&tmp.path),
        (count as u64 + 2) * page_size as u64,
        "the aborted transaction grew the file"
    );

    // Reopen: recovery restores every page and truncates the growth.
    {
        let mut c = tmp.open();
        assert_rows(&mut c, "SELECT id, v FROM t ORDER BY id", &expected_range(1, N));
        assert_scalar(&mut c, "SELECT count(*) FROM t", int(N));
    }
    assert_eq!(read_file(&tmp.path), committed, "database rolled back to the exact committed image");
    assert!(!tmp.journal().exists(), "the hot journal is removed once recovery completes");
}

// ===========================================================================
// (a2) A hot journal for a transaction that SHRANK the file -> recovery must
//      restore the freed pages, EXTENDING the file back to its pre-transaction size.
// ===========================================================================

/// The mirror of `hot_journal_rolls_back_interrupted_transaction`: an aborted
/// transaction that FREED pages (e.g. a `DROP`/`DELETE` that truncated the database
/// file to fewer pages) rather than growing it. The crash leaves the file SHORTER than
/// it was at transaction start, but the hot journal still holds the pre-image of every
/// page that existed then. Recovery must write those pre-images back — extending the
/// file past its crashed end — and restore the pre-transaction size, so every row
/// returns. This exercises the extend-on-restore path (writing pages beyond the
/// current EOF), which the growth case never hits (atomiccommit §4).
#[test]
fn hot_journal_restores_pages_freed_by_aborted_transaction() {
    const N: i64 = 400;
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)");
        insert_range(&mut db, 1, N);
    }
    let committed = read_file(&tmp.path);
    let page_size = db_page_size(&committed);
    let pages = split_pages(&committed, page_size);
    let count = pages.len() as u32;
    assert!(count >= 3, "need several pages so the freed-page restore is observable");

    // The journal holds the pre-image of every page that existed at transaction start.
    let records: Vec<(u32, Vec<u8>)> =
        (1..=count).map(|p| (p, pages[(p - 1) as usize].clone())).collect();
    let journal = encode_journal(page_size, count, 0x5EED_5217, count, None, &records);
    std::fs::write(tmp.journal(), &journal).unwrap();

    // The aborted transaction freed pages 2..: it scribbled page 1 (a new, smaller
    // header) and truncated the file down to a single page. Pages 2..count no longer
    // exist on disk — recovery must recreate them from the journal.
    write_at(&tmp.path, 0, &vec![0xFFu8; page_size as usize]);
    {
        let f = OpenOptions::new().write(true).open(&tmp.path).unwrap();
        f.set_len(page_size as u64).unwrap(); // shrink to one page
        f.sync_all().unwrap();
    }
    assert_eq!(file_len(&tmp.path), page_size as u64, "the aborted transaction shrank the file");

    {
        let mut c = tmp.open();
        assert_rows(&mut c, "SELECT id, v FROM t ORDER BY id", &expected_range(1, N));
        assert_scalar(&mut c, "SELECT count(*) FROM t", int(N));
    }
    assert_eq!(read_file(&tmp.path), committed, "freed pages restored; file extended to the pre-txn size");
    assert!(!tmp.journal().exists(), "the hot journal is removed once recovery completes");
}

// ===========================================================================
// (a3) A MULTI-SEGMENT hot journal -> every segment is replayed, each validated with
//      its OWN nonce.
// ===========================================================================

/// A rollback journal can hold more than one segment: when the page cache spills
/// mid-transaction, SQLite appends a fresh journal header (on the next sector boundary)
/// followed by more page records, each new segment carrying its own checksum nonce
/// (fileformat2 §3). Recovery must walk EVERY segment, not just the first, restoring all
/// of their pre-images. This crafts a two-segment journal — the pages split across the
/// segments under two DIFFERENT nonces — over a fully-scribbled database, and requires
/// the reopen to restore every page (so a recovery that stopped after segment one, or
/// validated segment two with the wrong nonce, would leave the later pages as garbage
/// and lose rows).
#[test]
fn multi_segment_hot_journal_replays_every_segment() {
    const N: i64 = 400;
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)");
        insert_range(&mut db, 1, N);
    }
    let committed = read_file(&tmp.path);
    let page_size = db_page_size(&committed);
    let pages = split_pages(&committed, page_size);
    let count = pages.len() as u32;
    // Need enough pages that BOTH segments carry row-bearing data pages, so dropping
    // either segment is observable as lost rows.
    assert!(count >= 4, "need >=4 pages so each segment holds data pages (got {count})");

    // Segment 1 restores pages 1..=k (nonce A); segment 2 restores pages k+1..=count
    // (nonce B). initial_db_pages is the pre-transaction size in BOTH headers (recovery
    // reads it from the first, but a faithful journal repeats it).
    let k = count / 2;
    let seg1_records: Vec<(u32, Vec<u8>)> =
        (1..=k).map(|p| (p, pages[(p - 1) as usize].clone())).collect();
    let seg2_records: Vec<(u32, Vec<u8>)> =
        (k + 1..=count).map(|p| (p, pages[(p - 1) as usize].clone())).collect();
    let mut journal = encode_journal(page_size, k, 0x5E6A_0001, count, None, &seg1_records);
    pad_to_sector(&mut journal); // the next segment header starts on a sector boundary
    let seg2 = encode_journal(page_size, count - k, 0x5E6A_0002, count, None, &seg2_records);
    journal.extend_from_slice(&seg2);
    std::fs::write(tmp.journal(), &journal).unwrap();

    // Scribble EVERY page: only replaying both segments can restore the committed image.
    let garbage = vec![0xFFu8; page_size as usize];
    for p in 1..=count {
        write_at(&tmp.path, (p as u64 - 1) * page_size as u64, &garbage);
    }

    {
        let mut c = tmp.open();
        assert_rows(&mut c, "SELECT id, v FROM t ORDER BY id", &expected_range(1, N));
        assert_scalar(&mut c, "SELECT count(*) FROM t", int(N));
    }
    assert_eq!(read_file(&tmp.path), committed, "both journal segments replayed to the committed image");
    assert!(!tmp.journal().exists(), "the hot journal is removed once recovery completes");
}

// ===========================================================================
// (b1) A TRUNCATED final journal record -> the valid prefix only; garbage not applied.
// ===========================================================================

/// A journal whose final page record is cut short (a crash while the journal itself was
/// still being written, before the database was modified). Recovery must replay the
/// valid prefix and STOP at the truncated record — never read its partial bytes as a
/// pre-image — leaving the committed image intact. The truncated record names a real
/// data page with garbage content, so if recovery wrongly applied it the rows on that
/// page would be destroyed; their survival proves the truncated record was rejected.
#[test]
fn truncated_final_journal_record_is_not_applied() {
    const N: i64 = 400;
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)");
        insert_range(&mut db, 1, N);
    }
    let committed = read_file(&tmp.path);
    let page_size = db_page_size(&committed);
    let pages = split_pages(&committed, page_size);
    let count = pages.len() as u32;
    assert!(count >= 3);

    // The database is NOT modified (a torn journal means the crash preceded any db
    // write). Record 1 is a genuine pre-image of page 1 (its replay is a safe no-op);
    // record 2 targets the LAST page with all-0xFF garbage and is then truncated.
    let last = count;
    let records: Vec<(u32, Vec<u8>)> =
        vec![(1, pages[0].clone()), (last, vec![0xFFu8; page_size as usize])];
    let mut journal = encode_journal(page_size, 2, 0x0BAD_0002, count, None, &records);
    // Cut off the last record's checksum and half its content: a torn final write.
    journal.truncate(journal.len() - (4 + (page_size as usize) / 2));
    std::fs::write(tmp.journal(), &journal).unwrap();

    {
        let mut c = tmp.open();
        assert_rows(&mut c, "SELECT id, v FROM t ORDER BY id", &expected_range(1, N));
        assert_scalar(&mut c, "SELECT count(*) FROM t", int(N));
    }
    assert_eq!(read_file(&tmp.path), committed, "committed image intact; torn record not applied");
    assert!(!tmp.journal().exists(), "hot journal removed after replaying the valid prefix");
}

// ===========================================================================
// (b2) A BAD-CHECKSUM final journal record -> rejected even though the count claims it.
// ===========================================================================

/// The header's page count claims two records, but the second is full-length with a
/// checksum that does not validate (a torn write: some sectors reached disk, others
/// did not). fileformat2 §3's checksum exists precisely to catch a record the count
/// claims but that did not fully land; recovery must reject it and replay only the
/// valid prefix. The bad record carries garbage for a real data page, so its rejection
/// is what keeps the rows intact.
#[test]
fn bad_checksum_final_journal_record_is_not_applied() {
    const N: i64 = 400;
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)");
        insert_range(&mut db, 1, N);
    }
    let committed = read_file(&tmp.path);
    let page_size = db_page_size(&committed);
    let pages = split_pages(&committed, page_size);
    let count = pages.len() as u32;
    assert!(count >= 3);

    let last = count;
    let records: Vec<(u32, Vec<u8>)> =
        vec![(1, pages[0].clone()), (last, vec![0xFFu8; page_size as usize])];
    let mut journal = encode_journal(page_size, 2, 0x0BAD_0003, count, None, &records);
    // Corrupt the final record's checksum (its last 4 bytes) so it cannot validate.
    let len = journal.len();
    journal[len - 1] ^= 0xFF;
    std::fs::write(tmp.journal(), &journal).unwrap();

    {
        let mut c = tmp.open();
        assert_rows(&mut c, "SELECT id, v FROM t ORDER BY id", &expected_range(1, N));
        assert_scalar(&mut c, "SELECT count(*) FROM t", int(N));
    }
    assert_eq!(read_file(&tmp.path), committed, "committed image intact; bad-checksum record rejected");
    assert!(!tmp.journal().exists(), "hot journal removed after replaying the valid prefix");
}

// ===========================================================================
// (c) An INCONSISTENT journal header -> the journal is not valid and is ignored.
// ===========================================================================

/// "A rollback journal is only considered to be valid if it exists and contains a valid
/// header" (fileformat2 §3). A journal with the right magic but a corrupt page-size
/// field is NOT a valid header, so recovery must ignore it entirely — importantly it
/// must NOT act on the header's `initial_db_pages` (set here to 1, which would truncate
/// the database to a single page and destroy every row if the header were trusted).
/// The committed database must be left completely intact.
#[test]
fn inconsistent_journal_header_is_ignored_not_trusted() {
    const N: i64 = 400;
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)");
        insert_range(&mut db, 1, N);
    }
    let committed = read_file(&tmp.path);
    let page_size = db_page_size(&committed);
    let pages = split_pages(&committed, page_size);
    let count = pages.len() as u32;
    assert!(count >= 3);

    // Valid magic, a genuine pre-image record for page 1 — but the page-size header
    // field is 1000 (not a power of two) and initial_db_pages is a catastrophic 1.
    // A correct implementation rejects the header and touches nothing.
    let records = vec![(1u32, pages[0].clone())];
    let journal = encode_journal(page_size, 1, 0x0BAD_000C, 1, Some(1000), &records);
    std::fs::write(tmp.journal(), &journal).unwrap();

    let mut c = tmp.open();
    assert_rows(&mut c, "SELECT id, v FROM t ORDER BY id", &expected_range(1, N));
    assert_scalar(&mut c, "SELECT count(*) FROM t", int(N));
    // The database file is byte-for-byte the committed image (never truncated).
    assert_eq!(read_file(&tmp.path), committed, "an invalid-header journal must not touch the db");
}

// ===========================================================================
// (c2) A record whose checksum was computed under a DIFFERENT nonce than the header
//      declares -> the per-transaction nonce mismatch is detected and it is not replayed.
// ===========================================================================

/// The rollback checksum is seeded with the journal header's nonce, and "a different
/// random nonce is used each time a rollback journal is created, which allows [the]
/// checksum to detect stale data" (fileformat2 §3). This models a stale record left in
/// a reused journal file whose header was rewritten with a fresh nonce: the record's
/// stored checksum was computed under the OLD nonce, so recovery — which recomputes it
/// under the header's (new) nonce — must find the mismatch, treat the record as
/// untrustworthy, and NOT replay it. The record carries garbage for a real data page,
/// so its rejection is what keeps the committed rows intact.
#[test]
fn stale_record_with_mismatched_nonce_is_not_replayed() {
    const N: i64 = 400;
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)");
        insert_range(&mut db, 1, N);
    }
    let committed = read_file(&tmp.path);
    let page_size = db_page_size(&committed);
    let pages = split_pages(&committed, page_size);
    let count = pages.len() as u32;
    assert!(count >= 3);

    // Build the journal with the OLD nonce (so the record's checksum uses it), naming a
    // real data page with garbage content, then rewrite ONLY the header's nonce field to
    // a fresh value. Now the header declares NEW_NONCE while the record is checksummed
    // under OLD_NONCE — exactly a stale record after a journal reuse.
    const OLD_NONCE: u32 = 0x1111_2222;
    const NEW_NONCE: u32 = 0x3333_4444;
    let last = count;
    let records = vec![(last, vec![0xFFu8; page_size as usize])];
    let mut journal = encode_journal(page_size, 1, OLD_NONCE, count, None, &records);
    journal[12..16].copy_from_slice(&NEW_NONCE.to_be_bytes()); // header nonce field
    std::fs::write(tmp.journal(), &journal).unwrap();

    {
        let mut c = tmp.open();
        assert_rows(&mut c, "SELECT id, v FROM t ORDER BY id", &expected_range(1, N));
        assert_scalar(&mut c, "SELECT count(*) FROM t", int(N));
    }
    assert_eq!(read_file(&tmp.path), committed, "stale (mismatched-nonce) record must not be replayed");
    assert!(!tmp.journal().exists(), "the hot journal is removed after recovery declines the stale record");
}

// ===========================================================================
// (c3) A journal header whose page COUNT exceeds the records actually present ->
//      the over-large count is self-correcting; replay stops at end of file.
// ===========================================================================

/// The header's page-count field can legitimately over-claim: a single-sync journal may
/// backfill a count for records that never fully reached disk (fileformat2 §3 makes the
/// per-record checksum, not a trusted count, the bound). Recovery must therefore treat
/// the count as an upper bound and stop when a claimed record runs past end of file,
/// rather than reading unrelated bytes as a pre-image. Here the header claims far more
/// records than exist, with a single valid pre-image for the last (row-bearing) page,
/// which was scribbled in the database: replaying just the present record restores it,
/// while a recovery that trusted the count and read past EOF would error or corrupt.
#[test]
fn over_large_journal_page_count_self_corrects_at_eof() {
    const N: i64 = 400;
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)");
        insert_range(&mut db, 1, N);
    }
    let committed = read_file(&tmp.path);
    let page_size = db_page_size(&committed);
    let pages = split_pages(&committed, page_size);
    let count = pages.len() as u32;
    assert!(count >= 3);

    // One valid record — the pre-image of the last (row-bearing) page — but the header
    // claims count+10 records. The file ends right after the single record, so records
    // 2..=count+10 are simply absent.
    let last = count;
    let records = vec![(last, pages[(last - 1) as usize].clone())];
    let journal = encode_journal(page_size, count + 10, 0x0BAD_00C3, count, None, &records);
    std::fs::write(tmp.journal(), &journal).unwrap();

    // The aborted transaction had overwritten the last page; only replaying the present
    // record restores it.
    write_at(&tmp.path, (last as u64 - 1) * page_size as u64, &vec![0xFFu8; page_size as usize]);

    {
        let mut c = tmp.open();
        assert_rows(&mut c, "SELECT id, v FROM t ORDER BY id", &expected_range(1, N));
        assert_scalar(&mut c, "SELECT count(*) FROM t", int(N));
    }
    assert_eq!(read_file(&tmp.path), committed, "present record replayed; the over-large count stopped at EOF");
    assert!(!tmp.journal().exists(), "the hot journal is removed once recovery completes");
}

// ===========================================================================
// WAL helpers.
// ===========================================================================

/// The `-wal` header is 32 bytes; a live log with any committed frame is larger, and a
/// completed checkpoint that resets the log truncates it back to exactly this
/// (fileformat2 §4.1 / walformat).
const WAL_HEADER_SIZE: u64 = 32;
/// Each WAL frame is a 24-byte frame header plus one page of data (fileformat2 §4.1).
const WAL_FRAME_HEADER: u64 = 24;

/// Run `PRAGMA journal_mode=WAL` from a throwaway connection and close it, so the NEXT
/// open uses the WAL backing (the "reopen model" documented in
/// `conformance_wal_durability.rs`). Asserts the pragma reports `wal`.
fn enter_wal_mode(tmp: &TempDb) {
    let mut a = tmp.open();
    let qr = query(&mut a, "PRAGMA journal_mode=WAL");
    match &qr.rows[0][0] {
        Value::Text(s) if s == "wal" => {}
        other => panic!("PRAGMA journal_mode=WAL should report wal, got {other:?}"),
    }
}

/// Assert `db` currently reports WAL journal mode (so a reopen really took the WAL path).
fn assert_wal_mode(db: &mut Connection) {
    let qr = query(db, "PRAGMA journal_mode");
    match &qr.rows[0][0] {
        Value::Text(s) if s == "wal" => {}
        other => panic!("expected journal_mode=wal, got {other:?}"),
    }
}

// ===========================================================================
// (d) WAL committed frames but no checkpoint -> recovered from the -wal on reopen.
// ===========================================================================

/// Several committed transactions in WAL mode (never checkpointed) leave their frames
/// in the `-wal`; the database file is not written by a WAL commit (fileformat2 §4).
/// After dropping every connection (freeing the in-process WAL coordinator) and
/// deleting the transient `-shm` — the state a crash leaves — a reopen must
/// reconstruct the committed snapshot from the `-wal` alone (fileformat2 §4.6: "After
/// a crash, the wal-index is reconstructed from the original WAL file"), including the
/// LATEST value of a page rewritten across transactions (§4.5 reader algorithm).
#[test]
fn wal_committed_frames_recovered_from_wal_after_crash() {
    const A_HI: i64 = 100;
    const B_HI: i64 = 200;
    let tmp = TempDb::new();
    enter_wal_mode(&tmp);

    let db_len_base;
    {
        let mut b = tmp.open();
        assert_wal_mode(&mut b);
        exec(&mut b, "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)");
        insert_range(&mut b, 1, A_HI); // transaction chain #1
        insert_range(&mut b, A_HI + 1, B_HI); // transaction chain #2
        // A cross-transaction UPDATE: page carrying id=50 is rewritten in a LATER frame,
        // so recovery must resolve the newest frame for it, not the original.
        exec(&mut b, "UPDATE t SET v = 'updated-0050' WHERE id = 50");
        db_len_base = file_len(&tmp.path);
    }

    // Frames really live in the -wal, and the WAL commits never wrote the db file.
    assert!(file_len(&tmp.wal()) > WAL_HEADER_SIZE, "committed frames present in the -wal");

    // Simulate the crash: the transient wal-index is gone; only the -wal remains.
    let _ = std::fs::remove_file(tmp.sidecar("-shm"));

    let mut c = tmp.open();
    assert_wal_mode(&mut c);
    // Every row survives, and id=50 shows the cross-transaction UPDATE (latest frame).
    let mut expected = expected_range(1, B_HI);
    expected[49] = vec![int(50), text("updated-0050")];
    assert_rows(&mut c, "SELECT id, v FROM t ORDER BY id", &expected);
    assert_scalar(&mut c, "SELECT count(*) FROM t", int(B_HI));
    assert_eq!(file_len(&tmp.path), db_len_base, "reopen did not silently checkpoint the db file");
}

// ===========================================================================
// (e1) A torn (short) trailing WAL frame -> ignored; the last valid commit wins.
// ===========================================================================

/// A partial frame appended after the last committed frame (an interrupted write that
/// never produced a valid frame). `scan` must stop at the last valid commit and ignore
/// the torn tail (fileformat2 §4.1: a frame is valid only with matching salts AND a
/// matching cumulative checksum), so every committed row survives and no phantom row
/// appears.
#[test]
fn torn_trailing_wal_frame_is_ignored_on_reopen() {
    const N: i64 = 150;
    let tmp = TempDb::new();
    enter_wal_mode(&tmp);
    {
        let mut b = tmp.open();
        assert_wal_mode(&mut b);
        exec(&mut b, "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)");
        insert_range(&mut b, 1, N);
    }
    let good_len = file_len(&tmp.wal());
    assert!(good_len > WAL_HEADER_SIZE, "committed frames present before the torn tail");

    // A short partial frame: fewer bytes than a whole frame, so `scan` cannot mistake
    // it for a valid one and stops at the prior commit.
    append(&tmp.wal(), &[0xFFu8; 37]);
    assert!(file_len(&tmp.wal()) > good_len, "torn bytes appended");

    let mut c = tmp.open();
    assert_wal_mode(&mut c);
    assert_rows(&mut c, "SELECT id, v FROM t ORDER BY id", &expected_range(1, N));
    assert_scalar(&mut c, "SELECT count(*) FROM t", int(N));
}

// ===========================================================================
// (e2) A corrupted final commit frame -> that transaction is discarded, priors survive.
// ===========================================================================

/// Two committed transactions; the SECOND transaction's final (commit) frame has one
/// page-data byte flipped, breaking its cumulative checksum. Per fileformat2 §4.5 the
/// reader uses "the last valid instance of page P that is followed by a commit frame or
/// is a commit frame itself", and frames after the last valid commit are ignored — so
/// the corrupted transaction is dropped WHOLESALE and only the first transaction's rows
/// survive. This is the torn-final-commit class of the real WAL corruption bug.
#[test]
fn corrupted_final_commit_frame_discards_only_that_transaction() {
    const A_HI: i64 = 100;
    const B_HI: i64 = 150;
    let tmp = TempDb::new();
    enter_wal_mode(&tmp);
    {
        let mut b = tmp.open();
        assert_wal_mode(&mut b);
        exec(&mut b, "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)");
        insert_range(&mut b, 1, A_HI); // transaction A (must survive)
        insert_range(&mut b, A_HI + 1, B_HI); // transaction B (its last frame is corrupted)
    }

    // Flip one byte inside the LAST frame's page data. A frame is 24 + page_size bytes;
    // its page data is the final page_size bytes of the file, so offset (len - page_size)
    // is the first page-data byte. Any flip there breaks that frame's cumulative checksum.
    let committed = read_file(&tmp.path);
    let page_size = db_page_size(&committed) as u64;
    let wal_len = file_len(&tmp.wal());
    assert!(
        wal_len >= WAL_HEADER_SIZE + 2 * (WAL_FRAME_HEADER + page_size),
        "expected at least two committed frames in the -wal (len {wal_len})"
    );
    let last_frame_data = wal_len - page_size;
    {
        let mut f = OpenOptions::new().read(true).write(true).open(tmp.wal()).unwrap();
        f.seek(SeekFrom::Start(last_frame_data)).unwrap();
        let mut byte = [0u8; 1];
        f.read_exact(&mut byte).unwrap();
        byte[0] ^= 0xFF;
        f.seek(SeekFrom::Start(last_frame_data)).unwrap();
        f.write_all(&byte).unwrap();
        f.sync_all().unwrap();
    }

    let mut c = tmp.open();
    assert_wal_mode(&mut c);
    // Transaction B is discarded in full; only transaction A's rows remain.
    assert_rows(&mut c, "SELECT id, v FROM t ORDER BY id", &expected_range(1, A_HI));
    assert_scalar(&mut c, "SELECT count(*) FROM t", int(A_HI));
    assert_scalar(&mut c, "SELECT count(*) FROM t WHERE id > 100", int(0));
}

// ===========================================================================
// (f) An interrupted checkpoint -> no data loss; the WAL/db stay consistent.
// ===========================================================================

/// A checkpoint copies WAL pages into the database file and only THEN, on a complete
/// drain with no readers, resets the `-wal` (fileformat2 §4.3/§4.4). A crash partway
/// leaves the database file partially written while the `-wal` is still fully intact
/// (the reset never ran). On reopen the WAL is authoritative, so every read resolves
/// through it and the half-written database pages are shadowed — no data is lost. A
/// subsequent checkpoint then completes idempotently. This is the single-connection
/// face of the checkpoint-vs-writer interleaving where the real WAL bug lived.
#[test]
fn interrupted_checkpoint_loses_no_data_and_recovers() {
    const N: i64 = 200;
    let tmp = TempDb::new();
    enter_wal_mode(&tmp);
    {
        let mut b = tmp.open();
        assert_wal_mode(&mut b);
        exec(&mut b, "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)");
        insert_range(&mut b, 1, N);
    }
    assert!(file_len(&tmp.wal()) > WAL_HEADER_SIZE, "committed frames live in the -wal");

    // Simulate an interrupted checkpoint: the database file gets partially overwritten
    // with garbage data pages (as a mid-copy checkpoint would), grown to hold them, but
    // the -wal is left fully intact (the crash preceded the reset). Page 1's header is
    // preserved so WAL-mode detection still works — a real checkpoint writes a valid
    // page 1, never garbage over the magic.
    let committed = read_file(&tmp.path);
    let page_size = db_page_size(&committed) as u64;
    let garbage_pages = 8u64;
    let garbage = vec![0xEEu8; (garbage_pages * page_size) as usize];
    // Overwrite pages 2.. (offset one page in) and grow the file — page 1 untouched.
    write_at(&tmp.path, page_size, &garbage);
    let wal_before = file_len(&tmp.wal());

    // Reopen over the half-checkpointed database: the intact WAL shadows the garbage db
    // pages, so every committed row is read back exactly.
    {
        let mut c = tmp.open();
        assert_wal_mode(&mut c);
        assert_rows(&mut c, "SELECT id, v FROM t ORDER BY id", &expected_range(1, N));
        assert_scalar(&mut c, "SELECT count(*) FROM t", int(N));
        assert_eq!(
            file_len(&tmp.wal()),
            wal_before,
            "the interrupted checkpoint left the -wal intact; reopen did not silently reset it"
        );

        // Completing the checkpoint now must drain the intact WAL over the garbage and
        // succeed — proving the partial db writes did not corrupt the WAL/db pairing.
        let qr = query(&mut c, "PRAGMA wal_checkpoint");
        match &qr.rows[0][0] {
            Value::Integer(0) => {}
            other => panic!("wal_checkpoint should report busy=0, got {other:?}"),
        }
        assert_rows(&mut c, "SELECT id, v FROM t ORDER BY id", &expected_range(1, N));
    }

    // Final reopen: with the checkpoint completed the rows now come from the database
    // file, still exact — end-to-end, the interrupted checkpoint lost nothing.
    let mut d = tmp.open();
    assert_wal_mode(&mut d);
    assert_rows(&mut d, "SELECT id, v FROM t ORDER BY id", &expected_range(1, N));
    assert_scalar(&mut d, "SELECT count(*) FROM t", int(N));
}

// ===========================================================================
// (g) A -wal with a CORRUPT header -> the WAL is ignored; the checkpointed database
//     base survives (the WAL twin of the invalid-journal-header case).
// ===========================================================================

/// "A WAL file ... is only considered valid if [its] header ... has a valid checksum"
/// (fileformat2 §4.1): a `-wal` whose header fails validation cannot be interpreted, so
/// recovery must ignore the entire WAL and fall back to the database file — never error,
/// and never serve garbage. Here a first transaction is checkpointed into the database
/// base, a second transaction's frames land in the freshly-reset `-wal`, and then the
/// WAL header's checksum is corrupted. On reopen the WAL is discarded, so the
/// checkpointed base (transaction one) survives intact; transaction two, which existed
/// only under the now-invalid header, is legitimately unrecoverable (fileformat2 §4).
#[test]
fn corrupt_wal_header_is_ignored_checkpointed_base_survives() {
    const A_HI: i64 = 100; // checkpointed into the db base
    const B_HI: i64 = 150; // only in the -wal under the (soon-corrupt) header
    let tmp = TempDb::new();
    enter_wal_mode(&tmp);
    {
        let mut b = tmp.open();
        assert_wal_mode(&mut b);
        exec(&mut b, "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)");
        insert_range(&mut b, 1, A_HI);
        // Drain transaction one into the database file and reset the -wal.
        let qr = query(&mut b, "PRAGMA wal_checkpoint");
        match &qr.rows[0][0] {
            Value::Integer(0) => {}
            other => panic!("wal_checkpoint should report busy=0, got {other:?}"),
        }
        // Transaction two appends fresh frames into the reset -wal (new header).
        insert_range(&mut b, A_HI + 1, B_HI);
    }
    assert!(
        file_len(&tmp.wal()) > WAL_HEADER_SIZE,
        "transaction two's frames are present in the -wal before corruption"
    );

    // Corrupt the checkpoint-sequence field of the WAL header (offset 12, fileformat2
    // §4.1). It is covered by the header checksum (computed over the first 24 bytes) so
    // this invalidates the header, yet — unlike the salts (16..24) or the checksum seed
    // (24..32) — it does NOT feed frame validation. So the ONLY thing standing between a
    // reopen and transaction two's (intact) frames is the header-checksum check itself:
    // the test isolates that guard rather than being masked by frame-level rejection.
    {
        let mut f = OpenOptions::new().read(true).write(true).open(tmp.wal()).unwrap();
        f.seek(SeekFrom::Start(12)).unwrap();
        let mut byte = [0u8; 1];
        f.read_exact(&mut byte).unwrap();
        byte[0] ^= 0xFF;
        f.seek(SeekFrom::Start(12)).unwrap();
        f.write_all(&byte).unwrap();
        f.sync_all().unwrap();
    }
    let _ = std::fs::remove_file(tmp.sidecar("-shm"));

    // Reopen: the invalid WAL header is ignored and the checkpointed base is authoritative.
    let mut c = tmp.open();
    assert_wal_mode(&mut c);
    assert_rows(&mut c, "SELECT id, v FROM t ORDER BY id", &expected_range(1, A_HI));
    assert_scalar(&mut c, "SELECT count(*) FROM t", int(A_HI));
    assert_scalar(&mut c, "SELECT count(*) FROM t WHERE id > 100", int(0));
}
