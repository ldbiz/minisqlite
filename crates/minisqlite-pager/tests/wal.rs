//! WAL-mode behavior tests, driven through the PUBLIC pager surface (`DiskPager` +
//! `Pager`) — the real path the engine uses, not internal helpers — plus `is_busy`
//! to recognize the retryable BUSY these tests assert (a behavioral seam exercised
//! only by these tests, not part of the engine's path).
//!
//! Single-connection: a WAL-mode round-trip survives a reopen, a checkpoint drains
//! the WAL back into the database and resets it, and a torn last frame is ignored on
//! reopen. Two-connection (all in ONE process on one canonical path, the documented
//! same-process coordination model): a second connection sees the first's commits, a
//! reader keeps its snapshot across a concurrent commit, a checkpoint is bounded by an
//! active reader and only resets once no reader is behind, and two writers serialize
//! (the second gets BUSY, never interleaves).

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use minisqlite_pager::codec::{DatabaseHeader, HEADER_SIZE, MAGIC};
use minisqlite_pager::{is_busy, CheckpointMode, DiskPager, Pager};

const PS: u32 = 512;

/// The `-wal` on-disk layout (walformat §4), kept as the single source of truth these tests
/// share so a frame-size change cannot silently drift between the frame-count helper and the
/// torn/corrupt-frame offset math: a fixed 32-byte file header, then fixed-size frames of a
/// 24-byte frame header plus one `PS`-byte page.
const WAL_HEADER_SIZE: u64 = 32;
const FRAME_HEADER_SIZE: u64 = 24;
const FRAME_STRIDE: u64 = FRAME_HEADER_SIZE + PS as u64;

/// A WAL-mode database plus its `-wal`/`-journal` siblings under the OS temp dir, all
/// removed on drop. Unique per test so parallel runs never collide.
struct Case {
    db: PathBuf,
    wal: PathBuf,
    journal: PathBuf,
}

impl Case {
    fn new(tag: &str) -> Case {
        static N: AtomicU32 = AtomicU32::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
        let mut db = std::env::temp_dir();
        db.push(format!("mspw-{tag}-{pid}-{n}-{nanos}.db"));
        let sib = |suffix: &str| {
            let mut s = db.as_os_str().to_os_string();
            s.push(suffix);
            PathBuf::from(s)
        };
        Case { wal: sib("-wal"), journal: sib("-journal"), db }
    }
}

impl Drop for Case {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.db);
        let _ = std::fs::remove_file(&self.wal);
        let _ = std::fs::remove_file(&self.journal);
    }
}

/// One page filled with a repeating byte.
fn marker(byte: u8) -> Vec<u8> {
    vec![byte; PS as usize]
}

/// The initial content of a crafted page (distinct per page id).
fn page_seed(id: u32) -> u8 {
    0xE0u8.wrapping_add(id as u8)
}

/// Craft a WAL-mode database file directly: a version-2 page-1 header declaring
/// `n_pages`, followed by `n_pages - 1` seeded content pages. Bypasses the pager so a
/// test can pin the exact page size and layout, and so `DiskPager::open` exercises the
/// real WAL-mode detection + open path on a file it did not itself create.
fn craft_wal_db(path: &Path, n_pages: u32) {
    let mut h = DatabaseHeader { page_size: PS, ..DatabaseHeader::default() };
    h.write_version = 2;
    h.read_version = 2;
    h.database_size_pages = n_pages; // counters default 0 == 0 ⇒ in-header size valid
    let mut buf: Vec<u8> = Vec::with_capacity((n_pages * PS) as usize);
    let mut page1 = vec![0u8; PS as usize];
    page1[..HEADER_SIZE].copy_from_slice(&h.to_bytes());
    buf.extend_from_slice(&page1);
    for id in 2..=n_pages {
        buf.extend_from_slice(&marker(page_seed(id)));
    }
    std::fs::write(path, &buf).unwrap();
}

fn file_len(path: &Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

/// Number of committed frames currently in the `-wal` file: `(len - 32) / (24 + PS)`
/// (walformat §4). This is exactly the pragma's `log` column — "the number of modified pages
/// that have been written to the write-ahead log file" (pragma.html #pragma_wal_checkpoint) —
/// so the checkpoint tests pin the report against the file they can see, without hardcoding how
/// many frames a given commit produced (a single one-page commit also appends a page-1 frame
/// for change-counter maintenance, so N one-page commits leave 2N frames).
///
/// This equals the in-memory `mxFrame` (and thus `rep.log`) only when the file holds whole,
/// valid frames with no torn tail — true for these tests, which only ever drive clean
/// commits. It would over-count against a deliberately torn/corrupted tail (see
/// `torn_last_frame_is_ignored_on_reopen` / `corrupted_trailing_frame_is_ignored_on_reopen`),
/// so do not use it to predict `rep.log` in a case that appends a partial frame.
fn wal_frame_count(path: &Path) -> u32 {
    let len = file_len(path);
    if len <= WAL_HEADER_SIZE {
        0
    } else {
        ((len - WAL_HEADER_SIZE) / FRAME_STRIDE) as u32
    }
}

fn is_sqlite_header(page: &[u8]) -> bool {
    page.len() >= HEADER_SIZE && &page[0..16] == MAGIC.as_slice()
}

#[test]
fn wal_roundtrip_survives_reopen_without_touching_db_file() {
    let case = Case::new("roundtrip");
    craft_wal_db(&case.db, 3); // pages 1..=3 in the db file

    {
        let mut p = DiskPager::open(&case.db).unwrap();
        assert_eq!(p.page_size(), PS, "page size from the crafted header");
        assert_eq!(p.page_count().unwrap(), 3);

        p.begin().unwrap();
        p.write_page(2, &marker(0xB2)).unwrap();
        let four = p.allocate_page().unwrap();
        assert_eq!(four, 4, "allocation grows the logical db");
        p.write_page(4, &marker(0xB4)).unwrap();
        p.commit().unwrap();

        // The writer sees its own commit immediately (no reopen, no begin_read).
        assert_eq!(p.page_count().unwrap(), 4);
        assert_eq!(p.read_page(2).unwrap(), &marker(0xB2)[..]);
        assert_eq!(p.read_page(4).unwrap(), &marker(0xB4)[..]);
    }

    // A commit in WAL mode appends to the -wal and does NOT write the db file.
    assert!(file_len(&case.wal) > 32, "commit appended frames to the -wal");
    assert_eq!(file_len(&case.db), 3 * PS as u64, "the db file was not grown by the commit");

    // Reopen: the WAL is read back, so the committed state (including the new page 4)
    // survives.
    let p = DiskPager::open(&case.db).unwrap();
    assert_eq!(p.page_count().unwrap(), 4, "logical size comes from the WAL after reopen");
    assert_eq!(p.read_page(2).unwrap(), &marker(0xB2)[..]);
    assert_eq!(p.read_page(4).unwrap(), &marker(0xB4)[..]);
    assert!(is_sqlite_header(p.read_page(1).unwrap()), "page 1 is still a valid header");
}

#[test]
fn checkpoint_drains_wal_into_db_and_resets() {
    let case = Case::new("checkpoint");
    craft_wal_db(&case.db, 2);

    {
        let mut p = DiskPager::open(&case.db).unwrap();
        p.begin().unwrap();
        p.write_page(2, &marker(0xC2)).unwrap();
        let three = p.allocate_page().unwrap();
        assert_eq!(three, 3);
        p.write_page(3, &marker(0xC3)).unwrap();
        p.commit().unwrap();

        assert!(file_len(&case.wal) > 32, "frames in the WAL before checkpoint");
        assert_eq!(file_len(&case.db), 2 * PS as u64, "db not yet grown");

        // A full checkpoint with no active reader drains the WAL into the db file and
        // resets the WAL (fresh header, truncated to 32 bytes).
        p.checkpoint(CheckpointMode::Passive).unwrap();
        assert_eq!(file_len(&case.wal), 32, "WAL reset to a bare header after a complete drain");
        assert_eq!(file_len(&case.db), 3 * PS as u64, "db file grew to hold the drained pages");
        assert_eq!(p.read_page(3).unwrap(), &marker(0xC3)[..], "checkpointed data readable");
    }

    // Reopen: the data now lives in the db file (WAL empty), and is intact.
    let p = DiskPager::open(&case.db).unwrap();
    assert_eq!(p.page_count().unwrap(), 3);
    assert_eq!(p.read_page(2).unwrap(), &marker(0xC2)[..]);
    assert_eq!(p.read_page(3).unwrap(), &marker(0xC3)[..]);
}

#[test]
fn torn_last_frame_is_ignored_on_reopen() {
    let case = Case::new("torn");
    craft_wal_db(&case.db, 2);
    {
        let mut p = DiskPager::open(&case.db).unwrap();
        p.begin().unwrap();
        p.write_page(2, &marker(0xD2)).unwrap();
        p.commit().unwrap(); // one durable commit frame
    }
    let good_len = file_len(&case.wal);

    // Simulate an interrupted append: a partial (torn) frame with no valid header or
    // checksum, shorter than a whole frame. `scan` must stop at the last good commit.
    {
        let mut wal = OpenOptions::new().append(true).open(&case.wal).unwrap();
        wal.write_all(&[0xFF; 37]).unwrap();
        wal.sync_all().unwrap();
    }
    assert!(file_len(&case.wal) > good_len, "torn bytes were appended");

    let p = DiskPager::open(&case.db).unwrap();
    assert_eq!(p.page_count().unwrap(), 2, "size from the last valid commit");
    assert_eq!(p.read_page(2).unwrap(), &marker(0xD2)[..], "committed data intact past the torn tail");
}

#[test]
fn second_connection_sees_first_connections_commit() {
    let case = Case::new("see-commit");
    craft_wal_db(&case.db, 2);
    let mut a = DiskPager::open(&case.db).unwrap();
    let mut b = DiskPager::open(&case.db).unwrap();

    a.begin().unwrap();
    a.write_page(2, &marker(0xA2)).unwrap();
    a.commit().unwrap();

    // A fresh read transaction on B resolves against the new WAL ceiling.
    b.begin_read().unwrap();
    assert_eq!(b.read_page(2).unwrap(), &marker(0xA2)[..], "B sees A's committed page");
    b.end_read().unwrap();
}

#[test]
fn reader_keeps_snapshot_while_writer_commits() {
    let case = Case::new("snapshot");
    craft_wal_db(&case.db, 2);
    let mut a = DiskPager::open(&case.db).unwrap();
    let mut b = DiskPager::open(&case.db).unwrap();

    // Commit #1: page 2 = 0xA1.
    a.begin().unwrap();
    a.write_page(2, &marker(0xA1)).unwrap();
    a.commit().unwrap();

    // B pins a read snapshot at commit #1.
    b.begin_read().unwrap();
    assert_eq!(b.read_page(2).unwrap(), &marker(0xA1)[..]);

    // Commit #2 (by A) appends frames beyond B's snapshot.
    a.begin().unwrap();
    a.write_page(2, &marker(0xA2)).unwrap();
    a.commit().unwrap();

    // B still reads its pinned snapshot — the writer's new commit is invisible.
    assert_eq!(b.read_page(2).unwrap(), &marker(0xA1)[..], "reader holds its snapshot");

    // A NEW read transaction on B advances the snapshot and sees commit #2.
    b.end_read().unwrap();
    b.begin_read().unwrap();
    assert_eq!(b.read_page(2).unwrap(), &marker(0xA2)[..], "a new read sees the new data");
    b.end_read().unwrap();
}

#[test]
fn checkpoint_is_bounded_by_active_reader_then_resets() {
    let case = Case::new("bounded-ckpt");
    craft_wal_db(&case.db, 2);
    let mut a = DiskPager::open(&case.db).unwrap();
    let mut b = DiskPager::open(&case.db).unwrap();

    // Commit #1 then a reader pins it; commit #2 lands beyond the reader.
    a.begin().unwrap();
    a.write_page(2, &marker(0xA1)).unwrap();
    a.commit().unwrap();
    b.begin_read().unwrap();
    assert_eq!(b.read_page(2).unwrap(), &marker(0xA1)[..]);
    a.begin().unwrap();
    a.write_page(2, &marker(0xA2)).unwrap();
    a.commit().unwrap();

    // A checkpoint runs while B is behind: it drains only up to B's mark, does NOT
    // reset the WAL, and leaves B's snapshot readable.
    let bounded = a.checkpoint(CheckpointMode::Passive).unwrap();
    assert!(
        !bounded.busy,
        "PASSIVE is never busy even with a reader behind and frames present (pragma.html: the busy-handler is never invoked in PASSIVE mode)"
    );
    assert!(file_len(&case.wal) > 32, "WAL not reset while a reader is behind mxFrame");
    assert_eq!(b.read_page(2).unwrap(), &marker(0xA1)[..], "reader unaffected by the bounded checkpoint");

    // Once the reader releases, a checkpoint drains fully and resets the WAL, losing
    // nothing: the newest committed data is still readable (now from the db file).
    b.end_read().unwrap();
    let reset = a.checkpoint(CheckpointMode::Passive).unwrap();
    assert!(!reset.busy, "PASSIVE is never busy (pragma.html)");
    assert_eq!(file_len(&case.wal), 32, "WAL reset once no reader is behind");
    b.begin_read().unwrap();
    assert_eq!(b.read_page(2).unwrap(), &marker(0xA2)[..], "latest data survives checkpoint + reset");
    b.end_read().unwrap();
}

#[test]
fn two_writers_serialize_second_gets_busy() {
    let case = Case::new("busy");
    craft_wal_db(&case.db, 2);
    let mut a = DiskPager::open(&case.db).unwrap();
    let mut b = DiskPager::open(&case.db).unwrap();

    // A holds the single write lock for its transaction.
    a.begin().unwrap();
    a.write_page(2, &marker(0xA2)).unwrap();

    // B's write attempt gets BUSY immediately (it never blocks or interleaves).
    let err = b.begin().unwrap_err();
    assert!(is_busy(&err), "the second writer gets BUSY: {err}");

    // A finishes atomically; only then can B write.
    a.commit().unwrap();
    b.begin().unwrap();
    b.write_page(2, &marker(0xB2)).unwrap();
    b.commit().unwrap();

    b.begin_read().unwrap();
    assert_eq!(b.read_page(2).unwrap(), &marker(0xB2)[..], "B's serialized write landed");
    b.end_read().unwrap();
}

#[test]
fn write_after_reset_uses_fresh_salts() {
    // The reset-then-write path (the fence's happy path): after a checkpoint resets
    // the WAL with fresh salts, the next commit appends to the fresh WAL using the new
    // header read back from shared state — never a stale copied salt — so the frames
    // validate on a later scan.
    let case = Case::new("post-reset");
    craft_wal_db(&case.db, 2);
    let mut a = DiskPager::open(&case.db).unwrap();

    a.begin().unwrap();
    a.write_page(2, &marker(0xA1)).unwrap();
    a.commit().unwrap();
    a.checkpoint(CheckpointMode::Passive).unwrap(); // full drain + reset (no readers)
    assert_eq!(file_len(&case.wal), 32, "WAL reset");

    // Append to the fresh WAL.
    a.begin().unwrap();
    a.write_page(2, &marker(0xA2)).unwrap();
    a.commit().unwrap();
    assert!(file_len(&case.wal) > 32, "post-reset commit re-grew the WAL");
    a.begin_read().unwrap();
    assert_eq!(a.read_page(2).unwrap(), &marker(0xA2)[..]);
    a.end_read().unwrap();
    drop(a);

    // Reopen scans the post-reset WAL: the fresh-salt frames validate and the newest
    // data is read back.
    let p = DiskPager::open(&case.db).unwrap();
    assert_eq!(p.read_page(2).unwrap(), &marker(0xA2)[..], "post-reset commit survives reopen");
}

#[test]
fn reader_snapshot_survives_checkpoint_reset_and_recommit() {
    // Snapshot isolation across a checkpoint/reset (the behaviors pinned here). A reader
    // whose mark EQUALS the current mxFrame makes a drain report `complete`, but the
    // WAL must NOT be reset while that reader is active: a later reset+recommit would
    // advance the db file past frames the reader must still resolve from the WAL, so it
    // would observe data committed AFTER its pinned snapshot. The reset must be deferred
    // until no reader is active (SQLite's WAL-restart rule).
    let case = Case::new("snap-reset");
    craft_wal_db(&case.db, 3); // db file: page1(size3), page2=0xE2, page3=0xE3
    let mut a = DiskPager::open(&case.db).unwrap();
    let mut b = DiskPager::open(&case.db).unwrap();

    // gen0 commit: touch only page 2 (page 3 stays 0xE3, lives ONLY in the db file).
    a.begin().unwrap();
    a.write_page(2, &marker(0xA2)).unwrap();
    a.commit().unwrap();

    // B pins a read snapshot at the gen0 ceiling. It must hold it until end_read.
    b.begin_read().unwrap();
    assert_eq!(b.read_page(2).unwrap(), &marker(0xA2)[..]);
    // B deliberately does NOT read page 3 yet, so nothing is cached for it.

    // Checkpoint #1: B's mark == mxFrame, so the drain reports complete — but B is an
    // active reader, so the WAL must NOT be reset.
    a.checkpoint(CheckpointMode::Passive).unwrap();

    // A modifies page 3 and checkpoints again. B's mark still bounds the drain, so
    // page 3's new value is NOT backfilled past B's snapshot.
    a.begin().unwrap();
    a.write_page(3, &marker(0xF3)).unwrap();
    a.commit().unwrap();
    a.checkpoint(CheckpointMode::Passive).unwrap();

    // B is STILL in its original read transaction; its snapshot predates page 3 ever
    // changing, so it must observe the ORIGINAL page 3 (page_seed(3) == 0xE3).
    assert_eq!(
        b.read_page(3).unwrap(),
        &marker(page_seed(3))[..],
        "reader must keep its snapshot: page 3 must be the pre-snapshot value, not 0xF3"
    );
    b.end_read().unwrap();
}

#[test]
fn dropped_writer_releases_the_write_lock() {
    // A connection dropped mid-transaction (no commit/rollback) must not strand the
    // single write lock. Real sqlite implicitly rolls back on close, so a sibling can
    // then take the lock; without `impl Drop for WalStore` the sibling would be BUSY
    // forever (a permanent liveness failure invisible at the leaking connection).
    let case = Case::new("drop-writelock");
    craft_wal_db(&case.db, 2);
    let mut a = DiskPager::open(&case.db).unwrap();
    let mut b = DiskPager::open(&case.db).unwrap(); // keeps the shared coordinator alive

    a.begin().unwrap(); // A takes the single write lock
    a.write_page(2, &marker(0xA2)).unwrap();
    drop(a); // dropped mid-transaction: no commit/rollback

    // The lock was released on drop, so B can begin AND commit its own transaction.
    b.begin().unwrap();
    b.write_page(2, &marker(0xB2)).unwrap();
    b.commit().unwrap();
    b.begin_read().unwrap();
    assert_eq!(
        b.read_page(2).unwrap(),
        &marker(0xB2)[..],
        "sibling proceeds after the writer was dropped mid-transaction"
    );
    b.end_read().unwrap();
}

#[test]
fn dropped_reader_does_not_pin_checkpoint_forever() {
    // A connection dropped without end_read must not leave a reader mark that pins
    // every future checkpoint's drain bound and forbids the WAL reset forever
    // (unbounded WAL growth surfacing far from the leaking connection).
    let case = Case::new("drop-reader");
    craft_wal_db(&case.db, 2);
    let mut a = DiskPager::open(&case.db).unwrap();
    let mut b = DiskPager::open(&case.db).unwrap();

    a.begin().unwrap();
    a.write_page(2, &marker(0xA2)).unwrap();
    a.commit().unwrap();

    // B pins a read snapshot then is dropped WITHOUT end_read.
    b.begin_read().unwrap();
    assert_eq!(b.read_page(2).unwrap(), &marker(0xA2)[..]);
    drop(b);

    // With B's mark released on drop, A's checkpoint fully drains AND resets the WAL.
    a.checkpoint(CheckpointMode::Passive).unwrap();
    assert_eq!(file_len(&case.wal), 32, "no live reader ⇒ checkpoint drains fully and resets");
    assert_eq!(a.read_page(2).unwrap(), &marker(0xA2)[..], "data intact after drain + reset");
}

#[test]
fn peer_commit_after_reset_validates_on_reopen() {
    // The headline fence scenario across TWO connections (the 16-year-bug class): A
    // checkpoints + resets the WAL (bumping generation + fresh salts), then B — a
    // DIFFERENT connection that cached the OLD header at open — commits new frames. B
    // must append with the FRESH salts read from shared state under the lock (never a
    // stale copied header), so its frames validate on a later scan / reopen.
    let case = Case::new("peer-after-reset");
    craft_wal_db(&case.db, 2);
    let mut a = DiskPager::open(&case.db).unwrap();
    let mut b = DiskPager::open(&case.db).unwrap();

    // A commits and checkpoints (full drain + reset with fresh salts).
    a.begin().unwrap();
    a.write_page(2, &marker(0xA2)).unwrap();
    a.commit().unwrap();
    a.checkpoint(CheckpointMode::Passive).unwrap();
    assert_eq!(file_len(&case.wal), 32, "A reset the WAL");

    // B (holding the OLD generation/salts from open) commits into the fresh WAL.
    b.begin().unwrap();
    b.write_page(2, &marker(0xB2)).unwrap();
    b.commit().unwrap();
    assert!(file_len(&case.wal) > 32, "B's commit re-grew the fresh WAL");
    assert_eq!(b.read_page(2).unwrap(), &marker(0xB2)[..], "B sees its own post-reset commit");

    drop(a);
    drop(b);

    // Reopen scans the post-reset WAL: B's fresh-salt frames validate.
    let p = DiskPager::open(&case.db).unwrap();
    assert_eq!(
        p.read_page(2).unwrap(),
        &marker(0xB2)[..],
        "a peer's post-reset commit survives reopen (fresh salts, not a stale copy)"
    );
}

#[test]
fn many_commits_then_read_resolves_latest_via_incremental_index() {
    // Exercises the incrementally-maintained index (extend_commit) across many commits
    // WITHOUT a checkpoint, so the WAL accumulates frames. This is the "fine on three
    // rows" shape the small tests miss: each commit must extend the index in place and
    // still resolve the LATEST page, and a full re-scan on reopen must agree.
    let case = Case::new("many-commits");
    craft_wal_db(&case.db, 2);
    let mut a = DiskPager::open(&case.db).unwrap();
    let m: u32 = 200;
    for i in 0..m {
        a.begin().unwrap();
        a.write_page(2, &marker(i as u8)).unwrap();
        a.commit().unwrap();
        // The writer observes its own newest commit immediately (incremental index).
        assert_eq!(a.read_page(2).unwrap(), &marker(i as u8)[..], "commit {i} is visible");
    }
    drop(a);

    // Reopen scans the whole accumulated WAL; the incremental index and a full scan
    // must agree on the newest value.
    let p = DiskPager::open(&case.db).unwrap();
    assert_eq!(
        p.read_page(2).unwrap(),
        &marker((m - 1) as u8)[..],
        "the latest of {m} accumulated commits resolves after reopen"
    );
}

#[test]
fn stale_base_page_cache_is_invalidated_after_peer_checkpoint() {
    // A connection must not serve a STALE base-image page after a PEER's checkpoint
    // backfilled a newer value for it and reset the WAL. Pins the db_generation
    // cache-invalidation in `LocalSnapshot::adopt`: without it, A keeps returning the
    // pre-checkpoint 0xE3 it cached from the db file. This is the two-connection
    // correctness property the suite otherwise left unwitnessed.
    let case = Case::new("stale-basecache");
    craft_wal_db(&case.db, 3); // page1(size3), page2=0xE2, page3=0xE3 (page 3 in the db file)
    let mut a = DiskPager::open(&case.db).unwrap();
    let mut b = DiskPager::open(&case.db).unwrap();

    // A reads page 3 from the db file, caching 0xE3 in its base-page cache.
    a.begin_read().unwrap();
    assert_eq!(a.read_page(3).unwrap(), &marker(page_seed(3))[..]); // 0xE3, now cached
    a.end_read().unwrap();

    // B overwrites page 3, commits, and checkpoints (full drain + reset): this
    // backfills page3=0xF3 into the db file and bumps the shared db_generation.
    b.begin().unwrap();
    b.write_page(3, &marker(0xF3)).unwrap();
    b.commit().unwrap();
    b.checkpoint(CheckpointMode::Passive).unwrap();
    assert_eq!(file_len(&case.wal), 32, "B fully drained and reset the WAL");

    // A's next read transaction must observe the NEW page 3: its stale cached 0xE3 is
    // dropped because the db_generation advanced under it.
    a.begin_read().unwrap();
    assert_eq!(
        a.read_page(3).unwrap(),
        &marker(0xF3)[..],
        "stale base-page cache invalidated after a peer's checkpoint backfilled a new value"
    );
    a.end_read().unwrap();
}

#[test]
fn corrupted_trailing_frame_is_ignored_on_reopen() {
    // Complements `torn_last_frame` (a short partial tail): here the trailing frame is
    // FULL-LENGTH but corrupted, so its cumulative checksum no longer matches. `scan`
    // must reject it (not accept a same-size-but-invalid frame) and fall back to the
    // last valid commit.
    let case = Case::new("corrupt-frame");
    craft_wal_db(&case.db, 2);
    {
        let mut p = DiskPager::open(&case.db).unwrap();
        p.begin().unwrap();
        p.write_page(2, &marker(0xD2)).unwrap();
        p.commit().unwrap(); // commit #1 (the one that must survive)
        p.begin().unwrap();
        p.write_page(2, &marker(0xD3)).unwrap();
        p.commit().unwrap(); // commit #2 (its trailing frame will be corrupted)
    }

    // Corrupt one byte inside the LAST frame of the -wal file. A WAL frame is a 24-byte
    // header + one page, so the last frame is the final `24 + PS` bytes; flipping any
    // byte in it (here the first page-data byte) breaks that frame's cumulative checksum.
    let stride = FRAME_STRIDE;
    let len = file_len(&case.wal);
    assert!(len >= WAL_HEADER_SIZE + 2 * stride, "expected at least two committed frames in the WAL");
    let last_frame_data = len - stride + FRAME_HEADER_SIZE;
    {
        use std::io::{Read, Seek, SeekFrom};
        let mut f = OpenOptions::new().read(true).write(true).open(&case.wal).unwrap();
        f.seek(SeekFrom::Start(last_frame_data)).unwrap();
        let mut byte = [0u8; 1];
        f.read_exact(&mut byte).unwrap();
        byte[0] ^= 0xFF;
        f.seek(SeekFrom::Start(last_frame_data)).unwrap();
        f.write_all(&byte).unwrap();
        f.sync_all().unwrap();
    }

    // Reopen: the corrupted trailing frame is rejected; the last VALID commit (#1) wins.
    let p = DiskPager::open(&case.db).unwrap();
    assert_eq!(
        p.read_page(2).unwrap(),
        &marker(0xD2)[..],
        "a corrupted full-length trailing frame is ignored; the prior commit survives"
    );
}

// ---------------------------------------------------------------------------
// LAZY, MODE-AWARE write lock (`begin_deferred`): a DEFERRED transaction pins a
// read snapshot only and defers the single write lock to its first write
// (lang_transaction §2.1/§2.2). These pin the concurrency behavior the eager
// `begin()` cannot express — a read-only DEFERRED transaction that does NOT
// block a concurrent writer, and a first-write upgrade that is BUSY when the
// lock is held or the snapshot is stale.
// ---------------------------------------------------------------------------

#[test]
fn deferred_read_txn_does_not_block_a_concurrent_commit() {
    // The headline fix: a `BEGIN DEFERRED; SELECT ...` holds NO write lock, so a
    // second connection can still `begin()` + commit. With the old EAGER begin,
    // A's begin would hold the write lock and B's begin would be BUSY. A also keeps
    // its pinned snapshot while B commits (WAL snapshot isolation), then releases
    // only its reader mark at commit.
    let case = Case::new("deferred-noblock");
    craft_wal_db(&case.db, 3);
    let mut a = DiskPager::open(&case.db).unwrap();
    let mut b = DiskPager::open(&case.db).unwrap();

    // Commit #1 (eager) establishes page 2 = 0xA1.
    a.begin().unwrap();
    a.write_page(2, &marker(0xA1)).unwrap();
    a.commit().unwrap();

    // A opens a DEFERRED transaction: a read snapshot, no write lock.
    a.begin_deferred().unwrap();
    assert_eq!(a.read_page(2).unwrap(), &marker(0xA1)[..], "deferred read sees the committed value");

    // B can take the write lock and commit WHILE A's deferred txn is open — the
    // property the eager begin cannot provide.
    b.begin().unwrap();
    b.write_page(2, &marker(0xB2)).unwrap();
    b.commit().unwrap();

    // A still observes its pinned snapshot, not B's newer commit.
    assert_eq!(
        a.read_page(2).unwrap(),
        &marker(0xA1)[..],
        "a reader in a DEFERRED transaction keeps its snapshot while another connection commits"
    );

    // A's read-only DEFERRED commit applies an empty overlay and releases the reader
    // mark; a fresh read then advances to B's commit.
    a.commit().unwrap();
    a.begin_read().unwrap();
    assert_eq!(a.read_page(2).unwrap(), &marker(0xB2)[..], "a new read sees the peer's commit");
    a.end_read().unwrap();
}

#[test]
fn deferred_first_write_acquires_lock_then_serializes_a_second_writer() {
    // The lazy upgrade: a DEFERRED txn takes no lock at begin, but its FIRST write
    // acquires the single write lock — after which a second connection's write is
    // BUSY (two writers serialize, never interleave).
    let case = Case::new("deferred-upgrade");
    craft_wal_db(&case.db, 2);
    let mut a = DiskPager::open(&case.db).unwrap();
    let mut b = DiskPager::open(&case.db).unwrap();

    a.begin_deferred().unwrap();
    // Before the first write, the lock is free: B could take it. We do NOT here (that
    // is the other test); instead A writes, which upgrades and takes the lock.
    a.write_page(2, &marker(0xA2)).unwrap();

    // Now A holds the write lock, so B's begin is BUSY.
    let err = b.begin().unwrap_err();
    assert!(is_busy(&err), "second writer is BUSY once the deferred txn upgraded: {err}");

    // A commits (releasing the lock); only then can B write.
    a.commit().unwrap();
    b.begin().unwrap();
    b.write_page(2, &marker(0xB2)).unwrap();
    b.commit().unwrap();
    b.begin_read().unwrap();
    assert_eq!(b.read_page(2).unwrap(), &marker(0xB2)[..], "B's serialized write landed");
    b.end_read().unwrap();
}

#[test]
fn deferred_first_write_is_busy_when_another_holds_the_write_lock() {
    // lang_transaction §2.1: upgrading a read transaction to a write transaction is
    // BUSY when another connection is already writing. B's DEFERRED begin succeeds
    // (read snapshot only), but its first write cannot take the held lock. The BUSY
    // leaves B's transaction read-only with its overlay untouched (clean, rollback-safe).
    let case = Case::new("deferred-busy-held");
    craft_wal_db(&case.db, 2);
    let mut a = DiskPager::open(&case.db).unwrap();
    let mut b = DiskPager::open(&case.db).unwrap();

    // A takes the single write lock (eager) and stages a write.
    a.begin().unwrap();
    a.write_page(2, &marker(0xA2)).unwrap();

    // B opens a DEFERRED txn — allowed even while A writes (no lock needed to read).
    b.begin_deferred().unwrap();
    assert_eq!(b.read_page(2).unwrap(), &marker(page_seed(2))[..], "B reads its pre-write snapshot");

    // B's first write tries to upgrade, but A holds the lock -> BUSY.
    let err = b.write_page(2, &marker(0xB2)).unwrap_err();
    assert!(is_busy(&err), "deferred first-write is BUSY while a peer holds the write lock: {err}");

    // B is still read-only: its rollback releases only the reader mark (no lock to drop),
    // and A can finish cleanly.
    b.rollback().unwrap();
    a.commit().unwrap();

    // After A's commit, B can retry and succeed (the lock is free now).
    b.begin_deferred().unwrap();
    b.write_page(2, &marker(0xB3)).unwrap();
    b.commit().unwrap();
    b.begin_read().unwrap();
    assert_eq!(b.read_page(2).unwrap(), &marker(0xB3)[..], "B's retried write landed after A freed the lock");
    b.end_read().unwrap();
}

#[test]
fn deferred_first_write_is_busy_when_snapshot_is_stale() {
    // lang_transaction §2.1: even if no connection currently HOLDS the write lock, a
    // read transaction that pinned an OLD snapshot cannot upgrade once another
    // connection has committed past it — the write would be built on a stale base, so
    // it is BUSY. B keeps its snapshot; it must re-begin to write on the new state.
    let case = Case::new("deferred-busy-stale");
    craft_wal_db(&case.db, 2);
    let mut a = DiskPager::open(&case.db).unwrap();
    let mut b = DiskPager::open(&case.db).unwrap();

    // Commit #1 by A: page 2 = 0xA1.
    a.begin().unwrap();
    a.write_page(2, &marker(0xA1)).unwrap();
    a.commit().unwrap();

    // B pins a DEFERRED read snapshot at commit #1.
    b.begin_deferred().unwrap();
    assert_eq!(b.read_page(2).unwrap(), &marker(0xA1)[..]);

    // Commit #2 by A advances the WAL past B's snapshot; A releases the lock cleanly.
    a.begin().unwrap();
    a.write_page(2, &marker(0xA2)).unwrap();
    a.commit().unwrap();

    // The write lock is FREE now, but B's snapshot is stale (mxFrame advanced), so B's
    // first-write upgrade is BUSY rather than silently writing on the old base.
    let err = b.write_page(2, &marker(0xB2)).unwrap_err();
    assert!(is_busy(&err), "deferred upgrade over a peer's intervening commit is BUSY: {err}");

    // B still holds its snapshot and can roll back; a fresh DEFERRED begin sees #2 and
    // can then write.
    assert_eq!(b.read_page(2).unwrap(), &marker(0xA1)[..], "B keeps its snapshot after the BUSY");
    b.rollback().unwrap();
    b.begin_deferred().unwrap();
    assert_eq!(b.read_page(2).unwrap(), &marker(0xA2)[..], "a re-begun deferred txn sees the latest");
    b.write_page(2, &marker(0xB2)).unwrap();
    b.commit().unwrap();
    b.begin_read().unwrap();
    assert_eq!(b.read_page(2).unwrap(), &marker(0xB2)[..], "B's write on the refreshed snapshot landed");
    b.end_read().unwrap();
}

#[test]
fn deferred_begin_matches_eager_for_a_single_connection() {
    // Single-connection equivalence: a DEFERRED transaction that writes behaves
    // exactly like an eager one — the lazy upgrade is invisible with no contender.
    // Guards against the deferred path corrupting the ordinary write/commit flow.
    let case = Case::new("deferred-single");
    craft_wal_db(&case.db, 2);
    let mut a = DiskPager::open(&case.db).unwrap();

    a.begin_deferred().unwrap();
    a.write_page(2, &marker(0x2A)).unwrap();
    let three = a.allocate_page().unwrap();
    assert_eq!(three, 3, "allocation inside a deferred txn grows the db");
    a.write_page(3, &marker(0x3A)).unwrap();
    a.commit().unwrap();

    // The writer sees its own commit immediately, and it survives a reopen.
    assert_eq!(a.page_count().unwrap(), 3);
    assert_eq!(a.read_page(2).unwrap(), &marker(0x2A)[..]);
    assert_eq!(a.read_page(3).unwrap(), &marker(0x3A)[..]);
    drop(a);

    let p = DiskPager::open(&case.db).unwrap();
    assert_eq!(p.page_count().unwrap(), 3, "deferred-txn commit survives reopen");
    assert_eq!(p.read_page(2).unwrap(), &marker(0x2A)[..]);
    assert_eq!(p.read_page(3).unwrap(), &marker(0x3A)[..]);
}

#[test]
fn deferred_rollback_after_upgrade_releases_the_write_lock() {
    // A DEFERRED txn that upgraded (took the lock) and then ROLLS BACK must release
    // the write lock, so a sibling can proceed — the rollback-path counterpart of the
    // commit release, for the lock_held branch specifically.
    let case = Case::new("deferred-rollback-lock");
    craft_wal_db(&case.db, 2);
    let mut a = DiskPager::open(&case.db).unwrap();
    let mut b = DiskPager::open(&case.db).unwrap();

    a.begin_deferred().unwrap();
    a.write_page(2, &marker(0xA2)).unwrap(); // upgrade: A now holds the lock
    let err = b.begin().unwrap_err();
    assert!(is_busy(&err), "sibling is BUSY while the upgraded deferred txn holds the lock");
    a.rollback().unwrap(); // must release the write lock

    // B can now take the lock and commit; the rolled-back write left no trace.
    b.begin().unwrap();
    b.write_page(2, &marker(0xB2)).unwrap();
    b.commit().unwrap();
    b.begin_read().unwrap();
    assert_eq!(
        b.read_page(2).unwrap(),
        &marker(0xB2)[..],
        "the rolled-back deferred write left no trace and released the lock"
    );
    b.end_read().unwrap();
}

#[test]
fn deferred_first_allocate_is_busy_when_another_holds_the_write_lock() {
    // Witnesses the lazy upgrade in `grow_one` (the ALLOCATE path), not just `stage_write`
    // (the write_page path the other BUSY tests use). A DEFERRED txn whose FIRST mutation
    // is an allocation must still take the single write lock, so it is BUSY while a peer
    // holds it — never a silent grow that skipped the lock. On a freshly crafted db (valid
    // header, empty freelist) `allocate_page` routes straight to `cow.grow_one`, so the
    // BUSY here can only come from `grow_one`'s `ensure_write_lock`: drop that call and the
    // allocate would succeed instead, reddening this test.
    let case = Case::new("deferred-busy-alloc");
    craft_wal_db(&case.db, 2);
    let mut a = DiskPager::open(&case.db).unwrap();
    let mut b = DiskPager::open(&case.db).unwrap();

    // A holds the single write lock (eager) without committing.
    a.begin().unwrap();
    a.write_page(2, &marker(0xA2)).unwrap();

    // B's DEFERRED begin only reads; its FIRST mutation is an allocate, which must
    // upgrade -> BUSY while A holds the lock.
    b.begin_deferred().unwrap();
    let err = b.allocate_page().unwrap_err();
    assert!(is_busy(&err), "deferred first-allocate is BUSY while a peer holds the write lock: {err}");

    // The BUSY left B read-only with its state pristine: rollback drops only the reader
    // mark, and A finishes cleanly.
    b.rollback().unwrap();
    a.commit().unwrap();

    // Once A frees the lock, B's deferred allocate upgrades and lands.
    b.begin_deferred().unwrap();
    let id = b.allocate_page().unwrap();
    assert_eq!(id, 3, "the retried allocation grows the db");
    b.commit().unwrap();
    assert_eq!(b.page_count().unwrap(), 3, "B's serialized allocation landed");
}

#[test]
fn deferred_first_page_mut_is_busy_when_another_holds_the_write_lock() {
    // Witnesses the lazy upgrade in `page_mut` (the IN-PLACE mutation path), not just
    // `stage_write`. A DEFERRED txn whose FIRST mutation is a `page_mut` must still take
    // the write lock, so it is BUSY while a peer holds it. Drop `ensure_write_lock` from
    // `page_mut` and this would hand back a writable borrow with no lock taken.
    let case = Case::new("deferred-busy-pagemut");
    craft_wal_db(&case.db, 2);
    let mut a = DiskPager::open(&case.db).unwrap();
    let mut b = DiskPager::open(&case.db).unwrap();

    a.begin().unwrap();
    a.write_page(2, &marker(0xA2)).unwrap();

    b.begin_deferred().unwrap();
    let err = b.page_mut(2).map(|_| ()).unwrap_err();
    assert!(is_busy(&err), "deferred first page_mut is BUSY while a peer holds the write lock: {err}");

    // Clean read-only rollback, then B's in-place edit lands once A frees the lock.
    b.rollback().unwrap();
    a.commit().unwrap();

    b.begin_deferred().unwrap();
    b.page_mut(2).unwrap()[0] = 0xB2;
    b.commit().unwrap();
    b.begin_read().unwrap();
    assert_eq!(b.read_page(2).unwrap()[0], 0xB2, "B's serialized in-place edit landed");
    b.end_read().unwrap();
}

#[test]
fn deferred_read_only_commit_releases_reader_mark_for_checkpoint() {
    // Witnesses the reader-mark release on a read-only DEFERRED `commit`. A read-only
    // deferred txn pins a reader mark at begin (via `begin_read`) and takes NO write lock;
    // its commit must release that mark. A checkpoint resets the WAL only when
    // `readers.is_empty()`, so a leaked mark would forbid the reset forever (the log
    // drains but never truncates). Drop the `read_pinned` branch's `end_read()` from
    // `commit` and the `wal_len == 32` assertion below goes red.
    let case = Case::new("deferred-ro-commit-release");
    craft_wal_db(&case.db, 2);
    let mut a = DiskPager::open(&case.db).unwrap();

    // A committed frame lives in the WAL before the checkpoint.
    a.begin().unwrap();
    a.write_page(2, &marker(0xA2)).unwrap();
    a.commit().unwrap();
    assert!(file_len(&case.wal) > 32, "a committed frame lives in the WAL before the checkpoint");

    // A read-only DEFERRED transaction: pins a reader mark, takes no write lock, and must
    // release the mark at commit (its overlay is empty, so nothing is appended).
    a.begin_deferred().unwrap();
    assert_eq!(a.read_page(2).unwrap(), &marker(0xA2)[..], "the deferred read sees the committed value");
    a.commit().unwrap();

    // With the mark released, the checkpoint fully drains AND resets the WAL.
    a.checkpoint(CheckpointMode::Passive).unwrap();
    assert_eq!(
        file_len(&case.wal),
        32,
        "a read-only deferred commit released its reader mark, so the checkpoint resets the WAL"
    );
    assert_eq!(a.read_page(2).unwrap(), &marker(0xA2)[..], "data intact after drain + reset");
}

#[test]
fn deferred_read_only_rollback_releases_reader_mark_for_checkpoint() {
    // The rollback counterpart of the commit test above: a read-only DEFERRED `rollback`
    // must also release the reader mark it pinned at begin, or a later checkpoint can
    // never reset the WAL. Drop the `read_pinned` branch's `end_read()` from `rollback`
    // and this goes red.
    let case = Case::new("deferred-ro-rollback-release");
    craft_wal_db(&case.db, 2);
    let mut a = DiskPager::open(&case.db).unwrap();

    a.begin().unwrap();
    a.write_page(2, &marker(0xA2)).unwrap();
    a.commit().unwrap();
    assert!(file_len(&case.wal) > 32, "a committed frame lives in the WAL before the checkpoint");

    // A read-only DEFERRED transaction that rolls back (no write ever attempted) must
    // still release its reader mark.
    a.begin_deferred().unwrap();
    assert_eq!(a.read_page(2).unwrap(), &marker(0xA2)[..], "the deferred read sees the committed value");
    a.rollback().unwrap();

    a.checkpoint(CheckpointMode::Passive).unwrap();
    assert_eq!(
        file_len(&case.wal),
        32,
        "a read-only deferred rollback released its reader mark, so the checkpoint resets the WAL"
    );
    assert_eq!(a.read_page(2).unwrap(), &marker(0xA2)[..], "data intact after drain + reset");
}

// ---------------------------------------------------------------------------
// CHECKPOINT MODE matrix (pragma.html #pragma_wal_checkpoint). The tests above
// only ever drive PASSIVE and discard the report; these pin the mode-specific
// contracts — TRUNCATE's zero-byte `-wal`, RESTART's 32-byte reset, NOOP's
// no-drain, and the `(busy, log, checkpointed)` result row across the RESTART/
// TRUNCATE/FULL busy gates that PASSIVE never reaches. Frame counts are read
// from the file via `wal_frame_count` so the assertions pin the
// observable (the pragma's result row + the post-checkpoint `-wal` state)
// rather than the exact number of frames a commit happened to append.
// ---------------------------------------------------------------------------

#[test]
fn truncate_zeroes_the_wal_and_reports_full_drain() {
    // pragma.html: TRUNCATE "works the same way as RESTART with the addition that the WAL file
    // is truncated to zero bytes upon successful completion." So a TRUNCATE with no active
    // reader/writer must leave the `-wal` at 0 bytes — NOT the 32-byte bare header a
    // PASSIVE/RESTART reset leaves — and report a full, non-busy drain.
    let case = Case::new("truncate-zero");
    craft_wal_db(&case.db, 2);
    let mut p = DiskPager::open(&case.db).unwrap();

    // Three one-page commits accumulate frames in the WAL with no active reader.
    for i in 0..3u32 {
        p.begin().unwrap();
        p.write_page(2, &marker(0xC0u8.wrapping_add(i as u8))).unwrap();
        p.commit().unwrap();
    }
    let last = 0xC0u8.wrapping_add(2);
    let n = wal_frame_count(&case.wal);
    assert!(n >= 3, "expected at least three committed frames before the checkpoint, got {n}");
    assert!(file_len(&case.wal) > 32, "frames accumulated in the -wal before the checkpoint");

    let rep = p.checkpoint(CheckpointMode::Truncate).unwrap();

    assert_eq!(
        file_len(&case.wal),
        0,
        "TRUNCATE must zero the -wal file, not merely reset it (pragma.html #pragma_wal_checkpoint)"
    );
    assert!(!rep.busy, "TRUNCATE with no reader or writer completes, so busy must be false (pragma.html)");
    assert_eq!(rep.log, Some(n), "log is the count of frames written to the WAL (pragma.html log column)");
    assert_eq!(
        rep.checkpointed,
        Some(n),
        "a full drain moved every frame back into the db (pragma.html checkpointed column)"
    );
    // The last commit's value was drained into the db file and is still readable.
    assert_eq!(
        p.read_page(2).unwrap(),
        &marker(last)[..],
        "the last commit's value survives the drain into the db file"
    );

    // The WAL is reusable: the next writer re-initializes a fresh header and appends.
    p.begin().unwrap();
    p.write_page(2, &marker(0xAB)).unwrap();
    p.commit().unwrap();
    assert!(file_len(&case.wal) > 32, "a post-TRUNCATE commit re-initializes a fresh WAL header and appends");
    p.begin_read().unwrap();
    assert_eq!(p.read_page(2).unwrap(), &marker(0xAB)[..], "a commit + read round-trips after TRUNCATE");
    p.end_read().unwrap();
}

#[test]
fn restart_resets_to_bare_header_not_zero() {
    // pragma.html: RESTART ensures "the next client to write to the database file restarts the
    // log file from the beginning" but — unlike TRUNCATE — does not truncate the `-wal` to zero.
    // A successful RESTART with no reader must leave a bare 32-byte WAL header, pinning RESTART
    // apart from TRUNCATE.
    let case = Case::new("restart-reset");
    craft_wal_db(&case.db, 2);
    let mut p = DiskPager::open(&case.db).unwrap();

    for i in 0..3u32 {
        p.begin().unwrap();
        p.write_page(2, &marker(0xC0u8.wrapping_add(i as u8))).unwrap();
        p.commit().unwrap();
    }
    let last = 0xC0u8.wrapping_add(2);
    let n = wal_frame_count(&case.wal);

    let rep = p.checkpoint(CheckpointMode::Restart).unwrap();

    assert_eq!(
        file_len(&case.wal),
        32,
        "RESTART resets to a bare 32-byte header, not zero bytes — only TRUNCATE zeroes the -wal (pragma.html)"
    );
    assert!(!rep.busy, "RESTART with no reader or writer completes, so busy must be false (pragma.html)");
    assert_eq!(rep.log, Some(n), "log is the frame count written to the WAL (pragma.html)");
    assert_eq!(rep.checkpointed, Some(n), "a full drain moved every frame back into the db (pragma.html)");
    assert_eq!(p.read_page(2).unwrap(), &marker(last)[..], "the last commit survives the drain");
}

#[test]
fn noop_reports_counts_without_draining_then_passive_drains() {
    // pragma.html: NOOP "does not checkpoint any frames. It is used to obtain the returned
    // values only." So NOOP must copy nothing, never reset, and report the current counts; a
    // following PASSIVE must then still drain normally.
    let case = Case::new("noop-counts");
    craft_wal_db(&case.db, 2);
    let mut p = DiskPager::open(&case.db).unwrap();

    for i in 0..3u32 {
        p.begin().unwrap();
        p.write_page(2, &marker(0xC0u8.wrapping_add(i as u8))).unwrap();
        p.commit().unwrap();
    }
    let last = 0xC0u8.wrapping_add(2);
    let n = wal_frame_count(&case.wal);
    let wal_len_before = file_len(&case.wal);
    let db_len_before = file_len(&case.db);

    let rep = p.checkpoint(CheckpointMode::Noop).unwrap();

    assert_eq!(
        file_len(&case.wal),
        wal_len_before,
        "NOOP copies no frame and never resets: the -wal is byte-length unchanged (pragma.html)"
    );
    assert_eq!(file_len(&case.db), db_len_before, "NOOP writes nothing back into the db file (pragma.html)");
    assert!(!rep.busy, "NOOP is never busy — it only obtains the returned values (pragma.html)");
    assert_eq!(rep.log, Some(n), "NOOP still reports the current WAL frame count (pragma.html log column)");
    assert_eq!(
        rep.checkpointed,
        Some(0),
        "nothing has been backfilled yet, so NOOP reports checkpointed = 0 (pragma.html checkpointed column)"
    );

    // A following PASSIVE checkpoint still drains and resets normally.
    let rep2 = p.checkpoint(CheckpointMode::Passive).unwrap();
    assert_eq!(file_len(&case.wal), 32, "a PASSIVE checkpoint after NOOP still drains and resets the WAL");
    assert!(!rep2.busy, "PASSIVE is never busy (pragma.html)");
    assert_eq!(rep2.log, Some(n), "PASSIVE reports the frame count NOOP left in place");
    assert_eq!(rep2.checkpointed, Some(n), "PASSIVE drained every frame back into the db");
    assert_eq!(p.read_page(2).unwrap(), &marker(last)[..], "data intact after NOOP then PASSIVE drain");
}

#[test]
fn noop_reports_the_pre_existing_partial_backfill_count() {
    // pragma.html: NOOP "is used to obtain the returned values only", and the checkpointed column
    // is "the number of pages ... successfully moved back into the database file". So NOOP must
    // report the frames ALREADY backfilled (nBackfill), not a constant 0. This drives the
    // `nbackfill.min(log_frames)` value with a partial backfill strictly between 0 and the log
    // size — the `noop_reports_counts_without_draining_then_passive_drains` case (nBackfill == 0)
    // cannot tell that formula apart from a hardcoded 0.
    let case = Case::new("noop-partial");
    craft_wal_db(&case.db, 2);
    let mut a = DiskPager::open(&case.db).unwrap();
    let mut b = DiskPager::open(&case.db).unwrap();

    // Commit #1, then B pins a read snapshot at that ceiling; commit #2 lands beyond it.
    a.begin().unwrap();
    a.write_page(2, &marker(0xA1)).unwrap();
    a.commit().unwrap();
    b.begin_read().unwrap();
    assert_eq!(b.read_page(2).unwrap(), &marker(0xA1)[..]);
    let mark = wal_frame_count(&case.wal);
    a.begin().unwrap();
    a.write_page(2, &marker(0xA2)).unwrap();
    a.commit().unwrap();
    let total = wal_frame_count(&case.wal);
    assert!(mark > 0 && mark < total, "partial-backfill precondition: 0 < mark ({mark}) < total ({total})");

    // A reader-bounded PASSIVE drain backfills exactly up to the reader's mark (nBackfill = mark).
    let drain = a.checkpoint(CheckpointMode::Passive).unwrap();
    assert_eq!(drain.checkpointed, Some(mark), "the reader-bounded PASSIVE drain backfilled up to the reader's mark");

    // NOOP now reports that existing backfill count (nBackfill.min(log)), copying nothing more.
    let wal_len_before = file_len(&case.wal);
    let rep = a.checkpoint(CheckpointMode::Noop).unwrap();
    assert!(!rep.busy, "NOOP is never busy (pragma.html)");
    assert_eq!(rep.log, Some(total), "NOOP reports the full WAL frame count (pragma.html log column)");
    assert_eq!(
        rep.checkpointed,
        Some(mark),
        "NOOP reports the frames already backfilled (nBackfill), not a constant 0 (pragma.html checkpointed column)"
    );
    assert_eq!(file_len(&case.wal), wal_len_before, "NOOP copies nothing further and does not reset");
}

#[test]
fn restart_is_busy_and_defers_reset_while_reader_active() {
    // pragma.html: RESTART "blocks (calls the busy-handler callback) until all readers are
    // finished with the log file", and the result row's first column is 1 iff a RESTART/FULL/
    // TRUNCATE was blocked from completing. So while a reader pins the log behind mxFrame,
    // RESTART must report busy=true and must NOT reset the WAL; after the reader releases it
    // completes with busy=false and resets.
    let case = Case::new("restart-busy-reader");
    craft_wal_db(&case.db, 2);
    let mut a = DiskPager::open(&case.db).unwrap();
    let mut b = DiskPager::open(&case.db).unwrap();

    // Commit #1, then B pins a read snapshot at that ceiling; commit #2 lands beyond it.
    a.begin().unwrap();
    a.write_page(2, &marker(0xA1)).unwrap();
    a.commit().unwrap();
    b.begin_read().unwrap();
    assert_eq!(b.read_page(2).unwrap(), &marker(0xA1)[..]);
    let mark = wal_frame_count(&case.wal); // the frames B pins (its mxFrame at begin_read)
    a.begin().unwrap();
    a.write_page(2, &marker(0xA2)).unwrap();
    a.commit().unwrap();
    let total = wal_frame_count(&case.wal);
    assert!(total > mark, "commit #2 advanced mxFrame beyond the reader's mark");

    // RESTART while B is behind: busy, a partial drain bounded by the reader's mark, no reset.
    let rep = a.checkpoint(CheckpointMode::Restart).unwrap();
    assert!(rep.busy, "RESTART is busy while a reader is not yet finished with the log (pragma.html)");
    assert!(file_len(&case.wal) > 32, "RESTART must NOT reset the WAL while a reader is behind mxFrame");
    assert_eq!(rep.log, Some(total), "log reports every frame in the WAL, reader-pinned or not (pragma.html)");
    assert_eq!(
        rep.checkpointed,
        Some(mark),
        "the bounded drain backfilled only up to the reader's mark (pragma.html checkpointed column)"
    );
    assert_eq!(
        b.read_page(2).unwrap(),
        &marker(0xA1)[..],
        "the reader's snapshot is unaffected by the bounded checkpoint"
    );

    // Once the reader releases, the same RESTART completes and resets the WAL.
    b.end_read().unwrap();
    let rep2 = a.checkpoint(CheckpointMode::Restart).unwrap();
    assert!(!rep2.busy, "RESTART completes (busy=false) once no reader is behind (pragma.html)");
    assert_eq!(file_len(&case.wal), 32, "RESTART resets the WAL to a bare header once no reader is behind");
    assert_eq!(rep2.log, Some(total), "log still counts the frames that were in the WAL");
    assert_eq!(rep2.checkpointed, Some(total), "the completing drain backfilled every frame");
    assert_eq!(a.read_page(2).unwrap(), &marker(0xA2)[..], "the latest data survives the drain + reset");
}

#[test]
fn truncate_is_busy_and_defers_zeroing_while_reader_active() {
    // pragma.html: TRUNCATE works as RESTART (blocks until readers finish with the log) plus
    // zeroing the `-wal` on success. While a reader is behind, TRUNCATE must be busy and must
    // NOT zero (nor reset) the file; after the reader releases it zeroes.
    let case = Case::new("truncate-busy-reader");
    craft_wal_db(&case.db, 2);
    let mut a = DiskPager::open(&case.db).unwrap();
    let mut b = DiskPager::open(&case.db).unwrap();

    a.begin().unwrap();
    a.write_page(2, &marker(0xA1)).unwrap();
    a.commit().unwrap();
    b.begin_read().unwrap();
    assert_eq!(b.read_page(2).unwrap(), &marker(0xA1)[..]);
    let mark = wal_frame_count(&case.wal);
    a.begin().unwrap();
    a.write_page(2, &marker(0xA2)).unwrap();
    a.commit().unwrap();
    let total = wal_frame_count(&case.wal);

    // TRUNCATE while B is behind: busy, and the file is neither zeroed nor reset.
    let rep = a.checkpoint(CheckpointMode::Truncate).unwrap();
    assert!(rep.busy, "TRUNCATE is busy while a reader is behind mxFrame (pragma.html)");
    assert!(
        file_len(&case.wal) > 32,
        "TRUNCATE must neither zero nor reset the WAL while a reader is behind (pragma.html)"
    );
    assert_eq!(rep.log, Some(total), "log reports every frame in the WAL (pragma.html)");
    assert_eq!(rep.checkpointed, Some(mark), "the bounded drain backfilled only up to the reader's mark (pragma.html)");

    // Once the reader releases, TRUNCATE completes and zeroes the -wal.
    b.end_read().unwrap();
    let rep2 = a.checkpoint(CheckpointMode::Truncate).unwrap();
    assert!(!rep2.busy, "TRUNCATE completes (busy=false) once no reader is behind (pragma.html)");
    assert_eq!(file_len(&case.wal), 0, "TRUNCATE zeroes the -wal once no reader is behind (pragma.html)");
    assert_eq!(a.read_page(2).unwrap(), &marker(0xA2)[..], "the latest data survives the drain + truncate");
}

#[test]
fn full_is_busy_while_a_writer_holds_the_write_lock() {
    // pragma.html: FULL "blocks (invokes the busy-handler callback) until there is no database
    // writer ...". So while another connection holds the write lock, a FULL checkpoint must
    // report busy=true (even though it can still backfill the already-committed frames); once
    // that writer commits, FULL completes with busy=false. This drives the `Full => write_locked`
    // sub-condition of the busy gate that PASSIVE-only tests never reach.
    let case = Case::new("full-busy-writer");
    craft_wal_db(&case.db, 2);
    let mut a = DiskPager::open(&case.db).unwrap();
    let mut b = DiskPager::open(&case.db).unwrap();

    // A commit puts frames in the WAL (satisfying the log_frames>0 guard).
    a.begin().unwrap();
    a.write_page(2, &marker(0xA1)).unwrap();
    a.commit().unwrap();
    let n = wal_frame_count(&case.wal);

    // B holds the single write lock (open write transaction, not yet committed).
    b.begin().unwrap();
    b.write_page(2, &marker(0xB2)).unwrap();

    // A's FULL checkpoint is blocked by the active writer.
    let rep = a.checkpoint(CheckpointMode::Full).unwrap();
    assert!(rep.busy, "FULL is busy while a writer holds the write lock (pragma.html: FULL blocks until no writer)");
    assert!(file_len(&case.wal) > 32, "FULL must not reset the WAL while a writer holds the lock");
    assert_eq!(rep.log, Some(n), "log counts the committed frames in the WAL (B's write is not yet committed)");
    assert_eq!(
        rep.checkpointed,
        Some(n),
        "a blocked FULL still backfills the already-committed frames; it just cannot reset (pragma.html)"
    );

    // Once B commits and releases the lock, FULL completes and resets.
    b.commit().unwrap();
    let total = wal_frame_count(&case.wal);
    assert!(total > n, "B's commit appended more frames");
    let rep2 = a.checkpoint(CheckpointMode::Full).unwrap();
    assert!(!rep2.busy, "FULL completes (busy=false) once no writer holds the lock (pragma.html)");
    assert_eq!(file_len(&case.wal), 32, "the completing FULL drains and resets the WAL");
    assert_eq!(rep2.log, Some(total), "log counts every frame now in the WAL (pragma.html)");
    assert_eq!(rep2.checkpointed, Some(total), "FULL drained every frame back into the db (pragma.html)");
}

#[test]
fn full_is_busy_while_a_reader_pins_the_log_behind_mxframe() {
    // The other half of the FULL busy gate (`Full => ... || !res.complete`): pragma.html says
    // FULL blocks until "all readers are reading from the most recent database snapshot". A
    // reader pinned behind mxFrame forces an incomplete drain, so FULL must report busy=true;
    // once the reader releases, the drain completes and busy=false.
    let case = Case::new("full-busy-reader");
    craft_wal_db(&case.db, 2);
    let mut a = DiskPager::open(&case.db).unwrap();
    let mut b = DiskPager::open(&case.db).unwrap();

    a.begin().unwrap();
    a.write_page(2, &marker(0xA1)).unwrap();
    a.commit().unwrap();
    b.begin_read().unwrap();
    assert_eq!(b.read_page(2).unwrap(), &marker(0xA1)[..]);
    let mark = wal_frame_count(&case.wal);
    a.begin().unwrap();
    a.write_page(2, &marker(0xA2)).unwrap();
    a.commit().unwrap();
    let total = wal_frame_count(&case.wal);

    // FULL cannot drain past the reader's mark, so it is blocked from completing.
    let rep = a.checkpoint(CheckpointMode::Full).unwrap();
    assert!(
        rep.busy,
        "FULL is busy while a reader pins the log behind mxFrame (pragma.html: readers must be on the latest snapshot)"
    );
    assert_eq!(rep.log, Some(total), "log reports every frame in the WAL (pragma.html)");
    assert_eq!(rep.checkpointed, Some(mark), "FULL backfilled only up to the reader's mark (pragma.html)");

    // Once the reader is gone, FULL completes.
    b.end_read().unwrap();
    let rep2 = a.checkpoint(CheckpointMode::Full).unwrap();
    assert!(!rep2.busy, "FULL completes once the reader is off the log (pragma.html)");
    assert_eq!(rep2.checkpointed, Some(total), "the completing FULL drained every frame");
}

#[test]
fn empty_wal_checkpoint_is_never_busy_for_any_mode() {
    // pragma.html: the result row's busy column is 1 only if a checkpoint "was blocked from
    // completing". With no committed frames there is nothing to block on, so EVERY mode reports
    // busy=false and log=0 (checkpointed=0). Pins the `log_frames > 0` gate for all modes.
    for mode in [
        CheckpointMode::Passive,
        CheckpointMode::Full,
        CheckpointMode::Restart,
        CheckpointMode::Truncate,
        CheckpointMode::Noop,
    ] {
        let case = Case::new(&format!("empty-ckpt-{mode:?}"));
        craft_wal_db(&case.db, 2);
        let mut p = DiskPager::open(&case.db).unwrap();

        let rep = p.checkpoint(mode).unwrap();
        assert!(!rep.busy, "an empty-WAL checkpoint is never busy for {mode:?} (pragma.html log_frames>0 gate)");
        assert_eq!(rep.log, Some(0), "an empty WAL has 0 frames for {mode:?} (pragma.html log column)");
        assert_eq!(rep.checkpointed, Some(0), "nothing to checkpoint in an empty WAL for {mode:?}");
    }
}

#[test]
fn checkpoint_on_the_empty_post_reset_wal_never_enters_the_reset_branch() {
    // Regression guard for a CROSS-CRATE invariant the checkpoint reset path relies
    // on. A reset (which rewrites/zeroes the WAL and, on TRUNCATE, drops the in-memory
    // header to `None`) only runs when the drain is COMPLETE. `wal_checkpoint` reports
    // `complete` ONLY for a non-empty log (`mx != 0 && safe == mx`, in minisqlite-wal),
    // so an empty WAL is never complete ⇒ the reset branch — which reads the now-`None`
    // header — is never re-entered before the next write re-initializes it. If a future
    // change in minisqlite-wal ever let an empty WAL report `complete`, the pager would
    // panic in that branch; because this pager is the linked library, that
    // panic would abort the embedding process. This pins the seam so such drift fails a
    // cargo-test here (loudly, in dev) instead.
    //
    // The sibling `empty_wal_checkpoint_is_never_busy_for_any_mode` only covers a
    // header=`None` reached at OPEN. This covers header=`None` reached at RUNTIME via a
    // TRUNCATE reset, plus that the store stays usable afterward.
    for mode in [
        CheckpointMode::Passive,
        CheckpointMode::Full,
        CheckpointMode::Restart,
        CheckpointMode::Truncate,
        CheckpointMode::Noop,
    ] {
        let case = Case::new(&format!("post-reset-ckpt-{mode:?}"));
        craft_wal_db(&case.db, 2);
        let mut p = DiskPager::open(&case.db).unwrap();

        // One commit, then TRUNCATE: a complete drain zeroes the -wal and drops the
        // in-memory header to `None` (walstore's TRUNCATE reset branch).
        p.begin().unwrap();
        p.write_page(2, &marker(0x5A)).unwrap();
        p.commit().unwrap();
        let first = p.checkpoint(CheckpointMode::Truncate).unwrap();
        assert!(!first.busy, "the initial TRUNCATE with no reader/writer completes");
        assert_eq!(file_len(&case.wal), 0, "TRUNCATE zeroes the -wal, leaving the header None");

        // Re-checkpoint the now-empty (header=None) WAL BEFORE any new write. This must
        // be a clean no-op — never a panic on the reset branch — for every mode.
        let rep = p.checkpoint(mode).unwrap();
        assert!(!rep.busy, "a checkpoint on the empty post-reset WAL is never busy for {mode:?}");
        assert_eq!(rep.log, Some(0), "no frames remain after the reset for {mode:?} (pragma.html log column)");
        assert_eq!(rep.checkpointed, Some(0), "nothing to drain on the empty post-reset WAL for {mode:?}");
        assert_eq!(file_len(&case.wal), 0, "a no-op checkpoint leaves the zeroed WAL untouched for {mode:?}");

        // The header=None state was a transient empty, not a corrupt store: the next
        // write re-initializes a fresh header and the value round-trips.
        p.begin().unwrap();
        p.write_page(2, &marker(0xB6)).unwrap();
        p.commit().unwrap();
        p.begin_read().unwrap();
        assert_eq!(
            p.read_page(2).unwrap(),
            &marker(0xB6)[..],
            "the store round-trips after a post-reset re-checkpoint for {mode:?}"
        );
        p.end_read().unwrap();
    }
}

// ---------------------------------------------------------------------------
// FULL auto_vacuum COMPACTION under WAL is atomic-in-commit AND crash-safe
// (single connection). The disk (rollback-journal) backing already pins this in
// `diskpager.rs::full_auto_vacuum_compaction_is_atomic_and_crash_safe`; the WAL
// backing had only GENERIC frame-atomicity crash tests (torn/corrupt trailing
// frame, above) and facade-level compaction CORRECTNESS
// (`minisqlite/tests/conformance_auto_vacuum.rs`). NOTHING drove a COMPACTING WAL
// commit — a smaller db_size plus a relocated/leaked page recorded in ONE commit
// frame — through crash recovery. This closes that gap for the single-connection
// durability by COMBINING the two existing patterns:
//   * the WAL crash/reopen harness here (`torn_last_frame_is_ignored_on_reopen`),
//   * the FULL-av compacting-commit setup from the disk test (a FULL av page-1
//     header naming table `t`, the ptrmap-skipping allocation, then a DROP-leak
//     commit `av_commit::finalize_and_compact` compacts + shrinks).
//
// A FULL av commit compacts INSIDE the user's own commit (see `av_commit.rs`), so
// under WAL the shrink rides the commit frame's db-size (`walstore.rs` `apply_commit`
// sets `db_size = new_count` on the last frame; walformat §4). This pins BOTH crash
// outcomes at the durable-write boundary:
//   (A) a TORN/incomplete trailing commit frame => reopen ignores it and recovers the
//       PRE-COMMIT (larger, un-compacted) image, value-exact — never a torn file.
//   (B) a CLEAN commit frame => reopen replays the WAL to the COMPACTED (smaller)
//       image (freelist_count == 0, logical page count reduced), value-exact — never
//       left un-compacted.
//
// RED-if-broken: if the compacting commit is reverted to non-compacting (finalize
// sweeps page 3 onto the freelist but does NOT truncate — the old two-phase design),
// the commit frame would carry db_size == 3 with freelist_count == 1 across more than
// one frame, so outcome (B)'s `page_count == 1`, `freelist_count == 0` and
// `wal_frames == 1` assertions all fail.
// ---------------------------------------------------------------------------

#[test]
fn full_auto_vacuum_wal_compaction_is_atomic_and_crash_safe() {
    use minisqlite_pager::codec::{encode_record, encode_table_leaf_cell, PageBuilder, PageType};
    use minisqlite_types::Value;

    // A page-1 `sqlite_schema` leaf carrying a WAL (version 2) FULL auto_vacuum header
    // (off52 = largest root != 0, off64 = 0). `Some(rp)` names one table `t` rooted at
    // `rp`; `None` is an EMPTY schema (the "DROP" that leaks the table's page). Mirrors
    // the disk test's `av_schema_page1` with the write/read version bytes set to WAL, so
    // a clean commit of it leaves a version-2 db on disk and the NEXT open selects WAL.
    fn av_schema_page1(root: Option<i64>, ps: u32) -> Vec<u8> {
        let mut b = PageBuilder::new(ps as usize, ps as usize, 1, PageType::LeafTable);
        if let Some(rp) = root {
            let rec = encode_record(&[
                Value::Text("table".into()),
                Value::Text("t".into()),
                Value::Text("t".into()),
                Value::Integer(rp),
                Value::Text("CREATE TABLE t(a)".into()),
            ]);
            let mut cell = Vec::new();
            encode_table_leaf_cell(rec.len() as u64, 1, &rec, None, &mut cell);
            assert!(b.add_cell(&cell), "schema row fits on page 1");
        }
        let mut page1 = b.finish();
        let hdr = DatabaseHeader {
            page_size: ps,
            write_version: 2, // version 2 read+write => the next open selects the WAL backing
            read_version: 2,
            largest_root_btree: root.map(|r| r as u32).unwrap_or(1),
            incremental_vacuum: 0,
            ..DatabaseHeader::default()
        };
        let head: &mut [u8; HEADER_SIZE] = (&mut page1[..HEADER_SIZE]).try_into().unwrap();
        hdr.write(head);
        page1
    }

    fn empty_leaf(page_no: u32, ps: u32) -> Vec<u8> {
        PageBuilder::new(ps as usize, ps as usize, page_no, PageType::LeafTable).finish()
    }

    fn db_header(page1: &[u8]) -> DatabaseHeader {
        let mut buf = [0u8; HEADER_SIZE];
        buf.copy_from_slice(&page1[..HEADER_SIZE]);
        DatabaseHeader::read(&buf).unwrap()
    }

    // WAL frame count for an arbitrary page size (the file-level `wal_frame_count`
    // hardcodes `PS == 512`; this setup uses the pager's default page size).
    fn wal_frames(path: &Path, ps: u32) -> u32 {
        let len = file_len(path);
        let stride = FRAME_HEADER_SIZE + ps as u64;
        if len <= WAL_HEADER_SIZE {
            0
        } else {
            ((len - WAL_HEADER_SIZE) / stride) as u32
        }
    }

    // Build a 3-page FULL auto_vacuum WAL database THROUGH the pager (the WAL analog of
    // the disk test's pager-built setup): a fresh file opens rollback-backed; writing a
    // version-2 av page 1 and committing cleanly leaves a version-2 db on disk (page 2 =
    // the ptrmap finalize derived, page 3 = table t's empty leaf), so the NEXT open
    // selects the WAL backing — no byte-patching of the file. Returns (page_size,
    // pre-commit image C). No `-wal` exists yet (the setup ran in rollback mode).
    fn build_full_av_wal_db(case: &Case) -> (u32, Vec<u8>) {
        let ps;
        {
            let mut p = DiskPager::open(&case.db).unwrap();
            ps = p.page_size();
            p.begin().unwrap();
            assert_eq!(p.allocate_page().unwrap(), 1);
            p.write_page(1, &av_schema_page1(Some(3), ps)).unwrap();
            // A FULL av allocator reserves page 2 for the ptrmap and hands back page 3 —
            // matching the rootpage named in the schema above.
            assert_eq!(p.allocate_page().unwrap(), 3, "av alloc skips the ptrmap page 2");
            p.write_page(3, &empty_leaf(3, ps)).unwrap();
            p.commit().unwrap();
            assert_eq!(p.page_count().unwrap(), 3, "3-page FULL av db; no compaction (no free pages)");
            let h = db_header(p.read_page(1).unwrap());
            assert_eq!(h.freelist_count, 0, "insert-only FULL: freelist empty");
            assert!(h.largest_root_btree != 0, "auto_vacuum header set (off52 != 0)");
            assert!(h.is_wal(), "the committed page-1 header is WAL mode (version 2)");
        }
        assert_eq!(file_len(&case.wal), 0, "the rollback-mode setup wrote no -wal");
        let committed = std::fs::read(&case.db).unwrap();
        assert_eq!(committed.len(), 3 * ps as usize, "pre-commit image C is exactly 3 pages");
        (ps, committed)
    }

    // ===================================================================
    // (B) A CLEAN compacting WAL commit replays to the COMPACTED image.
    // ===================================================================
    {
        let case = Case::new("wal-av-compact-clean");
        let (ps, _committed) = build_full_av_wal_db(&case);

        let compacted_page1: Vec<u8> = {
            let mut p = DiskPager::open(&case.db).unwrap();
            assert_eq!(p.page_count().unwrap(), 3, "WAL open sees the 3-page pre-commit image");

            // The compacting commit: rewrite page 1 to an EMPTY schema so `finalize`
            // sweeps the now-unreferenced page 3 onto the freelist, then
            // `finalize_and_compact` reclaims it and truncates — all in ONE WAL commit
            // whose commit frame records the shrunk db_size (walformat §4).
            p.begin().unwrap();
            p.write_page(1, &av_schema_page1(None, ps)).unwrap();
            p.commit().unwrap();

            // LOGICAL compacted state. In WAL mode the main db FILE lags the WAL until a
            // checkpoint, so the oracle is the WAL's db_size + the page-1 header carried in
            // the commit frame, NOT the raw db-file length. These three assertions are
            // exactly what go RED if the compacting commit is reverted to non-compacting
            // (finalize-only leaves db_size == 3, freelist_count == 1, over more than one frame).
            assert_eq!(p.page_count().unwrap(), 1, "atomic FULL WAL commit compacted 3 -> 1 (logical db_size)");
            assert_eq!(
                wal_frames(&case.wal, ps),
                1,
                "the compaction rode exactly ONE WAL commit frame (non-compacting would leave more)"
            );
            let h = db_header(p.read_page(1).unwrap());
            assert_eq!(h.freelist_count, 0, "a committed FULL db has freelist_count == 0 (compacted, not swept-only)");
            assert!(h.largest_root_btree != 0, "the compacted file is still an auto_vacuum db");
            assert!(h.is_wal(), "the compacting commit kept the WAL version bytes");
            p.read_page(1).unwrap().to_vec()
        };

        // Reopen (WAL). The writer pager was scoped to the block above, so it is dropped
        // before this open — which re-reads the -wal from disk and genuinely REPLAYS it
        // rather than sharing a still-live in-memory SharedWal. Keeping the writer's drop
        // before this reopen is what makes (B) a durability (replay) check, not an
        // in-memory-share check. The clean commit frame replays, so the reopened image is
        // the COMPACTED one, value-exact — never left un-compacted.
        let p = DiskPager::open(&case.db).unwrap();
        assert_eq!(p.page_count().unwrap(), 1, "reopen replays the WAL to the compacted 1-page image");
        assert_eq!(p.read_page(1).unwrap(), &compacted_page1[..], "page 1 replays byte-exact after the WAL reopen");
        let h = db_header(p.read_page(1).unwrap());
        assert_eq!(h.freelist_count, 0, "freelist_count == 0 survives the WAL reopen");
        assert!(h.largest_root_btree != 0, "still an auto_vacuum db after the WAL reopen");
        assert!(is_sqlite_header(p.read_page(1).unwrap()), "page 1 is still a valid header");
        assert!(p.read_page(2).is_err(), "the compacted db has no page 2 (out of range)");
    }

    // ===================================================================
    // (A) A TORN trailing commit frame reverts to the PRE-COMMIT image.
    // ===================================================================
    {
        let case = Case::new("wal-av-compact-torn");
        let (ps, committed) = build_full_av_wal_db(&case);

        // Drive the SAME compacting WAL commit, then drop the connection so the reopen
        // re-reads the -wal from disk (the shared coordinator is released on last drop).
        {
            let mut p = DiskPager::open(&case.db).unwrap();
            p.begin().unwrap();
            p.write_page(1, &av_schema_page1(None, ps)).unwrap();
            p.commit().unwrap();
            assert_eq!(p.page_count().unwrap(), 1, "the writer sees its own compacting commit");
            assert_eq!(wal_frames(&case.wal, ps), 1, "one compacting commit frame in the -wal before the crash");
        }

        // Simulate a crash that left the trailing commit frame incomplete (an append/fsync
        // interrupted mid-frame): truncate the -wal so its last frame is shorter than a
        // whole frame. walformat: an incomplete trailing frame is not a commit, so `scan`
        // stops before it — the WAL analog of the disk test's hot-journal crash BEFORE the
        // commit point.
        let stride = FRAME_HEADER_SIZE + ps as u64;
        assert_eq!(
            file_len(&case.wal),
            WAL_HEADER_SIZE + stride,
            "the -wal holds exactly one full frame before tearing"
        );
        let torn_len = WAL_HEADER_SIZE + stride / 2; // 32-byte header + a partial (torn) frame
        {
            let f = OpenOptions::new().write(true).open(&case.wal).unwrap();
            f.set_len(torn_len).unwrap();
            f.sync_all().unwrap();
        }
        assert_eq!(file_len(&case.wal), torn_len, "the sole commit frame was truncated to a torn partial");

        // Reopen: `scan` ignores the torn trailing frame, so recovery lands on the
        // PRE-COMMIT (larger, un-compacted) image — value-exact, never a torn/partial file.
        let p = DiskPager::open(&case.db).unwrap();
        assert_eq!(p.page_count().unwrap(), 3, "a torn compacting commit reverts to the 3-page pre-commit image");
        assert_eq!(
            p.read_page(1).unwrap(),
            &committed[0..ps as usize],
            "page 1 is the exact pre-commit header (never the compacted one)"
        );
        assert_eq!(
            p.read_page(3).unwrap(),
            &committed[2 * ps as usize..3 * ps as usize],
            "table t's leaf (page 3) is the exact pre-commit bytes (never dropped by the torn commit)"
        );
        let h = db_header(p.read_page(1).unwrap());
        assert!(h.largest_root_btree != 0, "the recovered pre-commit db is still auto_vacuum");
        assert_eq!(h.freelist_count, 0, "the recovered pre-commit FULL db has freelist_count == 0 (not a torn partial)");
        // WAL commits never write the db file, so the durable file stays the untouched
        // pre-commit image throughout — the torn commit corrupted nothing.
        assert_eq!(std::fs::read(&case.db).unwrap(), committed, "the db file remains the exact pre-commit image");
    }
}
