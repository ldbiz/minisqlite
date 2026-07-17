//! Conformance battery: on-disk **durability for values that SPILL
//! ONTO OVERFLOW PAGES**, driven end-to-end through the pinned facade
//! `minisqlite::Connection`.
//!
//! A value wider than a table b-tree leaf cell's local-payload threshold cannot
//! fit inline and spills the surplus onto a linked list of overflow pages
//! (`spec/sqlite-doc/fileformat2.html` §1.6 "the amount of payload that spills"
//! and §1.7 "Cell Payload Overflow Pages"). At the default 4096-byte page size
//! the table-leaf threshold is `X = U-35 = 4061` bytes, so every payload used
//! here (>= 5000 bytes) is guaranteed to overflow, while the deliberately mixed
//! ~2000-byte rows in [`many_large_rows_survive_reopen`] stay inline — exercising
//! overflow cells packed among inline cells.
//!
//! The gaps this file closes (none covered elsewhere):
//!  - a large value written THROUGH the engine survives a file close+reopen
//!    (sibling `minisqlite-btree/tests/overflow.rs` only ever uses an in-memory
//!    `MemPager`, never a real file across a reopen or through the journal);
//!  - a rolled-back large INSERT/UPDATE leaves NO trace on disk (the rollback
//!    journal must undo overflow-page writes and restore the pre-image —
//!    `spec/sqlite-doc/atomiccommit.html` §3.5).
//!
//! On-disk overflow FORMAT cross-validation (byte-exact, in BOTH directions)
//! lives compliantly in
//! `crates/minisqlite-pager/tests/ondisk_format_{read,write}.rs` against
//! hand-built, spec-derived fixtures. This suite never invokes real SQLite — it
//! validates durability of overflow-page values purely through our own engine's
//! write->close->reopen round-trip.
//!
//! DISCIPLINE: every expected value is derived from the SQLite documentation —
//! never from "whatever our engine currently returns". A failing case is the
//! intended signal of a real discrepancy; it is never weakened, deleted, or
//! `#[ignore]`-d, and no production code is changed to make it pass. Payloads are
//! DETERMINISTIC (an LCG seeded per row) so any
//! single wrong byte is caught and reproducible.
//!
//! Each case is its own `#[test]` (one durable behavior) so one discrepancy
//! fails exactly that case rather than masking the rest.

mod conformance;
use conformance::*;

use minisqlite::Connection;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Temp-file harness: a unique on-disk path per test, cleaned up even on panic.
// (Copied from `conformance_ondisk_roundtrip.rs`; a `.db` file is never committed.)
// ---------------------------------------------------------------------------

/// An RAII guard over a unique temporary database path. Every test gets its own
/// file (isolated + idempotent), and `Drop` deletes the database AND its
/// `-journal` / `-wal` / `-shm` sidecars — so cleanup happens even when an
/// assertion panics and nothing is left behind in the temp dir.
struct TempDb {
    path: PathBuf,
}

impl TempDb {
    /// Build a fresh, unique path under the system temp dir and remove any stale
    /// file (and sidecars) a previous crashed run might have left. Uniqueness
    /// combines the process id, a per-process atomic counter, and a nanosecond
    /// timestamp so parallel test threads never collide.
    fn new() -> TempDb {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
        let mut path = std::env::temp_dir();
        path.push(format!("minisqlite_ovf_{pid}_{n}_{nanos}.db"));
        let db = TempDb { path };
        db.remove_all();
        db
    }

    /// Open (or reopen) a file-backed connection to this path, panicking with
    /// context on failure so a test body can use it unwrapped.
    fn open(&self) -> Connection {
        let path: &Path = &self.path;
        Connection::open(path)
            .unwrap_or_else(|e| panic!("Connection::open failed for {path:?}: {e:?}"))
    }

    /// The `<path><suffix>` sibling (e.g. `-journal`), by appending to the FULL
    /// path including its extension — SQLite's own naming convention.
    fn sidecar(&self, suffix: &str) -> PathBuf {
        let mut s = self.path.as_os_str().to_os_string();
        s.push(suffix);
        PathBuf::from(s)
    }

    /// Delete the database file and every sidecar SQLite may have created.
    /// Missing files are ignored — the expected, non-broken state.
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
// Deterministic payload generators (an LCG, so a single wrong byte is caught and
// failures reproduce). `big_payload` is copied verbatim from the btree overflow
// test so both suites exercise the identical byte stream per seed.
// ---------------------------------------------------------------------------

/// A deterministic pseudo-random blob of `len` bytes, distinct per `seed` (an LCG
/// stream) so a round-trip checks every byte and cross-row contamination is
/// caught (row A's bytes surfacing for row B).
fn big_payload(len: usize, seed: u64) -> Vec<u8> {
    let mut v = Vec::with_capacity(len);
    let mut state = seed.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for _ in 0..len {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        v.push((state >> 40) as u8);
    }
    v
}

/// A deterministic ASCII string of `len` bytes over a safe alphabet (letters +
/// digits ONLY: no quote, backslash, whitespace, or newline). That keeps the
/// value embeddable in a single-quoted SQL literal — while still varying per
/// position (LCG) so a single wrong byte is caught, not just a length mismatch.
fn ascii_payload(len: usize, seed: u64) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let mut s = String::with_capacity(len);
    let mut state = seed.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for _ in 0..len {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let idx = (state >> 40) as usize % ALPHABET.len();
        s.push(ALPHABET[idx] as char);
    }
    s
}

/// Uppercase hex encoding of `bytes`, used to build `x'..'` SQL blob literals
/// (the only way to write exact bytes through SQL text).
fn to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        let _ = write!(s, "{b:02X}");
    }
    s
}

/// Insert a single blob value under `key` into table `t(k INTEGER PRIMARY KEY, b)`
/// using a hex literal (the only way to write exact bytes through SQL text).
fn insert_blob(db: &mut Connection, key: i64, bytes: &[u8]) {
    exec(db, &format!("INSERT INTO t(k, b) VALUES ({key}, x'{}')", to_hex(bytes)));
}

// ===========================================================================
// SECTION 1 — durability across reopen (facade only, no sqlite3 needed).
//
// Each case writes a large value through a file-backed `Connection`, closes it
// (drop), reopens the SAME file, and asserts the committed value came back
// byte-for-byte. This exercises the overflow WRITE + read path end to end plus
// single-connection durability.
// ===========================================================================

/// A 5000-byte blob (> the 4061-byte table-leaf threshold at page size 4096) must
/// spill onto overflow pages, and after a close+reopen read back byte-for-byte
/// with its BLOB storage class intact (fileformat2 §1.6/§1.7; datatype3 §3.1: a
/// no-declared-type column keeps the stored class).
#[test]
fn large_blob_survives_reopen() {
    let tmp = TempDb::new();
    let payload = big_payload(5_000, 1);
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, b)");
        insert_blob(&mut db, 1, &payload);
        assert_rows(&mut db, "SELECT b FROM t", &[vec![blob(&payload)]]);
    }

    let mut db = tmp.open();
    assert_rows(&mut db, "SELECT b FROM t", &[vec![blob(&payload)]]);
    assert_scalar(&mut db, "SELECT typeof(b) FROM t", text("blob"));
}

/// A 20000-byte TEXT value spans several overflow pages; after reopen the exact
/// characters and the TEXT storage class must survive (serial type is text; the
/// overflow chain carries the tail bytes — fileformat2 §1.7).
#[test]
fn large_text_survives_reopen() {
    let tmp = TempDb::new();
    let s = ascii_payload(20_000, 2);
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, b)");
        // The alphabet excludes `'`, so direct embedding is safe.
        exec(&mut db, &format!("INSERT INTO t(k, b) VALUES (1, '{s}')"));
        assert_rows(&mut db, "SELECT b FROM t", &[vec![text(&s)]]);
    }

    let mut db = tmp.open();
    assert_rows(&mut db, "SELECT b FROM t", &[vec![text(&s)]]);
    assert_scalar(&mut db, "SELECT typeof(b) FROM t", text("text"));
}

/// A 100000-byte blob spans MANY overflow pages (~24 at page size 4096). Reopen
/// must reassemble the whole chain byte-for-byte — a single dropped/duplicated
/// link would corrupt one interior 4092-byte span and be caught.
#[test]
fn very_large_blob_survives_reopen() {
    let tmp = TempDb::new();
    let payload = big_payload(100_000, 3);
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, b)");
        insert_blob(&mut db, 1, &payload);
    }

    let mut db = tmp.open();
    assert_rows(&mut db, "SELECT b FROM t", &[vec![blob(&payload)]]);
    assert_scalar(&mut db, "SELECT length(b) FROM t", int(payload.len() as i64));
    assert_scalar(&mut db, "SELECT typeof(b) FROM t", text("blob"));
}

/// ~30 rows, each a DISTINCT payload sized across [2000, 8000] bytes (a mix of
/// inline and overflowing), must ALL read back byte-exact after reopen. A distinct
/// per-row LCG seed makes a chain mixup or cross-row contamination (row A's bytes
/// under row B's rowid) fail on the offending row rather than hide.
#[test]
fn many_large_rows_survive_reopen() {
    let tmp = TempDb::new();
    const N: i64 = 30;
    // Deterministic per-row (len, bytes): len spreads across 2000..=8000 so some
    // rows stay inline (< 4061) and some overflow, interleaved by rowid.
    let rows: Vec<(i64, Vec<u8>)> = (1..=N)
        .map(|id| {
            let len = 2_000 + ((id as usize) * 211) % 6_001; // in [2000, 8000]
            (id, big_payload(len, id as u64))
        })
        .collect();
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, b)");
        for (id, bytes) in &rows {
            insert_blob(&mut db, *id, bytes);
        }
    }

    let mut db = tmp.open();
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(N));
    // Per-row assertion pinpoints exactly which row diverged.
    for (id, bytes) in &rows {
        assert_rows(
            &mut db,
            &format!("SELECT b FROM t WHERE k = {id}"),
            &[vec![blob(bytes)]],
        );
    }
}

/// Growing a row from a small inline value into a large one must allocate a fresh
/// overflow chain that persists across reopen; then shrinking it back must FREE
/// that chain so the old big bytes never resurface. Reopen after each step proves
/// the on-disk state, not just the in-cache state (fileformat2 §1.7; the freed
/// chain returns to the freelist per fileformat2 §1.5).
#[test]
fn update_growing_row_into_overflow_persists() {
    let tmp = TempDb::new();
    let small_1 = big_payload(5, 40);
    let big = big_payload(30_000, 41);
    let small_2 = big_payload(6, 42);
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, b)");
        insert_blob(&mut db, 1, &small_1);
        exec(&mut db, &format!("UPDATE t SET b = x'{}' WHERE k = 1", to_hex(&big)));
    }

    // Reopen #1: the grown-into-overflow value is durable.
    {
        let mut db = tmp.open();
        assert_rows(&mut db, "SELECT b FROM t", &[vec![blob(&big)]]);
        // Shrink back to a small inline value; the old overflow chain must be freed.
        exec(&mut db, &format!("UPDATE t SET b = x'{}' WHERE k = 1", to_hex(&small_2)));
    }

    // Reopen #2: only the new small value is present — the old big chain left no trace.
    let mut db = tmp.open();
    assert_rows(&mut db, "SELECT b FROM t", &[vec![blob(&small_2)]]);
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(1));
    assert_scalar(&mut db, "SELECT length(b) FROM t", int(small_2.len() as i64));
}

// ===========================================================================
// SECTION 2 — rollback durability (facade only): the rollback journal must undo
// overflow-page writes exactly (atomiccommit.html §3.5: the journal holds the
// original page content and a ROLLBACK restores the pre-transaction image).
// ===========================================================================

/// A large INSERT rolled back inside an explicit transaction must leave NO trace:
/// neither in the connection nor on disk after reopen. Only the row committed
/// before the transaction (in autocommit) survives — the aborted 100000-byte
/// value's overflow chain must be undone by the journal.
#[test]
fn rolled_back_large_insert_leaves_no_trace() {
    let tmp = TempDb::new();
    let small = big_payload(4, 50);
    let big = big_payload(100_000, 51);
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, b)");
        insert_blob(&mut db, 1, &small); // autocommit -> durable
        exec(&mut db, "BEGIN");
        insert_blob(&mut db, 2, &big); // to be rolled back
        exec(&mut db, "ROLLBACK");
        // Within the connection only the committed small row remains.
        assert_scalar(&mut db, "SELECT count(*) FROM t", int(1));
        assert_rows(&mut db, "SELECT b FROM t WHERE k = 1", &[vec![blob(&small)]]);
        assert_scalar(&mut db, "SELECT count(*) FROM t WHERE k = 2", int(0));
    }

    // After reopen, the rolled-back big row + its overflow chain left no trace.
    let mut db = tmp.open();
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(1));
    assert_rows(&mut db, "SELECT b FROM t WHERE k = 1", &[vec![blob(&small)]]);
    assert_scalar(&mut db, "SELECT count(*) FROM t WHERE k = 2", int(0));
}

/// Updating a committed large value to a DIFFERENT large value inside a
/// transaction, then rolling back, must restore the ORIGINAL bytes exactly —
/// both in the connection and after reopen. The journal must have captured the
/// original overflow pages and restored them (atomiccommit.html §3.5).
#[test]
fn rolled_back_large_update_restores_original() {
    let tmp = TempDb::new();
    let original = big_payload(10_000, 60);
    let replacement = big_payload(50_000, 61);
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, b)");
        insert_blob(&mut db, 1, &original); // autocommit -> durable
        exec(&mut db, "BEGIN");
        exec(&mut db, &format!("UPDATE t SET b = x'{}' WHERE k = 1", to_hex(&replacement)));
        // Mid-transaction the new value is visible...
        assert_rows(&mut db, "SELECT b FROM t WHERE k = 1", &[vec![blob(&replacement)]]);
        exec(&mut db, "ROLLBACK");
        // ...but ROLLBACK restores the original bytes.
        assert_rows(&mut db, "SELECT b FROM t WHERE k = 1", &[vec![blob(&original)]]);
    }

    let mut db = tmp.open();
    assert_rows(&mut db, "SELECT b FROM t WHERE k = 1", &[vec![blob(&original)]]);
    assert_scalar(&mut db, "SELECT length(b) FROM t", int(original.len() as i64));
}

// ===========================================================================
// BONUS (optional, stable) — freelist reuse proxy via on-disk file size.
// Deleting a large value frees its overflow chain to the freelist; reinserting an
// equal-size value must REUSE those freed pages rather than leak a second chain,
// so the file does not grow by another whole chain (fileformat2 §1.5 freelist).
// The slack (8 pages) is far below one 100000-byte chain (~24 pages), so a genuine
// leak fails loudly while normal structural variance passes.
// ===========================================================================

#[test]
fn big_delete_then_equal_reinsert_reuses_freed_space() {
    let tmp = TempDb::new();
    let first = big_payload(100_000, 80);
    let second = big_payload(100_000, 81); // same size, different bytes
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, b)");
        insert_blob(&mut db, 1, &first);
    }
    let size_after_first = std::fs::metadata(&tmp.path).expect("stat db").len();

    {
        let mut db = tmp.open();
        exec(&mut db, "DELETE FROM t"); // frees the overflow chain to the freelist
        insert_blob(&mut db, 1, &second); // must reuse the freed pages
    }
    let size_after_reinsert = std::fs::metadata(&tmp.path).expect("stat db").len();

    let slack = 8 * 4096; // structural pages; a leaked chain would add ~24 pages
    assert!(
        size_after_reinsert <= size_after_first + slack,
        "delete+equal-reinsert must reuse the freed overflow chain: file grew from \
         {size_after_first} to {size_after_reinsert} bytes (slack {slack}); a leak would add \
         a whole second ~100KB chain"
    );

    // And the reinserted value is exactly the new bytes (not a stale chain).
    let mut db = tmp.open();
    assert_rows(&mut db, "SELECT b FROM t", &[vec![blob(&second)]]);
}
