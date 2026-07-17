//! Conformance battery: **WAL-mode durability across a full close -> reopen** at
//! the pinned facade (`minisqlite::Connection`).
//!
//! This is the WAL analog of `conformance_ondisk_roundtrip.rs` (which covers
//! rollback-mode durability) and it closes a coverage gap: no other facade test
//! writes REAL schema + rows through the public API in WAL mode, closes ALL
//! connections, reopens, and asserts the rows survive — both with a LIVE `-wal`
//! (no checkpoint) and AFTER a checkpoint drains the `-wal` into the database file.
//! `conformance_concurrency.rs` covers cross-connection visibility and in-session
//! checkpoint, but never a full close -> reopen round-trip in WAL mode.
//!
//! ## The WAL "reopen model" (why every test uses a throwaway scope to enter WAL)
//! `PRAGMA journal_mode=WAL` writes the version-2 page-1 header via a *rollback-mode*
//! commit, but the connection that ran it KEEPS its rollback backing until it is
//! closed and reopened — the storage backing is fixed for the life of an open handle
//! (`minisqlite-pager::DiskPager` picks WAL vs rollback from the header at open time).
//! So WAL mode only takes effect on the NEXT `Connection::open`. Every case therefore:
//!   scope A: open, `PRAGMA journal_mode=WAL`, drop  (db is now version-2);
//!   scope B: open (now WAL), CREATE + INSERT, drop;
//!   scope C: open (WAL), SELECT and assert.
//! This mirrors `conformance_concurrency.rs`'s
//! `second_connection_sees_first_connections_commit_wal_mode`.
//!
//! ## Discipline
//! Expected values are derived from the SQLite documentation and the durability
//! contract (committed data survives a reopen; a rolled-back transaction leaves no
//! trace), never from "whatever the engine currently returns". A failing case is the
//! intended signal of a real WAL durability discrepancy — it is never weakened to
//! pass, and no `src/` file is changed to make it pass. Row values are DETERMINISTIC
//! (an LCG per id), so a single swapped/duplicated/corrupted row is caught and
//! reproducible, and assertions are VALUE-exact (not mere row counts).
//!
//! Real `sqlite3` is not used here: everything is validated through
//! `minisqlite::Connection` (open / execute / query) plus filesystem-level inspection
//! of the `.db` / `-wal` file sizes.
//!
//! Each case is its own `#[test]` (one durable behavior) so one discrepancy fails
//! exactly that case rather than masking the rest.
//!
//! Spec sources:
//! - `spec/sqlite-doc/wal.html`: a WAL database appends committed transactions to the
//!   `-wal` file; a reader resolves pages from the WAL over the database file; a
//!   checkpoint folds the WAL back into the database.
//! - `spec/sqlite-doc/pragma.html` #pragma_journal_mode / #pragma_wal_checkpoint.
//! - `spec/sqlite-doc/fileformat2.html` §1.3: write/read file-format versions of 2
//!   select WAL mode.
//! - `spec/sqlite-doc/atomiccommit.html`: a committed transaction is durable; a
//!   rolled-back one leaves the pre-transaction image.

mod conformance;
use conformance::*;

use minisqlite::{Connection, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Temp-file harness: a unique on-disk path per test, cleaned up even on panic.
// (Same idiom as `conformance_ondisk_roundtrip.rs`; a `.db`/`-wal` file is never
// committed to the repo.)
// ---------------------------------------------------------------------------

/// The `-wal` header is 32 bytes; a live log holds more than that once any commit
/// has appended a frame, and a completed checkpoint that resets the log truncates it
/// back to exactly this. Kept as a named constant so the "frames present" / "log
/// drained" thresholds read clearly (matches `minisqlite-pager`'s `WAL_HEADER_SIZE`).
const WAL_HEADER_SIZE: u64 = 32;

/// An RAII guard over a unique temporary database path. Every test gets its own file
/// (isolated + idempotent), and `Drop` deletes the database AND its `-journal` /
/// `-wal` / `-shm` sidecars — so cleanup happens even when an assertion panics and
/// nothing is left behind in the temp dir.
struct TempDb {
    path: PathBuf,
}

impl TempDb {
    /// Build a fresh, unique path under the system temp dir and remove any stale file
    /// (and sidecars) a previous crashed run might have left. Uniqueness combines the
    /// process id, a per-process atomic counter, and a nanosecond timestamp so
    /// parallel test threads never collide.
    fn new() -> TempDb {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
        let mut path = std::env::temp_dir();
        path.push(format!("minisqlite_waldur_{pid}_{n}_{nanos}.db"));
        let db = TempDb { path };
        db.remove_all();
        db
    }

    /// Open (or reopen) a file-backed connection to this path, panicking with context
    /// on failure so a test body can use it unwrapped.
    fn open(&self) -> Connection {
        let path: &Path = &self.path;
        Connection::open(path)
            .unwrap_or_else(|e| panic!("Connection::open failed for {path:?}: {e:?}"))
    }

    /// The `<path><suffix>` sibling (e.g. `-wal`), by appending to the FULL path
    /// including its extension — SQLite's own naming convention (`foo.db-wal`).
    fn sidecar(&self, suffix: &str) -> PathBuf {
        let mut s = self.path.as_os_str().to_os_string();
        s.push(suffix);
        PathBuf::from(s)
    }

    /// Byte length of the database file (0 if absent). See [`file_len`].
    fn db_len(&self) -> u64 {
        file_len(&self.path)
    }

    /// Byte length of the `-wal` sidecar (0 if absent). A committed WAL frame grows
    /// it past its 32-byte header; a completed checkpoint that drains+resets the log
    /// shrinks it back, so this is a filesystem-level witness of WAL activity. See
    /// [`file_len`].
    fn wal_len(&self) -> u64 {
        file_len(&self.sidecar("-wal"))
    }

    /// Delete the database file and every sidecar SQLite may have created. Missing
    /// files are ignored — the expected, non-broken state, not an error to surface.
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

/// Byte length of `path`, or 0 when it does not exist — an expected, non-broken state
/// (e.g. a `-wal` before any WAL write). A NON-`NotFound` IO error is a real
/// filesystem fault, not a durability signal, so it panics loudly rather than folding
/// into a misleading `0` that a size assertion could read as a genuine value.
fn file_len(path: &Path) -> u64 {
    match std::fs::metadata(path) {
        Ok(m) => m.len(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => 0,
        Err(e) => panic!("metadata({path:?}) failed unexpectedly: {e}"),
    }
}

// ---------------------------------------------------------------------------
// Deterministic row data + WAL helpers.
// ---------------------------------------------------------------------------

/// A deterministic TEXT value for row `id`: an LCG mix rendered as `row-<id>-<hex>`.
/// The alphabet is letters, digits, and `-` ONLY (no quote / backslash / whitespace),
/// so it embeds directly in a single-quoted SQL literal, while the hashed suffix
/// varies per id — a swapped, duplicated, or corrupted row fails the value-exact
/// assertion rather than slipping through a bare row count.
fn val_for(id: i64) -> String {
    let mut state = (id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let v = format!("row-{id:05}-{:012x}", state >> 16);
    // Enforce (not just document) the literal-safety invariant: `insert_range` embeds
    // this straight into a single-quoted SQL literal, so the value must stay within
    // `[A-Za-z0-9-]`. A future edit that widened the alphabet to include a quote or
    // backslash would malform the SQL and could let a bogus INSERT slip through — this assertion
    // turns that into a loud failure at the source instead.
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

/// Insert ids `lo..=hi` into `t(id, v)` with deterministic values, in bounded batches
/// (a single INSERT per 50 ids). Batching keeps the round-trip count low while still
/// producing several WAL commit frames, and no single statement is unreasonably large.
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

/// Run a `PRAGMA journal_mode[=…]` and return the single mode string it reports
/// (`"wal"`, `"delete"`, …). Pins the documented one-row / one-`journal_mode`-column
/// shape (pragma.html #pragma_journal_mode).
fn journal_mode(db: &mut Connection, sql: &str) -> String {
    let qr = query(db, sql);
    assert_eq!(qr.columns, vec!["journal_mode".to_string()], "one column named journal_mode: {sql}");
    assert_eq!(qr.rows.len(), 1, "PRAGMA journal_mode returns exactly one row: {sql}");
    match &qr.rows[0][0] {
        Value::Text(s) => s.clone(),
        other => panic!("PRAGMA journal_mode returned a non-text cell: {other:?}"),
    }
}

/// Put the database at `db`'s path into WAL mode from a THROWAWAY connection and close
/// it, so the NEXT open uses the WAL backing (see the module docs' "reopen model").
/// Asserts the pragma reports `wal` — if this fails, WAL mode never engaged and every
/// downstream assertion would be meaningless, so failing here pinpoints the cause.
fn enter_wal_mode(tmp: &TempDb) {
    let mut a = tmp.open();
    assert_eq!(
        journal_mode(&mut a, "PRAGMA journal_mode=WAL"),
        "wal",
        "PRAGMA journal_mode=WAL must report the resulting mode as wal"
    );
}

/// Run `PRAGMA wal_checkpoint`, asserting the documented one-row `(busy, log,
/// checkpointed)` shape with `busy = 0` (pragma.html #pragma_wal_checkpoint). Drains
/// the `-wal` back into the database file.
fn checkpoint(db: &mut Connection) {
    let qr = query(db, "PRAGMA wal_checkpoint");
    assert_eq!(qr.rows.len(), 1, "PRAGMA wal_checkpoint returns exactly one row");
    assert_eq!(qr.rows[0].len(), 3, "wal_checkpoint returns three columns (busy, log, checkpointed)");
    match &qr.rows[0][0] {
        Value::Integer(0) => {}
        other => panic!("wal_checkpoint should report busy=0, got {other:?}"),
    }
}

// ===========================================================================
// 1 — rows written in WAL mode survive close + reopen with a LIVE -wal.
// ===========================================================================

/// Write a table and 200 rows through a WAL-mode connection, close it WITHOUT a
/// checkpoint (so the commits stay in a live `-wal`), then reopen and require every
/// row back value-exact. This proves the reopen path rebuilds the WAL state from the
/// `-wal` file and resolves reads through it, merged with the database base image.
#[test]
fn wal_rows_survive_close_and_reopen_with_live_wal() {
    const N: i64 = 200;
    let tmp = TempDb::new();

    // scope A: enter WAL mode on the fresh database, then close.
    enter_wal_mode(&tmp);
    let db_len_after_wal_set = tmp.db_len();
    assert!(db_len_after_wal_set > 0, "the fresh db file exists after WAL was set");

    // scope B: now in WAL mode — create the schema and insert N rows, then close
    // without checkpointing so the frames remain in a live `-wal`.
    {
        let mut b = tmp.open();
        assert_eq!(journal_mode(&mut b, "PRAGMA journal_mode"), "wal", "scope B reopened in WAL mode");
        exec(&mut b, "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)");
        insert_range(&mut b, 1, N);
        // Sanity within the writing connection before it closes.
        assert_scalar(&mut b, "SELECT count(*) FROM t", int(N));
    }

    // The commits really went to the `-wal`, not the database file: the log holds
    // more than its 32-byte header, and the db file did NOT grow (a WAL-mode commit
    // never writes the database file — only a checkpoint does).
    let wal_len = tmp.wal_len();
    assert!(
        wal_len > WAL_HEADER_SIZE,
        "committed frames live in a live -wal before reopen (len {wal_len}, header {WAL_HEADER_SIZE})"
    );
    assert_eq!(
        tmp.db_len(),
        db_len_after_wal_set,
        "WAL-mode commits append to the -wal; they do not grow the database file"
    );

    // scope C: reopen (WAL) and require every row back, value-exact and in order.
    let mut c = tmp.open();
    assert_eq!(journal_mode(&mut c, "PRAGMA journal_mode"), "wal", "scope C reopened in WAL mode");
    assert_rows(&mut c, "SELECT id, v FROM t ORDER BY id", &expected_range(1, N));
    assert_scalar(&mut c, "SELECT count(*) FROM t", int(N));
}

// ===========================================================================
// 2 — rows survive close + reopen AFTER a checkpoint drains the -wal.
// ===========================================================================

/// Same writes as case 1, but `PRAGMA wal_checkpoint` folds the `-wal` back into the
/// database file before close. After reopen the rows are served from the database
/// file (with a reset/empty `-wal`) and must still be value-exact.
#[test]
fn wal_rows_survive_close_and_reopen_after_checkpoint() {
    const N: i64 = 200;
    let tmp = TempDb::new();

    enter_wal_mode(&tmp);

    {
        let mut b = tmp.open();
        assert_eq!(journal_mode(&mut b, "PRAGMA journal_mode"), "wal", "scope B reopened in WAL mode");
        exec(&mut b, "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)");
        insert_range(&mut b, 1, N);

        let wal_before = tmp.wal_len();
        assert!(wal_before > WAL_HEADER_SIZE, "frames present in the -wal before checkpoint (len {wal_before})");

        checkpoint(&mut b);

        // Pin the DRAIN: a completed checkpoint folds every frame into the db file and
        // resets the log, so the `-wal` shrinks. A no-op checkpoint would leave it.
        let wal_after = tmp.wal_len();
        assert!(
            wal_after < wal_before,
            "checkpoint drained+reset the -wal ({wal_before} -> {wal_after})"
        );
        assert_scalar(&mut b, "SELECT count(*) FROM t", int(N));
    }

    // scope C: reopen; rows now come from the database file, still value-exact.
    let mut c = tmp.open();
    assert_eq!(journal_mode(&mut c, "PRAGMA journal_mode"), "wal", "scope C still WAL mode after checkpoint");
    assert_rows(&mut c, "SELECT id, v FROM t ORDER BY id", &expected_range(1, N));
    assert_scalar(&mut c, "SELECT count(*) FROM t", int(N));
}

// ===========================================================================
// 3 — writes AFTER a checkpoint reset also survive; reopen reads the union.
// ===========================================================================

/// Insert batch 1, checkpoint (drain + reset the log), then insert batch 2 — whose
/// frames append into the freshly reset `-wal`. After close + reopen BOTH batches must
/// survive value-exact: batch 1 served from the database file, batch 2 resolved from
/// the post-reset `-wal`. This exercises writing fresh frames after a checkpoint reset
/// and reading their union with the checkpointed base on reopen.
#[test]
fn wal_rows_survive_reopen_across_checkpoint_and_more_writes() {
    const BATCH1_HI: i64 = 120; // ids 1..=120
    const BATCH2_LO: i64 = 121;
    const BATCH2_HI: i64 = 200; // ids 121..=200
    let tmp = TempDb::new();

    enter_wal_mode(&tmp);

    {
        let mut b = tmp.open();
        assert_eq!(journal_mode(&mut b, "PRAGMA journal_mode"), "wal", "scope B reopened in WAL mode");
        exec(&mut b, "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)");

        insert_range(&mut b, 1, BATCH1_HI); // batch 1 -> frames in the -wal
        checkpoint(&mut b); // drain batch 1 into the db file and reset the log
        insert_range(&mut b, BATCH2_LO, BATCH2_HI); // batch 2 -> fresh frames after the reset

        assert_scalar(&mut b, "SELECT count(*) FROM t", int(BATCH2_HI));
    }

    // Batch 2 was never checkpointed, so it lives in a live -wal on disk (batch 1 was
    // drained into the db file). Scope C's reopen therefore resolves the union from
    // the physical -wal + the db base — a genuine on-disk read, not an in-process
    // cache (the sole WAL connection dropped, tearing down its coordinator).
    let wal_len = tmp.wal_len();
    assert!(wal_len > WAL_HEADER_SIZE, "batch 2 lives in a live -wal before reopen (len {wal_len})");

    // scope C: reopen; the union of both batches must be present, value-exact.
    let mut c = tmp.open();
    assert_eq!(journal_mode(&mut c, "PRAGMA journal_mode"), "wal", "scope C reopened in WAL mode");
    assert_rows(&mut c, "SELECT id, v FROM t ORDER BY id", &expected_range(1, BATCH2_HI));
    assert_scalar(&mut c, "SELECT count(*) FROM t", int(BATCH2_HI));
}

// ===========================================================================
// 4 — a rolled-back explicit transaction in WAL mode leaves no trace on reopen.
// ===========================================================================

/// Commit a baseline in autocommit (durable WAL frames), then open an explicit
/// transaction that BOTH inserts new rows and deletes baseline rows and `ROLLBACK` it.
/// None of that transaction's work may reach the `-wal`, so after close + reopen ONLY
/// the baseline survives (atomiccommit.html: a rolled-back transaction leaves the
/// pre-transaction image). This is the WAL counterpart of
/// `conformance_ondisk_roundtrip.rs`'s `rolled_back_transaction_absent_after_reopen`,
/// and the first facade coverage of the explicit-BEGIN/ROLLBACK path in WAL mode
/// (the concurrency suite only drives autocommit).
#[test]
fn rolled_back_txn_in_wal_mode_leaves_no_trace_after_reopen() {
    const BASE: i64 = 100; // baseline ids 1..=100
    let tmp = TempDb::new();

    enter_wal_mode(&tmp);

    {
        let mut b = tmp.open();
        assert_eq!(journal_mode(&mut b, "PRAGMA journal_mode"), "wal", "scope B reopened in WAL mode");
        exec(&mut b, "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)");
        insert_range(&mut b, 1, BASE); // autocommit -> durable baseline
        // The baseline lives in a live -wal on disk (never checkpointed), so scope C's
        // reopen reads it from the physical -wal.
        let wal_baseline = tmp.wal_len();
        assert!(wal_baseline > WAL_HEADER_SIZE, "baseline lives in a live -wal (len {wal_baseline})");

        // An explicit transaction that would both add rows and remove baseline rows,
        // then rolls back: none of it may be committed to the WAL.
        exec(&mut b, "BEGIN");
        insert_range(&mut b, BASE + 1, BASE + 50); // would-be new rows
        exec(&mut b, "DELETE FROM t WHERE id <= 10"); // would-be deletions
        exec(&mut b, "ROLLBACK");

        // The rollback appended NOTHING to the physical -wal — it never reached
        // apply_commit — so the on-disk log is byte-for-byte the baseline's. A direct
        // witness that the aborted txn left no frames, stronger than only re-reading
        // the row set after reopen.
        let wal_after_rollback = tmp.wal_len();
        assert_eq!(
            wal_after_rollback, wal_baseline,
            "a rolled-back WAL-mode txn must append no frames to the -wal"
        );

        // Within the connection, only the baseline remains after the rollback.
        assert_scalar(&mut b, "SELECT count(*) FROM t", int(BASE));
        assert_rows(&mut b, "SELECT id, v FROM t ORDER BY id", &expected_range(1, BASE));
    }

    // scope C: reopen; the rolled-back changes left no trace — only baseline survives.
    let mut c = tmp.open();
    assert_eq!(journal_mode(&mut c, "PRAGMA journal_mode"), "wal", "scope C reopened in WAL mode");
    assert_rows(&mut c, "SELECT id, v FROM t ORDER BY id", &expected_range(1, BASE));
    assert_scalar(&mut c, "SELECT count(*) FROM t", int(BASE));
}
