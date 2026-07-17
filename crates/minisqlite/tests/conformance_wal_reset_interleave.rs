//! Two-connection WAL **reset-vs-reader/writer** conformance,
//! pinned at the public facade `minisqlite::Connection`. This is the sharp corner the
//! sibling interleaving battery (`conformance_concurrency_interleave.rs`) deliberately
//! leaves to this file: what a `PRAGMA wal_checkpoint(RESTART|TRUNCATE)` — a checkpoint
//! that wants to RESET the log — does while another connection is actively reading or
//! has a live snapshot behind the log tail. It is the interleaving where the historic
//! 16-year SQLite WAL corruption bug lived: a reset that reused the log from the
//! beginning while a reader still needed the frames it overwrote.
//!
//! Every case transcribes the TARGET straight from the SQLite documentation, never
//! from whatever the current engine returns:
//!
//! - `spec/sqlite-doc/wal.html` §2 (*Checkpointing*) + §4 fileformat2 §4.4 (*WAL reset*):
//!   a checkpoint copies committed frames back into the database file; the log may only
//!   be reset (reused from the beginning) once every frame is safely in the database AND
//!   no reader still needs a frame in the log, because a reset bumps the WAL salts and a
//!   reader that read a frame under the old salt must not read a new frame written over
//!   it. A WAL is identified by its salt values in the header.
//! - `spec/sqlite-doc/pragma.html` #pragma_wal_checkpoint: `PRAGMA wal_checkpoint(<mode>)`
//!   "returns a single row with three integer columns" — `(busy, log, checkpointed)`.
//!   `busy` "will be 1 if a RESTART or FULL or TRUNCATE checkpoint was blocked from
//!   completing, for example because another thread or process was actively using the
//!   database. In other words, busy will be set to 1 only if the equivalent call to
//!   [sqlite3_wal_checkpoint_v2()] would have returned SQLITE_BUSY." `log` is "the number
//!   of frames in the WAL" and `checkpointed` is "the number of frames in the WAL that
//!   have been checkpointed", or `-1` for each when not in WAL mode. The modes:
//!     * `PASSIVE` — "checkpoint as many frames as possible without waiting" and never
//!       blocks (busy is always 0); it does not force a reset.
//!     * `FULL` — block until all frames are checkpointed (busy=1 if it cannot).
//!     * `RESTART` — FULL, plus block until readers are done with the log so the next
//!       writer restarts it from the beginning (busy=1 while a reader still holds it).
//!     * `TRUNCATE` — "the same way as RESTART with the addition that the WAL file is
//!       truncated to zero bytes upon successful completion".
//!     * (a bare / unrecognized argument is PASSIVE).
//! - `spec/sqlite-doc/lang_transaction.html` §2.1: while a read transaction is open the
//!   connection "will continue to see an historic snapshot of the database" — a
//!   concurrent commit, and a concurrent checkpoint, are invisible to it.
//!
//! **Assertion style.** `busy` is the hard contract (the reset-blocked signal the whole
//! corner is about), so it is pinned EXACTLY. Row VALUES are pinned exactly (via
//! `select_x_sorted`, so a single stale/duplicated/mangled row fails). `log`/`checkpointed`
//! are pinned by their spec RELATIONSHIP (a complete drain has `checkpointed == log`; a
//! reader-bounded partial drain has `0 < checkpointed < log`; NOOP copies nothing so
//! `checkpointed == 0`) rather than a raw frame count, and the `-wal` size by its reset
//! RELATIONSHIP (TRUNCATE → 0 on success; unchanged while a reader blocks the reset) —
//! matching the sibling battery's convention, so a correct change to how many frames a
//! commit writes does not spuriously fail these tests.
//!
//! **Snapshot pin-timing** (same rule as the sibling battery): this engine pins a
//! `BEGIN [DEFERRED]` read snapshot at BEGIN time; to stay on the side of the boundary
//! where the engine and real SQLite agree, every reader here performs its FIRST read
//! before any concurrent connection commits, so its snapshot covers exactly the
//! pre-commit image under both pin rules.

use minisqlite::{Connection, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Temp-file harness (a unique on-disk path per test, cleaned up even on panic).
// Each Rust integration test is its own binary, so this small fixture is local
// rather than shared — the standard `tests/*.rs` idiom.
// ---------------------------------------------------------------------------

/// An RAII guard over a unique temporary database path. `Drop` deletes the database
/// AND its `-journal` / `-wal` / `-shm` sidecars, so nothing leaks even on a panic.
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
        path.push(format!("minisqlite_wal_reset_{pid}_{n}_{nanos}.db"));
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

/// A `Value` as an `i64` (panicking otherwise). `Value` deliberately does not implement
/// `PartialEq`, so tests destructure it rather than comparing directly.
fn as_int(v: &Value) -> i64 {
    match v {
        Value::Integer(n) => *n,
        other => panic!("expected an integer, got {other:?}"),
    }
}

/// Every `x` value in `t`, ascending, as `i64`s. Pins the exact surviving ROW SET (not
/// just the count), so a refused writer that left a phantom row, a checkpoint that
/// stranded a reader, or a reset that leaked a stale frame are all caught by value.
fn select_x_sorted(c: &mut Connection) -> Vec<i64> {
    let r = c.query("SELECT x FROM t ORDER BY x").unwrap();
    r.rows
        .iter()
        .map(|row| {
            assert_eq!(row.len(), 1, "SELECT x returns exactly one column");
            as_int(&row[0])
        })
        .collect()
}

/// Byte length of the `-wal` sidecar (0 if it does not exist). A committed frame grows
/// it; a checkpoint that resets the log shrinks it back toward its 32-byte header, and
/// TRUNCATE all the way to zero — so it is a filesystem-level witness of whether a reset
/// actually ran.
fn wal_len(tmp: &TempDb) -> u64 {
    std::fs::metadata(tmp.sidecar("-wal")).map(|m| m.len()).unwrap_or(0)
}

/// Switch a connection to WAL mode, asserting the documented one-row `"wal"` result.
fn to_wal(c: &mut Connection) {
    let r = c.query("PRAGMA journal_mode=WAL").unwrap();
    assert_eq!(r.rows.len(), 1, "PRAGMA journal_mode=WAL returns one row");
    match &r.rows[0][0] {
        Value::Text(s) => assert_eq!(s, "wal", "PRAGMA journal_mode=WAL reports \"wal\""),
        other => panic!("expected the text \"wal\", got {other:?}"),
    }
}

/// Create `t(x INTEGER)` and switch the file to WAL on a throwaway connection that is
/// then dropped. Both WAL mode and the empty table are committed to the shared file, so
/// the connections opened afterward operate in WAL mode over a clean, empty `-wal`.
fn setup(tmp: &TempDb) {
    let mut c = tmp.open();
    c.execute("CREATE TABLE t(x INTEGER)").unwrap();
    to_wal(&mut c);
}

/// Insert one row per value, each its own autocommit transaction (so each appends a
/// commit to the WAL). Runs on a caller-held OPEN connection so the appended frames stay
/// in the `-wal` (the last connection to close checkpoints the log away).
fn insert_each(c: &mut Connection, values: &[i64]) {
    for v in values {
        c.execute(&format!("INSERT INTO t VALUES ({v})")).unwrap();
    }
}

/// The three integer columns of a `PRAGMA wal_checkpoint(<mode>)` result row.
#[derive(Debug, Clone, Copy)]
struct Ckpt {
    busy: i64,
    log: i64,
    checkpointed: i64,
}

/// Run `PRAGMA wal_checkpoint(<mode>)` (bare when `mode` is empty) and return its one
/// documented row of three integers, asserting the shape (exactly one row, three
/// columns, all integers) unconditionally so a malformed result can never slip through.
fn checkpoint(c: &mut Connection, mode: &str) -> Ckpt {
    let sql = if mode.is_empty() {
        "PRAGMA wal_checkpoint".to_string()
    } else {
        format!("PRAGMA wal_checkpoint({mode})")
    };
    let r = c
        .query(&sql)
        .unwrap_or_else(|e| panic!("`{sql}` should be accepted in WAL mode, got {e:?}"));
    assert_eq!(r.rows.len(), 1, "`{sql}`: exactly one result row");
    assert_eq!(r.rows[0].len(), 3, "`{sql}`: three integer columns (busy, log, checkpointed)");
    Ckpt {
        busy: as_int(&r.rows[0][0]),
        log: as_int(&r.rows[0][1]),
        checkpointed: as_int(&r.rows[0][2]),
    }
}

/// Assert a checkpoint row is a COMPLETE drain: `busy == expected_busy`, the log is
/// non-empty, and every frame was folded back (`checkpointed == log`, pragma.html's
/// "the number of frames in the WAL that have been checkpointed" reaching all of them).
fn assert_complete_drain(row: Ckpt, expected_busy: i64, ctx: &str) {
    assert_eq!(row.busy, expected_busy, "{ctx}: busy column");
    assert!(row.log > 0, "{ctx}: the WAL holds frames (log {} > 0)", row.log);
    assert_eq!(
        row.checkpointed, row.log,
        "{ctx}: a complete drain folds every frame back (checkpointed {} == log {})",
        row.checkpointed, row.log
    );
}

/// Assert a checkpoint row is a reader-bounded PARTIAL drain: `busy == expected_busy`,
/// and `0 < checkpointed < log` — the drain advanced (the reader's snapshot mark is above
/// zero) but stopped below the log tail because a reader still needs the newer frames.
fn assert_partial_drain(row: Ckpt, expected_busy: i64, ctx: &str) {
    assert_eq!(row.busy, expected_busy, "{ctx}: busy column");
    assert!(
        row.checkpointed > 0 && row.checkpointed < row.log,
        "{ctx}: a reader-bounded partial drain (0 < checkpointed {} < log {})",
        row.checkpointed,
        row.log
    );
}

// ===========================================================================
// Scenario 1 — a RESTART / TRUNCATE checkpoint cannot reset the log while a
// reader holds a live snapshot: it reports busy == 1 and leaves the `-wal`
// un-reset, and the reader keeps reading its EXACT snapshot throughout.
// Spec: pragma.html #pragma_wal_checkpoint (busy == 1 when a RESTART/TRUNCATE
// checkpoint is blocked by an active reader) + wal.html §4.4 (the log may not be
// reset while a reader still needs a frame in it) + lang_transaction.html §2.1
// (the reader keeps its historic snapshot).
// ===========================================================================

/// Shared body for the two reset-seeking modes. A writer connection stays OPEN with the
/// seeded rows still in the `-wal`; a reader pins that exact snapshot; then the writer's
/// own connection issues `PRAGMA wal_checkpoint(<mode>)` in autocommit (holding no reader
/// mark of its own). Because a reader still holds the log, the reset is refused (busy==1)
/// and the `-wal` is NOT truncated, yet the drain itself still completes.
fn reset_checkpoint_is_busy_while_a_reader_pins_the_snapshot(mode: &str) {
    let tmp = TempDb::new();
    setup(&tmp);

    let mut w = tmp.open();
    let mut r = tmp.open();

    // Frames live in the -wal because `w` stays open (see `insert_each`).
    insert_each(&mut w, &[10, 20, 30]);

    // The reader pins its snapshot on all three rows — its FIRST read, before the
    // checkpoint runs (pin-timing note). No reader has been dropped, so this mark keeps
    // the log pinned for the checkpoint below.
    r.execute("BEGIN").unwrap();
    assert_eq!(select_x_sorted(&mut r), vec![10, 20, 30], "reader pins {{10,20,30}}");

    let pre = wal_len(&tmp);
    assert!(pre > 0, "the seeded frames are in the -wal before the checkpoint");

    // pragma.html: a RESTART/TRUNCATE checkpoint blocked by an active reader reports
    // busy == 1. The drain still folds every frame back (checkpointed == log) — only the
    // RESET is blocked, which is the whole distinction this scenario pins.
    let row = checkpoint(&mut w, mode);
    assert_complete_drain(row, 1, &format!("{mode} while a reader pins the snapshot"));

    // The reset did NOT happen: the -wal is unchanged (TRUNCATE did not zero it, RESTART
    // did not shrink it to its header). wal.html §4.4 forbids resetting past a reader.
    assert_eq!(
        wal_len(&tmp),
        pre,
        "{mode} must not reset the -wal while a reader holds the log"
    );

    // The reader keeps its EXACT snapshot across the blocked checkpoint — never stranded,
    // never advanced. Read twice to witness stability.
    assert_eq!(select_x_sorted(&mut r), vec![10, 20, 30], "reader unchanged right after the checkpoint");
    assert_eq!(select_x_sorted(&mut r), vec![10, 20, 30], "reader still holds its snapshot");

    // Once the reader ends, the same checkpoint now succeeds (busy == 0) and resets the
    // log — proving the earlier busy was purely the reader fence, and nothing corrupted.
    r.execute("COMMIT").unwrap();
    let after = checkpoint(&mut w, mode);
    assert_eq!(after.busy, 0, "{mode} succeeds once the reader has ended");
    if mode == "TRUNCATE" {
        assert_eq!(wal_len(&tmp), 0, "TRUNCATE zeroes the -wal on success");
    } else {
        assert!(wal_len(&tmp) < pre, "{mode} resets the -wal on success ({pre} -> shrunk)");
    }

    // No data loss anywhere: this connection and a fresh one both see all three rows.
    assert_eq!(select_x_sorted(&mut w), vec![10, 20, 30], "writer sees every row after the reset");
    let mut fresh = tmp.open();
    assert_eq!(select_x_sorted(&mut fresh), vec![10, 20, 30], "a fresh connection sees every row");
}

#[test]
fn restart_checkpoint_is_busy_while_a_reader_pins_the_snapshot() {
    reset_checkpoint_is_busy_while_a_reader_pins_the_snapshot("RESTART");
}

#[test]
fn truncate_checkpoint_is_busy_while_a_reader_pins_the_snapshot() {
    reset_checkpoint_is_busy_while_a_reader_pins_the_snapshot("TRUNCATE");
}

// ===========================================================================
// Scenario 2 — after the reader ends, a RESTART/TRUNCATE checkpoint succeeds
// (busy == 0), the -wal resets, and a writer that then commits NEW transactions
// reuses the log from the beginning (fresh salt/generation). A FRESH reader sees
// the fully-correct COMBINED state, with no stale pre-reset frame leaking back in.
// Spec: wal.html §4.4 (reset after a complete checkpoint with no reader; the next
// writer restarts the log; the salt change invalidates every pre-reset frame) +
// pragma.html #pragma_wal_checkpoint (TRUNCATE zeroes the -wal on success).
// ===========================================================================

#[test]
fn reset_succeeds_after_reader_ends_then_writer_reuses_the_log() {
    let tmp = TempDb::new();
    setup(&tmp);

    let mut w = tmp.open();
    let mut r = tmp.open();

    insert_each(&mut w, &[1, 2, 3]);

    // A reader pins {1,2,3}; while it holds the log, TRUNCATE is refused (busy == 1) and
    // the -wal is untouched — the guard scenario 1 pins, restated here as the precondition.
    r.execute("BEGIN").unwrap();
    assert_eq!(select_x_sorted(&mut r), vec![1, 2, 3], "reader pins {{1,2,3}}");
    let pre = wal_len(&tmp);
    let blocked = checkpoint(&mut w, "TRUNCATE");
    assert_eq!(blocked.busy, 1, "TRUNCATE is blocked while the reader holds the log");
    assert_eq!(wal_len(&tmp), pre, "the blocked TRUNCATE left the -wal untouched");

    // The reader ends: its snapshot mark is released, so nothing pins the log anymore.
    r.execute("COMMIT").unwrap();

    // Now TRUNCATE completes: busy == 0 and the -wal is zeroed (pragma.html). The three
    // rows now live in the database file, not the log.
    let done = checkpoint(&mut w, "TRUNCATE");
    assert_complete_drain(done, 0, "TRUNCATE after the reader ended");
    assert_eq!(wal_len(&tmp), 0, "TRUNCATE zeroed the -wal on success");

    // The writer commits NEW transactions. The log restarts from the beginning under a
    // fresh salt, so these frames occupy the same physical offsets the pre-reset frames
    // used — the exact hazard the salt/generation fence exists for.
    insert_each(&mut w, &[50, 51, 52]);
    assert!(wal_len(&tmp) > 0, "the reused log holds the new frames");

    // A FRESH connection (its own scan of the reset+reused log) sees the COMBINED state
    // EXACTLY — the checkpointed {1,2,3} from the db file plus the new {50,51,52} from the
    // reused log — with no stale pre-reset frame read back at a reused offset and no
    // duplicate. A leaked stale frame or a missed new frame changes this exact set.
    let mut fresh = tmp.open();
    assert_eq!(
        select_x_sorted(&mut fresh),
        vec![1, 2, 3, 50, 51, 52],
        "a fresh reader sees the exact combined state after reset + reuse"
    );

    // A second checkpoint (no reader) drains the reused log; the combined state is stable
    // and still exact through yet another fresh connection.
    let drain2 = checkpoint(&mut w, "RESTART");
    assert_complete_drain(drain2, 0, "RESTART drains the reused log with no reader");
    let mut fresh2 = tmp.open();
    assert_eq!(
        select_x_sorted(&mut fresh2),
        vec![1, 2, 3, 50, 51, 52],
        "the combined state is stable after draining the reused log"
    );
}

// ===========================================================================
// Scenario 3 — a reader holding an OLD snapshot (its mark below the log tail)
// stays isolated from a writer that commits far past it: repeated reads return
// the EXACT old snapshot, a concurrent RESTART checkpoint can only drain up to
// the reader's mark (busy == 1, partial) and cannot reset, and once the reader
// ends a fresh connection — and then a successful reset — see the full new state.
// This is the reader-mark / salt-generation fence: the reader never reads one of
// the newer frames (nor, structurally, a post-reset frame, since the reset is
// blocked while it is live) as if it belonged to its snapshot.
// Spec: lang_transaction.html §2.1 (historic snapshot) + wal.html §2/§4.4 (a
// checkpoint may not drain past the oldest reader's mark, nor reset while a reader
// holds the log; a WAL is identified by its salt values).
// ===========================================================================

#[test]
fn old_reader_snapshot_is_isolated_from_concurrent_writes_and_reset_is_blocked() {
    let tmp = TempDb::new();
    setup(&tmp);

    let mut w = tmp.open();
    let mut r = tmp.open();

    // The writer seeds {10,20} into the live -wal; the reader pins EXACTLY that as its
    // FIRST read (mark below the tail once the writer continues). Pin-timing note.
    insert_each(&mut w, &[10, 20]);
    r.execute("BEGIN").unwrap();
    assert_eq!(select_x_sorted(&mut r), vec![10, 20], "reader pins the OLD snapshot {{10,20}}");

    // The writer commits many more rows, growing the log well past the reader's mark.
    // Each is invisible to the reader's held snapshot (§2.1).
    let newer: Vec<i64> = (100..112).collect();
    insert_each(&mut w, &newer);

    // A RESTART checkpoint now: it may drain only up to the reader's mark (partial:
    // 0 < checkpointed < log) and cannot reset the log (busy == 1) — the reader still
    // needs the frames above its mark, so the log is neither reset nor drained past it.
    let pre = wal_len(&tmp);
    let row = checkpoint(&mut w, "RESTART");
    assert_partial_drain(row, 1, "RESTART with an old reader below the log tail");
    assert_eq!(wal_len(&tmp), pre, "the log is not reset while the old reader holds it");

    // The reader re-reads several times AFTER the heavy writes and the checkpoint attempt
    // and still sees EXACTLY its old snapshot — never one of the newer frames, never a
    // frame from the (blocked) reset. This is the isolation the fence guarantees.
    for _ in 0..3 {
        assert_eq!(
            select_x_sorted(&mut r),
            vec![10, 20],
            "the old reader keeps its exact snapshot across concurrent writes + checkpoint"
        );
    }

    // The reader ends; nothing pins the log now. A fresh connection sees the FULL new
    // state exactly (the seed plus every newer row).
    r.execute("COMMIT").unwrap();
    let mut expected: Vec<i64> = vec![10, 20];
    expected.extend(&newer);
    let mut fresh = tmp.open();
    assert_eq!(select_x_sorted(&mut fresh), expected, "a fresh connection sees the full new state");

    // And now that no reader holds the log, the reset finally succeeds and drains
    // everything, with the full state still exact afterward.
    let drained = checkpoint(&mut w, "RESTART");
    assert_complete_drain(drained, 0, "RESTART completes once the old reader has ended");
    assert!(wal_len(&tmp) < pre, "the log is reset once the old reader has ended");
    assert_eq!(select_x_sorted(&mut w), expected, "the writer sees the full state after the reset");
}

// ===========================================================================
// Scenario 4 — the PASSIVE/FULL vs RESTART/TRUNCATE distinction under a LIVE
// reader (pragma.html #pragma_wal_checkpoint). PASSIVE drains what it can and
// never blocks; FULL blocks only when it cannot drain the whole log; RESTART and
// TRUNCATE additionally want a reset and so block on ANY reader. Two readers pin
// different snapshots (latest vs old), and each mode's `(busy, log, checkpointed)`
// row is pinned for that reader, with the reader kept exact throughout. Each mode
// runs on its OWN fresh file so its row reflects a single checkpoint over an
// identical log (no nBackfill carry-over between modes).
//
// SCOPE NOTE — this scenario pins the mode distinction that is observable at the
// facade AND correct in this engine: `busy` under a live reader (no mode may reset
// past a reader, so `-wal` size is unchanged for every mode here for the same
// reason — the reader fence, NOT the mode). It deliberately does NOT pin the
// NO-READER case, where this engine has a KNOWN pre-existing divergence: the reset
// gate (`walstore.rs`: `did_reset = complete && !write_locked && readers.is_empty()`)
// is MODE-BLIND, so a completed PASSIVE/FULL checkpoint with no reader also resets
// and shrinks the `-wal`. pragma.html reserves the log restart for RESTART/TRUNCATE
// (and truncation for TRUNCATE alone), and wal.html §6 says a checkpoint "does not
// normally truncate the WAL file ... Instead, it merely causes SQLite to start
// overwriting the WAL file from the beginning" on the NEXT write. Fixing that is a
// writer-side deferred-rewind change that must also revisit
// `conformance_wal_durability.rs` (which asserts a bare checkpoint shrinks the
// `-wal`); it is a known follow-up, not pinned here, so this battery asserts
// nothing false about the no-reader path.
// ===========================================================================

/// Pin one mode's checkpoint row against a reader that pinned the LATEST snapshot
/// (its mark == the log tail): the drain can complete for every mode, but only
/// RESTART/TRUNCATE want a reset and so are blocked (busy == 1) by the live reader.
fn mode_row_under_latest_reader(mode: &str, expected_busy: i64) {
    let tmp = TempDb::new();
    setup(&tmp);
    let mut w = tmp.open();
    let mut r = tmp.open();

    insert_each(&mut w, &[10, 20, 30]);
    // The reader pins AFTER all writes, so its mark == the log tail (latest snapshot).
    r.execute("BEGIN").unwrap();
    assert_eq!(select_x_sorted(&mut r), vec![10, 20, 30], "latest reader pins {{10,20,30}}");
    let pre = wal_len(&tmp);

    let row = checkpoint(&mut w, mode);
    // A latest reader does not bound the drain, so every mode drains completely; the only
    // difference is whether a reset was wanted-but-blocked (busy).
    assert_complete_drain(row, expected_busy, &format!("{mode} under a latest reader"));
    // No mode resets while the reader is live (did-reset needs no active reader).
    assert_eq!(wal_len(&tmp), pre, "{mode} does not reset the -wal while the reader is live");
    assert_eq!(select_x_sorted(&mut r), vec![10, 20, 30], "the latest reader stays exact under {mode}");
    r.execute("COMMIT").unwrap();
}

#[test]
fn checkpoint_modes_under_a_latest_reader() {
    // PASSIVE/FULL complete without wanting a reset → busy 0. RESTART/TRUNCATE want a
    // reset the live reader forbids → busy 1 (yet the drain still completed).
    mode_row_under_latest_reader("PASSIVE", 0);
    mode_row_under_latest_reader("FULL", 0);
    mode_row_under_latest_reader("RESTART", 1);
    mode_row_under_latest_reader("TRUNCATE", 1);
}

/// Pin one mode's checkpoint row against a reader that pinned an OLD snapshot (its mark
/// below the log tail): the drain is bounded to the reader's mark (partial), so FULL now
/// also blocks (it cannot drain the whole log), while PASSIVE still never blocks.
fn mode_row_under_old_reader(mode: &str, expected_busy: i64) {
    let tmp = TempDb::new();
    setup(&tmp);
    let mut w = tmp.open();
    let mut r = tmp.open();

    insert_each(&mut w, &[10, 20]);
    // Reader pins {10,20} (mark below the tail once the writer continues).
    r.execute("BEGIN").unwrap();
    assert_eq!(select_x_sorted(&mut r), vec![10, 20], "old reader pins {{10,20}}");
    insert_each(&mut w, &[100, 101, 102]);
    let pre = wal_len(&tmp);

    let row = checkpoint(&mut w, mode);
    // The old reader bounds the drain below the tail: a partial drain (0 < checkpointed
    // < log) for every mode; PASSIVE stays busy 0, the reset/complete modes are busy 1.
    assert_partial_drain(row, expected_busy, &format!("{mode} under an old reader"));
    assert_eq!(wal_len(&tmp), pre, "{mode} does not reset the -wal while the old reader is live");
    assert_eq!(select_x_sorted(&mut r), vec![10, 20], "the old reader stays exact under {mode}");
    r.execute("COMMIT").unwrap();
}

#[test]
fn checkpoint_modes_under_an_old_reader() {
    // PASSIVE never blocks. FULL now blocks too (it cannot drain the whole log past the
    // old reader). RESTART/TRUNCATE block (cannot reset, and cannot even fully drain).
    mode_row_under_old_reader("PASSIVE", 0);
    mode_row_under_old_reader("FULL", 1);
    mode_row_under_old_reader("RESTART", 1);
    mode_row_under_old_reader("TRUNCATE", 1);
}

#[test]
fn noop_checkpoint_reports_counts_without_draining_under_a_reader() {
    // pragma.html: a NOOP wal_checkpoint copies no frame — it only reports the counts. It
    // never blocks (busy 0) and never resets, even with a live reader; `checkpointed` is 0
    // because nothing was folded back, while `log` still reports the frames present.
    let tmp = TempDb::new();
    setup(&tmp);
    let mut w = tmp.open();
    let mut r = tmp.open();

    insert_each(&mut w, &[10, 20, 30]);
    r.execute("BEGIN").unwrap();
    assert_eq!(select_x_sorted(&mut r), vec![10, 20, 30], "reader pins {{10,20,30}}");
    let pre = wal_len(&tmp);

    let row = checkpoint(&mut w, "NOOP");
    assert_eq!(row.busy, 0, "NOOP never blocks");
    assert!(row.log > 0, "NOOP still reports the frames present in the log");
    assert_eq!(row.checkpointed, 0, "NOOP copies no frame back (checkpointed == 0)");
    assert_eq!(wal_len(&tmp), pre, "NOOP leaves the -wal untouched");
    assert_eq!(select_x_sorted(&mut r), vec![10, 20, 30], "the reader is unaffected by NOOP");
    r.execute("COMMIT").unwrap();
}
