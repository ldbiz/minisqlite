//! Conformance battery: WAL / journalling control and **cross-connection
//! visibility** at the pinned facade (`minisqlite::Connection`).
//!
//! These cases pin the behavior that makes the WAL subsystem reachable through the
//! public API and that two connections to one on-disk file observe each other's
//! commits (the "each connection sees the other's commits" requirement).
//! Assertions are transcribed from the SQLite documentation, never
//! from whatever the engine currently returns.
//!
//! Spec sources:
//! - `spec/sqlite-doc/pragma.html` #pragma_journal_mode: the statement returns one
//!   row naming the resulting journal mode ("wal", "delete", "memory", ...).
//! - `spec/sqlite-doc/pragma.html` #pragma_wal_checkpoint: returns one row of three
//!   integers `(busy, log, checkpointed)`; a non-WAL database reports `(0, -1, -1)`.
//! - `spec/sqlite-doc/fileformat2.html` §1.3: file-format write/read versions of 2
//!   select WAL mode; the change counter (offset 24) advances on each commit.
//! - `spec/sqlite-doc/wal.html`: a WAL database's readers see committed transactions
//!   from other connections.
//!
//! Each case is its own `#[test]` so one discrepancy fails exactly that behavior.

use minisqlite::{Connection, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Temp-file harness: a unique on-disk path per test, cleaned up even on panic.
// (Each Rust integration test is its own binary, so this small fixture is kept
// local rather than shared — the standard idiom for `tests/*.rs`.)
// ---------------------------------------------------------------------------

/// An RAII guard over a unique temporary database path. `Drop` deletes the database
/// AND its `-journal` / `-wal` / `-shm` sidecars, so nothing is left behind even
/// when an assertion panics.
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
        path.push(format!("minisqlite_conc_{pid}_{n}_{nanos}.db"));
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

/// A `Value` as an `i64` (panicking otherwise). `Value` deliberately does not
/// implement `PartialEq`, so tests destructure it rather than comparing directly.
fn as_int(v: &Value) -> i64 {
    match v {
        Value::Integer(n) => *n,
        other => panic!("expected an integer, got {other:?}"),
    }
}

/// `SELECT count(*) FROM t` as an `i64` (the tables here are all named `t`).
fn count_t(c: &mut Connection) -> i64 {
    let r = c.query("SELECT count(*) FROM t").unwrap();
    assert_eq!(r.rows.len(), 1, "count(*) returns exactly one row");
    as_int(&r.rows[0][0])
}

/// Byte length of the `-wal` sidecar (0 if it does not exist). A committed WAL frame
/// grows it; a completed checkpoint that drains+resets the log shrinks it back to its
/// header, so the size is a filesystem-level witness of whether a checkpoint ran.
fn wal_len(tmp: &TempDb) -> u64 {
    std::fs::metadata(tmp.sidecar("-wal")).map(|m| m.len()).unwrap_or(0)
}

/// The single `journal_mode` string a `PRAGMA journal_mode[=…]` returns.
fn journal_mode(c: &mut Connection, sql: &str) -> String {
    let r = c.query(sql).unwrap();
    assert_eq!(r.columns, vec!["journal_mode".to_string()], "one column named journal_mode");
    assert_eq!(r.rows.len(), 1, "PRAGMA journal_mode returns exactly one row: {sql}");
    match &r.rows[0][0] {
        Value::Text(s) => s.clone(),
        other => panic!("expected a text mode, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Witness A: PRAGMA journal_mode is reachable and returns a naming row.
// ---------------------------------------------------------------------------

#[test]
fn journal_mode_wal_returns_one_wal_row() {
    // Test A: `PRAGMA journal_mode=WAL` must return exactly one row
    // naming the resulting mode "wal" (previously an unhandled no-op returning []).
    let tmp = TempDb::new();
    let mut c = tmp.open();
    c.execute("CREATE TABLE t(x)").unwrap();
    assert_eq!(journal_mode(&mut c, "PRAGMA journal_mode=WAL"), "wal");
}

#[test]
fn journal_mode_get_defaults_to_delete_on_a_fresh_file() {
    // A freshly created file database is in the rollback (delete) journal mode.
    let tmp = TempDb::new();
    let mut c = tmp.open();
    c.execute("CREATE TABLE t(x)").unwrap();
    assert_eq!(journal_mode(&mut c, "PRAGMA journal_mode"), "delete");
}

#[test]
fn journal_mode_wal_persists_across_reopen() {
    // Setting WAL writes the version-2 header; a reopened connection reads that
    // header and reports (and uses) WAL mode. This is the mechanism by which the
    // WAL backing becomes reachable through the public API.
    let tmp = TempDb::new();
    {
        let mut c = tmp.open();
        c.execute("CREATE TABLE t(x)").unwrap();
        assert_eq!(journal_mode(&mut c, "PRAGMA journal_mode=WAL"), "wal");
    }
    let mut c = tmp.open();
    assert_eq!(journal_mode(&mut c, "PRAGMA journal_mode"), "wal");
}

#[test]
fn journal_mode_can_switch_back_to_delete() {
    let tmp = TempDb::new();
    let mut c = tmp.open();
    c.execute("CREATE TABLE t(x)").unwrap();
    assert_eq!(journal_mode(&mut c, "PRAGMA journal_mode=WAL"), "wal");
    assert_eq!(journal_mode(&mut c, "PRAGMA journal_mode=DELETE"), "delete");
}

#[test]
fn in_memory_journal_mode_is_memory() {
    // An in-memory database reports journal_mode=memory and ignores a change,
    // matching SQLite (there is no backing file to reopen in another mode).
    let mut c = Connection::open_in_memory().unwrap();
    c.execute("CREATE TABLE t(x)").unwrap();
    assert_eq!(journal_mode(&mut c, "PRAGMA journal_mode"), "memory");
    assert_eq!(journal_mode(&mut c, "PRAGMA journal_mode=WAL"), "memory");
}

// ---------------------------------------------------------------------------
// PRAGMA wal_checkpoint returns the documented 3-integer row.
// ---------------------------------------------------------------------------

#[test]
fn wal_checkpoint_on_non_wal_reports_minus_one() {
    // On a rollback-mode database there is nothing to checkpoint; SQLite reports
    // (busy=0, log=-1, checkpointed=-1).
    let tmp = TempDb::new();
    let mut c = tmp.open();
    c.execute("CREATE TABLE t(x)").unwrap();
    let r = c.query("PRAGMA wal_checkpoint").unwrap();
    assert_eq!(r.rows.len(), 1, "wal_checkpoint returns exactly one row");
    let row = &r.rows[0];
    assert_eq!(row.len(), 3, "wal_checkpoint returns three columns");
    assert_eq!(
        (as_int(&row[0]), as_int(&row[1]), as_int(&row[2])),
        (0, -1, -1),
        "non-WAL wal_checkpoint reports (0, -1, -1)"
    );
}

#[test]
fn wal_checkpoint_in_wal_mode_runs_and_returns_a_row() {
    // In WAL mode a checkpoint runs (folding the log back into the db file) and
    // returns a busy=0 row; data remains intact after it.
    let tmp = TempDb::new();
    {
        let mut c = tmp.open();
        c.execute("CREATE TABLE t(x INTEGER)").unwrap();
        assert_eq!(journal_mode(&mut c, "PRAGMA journal_mode=WAL"), "wal");
    }
    let mut c = tmp.open();
    c.execute("INSERT INTO t VALUES (1), (2), (3)").unwrap();

    // The commit appended frames to the -wal, so it now holds more than its header.
    let wal_before = wal_len(&tmp);
    assert!(wal_before > 32, "committed frames live in the -wal before the checkpoint (len {wal_before})");

    let r = c.query("PRAGMA wal_checkpoint").unwrap();
    assert_eq!(r.rows.len(), 1, "wal_checkpoint returns one row in WAL mode");
    assert_eq!(as_int(&r.rows[0][0]), 0, "checkpoint reports busy=0");

    // Pin the DRAIN, not just the row shape: a completed checkpoint folds every frame
    // back into the db file and resets the log, so the -wal shrinks. A no-op checkpoint
    // would leave the frames in place and this assert would fail.
    let wal_after = wal_len(&tmp);
    assert!(wal_after < wal_before, "checkpoint drained+reset the -wal ({wal_before} -> {wal_after})");
    assert_eq!(count_t(&mut c), 3, "data survives a checkpoint");
}

// ---------------------------------------------------------------------------
// Witness B: two connections see each other's commits.
// ---------------------------------------------------------------------------

#[test]
fn second_connection_sees_first_connections_commit_rollback_mode() {
    // Test B (default rollback journal mode): c2 opens after c1's
    // first commit and must see c1's LATER commit too — not a snapshot frozen at
    // open. Real sqlite: c2 sees 2 rows after c1's second insert.
    let tmp = TempDb::new();
    let mut c1 = tmp.open();
    c1.execute("CREATE TABLE t(x INTEGER)").unwrap();
    c1.execute("INSERT INTO t VALUES (1)").unwrap();

    let mut c2 = tmp.open();
    assert_eq!(count_t(&mut c2), 1, "c2 sees the 1 row present when it opened");

    c1.execute("INSERT INTO t VALUES (2)").unwrap();
    assert_eq!(count_t(&mut c2), 2, "c2 sees c1's later commit (not frozen at open)");
}

#[test]
fn second_connection_sees_first_connections_commit_wal_mode() {
    // The WAL-mode counterpart: after the file is put in WAL mode, two
    // connections coordinate through the shared WAL, and a reader's per-statement
    // snapshot refresh makes the writer's commits visible.
    let tmp = TempDb::new();
    {
        let mut c = tmp.open();
        c.execute("CREATE TABLE t(x INTEGER)").unwrap();
        assert_eq!(journal_mode(&mut c, "PRAGMA journal_mode=WAL"), "wal");
    }

    let mut c1 = tmp.open();
    assert_eq!(journal_mode(&mut c1, "PRAGMA journal_mode"), "wal", "reopened in WAL mode");
    c1.execute("INSERT INTO t VALUES (1)").unwrap();

    let mut c2 = tmp.open();
    assert_eq!(count_t(&mut c2), 1, "c2 sees the row present at open (WAL)");

    c1.execute("INSERT INTO t VALUES (2)").unwrap();
    assert_eq!(count_t(&mut c2), 2, "c2 sees c1's later commit (WAL snapshot refresh)");
}

#[test]
fn writer_builds_on_the_other_connections_committed_state() {
    // A second connection that WRITES must build on the first connection's commit,
    // not a stale base — otherwise its own commit would drop rows. Both connections
    // insert; the final total must include every row.
    let tmp = TempDb::new();
    let mut c1 = tmp.open();
    c1.execute("CREATE TABLE t(x INTEGER)").unwrap();
    c1.execute("INSERT INTO t VALUES (1)").unwrap();

    let mut c2 = tmp.open();
    c2.execute("INSERT INTO t VALUES (2)").unwrap();
    c1.execute("INSERT INTO t VALUES (3)").unwrap();

    // Every connection, reading fresh, sees all three committed rows.
    assert_eq!(count_t(&mut c1), 3, "c1 sees all commits");
    assert_eq!(count_t(&mut c2), 3, "c2 sees all commits");

    let mut c3 = tmp.open();
    assert_eq!(count_t(&mut c3), 3, "a third fresh connection sees all commits");
}
