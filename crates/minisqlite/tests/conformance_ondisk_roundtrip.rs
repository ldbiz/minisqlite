//! Conformance battery: on-disk **durability round-trip** at the pinned facade.
//!
//! Every case here writes through a FILE-backed `minisqlite::Connection::open(path)`,
//! closes the connection (drops it), reopens the SAME file, and asserts the committed
//! state came back. This exercises the on-disk format's WRITE+read path end to end and
//! single-connection durability ("committed data survives a
//! reopen; a rolled-back transaction leaves no trace").
//!
//! Assertions are transcribed from the SQLite documentation, never from whatever the
//! engine currently returns: a failing case is the intended signal that the engine
//! diverges from the spec (a real durability/format discrepancy) and is left as a
//! genuine failing assertion rather than weakened to pass.
//!
//! Spec sources:
//! - `spec/sqlite-doc/atomiccommit.html`: a committed transaction is durable; an
//!   aborted/rolled-back one restores the pre-transaction image.
//! - `spec/sqlite-doc/fileformat2.html`: the page-1 header (§1.3, incl. the 4-byte
//!   `user_version` at offset 60) and the record serial types (§2.1) that must
//!   round-trip a value's storage class and bytes exactly.
//! - `spec/sqlite-doc/datatype3.html` §3.1: a column with no declared type has BLOB
//!   (NONE) affinity, so a stored value keeps its original storage class.
//! - `spec/sqlite-doc/lang_createtable.html` (ROWIDs): a freshly assigned rowid is one
//!   greater than the largest rowid currently in the table, so after reopen the next
//!   INSERT continues the sequence rather than restarting at 1.
//!
//! Each case is its own `#[test]` (one durable behavior) so one discrepancy fails
//! exactly that case rather than masking the rest.

mod conformance;
use conformance::*;

// A file-backed connection is opened DIRECTLY (the harness `mem()` is in-memory only);
// the harness assert helpers all take `&mut Connection`, so they work on it unchanged.
use minisqlite::{Connection, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Temp-file harness: a unique on-disk path per test, cleaned up even on panic.
// ---------------------------------------------------------------------------

/// An RAII guard over a unique temporary database path. Every test gets its own file
/// (isolated + idempotent), and `Drop` deletes the database AND its `-journal` /
/// `-wal` / `-shm` sidecars — so cleanup happens even when an assertion panics and
/// nothing is left behind in the temp dir. No `.db` file is ever committed.
struct TempDb {
    path: PathBuf,
}

impl TempDb {
    /// Build a fresh, unique path under the system temp dir and remove any stale file
    /// (and sidecars) that a previous crashed run might have left at the same name.
    /// Uniqueness combines the process id, a per-process atomic counter, and a
    /// nanosecond timestamp so parallel test threads never collide.
    fn new() -> TempDb {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
        let mut path = std::env::temp_dir();
        path.push(format!("minisqlite_conf_{pid}_{n}_{nanos}.db"));
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

    /// The `<path><suffix>` sibling (e.g. `-journal`), by appending to the FULL path
    /// including its extension — SQLite's own naming convention (`foo.db-journal`).
    fn sidecar(&self, suffix: &str) -> PathBuf {
        let mut s = self.path.as_os_str().to_os_string();
        s.push(suffix);
        PathBuf::from(s)
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

// ---------------------------------------------------------------------------
// Basic persistence: schema + data survive close and reopen.
// ---------------------------------------------------------------------------

#[test]
fn basic_persist_schema_and_data() {
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
        exec(&mut db, "INSERT INTO t VALUES (1, 'x'), (2, 'y')");
        // Sanity within the same connection before it is closed.
        assert_rows(
            &mut db,
            "SELECT a, b FROM t ORDER BY a",
            &[vec![int(1), text("x")], vec![int(2), text("y")]],
        );
    } // the connection is dropped here (closed)

    // Reopen the SAME path: both the schema and the rows must have survived.
    let mut db = tmp.open();
    assert_rows(
        &mut db,
        "SELECT a, b FROM t ORDER BY a",
        &[vec![int(1), text("x")], vec![int(2), text("y")]],
    );
}

#[test]
fn empty_table_schema_survives_reopen() {
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
    }

    // The schema persists even with no rows: the columns come back and the table is
    // empty (a durable CREATE TABLE with an empty b-tree, not a lost definition).
    let mut db = tmp.open();
    assert_columns(&mut db, "SELECT a, b FROM t", &["a", "b"]);
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
}

#[test]
fn multiple_tables_persist_across_reopen() {
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE a(x INTEGER)");
        exec(&mut db, "CREATE TABLE b(y TEXT)");
        exec(&mut db, "INSERT INTO a VALUES (10), (20)");
        exec(&mut db, "INSERT INTO b VALUES ('p'), ('q')");
    }

    let mut db = tmp.open();
    assert_rows(&mut db, "SELECT x FROM a ORDER BY x", &[vec![int(10)], vec![int(20)]]);
    assert_rows(&mut db, "SELECT y FROM b ORDER BY y", &[vec![text("p")], vec![text("q")]]);
}

#[test]
fn index_persists_across_reopen() {
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
        exec(&mut db, "CREATE INDEX idx_t_a ON t(a)");
        exec(&mut db, "INSERT INTO t VALUES (1, 'one'), (2, 'two'), (3, 'three')");
    }

    let mut db = tmp.open();
    // The index's `sqlite_schema` row survives (the table has no UNIQUE/PK constraint,
    // so `idx_t_a` is the only index row).
    assert_rows(
        &mut db,
        "SELECT type, name FROM sqlite_master WHERE type = 'index' ORDER BY name",
        &[vec![text("index"), text("idx_t_a")]],
    );
    // And a lookup on the indexed column still returns the right row after reopen.
    assert_rows(&mut db, "SELECT a, b FROM t WHERE a = 2", &[vec![int(2), text("two")]]);
}

// ---------------------------------------------------------------------------
// Value fidelity: every storage class round-trips exactly across a reopen.
// ---------------------------------------------------------------------------

#[test]
fn value_fidelity_all_storage_classes_survive_reopen() {
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        // Columns with NO declared type have BLOB (NONE) affinity (datatype3 §3.1), so
        // each value keeps its original storage class — nothing is coerced on store.
        exec(&mut db, "CREATE TABLE t(i, r, tx, b, n)");
        exec(&mut db, "INSERT INTO t VALUES (42, 3.5, 'hello', x'00ff10', NULL)");
    }

    let mut db = tmp.open();
    // Values round-trip exactly (value_eq is class-sensitive: Integer != Real).
    assert_rows(
        &mut db,
        "SELECT i, r, tx, b, n FROM t",
        &[vec![int(42), real(3.5), text("hello"), blob(&[0x00, 0xff, 0x10]), null()]],
    );
    // And the stored storage class is preserved (typeof reports the concrete variant).
    assert_rows(
        &mut db,
        "SELECT typeof(i), typeof(r), typeof(tx), typeof(b), typeof(n) FROM t",
        &[vec![text("integer"), text("real"), text("text"), text("blob"), text("null")]],
    );
}

#[test]
fn blob_bytes_survive_reopen_including_zero() {
    let tmp = TempDb::new();
    // A blob spanning 0x00 and 0xFF: catches any C-string-style truncation at the NUL
    // or sign-handling slip in the serial-type write/read path (fileformat2 §2.1).
    let bytes: &[u8] = &[0x00, 0x01, 0x7f, 0x80, 0xfe, 0xff];
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(b)");
        exec(&mut db, "INSERT INTO t VALUES (x'00017f80feff')");
    }

    let mut db = tmp.open();
    assert_rows(&mut db, "SELECT b FROM t", &[vec![blob(bytes)]]);
    assert_scalar(&mut db, "SELECT typeof(b) FROM t", text("blob"));
}

#[test]
fn real_value_survives_reopen_exactly() {
    let tmp = TempDb::new();
    // The closest f64 to pi (full mantissa). fileformat2 serial type 7 stores a REAL
    // as an 8-byte big-endian IEEE-754 double, so the exact bits must round-trip —
    // hence an EXACT compare (value_eq uses f64::total_cmp), not an approximate one.
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(r REAL)");
        exec(&mut db, "INSERT INTO t VALUES (3.141592653589793)");
    }

    let mut db = tmp.open();
    assert_rows(&mut db, "SELECT r FROM t", &[vec![real(3.141592653589793)]]);
    assert_scalar(&mut db, "SELECT typeof(r) FROM t", text("real"));
}

#[test]
fn integer_serial_type_widths_survive_reopen() {
    let tmp = TempDb::new();
    // fileformat2 §2.1 encodes an INTEGER in the smallest serial type that fits: types
    // 1/2/3/4/5/6 are 1/2/3/4/6/8 bytes, and the special types 8 and 9 store the values
    // 0 and 1 in ZERO payload bytes. A bug confined to one width (or to the 0/1
    // zero-length encodings) would round-trip the others fine, so we persist one row per
    // boundary and require every value AND its INTEGER class to survive the reopen.
    let values: &[i64] = &[
        0,                    // serial type 8 (zero-length payload)
        1,                    // serial type 9 (zero-length payload)
        -1,                   // type 1
        127,
        -128,                 // type 1 bounds
        128,
        -129,                 // type 2
        32767,
        -32768,               // type 2 bounds
        8388607,
        -8388608,             // type 3 bounds
        2147483647,
        -2147483648,          // type 4 bounds
        140737488355327,      // type 5 (2^47 - 1)
        9223372036854775807,  // type 6 (i64::MAX)
        -9223372036854775807, // type 6 (SQLite parses the i64::MIN literal as REAL, so we
                              // use -i64::MAX, the most-negative value writable as an int literal)
    ];

    {
        let mut db = tmp.open();
        // `v` has NONE affinity (no declared type), so an integer keeps its INTEGER class.
        exec(&mut db, "CREATE TABLE t(k INTEGER, v)");
        // One batched INSERT (not a query per row): `(0, 0), (1, 1), (2, -1), …`.
        let tuples: Vec<String> =
            values.iter().enumerate().map(|(k, v)| format!("({k}, {v})")).collect();
        exec(&mut db, &format!("INSERT INTO t VALUES {}", tuples.join(", ")));
    }

    let mut db = tmp.open();
    let expected: Vec<Vec<Value>> = values.iter().map(|&v| vec![int(v)]).collect();
    assert_rows(&mut db, "SELECT v FROM t ORDER BY k", &expected);
    // Every stored value came back with INTEGER storage class (not coerced to REAL/TEXT).
    assert_scalar(
        &mut db,
        "SELECT count(*) FROM t WHERE typeof(v) = 'integer'",
        int(values.len() as i64),
    );
}

#[test]
fn real_boundary_values_survive_reopen() {
    let tmp = TempDb::new();
    // Serial type 7 (fileformat2 §2.1) stores a REAL as 8 big-endian IEEE-754 bytes, so
    // the exact bit pattern must round-trip for negatives, extreme magnitudes, and a
    // denormal — value_eq compares via f64::total_cmp, so this is an EXACT bit check.
    // Each literal is formatted with Rust's shortest round-trippable Debug form, which
    // parses back to the identical f64 under IEEE round-to-nearest.
    let values: &[f64] = &[
        -1.5,
        2.25,
        1.0e308,                  // near f64::MAX magnitude
        5.0e-324,                 // smallest positive subnormal double
        -2.2250738585072014e-308, // -DBL_MIN (smallest-magnitude negative normal)
        -3.141592653589793e-10,
    ];

    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(k INTEGER, v)"); // NONE affinity keeps the REAL class
        let tuples: Vec<String> =
            values.iter().enumerate().map(|(k, v)| format!("({k}, {v:?})")).collect();
        exec(&mut db, &format!("INSERT INTO t VALUES {}", tuples.join(", ")));
    }

    let mut db = tmp.open();
    let expected: Vec<Vec<Value>> = values.iter().map(|&v| vec![real(v)]).collect();
    assert_rows(&mut db, "SELECT v FROM t ORDER BY k", &expected);
    assert_scalar(
        &mut db,
        "SELECT count(*) FROM t WHERE typeof(v) = 'real'",
        int(values.len() as i64),
    );
}

#[test]
fn negative_zero_real_survives_reopen() {
    let tmp = TempDb::new();
    // IEEE-754 distinguishes -0.0 from +0.0 by the sign bit, and serial type 7 stores all
    // 8 bytes, so the sign of zero must survive the reopen (value_eq via f64::total_cmp
    // treats -0.0 and +0.0 as DISTINCT). Isolated in its own test so a sign-of-zero
    // discrepancy is pinpointed rather than masked among the other real boundaries.
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(v)");
        exec(&mut db, "INSERT INTO t VALUES (-0.0)");
    }

    let mut db = tmp.open();
    assert_rows(&mut db, "SELECT v FROM t", &[vec![real(-0.0)]]);
    assert_scalar(&mut db, "SELECT typeof(v) FROM t", text("real"));
}

// ---------------------------------------------------------------------------
// PRAGMA user_version: stored in the page-1 header, survives reopen.
// ---------------------------------------------------------------------------

#[test]
fn pragma_user_version_persists_across_reopen() {
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(&mut db, "PRAGMA user_version = 7");
        // Reads back within the same connection.
        assert_rows(&mut db, "PRAGMA user_version", &[vec![int(7)]]);
    }

    // The value lives in the page-1 header (fileformat2 §1.3, offset 60), so a reopen
    // that re-reads the header must report the same 7.
    let mut db = tmp.open();
    assert_rows(&mut db, "PRAGMA user_version", &[vec![int(7)]]);
}

// ---------------------------------------------------------------------------
// Transactions: a committed txn persists; a rolled-back one leaves no trace.
// ---------------------------------------------------------------------------

#[test]
fn committed_transaction_persists_across_reopen() {
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(a INTEGER)");
        exec(&mut db, "BEGIN");
        exec(&mut db, "INSERT INTO t VALUES (10)");
        exec(&mut db, "INSERT INTO t VALUES (20)");
        exec(&mut db, "COMMIT");
    }

    let mut db = tmp.open();
    assert_rows_unordered(&mut db, "SELECT a FROM t", &[vec![int(10)], vec![int(20)]]);
}

#[test]
fn rolled_back_transaction_absent_after_reopen() {
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(a INTEGER)");
        exec(&mut db, "INSERT INTO t VALUES (1)"); // autocommit -> durable
        exec(&mut db, "BEGIN");
        exec(&mut db, "INSERT INTO t VALUES (2)"); // to be rolled back
        exec(&mut db, "ROLLBACK");
        // Within the connection, only the committed row remains.
        assert_rows_unordered(&mut db, "SELECT a FROM t", &[vec![int(1)]]);
    }

    // After reopen, the rolled-back row left no trace on disk.
    let mut db = tmp.open();
    assert_rows_unordered(&mut db, "SELECT a FROM t", &[vec![int(1)]]);
}

#[test]
fn rolled_back_ddl_absent_after_reopen() {
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE kept(a INTEGER)");
        exec(&mut db, "INSERT INTO kept VALUES (1)");
        exec(&mut db, "BEGIN");
        exec(&mut db, "CREATE TABLE gone(x INTEGER)"); // schema change to be rolled back
        exec(&mut db, "ROLLBACK");
        // Within the connection the rolled-back table is gone from the schema...
        assert_query_error(&mut db, "SELECT x FROM gone");
        // ...and the table created before the transaction is untouched.
        assert_rows(&mut db, "SELECT a FROM kept", &[vec![int(1)]]);
    }

    // After reopen, the rolled-back CREATE TABLE left no `sqlite_schema` row on disk: the
    // table is still absent and the pre-existing one still present. This exercises the
    // rollback-journal path for a schema change, not just for row data.
    let mut db = tmp.open();
    assert_query_error(&mut db, "SELECT x FROM gone");
    assert_rows(&mut db, "SELECT a FROM kept", &[vec![int(1)]]);
}

// ---------------------------------------------------------------------------
// ROWID sequence: after reopen a new INSERT continues from max(rowid)+1.
// ---------------------------------------------------------------------------

#[test]
fn integer_primary_key_sequence_continues_after_reopen() {
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT)");
        // `a` is the rowid alias; with no explicit value it auto-assigns 1, 2, 3.
        exec(&mut db, "INSERT INTO t(b) VALUES ('one'), ('two'), ('three')");
        assert_rows(
            &mut db,
            "SELECT a, b FROM t ORDER BY a",
            &[vec![int(1), text("one")], vec![int(2), text("two")], vec![int(3), text("three")]],
        );
    }

    // The next INSERT after reopen must continue at max(rowid)+1 == 4, not restart at 1
    // — the on-disk b-tree's largest key is what SQLite reads to pick the next rowid.
    let mut db = tmp.open();
    exec(&mut db, "INSERT INTO t(b) VALUES ('four')");
    assert_rows(
        &mut db,
        "SELECT a, b FROM t ORDER BY a",
        &[
            vec![int(1), text("one")],
            vec![int(2), text("two")],
            vec![int(3), text("three")],
            vec![int(4), text("four")],
        ],
    );
}

#[test]
fn implicit_rowid_sequence_continues_after_reopen() {
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(b TEXT)"); // a plain rowid table (no IPK)
        exec(&mut db, "INSERT INTO t(b) VALUES ('one'), ('two'), ('three')");
        assert_rows(
            &mut db,
            "SELECT rowid, b FROM t ORDER BY rowid",
            &[vec![int(1), text("one")], vec![int(2), text("two")], vec![int(3), text("three")]],
        );
    }

    let mut db = tmp.open();
    exec(&mut db, "INSERT INTO t(b) VALUES ('four')"); // must take rowid 4
    assert_rows(
        &mut db,
        "SELECT rowid, b FROM t ORDER BY rowid",
        &[
            vec![int(1), text("one")],
            vec![int(2), text("two")],
            vec![int(3), text("three")],
            vec![int(4), text("four")],
        ],
    );
}

// ---------------------------------------------------------------------------
// UPDATE / DELETE also persist across reopen (the write path both ways).
// ---------------------------------------------------------------------------

#[test]
fn update_persists_across_reopen() {
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
        exec(&mut db, "INSERT INTO t VALUES (1, 'old'), (2, 'keep')");
        exec(&mut db, "UPDATE t SET b = 'new' WHERE a = 1");
    }

    let mut db = tmp.open();
    assert_rows(
        &mut db,
        "SELECT a, b FROM t ORDER BY a",
        &[vec![int(1), text("new")], vec![int(2), text("keep")]],
    );
}

#[test]
fn delete_persists_across_reopen() {
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(a INTEGER)");
        exec(&mut db, "INSERT INTO t VALUES (1), (2), (3)");
        exec(&mut db, "DELETE FROM t WHERE a = 2");
    }

    let mut db = tmp.open();
    assert_rows_unordered(&mut db, "SELECT a FROM t", &[vec![int(1)], vec![int(3)]]);
}
