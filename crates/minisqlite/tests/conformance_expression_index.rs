//! Conformance battery: **indexes on expressions** (`CREATE INDEX i ON t(a+b)` /
//! `t(lower(a))`), end to end at the pinned `minisqlite` facade.
//!
//! Every assertion is transcribed from the SQLite documentation, never from what the
//! engine currently returns — a failing case is the intended signal that the engine
//! diverges from the spec; it is never weakened to pass.
//!
//! Spec sources:
//! - `spec/sqlite-doc/lang_createindex.html` §1.2 ("Indexes On Expressions") and the
//!   `expridx.html` companion: an index key may be a general expression over the indexed
//!   table's own columns; the index is a redundant, MAINTAINED copy keyed by that
//!   expression's value per row.
//! - `spec/sqlite-doc/lang_createindex.html` §1.1 ("Unique Indexes"): a UNIQUE index
//!   forbids duplicate entries — "Any attempt to insert a duplicate entry will result in
//!   an error" — while all NULLs are distinct. This is the lever these tests pull: a
//!   UNIQUE expression index can only raise a conflict if the engine actually MAINTAINS
//!   it (evaluates the key expression on INSERT/UPDATE/DELETE and on CREATE-INDEX
//!   backfill). A value-only `SELECT` would pass even over an UNMAINTAINED index (a
//!   SeqScan recomputes the expression), so it does NOT prove maintenance; a UNIQUE
//!   conflict does.
//! - `spec/sqlite-doc/lang_conflict.html`: the default ABORT algorithm raises
//!   `SQLITE_CONSTRAINT` and backs out the whole failing statement (prior committed
//!   statements are preserved).
//! - `spec/sqlite-doc/foreignkeys.html` §4.3 ("ON DELETE ... Actions"): `CASCADE` deletes
//!   each mapped child row and `SET NULL` nulls its FK columns; §6 treats those actions as
//!   trigger programs, so a child's indexes are MAINTAINED by them. An expression index on
//!   the child must therefore be evaluated and maintained through the cascade too (a UNIQUE
//!   expression index on the child is again the lever: a freed/moved key proves it). Gated on
//!   `PRAGMA foreign_keys` (default OFF).
//! - Single-connection durability: committed data — here including a
//!   reconstructed-on-load expression index — survives a reopen.
//!
//! SQLite's UNIQUE-violation message names the key COLUMNS for an ordinary index
//! (`table.col`), but an index on an EXPRESSION has no column name to print, so it reports
//! `index '<name>'` instead (verified against SQLite's error wording for expression
//! indexes). One case pins that exact detail; the rest assert the error KIND
//! (`Error::Constraint`, i.e. `SQLITE_CONSTRAINT`).
//!
//! Each behavior is its own `#[test]` so one discrepancy fails exactly that case.

mod conformance;
use conformance::*;

// `Connection` and `Error` are NOT re-exported through `conformance::*` (the harness
// imports them privately); pull them in directly. `Error` types the constraint-kind check;
// `Connection` types the file-backed reopen helper and the shared helpers below.
use minisqlite::{Connection, Error};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Shared helpers.
// ---------------------------------------------------------------------------

/// Assert `sql` fails specifically with a constraint error (`SQLITE_CONSTRAINT`), not
/// merely some error — a `Sql`/`Io`/`Format` error here would be the wrong KIND.
/// Returns the error so a caller can additionally inspect its text.
fn assert_constraint_error(db: &mut Connection, sql: &str) -> Error {
    let e = assert_exec_error(db, sql);
    assert!(
        matches!(e, Error::Constraint(_)),
        "expected a Constraint error (SQLITE_CONSTRAINT)\n  sql: {sql}\n  actual: {e:?}"
    );
    e
}

/// How many indexes named `name` the schema reports (0 or 1). Reads `sqlite_master`
/// (page 1 directly), so a rolled-back CREATE leaves nothing for it to see — the check the
/// backfill-atomicity cases rely on.
fn index_count(db: &mut Connection, name: &str) -> i64 {
    let sql = format!("SELECT count(*) FROM sqlite_master WHERE type='index' AND name='{name}'");
    let qr = query(db, &sql);
    match &qr.rows[0][0] {
        minisqlite::Value::Integer(n) => *n,
        other => panic!("count(*) should be an Integer, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// INSERT maintenance: a UNIQUE expression index rejects a duplicate KEY VALUE
// even when the raw column tuples differ.
// ---------------------------------------------------------------------------

#[test]
fn unique_expression_index_rejects_duplicate_key_on_insert() {
    // §1.2 + §1.1: the index keys on `a+b`, so (1,2) and (2,1) — different rows, SAME
    // key 3 — collide. The second INSERT must raise a UNIQUE error, which can only happen
    // if INSERT evaluates the key expression and maintains the index.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "CREATE UNIQUE INDEX u ON t(a + b)");
    exec(&mut db, "INSERT INTO t VALUES (1, 2)"); // a+b = 3
    let e = assert_constraint_error(&mut db, "INSERT INTO t VALUES (2, 1)"); // a+b = 3 → conflict

    // An expression index has no column to name, so SQLite reports the INDEX name.
    assert!(
        e.to_string().contains("UNIQUE constraint failed: index 'u'"),
        "an expression-index UNIQUE violation names the index, not a column; got {e:?}"
    );

    // ABORT backed out only the failing row; the first row stands, nothing was doubled.
    assert_rows_unordered(&mut db, "SELECT a, b FROM t", &[vec![int(1), int(2)]]);
}

#[test]
fn unique_expression_index_admits_distinct_key_values() {
    // The negative control: distinct key values never collide. (1,2)→3 and (3,4)→7 both
    // insert cleanly, so the rejection above is about the KEY, not a blanket refusal.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "CREATE UNIQUE INDEX u ON t(a + b)");
    exec(&mut db, "INSERT INTO t VALUES (1, 2), (3, 4)"); // keys 3, 7 — distinct
    assert_rows_unordered(
        &mut db,
        "SELECT a, b FROM t",
        &[vec![int(1), int(2)], vec![int(3), int(4)]],
    );
}

// ---------------------------------------------------------------------------
// DELETE maintenance: removing a row frees its expression key.
// ---------------------------------------------------------------------------

#[test]
fn unique_expression_index_delete_frees_the_key() {
    // After (1,2) occupies key 3, deleting it must REMOVE the index entry for key 3 — so a
    // later (2,1) [key 3] then inserts cleanly. If DELETE did not maintain the index, the
    // stale key-3 entry would keep rejecting the re-insert.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "CREATE UNIQUE INDEX u ON t(a + b)");
    exec(&mut db, "INSERT INTO t VALUES (1, 2)"); // key 3
    assert_constraint_error(&mut db, "INSERT INTO t VALUES (2, 1)"); // key 3 → conflict

    exec(&mut db, "DELETE FROM t WHERE a = 1"); // frees key 3
    exec(&mut db, "INSERT INTO t VALUES (2, 1)"); // key 3 now free → succeeds

    assert_rows_unordered(&mut db, "SELECT a, b FROM t", &[vec![int(2), int(1)]]);
}

// ---------------------------------------------------------------------------
// UPDATE maintenance: an assignment re-keys the row in the index.
// ---------------------------------------------------------------------------

#[test]
fn unique_expression_index_update_into_collision_fails() {
    // Rows (1,2)→3 and (3,4)→7. Updating the second so its key becomes 3 collides with the
    // first — a UNIQUE error. ABORT rolls the whole UPDATE back, so both original rows
    // remain unchanged.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "CREATE UNIQUE INDEX u ON t(a + b)");
    exec(&mut db, "INSERT INTO t VALUES (1, 2), (3, 4)"); // keys 3, 7

    // Set the (3,4) row to (2,1): new key 3, collides with (1,2)'s key 3.
    assert_constraint_error(&mut db, "UPDATE t SET a = 2, b = 1 WHERE a = 3");

    assert_rows_unordered(
        &mut db,
        "SELECT a, b FROM t",
        &[vec![int(1), int(2)], vec![int(3), int(4)]],
    );
}

#[test]
fn unique_expression_index_update_out_of_collision_succeeds() {
    // The complement: an UPDATE that moves a row to a still-distinct key succeeds, and the
    // vacated key becomes reusable — proving UPDATE removed the OLD entry and inserted the
    // NEW one (not merely added a second). Start (1,2)→3, (3,4)→7.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "CREATE UNIQUE INDEX u ON t(a + b)");
    exec(&mut db, "INSERT INTO t VALUES (1, 2), (3, 4)"); // keys 3, 7

    // Move (1,2) → (1,4): key 3 → 5. Now keys are {5, 7}.
    exec(&mut db, "UPDATE t SET b = 4 WHERE a = 1");
    assert_rows_unordered(
        &mut db,
        "SELECT a, b FROM t",
        &[vec![int(1), int(4)], vec![int(3), int(4)]],
    );

    // Key 3 is now free, so a fresh (0,3) [key 3] inserts cleanly.
    exec(&mut db, "INSERT INTO t VALUES (0, 3)");
    // Key 5 is now taken, so a fresh (2,3) [key 5] collides with (1,4).
    assert_constraint_error(&mut db, "INSERT INTO t VALUES (2, 3)");
    // Key 7 is still taken, so (5,2) [key 7] collides with (3,4).
    assert_constraint_error(&mut db, "INSERT INTO t VALUES (5, 2)");
}

// ---------------------------------------------------------------------------
// CREATE INDEX backfill maintenance: building the index over existing rows.
// ---------------------------------------------------------------------------

#[test]
fn backfill_unique_expression_index_then_insert_is_maintained() {
    // Rows exist FIRST, then the UNIQUE expression index is built (backfill). A later
    // colliding INSERT must be rejected — proving the backfill keyed the pre-existing row
    // by its expression AND the maintained path stays live afterward.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "INSERT INTO t VALUES (1, 2)"); // key 3, BEFORE the index exists
    exec(&mut db, "CREATE UNIQUE INDEX u ON t(a + b)"); // backfills key 3
    assert_constraint_error(&mut db, "INSERT INTO t VALUES (2, 1)"); // key 3 → conflict

    // A distinct-key INSERT still works through the maintained index.
    exec(&mut db, "INSERT INTO t VALUES (4, 5)"); // key 9
    assert_rows_unordered(
        &mut db,
        "SELECT a, b FROM t",
        &[vec![int(1), int(2)], vec![int(4), int(5)]],
    );
}

#[test]
fn backfill_unique_expression_index_detects_existing_duplicate_atomically() {
    // Two pre-existing rows already share key 3 ((1,2) and (2,1)). Building a UNIQUE
    // expression index over them MUST fail (§1.1) and, being atomic, leave NO index — no
    // sqlite_master row — so the name is free to re-create as a NON-unique index.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "INSERT INTO t VALUES (1, 2), (2, 1)"); // both key 3

    let e = assert_constraint_error(&mut db, "CREATE UNIQUE INDEX u ON t(a + b)");
    assert!(
        e.to_string().contains("UNIQUE constraint failed: index 'u'"),
        "a duplicate expression key among existing rows names the index; got {e:?}"
    );
    assert_eq!(index_count(&mut db, "u"), 0, "a failed UNIQUE backfill must leave no index row");

    // The name is free again: a non-unique expression index of the same name builds over
    // the same (duplicate-keyed) data, confirming the failed create was fully rolled back.
    exec(&mut db, "CREATE INDEX u ON t(a + b)");
    assert_eq!(index_count(&mut db, "u"), 1, "the re-created non-unique index is registered");
    assert_rows_unordered(
        &mut db,
        "SELECT a, b FROM t",
        &[vec![int(1), int(2)], vec![int(2), int(1)]],
    );
}

// ---------------------------------------------------------------------------
// Function-expression index: value correctness + case-folding maintenance.
// ---------------------------------------------------------------------------

#[test]
fn function_expression_index_query_is_value_correct() {
    // §1.2: `lower(s)` as an index key. A `WHERE lower(s) = 'app'` returns exactly the rows
    // whose `s` folds to 'app', regardless of whether the planner routes through the index
    // (value correctness — SeqScan recomputes `lower(s)` either way).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(s, n)");
    exec(&mut db, "CREATE INDEX i ON t(lower(s))");
    exec(&mut db, "INSERT INTO t VALUES ('App', 1), ('app', 2), ('APP', 3), ('banana', 4)");
    assert_rows_unordered(
        &mut db,
        "SELECT n FROM t WHERE lower(s) = 'app'",
        &[vec![int(1)], vec![int(2)], vec![int(3)]],
    );
    assert_rows_unordered(&mut db, "SELECT n FROM t WHERE lower(s) = 'banana'", &[vec![int(4)]]);
}

#[test]
fn unique_function_expression_index_folds_case_on_maintenance() {
    // A UNIQUE `lower(s)` index makes case-variant strings collide: after 'Foo' (key 'foo')
    // is stored, 'FOO' (key 'foo') must be rejected — proving the maintained key is the
    // EVALUATED `lower(s)`, not the raw column. A genuinely different string still inserts.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(s, n)");
    exec(&mut db, "CREATE UNIQUE INDEX u ON t(lower(s))");
    exec(&mut db, "INSERT INTO t VALUES ('Foo', 1)"); // key 'foo'
    assert_constraint_error(&mut db, "INSERT INTO t VALUES ('FOO', 2)"); // key 'foo' → conflict
    exec(&mut db, "INSERT INTO t VALUES ('bar', 3)"); // key 'bar' → distinct, ok
    assert_rows_unordered(
        &mut db,
        "SELECT s, n FROM t",
        &[vec![text("Foo"), int(1)], vec![text("bar"), int(3)]],
    );
}

// ---------------------------------------------------------------------------
// Durability: an expression index is RECONSTRUCTED on reopen and stays maintained.
// ---------------------------------------------------------------------------

/// An RAII temp database path (unique per test, cleaned up even on panic, incl. the
/// `-journal`/`-wal`/`-shm` sidecars). Mirrors the helper in `conformance_ondisk_roundtrip.rs`;
/// each test file owns its own so no `.db` file is ever committed.
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
        path.push(format!("minisqlite_exprindex_{pid}_{n}_{nanos}.db"));
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

#[test]
fn unique_expression_index_survives_reopen_and_stays_enforced() {
    // Durability: a committed UNIQUE expression index reconstructs on reopen (its
    // key expression is reparsed from the stored `CREATE INDEX` text), so both its data and
    // its ENFORCEMENT come back. Proving the conflict is still raised after close+reopen is
    // what shows the catalog LOADED the key expression, not just the b-tree bytes.
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(a, b)");
        exec(&mut db, "CREATE UNIQUE INDEX u ON t(a + b)");
        exec(&mut db, "INSERT INTO t VALUES (1, 2), (3, 5)"); // keys 3, 8
        // Enforced before close.
        assert_constraint_error(&mut db, "INSERT INTO t VALUES (2, 1)"); // key 3 → conflict
    } // connection dropped (closed)

    // Reopen the SAME file.
    let mut db = tmp.open();
    // The committed rows survived.
    assert_rows_unordered(
        &mut db,
        "SELECT a, b FROM t",
        &[vec![int(1), int(2)], vec![int(3), int(5)]],
    );
    // A value query over the reconstructed key is correct.
    assert_rows_unordered(&mut db, "SELECT a FROM t WHERE a + b = 8", &[vec![int(3)]]);
    // And the UNIQUE expression index is STILL enforced: (2,1) [key 3] must conflict.
    assert_constraint_error(&mut db, "INSERT INTO t VALUES (2, 1)");
    // A distinct-key INSERT still succeeds through the reconstructed, maintained index.
    exec(&mut db, "INSERT INTO t VALUES (10, 1)"); // key 11
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(3));
}

// ---------------------------------------------------------------------------
// FK ON DELETE action maintenance: a parent delete that CASCADEs / SET-NULLs into a
// child carrying an EXPRESSION index must MAINTAIN that child index (foreignkeys.html
// §4.3/§6 × lang_createindex.html §1.2). Turning expression-index creation on made this
// path reachable — a cascade into an expression-indexed child must not raise the internal
// "key expressions were not compiled for this path" error, it must behave like real SQLite.
// ---------------------------------------------------------------------------

#[test]
fn fk_cascade_delete_maintains_child_expression_index() {
    // ON DELETE CASCADE into a child with a UNIQUE expression index on `a + b`. Deleting the
    // parent must delete the child row AND remove its index entry (key 5), so the freed key
    // is reusable. Real SQLite cascades and drops the entry; the previous engine errored here
    // because the cascade supplied no compiled key expression to the child's index plan.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE c(a, b, pid INTEGER REFERENCES p(id) ON DELETE CASCADE)");
    exec(&mut db, "CREATE UNIQUE INDEX cu ON c(a + b)");
    exec(&mut db, "INSERT INTO p VALUES (1), (2)");
    exec(&mut db, "INSERT INTO c VALUES (2, 3, 1)"); // key a+b = 5, child of p=1

    // The child index is live before the delete: a distinct row with the SAME key 5 conflicts.
    assert_constraint_error(&mut db, "INSERT INTO c VALUES (4, 1, 2)"); // key 5 → conflict

    // Cascade: deleting p=1 removes the child row AND its expression-index entry.
    exec(&mut db, "DELETE FROM p WHERE id = 1");
    assert_scalar(&mut db, "SELECT count(*) FROM c", int(0)); // child cascaded away

    // Key 5 is now FREE — proving the cascade MAINTAINED (removed) the index entry, not just
    // the row. A fresh (4,1) [key 5] child of the surviving p=2 now inserts cleanly.
    exec(&mut db, "INSERT INTO c VALUES (4, 1, 2)");
    assert_rows_unordered(&mut db, "SELECT a, b, pid FROM c", &[vec![int(4), int(1), int(2)]]);
}

#[test]
fn fk_set_null_maintains_child_expression_index() {
    // ON DELETE SET NULL into a child whose UNIQUE expression index keys on `a + pid` (the FK
    // column). Nulling `pid` rewrites the child row and RECOMPUTES its key (a + NULL = NULL,
    // which a UNIQUE index treats as distinct), so the OLD key must be freed. The row SURVIVES
    // with pid NULL. Real SQLite performs the SET NULL rewrite and moves the entry; the
    // previous engine errored because the rewrite supplied no compiled key expression.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE c(a, b, pid INTEGER REFERENCES p(id) ON DELETE SET NULL)");
    exec(&mut db, "CREATE UNIQUE INDEX cu ON c(a + pid)");
    exec(&mut db, "INSERT INTO p VALUES (1), (2)");
    exec(&mut db, "INSERT INTO c VALUES (10, 99, 1)"); // key a+pid = 11, child of p=1

    // Live before the delete: a distinct row with the SAME key 11 conflicts.
    assert_constraint_error(&mut db, "INSERT INTO c VALUES (9, 0, 2)"); // key 9+2 = 11 → conflict

    // SET NULL: the child row survives with pid NULL; its key moves 11 → NULL.
    exec(&mut db, "DELETE FROM p WHERE id = 1");
    assert_rows_unordered(&mut db, "SELECT a, b, pid FROM c", &[vec![int(10), int(99), null()]]);

    // The OLD key 11 is now FREE — proving SET NULL removed (moved) the stale entry. A fresh
    // (9,0) [key 11] child of the surviving p=2 now inserts cleanly.
    exec(&mut db, "INSERT INTO c VALUES (9, 0, 2)");
    assert_rows_unordered(
        &mut db,
        "SELECT a, b, pid FROM c",
        &[vec![int(10), int(99), null()], vec![int(9), int(0), int(2)]],
    );
}

#[test]
fn fk_cascade_delete_maintains_child_function_expression_index() {
    // The function-index variant PROVES the cascade compiles the child's key expression over
    // the executor's builtin function registry: `lower(s)` must RESOLVE to key the child index
    // during the cascade (an arithmetic key needs no registry; a function key does). Deleting
    // the parent removes the child and its lower(s)='foo' entry, freeing the folded key.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE c(s, pid INTEGER REFERENCES p(id) ON DELETE CASCADE)");
    exec(&mut db, "CREATE UNIQUE INDEX cu ON c(lower(s))");
    exec(&mut db, "INSERT INTO p VALUES (1), (2)");
    exec(&mut db, "INSERT INTO c VALUES ('Foo', 1)"); // key 'foo', child of p=1

    // A case-variant string collides on lower(s) — the index is live.
    assert_constraint_error(&mut db, "INSERT INTO c VALUES ('FOO', 2)"); // key 'foo' → conflict

    // Cascade removes the child and evaluates lower(s) to drop its 'foo' entry.
    exec(&mut db, "DELETE FROM p WHERE id = 1");
    assert_scalar(&mut db, "SELECT count(*) FROM c", int(0));

    // 'foo' is free again — the cascade evaluated the function key to maintain the entry.
    exec(&mut db, "INSERT INTO c VALUES ('FOO', 2)"); // key 'foo', child of p=2 → succeeds
    assert_rows_unordered(&mut db, "SELECT s, pid FROM c", &[vec![text("FOO"), int(2)]]);
}
