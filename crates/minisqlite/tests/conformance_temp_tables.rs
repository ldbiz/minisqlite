//! Conformance battery: the **TEMP (temporary) schema** and the multi-database
//! namespace it introduces, exercised through the pinned `minisqlite::Connection`
//! facade.
//!
//! Every expectation here is TRANSCRIBED FROM THE SQLITE DOCS in `spec/sqlite-doc/`,
//! never from what the engine currently returns — a failing case is the intended
//! signal that the engine diverges from the spec; an assertion is never weakened to
//! pass.
//!
//! Spec sources (all under `spec/sqlite-doc/`):
//!   * `lang_createtable.html` (TEMP / TEMPORARY): "the table is only visible within
//!     that same database connection and is automatically deleted when the database
//!     connection is closed" and lives in a separate `temp` database — so a temp
//!     object is absent from `main`'s schema, is not written to the main file, and a
//!     `CREATE TEMP TABLE t` may coexist with a `main.t` (each in its own schema).
//!   * `schematab.html` §2 "Alternative Names": each database has its own schema
//!     table; the temp schema's is reachable as `temp.sqlite_master` /
//!     `sqlite_temp_master`, while an unqualified `sqlite_master` names the schema
//!     table of `main`. So a temp object appears in `temp.sqlite_master`, NOT in an
//!     unqualified `sqlite_master`.
//!   * `lang_naming.html` / name resolution: an unqualified object name is resolved
//!     `temp` first, then `main`, then attached databases — so a temp table SHADOWS a
//!     same-named main table for an unqualified reference, while `main.t` / `temp.t`
//!     reach each explicit schema.
//!   * `lang_createindex.html`: there is no `CREATE TEMP INDEX` — an index inherits
//!     the schema of the table it indexes, so an index on a temp table is a temp
//!     object automatically.
//!   * `pragma.html` #pragma_database_list: one row per open database — `main` at
//!     `seq` 0 and, once a temp object has created the temp database, `temp` at
//!     `seq` 1 with an empty `file`.
//!   * `lang_transaction.html`: a ROLLBACK reverts every change of the aborted
//!     transaction, INCLUDING temp-schema changes — so a temp table created and filled
//!     inside a rolled-back transaction leaves no trace.
//!
//! Each behavior is its own small `#[test]` so one discrepancy fails exactly that
//! case rather than masking the rest.

mod conformance;
use conformance::*;

use minisqlite::{Connection, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Temp-file harness: a unique on-disk path per test, cleaned up even on panic.
// Mirrors the RAII guard in `conformance_ondisk_roundtrip.rs`; kept local (not
// in the shared harness) because only the not-persisted test here needs a
// file-backed connection, and the shared `conformance/mod.rs` is a pinned seam.
// ---------------------------------------------------------------------------

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
        path.push(format!("minisqlite_temp_tables_{pid}_{n}_{nanos}.db"));
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

// ---------------------------------------------------------------------------
// The common data path must keep working (regressing it is forbidden).
// ---------------------------------------------------------------------------

#[test]
fn temp_table_basic_data_roundtrip() {
    // `CREATE TEMP TABLE t; INSERT; SELECT` must return the inserted data — the
    // common temp path that already worked before real temp semantics existed and
    // must never regress.
    let mut db = mem();
    exec(&mut db, "CREATE TEMP TABLE t(a, b)");
    exec(&mut db, "INSERT INTO t VALUES (1, 'x'), (2, 'y')");
    assert_rows(
        &mut db,
        "SELECT a, b FROM t ORDER BY a",
        &[vec![int(1), text("x")], vec![int(2), text("y")]],
    );
    // TEMPORARY is a synonym for TEMP (lang_createtable.html).
    exec(&mut db, "CREATE TEMPORARY TABLE u(v)");
    exec(&mut db, "INSERT INTO u VALUES (42)");
    assert_scalar(&mut db, "SELECT v FROM u", int(42));
}

#[test]
fn create_table_with_temp_qualifier_targets_temp_schema() {
    // A `temp.`-qualified CREATE (WITHOUT the TEMP keyword) also creates in the temp
    // schema — a distinct routing branch from the TEMP-keyword path, exercising
    // schema-qualified DDL. The object is a temp object: present in temp's schema
    // table, absent from main's, resolvable unqualified, and reported by database_list.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE temp.x(a)");
    exec(&mut db, "INSERT INTO temp.x VALUES (5)");

    assert_rows_unordered(
        &mut db,
        "SELECT name FROM temp.sqlite_master WHERE type = 'table'",
        &[vec![text("x")]],
    );
    assert_rows_unordered(
        &mut db,
        "SELECT name FROM sqlite_master WHERE type = 'table'",
        &[],
    );
    assert_scalar(&mut db, "SELECT a FROM x", int(5));
    assert_rows(
        &mut db,
        "PRAGMA database_list",
        &[
            vec![int(0), text("main"), text("")],
            vec![int(1), text("temp"), text("")],
        ],
    );
}

// ---------------------------------------------------------------------------
// Invisibility in main's schema table; visibility in temp's.
// ---------------------------------------------------------------------------

#[test]
fn temp_table_absent_from_main_schema_table() {
    // schematab.html §2: an unqualified `sqlite_master` (and the qualified
    // `main.sqlite_master`) names MAIN's schema table, so a temp table must NOT
    // appear there — only the main table does.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE keep(a)");
    exec(&mut db, "CREATE TEMP TABLE t(x)");

    assert_rows_unordered(
        &mut db,
        "SELECT name FROM sqlite_master WHERE type = 'table'",
        &[vec![text("keep")]],
    );
    assert_rows_unordered(
        &mut db,
        "SELECT name FROM main.sqlite_master WHERE type = 'table'",
        &[vec![text("keep")]],
    );
    // The modern alias `sqlite_schema` behaves the same as `sqlite_master`.
    assert_rows_unordered(
        &mut db,
        "SELECT name FROM sqlite_schema WHERE type = 'table'",
        &[vec![text("keep")]],
    );
}

#[test]
fn temp_object_present_in_temp_schema_table() {
    // schematab.html §2: the temp schema's table is reachable as
    // `temp.sqlite_master`, where the temp object IS listed (and the main object is
    // NOT — each schema table is per-database).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE keep(a)");
    exec(&mut db, "CREATE TEMP TABLE t(x)");

    assert_rows_unordered(
        &mut db,
        "SELECT name FROM temp.sqlite_master WHERE type = 'table'",
        &[vec![text("t")]],
    );
}

// ---------------------------------------------------------------------------
// Shadowing: a temp table shadows a same-named main table for unqualified refs.
// ---------------------------------------------------------------------------

#[test]
fn temp_table_shadows_same_named_main_table() {
    // lang_createtable.html + name resolution: `CREATE TABLE t` then
    // `CREATE TEMP TABLE t` must SUCCEED (distinct schemas, not "table t already
    // exists"), and an unqualified `t` resolves to the temp one (searched first),
    // while `main.t` / `temp.t` reach each explicit schema. Distinct column names
    // (x in main, y in temp) prove WHICH store answered.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x)");
    exec(&mut db, "INSERT INTO t VALUES (1)"); // into main (no temp yet)
    exec(&mut db, "CREATE TEMP TABLE t(y)"); // shadows main.t — must not error
    exec(&mut db, "INSERT INTO t VALUES (100)"); // unqualified -> temp

    // Unqualified `t` is the temp table: column y = 100.
    assert_rows(&mut db, "SELECT y FROM t", &[vec![int(100)]]);
    // `main.t` is the main table: column x = 1, unchanged by the temp insert.
    assert_rows(&mut db, "SELECT x FROM main.t", &[vec![int(1)]]);
    // `temp.t` is the temp table.
    assert_rows(&mut db, "SELECT y FROM temp.t", &[vec![int(100)]]);
    // The temp table has no column x, proving the unqualified name is NOT main's.
    assert_query_error(&mut db, "SELECT x FROM t");
}

// ---------------------------------------------------------------------------
// CREATE TEMP VIEW / TEMP TRIGGER, and an index on a temp table.
// ---------------------------------------------------------------------------

#[test]
fn create_temp_view_resolves_and_stays_in_temp() {
    // A TEMP VIEW lives in the temp schema, is queryable, and is absent from main's
    // schema table (schematab.html §2).
    let mut db = mem();
    exec(&mut db, "CREATE TEMP TABLE t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1), (2), (3)");
    exec(&mut db, "CREATE TEMP VIEW v AS SELECT a FROM t WHERE a > 1");

    assert_rows(&mut db, "SELECT a FROM v ORDER BY a", &[vec![int(2)], vec![int(3)]]);
    assert_rows_unordered(
        &mut db,
        "SELECT name FROM sqlite_master WHERE type = 'view'",
        &[],
    );
    assert_rows_unordered(
        &mut db,
        "SELECT name FROM temp.sqlite_master WHERE type = 'view'",
        &[vec![text("v")]],
    );
}

#[test]
fn index_on_temp_table_lives_in_temp_schema() {
    // lang_createindex.html: there is no `CREATE TEMP INDEX` keyword, so that syntax
    // is rejected; an index on a temp table is a temp object automatically and is
    // therefore absent from main's schema table but present in temp's.
    let mut db = mem();
    exec(&mut db, "CREATE TEMP TABLE t(a, b)");
    exec(&mut db, "INSERT INTO t VALUES (1, 'x'), (2, 'y')");
    // There is no `CREATE TEMP INDEX` keyword (lang_createindex.html): the parser rejects it.
    // Pin the reason so the test can't pass on some unrelated failure (a panic surfaced as an
    // error, a later semantic reject) — it must be THIS syntactic rejection.
    let e = assert_exec_error(&mut db, "CREATE TEMP INDEX bad ON t(a)");
    assert!(
        e.to_string().contains("TEMP is not valid with CREATE INDEX"),
        "expected the CREATE-TEMP-INDEX rejection reason, got: {e}"
    );

    exec(&mut db, "CREATE INDEX it ON t(a)");
    // The index is usable (equality lookup returns the row).
    assert_rows(&mut db, "SELECT b FROM t WHERE a = 2", &[vec![text("y")]]);
    // It is a temp object: absent from main, present in temp.
    assert_rows_unordered(
        &mut db,
        "SELECT name FROM sqlite_master WHERE type = 'index'",
        &[],
    );
    assert_rows_unordered(
        &mut db,
        "SELECT name FROM temp.sqlite_master WHERE type = 'index' AND name = 'it'",
        &[vec![text("it")]],
    );
}

#[test]
fn temp_trigger_on_temp_table_fires() {
    // A TEMP TRIGGER on a temp table (both in the temp schema) fires its action,
    // which writes another temp table — the same-namespace case. The trigger is a
    // temp object (absent from main's schema table).
    let mut db = mem();
    exec(&mut db, "CREATE TEMP TABLE t(a)");
    exec(&mut db, "CREATE TEMP TABLE log(msg)");
    exec(
        &mut db,
        "CREATE TEMP TRIGGER trg AFTER INSERT ON t BEGIN \
         INSERT INTO log VALUES ('fired'); END",
    );
    exec(&mut db, "INSERT INTO t VALUES (1), (2)");

    // Two inserts -> the AFTER INSERT trigger fired twice.
    assert_scalar(&mut db, "SELECT count(*) FROM log", int(2));
    assert_rows_unordered(
        &mut db,
        "SELECT name FROM sqlite_master WHERE type = 'trigger'",
        &[],
    );
}

// ---------------------------------------------------------------------------
// PRAGMA database_list reports main, and temp once it exists.
// ---------------------------------------------------------------------------

#[test]
fn pragma_database_list_reports_main_then_temp() {
    // pragma.html #pragma_database_list: before any temp object, only `main` (seq 0);
    // once a temp object creates the temp database, a `temp` row (seq 1, empty file)
    // follows it.
    let mut db = mem();
    assert_rows(
        &mut db,
        "PRAGMA database_list",
        &[vec![int(0), text("main"), text("")]],
    );

    exec(&mut db, "CREATE TEMP TABLE t(a)");
    assert_rows(
        &mut db,
        "PRAGMA database_list",
        &[
            vec![int(0), text("main"), text("")],
            vec![int(1), text("temp"), text("")],
        ],
    );
}

// ---------------------------------------------------------------------------
// Cross-namespace read and write.
// ---------------------------------------------------------------------------

#[test]
fn cross_namespace_read_temp_join_main() {
    // A join across the two schemas reads BOTH pagers: the executor opens each base
    // cursor on the store the binder stamped, so a `temp JOIN main` is correct.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE m(a)");
    exec(&mut db, "INSERT INTO m VALUES (1), (2)");
    exec(&mut db, "CREATE TEMP TABLE tt(b)");
    exec(&mut db, "INSERT INTO tt VALUES (10), (20)");

    assert_rows(
        &mut db,
        "SELECT m.a, tt.b FROM m, tt ORDER BY m.a, tt.b",
        &[
            vec![int(1), int(10)],
            vec![int(1), int(20)],
            vec![int(2), int(10)],
            vec![int(2), int(20)],
        ],
    );
}

#[test]
fn cross_namespace_insert_main_from_temp() {
    // `INSERT INTO main_table SELECT FROM temp_table`: the two-phase DML reads the
    // TEMP source (phase 1) and writes the MAIN target (phase 2). The temp source is
    // left unchanged.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE m(a)");
    exec(&mut db, "INSERT INTO m VALUES (1)");
    exec(&mut db, "CREATE TEMP TABLE tt(b)");
    exec(&mut db, "INSERT INTO tt VALUES (10), (20)");

    exec(&mut db, "INSERT INTO m SELECT b FROM tt");
    assert_rows(&mut db, "SELECT a FROM m ORDER BY a", &[vec![int(1)], vec![int(10)], vec![int(20)]]);
    // The temp source is untouched by the write to main.
    assert_rows(&mut db, "SELECT b FROM tt ORDER BY b", &[vec![int(10)], vec![int(20)]]);
}

#[test]
fn cross_namespace_insert_temp_from_main() {
    // The mirror: `INSERT INTO temp_table SELECT FROM main_table` reads MAIN and
    // writes TEMP, leaving main unchanged.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE m(a)");
    exec(&mut db, "INSERT INTO m VALUES (1), (2)");
    exec(&mut db, "CREATE TEMP TABLE tt(b)");

    exec(&mut db, "INSERT INTO tt SELECT a FROM m");
    assert_rows(&mut db, "SELECT b FROM tt ORDER BY b", &[vec![int(1)], vec![int(2)]]);
    assert_rows(&mut db, "SELECT a FROM m ORDER BY a", &[vec![int(1)], vec![int(2)]]);
}

// ---------------------------------------------------------------------------
// Not persisted: a temp object is never written to the main file.
// ---------------------------------------------------------------------------

#[test]
fn temp_table_not_persisted_across_reopen() {
    // lang_createtable.html: a temp table is deleted when the connection closes and
    // is never written to the main database file. So after close+reopen of a
    // file-backed database, the main table survives but the temp table is gone.
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE keep(a)");
        exec(&mut db, "INSERT INTO keep VALUES (7)");
        exec(&mut db, "CREATE TEMP TABLE t(x)");
        exec(&mut db, "INSERT INTO t VALUES (1)");
        // Both are visible within the live connection.
        assert_scalar(&mut db, "SELECT a FROM keep", int(7));
        assert_scalar(&mut db, "SELECT x FROM t", int(1));
    } // connection dropped (closed)

    let mut db = tmp.open();
    // The main table and its data survived.
    assert_scalar(&mut db, "SELECT a FROM keep", int(7));
    // The temp table did NOT persist: it is not in the reopened schema, and no temp
    // database exists yet in the fresh connection.
    assert_query_error(&mut db, "SELECT x FROM t");
    assert_rows_unordered(
        &mut db,
        "SELECT name FROM sqlite_master WHERE type = 'table'",
        &[vec![text("keep")]],
    );
    assert_rows(&mut db, "PRAGMA database_list", &[vec![int(0), text("main"), text_path(&tmp.path)]]);
}

/// The absolute path string `PRAGMA database_list` reports for a file-backed main
/// database (canonicalized when the file exists, as the pragma does).
fn text_path(path: &Path) -> Value {
    let s = std::fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .into_owned();
    text(&s)
}

// ---------------------------------------------------------------------------
// Transactions span the temp schema: ROLLBACK undoes temp writes.
// ---------------------------------------------------------------------------

#[test]
fn rollback_undoes_temp_table_creation_and_writes() {
    // lang_transaction.html: a ROLLBACK reverts every change of the aborted
    // transaction, including temp-schema changes. A temp table created and filled
    // inside a rolled-back transaction leaves no trace.
    let mut db = mem();
    exec(&mut db, "BEGIN");
    exec(&mut db, "CREATE TEMP TABLE t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1), (2)");
    // Visible mid-transaction.
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2));
    exec(&mut db, "ROLLBACK");

    // No trace: the temp table is gone after rollback.
    assert_query_error(&mut db, "SELECT * FROM t");
    // And the name may be reused by a fresh CREATE (the old definition is truly gone).
    exec(&mut db, "CREATE TEMP TABLE t(z)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
}

#[test]
fn rollback_undoes_temp_writes_to_existing_temp_table() {
    // A temp table created (and committed) BEFORE a transaction keeps its
    // pre-transaction contents when a later transaction that wrote to it is rolled
    // back — the temp pager participates in the connection-wide rollback.
    let mut db = mem();
    exec(&mut db, "CREATE TEMP TABLE t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1)");

    exec(&mut db, "BEGIN");
    exec(&mut db, "INSERT INTO t VALUES (2), (3)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(3));
    exec(&mut db, "ROLLBACK");

    // Only the pre-transaction row remains.
    assert_rows(&mut db, "SELECT a FROM t", &[vec![int(1)]]);
}

#[test]
fn commit_keeps_temp_writes_across_namespaces() {
    // The COMMIT counterpart: a transaction that writes BOTH a main and a temp table
    // commits both atomically (the temp pager commits alongside main).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE m(a)");
    exec(&mut db, "CREATE TEMP TABLE t(b)");

    exec(&mut db, "BEGIN");
    exec(&mut db, "INSERT INTO m VALUES (1)");
    exec(&mut db, "INSERT INTO t VALUES (2)");
    exec(&mut db, "COMMIT");

    assert_scalar(&mut db, "SELECT a FROM m", int(1));
    assert_scalar(&mut db, "SELECT b FROM t", int(2));
}

// ---------------------------------------------------------------------------
// A `temp.`-qualified write DDL issued before the temp schema is materialized is a
// clean SQL error (never a crash), and — this is the case not covered elsewhere —
// that error INSIDE an explicit transaction leaves the transaction open and usable.
// The full autocommit matrix (every DROP/ALTER/CREATE INDEX kind, IF EXISTS no-ops,
// and that temp stays unmaterialized) lives in `conformance_qualified_ddl_temp.rs`;
// this pins only the transaction-interaction slice that file does not.
// ---------------------------------------------------------------------------

#[test]
fn temp_qualified_ddl_error_inside_transaction_leaves_txn_usable() {
    // `DROP TABLE temp.t` before temp exists reports "no such table" (a plain SQL
    // error, not a crash and not a transaction abort). Issued inside a `BEGIN`, the
    // transaction stays open and coherent: a subsequent write + COMMIT still succeeds.
    let mut db = mem();
    exec(&mut db, "BEGIN");
    let e = assert_exec_error(&mut db, "DROP TABLE temp.t");
    assert!(
        e.to_string().contains("no such table"),
        "expected 'no such table', got: {e}"
    );
    exec(&mut db, "CREATE TABLE m(a)");
    exec(&mut db, "INSERT INTO m VALUES (1)");
    exec(&mut db, "COMMIT");
    assert_scalar(&mut db, "SELECT a FROM m", int(1));
}
