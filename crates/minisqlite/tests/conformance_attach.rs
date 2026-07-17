//! Conformance battery: **ATTACH / DETACH DATABASE** and the attached-database name
//! resolution it introduces, exercised through the pinned `minisqlite::Connection` facade.
//!
//! Every expectation here is TRANSCRIBED FROM THE SQLITE DOCS in `spec/sqlite-doc/`
//! (cross-checked against real sqlite's documented behavior), never from what the engine
//! currently returns — a failing case is the intended signal that the engine diverges from
//! the spec; an assertion is never weakened to pass.
//!
//! Spec sources (all under `spec/sqlite-doc/`):
//!   * `lang_attach.html`: `ATTACH DATABASE <file> AS <schema>` adds a database file to the
//!     connection; `:memory:` gives an in-memory database and an empty string a new
//!     temporary database; the names `main`/`temp` cannot be attached; a schema-qualified
//!     `schema.table` reaches an attached database, and an unqualified name that is unique
//!     needs no qualifier — when two attached databases share a name, "the table chosen is
//!     the one in the database that was least recently attached" (earliest attach wins).
//!     There is a limit (default 10) on the number of attached databases.
//!   * `lang_detach.html`: `DETACH DATABASE <schema>` removes a database previously
//!     attached; `main`/`temp` cannot be detached.
//!   * `lang_naming.html` / name resolution: an unqualified object resolves `temp`, then
//!     `main`, then attached databases in attach order — so `main` shadows an attached db
//!     for a shared name, and `temp` shadows both.
//!   * `pragma.html` #pragma_database_list: one row per open database — `main` at `seq` 0,
//!     `temp` at `seq` 1 (only once its schema is materialized), attached databases at
//!     `seq` 2.. with their backing file path (empty for `:memory:`).
//!
//! In-transaction ATTACH/DETACH follows modern SQLite (ATTACH/DETACH have been permitted
//! inside a transaction since 3.21.0, and the repo spec imposes no prohibition): ATTACH
//! enrolls the new database in the open transaction, and DETACH of a database participating
//! in the open transaction is rejected as "locked".
//!
//! Each behavior is its own small `#[test]` so one discrepancy fails exactly that case.

mod conformance;
use conformance::*;

use minisqlite::Connection;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Temp-file harness: unique on-disk paths per test, cleaned up even on panic.
// A single guard mints many paths (a test may need a main file plus one or more
// attached files); every minted path (and its journal/WAL sidecars) is removed on
// drop. Kept local because the shared `conformance/mod.rs` is a pinned seam.
// ---------------------------------------------------------------------------

struct TempFiles {
    paths: Vec<PathBuf>,
}

impl TempFiles {
    fn new() -> TempFiles {
        TempFiles { paths: Vec::new() }
    }

    /// Mint a fresh, unique, currently-absent database path and remember it for cleanup.
    fn path(&mut self, tag: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
        let mut path = std::env::temp_dir();
        path.push(format!("minisqlite_attach_{tag}_{pid}_{n}_{nanos}.db"));
        remove_all(&path);
        self.paths.push(path.clone());
        path
    }
}

impl Drop for TempFiles {
    fn drop(&mut self) {
        for p in &self.paths {
            remove_all(p);
        }
    }
}

fn remove_all(path: &Path) {
    let _ = std::fs::remove_file(path);
    for suffix in ["-journal", "-wal", "-shm"] {
        let mut s = path.as_os_str().to_os_string();
        s.push(suffix);
        let _ = std::fs::remove_file(PathBuf::from(s));
    }
}

/// Open a file-backed connection, panicking with context on failure.
fn open_file(path: &Path) -> Connection {
    Connection::open(path).unwrap_or_else(|e| panic!("Connection::open failed for {path:?}: {e:?}"))
}

/// A single-quoted SQL string literal for a path (with `'` doubled, so a path that somehow
/// contained a quote still parses). ATTACH takes the filename as a constant string.
fn sql_lit(path: &Path) -> String {
    format!("'{}'", path.to_string_lossy().replace('\'', "''"))
}

/// The absolute-path string `PRAGMA database_list` reports for a file-backed database — the
/// OS canonicalization of the path, the same reference SQLite uses (falls back to the path
/// as given if canonicalization fails, matching the engine's own display).
fn canonical_display(path: &Path) -> String {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf()).to_string_lossy().into_owned()
}

/// `db.execute(sql)` must be `Err` and its message must contain `needle` (case-insensitive).
/// A bare `is_err()` passes on ANY error — including the wrong one — so pinning a stable
/// substring stops a regression that fails for the wrong reason from passing unnoticed.
#[track_caller]
fn assert_exec_error_contains(db: &mut Connection, sql: &str, needle: &str) {
    match db.execute(sql) {
        Ok(()) => panic!("expected an error containing {needle:?}, but `{sql}` succeeded"),
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.to_ascii_lowercase().contains(&needle.to_ascii_lowercase()),
                "error for `{sql}` was {msg:?}, expected to contain {needle:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// ATTACH opens a store you can read and write.
// ---------------------------------------------------------------------------

#[test]
fn attach_file_db_reads_and_writes() {
    // lang_attach.html: ATTACH adds a database file (created if absent); it can be read from
    // and written to via a schema qualifier.
    let mut tf = TempFiles::new();
    let aux = tf.path("rw");
    let mut db = mem();
    exec(&mut db, &format!("ATTACH DATABASE {} AS aux", sql_lit(&aux)));
    exec(&mut db, "CREATE TABLE aux.t(a, b)");
    exec(&mut db, "INSERT INTO aux.t VALUES (1, 'x'), (2, 'y')");
    assert_rows_unordered(
        &mut db,
        "SELECT a, b FROM aux.t",
        &[vec![int(1), text("x")], vec![int(2), text("y")]],
    );
}

#[test]
fn attach_memory_db() {
    // lang_attach.html §2: the special name ":memory:" attaches an in-memory database.
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.t(a)");
    exec(&mut db, "INSERT INTO aux.t VALUES (42)");
    assert_scalar(&mut db, "SELECT a FROM aux.t", int(42));
}

#[test]
fn attach_empty_string_is_a_transient_database() {
    // lang_attach.html §2: "If the filename is an empty string, then a new temporary
    // database is created." This build backs that transient database in memory (a private
    // on-disk temp file is a documented narrow gap); observably it is a usable, writable,
    // non-file database — its `database_list` file is empty like `:memory:`.
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE '' AS aux");
    exec(&mut db, "CREATE TABLE aux.t(a)");
    exec(&mut db, "INSERT INTO aux.t VALUES (5)");
    assert_scalar(&mut db, "SELECT a FROM aux.t", int(5));
    assert_rows(
        &mut db,
        "PRAGMA database_list",
        &[vec![int(0), text("main"), text("")], vec![int(2), text("aux"), text("")]],
    );
}

#[test]
fn attach_key_clause_is_accepted_and_ignored() {
    // The optional `KEY <expr>` clause selects an encryption key. This build has no
    // encryption, so KEY is documented as accepted-and-ignored: the ATTACH SUCCEEDS (it is
    // not a parse/exec error) and yields an ordinary, usable unencrypted database — rather
    // than erroring or silently doing nothing.
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux KEY 'secret'");
    exec(&mut db, "CREATE TABLE aux.t(a)");
    exec(&mut db, "INSERT INTO aux.t VALUES (9)");
    assert_scalar(&mut db, "SELECT a FROM aux.t", int(9));
}

#[test]
fn attach_alias_accepts_identifier_and_string_forms() {
    // The AS <schema> operand is a schema-name, which the parser leaves as a bare identifier
    // (`AS aux`) or a quoted string (`AS 'aux2'`); both name the attached database.
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "ATTACH DATABASE ':memory:' AS 'aux2'");
    exec(&mut db, "CREATE TABLE aux.a(x)");
    exec(&mut db, "CREATE TABLE aux2.b(y)");
    exec(&mut db, "INSERT INTO aux.a VALUES (1)");
    exec(&mut db, "INSERT INTO aux2.b VALUES (2)");
    assert_scalar(&mut db, "SELECT x FROM aux.a", int(1));
    assert_scalar(&mut db, "SELECT y FROM aux2.b", int(2));
}

// ---------------------------------------------------------------------------
// Name resolution: qualified reaches the store; unqualified follows precedence.
// ---------------------------------------------------------------------------

#[test]
fn qualified_alias_resolution_reaches_distinct_stores() {
    // A `schema.table` qualifier reaches exactly that database. `main.t` and `aux.t` are
    // DIFFERENT tables in different databases; each qualified reference sees only its own.
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE t(a)"); // unqualified create -> main
    exec(&mut db, "INSERT INTO t VALUES (99)");
    exec(&mut db, "CREATE TABLE aux.t(a)");
    exec(&mut db, "INSERT INTO aux.t VALUES (10), (20)");

    assert_scalar(&mut db, "SELECT a FROM main.t", int(99));
    assert_rows_unordered(&mut db, "SELECT a FROM aux.t", &[vec![int(10)], vec![int(20)]]);
}

#[test]
fn unqualified_precedence_temp_then_main_then_attached_in_order() {
    // lang_naming.html + lang_attach.html: an unqualified name resolves temp, then main,
    // then attached databases in attach order (least-recently-attached wins).
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS a1");
    exec(&mut db, "ATTACH DATABASE ':memory:' AS a2");

    // main beats an attached database for a shared name.
    exec(&mut db, "CREATE TABLE p(m)");
    exec(&mut db, "INSERT INTO p VALUES ('main')");
    exec(&mut db, "CREATE TABLE a1.p(m)");
    exec(&mut db, "INSERT INTO a1.p VALUES ('a1')");
    assert_scalar(&mut db, "SELECT m FROM p", text("main"));

    // For a name only in attached databases, the EARLIEST attached (a1) wins over a2.
    exec(&mut db, "CREATE TABLE a1.q(m)");
    exec(&mut db, "INSERT INTO a1.q VALUES ('a1q')");
    exec(&mut db, "CREATE TABLE a2.q(m)");
    exec(&mut db, "INSERT INTO a2.q VALUES ('a2q')");
    assert_scalar(&mut db, "SELECT m FROM q", text("a1q"));

    // temp shadows both main and attached for the shared name `p`.
    exec(&mut db, "CREATE TEMP TABLE p(m)");
    exec(&mut db, "INSERT INTO p VALUES ('temp')");
    assert_scalar(&mut db, "SELECT m FROM p", text("temp"));
    // The shadowed copies are still reachable by qualifier.
    assert_scalar(&mut db, "SELECT m FROM main.p", text("main"));
    assert_scalar(&mut db, "SELECT m FROM a1.p", text("a1"));
}

#[test]
fn cross_db_read_join() {
    // A single query may join a main table with an attached table.
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE t(id, name)");
    exec(&mut db, "INSERT INTO t VALUES (1, 'a'), (2, 'b')");
    exec(&mut db, "CREATE TABLE aux.u(id, tag)");
    exec(&mut db, "INSERT INTO aux.u VALUES (1, 'x'), (2, 'y')");

    assert_rows_unordered(
        &mut db,
        "SELECT t.name, u.tag FROM main.t JOIN aux.u ON t.id = u.id",
        &[vec![text("a"), text("x")], vec![text("b"), text("y")]],
    );
}

#[test]
fn single_attached_autocommit_write_leaves_main_untouched() {
    // A write to `aux.t` targets only the attached store; a same-named `main.t` is unchanged
    // (each namespace's autocommit transaction brackets only its own pager).
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    exec(&mut db, "CREATE TABLE aux.t(a)");
    exec(&mut db, "INSERT INTO aux.t VALUES (2)");

    assert_scalar(&mut db, "SELECT a FROM main.t", int(1));
    assert_scalar(&mut db, "SELECT a FROM aux.t", int(2));
}

// ---------------------------------------------------------------------------
// PRAGMA database_list.
// ---------------------------------------------------------------------------

#[test]
fn database_list_reports_attached_with_seq_name_file() {
    // pragma.html #pragma_database_list: `main` at seq 0 (empty file for an in-memory main)
    // and the attached file database at seq 2 with its absolute path. The reserved-but-unused
    // `temp` slot (seq 1) is NOT listed — it appears only once a temp object materializes it.
    let mut tf = TempFiles::new();
    let aux = tf.path("dblist");
    let mut db = mem();
    exec(&mut db, &format!("ATTACH DATABASE {} AS aux", sql_lit(&aux)));

    assert_rows(
        &mut db,
        "PRAGMA database_list",
        &[
            vec![int(0), text("main"), text("")],
            vec![int(2), text("aux"), text(&canonical_display(&aux))],
        ],
    );
}

#[test]
fn database_list_temp_hidden_until_materialized_even_after_attach() {
    // ATTACH reserves the temp slot (index 1) so attached databases land at 2.., but that
    // reservation must NOT make an unused `temp` appear. `temp` shows only once a user temp
    // object materializes it — then it appears at seq 1, between main and the attached db.
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    // temp still hidden despite the attach-time reservation.
    assert_rows(
        &mut db,
        "PRAGMA database_list",
        &[vec![int(0), text("main"), text("")], vec![int(2), text("aux"), text("")]],
    );

    exec(&mut db, "CREATE TEMP TABLE t(a)");
    assert_rows(
        &mut db,
        "PRAGMA database_list",
        &[
            vec![int(0), text("main"), text("")],
            vec![int(1), text("temp"), text("")],
            vec![int(2), text("aux"), text("")],
        ],
    );
}

// ---------------------------------------------------------------------------
// DETACH.
// ---------------------------------------------------------------------------

#[test]
fn detach_removes_the_database() {
    // After DETACH, the alias no longer resolves — a qualified reference to it errors.
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.t(a)");
    exec(&mut db, "INSERT INTO aux.t VALUES (1)");
    assert_scalar(&mut db, "SELECT a FROM aux.t", int(1));

    exec(&mut db, "DETACH DATABASE aux");
    // The attached namespace is gone: `aux.t` fails specifically because the table (its
    // whole database) no longer resolves — pin the reason so a wrong-error regression
    // (e.g. a panic surfaced as some other error) does not pass unnoticed.
    let e = assert_query_error(&mut db, "SELECT a FROM aux.t");
    assert!(
        e.to_string().to_ascii_lowercase().contains("no such table"),
        "expected 'no such table' after detach, got: {e}"
    );
    // database_list no longer lists it (only main).
    assert_rows(&mut db, "PRAGMA database_list", &[vec![int(0), text("main"), text("")]]);
}

#[test]
fn detach_unknown_alias_errors() {
    // lang_detach.html: detaching a name that is not attached is an error.
    let mut db = mem();
    assert_exec_error_contains(&mut db, "DETACH DATABASE nope", "no such database");
}

#[test]
fn reattach_after_detach_reuses_the_slot() {
    // A detached alias can be attached again (the registry entry was truly removed).
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "DETACH DATABASE aux");
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.t(a)");
    exec(&mut db, "INSERT INTO aux.t VALUES (7)");
    assert_scalar(&mut db, "SELECT a FROM aux.t", int(7));
}

// ---------------------------------------------------------------------------
// Reserved names, duplicates, and the attach limit.
// ---------------------------------------------------------------------------

#[test]
fn cannot_attach_main_or_temp() {
    // lang_attach.html: "The main and temp databases cannot be attached or detached." SQLite
    // reports these (like a duplicate alias) as the name already being in use.
    let mut db = mem();
    assert_exec_error_contains(&mut db, "ATTACH DATABASE ':memory:' AS main", "already in use");
    assert_exec_error_contains(&mut db, "ATTACH DATABASE ':memory:' AS temp", "already in use");
    assert_exec_error_contains(&mut db, "ATTACH DATABASE ':memory:' AS temporary", "already in use");
}

#[test]
fn cannot_detach_main_or_temp() {
    // lang_detach.html: main/temp cannot be detached.
    let mut db = mem();
    assert_exec_error_contains(&mut db, "DETACH DATABASE main", "cannot detach");
    assert_exec_error_contains(&mut db, "DETACH DATABASE temp", "cannot detach");
    assert_exec_error_contains(&mut db, "DETACH DATABASE temporary", "cannot detach");
}

#[test]
fn duplicate_alias_errors() {
    // Attaching a second database under a live alias is rejected — the name is in use.
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    assert_exec_error_contains(&mut db, "ATTACH DATABASE ':memory:' AS aux", "already in use");
    // Case-insensitively, too (SQL identifiers fold over ASCII).
    assert_exec_error_contains(&mut db, "ATTACH DATABASE ':memory:' AS AUX", "already in use");
}

#[test]
fn too_many_attached_databases_errors() {
    // lang_attach.html: there is a limit (default 10) on simultaneously attached databases.
    let mut db = mem();
    for i in 0..10 {
        exec(&mut db, &format!("ATTACH DATABASE ':memory:' AS aux{i}"));
    }
    // The 11th exceeds the limit.
    assert_exec_error_contains(
        &mut db,
        "ATTACH DATABASE ':memory:' AS aux10",
        "too many attached databases",
    );
}

// ---------------------------------------------------------------------------
// ATTACH / DETACH interaction with transactions (modern sqlite: allowed).
// ---------------------------------------------------------------------------

#[test]
fn attach_within_transaction_joins_and_commits() {
    // ATTACH inside a transaction is allowed (sqlite 3.21.0+); the new database joins the
    // open transaction, so a write to it made before COMMIT is durable after it.
    let mut db = mem();
    exec(&mut db, "BEGIN");
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.t(a)");
    exec(&mut db, "INSERT INTO aux.t VALUES (1)");
    exec(&mut db, "COMMIT");
    assert_scalar(&mut db, "SELECT a FROM aux.t", int(1));
}

#[test]
fn attached_store_participates_in_rollback() {
    // An attached database enrolled in the connection's transaction is rolled back with it:
    // a write to `aux.t` inside a rolled-back transaction leaves no trace.
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.t(a)");
    exec(&mut db, "INSERT INTO aux.t VALUES (1)"); // committed (autocommit)

    exec(&mut db, "BEGIN");
    exec(&mut db, "INSERT INTO aux.t VALUES (2)");
    assert_rows_unordered(&mut db, "SELECT a FROM aux.t", &[vec![int(1)], vec![int(2)]]);
    exec(&mut db, "ROLLBACK");

    // Only the pre-transaction row survives — the attached pager rolled back with main.
    assert_rows(&mut db, "SELECT a FROM aux.t", &[vec![int(1)]]);
}

#[test]
fn detach_within_transaction_is_locked() {
    // A database participating in the open transaction cannot be detached ("locked"); once
    // the transaction ends the detach succeeds.
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "BEGIN");
    assert_exec_error_contains(&mut db, "DETACH DATABASE aux", "locked");
    exec(&mut db, "ROLLBACK");
    exec(&mut db, "DETACH DATABASE aux");
}

// ---------------------------------------------------------------------------
// Multi-file transactions (the DEFER boundary: sequential per-file commit persists
// both files on a normal commit; crash-atomicity across files is a documented,
// SILENT-at-runtime gap on `exec_commit`, out of scope here — no loud error by design).
// ---------------------------------------------------------------------------

#[test]
fn multi_file_transaction_commits_and_persists() {
    // A single BEGIN..COMMIT that writes BOTH a durable main file and a durable attached
    // file commits both: after reopening each database standalone, both writes are present.
    //
    // This pins PERSISTENCE on a clean commit ONLY. It does NOT assert crash-atomicity
    // across the two files — that needs a super-journal (master journal) and is a
    // documented SILENT-at-runtime gap on `exec_commit`, out of scope here. NO loud error
    // is expected by design: sequential per-file commit matches real sqlite on a normal
    // commit, so the COMMIT below succeeds silently.
    let mut tf = TempFiles::new();
    let main_path = tf.path("multimain");
    let aux_path = tf.path("multiaux");
    {
        let mut db = open_file(&main_path);
        exec(&mut db, &format!("ATTACH DATABASE {} AS aux", sql_lit(&aux_path)));
        exec(&mut db, "CREATE TABLE m(a)");
        exec(&mut db, "CREATE TABLE aux.a(b)");
        exec(&mut db, "BEGIN");
        exec(&mut db, "INSERT INTO m VALUES (1)");
        exec(&mut db, "INSERT INTO aux.a VALUES (2)");
        exec(&mut db, "COMMIT");
    }
    // Reopen each file on its own; both committed rows survived.
    {
        let mut main_db = open_file(&main_path);
        assert_scalar(&mut main_db, "SELECT a FROM m", int(1));
    }
    {
        let mut aux_db = open_file(&aux_path);
        assert_scalar(&mut aux_db, "SELECT b FROM a", int(2));
    }
}

// ---------------------------------------------------------------------------
// On-disk round trips through the file format (both directions of this engine).
// ---------------------------------------------------------------------------

#[test]
fn attach_reads_back_a_file_this_engine_wrote() {
    // A database written by a standalone connection is readable when ATTACHed to another
    // connection — this engine's own write side round-trips through the ATTACH read path.
    let mut tf = TempFiles::new();
    let path = tf.path("roundtrip");
    {
        let mut writer = open_file(&path);
        exec(&mut writer, "CREATE TABLE t(a, b)");
        exec(&mut writer, "INSERT INTO t VALUES (1, 'one'), (2, 'two')");
    }
    // Fresh connection, attach the file just written, read it back through the alias.
    let mut db = mem();
    exec(&mut db, &format!("ATTACH DATABASE {} AS aux", sql_lit(&path)));
    assert_rows_unordered(
        &mut db,
        "SELECT a, b FROM aux.t",
        &[vec![int(1), text("one")], vec![int(2), text("two")]],
    );
}

#[test]
fn write_attached_file_then_reopen_standalone_persists() {
    // Writing to an attached file database persists it: after closing the connection, the
    // file opened standalone (as its own `main`) contains the rows written through the alias.
    let mut tf = TempFiles::new();
    let path = tf.path("persist");
    {
        let mut db = mem();
        exec(&mut db, &format!("ATTACH DATABASE {} AS aux", sql_lit(&path)));
        exec(&mut db, "CREATE TABLE aux.t(a)");
        exec(&mut db, "INSERT INTO aux.t VALUES (11), (22)");
    }
    // Reopen the attached file as a standalone database; the data is there under `main`.
    let mut standalone = open_file(&path);
    assert_rows_unordered(&mut standalone, "SELECT a FROM t", &[vec![int(11)], vec![int(22)]]);
}
