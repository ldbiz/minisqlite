//! End-to-end tests for the engine integrator, driven through the public
//! `minisqlite_engine::open_in_memory()` + [`Engine`] surface (the same route the
//! facade `minisqlite::Connection` uses). These exercise the WIRING — parse → plan
//! → execute over the real catalog + pager — not the individual seams, which have
//! their own tests. They pin what the engine must do with the planner as it stands
//! today (SELECT, no-FROM, VALUES) plus engine-level DDL / transactions / PRAGMA.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use minisqlite_engine::{open, open_in_memory, Engine};
use minisqlite_types::{Error, Value};

/// A fresh in-memory engine, or panic — every test starts from one.
fn mem() -> Box<dyn Engine> {
    open_in_memory().expect("open_in_memory should succeed")
}

/// Assert a result is an `Err` whose message mentions `needle` (case-insensitive).
/// A bare `.is_err()` passes on ANY error — including a spurious or wrong-reason one
/// — so pinning a stable substring of the intended message stops a regression that
/// fails for the WRONG reason from being masked.
#[track_caller]
fn assert_err_contains<T>(result: Result<T, Error>, needle: &str) {
    match result {
        Ok(_) => panic!("expected an error mentioning {needle:?}, got Ok"),
        Err(e) => {
            let msg = e.to_string().to_ascii_lowercase();
            assert!(
                msg.contains(&needle.to_ascii_lowercase()),
                "error {e:?} does not mention {needle:?}"
            );
        }
    }
}

/// A unique database path under the OS temp dir (unique per call so parallel test
/// threads never collide).
fn unique_db_path(tag: &str) -> PathBuf {
    static N: AtomicU32 = AtomicU32::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
    let mut p = std::env::temp_dir();
    p.push(format!("mse-engine-{tag}-{pid}-{n}-{nanos}.db"));
    p
}

/// Remove a test database and its rollback journal sibling.
fn cleanup(path: &Path) {
    let _ = std::fs::remove_file(path);
    let mut journal = path.as_os_str().to_os_string();
    journal.push("-journal");
    let _ = std::fs::remove_file(PathBuf::from(journal));
}

// ----- SELECT through the planner + executor --------------------------------

#[test]
fn select_constant_returns_one_row() {
    let mut db = mem();
    let r = db.query("SELECT 1").unwrap();
    assert_eq!(r.rows.len(), 1);
    assert_eq!(r.rows[0].len(), 1);
    assert!(matches!(r.rows[0][0], Value::Integer(1)));
}

#[test]
fn select_expression_and_text_literal() {
    let mut db = mem();
    let r = db.query("SELECT 1+2, 'hi'").unwrap();
    assert_eq!(r.rows.len(), 1);
    assert!(matches!(r.rows[0][0], Value::Integer(3)));
    assert!(matches!(&r.rows[0][1], Value::Text(s) if s == "hi"));
}

// ----- CREATE TABLE (engine-level DDL via the catalog) ----------------------

#[test]
fn create_table_duplicate_errors_but_if_not_exists_is_a_noop() {
    let mut db = mem();
    db.execute("CREATE TABLE t(a INTEGER, b TEXT)").unwrap();

    // A second identical CREATE is an error (the table already exists). The failed
    // implicit transaction must leave no partial write — the table stays usable.
    assert!(db.execute("CREATE TABLE t(a INTEGER, b TEXT)").is_err());

    // IF NOT EXISTS on an existing table is a no-op success.
    db.execute("CREATE TABLE IF NOT EXISTS t(a INTEGER, b TEXT)").unwrap();

    // The empty table selects zero rows, and the result carries the column names.
    let r = db.query("SELECT a, b FROM t").unwrap();
    assert_eq!(r.columns, vec!["a".to_string(), "b".to_string()]);
    assert_eq!(r.rows.len(), 0);
}

// ----- explicit transactions ------------------------------------------------

#[test]
fn begin_create_commit_persists_the_table() {
    let mut db = mem();
    db.execute("BEGIN").unwrap();
    db.execute("CREATE TABLE u(x)").unwrap();
    db.execute("COMMIT").unwrap();
    let r = db.query("SELECT x FROM u").unwrap();
    assert_eq!(r.columns, vec!["x".to_string()]);
    assert_eq!(r.rows.len(), 0);
}

#[test]
fn begin_create_rollback_leaves_the_table_absent() {
    let mut db = mem();
    db.execute("BEGIN").unwrap();
    db.execute("CREATE TABLE u(x)").unwrap();
    db.execute("ROLLBACK").unwrap();

    // The rolled-back table is gone from storage AND from the schema cache, so a
    // select from it errors with "no such table". If the cache were not resynced
    // after rollback, the planner would still "see" u and this would not error.
    assert!(db.query("SELECT x FROM u").is_err());

    // And the name is free again: re-creating it must succeed (proving the cache
    // truly forgot the rolled-back table, not just storage).
    db.execute("CREATE TABLE u(x)").unwrap();
}

#[test]
fn commit_or_rollback_without_a_transaction_errors() {
    let mut db = mem();
    // Pin the REASON (no active transaction), not merely that it errored.
    assert_err_contains(db.execute("COMMIT"), "no transaction is active");
    assert_err_contains(db.execute("ROLLBACK"), "no transaction is active");
}

#[test]
fn nested_begin_errors() {
    let mut db = mem();
    db.execute("BEGIN").unwrap();
    assert_err_contains(db.execute("BEGIN"), "within a transaction");
    db.execute("ROLLBACK").unwrap();
}

#[test]
fn savepoint_release_rollback_to_are_wired() {
    // SAVEPOINT / RELEASE / ROLLBACK TO are implemented end-to-end (the pager
    // savepoint seam + the engine name->depth stack), so they SUCCEED — this is the
    // positive counterpart of the old "reported gap" guard. A bare SAVEPOINT in
    // autocommit starts a transaction that persists until its RELEASE.
    let mut db = mem();
    db.execute("CREATE TABLE t(a)").unwrap();

    db.execute("SAVEPOINT sp1").unwrap(); // starts the transaction
    db.execute("INSERT INTO t VALUES (1)").unwrap();
    db.execute("SAVEPOINT sp2").unwrap();
    db.execute("INSERT INTO t VALUES (2)").unwrap();
    // ROLLBACK TO undoes only the work after sp2, keeping the savepoint and the txn.
    db.execute("ROLLBACK TO sp2").unwrap();
    let r = db.query("SELECT count(*) FROM t").unwrap();
    assert!(matches!(r.rows[0][0], Value::Integer(1)), "ROLLBACK TO sp2 undoes only the 2nd insert");

    // RELEASE the outermost savepoint (which started the transaction) commits it.
    db.execute("RELEASE sp1").unwrap();

    // GUARD (outermost RELEASE == COMMIT, and it truly ENDS the pager transaction):
    // a following autocommit INSERT must open a FRESH implicit transaction. If the
    // savepoint-started RELEASE had only cleared the stack WITHOUT committing, the
    // pager transaction would still be open and this INSERT's implicit `begin()`
    // would error ("a transaction is already active"). So this line failing is
    // the signal that RELEASE stopped committing.
    db.execute("INSERT INTO t VALUES (3)").unwrap();
    let r = db.query("SELECT count(*) FROM t").unwrap();
    assert!(matches!(r.rows[0][0], Value::Integer(2)), "kept row 1 + autocommitted row 3 == 2");
}

#[test]
fn rollback_to_savepoint_reverts_ddl_and_resyncs_the_catalog() {
    // A CREATE TABLE made inside a savepoint is reverted by ROLLBACK TO: afterwards
    // the table is gone from BOTH storage AND the schema cache (the catalog is
    // reloaded from the reverted page 1, exactly as the full-ROLLBACK path does), so a
    // SELECT from it errors. Without that resync the planner would still "see" the
    // table and this would wrongly succeed.
    let mut db = mem();
    db.execute("SAVEPOINT sp1").unwrap();
    db.execute("CREATE TABLE tmp(a)").unwrap();
    db.execute("INSERT INTO tmp VALUES (1)").unwrap();
    assert!(db.query("SELECT a FROM tmp").is_ok(), "the table exists after CREATE inside the savepoint");

    db.execute("ROLLBACK TO sp1").unwrap();
    assert!(
        db.query("SELECT a FROM tmp").is_err(),
        "ROLLBACK TO must drop the created table from the schema cache, not just storage"
    );
    // The savepoint stays open after ROLLBACK TO; RELEASE ends the (now empty) txn.
    db.execute("RELEASE sp1").unwrap();
}

// ----- DROP (wired through the catalog's drop_object seam) -------------------

#[test]
fn drop_table_removes_it() {
    let mut db = mem();
    db.execute("CREATE TABLE t(a)").unwrap();
    db.query("SELECT a FROM t").unwrap(); // present before the drop
    db.execute("DROP TABLE t").unwrap();
    assert!(db.query("SELECT a FROM t").is_err(), "a dropped table is gone");
    // DROP IF EXISTS on a missing table is a no-op success.
    db.execute("DROP TABLE IF EXISTS t").unwrap();
}

// ----- CREATE INDEX (engine-level DDL via the catalog) ----------------------

#[test]
fn create_index_is_wired_to_the_catalog() {
    let mut db = mem();
    // An index on a MISSING table must error — proof the arm truly calls
    // catalog.create_index (a faked `Ok(None)` would skip that validation and
    // silently succeed). This guards the dispatch arm against being neutered.
    assert!(
        db.execute("CREATE INDEX idx ON missing(a)").is_err(),
        "CREATE INDEX on a non-existent table must error"
    );
    db.execute("CREATE TABLE t(a INTEGER, b TEXT)").unwrap();
    db.execute("CREATE INDEX idx_t_a ON t(a)").unwrap();
    // The table still reads correctly with the index registered.
    let r = db.query("SELECT a, b FROM t").unwrap();
    assert_eq!(r.columns, vec!["a".to_string(), "b".to_string()]);
}

// ----- ALTER TABLE (engine-level DDL via the catalog) -----------------------

#[test]
fn alter_table_is_wired_to_the_catalog() {
    let mut db = mem();
    // ALTER on a MISSING table must error — proof the arm truly calls
    // catalog.alter_table (a faked `Ok(None)` would skip that validation and
    // silently succeed). This guards the dispatch arm against being re-stubbed:
    // if a future edit reverts ALTER to "not yet supported", every assert below
    // flips, catching the regression at the engine layer (not only via the facade
    // conformance suite). All four actions are exercised: ADD / RENAME / DROP COLUMN.
    assert!(
        db.execute("ALTER TABLE missing ADD COLUMN c INTEGER").is_err(),
        "ALTER TABLE on a non-existent table must error"
    );
    db.execute("CREATE TABLE t(a INTEGER, b TEXT)").unwrap();

    // ADD COLUMN: the new column becomes visible in a SELECT.
    db.execute("ALTER TABLE t ADD COLUMN c INTEGER DEFAULT 7").unwrap();
    let r = db.query("SELECT a, b, c FROM t").unwrap();
    assert_eq!(r.columns, vec!["a".to_string(), "b".to_string(), "c".to_string()]);

    // RENAME COLUMN: the old name stops resolving, the new one resolves.
    db.execute("ALTER TABLE t RENAME COLUMN b TO renamed").unwrap();
    assert!(
        db.query("SELECT b FROM t").is_err(),
        "the pre-rename column name must no longer resolve"
    );
    db.query("SELECT renamed FROM t").unwrap();

    // DROP COLUMN: the dropped column stops resolving; the survivors remain.
    db.execute("ALTER TABLE t DROP COLUMN c").unwrap();
    assert!(
        db.query("SELECT c FROM t").is_err(),
        "the dropped column must no longer resolve"
    );
    let r = db.query("SELECT a, renamed FROM t").unwrap();
    assert_eq!(r.columns, vec!["a".to_string(), "renamed".to_string()]);
}

// ----- unsupported DDL / utility: honest loud gaps, never faked success ------

#[test]
fn unsupported_statements_are_loud_gaps_not_faked_success() {
    // Each of these parses to a `Statement` variant the engine dispatches, but no
    // seam supports it yet, so the engine must return a loud "not yet supported"
    // error. Flipping any arm to a silent `Ok(None)` (fake success) would make
    // `execute` return `Ok` and fail the substring assertion here — locking the
    // honest-gap contract for every arm at once.
    // ALTER TABLE is NOT in this list: it is now routed to the catalog seam and
    // supported (see the alter-table conformance suite), so it is no longer a gap.
    // Plain VACUUM / REINDEX are also NOT here: they have no observable effect for this
    // engine, so they are accepted no-ops (see `maintenance_statements_are_accepted_noops`).
    // ANALYZE is NOT here either: it now gathers table/index statistics into
    // `sqlite_stat1` (see the analyze conformance suite), so it is a supported
    // statement — asserted positively below so it can never regress back to a loud gap.
    // VACUUM ... INTO is NOT here anymore either: it now writes a database copy (see
    // `vacuum_into_writes_a_readable_copy` and the `conformance_vacuum_into` suite). Only
    // EXPLAIN remains a gap here, whose output is VDBE-specific.
    let mut db = mem();
    db.execute("CREATE TABLE t(a INTEGER)").unwrap();
    for sql in ["EXPLAIN SELECT 1", "EXPLAIN QUERY PLAN SELECT 1"] {
        assert_err_contains(db.execute(sql), "not yet supported");
    }
    // ANALYZE succeeds now (creating/populating sqlite_stat1), rather than erroring —
    // the depth is in the analyze conformance suite; this guards the gap-list test's own
    // contract from the opposite direction (a now-supported statement is not faked as a gap).
    db.execute("ANALYZE").unwrap();
    db.execute("ANALYZE t").unwrap();
}

#[test]
fn maintenance_statements_are_accepted_noops() {
    // Plain VACUUM (`VACUUM` / `VACUUM <schema>`) and REINDEX are maintenance statements
    // that change only PHYSICAL storage layout, never query-visible content: they preserve
    // every row/index/rowid and create no object, so each is accepted as a successful
    // no-op (returning no rows) rather than aborting the script — and must leave existing
    // data exactly as it was. (The physical rework real sqlite does — freelist reclaim /
    // file shrink / index repack — is a documented deferred gap; see the dispatch comment.
    // ANALYZE is NOT a no-op either, but it is implemented rather than a gap: it WRITES
    // statistics into `sqlite_stat1` (see the analyze conformance suite). VACUUM ... INTO
    // is likewise implemented — it writes a full database copy to the named file (see
    // `vacuum_into_writes_a_readable_copy` and the `conformance_vacuum_into` suite) — so
    // it is not a gap either; only the plain `VACUUM` form is the accepted no-op here.)
    let mut db = mem();
    db.execute("CREATE TABLE t(a INTEGER)").unwrap();
    db.execute("INSERT INTO t VALUES (1), (2), (3)").unwrap();
    for sql in ["VACUUM", "VACUUM main", "REINDEX", "REINDEX t"] {
        db.execute(sql).unwrap_or_else(|e| panic!("{sql} should be an accepted no-op, got {e:?}"));
    }
    // The rows are untouched by the maintenance statements above.
    let r = db.query("SELECT a FROM t ORDER BY a").unwrap();
    assert_eq!(r.rows.len(), 3, "VACUUM/REINDEX must preserve all rows");
}

#[test]
fn vacuum_into_writes_a_readable_copy() {
    // `VACUUM INTO '<file>'` writes a full copy of the database to the named file. This
    // engine now IMPLEMENTS the copy (see the `conformance_vacuum_into` suite for the
    // fidelity depth), so `execute` must SUCCEED and produce a file that reopens with the
    // same rows — not a silent `Ok` that creates nothing, and not the old "not yet
    // supported" gap. The reopen-and-compare below is the anti-fake check: a no-op copy
    // would leave the file absent or empty and fail here.
    let out = unique_db_path("vacuum-into");
    cleanup(&out); // start from an absent target
    let mut db = mem();
    db.execute("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT)").unwrap();
    db.execute("INSERT INTO t VALUES (1, 'one'), (2, 'two'), (3, 'three')").unwrap();
    let sql = format!("VACUUM INTO '{}'", out.display());
    db.execute(&sql).expect("VACUUM INTO should succeed");
    assert!(out.exists(), "VACUUM INTO must create the copy file");

    // Reopen the copy with a fresh engine (real on-disk read-back, not the source's
    // in-memory state) and confirm identical content.
    {
        let mut copy = open(&out).expect("the vacuumed copy must reopen");
        let r = copy.query("SELECT a, b FROM t ORDER BY a").unwrap();
        assert_eq!(r.rows.len(), 3, "all rows copied");
        assert!(matches!(&r.rows[0][0], Value::Integer(1)));
        assert!(matches!(&r.rows[0][1], Value::Text(s) if s == "one"));
        assert!(matches!(&r.rows[2][0], Value::Integer(3)));
        assert!(matches!(&r.rows[2][1], Value::Text(s) if s == "three"));
    }

    // The source is only READ: it is still queryable and writable afterward.
    let r = db.query("SELECT count(*) FROM t").unwrap();
    assert!(matches!(&r.rows[0][0], Value::Integer(3)), "source unchanged");
    db.execute("INSERT INTO t VALUES (4, 'four')").expect("source still writable");

    cleanup(&out);
}

#[test]
fn integrity_and_quick_check_report_ok() {
    // `PRAGMA integrity_check` / `quick_check` return a single `ok` row for a sound
    // database (this engine only ever writes well-formed files), NOT an empty result —
    // SQLite always yields at least one row, so an empty set would itself diverge. The
    // result column is named after the pragma. Pins the change from the pre-existing
    // empty-result behavior so a regression back to it fails.
    let mut db = mem();
    db.execute("CREATE TABLE t(a INTEGER)").unwrap();
    db.execute("INSERT INTO t VALUES (1), (2)").unwrap();
    for pragma in ["integrity_check", "quick_check"] {
        let r = db.query(&format!("PRAGMA {pragma}")).unwrap();
        assert_eq!(r.columns, vec![pragma.to_string()], "column named after the pragma");
        assert_eq!(r.rows.len(), 1, "exactly one row (never empty)");
        assert!(matches!(&r.rows[0][0], Value::Text(s) if s == "ok"), "reports ok, got {:?}", r.rows[0][0]);
    }
}

#[test]
fn database_list_reports_the_main_database() {
    // `PRAGMA database_list` always yields at least the `main` row (seq 0). For an
    // in-memory database the `file` column is the empty string, as SQLite reports for
    // `:memory:`. Returning a row (not an empty set) matches SQLite.
    let mut db = mem();
    let r = db.query("PRAGMA database_list").unwrap();
    assert_eq!(r.columns, vec!["seq".to_string(), "name".to_string(), "file".to_string()]);
    assert_eq!(r.rows.len(), 1, "exactly the main database");
    assert!(matches!(&r.rows[0][0], Value::Integer(0)), "seq 0");
    assert!(matches!(&r.rows[0][1], Value::Text(s) if s == "main"), "name main");
    assert!(matches!(&r.rows[0][2], Value::Text(s) if s.is_empty()), "in-memory file is empty");
}

#[test]
fn page_count_and_freelist_count_report_integers() {
    // `PRAGMA page_count` reports the live page total from the pager (>= 1 for any
    // formatted database); `PRAGMA freelist_count` reports the header's free-page count
    // (0 for a freshly built database). Both return one integer row, not the empty set
    // the unknown-pragma catch-all would otherwise give — the divergence these fix.
    fn page_count(db: &mut dyn Engine) -> i64 {
        let r = db.query("PRAGMA page_count").unwrap();
        assert_eq!(r.columns, vec!["page_count".to_string()]);
        assert_eq!(r.rows.len(), 1, "exactly one row (never empty)");
        match &r.rows[0][0] {
            Value::Integer(n) => *n,
            other => panic!("expected an integer page_count, got {other:?}"),
        }
    }
    let mut db = mem();
    db.execute("CREATE TABLE t(a INTEGER)").unwrap();
    let before = page_count(db.as_mut());
    assert!(before >= 1, "a formatted db has at least page 1, got {before}");

    // The count is LIVE, not a constant: a blob far larger than one page spills onto
    // overflow pages, so page_count must grow. This defeats a stub that hardcodes a
    // fixed value while still satisfying the `>= 1` wiring check.
    db.execute("CREATE TABLE big(b BLOB)").unwrap();
    db.execute("INSERT INTO big VALUES (zeroblob(50000))").unwrap();
    let after = page_count(db.as_mut());
    assert!(after > before, "page_count must grow after adding overflow pages ({before} -> {after})");

    let fc = db.query("PRAGMA freelist_count").unwrap();
    assert_eq!(fc.columns, vec!["freelist_count".to_string()]);
    assert_eq!(fc.rows.len(), 1, "exactly one row (never empty)");
    assert!(
        matches!(&fc.rows[0][0], Value::Integer(0)),
        "no page has been freed, so the free-list is empty, got {:?}",
        fc.rows[0][0]
    );
}

// ----- CREATE VIEW / CREATE TRIGGER (wired through the catalog) --------------

#[test]
fn create_view_is_wired_to_the_catalog() {
    let mut db = mem();
    db.execute("CREATE TABLE t(a INTEGER, b TEXT)").unwrap();
    // The catalog persists a `type='view'` row; the view is then visible in the
    // schema table (proof the arm truly called catalog.create_view, not a faked
    // `Ok(None)` that would skip the write).
    db.execute("CREATE VIEW v AS SELECT a FROM t").unwrap();
    let r = db
        .query("SELECT type, name, tbl_name FROM sqlite_master WHERE name = 'v'")
        .unwrap();
    assert_eq!(r.rows.len(), 1, "the created view must appear in sqlite_master");
    assert!(matches!(&r.rows[0][0], Value::Text(s) if s == "view"));
    assert!(matches!(&r.rows[0][1], Value::Text(s) if s == "v"));
    // For a view, tbl_name is a copy of the name (schematab.html §3).
    assert!(matches!(&r.rows[0][2], Value::Text(s) if s == "v"));

    // A DROP after the CREATE removes the schema row (routes through drop_object).
    db.execute("DROP VIEW v").unwrap();
    let after = db.query("SELECT count(*) FROM sqlite_master WHERE name = 'v'").unwrap();
    assert!(matches!(after.rows[0][0], Value::Integer(0)), "DROP VIEW removes the row");
}

#[test]
fn create_trigger_is_wired_to_the_catalog() {
    let mut db = mem();
    db.execute("CREATE TABLE t(a INTEGER, b TEXT)").unwrap();
    // A trigger on a MISSING table must error — proof the arm truly calls
    // catalog.create_trigger (which validates the target table), not a faked `Ok`.
    assert!(
        db.execute("CREATE TRIGGER trg AFTER INSERT ON missing BEGIN SELECT 1; END").is_err(),
        "CREATE TRIGGER on a non-existent target table must error"
    );

    db.execute("CREATE TRIGGER trg AFTER INSERT ON t BEGIN SELECT 1; END").unwrap();
    let r = db
        .query("SELECT type, name, tbl_name FROM sqlite_master WHERE name = 'trg'")
        .unwrap();
    assert_eq!(r.rows.len(), 1, "the created trigger must appear in sqlite_master");
    assert!(matches!(&r.rows[0][0], Value::Text(s) if s == "trigger"));
    assert!(matches!(&r.rows[0][1], Value::Text(s) if s == "trg"));
    // For a trigger, tbl_name is the table that fires it (schematab.html §3).
    assert!(matches!(&r.rows[0][2], Value::Text(s) if s == "t"));

    // A DROP after the CREATE removes the schema row (routes through drop_object).
    db.execute("DROP TRIGGER trg").unwrap();
    let after = db.query("SELECT count(*) FROM sqlite_master WHERE name = 'trg'").unwrap();
    assert!(matches!(after.rows[0][0], Value::Integer(0)), "DROP TRIGGER removes the row");
}

// ----- PRAGMA ---------------------------------------------------------------

#[test]
fn unknown_pragma_is_a_noop_not_an_error() {
    let mut db = mem();
    // Both the bare and the assignment forms are silently ignored (SQLite behavior).
    db.execute("PRAGMA nonsense_xyz").unwrap();
    db.execute("PRAGMA nonsense_xyz = 5").unwrap();
    let r = db.query("PRAGMA nonsense_xyz").unwrap();
    assert_eq!(r.rows.len(), 0, "an unknown pragma returns no rows");
    assert_eq!(r.columns.len(), 0);
}

#[test]
fn user_version_defaults_to_zero_and_roundtrips() {
    let mut db = mem();
    let r = db.query("PRAGMA user_version").unwrap();
    assert_eq!(r.columns, vec!["user_version".to_string()]);
    assert_eq!(r.rows.len(), 1);
    assert!(matches!(r.rows[0][0], Value::Integer(0)), "fresh db has user_version 0");

    db.execute("PRAGMA user_version = 42").unwrap();
    let r = db.query("PRAGMA user_version").unwrap();
    assert!(matches!(r.rows[0][0], Value::Integer(42)), "set value reads back");
}

// ----- other page-1 header-field PRAGMAs ------------------------------------

#[test]
fn application_id_defaults_to_zero_and_roundtrips() {
    let mut db = mem();
    let r = db.query("PRAGMA application_id").unwrap();
    assert_eq!(r.columns, vec!["application_id".to_string()]);
    assert!(matches!(r.rows[0][0], Value::Integer(0)), "fresh db has application_id 0");

    // 0x01020304 makes all four header bytes distinct, so a byte-order slip would
    // read back a different integer.
    db.execute("PRAGMA application_id = 16909060").unwrap();
    let r = db.query("PRAGMA application_id").unwrap();
    assert!(matches!(r.rows[0][0], Value::Integer(16909060)), "application_id reads back");
}

#[test]
fn schema_version_pragma_force_sets_the_cookie() {
    let mut db = mem();
    // schema_version force-writes the schema cookie (off 40) directly; issued after a
    // table exists, exactly as the on-disk conformance case does.
    db.execute("CREATE TABLE t(a)").unwrap();
    db.execute("PRAGMA schema_version = 424242").unwrap();
    let r = db.query("PRAGMA schema_version").unwrap();
    assert_eq!(r.columns, vec!["schema_version".to_string()]);
    assert!(matches!(r.rows[0][0], Value::Integer(424242)), "force-set cookie reads back");
}

#[test]
fn default_cache_size_pragma_roundtrips() {
    let mut db = mem();
    db.execute("PRAGMA default_cache_size = 500").unwrap();
    let r = db.query("PRAGMA default_cache_size").unwrap();
    assert_eq!(r.columns, vec!["default_cache_size".to_string()]);
    assert!(matches!(r.rows[0][0], Value::Integer(500)), "default_cache_size reads back");
}

#[test]
fn page_size_defaults_to_4096_on_get() {
    let mut db = mem();
    let r = db.query("PRAGMA page_size").unwrap();
    assert_eq!(r.columns, vec!["page_size".to_string()]);
    assert!(matches!(r.rows[0][0], Value::Integer(4096)), "default page size is 4096");
}

#[test]
fn page_size_set_on_empty_db_takes_effect_and_pager_is_usable() {
    let mut db = mem();
    db.execute("PRAGMA page_size = 8192").unwrap();
    let r = db.query("PRAGMA page_size").unwrap();
    assert!(matches!(r.rows[0][0], Value::Integer(8192)), "page size changes on an empty db");
    // The rebuilt pager must be usable at the new size: a table can be created and read.
    db.execute("CREATE TABLE t(a)").unwrap();
    db.execute("INSERT INTO t VALUES (1)").unwrap();
    let r = db.query("SELECT a FROM t").unwrap();
    assert_eq!(r.rows.len(), 1, "the 8192-byte pager stores and reads a row");
}

#[test]
fn page_size_set_after_a_table_exists_is_a_noop() {
    let mut db = mem();
    db.execute("CREATE TABLE t(a)").unwrap();
    // A data page now exists, so the size is fixed: the pragma is silently ignored.
    db.execute("PRAGMA page_size = 8192").unwrap();
    let r = db.query("PRAGMA page_size").unwrap();
    assert!(matches!(r.rows[0][0], Value::Integer(4096)), "page size is fixed once a table exists");
}

#[test]
fn page_size_invalid_values_are_silently_ignored() {
    let mut db = mem();
    // Non-power-of-two, below the 512 minimum, and a power of two above the 65536
    // maximum — each must be a no-op (SQLite does not error), not a panic in the
    // pager's page-size assertion.
    for bad in ["1000", "0", "131072"] {
        db.execute(&format!("PRAGMA page_size = {bad}")).unwrap();
        let r = db.query("PRAGMA page_size").unwrap();
        assert!(
            matches!(r.rows[0][0], Value::Integer(4096)),
            "invalid page size {bad} must be ignored, leaving 4096"
        );
    }
}

#[test]
fn encoding_and_auto_vacuum_set_on_empty_db_are_honored_then_fixed() {
    // GET honestly reports the engine's real state (UTF-8 / auto-vacuum off on a fresh
    // db), matching real sqlite. Both settings are creation-time properties (pragma.html
    // "encoding" / "auto_vacuum"): a set on a still-empty database is HONORED and the last
    // one while empty wins, then the property is FIXED once the first object exists (a later
    // set is a silent no-op). `PRAGMA auto_vacuum` now writes real ptrmap-backed header
    // fields (offset 52/64), so the mode round-trips instead of being an accepted no-op.
    let mut db = mem();

    let enc = db.query("PRAGMA encoding").unwrap();
    assert_eq!(enc.columns, vec!["encoding".to_string()]);
    assert!(
        matches!(&enc.rows[0][0], Value::Text(s) if s == "UTF-8"),
        "fresh db reports encoding UTF-8"
    );
    let av = db.query("PRAGMA auto_vacuum").unwrap();
    assert_eq!(av.columns, vec!["auto_vacuum".to_string()]);
    assert!(matches!(av.rows[0][0], Value::Integer(0)), "fresh db reports auto_vacuum 0 (none)");

    // encoding SET on an EMPTY database is honored; the last set while empty wins.
    db.execute("PRAGMA encoding = 'UTF-16le'").unwrap();
    assert!(
        matches!(&db.query("PRAGMA encoding").unwrap().rows[0][0], Value::Text(s) if s == "UTF-16le"),
        "encoding = UTF-16le is honored on an empty database"
    );
    db.execute("PRAGMA encoding = 'UTF-16be'").unwrap();
    assert!(
        matches!(&db.query("PRAGMA encoding").unwrap().rows[0][0], Value::Text(s) if s == "UTF-16be"),
        "a second encoding set still applies while the database is empty"
    );

    // auto_vacuum SET on the still-empty database is now HONORED and round-trips; the last
    // set while empty wins (FULL, then INCREMENTAL), exactly like encoding above.
    db.execute("PRAGMA auto_vacuum = 1").unwrap();
    assert!(
        matches!(db.query("PRAGMA auto_vacuum").unwrap().rows[0][0], Value::Integer(1)),
        "auto_vacuum = 1 (FULL) is honored on an empty database"
    );
    db.execute("PRAGMA auto_vacuum = 2").unwrap();
    assert!(
        matches!(db.query("PRAGMA auto_vacuum").unwrap().rows[0][0], Value::Integer(2)),
        "a second auto_vacuum set still applies while the database is empty (INCREMENTAL)"
    );

    // Once an object exists BOTH properties are FIXED: a later set is a silent no-op, as in
    // sqlite (encoding "cannot be changed after … created"; auto_vacuum "must be turned on
    // before any tables are created").
    db.execute("CREATE TABLE t(a)").unwrap();
    db.execute("PRAGMA encoding = 'UTF-8'").unwrap();
    assert!(
        matches!(&db.query("PRAGMA encoding").unwrap().rows[0][0], Value::Text(s) if s == "UTF-16be"),
        "encoding is fixed at creation: a set after the first table is a no-op"
    );
    db.execute("PRAGMA auto_vacuum = 0").unwrap();
    assert!(
        matches!(db.query("PRAGMA auto_vacuum").unwrap().rows[0][0], Value::Integer(2)),
        "auto_vacuum is fixed at creation: a set after the first table is a no-op (stays INCREMENTAL)"
    );
}

#[test]
fn signed_header_pragmas_reinterpret_the_high_bit() {
    // The signed-32-bit reinterpret (`v as i32 as u32` on set, `u32 as i32 as i64` on
    // get) must treat these fields as SIGNED, matching sqlite. A value with bit 31 set
    // (2^31) round-trips as the negative i32, not the unsigned 2^31 — a future edit
    // that dropped the `as i32` and read the field unsigned would fail here.
    let mut db = mem();
    db.execute("PRAGMA application_id = 2147483648").unwrap(); // 0x8000_0000
    assert!(
        matches!(db.query("PRAGMA application_id").unwrap().rows[0][0], Value::Integer(-2147483648)),
        "application_id is signed: 0x80000000 reads back as i32::MIN"
    );
}

#[test]
fn page_size_inside_a_transaction_is_a_noop() {
    // A page-size change rebuilds the pager; doing that mid-transaction would strand
    // the open transaction on the discarded pager. Like SQLite, the pragma is a silent
    // no-op inside a transaction, and the connection stays consistent — COMMIT still
    // works against the never-swapped pager, and normal work continues afterwards.
    let mut db = mem();
    db.execute("BEGIN").unwrap();
    db.execute("PRAGMA page_size = 8192").unwrap();
    assert!(
        matches!(db.query("PRAGMA page_size").unwrap().rows[0][0], Value::Integer(4096)),
        "page_size is ignored inside a transaction"
    );
    db.execute("COMMIT").unwrap(); // would error if the pager had been swapped mid-txn
    db.execute("CREATE TABLE t(a)").unwrap();
    db.execute("INSERT INTO t VALUES (1)").unwrap();
    assert_eq!(db.query("SELECT a FROM t").unwrap().rows.len(), 1, "connection still usable");
}

// ----- PRAGMA foreign_keys / foreign_key_list -------------------------------

/// A comparable projection of a result cell (`Value` is not `PartialEq`), so a whole
/// `foreign_key_list` row can be asserted at once. Only the storage classes that pragma
/// emits are represented; any other is a test bug worth a panic.
#[derive(Debug, PartialEq)]
enum Cell {
    Int(i64),
    Text(String),
    Null,
}

fn cell(v: &Value) -> Cell {
    match v {
        Value::Integer(i) => Cell::Int(*i),
        Value::Text(s) => Cell::Text(s.clone()),
        Value::Null => Cell::Null,
        other => panic!("unexpected value in a pragma row: {other:?}"),
    }
}

fn rows_as_cells(r: &minisqlite_types::QueryResult) -> Vec<Vec<Cell>> {
    r.rows.iter().map(|row| row.iter().map(cell).collect()).collect()
}

#[test]
fn foreign_keys_defaults_off_and_roundtrips() {
    let mut db = mem();
    // SQLite defaults foreign-key enforcement OFF (pragma.html, 3.6.19+).
    let r = db.query("PRAGMA foreign_keys").unwrap();
    assert_eq!(r.columns, vec!["foreign_keys".to_string()]);
    assert_eq!(r.rows.len(), 1);
    assert!(matches!(r.rows[0][0], Value::Integer(0)), "fresh db reports foreign_keys 0 (off)");

    db.execute("PRAGMA foreign_keys = ON").unwrap();
    assert!(matches!(db.query("PRAGMA foreign_keys").unwrap().rows[0][0], Value::Integer(1)), "ON -> 1");

    db.execute("PRAGMA foreign_keys = OFF").unwrap();
    assert!(matches!(db.query("PRAGMA foreign_keys").unwrap().rows[0][0], Value::Integer(0)), "OFF -> 0");

    // The 1 / 0 numeric spellings work too.
    db.execute("PRAGMA foreign_keys = 1").unwrap();
    assert!(matches!(db.query("PRAGMA foreign_keys").unwrap().rows[0][0], Value::Integer(1)), "= 1 -> 1");
    db.execute("PRAGMA foreign_keys = 0").unwrap();
    assert!(matches!(db.query("PRAGMA foreign_keys").unwrap().rows[0][0], Value::Integer(0)), "= 0 -> 0");
}

#[test]
fn foreign_keys_set_is_a_noop_within_a_transaction() {
    // SQLite forbids changing FK enforcement mid-transaction (pragma.html): the SET is a
    // silent no-op while a transaction is pending; the GET still reports the live value.
    let mut db = mem();
    db.execute("BEGIN").unwrap();
    db.execute("PRAGMA foreign_keys = ON").unwrap(); // no-op: a transaction is active
    assert!(
        matches!(db.query("PRAGMA foreign_keys").unwrap().rows[0][0], Value::Integer(0)),
        "foreign_keys stays off when set inside a transaction"
    );
    db.execute("COMMIT").unwrap();
    // Outside a transaction the same statement now takes effect.
    db.execute("PRAGMA foreign_keys = ON").unwrap();
    assert!(matches!(db.query("PRAGMA foreign_keys").unwrap().rows[0][0], Value::Integer(1)));
}

#[test]
fn foreign_key_list_reports_single_and_multi_column_fks_with_reverse_id_numbering() {
    // SQLite numbers foreign keys with the LAST-declared getting id 0 (see pragma.html /
    // pragma.c). Two table-level FKs: FK(a,b)->parent is declared first, FK(c)->parent
    // second, so the single-column FK(c) is id 0 and the composite FK(a,b) is id 1. A
    // multi-column FK emits one row per child column (seq 0,1). Column order is exactly
    // id, seq, table, from, to, on_update, on_delete, match.
    let mut db = mem();
    db.execute("CREATE TABLE parent(x INTEGER PRIMARY KEY, y)").unwrap();
    db.execute(
        "CREATE TABLE t(a, b, c, \
         FOREIGN KEY(a, b) REFERENCES parent(x, y) ON DELETE CASCADE, \
         FOREIGN KEY(c) REFERENCES parent(x) ON UPDATE SET NULL)",
    )
    .unwrap();

    let r = db.query("PRAGMA foreign_key_list(t)").unwrap();
    assert_eq!(
        r.columns,
        vec![
            "id".to_string(),
            "seq".to_string(),
            "table".to_string(),
            "from".to_string(),
            "to".to_string(),
            "on_update".to_string(),
            "on_delete".to_string(),
            "match".to_string(),
        ]
    );
    let t = |s: &str| Cell::Text(s.to_string());
    assert_eq!(
        rows_as_cells(&r),
        vec![
            // id 0 == the LAST-declared FK: the single-column FK(c) -> parent(x).
            vec![Cell::Int(0), Cell::Int(0), t("parent"), t("c"), t("x"), t("SET NULL"), t("NO ACTION"), t("NONE")],
            // id 1 == the composite FK(a,b) -> parent(x,y), one row per child column.
            vec![Cell::Int(1), Cell::Int(0), t("parent"), t("a"), t("x"), t("NO ACTION"), t("CASCADE"), t("NONE")],
            vec![Cell::Int(1), Cell::Int(1), t("parent"), t("b"), t("y"), t("NO ACTION"), t("CASCADE"), t("NONE")],
        ]
    );
}

#[test]
fn foreign_key_list_id_ordering_mixes_column_and_table_level_fks() {
    // The subtlest slice of the id numbering (diffed byte-for-byte against real sqlite):
    // SQLite builds its per-table FK list by PREPENDING each constraint as it is parsed —
    // column-level `REFERENCES` first, in column order, then the table-level `FOREIGN KEY`s
    // — and `foreign_key_list` walks that list head-first, so the ids come out in REVERSE
    // of declaration order regardless of whether an FK is column- or table-level. Here `a`'s
    // column-level FK is declared first, then table FK(b), then table FK(c): the last one
    // (c) is id 0, then b is id 1, and the column-level FK on `a` sinks to the highest id 2.
    // Our builder records foreign_keys in declaration order [column-level…, table-level…]
    // and the pragma reverses it, which must reproduce exactly this interleaving.
    let mut db = mem();
    db.execute("CREATE TABLE pa(i INTEGER PRIMARY KEY)").unwrap();
    db.execute("CREATE TABLE pc(i INTEGER PRIMARY KEY)").unwrap();
    db.execute("CREATE TABLE pd(i INTEGER PRIMARY KEY)").unwrap();
    db.execute(
        "CREATE TABLE t(a REFERENCES pa(i), b, c, \
         FOREIGN KEY(b) REFERENCES pc(i), \
         FOREIGN KEY(c) REFERENCES pd(i))",
    )
    .unwrap();
    let r = db.query("PRAGMA foreign_key_list(t)").unwrap();
    let t = |s: &str| Cell::Text(s.to_string());
    assert_eq!(
        rows_as_cells(&r),
        vec![
            // id 0 == the LAST-declared FK overall: the table-level FK(c) -> pd.
            vec![Cell::Int(0), Cell::Int(0), t("pd"), t("c"), t("i"), t("NO ACTION"), t("NO ACTION"), t("NONE")],
            // id 1 == the first table-level FK(b) -> pc.
            vec![Cell::Int(1), Cell::Int(0), t("pc"), t("b"), t("i"), t("NO ACTION"), t("NO ACTION"), t("NONE")],
            // id 2 == the column-level FK on `a` -> pa, which sinks to the highest id.
            vec![Cell::Int(2), Cell::Int(0), t("pa"), t("a"), t("i"), t("NO ACTION"), t("NO ACTION"), t("NONE")],
        ]
    );
}

#[test]
fn foreign_key_list_to_is_null_when_referencing_parent_primary_key() {
    // A `REFERENCES parent` with no explicit column list targets the parent's PRIMARY KEY;
    // SQLite prints `to` as NULL for such an FK (not the resolved PK column name).
    let mut db = mem();
    db.execute("CREATE TABLE parent(x INTEGER PRIMARY KEY)").unwrap();
    db.execute("CREATE TABLE t(a REFERENCES parent)").unwrap();
    let r = db.query("PRAGMA foreign_key_list(t)").unwrap();
    let t = |s: &str| Cell::Text(s.to_string());
    assert_eq!(
        rows_as_cells(&r),
        vec![vec![
            Cell::Int(0),
            Cell::Int(0),
            t("parent"),
            t("a"),
            Cell::Null,
            t("NO ACTION"),
            t("NO ACTION"),
            t("NONE"),
        ]]
    );
}

#[test]
fn foreign_key_list_is_empty_for_a_table_without_fks_and_for_a_missing_table() {
    // A table with no foreign keys yields the fixed columns and zero rows; likewise a
    // completely unknown table (SQLite returns an empty result set, not an error).
    let mut db = mem();
    db.execute("CREATE TABLE nofk(a, b)").unwrap();
    let r = db.query("PRAGMA foreign_key_list(nofk)").unwrap();
    assert_eq!(r.columns.len(), 8, "the fixed column set is always declared");
    assert_eq!(r.rows.len(), 0, "no foreign keys -> no rows");

    let missing = db.query("PRAGMA foreign_key_list(does_not_exist)").unwrap();
    assert_eq!(missing.columns.len(), 8);
    assert_eq!(missing.rows.len(), 0, "a missing table -> columns and no rows");
}

// ----- query() result-set selection over a program --------------------------

#[test]
fn query_returns_the_last_result_producing_statement() {
    let mut db = mem();
    let r = db.query("SELECT 1; SELECT 2").unwrap();
    assert_eq!(r.rows.len(), 1);
    assert!(matches!(r.rows[0][0], Value::Integer(2)), "last SELECT's result is returned");
}

#[test]
fn query_of_only_ddl_returns_an_empty_result_set() {
    let mut db = mem();
    let r = db.query("CREATE TABLE t(a)").unwrap();
    assert_eq!(r.columns.len(), 0);
    assert_eq!(r.rows.len(), 0);
}

// ----- DML is a reported gap until the planner compiles it ------------------

#[test]
fn insert_routes_through_the_planner_without_panicking() {
    // DML is planned + executed by the seams, not the engine; the engine's job is to
    // route INSERT there, wrap it in an implicit transaction, and surface the result.
    // This test is evergreen across the "INSERT planning lands" transition:
    //   * While minisqlite-plan's INSERT compile is a stub, `plan()` returns an Err
    //     BEFORE any write — so the implicit txn never opens and the table stays empty.
    //   * Once DML planning lands, the same call inserts a row and (autocommit) commits
    //     it, so a following SELECT sees it — with NO engine change.
    // Either way the engine must not panic and must leave a consistent, usable state.
    let mut db = mem();
    db.execute("CREATE TABLE t(a INTEGER, b TEXT)").unwrap();
    match db.execute("INSERT INTO t VALUES (1, 'x')") {
        Ok(()) => {
            let r = db.query("SELECT a, b FROM t").unwrap();
            assert_eq!(r.rows.len(), 1, "a committed INSERT must be readable back");
            assert!(matches!(r.rows[0][0], Value::Integer(1)));
        }
        Err(_) => {
            // Planning still a stub: the failed statement wrote nothing (durability).
            let r = db.query("SELECT a, b FROM t").unwrap();
            assert_eq!(r.rows.len(), 0, "a failed INSERT must leave no partial write");
        }
    }
}

// ----- on-disk open(): create, close, reopen (durability) -------------------

#[test]
fn on_disk_schema_and_user_version_survive_reopen() {
    let path = unique_db_path("persist");

    // First connection: format a fresh file, create a table, set the user version.
    {
        let mut db = open(&path).expect("open a fresh on-disk database");
        db.execute("CREATE TABLE t(a INTEGER, b TEXT)").unwrap();
        db.execute("PRAGMA user_version = 7").unwrap();
        // Usable within the same connection right away.
        let r = db.query("SELECT a, b FROM t").unwrap();
        assert_eq!(r.columns, vec!["a".to_string(), "b".to_string()]);
    }

    // Reopen: the schema (via catalog.load from page 1) and the header user_version
    // must both be recovered from the committed file.
    {
        let mut db = open(&path).expect("reopen the existing on-disk database");
        let r = db.query("SELECT a, b FROM t").unwrap();
        assert_eq!(r.columns, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(r.rows.len(), 0);
        let uv = db.query("PRAGMA user_version").unwrap();
        assert!(matches!(uv.rows[0][0], Value::Integer(7)), "user_version persisted across reopen");
    }

    cleanup(&path);
}

#[test]
fn savepoint_started_release_survives_reopen() {
    // The disk-backing durability criterion for savepoints: a SAVEPOINT that STARTS
    // the transaction, followed by an outermost RELEASE, commits through the same
    // durable path as COMMIT, so the rows survive closing and reopening the file. A
    // RELEASE that failed to commit would abandon the open transaction on close and
    // the rows would vanish on reopen (count 0).
    let path = unique_db_path("savepoint-persist");

    {
        let mut db = open(&path).expect("open a fresh on-disk database");
        db.execute("CREATE TABLE t(a)").unwrap(); // autocommitted, exists after reopen
        db.execute("SAVEPOINT sp1").unwrap(); // starts a transaction
        db.execute("INSERT INTO t VALUES (1)").unwrap();
        db.execute("INSERT INTO t VALUES (2)").unwrap();
        db.execute("RELEASE sp1").unwrap(); // outermost RELEASE == durable COMMIT
    }

    {
        let mut db = open(&path).expect("reopen the existing on-disk database");
        let r = db.query("SELECT count(*) FROM t").unwrap();
        assert!(
            matches!(r.rows[0][0], Value::Integer(2)),
            "the savepoint-started RELEASE persisted both rows across reopen"
        );
    }

    cleanup(&path);
}

// ----- on-disk page_size (the header off-16 must match the real layout) ------

#[test]
fn on_disk_page_size_pragma_writes_header_and_lays_out_the_file() {
    // The engine-level counterpart of the facade conformance byte-check: `PRAGMA
    // page_size = N` before any table both records N at header offset 16 (big-endian)
    // AND lays the file out in N-byte pages (length a whole multiple of N). A
    // regression that only patched the header byte — leaving the pager at 4096 —
    // writes a file real sqlite could not read and is caught by the length assertion.
    let path = unique_db_path("pagesize8k");
    {
        let mut db = open(&path).expect("open a fresh on-disk database");
        db.execute("PRAGMA page_size = 8192").unwrap();
        db.execute("CREATE TABLE t(a)").unwrap();
    }
    let raw = std::fs::read(&path).expect("read the db file");
    let off16 = u16::from_be_bytes([raw[16], raw[17]]);
    assert_eq!(off16, 8192, "header offset 16 records the new page size 8192");
    assert_eq!(raw.len() % 8192, 0, "file laid out in whole 8192-byte pages, got len {}", raw.len());
    assert!(raw.len() >= 8192, "at least one 8192-byte page written");
    cleanup(&path);
}

#[test]
fn on_disk_page_size_65536_uses_sentinel_and_reopens() {
    // 65536 does not fit the 2-byte off-16 field, so it is stored as the sentinel 1;
    // the file must still be laid out in 65536-byte pages, and the database must
    // reopen and recover its schema at that size — proving the written header is one a
    // conformant reader accepts, not a 4096-page file mislabeled 65536.
    let path = unique_db_path("pagesize64k");
    {
        let mut db = open(&path).expect("open a fresh on-disk database");
        db.execute("PRAGMA page_size = 65536").unwrap();
        db.execute("CREATE TABLE t(a)").unwrap();
    }
    let raw = std::fs::read(&path).expect("read the db file");
    let off16 = u16::from_be_bytes([raw[16], raw[17]]);
    assert_eq!(off16, 1, "65536 is stored as the sentinel value 1 at offset 16");
    assert_eq!(raw.len() % 65536, 0, "file laid out in 65536-byte pages, got len {}", raw.len());
    {
        let mut db = open(&path).expect("reopen the 65536-page database");
        let r = db.query("PRAGMA page_size").unwrap();
        assert!(matches!(r.rows[0][0], Value::Integer(65536)), "reopened db reports 65536");
        db.query("SELECT a FROM t").unwrap(); // schema recovered at the new page size
    }
    cleanup(&path);
}

#[test]
fn on_disk_default_page_size_is_unchanged_without_the_pragma() {
    // Guard the "default fresh db is byte-unchanged" criterion: with NO page_size
    // pragma the on-disk header keeps 4096 at offset 16 and the file is 4096-page.
    let path = unique_db_path("pagesizedefault");
    {
        let mut db = open(&path).expect("open a fresh on-disk database");
        db.execute("CREATE TABLE t(a)").unwrap();
    }
    let raw = std::fs::read(&path).expect("read the db file");
    let off16 = u16::from_be_bytes([raw[16], raw[17]]);
    assert_eq!(off16, 4096, "a db created without the page_size pragma stays 4096");
    assert_eq!(raw.len() % 4096, 0, "file laid out in 4096-byte pages");
    cleanup(&path);
}

#[test]
fn on_disk_encoding_and_auto_vacuum_set_write_real_header_and_ptrmap() {
    // Pinned at the byte level: `PRAGMA encoding = 'UTF-16le'` on a still-empty database
    // writes off 56 (text encoding) to 2 — the encoding is fixed at creation (pragma.html
    // "encoding") and the record codec is encoding-aware, so the file is real-sqlite-readable
    // UTF-16le. `PRAGMA auto_vacuum = 1` (FULL) on the still-empty database is HONORED and
    // backed by a real ptrmap structure: after the first table, off 52 (largest root b-tree)
    // becomes the table's root page 3 (page 2 is reserved as the first ptrmap page), off 64
    // (incremental flag) stays 0 for FULL, and page 2 holds page 3's back-pointer entry
    // (type 1 root, parent 0) — NOT fabricated header bytes, but a cross-readable layout.
    let path = unique_db_path("encoding-set-bytes");
    {
        let mut db = open(&path).expect("open a fresh on-disk database");
        db.execute("PRAGMA encoding = 'UTF-16le'").unwrap();
        db.execute("PRAGMA auto_vacuum = 1").unwrap();
        // Force a real on-disk commit so the header + ptrmap bytes are actually written.
        db.execute("CREATE TABLE t(a)").unwrap();
    }
    let raw = std::fs::read(&path).expect("read the db file");
    let be32 = |off: usize| u32::from_be_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
    let page_size = 4096usize; // default: no page_size pragma set
    assert_eq!(be32(56), 2, "off 56 text-encoding is 2 (UTF-16le) after the honored set");
    assert_eq!(be32(52), 3, "off 52 largest-root-btree is the table root (page 3) in auto_vacuum");
    assert_eq!(be32(64), 0, "off 64 incremental-vacuum is 0 for FULL auto_vacuum");
    // Page 2 is the first ptrmap page; page 3's entry sits at its byte offset 0 (§1.8):
    // type 1 (b-tree root), back-pointer 0. This is the anti-"painted header" check.
    let ptrmap = page_size; // page 2 starts after page 1
    assert_eq!(raw[ptrmap], 1, "page 3's ptrmap entry type is 1 (b-tree root)");
    assert_eq!(be32(ptrmap + 1), 0, "page 3's ptrmap back-pointer is 0 (a root)");
    cleanup(&path);
}
