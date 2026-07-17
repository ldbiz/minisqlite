//! Conformance battery: **DDL, constraints, ROWID, and PRAGMA**.
//!
//! Every expected result here is TRANSCRIBED FROM THE SQLITE DOCS in
//! `spec/sqlite-doc/`, never from what the engine currently returns — a failing
//! case is the intended signal that the engine diverges from the spec. A case that
//! reveals an engine bug is left as a genuine failing assertion rather than weakened
//! to pass.
//!
//! Spec sources (all under `spec/sqlite-doc/`):
//!   * `lang_createtable.html` §2 — it is an error to CREATE a table whose name
//!     already names a table/index/view; with `IF NOT EXISTS` a pre-existing
//!     table makes the statement a no-op "(and no error message is returned)".
//!   * `lang_createtable.html` §3.2 "The DEFAULT clause" — with no explicit
//!     DEFAULT the column default is NULL; an explicit constant DEFAULT is used
//!     "directly in the new row" for any column an INSERT does not supply.
//!   * `lang_createtable.html` §3.5 "The PRIMARY KEY" / §3.6 "UNIQUE
//!     constraints" — each row must hold a unique combination of the key
//!     columns, and "NULL values are considered distinct from all other values,
//!     including other NULLs" (so a UNIQUE column admits many NULLs). A
//!     duplicate key is "a constraint violation".
//!   * `lang_createtable.html` §3.8 "NOT NULL constraints" — setting a NOT NULL
//!     column to NULL on insert "causes a constraint violation".
//!   * `lang_createtable.html` §4.1 — the default conflict resolution for
//!     PRIMARY KEY / UNIQUE / NOT NULL / CHECK is ABORT, i.e. the statement
//!     errors. The engine reports these as `Error::Constraint(_)` (see
//!     `minisqlite-types`'s error taxonomy / SQLITE_CONSTRAINT).
//!   * `lang_createtable.html` §5 "ROWIDs and the INTEGER PRIMARY KEY" — every
//!     rowid table row has a 64-bit integer key reachable as `rowid`, `oid`, or
//!     `_rowid_`; a lone `INTEGER PRIMARY KEY` column is an alias for that rowid,
//!     so an omitted value is auto-assigned and `typeof` is `integer`.
//!   * `autoinc.html` — the auto-assigned rowid is "one larger than the largest
//!     ROWID in the table prior to the insert. If the table is initially empty,
//!     then a ROWID of 1 is used."
//!   * `lang_createindex.html` §1 — `CREATE INDEX ... IF NOT EXISTS` is a no-op
//!     when the name already exists; without it a duplicate name is an error.
//!   * `lang_droptable.html` — DROP TABLE removes the table (a later query is an
//!     error); the optional `IF EXISTS` "suppresses the error that would
//!     normally result if the table does not exist".
//!   * `lang_dropindex.html` — DROP INDEX removes an index created by CREATE
//!     INDEX.
//!   * `pragma.html#pragma_user_version` — `PRAGMA user_version` gets/sets the
//!     user-version integer in the database header (0 for a fresh database).
//!   * `pragma.html#pragma_table_info` — `PRAGMA table_info(t)` returns one row
//!     per normal column. (PROBE.)
//!   * `schematab.html` — the schema is visible as `sqlite_schema` (a.k.a. the
//!     legacy alias `sqlite_master`), one row per schema object with a `type`
//!     and `name`. (PROBE.)
//!
//! Each case is its own small `#[test]` so an unsupported feature fails exactly
//! that case and never masks the rest. Constraint cases are split in two: one
//! test pins the documented fact that the violation *errors at all* (robust),
//! and a second pins that it is classified as `Error::Constraint(_)` (precise),
//! so a wrong error *kind* is recorded without hiding that the abort happened.

mod conformance;

use conformance::*;
// `Error` is matched directly to check that a constraint violation is reported
// as `Error::Constraint(_)` (the SQLITE_CONSTRAINT family). `Value` has no
// `PartialEq`, so every value comparison still goes through the harness.
use minisqlite::Error;

// ===========================================================================
// CREATE TABLE — basics and round-trip (lang_createtable.html §3).
// ===========================================================================

#[test]
fn create_table_then_column_names() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    assert_columns(&mut db, "SELECT a, b FROM t", &["a", "b"]);
}

#[test]
fn create_table_insert_select_roundtrip() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "INSERT INTO t VALUES (1, 'x')");
    assert_rows(&mut db, "SELECT a, b FROM t", &[vec![int(1), text("x")]]);
}

#[test]
fn create_table_select_star_column_names() {
    // `*` expands to the table's columns in declaration order.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    assert_columns(&mut db, "SELECT * FROM t", &["a", "b"]);
}

#[test]
fn create_table_with_types_roundtrip() {
    // Declared types are allowed on columns (they set affinity); the values
    // round-trip. Column names ignore the type token.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
    exec(&mut db, "INSERT INTO t VALUES (7, 'q')");
    assert_rows(&mut db, "SELECT a, b FROM t", &[vec![int(7), text("q")]]);
}

// ===========================================================================
// CREATE TABLE — duplicate name and IF NOT EXISTS (lang_createtable.html §2).
// ===========================================================================

#[test]
fn duplicate_table_is_error() {
    // §2: "usually an error to attempt to create a new table in a database that
    // already contains a table ... of the same name."
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    assert_exec_error(&mut db, "CREATE TABLE t(a, b)");
}

#[test]
fn create_table_if_not_exists_second_is_noop_and_keeps_data() {
    // §2: with IF NOT EXISTS a pre-existing table makes CREATE "simply has no
    // effect (and no error message is returned)" — so the row survives.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "INSERT INTO t VALUES (5)");
    exec(&mut db, "CREATE TABLE IF NOT EXISTS t(a)");
    assert_scalar(&mut db, "SELECT a FROM t", int(5));
}

#[test]
fn create_table_if_not_exists_on_fresh_name_creates() {
    // With no pre-existing `t`, IF NOT EXISTS still creates the table.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE IF NOT EXISTS t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    assert_scalar(&mut db, "SELECT a FROM t", int(1));
}

// ===========================================================================
// NOT NULL constraint (lang_createtable.html §3.8, §4.1).
// ===========================================================================

#[test]
fn not_null_accepts_non_null() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER NOT NULL)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    assert_scalar(&mut db, "SELECT a FROM t", int(1));
}

#[test]
fn not_null_insert_null_is_error() {
    // §3.8: setting a NOT NULL column to NULL on insert "causes a constraint
    // violation". Here we pin only that it errors (the robust half).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER NOT NULL)");
    assert_exec_error(&mut db, "INSERT INTO t VALUES (NULL)");
}

#[test]
fn not_null_violation_is_constraint_error() {
    // §4.1: the violation aborts with a CONSTRAINT error; the engine models this
    // as `Error::Constraint(_)`.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER NOT NULL)");
    let e = assert_exec_error(&mut db, "INSERT INTO t VALUES (NULL)");
    assert!(
        matches!(e, Error::Constraint(_)),
        "NOT NULL violation should be Error::Constraint(_); got {e:?}"
    );
}

// ===========================================================================
// UNIQUE constraint (lang_createtable.html §3.6, §4.1; lang_createindex §1.1).
// ===========================================================================

#[test]
fn unique_accepts_distinct_values() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a UNIQUE)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    exec(&mut db, "INSERT INTO t VALUES (2)");
    assert_rows_unordered(&mut db, "SELECT a FROM t", &[vec![int(1)], vec![int(2)]]);
}

#[test]
fn unique_duplicate_is_error() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a UNIQUE)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    assert_exec_error(&mut db, "INSERT INTO t VALUES (1)");
}

#[test]
fn unique_duplicate_is_constraint_error() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a UNIQUE)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    let e = assert_exec_error(&mut db, "INSERT INTO t VALUES (1)");
    assert!(
        matches!(e, Error::Constraint(_)),
        "UNIQUE violation should be Error::Constraint(_); got {e:?}"
    );
}

#[test]
fn unique_allows_multiple_nulls() {
    // §3.6 / lang_createindex §1.1: "all NULL values are considered different
    // from all other NULL values and are thus unique" — so two NULLs in a UNIQUE
    // column are NOT a violation, and both rows survive.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a UNIQUE)");
    exec(&mut db, "INSERT INTO t VALUES (NULL)");
    exec(&mut db, "INSERT INTO t VALUES (NULL)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2));
}

// ===========================================================================
// PRIMARY KEY constraint and INTEGER PRIMARY KEY (lang_createtable §3.5, §5).
// ===========================================================================

#[test]
fn primary_key_accepts_distinct_values() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, b)");
    exec(&mut db, "INSERT INTO t VALUES (1, 'x')");
    exec(&mut db, "INSERT INTO t VALUES (2, 'y')");
    assert_rows_unordered(
        &mut db,
        "SELECT a, b FROM t",
        &[vec![int(1), text("x")], vec![int(2), text("y")]],
    );
}

#[test]
fn primary_key_duplicate_is_error() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, b)");
    exec(&mut db, "INSERT INTO t VALUES (1, 'x')");
    assert_exec_error(&mut db, "INSERT INTO t VALUES (1, 'y')");
}

#[test]
fn primary_key_duplicate_is_constraint_error() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, b)");
    exec(&mut db, "INSERT INTO t VALUES (1, 'x')");
    let e = assert_exec_error(&mut db, "INSERT INTO t VALUES (1, 'y')");
    assert!(
        matches!(e, Error::Constraint(_)),
        "PRIMARY KEY violation should be Error::Constraint(_); got {e:?}"
    );
}

#[test]
fn integer_primary_key_auto_assigns_rowid() {
    // §5: an omitted INTEGER PRIMARY KEY value is auto-assigned; the first row in
    // an empty table gets rowid 1 (autoinc.html), and `a` aliases that rowid.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, b)");
    exec(&mut db, "INSERT INTO t(b) VALUES ('z')");
    assert_scalar(&mut db, "SELECT a FROM t", int(1));
}

#[test]
fn integer_primary_key_typeof_is_integer() {
    // §5: an integer primary key / rowid column holds only integer values.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, b)");
    exec(&mut db, "INSERT INTO t(b) VALUES ('z')");
    assert_scalar(&mut db, "SELECT typeof(a) FROM t", text("integer"));
}

#[test]
fn integer_primary_key_second_row_is_two() {
    // Second auto-assigned rowid is one larger than the largest existing rowid.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, b)");
    exec(&mut db, "INSERT INTO t(b) VALUES ('y')");
    exec(&mut db, "INSERT INTO t(b) VALUES ('z')");
    assert_rows(
        &mut db,
        "SELECT a, b FROM t ORDER BY a",
        &[vec![int(1), text("y")], vec![int(2), text("z")]],
    );
}

// ===========================================================================
// DEFAULT clause (lang_createtable.html §3.2).
// ===========================================================================

#[test]
fn default_values_uses_column_defaults() {
    // §3.2: INSERT ... DEFAULT VALUES stores each column's default directly.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER DEFAULT 7, b TEXT DEFAULT 'q')");
    exec(&mut db, "INSERT INTO t DEFAULT VALUES");
    assert_rows(&mut db, "SELECT a, b FROM t", &[vec![int(7), text("q")]]);
}

#[test]
fn default_fills_omitted_column() {
    // §3.2: a column not supplied by the INSERT takes its DEFAULT.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER DEFAULT 7, b TEXT DEFAULT 'q')");
    exec(&mut db, "INSERT INTO t(b) VALUES ('z')");
    assert_rows(&mut db, "SELECT a, b FROM t", &[vec![int(7), text("z")]]);
}

#[test]
fn absent_default_clause_is_null() {
    // §3.2: "If there is no explicit DEFAULT clause ... then the default value of
    // the column is NULL."
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "INSERT INTO t(a) VALUES (1)");
    assert_rows(&mut db, "SELECT a, b FROM t", &[vec![int(1), null()]]);
}

// ===========================================================================
// ROWID (lang_createtable.html §5; autoinc.html).
// ===========================================================================

#[test]
fn rowid_auto_increments_from_one() {
    // §5 + autoinc.html: rowids are auto-assigned 1, 2, ... for a fresh table.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x)");
    exec(&mut db, "INSERT INTO t VALUES (10), (20)");
    assert_rows(
        &mut db,
        "SELECT rowid, x FROM t ORDER BY rowid",
        &[vec![int(1), int(10)], vec![int(2), int(20)]],
    );
}

#[test]
fn rowid_aliases_oid_and_underscore_rowid() {
    // §5: the rowid is reachable via the case-independent names "rowid", "oid",
    // and "_rowid_"; on a table with no user column of those names they all
    // return the same integer key.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x)");
    exec(&mut db, "INSERT INTO t VALUES (10)");
    assert_rows(
        &mut db,
        "SELECT rowid, oid, _rowid_ FROM t",
        &[vec![int(1), int(1), int(1)]],
    );
}

#[test]
fn explicit_rowid_is_used() {
    // §5: an INSERT may supply the rowid value directly.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x)");
    exec(&mut db, "INSERT INTO t(rowid, x) VALUES (42, 99)");
    assert_rows(&mut db, "SELECT rowid, x FROM t", &[vec![int(42), int(99)]]);
}

// ===========================================================================
// CREATE INDEX / DROP INDEX (lang_createindex.html, lang_dropindex.html).
// ===========================================================================

#[test]
fn create_index_then_equality_query_is_correct() {
    // An index must not change query results (it is a performance structure).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "CREATE INDEX idx ON t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1,'x'),(2,'y'),(3,'z')");
    assert_scalar(&mut db, "SELECT b FROM t WHERE a = 2", text("y"));
}

#[test]
fn index_created_after_rows_is_correct() {
    // Building the index on an already-populated table keeps lookups correct.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "INSERT INTO t VALUES (1,'x'),(2,'y'),(3,'z')");
    exec(&mut db, "CREATE INDEX idx ON t(a)");
    assert_scalar(&mut db, "SELECT b FROM t WHERE a = 3", text("z"));
}

#[test]
fn drop_index_then_query_still_correct() {
    // After DROP INDEX the same query still returns the right rows (full scan).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "CREATE INDEX idx ON t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1,'x'),(2,'y'),(3,'z')");
    exec(&mut db, "DROP INDEX idx");
    assert_scalar(&mut db, "SELECT b FROM t WHERE a = 2", text("y"));
}

#[test]
fn create_index_if_not_exists_is_idempotent() {
    // lang_createindex §1: IF NOT EXISTS on an existing index name is a no-op.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "CREATE INDEX IF NOT EXISTS idx ON t(a)");
    exec(&mut db, "CREATE INDEX IF NOT EXISTS idx ON t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1,'x')");
    assert_scalar(&mut db, "SELECT b FROM t WHERE a = 1", text("x"));
}

#[test]
fn duplicate_index_name_is_error() {
    // Without IF NOT EXISTS, re-declaring an index name is an error.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "CREATE INDEX idx ON t(a)");
    assert_exec_error(&mut db, "CREATE INDEX idx ON t(a)");
}

#[test]
fn unique_index_duplicate_is_error() {
    // lang_createindex §1.1: a UNIQUE index rejects a duplicate entry. Robust
    // half — pins only that the second insert aborts (mirrors the split used for
    // the column constraints above).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "CREATE UNIQUE INDEX idx ON t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1, 'x')");
    assert_exec_error(&mut db, "INSERT INTO t VALUES (1, 'y')");
}

#[test]
fn unique_index_duplicate_is_constraint_error() {
    // Precise half — the abort is classified as `Error::Constraint(_)`.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "CREATE UNIQUE INDEX idx ON t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1, 'x')");
    let e = assert_exec_error(&mut db, "INSERT INTO t VALUES (1, 'y')");
    assert!(
        matches!(e, Error::Constraint(_)),
        "UNIQUE index violation should be Error::Constraint(_); got {e:?}"
    );
}

// ===========================================================================
// DROP TABLE (lang_droptable.html).
// ===========================================================================

#[test]
fn drop_table_then_query_is_error() {
    // The table is "completely removed"; a later query finds no such table.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "DROP TABLE t");
    assert_query_error(&mut db, "SELECT a FROM t");
}

#[test]
fn drop_table_if_exists_missing_is_noop() {
    // "The optional IF EXISTS clause suppresses the error that would normally
    // result if the table does not exist." -> no error.
    let mut db = mem();
    exec(&mut db, "DROP TABLE IF EXISTS nope");
}

#[test]
fn drop_table_missing_is_error() {
    // Without IF EXISTS, dropping a non-existent table is an error.
    let mut db = mem();
    assert_exec_error(&mut db, "DROP TABLE nope");
}

#[test]
fn drop_table_then_recreate_is_ok() {
    // After a DROP the name is free again, so a fresh CREATE succeeds.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    exec(&mut db, "DROP TABLE t");
    exec(&mut db, "CREATE TABLE t(a)");
    // The recreated table is empty.
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
}

// ===========================================================================
// PRAGMA user_version (pragma.html#pragma_user_version).
// ===========================================================================

#[test]
fn pragma_user_version_default_is_zero() {
    // The user-version integer in the database header is 0 for a fresh database.
    let mut db = mem();
    assert_scalar(&mut db, "PRAGMA user_version", int(0));
}

#[test]
fn pragma_user_version_set_then_get() {
    // `PRAGMA user_version = N` sets it; a re-read returns N.
    let mut db = mem();
    exec(&mut db, "PRAGMA user_version = 42");
    assert_scalar(&mut db, "PRAGMA user_version", int(42));
}

// ===========================================================================
// PROBE — schema introspection. These pin documented behavior but exercise
// surface the engine may not implement yet.
// ===========================================================================

#[test]
fn probe_sqlite_master_lists_tables() {
    // schematab.html: user tables appear in the schema table with type='table'.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    assert_rows_unordered(
        &mut db,
        "SELECT name FROM sqlite_master WHERE type = 'table'",
        &[vec![text("t")]],
    );
}

#[test]
fn probe_sqlite_schema_lists_tables() {
    // `sqlite_schema` is the modern name for the same schema table.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    assert_rows_unordered(
        &mut db,
        "SELECT name FROM sqlite_schema WHERE type = 'table'",
        &[vec![text("t")]],
    );
}

#[test]
fn probe_table_info_one_row_per_column() {
    // pragma.html#pragma_table_info: "returns one row for each normal column."
    // Robust half — pins only the cardinality (one row per column).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    let qr = query(&mut db, "PRAGMA table_info(t)");
    assert_eq!(
        qr.rows.len(),
        2,
        "PRAGMA table_info(t) should return one row per column (2); got {:?}",
        qr.rows
    );
}

#[test]
fn probe_table_info_columns_and_values() {
    // pragma.html#pragma_table_info: each row is (cid, name, type, notnull,
    // dflt_value, pk). For `t(a, b)` with no declared types or constraints the
    // documented values are fully determined: cid is the 0-based result rank,
    // `type` is '' when none is given, `notnull`/`pk` are 0, and `dflt_value` is
    // NULL (no DEFAULT). Precise half — pins the payload, not just the count, so
    // a stub returning two arbitrary rows cannot pass.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    assert_rows(
        &mut db,
        "PRAGMA table_info(t)",
        &[
            vec![int(0), text("a"), text(""), int(0), null(), int(0)],
            vec![int(1), text("b"), text(""), int(0), null(), int(0)],
        ],
    );
}
