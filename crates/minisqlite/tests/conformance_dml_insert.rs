//! Conformance battery: the SQLite **INSERT** statement in all its forms, plus
//! the **ON CONFLICT** (`INSERT OR ...`) resolution algorithms.
//!
//! Every assertion is transcribed from the SQLite documentation, never from what
//! the engine currently returns — a failing case is the intended signal that the
//! engine diverges from the spec. A case that reveals an engine bug is left as a
//! genuine failing assertion rather than weakened to pass.
//!
//! Spec sources:
//! - `spec/sqlite-doc/lang_insert.html`: the three INSERT forms. (1) `VALUES` —
//!   with no column list the number of values must equal the number of table
//!   columns; with a column list the number of values must match the listed
//!   columns and any omitted column takes its `DEFAULT` (or NULL if none). (2)
//!   `INSERT ... SELECT` — one new row per SELECT result row. (3) `DEFAULT VALUES`
//!   — one row, every column its default or NULL. The leading `INSERT` keyword may
//!   be replaced by `REPLACE` or `INSERT OR <action>`; bare `REPLACE` is an alias
//!   for `INSERT OR REPLACE`.
//! - `spec/sqlite-doc/lang_conflict.html`: the five algorithms ROLLBACK, ABORT,
//!   FAIL, IGNORE, REPLACE (default ABORT). ABORT raises `SQLITE_CONSTRAINT` and
//!   backs out the whole failing statement (prior statements preserved). FAIL
//!   raises `SQLITE_CONSTRAINT` but keeps rows the failed statement wrote before
//!   the conflict. IGNORE skips only the offending row and returns no error for
//!   UNIQUE / NOT NULL / PRIMARY KEY violations. REPLACE deletes the pre-existing
//!   conflicting row(s) then inserts, leaving the row count unchanged. ROLLBACK
//!   raises `SQLITE_CONSTRAINT` and rolls back the active transaction, or behaves
//!   like ABORT when no transaction is active.
//! - `spec/sqlite-doc/lang_returning.html`: `RETURNING` yields one result row per
//!   inserted row; its expressions are like SELECT columns (an `AS` names the
//!   column, `*` expands to all table columns); for INSERT the column references
//!   are the values *after* the change. The output row order is explicitly
//!   arbitrary, so multi-row RETURNING is compared as a multiset.
//! - `spec/sqlite-doc/lang_createtable.html#rowid` and `autoinc.html`: a single
//!   PRIMARY KEY column whose declared type is exactly `INTEGER` is an alias for
//!   the rowid. Inserting NULL (or omitting the column) auto-assigns a rowid one
//!   larger than the largest rowid currently in the table (1 for an empty table).
//!
//! Each case is its own `#[test]` so an unsupported feature fails exactly that
//! case rather than masking the rest, and every INSERT's effect is confirmed with
//! a follow-up `SELECT ... ORDER BY` (deterministic order).

mod conformance;
use conformance::*;

// `Connection` and `Error` are NOT re-exported through `conformance::*` (the
// harness imports them privately), so pull them in directly. `Error` is needed to
// assert a constraint violation is specifically `Error::Constraint` (SQLITE_CONSTRAINT),
// not merely some error; `Connection` types the shared helper below.
use minisqlite::{Connection, Error};

/// Assert that executing `sql` fails with a *constraint* error specifically.
///
/// `lang_conflict.html`: a constraint violation under the default ABORT algorithm
/// aborts the statement with an `SQLITE_CONSTRAINT` error. The engine models that
/// as `Error::Constraint`; a `Sql`/`Io`/`Format` error here would be the wrong
/// *kind*, so the match is on the variant.
fn assert_constraint_error(db: &mut Connection, sql: &str) {
    let e = assert_exec_error(db, sql);
    assert!(
        matches!(e, Error::Constraint(_)),
        "expected a Constraint error (SQLITE_CONSTRAINT)\n  sql: {sql}\n  actual: {e:?}"
    );
}

/// Assert that executing `sql` fails specifically with an ordinary SQL error
/// (`SQLITE_ERROR` / `Error::Sql`).
///
/// SQLite raises `SQLITE_ERROR` (primary code 1) for a *malformed* statement — a
/// value-count mismatch, an unknown column, a missing table — as distinct from a
/// constraint violation (`SQLITE_CONSTRAINT`). Pinning the variant catches a
/// wrong-*kind* regression (e.g. a missing table reported as a constraint error).
fn assert_sql_error(db: &mut Connection, sql: &str) {
    let e = assert_exec_error(db, sql);
    assert!(
        matches!(e, Error::Sql(_)),
        "expected a SQL error (SQLITE_ERROR)\n  sql: {sql}\n  actual: {e:?}"
    );
}

// ---------------------------------------------------------------------------
// Form 1: INSERT ... VALUES — lang_insert.html.
// ---------------------------------------------------------------------------

#[test]
fn insert_single_row_values() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
    exec(&mut db, "INSERT INTO t VALUES (1,'x')");
    assert_rows(&mut db, "SELECT a,b FROM t ORDER BY a", &[vec![int(1), text("x")]]);
}

#[test]
fn insert_multi_row_values() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
    exec(&mut db, "INSERT INTO t VALUES (1,'x')");
    exec(&mut db, "INSERT INTO t VALUES (2,'y'),(3,'z')");
    assert_rows(
        &mut db,
        "SELECT a,b FROM t ORDER BY a",
        &[
            vec![int(1), text("x")],
            vec![int(2), text("y")],
            vec![int(3), text("z")],
        ],
    );
}

#[test]
fn insert_values_are_evaluated_expressions() {
    // Each term of the VALUES list is an expression, evaluated before storage.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1+1, 3*4)");
    assert_rows(&mut db, "SELECT a,b FROM t", &[vec![int(2), int(12)]]);
}

// ---------------------------------------------------------------------------
// Column lists: explicit / reordered / partial — lang_insert.html.
// "Each of the named columns of the new row is populated with the results of
// evaluating the corresponding VALUES expression. Table columns that do not
// appear in the column list are populated with the default column value ... or
// with NULL if no default value is specified."
// ---------------------------------------------------------------------------

#[test]
fn insert_explicit_column_list_reordered() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
    // Column list (b,a) means the values map by name, not position.
    exec(&mut db, "INSERT INTO t(b,a) VALUES ('q',4)");
    assert_rows(&mut db, "SELECT a,b FROM t WHERE a=4", &[vec![int(4), text("q")]]);
}

#[test]
fn insert_partial_column_list_fills_null() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE u(a,b,c)");
    exec(&mut db, "INSERT INTO u(a) VALUES(1)");
    // b and c have no default, so they are NULL.
    assert_rows(&mut db, "SELECT a,b,c FROM u", &[vec![int(1), null(), null()]]);
}

#[test]
fn insert_partial_column_list_uses_default() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE d(a INTEGER, b TEXT DEFAULT 'zz')");
    exec(&mut db, "INSERT INTO d(a) VALUES(1)");
    // b is omitted, so it takes its declared DEFAULT.
    assert_rows(&mut db, "SELECT a,b FROM d", &[vec![int(1), text("zz")]]);
}

// ---------------------------------------------------------------------------
// Form 3: INSERT ... DEFAULT VALUES — lang_insert.html.
// "Each column of the new row is populated with its default value, or with a
// NULL if no default value is specified."
// ---------------------------------------------------------------------------

#[test]
fn insert_default_values_uses_declared_defaults() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE d(a INTEGER DEFAULT 5, b TEXT DEFAULT 'z')");
    exec(&mut db, "INSERT INTO d DEFAULT VALUES");
    assert_rows(&mut db, "SELECT a,b FROM d", &[vec![int(5), text("z")]]);
}

#[test]
fn insert_default_values_without_defaults_are_null() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE d(a, b)");
    exec(&mut db, "INSERT INTO d DEFAULT VALUES");
    assert_rows(&mut db, "SELECT a,b FROM d", &[vec![null(), null()]]);
}

// ---------------------------------------------------------------------------
// Form 2: INSERT ... SELECT — lang_insert.html.
// "A new entry is inserted into the table for each row of data returned by
// executing the SELECT statement."
// ---------------------------------------------------------------------------

#[test]
fn insert_select_from_other_table() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE src(a INTEGER, b TEXT)");
    exec(&mut db, "INSERT INTO src VALUES(1,'x'),(2,'y')");
    exec(&mut db, "CREATE TABLE dst(a INTEGER, b TEXT)");
    exec(&mut db, "INSERT INTO dst SELECT a, b FROM src");
    assert_rows(
        &mut db,
        "SELECT a,b FROM dst ORDER BY a",
        &[vec![int(1), text("x")], vec![int(2), text("y")]],
    );
}

#[test]
fn insert_select_with_reordering_column_list() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE src(x INTEGER, y TEXT)");
    exec(&mut db, "INSERT INTO src VALUES(1,'a'),(2,'b')");
    exec(&mut db, "CREATE TABLE dst(a INTEGER, b TEXT)");
    // Column list (b,a) maps SELECT's first result column to b, second to a.
    exec(&mut db, "INSERT INTO dst(b,a) SELECT y, x FROM src");
    assert_rows(
        &mut db,
        "SELECT a,b FROM dst ORDER BY a",
        &[vec![int(1), text("a")], vec![int(2), text("b")]],
    );
}

#[test]
fn insert_select_from_same_table() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
    exec(&mut db, "INSERT INTO t VALUES(1,'x'),(2,'y'),(3,'z')");
    // Same-table INSERT..SELECT with a transform: the pre-existing rows a<3 (a=1,2)
    // produce (11,'x') and (12,'y'). (WHERE a<3 also excludes the new rows, so this
    // case does not by itself pin the no-self-feedback guarantee — the dedicated
    // self-copy test below does that.)
    exec(&mut db, "INSERT INTO t SELECT a+10, b FROM t WHERE a<3");
    assert_rows(
        &mut db,
        "SELECT a,b FROM t ORDER BY a",
        &[
            vec![int(1), text("x")],
            vec![int(2), text("y")],
            vec![int(3), text("z")],
            vec![int(11), text("x")],
            vec![int(12), text("y")],
        ],
    );
}

#[test]
fn insert_select_self_copy_reads_only_original_rows() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES(1),(2),(3)");
    // The SELECT reads the SAME table the INSERT writes, with NO predicate that
    // would exclude a fed-back row. A correct engine reads only the 3 pre-existing
    // rows (no "Halloween problem" where a row inserted by this statement is
    // re-read and re-inserted), so the table ends with exactly 6 rows — never more,
    // and the statement must terminate rather than loop.
    exec(&mut db, "INSERT INTO t SELECT a FROM t");
    assert_rows(
        &mut db,
        "SELECT a FROM t ORDER BY a",
        &[
            vec![int(1)],
            vec![int(1)],
            vec![int(2)],
            vec![int(2)],
            vec![int(3)],
            vec![int(3)],
        ],
    );
}

// ---------------------------------------------------------------------------
// RETURNING — lang_returning.html.
// One result row per inserted row; expressions like SELECT columns; `*` expands
// to all columns; for INSERT the values are those *after* the change.
// ---------------------------------------------------------------------------

#[test]
fn insert_returning_named_columns() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE r(id INTEGER PRIMARY KEY, v TEXT)");
    // id is auto-assigned to 1; RETURNING reports the post-insert values.
    assert_rows(
        &mut db,
        "INSERT INTO r(v) VALUES('a') RETURNING id, v",
        &[vec![int(1), text("a")]],
    );
    // Effect persisted.
    assert_rows(&mut db, "SELECT id,v FROM r ORDER BY id", &[vec![int(1), text("a")]]);
}

#[test]
fn insert_returning_star_expands_all_columns() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE r(id INTEGER PRIMARY KEY, v TEXT)");
    // `*` expands to (id, v) in table order.
    assert_rows(
        &mut db,
        "INSERT INTO r(v) VALUES('a') RETURNING *",
        &[vec![int(1), text("a")]],
    );
    assert_rows(&mut db, "SELECT id,v FROM r ORDER BY id", &[vec![int(1), text("a")]]);
}

#[test]
fn insert_returning_reports_column_names() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE r(id INTEGER PRIMARY KEY, v TEXT)");
    // Single query call (INSERT RETURNING is not idempotent, so don't re-run it):
    // check the reported column names off the same result.
    let qr = query(&mut db, "INSERT INTO r(v) VALUES('a') RETURNING id, v");
    let cols: Vec<&str> = qr.columns.iter().map(String::as_str).collect();
    assert_eq!(cols, ["id", "v"], "RETURNING column names follow the projected columns");
    // RETURNING must also persist the row, not merely project it.
    assert_rows(&mut db, "SELECT id,v FROM r", &[vec![int(1), text("a")]]);
}

#[test]
fn insert_returning_expression_and_alias() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE r(id INTEGER PRIMARY KEY, v TEXT)");
    let qr = query(
        &mut db,
        "INSERT INTO r(v) VALUES('a') RETURNING id AS pk, v || '!' AS vv",
    );
    assert_eq!(qr.rows.len(), 1, "one inserted row -> one RETURNING row");
    // Pin the row width before indexing [0][1], so a wrong shape fails with a
    // legible assertion rather than an index-out-of-bounds panic.
    assert_eq!(qr.rows[0].len(), 2, "two RETURNING expressions -> two columns");
    assert!(value_eq(&qr.rows[0][0], &int(1)), "pk = post-insert id");
    assert!(value_eq(&qr.rows[0][1], &text("a!")), "vv = v with '!' appended");
    let cols: Vec<&str> = qr.columns.iter().map(String::as_str).collect();
    assert_eq!(cols, ["pk", "vv"], "AS clauses name the RETURNING columns");
    // The insert is persisted with the stored (not the projected-expression) value.
    assert_rows(&mut db, "SELECT id,v FROM r", &[vec![int(1), text("a")]]);
}

#[test]
fn insert_multi_row_returning_reports_every_row() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE r(id INTEGER PRIMARY KEY, v TEXT)");
    // RETURNING row order is spec-arbitrary, so compare as a multiset.
    assert_rows_unordered(
        &mut db,
        "INSERT INTO r(v) VALUES('a'),('b') RETURNING id, v",
        &[vec![int(1), text("a")], vec![int(2), text("b")]],
    );
    assert_rows(
        &mut db,
        "SELECT id,v FROM r ORDER BY id",
        &[vec![int(1), text("a")], vec![int(2), text("b")]],
    );
}

// ---------------------------------------------------------------------------
// rowid / INTEGER PRIMARY KEY auto-assignment —
// lang_createtable.html#rowid, autoinc.html.
// ---------------------------------------------------------------------------

#[test]
fn insert_integer_primary_key_autoassigns_sequential() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE k(id INTEGER PRIMARY KEY, v)");
    exec(&mut db, "INSERT INTO k(v) VALUES('a'),('b')");
    // Empty table starts at rowid 1, then largest+1.
    assert_rows(
        &mut db,
        "SELECT id,v FROM k ORDER BY id",
        &[vec![int(1), text("a")], vec![int(2), text("b")]],
    );
}

#[test]
fn insert_explicit_null_id_autoassigns() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE k(id INTEGER PRIMARY KEY, v)");
    // An explicit NULL for the rowid alias is auto-assigned, just like omitting it.
    exec(&mut db, "INSERT INTO k(id, v) VALUES(NULL, 'a')");
    assert_rows(&mut db, "SELECT id,v FROM k ORDER BY id", &[vec![int(1), text("a")]]);
}

#[test]
fn insert_explicit_integer_primary_key_value_is_stored() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE k(id INTEGER PRIMARY KEY, v)");
    exec(&mut db, "INSERT INTO k VALUES(42, 'a')");
    assert_rows(&mut db, "SELECT id,v FROM k ORDER BY id", &[vec![int(42), text("a")]]);
}

#[test]
fn insert_explicit_id_then_null_continues_from_max_plus_one() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE k(id INTEGER PRIMARY KEY, v)");
    exec(&mut db, "INSERT INTO k VALUES(10,'x')");
    // Auto rowid is one larger than the largest currently in use (10) -> 11.
    exec(&mut db, "INSERT INTO k(v) VALUES('y')");
    assert_rows(
        &mut db,
        "SELECT id,v FROM k ORDER BY id",
        &[vec![int(10), text("x")], vec![int(11), text("y")]],
    );
}

#[test]
fn insert_rowid_autoassigned_for_plain_table() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(v TEXT)");
    exec(&mut db, "INSERT INTO t VALUES('a'),('b')");
    // Every rowid table has an implicit rowid accessible by that name, starting at 1.
    assert_rows(
        &mut db,
        "SELECT rowid, v FROM t ORDER BY rowid",
        &[vec![int(1), text("a")], vec![int(2), text("b")]],
    );
}

#[test]
fn insert_non_integer_text_into_integer_primary_key_is_error() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE k(id INTEGER PRIMARY KEY, v)");
    // lang_createtable.html#rowid: an INTEGER PRIMARY KEY (rowid alias) can only
    // hold integers. A TEXT value that cannot be losslessly converted to an integer
    // is a "datatype mismatch" and the statement is aborted. (The facade Error has
    // no dedicated MISMATCH variant, so only that it errors is asserted.)
    let _ = assert_exec_error(&mut db, "INSERT INTO k VALUES('notint','a')");
    assert_scalar(&mut db, "SELECT count(*) FROM k", int(0));
}

// ---------------------------------------------------------------------------
// Constraint violations under the default (ABORT) algorithm — the returned
// Error must be a Constraint kind. lang_conflict.html / lang_createtable.html.
// ---------------------------------------------------------------------------

#[test]
fn insert_unique_violation_is_constraint_error() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t2(a UNIQUE)");
    exec(&mut db, "INSERT INTO t2 VALUES(1)");
    assert_constraint_error(&mut db, "INSERT INTO t2 VALUES(1)");
    // The failed insert changed nothing.
    assert_scalar(&mut db, "SELECT count(*) FROM t2", int(1));
}

#[test]
fn insert_not_null_violation_is_constraint_error() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t3(a NOT NULL)");
    assert_constraint_error(&mut db, "INSERT INTO t3 VALUES(NULL)");
    assert_scalar(&mut db, "SELECT count(*) FROM t3", int(0));
}

#[test]
fn insert_primary_key_dup_is_constraint_error() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE k(id INTEGER PRIMARY KEY, v)");
    exec(&mut db, "INSERT INTO k VALUES(1,'a')");
    assert_constraint_error(&mut db, "INSERT INTO k VALUES(1,'dup')");
    // Original row intact, no duplicate.
    assert_rows(&mut db, "SELECT id,v FROM k ORDER BY id", &[vec![int(1), text("a")]]);
}

#[test]
fn insert_text_primary_key_dup_is_constraint_error() {
    // A non-integer PRIMARY KEY is a unique index, not a rowid alias, but a
    // duplicate is still a PRIMARY KEY constraint violation.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a TEXT PRIMARY KEY, b)");
    exec(&mut db, "INSERT INTO t VALUES('x', 1)");
    assert_constraint_error(&mut db, "INSERT INTO t VALUES('x', 2)");
    assert_rows(&mut db, "SELECT a,b FROM t", &[vec![text("x"), int(1)]]);
}

// ---------------------------------------------------------------------------
// Malformed INSERT: value-count and schema errors — lang_insert.html. These are
// ordinary SQL errors (SQLITE_ERROR / Error::Sql), not constraint violations. In
// addition to the error kind, each case confirms the failed statement left NO
// partial row behind (an INSERT that errors must be atomic).
// ---------------------------------------------------------------------------

#[test]
fn insert_too_few_values_is_error() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a,b)");
    // 2 columns, 1 value, no column list -> counts must match.
    assert_sql_error(&mut db, "INSERT INTO t VALUES(1)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
}

#[test]
fn insert_too_many_values_is_error() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a,b)");
    assert_sql_error(&mut db, "INSERT INTO t VALUES(1,2,3)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
}

#[test]
fn insert_column_list_count_mismatch_is_error() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a,b)");
    // 2 named columns, 1 value.
    assert_sql_error(&mut db, "INSERT INTO t(a,b) VALUES(1)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
}

#[test]
fn insert_unknown_column_is_error() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a,b)");
    assert_sql_error(&mut db, "INSERT INTO t(a,zzz) VALUES(1,2)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
}

#[test]
fn insert_into_missing_table_is_error() {
    let mut db = mem();
    // No such table -> SQLITE_ERROR. (No count check: the table does not exist.)
    assert_sql_error(&mut db, "INSERT INTO nope VALUES(1)");
}

// ---------------------------------------------------------------------------
// Conflict clauses: INSERT OR IGNORE — lang_conflict.html.
// "the IGNORE resolution algorithm skips the one row that contains the
// constraint violation and continues processing subsequent rows ... No error is
// returned for uniqueness, NOT NULL, and UNIQUE constraint errors."
// ---------------------------------------------------------------------------

#[test]
fn insert_or_ignore_skips_conflicting_row_without_error() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a UNIQUE)");
    exec(&mut db, "INSERT INTO t VALUES(1)");
    // No error, and the conflicting row is silently skipped (count unchanged).
    exec(&mut db, "INSERT OR IGNORE INTO t VALUES(1)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(1));
}

#[test]
fn insert_or_ignore_keeps_other_rows_in_batch() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a UNIQUE)");
    exec(&mut db, "INSERT INTO t VALUES(1)");
    // Of (2),(1),(3): 1 conflicts and is skipped; 2 and 3 are inserted normally.
    exec(&mut db, "INSERT OR IGNORE INTO t VALUES(2),(1),(3)");
    assert_rows(
        &mut db,
        "SELECT a FROM t ORDER BY a",
        &[vec![int(1)], vec![int(2)], vec![int(3)]],
    );
}

#[test]
fn insert_or_ignore_skips_not_null_violation() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a NOT NULL)");
    // IGNORE returns no error for a NOT NULL violation; the row is skipped.
    exec(&mut db, "INSERT OR IGNORE INTO t VALUES(NULL)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
}

#[test]
fn insert_or_ignore_skips_check_violation() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x INTEGER CHECK(x > 0))");
    // A CHECK participates in the statement-level conflict algorithm exactly like NOT NULL
    // and UNIQUE: the OR clause overrides the CREATE TABLE default (ABORT) — lang_conflict
    // .html, "The algorithm specified in the OR clause of an INSERT or UPDATE overrides any
    // algorithm specified in a CREATE TABLE" (its FAIL entry lists CHECK among the governed
    // constraints). So OR IGNORE skips the one CHECK-violating row (-1) with NO error and
    // inserts the surrounding rows (5, 10); it must NOT abort + roll the whole batch back.
    exec(&mut db, "INSERT OR IGNORE INTO t VALUES(5),(-1),(10)");
    assert_rows(&mut db, "SELECT x FROM t ORDER BY x", &[vec![int(5)], vec![int(10)]]);
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2));
}

#[test]
fn insert_or_ignore_skips_primary_key_conflict() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE k(id INTEGER PRIMARY KEY, v TEXT)");
    exec(&mut db, "INSERT INTO k VALUES(1,'a')");
    // A PRIMARY KEY (rowid) conflict is skipped just like a UNIQUE one; the
    // pre-existing row is left untouched (not overwritten, unlike REPLACE).
    exec(&mut db, "INSERT OR IGNORE INTO k VALUES(1,'b')");
    assert_rows(&mut db, "SELECT id,v FROM k ORDER BY id", &[vec![int(1), text("a")]]);
    assert_scalar(&mut db, "SELECT count(*) FROM k", int(1));
}

// ---------------------------------------------------------------------------
// Conflict clauses: INSERT OR REPLACE / REPLACE — lang_conflict.html,
// lang_insert.html. On a UNIQUE/PRIMARY KEY conflict, REPLACE deletes the
// pre-existing conflicting row then inserts the new one; the row count is
// unchanged. Bare REPLACE is an alias for INSERT OR REPLACE.
// ---------------------------------------------------------------------------

#[test]
fn insert_or_replace_replaces_conflicting_row() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY, v)");
    exec(&mut db, "INSERT INTO p VALUES(1,'a')");
    exec(&mut db, "INSERT OR REPLACE INTO p VALUES(1,'b')");
    assert_rows(&mut db, "SELECT v FROM p WHERE id=1", &[vec![text("b")]]);
    assert_scalar(&mut db, "SELECT count(*) FROM p", int(1));
}

#[test]
fn replace_into_is_alias_for_insert_or_replace() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY, v)");
    exec(&mut db, "INSERT INTO p VALUES(1,'a')");
    exec(&mut db, "REPLACE INTO p VALUES(1,'b')");
    assert_rows(&mut db, "SELECT v FROM p WHERE id=1", &[vec![text("b")]]);
    assert_scalar(&mut db, "SELECT count(*) FROM p", int(1));
}

#[test]
fn insert_or_replace_on_unique_index() {
    // REPLACE also resolves a plain UNIQUE conflict, not only PRIMARY KEY.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a UNIQUE, b TEXT)");
    exec(&mut db, "INSERT INTO t VALUES(1,'a')");
    exec(&mut db, "INSERT OR REPLACE INTO t VALUES(1,'b')");
    assert_rows(&mut db, "SELECT a,b FROM t", &[vec![int(1), text("b")]]);
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(1));
}

#[test]
fn insert_or_replace_returning_reports_inserted_row() {
    // RETURNING reports rows directly inserted; the REPLACE-deleted old row is a
    // conflict-resolution side effect and is not reported (lang_returning.html:
    // "only returns rows that are directly modified by the ... INSERT").
    let mut db = mem();
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY, v TEXT)");
    exec(&mut db, "INSERT INTO p VALUES(1,'a')");
    assert_rows(
        &mut db,
        "INSERT OR REPLACE INTO p VALUES(1,'b') RETURNING id, v",
        &[vec![int(1), text("b")]],
    );
    assert_rows(&mut db, "SELECT id,v FROM p", &[vec![int(1), text("b")]]);
}

// ---------------------------------------------------------------------------
// Conflict clauses: INSERT OR ABORT / FAIL / ROLLBACK — lang_conflict.html.
// These distinguish how much of the failing statement (and transaction) is
// undone.
// ---------------------------------------------------------------------------

#[test]
fn insert_or_abort_errors_on_conflict() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a UNIQUE)");
    exec(&mut db, "INSERT INTO t VALUES(1)");
    assert_constraint_error(&mut db, "INSERT OR ABORT INTO t VALUES(1)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(1));
}

#[test]
fn insert_or_abort_backs_out_whole_statement() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a UNIQUE)");
    exec(&mut db, "INSERT INTO t VALUES(1)");
    // Row (2) is written, then (1) conflicts. ABORT (the default) backs out ALL
    // changes made by this statement, so (2) is discarded too.
    assert_constraint_error(&mut db, "INSERT INTO t VALUES(2),(1)");
    assert_rows(&mut db, "SELECT a FROM t ORDER BY a", &[vec![int(1)]]);
}

#[test]
fn insert_or_fail_keeps_rows_before_the_conflict() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a UNIQUE)");
    exec(&mut db, "INSERT INTO t VALUES(1)");
    // Row (2) is written, then (1) conflicts. FAIL does NOT back out prior rows
    // of the failed statement, so (2) survives while the statement still errors.
    assert_constraint_error(&mut db, "INSERT OR FAIL INTO t VALUES(2),(1)");
    assert_rows(&mut db, "SELECT a FROM t ORDER BY a", &[vec![int(1)], vec![int(2)]]);
}

#[test]
fn insert_or_rollback_without_transaction_acts_like_abort() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a UNIQUE)");
    exec(&mut db, "INSERT INTO t VALUES(1)");
    // No active transaction, so ROLLBACK degrades to ABORT: whole statement undone.
    assert_constraint_error(&mut db, "INSERT OR ROLLBACK INTO t VALUES(2),(1)");
    assert_rows(&mut db, "SELECT a FROM t ORDER BY a", &[vec![int(1)]]);
}

#[test]
fn insert_or_rollback_rolls_back_active_transaction() {
    // ROLLBACK's defining behavior: on conflict it rolls back the whole active
    // transaction, discarding changes made by earlier statements in that
    // transaction (depends on BEGIN/ROLLBACK transaction support).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a UNIQUE)");
    exec(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "BEGIN");
    exec(&mut db, "INSERT INTO t VALUES(2)");
    assert_constraint_error(&mut db, "INSERT OR ROLLBACK INTO t VALUES(1)");
    // The transaction (including the (2) row) is rolled back; only {1} remains.
    assert_rows(&mut db, "SELECT a FROM t ORDER BY a", &[vec![int(1)]]);
}
