//! Conformance battery: **column and table CONSTRAINTS** — NOT NULL, UNIQUE,
//! PRIMARY KEY, and CHECK — under the default (ABORT) conflict algorithm.
//!
//! Every expected result here is transcribed from the SQLite documentation in
//! `spec/sqlite-doc/`, not from what this engine returns; a case that reveals an
//! engine bug is left as a genuine failing assertion rather than weakened to
//! pass.
//!
//! This file exists to close a real coverage blind spot: CHECK constraints have
//! no coverage elsewhere in `crates/minisqlite/tests/`, and the NULL-quirk
//! semantics of every constraint class (NULLs distinct for UNIQUE/PRIMARY KEY,
//! CHECK passing on NULL, the PRIMARY-KEY-allows-NULL bug-compat behavior) are
//! not pinned systematically. It complements — and does not duplicate — the
//! conflict-clause coverage (`INSERT OR IGNORE/REPLACE/FAIL/ROLLBACK`) already in
//! `conformance_dml_insert.rs`; here we only exercise the constraints themselves
//! under the default behavior.
//!
//! Spec sources (all under `spec/sqlite-doc/`):
//!   * `lang_createtable.html` §3.5 "The PRIMARY KEY" — each row must have a
//!     unique combination of primary-key values; for uniqueness "NULL values are
//!     considered distinct from all other values, including other NULLs". QUIRK:
//!     "According to the SQL standard, PRIMARY KEY should always imply NOT NULL.
//!     Unfortunately, due to a bug in some early versions, this is not the case
//!     in SQLite. Unless the column is an INTEGER PRIMARY KEY or the table is a
//!     WITHOUT ROWID table or a STRICT table or the column is declared NOT NULL,
//!     SQLite allows NULL values in a PRIMARY KEY column."
//!   * `lang_createtable.html` §3.6 "UNIQUE constraints" — each row must hold a
//!     unique combination of the UNIQUE columns; "For the purposes of UNIQUE
//!     constraints, NULL values are considered distinct from all other values,
//!     including other NULLs" (so a UNIQUE column admits many NULLs).
//!   * `lang_createtable.html` §3.7 "CHECK constraints" — the expression "is
//!     evaluated and cast to a NUMERIC value in the same way as a CAST
//!     expression. If the result is zero (integer value 0 or real value 0.0),
//!     then a constraint violation has occurred. If the CHECK expression
//!     evaluates to NULL, or any other non-zero value, it is not a constraint
//!     violation." Verified on write (INSERT/UPDATE), not on read.
//!   * `lang_createtable.html` §3.8 "NOT NULL constraints" — "a NOT NULL
//!     constraint dictates that the associated column may not contain a NULL
//!     value. Attempting to set the column value to NULL when inserting a new row
//!     or updating an existing one causes a constraint violation."
//!   * `lang_createtable.html` §4.1 "Response to constraint violations" — with no
//!     explicit conflict-clause the default algorithm is ABORT; "The conflict
//!     resolution algorithm for CHECK constraints is always ABORT."
//!   * `lang_createtable.html` §5 "ROWIDs and the INTEGER PRIMARY KEY" — a lone
//!     `INTEGER PRIMARY KEY` column aliases the rowid; "If an INSERT statement
//!     attempts to insert a NULL value into a rowid or integer primary key
//!     column, the system chooses an integer value to use as the rowid
//!     automatically."
//!   * `autoinc.html` — the auto-assigned rowid for an initially empty table is 1.
//!   * `lang_conflict.html` (ABORT) — "the ABORT resolution algorithm aborts the
//!     current SQL statement with an SQLITE_CONSTRAINT error and backs out any
//!     changes made by the current SQL statement; but changes caused by prior
//!     SQL statements within the same transaction are preserved". This is why
//!     every violation test below confirms the table is UNCHANGED afterwards.
//!
//! Testing method: the primary bar for a violation is that the
//! statement ERRORS *and* the table is left exactly as before (a follow-up
//! `SELECT`), which is what the ABORT algorithm guarantees. A dedicated
//! "error classification" section additionally pins that the error is
//! specifically `Error::Constraint(_)` (SQLITE_CONSTRAINT) for one representative
//! case per class.
//! Each case is its own small `#[test]` so an unenforced constraint fails exactly
//! that case rather than masking the rest.

mod conformance;
use conformance::*;

// `Error` is not re-exported through `conformance::*` (the harness imports it
// privately), so pull it in directly for the error-classification section, which
// pins that a violation is specifically `Error::Constraint(_)`.
use minisqlite::Error;

/// Assert a returned error is specifically the constraint class
/// (`SQLITE_CONSTRAINT`). Used only by the classification section; the behavioral
/// sections rely on the more robust "errors + data unchanged" bar so that a
/// merely-mis-tagged (but correctly enforced) violation is still visible as
/// enforced.
fn assert_is_constraint(e: Error, sql: &str) {
    assert!(
        matches!(e, Error::Constraint(_)),
        "expected Error::Constraint (SQLITE_CONSTRAINT)\n  sql: {sql}\n  actual: {e:?}"
    );
}

// ===========================================================================
// NOT NULL — lang_createtable.html §3.8.
// "a NOT NULL constraint dictates that the associated column may not contain a
// NULL value. Attempting to set the column value to NULL when inserting a new
// row or updating an existing one causes a constraint violation."
// ===========================================================================

#[test]
fn not_null_rejects_null_insert() {
    // §3.8: inserting NULL into a NOT NULL column is a constraint violation; under
    // the default ABORT the statement errors and writes nothing.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INT NOT NULL)");
    assert_exec_error(&mut db, "INSERT INTO t VALUES(NULL)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
}

#[test]
fn not_null_accepts_non_null_insert() {
    // §3.8: a non-NULL value satisfies the constraint and is stored.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INT NOT NULL)");
    exec(&mut db, "INSERT INTO t VALUES(5)");
    assert_rows(&mut db, "SELECT a FROM t", &[vec![int(5)]]);
}

#[test]
fn not_null_enforced_on_update() {
    // §3.8: NOT NULL is enforced on UPDATE too ("updating an existing one"). The
    // UPDATE errors and the pre-existing row is left unchanged (ABORT).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INT NOT NULL)");
    exec(&mut db, "INSERT INTO t VALUES(5)");
    assert_exec_error(&mut db, "UPDATE t SET a=NULL");
    assert_rows(&mut db, "SELECT a FROM t", &[vec![int(5)]]);
}

#[test]
fn not_null_update_to_valid_value_succeeds() {
    // §3.8: an UPDATE to another non-NULL value is not a violation.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INT NOT NULL)");
    exec(&mut db, "INSERT INTO t VALUES(5)");
    exec(&mut db, "UPDATE t SET a=9");
    assert_rows(&mut db, "SELECT a FROM t", &[vec![int(9)]]);
}

#[test]
fn not_null_column_with_default_supplies_value() {
    // §3.8 + §3.2: a NOT NULL column WITH a DEFAULT is satisfied by the default
    // when the INSERT omits the column, so DEFAULT VALUES stores 7 (no error).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INT NOT NULL DEFAULT 7)");
    exec(&mut db, "INSERT INTO t DEFAULT VALUES");
    assert_rows(&mut db, "SELECT a FROM t", &[vec![int(7)]]);
}

#[test]
fn not_null_default_values_without_default_is_error() {
    // §3.8: with no DEFAULT, an omitted NOT NULL column would take NULL, which
    // violates the constraint — so INSERT ... DEFAULT VALUES errors and stores
    // nothing.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INT NOT NULL)");
    assert_exec_error(&mut db, "INSERT INTO t DEFAULT VALUES");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
}

#[test]
fn not_null_is_per_column() {
    // §3.8: NOT NULL applies only to the column it decorates. `a` is NOT NULL, `b`
    // is nullable: (NULL,1) violates, (1,NULL) is fine.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INT NOT NULL, b INT)");
    assert_exec_error(&mut db, "INSERT INTO t VALUES(NULL, 1)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
    exec(&mut db, "INSERT INTO t VALUES(1, NULL)");
    assert_rows(&mut db, "SELECT a, b FROM t", &[vec![int(1), null()]]);
}

// ===========================================================================
// UNIQUE — lang_createtable.html §3.6.
// "each row must contain a unique combination of values in the columns
// identified by the UNIQUE constraint. For the purposes of UNIQUE constraints,
// NULL values are considered distinct from all other values, including other
// NULLs."
// ===========================================================================

#[test]
fn unique_column_rejects_duplicate() {
    // §3.6: a second identical value is a constraint violation; ABORT keeps the
    // table at its pre-insert contents. Pin the surviving *value*, not just the
    // count, so an engine that wrongly kept the second row (or mangled the first)
    // is caught, not only one that changed the row count.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a UNIQUE)");
    exec(&mut db, "INSERT INTO t VALUES(1)");
    assert_exec_error(&mut db, "INSERT INTO t VALUES(1)");
    assert_rows(&mut db, "SELECT a FROM t", &[vec![int(1)]]);
}

#[test]
fn unique_column_accepts_distinct_values() {
    // §3.6: distinct values are not a violation.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a UNIQUE)");
    exec(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "INSERT INTO t VALUES(2)");
    assert_rows_unordered(&mut db, "SELECT a FROM t", &[vec![int(1)], vec![int(2)]]);
}

#[test]
fn unique_allows_multiple_nulls() {
    // §3.6: "NULL values are considered distinct from all other values, including
    // other NULLs" — two NULLs in a UNIQUE column are NOT a violation, so both
    // rows survive.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a UNIQUE)");
    exec(&mut db, "INSERT INTO t VALUES(NULL)");
    exec(&mut db, "INSERT INTO t VALUES(NULL)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2));
    assert_rows_unordered(&mut db, "SELECT a FROM t", &[vec![null()], vec![null()]]);
}

#[test]
fn unique_multi_column_table_constraint() {
    // §3.6: a table-level UNIQUE(a,b) constrains the *combination*. (1,2) then a
    // duplicate (1,2) violates, but (1,3) shares only one component and is fine.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b, UNIQUE(a, b))");
    exec(&mut db, "INSERT INTO t VALUES(1, 2)");
    assert_exec_error(&mut db, "INSERT INTO t VALUES(1, 2)");
    exec(&mut db, "INSERT INTO t VALUES(1, 3)");
    assert_rows_unordered(
        &mut db,
        "SELECT a, b FROM t",
        &[vec![int(1), int(2)], vec![int(1), int(3)]],
    );
}

#[test]
fn unique_multi_column_allows_null_component_duplicates() {
    // §3.6: because NULLs are distinct, two rows (1,NULL) and (1,NULL) are
    // distinct *combinations* for a composite UNIQUE(a,b) and both are allowed.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b, UNIQUE(a, b))");
    exec(&mut db, "INSERT INTO t VALUES(1, NULL)");
    exec(&mut db, "INSERT INTO t VALUES(1, NULL)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2));
}

#[test]
fn unique_update_creating_duplicate_is_error() {
    // §3.6 + §4.1: an UPDATE that would collide with an existing value violates
    // UNIQUE; ABORT leaves the updated row unchanged.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a UNIQUE)");
    exec(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "INSERT INTO t VALUES(2)");
    assert_exec_error(&mut db, "UPDATE t SET a=1 WHERE a=2");
    assert_rows_unordered(&mut db, "SELECT a FROM t", &[vec![int(1)], vec![int(2)]]);
}

// ===========================================================================
// PRIMARY KEY — lang_createtable.html §3.5, §5.
// Uniqueness with NULLs distinct; the documented SQLite quirk that a non-INTEGER
// PRIMARY KEY column *allows* NULLs (PRIMARY KEY does not imply NOT NULL); and
// the INTEGER PRIMARY KEY exception that a NULL is auto-assigned a rowid.
// ===========================================================================

#[test]
fn integer_primary_key_rejects_explicit_duplicate() {
    // §3.5/§5: an explicit duplicate INTEGER PRIMARY KEY (rowid) value violates;
    // ABORT leaves the original row intact.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, b)");
    exec(&mut db, "INSERT INTO t VALUES(1, 'x')");
    assert_exec_error(&mut db, "INSERT INTO t VALUES(1, 'y')");
    assert_rows(&mut db, "SELECT a, b FROM t", &[vec![int(1), text("x")]]);
}

#[test]
fn text_primary_key_rejects_duplicate() {
    // §3.5: a non-integer single-column PRIMARY KEY is a unique key; a duplicate
    // is a violation. ABORT keeps only the original row — pin its value.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a TEXT PRIMARY KEY)");
    exec(&mut db, "INSERT INTO t VALUES('x')");
    assert_exec_error(&mut db, "INSERT INTO t VALUES('x')");
    assert_rows(&mut db, "SELECT a FROM t", &[vec![text("x")]]);
}

#[test]
fn primary_key_multi_column_rejects_duplicate() {
    // §3.5: a table-level PRIMARY KEY(a,b) constrains the combination — a
    // duplicate (1,2) violates while (1,3) is fine.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b, PRIMARY KEY(a, b))");
    exec(&mut db, "INSERT INTO t VALUES(1, 2)");
    assert_exec_error(&mut db, "INSERT INTO t VALUES(1, 2)");
    exec(&mut db, "INSERT INTO t VALUES(1, 3)");
    assert_rows_unordered(
        &mut db,
        "SELECT a, b FROM t",
        &[vec![int(1), int(2)], vec![int(1), int(3)]],
    );
}

#[test]
fn text_primary_key_allows_null() {
    // §3.5 QUIRK: PRIMARY KEY does NOT imply NOT NULL in SQLite. A TEXT PRIMARY
    // KEY column (not INTEGER PK, not WITHOUT ROWID, not declared NOT NULL) admits
    // a NULL — this documented bug-compat behavior must hold.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a TEXT PRIMARY KEY)");
    exec(&mut db, "INSERT INTO t VALUES(NULL)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(1));
    assert_rows(&mut db, "SELECT a FROM t", &[vec![null()]]);
}

#[test]
fn text_primary_key_allows_multiple_nulls() {
    // §3.5 QUIRK + "NULL values are considered distinct" — a second NULL in a TEXT
    // PRIMARY KEY column is also allowed (NULLs never collide), so both survive.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a TEXT PRIMARY KEY)");
    exec(&mut db, "INSERT INTO t VALUES(NULL)");
    exec(&mut db, "INSERT INTO t VALUES(NULL)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2));
    assert_rows_unordered(&mut db, "SELECT a FROM t", &[vec![null()], vec![null()]]);
}

#[test]
fn composite_primary_key_allows_null_combinations() {
    // §3.5 QUIRK + NULLs distinct: on a rowid table a composite PRIMARY KEY(a,b)
    // allows NULL components, and (1,NULL),(1,NULL) are distinct combinations, so
    // both are allowed.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b, PRIMARY KEY(a, b))");
    exec(&mut db, "INSERT INTO t VALUES(1, NULL)");
    exec(&mut db, "INSERT INTO t VALUES(1, NULL)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2));
}

#[test]
fn integer_primary_key_null_autoassigns_integer() {
    // §5 EXCEPTION: "If an INSERT statement attempts to insert a NULL value into a
    // rowid or integer primary key column, the system chooses an integer value".
    // On an empty table the chosen rowid is 1 (autoinc.html), so `a` is a non-NULL
    // integer — NOT a NULL and NOT an error (unlike a TEXT PRIMARY KEY).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY)");
    exec(&mut db, "INSERT INTO t VALUES(NULL)");
    assert_rows(&mut db, "SELECT a FROM t", &[vec![int(1)]]);
}

#[test]
fn integer_primary_key_repeated_null_autoassigns_distinct() {
    // §5: repeated NULL inserts into an INTEGER PRIMARY KEY each auto-assign a
    // fresh rowid (1 then 2) — no duplicate, no error.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY)");
    exec(&mut db, "INSERT INTO t VALUES(NULL)");
    exec(&mut db, "INSERT INTO t VALUES(NULL)");
    assert_rows(&mut db, "SELECT a FROM t ORDER BY a", &[vec![int(1)], vec![int(2)]]);
}

#[test]
fn text_primary_key_not_null_rejects_null() {
    // §3.5 QUIRK boundary: an explicit NOT NULL on a TEXT PRIMARY KEY overrides
    // the allows-NULL behavior, so a NULL insert IS rejected (NOT NULL wins).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a TEXT PRIMARY KEY NOT NULL)");
    assert_exec_error(&mut db, "INSERT INTO t VALUES(NULL)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
}

#[test]
fn integer_primary_key_enforced_on_update() {
    // §3.5/§5: PRIMARY KEY uniqueness is enforced on UPDATE, not only INSERT
    // ("modify the table content so that two or more rows have identical primary
    // key values"). Updating one row's key to collide with another is a violation;
    // ABORT leaves both rows unchanged. This mirrors the on-UPDATE cases the file
    // has for NOT NULL / UNIQUE / CHECK, applied here to PRIMARY KEY.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, b)");
    exec(&mut db, "INSERT INTO t VALUES(1, 'x')");
    exec(&mut db, "INSERT INTO t VALUES(2, 'y')");
    assert_exec_error(&mut db, "UPDATE t SET a=1 WHERE a=2");
    assert_rows_unordered(
        &mut db,
        "SELECT a, b FROM t",
        &[vec![int(1), text("x")], vec![int(2), text("y")]],
    );
}

#[test]
fn text_primary_key_enforced_on_update() {
    // §3.5: the same on-UPDATE enforcement for a non-integer PRIMARY KEY.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a TEXT PRIMARY KEY)");
    exec(&mut db, "INSERT INTO t VALUES('x')");
    exec(&mut db, "INSERT INTO t VALUES('y')");
    assert_exec_error(&mut db, "UPDATE t SET a='x' WHERE a='y'");
    assert_rows_unordered(&mut db, "SELECT a FROM t", &[vec![text("x")], vec![text("y")]]);
}

// ===========================================================================
// CHECK — lang_createtable.html §3.7.
// The expression is cast to NUMERIC like CAST: a result of 0 / 0.0 is a
// violation; NULL or any other non-zero value is NOT a violation. Verified on
// INSERT and UPDATE; column and table CHECKs behave identically.
//
// NOTE: CHECK parses today but may not be ENFORCED by the engine yet, so several
// of these violating-insert cases may FAIL against the current engine (the insert
// wrongly succeeds). That is expected: the cases stay spec-correct so the gap is
// visible.
// ===========================================================================

#[test]
fn check_accepts_value_satisfying_predicate() {
    // §3.7: `x>0` yields 1 (true) for x=5 — a non-zero value, not a violation.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x INT CHECK(x>0))");
    exec(&mut db, "INSERT INTO t VALUES(5)");
    assert_rows(&mut db, "SELECT x FROM t", &[vec![int(5)]]);
}

#[test]
fn check_rejects_value_violating_predicate() {
    // §3.7: `x>0` yields 0 (false) for x=-1 — a violation; ABORT stores nothing.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x INT CHECK(x>0))");
    assert_exec_error(&mut db, "INSERT INTO t VALUES(-1)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
}

#[test]
fn check_rejects_zero_boundary() {
    // §3.7: `x>0` at the boundary x=0 yields 0 (false) — a violation.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x INT CHECK(x>0))");
    assert_exec_error(&mut db, "INSERT INTO t VALUES(0)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
}

#[test]
fn check_passes_when_expr_is_null() {
    // §3.7 KEY QUIRK: "If the CHECK expression evaluates to NULL ... it is not a
    // constraint violation." With x NULL, `x>0` is NULL, so the row is accepted.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x INT CHECK(x>0))");
    exec(&mut db, "INSERT INTO t VALUES(NULL)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(1));
    assert_rows(&mut db, "SELECT x FROM t", &[vec![null()]]);
}

#[test]
fn check_expr_result_zero_is_violation() {
    // §3.7: a bare `CHECK(x)` uses the column value itself; integer 0 is a
    // violation ("the result is zero (integer value 0 ...)").
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x, CHECK(x))");
    assert_exec_error(&mut db, "INSERT INTO t VALUES(0)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
}

#[test]
fn check_expr_result_real_zero_is_violation() {
    // §3.7: real 0.0 is equally a violation ("... or real value 0.0").
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x, CHECK(x))");
    assert_exec_error(&mut db, "INSERT INTO t VALUES(0.0)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
}

#[test]
fn check_expr_positive_nonzero_passes() {
    // §3.7: a bare `CHECK(x)` accepts any non-zero value; 7 is stored.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x, CHECK(x))");
    exec(&mut db, "INSERT INTO t VALUES(7)");
    assert_rows(&mut db, "SELECT x FROM t", &[vec![int(7)]]);
}

#[test]
fn check_expr_negative_nonzero_passes() {
    // §3.7: "any other non-zero value" is not a violation — a NEGATIVE value
    // passes a bare `CHECK(x)` (only zero fails, not "not positive").
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x, CHECK(x))");
    exec(&mut db, "INSERT INTO t VALUES(-3)");
    assert_rows(&mut db, "SELECT x FROM t", &[vec![int(-3)]]);
}

#[test]
fn check_expr_bare_null_passes() {
    // §3.7 QUIRK: a bare `CHECK(x)` with x NULL evaluates to NULL — not a
    // violation — so the NULL row is accepted.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x, CHECK(x))");
    exec(&mut db, "INSERT INTO t VALUES(NULL)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(1));
    assert_rows(&mut db, "SELECT x FROM t", &[vec![null()]]);
}

#[test]
fn check_table_level_across_columns_ok() {
    // §3.7: a table CHECK may reference several columns; (1,2) satisfies `a<b`.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INT, b INT, CHECK(a<b))");
    exec(&mut db, "INSERT INTO t VALUES(1, 2)");
    assert_rows(&mut db, "SELECT a, b FROM t", &[vec![int(1), int(2)]]);
}

#[test]
fn check_table_level_across_columns_violation() {
    // §3.7: (2,1) makes `a<b` false (0) — a violation; ABORT stores nothing.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INT, b INT, CHECK(a<b))");
    assert_exec_error(&mut db, "INSERT INTO t VALUES(2, 1)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
}

#[test]
fn check_table_level_equal_is_violation() {
    // §3.7 boundary: (1,1) makes `a<b` false (0) — a violation.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INT, b INT, CHECK(a<b))");
    assert_exec_error(&mut db, "INSERT INTO t VALUES(1, 1)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
}

#[test]
fn check_enforced_on_update() {
    // §3.7 + §4.1: CHECK is verified on UPDATE too. Updating x to -1 violates
    // `x>0`; ABORT leaves the row unchanged.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x INT CHECK(x>0))");
    exec(&mut db, "INSERT INTO t VALUES(5)");
    assert_exec_error(&mut db, "UPDATE t SET x=-1");
    assert_rows(&mut db, "SELECT x FROM t", &[vec![int(5)]]);
}

#[test]
fn check_update_to_valid_value_succeeds() {
    // §3.7: an UPDATE to another satisfying value (10) is not a violation.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x INT CHECK(x>0))");
    exec(&mut db, "INSERT INTO t VALUES(5)");
    exec(&mut db, "UPDATE t SET x=10");
    assert_rows(&mut db, "SELECT x FROM t", &[vec![int(10)]]);
}

#[test]
fn check_multiple_constraints_each_enforced() {
    // §3.7: a table may carry several CHECKs; each is enforced independently.
    // (5,5) satisfies both; (5,20) fails `y<10`; (-1,5) fails `x>0`. After the two
    // failures only (5,5) remains.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x INT CHECK(x>0), y INT CHECK(y<10))");
    exec(&mut db, "INSERT INTO t VALUES(5, 5)");
    assert_exec_error(&mut db, "INSERT INTO t VALUES(5, 20)");
    assert_exec_error(&mut db, "INSERT INTO t VALUES(-1, 5)");
    assert_rows(&mut db, "SELECT x, y FROM t", &[vec![int(5), int(5)]]);
}

// ===========================================================================
// Interaction sanity — the default ABORT algorithm (lang_conflict.html).
// "backs out any changes made by the current SQL statement; but changes caused
// by prior SQL statements within the same transaction are preserved."
// ===========================================================================

#[test]
fn failed_insert_preserves_earlier_statement_rows() {
    // ABORT backs out only the FAILING statement, leaving rows already committed
    // by prior (autocommit) statements intact — after a dup fails, {1,2} remain.
    // The stronger clause "changes caused by prior SQL statements *within the same
    // transaction* are preserved and the transaction remains active" is pinned by
    // the BEGIN-based test below; this one is the autocommit case.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a UNIQUE)");
    exec(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "INSERT INTO t VALUES(2)");
    assert_exec_error(&mut db, "INSERT INTO t VALUES(1)");
    assert_rows_unordered(&mut db, "SELECT a FROM t", &[vec![int(1)], vec![int(2)]]);
}

#[test]
fn abort_preserves_prior_statements_within_transaction() {
    // lang_conflict.html (ABORT): a violation "aborts the current SQL statement
    // ... and backs out any changes made by the current SQL statement; but changes
    // caused by prior SQL statements within the same transaction are preserved and
    // the transaction remains active." Inside an open BEGIN, INSERT(2) precedes the
    // failing dup INSERT(1): only the dup is rolled back, (2) survives, the
    // transaction stays active, and the following COMMIT persists {1,2}.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a UNIQUE)");
    exec(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "BEGIN");
    exec(&mut db, "INSERT INTO t VALUES(2)");
    assert_exec_error(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "COMMIT");
    assert_rows_unordered(&mut db, "SELECT a FROM t", &[vec![int(1)], vec![int(2)]]);
}

#[test]
fn multi_row_check_violation_backs_out_entire_statement() {
    // ABORT "backs out any changes made by the current SQL statement": in a
    // multi-row INSERT where the last value (-1) violates `x>0`, the earlier
    // values (5,10) written by the same statement are ALSO rolled back — the
    // table stays empty.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x INT CHECK(x>0))");
    assert_exec_error(&mut db, "INSERT INTO t VALUES(5),(10),(-1)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
}

#[test]
fn failed_update_does_not_partially_apply() {
    // ABORT applied to UPDATE: `x = x - 7` would take 5 -> -2 (violates `x>0`) and
    // 10 -> 3 (valid). Because one row violates, the WHOLE update is backed out —
    // even the would-be-valid change to 10 — so both rows keep their old values.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x INT CHECK(x>0))");
    exec(&mut db, "INSERT INTO t VALUES(5),(10)");
    assert_exec_error(&mut db, "UPDATE t SET x = x - 7");
    assert_rows_unordered(&mut db, "SELECT x FROM t", &[vec![int(5)], vec![int(10)]]);
}

// ===========================================================================
// Error classification — the violation is specifically SQLITE_CONSTRAINT
// (Error::Constraint), not an ordinary SQL/type error. One representative case
// per class. lang_createtable.html §4.1 / minisqlite-types error taxonomy.
// ===========================================================================

#[test]
fn not_null_violation_is_constraint_error() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INT NOT NULL)");
    // Bind the SQL once so the errors-check and the kind-check can never cite
    // divergent SQL in a panic message.
    let sql = "INSERT INTO t VALUES(NULL)";
    let e = assert_exec_error(&mut db, sql);
    assert_is_constraint(e, sql);
}

#[test]
fn unique_violation_is_constraint_error() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a UNIQUE)");
    exec(&mut db, "INSERT INTO t VALUES(1)");
    let sql = "INSERT INTO t VALUES(1)";
    let e = assert_exec_error(&mut db, sql);
    assert_is_constraint(e, sql);
}

#[test]
fn primary_key_violation_is_constraint_error() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, b)");
    exec(&mut db, "INSERT INTO t VALUES(1, 'x')");
    let sql = "INSERT INTO t VALUES(1, 'y')";
    let e = assert_exec_error(&mut db, sql);
    assert_is_constraint(e, sql);
}

#[test]
fn check_violation_is_constraint_error() {
    // A column CHECK violation must be SQLITE_CONSTRAINT (Error::Constraint), not a
    // parse/type error. If CHECK is unenforced this fails at `assert_exec_error`
    // (the insert wrongly succeeds).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x INT CHECK(x>0))");
    let sql = "INSERT INTO t VALUES(-1)";
    let e = assert_exec_error(&mut db, sql);
    assert_is_constraint(e, sql);
}

#[test]
fn check_table_level_violation_is_constraint_error() {
    // Companion to the column-CHECK classifier: a TABLE-level CHECK violation must
    // ALSO be SQLITE_CONSTRAINT, so an engine that enforced column CHECKs but not
    // table CHECKs is still caught here on the error *kind* (not only on the
    // errors-at-all bar the behavioral section uses).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INT, b INT, CHECK(a<b))");
    let sql = "INSERT INTO t VALUES(2, 1)";
    let e = assert_exec_error(&mut db, sql);
    assert_is_constraint(e, sql);
}
