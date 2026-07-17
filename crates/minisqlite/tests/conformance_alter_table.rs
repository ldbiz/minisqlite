//! Conformance battery: **ALTER TABLE**.
//!
//! Every expected result here is TRANSCRIBED FROM THE SQLITE DOCS in
//! `spec/sqlite-doc/lang_altertable.html`, never from what the engine currently
//! returns — a failing case is the intended signal that the engine diverges from
//! the spec. Assertions are left as genuine failing checks rather than weakened
//! to pass; a real divergence stays spec-correct.
//!
//! Spec sources (all `spec/sqlite-doc/lang_altertable.html`):
//!   * §2 "ALTER TABLE RENAME" — "The RENAME TO syntax changes the name of
//!     table-name to new-table-name." "If the table being renamed has triggers or
//!     indices, then these remain attached to the table after it has been renamed."
//!   * §3 "ALTER TABLE RENAME COLUMN" — "The RENAME COLUMN TO syntax changes the
//!     column-name of table table-name into new-column-name. The column name is
//!     changed both within the table definition itself and also within all indexes,
//!     triggers, and views that reference the column." (SQLite 3.25.0+.)
//!   * §4 "ALTER TABLE ADD COLUMN" — "The ADD COLUMN syntax is used to add a new
//!     column to an existing table. The new column is always appended to the end of
//!     the list of existing columns." Restrictions: "The column may not have a
//!     PRIMARY KEY or UNIQUE constraint."; "The column may not have a default value
//!     of CURRENT_TIME, CURRENT_DATE, CURRENT_TIMESTAMP, or an expression in
//!     parentheses."; "If a NOT NULL constraint is specified, then the column must
//!     have a default value other than NULL." And: "No changes are made to table
//!     content for renames or column addition without constraints." — so pre-existing
//!     rows read the new column's DEFAULT (NULL when none is declared).
//!   * §5 "ALTER TABLE DROP COLUMN" — "The DROP COLUMN syntax is used to remove an
//!     existing column from a table. The DROP COLUMN command removes the named
//!     column from the table, and rewrites its content to purge the data associated
//!     with that column." (SQLite 3.35.0+.)
//!
//! Each case is its own small `#[test]` so an unsupported or divergent feature fails
//! exactly that case and never masks the rest. After every ALTER, a follow-up SELECT
//! (with ORDER BY where several rows are compared) verifies the effect, so a silent
//! no-op is caught as a wrong result, not passed over.

mod conformance;

use conformance::*;

// ===========================================================================
// ADD COLUMN — basics (§4). The new column is appended to the end; with no
// DEFAULT, pre-existing rows read NULL for it.
// ===========================================================================

#[test]
fn add_column_appends_and_existing_rows_read_null() {
    // §4: a new column with no DEFAULT reads NULL in every pre-existing row.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1),(2)");
    exec(&mut db, "ALTER TABLE t ADD COLUMN b TEXT");
    assert_rows(
        &mut db,
        "SELECT a, b FROM t ORDER BY a",
        &[vec![int(1), null()], vec![int(2), null()]],
    );
}

#[test]
fn add_column_shows_in_select_star_columns() {
    // §4: the new column is "appended to the end of the list of existing columns",
    // so `SELECT *` expands to the old columns followed by the new one.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1),(2)");
    exec(&mut db, "ALTER TABLE t ADD COLUMN b TEXT");
    assert_columns(&mut db, "SELECT * FROM t", &["a", "b"]);
}

#[test]
fn add_column_optional_column_keyword_omitted() {
    // The COLUMN keyword is optional: `ADD c INT` (no COLUMN word) adds a column too.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    exec(&mut db, "ALTER TABLE t ADD c INT");
    assert_columns(&mut db, "SELECT * FROM t", &["a", "c"]);
    // The pre-existing row reads NULL for the added no-default column.
    assert_rows(&mut db, "SELECT a, c FROM t", &[vec![int(1), null()]]);
}

// ===========================================================================
// ADD COLUMN — DEFAULT on pre-existing rows (§4). SQLite makes "no changes to
// table content" on ADD COLUMN, so a pre-existing row reads the declared DEFAULT.
// ===========================================================================

#[test]
fn add_column_text_default_read_by_existing_rows() {
    // §4: existing rows must read the DEFAULT ('x'), not NULL, for the added column.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    exec(&mut db, "ALTER TABLE t ADD COLUMN d TEXT DEFAULT 'x'");
    assert_rows(&mut db, "SELECT a, d FROM t", &[vec![int(1), text("x")]]);
}

#[test]
fn add_column_numeric_default_read_by_existing_rows() {
    // §4: a numeric DEFAULT (7) is likewise read by every pre-existing row.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    exec(&mut db, "ALTER TABLE t ADD COLUMN e INT DEFAULT 7");
    assert_scalar(&mut db, "SELECT e FROM t WHERE a = 1", int(7));
}

#[test]
fn add_column_default_used_by_new_insert_omitting_the_column() {
    // A new INSERT that does not supply the added column takes its DEFAULT — the
    // ordinary DEFAULT-clause behavior, now for a column introduced by ALTER.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    exec(&mut db, "ALTER TABLE t ADD COLUMN d TEXT DEFAULT 'x'");
    exec(&mut db, "INSERT INTO t(a) VALUES (2)");
    assert_scalar(&mut db, "SELECT d FROM t WHERE a = 2", text("x"));
}

#[test]
fn add_column_then_full_row_insert_accepts_new_columns() {
    // After ADD COLUMN, a subsequent INSERT may name and supply the new columns.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    exec(&mut db, "ALTER TABLE t ADD COLUMN d TEXT DEFAULT 'x'");
    exec(&mut db, "ALTER TABLE t ADD COLUMN e INT DEFAULT 7");
    exec(&mut db, "INSERT INTO t(a, d, e) VALUES (9, 'y', 3)");
    assert_rows(
        &mut db,
        "SELECT a, d, e FROM t WHERE a = 9",
        &[vec![int(9), text("y"), int(3)]],
    );
}

// ===========================================================================
// ADD COLUMN — NOT NULL with a non-NULL default (§4). A NOT NULL column is
// allowed *when* it carries a non-NULL default; existing rows then read that
// default.
// ===========================================================================

#[test]
fn add_column_not_null_with_default_succeeds() {
    // §4: "If a NOT NULL constraint is specified, then the column must have a
    // default value other than NULL." A DEFAULT 0 satisfies that, so the ADD works.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    exec(&mut db, "ALTER TABLE t ADD COLUMN n INTEGER NOT NULL DEFAULT 0");
    assert_columns(&mut db, "SELECT * FROM t", &["a", "n"]);
}

#[test]
fn add_column_not_null_default_read_by_existing_rows() {
    // §4: with "no changes ... to table content", the pre-existing row reads the
    // NOT NULL column's default value (0), not NULL.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    exec(&mut db, "ALTER TABLE t ADD COLUMN n INTEGER NOT NULL DEFAULT 0");
    assert_scalar(&mut db, "SELECT n FROM t WHERE a = 1", int(0));
}

// ===========================================================================
// ADD COLUMN — restrictions (§4). These forms are documented errors; each must
// fail. `assert_exec_error` pins that the statement errors at all; the follow-up
// `assert_columns` pins that the rejected column left NO partial effect — SQLite
// validates the restriction before touching the table, so the column list is
// unchanged. Together they turn "something errored" into "the right thing was
// rejected with no leak". The spec rule violated is cited on each case.
// ===========================================================================

#[test]
fn add_column_unique_is_error() {
    // §4: "The column may not have a PRIMARY KEY or UNIQUE constraint."
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    assert_exec_error(&mut db, "ALTER TABLE t ADD COLUMN u INTEGER UNIQUE");
    assert_columns(&mut db, "SELECT * FROM t", &["a"]);
}

#[test]
fn add_column_primary_key_is_error() {
    // §4: "The column may not have a PRIMARY KEY or UNIQUE constraint."
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    assert_exec_error(&mut db, "ALTER TABLE t ADD COLUMN p INTEGER PRIMARY KEY");
    assert_columns(&mut db, "SELECT * FROM t", &["a"]);
}

#[test]
fn add_column_not_null_without_default_is_error() {
    // §4: "If a NOT NULL constraint is specified, then the column must have a
    // default value other than NULL." No default at all => the ADD must fail.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    assert_exec_error(&mut db, "ALTER TABLE t ADD COLUMN n INTEGER NOT NULL");
    assert_columns(&mut db, "SELECT * FROM t", &["a"]);
}

#[test]
fn add_column_default_current_timestamp_is_error() {
    // §4: "The column may not have a default value of CURRENT_TIME, CURRENT_DATE,
    // CURRENT_TIMESTAMP, or an expression in parentheses." (The CURRENT_TIMESTAMP
    // form.)
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    assert_exec_error(&mut db, "ALTER TABLE t ADD COLUMN ts TEXT DEFAULT CURRENT_TIMESTAMP");
    assert_columns(&mut db, "SELECT * FROM t", &["a"]);
}

#[test]
fn add_column_default_current_time_is_error() {
    // §4: the same rule bars a CURRENT_TIME default.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    assert_exec_error(&mut db, "ALTER TABLE t ADD COLUMN ct TEXT DEFAULT CURRENT_TIME");
    assert_columns(&mut db, "SELECT * FROM t", &["a"]);
}

#[test]
fn add_column_default_current_date_is_error() {
    // §4: the same rule bars a CURRENT_DATE default.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    assert_exec_error(&mut db, "ALTER TABLE t ADD COLUMN cd TEXT DEFAULT CURRENT_DATE");
    assert_columns(&mut db, "SELECT * FROM t", &["a"]);
}

#[test]
fn add_column_default_parenthesized_expr_is_error() {
    // §4: the same rule bars "an expression in parentheses" as a default.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    assert_exec_error(&mut db, "ALTER TABLE t ADD COLUMN x INTEGER DEFAULT (1 + 2)");
    assert_columns(&mut db, "SELECT * FROM t", &["a"]);
}

#[test]
fn add_column_stored_generated_is_error() {
    // §4: "The column may not be GENERATED ALWAYS ... STORED, though VIRTUAL
    // columns are allowed." A STORED generated column is rejected.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    assert_exec_error(
        &mut db,
        "ALTER TABLE t ADD COLUMN g INTEGER GENERATED ALWAYS AS (a + 1) STORED",
    );
    assert_columns(&mut db, "SELECT * FROM t", &["a"]);
}

// ===========================================================================
// ADD COLUMN — other errors (§4).
// ===========================================================================

#[test]
fn add_column_to_missing_table_is_error() {
    // Altering a table that does not exist is an error ("no such table").
    let mut db = mem();
    assert_exec_error(&mut db, "ALTER TABLE nope ADD COLUMN x INT");
}

#[test]
fn add_column_duplicate_name_is_error() {
    // A column-def whose name already exists on the table is a duplicate-column
    // error (the new column must be distinct from the existing ones).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    assert_exec_error(&mut db, "ALTER TABLE t ADD COLUMN a INT");
    // The rejected duplicate must not have been appended — still one column `a`.
    assert_columns(&mut db, "SELECT * FROM t", &["a"]);
}

// ===========================================================================
// RENAME TO (§2). The table is renamed; data is preserved, the old name is gone,
// and attached indices continue to work under the new name.
// ===========================================================================

#[test]
fn rename_to_preserves_data_under_new_name() {
    // §2: the name changes to new-table-name; the table's rows are unaffected.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "INSERT INTO t VALUES (1, 'x')");
    exec(&mut db, "ALTER TABLE t RENAME TO t2");
    assert_rows(&mut db, "SELECT a, b FROM t2", &[vec![int(1), text("x")]]);
}

#[test]
fn rename_to_old_name_no_longer_resolves() {
    // §2: after the rename the old name is gone — a query on it is an error
    // ("no such table: t").
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "INSERT INTO t VALUES (1, 'x')");
    exec(&mut db, "ALTER TABLE t RENAME TO t2");
    assert_query_error(&mut db, "SELECT a FROM t");
}

#[test]
fn rename_to_keeps_index_attached() {
    // §2: "If the table being renamed has triggers or indices, then these remain
    // attached to the table after it has been renamed." The index still serves a
    // lookup on the renamed table.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "CREATE INDEX ix ON t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1),(2)");
    exec(&mut db, "ALTER TABLE t RENAME TO t2");
    assert_rows(&mut db, "SELECT a FROM t2 WHERE a = 2 ORDER BY a", &[vec![int(2)]]);
}

#[test]
fn rename_to_preserves_unique_index_enforcement() {
    // §2: indices "remain attached to the table after it has been renamed." A bare
    // SELECT cannot distinguish a surviving index from a full scan, so this pins the
    // index's OBSERVABLE effect instead: a UNIQUE index still rejects a duplicate
    // under the new table name. Had the rename dropped or detached the index, the
    // duplicate INSERT would be accepted and this assertion would (correctly) fail.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "CREATE UNIQUE INDEX ix ON t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1),(2)");
    exec(&mut db, "ALTER TABLE t RENAME TO t2");
    assert_exec_error(&mut db, "INSERT INTO t2 VALUES (2)");
}

#[test]
fn rename_to_existing_name_is_error() {
    // §2: the new name must be free; renaming onto a name that already names another
    // table (or index) is an error ("there is already another table or index with
    // this name").
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "CREATE TABLE u(a)");
    assert_exec_error(&mut db, "ALTER TABLE t RENAME TO u");
}

#[test]
fn rename_to_own_name_is_error() {
    // §2: the target name must be free across the whole schema namespace; renaming a
    // table onto its OWN name still collides (the name is in use), so it errors — a
    // slightly different path than the cross-object collision above.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    assert_exec_error(&mut db, "ALTER TABLE t RENAME TO t");
}

// ===========================================================================
// RENAME COLUMN (§3). SQLite 3.25.0+ renames a column in place; data is
// preserved. NOTE: the engine currently reports "RENAME COLUMN not yet
// supported"; these spec-correct cases therefore FAIL today — that failure is the
// intended signal. They are NOT weakened to match the engine, and NOT
// `assert_exec_error` (the spec says the statement SUCCEEDS).
// ===========================================================================

#[test]
fn rename_column_changes_the_column_name() {
    // §3: RENAME COLUMN b TO c changes the column name in the table definition.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "INSERT INTO t VALUES (1, 2)");
    exec(&mut db, "ALTER TABLE t RENAME COLUMN b TO c");
    assert_columns(&mut db, "SELECT * FROM t", &["a", "c"]);
}

#[test]
fn rename_column_preserves_row_data() {
    // §3: renaming a column does not alter row content — the value is read back
    // under the new column name.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "INSERT INTO t VALUES (1, 2)");
    exec(&mut db, "ALTER TABLE t RENAME COLUMN b TO c");
    assert_rows(&mut db, "SELECT a, c FROM t", &[vec![int(1), int(2)]]);
}

#[test]
fn rename_column_optional_column_keyword_omitted() {
    // §3: the COLUMN keyword is optional — `RENAME b TO c` also renames a column.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "INSERT INTO t VALUES (1, 2)");
    exec(&mut db, "ALTER TABLE t RENAME b TO c");
    assert_columns(&mut db, "SELECT * FROM t", &["a", "c"]);
}

// ===========================================================================
// DROP COLUMN (§5). SQLite 3.35.0+ removes a column and purges its data. NOTE:
// the engine currently reports "DROP COLUMN not yet supported"; these
// spec-correct cases therefore FAIL today — that failure is the intended signal.
// NOT weakened; NOT `assert_exec_error`.
// ===========================================================================

#[test]
fn drop_column_removes_the_column() {
    // §5: DROP COLUMN b removes b, leaving the other columns in order.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b, c)");
    exec(&mut db, "ALTER TABLE t DROP COLUMN b");
    assert_columns(&mut db, "SELECT * FROM t", &["a", "c"]);
}

#[test]
fn drop_column_preserves_other_columns_data() {
    // §5: the remaining columns keep their values after the dropped column's data
    // is purged.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b, c)");
    exec(&mut db, "INSERT INTO t VALUES (1, 2, 3)");
    exec(&mut db, "ALTER TABLE t DROP COLUMN b");
    assert_rows(&mut db, "SELECT a, c FROM t", &[vec![int(1), int(3)]]);
}

#[test]
fn drop_column_optional_column_keyword_omitted() {
    // §5: the COLUMN keyword is optional — `DROP b` also drops a column.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b, c)");
    exec(&mut db, "ALTER TABLE t DROP b");
    assert_columns(&mut db, "SELECT * FROM t", &["a", "c"]);
}
