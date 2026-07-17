//! Conformance battery: **UPDATE** and **DELETE** DML, plus **RETURNING** and the
//! `changes()` function.
//!
//! Every assertion here is transcribed from the SQLite documentation, never from
//! whatever the engine currently returns — a failing case is the intended signal
//! that the engine diverges from the spec.
//!
//! Spec sources:
//! - `spec/sqlite-doc/lang_update.html` §1–2: an UPDATE modifies zero or more rows;
//!   with no WHERE clause *all* rows are modified, otherwise only rows for which the
//!   WHERE boolean expression is TRUE (it is not an error if none match — zero rows
//!   are affected). Each `SET column-name = expr` assigns the evaluated scalar to the
//!   named column for every affected row; columns not listed are left unmodified; if a
//!   column-name appears more than once, all but the rightmost occurrence is ignored;
//!   scalar expressions may reference columns of the row being updated and are all
//!   evaluated *before* any assignment is made.
//! - `spec/sqlite-doc/lang_delete.html` §1: DELETE removes records; with no WHERE all
//!   records are deleted, otherwise only rows for which the WHERE expression is TRUE —
//!   rows for which it is FALSE or NULL are retained. §4: even under the truncate
//!   optimization (WHERE and RETURNING both omitted) `changes()` still reports the
//!   number of deleted rows (since version 3.6.5).
//! - `spec/sqlite-doc/lang_returning.html` §2: a RETURNING clause returns one result
//!   row per row deleted/updated; `*` expands to all non-hidden columns; for UPDATE the
//!   referenced column values are those *after* the change, for DELETE those *before*
//!   the delete. §2.1: the output order of RETURNING rows is arbitrary (so multi-row
//!   RETURNING is asserted here with the unordered/multiset comparison). A statement
//!   that modifies zero rows produces zero RETURNING rows.
//! - `spec/sqlite-doc/lang_corefunc.html` `changes()`: returns the number of rows
//!   changed, inserted, or deleted by the most recently completed INSERT, DELETE, or
//!   UPDATE statement (exclusive of lower-level triggers).
//!
//! Each case is its own `#[test]` so an unsupported operation fails exactly that case
//! rather than masking the rest. Every mutation is verified with a follow-up SELECT
//! (ORDER BY for determinism).

mod conformance;
use conformance::*;

// Needed only to name the `Connection` type in the shared fixture's signature; every
// value comparison still goes through the harness (Value has no PartialEq).
use minisqlite::Connection;

/// A fresh in-memory database preloaded with the fixture table shared by every case:
///
/// ```text
/// id | name | v
///  1 | 'a'  | 10
///  2 | 'b'  | 20
///  3 | 'c'  | 30
///  4 | 'a'  | 40
/// ```
///
/// `name='a'` deliberately matches two rows (1 and 4) so multi-row WHERE, multi-row
/// RETURNING, and `changes()` counting can all be exercised against the same shape.
fn fixture() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, name TEXT, v INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1,'a',10),(2,'b',20),(3,'c',30),(4,'a',40)");
    db
}

// =============================================================================
// UPDATE — lang_update.html §1–2
// =============================================================================

/// WHERE matches exactly one row: only that row's listed column changes; the row's
/// other columns and every other row are left unmodified (§2).
#[test]
fn update_where_matches_single_row() {
    let mut db = fixture();
    exec(&mut db, "UPDATE t SET v=99 WHERE id=2");
    assert_rows(&mut db, "SELECT v FROM t WHERE id=2", &[vec![int(99)]]);
    // Unlisted column (name) of the updated row is unchanged; other rows untouched.
    assert_rows(
        &mut db,
        "SELECT id, name, v FROM t ORDER BY id",
        &[
            vec![int(1), text("a"), int(10)],
            vec![int(2), text("b"), int(99)],
            vec![int(3), text("c"), int(30)],
            vec![int(4), text("a"), int(40)],
        ],
    );
}

/// WHERE matches multiple rows (name='a' → ids 1 and 4): every matching row is updated
/// and non-matching rows are left alone (§2).
#[test]
fn update_where_matches_multiple_rows() {
    let mut db = fixture();
    exec(&mut db, "UPDATE t SET v=0 WHERE name='a'");
    assert_rows(
        &mut db,
        "SELECT id, v FROM t ORDER BY id",
        &[
            vec![int(1), int(0)],
            vec![int(2), int(20)],
            vec![int(3), int(30)],
            vec![int(4), int(0)],
        ],
    );
}

/// No WHERE clause: all rows are modified (§2). The assignment reads the row's own
/// value, exercising the "expressions may refer to columns of the row being updated"
/// rule for every row.
#[test]
fn update_all_rows_no_where() {
    let mut db = fixture();
    exec(&mut db, "UPDATE t SET v=v+1");
    assert_rows(
        &mut db,
        "SELECT id, v FROM t ORDER BY id",
        &[
            vec![int(1), int(11)],
            vec![int(2), int(21)],
            vec![int(3), int(31)],
            vec![int(4), int(41)],
        ],
    );
}

/// Multiple columns in one SET list are all assigned for the affected row (§2).
#[test]
fn update_multiple_columns() {
    let mut db = fixture();
    exec(&mut db, "UPDATE t SET name='z', v=7 WHERE id=3");
    assert_rows(
        &mut db,
        "SELECT id, name, v FROM t WHERE id=3",
        &[vec![int(3), text("z"), int(7)]],
    );
    // Every other row unchanged.
    assert_rows(
        &mut db,
        "SELECT id, name, v FROM t ORDER BY id",
        &[
            vec![int(1), text("a"), int(10)],
            vec![int(2), text("b"), int(20)],
            vec![int(3), text("z"), int(7)],
            vec![int(4), text("a"), int(40)],
        ],
    );
}

/// The SET expression may be a function of the column being assigned (§2): v*2 reads
/// the old v (10) and writes 20.
#[test]
fn update_expression_of_same_column() {
    let mut db = fixture();
    exec(&mut db, "UPDATE t SET v=v*2 WHERE id=1");
    assert_rows(&mut db, "SELECT v FROM t WHERE id=1", &[vec![int(20)]]);
    // Only the matched row changed; every other row and column is intact.
    assert_rows(
        &mut db,
        "SELECT id, name, v FROM t ORDER BY id",
        &[
            vec![int(1), text("a"), int(20)],
            vec![int(2), text("b"), int(20)],
            vec![int(3), text("c"), int(30)],
            vec![int(4), text("a"), int(40)],
        ],
    );
}

/// The SET expression may reference a different column of the same row (§2):
/// v = id*100 for id=3 writes 300.
#[test]
fn update_expression_of_other_column() {
    let mut db = fixture();
    exec(&mut db, "UPDATE t SET v = id*100 WHERE id=3");
    assert_rows(&mut db, "SELECT v FROM t WHERE id=3", &[vec![int(300)]]);
    assert_rows(
        &mut db,
        "SELECT id, name, v FROM t ORDER BY id",
        &[
            vec![int(1), text("a"), int(10)],
            vec![int(2), text("b"), int(20)],
            vec![int(3), text("c"), int(300)],
            vec![int(4), text("a"), int(40)],
        ],
    );
}

/// A column may be set to NULL (§2); the stored value becomes the NULL storage class.
#[test]
fn update_set_column_to_null() {
    let mut db = fixture();
    exec(&mut db, "UPDATE t SET name=NULL WHERE id=1");
    assert_rows(&mut db, "SELECT name FROM t WHERE id=1", &[vec![null()]]);
    assert_rows(
        &mut db,
        "SELECT id, name, v FROM t ORDER BY id",
        &[
            vec![int(1), null(), int(10)],
            vec![int(2), text("b"), int(20)],
            vec![int(3), text("c"), int(30)],
            vec![int(4), text("a"), int(40)],
        ],
    );
}

/// UPDATE applies the target column's type affinity to the stored value, exactly as
/// INSERT does. Column `v` is declared INTEGER (INTEGER affinity), so assigning the text
/// '7' stores it as the INTEGER 7, not TEXT (datatype3.html §3: INTEGER affinity behaves
/// like NUMERIC — a well-formed integer literal in text form is converted to INTEGER).
#[test]
fn update_set_applies_integer_affinity_to_text() {
    let mut db = fixture();
    exec(&mut db, "UPDATE t SET v='7' WHERE id=1");
    assert_rows(
        &mut db,
        "SELECT typeof(v), v FROM t WHERE id=1",
        &[vec![text("integer"), int(7)]],
    );
}

/// A boolean WHERE with OR selects the union of matching rows (§2).
#[test]
fn update_where_or_predicate() {
    let mut db = fixture();
    exec(&mut db, "UPDATE t SET v=0 WHERE id=1 OR id=3");
    assert_rows(
        &mut db,
        "SELECT id, v FROM t ORDER BY id",
        &[
            vec![int(1), int(0)],
            vec![int(2), int(20)],
            vec![int(3), int(0)],
            vec![int(4), int(40)],
        ],
    );
}

/// "If a single column-name appears more than once in the list of assignment
/// expressions, all but the rightmost occurrence is ignored" (§2): `v=1, v=2` → 2.
#[test]
fn update_duplicate_column_rightmost_wins() {
    let mut db = fixture();
    exec(&mut db, "UPDATE t SET v=1, v=2 WHERE id=1");
    assert_rows(&mut db, "SELECT v FROM t WHERE id=1", &[vec![int(2)]]);
    assert_rows(
        &mut db,
        "SELECT id, name, v FROM t ORDER BY id",
        &[
            vec![int(1), text("a"), int(2)],
            vec![int(2), text("b"), int(20)],
            vec![int(3), text("c"), int(30)],
            vec![int(4), text("a"), int(40)],
        ],
    );
}

/// "It is not an error if the WHERE clause does not evaluate to true for any row" —
/// the statement simply affects zero rows and the table is unchanged (§2).
#[test]
fn update_no_matching_rows_is_not_an_error() {
    let mut db = fixture();
    exec(&mut db, "UPDATE t SET v=99 WHERE id=999");
    assert_rows(
        &mut db,
        "SELECT id, name, v FROM t ORDER BY id",
        &[
            vec![int(1), text("a"), int(10)],
            vec![int(2), text("b"), int(20)],
            vec![int(3), text("c"), int(30)],
            vec![int(4), text("a"), int(40)],
        ],
    );
}

/// A WHERE that evaluates to NULL for every row updates nothing: only rows for which
/// the WHERE boolean expression is TRUE are affected (§2), and NULL is not TRUE. This
/// mirrors the DELETE "false or NULL are retained" rule (lang_delete.html §1) on the
/// UPDATE side, whose WHERE evaluation is separately implemented.
#[test]
fn update_where_null_leaves_all_rows_unchanged() {
    let mut db = fixture();
    exec(&mut db, "UPDATE t SET v=0 WHERE NULL");
    assert_rows(
        &mut db,
        "SELECT id, name, v FROM t ORDER BY id",
        &[
            vec![int(1), text("a"), int(10)],
            vec![int(2), text("b"), int(20)],
            vec![int(3), text("c"), int(30)],
            vec![int(4), text("a"), int(40)],
        ],
    );
}

/// "The scalar expressions may refer to columns of the row being updated. In this case
/// all scalar expressions are evaluated before any assignments are made" (§2). With
/// `SET v=id, id=v+100` on the id=2 row, both right-hand sides must read the *pre*-update
/// values (id=2, v=20), giving v=2 and id=120. A left-to-right evaluate-and-assign would
/// instead compute id from the already-reassigned v (2+100=102), so id=120 discriminates
/// the correct ordering. (This assignment also relocates the INTEGER PRIMARY KEY row.)
#[test]
fn update_evaluates_expressions_before_assigning() {
    let mut db = fixture();
    exec(&mut db, "UPDATE t SET v=id, id=v+100 WHERE id=2");
    assert_rows(
        &mut db,
        "SELECT id, name, v FROM t ORDER BY id",
        &[
            vec![int(1), text("a"), int(10)],
            vec![int(3), text("c"), int(30)],
            vec![int(4), text("a"), int(40)],
            vec![int(120), text("b"), int(2)],
        ],
    );
    assert_scalar(&mut db, "SELECT count(*) FROM t WHERE id=2", int(0));
}

/// Updating the INTEGER PRIMARY KEY (a rowid alias) relocates the row (§2): after
/// `SET id=99 WHERE id=1` the row is found at id=99 and no longer at id=1. This
/// exercises the b-tree relocation path of the newly-implemented UPDATE planner in
/// isolation (no cross-column dependency).
#[test]
fn update_changing_primary_key_relocates_row() {
    let mut db = fixture();
    exec(&mut db, "UPDATE t SET id=99 WHERE id=1");
    assert_rows(
        &mut db,
        "SELECT id, name, v FROM t WHERE id=99",
        &[vec![int(99), text("a"), int(10)]],
    );
    assert_scalar(&mut db, "SELECT count(*) FROM t WHERE id=1", int(0));
    assert_rows(
        &mut db,
        "SELECT id, name, v FROM t ORDER BY id",
        &[
            vec![int(2), text("b"), int(20)],
            vec![int(3), text("c"), int(30)],
            vec![int(4), text("a"), int(40)],
            vec![int(99), text("a"), int(10)],
        ],
    );
}

// ---- UPDATE ... RETURNING — lang_returning.html §2 --------------------------

/// UPDATE ... RETURNING yields one row per updated row, with the listed columns'
/// values *after* the change (§2). Single row → ordered comparison is deterministic.
#[test]
fn update_returning_id_and_new_value() {
    let mut db = fixture();
    assert_rows(
        &mut db,
        "UPDATE t SET v=5 WHERE id=1 RETURNING id, v",
        &[vec![int(1), int(5)]],
    );
    // Verify the update actually persisted.
    assert_rows(&mut db, "SELECT v FROM t WHERE id=1", &[vec![int(5)]]);
}

/// RETURNING columns for UPDATE reflect the value *after* the change even when the SET
/// is an expression: v=20 becomes 120 and RETURNING reports 120 (§2).
#[test]
fn update_returning_reports_value_after_change() {
    let mut db = fixture();
    assert_rows(
        &mut db,
        "UPDATE t SET v = v + 100 WHERE id=2 RETURNING v",
        &[vec![int(120)]],
    );
    assert_rows(&mut db, "SELECT v FROM t WHERE id=2", &[vec![int(120)]]);
}

/// `RETURNING *` expands to all non-hidden columns of the updated row, with post-change
/// values (§2): id and name unchanged, v updated.
#[test]
fn update_returning_star_expands_all_columns() {
    let mut db = fixture();
    assert_rows(
        &mut db,
        "UPDATE t SET v=5 WHERE id=1 RETURNING *",
        &[vec![int(1), text("a"), int(5)]],
    );
    // Confirm the change persisted, matching the "verify every mutation" discipline.
    assert_rows(
        &mut db,
        "SELECT id, name, v FROM t WHERE id=1",
        &[vec![int(1), text("a"), int(5)]],
    );
}

/// When an UPDATE ... RETURNING affects several rows, the output order is arbitrary
/// (§2.1), so the returned ids {1,4} are compared as a multiset.
#[test]
fn update_returning_multiple_rows_unordered() {
    let mut db = fixture();
    assert_rows_unordered(
        &mut db,
        "UPDATE t SET v=0 WHERE name='a' RETURNING id",
        &[vec![int(1)], vec![int(4)]],
    );
    // Both matching rows persisted their update.
    assert_rows(
        &mut db,
        "SELECT id, v FROM t ORDER BY id",
        &[
            vec![int(1), int(0)],
            vec![int(2), int(20)],
            vec![int(3), int(30)],
            vec![int(4), int(0)],
        ],
    );
}

/// An UPDATE that matches no rows returns zero RETURNING rows and leaves the table
/// unchanged (§2 / lang_returning.html §2).
#[test]
fn update_returning_no_match_is_empty() {
    let mut db = fixture();
    assert_rows(&mut db, "UPDATE t SET v=99 WHERE id=999 RETURNING id, v", &[]);
    assert_rows(
        &mut db,
        "SELECT id, v FROM t ORDER BY id",
        &[
            vec![int(1), int(10)],
            vec![int(2), int(20)],
            vec![int(3), int(30)],
            vec![int(4), int(40)],
        ],
    );
}

/// A RETURNING clause defines result-column names like a SELECT, and an `AS` alias
/// determines the column name (lang_returning.html §2). Pin the reported names/aliases.
#[test]
fn update_returning_column_names() {
    let mut db = fixture();
    assert_columns(
        &mut db,
        "UPDATE t SET v=5 WHERE id=1 RETURNING id AS k, v AS w",
        &["k", "w"],
    );
}

/// A RETURNING item may be an arbitrary expression, not just a bare column or `*`
/// (lang_returning.html §2). For UPDATE it is evaluated over the post-change row, so
/// `v+1` on the new v=5 yields 6.
#[test]
fn update_returning_expression_value() {
    let mut db = fixture();
    assert_rows(
        &mut db,
        "UPDATE t SET v=5 WHERE id=1 RETURNING v+1",
        &[vec![int(6)]],
    );
    assert_rows(&mut db, "SELECT v FROM t WHERE id=1", &[vec![int(5)]]);
}

// =============================================================================
// DELETE — lang_delete.html §1
// =============================================================================

/// WHERE matches one row: that row is removed and the rest remain (§1).
#[test]
fn delete_where_matches_single_row() {
    let mut db = fixture();
    exec(&mut db, "DELETE FROM t WHERE id=2");
    assert_rows(
        &mut db,
        "SELECT id FROM t ORDER BY id",
        &[vec![int(1)], vec![int(3)], vec![int(4)]],
    );
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(3));
}

/// WHERE matches multiple rows (name='a' → ids 1 and 4): all matching rows removed (§1).
#[test]
fn delete_where_matches_multiple_rows() {
    let mut db = fixture();
    exec(&mut db, "DELETE FROM t WHERE name='a'");
    assert_rows(
        &mut db,
        "SELECT id FROM t ORDER BY id",
        &[vec![int(2)], vec![int(3)]],
    );
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2));
}

/// No WHERE clause: all records are deleted (§1). Exercises the truncate-optimization
/// path (WHERE and RETURNING both omitted).
#[test]
fn delete_all_rows_no_where() {
    let mut db = fixture();
    exec(&mut db, "DELETE FROM t");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
    assert_rows(&mut db, "SELECT id FROM t ORDER BY id", &[]);
}

/// A compound boolean WHERE (id>=2 AND id<=3) deletes exactly the rows for which it is
/// TRUE (§1).
#[test]
fn delete_compound_where_predicate() {
    let mut db = fixture();
    exec(&mut db, "DELETE FROM t WHERE id >= 2 AND id <= 3");
    assert_rows(
        &mut db,
        "SELECT id FROM t ORDER BY id",
        &[vec![int(1)], vec![int(4)]],
    );
}

/// "Rows for which the expression is false ... are retained" (§1): a WHERE that is FALSE
/// for every row deletes nothing.
#[test]
fn delete_where_false_retains_all_rows() {
    let mut db = fixture();
    exec(&mut db, "DELETE FROM t WHERE name='nope'");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(4));
}

/// "Rows for which the expression is ... NULL are retained" (§1): a WHERE that is NULL
/// for every row deletes nothing.
#[test]
fn delete_where_null_retains_all_rows() {
    let mut db = fixture();
    exec(&mut db, "DELETE FROM t WHERE NULL");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(4));
}

/// A DELETE whose WHERE matches nothing removes nothing and is not an error (§1).
#[test]
fn delete_no_matching_rows_removes_nothing() {
    let mut db = fixture();
    exec(&mut db, "DELETE FROM t WHERE id=999");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(4));
    assert_rows(
        &mut db,
        "SELECT id FROM t ORDER BY id",
        &[vec![int(1)], vec![int(2)], vec![int(3)], vec![int(4)]],
    );
}

/// A DELETE WHERE predicate may contain a subquery (lang_delete.html §1 boolean
/// expression; the IN operator, lang_expr.html): `id IN (SELECT id FROM t WHERE
/// name='a')` matches ids 1 and 4, so those rows are deleted and 2,3 remain. Probes the
/// newly-implemented DELETE planner against a correlated-free subquery predicate.
#[test]
fn delete_where_in_subquery() {
    let mut db = fixture();
    exec(
        &mut db,
        "DELETE FROM t WHERE id IN (SELECT id FROM t WHERE name='a')",
    );
    assert_rows(
        &mut db,
        "SELECT id FROM t ORDER BY id",
        &[vec![int(2)], vec![int(3)]],
    );
}

// ---- DELETE ... RETURNING — lang_returning.html §2 --------------------------

/// DELETE ... RETURNING yields one row per deleted row, with the columns' values
/// *before* the delete (§2): the deleted row (3,'c') is reported, then it is gone.
#[test]
fn delete_returning_reports_values_before_delete() {
    let mut db = fixture();
    assert_rows(
        &mut db,
        "DELETE FROM t WHERE id=3 RETURNING id, name",
        &[vec![int(3), text("c")]],
    );
    assert_scalar(&mut db, "SELECT count(*) FROM t WHERE id=3", int(0));
    assert_rows(
        &mut db,
        "SELECT id FROM t ORDER BY id",
        &[vec![int(1)], vec![int(2)], vec![int(4)]],
    );
}

/// `RETURNING *` on DELETE expands to all non-hidden columns, with the pre-delete row
/// values (§2).
#[test]
fn delete_returning_star_expands_all_columns() {
    let mut db = fixture();
    assert_rows(
        &mut db,
        "DELETE FROM t WHERE id=1 RETURNING *",
        &[vec![int(1), text("a"), int(10)]],
    );
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(3));
    // The right row (id=1) was removed, not some other row.
    assert_rows(
        &mut db,
        "SELECT id FROM t ORDER BY id",
        &[vec![int(2)], vec![int(3)], vec![int(4)]],
    );
}

/// A multi-row DELETE ... RETURNING has arbitrary output order (§2.1), so the returned
/// ids {1,4} are compared as a multiset; both rows are then gone.
#[test]
fn delete_returning_multiple_rows_unordered() {
    let mut db = fixture();
    assert_rows_unordered(
        &mut db,
        "DELETE FROM t WHERE name='a' RETURNING id",
        &[vec![int(1)], vec![int(4)]],
    );
    assert_rows(
        &mut db,
        "SELECT id FROM t ORDER BY id",
        &[vec![int(2)], vec![int(3)]],
    );
}

/// A DELETE that matches no rows returns zero RETURNING rows and removes nothing (§2).
#[test]
fn delete_returning_no_match_is_empty() {
    let mut db = fixture();
    assert_rows(&mut db, "DELETE FROM t WHERE id=999 RETURNING id", &[]);
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(4));
}

/// A RETURNING item may be an arbitrary expression (lang_returning.html §2). For DELETE
/// it is evaluated over the *pre*-delete row, so `v*2` on the doomed row's old v=20
/// yields 40, and the row is then gone.
#[test]
fn delete_returning_expression_before_value() {
    let mut db = fixture();
    assert_rows(
        &mut db,
        "DELETE FROM t WHERE id=2 RETURNING v*2",
        &[vec![int(40)]],
    );
    assert_scalar(&mut db, "SELECT count(*) FROM t WHERE id=2", int(0));
}

/// DELETE RETURNING reports SELECT-like column names, and `AS` sets an alias
/// (lang_returning.html §2).
#[test]
fn delete_returning_column_names() {
    let mut db = fixture();
    assert_columns(
        &mut db,
        "DELETE FROM t WHERE id=2 RETURNING id AS pk, name AS nm",
        &["pk", "nm"],
    );
}

// =============================================================================
// changes() — lang_corefunc.html: rows changed by the most recent DML statement.
// =============================================================================

/// After an UPDATE that modifies two rows, `changes()` reports 2.
#[test]
fn changes_after_update_multiple_rows() {
    let mut db = fixture();
    exec(&mut db, "UPDATE t SET v=0 WHERE name='a'");
    assert_scalar(&mut db, "SELECT changes()", int(2));
}

/// After an UPDATE with no WHERE, `changes()` reports every row modified. A fifth row
/// is inserted first so the expected count (5) differs from both the fixture's own
/// INSERT count (4) and any single-row count — the test stands on its own rather than
/// coincidentally matching a stale counter.
#[test]
fn changes_after_update_all_rows() {
    let mut db = fixture();
    exec(&mut db, "INSERT INTO t VALUES (5,'e',50)");
    exec(&mut db, "UPDATE t SET v=v+1");
    assert_scalar(&mut db, "SELECT changes()", int(5));
}

/// An UPDATE that matches no rows sets `changes()` to 0.
#[test]
fn changes_after_update_no_match() {
    let mut db = fixture();
    exec(&mut db, "UPDATE t SET v=99 WHERE id=999");
    assert_scalar(&mut db, "SELECT changes()", int(0));
}

/// `changes()` counts rows *matched/processed* by the UPDATE, not only rows whose stored
/// value actually differs. c3ref/changes.html: it returns "the number of rows modified
/// ... by the most recently completed ... UPDATE"; SQLite writes every matched row
/// regardless of whether the new value equals the old, so a no-op `SET v=v` on the two
/// name='a' rows still reports 2. A planner that skipped unchanged rows would report 0.
#[test]
fn changes_after_noop_update_counts_matched_rows() {
    let mut db = fixture();
    exec(&mut db, "UPDATE t SET v=v WHERE name='a'");
    assert_scalar(&mut db, "SELECT changes()", int(2));
}

/// After deleting a single row, `changes()` reports 1.
#[test]
fn changes_after_delete_single_row() {
    let mut db = fixture();
    exec(&mut db, "DELETE FROM t WHERE id=2");
    assert_scalar(&mut db, "SELECT changes()", int(1));
}

/// Deleting every row with no WHERE reports the full count in `changes()`. Per
/// lang_delete.html §4 the truncate optimization still reports the deleted-row count
/// (fixed in version 3.6.5), so this is the row count, not 0. A fifth row is inserted
/// first so the expected 5 differs from the fixture's own INSERT count (4), making the
/// test self-standing rather than coincidentally matching a stale counter.
#[test]
fn changes_after_delete_all_rows() {
    let mut db = fixture();
    exec(&mut db, "INSERT INTO t VALUES (5,'e',50)");
    exec(&mut db, "DELETE FROM t");
    assert_scalar(&mut db, "SELECT changes()", int(5));
}

/// A DELETE that matches no rows sets `changes()` to 0.
#[test]
fn changes_after_delete_no_match() {
    let mut db = fixture();
    exec(&mut db, "DELETE FROM t WHERE id=999");
    assert_scalar(&mut db, "SELECT changes()", int(0));
}
