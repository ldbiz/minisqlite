//! Conformance battery: **UPSERT (`ON CONFLICT ... DO ...`) into a WITHOUT ROWID table**
//! (hole (a) of the WITHOUT ROWID feature set). Real `sqlite3` runs an UPSERT on a
//! WITHOUT ROWID table exactly as on a rowid table; the engine used to refuse every one
//! with a loud "INSERT ... ON CONFLICT (UPSERT) into WITHOUT ROWID table ... is not yet
//! supported".
//!
//! The load-bearing subtlety this battery pins: a WITHOUT ROWID row is identified by its
//! PRIMARY KEY (there is NO rowid), so the conflict is detected against the PRIMARY KEY
//! b-tree and a matched `DO UPDATE` rewrites the existing row via delete-old-key /
//! insert-new-record — the same choreography a WR `UPDATE` uses. The `DO UPDATE` SET/WHERE
//! bind over the combined row `existing(N) ++ excluded(N)` (width `N`, not the rowid path's
//! `N+1`), so a bare / `table.`-qualified column reads the EXISTING row and `excluded.col`
//! the would-be-inserted row (a width slip would swap the two halves and surface here).
//!
//! Every expected value is TRANSCRIBED FROM THE SQLITE DOCS, never from what the engine
//! currently returns; a spec-correct case is never weakened or `#[ignore]`d to make it
//! pass. Each case is its own `#[test]`.
//!
//! Spec sources (all under `spec/sqlite-doc/`):
//!   * `lang_upsert.html`: "If the insert operation would cause the conflict target
//!     uniqueness constraint to fail, then the insert is omitted and the corresponding
//!     DO NOTHING or DO UPDATE operation is performed instead." "Column names in the
//!     expressions of a DO UPDATE refer to the original unchanged value of the column,
//!     before the attempted INSERT. To use the value that would have been inserted ...
//!     add the special 'excluded.' table qualifier." "The only use for the WHERE clause
//!     at the end of the DO UPDATE is to optionally change the DO UPDATE into a no-op."
//!     "Only a single ON CONFLICT clause, specifically the first ON CONFLICT clause with a
//!     matching conflict target, may run for each row." "In the case of a multi-row insert,
//!     the upsert decision is made separately for each row." "The conflict resolution
//!     algorithm for the update operation of the DO UPDATE clause is always ABORT ... If
//!     the DO UPDATE clause encounters any constraint violation, the entire INSERT
//!     statement rolls back and halts." "UPSERT does not intervene for failed NOT NULL,
//!     CHECK, or foreign key constraints."
//!   * `withoutrowid.html`: a WITHOUT ROWID table has no rowid; it stores rows in a
//!     PRIMARY KEY index b-tree (so a plain scan visits rows in PRIMARY KEY order), and an
//!     `INTEGER PRIMARY KEY` on a WR table is an ordinary key column, NOT a rowid alias.

mod conformance;
use conformance::*;

// -------------------------------------------------------------------------------------
// DO NOTHING — a conflict skips the row, leaving the existing row untouched, no error.
// -------------------------------------------------------------------------------------

#[test]
fn pk_conflict_do_nothing_keeps_the_existing_row() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k TEXT PRIMARY KEY, v TEXT) WITHOUT ROWID");
    exec(&mut db, "INSERT INTO t VALUES ('a', 'one')");
    // The second row collides on the PRIMARY KEY 'a'; DO NOTHING drops it silently.
    exec(&mut db, "INSERT INTO t VALUES ('a', 'two') ON CONFLICT(k) DO NOTHING");
    assert_rows(&mut db, "SELECT k, v FROM t", &[vec![text("a"), text("one")]]);
}

#[test]
fn target_omitted_do_nothing_matches_any_conflict() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k TEXT PRIMARY KEY, v TEXT) WITHOUT ROWID");
    exec(&mut db, "INSERT INTO t VALUES ('a', 'one')");
    // A target-omitted ON CONFLICT fires for any uniqueness conflict (lang_upsert.html §2).
    exec(&mut db, "INSERT INTO t VALUES ('a', 'two') ON CONFLICT DO NOTHING");
    assert_rows(&mut db, "SELECT k, v FROM t", &[vec![text("a"), text("one")]]);
}

// -------------------------------------------------------------------------------------
// DO UPDATE — excluded.* is the would-be-inserted value; a bare column is the existing.
// -------------------------------------------------------------------------------------

#[test]
fn pk_conflict_do_update_overwrites_with_excluded() {
    // The Alice/phonenumber example (lang_upsert.html §2.1), on a WITHOUT ROWID table.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE phonebook(name TEXT PRIMARY KEY, phonenumber TEXT) WITHOUT ROWID");
    exec(&mut db, "INSERT INTO phonebook VALUES ('Alice', '704-555-1212')");
    exec(
        &mut db,
        "INSERT INTO phonebook(name, phonenumber) VALUES ('Alice', '555-0000') \
         ON CONFLICT(name) DO UPDATE SET phonenumber = excluded.phonenumber",
    );
    // Existing Alice overwritten with the would-be-inserted phone number; still one row.
    assert_rows(&mut db, "SELECT name, phonenumber FROM phonebook", &[vec![text("Alice"), text("555-0000")]]);
}

#[test]
fn do_update_bare_column_reads_the_original_value() {
    // The vocabulary/count example (lang_upsert.html §2.1): `count+1` reads the EXISTING
    // count. Insert once (DEFAULT 1), then upsert twice to reach 3, proving each DO UPDATE
    // sees the current stored value. The qualified `vocabulary.count` form must also work.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE vocabulary(word TEXT PRIMARY KEY, count INT DEFAULT 1) WITHOUT ROWID");
    exec(&mut db, "INSERT INTO vocabulary(word) VALUES ('jovial') ON CONFLICT(word) DO UPDATE SET count = count + 1");
    assert_rows(&mut db, "SELECT word, count FROM vocabulary", &[vec![text("jovial"), int(1)]]);
    exec(&mut db, "INSERT INTO vocabulary(word) VALUES ('jovial') ON CONFLICT(word) DO UPDATE SET count = count + 1");
    assert_rows(&mut db, "SELECT word, count FROM vocabulary", &[vec![text("jovial"), int(2)]]);
    // The `table.column` form is equivalent (SQLite accepts either; PostgreSQL requires this one).
    exec(&mut db, "INSERT INTO vocabulary(word) VALUES ('jovial') ON CONFLICT(word) DO UPDATE SET count = vocabulary.count + 1");
    assert_rows(&mut db, "SELECT word, count FROM vocabulary", &[vec![text("jovial"), int(3)]]);
}

#[test]
fn do_update_can_read_both_existing_and_excluded_in_one_expression() {
    // A single assignment mixes the existing value (bare `v`) and the would-be-inserted
    // value (`excluded.v`), confirming the combined `existing(N) ++ excluded(N)` binding.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k TEXT PRIMARY KEY, v TEXT) WITHOUT ROWID");
    exec(&mut db, "INSERT INTO t VALUES ('a', 'one')");
    exec(
        &mut db,
        "INSERT INTO t VALUES ('a', 'two') ON CONFLICT(k) DO UPDATE SET v = v || '+' || excluded.v",
    );
    assert_rows(&mut db, "SELECT k, v FROM t", &[vec![text("a"), text("one+two")]]);
}

#[test]
fn target_omitted_do_update_matches_any_conflict() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k TEXT PRIMARY KEY, v TEXT) WITHOUT ROWID");
    exec(&mut db, "INSERT INTO t VALUES ('a', 'one')");
    exec(&mut db, "INSERT INTO t VALUES ('a', 'two') ON CONFLICT DO UPDATE SET v = excluded.v");
    assert_rows(&mut db, "SELECT k, v FROM t", &[vec![text("a"), text("two")]]);
}

#[test]
fn no_conflict_upsert_inserts_normally() {
    // No uniqueness conflict → the row is inserted and the DO UPDATE never runs.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k TEXT PRIMARY KEY, v TEXT) WITHOUT ROWID");
    exec(&mut db, "INSERT INTO t VALUES ('a', 'one')");
    exec(&mut db, "INSERT INTO t VALUES ('b', 'two') ON CONFLICT(k) DO UPDATE SET v = 'never'");
    assert_rows(
        &mut db,
        "SELECT k, v FROM t",
        &[vec![text("a"), text("one")], vec![text("b"), text("two")]],
    );
}

// -------------------------------------------------------------------------------------
// WHERE gate — a false/NULL predicate makes the DO UPDATE a no-op (lang_upsert.html §2.1).
// -------------------------------------------------------------------------------------

#[test]
fn do_update_where_gates_the_rewrite() {
    // The phonebook2/validDate example: only update when the new validDate is newer.
    let mut db = mem();
    exec(
        &mut db,
        "CREATE TABLE phonebook2(name TEXT PRIMARY KEY, phonenumber TEXT, validDate TEXT) WITHOUT ROWID",
    );
    exec(&mut db, "INSERT INTO phonebook2 VALUES ('Alice', '111', '2018-05-08')");
    // Older validDate → WHERE false → no-op (row unchanged).
    exec(
        &mut db,
        "INSERT INTO phonebook2 VALUES ('Alice', '999', '2017-01-01') \
         ON CONFLICT(name) DO UPDATE SET phonenumber = excluded.phonenumber, validDate = excluded.validDate \
         WHERE excluded.validDate > phonebook2.validDate",
    );
    assert_rows(
        &mut db,
        "SELECT name, phonenumber, validDate FROM phonebook2",
        &[vec![text("Alice"), text("111"), text("2018-05-08")]],
    );
    // Newer validDate → WHERE true → the row is updated.
    exec(
        &mut db,
        "INSERT INTO phonebook2 VALUES ('Alice', '222', '2019-01-01') \
         ON CONFLICT(name) DO UPDATE SET phonenumber = excluded.phonenumber, validDate = excluded.validDate \
         WHERE excluded.validDate > phonebook2.validDate",
    );
    assert_rows(
        &mut db,
        "SELECT name, phonenumber, validDate FROM phonebook2",
        &[vec![text("Alice"), text("222"), text("2019-01-01")]],
    );
}

// -------------------------------------------------------------------------------------
// Composite PRIMARY KEY, INTEGER PRIMARY KEY (a WR key column, not a rowid alias).
// -------------------------------------------------------------------------------------

#[test]
fn composite_pk_do_update_matches_the_key_set_order_insensitively() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INT, b INT, v TEXT, PRIMARY KEY(a, b)) WITHOUT ROWID");
    exec(&mut db, "INSERT INTO t VALUES (1, 2, 'x')");
    // ON CONFLICT names the key columns in a DIFFERENT order — a conflict target names a
    // constraint by its column SET, so (b, a) still resolves to the (a, b) primary key.
    exec(&mut db, "INSERT INTO t VALUES (1, 2, 'y') ON CONFLICT(b, a) DO UPDATE SET v = excluded.v");
    assert_rows(&mut db, "SELECT a, b, v FROM t", &[vec![int(1), int(2), text("y")]]);
    // A different composite key does not conflict → inserted as a new row.
    exec(&mut db, "INSERT INTO t VALUES (1, 3, 'z') ON CONFLICT(a, b) DO UPDATE SET v = excluded.v");
    assert_rows(
        &mut db,
        "SELECT a, b, v FROM t ORDER BY a, b",
        &[vec![int(1), int(2), text("y")], vec![int(1), int(3), text("z")]],
    );
}

#[test]
fn integer_primary_key_wr_do_update() {
    // On a WITHOUT ROWID table an INTEGER PRIMARY KEY is an ordinary key column (not a rowid
    // alias — withoutrowid.html), so the UPSERT resolves it exactly like any single-col PK.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT) WITHOUT ROWID");
    exec(&mut db, "INSERT INTO t VALUES (1, 'one')");
    exec(&mut db, "INSERT INTO t VALUES (1, 'two') ON CONFLICT(id) DO UPDATE SET v = excluded.v");
    assert_rows(&mut db, "SELECT id, v FROM t", &[vec![int(1), text("two")]]);
}

// -------------------------------------------------------------------------------------
// A PRIMARY KEY-changing DO UPDATE re-keys the row (supported on WR, unlike the rowid path).
// -------------------------------------------------------------------------------------

#[test]
fn do_update_that_changes_the_primary_key_rekeys_the_row() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k TEXT PRIMARY KEY, v TEXT) WITHOUT ROWID");
    exec(&mut db, "INSERT INTO t VALUES ('a', 'one')");
    // Conflict on 'a'; the DO UPDATE moves the row to PRIMARY KEY 'z'.
    exec(
        &mut db,
        "INSERT INTO t VALUES ('a', 'two') ON CONFLICT(k) DO UPDATE SET k = 'z', v = excluded.v",
    );
    assert_rows(&mut db, "SELECT k, v FROM t", &[vec![text("z"), text("two")]]);
    // The old key is gone (the row moved, not duplicated).
    assert_rows(&mut db, "SELECT count(*) FROM t WHERE k = 'a'", &[vec![int(0)]]);
}

#[test]
fn do_update_pk_change_colliding_with_another_row_aborts_unchanged() {
    // DO UPDATE is OR ABORT (lang_upsert.html §3): a PK change onto an EXISTING different
    // row is a uniqueness violation, so the statement fails and nothing changes.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k TEXT PRIMARY KEY, v TEXT) WITHOUT ROWID");
    exec(&mut db, "INSERT INTO t VALUES ('a', 'one')");
    exec(&mut db, "INSERT INTO t VALUES ('z', 'zed')");
    assert_exec_error(
        &mut db,
        "INSERT INTO t VALUES ('a', 'two') ON CONFLICT(k) DO UPDATE SET k = 'z'",
    );
    // Both original rows survive intact (the failed rewrite wrote nothing).
    assert_rows(
        &mut db,
        "SELECT k, v FROM t ORDER BY k",
        &[vec![text("a"), text("one")], vec![text("z"), text("zed")]],
    );
}

// -------------------------------------------------------------------------------------
// Secondary UNIQUE index as the conflict target.
// -------------------------------------------------------------------------------------

#[test]
fn secondary_unique_index_do_update() {
    // The conflict is on a UNIQUE secondary index (email), not the PRIMARY KEY: the DO
    // UPDATE rewrites the EXISTING row that owns that email, and the candidate 'b' is not
    // inserted (its would-be email duplicated an existing one).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k TEXT PRIMARY KEY, email TEXT, hits INT DEFAULT 0) WITHOUT ROWID");
    exec(&mut db, "CREATE UNIQUE INDEX t_email ON t(email)");
    exec(&mut db, "INSERT INTO t VALUES ('a', 'x@y', 0)");
    exec(
        &mut db,
        "INSERT INTO t VALUES ('b', 'x@y', 0) ON CONFLICT(email) DO UPDATE SET hits = hits + 1",
    );
    // The existing 'a' row (which owns x@y) is the one updated; 'b' never lands.
    assert_rows(&mut db, "SELECT k, email, hits FROM t", &[vec![text("a"), text("x@y"), int(1)]]);
}

#[test]
fn secondary_unique_index_do_nothing() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k TEXT PRIMARY KEY, email TEXT) WITHOUT ROWID");
    exec(&mut db, "CREATE UNIQUE INDEX t_email ON t(email)");
    exec(&mut db, "INSERT INTO t VALUES ('a', 'x@y')");
    exec(&mut db, "INSERT INTO t VALUES ('b', 'x@y') ON CONFLICT(email) DO NOTHING");
    assert_rows(&mut db, "SELECT k, email FROM t", &[vec![text("a"), text("x@y")]]);
}

#[test]
fn do_update_secondary_unique_collision_aborts_unchanged() {
    // A DO UPDATE (OR ABORT) that would push the rewritten row's UNIQUE column onto a
    // DIFFERENT existing row's value is a violation: the whole statement fails, no change.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k TEXT PRIMARY KEY, email TEXT) WITHOUT ROWID");
    exec(&mut db, "CREATE UNIQUE INDEX t_email ON t(email)");
    exec(&mut db, "INSERT INTO t VALUES ('a', 'a@y')");
    exec(&mut db, "INSERT INTO t VALUES ('b', 'b@y')");
    // Conflict on PK 'a'; the SET would move a's email onto b's existing 'b@y'.
    assert_exec_error(
        &mut db,
        "INSERT INTO t VALUES ('a', 'ignored') ON CONFLICT(k) DO UPDATE SET email = 'b@y'",
    );
    assert_rows(
        &mut db,
        "SELECT k, email FROM t ORDER BY k",
        &[vec![text("a"), text("a@y")], vec![text("b"), text("b@y")]],
    );
}

// -------------------------------------------------------------------------------------
// OR ABORT: a DO UPDATE that breaks NOT NULL / CHECK aborts, leaving the row unchanged.
// -------------------------------------------------------------------------------------

#[test]
fn do_update_breaking_not_null_aborts_unchanged() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k TEXT PRIMARY KEY, v TEXT NOT NULL) WITHOUT ROWID");
    exec(&mut db, "INSERT INTO t VALUES ('a', 'one')");
    assert_exec_error(
        &mut db,
        "INSERT INTO t VALUES ('a', 'two') ON CONFLICT(k) DO UPDATE SET v = NULL",
    );
    assert_rows(&mut db, "SELECT k, v FROM t", &[vec![text("a"), text("one")]]);
}

#[test]
fn do_update_breaking_check_aborts_unchanged() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k TEXT PRIMARY KEY, n INT CHECK(n >= 0)) WITHOUT ROWID");
    exec(&mut db, "INSERT INTO t VALUES ('a', 5)");
    assert_exec_error(
        &mut db,
        "INSERT INTO t VALUES ('a', 1) ON CONFLICT(k) DO UPDATE SET n = -1",
    );
    assert_rows(&mut db, "SELECT k, n FROM t", &[vec![text("a"), int(5)]]);
}

// -------------------------------------------------------------------------------------
// Chained clauses: the FIRST clause with a matching target fires; later ones are bypassed.
// -------------------------------------------------------------------------------------

#[test]
fn chained_clauses_first_matching_target_wins() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k TEXT PRIMARY KEY, u TEXT, v TEXT) WITHOUT ROWID");
    exec(&mut db, "CREATE UNIQUE INDEX t_u ON t(u)");
    exec(&mut db, "INSERT INTO t VALUES ('a', 'uu', 'one')");
    // The candidate conflicts on the PRIMARY KEY 'a' only (its u='xx' does not duplicate
    // 'uu'). The u-targeted clause does not match; the k-targeted clause does → v='by_k'.
    exec(
        &mut db,
        "INSERT INTO t VALUES ('a', 'xx', 'X') \
         ON CONFLICT(u) DO UPDATE SET v = 'by_u' \
         ON CONFLICT(k) DO UPDATE SET v = 'by_k'",
    );
    assert_rows(&mut db, "SELECT k, u, v FROM t", &[vec![text("a"), text("uu"), text("by_k")]]);
}

// -------------------------------------------------------------------------------------
// Multi-row INSERT: the upsert decision is made separately per row (lang_upsert.html §2),
// so a later row conflicting with an earlier one just inserted updates it.
// -------------------------------------------------------------------------------------

#[test]
fn multi_row_second_row_updates_the_first() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k TEXT PRIMARY KEY, v TEXT) WITHOUT ROWID");
    exec(
        &mut db,
        "INSERT INTO t VALUES ('a', 'one'), ('a', 'two') ON CONFLICT(k) DO UPDATE SET v = excluded.v",
    );
    // Row 1 inserts ('a','one'); row 2 conflicts with it and updates v to 'two'.
    assert_rows(&mut db, "SELECT k, v FROM t", &[vec![text("a"), text("two")]]);
}

// -------------------------------------------------------------------------------------
// RETURNING reflects the post-update row; generated columns recompute on DO UPDATE.
// -------------------------------------------------------------------------------------

#[test]
fn do_update_returning_gives_the_post_update_values() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k TEXT PRIMARY KEY, v TEXT) WITHOUT ROWID");
    exec(&mut db, "INSERT INTO t VALUES ('a', 'one')");
    let qr = query(
        &mut db,
        "INSERT INTO t VALUES ('a', 'two') ON CONFLICT(k) DO UPDATE SET v = excluded.v RETURNING k, v",
    );
    let mut rows = qr.rows;
    assert!(
        rows.len() == 1 && rows[0].len() == 2,
        "RETURNING should yield exactly one 2-column row, got {rows:?}"
    );
    let row = rows.pop().unwrap();
    assert!(value_eq(&row[0], &text("a")) && value_eq(&row[1], &text("two")), "got {row:?}");
}

#[test]
fn do_update_recomputes_a_stored_generated_column() {
    // A STORED generated column depends on a base column; a DO UPDATE that changes the base
    // must recompute the generated value (gencol.html), just as a plain UPDATE would.
    let mut db = mem();
    exec(
        &mut db,
        "CREATE TABLE t(k TEXT PRIMARY KEY, base INT, g INT AS (base * 2) STORED) WITHOUT ROWID",
    );
    exec(&mut db, "INSERT INTO t(k, base) VALUES ('a', 5)");
    assert_rows(&mut db, "SELECT k, base, g FROM t", &[vec![text("a"), int(5), int(10)]]);
    exec(
        &mut db,
        "INSERT INTO t(k, base) VALUES ('a', 100) ON CONFLICT(k) DO UPDATE SET base = excluded.base",
    );
    assert_rows(&mut db, "SELECT k, base, g FROM t", &[vec![text("a"), int(100), int(200)]]);
}
