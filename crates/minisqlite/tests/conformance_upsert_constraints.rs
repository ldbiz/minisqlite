//! Conformance battery: the UPSERT `DO UPDATE` path's interaction with the OTHER
//! constraints — NOT NULL, CHECK, the conflict-target-must-name-a-uniqueness-constraint
//! rule — and its fail-closed handling of the descoped rowid-MOVE corner.
//!
//! `conformance_upsert.rs` deliberately does NOT probe these (its header note §2 says it
//! "deliberately does NOT probe NOT NULL / CHECK interaction"), so they live here in a
//! separate file that only ADDS coverage — it never edits or weakens that battery.
//!
//! Every expected behavior is transcribed from `spec/sqlite-doc/lang_upsert.html`:
//!   * §3: "the DO UPDATE ... conflict-resolution algorithm is always ABORT" ("DO UPDATE OR
//!     ABORT"). So a SET that nulls a NOT NULL column or breaks a CHECK ABORTs and writes
//!     nothing — the same as an ordinary UPDATE, NOT a silent bad-data write.
//!   * §2: the conflict target "specifies ... a uniqueness constraint." A target naming a
//!     column with no PRIMARY KEY / UNIQUE / unique index does not match a constraint, so
//!     real sqlite errors at prepare rather than accepting it.
//!
//! Plus one KNOWN-LIMITATION pin: a `DO UPDATE` that reassigns the INTEGER PRIMARY KEY (a
//! rowid MOVE) is not yet supported and is rejected LOUDLY — the honest fail-closed choice
//! over the silent store corruption a naive move would cause. This documents the descope and
//! guards against a regression that would silently corrupt instead.

mod conformance;
use conformance::*;

/// Assert that executing `sql` FAILS and its error's debug rendering contains `needle`.
/// (Errors are compared by substring, never exact wording — real sqlite distinguishes
/// success from error, not the message text.)
fn assert_exec_err(db: &mut minisqlite::Connection, sql: &str, needle: &str) {
    match db.execute(sql) {
        Ok(_) => panic!("expected an error containing {needle:?}, but it succeeded\n  sql: {sql}"),
        Err(e) => {
            let msg = format!("{e:?}");
            assert!(
                msg.contains(needle),
                "expected error containing {needle:?}\n  sql: {sql}\n  got:  {msg}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// §3 — DO UPDATE OR ABORT: NOT NULL / CHECK on the REWRITTEN row.
// ---------------------------------------------------------------------------

/// A `DO UPDATE` that sets a `NOT NULL` column to NULL ABORTs (it does not store NULL), and
/// leaves the pre-existing row untouched.
#[test]
fn do_update_setting_not_null_column_to_null_aborts() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT NOT NULL)");
    exec(&mut db, "INSERT INTO t VALUES(1, 'a')");
    assert_exec_err(
        &mut db,
        "INSERT INTO t VALUES(1, 'b') ON CONFLICT(id) DO UPDATE SET v = NULL",
        "NOT NULL",
    );
    // Aborted before any write: the original row stands.
    assert_rows(&mut db, "SELECT id, v FROM t", &[vec![int(1), text("a")]]);
}

/// A `DO UPDATE` that makes a `CHECK` fail ABORTs, and leaves the pre-existing row untouched.
#[test]
fn do_update_violating_check_aborts() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v INT CHECK(v > 0))");
    exec(&mut db, "INSERT INTO t VALUES(1, 5)");
    assert_exec_err(
        &mut db,
        "INSERT INTO t VALUES(1, 5) ON CONFLICT(id) DO UPDATE SET v = -1",
        "CHECK",
    );
    assert_rows(&mut db, "SELECT id, v FROM t", &[vec![int(1), int(5)]]);
}

/// The complement: a `DO UPDATE` whose new row SATISFIES both NOT NULL and CHECK succeeds and
/// rewrites in place — so the enforcement above does not false-positive on a valid update. The
/// bystander column `w` (`w + 1`) also confirms the update reads the EXISTING row (5 -> 6).
#[test]
fn do_update_satisfying_constraints_rewrites_in_place() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT NOT NULL, w INT CHECK(w >= 0))");
    exec(&mut db, "INSERT INTO t VALUES(1, 'a', 5)");
    exec(
        &mut db,
        "INSERT INTO t VALUES(1, 'b', 99) ON CONFLICT(id) DO UPDATE SET v = excluded.v, w = w + 1",
    );
    assert_rows(&mut db, "SELECT id, v, w FROM t", &[vec![int(1), text("b"), int(6)]]);
}

// ---------------------------------------------------------------------------
// §2 — the conflict target must name a uniqueness constraint.
// ---------------------------------------------------------------------------

/// `ON CONFLICT(v)` where `v` has no PRIMARY KEY / UNIQUE / unique index does not name a
/// uniqueness constraint — real sqlite errors at prepare, so nothing is inserted.
#[test]
fn on_conflict_target_without_a_uniqueness_constraint_is_rejected() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)");
    assert_exec_err(
        &mut db,
        "INSERT INTO t(id, v) VALUES(1, 'a') ON CONFLICT(v) DO NOTHING",
        "does not match any PRIMARY KEY or UNIQUE constraint",
    );
    assert_rows(&mut db, "SELECT count(*) FROM t", &[vec![int(0)]]);
}

/// The complement: a target that DOES name a UNIQUE column is accepted and drives the upsert
/// (so the validation above does not reject valid non-PK targets).
#[test]
fn on_conflict_target_naming_a_unique_column_is_accepted() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, u INT UNIQUE, v TEXT)");
    exec(&mut db, "INSERT INTO t VALUES(1, 10, 'a')");
    exec(
        &mut db,
        "INSERT INTO t VALUES(2, 10, 'b') ON CONFLICT(u) DO UPDATE SET v = excluded.v",
    );
    // u=10 conflicts the existing row (id 1): DO UPDATE rewrites its v to 'b' in place; no new
    // row (id 2) is inserted.
    assert_rows(&mut db, "SELECT id, u, v FROM t", &[vec![int(1), int(10), text("b")]]);
}

// ---------------------------------------------------------------------------
// Known limitation (fail-closed): rowid-MOVE DO UPDATE is rejected, not corrupting.
// ---------------------------------------------------------------------------

/// Reassigning the INTEGER PRIMARY KEY in a `DO UPDATE` (a rowid MOVE) is not yet supported
/// and is rejected LOUDLY. Supporting a move safely needs the auto-rowid high-water to learn
/// the moved-to rowid; without that a later auto-assigned row in the same statement could be
/// handed the vacated rowid and (because a table insert replaces on a rowid collision) silently
/// overwrite the moved row. This pins the fail-closed behavior AND that the store stays
/// internally consistent (the row reads back identically by table scan and by the unique index).
#[test]
fn do_update_moving_rowid_is_rejected_not_corrupted() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, u INT UNIQUE)");
    exec(&mut db, "INSERT INTO t VALUES(5, 10)");
    // (NULL,10) auto-assigns a rowid, conflicts the existing (5,10) on u, and its DO UPDATE
    // SET id=6 would MOVE row 5 -> 6.
    assert_exec_err(
        &mut db,
        "INSERT INTO t VALUES(NULL, 10) ON CONFLICT(u) DO UPDATE SET id = 6",
        "not yet supported",
    );
    assert_rows(&mut db, "SELECT id, u FROM t", &[vec![int(5), int(10)]]);
    assert_rows(&mut db, "SELECT id, u FROM t WHERE u = 10", &[vec![int(5), int(10)]]);
}
