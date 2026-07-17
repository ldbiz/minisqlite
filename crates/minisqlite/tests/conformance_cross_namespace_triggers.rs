//! Conformance battery: **cross-namespace trigger / INSTEAD-OF action routing**,
//! exercised through the pinned `minisqlite::Connection` facade.
//!
//! A fired trigger action (or an `INSTEAD OF` view-trigger action) must write ITS
//! OWN namespace — the store the action's target table resolves to — not a single
//! pinned namespace. Every expectation here is derived from the SQLite docs in
//! `spec/sqlite-doc/`, never from what the engine happens to return; a failing case
//! is the intended signal the engine diverges from the spec.
//!
//! The final section verifies autocommit STATEMENT-ABORT ATOMICITY across namespaces: an
//! aborted autocommit statement backs out its ENTIRE effect — including a trigger's /
//! INSTEAD-OF's cross-namespace writes — because the whole statement is one atomic unit
//! (lang_transaction.html: a statement is atomic; lang_createtrigger.html §6: a fired
//! trigger's actions run within the firing statement, so a `RAISE(ABORT)` or a
//! constraint failure "performs the ON CONFLICT processing" and undoes the statement's
//! changes). Those writes reach stores other than the top-level DML target at runtime,
//! so the engine's implicit transaction must bracket EVERY live namespace for the abort
//! to be atomic.
//!
//! Spec sources (`spec/sqlite-doc/`):
//!   * `lang_createtrigger.html` §"unqualified table name": a trigger action's
//!     INSERT/UPDATE/DELETE target MUST be UNQUALIFIED (`tab`, never `db.tab`); the
//!     name resolves by the normal search order.
//!   * `lang_createtrigger.html` §scope: a NON-TEMP trigger may only modify a table
//!     in the SAME database as the table/view it is attached to; a TEMP trigger "is
//!     allowed to query or modify any table in any ATTACH-ed database" — so a TEMP
//!     trigger is the reachable vehicle for a cross-namespace action.
//!   * `lang_naming.html`: an unqualified name resolves `temp`, then `main`, then
//!     attached databases in attach order. So which store an action writes is fully
//!     determined by which namespaces hold a table of that name.
//!   * `lang_attach.html`: `ATTACH ':memory:' AS aux` adds an in-memory database.
//!
//! DISCRIMINATOR vs GUARD. Before this fix the fire phase pinned a SINGLE namespace
//! (INSTEAD OF pinned `main`; a regular trigger pinned the firing DML's own store),
//! so any action naming a DIFFERENT namespace FAILED CLOSED with a "cannot reach
//! namespace" error. The DISCRIMINATOR tests below are exactly those cross-namespace
//! actions: each errored before the fix and now writes the correct store. The GUARD
//! tests are the cases that already worked (an action writing the pinned namespace,
//! a same-namespace trigger) and must keep working under per-action routing.
//!
//! To PROVE which store was written, data is distinct per namespace and reads are
//! schema-qualified (`main.x`, `temp.x`, `aux.x`); where a same-named spacer would
//! not change name resolution, one is planted in the "wrong" store and asserted
//! untouched.
//!
//! COVERAGE BOUNDARY. Every cross-namespace ACTION direction is exercised: an INSTEAD OF
//! redirect on a temp view writing temp or main, on an attached view writing that attached db
//! OR writing main; and a TEMP trigger writing main or an attached db, across
//! INSERT/UPDATE/DELETE and both BEFORE/AFTER — including the two directions that require a
//! TEMP trigger ATTACHED to a non-temp object (spec §"TEMP Triggers on Non-TEMP Tables"):
//! `instead_of_temp_trigger_on_attached_view_action_writes_main` and
//! `temp_trigger_on_main_table_action_writes_attached`. This file focuses on the fire-phase
//! per-ACTION routing; the ATTACHMENT side of that feature (binding a TEMP trigger to a
//! main/attached table or view, shadowing, non-persistence, DROP cascade) is proven in the
//! sibling `conformance_temp_trigger_on_nontemp.rs`.

mod conformance;
use conformance::*;

// ===========================================================================
// INSTEAD OF view triggers — DISCRIMINATORS (errored under the old `main` pin).
// ===========================================================================

#[test]
fn instead_of_temp_view_action_writes_temp_table() {
    // A TEMP view's INSTEAD OF INSERT trigger whose action writes a TEMP table.
    // `main.tdata` holds a spacer; `temp.tdata` shadows it, so the unqualified
    // `INSERT INTO tdata` in the trigger body resolves to TEMP (search order
    // temp->main). The OLD `main` pin sent this write to `main` (or errored); the
    // fix routes it to `temp`. Proof: `temp.tdata` gains the row, `main.tdata` does
    // not.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE tdata(x)"); // main.tdata
    exec(&mut db, "INSERT INTO tdata VALUES (111)"); // main spacer
    exec(&mut db, "CREATE TEMP TABLE tdata(x)"); // temp.tdata shadows main
    exec(&mut db, "CREATE TEMP VIEW tv AS SELECT x FROM tdata");
    // A trigger on a TEMP view lives in the temp schema, so it is created as TEMP;
    // its unqualified `tv` resolves there.
    exec(
        &mut db,
        "CREATE TEMP TRIGGER tvi INSTEAD OF INSERT ON tv \
         BEGIN INSERT INTO tdata VALUES (NEW.x); END",
    );

    exec(&mut db, "INSERT INTO tv VALUES (777)");

    assert_rows(&mut db, "SELECT x FROM temp.tdata", &[vec![int(777)]]);
    // The main spacer is untouched — the write did NOT land in `main`.
    assert_rows(&mut db, "SELECT x FROM main.tdata", &[vec![int(111)]]);
}

#[test]
fn instead_of_attached_view_action_writes_attached_table() {
    // An INSTEAD OF INSERT trigger on an ATTACHED view whose action writes the
    // ATTACHED database's table. `adata` exists only in `aux`, so the unqualified
    // action resolves there (temp->main->aux). The old `main` pin could not reach
    // `aux` and failed closed; the fix writes `aux.adata`.
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.adata(x)");
    exec(&mut db, "CREATE VIEW aux.av AS SELECT x FROM aux.adata");
    // The trigger lives in the `aux` schema (its qualified name), so both its ON
    // target `av` and its action `adata` resolve within `aux` (a non-TEMP trigger is
    // confined to its own database). The action's stamped db is `aux`.
    exec(
        &mut db,
        "CREATE TRIGGER aux.avi INSTEAD OF INSERT ON av \
         BEGIN INSERT INTO adata VALUES (NEW.x); END",
    );

    exec(&mut db, "INSERT INTO aux.av VALUES (888)");

    assert_rows(&mut db, "SELECT x FROM aux.adata", &[vec![int(888)]]);
}

// ===========================================================================
// INSTEAD OF view triggers — GUARDS (worked under the old pin; must still work).
// ===========================================================================

#[test]
fn instead_of_main_view_action_writes_main_table() {
    // The plain main-schema case: a main view, a main INSTEAD OF trigger, a main
    // base table. This always worked and must keep working (regression guard).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE base(x)");
    exec(&mut db, "CREATE VIEW v AS SELECT x FROM base");
    exec(
        &mut db,
        "CREATE TRIGGER vi INSTEAD OF INSERT ON v \
         BEGIN INSERT INTO base VALUES (NEW.x); END",
    );

    exec(&mut db, "INSERT INTO v VALUES (42)");

    assert_rows(&mut db, "SELECT x FROM main.base", &[vec![int(42)]]);
}

#[test]
fn instead_of_temp_view_action_writes_main_table() {
    // A TEMP view's INSTEAD OF trigger whose action writes a MAIN table. `mdata`
    // exists only in `main`, so the unqualified action resolves to `main`. This
    // WORKED under the old `main` pin and must still work: per-action routing must
    // still reach `main` correctly (it is not a regression to lose the pin).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE mdata(x)"); // main only
    exec(&mut db, "CREATE TEMP VIEW mv AS SELECT x FROM mdata");
    exec(
        &mut db,
        "CREATE TEMP TRIGGER mvi INSTEAD OF INSERT ON mv \
         BEGIN INSERT INTO mdata VALUES (NEW.x); END",
    );

    exec(&mut db, "INSERT INTO mv VALUES (555)");

    assert_rows(&mut db, "SELECT x FROM main.mdata", &[vec![int(555)]]);
}

#[test]
fn instead_of_temp_trigger_on_attached_view_action_writes_main() {
    // An attached-view INSTEAD OF action that writes MAIN. Reachable via "TEMP Triggers on
    // Non-TEMP Tables" (spec §7): only a TEMP trigger may cross namespaces, and a TEMP trigger
    // may now attach to a non-temp (here attached) view via a schema-qualified `ON aux.av`.
    // The redirect's action target `mdata` is UNQUALIFIED (§2.1) and exists only in `main`, so
    // it resolves to `main` — the per-action routing must reach `main` from a trigger bound to
    // `aux`. (Attachment of this cross-namespace TEMP trigger is proven in
    // `conformance_temp_trigger_on_nontemp.rs`; here the focus is the ACTION crossing to main.)
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE mdata(x)"); // main.mdata — the sole holder, so the action resolves to main
    exec(&mut db, "CREATE TABLE aux.abase(x)");
    exec(&mut db, "CREATE VIEW aux.av AS SELECT x FROM aux.abase");
    exec(
        &mut db,
        "CREATE TEMP TRIGGER avm INSTEAD OF INSERT ON aux.av \
         BEGIN INSERT INTO mdata VALUES (NEW.x); END",
    );

    exec(&mut db, "INSERT INTO aux.av VALUES (321)");

    assert_rows(&mut db, "SELECT x FROM main.mdata", &[vec![int(321)]]);
    assert_rows(&mut db, "SELECT x FROM aux.abase", &[]); // INSTEAD OF -> no actual base insert
}

// ===========================================================================
// Regular (BEFORE/AFTER) triggers — DISCRIMINATORS.
// A TEMP trigger's body may modify any attached database; before the fix the
// action was pinned to the FIRING DML's own store, so any other-namespace target
// failed closed. Each op (INSERT/UPDATE/DELETE) and each timing (BEFORE/AFTER)
// exercises a distinct fire site that now routes per-action.
// ===========================================================================

#[test]
fn temp_trigger_after_insert_on_temp_table_writes_main() {
    // Firing DML is on TEMP `tsrc`; the AFTER INSERT action writes MAIN `mlog`
    // (only `main` holds `mlog`). Old code pinned the action to `temp` and failed;
    // the fix writes `main`. The spacer proves the write landed in `main`.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE mlog(x)"); // main
    exec(&mut db, "INSERT INTO mlog VALUES (111)"); // main spacer
    exec(&mut db, "CREATE TEMP TABLE tsrc(x)"); // temp
    exec(
        &mut db,
        "CREATE TEMP TRIGGER ti AFTER INSERT ON tsrc \
         BEGIN INSERT INTO mlog VALUES (NEW.x); END",
    );

    exec(&mut db, "INSERT INTO tsrc VALUES (333)");

    assert_rows(&mut db, "SELECT x FROM main.mlog ORDER BY x", &[vec![int(111)], vec![int(333)]]);
    assert_rows(&mut db, "SELECT x FROM temp.tsrc", &[vec![int(333)]]);
}

#[test]
fn temp_trigger_before_insert_on_temp_table_writes_main() {
    // BEFORE-timing variant of the INSERT fire site: a TEMP BEFORE INSERT trigger
    // on `tsrc` writes MAIN `mlog`. Covers the BEFORE fire path in the insert op.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE mlog(x)");
    exec(&mut db, "CREATE TEMP TABLE tsrc(x)");
    exec(
        &mut db,
        "CREATE TEMP TRIGGER tb BEFORE INSERT ON tsrc \
         BEGIN INSERT INTO mlog VALUES (NEW.x); END",
    );

    exec(&mut db, "INSERT INTO tsrc VALUES (222)");

    assert_rows(&mut db, "SELECT x FROM main.mlog", &[vec![int(222)]]);
    assert_rows(&mut db, "SELECT x FROM temp.tsrc", &[vec![int(222)]]);
}

#[test]
fn temp_trigger_after_update_on_temp_table_writes_main() {
    // The UPDATE fire site: a TEMP AFTER UPDATE trigger on `tsrc` writes MAIN
    // `mlog` with the NEW value. Covers the update op's cross-namespace fire.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE mlog(x)");
    exec(&mut db, "CREATE TEMP TABLE tsrc(x)");
    exec(&mut db, "INSERT INTO tsrc VALUES (10)");
    exec(
        &mut db,
        "CREATE TEMP TRIGGER tu AFTER UPDATE ON tsrc \
         BEGIN INSERT INTO mlog VALUES (NEW.x); END",
    );

    exec(&mut db, "UPDATE tsrc SET x = 20");

    assert_rows(&mut db, "SELECT x FROM main.mlog", &[vec![int(20)]]);
    assert_rows(&mut db, "SELECT x FROM temp.tsrc", &[vec![int(20)]]);
}

#[test]
fn temp_trigger_after_delete_on_temp_table_writes_main() {
    // The DELETE fire site: a TEMP AFTER DELETE trigger on `tsrc` writes MAIN
    // `mlog` with the OLD value. Covers the delete op's cross-namespace fire.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE mlog(x)");
    exec(&mut db, "CREATE TEMP TABLE tsrc(x)");
    exec(&mut db, "INSERT INTO tsrc VALUES (30)");
    exec(
        &mut db,
        "CREATE TEMP TRIGGER td AFTER DELETE ON tsrc \
         BEGIN INSERT INTO mlog VALUES (OLD.x); END",
    );

    exec(&mut db, "DELETE FROM tsrc");

    assert_rows(&mut db, "SELECT x FROM main.mlog", &[vec![int(30)]]);
    assert_rows(&mut db, "SELECT x FROM temp.tsrc", &[]);
}

#[test]
fn temp_trigger_on_main_table_action_writes_attached() {
    // The reverse regular-trigger direction: firing DML on a MAIN table whose trigger action
    // writes a DIFFERENT namespace. Reachable via "TEMP Triggers on Non-TEMP Tables" (spec §7):
    // a TEMP AFTER INSERT trigger BOUND to `main.msrc` whose UNQUALIFIED action target `alog`
    // exists only in `aux`, so it resolves to the attached database. Proves the per-action
    // routing reaches an attached store from a trigger bound to `main`.
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE msrc(x)"); // main — firing DML target
    exec(&mut db, "CREATE TABLE aux.alog(x)"); // attached — action target (sole holder)
    exec(
        &mut db,
        "CREATE TEMP TRIGGER tma AFTER INSERT ON main.msrc \
         BEGIN INSERT INTO alog VALUES (NEW.x); END",
    );

    exec(&mut db, "INSERT INTO msrc VALUES (654)");

    assert_rows(&mut db, "SELECT x FROM aux.alog", &[vec![int(654)]]);
    assert_rows(&mut db, "SELECT x FROM main.msrc", &[vec![int(654)]]);
}

#[test]
fn temp_trigger_on_temp_table_writes_attached() {
    // Firing DML is on TEMP `tsrc`; the TEMP AFTER INSERT trigger writes an
    // ATTACHED table `alog` (only `aux` holds it, so the action resolves to `aux`).
    // A third distinct namespace, proving routing is general — not a two-way toggle.
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.alog(x)");
    exec(&mut db, "CREATE TEMP TABLE tsrc(x)");
    exec(
        &mut db,
        "CREATE TEMP TRIGGER ta AFTER INSERT ON tsrc \
         BEGIN INSERT INTO alog VALUES (NEW.x); END",
    );

    exec(&mut db, "INSERT INTO tsrc VALUES (999)");

    assert_rows(&mut db, "SELECT x FROM aux.alog", &[vec![int(999)]]);
}

// ===========================================================================
// Regular triggers — GUARD (the common same-namespace case must still fire).
// ===========================================================================

#[test]
fn same_namespace_main_trigger_still_fires() {
    // A main trigger on a main table writing a main table — the overwhelmingly
    // common case. Per-action routing must not disturb it (regression guard).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x)");
    exec(&mut db, "CREATE TABLE tlog(x)");
    exec(
        &mut db,
        "CREATE TRIGGER g AFTER INSERT ON t \
         BEGIN INSERT INTO tlog VALUES (NEW.x); END",
    );

    exec(&mut db, "INSERT INTO t VALUES (1), (2)");

    assert_rows(&mut db, "SELECT x FROM tlog ORDER BY x", &[vec![int(1)], vec![int(2)]]);
}

// ===========================================================================
// PER-ACTION (not per-program) routing: one trigger body, two actions, two
// namespaces. This is the defining property of the change — routing is derived
// from EACH action's own `node.db`, not once per firing program.
// ===========================================================================

#[test]
fn one_trigger_body_routes_each_action_to_its_own_namespace() {
    // A single TEMP trigger whose body writes MAIN then ATTACHED in the same fire.
    // If routing were per-PROGRAM (one namespace per firing) instead of per-ACTION,
    // one of the two writes would land in the wrong store; asserting BOTH stores
    // pins that each action routes independently within one trigger body.
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE mlog(x)"); // main
    exec(&mut db, "CREATE TABLE aux.alog(x)"); // attached
    exec(&mut db, "CREATE TEMP TABLE tsrc(x)"); // temp
    exec(
        &mut db,
        "CREATE TEMP TRIGGER two AFTER INSERT ON tsrc BEGIN \
         INSERT INTO mlog VALUES (NEW.x); \
         INSERT INTO alog VALUES (NEW.x + 1); END",
    );

    exec(&mut db, "INSERT INTO tsrc VALUES (10)");

    assert_rows(&mut db, "SELECT x FROM main.mlog", &[vec![int(10)]]);
    assert_rows(&mut db, "SELECT x FROM aux.alog", &[vec![int(11)]]);
}

// ===========================================================================
// AUTOCOMMIT STATEMENT-ABORT ATOMICITY across namespaces.
//
// An aborted autocommit statement backs out its ENTIRE effect — a fired trigger's /
// INSTEAD-OF's cross-namespace writes included — because the statement is one atomic
// unit (lang_transaction.html; lang_createtrigger.html §6: a RAISE(ABORT) or a
// constraint failure inside a fired trigger performs the ON CONFLICT processing and
// undoes the statement's changes). A fired action reaches a namespace OTHER than the
// top-level DML target at RUNTIME (per-action routing, nested-trigger recursion, FK
// cascades), so the engine's implicit transaction must bracket EVERY live namespace,
// not just the target — otherwise an un-bracketed store's copy-on-write autocommit
// persists the write immediately and the abort cannot reach it.
//
// The DISCRIMINATORS below abort a statement AFTER a cross-namespace write has
// succeeded and assert every touched store rolls back; the GUARDS fence the
// boundaries — OR FAIL still keeps its pre-conflict rows across namespaces, the happy
// path still commits, a same-namespace abort still rolls back, and the single-`main`
// hot path is unchanged. To catch a PARTIAL fix, each covers a distinct fire site
// (INSERT/UPDATE/DELETE, INSTEAD OF) and a distinct abort cause (CHECK, UNIQUE, NOT
// NULL, FK, RAISE(ABORT)).
// ===========================================================================

/// Regular cross-namespace trigger (TEMP trigger on a temp table → main) under a
/// mid-statement ABORT: the trigger fired for the pre-abort rows, writing `main.mlog`;
/// the abort must back those cross-namespace writes out with the rest of the statement,
/// leaving both the firing temp table and `main.mlog` empty.
#[test]
fn autocommit_regular_trigger_cross_namespace_abort_rolls_back() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE mlog(x)"); // main
    exec(&mut db, "CREATE TEMP TABLE tsrc(x CHECK (x <> 3))"); // temp
    exec(
        &mut db,
        "CREATE TEMP TRIGGER ta AFTER INSERT ON tsrc \
         BEGIN INSERT INTO mlog VALUES (NEW.x); END",
    );

    // Rows (1),(2) fire the trigger (writing main.mlog); row (3) violates the temp CHECK,
    // so the whole multi-row INSERT ABORTs.
    assert_exec_error(&mut db, "INSERT INTO tsrc VALUES (1),(2),(3)");

    // The firing temp table (the top-level target) rolls back …
    assert_rows(&mut db, "SELECT x FROM temp.tsrc", &[]);
    // … and so does the trigger's cross-namespace write to main.
    assert_rows(&mut db, "SELECT x FROM main.mlog", &[]);
}

/// INSTEAD OF path under a mid-statement ABORT: a temp view's redirect writes
/// `temp.tdata`; the abort must back the redirected writes out, leaving it empty. (The
/// autocommit bracket spans every live namespace, so the temp redirect is rolled back
/// even though the top-level statement targets a view — a redirect is not the plan's
/// nominal target.)
#[test]
fn autocommit_instead_of_cross_namespace_abort_rolls_back() {
    let mut db = mem();
    exec(&mut db, "CREATE TEMP TABLE tdata(x CHECK (x <> 3))");
    exec(&mut db, "CREATE TEMP VIEW tv AS SELECT x FROM tdata");
    exec(
        &mut db,
        "CREATE TEMP TRIGGER tvi INSTEAD OF INSERT ON tv \
         BEGIN INSERT INTO tdata VALUES (NEW.x); END",
    );

    // Rows (1),(2) redirect into temp.tdata; row (3)'s redirect violates the CHECK,
    // aborting the whole statement.
    assert_exec_error(&mut db, "INSERT INTO tv VALUES (1),(2),(3)");

    assert_rows(&mut db, "SELECT x FROM temp.tdata", &[]);
}

/// UPDATE fire site: a TEMP AFTER UPDATE trigger writes `main.mlog` (a SUCCEEDING
/// cross-namespace write) and then `RAISE(ABORT)`s. The abort must undo BOTH the
/// trigger's committed-looking main write and the firing UPDATE itself. A single firing
/// row makes the "write-then-abort" order deterministic (no dependence on multi-row
/// processing order).
#[test]
fn autocommit_update_fired_action_cross_namespace_abort_rolls_back() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE mlog(x)"); // main
    exec(&mut db, "CREATE TEMP TABLE tsrc(x)"); // temp
    exec(&mut db, "INSERT INTO tsrc VALUES (10)");
    exec(
        &mut db,
        "CREATE TEMP TRIGGER tu AFTER UPDATE ON tsrc BEGIN \
         INSERT INTO mlog VALUES (NEW.x); \
         SELECT RAISE(ABORT, 'stop'); END",
    );

    assert_exec_error(&mut db, "UPDATE tsrc SET x = 20");

    // The cross-namespace write mlog(20) succeeded before the RAISE, then rolled back.
    assert_rows(&mut db, "SELECT x FROM main.mlog", &[]);
    // The firing UPDATE is undone: tsrc keeps its original value.
    assert_rows(&mut db, "SELECT x FROM temp.tsrc", &[vec![int(10)]]);
}

/// DELETE fire site: a TEMP AFTER DELETE trigger writes `main.mlog` with the OLD value
/// (a succeeding cross-namespace write) and then `RAISE(ABORT)`s. The abort must undo
/// the main write and re-instate the deleted row. Single firing row for determinism.
#[test]
fn autocommit_delete_fired_action_cross_namespace_abort_rolls_back() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE mlog(x)"); // main
    exec(&mut db, "CREATE TEMP TABLE tsrc(x)"); // temp
    exec(&mut db, "INSERT INTO tsrc VALUES (30)");
    exec(
        &mut db,
        "CREATE TEMP TRIGGER td AFTER DELETE ON tsrc BEGIN \
         INSERT INTO mlog VALUES (OLD.x); \
         SELECT RAISE(ABORT, 'stop'); END",
    );

    assert_exec_error(&mut db, "DELETE FROM tsrc");

    assert_rows(&mut db, "SELECT x FROM main.mlog", &[]);
    // The DELETE is undone: the row survives.
    assert_rows(&mut db, "SELECT x FROM temp.tsrc", &[vec![int(30)]]);
}

/// RAISE(ABORT) as the direct abort cause on the INSERT fire site: the trigger writes
/// `main.mlog` (succeeds) and then raises. The abort undoes the main write and the
/// firing insert. This isolates the RAISE(ABORT) cause from the constraint-based cases.
#[test]
fn autocommit_raise_abort_in_trigger_body_rolls_back_cross_namespace() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE mlog(x)"); // main
    exec(&mut db, "CREATE TEMP TABLE tsrc(x)"); // temp
    exec(
        &mut db,
        "CREATE TEMP TRIGGER ta AFTER INSERT ON tsrc BEGIN \
         INSERT INTO mlog VALUES (NEW.x); \
         SELECT RAISE(ABORT, 'stop'); END",
    );

    assert_exec_error(&mut db, "INSERT INTO tsrc VALUES (5)");

    assert_rows(&mut db, "SELECT x FROM main.mlog", &[]);
    assert_rows(&mut db, "SELECT x FROM temp.tsrc", &[]);
}

/// A UNIQUE violation (not CHECK) as the abort cause. Rows (1),(2) fire the trigger,
/// writing `main.mlog`; row (1) again violates `tsrc`'s UNIQUE, aborting the statement.
/// The trigger's cross-namespace writes for the pre-abort rows must roll back too.
#[test]
fn autocommit_unique_violation_abort_rolls_back_cross_namespace() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE mlog(x)"); // main
    exec(&mut db, "CREATE TEMP TABLE tsrc(x UNIQUE)"); // temp
    exec(
        &mut db,
        "CREATE TEMP TRIGGER ta AFTER INSERT ON tsrc \
         BEGIN INSERT INTO mlog VALUES (NEW.x); END",
    );

    assert_exec_error(&mut db, "INSERT INTO tsrc VALUES (1),(2),(1)");

    assert_rows(&mut db, "SELECT x FROM main.mlog", &[]);
    assert_rows(&mut db, "SELECT x FROM temp.tsrc", &[]);
}

/// A NOT NULL violation as the abort cause. Rows (1),(2) fire the trigger; row (NULL)
/// violates `tsrc`'s NOT NULL, aborting the statement — the pre-abort cross-namespace
/// writes roll back.
#[test]
fn autocommit_not_null_violation_abort_rolls_back_cross_namespace() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE mlog(x)"); // main
    exec(&mut db, "CREATE TEMP TABLE tsrc(x NOT NULL)"); // temp
    exec(
        &mut db,
        "CREATE TEMP TRIGGER ta AFTER INSERT ON tsrc \
         BEGIN INSERT INTO mlog VALUES (NEW.x); END",
    );

    assert_exec_error(&mut db, "INSERT INTO tsrc VALUES (1),(2),(NULL)");

    assert_rows(&mut db, "SELECT x FROM main.mlog", &[]);
    assert_rows(&mut db, "SELECT x FROM temp.tsrc", &[]);
}

/// A FOREIGN KEY violation as the abort cause (`PRAGMA foreign_keys = ON`; SQLite's FK
/// enforcement is per-connection and confined to one database, so parent/child live in
/// `main`). A TEMP trigger on `temp.tsrc` inserts each fired value into the FK child
/// `main.child`; rows (1),(2) reference existing parents and write, row (9) has no
/// parent → the child insert violates the FK and aborts. The child rows written for the
/// pre-abort rows (a cross-namespace write) must roll back, and the firing temp inserts
/// too.
#[test]
fn autocommit_fk_violation_abort_rolls_back_cross_namespace() {
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE parent(id INTEGER PRIMARY KEY)"); // main
    exec(&mut db, "INSERT INTO parent VALUES (1),(2)");
    exec(&mut db, "CREATE TABLE child(x INTEGER REFERENCES parent(id))"); // main FK child
    exec(&mut db, "CREATE TEMP TABLE tsrc(x)"); // temp firing table
    exec(
        &mut db,
        "CREATE TEMP TRIGGER ta AFTER INSERT ON tsrc \
         BEGIN INSERT INTO child VALUES (NEW.x); END",
    );

    assert_exec_error(&mut db, "INSERT INTO tsrc VALUES (1),(2),(9)");

    // The cross-namespace child rows written for the pre-abort rows (1),(2) roll back.
    assert_rows(&mut db, "SELECT x FROM main.child", &[]);
    assert_rows(&mut db, "SELECT x FROM temp.tsrc", &[]);
    // The FK parent is untouched by the aborted statement.
    assert_rows(&mut db, "SELECT id FROM main.parent ORDER BY id", &[vec![int(1)], vec![int(2)]]);
}

/// INSTEAD OF redirect to a `main` table under abort: a temp view redirects into
/// `main.mdata`; rows (1),(2) redirect and write, row (3) violates `main.mdata`'s CHECK
/// and aborts the statement, so the redirected `main` writes must roll back.
///
/// This is a GUARD, not a discriminator of the historical bug: an INSTEAD-OF redirect
/// resolves to `main` (index 0), which the OLD single-target bracket (`InsteadOf → main`)
/// already covered, so this direction rolled back correctly both before AND after the fix
/// (it is not among the cases that go red on a single-target revert). It pins that the
/// main-redirect direction still rolls back under bracket-all. The discriminating
/// INSTEAD-OF case is `autocommit_instead_of_cross_namespace_abort_rolls_back` (a
/// temp→temp redirect, whose write to `temp` — index 1 — the old `main` pin missed).
#[test]
fn autocommit_instead_of_redirect_to_main_abort_rolls_back() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE mdata(x CHECK (x <> 3))"); // main, with CHECK
    exec(&mut db, "CREATE TEMP VIEW mv AS SELECT x FROM mdata"); // temp view over main
    exec(
        &mut db,
        "CREATE TEMP TRIGGER mvi INSTEAD OF INSERT ON mv \
         BEGIN INSERT INTO mdata VALUES (NEW.x); END",
    );

    assert_exec_error(&mut db, "INSERT INTO mv VALUES (1),(2),(3)");

    assert_rows(&mut db, "SELECT x FROM main.mdata", &[]);
}

/// NESTED-TRIGGER recursion reaching a namespace the top-level plan never names — the
/// case that justifies bracketing ALL live pagers over a static plan write-set walk.
/// `PRAGMA recursive_triggers = ON` lets a fired trigger's action fire the target's OWN
/// triggers at RUNTIME (lang_createtrigger.html §recursion; pragma.html). Chain: a temp
/// trigger on `temp.tsrc` writes `temp.tmid` (a DIRECT action, still in `temp`); a temp
/// trigger on `temp.tmid` then writes `main.m` — but that second trigger is discovered
/// only at runtime (trigger depth 2), so it is NOT in the statement's compiled one-level
/// action set. Rows (1),(2) drive the full chain into `main.m`; row (3) trips `temp.tsrc`'s
/// CHECK and aborts.
///
/// A static write-set walk of the compiled plan would see only `{temp}` (the target plus
/// the one-level action into `temp.tmid`) and leave `main.m` un-bracketed, so the nested
/// write would survive the abort. Bracketing every live namespace covers `main` no matter
/// how deep the chain that reaches it — RED under both the old single-target bracket and a
/// hypothetical static write-set walk, GREEN only when all live namespaces are bracketed.
#[test]
fn autocommit_nested_trigger_deep_chain_cross_namespace_abort_rolls_back() {
    let mut db = mem();
    exec(&mut db, "PRAGMA recursive_triggers = ON");
    exec(&mut db, "CREATE TABLE m(x)"); // main — reached ONLY by the depth-2 nested trigger
    exec(&mut db, "CREATE TEMP TABLE tsrc(x CHECK (x <> 3))"); // temp, top-level target
    exec(&mut db, "CREATE TEMP TABLE tmid(x)"); // temp, the direct action's target
    exec(
        &mut db,
        "CREATE TEMP TRIGGER ta AFTER INSERT ON tsrc \
         BEGIN INSERT INTO tmid VALUES (NEW.x); END",
    );
    exec(
        &mut db,
        "CREATE TEMP TRIGGER tb AFTER INSERT ON tmid \
         BEGIN INSERT INTO m VALUES (NEW.x); END",
    );

    // Rows (1),(2): tsrc → ta → tmid → tb → main.m. Row (3) trips tsrc's CHECK → ABORT.
    assert_exec_error(&mut db, "INSERT INTO tsrc VALUES (1),(2),(3)");

    // Every store the deep chain touched rolls back — including `main`, reached only by
    // the runtime-discovered depth-2 trigger.
    assert_rows(&mut db, "SELECT x FROM temp.tsrc", &[]);
    assert_rows(&mut db, "SELECT x FROM temp.tmid", &[]);
    assert_rows(&mut db, "SELECT x FROM main.m", &[]);
}

/// Positive control for the nested-chain discriminator: WITHOUT the abort, the same
/// depth-2 chain COMMITS into all three tables — proving the nested trigger really fires
/// and reaches `main`, so the discriminator's empty `main.m` means "rolled back", not
/// "never written". Guards against a false pass if nested recursion silently did not fire.
#[test]
fn autocommit_nested_trigger_deep_chain_no_abort_commits_all() {
    let mut db = mem();
    exec(&mut db, "PRAGMA recursive_triggers = ON");
    exec(&mut db, "CREATE TABLE m(x)");
    exec(&mut db, "CREATE TEMP TABLE tsrc(x)");
    exec(&mut db, "CREATE TEMP TABLE tmid(x)");
    exec(
        &mut db,
        "CREATE TEMP TRIGGER ta AFTER INSERT ON tsrc \
         BEGIN INSERT INTO tmid VALUES (NEW.x); END",
    );
    exec(
        &mut db,
        "CREATE TEMP TRIGGER tb AFTER INSERT ON tmid \
         BEGIN INSERT INTO m VALUES (NEW.x); END",
    );

    exec(&mut db, "INSERT INTO tsrc VALUES (1),(2)");

    assert_rows(&mut db, "SELECT x FROM temp.tsrc ORDER BY x", &[vec![int(1)], vec![int(2)]]);
    assert_rows(&mut db, "SELECT x FROM temp.tmid ORDER BY x", &[vec![int(1)], vec![int(2)]]);
    assert_rows(&mut db, "SELECT x FROM main.m ORDER BY x", &[vec![int(1)], vec![int(2)]]);
}

// ---------------------------------------------------------------------------
// GUARDS — the boundaries the abort-atomicity fix must NOT overshoot.
// ---------------------------------------------------------------------------

/// OR FAIL is NOT a full abort: it keeps the rows written before the conflict
/// (lang_conflict.html "FAIL"). Across namespaces that must hold for BOTH the firing
/// table AND the trigger's cross-namespace writes — the bracket commits every live
/// namespace on the FAIL path, it does not roll them back. Rows (1),(2) succeed (temp
/// and main each gain them); row (3) trips the CHECK and FAIL stops there.
#[test]
fn autocommit_or_fail_cross_namespace_keeps_preconflict_rows() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE mlog(x)"); // main
    exec(&mut db, "CREATE TEMP TABLE tsrc(x CHECK (x <> 3))"); // temp
    exec(
        &mut db,
        "CREATE TEMP TRIGGER ta AFTER INSERT ON tsrc \
         BEGIN INSERT INTO mlog VALUES (NEW.x); END",
    );

    // OR FAIL still errors on the conflict, but keeps the pre-conflict rows.
    assert_exec_error(&mut db, "INSERT OR FAIL INTO tsrc VALUES (1),(2),(3)");

    assert_rows(&mut db, "SELECT x FROM temp.tsrc ORDER BY x", &[vec![int(1)], vec![int(2)]]);
    assert_rows(&mut db, "SELECT x FROM main.mlog ORDER BY x", &[vec![int(1)], vec![int(2)]]);
}

/// The happy path (no abort) still COMMITS a cross-namespace write: a TEMP trigger on
/// `temp.tsrc` writes `main.mlog`, and a successful multi-row insert persists in both
/// stores. Guards that bracketing every namespace did not accidentally drop the writes.
#[test]
fn autocommit_cross_namespace_no_abort_still_commits() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE mlog(x)"); // main
    exec(&mut db, "CREATE TEMP TABLE tsrc(x)"); // temp
    exec(
        &mut db,
        "CREATE TEMP TRIGGER ta AFTER INSERT ON tsrc \
         BEGIN INSERT INTO mlog VALUES (NEW.x); END",
    );

    exec(&mut db, "INSERT INTO tsrc VALUES (1),(2)");

    assert_rows(&mut db, "SELECT x FROM temp.tsrc ORDER BY x", &[vec![int(1)], vec![int(2)]]);
    assert_rows(&mut db, "SELECT x FROM main.mlog ORDER BY x", &[vec![int(1)], vec![int(2)]]);
}

/// A same-namespace abort (all in `main`) still rolls back — this always worked (the
/// target namespace was bracketed) and must keep working under bracket-all.
#[test]
fn autocommit_same_namespace_abort_rolls_back() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x CHECK (x <> 3))"); // main
    exec(&mut db, "CREATE TABLE tlog(x)"); // main
    exec(
        &mut db,
        "CREATE TRIGGER g AFTER INSERT ON t \
         BEGIN INSERT INTO tlog VALUES (NEW.x); END",
    );

    assert_exec_error(&mut db, "INSERT INTO t VALUES (1),(2),(3)");

    assert_rows(&mut db, "SELECT x FROM t", &[]);
    assert_rows(&mut db, "SELECT x FROM tlog", &[]);
}

/// The single-`main` hot path (no temp/attached, no trigger) still rolls a plain
/// multi-row abort back. With only `main` live the bracket is exactly the single-pager
/// begin/rollback it generalizes; this pins that the len==1 path is unchanged.
#[test]
fn autocommit_single_main_plain_abort_rolls_back() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x CHECK (x <> 3))"); // main only

    assert_exec_error(&mut db, "INSERT INTO t VALUES (1),(2),(3)");

    assert_rows(&mut db, "SELECT x FROM t", &[]);
}

/// After an aborted cross-namespace statement, a SUBSEQUENT statement must commit
/// normally — proving the rollback-all left NO live pager stuck in an open transaction
/// (a leaked implicit txn would make the next begin-all fail with "a transaction is
/// already active"). This fences the begin-all/rollback-all cleanup contract.
#[test]
fn autocommit_abort_does_not_leak_open_transaction() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE mlog(x)"); // main
    exec(&mut db, "CREATE TEMP TABLE tsrc(x CHECK (x <> 3))"); // temp
    exec(
        &mut db,
        "CREATE TEMP TRIGGER ta AFTER INSERT ON tsrc \
         BEGIN INSERT INTO mlog VALUES (NEW.x); END",
    );

    // First statement aborts and rolls back every namespace.
    assert_exec_error(&mut db, "INSERT INTO tsrc VALUES (1),(2),(3)");

    // A later valid cross-namespace statement commits cleanly (no leaked open txn).
    exec(&mut db, "INSERT INTO tsrc VALUES (5)");
    assert_rows(&mut db, "SELECT x FROM temp.tsrc", &[vec![int(5)]]);
    assert_rows(&mut db, "SELECT x FROM main.mlog", &[vec![int(5)]]);
}
