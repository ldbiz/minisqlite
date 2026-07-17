//! Conformance battery: **TEMP triggers on non-TEMP tables** (cross-namespace trigger
//! attachment), exercised through the pinned `minisqlite::Connection` facade.
//!
//! Every expectation is derived from `spec/sqlite-doc/lang_createtrigger.html`, never from
//! what the engine happens to return.
//!
//! Spec basis:
//!   * §7 "TEMP Triggers on Non-TEMP Tables": "it is possible to create a TEMP TRIGGER on
//!     a table in another database"; the recommended form qualifies the target
//!     (`CREATE TEMP TRIGGER ex1 AFTER INSERT ON main.tab1 BEGIN ... END`). An UNQUALIFIED
//!     target resolves by search order (temp → main → attached) AT CREATE TIME and BINDS to
//!     the resolved target's database — so the trigger fires for DML on THAT specific
//!     `(database, table)`, not on a same-named table in another database.
//!   * §2: a NON-TEMP trigger's action target must be in the same database as the object the
//!     trigger is attached to ("TEMP triggers are not subject to the same-database rule"). A
//!     non-TEMP trigger whose ON-target is EXPLICITLY qualified to a different database than
//!     the trigger is rejected (SQLite's schema fixer:
//!     "trigger NAME cannot reference objects in database DB").
//!   * §3: BEFORE/AFTER fire only on tables; INSTEAD OF fires only on views.
//!   * A TEMP trigger lives in the transient per-connection `temp` schema, so it is NOT
//!     written to `main.sqlite_master` and does NOT survive a close → reopen.
//!
//! To PROVE a trigger fired (and which table it was bound to), each trigger body appends the
//! affected value to a dedicated audit table; the audit contents are then asserted. Data is
//! distinct per namespace and reads are schema-qualified (`main.x`, `temp.x`, `aux.x`) so the
//! test pins exactly which store was touched.

mod conformance;
use conformance::*;

use minisqlite::Connection;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

// ===========================================================================
// 1 — qualified `ON main.t` (the spec's recommended form) parses, binds to
//     (main, t), and fires on `INSERT INTO t`.
// ===========================================================================

#[test]
fn qualified_temp_trigger_on_main_table_fires() {
    // §7 example shape: `CREATE TEMP TRIGGER ... AFTER INSERT ON main.tab ...`. The trigger
    // lives in temp but binds to main.t; an INSERT into main.t must fire it.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)"); // main.t
    exec(&mut db, "CREATE TABLE audit(v)"); // main.audit (the trigger body's log)
    exec(
        &mut db,
        "CREATE TEMP TRIGGER trg AFTER INSERT ON main.t \
         BEGIN INSERT INTO audit VALUES (NEW.a); END",
    );

    exec(&mut db, "INSERT INTO t VALUES (42)");

    assert_rows(&mut db, "SELECT v FROM main.audit", &[vec![int(42)]]);
    assert_rows(&mut db, "SELECT a FROM main.t", &[vec![int(42)]]);
}

// ===========================================================================
// 2 — unqualified `ON t` where `t` exists only in main resolves to main at
//     create time and fires on main.t.
// ===========================================================================

#[test]
fn unqualified_temp_trigger_resolves_to_main_and_fires() {
    // With no temp.t to shadow it, `ON t` resolves (temp → main → attached) to main.t and
    // binds there. Firing on main.t proves the bind.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)"); // main.t only
    exec(&mut db, "CREATE TABLE audit(v)");
    exec(
        &mut db,
        "CREATE TEMP TRIGGER trg AFTER INSERT ON t \
         BEGIN INSERT INTO audit VALUES (NEW.a); END",
    );

    exec(&mut db, "INSERT INTO t VALUES (7)");
    exec(&mut db, "INSERT INTO main.t VALUES (8)"); // the qualified spelling reaches the same table

    assert_rows_unordered(&mut db, "SELECT v FROM audit", &[vec![int(7)], vec![int(8)]]);
}

// ===========================================================================
// 3 — SHADOWING: with main.t AND temp.t present, `ON main.t` fires ONLY on
//     main.t, `ON temp.t` (and unqualified `ON t`, temp-first) ONLY on temp.t.
// ===========================================================================

#[test]
fn shadowing_binds_each_trigger_to_its_own_namespace() {
    // The trigger is bound to a specific (db, name), not a bare name. Two same-named tables
    // (main.t, temp.t) each carry a TEMP trigger bound to THAT namespace; a write to one must
    // fire only its own trigger. This is the core shadowing-correctness case.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)"); // main.t
    exec(&mut db, "CREATE TEMP TABLE t(a)"); // temp.t shadows main.t for a bare name
    exec(&mut db, "CREATE TABLE maud(v)"); // main audit (for the main-bound trigger)
    exec(&mut db, "CREATE TEMP TABLE taud(v)"); // temp audit (for the temp-bound trigger)
    exec(
        &mut db,
        "CREATE TEMP TRIGGER trg_main AFTER INSERT ON main.t \
         BEGIN INSERT INTO maud VALUES (NEW.a); END",
    );
    exec(
        &mut db,
        "CREATE TEMP TRIGGER trg_temp AFTER INSERT ON temp.t \
         BEGIN INSERT INTO taud VALUES (NEW.a); END",
    );

    exec(&mut db, "INSERT INTO main.t VALUES (10)"); // fires trg_main only
    exec(&mut db, "INSERT INTO temp.t VALUES (20)"); // fires trg_temp only
    exec(&mut db, "INSERT INTO t VALUES (30)"); // bare name -> temp.t (search order) -> trg_temp

    // Proof: each audit saw only its own namespace's inserts.
    assert_rows(&mut db, "SELECT v FROM main.maud", &[vec![int(10)]]);
    assert_rows_unordered(&mut db, "SELECT v FROM temp.taud", &[vec![int(20)], vec![int(30)]]);
    // And the tables themselves hold the expected per-namespace rows.
    assert_rows(&mut db, "SELECT a FROM main.t", &[vec![int(10)]]);
    assert_rows_unordered(&mut db, "SELECT a FROM temp.t", &[vec![int(20)], vec![int(30)]]);
}

// ===========================================================================
// 4 — TEMP trigger on an ATTACHed table fires on `INSERT INTO aux.u`.
// ===========================================================================

#[test]
fn temp_trigger_on_attached_table_fires() {
    // §7: "A TEMP trigger is allowed to query or modify any table in any ATTACH-ed database."
    // Symmetrically it may be ATTACHED to a table in an attached database.
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.u(x)");
    exec(&mut db, "CREATE TABLE audit(v)"); // main.audit
    exec(
        &mut db,
        "CREATE TEMP TRIGGER trg AFTER INSERT ON aux.u \
         BEGIN INSERT INTO audit VALUES (NEW.x); END",
    );

    exec(&mut db, "INSERT INTO aux.u VALUES (99)");

    assert_rows(&mut db, "SELECT v FROM audit", &[vec![int(99)]]);
    // A shadowing main.u would NOT be affected — plant one and prove it is untouched.
    exec(&mut db, "CREATE TABLE u(x)"); // main.u, same bare name, no trigger
    exec(&mut db, "INSERT INTO main.u VALUES (1)"); // must NOT fire the aux-bound trigger
    assert_rows(&mut db, "SELECT v FROM audit", &[vec![int(99)]]);
}

// ===========================================================================
// 5 — TEMP INSTEAD OF trigger on a main VIEW, and on an attached VIEW.
// ===========================================================================

#[test]
fn temp_instead_of_trigger_on_main_view_fires() {
    // §3: INSTEAD OF fires only on a view. A TEMP INSTEAD OF trigger on a main view redirects
    // the INSERT: no row lands in the base table, the trigger body runs instead.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE base(a)");
    exec(&mut db, "CREATE VIEW v AS SELECT a FROM base"); // main.v
    exec(&mut db, "CREATE TABLE audit(v)");
    exec(
        &mut db,
        "CREATE TEMP TRIGGER trg INSTEAD OF INSERT ON main.v \
         BEGIN INSERT INTO audit VALUES (NEW.a); END",
    );

    exec(&mut db, "INSERT INTO v VALUES (5)");

    assert_rows(&mut db, "SELECT v FROM audit", &[vec![int(5)]]);
    assert_rows(&mut db, "SELECT a FROM base", &[]); // INSTEAD OF -> no actual insert
}

#[test]
fn temp_instead_of_trigger_on_attached_view_fires() {
    // The previously-unreachable case: an attached-db view whose INSTEAD OF action is served by
    // a TEMP trigger bound across namespaces.
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.abase(a)");
    exec(&mut db, "CREATE VIEW aux.av AS SELECT a FROM aux.abase");
    exec(&mut db, "CREATE TABLE audit(v)"); // main.audit
    exec(
        &mut db,
        "CREATE TEMP TRIGGER trg INSTEAD OF INSERT ON aux.av \
         BEGIN INSERT INTO audit VALUES (NEW.a); END",
    );

    exec(&mut db, "INSERT INTO aux.av VALUES (13)");

    assert_rows(&mut db, "SELECT v FROM audit", &[vec![int(13)]]);
    assert_rows(&mut db, "SELECT a FROM aux.abase", &[]); // redirected, no base insert
}

// ===========================================================================
// 6 — BEFORE and AFTER timings, and INSERT/UPDATE/DELETE events, all fire for
//     a cross-namespace-bound TEMP trigger.
// ===========================================================================

#[test]
fn temp_trigger_fires_for_before_after_and_all_events_on_main() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)"); // main.t
    exec(&mut db, "CREATE TABLE audit(tag, v)");
    exec(
        &mut db,
        "CREATE TEMP TRIGGER tb BEFORE INSERT ON main.t \
         BEGIN INSERT INTO audit VALUES ('bins', NEW.a); END",
    );
    exec(
        &mut db,
        "CREATE TEMP TRIGGER ti AFTER INSERT ON main.t \
         BEGIN INSERT INTO audit VALUES ('ains', NEW.a); END",
    );
    exec(
        &mut db,
        "CREATE TEMP TRIGGER tu AFTER UPDATE ON main.t \
         BEGIN INSERT INTO audit VALUES ('upd', NEW.a); END",
    );
    exec(
        &mut db,
        "CREATE TEMP TRIGGER td AFTER DELETE ON main.t \
         BEGIN INSERT INTO audit VALUES ('del', OLD.a); END",
    );

    exec(&mut db, "INSERT INTO t VALUES (5)"); // tb + ti
    exec(&mut db, "UPDATE t SET a = 6 WHERE a = 5"); // tu
    exec(&mut db, "DELETE FROM t WHERE a = 6"); // td

    assert_rows_unordered(
        &mut db,
        "SELECT tag, v FROM audit",
        &[
            vec![text("bins"), int(5)],
            vec![text("ains"), int(5)],
            vec![text("upd"), int(6)],
            vec![text("del"), int(6)],
        ],
    );
}

// ===========================================================================
// 7 — NEGATIVE: a NON-TEMP trigger whose ON-target is in a different database
//     is rejected (§2 / SQLite's schema fixer).
// ===========================================================================

#[test]
fn nontemp_trigger_targeting_another_database_is_rejected() {
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.t(x)");

    // Trigger explicitly in main, target explicitly in aux: cross-database, rejected.
    let e1 = assert_exec_error(
        &mut db,
        "CREATE TRIGGER main.trg AFTER INSERT ON aux.t BEGIN SELECT 1; END",
    );
    assert!(
        format!("{e1:?}").contains("cannot reference objects in database"),
        "expected SQLite's cross-database fixer message, got: {e1:?}"
    );

    // Unqualified trigger name defaults to main; target explicitly aux: also cross-database.
    let e2 = assert_exec_error(
        &mut db,
        "CREATE TRIGGER trg2 AFTER INSERT ON aux.t BEGIN SELECT 1; END",
    );
    assert!(
        format!("{e2:?}").contains("cannot reference objects in database"),
        "expected SQLite's cross-database fixer message, got: {e2:?}"
    );

    // GUARD (no regression): a non-TEMP trigger whose target is in the trigger's OWN database
    // is fine — `aux.trg ON aux.t` still creates and fires.
    exec(&mut db, "CREATE TABLE aux.aud(v)");
    exec(
        &mut db,
        "CREATE TRIGGER aux.trg AFTER INSERT ON aux.t \
         BEGIN INSERT INTO aud VALUES (NEW.x); END",
    );
    exec(&mut db, "INSERT INTO aux.t VALUES (3)");
    assert_rows(&mut db, "SELECT v FROM aux.aud", &[vec![int(3)]]);
}

// ===========================================================================
// 8 — NOT PERSISTED: absent from main.sqlite_master; gone after a close →
//     reopen of a file-backed connection, while main.t and its data remain.
// ===========================================================================

#[test]
fn temp_trigger_on_main_is_not_persisted_and_gone_after_reopen() {
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(a)"); // persisted (main)
        exec(&mut db, "CREATE TABLE audit(v)"); // persisted (main)
        exec(&mut db, "INSERT INTO t VALUES (1)");
        exec(
            &mut db,
            "CREATE TEMP TRIGGER trg AFTER INSERT ON main.t \
             BEGIN INSERT INTO audit VALUES (NEW.a); END",
        );
        exec(&mut db, "INSERT INTO t VALUES (2)"); // fires trg -> audit gets 2

        assert_rows(&mut db, "SELECT v FROM audit", &[vec![int(2)]]);
        // The TEMP trigger is NOT in the persisted main schema.
        assert_scalar(
            &mut db,
            "SELECT count(*) FROM main.sqlite_master WHERE type='trigger' AND name='trg'",
            int(0),
        );
    }
    {
        // Reopen: the temp store is fresh, so the trigger is gone; main.t and its data remain.
        let mut db = tmp.open();
        assert_scalar(
            &mut db,
            "SELECT count(*) FROM main.sqlite_master WHERE type='trigger' AND name='trg'",
            int(0),
        );
        // The trigger no longer fires: inserting a new row adds nothing to audit.
        exec(&mut db, "INSERT INTO t VALUES (3)");
        assert_rows(&mut db, "SELECT v FROM audit", &[vec![int(2)]]); // unchanged -> did not fire
        assert_rows_unordered(
            &mut db,
            "SELECT a FROM t",
            &[vec![int(1)], vec![int(2)], vec![int(3)]],
        );
        // And DROP TRIGGER now finds nothing.
        assert_exec_error(&mut db, "DROP TRIGGER trg");
    }
}

// ===========================================================================
// 9 — DROP TRIGGER removes the temp-stored cross-namespace trigger.
// ===========================================================================

#[test]
fn drop_trigger_removes_cross_namespace_temp_trigger() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "CREATE TABLE audit(v)");
    exec(
        &mut db,
        "CREATE TEMP TRIGGER trg AFTER INSERT ON main.t \
         BEGIN INSERT INTO audit VALUES (NEW.a); END",
    );

    exec(&mut db, "INSERT INTO t VALUES (1)"); // fires -> audit=[1]
    assert_rows(&mut db, "SELECT v FROM audit", &[vec![int(1)]]);

    exec(&mut db, "DROP TRIGGER trg"); // finds & removes the temp-stored trigger
    exec(&mut db, "INSERT INTO t VALUES (2)"); // no longer fires

    assert_rows(&mut db, "SELECT v FROM audit", &[vec![int(1)]]); // unchanged
    assert_rows_unordered(&mut db, "SELECT a FROM t", &[vec![int(1)], vec![int(2)]]);
    // A second DROP is now a no-such-trigger error.
    assert_exec_error(&mut db, "DROP TRIGGER trg");
}

// ===========================================================================
// 10 — DROP TABLE main.t cascade-drops a TEMP trigger bound to it.
// ===========================================================================

#[test]
fn drop_table_cascades_cross_namespace_temp_trigger() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "CREATE TABLE audit(v)");
    exec(
        &mut db,
        "CREATE TEMP TRIGGER trg AFTER INSERT ON main.t \
         BEGIN INSERT INTO audit VALUES (NEW.a); END",
    );
    exec(&mut db, "INSERT INTO t VALUES (1)"); // proves it fires pre-drop
    assert_rows(&mut db, "SELECT v FROM audit", &[vec![int(1)]]);

    exec(&mut db, "DROP TABLE t"); // must cascade the temp trigger bound to (main, t)

    // The trigger is gone: recreate t, insert, and audit must not grow.
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "INSERT INTO t VALUES (2)");
    assert_rows(&mut db, "SELECT v FROM audit", &[vec![int(1)]]); // unchanged -> cascade removed it
    // DROP TRIGGER proves the cascade actually removed the trigger row.
    assert_exec_error(&mut db, "DROP TRIGGER trg");
}

#[test]
fn drop_table_does_not_cascade_a_trigger_bound_to_a_same_named_table_in_another_db() {
    // Shadowing on the DROP side: a TEMP trigger bound to (main, t) must survive
    // `DROP TABLE temp.t` (same bare name, different database).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)"); // main.t
    exec(&mut db, "CREATE TEMP TABLE t(a)"); // temp.t (same bare name)
    exec(&mut db, "CREATE TABLE audit(v)");
    exec(
        &mut db,
        "CREATE TEMP TRIGGER trg AFTER INSERT ON main.t \
         BEGIN INSERT INTO audit VALUES (NEW.a); END",
    );

    exec(&mut db, "DROP TABLE temp.t"); // must NOT cascade the (main, t)-bound trigger

    // The trigger still fires on main.t.
    exec(&mut db, "INSERT INTO main.t VALUES (5)");
    assert_rows(&mut db, "SELECT v FROM audit", &[vec![int(5)]]);
}

// ===========================================================================
// 11 — Atomicity: an aborted autocommit statement that fired a cross-namespace
//      bound TEMP trigger backs out the WHOLE statement (the trigger's writes too).
// ===========================================================================

#[test]
fn aborted_statement_backs_out_cross_namespace_temp_trigger_writes() {
    // A fired trigger's actions run within the firing statement (§, lang_transaction.html:
    // a statement is atomic). A RAISE(ABORT) inside the trigger must undo the statement's
    // changes AND the trigger's own writes.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "CREATE TABLE audit(v)");
    exec(
        &mut db,
        "CREATE TEMP TRIGGER trg AFTER INSERT ON main.t \
         BEGIN INSERT INTO audit VALUES (NEW.a); SELECT RAISE(ABORT, 'stop'); END",
    );

    assert_exec_error(&mut db, "INSERT INTO t VALUES (1)");

    // Both the top-level insert and the trigger's audit write are rolled back.
    assert_rows(&mut db, "SELECT a FROM t", &[]);
    assert_rows(&mut db, "SELECT v FROM audit", &[]);
}

// ===========================================================================
// 12 — Races/TOCTOU: the (db, table) binding is a stable NAME, re-resolved
//      every statement, so DETACH remapping DbIndex values never breaks it.
// ===========================================================================

#[test]
fn temp_trigger_on_attached_keeps_firing_after_detach_of_lower_db() {
    // Witness A: DETACHing a LOWER-indexed attached db shifts the target's
    // DbIndex down. A binding cached as a numeric index would go stale and the trigger would
    // silently stop firing; a NAME-based binding (re-resolved against the live registry) keeps
    // firing on the real target.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE log(x)"); // main.log (the trigger's action target)
    exec(&mut db, "ATTACH DATABASE ':memory:' AS a"); // -> index 2
    exec(&mut db, "ATTACH DATABASE ':memory:' AS b"); // -> index 3
    exec(&mut db, "CREATE TABLE b.u(x)");
    exec(
        &mut db,
        "CREATE TEMP TRIGGER trg AFTER INSERT ON b.u \
         BEGIN INSERT INTO log VALUES (NEW.x); END",
    );
    exec(&mut db, "INSERT INTO b.u VALUES (1)"); // fires pre-detach
    assert_rows(&mut db, "SELECT x FROM main.log", &[vec![int(1)]]);

    exec(&mut db, "DETACH DATABASE a"); // Vec::remove(2): b shifts index 3 -> 2

    exec(&mut db, "INSERT INTO b.u VALUES (2)"); // must STILL fire on b.u
    assert_rows_unordered(&mut db, "SELECT x FROM main.log", &[vec![int(1)], vec![int(2)]]);
}

#[test]
fn attach_slot_reuse_after_detach_does_not_misfire_on_the_wrong_db() {
    // Witness B: after DETACHing a lower db and ATTACHing a new one that reuses
    // the freed numeric slot, a trigger bound to the ORIGINAL db must not fire on the NEW db's
    // same-named table (the cross-db corruption a cached numeric index would cause).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE log(x)");
    exec(&mut db, "ATTACH DATABASE ':memory:' AS a"); // index 2
    exec(&mut db, "ATTACH DATABASE ':memory:' AS b"); // index 3
    exec(&mut db, "CREATE TABLE b.u(x)");
    exec(
        &mut db,
        "CREATE TEMP TRIGGER trg AFTER INSERT ON b.u \
         BEGIN INSERT INTO log VALUES (NEW.x); END",
    );

    exec(&mut db, "DETACH DATABASE a"); // frees index 2; b shifts 3 -> 2
    exec(&mut db, "ATTACH DATABASE ':memory:' AS c"); // c -> index 3 (b's vacated numeric slot)
    exec(&mut db, "CREATE TABLE c.u(x)");

    exec(&mut db, "INSERT INTO c.u VALUES (99)"); // MUST NOT fire trg (bound to b.u, not c.u)
    assert_rows(&mut db, "SELECT x FROM main.log", &[]); // no cross-db misfire

    exec(&mut db, "INSERT INTO b.u VALUES (1)"); // the real target still fires
    assert_rows(&mut db, "SELECT x FROM main.log", &[vec![int(1)]]);
}

// ===========================================================================
// 13 — Reload: a temp rollback-resync reloads the temp catalog, and the
//      binding must be reconstructed from the stored SQL, never coerced to MAIN.
// ===========================================================================

#[test]
fn attached_bound_temp_trigger_still_fires_after_rollback() {
    // An explicit ROLLBACK reloads EVERY catalog (incl. temp), so `load_trigger_row` must
    // reconstruct the attached-db binding from the stored SQL's qualifier
    // (`ForeignSchema("aux")`), never the old silent MAIN default. Passes before AND after the
    // unrelated ROLLBACK.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE log(x)"); // main-only action target
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.u(x)");
    exec(
        &mut db,
        "CREATE TEMP TRIGGER trg AFTER INSERT ON aux.u \
         BEGIN INSERT INTO log VALUES (NEW.x); END",
    );
    exec(&mut db, "INSERT INTO aux.u VALUES (1)");
    assert_rows(&mut db, "SELECT x FROM main.log", &[vec![int(1)]]); // fires (correct)

    // An unrelated explicit transaction reload-resyncs the temp catalog.
    exec(&mut db, "BEGIN");
    exec(&mut db, "ROLLBACK");

    exec(&mut db, "INSERT INTO aux.u VALUES (2)");
    // Must still fire on aux.u — the binding reconstructed as the attach alias, not main.
    assert_rows_unordered(&mut db, "SELECT x FROM main.log", &[vec![int(1)], vec![int(2)]]);
}

#[test]
fn unqualified_bound_temp_trigger_still_fires_after_rollback() {
    // The ForeignUnqualified reload path: an unqualified `ON t` (resolved to main at create) is
    // rebuilt as ForeignUnqualified on the temp rollback-resync and re-resolved by search order
    // at fire time, so it keeps firing on main.t (no coercion to a wrong or same-store binding).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)"); // main.t only
    exec(&mut db, "CREATE TABLE audit(v)"); // main.audit
    exec(
        &mut db,
        "CREATE TEMP TRIGGER trg AFTER INSERT ON t \
         BEGIN INSERT INTO audit VALUES (NEW.a); END",
    );
    exec(&mut db, "INSERT INTO t VALUES (1)");
    assert_rows(&mut db, "SELECT v FROM audit", &[vec![int(1)]]);

    exec(&mut db, "BEGIN");
    exec(&mut db, "ROLLBACK"); // reloads temp -> trg rebuilt as ForeignUnqualified

    exec(&mut db, "INSERT INTO t VALUES (2)");
    assert_rows_unordered(&mut db, "SELECT v FROM audit", &[vec![int(1)], vec![int(2)]]);
}

// ===========================================================================
// 14 — NEGATIVE: a schema-QUALIFIED TEMP-trigger ON-target that is ABSENT in
//      its named db ERRORS (qualifier-preserving), never silently rebinds.
// ===========================================================================

#[test]
fn qualified_temp_trigger_on_absent_main_target_errors() {
    // A qualified `ON main.t` with main.t ABSENT (but a shadowing temp.t present) must ERROR —
    // real sqlite honors the qualifier (`sqlite3SrcListLookup`) — not create+bind to temp.t.
    let mut db = mem();
    exec(&mut db, "CREATE TEMP TABLE t(a)"); // temp.t present; main.t ABSENT
    exec(&mut db, "CREATE TEMP TABLE audit(v)");
    let e = assert_exec_error(
        &mut db,
        "CREATE TEMP TRIGGER trg AFTER INSERT ON main.t \
         BEGIN INSERT INTO audit VALUES (NEW.a); END",
    );
    assert!(
        format!("{e:?}").contains("no such table: main.t"),
        "expected qualifier-preserving 'no such table: main.t', got: {e:?}"
    );
    // It did NOT silently bind to temp.t: an insert into temp.t fires nothing, DROP TRIGGER misses.
    exec(&mut db, "INSERT INTO t VALUES (1)");
    assert_rows(&mut db, "SELECT v FROM audit", &[]);
    assert_exec_error(&mut db, "DROP TRIGGER trg");
}

#[test]
fn qualified_temp_trigger_on_absent_attached_target_errors() {
    // The same rule for an ATTACH qualifier: `ON aux.t` with aux.t ABSENT must ERROR, not
    // rebind to a shadowing temp.t.
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TEMP TABLE t(a)"); // temp.t present (shadowing bare name); aux.t ABSENT
    exec(&mut db, "CREATE TEMP TABLE audit(v)");
    let e = assert_exec_error(
        &mut db,
        "CREATE TEMP TRIGGER trg AFTER INSERT ON aux.t \
         BEGIN INSERT INTO audit VALUES (NEW.a); END",
    );
    assert!(
        format!("{e:?}").contains("no such table: aux.t"),
        "expected qualifier-preserving 'no such table: aux.t', got: {e:?}"
    );
    exec(&mut db, "INSERT INTO t VALUES (1)");
    assert_rows(&mut db, "SELECT v FROM audit", &[]);
    assert_exec_error(&mut db, "DROP TRIGGER trg");
}

#[test]
fn trigger_on_unknown_schema_qualifier_errors_without_misbinding() {
    // A qualified ON-target naming an UNKNOWN database must error, never silently bind to a
    // same-named table in the trigger's own store (the String -> QualifiedName parse change made
    // `ON db.tbl` reachable for non-TEMP triggers too, so this pins the unknown-db behavior).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)"); // main.t exists (the would-be silent-misbind target)
    let e = assert_exec_error(
        &mut db,
        "CREATE TRIGGER trg AFTER INSERT ON bogus.t BEGIN SELECT 1; END",
    );
    assert!(
        format!("{e:?}").contains("unknown database bogus"),
        "expected 'unknown database bogus', got: {e:?}"
    );
    assert_exec_error(&mut db, "DROP TRIGGER trg"); // nothing was created
}

// ===========================================================================
// 15 — §7 reattach-on-schema-change: an UNQUALIFIED cross-namespace temp
//      trigger re-resolves by search order, so a later shadowing temp.t
//      "reattaches" it (fires on temp.t, no longer on main.t).
// ===========================================================================

#[test]
fn unqualified_temp_trigger_reattaches_to_a_later_shadowing_temp_table() {
    // The spec's documented footgun (lang_createtrigger.html §7): an unqualified TEMP-trigger
    // target re-resolves whenever the schema changes. Binding NAME-based (not to a fixed db)
    // makes this correct — creating a shadowing temp.t reattaches the trigger to temp.t (it
    // fires there and NO LONGER on main.t). Previously it silently fired on NEITHER after the
    // shadow appeared; this pins the fix.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)"); // main.t (the create-time resolved target)
    exec(&mut db, "CREATE TABLE audit(v)"); // main.audit (the trigger body's log)
    exec(
        &mut db,
        "CREATE TEMP TRIGGER trg AFTER INSERT ON t \
         BEGIN INSERT INTO audit VALUES (NEW.a); END", // unqualified -> bound to main.t for now
    );
    exec(&mut db, "INSERT INTO main.t VALUES (1)"); // bound to main.t -> fires
    assert_rows(&mut db, "SELECT v FROM audit", &[vec![int(1)]]);

    exec(&mut db, "CREATE TEMP TABLE t(a)"); // a schema change: temp.t now shadows main.t

    exec(&mut db, "INSERT INTO temp.t VALUES (20)"); // reattached -> fires on temp.t
    exec(&mut db, "INSERT INTO main.t VALUES (99)"); // no longer bound to main.t -> does NOT fire

    // audit saw the pre-shadow main.t insert (1) and the post-shadow temp.t insert (20), but
    // NOT the post-shadow main.t insert (99).
    assert_rows_unordered(&mut db, "SELECT v FROM audit", &[vec![int(1)], vec![int(20)]]);
}

// ===========================================================================
// 16 — DROP TABLE aux.u cascades a TEMP trigger bound to the ATTACHED table
//      (the name-based cascade for a ForeignSchema(alias) binding).
// ===========================================================================

#[test]
fn drop_table_cascades_a_temp_trigger_bound_to_an_attached_table() {
    // Test 10 covers the `main`-bound cascade; this covers an ATTACHED binding, whose alias is
    // resolved through the live registry by the engine's name-based victim resolution.
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.u(x)");
    exec(&mut db, "CREATE TABLE audit(v)"); // main.audit
    exec(
        &mut db,
        "CREATE TEMP TRIGGER trg AFTER INSERT ON aux.u \
         BEGIN INSERT INTO audit VALUES (NEW.x); END",
    );
    exec(&mut db, "INSERT INTO aux.u VALUES (1)"); // proves it fires pre-drop
    assert_rows(&mut db, "SELECT v FROM audit", &[vec![int(1)]]);

    exec(&mut db, "DROP TABLE aux.u"); // must cascade the temp trigger bound to (aux, u)

    // The trigger is gone: recreate aux.u, insert, and audit must not grow.
    exec(&mut db, "CREATE TABLE aux.u(x)");
    exec(&mut db, "INSERT INTO aux.u VALUES (2)");
    assert_rows(&mut db, "SELECT v FROM audit", &[vec![int(1)]]); // unchanged -> cascade removed it
    assert_exec_error(&mut db, "DROP TRIGGER trg"); // proves the cascade removed the row
}

// ---------------------------------------------------------------------------
// File-backed temp-database helper (mirrors the pattern in the sibling on-disk
// conformance files; each file owns its own so no `.db` is ever committed).
// ---------------------------------------------------------------------------

struct TempDb {
    path: PathBuf,
}

impl TempDb {
    fn new() -> TempDb {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
        let mut path = std::env::temp_dir();
        path.push(format!("minisqlite_temptrig_{pid}_{n}_{nanos}.db"));
        let db = TempDb { path };
        db.remove_all();
        db
    }

    fn open(&self) -> Connection {
        let path: &Path = &self.path;
        Connection::open(path)
            .unwrap_or_else(|e| panic!("Connection::open failed for {path:?}: {e:?}"))
    }

    fn sidecar(&self, suffix: &str) -> PathBuf {
        let mut s = self.path.as_os_str().to_os_string();
        s.push(suffix);
        PathBuf::from(s)
    }

    fn remove_all(&self) {
        let _ = std::fs::remove_file(&self.path);
        for suffix in ["-journal", "-wal", "-shm"] {
            let _ = std::fs::remove_file(self.sidecar(suffix));
        }
    }
}

impl Drop for TempDb {
    fn drop(&mut self) {
        self.remove_all();
    }
}
