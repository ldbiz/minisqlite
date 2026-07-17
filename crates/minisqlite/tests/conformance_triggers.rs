//! Conformance battery: TRIGGER FIRING semantics — the run-time behavior of a
//! `CREATE TRIGGER` when its DML event occurs.
//!
//! Every expected value is TRANSCRIBED FROM THE SPEC, `spec/sqlite-doc/lang_createtrigger.html`,
//! never from what the engine returns. The binding rules pinned below (verbatim from the
//! sections named) are:
//!
//!   * §2: "Each trigger must specify that it will fire for one of the following operations:
//!     DELETE, INSERT, UPDATE. The trigger fires once for each row that is deleted, inserted,
//!     or updated." — so a trigger runs its body ONCE PER AFFECTED ROW.
//!   * §2: "If the 'UPDATE OF column-name' syntax is used, then the trigger will only fire if
//!     column-name appears on the left-hand side of one of the terms in the SET clause of the
//!     UPDATE statement." — an `UPDATE OF a` trigger fires for `SET a=…`, not for `SET b=…`.
//!   * §2: "At this time SQLite supports only FOR EACH ROW triggers, not FOR EACH STATEMENT
//!     triggers. Hence explicitly specifying FOR EACH ROW is optional." — writing or omitting
//!     `FOR EACH ROW` behaves identically; there is no per-statement trigger.
//!   * §2: OLD/NEW validity table — INSERT ⇒ NEW references valid; UPDATE ⇒ NEW and OLD valid;
//!     DELETE ⇒ OLD references valid.
//!   * §2: "If a WHEN clause is supplied, the SQL statements specified are only executed if the
//!     WHEN clause is true. If no WHEN clause is supplied, the SQL statements are executed every
//!     time the trigger fires."
//!   * §2: "The BEFORE or AFTER keyword determines when the trigger actions will be executed …
//!     BEFORE is the default when neither keyword is present."
//!   * §3: "BEFORE and AFTER triggers work only on ordinary tables. INSTEAD OF triggers work
//!     only on views. If an INSTEAD OF INSERT trigger exists on a view, then it is possible to
//!     execute an INSERT statement against that view. No actual insert occurs. Instead, the
//!     statements contained within the trigger are run." (Likewise INSTEAD OF UPDATE / DELETE.)
//!   * §4: the worked `update_customer_address` (UPDATE OF, default timing) and the INSTEAD OF
//!     UPDATE `cust_addr_chng` examples.
//!   * §5: "programmers are encouraged to prefer AFTER triggers over BEFORE triggers" — a BEFORE
//!     trigger that modifies the row it fires on is UNDEFINED, so every well-defined post-state
//!     test here uses an AFTER trigger (or a BEFORE trigger that only writes ANOTHER table).
//!   * §6: "When one of RAISE(ROLLBACK,…), RAISE(ABORT,…) or RAISE(FAIL,…) is called … the
//!     specified ON CONFLICT processing is performed and the current query terminates. An error
//!     code of SQLITE_CONSTRAINT is returned to the application …"
//!
//! ENGINE GAP (WHY these tests fail today). `CREATE TRIGGER` SUCCEEDS — the catalog registers
//! a `type='trigger'` schema row — and the trigger body is even COMPILED at plan time and attached
//! to the DML node. But the EXECUTOR does not yet FIRE triggers on DML: it never runs the attached
//! trigger programs. So you create a trigger, run the DML, and the trigger's side effect (the audit
//! row, the redirected write) never happens. Each test below asserts the SPEC-CORRECT POST-FIRE
//! STATE — the side effect DID happen — so it fails now and will pass, with no test change, the
//! moment the executor gains firing. The spec-correct assertion is never weakened to make it pass,
//! the current un-fired state is never asserted as the expectation, and none of these are
//! `#[ignore]`d.
//!
//! RULE (the #1 correctness rule for this file): always assert the state as if the trigger FIRED
//! (the audit row EXISTS, the redirected write HAPPENED, the count reflects one firing per row).
//! NEVER assert the un-fired state ("audit is empty", "count == 0") — that would pass today and
//! fail when firing lands, exactly backwards. The two tests that must observe a NON-firing (an
//! `UPDATE OF` on an un-named column, and a DROPed trigger) ANCHOR that observation to a
//! proven-firing baseline earlier in the SAME test, so neither is a bare un-fired gap-pin.
//!
//! COMPOUND cases (marked `// COMPOUND: also needs <X>`) depend on a second unimplemented feature
//! besides firing (view-target INSTEAD OF routing, the RAISE() function, the recursive_triggers
//! pragma); each case names the extra feature it needs and may stay red until BOTH land.

mod conformance;
use conformance::*;

// =============================================================================
// CATEGORY 1 — CORE FIRING. Clean single-feature loud-signals: a trigger writes
// an `audit`/`log` side table, and each test asserts that side table's contents
// AS IF the trigger fired. All fail today (the executor does not fire) and pass
// on landing, with no COMPOUND second feature.
// =============================================================================

/// §2 ("fires once for each row … inserted") + INSERT⇒NEW valid. An AFTER INSERT trigger
/// writes NEW.a to `log`; a single-row INSERT fires it once, so `log` holds exactly (7).
#[test]
fn after_insert_fires_once_and_logs_new_value() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "CREATE TABLE log(v INTEGER)");
    exec(&mut db, "CREATE TRIGGER tr AFTER INSERT ON t BEGIN INSERT INTO log(v) VALUES(NEW.a); END");
    exec(&mut db, "INSERT INTO t(a) VALUES(7)");
    assert_rows(&mut db, "SELECT v FROM log", &[vec![int(7)]]);
}

/// §2 (UPDATE ⇒ NEW and OLD references valid). An AFTER UPDATE trigger logs BOTH the OLD and
/// the NEW value of the changed row, so updating a=1 to a=2 records the transition (1, 2).
#[test]
fn after_update_logs_old_and_new_values() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "CREATE TABLE log(old_a INTEGER, new_a INTEGER)");
    exec(&mut db, "INSERT INTO t(a) VALUES(1)");
    exec(
        &mut db,
        "CREATE TRIGGER tr AFTER UPDATE ON t BEGIN INSERT INTO log(old_a, new_a) VALUES(OLD.a, NEW.a); END",
    );
    exec(&mut db, "UPDATE t SET a = 2");
    assert_rows(&mut db, "SELECT old_a, new_a FROM log", &[vec![int(1), int(2)]]);
}

/// §2 (DELETE ⇒ OLD references valid). An AFTER DELETE trigger logs OLD.a, so deleting the row
/// a=9 records 9 in `log` (the deleted row's value is visible through OLD).
#[test]
fn after_delete_logs_old_value() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "CREATE TABLE log(v INTEGER)");
    exec(&mut db, "INSERT INTO t(a) VALUES(9)");
    exec(&mut db, "CREATE TRIGGER tr AFTER DELETE ON t BEGIN INSERT INTO log(v) VALUES(OLD.a); END");
    exec(&mut db, "DELETE FROM t WHERE a = 9");
    assert_rows(&mut db, "SELECT v FROM log", &[vec![int(9)]]);
}

/// §2 ("fires once for each row that is … inserted"). A multi-row INSERT (3 rows) fires the
/// FOR EACH ROW trigger three times, so `log` gains one row per inserted row: {1, 2, 3}.
#[test]
fn for_each_row_fires_once_per_inserted_row() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "CREATE TABLE log(v INTEGER)");
    exec(
        &mut db,
        "CREATE TRIGGER tr AFTER INSERT ON t FOR EACH ROW BEGIN INSERT INTO log(v) VALUES(NEW.a); END",
    );
    exec(&mut db, "INSERT INTO t(a) VALUES(1), (2), (3)");
    assert_rows_unordered(&mut db, "SELECT v FROM log", &[vec![int(1)], vec![int(2)], vec![int(3)]]);
}

/// §2 ("fires once for each row that is … updated"). An UPDATE matching 3 rows fires the trigger
/// three times; each firing logs the row's NEW value, so `log` holds all three new values.
#[test]
fn for_each_row_fires_once_per_updated_row() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "CREATE TABLE log(v INTEGER)");
    exec(&mut db, "INSERT INTO t(a) VALUES(1), (2), (3)");
    exec(
        &mut db,
        "CREATE TRIGGER tr AFTER UPDATE ON t BEGIN INSERT INTO log(v) VALUES(NEW.a); END",
    );
    // `SET a = a + 10` bumps every row, so the three NEW values are 11, 12, 13.
    exec(&mut db, "UPDATE t SET a = a + 10");
    assert_rows_unordered(&mut db, "SELECT v FROM log", &[vec![int(11)], vec![int(12)], vec![int(13)]]);
}

/// §2 ("fires once for each row that is deleted"). A DELETE matching 2 rows fires the trigger
/// twice; each firing logs OLD.a, so `log` holds both deleted values.
#[test]
fn for_each_row_fires_once_per_deleted_row() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "CREATE TABLE log(v INTEGER)");
    exec(&mut db, "INSERT INTO t(a) VALUES(1), (2), (3)");
    exec(&mut db, "CREATE TRIGGER tr AFTER DELETE ON t BEGIN INSERT INTO log(v) VALUES(OLD.a); END");
    // Deletes rows 1 and 2 (a < 3), leaving 3; the trigger fires once per deleted row.
    exec(&mut db, "DELETE FROM t WHERE a < 3");
    assert_rows_unordered(&mut db, "SELECT v FROM log", &[vec![int(1)], vec![int(2)]]);
}

/// §2 ("explicitly specifying FOR EACH ROW is optional"). A trigger written WITHOUT `FOR EACH
/// ROW` fires per row exactly like one with it — a 2-row INSERT still logs both rows.
#[test]
fn for_each_row_keyword_is_optional() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "CREATE TABLE log(v INTEGER)");
    // No FOR EACH ROW clause — SQLite has only per-row triggers, so this is per-row too.
    exec(&mut db, "CREATE TRIGGER tr AFTER INSERT ON t BEGIN INSERT INTO log(v) VALUES(NEW.a); END");
    exec(&mut db, "INSERT INTO t(a) VALUES(4), (5)");
    assert_rows_unordered(&mut db, "SELECT v FROM log", &[vec![int(4)], vec![int(5)]]);
}

/// §2 ("BEFORE is the default when neither keyword is present"). A trigger written with NEITHER
/// BEFORE nor AFTER is accepted and fires, logging NEW.a. NOTE: this pins that a no-timing-keyword
/// trigger IS accepted and DOES fire — it does NOT, on its own, observe that the timing is BEFORE
/// rather than AFTER (its body writes another table, so the insert order is not visible here); a
/// well-defined BEFORE-vs-AFTER discriminator is subtle (§5) and is flagged as a follow-up.
#[test]
fn trigger_without_timing_keyword_is_accepted_and_fires() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "CREATE TABLE log(v INTEGER)");
    // No BEFORE/AFTER keyword ⇒ BEFORE by default (§2); either way it must fire.
    exec(&mut db, "CREATE TRIGGER tr INSERT ON t BEGIN INSERT INTO log(v) VALUES(NEW.a); END");
    exec(&mut db, "INSERT INTO t(a) VALUES(8)");
    assert_rows(&mut db, "SELECT v FROM log", &[vec![int(8)]]);
}

/// §2 ("the SQL statements … are only executed if the WHEN clause is true"). With `WHEN NEW.a >
/// 10` a 2-row INSERT of {5, 20} fires the body only for the row that satisfies WHEN, so `log`
/// holds exactly (20) — the 5 row is gated out. (Asserting the EXACT set {20}, not "log empty",
/// keeps this RED today: today log is {} which is not {20}.)
#[test]
fn when_clause_gates_firing_per_row() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "CREATE TABLE log(v INTEGER)");
    exec(
        &mut db,
        "CREATE TRIGGER tr AFTER INSERT ON t WHEN NEW.a > 10 BEGIN INSERT INTO log(v) VALUES(NEW.a); END",
    );
    exec(&mut db, "INSERT INTO t(a) VALUES(5), (20)");
    assert_rows(&mut db, "SELECT v FROM log", &[vec![int(20)]]);
}

/// §2 ("If no WHEN clause is supplied, the SQL statements are executed every time the trigger
/// fires"). The companion to the WHEN test: with NO WHEN, the SAME {5, 20} INSERT fires for BOTH
/// rows, so `log` holds {5, 20}.
#[test]
fn absent_when_clause_fires_for_every_row() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "CREATE TABLE log(v INTEGER)");
    exec(&mut db, "CREATE TRIGGER tr AFTER INSERT ON t BEGIN INSERT INTO log(v) VALUES(NEW.a); END");
    exec(&mut db, "INSERT INTO t(a) VALUES(5), (20)");
    assert_rows_unordered(&mut db, "SELECT v FROM log", &[vec![int(5)], vec![int(20)]]);
}

/// §2 (WHEN may access OLD and NEW on an UPDATE, both being valid). `WHEN NEW.a > OLD.a` fires the
/// body only when the update INCREASES a. Seeded at 5 then updated to 10 (an increase), WHEN is
/// true, so the body logs the new value 10.
#[test]
fn when_clause_can_reference_old_and_new() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "CREATE TABLE log(v INTEGER)");
    exec(&mut db, "INSERT INTO t(a) VALUES(5)");
    exec(
        &mut db,
        "CREATE TRIGGER tr AFTER UPDATE ON t WHEN NEW.a > OLD.a BEGIN INSERT INTO log(v) VALUES(NEW.a); END",
    );
    exec(&mut db, "UPDATE t SET a = 10");
    assert_rows(&mut db, "SELECT v FROM log", &[vec![int(10)]]);
}

/// §2 ("'UPDATE OF column-name' … will only fire if column-name appears on the left-hand side of
/// one of the terms in the SET clause"). An `UPDATE OF a` trigger fires when the UPDATE assigns a,
/// so `SET a = 100` logs the new value.
#[test]
fn update_of_named_column_fires() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b INTEGER)");
    exec(&mut db, "CREATE TABLE log(v INTEGER)");
    exec(&mut db, "INSERT INTO t(a, b) VALUES(1, 2)");
    exec(
        &mut db,
        "CREATE TRIGGER tr AFTER UPDATE OF a ON t BEGIN INSERT INTO log(v) VALUES(NEW.a); END",
    );
    exec(&mut db, "UPDATE t SET a = 100");
    assert_rows(&mut db, "SELECT v FROM log", &[vec![int(100)]]);
}

/// §2 (the converse of the `UPDATE OF` rule: it does NOT fire when the named column is not
/// assigned). ANCHORED so it is not a bare un-fired gap-pin: first `SET a = …` PROVES the trigger
/// fires (log count 1 — RED today), then `SET b = …` (b un-named) must NOT fire, so the count
/// stays 1 relative to that proven baseline (the second UPDATE adds nothing).
#[test]
fn update_of_unnamed_column_does_not_fire_relative_to_a_fired_baseline() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b INTEGER)");
    exec(&mut db, "CREATE TABLE log(v INTEGER)");
    exec(&mut db, "INSERT INTO t(a, b) VALUES(1, 2)");
    exec(
        &mut db,
        "CREATE TRIGGER tr AFTER UPDATE OF a ON t BEGIN INSERT INTO log(v) VALUES(NEW.a); END",
    );
    // Baseline: assigning the NAMED column `a` fires exactly once (RED today — log is empty now).
    exec(&mut db, "UPDATE t SET a = 100");
    assert_scalar(&mut db, "SELECT count(*) FROM log", int(1));
    // Assigning only the UN-named column `b` must NOT fire, so the count is unchanged from the
    // proven-fired baseline of 1 (this is a no-fire measured against a firing, not an empty pin).
    exec(&mut db, "UPDATE t SET b = 200");
    assert_scalar(&mut db, "SELECT count(*) FROM log", int(1));
}

/// §2 ("only fire if column-name appears … in the SET clause") for a MULTI-column `UPDATE OF a, b`:
/// the trigger fires if ANY named column is assigned. Here only `b` is assigned, and b IS named, so
/// it fires and logs the new b value.
#[test]
fn update_of_multiple_columns_fires_when_any_named_column_assigned() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b INTEGER, c INTEGER)");
    exec(&mut db, "CREATE TABLE log(v INTEGER)");
    exec(&mut db, "INSERT INTO t(a, b, c) VALUES(1, 2, 3)");
    exec(
        &mut db,
        "CREATE TRIGGER tr AFTER UPDATE OF a, b ON t BEGIN INSERT INTO log(v) VALUES(NEW.b); END",
    );
    exec(&mut db, "UPDATE t SET b = 20");
    assert_rows(&mut db, "SELECT v FROM log", &[vec![int(20)]]);
}

/// §2 ("Triggers are database operations that are automatically performed"): two independent
/// triggers on the SAME event BOTH fire. Each tags its `log` row, so an INSERT of a=5 produces
/// both ('t1', 5) and ('t2', 5).
#[test]
fn multiple_triggers_on_same_event_all_fire() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "CREATE TABLE log(src TEXT, v INTEGER)");
    exec(&mut db, "CREATE TRIGGER tr1 AFTER INSERT ON t BEGIN INSERT INTO log(src, v) VALUES('t1', NEW.a); END");
    exec(&mut db, "CREATE TRIGGER tr2 AFTER INSERT ON t BEGIN INSERT INTO log(src, v) VALUES('t2', NEW.a); END");
    exec(&mut db, "INSERT INTO t(a) VALUES(5)");
    assert_rows_unordered(
        &mut db,
        "SELECT src, v FROM log",
        &[vec![text("t1"), int(5)], vec![text("t2"), int(5)]],
    );
}

/// §2 (a trigger body is a sequence of statements, all run per firing). A body with TWO INSERTs
/// runs BOTH, so one firing writes to both `log_a` and `log_b`.
#[test]
fn multi_statement_trigger_body_runs_every_statement() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "CREATE TABLE log_a(v INTEGER)");
    exec(&mut db, "CREATE TABLE log_b(v INTEGER)");
    exec(
        &mut db,
        "CREATE TRIGGER tr AFTER INSERT ON t BEGIN \
         INSERT INTO log_a(v) VALUES(NEW.a); \
         INSERT INTO log_b(v) VALUES(NEW.a + 1); END",
    );
    exec(&mut db, "INSERT INTO t(a) VALUES(10)");
    assert_rows(&mut db, "SELECT v FROM log_a", &[vec![int(10)]]);
    assert_rows(&mut db, "SELECT v FROM log_b", &[vec![int(11)]]);
}

/// §2 (the WHEN clause and body "may access elements of the row" via NEW): a body may compute an
/// EXPRESSION over NEW columns. `NEW.a + NEW.b` is evaluated per firing, so inserting (3, 4) logs 7.
#[test]
fn trigger_body_computes_expression_over_new() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b INTEGER)");
    exec(&mut db, "CREATE TABLE log(total INTEGER)");
    exec(
        &mut db,
        "CREATE TRIGGER tr AFTER INSERT ON t BEGIN INSERT INTO log(total) VALUES(NEW.a + NEW.b); END",
    );
    exec(&mut db, "INSERT INTO t(a, b) VALUES(3, 4)");
    assert_rows(&mut db, "SELECT total FROM log", &[vec![int(7)]]);
}

/// §2 (INSERT ⇒ NEW valid): a single firing can read MULTIPLE NEW columns into one logged row.
/// Inserting (1, 'x') logs the pair (1, 'x') — pinning that every NEW column is available, not
/// just the first.
#[test]
fn insert_trigger_logs_multiple_new_columns() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
    exec(&mut db, "CREATE TABLE log(a INTEGER, b TEXT)");
    exec(
        &mut db,
        "CREATE TRIGGER tr AFTER INSERT ON t BEGIN INSERT INTO log(a, b) VALUES(NEW.a, NEW.b); END",
    );
    exec(&mut db, "INSERT INTO t(a, b) VALUES(1, 'x')");
    assert_rows(&mut db, "SELECT a, b FROM log", &[vec![int(1), text("x")]]);
}

/// §2 ("Triggers are removed using the DROP TRIGGER statement"). ANCHORED so it is not a bare
/// un-fired gap-pin: the first INSERT PROVES the trigger fires (log count 1 — RED today); after
/// `DROP TRIGGER`, a second INSERT must add NOTHING, so the count stays 1 relative to that proven
/// baseline (the dropped trigger no longer fires).
#[test]
fn drop_trigger_stops_further_firing_relative_to_a_fired_baseline() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "CREATE TABLE log(v INTEGER)");
    exec(&mut db, "CREATE TRIGGER tr AFTER INSERT ON t BEGIN INSERT INTO log(v) VALUES(NEW.a); END");
    // Baseline: the trigger fires on this INSERT (log count 1 — fails today, passes on firing).
    exec(&mut db, "INSERT INTO t(a) VALUES(1)");
    assert_scalar(&mut db, "SELECT count(*) FROM log", int(1));
    // After DROP, the next INSERT must NOT fire, so the count is unchanged from the fired baseline.
    exec(&mut db, "DROP TRIGGER tr");
    exec(&mut db, "INSERT INTO t(a) VALUES(2)");
    assert_scalar(&mut db, "SELECT count(*) FROM log", int(1));
}

// =============================================================================
// CATEGORY 2 — COMPOUND. Still loud-signal (fails now, passes on landing), but each
// depends on a SECOND unimplemented feature besides firing, so it may stay failing
// until BOTH land. Each is marked with the extra feature it needs.
// =============================================================================

/// §3 ("If an INSTEAD OF INSERT trigger exists on a view … No actual insert occurs. Instead, the
/// statements contained within the trigger are run"). Inserting into the view runs the body, which
/// writes the BASE table, so `base` gains (1, 'x') even though the view itself is not directly
/// insertable.
// COMPOUND: also needs view-target INSTEAD OF acceptance + view-DML → INSTEAD OF routing (views
// themselves already work; only the trigger firing/routing is missing).
#[test]
fn instead_of_insert_on_view_redirects_to_base_table() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE base(id INTEGER, val TEXT)");
    exec(&mut db, "CREATE VIEW v AS SELECT id, val FROM base");
    exec(
        &mut db,
        "CREATE TRIGGER tr INSTEAD OF INSERT ON v BEGIN INSERT INTO base(id, val) VALUES(NEW.id, NEW.val); END",
    );
    exec(&mut db, "INSERT INTO v(id, val) VALUES(1, 'x')");
    assert_rows_unordered(&mut db, "SELECT id, val FROM base", &[vec![int(1), text("x")]]);
}

/// §3 + §4 (the `cust_addr_chng` worked example): an INSTEAD OF UPDATE trigger on a view redirects
/// the update to the base table. Updating the view's cust_addr for cust_id=1 runs the body
/// `UPDATE customer SET cust_addr=NEW.cust_addr WHERE cust_id=NEW.cust_id`, so the base customer
/// row's address becomes 'new addr'.
// COMPOUND: also needs view-target INSTEAD OF acceptance + view-DML → INSTEAD OF routing.
#[test]
fn instead_of_update_on_view_redirects_to_base_table() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE customer(cust_id INTEGER PRIMARY KEY, cust_name TEXT, cust_addr TEXT)");
    exec(&mut db, "INSERT INTO customer(cust_id, cust_name, cust_addr) VALUES(1, 'Jack', 'old addr')");
    exec(&mut db, "CREATE VIEW customer_address AS SELECT cust_id, cust_addr FROM customer");
    exec(
        &mut db,
        "CREATE TRIGGER cust_addr_chng INSTEAD OF UPDATE OF cust_addr ON customer_address \
         BEGIN UPDATE customer SET cust_addr = NEW.cust_addr WHERE cust_id = NEW.cust_id; END",
    );
    exec(&mut db, "UPDATE customer_address SET cust_addr = 'new addr' WHERE cust_id = 1");
    assert_rows(&mut db, "SELECT cust_id, cust_addr FROM customer", &[vec![int(1), text("new addr")]]);
}

/// §3 ("INSTEAD OF DELETE … work[s] the same way … against views"). Deleting from the view runs the
/// INSTEAD OF DELETE body, which deletes the matching BASE row via OLD, so `base` loses the row.
// COMPOUND: also needs view-target INSTEAD OF acceptance + view-DML → INSTEAD OF routing.
#[test]
fn instead_of_delete_on_view_redirects_to_base_table() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE base(id INTEGER, val TEXT)");
    exec(&mut db, "INSERT INTO base(id, val) VALUES(1, 'a'), (2, 'b')");
    exec(&mut db, "CREATE VIEW v AS SELECT id, val FROM base");
    exec(
        &mut db,
        "CREATE TRIGGER tr INSTEAD OF DELETE ON v BEGIN DELETE FROM base WHERE id = OLD.id; END",
    );
    exec(&mut db, "DELETE FROM v WHERE id = 1");
    // Only base row 2 remains after the redirected delete of id=1.
    assert_rows(&mut db, "SELECT id, val FROM base", &[vec![int(2), text("b")]]);
}

/// §2 (WHEN gates the body) + §6 (RAISE(ABORT,…) "the specified ON CONFLICT processing is performed
/// and the current query terminates. An error code of SQLITE_CONSTRAINT is returned"). One BEFORE
/// INSERT trigger, gated by `WHEN NEW.a < 0`, exercises BOTH branches of the RAISE(ABORT) contract:
///   * a valid row (a=5, WHEN false) is ADMITTED — the RAISE body never runs, so the INSERT succeeds
///     and the row is present;
///   * a forbidden row (a=-1, WHEN true) fires RAISE(ABORT), so THAT INSERT errors and writes
///     nothing — the table stays unchanged, still holding only the admitted valid row.
/// The valid-row step fails today (the firing INSERT errors at plan time because RAISE() does not
/// yet bind), so this is a genuine fail→pass signal — NOT a coincidentally-passing pin. The
/// forbidden-row step then guards the "errors AND data unchanged" bar: it fails if a landed
/// RAISE(ABORT) does not error, or errors but fails to roll the offending row back.
// COMPOUND: also needs the RAISE() function (its argument does not yet bind), on top of firing and
// per-row WHEN evaluation.
#[test]
fn before_insert_raise_abort_blocks_forbidden_row_but_admits_valid_row() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(
        &mut db,
        "CREATE TRIGGER tr BEFORE INSERT ON t WHEN NEW.a < 0 \
         BEGIN SELECT RAISE(ABORT, 'negative not allowed'); END",
    );
    // A valid row fails WHEN (5 < 0 is false), so the RAISE body never runs and the INSERT proceeds.
    exec(&mut db, "INSERT INTO t(a) VALUES(5)");
    assert_rows(&mut db, "SELECT a FROM t", &[vec![int(5)]]);
    // A forbidden row satisfies WHEN, firing RAISE(ABORT): the INSERT must error …
    assert_exec_error(&mut db, "INSERT INTO t(a) VALUES(-1)");
    // … and write nothing — the table is unchanged, still holding only the admitted valid row.
    assert_rows(&mut db, "SELECT a FROM t", &[vec![int(5)]]);
}

/// §RAISE: "When RAISE(IGNORE) is called, the remainder of the current trigger program, the
/// statement that caused the trigger program to execute and any subsequent trigger programs
/// that would have been executed are abandoned. No database changes are rolled back."
/// (lang_createtrigger.html). The classic row-FILTER idiom: a BEFORE INSERT trigger gated by
/// `WHEN NEW.x < 0` fires `RAISE(IGNORE)`, so a multi-row insert silently DROPS the negative
/// rows and keeps the rest — WITHOUT erroring the statement (unlike RAISE(ABORT)). Here
/// inserting 1,-1,2,-5,3 leaves exactly {1,2,3}.
#[test]
fn before_insert_raise_ignore_skips_the_row_without_error() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE g(x INTEGER)");
    exec(
        &mut db,
        "CREATE TRIGGER tr BEFORE INSERT ON g WHEN NEW.x < 0 BEGIN SELECT RAISE(IGNORE); END",
    );
    // One multi-row INSERT: the negatives fire RAISE(IGNORE) (skipped, no error), the rest land.
    exec(&mut db, "INSERT INTO g(x) VALUES(1),(-1),(2),(-5),(3)");
    assert_rows(&mut db, "SELECT x FROM g ORDER BY x", &[vec![int(1)], vec![int(2)], vec![int(3)]]);
}

/// §RAISE (IGNORE on an UPDATE): a BEFORE UPDATE trigger that fires `RAISE(IGNORE)` for a
/// forbidden new value ABANDONS that row's update (the existing row is left untouched, no
/// error) while other rows update normally. Rows start at 1,2,3; `UPDATE g SET x = x + 10`
/// with a `WHEN NEW.x >= 13` guard skips the row that would become 13 (from 3) and updates
/// the rest, so the table ends {11, 12, 3}.
#[test]
fn before_update_raise_ignore_abandons_only_the_guarded_row() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE g(x INTEGER)");
    exec(&mut db, "INSERT INTO g(x) VALUES(1),(2),(3)");
    exec(
        &mut db,
        "CREATE TRIGGER tr BEFORE UPDATE ON g WHEN NEW.x >= 13 BEGIN SELECT RAISE(IGNORE); END",
    );
    exec(&mut db, "UPDATE g SET x = x + 10");
    assert_rows(&mut db, "SELECT x FROM g ORDER BY x", &[vec![int(3)], vec![int(11)], vec![int(12)]]);
}

/// §RAISE (IGNORE on a DELETE): a BEFORE DELETE trigger firing `RAISE(IGNORE)` for a
/// protected row ABANDONS that row's delete (it survives, no error) while unprotected rows
/// are deleted. Rows 1,2,3 with `WHEN OLD.x = 2` protecting row 2: `DELETE FROM g` removes
/// 1 and 3 and keeps 2.
#[test]
fn before_delete_raise_ignore_protects_the_guarded_row() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE g(x INTEGER)");
    exec(&mut db, "INSERT INTO g(x) VALUES(1),(2),(3)");
    exec(
        &mut db,
        "CREATE TRIGGER tr BEFORE DELETE ON g WHEN OLD.x = 2 BEGIN SELECT RAISE(IGNORE); END",
    );
    exec(&mut db, "DELETE FROM g");
    assert_rows(&mut db, "SELECT x FROM g", &[vec![int(2)]]);
}

/// §RAISE (IGNORE does not roll back prior changes): "No database changes are rolled back."
/// A BEFORE INSERT trigger writes an audit row into ANOTHER table and THEN fires
/// `RAISE(IGNORE)` on the guarded row. The audit write it made before the RAISE survives,
/// and only the guarded row's own insert is abandoned — proving IGNORE abandons the row's
/// operation, not the side effects the trigger body already performed.
#[test]
fn raise_ignore_keeps_side_effects_made_before_it() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE g(x INTEGER)");
    exec(&mut db, "CREATE TABLE audit(x INTEGER)");
    exec(
        &mut db,
        "CREATE TRIGGER tr BEFORE INSERT ON g WHEN NEW.x < 0 \
         BEGIN INSERT INTO audit(x) VALUES(NEW.x); SELECT RAISE(IGNORE); END",
    );
    exec(&mut db, "INSERT INTO g(x) VALUES(1),(-1),(2)");
    // g dropped the negative; audit KEPT the row the trigger wrote before RAISE(IGNORE).
    assert_rows(&mut db, "SELECT x FROM g ORDER BY x", &[vec![int(1)], vec![int(2)]]);
    assert_rows(&mut db, "SELECT x FROM audit", &[vec![int(-1)]]);
}

// =============================================================================
// CATEGORY 3 — RECURSION (one case). A trigger whose body causes the same event
// recurses when `PRAGMA recursive_triggers=ON`; the WHEN clause bounds the depth.
// =============================================================================

/// §2 (a trigger action is itself DML that can fire triggers) with `PRAGMA recursive_triggers=ON`:
/// an AFTER INSERT trigger on `t` that inserts NEW.n+1 back into `t`, gated by `WHEN NEW.n < 3`,
/// recurses in a BOUNDED chain. Seeding n=1 fires: 1→2→3, stopping when WHEN(3<3) is false, so `t`
/// ends holding {1, 2, 3}. (RED today: the executor does not fire, so `t` holds only the seed {1}.)
// COMPOUND: also needs the recursive_triggers pragma honored (an unknown pragma is a silent no-op
// today) in addition to firing. The WHEN bound keeps the recursion finite and well-defined.
#[test]
fn recursive_trigger_bounded_by_when_with_pragma() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(n INTEGER)");
    exec(
        &mut db,
        "CREATE TRIGGER tr AFTER INSERT ON t WHEN NEW.n < 3 BEGIN INSERT INTO t(n) VALUES(NEW.n + 1); END",
    );
    exec(&mut db, "PRAGMA recursive_triggers = ON");
    exec(&mut db, "INSERT INTO t(n) VALUES(1)");
    assert_rows_unordered(&mut db, "SELECT n FROM t", &[vec![int(1)], vec![int(2)], vec![int(3)]]);
}
