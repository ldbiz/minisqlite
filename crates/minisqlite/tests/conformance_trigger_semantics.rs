//! Trigger-firing SEMANTICS that the base-table `conformance_triggers.rs` battery does
//! not pin — added here in a separate file so the base spec-transcribed suite stays
//! untouched. Each test asserts the SQLite-documented observable behavior through the
//! real facade:
//!
//! * `changes()` / `last_insert_rowid()` EXCLUDE a trigger program's own writes, while
//!   `total_changes()` INCLUDES them (spec/sqlite-doc/lang_corefunc.html,
//!   c3ref/changes.html, c3ref/last_insert_rowid.html, c3ref/total_changes.html);
//! * `PRAGMA recursive_triggers` gates runtime recursion, defaulting OFF (pragma.html):
//!   a DML inside a trigger body fires no further triggers unless it is ON;
//! * a trigger ACTION that is an UPDATE / DELETE on ANOTHER base table runs its
//!   `base = outer.len()` offset arithmetic (exercised only when the action's scan carries
//!   the OLD/NEW frame — never at top level), including OLD/NEW refs below `base`;
//! * a trigger-action uncorrelated subquery does NOT collide with the enclosing
//!   statement's cached subquery of the same `SubqueryId` (the per-action cache reset).

mod conformance;
use conformance::*;

// =============================================================================
// changes() / last_insert_rowid() / total_changes() across the trigger boundary
// =============================================================================

/// `changes()` is "exclusive of statements in lower-level triggers"
/// (lang_corefunc.html): a single-row INSERT that fires an AFTER trigger inserting
/// elsewhere reports `changes() == 1`, not 2. (The trigger DID fire — asserted via `log`.)
#[test]
fn changes_excludes_trigger_caused_writes() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "CREATE TABLE log(v INTEGER)");
    exec(&mut db, "CREATE TRIGGER tr AFTER INSERT ON t BEGIN INSERT INTO log(v) VALUES(NEW.a); END");
    exec(&mut db, "INSERT INTO t(a) VALUES(7)");

    // The trigger fired (log has the row) and the top-level row landed …
    assert_rows(&mut db, "SELECT v FROM log", &[vec![int(7)]]);
    assert_rows(&mut db, "SELECT a FROM t", &[vec![int(7)]]);
    // … but changes() counts ONLY the top-level INSERT's one row.
    assert_scalar(&mut db, "SELECT changes()", int(1));
}

/// `total_changes()` "counts changes made by all trigger contexts"
/// (c3ref/total_changes.html): the SAME statement that reports `changes() == 1` reports
/// `total_changes() == 2` (the top-level row plus the trigger's row).
#[test]
fn total_changes_includes_trigger_caused_writes() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "CREATE TABLE log(v INTEGER)");
    exec(&mut db, "CREATE TRIGGER tr AFTER INSERT ON t BEGIN INSERT INTO log(v) VALUES(NEW.a); END");
    exec(&mut db, "INSERT INTO t(a) VALUES(7)");

    assert_scalar(&mut db, "SELECT changes()", int(1));
    assert_scalar(&mut db, "SELECT total_changes()", int(2));
}

/// `last_insert_rowid()` "reverts to what it was before the trigger was fired" once the
/// trigger program ends (c3ref/last_insert_rowid.html). `log` is pre-seeded so its
/// trigger-written rowid (4) is distinct from the top-level row's rowid (1); after the
/// statement, `last_insert_rowid()` must be 1, not 4.
#[test]
fn last_insert_rowid_reverts_after_trigger_program() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "CREATE TABLE log(v INTEGER)");
    exec(&mut db, "INSERT INTO log(v) VALUES(10),(20),(30)"); // log rowids 1,2,3
    exec(&mut db, "CREATE TRIGGER tr AFTER INSERT ON t BEGIN INSERT INTO log(v) VALUES(NEW.a); END");
    exec(&mut db, "INSERT INTO t(a) VALUES(7)"); // t.rowid = 1; the trigger writes log rowid 4

    // The trigger wrote log rowid 4 (four rows now) …
    assert_scalar(&mut db, "SELECT count(*) FROM log", int(4));
    // … but last_insert_rowid() reverted to the top-level INSERT's rowid (t.rowid = 1).
    assert_scalar(&mut db, "SELECT last_insert_rowid()", int(1));
}

// =============================================================================
// PRAGMA recursive_triggers gate (defaults OFF)
// =============================================================================

/// recursive_triggers defaults OFF (pragma.html): a self-writing AFTER INSERT trigger with
/// NO `WHEN` bound must fire exactly ONCE (the nested INSERT does not re-fire it), so
/// seeding n=1 leaves `t` = {1, 2}. With recursion wrongly always-on this would recurse to
/// the depth cap and roll the whole statement back to empty.
#[test]
fn recursive_triggers_default_off_fires_one_level() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(n INTEGER)");
    exec(&mut db, "CREATE TRIGGER tr AFTER INSERT ON t BEGIN INSERT INTO t(n) VALUES(NEW.n + 1); END");
    exec(&mut db, "INSERT INTO t(n) VALUES(1)");
    assert_rows_unordered(&mut db, "SELECT n FROM t", &[vec![int(1)], vec![int(2)]]);
}

/// The pragma gates recursion BOTH ways. With a `WHEN NEW.n < 3` bound: ON recurses the
/// full 1→2→3 chain; OFF (even though the WHEN would admit deeper) fires just one level to
/// {1, 2}. Two connections so neither pollutes the other.
#[test]
fn recursive_triggers_pragma_gates_recursion_both_ways() {
    // recursion ON: the bounded chain runs to completion.
    let mut on = mem();
    exec(&mut on, "CREATE TABLE t(n INTEGER)");
    exec(
        &mut on,
        "CREATE TRIGGER tr AFTER INSERT ON t WHEN NEW.n < 3 BEGIN INSERT INTO t(n) VALUES(NEW.n + 1); END",
    );
    exec(&mut on, "PRAGMA recursive_triggers = ON");
    exec(&mut on, "INSERT INTO t(n) VALUES(1)");
    assert_rows_unordered(&mut on, "SELECT n FROM t", &[vec![int(1)], vec![int(2)], vec![int(3)]]);

    // recursion OFF (explicit): the same trigger fires only one level even though WHEN
    // would admit n=2's insert to recurse.
    let mut off = mem();
    exec(&mut off, "CREATE TABLE t(n INTEGER)");
    exec(
        &mut off,
        "CREATE TRIGGER tr AFTER INSERT ON t WHEN NEW.n < 3 BEGIN INSERT INTO t(n) VALUES(NEW.n + 1); END",
    );
    exec(&mut off, "PRAGMA recursive_triggers = OFF");
    exec(&mut off, "INSERT INTO t(n) VALUES(1)");
    assert_rows_unordered(&mut off, "SELECT n FROM t", &[vec![int(1)], vec![int(2)]]);
}

/// GET form reports the current setting (0 default, 1 after `= ON`), matching sqlite's
/// one-row `recursive_triggers` result.
#[test]
fn pragma_recursive_triggers_get_reports_state() {
    let mut db = mem();
    assert_scalar(&mut db, "PRAGMA recursive_triggers", int(0));
    exec(&mut db, "PRAGMA recursive_triggers = ON");
    assert_scalar(&mut db, "PRAGMA recursive_triggers", int(1));
    exec(&mut db, "PRAGMA recursive_triggers = OFF");
    assert_scalar(&mut db, "PRAGMA recursive_triggers", int(0));
}

// =============================================================================
// Trigger ACTION that is an UPDATE / DELETE on another base table (base > 0)
// =============================================================================

/// A trigger whose action is an UPDATE on ANOTHER base table: the action's scan carries the
/// firing OLD/NEW frame as its correlated outer, so the UPDATE runs its `base = outer.len()`
/// offset path (rowid at `base + n`, index key `old[base..base + n]`, width check
/// `base + n + 1`) — exercised ONLY here, never at top level. Two inserts fire it twice, so
/// the counter reaches 2.
#[test]
fn nested_update_action_on_other_table_uses_frame_offset() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "CREATE TABLE ctr(n INTEGER)");
    exec(&mut db, "INSERT INTO ctr VALUES(0)");
    exec(&mut db, "CREATE TRIGGER tr AFTER INSERT ON t BEGIN UPDATE ctr SET n = n + 1; END");
    exec(&mut db, "INSERT INTO t VALUES(1),(2)");
    assert_scalar(&mut db, "SELECT n FROM ctr", int(2));
    // The counter's changes are trigger-caused, so the outer INSERT's changes() is its own
    // two rows. total_changes() is connection-lifetime: the initial `INSERT INTO ctr` (1) +
    // the two t inserts (2) + the two nested ctr updates (2) = 5.
    assert_scalar(&mut db, "SELECT changes()", int(2));
    assert_scalar(&mut db, "SELECT total_changes()", int(5));
}

/// A trigger whose action is a DELETE on ANOTHER base table, with the WHERE referencing
/// `NEW` (a frame ref BELOW `base`) alongside the table's own column (at `base`): pins that
/// the delete's `base` slicing resolves the table row while the frame ref resolves the
/// firing row. Inserting a=2 deletes exactly `other.id = 2`, leaving {1, 3}.
#[test]
fn nested_delete_action_on_other_table_resolves_frame_and_base() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "CREATE TABLE other(id INTEGER)");
    exec(&mut db, "INSERT INTO other VALUES(1),(2),(3)");
    exec(&mut db, "CREATE TRIGGER tr AFTER INSERT ON t BEGIN DELETE FROM other WHERE id = NEW.a; END");
    exec(&mut db, "INSERT INTO t VALUES(2)");
    assert_rows_unordered(&mut db, "SELECT id FROM other", &[vec![int(1)], vec![int(3)]]);
}

// =============================================================================
// Trigger-action subquery cache isolation (per-action reset)
// =============================================================================

/// WITNESS A — a valid statement must NOT error from a stale subquery-cache entry. The
/// enclosing DELETE caches an uncorrelated SCALAR subquery `(SELECT max(g) FROM gate)` at
/// SubqueryId 0; the AFTER-DELETE action's own uncorrelated `IN (SELECT g FROM gate)` also
/// has id 0. Without the per-action cache reset the action reads the Scalar entry and aborts
/// with a "cached subquery kind mismatch"; with it, `t` empties and `audit` gets {1}.
#[test]
fn action_subquery_does_not_collide_with_enclosing_scalar() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE gate(g)");
    exec(&mut db, "INSERT INTO gate VALUES (1)");
    exec(&mut db, "CREATE TABLE t(id)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    exec(&mut db, "CREATE TABLE audit(id)");
    exec(
        &mut db,
        "CREATE TRIGGER tr AFTER DELETE ON t \
         BEGIN INSERT INTO audit SELECT OLD.id WHERE OLD.id IN (SELECT g FROM gate); END",
    );
    exec(&mut db, "DELETE FROM t WHERE id = (SELECT max(g) FROM gate)");

    assert_rows(&mut db, "SELECT id FROM t", &[]);
    assert_rows(&mut db, "SELECT id FROM audit", &[vec![int(1)]]);
}

/// WITNESS B — a trigger-action subquery must compute over ITS OWN data, not silently reuse
/// the enclosing statement's cached set of the same kind at the same id. The enclosing
/// INSERT caches `IN (SELECT x FROM a)` = {5} at id 0; the action's `IN (SELECT y FROM b)` =
/// {99} is also id 0. Without the per-action reset the action reuses {5}, so `5 IN {5}` is
/// true and `log` wrongly gets a row; with it, `5 IN {99}` is false and `log` stays empty.
#[test]
fn action_subquery_does_not_read_enclosing_in_set() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE a(x)");
    exec(&mut db, "INSERT INTO a VALUES (5)");
    exec(&mut db, "CREATE TABLE b(y)");
    exec(&mut db, "INSERT INTO b VALUES (99)");
    exec(&mut db, "CREATE TABLE t2(v)");
    exec(&mut db, "CREATE TABLE log(v)");
    exec(
        &mut db,
        "CREATE TRIGGER tr2 AFTER INSERT ON t2 \
         BEGIN INSERT INTO log SELECT NEW.v WHERE NEW.v IN (SELECT y FROM b); END",
    );
    exec(&mut db, "INSERT INTO t2(v) SELECT 5 WHERE 5 IN (SELECT x FROM a)");

    // The enclosing INSERT did land its row (5 IN {5}) …
    assert_rows(&mut db, "SELECT v FROM t2", &[vec![int(5)]]);
    // … but the action saw its OWN set {99}: 5 not in {99}, so log stays empty.
    assert_scalar(&mut db, "SELECT count(*) FROM log", int(0));
}

// =============================================================================
// Auto-rowid high-water TOCTOU under trigger re-entrancy (same-table action)
// =============================================================================
//
// The write loop seeds an auto-rowid high-water ONCE and advances it only by its own
// writes. A trigger whose action inserts into the SAME auto-rowid table advances the
// table's real max behind that high-water, so the outer loop's next auto rowid can land on
// a rowid the trigger already wrote — and because an auto rowid skips the uniqueness probe,
// table_insert would SILENTLY REPLACE the trigger's row. The fix re-reads the live max
// before each auto write when triggers fire, so the outer row takes a fresh rowid instead.

/// AFTER, multi-row (fully well-defined, no recursion — recursive_triggers default OFF):
/// each outer row's AFTER trigger inserts NEW.v+100 into the same table. All four rows must
/// survive at distinct rowids: [1, 101, 2, 102] by rowid. A stale high-water overwrites the
/// first trigger row (giving the buggy [1, 2, 102]).
#[test]
fn after_trigger_same_table_insert_keeps_distinct_auto_rowids() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(v INTEGER)");
    exec(&mut db, "CREATE TRIGGER tr AFTER INSERT ON t BEGIN INSERT INTO t(v) VALUES(NEW.v + 100); END");
    exec(&mut db, "INSERT INTO t(v) VALUES(1),(2)");
    assert_rows(
        &mut db,
        "SELECT v FROM t ORDER BY rowid",
        &[vec![int(1)], vec![int(101)], vec![int(2)], vec![int(102)]],
    );
    // Corroborating signal: the b-tree has every counted row (no lost write) — count()
    // matches total_changes() (4 inserts: 2 outer + 2 trigger), proving no silent REPLACE.
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(4));
    assert_scalar(&mut db, "SELECT total_changes()", int(4));
}

/// BEFORE, single-row: the BEFORE trigger inserts NEW.v+100 into the same table before the
/// base row is written. Both rows must survive: the trigger's row takes rowid 1, the outer
/// row is re-assigned rowid 2, so ordered by rowid it is [101, 1]. A stale high-water writes
/// the outer row over the trigger's rowid-1 row (giving the buggy single [1]).
#[test]
fn before_trigger_same_table_insert_keeps_distinct_auto_rowids() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(v INTEGER)");
    exec(&mut db, "CREATE TRIGGER tr BEFORE INSERT ON t BEGIN INSERT INTO t(v) VALUES(NEW.v + 100); END");
    exec(&mut db, "INSERT INTO t(v) VALUES(1)");
    assert_rows(&mut db, "SELECT v FROM t ORDER BY rowid", &[vec![int(101)], vec![int(1)]]);
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2));
    assert_scalar(&mut db, "SELECT total_changes()", int(2));
}

/// The same TOCTOU against an INTEGER PRIMARY KEY (rowid-alias) table: the auto-assigned
/// alias value must also be re-derived, and the re-derived rowid must show through the alias
/// column. AFTER trigger inserts into the same table; all rows survive at distinct ids.
#[test]
fn after_trigger_same_table_insert_rederives_rowid_alias() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v INTEGER)");
    // recursive_triggers is OFF by default, so the nested INSERT does not re-fire tr.
    exec(&mut db, "CREATE TRIGGER tr AFTER INSERT ON t BEGIN INSERT INTO t(v) VALUES(NEW.v + 100); END");
    exec(&mut db, "INSERT INTO t(v) VALUES(1),(2)");
    // Rows by id: (1,v=1),(2,v=101),(3,v=2),(4,v=102). The alias id reflects the real rowid.
    assert_rows(
        &mut db,
        "SELECT id, v FROM t ORDER BY id",
        &[
            vec![int(1), int(1)],
            vec![int(2), int(101)],
            vec![int(3), int(2)],
            vec![int(4), int(102)],
        ],
    );
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(4));
}
