//! Conformance: a firing statement's leading `WITH` (CTE) must NOT leak into the
//! bodies of the triggers it fires.
//!
//! Spec basis (`spec/sqlite-doc/lang_with.html` §1): a CTE "act[s] like a temporary
//! view that exists only for the duration of a single SQL statement." A trigger body is
//! a SEPARATE compiled program (`lang_createtrigger.html`), so the CTEs of the statement
//! that FIRES a trigger are invisible inside that trigger. An unqualified name in a
//! trigger body therefore binds the base table/view of that name, never a same-named CTE
//! of the firing statement.
//!
//! Why this is a real hazard in THIS engine: CTE visibility is carried on a thread-local
//! scope stack that `INSERT`/`UPDATE`/`DELETE` planning pushes the firing statement's
//! CTEs onto, while trigger-action bodies are compiled (during that same planning) under
//! a fresh planning context that shares that thread-local stack. If the firing
//! statement's CTEs are still on the stack when a trigger body is compiled, the body's
//! `FROM <name>` resolves to the leaked CTE — shadowing the base table and lowering to a
//! CTE reference whose id belongs to the FIRING statement's plan, not the trigger
//! action's own (empty) plan: a dangling reference, i.e. a malformed plan.
//!
//! The witness below makes the leak a HARD, unambiguous failure: the base table `c` has
//! TWO columns `(a, b)` but the firing CTE `c` has only `a`. The trigger body selects
//! `a, b`, so if `c` leaks into the trigger the bind of `b` fails with `no such column:
//! b` at plan time and the whole statement errors — where real SQLite (and a correctly
//! isolated planner) binds the base table and succeeds. Each firing statement also USES
//! its own CTE (INSERT source / UPDATE SET subquery / DELETE WHERE subquery), so these
//! cases jointly pin BOTH halves of the scope rule: the CTE is visible to the firing
//! statement's own body, and invisible to the trigger it fires.
//!
//! (Triggers are compiled at plan time here even though the executor does not yet FIRE
//! them, so the leak surfaces at plan/execute time regardless — these assertions do not
//! depend on trigger execution.)

mod conformance;

use conformance::*;

#[test]
fn firing_insert_cte_does_not_leak_into_after_insert_trigger_body() {
    let mut db = mem();
    // A REAL base table `c` with two columns; the trigger intends to read it.
    exec(&mut db, "CREATE TABLE c(a INTEGER, b INTEGER)");
    exec(&mut db, "CREATE TABLE t(x INTEGER)");
    exec(&mut db, "CREATE TABLE sink(a INTEGER, b INTEGER)");
    exec(
        &mut db,
        "CREATE TRIGGER trg_ins AFTER INSERT ON t BEGIN INSERT INTO sink SELECT a, b FROM c; END",
    );

    // The firing INSERT carries a leading `WITH c` (only column `a`) that its OWN source
    // legitimately uses, then fires `trg_ins`. The trigger's `FROM c` must bind the base
    // table `c(a, b)`, NOT the leaked 1-column CTE — otherwise `b` is unresolved and this
    // errors at plan time.
    exec(&mut db, "WITH c AS (SELECT 1 AS a) INSERT INTO t SELECT a FROM c");

    // The INSERT's own body saw the CTE (x = 1 came from it); the statement did not error.
    assert_rows(&mut db, "SELECT x FROM t", &[vec![int(1)]]);
}

#[test]
fn firing_update_cte_does_not_leak_into_after_update_trigger_body() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE c(a INTEGER, b INTEGER)");
    exec(&mut db, "CREATE TABLE t(x INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (0)");
    exec(&mut db, "CREATE TABLE sink(a INTEGER, b INTEGER)");
    exec(
        &mut db,
        "CREATE TRIGGER trg_upd AFTER UPDATE ON t BEGIN INSERT INTO sink SELECT a, b FROM c; END",
    );

    // The firing UPDATE uses the CTE in its SET subquery (x <- 5), then fires `trg_upd`,
    // whose `FROM c` must bind the base table `c(a, b)` and not the 1-column CTE.
    exec(&mut db, "WITH c AS (SELECT 5 AS a) UPDATE t SET x = (SELECT a FROM c)");

    assert_rows(&mut db, "SELECT x FROM t", &[vec![int(5)]]);
}

#[test]
fn firing_delete_cte_does_not_leak_into_after_delete_trigger_body() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE c(a INTEGER, b INTEGER)");
    exec(&mut db, "CREATE TABLE t(x INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1), (2)");
    exec(&mut db, "CREATE TABLE sink(a INTEGER, b INTEGER)");
    exec(
        &mut db,
        "CREATE TRIGGER trg_del AFTER DELETE ON t BEGIN INSERT INTO sink SELECT a, b FROM c; END",
    );

    // The firing DELETE uses the CTE in its WHERE subquery (deletes x = 1), then fires
    // `trg_del`, whose `FROM c` must bind the base table `c(a, b)` and not the CTE.
    exec(&mut db, "WITH c AS (SELECT 1 AS a) DELETE FROM t WHERE x IN (SELECT a FROM c)");

    assert_rows(&mut db, "SELECT x FROM t", &[vec![int(2)]]);
}
