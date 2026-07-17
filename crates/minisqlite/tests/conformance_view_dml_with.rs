//! Conformance: a leading `WITH` (common table expression) on a view DML that is
//! redirected through an `INSTEAD OF` trigger.
//!
//! Spec basis:
//!   * `lang_with.html` §1 — a `WITH` clause may prefix `SELECT`, `INSERT`, `UPDATE`, and
//!     `DELETE`, and each CTE "act[s] like a temporary view that exists only for the
//!     duration of a single SQL statement" — the *firing* statement. So a CTE is visible to
//!     that statement's source / WHERE / SET (and their subqueries), but NOT to the separate
//!     statements inside a trigger the DML fires.
//!   * `lang_createtrigger.html` §3 — an `INSTEAD OF <event>` trigger on a view makes the
//!     view updatable for that event; the DML fires the trigger body FOR EACH affected view
//!     row and performs no base write of its own.
//!
//! A view DML combined with `WITH` previously returned a loud "WITH on a view … is not yet
//! supported"; real SQLite accepts it. These tests drive the redirected INSERT source, the
//! DELETE / UPDATE WHERE + SET (including a scalar subquery over a CTE), the recursive-CTE
//! path, and — critically — the isolation invariant that the firing statement's CTE is
//! INVISIBLE inside the fired `INSTEAD OF` body.
//!
//! Expected values are hand-derived from the spec, not from observing this engine.

mod conformance;
use conformance::{assert_exec_error, assert_rows_unordered, exec, int, mem, text};

/// Seed a base table `base(id, val)`, a pass-through view `v(id, val)`, and the three
/// `INSTEAD OF` triggers that redirect each event to `base`.
fn setup_passthrough(db: &mut minisqlite::Connection) {
    exec(db, "CREATE TABLE base(id INTEGER, val TEXT)");
    exec(db, "CREATE VIEW v(id, val) AS SELECT id, val FROM base");
    exec(
        db,
        "CREATE TRIGGER tr_ins INSTEAD OF INSERT ON v \
         BEGIN INSERT INTO base(id, val) VALUES(NEW.id, NEW.val); END",
    );
    exec(
        db,
        "CREATE TRIGGER tr_del INSTEAD OF DELETE ON v \
         BEGIN DELETE FROM base WHERE id = OLD.id; END",
    );
    exec(
        db,
        "CREATE TRIGGER tr_upd INSTEAD OF UPDATE ON v \
         BEGIN UPDATE base SET val = NEW.val WHERE id = NEW.id; END",
    );
}

/// `WITH c AS (VALUES …) INSERT INTO v SELECT … FROM c`: the CTE is the redirected INSERT's
/// row source. Each CTE row becomes a NEW row that fires `INSTEAD OF INSERT`, so both land
/// in `base`.
#[test]
fn with_cte_source_on_view_insert() {
    let mut db = mem();
    setup_passthrough(&mut db);
    exec(
        &mut db,
        "WITH c(id, val) AS (VALUES (1, 'a'), (2, 'b')) \
         INSERT INTO v(id, val) SELECT id, val FROM c",
    );
    assert_rows_unordered(
        &mut db,
        "SELECT id, val FROM base",
        &[vec![int(1), text("a")], vec![int(2), text("b")]],
    );
}

/// `WITH doomed AS (VALUES …) DELETE FROM v WHERE id IN (SELECT … FROM doomed)`: the CTE is
/// referenced by a subquery in the WHERE. The matching view rows (ids 1 and 3) each fire
/// `INSTEAD OF DELETE`, so only id 2 survives.
#[test]
fn with_cte_in_where_on_view_delete() {
    let mut db = mem();
    setup_passthrough(&mut db);
    exec(&mut db, "INSERT INTO base VALUES (1, 'a'), (2, 'b'), (3, 'c')");
    exec(
        &mut db,
        "WITH doomed(x) AS (VALUES (1), (3)) \
         DELETE FROM v WHERE id IN (SELECT x FROM doomed)",
    );
    assert_rows_unordered(&mut db, "SELECT id, val FROM base", &[vec![int(2), text("b")]]);
}

/// `WITH newval AS (VALUES …) UPDATE v SET val = (SELECT n FROM newval) WHERE id = 1`: the
/// CTE is referenced by a scalar subquery in the SET. The matching view row (id 1) fires
/// `INSTEAD OF UPDATE` with NEW.val = 'Z'; id 2 is untouched.
#[test]
fn with_cte_in_set_on_view_update() {
    let mut db = mem();
    setup_passthrough(&mut db);
    exec(&mut db, "INSERT INTO base VALUES (1, 'a'), (2, 'b')");
    exec(
        &mut db,
        "WITH newval(n) AS (VALUES ('Z')) \
         UPDATE v SET val = (SELECT n FROM newval) WHERE id = 1",
    );
    assert_rows_unordered(
        &mut db,
        "SELECT id, val FROM base",
        &[vec![int(1), text("Z")], vec![int(2), text("b")]],
    );
}

/// A `WITH RECURSIVE` source on a view INSERT: the recursive CTE generates 1..=5, and each
/// generated row fires `INSTEAD OF INSERT`. This exercises the recursive-CTE lowering
/// (seed + step + fixpoint) through the view-DML frame source.
#[test]
fn with_recursive_source_on_view_insert() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE base(n INTEGER)");
    exec(&mut db, "CREATE VIEW v(n) AS SELECT n FROM base");
    exec(
        &mut db,
        "CREATE TRIGGER tr_ins INSTEAD OF INSERT ON v \
         BEGIN INSERT INTO base(n) VALUES(NEW.n); END",
    );
    exec(
        &mut db,
        "WITH RECURSIVE seq(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM seq WHERE n < 5) \
         INSERT INTO v(n) SELECT n FROM seq",
    );
    assert_rows_unordered(
        &mut db,
        "SELECT n FROM base",
        &[vec![int(1)], vec![int(2)], vec![int(3)], vec![int(4)], vec![int(5)]],
    );
}

/// Safety: a leading `WITH` does NOT make a view updatable. A view with no matching
/// `INSTEAD OF` trigger still errors with SQLite's `cannot modify … because it is a view`,
/// exactly as it would without the `WITH`.
#[test]
fn with_does_not_make_a_nontriggered_view_updatable() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE base(id INTEGER)");
    exec(&mut db, "CREATE VIEW v AS SELECT id FROM base");
    let e = assert_exec_error(
        &mut db,
        "WITH c(x) AS (VALUES (1)) INSERT INTO v(id) SELECT x FROM c",
    );
    assert!(e.to_string().contains("cannot modify v because it is a view"), "got {e}");
}

/// The firing statement's CTE is INVISIBLE inside the fired `INSTEAD OF` body
/// (`lang_with.html` §1: a CTE exists only for the firing statement). Here a REAL base table
/// `c` and a firing CTE `c` share a name: the INSERT source `FROM c` resolves the CTE (one
/// NEW row), while the trigger body's `FROM c` must resolve the base TABLE `c`. So `base`
/// gets the table row (99, 'from_table'), never the CTE row (1, 'from_cte').
#[test]
fn firing_statement_cte_is_invisible_inside_the_instead_of_body() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE base(id INTEGER, val TEXT)");
    exec(&mut db, "CREATE TABLE c(id INTEGER, val TEXT)");
    exec(&mut db, "INSERT INTO c VALUES (99, 'from_table')");
    exec(&mut db, "CREATE VIEW v(id, val) AS SELECT id, val FROM base");
    // The trigger body reads `FROM c` — this must bind the base table `c`, not the firing
    // statement's CTE. It fires once (one NEW row) and copies table `c` into `base`.
    exec(
        &mut db,
        "CREATE TRIGGER tr_ins INSTEAD OF INSERT ON v \
         BEGIN INSERT INTO base(id, val) SELECT id, val FROM c; END",
    );
    exec(
        &mut db,
        "WITH c(id, val) AS (VALUES (1, 'from_cte')) \
         INSERT INTO v(id, val) SELECT id, val FROM c",
    );
    assert_rows_unordered(
        &mut db,
        "SELECT id, val FROM base",
        &[vec![int(99), text("from_table")]],
    );
}
