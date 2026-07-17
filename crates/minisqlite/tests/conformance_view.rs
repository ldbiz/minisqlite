//! Conformance battery: `CREATE VIEW` expansion (spec
//! `spec/sqlite-doc/lang_createview.html`). A view is a "pre-packaged SELECT
//! statement" — referencing it in a query is exactly equivalent to inlining that
//! SELECT as a derived table, so every expected value here is what the underlying
//! SELECT would return, TRANSCRIBED FROM THE SPEC and SQL semantics, never from what
//! the engine happens to emit:
//!
//! - "A view is … a pre-packaged SELECT statement." Selecting from a view returns the
//!   rows of its stored SELECT.
//! - The optional `CREATE VIEW name(col-list) AS …` renames the view's output columns;
//!   the arity of that list must equal the SELECT's output width.
//! - "You cannot DELETE, INSERT, or UPDATE a view" (without an INSTEAD OF trigger,
//!   which is a separate feature): the spec says a view is read-only, and real SQLite
//!   rejects DML on one with `cannot modify <view> because it is a view`.
//! - A view whose definition refers to itself (directly or transitively) is
//!   "circularly defined" and real SQLite errors `view <name> is circularly defined`
//!   rather than looping.
//! - A view is a self-contained schema object: its body resolves against the schema
//!   (base tables and other views) and its OWN `WITH`, never the CTEs of the query that
//!   references it.
//!
//! Expected values are transcribed from the SQLite documentation, not from what this
//! engine returns; a case that reveals an engine bug is left as a genuine failing
//! assertion rather than weakened to pass. Cases are split into many small `#[test]` fns
//! so one failure never masks the rest.
//!
//! TERMINATION: the circular-view cases rely on the planner's recursion guard to reject
//! (not loop). Each is isolated in its own `#[test]`, so if a future engine regresses
//! and loops, only that case is affected. `Value` has no `PartialEq`; every comparison
//! goes through the shared harness.

mod conformance;

use conformance::*;
use minisqlite::Connection;

// =============================================================================
// Section A — a view expands to the rows of its underlying SELECT.
// =============================================================================

/// Seed a two-column base table `t(a INTEGER, b TEXT)` with three rows.
fn seed_t(db: &mut Connection) {
    exec(db, "CREATE TABLE t(a INTEGER, b TEXT)");
    exec(db, "INSERT INTO t VALUES (1, 'x'), (2, 'y'), (3, 'z')");
}

#[test]
fn select_star_from_view_returns_underlying_rows() {
    // The spec's core promise: `SELECT * FROM v` is the view's stored SELECT.
    let mut db = mem();
    seed_t(&mut db);
    exec(&mut db, "CREATE VIEW v AS SELECT a, b FROM t");
    assert_rows(
        &mut db,
        "SELECT * FROM v ORDER BY a",
        &[vec![int(1), text("x")], vec![int(2), text("y")], vec![int(3), text("z")]],
    );
}

#[test]
fn view_output_column_names_come_from_the_select() {
    // With no explicit column list, the view's columns are the SELECT's names.
    let mut db = mem();
    seed_t(&mut db);
    exec(&mut db, "CREATE VIEW v AS SELECT a, b FROM t");
    assert_columns(&mut db, "SELECT * FROM v", &["a", "b"]);
}

#[test]
fn view_select_expression_alias_names_the_column() {
    // A view over an expression takes the expression's alias as its column name.
    let mut db = mem();
    seed_t(&mut db);
    exec(&mut db, "CREATE VIEW v AS SELECT a + 10 AS a10 FROM t");
    assert_columns(&mut db, "SELECT * FROM v", &["a10"]);
    assert_rows(
        &mut db,
        "SELECT a10 FROM v ORDER BY a10",
        &[vec![int(11)], vec![int(12)], vec![int(13)]],
    );
}

#[test]
fn explicit_column_list_renames_view_columns() {
    // `CREATE VIEW v(x, y) AS …` renames the outputs; the rows are unchanged.
    let mut db = mem();
    seed_t(&mut db);
    exec(&mut db, "CREATE VIEW v(x, y) AS SELECT a, b FROM t");
    assert_columns(&mut db, "SELECT * FROM v", &["x", "y"]);
    assert_rows(
        &mut db,
        "SELECT x, y FROM v ORDER BY x",
        &[vec![int(1), text("x")], vec![int(2), text("y")], vec![int(3), text("z")]],
    );
}

#[test]
fn view_can_be_filtered_and_projected_by_the_outer_query() {
    // A view is a source like any other: the outer query may add WHERE and pick columns.
    let mut db = mem();
    seed_t(&mut db);
    exec(&mut db, "CREATE VIEW v AS SELECT a, b FROM t");
    assert_rows(&mut db, "SELECT b FROM v WHERE a >= 2 ORDER BY a", &[vec![text("y")], vec![text("z")]]);
}

#[test]
fn view_body_may_carry_its_own_where() {
    // The view's own WHERE filters before the outer query ever sees the rows.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE people(name TEXT, age INTEGER)");
    exec(&mut db, "INSERT INTO people VALUES ('ann', 30), ('bo', 12), ('cy', 18)");
    exec(&mut db, "CREATE VIEW adults AS SELECT name FROM people WHERE age >= 18");
    assert_rows_unordered(&mut db, "SELECT name FROM adults", &[vec![text("ann")], vec![text("cy")]]);
}

#[test]
fn view_over_a_join_expands() {
    // A view whose body is a join materializes and scans correctly.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE emp(id INTEGER, dept INTEGER)");
    exec(&mut db, "CREATE TABLE dept(id INTEGER, name TEXT)");
    exec(&mut db, "INSERT INTO emp VALUES (1, 10), (2, 20)");
    exec(&mut db, "INSERT INTO dept VALUES (10, 'sales'), (20, 'eng')");
    exec(
        &mut db,
        "CREATE VIEW ed AS SELECT emp.id AS eid, dept.name AS dname FROM emp JOIN dept ON emp.dept = dept.id",
    );
    assert_rows(
        &mut db,
        "SELECT eid, dname FROM ed ORDER BY eid",
        &[vec![int(1), text("sales")], vec![int(2), text("eng")]],
    );
}

#[test]
fn view_over_an_aggregate_expands() {
    // An aggregate view yields its single computed row.
    let mut db = mem();
    seed_t(&mut db);
    exec(&mut db, "CREATE VIEW stats AS SELECT count(*) AS n, sum(a) AS s FROM t");
    assert_rows(&mut db, "SELECT n, s FROM stats", &[vec![int(3), int(6)]]);
}

#[test]
fn outer_query_may_aggregate_over_a_view() {
    // The view is a plain row source, so the outer query can aggregate it.
    let mut db = mem();
    seed_t(&mut db);
    exec(&mut db, "CREATE VIEW v AS SELECT a, b FROM t");
    assert_scalar(&mut db, "SELECT count(*) FROM v", int(3));
    assert_scalar(&mut db, "SELECT sum(a) FROM v", int(6));
}

#[test]
fn view_may_be_aliased_in_from() {
    // A view reference accepts an alias, and the alias qualifies its columns.
    let mut db = mem();
    seed_t(&mut db);
    exec(&mut db, "CREATE VIEW v AS SELECT a, b FROM t");
    assert_rows(
        &mut db,
        "SELECT x.a, x.b FROM v AS x WHERE x.a = 2",
        &[vec![int(2), text("y")]],
    );
}

#[test]
fn view_joined_with_a_base_table() {
    // A view participates in a join with a base table like any derived table.
    let mut db = mem();
    seed_t(&mut db);
    exec(&mut db, "CREATE TABLE u(a INTEGER, label TEXT)");
    exec(&mut db, "INSERT INTO u VALUES (1, 'one'), (3, 'three')");
    exec(&mut db, "CREATE VIEW v AS SELECT a, b FROM t");
    assert_rows(
        &mut db,
        "SELECT v.a, v.b, u.label FROM v JOIN u ON v.a = u.a ORDER BY v.a",
        &[vec![int(1), text("x"), text("one")], vec![int(3), text("z"), text("three")]],
    );
}

// =============================================================================
// Section B — a view referencing another view expands transitively.
// =============================================================================

#[test]
fn view_referencing_a_view_expands_transitively() {
    // v2 is defined over v1; selecting v2 pulls through both.
    let mut db = mem();
    seed_t(&mut db);
    exec(&mut db, "CREATE VIEW v1 AS SELECT a, b FROM t WHERE a >= 2");
    exec(&mut db, "CREATE VIEW v2 AS SELECT a FROM v1 WHERE a <= 2");
    assert_rows(&mut db, "SELECT a FROM v2", &[vec![int(2)]]);
}

#[test]
fn three_level_view_chain_expands() {
    // A longer chain (v3 → v2 → v1 → t) still resolves to the base rows.
    let mut db = mem();
    seed_t(&mut db);
    exec(&mut db, "CREATE VIEW v1 AS SELECT a FROM t");
    exec(&mut db, "CREATE VIEW v2 AS SELECT a FROM v1");
    exec(&mut db, "CREATE VIEW v3 AS SELECT a FROM v2");
    assert_rows(&mut db, "SELECT a FROM v3 ORDER BY a", &[vec![int(1)], vec![int(2)], vec![int(3)]]);
}

// =============================================================================
// Section C — a view is self-contained: its body does not see the referencing
// query's CTEs, but its OWN `WITH` works.
// =============================================================================

#[test]
fn view_body_resolves_base_table_not_an_outer_cte_of_the_same_name() {
    // The view body's `FROM src` must bind to the BASE table `src`, never to an outer
    // CTE that happens to share the name — a view is compiled against the schema, not
    // against the referencing statement's WITH. Distinct values make the binding
    // observable: the base table holds 99, the outer CTE holds 1.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE src(a INTEGER)");
    exec(&mut db, "INSERT INTO src VALUES (99)");
    exec(&mut db, "CREATE VIEW v AS SELECT a FROM src");
    assert_rows(
        &mut db,
        "WITH src AS (SELECT 1 AS a) SELECT a FROM v",
        &[vec![int(99)]],
    );
}

#[test]
fn view_may_define_its_own_cte() {
    // Isolation of the outer CTE scope must not break the view's OWN `WITH`.
    let mut db = mem();
    exec(&mut db, "CREATE VIEW v AS WITH c AS (SELECT 7 AS n) SELECT n FROM c");
    assert_rows(&mut db, "SELECT n FROM v", &[vec![int(7)]]);
}

// =============================================================================
// Section D — errors: circular definitions and DML on a view.
// =============================================================================

#[test]
fn directly_self_referential_view_errors_circular() {
    // `CREATE VIEW v AS SELECT * FROM v` creates fine (the body is not resolved at
    // create time), but expanding it must error — and NOT loop.
    let mut db = mem();
    exec(&mut db, "CREATE VIEW v AS SELECT * FROM v");
    let e = assert_query_error(&mut db, "SELECT * FROM v");
    assert!(
        e.to_string().contains("circularly defined"),
        "expected a circular-view error, got: {e}"
    );
}

#[test]
fn mutually_recursive_views_error_circular() {
    // a → b → a is a transitive cycle; the guard must catch it across both views.
    let mut db = mem();
    exec(&mut db, "CREATE VIEW a AS SELECT * FROM b");
    exec(&mut db, "CREATE VIEW b AS SELECT * FROM a");
    let e = assert_query_error(&mut db, "SELECT * FROM a");
    assert!(
        e.to_string().contains("circularly defined"),
        "expected a circular-view error, got: {e}"
    );
}

#[test]
fn self_referential_view_nested_in_a_larger_query_still_errors() {
    // The cycle must be caught even when the view is not the sole FROM source.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    exec(&mut db, "CREATE VIEW v AS SELECT a FROM v");
    let e = assert_query_error(&mut db, "SELECT t.a FROM t JOIN v ON t.a = v.a");
    assert!(
        e.to_string().contains("circularly defined"),
        "expected a circular-view error, got: {e}"
    );
}

#[test]
fn insert_into_a_view_is_rejected() {
    // lang_createview.html: a view is read-only. Real SQLite: "cannot modify v because
    // it is a view".
    let mut db = mem();
    seed_t(&mut db);
    exec(&mut db, "CREATE VIEW v AS SELECT a, b FROM t");
    let e = assert_exec_error(&mut db, "INSERT INTO v VALUES (4, 'w')");
    assert!(
        e.to_string().contains("cannot modify v because it is a view"),
        "expected the read-only-view message, got: {e}"
    );
}

#[test]
fn update_of_a_view_is_rejected() {
    let mut db = mem();
    seed_t(&mut db);
    exec(&mut db, "CREATE VIEW v AS SELECT a, b FROM t");
    let e = assert_exec_error(&mut db, "UPDATE v SET b = 'q' WHERE a = 1");
    assert!(
        e.to_string().contains("cannot modify v because it is a view"),
        "expected the read-only-view message, got: {e}"
    );
}

#[test]
fn delete_from_a_view_is_rejected() {
    let mut db = mem();
    seed_t(&mut db);
    exec(&mut db, "CREATE VIEW v AS SELECT a, b FROM t");
    let e = assert_exec_error(&mut db, "DELETE FROM v WHERE a = 1");
    assert!(
        e.to_string().contains("cannot modify v because it is a view"),
        "expected the read-only-view message, got: {e}"
    );
}

#[test]
fn read_only_rejection_uses_the_views_stored_case() {
    // The message carries the view's stored (creation-case) name, like SQLite's
    // `pTab->zName`, regardless of how the DML spelled it.
    let mut db = mem();
    seed_t(&mut db);
    exec(&mut db, "CREATE VIEW MyView AS SELECT a FROM t");
    let e = assert_exec_error(&mut db, "DELETE FROM myview");
    assert!(
        e.to_string().contains("cannot modify MyView because it is a view"),
        "expected the stored-case view name in the message, got: {e}"
    );
}

#[test]
fn dml_on_a_missing_name_is_still_no_such_table() {
    // The view branch must not swallow the ordinary missing-table error.
    let mut db = mem();
    let e = assert_exec_error(&mut db, "INSERT INTO nope VALUES (1)");
    assert!(e.to_string().contains("no such table"), "expected no-such-table, got: {e}");
}

#[test]
fn explicit_column_list_arity_mismatch_is_an_error() {
    // The declared column list must match the SELECT's output width; a mismatch is a
    // loud error at expansion, never a silently mis-shaped view.
    let mut db = mem();
    seed_t(&mut db);
    exec(&mut db, "CREATE VIEW v(x, y, z) AS SELECT a, b FROM t");
    let e = assert_query_error(&mut db, "SELECT * FROM v");
    assert!(
        e.to_string().contains("expected 3 columns")
            && e.to_string().contains("but got 2"),
        "expected an arity-mismatch error, got: {e}"
    );
}

// =============================================================================
// Section E — a view survives the schema being reloaded from disk (the stored
// SQL is the source of truth). This closes the loop that a view is not a plan
// baked at create time but re-parsed on every reference.
// =============================================================================

#[test]
fn view_reference_reparses_stored_sql_on_each_use() {
    // Referencing the same view twice in one statement expands it twice from its stored
    // text (once per reference), and both agree.
    let mut db = mem();
    seed_t(&mut db);
    exec(&mut db, "CREATE VIEW v AS SELECT a FROM t");
    assert_rows(
        &mut db,
        "SELECT l.a, r.a FROM v AS l JOIN v AS r ON l.a = r.a ORDER BY l.a",
        &[vec![int(1), int(1)], vec![int(2), int(2)], vec![int(3), int(3)]],
    );
}

// =============================================================================
// Section F — regression guards: view-body shapes on the
// SUCCESS path, phase-agreement for schema-qualified names, outer-CTE scope
// restore after a view, and DML error precedence over unsupported-clause gaps.
// =============================================================================

#[test]
fn view_body_select_star_expands_the_base_columns() {
    // A `SELECT *` in the view BODY (not the referencing query) expands to the base
    // table's columns on a NON-error path — the common star-in-body shape that the
    // other tests only hit via the self-referential error case.
    let mut db = mem();
    seed_t(&mut db);
    exec(&mut db, "CREATE VIEW v AS SELECT * FROM t");
    assert_columns(&mut db, "SELECT * FROM v", &["a", "b"]);
    assert_rows(
        &mut db,
        "SELECT * FROM v ORDER BY a",
        &[vec![int(1), text("x")], vec![int(2), text("y")], vec![int(3), text("z")]],
    );
}

#[test]
fn view_body_order_by_and_limit_are_applied() {
    // The view body may carry its own ORDER BY / LIMIT; expansion preserves them. The
    // body keeps the two LARGEST `a` (3, 2); the outer ORDER BY then re-sorts ascending,
    // so a dropped LIMIT (row a=1 appearing) or ignored body ORDER BY would show.
    let mut db = mem();
    seed_t(&mut db);
    exec(&mut db, "CREATE VIEW top2 AS SELECT a FROM t ORDER BY a DESC LIMIT 2");
    assert_rows(&mut db, "SELECT a FROM top2 ORDER BY a", &[vec![int(2)], vec![int(3)]]);
}

#[test]
fn schema_qualified_view_name_bypasses_a_same_named_cte() {
    // A schema-qualified reference (`main.v`) is ALWAYS a schema object, never a CTE — in
    // BOTH planning phases. So an outer `WITH v AS (…)` of the same bare name must not
    // shadow `main.v`; real SQLite resolves the VIEW. Regression guard for the Phase-1 /
    // Phase-2 precedence split: Phase 2 must gate its CTE lookup on `schema.is_none()`
    // exactly as Phase 1 does, else it silently scans the 1-column CTE body while Phase 1
    // sized the source at the view's 2 columns.
    let mut db = mem();
    seed_t(&mut db);
    exec(&mut db, "CREATE VIEW v AS SELECT a, b FROM t");
    assert_columns(&mut db, "WITH v AS (SELECT 1 AS a) SELECT * FROM main.v", &["a", "b"]);
    assert_rows(
        &mut db,
        "WITH v AS (SELECT 1 AS a) SELECT * FROM main.v ORDER BY a",
        &[vec![int(1), text("x")], vec![int(2), text("y")], vec![int(3), text("z")]],
    );
}

#[test]
fn an_outer_cte_referenced_alongside_a_view_still_resolves() {
    // The expander sets the outer CTE scope ASIDE while compiling the view body and
    // RESTORES it on drop. So an outer CTE joined to the view must still resolve after the
    // view's isolation ends — a failure to restore would make `c` an unknown table.
    // Distinct values (view yields 1/2/3, CTE yields 2) prove each side binds correctly.
    let mut db = mem();
    seed_t(&mut db);
    exec(&mut db, "CREATE VIEW v AS SELECT a FROM t");
    assert_rows(
        &mut db,
        "WITH c AS (SELECT 2 AS a) SELECT v.a FROM v JOIN c ON v.a = c.a",
        &[vec![int(2)]],
    );
}

#[test]
fn update_of_a_view_reports_view_error_before_the_unsupported_from_gap() {
    // `UPDATE <view> … FROM t` is BOTH a view target and an (unsupported) UPDATE-FROM.
    // SQLite resolves the target first, so the read-only-view error wins over the FROM
    // gap. (Ordering guard: target resolution must precede the clause-gap checks.)
    let mut db = mem();
    seed_t(&mut db);
    exec(&mut db, "CREATE TABLE u(a INTEGER)");
    exec(&mut db, "CREATE VIEW v AS SELECT a, b FROM t");
    let e = assert_exec_error(&mut db, "UPDATE v SET b = 'q' FROM u");
    assert!(
        e.to_string().contains("cannot modify v because it is a view"),
        "expected the read-only-view message to win over the FROM gap, got: {e}"
    );
}

#[test]
fn delete_from_a_view_reports_view_error_before_the_unsupported_with_gap() {
    // `WITH c AS (…) DELETE FROM <view>` is BOTH a view target and an (unsupported)
    // leading WITH. SQLite resolves the target first, so the read-only-view error wins.
    let mut db = mem();
    seed_t(&mut db);
    exec(&mut db, "CREATE VIEW v AS SELECT a, b FROM t");
    let e = assert_exec_error(&mut db, "WITH c AS (SELECT 1) DELETE FROM v");
    assert!(
        e.to_string().contains("cannot modify v because it is a view"),
        "expected the read-only-view message to win over the WITH gap, got: {e}"
    );
}
