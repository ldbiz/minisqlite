//! Conformance battery: **cross-namespace (main / temp / attached) READ shapes** — a
//! DELETE whose WHERE subquery reads another namespace, a CTE whose body reads an
//! ATTACHed database, and a single-statement 3-way JOIN spanning main + temp + attached —
//! exercised through the pinned `minisqlite::Connection` facade.
//!
//! These three shapes all route their reads through the base-table namespace-stamping path
//! (the read view a statement uses is the whole connection slice, so a subquery / CTE body /
//! join arm may name any database the connection can reach). Sibling suites already pin the
//! adjacent shapes — INSERT..SELECT, a cross-db JOIN, and an IN-subquery in a SELECT
//! (`conformance_attach.rs`); an UPDATE-SET / RETURNING cross-db subquery
//! (`conformance_cross_namespace_dml_subquery.rs`) — but these three read shapes were not
//! pinned by any committed test, so a regression on them would go uncaught. This file pins
//! them. Each expectation is DERIVED FROM THE SPEC in `spec/sqlite-doc/`, never from what the
//! engine happens to return; an assertion is never weakened to pass.
//!
//! Spec sources (`spec/sqlite-doc/`):
//!   * `lang_delete.html`: "If a WHERE clause is supplied, then only those rows for which the
//!     WHERE clause boolean expression is true are deleted." The WHERE is a full boolean
//!     `expr`, so an `x IN (SELECT … FROM other.namespace)` subquery is legal there and reads
//!     that other namespace — the delete target and the subquery source are independent.
//!   * `lang_with.html`: an ordinary CTE "works as if it were a view that exists for the
//!     duration of a single statement" (§2); its body is an ordinary SELECT, so it may read an
//!     attached (or temp) database like any other query.
//!   * `lang_select.html`: a `join-clause` combines two or more tables; nothing restricts the
//!     arms to one database, so a single SELECT may join main + temp + attached tables.
//!   * `lang_attach.html`: `ATTACH ':memory:' AS aux` adds an addressable in-memory database
//!     whose tables are reached as `aux.tbl`.
//!   * `lang_naming.html`: an unqualified name resolves temp, then main, then attached
//!     databases in attach order; "If a schema name is specified, then only that one schema is
//!     searched for the named object" — so a `main.t` qualifier reads main even when a
//!     same-named `temp.t` shadow exists. Namespace indices: main = 0, temp = 1, first
//!     attached = 2.
//!
//! DISCRIMINATORS: every namespace holds DISTINCT values, so a read that lands in the wrong
//! namespace produces a wrong result set and fails the assertion loudly (rather than silently
//! agreeing by coincidence). Each behavior is its own small `#[test]`, so one discrepancy
//! fails exactly that case. `Value` has no `PartialEq`; all comparisons go through the shared
//! harness helpers.

mod conformance;
use conformance::*;

// ===========================================================================
// Shape 1 — cross-db DELETE whose WHERE subquery READS ANOTHER namespace.
// The delete target and the subquery source are independent stores; the read
// source must NOT be modified by the delete. (`lang_delete.html`: the WHERE is a
// boolean expr, so the subquery is legal; `lang_attach.html` / `lang_naming.html`
// for the `schema.object` resolution.)
// ===========================================================================

#[test]
fn delete_attached_where_in_subquery_reads_main() {
    // DELETE from the ATTACHed `aux.u` (db 2) driven by an IN-subquery
    // that reads `main.t` (db 0). The matching rows {1,2,3} are removed from aux.u, leaving
    // {4}; `main.t` (the READ source) is untouched — a delete does not write its read source.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1),(2),(3)");
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.u(x INTEGER)");
    exec(&mut db, "INSERT INTO aux.u VALUES (1),(2),(3),(4)");

    exec(&mut db, "DELETE FROM aux.u WHERE x IN (SELECT a FROM main.t)");

    // aux.u lost exactly the rows whose x matched a main.t value; 4 (no match) survives.
    assert_rows(&mut db, "SELECT x FROM aux.u", &[vec![int(4)]]);
    // main.t, the read source, is unchanged.
    assert_rows_unordered(
        &mut db,
        "SELECT a FROM main.t",
        &[vec![int(1)], vec![int(2)], vec![int(3)]],
    );
}

#[test]
fn delete_main_where_in_subquery_reads_attached() {
    // Reverse direction of the same shape: DELETE from `main.t` (db 0) driven by an
    // IN-subquery reading the ATTACHed `aux.u` (db 2). Distinct source values (only {2,3} in
    // aux.u) make the delete selective: main.t keeps {1}. The attached READ source is
    // unchanged. Pins that the cross-namespace DELETE subquery works in BOTH directions, not
    // only aux-target/main-source.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1),(2),(3)");
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.u(x INTEGER)");
    exec(&mut db, "INSERT INTO aux.u VALUES (2),(3)");

    exec(&mut db, "DELETE FROM main.t WHERE a IN (SELECT x FROM aux.u)");

    // Only 1 (no match in aux.u) remains in main.t.
    assert_rows(&mut db, "SELECT a FROM main.t", &[vec![int(1)]]);
    // aux.u, the read source, is unchanged.
    assert_rows_unordered(&mut db, "SELECT x FROM aux.u", &[vec![int(2)], vec![int(3)]]);
}

#[test]
fn delete_qualified_main_source_not_shadowed_by_temp() {
    // Discriminator for `lang_naming.html`'s qualifier rule: a same-named `temp.t` shadow
    // exists with DISTINCT values, but the subquery names `main.t` explicitly, so it must read
    // main ({1,2,3}), not the temp shadow ({4}). The DELETE therefore removes {1,2,3} from
    // aux.u → {4}. If the qualifier were wrongly resolved to the temp shadow, only {4} would be
    // deleted and aux.u would read {1,2,3}, failing this assertion loudly. The temp shadow is
    // itself untouched.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1),(2),(3)");
    exec(&mut db, "CREATE TEMP TABLE t(a INTEGER)"); // temp.t shadows main.t for unqualified names
    exec(&mut db, "INSERT INTO temp.t VALUES (4)");
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.u(x INTEGER)");
    exec(&mut db, "INSERT INTO aux.u VALUES (1),(2),(3),(4)");

    exec(&mut db, "DELETE FROM aux.u WHERE x IN (SELECT a FROM main.t)");

    // main.t ({1,2,3}) drove the delete, not the temp shadow ({4}); only 4 survives in aux.u.
    assert_rows(&mut db, "SELECT x FROM aux.u", &[vec![int(4)]]);
    // Both sources are unchanged.
    assert_rows_unordered(
        &mut db,
        "SELECT a FROM main.t",
        &[vec![int(1)], vec![int(2)], vec![int(3)]],
    );
    assert_rows(&mut db, "SELECT a FROM temp.t", &[vec![int(4)]]);
}

// ===========================================================================
// Shape 2 — a CTE whose body READS an ATTACHed (or temp) database. The ordinary
// CTE is "a view that exists for the duration of a single statement"
// (`lang_with.html` §2), so its body can read any namespace the connection names.
// ===========================================================================

#[test]
fn cte_over_attached_db_count() {
    // A CTE body filters the ATTACHed `aux.u` (db 2); the outer query counts the
    // CTE rows. `x > 5` keeps {6,7} out of {5,6,7}, so count(*) = 2.
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.u(x INTEGER)");
    exec(&mut db, "INSERT INTO aux.u VALUES (5),(6),(7)");

    assert_scalar(
        &mut db,
        "WITH c AS (SELECT x FROM aux.u WHERE x > 5) SELECT count(*) FROM c",
        int(2),
    );
}

#[test]
fn cte_over_attached_db_row_values() {
    // Same CTE, but assert the actual VALUES flow through (not just a count that a broken read
    // could also produce by coincidence): the CTE over `aux.u WHERE x > 5` yields exactly
    // {6,7}, projected in order.
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.u(x INTEGER)");
    exec(&mut db, "INSERT INTO aux.u VALUES (5),(6),(7)");

    assert_rows(
        &mut db,
        "WITH c AS (SELECT x FROM aux.u WHERE x > 5) SELECT x FROM c ORDER BY x",
        &[vec![int(6)], vec![int(7)]],
    );
}

#[test]
fn cte_over_temp_db_row_values() {
    // Broaden the CTE shape to the `temp` namespace (db 1): the CTE body reads a
    // `CREATE TEMP TABLE`. Distinct values ({15,16,17}) pin that the CTE reaches temp, not
    // some other store; `x > 15` yields {16,17}. Pins the CTE-over-temp read path alongside
    // CTE-over-attached.
    let mut db = mem();
    exec(&mut db, "CREATE TEMP TABLE u(x INTEGER)");
    exec(&mut db, "INSERT INTO temp.u VALUES (15),(16),(17)");

    assert_rows(
        &mut db,
        "WITH c AS (SELECT x FROM temp.u WHERE x > 15) SELECT x FROM c ORDER BY x",
        &[vec![int(16)], vec![int(17)]],
    );
}

#[test]
fn cte_qualified_attached_source_not_shadowed_by_main() {
    // Discriminator for `lang_naming.html`'s qualifier rule INSIDE a CTE body — the CTE analog
    // of `delete_qualified_main_source_not_shadowed_by_temp`. An unqualified `u` resolves to
    // `main` (search order temp → main → attached), so a `main.u` decoy with DISTINCT values
    // ({90,91}) sits AHEAD of `aux.u` ({5,6,7}) in the search order. The CTE body names `aux.u`
    // explicitly, so it must read the ATTACHed table → {6,7} after `x > 5`, NOT the main decoy.
    // If the qualifier were dropped or mis-resolved to the main decoy, the CTE would yield
    // {90,91} and fail loudly. This pins that the CTE resolves the RIGHT namespace under
    // shadowing, not merely that it can reach an attached store.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE u(x INTEGER)"); // main.u decoy: ahead of aux in the search order
    exec(&mut db, "INSERT INTO u VALUES (90),(91)");
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.u(x INTEGER)");
    exec(&mut db, "INSERT INTO aux.u VALUES (5),(6),(7)");

    assert_rows(
        &mut db,
        "WITH c AS (SELECT x FROM aux.u WHERE x > 5) SELECT x FROM c ORDER BY x",
        &[vec![int(6)], vec![int(7)]],
    );
    // The main decoy is distinct and untouched — confirms it really sits in main (so reading it
    // instead of aux.u would have produced a different, failing result set).
    assert_rows(&mut db, "SELECT x FROM main.u ORDER BY x", &[vec![int(90)], vec![int(91)]]);
}

// ===========================================================================
// Shape 3 — a single-statement 3-way JOIN spanning main + temp + attached. Each
// namespace holds a DISTINCT text value ('m1' in main, 't1' in temp, 'a1' in the
// attached db), so the joined row proves all three stores were read and joined in
// one statement. (`lang_select.html` join-clause; `lang_attach.html` /
// `lang_naming.html` for the cross-db table references.)
// ===========================================================================

#[test]
fn three_way_join_main_temp_attached_single_row() {
    // Join `main.m` ⋈ `temp.tp` ⋈ `aux.c` on a shared id, projecting one column
    // from each namespace. The single matching id (1) yields exactly one row carrying the
    // distinct per-namespace discriminators ('m1','t1','a1').
    let mut db = mem();
    exec(&mut db, "CREATE TABLE m(id INTEGER, v TEXT)");
    exec(&mut db, "INSERT INTO m VALUES (1,'m1')");
    exec(&mut db, "CREATE TEMP TABLE tp(id INTEGER, w TEXT)");
    exec(&mut db, "INSERT INTO tp VALUES (1,'t1')");
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.c(id INTEGER, z TEXT)");
    exec(&mut db, "INSERT INTO aux.c VALUES (1,'a1')");

    assert_rows(
        &mut db,
        "SELECT m.v, tp.w, aux.c.z FROM main.m \
         JOIN temp.tp ON m.id = tp.id \
         JOIN aux.c ON m.id = aux.c.id",
        &[vec![text("m1"), text("t1"), text("a1")]],
    );
}

#[test]
fn three_way_join_main_temp_attached_multiple_rows() {
    // The same 3-way cross-namespace join returning MORE than one row: two matching ids each
    // pick their own per-namespace value, so the join must pair them correctly across all
    // three stores (id 1 → m1/t1/a1, id 2 → m2/t2/a2). Distinct values per (id, namespace)
    // make a mispaired join fail loudly.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE m(id INTEGER, v TEXT)");
    exec(&mut db, "INSERT INTO m VALUES (1,'m1'),(2,'m2')");
    exec(&mut db, "CREATE TEMP TABLE tp(id INTEGER, w TEXT)");
    exec(&mut db, "INSERT INTO tp VALUES (1,'t1'),(2,'t2')");
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.c(id INTEGER, z TEXT)");
    exec(&mut db, "INSERT INTO aux.c VALUES (1,'a1'),(2,'a2')");

    assert_rows_unordered(
        &mut db,
        "SELECT m.v, tp.w, aux.c.z FROM main.m \
         JOIN temp.tp ON m.id = tp.id \
         JOIN aux.c ON m.id = aux.c.id",
        &[
            vec![text("m1"), text("t1"), text("a1")],
            vec![text("m2"), text("t2"), text("a2")],
        ],
    );
}

#[test]
fn three_way_join_main_temp_attached_no_match_empty() {
    // The join must actually FILTER across all three stores, not merely reach them: here the
    // attached arm's id (2) does not match the main/temp id (1), so the inner join yields NO
    // rows. An empty result proves the `aux.c` predicate is enforced — a join that ignored the
    // attached arm's key would wrongly emit a row.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE m(id INTEGER, v TEXT)");
    exec(&mut db, "INSERT INTO m VALUES (1,'m1')");
    exec(&mut db, "CREATE TEMP TABLE tp(id INTEGER, w TEXT)");
    exec(&mut db, "INSERT INTO tp VALUES (1,'t1')");
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.c(id INTEGER, z TEXT)");
    exec(&mut db, "INSERT INTO aux.c VALUES (2,'a1')"); // id 2 does not match main/temp id 1

    assert_rows(
        &mut db,
        "SELECT m.v, tp.w, aux.c.z FROM main.m \
         JOIN temp.tp ON m.id = tp.id \
         JOIN aux.c ON m.id = aux.c.id",
        &[],
    );
}
