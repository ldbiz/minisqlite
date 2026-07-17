//! Real-catalog fidelity for the `ORDER BY` -> scan-order skip (`compile::order_scan`).
//!
//! The plan-shape unit tests in `src/compile/order_scan.rs` prove the skip against a
//! HAND-BUILT static `IdxCatalog`. This integration test proves the SAME skips fire (and
//! the same must-not-skip cases keep the Sort) against the REAL [`SchemaCatalog`] — the
//! store the engine actually uses — populated by running real `CREATE TABLE` / `CREATE
//! INDEX` DDL through a real pager, exactly as the engine does.
//!
//! Why this test exists (the gap it closes): the facade behavioral tests
//! (`minisqlite/tests/conformance_order_by_index.rs`) can only observe RESULTS, which stay
//! CORRECT whether the `Sort` is skipped OR kept. So a silent FAILED skip — the real
//! catalog populating an `IndexDef` (`key_exprs` / `key_columns` / collation / `partial`)
//! differently from the static stub, so the optimization never fires in production — would
//! leave every facade test passing while the whole point of the optimization (the perf win)
//! was lost. `EXPLAIN QUERY PLAN` is not yet supported, so the plan can't be inspected
//! through the facade; here we compile against the real catalog and assert on the `Plan`
//! tree directly.

use minisqlite_btree::init_database;
use minisqlite_catalog::{Catalog, SchemaCatalog};
use minisqlite_pager::MemPager;
use minisqlite_plan::{Plan, PlanNode, Planner, QueryPlanner};
use minisqlite_sql::{parse, Statement};

/// A fresh in-memory database image (page 1 formatted) plus an empty catalog cache — the
/// same setup the catalog crate's own write-path tests use.
fn fresh() -> (MemPager, SchemaCatalog) {
    let mut pager = MemPager::new(4096);
    init_database(&mut pager).expect("format page 1");
    (pager, SchemaCatalog::new())
}

/// Run one DDL statement (`CREATE TABLE` / `CREATE INDEX`) through the real catalog write
/// path, so the resulting `TableDef` / `IndexDef` are exactly what the engine builds.
fn ddl(cat: &mut SchemaCatalog, pager: &mut MemPager, sql: &str) {
    let ast = parse(sql).expect("parse ddl");
    match &ast.statements[0] {
        Statement::CreateTable(stmt) => cat.create_table(pager, stmt, sql).expect("create table"),
        Statement::CreateIndex(stmt) => cat.create_index(pager, stmt, sql).expect("create index"),
        other => panic!("expected CREATE TABLE/INDEX, got {other:?}"),
    }
}

/// Compile a SELECT against the real catalog through the real `QueryPlanner`.
fn plan(cat: &SchemaCatalog, sql: &str) -> Plan {
    let ast = parse(sql).expect("parse select");
    let stmt = ast.statements.first().expect("one statement");
    QueryPlanner::new().plan(stmt, cat).expect("plan")
}

/// Whether a `Sort` appears anywhere on the (linear, single-table) operator chain from the
/// root — the same descent the `order_scan` unit tests use.
fn has_sort(node: &PlanNode) -> bool {
    match node {
        PlanNode::Sort { .. } => true,
        PlanNode::Project { input, .. }
        | PlanNode::Filter { input, .. }
        | PlanNode::Distinct { input, .. }
        | PlanNode::Limit { input, .. } => has_sort(input),
        _ => false,
    }
}

#[track_caller]
fn assert_skipped(cat: &SchemaCatalog, sql: &str) {
    assert!(!has_sort(&plan(cat, sql).root), "expected the Sort to be SKIPPED: {sql}");
}

#[track_caller]
fn assert_kept(cat: &SchemaCatalog, sql: &str) {
    assert!(has_sort(&plan(cat, sql).root), "expected the Sort to be KEPT: {sql}");
}

// ===================== MUST-SKIP against the real catalog =======================

#[test]
fn rowid_and_intpk_order_skip_sort_against_real_catalog() {
    let (mut p, mut cat) = fresh();
    ddl(&mut cat, &mut p, "CREATE TABLE t(a INTEGER, b INTEGER)");
    ddl(&mut cat, &mut p, "CREATE TABLE u(id INTEGER PRIMARY KEY, v TEXT)");
    // rowid needs no index; ASC = forward table walk, DESC = reverse rowid scan.
    assert_skipped(&cat, "SELECT a, b FROM t ORDER BY rowid");
    assert_skipped(&cat, "SELECT a, b FROM t ORDER BY rowid DESC");
    // An INTEGER PRIMARY KEY aliases the rowid, so ORDER BY it is the rowid order.
    assert_skipped(&cat, "SELECT id, v FROM u ORDER BY id");
    assert_skipped(&cat, "SELECT id, v FROM u ORDER BY id DESC");
}

#[test]
fn indexed_column_asc_skips_sort_against_real_catalog() {
    let (mut p, mut cat) = fresh();
    ddl(&mut cat, &mut p, "CREATE TABLE t(seq INTEGER, a INTEGER, b INTEGER)");
    ddl(&mut cat, &mut p, "CREATE INDEX ia ON t(a)");
    // The headline case: a forward full index scan already yields `a` order.
    assert_skipped(&cat, "SELECT a, seq FROM t ORDER BY a");
    // Hidden ORDER BY column (`a` is not in the SELECT list) still traces to the index.
    assert_skipped(&cat, "SELECT b FROM t ORDER BY a");
    // With a LIMIT (the streaming / early-stop shape) the Limit is kept, the Sort dropped.
    assert_skipped(&cat, "SELECT a, seq FROM t ORDER BY a LIMIT 5");
    // A WHERE range on the same column continues the index in order.
    assert_skipped(&cat, "SELECT a FROM t WHERE a >= 3 ORDER BY a");
}

#[test]
fn indexed_text_column_skips_sort_against_real_catalog() {
    let (mut p, mut cat) = fresh();
    ddl(&mut cat, &mut p, "CREATE TABLE ct(seq INTEGER, s TEXT)");
    ddl(&mut cat, &mut p, "CREATE INDEX ics ON ct(s)");
    // A default-BINARY TEXT column served by its index (the index's byte order equals the
    // Sort's Binary text order), both when `s` is projected and when it is hidden.
    assert_skipped(&cat, "SELECT s FROM ct ORDER BY s");
    assert_skipped(&cat, "SELECT seq FROM ct ORDER BY s");
}

#[test]
fn composite_index_serves_order_against_real_catalog() {
    let (mut p, mut cat) = fresh();
    ddl(&mut cat, &mut p, "CREATE TABLE t(a INTEGER, b INTEGER)");
    ddl(&mut cat, &mut p, "CREATE INDEX iab ON t(a, b)");
    // WHERE a = 1 seeks the composite index; ORDER BY b continues it after the eq-prefix.
    assert_skipped(&cat, "SELECT b FROM t WHERE a = 1 ORDER BY b");
    // ORDER BY a, b matches the WHOLE (a, b) index (== key count) -> a full index scan.
    assert_skipped(&cat, "SELECT a, b FROM t ORDER BY a, b");
}

// ============= MUST-NOT-SKIP: the Sort is correctly KEPT (real catalog) ==========

#[test]
fn desc_secondary_index_keeps_sort_against_real_catalog() {
    let (mut p, mut cat) = fresh();
    ddl(&mut cat, &mut p, "CREATE TABLE t(a INTEGER, b INTEGER)");
    ddl(&mut cat, &mut p, "CREATE INDEX ia ON t(a)");
    // DESC over a (dup-valued) secondary index: a reverse index walk reverses ties, so the
    // Sort MUST stay (byte-identity). This is the deviation from the literal
    // must-skip example — pinned here against the real catalog too.
    assert_kept(&cat, "SELECT a FROM t ORDER BY a DESC");
}

#[test]
fn non_servable_orders_keep_sort_against_real_catalog() {
    let (mut p, mut cat) = fresh();
    ddl(&mut cat, &mut p, "CREATE TABLE t(a INTEGER, b INTEGER)");
    ddl(&mut cat, &mut p, "CREATE INDEX iab ON t(a, b)");
    // Non-leading index column (no a= equality) can't use the (a, b) index.
    assert_kept(&cat, "SELECT b FROM t ORDER BY b");
    // ORDER BY a over ONLY a wider (a, b) index: a full scan tie-breaks by b before the
    // rowid, so it differs from SeqScan + stable sort — keep the Sort.
    assert_kept(&cat, "SELECT a FROM t ORDER BY a");
    // Mixed direction: one b-tree walk cannot serve ASC then DESC.
    assert_kept(&cat, "SELECT a, b FROM t ORDER BY a ASC, b DESC");
    // A computed key is not a base-table column reference.
    assert_kept(&cat, "SELECT a FROM t ORDER BY a + 0");
}

#[test]
fn nocase_and_partial_indexes_keep_sort_against_real_catalog() {
    let (mut p, mut cat) = fresh();
    ddl(&mut cat, &mut p, "CREATE TABLE t(a INTEGER)");
    ddl(&mut cat, &mut p, "CREATE INDEX ia_pos ON t(a) WHERE a > 0");
    // A PARTIAL index covers only some rows, so a full scan would MISS rows — keep the Sort.
    assert_kept(&cat, "SELECT a FROM t ORDER BY a");

    let (mut p2, mut cat2) = fresh();
    ddl(&mut cat2, &mut p2, "CREATE TABLE tc(s TEXT COLLATE NOCASE)");
    ddl(&mut cat2, &mut p2, "CREATE INDEX ixs ON tc(s)");
    // A NOCASE-collated column cannot be served by the BINARY index order.
    assert_kept(&cat2, "SELECT s FROM tc ORDER BY s");
}

#[test]
fn aggregate_and_distinct_keep_sort_against_real_catalog() {
    let (mut p, mut cat) = fresh();
    ddl(&mut cat, &mut p, "CREATE TABLE t(a INTEGER)");
    ddl(&mut cat, &mut p, "CREATE INDEX ia ON t(a)");
    // Aggregation / DISTINCT re-order rows relative to the scan, so the Sort stays. (These
    // are gated off before `order_scan` even runs; pinned here for the real-catalog route.)
    assert_kept(&cat, "SELECT a, count(*) FROM t GROUP BY a ORDER BY a");
    assert_kept(&cat, "SELECT DISTINCT a FROM t ORDER BY a");
}
