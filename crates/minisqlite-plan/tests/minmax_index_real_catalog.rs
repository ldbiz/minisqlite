//! Real-catalog fidelity for the MIN/MAX index/rowid seek (`compile::minmax_index`).
//!
//! The plan-shape unit tests in `src/compile/minmax_index.rs` prove the rewrite against a
//! HAND-BUILT static `Cat`. This integration test proves the SAME rewrites fire (and the
//! same ineligible cases keep the full-scan `Aggregate`) against the REAL [`SchemaCatalog`]
//! — the store the engine actually uses — populated by running real `CREATE TABLE` / `CREATE
//! INDEX` DDL through a real pager.
//!
//! Why this test exists (the gap it closes): the facade behavioral tests
//! (`minisqlite/tests/conformance_minmax_index.rs`) can only observe RESULTS, which stay
//! CORRECT whether the seek fires OR the scan runs. So a silent FAILED rewrite — the real
//! catalog populating an `IndexDef` (`key_columns` / collation / `partial`) or a `TableDef`
//! (`rowid_alias`) differently from the static stub, so the seek never fires in production —
//! would leave every facade test green while the perf win was lost AND the seek executor
//! went unexercised. `EXPLAIN QUERY PLAN` is not supported, so here we compile against the
//! real catalog and assert on the `Plan` tree directly (mirrors `order_scan_real_catalog`).

use minisqlite_btree::init_database;
use minisqlite_catalog::{Catalog, SchemaCatalog};
use minisqlite_pager::MemPager;
use minisqlite_plan::{MinMaxSeek, MinMaxSource, Plan, PlanNode, Planner, QueryPlanner};
use minisqlite_sql::{parse, Statement};

/// A fresh in-memory database image (page 1 formatted) plus an empty catalog cache.
fn fresh() -> (MemPager, SchemaCatalog) {
    let mut pager = MemPager::new(4096);
    init_database(&mut pager).expect("format page 1");
    (pager, SchemaCatalog::new())
}

/// Run one DDL statement through the real catalog write path, so the resulting
/// `TableDef` / `IndexDef` are exactly what the engine builds.
fn ddl(cat: &mut SchemaCatalog, pager: &mut MemPager, sql: &str) {
    let ast = parse(sql).expect("parse ddl");
    match &ast.statements[0] {
        Statement::CreateTable(stmt) => cat.create_table(pager, stmt, sql).expect("create table"),
        Statement::CreateIndex(stmt) => cat.create_index(pager, stmt, sql).expect("create index"),
        other => panic!("expected CREATE TABLE/INDEX, got {other:?}"),
    }
}

fn plan(cat: &SchemaCatalog, sql: &str) -> Plan {
    let ast = parse(sql).expect("parse select");
    let stmt = ast.statements.first().expect("one statement");
    QueryPlanner::new().plan(stmt, cat).expect("plan")
}

/// Descend the row-preserving wrappers to the aggregate-or-seek node.
fn agg_or_seek(node: &PlanNode) -> &PlanNode {
    match node {
        PlanNode::MinMaxSeek(_) | PlanNode::Aggregate(_) => node,
        PlanNode::Project { input, .. }
        | PlanNode::Filter { input, .. }
        | PlanNode::Sort { input, .. }
        | PlanNode::Distinct { input, .. }
        | PlanNode::Limit { input, .. } => agg_or_seek(input),
        other => panic!("no Aggregate/MinMaxSeek on the chain, reached {other:?}"),
    }
}

#[track_caller]
fn seek_of(cat: &SchemaCatalog, sql: &str) -> MinMaxSeek {
    let p = plan(cat, sql);
    match agg_or_seek(&p.root) {
        PlanNode::MinMaxSeek(m) => m.clone(),
        other => panic!("expected a MinMaxSeek for `{sql}`, got {other:?}"),
    }
}

#[track_caller]
fn assert_aggregate(cat: &SchemaCatalog, sql: &str) {
    match agg_or_seek(&plan(cat, sql).root) {
        PlanNode::Aggregate(_) => {}
        PlanNode::MinMaxSeek(m) => panic!("expected the full-scan Aggregate for `{sql}`, got MinMaxSeek {m:?}"),
        _ => unreachable!(),
    }
}

// ===================== FIRES against the real catalog ===========================

#[test]
fn indexed_integer_min_max_seeks_the_index() {
    let (mut p, mut cat) = fresh();
    ddl(&mut cat, &mut p, "CREATE TABLE t(x INTEGER)");
    ddl(&mut cat, &mut p, "CREATE INDEX ix ON t(x)");
    let mn = seek_of(&cat, "SELECT MIN(x) FROM t");
    assert!(!mn.is_max);
    assert!(matches!(mn.source, MinMaxSource::Index { .. }), "MIN(x) uses the index");
    let mx = seek_of(&cat, "SELECT MAX(x) FROM t");
    assert!(mx.is_max);
    assert!(matches!(mx.source, MinMaxSource::Index { .. }), "MAX(x) uses the index");
}

#[test]
fn indexed_binary_text_min_max_seeks_the_index() {
    let (mut p, mut cat) = fresh();
    ddl(&mut cat, &mut p, "CREATE TABLE ct(s TEXT)");
    ddl(&mut cat, &mut p, "CREATE INDEX ics ON ct(s)");
    assert!(matches!(seek_of(&cat, "SELECT MIN(s) FROM ct").source, MinMaxSource::Index { .. }));
    assert!(matches!(seek_of(&cat, "SELECT MAX(s) FROM ct").source, MinMaxSource::Index { .. }));
}

#[test]
fn rowid_alias_and_keyword_seek_the_table_btree() {
    let (mut p, mut cat) = fresh();
    ddl(&mut cat, &mut p, "CREATE TABLE u(id INTEGER PRIMARY KEY, v TEXT)");
    ddl(&mut cat, &mut p, "CREATE TABLE t(a INTEGER)");
    // INTEGER PRIMARY KEY aliases the rowid → the table b-tree seek.
    assert!(matches!(seek_of(&cat, "SELECT MIN(id) FROM u").source, MinMaxSource::Rowid));
    assert!(matches!(seek_of(&cat, "SELECT MAX(id) FROM u").source, MinMaxSource::Rowid));
    // The bare rowid keywords likewise, with no INTEGER PK alias.
    assert!(matches!(seek_of(&cat, "SELECT MAX(rowid) FROM t").source, MinMaxSource::Rowid));
    assert!(matches!(seek_of(&cat, "SELECT MIN(_rowid_) FROM t").source, MinMaxSource::Rowid));
    assert!(matches!(seek_of(&cat, "SELECT MAX(oid) FROM t").source, MinMaxSource::Rowid));
}

#[test]
fn outer_expression_keeps_the_seek_under_a_project() {
    let (mut p, mut cat) = fresh();
    ddl(&mut cat, &mut p, "CREATE TABLE t(x INTEGER)");
    ddl(&mut cat, &mut p, "CREATE INDEX ix ON t(x)");
    let root = plan(&cat, "SELECT MAX(x)+1 FROM t").root;
    assert!(matches!(&root, PlanNode::Project { .. }), "outer expr keeps its Project");
    assert!(seek_of(&cat, "SELECT MAX(x)+1 FROM t").is_max);
}

#[test]
fn leftmost_column_of_a_composite_index_seeks() {
    let (mut p, mut cat) = fresh();
    ddl(&mut cat, &mut p, "CREATE TABLE t(x INTEGER, y INTEGER)");
    ddl(&mut cat, &mut p, "CREATE INDEX ixy ON t(x, y)");
    // §13 "left-most column of an index": the leading column `x` of the real composite
    // `IndexDef` is served by the seek...
    assert!(matches!(seek_of(&cat, "SELECT MIN(x) FROM t").source, MinMaxSource::Index { .. }));
    assert!(matches!(seek_of(&cat, "SELECT MAX(x) FROM t").source, MinMaxSource::Index { .. }));
    // ...while the trailing column `y` is not (no index presents y's extremum in order).
    assert_aggregate(&cat, "SELECT MIN(y) FROM t");
    assert_aggregate(&cat, "SELECT MAX(y) FROM t");
}

// ============= DECLINES against the real catalog (Aggregate kept) ================

#[test]
fn no_index_declines() {
    let (mut p, mut cat) = fresh();
    ddl(&mut cat, &mut p, "CREATE TABLE t(x INTEGER)");
    assert_aggregate(&cat, "SELECT MIN(x) FROM t");
    assert_aggregate(&cat, "SELECT MAX(x) FROM t");
}

#[test]
fn nocase_column_declines_even_with_an_index() {
    let (mut p, mut cat) = fresh();
    ddl(&mut cat, &mut p, "CREATE TABLE tc(s TEXT COLLATE NOCASE)");
    ddl(&mut cat, &mut p, "CREATE INDEX ixs ON tc(s)");
    assert_aggregate(&cat, "SELECT MIN(s) FROM tc");
    assert_aggregate(&cat, "SELECT MAX(s) FROM tc");
}

#[test]
fn desc_index_declines() {
    let (mut p, mut cat) = fresh();
    ddl(&mut cat, &mut p, "CREATE TABLE t(x INTEGER)");
    ddl(&mut cat, &mut p, "CREATE INDEX ix ON t(x DESC)");
    assert_aggregate(&cat, "SELECT MIN(x) FROM t");
    assert_aggregate(&cat, "SELECT MAX(x) FROM t");
}

#[test]
fn partial_index_declines() {
    let (mut p, mut cat) = fresh();
    ddl(&mut cat, &mut p, "CREATE TABLE t(x INTEGER)");
    ddl(&mut cat, &mut p, "CREATE INDEX ixp ON t(x) WHERE x > 0");
    assert_aggregate(&cat, "SELECT MIN(x) FROM t");
}

#[test]
fn blob_affinity_column_declines() {
    let (mut p, mut cat) = fresh();
    ddl(&mut cat, &mut p, "CREATE TABLE t(b BLOB)");
    ddl(&mut cat, &mut p, "CREATE INDEX ixb ON t(b)");
    assert_aggregate(&cat, "SELECT MIN(b) FROM t");
    // A truly typeless column is BLOB affinity too.
    let (mut p2, mut cat2) = fresh();
    ddl(&mut cat2, &mut p2, "CREATE TABLE z(v)");
    ddl(&mut cat2, &mut p2, "CREATE INDEX ixz ON z(v)");
    assert_aggregate(&cat2, "SELECT MAX(v) FROM z");
}

#[test]
fn without_rowid_table_declines() {
    let (mut p, mut cat) = fresh();
    ddl(&mut cat, &mut p, "CREATE TABLE w(k INTEGER PRIMARY KEY, v TEXT) WITHOUT ROWID");
    // A WITHOUT ROWID table has no integer rowid; the seek executor models neither the PK
    // b-tree nor a WR secondary index (whose trailing key is the PK, not a rowid), so it
    // declines to the correct scan even on the PRIMARY KEY column — a documented boundary.
    assert_aggregate(&cat, "SELECT MIN(k) FROM w");
    assert_aggregate(&cat, "SELECT MAX(k) FROM w");
}

#[test]
fn ineligible_query_shapes_decline() {
    let (mut p, mut cat) = fresh();
    ddl(&mut cat, &mut p, "CREATE TABLE t(x INTEGER, y INTEGER)");
    ddl(&mut cat, &mut p, "CREATE INDEX ix ON t(x)");
    assert_aggregate(&cat, "SELECT MAX(x), COUNT(*) FROM t");
    assert_aggregate(&cat, "SELECT MAX(x), MIN(x) FROM t");
    assert_aggregate(&cat, "SELECT MAX(x+1) FROM t");
    assert_aggregate(&cat, "SELECT MAX(DISTINCT x) FROM t");
    assert_aggregate(&cat, "SELECT MAX(x) FROM t WHERE x < 5");
    assert_aggregate(&cat, "SELECT MAX(x) FROM t GROUP BY y");
    assert_aggregate(&cat, "SELECT COUNT(x) FROM t");
    assert_aggregate(&cat, "SELECT SUM(x) FROM t");
}
