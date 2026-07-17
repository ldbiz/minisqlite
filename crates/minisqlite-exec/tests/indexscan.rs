//! Integration tests for the `IndexScan` access-path operator, run over a real
//! `MemPager` + `SchemaCatalog`. The index is created through the real catalog path
//! and then populated MANUALLY (index maintenance during `INSERT` is a separate DML
//! task and may not have landed): each entry is `encode_record([a, rowid])` inserted
//! into the index b-tree via `index_insert`.
//!
//! The index stores only `(a, rowid)`, so the emitted `b` column and the trailing
//! rowid can only be correct if the operator fetches the row from the TABLE by the
//! rowid recorded in the index entry — which is exactly what these tests assert.
//!
//! Coverage beyond the single-column `i`:
//! * a CORRELATED seek whose `eq_prefix` references the `outer` row, driven through a
//!   real `IndexNestedLoop` join ([`correlated_join`]) — the index-nested-loop path;
//! * a COMPOSITE seek (equality on the leading column + a range on the next) over a
//!   two-column index `i2 ON t(a, b)` ([`setup_ab`]), forward and reverse;
//! * the defensive `Err` paths — an over-wide prefix, a range bound with a full-width
//!   prefix, and a corrupt index entry (non-integer / dangling rowid).

use minisqlite_btree::{index_insert, init_database, table_insert};
use minisqlite_catalog::{Catalog, SchemaCatalog};
use minisqlite_exec::{Executor, Runtime, StreamingExecutor};
use minisqlite_expr::EvalExpr;
use minisqlite_fileformat::encode_record;
use minisqlite_pager::{MemPager, Pager};
use minisqlite_plan::{
    IndexOp, IndexScan, Join, JoinStrategy, JoinType, Plan, PlanNode, RangeBound, ScanDirection,
};
use minisqlite_sql::{parse, Statement};
use minisqlite_types::Value;
use minisqlite_exec::PagerSet;
use minisqlite_types::DbIndex;

// ----- fixtures ------------------------------------------------------------

/// The seed rows as `(rowid, a, b)`. Duplicate `a` values (10 and 20 each appear
/// twice) exercise value-group handling; the ordered index is
/// `(10,r2),(10,r4),(20,r1),(20,r5),(30,r3)`.
const ROWS: &[(i64, i64, &str)] =
    &[(1, 20, "r1"), (2, 10, "r2"), (3, 30, "r3"), (4, 10, "r4"), (5, 20, "r5")];

/// A fresh in-memory database with table `t(a INTEGER, b TEXT)`, its rows seeded, and
/// an index `i ON t(a)` created and populated by hand.
fn setup() -> (MemPager, SchemaCatalog) {
    let (mut pager, mut cat) = fresh_t();

    // Create the index (catalog allocates an empty index b-tree; it does NOT backfill).
    create_index(&mut cat, &mut pager, "CREATE INDEX i ON t(a)");
    let idx_root = cat.index("i").unwrap().unwrap().root_page;

    // Populate the index b-tree by hand: key = [a, rowid].
    pager.begin().unwrap();
    for (rowid, a, _b) in ROWS {
        let key = encode_record(&[Value::Integer(*a), Value::Integer(*rowid)]);
        index_insert(&mut pager, idx_root, &key).unwrap();
    }
    pager.commit().unwrap();

    (pager, cat)
}

fn create_index(cat: &mut SchemaCatalog, pager: &mut MemPager, sql: &str) {
    let ast = parse(sql).unwrap();
    let Statement::CreateIndex(stmt) = &ast.statements[0] else {
        panic!("not a CREATE INDEX: {sql}");
    };
    cat.create_index(pager, stmt, sql).unwrap();
}

/// A fresh database with table `t(a INTEGER, b TEXT)` seeded with [`ROWS`] and its rows
/// present in the table b-tree. Shared by the single-column, two-column, and
/// corrupt-entry fixtures so the base data is identical across them.
fn fresh_t() -> (MemPager, SchemaCatalog) {
    let mut pager = MemPager::new(4096);
    init_database(&mut pager).unwrap();
    let mut cat = SchemaCatalog::new();

    let sql = "CREATE TABLE t(a INTEGER, b TEXT)";
    let ast = parse(sql).unwrap();
    let Statement::CreateTable(stmt) = &ast.statements[0] else {
        panic!("not a CREATE TABLE");
    };
    cat.create_table(&mut pager, stmt, sql).unwrap();
    let table_root = cat.table("t").unwrap().unwrap().root_page;

    pager.begin().unwrap();
    for (rowid, a, b) in ROWS {
        let rec = encode_record(&[Value::Integer(*a), Value::Text((*b).to_string())]);
        table_insert(&mut pager, table_root, *rowid, &rec).unwrap();
    }
    pager.commit().unwrap();

    (pager, cat)
}

/// Like [`setup`] but the index is `i2 ON t(a, b)` — a TWO-column index whose keys are
/// `[a, b, rowid]`. This is the fixture that can exercise a `Seek` with a non-empty
/// `eq_prefix` (equality on `a`) AND a range on the NEXT column (`b`): the composite
/// multi-column seek a single-column index physically cannot reach.
fn setup_ab() -> (MemPager, SchemaCatalog) {
    let (mut pager, mut cat) = fresh_t();
    create_index(&mut cat, &mut pager, "CREATE INDEX i2 ON t(a, b)");
    let idx_root = cat.index("i2").unwrap().unwrap().root_page;

    // Key = [a, b, rowid]; the b-tree orders these under Binary, so within an `a` group
    // the entries are `b`-ascending (ties by rowid).
    pager.begin().unwrap();
    for (rowid, a, b) in ROWS {
        let key = encode_record(&[
            Value::Integer(*a),
            Value::Text((*b).to_string()),
            Value::Integer(*rowid),
        ]);
        index_insert(&mut pager, idx_root, &key).unwrap();
    }
    pager.commit().unwrap();

    (pager, cat)
}

/// Table `t` seeded normally (so rowids 1..=5 exist), index `i ON t(a)` created, then a
/// single RAW key record inserted into `i` by hand — used to inject one corrupt index
/// entry the operator must reject loudly (a non-integer rowid, or a rowid with no table
/// row). Only the one injected entry is present in the index.
fn db_with_corrupt_index_entry(entry: &[Value]) -> (MemPager, SchemaCatalog) {
    let (mut pager, mut cat) = fresh_t();
    create_index(&mut cat, &mut pager, "CREATE INDEX i ON t(a)");
    let idx_root = cat.index("i").unwrap().unwrap().root_page;

    pager.begin().unwrap();
    index_insert(&mut pager, idx_root, &encode_record(entry)).unwrap();
    pager.commit().unwrap();

    (pager, cat)
}

/// Drain a plan to completion and collect the result rows.
fn run(plan: &Plan, cat: &SchemaCatalog, pager: &mut MemPager) -> Vec<Vec<Value>> {
    let mut rt = Runtime::new();
    let mut exec = StreamingExecutor;
    let mut cur = exec.execute(plan, cat, PagerSet::One { db: DbIndex::MAIN, pager: &mut *pager }).unwrap();
    let mut out = Vec::new();
    while let Some(row) = cur.next_row(&mut rt).unwrap() {
        out.push(row);
    }
    out
}

/// Drive a plan expecting a failure somewhere along the way — either at build time
/// (`execute`) or while draining. Returns the error (or `Ok(())` if none occurred), so
/// a test can assert the operator fails LOUD on a malformed plan / corrupt entry rather
/// than silently returning wrong or empty rows.
fn run_err(plan: &Plan, cat: &SchemaCatalog, pager: &mut MemPager) -> minisqlite_types::Result<()> {
    let mut rt = Runtime::new();
    let mut exec = StreamingExecutor;
    let mut cur = exec.execute(plan, cat, PagerSet::One { db: DbIndex::MAIN, pager: &mut *pager })?;
    while cur.next_row(&mut rt)?.is_some() {}
    Ok(())
}

// ----- plan-building helpers -----------------------------------------------

fn lit(v: Value) -> EvalExpr {
    EvalExpr::Literal(v)
}

/// A correlated reference to register `i` of the row the operator is evaluated against
/// — here the OUTER row supplied by an enclosing index-nested-loop drive.
fn col(i: usize) -> EvalExpr {
    EvalExpr::Column(i)
}

fn plan(root: PlanNode) -> Plan {
    Plan {
        root,
        result_columns: vec!["a".into(), "b".into()],
        ctes: Vec::new(),
        subqueries: Vec::new(),
        mutates: false,
        generated: Vec::new(),
    }
}

fn index_node(op: IndexOp, direction: ScanDirection) -> PlanNode {
    PlanNode::IndexScan(IndexScan { db: DbIndex::MAIN,
        table: "t".into(),
        column_count: 2,
        index: "i".into(),
        op,
        direction,
        covering: false,
    })
}

fn fullscan(direction: ScanDirection) -> PlanNode {
    index_node(IndexOp::FullScan, direction)
}

fn seek(
    eq_prefix: Vec<EvalExpr>,
    low: Option<RangeBound>,
    high: Option<RangeBound>,
    direction: ScanDirection,
) -> PlanNode {
    index_node(IndexOp::Seek { eq_prefix, low, high }, direction)
}

fn bound(v: i64, inclusive: bool) -> RangeBound {
    RangeBound { value: lit(Value::Integer(v)), inclusive }
}

/// An `IndexScan` over the TWO-column index `i2 ON t(a, b)` (see [`setup_ab`]).
fn index2_node(op: IndexOp, direction: ScanDirection) -> PlanNode {
    PlanNode::IndexScan(IndexScan { db: DbIndex::MAIN,
        table: "t".into(),
        column_count: 2,
        index: "i2".into(),
        op,
        direction,
        covering: false,
    })
}

fn seek2(
    eq_prefix: Vec<EvalExpr>,
    low: Option<RangeBound>,
    high: Option<RangeBound>,
    direction: ScanDirection,
) -> PlanNode {
    index2_node(IndexOp::Seek { eq_prefix, low, high }, direction)
}

/// A TEXT range bound (for the `b` column of the two-column index).
fn tbound(s: &str, inclusive: bool) -> RangeBound {
    RangeBound { value: lit(Value::Text(s.into())), inclusive }
}

// ----- assertions ----------------------------------------------------------

fn int(v: &Value) -> i64 {
    match v {
        Value::Integer(i) => *i,
        other => panic!("expected Integer, got {other:?}"),
    }
}

fn text(v: &Value) -> &str {
    match v {
        Value::Text(s) => s,
        other => panic!("expected Text, got {other:?}"),
    }
}

/// Extract `(a, b, rowid)` triples from result rows, asserting the width-`N+1` shape.
/// `b` and `rowid` come only from the TABLE fetch (the index holds just `(a, rowid)`),
/// so a correct triple proves the by-rowid table lookup ran.
fn triples(rows: &[Vec<Value>]) -> Vec<(i64, String, i64)> {
    rows.iter()
        .map(|r| {
            assert_eq!(r.len(), 3, "an index scan emits width N+1 = 3 for a 2-column table");
            (int(&r[0]), text(&r[1]).to_string(), int(&r[2]))
        })
        .collect()
}

/// The rowid (trailing register) of each row.
fn rowids(rows: &[Vec<Value>]) -> Vec<i64> {
    rows.iter().map(|r| int(&r[r.len() - 1])).collect()
}

fn t(a: i64, b: &str, rowid: i64) -> (i64, String, i64) {
    (a, b.to_string(), rowid)
}

// ----- (1) FullScan --------------------------------------------------------

#[test]
fn fullscan_forward_yields_all_rows_in_index_order() {
    let (mut pager, cat) = setup();
    let rows = run(&plan(fullscan(ScanDirection::Forward)), &cat, &mut pager);
    assert_eq!(
        triples(&rows),
        vec![
            t(10, "r2", 2),
            t(10, "r4", 4),
            t(20, "r1", 1),
            t(20, "r5", 5),
            t(30, "r3", 3),
        ],
        "forward full index scan is a-ascending, ties broken by rowid; b/rowid from the table"
    );
}

#[test]
fn fullscan_reverse_yields_all_rows_descending() {
    let (mut pager, cat) = setup();
    let rows = run(&plan(fullscan(ScanDirection::Reverse)), &cat, &mut pager);
    assert_eq!(
        triples(&rows),
        vec![
            t(30, "r3", 3),
            t(20, "r5", 5),
            t(20, "r1", 1),
            t(10, "r4", 4),
            t(10, "r2", 2),
        ],
        "reverse full index scan is the exact reverse of forward"
    );
}

// ----- (2) equality seek ---------------------------------------------------

#[test]
fn seek_eq_forward_returns_exactly_the_group() {
    let (mut pager, cat) = setup();
    let node = seek(vec![lit(Value::Integer(10))], None, None, ScanDirection::Forward);
    let rows = run(&plan(node), &cat, &mut pager);
    assert_eq!(triples(&rows), vec![t(10, "r2", 2), t(10, "r4", 4)], "a = 10 group, rowid order");
}

#[test]
fn seek_eq_reverse_returns_group_descending() {
    // Exercises the reverse group-top walk (no upper bound, equality prefix).
    let (mut pager, cat) = setup();
    let node = seek(vec![lit(Value::Integer(20))], None, None, ScanDirection::Reverse);
    let rows = run(&plan(node), &cat, &mut pager);
    assert_eq!(triples(&rows), vec![t(20, "r5", 5), t(20, "r1", 1)], "a = 20 group, descending");
}

#[test]
fn seek_eq_absent_value_is_empty() {
    let (mut pager, cat) = setup();
    let node = seek(vec![lit(Value::Integer(99))], None, None, ScanDirection::Forward);
    assert!(run(&plan(node), &cat, &mut pager).is_empty(), "no a = 99 entries");
}

// ----- (3) leading-column range (eq_prefix empty) --------------------------

#[test]
fn range_forward_inclusive_both_bounds() {
    let (mut pager, cat) = setup();
    let node = seek(vec![], Some(bound(10, true)), Some(bound(20, true)), ScanDirection::Forward);
    let rows = run(&plan(node), &cat, &mut pager);
    assert_eq!(rowids(&rows), vec![2, 4, 1, 5], "10 <= a <= 20 ascending");
}

#[test]
fn range_forward_exclusive_high_drops_boundary_group() {
    // a < 20 excludes the entire a = 20 group.
    let (mut pager, cat) = setup();
    let node = seek(vec![], None, Some(bound(20, false)), ScanDirection::Forward);
    let rows = run(&plan(node), &cat, &mut pager);
    assert_eq!(rowids(&rows), vec![2, 4], "a < 20 keeps only the a = 10 group");
}

#[test]
fn range_forward_exclusive_low_skips_boundary_group() {
    // a > 10 skips the a = 10 group the seek lands in, then runs to the end (no high).
    let (mut pager, cat) = setup();
    let node = seek(vec![], Some(bound(10, false)), None, ScanDirection::Forward);
    let rows = run(&plan(node), &cat, &mut pager);
    assert_eq!(rowids(&rows), vec![1, 5, 3], "a > 10 ascending to the end");
}

#[test]
fn range_reverse_inclusive_high_present_walks_group_top() {
    // Upper bound value (20) is PRESENT: reverse start walks to the top of the a = 20
    // group (seek_ge + walk_to_top). 10 <= a <= 20 descending.
    let (mut pager, cat) = setup();
    let node = seek(vec![], Some(bound(10, true)), Some(bound(20, true)), ScanDirection::Reverse);
    let rows = run(&plan(node), &cat, &mut pager);
    assert_eq!(rowids(&rows), vec![5, 1, 4, 2], "descending, a = 20 group included and on top");
}

#[test]
fn range_reverse_inclusive_high_absent_steps_back() {
    // Upper bound value (25) is ABSENT: reverse start is `seek_ge(25)` (lands on a = 30,
    // above range) then `prev()` back to the a = 20 group top. a <= 25 descending.
    let (mut pager, cat) = setup();
    let node = seek(vec![], None, Some(bound(25, true)), ScanDirection::Reverse);
    let rows = run(&plan(node), &cat, &mut pager);
    assert_eq!(rowids(&rows), vec![5, 1, 4, 2], "a <= 25 descending (30 excluded)");
}

#[test]
fn range_reverse_exclusive_high_uses_seek_le() {
    // Exclusive upper bound: reverse start is `seek_le([20])`, which lands strictly
    // below the a = 20 group (a < 20). Descending a = 10 group.
    let (mut pager, cat) = setup();
    let node = seek(vec![], None, Some(bound(20, false)), ScanDirection::Reverse);
    let rows = run(&plan(node), &cat, &mut pager);
    assert_eq!(rowids(&rows), vec![4, 2], "a < 20 descending");
}

#[test]
fn range_reverse_low_only_starts_at_max() {
    // No upper bound, empty prefix: reverse start is `last()`. a >= 15 descending, and
    // the exclusive/inclusive low STOPS the scan when it drops below 15.
    let (mut pager, cat) = setup();
    let node = seek(vec![], Some(bound(15, true)), None, ScanDirection::Reverse);
    let rows = run(&plan(node), &cat, &mut pager);
    assert_eq!(rowids(&rows), vec![3, 5, 1], "a >= 15 descending, stops before a = 10");
}

// ----- (4) empty range -----------------------------------------------------

#[test]
fn range_empty_when_low_exceeds_high() {
    // low = 25 > high = 15: no entry can satisfy both — a real empty result, not a hang.
    let (mut pager, cat) = setup();
    let node = seek(vec![], Some(bound(25, true)), Some(bound(15, true)), ScanDirection::Forward);
    assert!(run(&plan(node), &cat, &mut pager).is_empty(), "25 <= a <= 15 is empty");
}

// ----- (5) correlated / index-nested-loop drive: eq_prefix references `outer` -----

/// An `IndexNestedLoop` join whose LEFT is a `Values` list of seek keys and whose RIGHT
/// is an `IndexScan` with `eq_prefix = [Column(0)]`. The join builds the right leaf with
/// `outer = &left_row` (see the executor's index-nested-loop cursor), so the IndexScan's
/// `prepare` MUST evaluate the prefix against that outer row. This is the correlated
/// path that a top-level `lit(...)` seek (every other test here) never exercises.
///
/// The join emits `left ++ [a, b, rowid]` (width 1 + 3 = 4). Asserting `left_key == a`
/// proves the seek used the outer value: a bug that evaluated the prefix against an
/// empty outer would error on `Column(0)` (out-of-range) or seek on `NULL` and return
/// nothing, so this would fail rather than pass silently.
fn correlated_join(seek_keys: &[i64]) -> PlanNode {
    let left_rows: Vec<Vec<EvalExpr>> =
        seek_keys.iter().map(|k| vec![lit(Value::Integer(*k))]).collect();
    let right = index_node(
        IndexOp::Seek { eq_prefix: vec![col(0)], low: None, high: None },
        ScanDirection::Forward,
    );
    PlanNode::Join(Join {
        left: Box::new(PlanNode::Values { rows: left_rows }),
        left_width: 1,
        right: Box::new(right),
        right_width: 3, // the IndexScan leaf emits width N+1 = 3
        join_type: JoinType::Inner,
        on: None, // the correlated seek IS the equality; no residual predicate
        strategy: JoinStrategy::IndexNestedLoop,
    })
}

#[test]
fn correlated_seek_evaluates_eq_prefix_against_outer() {
    let (mut pager, cat) = setup();
    let rows = run(&plan(correlated_join(&[10, 20, 99])), &cat, &mut pager);

    // left 10 -> the a = 10 group {r2, r4}; left 20 -> {r1, r5}; left 99 -> absent, so an
    // Inner join drops it. Combined row is [left_key, a, b, rowid].
    let got: Vec<(i64, i64, String, i64)> = rows
        .iter()
        .map(|r| {
            assert_eq!(r.len(), 4, "combined width = left(1) + IndexScan local(3)");
            (int(&r[0]), int(&r[1]), text(&r[2]).to_string(), int(&r[3]))
        })
        .collect();
    assert_eq!(
        got,
        vec![
            (10, 10, "r2".to_string(), 2),
            (10, 10, "r4".to_string(), 4),
            (20, 20, "r1".to_string(), 1),
            (20, 20, "r5".to_string(), 5),
        ],
        "each left row drives a fresh seek keyed off the outer value"
    );
    assert!(
        got.iter().all(|(left_key, a, _, _)| left_key == a),
        "the seeked `a` equals the outer seek key — the prefix was evaluated against `outer`"
    );
}

// ----- (6) composite seek: equality prefix + range on the NEXT column ------

// These need the two-column index `i2 ON t(a, b)` ([`setup_ab`]): with `key_columns == 2`
// a `Seek { eq_prefix: [a], low/high on b }` reaches `classify`'s `p < key_columns`
// bound branch for `p == 1` — the canonical multi-column seek (equality on the leading
// column, range on the next) that the single-column fixture cannot express. Within an
// `a` group the entries are `b`-ascending: a=10 -> {r2, r4}, a=20 -> {r1, r5}.

#[test]
fn composite_seek_forward_inclusive_range_on_next_column() {
    let (mut pager, cat) = setup_ab();
    let node = seek2(
        vec![lit(Value::Integer(10))],
        Some(tbound("r2", true)),
        Some(tbound("r4", true)),
        ScanDirection::Forward,
    );
    let rows = run(&plan(node), &cat, &mut pager);
    assert_eq!(
        triples(&rows),
        vec![t(10, "r2", 2), t(10, "r4", 4)],
        "a = 10 AND 'r2' <= b <= 'r4' keeps the whole a = 10 group, b-ascending"
    );
}

#[test]
fn composite_seek_forward_exclusive_high_on_next_column() {
    let (mut pager, cat) = setup_ab();
    let node = seek2(
        vec![lit(Value::Integer(10))],
        None,
        Some(tbound("r4", false)),
        ScanDirection::Forward,
    );
    let rows = run(&plan(node), &cat, &mut pager);
    assert_eq!(
        triples(&rows),
        vec![t(10, "r2", 2)],
        "a = 10 AND b < 'r4' drops the r4 boundary within the group"
    );
}

#[test]
fn composite_seek_forward_exclusive_low_on_next_column() {
    let (mut pager, cat) = setup_ab();
    let node = seek2(
        vec![lit(Value::Integer(20))],
        Some(tbound("r1", false)),
        None,
        ScanDirection::Forward,
    );
    let rows = run(&plan(node), &cat, &mut pager);
    assert_eq!(
        triples(&rows),
        vec![t(20, "r5", 5)],
        "a = 20 AND b > 'r1' skips the r1 boundary and keeps r5"
    );
}

#[test]
fn composite_seek_reverse_inclusive_high_on_next_column() {
    // Reverse over the composite seek exercises the group-top walk with a TWO-column
    // prefix: the start is the top of the a = 20 group at b = 'r5'.
    let (mut pager, cat) = setup_ab();
    let node = seek2(
        vec![lit(Value::Integer(20))],
        None,
        Some(tbound("r5", true)),
        ScanDirection::Reverse,
    );
    let rows = run(&plan(node), &cat, &mut pager);
    assert_eq!(
        triples(&rows),
        vec![t(20, "r5", 5), t(20, "r1", 1)],
        "a = 20 AND b <= 'r5' descending"
    );
}

#[test]
fn composite_seek_reverse_exclusive_low_on_next_column() {
    // Independently pins the REVERSE composite bound at p = 1: `a = 20 AND b > 'r1'`
    // descending drops the r1 boundary (the reverse near-bound STOPs the scan), leaving
    // only r5. Unlike the inclusive-range reverse case (whole group in range), here a
    // bound-ignored impl would wrongly also emit r1 — so this distinguishes the branch.
    let (mut pager, cat) = setup_ab();
    let node = seek2(
        vec![lit(Value::Integer(20))],
        Some(tbound("r1", false)),
        None,
        ScanDirection::Reverse,
    );
    let rows = run(&plan(node), &cat, &mut pager);
    assert_eq!(
        triples(&rows),
        vec![t(20, "r5", 5)],
        "a = 20 AND b > 'r1' descending keeps only r5"
    );
}

// ----- (7) defensive Err paths (malformed plan / corrupt index entry) ------

#[test]
fn seek_prefix_wider_than_index_is_a_loud_error() {
    // Index `i` has ONE column; a two-value equality prefix is a planner bug and must
    // fail closed, not silently return no rows.
    let (mut pager, cat) = setup();
    let node = seek(
        vec![lit(Value::Integer(10)), lit(Value::Integer(20))],
        None,
        None,
        ScanDirection::Forward,
    );
    assert!(
        run_err(&plan(node), &cat, &mut pager).is_err(),
        "eq_prefix wider than the index must error"
    );
}

#[test]
fn seek_range_bound_with_full_width_prefix_is_a_loud_error() {
    // The equality prefix covers BOTH columns of `i2`, leaving no column for the range
    // bound to apply to — a contradictory plan that must fail closed rather than have
    // the bound silently ignored.
    let (mut pager, cat) = setup_ab();
    let node = seek2(
        vec![lit(Value::Integer(10)), lit(Value::Text("r2".into()))],
        None,
        Some(tbound("r9", true)),
        ScanDirection::Forward,
    );
    assert!(
        run_err(&plan(node), &cat, &mut pager).is_err(),
        "a range bound with a full-width equality prefix must error"
    );
}

#[test]
fn non_integer_rowid_in_index_entry_is_a_loud_error() {
    // The trailing value of an index key must be the integer rowid; a TEXT there is
    // corruption, so the fetch must error rather than skip or panic.
    let (mut pager, cat) =
        db_with_corrupt_index_entry(&[Value::Integer(10), Value::Text("notrowid".into())]);
    assert!(
        run_err(&plan(fullscan(ScanDirection::Forward)), &cat, &mut pager).is_err(),
        "a non-integer trailing rowid must error"
    );
}

#[test]
fn dangling_index_entry_missing_table_row_is_a_loud_error() {
    // Entry points at rowid 9999, which has no table row (a dangling entry / index-
    // maintenance bug). Fail loud rather than silently drop it.
    let (mut pager, cat) =
        db_with_corrupt_index_entry(&[Value::Integer(10), Value::Integer(9999)]);
    assert!(
        run_err(&plan(fullscan(ScanDirection::Forward)), &cat, &mut pager).is_err(),
        "a rowid with no table row must error"
    );
}
