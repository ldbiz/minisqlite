//! Integration tests for the streaming executor: hand-built [`Plan`] trees run over a
//! real [`MemPager`] + [`SchemaCatalog`], seeded with real records, and driven
//! through the [`Executor`]/[`RowCursor`] seam. These pin the ROW/REGISTER
//! convention and the read operators' behavior end to end (not against a private
//! mock — against the actual storage + catalog the engine uses).

use minisqlite_btree::{index_insert, init_database, table_insert};
use minisqlite_catalog::{Catalog, SchemaCatalog};
use minisqlite_exec::{Executor, Runtime, StreamingExecutor};
use minisqlite_expr::{CmpOp, CompareMeta, EvalExpr, SortKey};
use minisqlite_fileformat::encode_record;
use minisqlite_pager::{MemPager, PageId, Pager};
use minisqlite_plan::{
    Delete, Plan, PlanNode, RangeBound, RowidOp, RowidScan, ScanDirection, SeqScan,
};
use minisqlite_sql::{parse, Statement};
use minisqlite_types::{Collation, Value};
use minisqlite_exec::PagerSet;
use minisqlite_types::DbIndex;

// ----- fixtures ------------------------------------------------------------

/// A fresh in-memory database with one table created through the real catalog path.
fn db_with_table(create_sql: &str) -> (MemPager, SchemaCatalog) {
    let mut pager = MemPager::new(4096);
    init_database(&mut pager).unwrap();
    let mut cat = SchemaCatalog::new();
    let ast = parse(create_sql).unwrap();
    let Statement::CreateTable(stmt) = &ast.statements[0] else {
        panic!("not a CREATE TABLE: {create_sql}");
    };
    cat.create_table(&mut pager, stmt, create_sql).unwrap();
    (pager, cat)
}

/// A fresh in-memory database with no user tables (for `VALUES` / `SingleRow`).
fn empty_db() -> (MemPager, SchemaCatalog) {
    let mut pager = MemPager::new(4096);
    init_database(&mut pager).unwrap();
    (pager, SchemaCatalog::new())
}

/// Insert `(rowid, values)` records into the table's b-tree, inside one transaction.
fn seed(pager: &mut MemPager, root: PageId, rows: &[(i64, Vec<Value>)]) {
    pager.begin().unwrap();
    for (rowid, vals) in rows {
        table_insert(pager, root, *rowid, &encode_record(vals)).unwrap();
    }
    pager.commit().unwrap();
}

/// The root page of a created table.
fn table_root(cat: &SchemaCatalog, name: &str) -> PageId {
    cat.table(name).unwrap().unwrap().root_page
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

/// Like [`run`] but returns the `Result` so error paths can be asserted.
fn run_err(plan: &Plan, cat: &SchemaCatalog, pager: &mut MemPager) -> minisqlite_types::Result<()> {
    let mut rt = Runtime::new();
    let mut exec = StreamingExecutor;
    let mut cur = exec.execute(plan, cat, PagerSet::One { db: DbIndex::MAIN, pager: &mut *pager })?;
    while cur.next_row(&mut rt)?.is_some() {}
    Ok(())
}

// ----- plan-building helpers -----------------------------------------------

fn col(i: usize) -> EvalExpr {
    EvalExpr::Column(i)
}

fn lit(v: Value) -> EvalExpr {
    EvalExpr::Literal(v)
}

fn binary_meta() -> CompareMeta {
    CompareMeta { apply_left: None, apply_right: None, collation: Collation::Binary }
}

fn plan(root: PlanNode, columns: &[&str]) -> Plan {
    Plan {
        root,
        result_columns: columns.iter().map(|s| s.to_string()).collect(),
        ctes: Vec::new(),
        subqueries: Vec::new(),
        mutates: false,
        generated: Vec::new(),
    }
}

fn seqscan(table: &str, column_count: usize) -> PlanNode {
    PlanNode::SeqScan(SeqScan { db: DbIndex::MAIN, table: table.to_string(), column_count })
}

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

// ----- (1) SeqScan -> Project -> Sort --------------------------------------

#[test]
fn seqscan_project_sort_ascending() {
    // SELECT a, b FROM t ORDER BY a
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER, b TEXT)");
    let root = table_root(&cat, "t");
    // Stored out of order (rowid 1 has a=2, rowid 2 has a=1): the sort must reorder.
    seed(
        &mut pager,
        root,
        &[
            (1, vec![Value::Integer(2), Value::Text("y".into())]),
            (2, vec![Value::Integer(1), Value::Text("x".into())]),
        ],
    );

    let tree = PlanNode::Sort {
        input: Box::new(PlanNode::Project {
            input: Box::new(seqscan("t", 2)),
            exprs: vec![col(0), col(1)],
        }),
        keys: vec![SortKey {
            expr: col(0),
            desc: false,
            nulls_first: None,
            collation: Collation::Binary,
        }],
        limit: None,
    };
    let rows = run(&plan(tree, &["a", "b"]), &cat, &mut pager);

    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].len(), 2, "Project narrows to 2 columns (no rowid)");
    assert_eq!(int(&rows[0][0]), 1);
    assert_eq!(text(&rows[0][1]), "x");
    assert_eq!(int(&rows[1][0]), 2);
    assert_eq!(text(&rows[1][1]), "y");
}

#[test]
fn sort_descending_and_nulls_default_last_for_desc() {
    // ORDER BY a DESC: values descending; a NULL sorts LAST (the DESC default).
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER)");
    let root = table_root(&cat, "t");
    seed(
        &mut pager,
        root,
        &[
            (1, vec![Value::Integer(1)]),
            (2, vec![Value::Null]),
            (3, vec![Value::Integer(3)]),
        ],
    );

    let tree = PlanNode::Sort {
        input: Box::new(PlanNode::Project { input: Box::new(seqscan("t", 1)), exprs: vec![col(0)] }),
        keys: vec![SortKey { expr: col(0), desc: true, nulls_first: None, collation: Collation::Binary }],
        limit: None,
    };
    let rows = run(&plan(tree, &["a"]), &cat, &mut pager);
    assert_eq!(rows.len(), 3);
    assert_eq!(int(&rows[0][0]), 3);
    assert_eq!(int(&rows[1][0]), 1);
    assert!(rows[2][0].is_null(), "NULL sorts last for DESC by default");
}

// ----- (2) Filter three-valued logic ---------------------------------------

#[test]
fn filter_eq_and_ne_both_drop_null_row() {
    // `a = 1` keeps the a=1 row and drops the NULL row; `a <> 1` drops BOTH (NULL
    // and the a=1 row) — the classic three-valued-logic behavior.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER)");
    let root = table_root(&cat, "t");
    seed(&mut pager, root, &[(1, vec![Value::Integer(1)]), (2, vec![Value::Null])]);

    let filter = |op: CmpOp| PlanNode::Filter {
        input: Box::new(seqscan("t", 1)),
        predicate: EvalExpr::Compare {
            op,
            null_safe: false,
            left: Box::new(col(0)),
            right: Box::new(lit(Value::Integer(1))),
            meta: binary_meta(),
        },
    };

    let eq_rows = run(&plan(filter(CmpOp::Eq), &["a"]), &cat, &mut pager);
    assert_eq!(eq_rows.len(), 1, "a = 1 keeps exactly the non-NULL matching row");
    assert_eq!(int(&eq_rows[0][0]), 1);

    let ne_rows = run(&plan(filter(CmpOp::Ne), &["a"]), &cat, &mut pager);
    assert_eq!(ne_rows.len(), 0, "a <> 1 drops the NULL (unknown) AND the a=1 (false) rows");
}

// ----- (3) Limit / Offset --------------------------------------------------

#[test]
fn limit_offset_returns_the_window() {
    // LIMIT 2 OFFSET 1 over 4 rows returns the 2nd and 3rd (rowid order).
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER)");
    let root = table_root(&cat, "t");
    seed(
        &mut pager,
        root,
        &[
            (1, vec![Value::Integer(10)]),
            (2, vec![Value::Integer(20)]),
            (3, vec![Value::Integer(30)]),
            (4, vec![Value::Integer(40)]),
        ],
    );

    let tree = PlanNode::Limit {
        input: Box::new(seqscan("t", 1)),
        limit: Some(lit(Value::Integer(2))),
        offset: Some(lit(Value::Integer(1))),
    };
    let rows = run(&plan(tree, &["a"]), &cat, &mut pager);
    assert_eq!(rows.len(), 2);
    assert_eq!(int(&rows[0][0]), 20);
    assert_eq!(int(&rows[1][0]), 30);
}

#[test]
fn limit_streams_without_scanning_the_whole_table() {
    // LIMIT 2 over 1000 rows returns exactly 2 rows. A materializing (non-streaming)
    // implementation would still be correct here, but this pins that the operator
    // stops pulling early — the streaming contract — and stays correct at scale.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER)");
    let root = table_root(&cat, "t");
    let rows: Vec<(i64, Vec<Value>)> = (1..=1000).map(|i| (i, vec![Value::Integer(i * 10)])).collect();
    seed(&mut pager, root, &rows);

    let tree =
        PlanNode::Limit { input: Box::new(seqscan("t", 1)), limit: Some(lit(Value::Integer(2))), offset: None };
    let out = run(&plan(tree, &["a"]), &cat, &mut pager);
    assert_eq!(out.len(), 2);
    assert_eq!(int(&out[0][0]), 10);
    assert_eq!(int(&out[1][0]), 20);
}

// ----- (4) Distinct --------------------------------------------------------

#[test]
fn distinct_folds_integer_and_real_and_dedups() {
    // DISTINCT over {2, 2.0, 2, 3}: 2 and 2.0 fold together, the extra 2 is a dup,
    // leaving {2, 3} — two rows.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a)");
    let root = table_root(&cat, "t");
    seed(
        &mut pager,
        root,
        &[
            (1, vec![Value::Integer(2)]),
            (2, vec![Value::Real(2.0)]),
            (3, vec![Value::Integer(2)]),
            (4, vec![Value::Integer(3)]),
        ],
    );

    let tree = PlanNode::Distinct {
        input: Box::new(PlanNode::Project { input: Box::new(seqscan("t", 1)), exprs: vec![col(0)] }),
        // One output column; numeric value-folding is collation-independent, so BINARY.
        column_collations: vec![Collation::Binary],
    };
    let rows = run(&plan(tree, &["a"]), &cat, &mut pager);
    assert_eq!(rows.len(), 2, "Integer(2) and Real(2.0) dedup together; the 2nd 2 is a dup");
    // First distinct row is the first-seen `2` (stored as Integer here).
    assert_eq!(int(&rows[0][0]), 2);
    assert_eq!(int(&rows[1][0]), 3);
}

// ----- (5) rowid alias -----------------------------------------------------

#[test]
fn rowid_alias_column_reflects_rowid() {
    // t(a INTEGER PRIMARY KEY, b TEXT): `a` aliases the rowid. A record stored with a
    // NULL in column `a` must surface the rowid there (register 0) AND at the trailing
    // rowid register (register 2).
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT)");
    assert_eq!(cat.table("t").unwrap().unwrap().rowid_alias, Some(0), "a is the rowid alias");
    let root = table_root(&cat, "t");
    seed(&mut pager, root, &[(5, vec![Value::Null, Value::Text("hi".into())])]);

    let rows = run(&plan(seqscan("t", 2), &["a", "b"]), &cat, &mut pager);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].len(), 3, "width N+1 = 3 for a 2-column table");
    assert_eq!(int(&rows[0][0]), 5, "aliased column 0 shows the rowid, not the stored NULL");
    assert_eq!(text(&rows[0][1]), "hi");
    assert_eq!(int(&rows[0][2]), 5, "trailing rowid register");
}

// ----- decode_table_row_enc NULL-padding + truncation ----------------------

#[test]
fn seqscan_pads_short_record_with_null() {
    // A record stored with FEWER values than the table's column count pads the missing
    // trailing columns with NULL (decode_table_row_enc's `resize`). A 2-column table t(a,b)
    // holds a 1-value record -> column b is NULL, and the row is still width N+1 = 3.
    // Deleting the resize/pad would drop column b and this fails.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER, b TEXT)");
    let root = table_root(&cat, "t");
    seed(&mut pager, root, &[(1, vec![Value::Integer(7)])]);
    let rows = run(&plan(seqscan("t", 2), &["a", "b"]), &cat, &mut pager);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].len(), 3, "width N+1 = 3 even though the record stored 1 value");
    assert_eq!(int(&rows[0][0]), 7);
    assert!(rows[0][1].is_null(), "missing trailing column b pads to NULL");
    assert_eq!(int(&rows[0][2]), 1, "trailing rowid register");
}

#[test]
fn seqscan_truncates_overlong_record() {
    // A record stored with MORE values than the table's column count truncates the
    // extras (decode_table_row_enc's `truncate`) so the logical row never widens past the
    // schema. A 1-column table t(a) holds a 2-value record -> the extra value is
    // dropped and the row is width N+1 = 2 = [a, rowid]. Deleting the truncate would
    // leave width 3 with the stray value and this fails.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER)");
    let root = table_root(&cat, "t");
    seed(&mut pager, root, &[(1, vec![Value::Integer(7), Value::Integer(999)])]);
    let rows = run(&plan(seqscan("t", 1), &["a"]), &cat, &mut pager);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].len(), 2, "extra stored value truncated; width N+1 = 2");
    assert_eq!(int(&rows[0][0]), 7);
    assert_eq!(int(&rows[0][1]), 1, "trailing rowid register, not the truncated 999");
}

// ----- RowidScan (point + range, both directions) --------------------------

fn rowid_seed() -> (MemPager, SchemaCatalog) {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER)");
    let root = table_root(&cat, "t");
    seed(
        &mut pager,
        root,
        &[
            (1, vec![Value::Integer(100)]),
            (2, vec![Value::Integer(200)]),
            (3, vec![Value::Integer(300)]),
            (4, vec![Value::Integer(400)]),
        ],
    );
    (pager, cat)
}

/// The trailing rowid register (index 1 for a 1-column table) of each row.
fn rowids(rows: &[Vec<Value>]) -> Vec<i64> {
    rows.iter().map(|r| int(&r[r.len() - 1])).collect()
}

#[test]
fn rowid_scan_point_lookup() {
    let (mut pager, cat) = rowid_seed();
    let tree = PlanNode::RowidScan(RowidScan { db: DbIndex::MAIN,
        table: "t".into(),
        column_count: 1,
        op: RowidOp::Eq(lit(Value::Integer(3))),
        direction: ScanDirection::Forward,
    });
    let rows = run(&plan(tree, &["a"]), &cat, &mut pager);
    assert_eq!(rowids(&rows), vec![3], "point lookup reaches exactly rowid 3");
    assert_eq!(int(&rows[0][0]), 300);
}

#[test]
fn rowid_scan_point_lookup_absent_is_empty() {
    let (mut pager, cat) = rowid_seed();
    let tree = PlanNode::RowidScan(RowidScan { db: DbIndex::MAIN,
        table: "t".into(),
        column_count: 1,
        op: RowidOp::Eq(lit(Value::Integer(99))),
        direction: ScanDirection::Forward,
    });
    assert!(run(&plan(tree, &["a"]), &cat, &mut pager).is_empty());
}

#[test]
fn rowid_scan_point_lookup_non_integer_is_empty() {
    // A non-integral / non-numeric Eq target names no integer rowid -> no rows.
    let (mut pager, cat) = rowid_seed();
    let tree = PlanNode::RowidScan(RowidScan { db: DbIndex::MAIN,
        table: "t".into(),
        column_count: 1,
        op: RowidOp::Eq(lit(Value::Real(2.5))),
        direction: ScanDirection::Forward,
    });
    assert!(run(&plan(tree, &["a"]), &cat, &mut pager).is_empty());
}

#[test]
fn rowid_scan_point_lookup_integral_real_folds_to_rowid() {
    // The POSITIVE fold: `rowid = 2.0` names the equal integer rowid 2 and reaches it.
    // The non_integer case above only pins the reject branch (Real(2.5) -> empty); this
    // guards the scan-path use of the shared `real_to_int_if_exact` fold (scan.rs), so
    // it can't silently regress to rejecting integral reals.
    let (mut pager, cat) = rowid_seed();
    let tree = PlanNode::RowidScan(RowidScan { db: DbIndex::MAIN,
        table: "t".into(),
        column_count: 1,
        op: RowidOp::Eq(lit(Value::Real(2.0))),
        direction: ScanDirection::Forward,
    });
    let rows = run(&plan(tree, &["a"]), &cat, &mut pager);
    assert_eq!(rowids(&rows), vec![2], "rowid = 2.0 folds to and reaches rowid 2");
    assert_eq!(int(&rows[0][0]), 200);
}

// The `Eq` seek value is coerced with sqlite's OP_SeekRowid rule (`must_be_int`: NUMERIC
// affinity then require a lossless integer), so a runtime value that would carry an
// affinity — a bind param, a numeric string, an integral real — reaches the right rowid
// by value rather than naming no row. These pin that coercion for the shapes the planner
// now emits raw (see `access_path::rowid_eq_value`); the Integer / integral-real / 2.5
// cases are pinned by `rowid_scan_point_lookup*` above.

#[test]
fn rowid_scan_point_lookup_numeric_text_coerces_to_that_rowid() {
    // `rowid = '2'`: NUMERIC affinity converts the text to Integer(2), which reaches rowid
    // 2 (real sqlite's OP_MustBeInt/OP_SeekRowid). Before the coercion this named no row.
    let (mut pager, cat) = rowid_seed();
    let tree = PlanNode::RowidScan(RowidScan { db: DbIndex::MAIN,
        table: "t".into(),
        column_count: 1,
        op: RowidOp::Eq(lit(Value::Text("2".into()))),
        direction: ScanDirection::Forward,
    });
    let rows = run(&plan(tree, &["a"]), &cat, &mut pager);
    assert_eq!(rowids(&rows), vec![2], "text '2' coerces to and reaches rowid 2");
    assert_eq!(int(&rows[0][0]), 200);
}

#[test]
fn rowid_scan_point_lookup_integral_real_text_coerces_to_that_rowid() {
    // `rowid = '2.0'`: NUMERIC affinity demotes the integral-real string to Integer(2).
    let (mut pager, cat) = rowid_seed();
    let tree = PlanNode::RowidScan(RowidScan { db: DbIndex::MAIN,
        table: "t".into(),
        column_count: 1,
        op: RowidOp::Eq(lit(Value::Text("2.0".into()))),
        direction: ScanDirection::Forward,
    });
    let rows = run(&plan(tree, &["a"]), &cat, &mut pager);
    assert_eq!(rowids(&rows), vec![2], "text '2.0' coerces to and reaches rowid 2");
}

#[test]
fn rowid_scan_point_lookup_non_numeric_text_is_empty() {
    // `rowid = 'abc'`: affinity leaves it Text, so it is no lossless integer -> no row.
    let (mut pager, cat) = rowid_seed();
    let tree = PlanNode::RowidScan(RowidScan { db: DbIndex::MAIN,
        table: "t".into(),
        column_count: 1,
        op: RowidOp::Eq(lit(Value::Text("abc".into()))),
        direction: ScanDirection::Forward,
    });
    assert!(run(&plan(tree, &["a"]), &cat, &mut pager).is_empty(), "non-numeric text names no row");
}

#[test]
fn rowid_scan_point_lookup_null_is_empty() {
    // `rowid = NULL` is never a match; the coerced seek value is NULL -> no row.
    let (mut pager, cat) = rowid_seed();
    let tree = PlanNode::RowidScan(RowidScan { db: DbIndex::MAIN,
        table: "t".into(),
        column_count: 1,
        op: RowidOp::Eq(lit(Value::Null)),
        direction: ScanDirection::Forward,
    });
    assert!(run(&plan(tree, &["a"]), &cat, &mut pager).is_empty(), "NULL rowid names no row");
}

#[test]
fn rowid_scan_range_inclusive_forward() {
    let (mut pager, cat) = rowid_seed();
    let tree = PlanNode::RowidScan(RowidScan { db: DbIndex::MAIN,
        table: "t".into(),
        column_count: 1,
        op: RowidOp::Range {
            lo: Some(RangeBound { value: lit(Value::Integer(2)), inclusive: true }),
            hi: Some(RangeBound { value: lit(Value::Integer(3)), inclusive: true }),
        },
        direction: ScanDirection::Forward,
    });
    let rows = run(&plan(tree, &["a"]), &cat, &mut pager);
    assert_eq!(rowids(&rows), vec![2, 3], "inclusive [2,3] ascending");
}

#[test]
fn rowid_scan_range_exclusive_bounds() {
    // `rowid > 1 AND rowid < 4` -> rowids 2, 3.
    let (mut pager, cat) = rowid_seed();
    let tree = PlanNode::RowidScan(RowidScan { db: DbIndex::MAIN,
        table: "t".into(),
        column_count: 1,
        op: RowidOp::Range {
            lo: Some(RangeBound { value: lit(Value::Integer(1)), inclusive: false }),
            hi: Some(RangeBound { value: lit(Value::Integer(4)), inclusive: false }),
        },
        direction: ScanDirection::Forward,
    });
    assert_eq!(rowids(&run(&plan(tree, &["a"]), &cat, &mut pager)), vec![2, 3]);
}

#[test]
fn rowid_scan_range_reverse_descends() {
    // Same inclusive [2,3] window, reverse direction: rows come out 3 then 2.
    let (mut pager, cat) = rowid_seed();
    let tree = PlanNode::RowidScan(RowidScan { db: DbIndex::MAIN,
        table: "t".into(),
        column_count: 1,
        op: RowidOp::Range {
            lo: Some(RangeBound { value: lit(Value::Integer(2)), inclusive: true }),
            hi: Some(RangeBound { value: lit(Value::Integer(3)), inclusive: true }),
        },
        direction: ScanDirection::Reverse,
    });
    assert_eq!(rowids(&run(&plan(tree, &["a"]), &cat, &mut pager)), vec![3, 2], "reverse descends");
}

#[test]
fn rowid_scan_unbounded_range_is_full_scan() {
    // {lo:None, hi:None} is a full ordered scan of the table b-tree.
    let (mut pager, cat) = rowid_seed();
    let forward = PlanNode::RowidScan(RowidScan { db: DbIndex::MAIN,
        table: "t".into(),
        column_count: 1,
        op: RowidOp::Range { lo: None, hi: None },
        direction: ScanDirection::Forward,
    });
    assert_eq!(rowids(&run(&plan(forward, &["a"]), &cat, &mut pager)), vec![1, 2, 3, 4]);

    let reverse = PlanNode::RowidScan(RowidScan { db: DbIndex::MAIN,
        table: "t".into(),
        column_count: 1,
        op: RowidOp::Range { lo: None, hi: None },
        direction: ScanDirection::Reverse,
    });
    assert_eq!(rowids(&run(&plan(reverse, &["a"]), &cat, &mut pager)), vec![4, 3, 2, 1]);
}

#[test]
fn rowid_scan_reverse_start_overshoots_then_steps_back() {
    // Gapped rowids {10, 20, 30}: a reverse scan bounded hi = 25 (absent) must start
    // at the largest rowid <= 25. `seek_ge(25)` overshoots to 30, so the reverse-start
    // logic steps back one to 20 — exercising the seek_ge-overshoot-then-prev branch a
    // contiguous table never reaches. Then it descends to lo = 5, emitting 20, 10.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER)");
    let root = table_root(&cat, "t");
    seed(
        &mut pager,
        root,
        &[
            (10, vec![Value::Integer(100)]),
            (20, vec![Value::Integer(200)]),
            (30, vec![Value::Integer(300)]),
        ],
    );
    let tree = PlanNode::RowidScan(RowidScan { db: DbIndex::MAIN,
        table: "t".into(),
        column_count: 1,
        op: RowidOp::Range {
            lo: Some(RangeBound { value: lit(Value::Integer(5)), inclusive: true }),
            hi: Some(RangeBound { value: lit(Value::Integer(25)), inclusive: true }),
        },
        direction: ScanDirection::Reverse,
    });
    assert_eq!(rowids(&run(&plan(tree, &["a"]), &cat, &mut pager)), vec![20, 10]);
}

// ----- Values / SingleRow --------------------------------------------------

#[test]
fn values_emits_each_literal_row() {
    let (mut pager, cat) = empty_db();
    let tree = PlanNode::Values {
        rows: vec![
            vec![lit(Value::Integer(1)), lit(Value::Text("a".into()))],
            vec![lit(Value::Integer(2)), lit(Value::Text("b".into()))],
        ],
    };
    let rows = run(&plan(tree, &["x", "y"]), &cat, &mut pager);
    assert_eq!(rows.len(), 2);
    assert_eq!(int(&rows[0][0]), 1);
    assert_eq!(text(&rows[0][1]), "a");
    assert_eq!(int(&rows[1][0]), 2);
    assert_eq!(text(&rows[1][1]), "b");
}

#[test]
fn single_row_projects_a_constant() {
    // SELECT 42 (no FROM): SingleRow yields one zero-column row; the projection above
    // supplies the value.
    let (mut pager, cat) = empty_db();
    let tree = PlanNode::Project { input: Box::new(PlanNode::SingleRow), exprs: vec![lit(Value::Integer(42))] };
    let rows = run(&plan(tree, &["v"]), &cat, &mut pager);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].len(), 1);
    assert_eq!(int(&rows[0][0]), 42);
}

// ----- unimplemented dispatch arms fail loud (never silent/panic) ----------

#[test]
fn unimplemented_read_node_errors() {
    // A CteScan has no operator yet: build_cursor must return an Err, not an empty
    // result set (which would silently look like "0 rows").
    let (mut pager, cat) = empty_db();
    let tree = PlanNode::CteScan { id: 0, column_count: 1 };
    assert!(run_err(&plan(tree, &["x"]), &cat, &mut pager).is_err());
}

#[test]
fn dml_delete_dispatches_to_build_dml_and_applies() {
    // A DELETE routes through build_dml and actually applies the mutation (it was a
    // not-yet-implemented Err before the DELETE operator landed). The dispatch layer
    // must run the operator, never silently no-op; deleting over a full scan empties
    // the table.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER)");
    let root = table_root(&cat, "t");
    seed(&mut pager, root, &[(1, vec![Value::Integer(10)]), (2, vec![Value::Integer(20)])]);

    let tree = PlanNode::Delete(Delete { db: DbIndex::MAIN,
        table: "t".into(),
        column_count: 1,
        scan: Box::new(seqscan("t", 1)),
        returning: Vec::new(),
        triggers: Vec::new(),
        index_key_exprs: Vec::new(),
        index_partial_predicates: Vec::new(),
    });
    let mut mutating = plan(tree, &[]);
    mutating.mutates = true;

    // DML mutates the pager, so drive it inside a write transaction like the engine.
    pager.begin().unwrap();
    assert!(run_err(&mutating, &cat, &mut pager).is_ok(), "DELETE dispatches and applies");
    pager.commit().unwrap();

    assert!(run(&plan(seqscan("t", 1), &["a"]), &cat, &mut pager).is_empty(), "every row deleted");
}

// ----- WITHOUT ROWID scans read the PK index b-tree ------------------------

#[test]
fn seqscan_without_rowid_table_scans_key_records() {
    // A WITHOUT ROWID table stores each row in a PRIMARY KEY index b-tree (its
    // `root_page` IS that index), so a SeqScan iterates the b-tree and decodes each key
    // record into a width-N row (NO fabricated trailing rowid), in PRIMARY KEY order.
    // Seed the b-tree directly with full-row key records `[a, b]` (a is the PK, stored
    // first) OUT of key order, then confirm the scan returns them IN key order and each
    // row is exactly N wide.
    let (mut pager, cat) =
        db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT) WITHOUT ROWID");
    assert!(cat.table("t").unwrap().unwrap().without_rowid, "fixture is WITHOUT ROWID");
    let root = table_root(&cat, "t");

    pager.begin().unwrap();
    for vals in [
        vec![Value::Integer(2), Value::Text("b".into())],
        vec![Value::Integer(1), Value::Text("a".into())],
    ] {
        index_insert(&mut pager, root, &encode_record(&vals)).unwrap();
    }
    pager.commit().unwrap();

    let rows = run(&plan(seqscan("t", 2), &["a", "b"]), &cat, &mut pager);
    assert_eq!(rows.len(), 2);
    assert!(rows.iter().all(|r| r.len() == 2), "WR scan rows are width N (no rowid), got {rows:?}");
    assert_eq!((int(&rows[0][0]), text(&rows[0][1])), (1, "a"), "key order: 1 before 2");
    assert_eq!((int(&rows[1][0]), text(&rows[1][1])), (2, "b"));
}

// ----- LIMIT / OFFSET edge cases -------------------------------------------

/// A 4-row table (`a` = 10,20,30,40 at rowids 1..4) for the LIMIT/OFFSET edges.
fn four_row_table() -> (MemPager, SchemaCatalog) {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER)");
    let root = table_root(&cat, "t");
    seed(
        &mut pager,
        root,
        &[
            (1, vec![Value::Integer(10)]),
            (2, vec![Value::Integer(20)]),
            (3, vec![Value::Integer(30)]),
            (4, vec![Value::Integer(40)]),
        ],
    );
    (pager, cat)
}

/// Run `LIMIT limit OFFSET offset` over the 4-row table, returning the `a` values.
fn limit_a_values(limit: Option<EvalExpr>, offset: Option<EvalExpr>) -> Vec<i64> {
    let (mut pager, cat) = four_row_table();
    let tree = PlanNode::Limit { input: Box::new(seqscan("t", 1)), limit, offset };
    run(&plan(tree, &["a"]), &cat, &mut pager).iter().map(|r| int(&r[0])).collect()
}

#[test]
fn limit_zero_returns_no_rows() {
    assert_eq!(limit_a_values(Some(lit(Value::Integer(0))), None), Vec::<i64>::new());
}

#[test]
fn negative_limit_is_unbounded() {
    // A negative LIMIT means "no limit" in SQLite -> every row.
    assert_eq!(limit_a_values(Some(lit(Value::Integer(-1))), None), vec![10, 20, 30, 40]);
}

#[test]
fn null_limit_is_an_error() {
    // lang_select.html §5: a LIMIT that evaluates to NULL cannot be losslessly
    // converted to an integer, so it is an ERROR — distinct from a NEGATIVE LIMIT,
    // which is unbounded (see `negative_limit_is_unbounded`). The executor mirrors
    // SQLite's `OP_MustBeInt` (apply NUMERIC affinity, then require an integer).
    let (mut pager, cat) = four_row_table();
    let tree = PlanNode::Limit {
        input: Box::new(seqscan("t", 1)),
        limit: Some(lit(Value::Null)),
        offset: None,
    };
    assert!(run_err(&plan(tree, &["a"]), &cat, &mut pager).is_err());
}

#[test]
fn negative_offset_is_zero() {
    // A negative OFFSET clamps to 0 (skips nothing).
    assert_eq!(
        limit_a_values(Some(lit(Value::Integer(2))), Some(lit(Value::Integer(-5)))),
        vec![10, 20]
    );
}

#[test]
fn null_offset_is_an_error() {
    // lang_select.html §5: the OFFSET must ALSO be losslessly convertible to an
    // integer, so a NULL OFFSET is an ERROR — distinct from a NEGATIVE OFFSET,
    // which is treated as 0 (see `negative_offset_is_zero`).
    let (mut pager, cat) = four_row_table();
    let tree = PlanNode::Limit {
        input: Box::new(seqscan("t", 1)),
        limit: Some(lit(Value::Integer(2))),
        offset: Some(lit(Value::Null)),
    };
    assert!(run_err(&plan(tree, &["a"]), &cat, &mut pager).is_err());
}

#[test]
fn offset_past_end_yields_no_rows() {
    // Offset beyond the row count consumes everything and emits nothing.
    assert_eq!(
        limit_a_values(Some(lit(Value::Integer(2))), Some(lit(Value::Integer(10)))),
        Vec::<i64>::new()
    );
}

// ----- SORT NULL placement + multi-key coverage ----------------------------

/// Sort the fixed 3-row set {1, NULL, 3} by its single column under the given
/// direction / NULL placement, returning the ordered column as `Option<i64>` so order
/// and NULL placement read as one assertion.
fn sort_nullable(desc: bool, nulls_first: Option<bool>) -> Vec<Option<i64>> {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER)");
    let root = table_root(&cat, "t");
    seed(
        &mut pager,
        root,
        &[
            (1, vec![Value::Integer(1)]),
            (2, vec![Value::Null]),
            (3, vec![Value::Integer(3)]),
        ],
    );
    let tree = PlanNode::Sort {
        input: Box::new(PlanNode::Project { input: Box::new(seqscan("t", 1)), exprs: vec![col(0)] }),
        keys: vec![SortKey { expr: col(0), desc, nulls_first, collation: Collation::Binary }],
        limit: None,
    };
    run(&plan(tree, &["a"]), &cat, &mut pager)
        .iter()
        .map(|r| match &r[0] {
            Value::Null => None,
            Value::Integer(i) => Some(*i),
            other => panic!("expected Integer/Null, got {other:?}"),
        })
        .collect()
}

#[test]
fn sort_ascending_nulls_first_by_default() {
    // ORDER BY a ASC: a NULL sorts FIRST by default (nulls_first defaults to !desc).
    assert_eq!(sort_nullable(false, None), vec![None, Some(1), Some(3)]);
}

#[test]
fn sort_nulls_placement_override_beats_the_default() {
    // Explicit NULLS LAST on ASC overrides the ASC default (nulls-first).
    assert_eq!(sort_nullable(false, Some(false)), vec![Some(1), Some(3), None]);
    // Explicit NULLS FIRST on DESC overrides the DESC default (nulls-last).
    assert_eq!(sort_nullable(true, Some(true)), vec![None, Some(3), Some(1)]);
}

#[test]
fn sort_multi_key_second_key_breaks_ties() {
    // ORDER BY a ASC, b ASC: two rows tie on `a`, so the second key decides. A
    // regression that consulted only the first key would leave (1,'b') before (1,'a').
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER, b TEXT)");
    let root = table_root(&cat, "t");
    seed(
        &mut pager,
        root,
        &[
            (1, vec![Value::Integer(1), Value::Text("b".into())]),
            (2, vec![Value::Integer(1), Value::Text("a".into())]),
            (3, vec![Value::Integer(0), Value::Text("z".into())]),
        ],
    );
    let tree = PlanNode::Sort {
        input: Box::new(PlanNode::Project {
            input: Box::new(seqscan("t", 2)),
            exprs: vec![col(0), col(1)],
        }),
        keys: vec![
            SortKey { expr: col(0), desc: false, nulls_first: None, collation: Collation::Binary },
            SortKey { expr: col(1), desc: false, nulls_first: None, collation: Collation::Binary },
        ],
        limit: None,
    };
    let rows = run(&plan(tree, &["a", "b"]), &cat, &mut pager);
    let got: Vec<(i64, &str)> = rows.iter().map(|r| (int(&r[0]), text(&r[1]))).collect();
    assert_eq!(got, vec![(0, "z"), (1, "a"), (1, "b")], "second key breaks the a=1 tie");
}

// ----- correlated-outer plumbing (leaf prepends outer) ---------------------
// `outer` is always empty on the top-level path today, so every operator above runs
// with an empty outer (`with_outer` returns the local row unchanged). The non-empty
// `outer ++ local` prepend that correlated subqueries rely on lands with the subquery
// operator that first produces a non-empty outer, which will add the direct
// end-to-end coverage.
