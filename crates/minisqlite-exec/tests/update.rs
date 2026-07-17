//! Integration tests for the `UPDATE` DML operator: hand-built [`Plan`] trees run
//! over a real [`MemPager`] + [`SchemaCatalog`] through the [`Executor`]/[`RowCursor`]
//! seam, then read back through the real storage/catalog (not a private mock). They
//! pin the operator end to end: assignment against the OLD row, affinity on store,
//! filtered updates, index maintenance (in-place AND rowid-changing), `ON CONFLICT`,
//! `RETURNING`, and the `Runtime` change counter.
//!
//! The operator assumes an open write transaction (the engine opens one for DML), so
//! each apply is wrapped in `begin`/`commit`; read-back scans need no transaction.

use minisqlite_btree::{init_database, IndexCursor};
use minisqlite_catalog::{Catalog, SchemaCatalog};
use minisqlite_exec::{Executor, Runtime, StreamingExecutor};
use minisqlite_expr::{ArithOp, CmpOp, CompareMeta, EvalExpr};
use minisqlite_pager::{MemPager, Pager};
use minisqlite_plan::{
    CheckConstraint, Insert, IndexOp, IndexScan, OnConflict, Plan, PlanNode, ScanDirection, SeqScan,
    Update,
};
use minisqlite_sql::{parse, Statement};
use minisqlite_types::{affinity_of_declared_type, Affinity, Collation, Error, Value};
use minisqlite_exec::PagerSet;
use minisqlite_types::DbIndex;

// ----- fixtures ------------------------------------------------------------

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

fn create_index(pager: &mut MemPager, cat: &mut SchemaCatalog, sql: &str) {
    let ast = parse(sql).unwrap();
    let Statement::CreateIndex(stmt) = &ast.statements[0] else {
        panic!("not a CREATE INDEX: {sql}");
    };
    cat.create_index(pager, stmt, sql).unwrap();
}

fn affinities(cat: &SchemaCatalog, table: &str) -> Vec<Affinity> {
    cat.table(table)
        .unwrap()
        .unwrap()
        .columns
        .iter()
        .map(|c| affinity_of_declared_type(c.declared_type.as_deref()))
        .collect()
}

fn col(i: usize) -> EvalExpr {
    EvalExpr::Column(i)
}

fn lit(v: Value) -> EvalExpr {
    EvalExpr::Literal(v)
}

fn arith(op: ArithOp, l: EvalExpr, r: EvalExpr) -> EvalExpr {
    EvalExpr::Arith { op, left: Box::new(l), right: Box::new(r) }
}

/// `left <op> right` with plain binary collation and no operand affinity.
fn cmp(op: CmpOp, left: EvalExpr, right: EvalExpr) -> EvalExpr {
    EvalExpr::Compare {
        op,
        null_safe: false,
        left: Box::new(left),
        right: Box::new(right),
        meta: CompareMeta { apply_left: None, apply_right: None, collation: Collation::Binary },
    }
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

fn seqscan(table: &str, n: usize) -> PlanNode {
    PlanNode::SeqScan(SeqScan { db: DbIndex::MAIN, table: table.to_string(), column_count: n })
}

/// Build an `UPDATE` plan over the given scan subtree.
fn update_plan(
    table: &str,
    n: usize,
    assignments: Vec<(usize, EvalExpr)>,
    scan: PlanNode,
    column_affinities: Vec<Affinity>,
    on_conflict: OnConflict,
    returning: Vec<EvalExpr>,
) -> Plan {
    Plan {
        root: PlanNode::Update(Update { db: DbIndex::MAIN,
            table: table.to_string(),
            column_count: n,
            assignments,
            scan: Box::new(scan),
            column_affinities,
            on_conflict,
            returning,
            triggers: Vec::new(),
            checks: Vec::new(),
            index_key_exprs: Vec::new(),
            index_partial_predicates: Vec::new(),
        }),
        result_columns: Vec::new(),
        ctes: Vec::new(),
        subqueries: Vec::new(),
        mutates: true,
        generated: Vec::new(),
    }
}

/// Seed rows through the real `INSERT` operator (positional, auto rowids unless an
/// explicit alias value is given), so tests exercise the same write path they verify.
fn seed(cat: &SchemaCatalog, pager: &mut MemPager, table: &str, n: usize, rows: Vec<Vec<EvalExpr>>) {
    let plan = Plan {
        root: PlanNode::Insert(Insert { db: DbIndex::MAIN,
            table: table.to_string(),
            column_count: n,
            columns: None,
            source: Box::new(PlanNode::Values { rows }),
            source_width: n,
            column_affinities: affinities(cat, table),
            on_conflict: OnConflict::Abort,
            returning: Vec::new(),
            triggers: Vec::new(),
            checks: Vec::new(),
            rowid_source: None,
            upsert: None,
            index_key_exprs: Vec::new(),
            index_partial_predicates: Vec::new(),
        }),
        result_columns: Vec::new(),
        ctes: Vec::new(),
        subqueries: Vec::new(),
        mutates: true,
        generated: Vec::new(),
    };
    let mut rt = Runtime::new();
    run(&plan, cat, pager, &mut rt);
}

/// Apply a mutating plan inside a write transaction, draining any `RETURNING` rows and
/// threading the caller's `Runtime` so the change counters can be read afterward.
fn run(plan: &Plan, cat: &SchemaCatalog, pager: &mut MemPager, rt: &mut Runtime) -> Vec<Vec<Value>> {
    pager.begin().unwrap();
    let mut exec = StreamingExecutor;
    let mut out = Vec::new();
    {
        let mut cur = exec.execute(plan, cat, PagerSet::One { db: DbIndex::MAIN, pager: &mut *pager }).unwrap();
        while let Some(row) = cur.next_row(rt).unwrap() {
            out.push(row);
        }
    }
    pager.commit().unwrap();
    out
}

/// Like [`run`] but for the error path: rolls the failed statement back (as the engine
/// would on a constraint violation) and returns the error.
fn run_err(plan: &Plan, cat: &SchemaCatalog, pager: &mut MemPager) -> Error {
    let mut rt = Runtime::new();
    pager.begin().unwrap();
    let mut exec = StreamingExecutor;
    let mut err = None;
    {
        match exec.execute(plan, cat, PagerSet::One { db: DbIndex::MAIN, pager: &mut *pager }) {
            Ok(mut cur) => loop {
                match cur.next_row(&mut rt) {
                    Ok(Some(_)) => {}
                    Ok(None) => break,
                    Err(e) => {
                        err = Some(e);
                        break;
                    }
                }
            },
            Err(e) => err = Some(e),
        }
    }
    pager.rollback().unwrap();
    err.expect("expected the UPDATE to error")
}

/// Full table scan back through the executor. Rows are `[c0, …, c_{N-1}, rowid]`,
/// ordered by ascending rowid.
fn scan_all(cat: &SchemaCatalog, pager: &mut MemPager, table: &str, n: usize) -> Vec<Vec<Value>> {
    let plan = Plan {
        root: seqscan(table, n),
        result_columns: Vec::new(),
        ctes: Vec::new(),
        subqueries: Vec::new(),
        mutates: false,
        generated: Vec::new(),
    };
    let mut rt = Runtime::new();
    let mut exec = StreamingExecutor;
    let mut cur = exec.execute(&plan, cat, PagerSet::One { db: DbIndex::MAIN, pager: &mut *pager }).unwrap();
    let mut out = Vec::new();
    while let Some(row) = cur.next_row(&mut rt).unwrap() {
        out.push(row);
    }
    out
}

/// Read a table row via an index equality seek (`index = value`), returning the
/// `[cols.., rowid]` rows the seek reaches. Proves the index entry exists and points
/// at the right table row.
fn index_eq_rows(
    cat: &SchemaCatalog,
    pager: &mut MemPager,
    table: &str,
    n: usize,
    index: &str,
    value: Value,
) -> Vec<Vec<Value>> {
    let plan = Plan {
        root: PlanNode::IndexScan(IndexScan { db: DbIndex::MAIN,
            table: table.to_string(),
            column_count: n,
            index: index.to_string(),
            op: IndexOp::Seek { eq_prefix: vec![lit(value)], low: None, high: None },
            direction: ScanDirection::Forward,
            covering: false,
        }),
        result_columns: Vec::new(),
        ctes: Vec::new(),
        subqueries: Vec::new(),
        mutates: false,
        generated: Vec::new(),
    };
    let mut rt = Runtime::new();
    let mut exec = StreamingExecutor;
    let mut cur = exec.execute(&plan, cat, PagerSet::One { db: DbIndex::MAIN, pager: &mut *pager }).unwrap();
    let mut out = Vec::new();
    while let Some(row) = cur.next_row(&mut rt).unwrap() {
        out.push(row);
    }
    out
}

// ----- (1) simple SET over a full scan --------------------------------------

#[test]
fn simple_set_updates_every_row_and_counts_changes() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER, b TEXT)");
    seed(
        &cat,
        &mut pager,
        "t",
        2,
        vec![
            vec![lit(Value::Integer(10)), lit(Value::Text("x".into()))],
            vec![lit(Value::Integer(20)), lit(Value::Text("y".into()))],
        ],
    );
    let mut rt = Runtime::new();
    // UPDATE t SET a = 99
    let plan = update_plan(
        "t",
        2,
        vec![(0, lit(Value::Integer(99)))],
        seqscan("t", 2),
        affinities(&cat, "t"),
        OnConflict::Abort,
        Vec::new(),
    );
    let out = run(&plan, &cat, &mut pager, &mut rt);
    assert!(out.is_empty(), "no RETURNING clause yields no rows");
    assert_eq!(rt.changes(), 2, "both rows updated");

    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 2);
    // a set to 99 on both; b and rowid unchanged.
    assert_eq!((int(&rows[0][0]), text(&rows[0][1]), int(&rows[0][2])), (99, "x", 1));
    assert_eq!((int(&rows[1][0]), text(&rows[1][1]), int(&rows[1][2])), (99, "y", 2));
}

// ----- (2) assignments read the PRE-UPDATE row ------------------------------

#[test]
fn set_a_equals_a_plus_one_increments() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER)");
    seed(&cat, &mut pager, "t", 1, vec![vec![lit(Value::Integer(5))], vec![lit(Value::Integer(41))]]);
    let mut rt = Runtime::new();
    // UPDATE t SET a = a + 1
    let plan = update_plan(
        "t",
        1,
        vec![(0, arith(ArithOp::Add, col(0), lit(Value::Integer(1))))],
        seqscan("t", 1),
        affinities(&cat, "t"),
        OnConflict::Abort,
        Vec::new(),
    );
    run(&plan, &cat, &mut pager, &mut rt);

    let rows = scan_all(&cat, &mut pager, "t", 1);
    assert_eq!((int(&rows[0][0]), int(&rows[1][0])), (6, 42));
}

#[test]
fn multi_assignment_uses_pre_update_values_for_all() {
    // UPDATE t SET a = a + 1, b = a  — SQLite uses the PRE-update `a` for BOTH, so `b`
    // becomes the OLD `a` (stored as text under b's TEXT affinity), not `a + 1`.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER, b TEXT)");
    seed(&cat, &mut pager, "t", 2, vec![vec![lit(Value::Integer(7)), lit(Value::Text("orig".into()))]]);
    let mut rt = Runtime::new();
    let plan = update_plan(
        "t",
        2,
        vec![(0, arith(ArithOp::Add, col(0), lit(Value::Integer(1)))), (1, col(0))],
        seqscan("t", 2),
        affinities(&cat, "t"),
        OnConflict::Abort,
        Vec::new(),
    );
    run(&plan, &cat, &mut pager, &mut rt);

    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1);
    assert_eq!(int(&rows[0][0]), 8, "a = old a + 1");
    // b = old a (7) coerced to TEXT affinity → "7", NOT the new a (8).
    assert_eq!(text(&rows[0][1]), "7", "b took the PRE-update a, as text");
}

// ----- (3) affinity on the assigned value -----------------------------------

#[test]
fn assigned_text_gets_integer_affinity() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER)");
    seed(&cat, &mut pager, "t", 1, vec![vec![lit(Value::Integer(1))]]);
    let mut rt = Runtime::new();
    // UPDATE t SET a = '123' — INTEGER affinity coerces the text to Integer(123).
    let plan = update_plan(
        "t",
        1,
        vec![(0, lit(Value::Text("123".into())))],
        seqscan("t", 1),
        affinities(&cat, "t"),
        OnConflict::Abort,
        Vec::new(),
    );
    run(&plan, &cat, &mut pager, &mut rt);

    let rows = scan_all(&cat, &mut pager, "t", 1);
    assert!(
        matches!(rows[0][0], Value::Integer(123)),
        "Text \"123\" coerced to Integer(123) by INTEGER affinity, got {:?}",
        rows[0][0]
    );
}

// ----- (4) filtered UPDATE only touches matching rows -----------------------

#[test]
fn filtered_update_changes_only_matching_rows() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER, b INTEGER)");
    seed(
        &cat,
        &mut pager,
        "t",
        2,
        vec![
            vec![lit(Value::Integer(1)), lit(Value::Integer(100))],
            vec![lit(Value::Integer(2)), lit(Value::Integer(200))],
            vec![lit(Value::Integer(1)), lit(Value::Integer(300))],
        ],
    );
    let mut rt = Runtime::new();
    // UPDATE t SET b = 0 WHERE a = 1  — Filter(a == 1) over the scan.
    let scan = PlanNode::Filter {
        input: Box::new(seqscan("t", 2)),
        predicate: cmp(CmpOp::Eq, col(0), lit(Value::Integer(1))),
    };
    let plan = update_plan(
        "t",
        2,
        vec![(1, lit(Value::Integer(0)))],
        scan,
        affinities(&cat, "t"),
        OnConflict::Abort,
        Vec::new(),
    );
    run(&plan, &cat, &mut pager, &mut rt);
    assert_eq!(rt.changes(), 2, "the two a=1 rows matched");

    let rows = scan_all(&cat, &mut pager, "t", 2);
    // rowid 1 (a=1) → b=0; rowid 2 (a=2) → b unchanged 200; rowid 3 (a=1) → b=0.
    assert_eq!((int(&rows[0][0]), int(&rows[0][1])), (1, 0));
    assert_eq!((int(&rows[1][0]), int(&rows[1][1])), (2, 200));
    assert_eq!((int(&rows[2][0]), int(&rows[2][1])), (1, 0));
}

// ----- (5) index maintenance, no rowid change -------------------------------

#[test]
fn index_entry_moves_when_indexed_column_changes() {
    let (mut pager, mut cat) = db_with_table("CREATE TABLE t(a INTEGER, b TEXT)");
    create_index(&mut pager, &mut cat, "CREATE INDEX idx_a ON t(a)");
    seed(&cat, &mut pager, "t", 2, vec![vec![lit(Value::Integer(5)), lit(Value::Text("row".into()))]]);
    let mut rt = Runtime::new();
    // UPDATE t SET a = 7
    let plan = update_plan(
        "t",
        2,
        vec![(0, lit(Value::Integer(7)))],
        seqscan("t", 2),
        affinities(&cat, "t"),
        OnConflict::Abort,
        Vec::new(),
    );
    run(&plan, &cat, &mut pager, &mut rt);

    // The OLD index key (a=5) is gone; the NEW key (a=7) reaches the row.
    let old = index_eq_rows(&cat, &mut pager, "t", 2, "idx_a", Value::Integer(5));
    assert!(old.is_empty(), "old index entry a=5 was removed");
    let new = index_eq_rows(&cat, &mut pager, "t", 2, "idx_a", Value::Integer(7));
    assert_eq!(new.len(), 1, "new index entry a=7 reaches the row");
    assert_eq!((int(&new[0][0]), text(&new[0][1]), int(&new[0][2])), (7, "row", 1));
}

/// The UPDATE's own scan SOURCE is a full `IndexScan` over the SAME index it rewrites.
/// Phase-1 buffering (drain the entire scan before any write) is load-bearing here: if
/// the operator streamed the index cursor while mutating that index, it would
/// re-encounter its own freshly-inserted entries (`a` 1→11 lands ahead of the forward
/// cursor) and reprocess them — a runaway that double-counts or loops. Buffering pins
/// the pre-update row set, so each original row is updated EXACTLY once. Without the
/// buffer this test's `changes() == 3` and value assertions fail.
#[test]
fn update_over_indexscan_on_the_rewritten_index_updates_each_row_once() {
    let (mut pager, mut cat) = db_with_table("CREATE TABLE t(a INTEGER, b TEXT)");
    create_index(&mut pager, &mut cat, "CREATE INDEX idx_a ON t(a)");
    seed(
        &cat,
        &mut pager,
        "t",
        2,
        vec![
            vec![lit(Value::Integer(1)), lit(Value::Text("x".into()))],
            vec![lit(Value::Integer(2)), lit(Value::Text("y".into()))],
            vec![lit(Value::Integer(3)), lit(Value::Text("z".into()))],
        ],
    );
    let mut rt = Runtime::new();
    // UPDATE t SET a = a + 10, driven by a FULL scan of idx_a (the index being rewritten).
    let scan = PlanNode::IndexScan(IndexScan { db: DbIndex::MAIN,
        table: "t".into(),
        column_count: 2,
        index: "idx_a".into(),
        op: IndexOp::FullScan,
        direction: ScanDirection::Forward,
        covering: false,
    });
    let plan = update_plan(
        "t",
        2,
        vec![(0, arith(ArithOp::Add, col(0), lit(Value::Integer(10))))],
        scan,
        affinities(&cat, "t"),
        OnConflict::Abort,
        Vec::new(),
    );
    run(&plan, &cat, &mut pager, &mut rt);
    // Exactly three rows changed — a reprocessed row would over-count.
    assert_eq!(rt.changes(), 3, "each pre-update row updated exactly once");

    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 3, "still three rows, none duplicated or dropped");
    assert_eq!((int(&rows[0][0]), text(&rows[0][1]), int(&rows[0][2])), (11, "x", 1));
    assert_eq!((int(&rows[1][0]), text(&rows[1][1]), int(&rows[1][2])), (12, "y", 2));
    assert_eq!((int(&rows[2][0]), text(&rows[2][1]), int(&rows[2][2])), (13, "z", 3));

    // The index holds ONLY the new keys; the old keys are gone (not lingering).
    for old_v in [1, 2, 3] {
        assert!(
            index_eq_rows(&cat, &mut pager, "t", 2, "idx_a", Value::Integer(old_v)).is_empty(),
            "old index key a={old_v} removed"
        );
    }
    for new_v in [11, 12, 13] {
        assert_eq!(
            index_eq_rows(&cat, &mut pager, "t", 2, "idx_a", Value::Integer(new_v)).len(),
            1,
            "new index key a={new_v} present exactly once"
        );
    }
}

// ----- (6) rowid change: row and index entries follow the new rowid ---------

#[test]
fn updating_integer_primary_key_moves_row_and_index_entries() {
    let (mut pager, mut cat) =
        db_with_table("CREATE TABLE t(id INTEGER PRIMARY KEY, b TEXT)");
    assert_eq!(cat.table("t").unwrap().unwrap().rowid_alias, Some(0), "id is the rowid alias");
    create_index(&mut pager, &mut cat, "CREATE INDEX idx_b ON t(b)");
    // Seed id=1, b='hi' (explicit rowid via the alias column).
    seed(&cat, &mut pager, "t", 2, vec![vec![lit(Value::Integer(1)), lit(Value::Text("hi".into()))]]);
    let mut rt = Runtime::new();
    // UPDATE t SET id = 100
    let plan = update_plan(
        "t",
        2,
        vec![(0, lit(Value::Integer(100)))],
        seqscan("t", 2),
        affinities(&cat, "t"),
        OnConflict::Abort,
        Vec::new(),
    );
    run(&plan, &cat, &mut pager, &mut rt);
    assert_eq!(rt.changes(), 1);

    // The table row now lives at rowid 100 and is gone from rowid 1.
    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1);
    // The alias column reads back AS the rowid (stored NULL, refilled on read).
    assert_eq!((int(&rows[0][0]), text(&rows[0][1]), int(&rows[0][2])), (100, "hi", 100));

    // The index entry on b now carries the NEW rowid (its trailing rowid moved 1→100).
    let idx_root = cat.index("idx_b").unwrap().unwrap().root_page;
    let mut cur = IndexCursor::open(&pager, idx_root).unwrap();
    assert!(cur.first().unwrap(), "one index entry present");
    let key = minisqlite_fileformat::decode_record(&cur.key().unwrap());
    assert_eq!(text(&key[0]), "hi");
    assert_eq!(int(&key[1]), 100, "index entry points at the new rowid");
    assert!(!cur.next().unwrap(), "exactly one index entry");
}

/// A rowid change while a UNIQUE indexed column is UNCHANGED must NOT self-conflict.
/// This is the case where passing the NEW rowid to the uniqueness probe would wrongly
/// flag the row's own (old-rowid-keyed) entry as a foreign duplicate.
#[test]
fn rowid_change_with_unchanged_unique_column_is_not_a_conflict() {
    let (mut pager, mut cat) =
        db_with_table("CREATE TABLE t(id INTEGER PRIMARY KEY, a INTEGER)");
    create_index(&mut pager, &mut cat, "CREATE UNIQUE INDEX uq_a ON t(a)");
    seed(&cat, &mut pager, "t", 2, vec![vec![lit(Value::Integer(1)), lit(Value::Integer(5))]]);
    let mut rt = Runtime::new();
    // UPDATE t SET id = 100  (a stays 5). Must succeed, not raise UNIQUE.
    let plan = update_plan(
        "t",
        2,
        vec![(0, lit(Value::Integer(100)))],
        seqscan("t", 2),
        affinities(&cat, "t"),
        OnConflict::Abort,
        Vec::new(),
    );
    run(&plan, &cat, &mut pager, &mut rt);
    assert_eq!(rt.changes(), 1);

    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1);
    assert_eq!((int(&rows[0][0]), int(&rows[0][1]), int(&rows[0][2])), (100, 5, 100));
    // The UNIQUE index entry for a=5 now carries rowid 100.
    let hits = index_eq_rows(&cat, &mut pager, "t", 2, "uq_a", Value::Integer(5));
    assert_eq!(hits.len(), 1);
    assert_eq!(int(&hits[0][2]), 100);
}

// ----- (7) ON CONFLICT on a UNIQUE violation --------------------------------

#[test]
fn or_ignore_skips_the_unique_conflict_and_leaves_the_row() {
    let (mut pager, mut cat) = db_with_table("CREATE TABLE t(a INTEGER, b TEXT)");
    create_index(&mut pager, &mut cat, "CREATE UNIQUE INDEX uq_a ON t(a)");
    // rowid 1: a=5, rowid 2: a=9.
    seed(
        &cat,
        &mut pager,
        "t",
        2,
        vec![
            vec![lit(Value::Integer(5)), lit(Value::Text("first".into()))],
            vec![lit(Value::Integer(9)), lit(Value::Text("second".into()))],
        ],
    );
    let mut rt = Runtime::new();
    // UPDATE OR IGNORE t SET a = 9 WHERE rowid = 1  — collides with rowid 2's a=9, so
    // the row is skipped and left as a=5.
    let scan = PlanNode::Filter {
        input: Box::new(seqscan("t", 2)),
        predicate: cmp(CmpOp::Eq, col(2), lit(Value::Integer(1))),
    };
    let plan = update_plan(
        "t",
        2,
        vec![(0, lit(Value::Integer(9)))],
        scan,
        affinities(&cat, "t"),
        OnConflict::Ignore,
        Vec::new(),
    );
    run(&plan, &cat, &mut pager, &mut rt);
    assert_eq!(rt.changes(), 0, "the conflicting row was skipped");

    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 2);
    assert_eq!((int(&rows[0][0]), text(&rows[0][1])), (5, "first"), "rowid 1 untouched");
    assert_eq!((int(&rows[1][0]), text(&rows[1][1])), (9, "second"), "rowid 2 untouched");
}

#[test]
fn or_abort_errors_on_the_unique_conflict_and_rolls_back() {
    let (mut pager, mut cat) = db_with_table("CREATE TABLE t(a INTEGER, b TEXT)");
    create_index(&mut pager, &mut cat, "CREATE UNIQUE INDEX uq_a ON t(a)");
    seed(
        &cat,
        &mut pager,
        "t",
        2,
        vec![
            vec![lit(Value::Integer(5)), lit(Value::Text("first".into()))],
            vec![lit(Value::Integer(9)), lit(Value::Text("second".into()))],
        ],
    );
    // UPDATE t SET a = 9 WHERE rowid = 1 (default Abort) → UNIQUE constraint error.
    let scan = PlanNode::Filter {
        input: Box::new(seqscan("t", 2)),
        predicate: cmp(CmpOp::Eq, col(2), lit(Value::Integer(1))),
    };
    let plan = update_plan(
        "t",
        2,
        vec![(0, lit(Value::Integer(9)))],
        scan,
        affinities(&cat, "t"),
        OnConflict::Abort,
        Vec::new(),
    );
    let err = run_err(&plan, &cat, &mut pager);
    // Assert the KIND, not merely that it errored: a wrong constraint kind
    // (PrimaryKey/NotNull) must NOT pass. `Error::Constraint` folds the kind into the
    // message ("UNIQUE constraint failed: <detail>"), and the detail is the index's
    // `table.col`, so pin both.
    match &err {
        Error::Constraint(m) => {
            assert!(m.starts_with("UNIQUE constraint failed"), "expected UNIQUE kind, got {m:?}");
            assert!(m.contains("t.a"), "expected the uq_a index detail t.a, got {m:?}");
        }
        other => panic!("expected Error::Constraint, got {other:?}"),
    }

    // The rolled-back statement left both original rows intact.
    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!((int(&rows[0][0]), text(&rows[0][1])), (5, "first"));
    assert_eq!((int(&rows[1][0]), text(&rows[1][1])), (9, "second"));
}

/// A multi-row UPDATE where an EARLIER row writes successfully and a LATER row then
/// conflicts under Abort. Two things are pinned: (1) the conflict is detected against
/// LIVE state — row 2 sees row 1's just-written entry, not the phase-1 snapshot — and
/// (2) the transaction rollback undoes row 1's PARTIAL write, so no half-applied
/// statement survives. (Non-alias table, so the implicit-rowid path is exercised too.)
#[test]
fn abort_after_a_partial_write_rolls_the_whole_statement_back() {
    let (mut pager, mut cat) = db_with_table("CREATE TABLE t(a INTEGER)");
    create_index(&mut pager, &mut cat, "CREATE UNIQUE INDEX uq_a ON t(a)");
    // rowid 1: a=1, rowid 2: a=2.
    seed(
        &cat,
        &mut pager,
        "t",
        1,
        vec![vec![lit(Value::Integer(1))], vec![lit(Value::Integer(2))]],
    );
    // UPDATE t SET a = 100 (Abort). Row 1 (rowid 1): 1→100 succeeds and writes its entry.
    // Row 2 (rowid 2): 2→100 then collides with row 1's LIVE a=100 → UNIQUE error.
    let plan = update_plan(
        "t",
        1,
        vec![(0, lit(Value::Integer(100)))],
        seqscan("t", 1),
        affinities(&cat, "t"),
        OnConflict::Abort,
        Vec::new(),
    );
    let err = run_err(&plan, &cat, &mut pager);
    match &err {
        Error::Constraint(m) => {
            assert!(m.starts_with("UNIQUE constraint failed"), "expected UNIQUE kind, got {m:?}")
        }
        other => panic!("expected Error::Constraint, got {other:?}"),
    }
    // Rollback undid row 1's partial write too: both rows are back at their originals.
    let rows = scan_all(&cat, &mut pager, "t", 1);
    assert_eq!(rows.len(), 2);
    assert_eq!((int(&rows[0][0]), int(&rows[0][1])), (1, 1));
    assert_eq!((int(&rows[1][0]), int(&rows[1][1])), (2, 2));
    // The index is back to the originals too: a=100 reaches nothing; a=1 and a=2 one each.
    assert!(
        index_eq_rows(&cat, &mut pager, "t", 1, "uq_a", Value::Integer(100)).is_empty(),
        "the rolled-back a=100 entry is gone"
    );
    assert_eq!(index_eq_rows(&cat, &mut pager, "t", 1, "uq_a", Value::Integer(1)).len(), 1);
    assert_eq!(index_eq_rows(&cat, &mut pager, "t", 1, "uq_a", Value::Integer(2)).len(), 1);
}

#[test]
fn no_op_update_of_a_unique_column_does_not_self_conflict() {
    // Updating a UNIQUE column to its OWN current value (same rowid) must not be flagged
    // as a duplicate against the row's own not-yet-deleted index entry.
    let (mut pager, mut cat) = db_with_table("CREATE TABLE t(a INTEGER, b TEXT)");
    create_index(&mut pager, &mut cat, "CREATE UNIQUE INDEX uq_a ON t(a)");
    seed(&cat, &mut pager, "t", 2, vec![vec![lit(Value::Integer(5)), lit(Value::Text("row".into()))]]);
    let mut rt = Runtime::new();
    // UPDATE t SET a = 5 (no-op on the value) — must succeed and count as a change.
    let plan = update_plan(
        "t",
        2,
        vec![(0, lit(Value::Integer(5)))],
        seqscan("t", 2),
        affinities(&cat, "t"),
        OnConflict::Abort,
        Vec::new(),
    );
    run(&plan, &cat, &mut pager, &mut rt);
    assert_eq!(rt.changes(), 1);
    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!((int(&rows[0][0]), text(&rows[0][1])), (5, "row"));
}

// ----- (8) RETURNING reflects the UPDATED row -------------------------------

#[test]
fn returning_reflects_the_updated_values() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER, b TEXT)");
    seed(&cat, &mut pager, "t", 2, vec![vec![lit(Value::Integer(1)), lit(Value::Text("old".into()))]]);
    let mut rt = Runtime::new();
    // UPDATE t SET a = 42 RETURNING a, b, rowid
    let plan = update_plan(
        "t",
        2,
        vec![(0, lit(Value::Integer(42)))],
        seqscan("t", 2),
        affinities(&cat, "t"),
        OnConflict::Abort,
        vec![col(0), col(1), col(2)],
    );
    let out = run(&plan, &cat, &mut pager, &mut rt);
    assert_eq!(out.len(), 1);
    assert_eq!(rt.changes(), 1, "the single row was updated");
    // a is the NEW value; b is unchanged; the trailing register is the rowid.
    assert_eq!((int(&out[0][0]), text(&out[0][1]), int(&out[0][2])), (42, "old", 1));
}

#[test]
fn returning_reflects_the_new_rowid_after_a_rowid_change() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(id INTEGER PRIMARY KEY, b TEXT)");
    seed(&cat, &mut pager, "t", 2, vec![vec![lit(Value::Integer(3)), lit(Value::Text("v".into()))]]);
    let mut rt = Runtime::new();
    // UPDATE t SET id = 50 RETURNING id, rowid — both reflect the NEW rowid 50.
    let plan = update_plan(
        "t",
        2,
        vec![(0, lit(Value::Integer(50)))],
        seqscan("t", 2),
        affinities(&cat, "t"),
        OnConflict::Abort,
        vec![col(0), col(2)],
    );
    let out = run(&plan, &cat, &mut pager, &mut rt);
    assert_eq!(out.len(), 1);
    assert_eq!((int(&out[0][0]), int(&out[0][1])), (50, 50), "alias column and rowid are the new rowid");
}

// ----- (9) rowid alias (INTEGER PRIMARY KEY) datatype coercion ---------------
// lang_createtable.html §5: an UPDATE cannot set the rowid alias to a NULL, a blob, a
// fractional/out-of-range real, or a non-numeric string — each is a "datatype mismatch"
// that aborts, leaving the row untouched. A genuine no-op (the alias is never assigned)
// and an integer / integral-real assignment are exact. (This replaces the former
// "documented narrowing" where a non-integer alias assignment silently kept the old
// rowid, masking the error real sqlite raises.)

/// Seed `t(id INTEGER PRIMARY KEY, b)` with one row at rowid 3, run `UPDATE t SET id =
/// <bad>`, and assert it errors with a datatype mismatch, leaving the row at rowid 3.
fn assert_update_alias_mismatch(bad: Value) {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(id INTEGER PRIMARY KEY, b TEXT)");
    seed(&cat, &mut pager, "t", 2, vec![vec![lit(Value::Integer(3)), lit(Value::Text("v".into()))]]);
    let plan = update_plan(
        "t",
        2,
        vec![(0, lit(bad.clone()))],
        seqscan("t", 2),
        affinities(&cat, "t"),
        OnConflict::Abort,
        Vec::new(),
    );
    let err = run_err(&plan, &cat, &mut pager);
    match err {
        Error::Sql(m) => assert_eq!(m, "datatype mismatch", "value {bad:?}"),
        other => panic!("expected Error::Sql(\"datatype mismatch\") for {bad:?}, got {other:?}"),
    }
    // Aborted: the row is untouched at rowid 3.
    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1);
    assert_eq!(
        (int(&rows[0][0]), text(&rows[0][1]), int(&rows[0][2])),
        (3, "v", 3),
        "row unmoved for {bad:?}"
    );
}

#[test]
fn set_rowid_alias_to_null_is_datatype_mismatch() {
    // Real sqlite: an UPDATE cannot null the rowid. (Formerly kept the old rowid.)
    assert_update_alias_mismatch(Value::Null);
}

#[test]
fn set_rowid_alias_to_non_numeric_text_is_datatype_mismatch() {
    assert_update_alias_mismatch(Value::Text("notint".into()));
}

#[test]
fn set_rowid_alias_to_fractional_real_is_datatype_mismatch() {
    assert_update_alias_mismatch(Value::Real(2.5));
}

#[test]
fn set_rowid_alias_to_blob_is_datatype_mismatch() {
    assert_update_alias_mismatch(Value::Blob(vec![9]));
}

#[test]
fn noop_update_on_alias_table_leaves_rowids_unchanged() {
    // The alias is never assigned (only b is), so new_vals[alias] stays Integer(old_rowid)
    // and coerces back to it — no relocation and no spurious mismatch. This is the case
    // the "require an integer" rule must NOT break.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(id INTEGER PRIMARY KEY, b TEXT)");
    seed(
        &cat,
        &mut pager,
        "t",
        2,
        vec![
            vec![lit(Value::Integer(1)), lit(Value::Text("a".into()))],
            vec![lit(Value::Integer(2)), lit(Value::Text("b".into()))],
        ],
    );
    let mut rt = Runtime::new();
    let plan = update_plan(
        "t",
        2,
        vec![(1, lit(Value::Text("z".into())))], // assign b only, never the alias
        seqscan("t", 2),
        affinities(&cat, "t"),
        OnConflict::Abort,
        Vec::new(),
    );
    run(&plan, &cat, &mut pager, &mut rt);
    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 2);
    assert_eq!((int(&rows[0][0]), text(&rows[0][1]), int(&rows[0][2])), (1, "z", 1));
    assert_eq!((int(&rows[1][0]), text(&rows[1][1]), int(&rows[1][2])), (2, "z", 2));
}

#[test]
fn set_rowid_alias_to_integral_real_relocates_row() {
    // An exactly-integral real moves the rowid (10.0 -> 10) like an integer assignment,
    // neither rejected nor truncated.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(id INTEGER PRIMARY KEY, b TEXT)");
    seed(&cat, &mut pager, "t", 2, vec![vec![lit(Value::Integer(1)), lit(Value::Text("a".into()))]]);
    let mut rt = Runtime::new();
    let plan = update_plan(
        "t",
        2,
        vec![(0, lit(Value::Real(10.0)))],
        seqscan("t", 2),
        affinities(&cat, "t"),
        OnConflict::Abort,
        Vec::new(),
    );
    run(&plan, &cat, &mut pager, &mut rt);
    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1);
    assert_eq!((int(&rows[0][0]), text(&rows[0][1]), int(&rows[0][2])), (10, "a", 10));
}

// ----- (10) WITHOUT ROWID routes to the WR write path -----------------------

#[test]
fn update_over_without_rowid_table_column_pk_succeeds() {
    // UPDATE resolves its base table through the WR-aware `resolve_base_table` gate and
    // branches on `def.without_rowid`, so an UPDATE of a WITHOUT ROWID table now REWRITES
    // the PRIMARY KEY index b-tree (it no longer fails closed). This is the smoke test that
    // a WR UPDATE routes to the WR path and rewrites the correct row; the exhaustive WR
    // cases (PK-changing updates, ON CONFLICT resolution, composite PK, RETURNING) live in
    // `tests/without_rowid_update.rs`. A column-level `INTEGER PRIMARY KEY WITHOUT ROWID`
    // recovers its PK from `TableDef::primary_key` (via `ops::without_rowid::wr_pk_columns`),
    // the single ordered authority for every PK form.
    let (mut pager, cat) =
        db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT) WITHOUT ROWID");
    seed(
        &cat,
        &mut pager,
        "t",
        2,
        vec![
            vec![lit(Value::Integer(1)), lit(Value::Text("x".into()))],
            vec![lit(Value::Integer(2)), lit(Value::Text("y".into()))],
        ],
    );
    let mut rt = Runtime::new();
    run(
        &update_plan(
            "t",
            2,
            vec![(1, lit(Value::Text("z".into())))],
            seqscan("t", 2),
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    assert_eq!(rt.changes(), 2, "both WR rows updated");
    // A WR scan emits width-N rows `[a, b]` (no rowid register), ordered by PRIMARY KEY.
    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 2, "no rows lost or duplicated");
    assert_eq!((int(&rows[0][0]), text(&rows[0][1])), (1, "z"), "PK 1 kept, b rewritten");
    assert_eq!((int(&rows[1][0]), text(&rows[1][1])), (2, "z"), "PK 2 kept, b rewritten");
}

// ----- (11) CHECK on the INTEGER PRIMARY KEY alias reads the rowid register --
//
// UPDATE evaluates its CHECK predicates over the POST-assignment `[c0..c_{N-1}, rowid]`
// row, where an INTEGER PRIMARY KEY alias column (and a bare `rowid`) resolves to the
// TRAILING register N (minisqlite_plan::bind::scope), NOT its column-i slot. So the operator
// must extend the width-N new row with the new rowid before enforcing checks. These pin it
// on `t(a INTEGER PRIMARY KEY, b TEXT)` (N == 2, rowid register Column(2)) with a bare
// `CHECK(a)` bound to Column(2): a row seeded at rowid 0, updated on the NON-alias column b
// (so the rowid stays 0), violates (truth(0) == false); at rowid 5 it passes. Evaluating
// over the width-2 new row instead would read Column(2) out of range — not this clean CHECK
// violation — so these fail without the rowid-register layout.

/// `UPDATE t SET b = <newb>` over a full scan of `t(a INTEGER PRIMARY KEY, b TEXT)`, carrying
/// a bare `CHECK(a)` == Column(2) (the trailing rowid register for a 2-column table). The
/// alias column is never assigned, so the rowid is unchanged and the check reads it directly.
fn update_setb_with_alias_check(cat: &SchemaCatalog, newb: &str) -> Plan {
    Plan {
        root: PlanNode::Update(Update { db: DbIndex::MAIN,
            table: "t".to_string(),
            column_count: 2,
            assignments: vec![(1, lit(Value::Text(newb.into())))],
            scan: Box::new(seqscan("t", 2)),
            column_affinities: affinities(cat, "t"),
            on_conflict: OnConflict::Abort,
            returning: Vec::new(),
            triggers: Vec::new(),
            checks: vec![CheckConstraint { expr: col(2), detail: "t".into() }],
            index_key_exprs: Vec::new(),
            index_partial_predicates: Vec::new(),
        }),
        result_columns: Vec::new(),
        ctes: Vec::new(),
        subqueries: Vec::new(),
        mutates: true,
        generated: Vec::new(),
    }
}

#[test]
fn check_on_rowid_alias_violation_on_update_reads_rowid_register() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT)");
    // Seed at rowid 0 (explicit alias 0); CHECK(a) == Column(2) will read this rowid.
    seed(&cat, &mut pager, "t", 2, vec![vec![lit(Value::Integer(0)), lit(Value::Text("x".into()))]]);
    // UPDATE SET b='new' leaves the rowid at 0, so CHECK(a) reads Column(2) == 0 -> violation.
    let err = run_err(&update_setb_with_alias_check(&cat, "new"), &cat, &mut pager);
    assert!(
        matches!(err, Error::Constraint(_)),
        "CHECK(a) with rowid 0 is a Constraint violation on UPDATE (reads the rowid register), got {err:?}"
    );
    // Rolled back: b is still 'x' at rowid 0.
    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1);
    assert_eq!((int(&rows[0][2]), text(&rows[0][1])), (0, "x"), "the failed update did not apply");
}

#[test]
fn check_on_rowid_alias_nonzero_on_update_passes() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT)");
    // Seed at rowid 5; CHECK(a) == Column(2) reads 5 -> passes.
    seed(&cat, &mut pager, "t", 2, vec![vec![lit(Value::Integer(5)), lit(Value::Text("x".into()))]]);
    let mut rt = Runtime::new();
    run(&update_setb_with_alias_check(&cat, "new"), &cat, &mut pager, &mut rt);
    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1);
    assert_eq!((int(&rows[0][2]), text(&rows[0][1])), (5, "new"), "the satisfying update applied");
}

// ----- (12) OR IGNORE skips a CHECK-violating row, keeps the rest -----------
//
// A statement-level `UPDATE OR IGNORE` overrides the ABORT default for a CHECK just as for
// NOT NULL/UNIQUE (lang_conflict.html: the OR clause overrides the CREATE TABLE default),
// so a row whose POST-assignment value violates a CHECK is SKIPPED (left untouched, no
// error) while the other rows update — it must NOT abort + roll back the whole statement.
// Reproduces the witness `CREATE TABLE t(x INT CHECK(x>0)); INSERT INTO t
// VALUES(5),(10); UPDATE OR IGNORE t SET x = x - 7` -> rows {3,5} (5->-2 skipped, 10->3
// applied).
//
// Regression guard: before enforce_checks took `on_conflict`, the -2 row raised
// SQLITE_CONSTRAINT mid-loop and the implicit-txn rollback reverted BOTH rows (unchanged +
// error). `run` unwraps execute/pull, so that pre-fix path would PANIC here.

#[test]
fn update_or_ignore_skips_check_violating_row_keeps_the_rest() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(x INTEGER)");
    seed(&cat, &mut pager, "t", 1, vec![vec![lit(Value::Integer(5))], vec![lit(Value::Integer(10))]]);
    let mut rt = Runtime::new();
    // CHECK(x > 0) over Column(0) (no rowid alias, N == 1); SET x = x - 7.
    let check = CheckConstraint {
        expr: cmp(CmpOp::Gt, col(0), lit(Value::Integer(0))),
        detail: "t.x".into(),
    };
    let plan = Plan {
        root: PlanNode::Update(Update { db: DbIndex::MAIN,
            table: "t".to_string(),
            column_count: 1,
            assignments: vec![(0, arith(ArithOp::Sub, col(0), lit(Value::Integer(7))))],
            scan: Box::new(seqscan("t", 1)),
            column_affinities: affinities(&cat, "t"),
            on_conflict: OnConflict::Ignore,
            returning: Vec::new(),
            triggers: Vec::new(),
            checks: vec![check],
            index_key_exprs: Vec::new(),
            index_partial_predicates: Vec::new(),
        }),
        result_columns: Vec::new(),
        ctes: Vec::new(),
        subqueries: Vec::new(),
        mutates: true,
        generated: Vec::new(),
    };
    // No error — the row whose new value violates is skipped, not an abort.
    run(&plan, &cat, &mut pager, &mut rt);
    assert_eq!(rt.changes(), 1, "only the 10->3 row updated; the 5->-2 row was skipped, not counted");
    let mut rows = scan_all(&cat, &mut pager, "t", 1);
    rows.sort_by_key(|r| int(&r[0]));
    assert_eq!(rows.len(), 2);
    // 10 -> 3 applied; 5 left untouched (its update to -2 would have violated x > 0).
    assert_eq!(int(&rows[0][0]), 3, "the 10 row updated to 3");
    assert_eq!(int(&rows[1][0]), 5, "the 5 row was left untouched (its -2 result violated)");
}
