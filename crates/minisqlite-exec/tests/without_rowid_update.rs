//! Integration tests for `UPDATE` on a WITHOUT ROWID (WR) table: hand-built [`Plan`]
//! trees run over a real [`MemPager`] + [`SchemaCatalog`] through the
//! [`Executor`]/[`RowCursor`] seam, then read back through the real WR scan (not a mock).
//!
//! A WR table stores each row in the PRIMARY KEY index b-tree keyed by its PK, with no
//! integer rowid (withoutrowid.html; fileformat2 §2.4). These pin the WR UPDATE path end
//! to end: a non-PK update, a PK-changing update (new key + old key gone), the three
//! `ON CONFLICT` resolutions of a PK collision (ABORT/REPLACE/IGNORE), the implicit PK
//! `NOT NULL`, a composite PRIMARY KEY, `RETURNING` the post-update row, the change
//! counter, an `UPDATE OR REPLACE` cascade that collapses a shifting PK (the `invalidated`
//! + live re-read branch), assignment affinity coercion, parent-side FK `ON UPDATE CASCADE`
//! from a WR parent into a rowid child, a WR table with a UNIQUE secondary index (the UPDATE
//! MOVES the row's index entry and enforces UNIQUE on the new value — fileformat2 §2.5.1),
//! and a WR table carrying its own UPDATE trigger (the UPDATE completes with the trigger
//! attached and rewrites the row — an empty-body trigger cannot prove firing). Expected
//! behavior is derived from
//! `lang_update.html` + `withoutrowid.html` + `foreignkeys.html`.
//!
//! The operator assumes an open write transaction (the engine opens one for DML), so each
//! apply is wrapped in `begin`/`commit`; read-back scans need no transaction. Rows are
//! seeded through the real WR `INSERT` operator so the tests exercise the same write path
//! they verify.

use minisqlite_btree::init_database;
use minisqlite_catalog::{Catalog, SchemaCatalog};
use minisqlite_exec::{Executor, PagerSet, Runtime, StreamingExecutor};
use minisqlite_expr::{ArithOp, CmpOp, CompareMeta, EvalExpr};
use minisqlite_pager::{MemPager, Pager};
use minisqlite_plan::{
    Insert, OnConflict, Plan, PlanNode, SeqScan, TriggerProgram, TriggerTiming, Update,
};
use minisqlite_sql::{parse, Statement};
use minisqlite_types::{affinity_of_declared_type, Affinity, Collation, DbIndex, Error, Value};

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

/// Create an ADDITIONAL table in an existing db (e.g. an FK child referencing a WR parent).
fn create_table(pager: &mut MemPager, cat: &mut SchemaCatalog, create_sql: &str) {
    let ast = parse(create_sql).unwrap();
    let Statement::CreateTable(stmt) = &ast.statements[0] else {
        panic!("not a CREATE TABLE: {create_sql}");
    };
    cat.create_table(pager, stmt, create_sql).unwrap();
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

fn txt(s: &str) -> EvalExpr {
    lit(Value::Text(s.into()))
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

/// A `Filter(predicate)` over the whole-table WR scan — the shape the planner emits for a
/// WR `UPDATE ... WHERE` (the whole WHERE is a residual filter over a width-N seq scan).
fn filtered(table: &str, n: usize, predicate: EvalExpr) -> PlanNode {
    PlanNode::Filter { input: Box::new(seqscan(table, n)), predicate }
}

#[allow(clippy::too_many_arguments)]
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
        root: PlanNode::Update(Update {
            db: DbIndex::MAIN,
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

/// An `UPDATE` plan carrying a single (synthetic) trigger — the shape a `CREATE TRIGGER
/// ... UPDATE ON <wr_table>` reaches the executor as. The WR path fires the row's UPDATE
/// triggers with a width-N `OLD`/`NEW` frame (`trigger::build_frame_wr`), so this drives that
/// firing path; an empty action body is a no-op, so the row is rewritten as normal.
fn update_plan_with_trigger(
    table: &str,
    n: usize,
    assignments: Vec<(usize, EvalExpr)>,
    scan: PlanNode,
    column_affinities: Vec<Affinity>,
) -> Plan {
    Plan {
        root: PlanNode::Update(Update {
            db: DbIndex::MAIN,
            table: table.to_string(),
            column_count: n,
            assignments,
            scan: Box::new(scan),
            column_affinities,
            on_conflict: OnConflict::Abort,
            returning: Vec::new(),
            triggers: vec![TriggerProgram {
                name: "trg".into(),
                timing: TriggerTiming::After,
                when: None,
                actions: Vec::new(),
            }],
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

/// Seed rows through the real WR `INSERT` operator, so tests exercise the same write path
/// they verify (a WR table has no rowid — the row IS its PRIMARY KEY index record).
fn seed(cat: &SchemaCatalog, pager: &mut MemPager, table: &str, n: usize, rows: Vec<Vec<EvalExpr>>) {
    let plan = Plan {
        root: PlanNode::Insert(Insert {
            db: DbIndex::MAIN,
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

/// Apply a mutating plan inside a write transaction, returning any `RETURNING` rows and
/// threading the caller's `Runtime` so the change counters can be read afterward.
fn run(plan: &Plan, cat: &SchemaCatalog, pager: &mut MemPager, rt: &mut Runtime) -> Vec<Vec<Value>> {
    pager.begin().unwrap();
    let mut exec = StreamingExecutor;
    let mut out = Vec::new();
    {
        let mut cur = exec
            .execute(plan, cat, PagerSet::One { db: DbIndex::MAIN, pager: &mut *pager })
            .unwrap();
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

/// Full WR scan back through the executor. A WR scan emits width-N rows `[c0..c_{N-1}]`
/// (no rowid register), ordered by the PRIMARY KEY (the b-tree key order).
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
    let mut cur =
        exec.execute(&plan, cat, PagerSet::One { db: DbIndex::MAIN, pager: &mut *pager }).unwrap();
    let mut out = Vec::new();
    while let Some(row) = cur.next_row(&mut rt).unwrap() {
        out.push(row);
    }
    out
}

// ----- (a) non-PK column update, PK unchanged -------------------------------

#[test]
fn update_non_pk_column_keeps_pk_and_rewrites_value() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a TEXT PRIMARY KEY, b INTEGER) WITHOUT ROWID");
    seed(&cat, &mut pager, "t", 2, vec![vec![txt("k"), lit(Value::Integer(1))], vec![txt("m"), lit(Value::Integer(2))]]);

    // UPDATE t SET b = 99 WHERE a = 'k'.
    let mut rt = Runtime::new();
    run(
        &update_plan(
            "t",
            2,
            vec![(1, lit(Value::Integer(99)))],
            filtered("t", 2, cmp(CmpOp::Eq, col(0), txt("k"))),
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    assert_eq!(rt.changes(), 1, "only the a='k' row updated");

    // Re-scan: 'k' still present at its PK with the new b; 'm' untouched. WR scan order is
    // by PK, so 'k' (b now 99) precedes 'm' (b 2).
    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 2, "no row lost or duplicated");
    assert_eq!((text(&rows[0][0]), int(&rows[0][1])), ("k", 99), "PK 'k' kept, b rewritten");
    assert_eq!((text(&rows[1][0]), int(&rows[1][1])), ("m", 2), "PK 'm' untouched");
}

// ----- (b) PK column update to a NEW unique value ---------------------------

#[test]
fn update_pk_to_new_value_moves_the_row() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a TEXT PRIMARY KEY, b INTEGER) WITHOUT ROWID");
    seed(&cat, &mut pager, "t", 2, vec![vec![txt("k"), lit(Value::Integer(1))], vec![txt("m"), lit(Value::Integer(2))]]);

    // UPDATE t SET a = 'z' WHERE a = 'k' — the PK moves, carrying b along.
    let mut rt = Runtime::new();
    run(
        &update_plan(
            "t",
            2,
            vec![(0, txt("z"))],
            filtered("t", 2, cmp(CmpOp::Eq, col(0), txt("k"))),
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    assert_eq!(rt.changes(), 1);

    // 'k' is gone; the row is now found under the new PK 'z' (b carried over). Scan order:
    // 'm' (b 2) then 'z' (b 1).
    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 2);
    assert!(!rows.iter().any(|r| text(&r[0]) == "k"), "old PK 'k' no longer present");
    assert_eq!((text(&rows[0][0]), int(&rows[0][1])), ("m", 2));
    assert_eq!((text(&rows[1][0]), int(&rows[1][1])), ("z", 1), "row found under new PK 'z'");
}

// ----- (c) PK collision under ABORT (default) -------------------------------

#[test]
fn update_pk_collision_abort_errors_and_leaves_store_unchanged() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a TEXT PRIMARY KEY, b INTEGER) WITHOUT ROWID");
    seed(&cat, &mut pager, "t", 2, vec![vec![txt("k"), lit(Value::Integer(1))], vec![txt("m"), lit(Value::Integer(2))]]);

    // UPDATE t SET a = 'm' WHERE a = 'k' — collides with the existing 'm' row.
    let err = run_err(
        &update_plan(
            "t",
            2,
            vec![(0, txt("m"))],
            filtered("t", 2, cmp(CmpOp::Eq, col(0), txt("k"))),
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
    );
    match err {
        Error::Constraint(m) => {
            assert!(m.starts_with("PRIMARY KEY constraint failed"), "PK violation, got {m:?}");
            assert!(m.contains("t.a"), "names the PK column, got {m:?}");
        }
        other => panic!("expected a PRIMARY KEY Constraint error, got {other:?}"),
    }
    // The statement rolled back: both original rows survive unchanged.
    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 2, "no row added or removed");
    assert_eq!((text(&rows[0][0]), int(&rows[0][1])), ("k", 1));
    assert_eq!((text(&rows[1][0]), int(&rows[1][1])), ("m", 2));
}

// ----- (d) PK collision under REPLACE ---------------------------------------

#[test]
fn update_pk_collision_replace_removes_victim() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a TEXT PRIMARY KEY, b INTEGER) WITHOUT ROWID");
    seed(&cat, &mut pager, "t", 2, vec![vec![txt("k"), lit(Value::Integer(1))], vec![txt("m"), lit(Value::Integer(2))]]);

    // UPDATE OR REPLACE t SET a = 'm' WHERE a = 'k' — the existing 'm' (b=2) is the victim
    // and is deleted; the updated row (b=1) takes its PK. The REPLACE deletion is NOT
    // counted as a change (only the update is).
    let mut rt = Runtime::new();
    run(
        &update_plan(
            "t",
            2,
            vec![(0, txt("m"))],
            filtered("t", 2, cmp(CmpOp::Eq, col(0), txt("k"))),
            affinities(&cat, "t"),
            OnConflict::Replace,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    assert_eq!(rt.changes(), 1, "the update counts once; the victim delete does not");

    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1, "victim removed, 'k' moved to 'm' — one row remains");
    assert_eq!((text(&rows[0][0]), int(&rows[0][1])), ("m", 1), "updated row present, victim's b gone");
}

// ----- (e) PK collision under IGNORE ----------------------------------------

#[test]
fn update_pk_collision_ignore_leaves_row_unchanged() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a TEXT PRIMARY KEY, b INTEGER) WITHOUT ROWID");
    seed(&cat, &mut pager, "t", 2, vec![vec![txt("k"), lit(Value::Integer(1))], vec![txt("m"), lit(Value::Integer(2))]]);

    // UPDATE OR IGNORE t SET a = 'm' WHERE a = 'k' — the collision skips this row silently.
    let mut rt = Runtime::new();
    run(
        &update_plan(
            "t",
            2,
            vec![(0, txt("m"))],
            filtered("t", 2, cmp(CmpOp::Eq, col(0), txt("k"))),
            affinities(&cat, "t"),
            OnConflict::Ignore,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    assert_eq!(rt.changes(), 0, "the colliding row was skipped, none updated");

    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 2, "both original rows intact");
    assert_eq!((text(&rows[0][0]), int(&rows[0][1])), ("k", 1), "'k' left unchanged");
    assert_eq!((text(&rows[1][0]), int(&rows[1][1])), ("m", 2), "'m' left unchanged");
}

// ----- (f) SET a PK column to NULL ------------------------------------------

#[test]
fn update_pk_to_null_is_a_not_null_violation() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a TEXT PRIMARY KEY, b INTEGER) WITHOUT ROWID");
    seed(&cat, &mut pager, "t", 2, vec![vec![txt("k"), lit(Value::Integer(1))]]);

    // UPDATE t SET a = NULL WHERE a = 'k' — every PK column of a WR table is implicitly
    // NOT NULL (withoutrowid.html).
    let err = run_err(
        &update_plan(
            "t",
            2,
            vec![(0, lit(Value::Null))],
            filtered("t", 2, cmp(CmpOp::Eq, col(0), txt("k"))),
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
    );
    match err {
        Error::Constraint(m) => {
            assert!(m.starts_with("NOT NULL constraint failed"), "NOT NULL violation, got {m:?}");
            assert!(m.contains("t.a"), "names the PK column, got {m:?}");
        }
        other => panic!("expected a NOT NULL Constraint error, got {other:?}"),
    }
    // Rolled back: the row is untouched.
    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1);
    assert_eq!((text(&rows[0][0]), int(&rows[0][1])), ("k", 1));
}

// ----- (g) composite PRIMARY KEY --------------------------------------------

#[test]
fn update_composite_pk_component_moves_the_row() {
    // PK is (c, a) → storage key order [c, a], then b. Schema order stays [a, b, c].
    let (mut pager, cat) =
        db_with_table("CREATE TABLE t(a INTEGER, b INTEGER, c TEXT, PRIMARY KEY(c, a)) WITHOUT ROWID");
    seed(
        &cat,
        &mut pager,
        "t",
        3,
        vec![
            vec![lit(Value::Integer(1)), lit(Value::Integer(10)), txt("x")],
            vec![lit(Value::Integer(2)), lit(Value::Integer(20)), txt("y")],
        ],
    );

    // First, a non-PK update over the composite key: SET b = 99 WHERE a = 1 AND c = 'x'.
    let mut rt = Runtime::new();
    run(
        &update_plan(
            "t",
            3,
            vec![(1, lit(Value::Integer(99)))],
            filtered(
                "t",
                3,
                EvalExpr::And(
                    Box::new(cmp(CmpOp::Eq, col(0), lit(Value::Integer(1)))),
                    Box::new(cmp(CmpOp::Eq, col(2), txt("x"))),
                ),
            ),
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    assert_eq!(rt.changes(), 1);

    // Then move a PK COMPONENT: SET a = 5 WHERE c = 'x' — PK ('x',1) → ('x',5).
    let mut rt2 = Runtime::new();
    run(
        &update_plan(
            "t",
            3,
            vec![(0, lit(Value::Integer(5)))],
            filtered("t", 3, cmp(CmpOp::Eq, col(2), txt("x"))),
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt2,
    );
    assert_eq!(rt2.changes(), 1);

    // Scan order by (c, a): ('x',5) then ('y',2). Schema-order rows are [a, b, c].
    let rows = scan_all(&cat, &mut pager, "t", 3);
    assert_eq!(rows.len(), 2);
    assert_eq!(
        (int(&rows[0][0]), int(&rows[0][1]), text(&rows[0][2])),
        (5, 99, "x"),
        "composite PK component moved (a:1->5), non-PK b kept its earlier update"
    );
    assert_eq!((int(&rows[1][0]), int(&rows[1][1]), text(&rows[1][2])), (2, 20, "y"), "other row untouched");
}

// ----- (h) RETURNING returns the NEW (post-update) row ----------------------

#[test]
fn returning_yields_post_update_values_including_a_changed_pk() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a TEXT PRIMARY KEY, b INTEGER) WITHOUT ROWID");
    seed(&cat, &mut pager, "t", 2, vec![vec![txt("k"), lit(Value::Integer(1))]]);

    // UPDATE t SET a = 'z', b = b + 10 WHERE a = 'k' RETURNING a, b — the NEW PK and value.
    let mut rt = Runtime::new();
    let out = run(
        &update_plan(
            "t",
            2,
            vec![(0, txt("z")), (1, arith(ArithOp::Add, col(1), lit(Value::Integer(10))))],
            filtered("t", 2, cmp(CmpOp::Eq, col(0), txt("k"))),
            affinities(&cat, "t"),
            OnConflict::Abort,
            vec![col(0), col(1)],
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    assert_eq!(out.len(), 1, "one RETURNING row");
    assert_eq!((text(&out[0][0]), int(&out[0][1])), ("z", 11), "RETURNING is the post-update row");
    // And the store reflects the same post-update row.
    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1);
    assert_eq!((text(&rows[0][0]), int(&rows[0][1])), ("z", 11));
}

// ----- (i) change counter counts exactly the updated rows -------------------

#[test]
fn changes_counts_exactly_the_updated_rows() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a TEXT PRIMARY KEY, b INTEGER) WITHOUT ROWID");
    seed(
        &cat,
        &mut pager,
        "t",
        2,
        vec![
            vec![txt("k"), lit(Value::Integer(5))],
            vec![txt("m"), lit(Value::Integer(5))],
            vec![txt("p"), lit(Value::Integer(9))],
        ],
    );

    // UPDATE t SET b = 0 WHERE b = 5 — matches 'k' and 'm', not 'p'.
    let mut rt = Runtime::new();
    run(
        &update_plan(
            "t",
            2,
            vec![(1, lit(Value::Integer(0)))],
            filtered("t", 2, cmp(CmpOp::Eq, col(1), lit(Value::Integer(5)))),
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    assert_eq!(rt.changes(), 2, "exactly the two b=5 rows counted");

    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 3, "no rows added or removed");
    assert_eq!((text(&rows[0][0]), int(&rows[0][1])), ("k", 0));
    assert_eq!((text(&rows[1][0]), int(&rows[1][1])), ("m", 0));
    assert_eq!((text(&rows[2][0]), int(&rows[2][1])), ("p", 9), "non-matching row untouched");
}

// ----- (j) OR REPLACE cascade over the whole table --------------------------

#[test]
fn or_replace_cascade_collapses_a_shifting_pk() {
    // `UPDATE OR REPLACE t SET a = a + 1` over `{1,2,3}` is the classic REPLACE cascade:
    // SQLite records the PK set in pass 1, then in pass 2 re-reads each key LIVE and applies
    // SET to the CURRENT contents. 1→2 evicts the row at 2; the pass-2 seek of key 2 then
    // finds the MOVED row and pushes it to 3 (evicting 3); the seek of key 3 finds it again
    // and pushes it to 4. Only the row that started at 1 survives, at PK 4, and every one of
    // the three keys counts as an update. This is exactly the `invalidated` + live re-read
    // path in the WR operator (a buffered row whose OLD PK an earlier victim-delete touched
    // is reprocessed from its current contents), so it pins that subtle branch.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b INTEGER) WITHOUT ROWID");
    seed(
        &cat,
        &mut pager,
        "t",
        2,
        vec![
            vec![lit(Value::Integer(1)), lit(Value::Integer(10))],
            vec![lit(Value::Integer(2)), lit(Value::Integer(20))],
            vec![lit(Value::Integer(3)), lit(Value::Integer(30))],
        ],
    );

    let mut rt = Runtime::new();
    run(
        &update_plan(
            "t",
            2,
            vec![(0, arith(ArithOp::Add, col(0), lit(Value::Integer(1))))],
            seqscan("t", 2), // no WHERE: every row
            affinities(&cat, "t"),
            OnConflict::Replace,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    assert_eq!(rt.changes(), 3, "all three keys are processed as updates");

    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1, "the cascade collapses to a single surviving row");
    assert_eq!(
        (int(&rows[0][0]), int(&rows[0][1])),
        (4, 10),
        "the row that started at PK 1 (b=10) survives at PK 4; the others were evicted"
    );
}

// ----- (k) WR with a UNIQUE secondary index: UPDATE maintains it ------------

#[test]
fn update_wr_maintains_unique_secondary_index() {
    // A WR table with a UNIQUE secondary column carries a secondary index keyed by
    // `[indexed cols.., trailing PK..]` (fileformat2 §2.5.1). An UPDATE that changes the
    // indexed column MOVES that row's index entry (old value freed, new value taken) and the
    // UNIQUE constraint is enforced on the new value. Proven behaviorally: after moving b
    // 5→99, re-using b=5 is free (its entry was removed) while moving a second row onto 99
    // collides.
    let (mut pager, cat) =
        db_with_table("CREATE TABLE t(a TEXT PRIMARY KEY, b INTEGER UNIQUE) WITHOUT ROWID");
    seed(
        &cat,
        &mut pager,
        "t",
        2,
        vec![vec![txt("k"), lit(Value::Integer(5))], vec![txt("m"), lit(Value::Integer(6))]],
    );

    // UPDATE t SET b = 99 WHERE a = 'k' — moves k's index entry from b=5 to b=99.
    let mut rt = Runtime::new();
    run(
        &update_plan(
            "t",
            2,
            vec![(1, lit(Value::Integer(99)))],
            filtered("t", 2, cmp(CmpOp::Eq, col(0), txt("k"))),
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    assert_eq!(rt.changes(), 1);
    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!((text(&rows[0][0]), int(&rows[0][1])), ("k", 99), "k now holds b=99");
    assert_eq!((text(&rows[1][0]), int(&rows[1][1])), ("m", 6), "m untouched");

    // The old b=5 entry was removed by the move: inserting a new row reusing b=5 succeeds.
    seed(&cat, &mut pager, "t", 2, vec![vec![txt("z"), lit(Value::Integer(5))]]);
    assert_eq!(scan_all(&cat, &mut pager, "t", 2).len(), 3, "b=5 was free to reuse after the move");

    // UNIQUE is enforced on the UPDATE's NEW value: moving m's b onto the existing 99 errors.
    let err = run_err(
        &update_plan(
            "t",
            2,
            vec![(1, lit(Value::Integer(99)))],
            filtered("t", 2, cmp(CmpOp::Eq, col(0), txt("m"))),
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
    );
    match err {
        Error::Constraint(m) => {
            assert!(m.contains("UNIQUE constraint failed"), "names the UNIQUE violation, got {m:?}");
        }
        other => panic!("expected a UNIQUE constraint error, got {other:?}"),
    }
}

// ----- (l) WR carrying an UPDATE trigger: the UPDATE completes ---------------

#[test]
fn update_wr_completes_with_a_trigger_attached() {
    // A WR table can carry UPDATE triggers. This pins that the WR UPDATE path COMPLETES with a
    // trigger ATTACHED — it does not fail closed: the plan carries one `AFTER UPDATE` program
    // and the UPDATE runs to completion, rewriting the row. The trigger body is EMPTY, whose
    // observable effect is identical whether the trigger fires or is skipped, so this test
    // proves completion-with-a-trigger, NOT that the trigger actually fires. Real WR trigger
    // firing (width-N `OLD`/`NEW` binding, a non-empty body) is covered by
    // `crates/minisqlite/tests/conformance_wr_triggers.rs`.
    let (mut pager, cat) =
        db_with_table("CREATE TABLE t(a TEXT PRIMARY KEY, b INTEGER) WITHOUT ROWID");
    seed(&cat, &mut pager, "t", 2, vec![vec![lit(Value::Text("x".into())), lit(Value::Integer(1))]]);

    let mut rt = Runtime::new();
    run(
        &update_plan_with_trigger(
            "t",
            2,
            vec![(1, lit(Value::Integer(0)))],
            seqscan("t", 2),
            affinities(&cat, "t"),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    assert_eq!(rt.changes(), 1, "the WR UPDATE completed with a trigger attached and rewrote the row");

    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1, "still one row after the update");
    assert_eq!(text(&rows[0][0]), "x", "PK unchanged");
    assert_eq!(int(&rows[0][1]), 0, "b updated to 0 with a trigger attached");
}

// ----- (m) assignment affinity coercion -------------------------------------

#[test]
fn assigned_numeric_text_gets_integer_affinity() {
    // An UPDATE assignment applies the target column's affinity to the new value
    // (lang_update.html / datatype3.html §3): assigning the TEXT literal '42' to an
    // INTEGER-affinity column stores Integer(42), not Text("42"). Pins that the WR path
    // threads `column_affinities` through `apply_affinity` exactly as the rowid path.
    let (mut pager, cat) =
        db_with_table("CREATE TABLE t(a TEXT PRIMARY KEY, b INTEGER) WITHOUT ROWID");
    seed(&cat, &mut pager, "t", 2, vec![vec![txt("k"), lit(Value::Integer(1))]]);

    let mut rt = Runtime::new();
    run(
        &update_plan(
            "t",
            2,
            vec![(1, txt("42"))],
            filtered("t", 2, cmp(CmpOp::Eq, col(0), txt("k"))),
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    assert_eq!(rt.changes(), 1);

    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1);
    assert_eq!(text(&rows[0][0]), "k", "PK unchanged");
    // `int` panics unless the stored value is the `Integer` variant, so this pins BOTH the
    // numeric value AND that the TEXT '42' was coerced (not stored as Text) by INTEGER affinity.
    assert_eq!(
        int(&rows[0][1]),
        42,
        "TEXT '42' assigned to an INTEGER column is coerced to Integer(42), not stored as Text"
    );
}

// ----- (n) WR parent ON UPDATE CASCADE into a rowid child -------------------

#[test]
fn update_on_wr_parent_cascades_to_rowid_child() {
    // The WR update path calls `enforce_parent_update` (parent side) exactly as the rowid
    // path does. With `PRAGMA foreign_keys` ON, changing a WR parent row's referenced key must
    // CASCADE into a ROWID child that references it: the matching child rows follow the new
    // key, the non-matching one is untouched, and — since `sqlite3_changes()` excludes
    // FK-action rows — only the parent update advances `changes()` (the cascade does not). A
    // wrong `logical` slice/arg would drop or mis-target it.
    let (mut pager, mut cat) =
        db_with_table("CREATE TABLE p(a TEXT PRIMARY KEY, b INTEGER) WITHOUT ROWID");
    create_table(
        &mut pager,
        &mut cat,
        "CREATE TABLE c(pref TEXT, note TEXT, FOREIGN KEY(pref) REFERENCES p(a) ON UPDATE CASCADE)",
    );

    // Seed under the default FK-OFF runtime: WR parent rows, then rowid child rows (two at
    // 'x', one at 'y').
    seed(
        &cat,
        &mut pager,
        "p",
        2,
        vec![vec![txt("x"), lit(Value::Integer(1))], vec![txt("y"), lit(Value::Integer(2))]],
    );
    seed(
        &cat,
        &mut pager,
        "c",
        2,
        vec![vec![txt("x"), txt("c1")], vec![txt("x"), txt("c2")], vec![txt("y"), txt("c3")]],
    );

    // UPDATE p SET a = 'z' WHERE a = 'x', foreign_keys ON: the two pref='x' children cascade
    // to 'z'.
    let mut rt = Runtime::new();
    rt.set_foreign_keys(true);
    run(
        &update_plan(
            "p",
            2,
            vec![(0, txt("z"))],
            filtered("p", 2, cmp(CmpOp::Eq, col(0), txt("x"))),
            affinities(&cat, "p"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    assert_eq!(rt.changes(), 1, "changes() counts only the parent row, not cascaded children");

    // Parent 'x' moved to 'z'; 'y' untouched. WR scan order by PK: 'y' then 'z'.
    let prows = scan_all(&cat, &mut pager, "p", 2);
    let pgot: Vec<(&str, i64)> = prows.iter().map(|r| (text(&r[0]), int(&r[1]))).collect();
    assert_eq!(pgot, vec![("y", 2), ("z", 1)], "parent PK moved x->z, y intact");

    // Exactly the two pref='x' children now read 'z'; pref='y' unchanged. A ROWID child scans
    // back with its columns at 0/1, so read columns 0 and 1.
    let crows = scan_all(&cat, &mut pager, "c", 2);
    let cgot: Vec<(&str, &str)> = crows.iter().map(|r| (text(&r[0]), text(&r[1]))).collect();
    assert_eq!(
        cgot,
        vec![("z", "c1"), ("z", "c2"), ("y", "c3")],
        "ON UPDATE CASCADE moved exactly the pref='x' children to 'z'"
    );
}
