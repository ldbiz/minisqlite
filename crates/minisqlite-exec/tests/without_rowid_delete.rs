//! Integration tests for `DELETE` over a WITHOUT ROWID (WR) table: hand-built [`Plan`]
//! trees run over a real [`MemPager`] + [`SchemaCatalog`] through the [`Executor`] /
//! [`RowCursor`] seam, then read back through the real storage/catalog (not a private
//! mock). A WR table stores each row in its PRIMARY KEY index b-tree at `root_page` (no
//! integer rowid — `withoutrowid.html`), so a DELETE removes the row's PRIMARY KEY entry.
//!
//! These pin the WR delete path end to end and DERIVE expected behavior from
//! `spec/sqlite-doc/lang_delete.html` + `withoutrowid.html`: (a) a filtered single-row
//! delete on a single-column TEXT PK, (b) a multi-row delete, (c) delete-all leaving an
//! empty PK b-tree, (d) a composite `PRIMARY KEY(c, a)`, (e) `RETURNING` the deleted
//! width-N row, (f) the `changes()` / `total_changes()` counters counting EXACTLY the rows
//! removed, (g) a WR table with a secondary (UNIQUE) index — the DELETE removes the row's
//! secondary-index entry as well as its PK entry (fileformat2 §2.5.1), (h) a WR table
//! carrying a DELETE trigger — the DELETE completes with the trigger attached and removes the
//! row (an empty-body trigger cannot prove firing), (i) a WR PARENT cascading `ON DELETE
//! CASCADE` into a ROWID child
//! under `PRAGMA foreign_keys=ON`, and (j) the fail-closed guard when a WR parent's FK child
//! is itself WITHOUT ROWID. (j) pins the remaining fail-closed guard and (i)/(j) the
//! `enforce_parent_delete` parent-side FK call the WR path makes — behavior the exec op adds,
//! so it is exercised, not assumed.
//!
//! Rows are seeded through the real `INSERT` operator (so the PK b-tree AND every secondary
//! index are populated exactly as production writes them). The FK cases (i)/(j) seed with the
//! pragma OFF (its default), so the child-side parent-key check is skipped and the row simply
//! lands; the DELETE then runs with `foreign_keys` ON to drive the parent-side action. DML
//! assumes an open write transaction (the engine opens one), so each apply is wrapped in
//! `begin`/`commit`; read-back scans need no transaction.

use minisqlite_btree::{init_database, IndexCursor};
use minisqlite_catalog::{Catalog, SchemaCatalog};
use minisqlite_exec::{Executor, PagerSet, Runtime, StreamingExecutor};
use minisqlite_expr::{CmpOp, CompareMeta, EvalExpr};
use minisqlite_pager::{MemPager, PageId, Pager};
use minisqlite_plan::{
    Delete, Insert, OnConflict, Plan, PlanNode, SeqScan, TriggerProgram, TriggerTiming,
};
use minisqlite_sql::{parse, Statement};
use minisqlite_types::{affinity_of_declared_type, Affinity, Collation, DbIndex, Error, Value};

// ----- fixtures ------------------------------------------------------------

/// A fresh in-memory database with one table created through the real catalog path.
fn db_with_table(create_sql: &str) -> (MemPager, SchemaCatalog) {
    let mut pager = MemPager::new(4096);
    init_database(&mut pager).unwrap();
    let mut cat = SchemaCatalog::new();
    create_table(&mut pager, &mut cat, create_sql);
    (pager, cat)
}

/// Create an ADDITIONAL table in an existing db through the real catalog path (used by the
/// FK cases, which need a parent AND a child in one database). A `FOREIGN KEY` referencing a
/// WITHOUT ROWID parent is recorded at DDL time and only validated when enforcement runs, so
/// this succeeds; the WR-parent/WR-child limits surface at DELETE time, not here.
fn create_table(pager: &mut MemPager, cat: &mut SchemaCatalog, create_sql: &str) {
    let ast = parse(create_sql).unwrap();
    let Statement::CreateTable(stmt) = &ast.statements[0] else {
        panic!("not a CREATE TABLE: {create_sql}");
    };
    cat.create_table(pager, stmt, create_sql).unwrap();
}

/// The per-column affinities a planner would attach, derived from the declared types.
fn affinities(cat: &SchemaCatalog, table: &str) -> Vec<Affinity> {
    cat.table(table)
        .unwrap()
        .unwrap()
        .columns
        .iter()
        .map(|c| affinity_of_declared_type(c.declared_type.as_deref()))
        .collect()
}

/// The WR table's PRIMARY KEY index b-tree root (the table's own `root_page`).
fn table_root(cat: &SchemaCatalog, table: &str) -> PageId {
    cat.table(table).unwrap().unwrap().root_page
}

/// The root page of the WR table's single secondary index (the auto-index a `UNIQUE`
/// column creates), so a test can count its entries at the storage layer and prove the
/// DELETE maintained it — independent of the read path.
fn index_root(cat: &SchemaCatalog, table: &str) -> PageId {
    cat.indexes_on(table).unwrap()[0].root_page
}

// ----- expression / node helpers -------------------------------------------

fn col(i: usize) -> EvalExpr {
    EvalExpr::Column(i)
}

fn lit(v: Value) -> EvalExpr {
    EvalExpr::Literal(v)
}

/// `left = right` under BINARY collation — the WHERE predicate for the filtered tests.
fn eq(left: EvalExpr, right: EvalExpr) -> EvalExpr {
    EvalExpr::Compare {
        op: CmpOp::Eq,
        null_safe: false,
        left: Box::new(left),
        right: Box::new(right),
        meta: CompareMeta { apply_left: None, apply_right: None, collation: Collation::Binary },
    }
}

fn seqscan(table: &str, n: usize) -> PlanNode {
    PlanNode::SeqScan(SeqScan { db: DbIndex::MAIN, table: table.to_string(), column_count: n })
}

/// A `Filter` node wrapping `input` with `predicate` (the residual WHERE a WR DELETE plan
/// carries — the planner keeps the whole WHERE as a residual `Filter` over the WR scan).
fn filter(input: PlanNode, predicate: EvalExpr) -> PlanNode {
    PlanNode::Filter { input: Box::new(input), predicate }
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

// ----- plan-building helpers -----------------------------------------------

/// An `INSERT` plan over a literal `VALUES` source, used to seed WR rows through the real
/// write path (so each row lands in the PRIMARY KEY b-tree exactly as production writes it).
fn insert_plan(table: &str, n: usize, rows: Vec<Vec<EvalExpr>>, affs: Vec<Affinity>) -> Plan {
    let source_width = rows.first().map(|r| r.len()).unwrap_or(n);
    Plan {
        root: PlanNode::Insert(Insert {
            db: DbIndex::MAIN,
            table: table.to_string(),
            column_count: n,
            columns: None,
            source: Box::new(PlanNode::Values { rows }),
            source_width,
            column_affinities: affs,
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
    }
}

/// A `DELETE` plan over a `scan` subtree (a WR `SeqScan`, or a `Filter` over one),
/// optionally with `RETURNING` bound over the width-N WR row `[c0..c_{N-1}]`.
fn delete_plan(table: &str, n: usize, scan: PlanNode, returning: Vec<EvalExpr>) -> Plan {
    Plan {
        root: PlanNode::Delete(Delete {
            db: DbIndex::MAIN,
            table: table.to_string(),
            column_count: n,
            scan: Box::new(scan),
            returning,
            triggers: Vec::new(),
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

/// A `DELETE` plan whose `triggers` list is NON-EMPTY — the shape a WR DELETE fires its
/// trigger through. One `AFTER` program with an empty action list is enough to drive the
/// firing path (`effective_triggers` returns the borrowed non-empty slice; the WR op builds
/// the width-N `OLD` frame via `build_frame_wr` and fires it — an empty body is a no-op, so
/// the delete completes). This is how a top-level DELETE plan carries the target's own DELETE
/// triggers (the planner attaches them), so a real `CREATE TRIGGER ... AFTER DELETE ON
/// <wr_table>` reaches the executor exactly this shape.
fn delete_plan_with_trigger(table: &str, n: usize, scan: PlanNode) -> Plan {
    Plan {
        root: PlanNode::Delete(Delete {
            db: DbIndex::MAIN,
            table: table.to_string(),
            column_count: n,
            scan: Box::new(scan),
            returning: Vec::new(),
            triggers: vec![TriggerProgram {
                name: "trg".into(),
                timing: TriggerTiming::After,
                when: None,
                actions: Vec::new(),
            }],
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

// ----- runners -------------------------------------------------------------

/// Apply a DML plan inside a write transaction, collecting any `RETURNING` rows and
/// threading the caller's `Runtime` so its change counters can be read afterward.
fn run_dml(plan: &Plan, cat: &SchemaCatalog, pager: &mut MemPager, rt: &mut Runtime) -> Vec<Vec<Value>> {
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

/// Apply a DML plan expected to FAIL under a fresh FK-OFF [`Runtime`]. Mirrors the engine's
/// abort path: the statement is rolled back and the error returned, so a fail-closed guard
/// can be asserted instead of unwrapped into a panic. Delegates to [`run_dml_err_rt`] — the
/// FK-enforcement cases thread their own FK-ON `Runtime` through that directly.
fn run_dml_err(plan: &Plan, cat: &SchemaCatalog, pager: &mut MemPager) -> Error {
    run_dml_err_rt(plan, cat, pager, &mut Runtime::new())
}

/// Like [`run_dml_err`] but threads a caller-configured [`Runtime`] (e.g. one with
/// `PRAGMA foreign_keys` ON) so an FK-enforcement fail-closed path can be asserted — the
/// no-arg `run_dml_err` builds a fresh FK-OFF `Runtime`, under which the FK path is a no-op.
fn run_dml_err_rt(
    plan: &Plan,
    cat: &SchemaCatalog,
    pager: &mut MemPager,
    rt: &mut Runtime,
) -> Error {
    pager.begin().unwrap();
    let mut exec = StreamingExecutor;
    let mut err = None;
    {
        match exec.execute(plan, cat, PagerSet::One { db: DbIndex::MAIN, pager: &mut *pager }) {
            Ok(mut cur) => loop {
                match cur.next_row(rt) {
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
    err.expect("expected the DELETE to error")
}

/// Full WR table scan back through the executor: rows are width-N `[c0..c_{N-1}]` (NO
/// trailing rowid), in PRIMARY KEY (storage) order.
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
    let mut cur = exec
        .execute(&plan, cat, PagerSet::One { db: DbIndex::MAIN, pager: &mut *pager })
        .unwrap();
    let mut out = Vec::new();
    while let Some(row) = cur.next_row(&mut rt).unwrap() {
        out.push(row);
    }
    out
}

/// Count the entries in the PRIMARY KEY b-tree directly (independent of the scan operator),
/// so "the store is empty/consistent" is proven at the storage layer, not just via a re-scan.
fn pk_entry_count(pager: &MemPager, root: PageId) -> usize {
    let mut cur = IndexCursor::open(pager, root).unwrap();
    let mut count = 0;
    if cur.first().unwrap() {
        loop {
            count += 1;
            if !cur.next().unwrap() {
                break;
            }
        }
    }
    count
}

// ----- (a) single-column TEXT PK: a filtered single-row delete ---------------

#[test]
fn delete_one_row_by_text_pk_keeps_the_others() {
    // DELETE FROM t WHERE a = 'y' removes exactly the 'y' row; 'x' and 'z' survive, in
    // PRIMARY KEY order. The WR scan row is width-N `[a, b]` (no rowid), so the WHERE and
    // the store read back at width 2.
    let (mut pager, cat) =
        db_with_table("CREATE TABLE t(a TEXT PRIMARY KEY, b INTEGER) WITHOUT ROWID");
    let mut seed_rt = Runtime::new();
    run_dml(
        &insert_plan(
            "t",
            2,
            vec![
                vec![lit(Value::Text("x".into())), lit(Value::Integer(1))],
                vec![lit(Value::Text("y".into())), lit(Value::Integer(2))],
                vec![lit(Value::Text("z".into())), lit(Value::Integer(3))],
            ],
            affinities(&cat, "t"),
        ),
        &cat,
        &mut pager,
        &mut seed_rt,
    );
    assert_eq!(scan_all(&cat, &mut pager, "t", 2).len(), 3, "seeded three WR rows");

    let scan = filter(seqscan("t", 2), eq(col(0), lit(Value::Text("y".into()))));
    let mut del_rt = Runtime::new();
    let out = run_dml(&delete_plan("t", 2, scan, Vec::new()), &cat, &mut pager, &mut del_rt);
    assert!(out.is_empty(), "no RETURNING clause yields no rows");
    assert_eq!(del_rt.changes(), 1, "exactly the a='y' row matched");

    let rows = scan_all(&cat, &mut pager, "t", 2);
    let got: Vec<(&str, i64)> = rows.iter().map(|r| (text(&r[0]), int(&r[1]))).collect();
    assert_eq!(got, vec![("x", 1), ("z", 3)], "the a='y' row is gone; the rest remain in PK order");
    // The store itself holds exactly two PK entries — the deleted key was really removed.
    assert_eq!(pk_entry_count(&pager, table_root(&cat, "t")), 2);
}

// ----- (b) delete multiple matching rows -----------------------------------

#[test]
fn delete_multiple_matching_rows() {
    // DELETE FROM t WHERE b = 1 matches two of three rows; both are removed and counted,
    // the non-matching row survives.
    let (mut pager, cat) =
        db_with_table("CREATE TABLE t(a TEXT PRIMARY KEY, b INTEGER) WITHOUT ROWID");
    let mut seed_rt = Runtime::new();
    run_dml(
        &insert_plan(
            "t",
            2,
            vec![
                vec![lit(Value::Text("x".into())), lit(Value::Integer(1))],
                vec![lit(Value::Text("y".into())), lit(Value::Integer(1))],
                vec![lit(Value::Text("z".into())), lit(Value::Integer(2))],
            ],
            affinities(&cat, "t"),
        ),
        &cat,
        &mut pager,
        &mut seed_rt,
    );

    let scan = filter(seqscan("t", 2), eq(col(1), lit(Value::Integer(1))));
    let mut del_rt = Runtime::new();
    run_dml(&delete_plan("t", 2, scan, Vec::new()), &cat, &mut pager, &mut del_rt);
    assert_eq!(del_rt.changes(), 2, "both b=1 rows were deleted");

    let rows = scan_all(&cat, &mut pager, "t", 2);
    let got: Vec<(&str, i64)> = rows.iter().map(|r| (text(&r[0]), int(&r[1]))).collect();
    assert_eq!(got, vec![("z", 2)], "only the b=2 row remains");
    assert_eq!(pk_entry_count(&pager, table_root(&cat, "t")), 1);
}

// ----- (c) delete all rows: empty PK b-tree afterward ----------------------

#[test]
fn delete_all_empties_the_pk_btree() {
    // DELETE FROM t (a full WR SeqScan, no filter) removes every row; the scan reads back
    // empty AND the PRIMARY KEY b-tree holds zero entries (proven directly, not only via
    // the scan operator).
    let (mut pager, cat) =
        db_with_table("CREATE TABLE t(a TEXT PRIMARY KEY, b INTEGER) WITHOUT ROWID");
    let mut seed_rt = Runtime::new();
    run_dml(
        &insert_plan(
            "t",
            2,
            vec![
                vec![lit(Value::Text("x".into())), lit(Value::Integer(1))],
                vec![lit(Value::Text("y".into())), lit(Value::Integer(2))],
                vec![lit(Value::Text("z".into())), lit(Value::Integer(3))],
            ],
            affinities(&cat, "t"),
        ),
        &cat,
        &mut pager,
        &mut seed_rt,
    );

    let mut del_rt = Runtime::new();
    run_dml(&delete_plan("t", 2, seqscan("t", 2), Vec::new()), &cat, &mut pager, &mut del_rt);
    assert_eq!(del_rt.changes(), 3, "all three rows deleted");
    assert!(scan_all(&cat, &mut pager, "t", 2).is_empty(), "the table scans back empty");
    assert_eq!(pk_entry_count(&pager, table_root(&cat, "t")), 0, "the PK b-tree holds no entries");
}

// ----- (d) composite PRIMARY KEY(c, a): the correct row is removed ---------

#[test]
fn delete_composite_pk_removes_the_correct_row() {
    // PRIMARY KEY(c, a): storage/scan order is by (c, a). Rows (schema `[a, b, c]`):
    //   (10, 100, 1) -> PK (c=1, a=10)
    //   (20, 200, 1) -> PK (c=1, a=20)
    //   (10, 300, 2) -> PK (c=2, a=10)
    // DELETE the row with a=10 AND c=1 (nested filters, avoiding an AND expr): only PK
    // (1, 10) is removed. The other two survive, in PK (c, a) order: (1,20) then (2,10).
    let (mut pager, cat) = db_with_table(
        "CREATE TABLE t(a INTEGER, b INTEGER, c INTEGER, PRIMARY KEY(c, a)) WITHOUT ROWID",
    );
    let mut seed_rt = Runtime::new();
    run_dml(
        &insert_plan(
            "t",
            3,
            vec![
                vec![lit(Value::Integer(10)), lit(Value::Integer(100)), lit(Value::Integer(1))],
                vec![lit(Value::Integer(20)), lit(Value::Integer(200)), lit(Value::Integer(1))],
                vec![lit(Value::Integer(10)), lit(Value::Integer(300)), lit(Value::Integer(2))],
            ],
            affinities(&cat, "t"),
        ),
        &cat,
        &mut pager,
        &mut seed_rt,
    );

    // WHERE a = 10 AND c = 1, expressed as two stacked residual Filters over the WR scan.
    let scan = filter(
        filter(seqscan("t", 3), eq(col(0), lit(Value::Integer(10)))),
        eq(col(2), lit(Value::Integer(1))),
    );
    let mut del_rt = Runtime::new();
    run_dml(&delete_plan("t", 3, scan, Vec::new()), &cat, &mut pager, &mut del_rt);
    assert_eq!(del_rt.changes(), 1, "exactly the (c=1, a=10) row matched");

    let rows = scan_all(&cat, &mut pager, "t", 3);
    let got: Vec<(i64, i64, i64)> =
        rows.iter().map(|r| (int(&r[0]), int(&r[1]), int(&r[2]))).collect();
    assert_eq!(
        got,
        vec![(20, 200, 1), (10, 300, 2)],
        "the (1,10) key is removed; the survivors read back in PK (c, a) order"
    );
    assert_eq!(pk_entry_count(&pager, table_root(&cat, "t")), 2);
}

// ----- (e) RETURNING the deleted width-N row -------------------------------

#[test]
fn delete_returning_yields_the_deleted_row_columns() {
    // RETURNING a, b over a WR DELETE binds against the width-N row `[a, b]` (no rowid
    // register). Deleting a='x' returns exactly ('x', 1); the a='y' row is untouched.
    let (mut pager, cat) =
        db_with_table("CREATE TABLE t(a TEXT PRIMARY KEY, b INTEGER) WITHOUT ROWID");
    let mut seed_rt = Runtime::new();
    run_dml(
        &insert_plan(
            "t",
            2,
            vec![
                vec![lit(Value::Text("x".into())), lit(Value::Integer(1))],
                vec![lit(Value::Text("y".into())), lit(Value::Integer(2))],
            ],
            affinities(&cat, "t"),
        ),
        &cat,
        &mut pager,
        &mut seed_rt,
    );

    let scan = filter(seqscan("t", 2), eq(col(0), lit(Value::Text("x".into()))));
    let mut del_rt = Runtime::new();
    let out = run_dml(
        &delete_plan("t", 2, scan, vec![col(0), col(1)]),
        &cat,
        &mut pager,
        &mut del_rt,
    );
    assert_eq!(out.len(), 1, "one row deleted yields one RETURNING row");
    assert_eq!((text(&out[0][0]), int(&out[0][1])), ("x", 1), "RETURNING is the deleted row's columns");
    assert_eq!(out[0].len(), 2, "the WR RETURNING row is width N, no trailing rowid");
    assert_eq!(del_rt.changes(), 1);

    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1, "only the a='y' row remains");
    assert_eq!((text(&rows[0][0]), int(&rows[0][1])), ("y", 2));
}

// ----- (f) changes()/total_changes() count EXACTLY the rows removed --------

#[test]
fn change_counters_count_exactly_the_rows_removed() {
    // A partial-match DELETE bumps both `changes()` and `total_changes()` by EXACTLY the
    // number of rows actually removed (2 of 3), never by the rows scanned. A follow-up
    // DELETE matching nothing leaves a fresh counter at 0 and removes no row.
    let (mut pager, cat) =
        db_with_table("CREATE TABLE t(a TEXT PRIMARY KEY, b INTEGER) WITHOUT ROWID");
    let mut seed_rt = Runtime::new();
    run_dml(
        &insert_plan(
            "t",
            2,
            vec![
                vec![lit(Value::Text("x".into())), lit(Value::Integer(1))],
                vec![lit(Value::Text("y".into())), lit(Value::Integer(1))],
                vec![lit(Value::Text("z".into())), lit(Value::Integer(2))],
            ],
            affinities(&cat, "t"),
        ),
        &cat,
        &mut pager,
        &mut seed_rt,
    );

    // Delete the two b=1 rows under a fresh Runtime: both counters are exactly 2.
    let scan = filter(seqscan("t", 2), eq(col(1), lit(Value::Integer(1))));
    let mut del_rt = Runtime::new();
    run_dml(&delete_plan("t", 2, scan, Vec::new()), &cat, &mut pager, &mut del_rt);
    assert_eq!(del_rt.changes(), 2, "changes() == rows removed");
    assert_eq!(del_rt.total_changes(), 2, "total_changes() == rows removed");

    // A DELETE matching nothing (b=999) removes nothing and counts nothing.
    let miss = filter(seqscan("t", 2), eq(col(1), lit(Value::Integer(999))));
    let mut miss_rt = Runtime::new();
    run_dml(&delete_plan("t", 2, miss, Vec::new()), &cat, &mut pager, &mut miss_rt);
    assert_eq!(miss_rt.changes(), 0, "a whole-miss delete counts nothing");
    assert_eq!(miss_rt.total_changes(), 0);
    assert_eq!(scan_all(&cat, &mut pager, "t", 2).len(), 1, "only the untouched b=2 row remains");
}

// ----- (g) a WR table with a secondary (UNIQUE) index: DELETE maintains it -

#[test]
fn delete_on_wr_table_maintains_secondary_index() {
    // A `UNIQUE` column constraint on a WR table creates a secondary index keyed by
    // `[indexed cols.., trailing PK..]` (fileformat2 §2.5.1). A DELETE must remove the row's
    // secondary-index entry as well as its PK-b-tree row — proven at the storage layer (the
    // index entry count drops in lockstep with the PK count) AND behaviorally (re-inserting
    // the just-freed `b` value succeeds under the UNIQUE index, which is only possible if the
    // old entry was actually removed; a still-present duplicate would collide).
    let (mut pager, cat) =
        db_with_table("CREATE TABLE t(a TEXT PRIMARY KEY, b INTEGER UNIQUE) WITHOUT ROWID");
    let root = table_root(&cat, "t");
    let idx_root = index_root(&cat, "t");

    // Seed two rows through the REAL INSERT operator — WR secondary-index maintenance now
    // populates the auto-index exactly as production does.
    let mut seed_rt = Runtime::new();
    run_dml(
        &insert_plan(
            "t",
            2,
            vec![
                vec![lit(Value::Text("x".into())), lit(Value::Integer(1))],
                vec![lit(Value::Text("y".into())), lit(Value::Integer(2))],
            ],
            affinities(&cat, "t"),
        ),
        &cat,
        &mut pager,
        &mut seed_rt,
    );
    assert_eq!(pk_entry_count(&pager, root), 2, "two rows before the delete");
    assert_eq!(pk_entry_count(&pager, idx_root), 2, "two secondary-index entries before the delete");

    // DELETE FROM t WHERE a = 'x'.
    let mut del_rt = Runtime::new();
    let scan = filter(seqscan("t", 2), eq(col(0), lit(Value::Text("x".into()))));
    run_dml(&delete_plan("t", 2, scan, Vec::new()), &cat, &mut pager, &mut del_rt);
    assert_eq!(del_rt.changes(), 1, "one row deleted");
    assert_eq!(pk_entry_count(&pager, root), 1, "the PK row is gone");
    assert_eq!(pk_entry_count(&pager, idx_root), 1, "the secondary-index entry is gone too");

    // Only the untouched b=2 row remains, intact.
    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1, "one row survives");
    assert_eq!((text(&rows[0][0]), int(&rows[0][1])), ("y", 2));

    // The removed entry is truly gone: re-inserting b=1 (freed by the delete) succeeds.
    let mut reinsert_rt = Runtime::new();
    run_dml(
        &insert_plan(
            "t",
            2,
            vec![vec![lit(Value::Text("z".into())), lit(Value::Integer(1))]],
            affinities(&cat, "t"),
        ),
        &cat,
        &mut pager,
        &mut reinsert_rt,
    );
    assert_eq!(pk_entry_count(&pager, idx_root), 2, "re-inserting the freed b=1 succeeds");

    // The surviving b=2 entry is still enforced: a duplicate b=2 insert errors UNIQUE.
    let dup = run_dml_err(
        &insert_plan(
            "t",
            2,
            vec![vec![lit(Value::Text("w".into())), lit(Value::Integer(2))]],
            affinities(&cat, "t"),
        ),
        &cat,
        &mut pager,
    );
    match dup {
        Error::Constraint(m) => {
            assert!(m.contains("UNIQUE constraint failed"), "names the UNIQUE violation, got {m:?}");
        }
        other => panic!("expected a UNIQUE constraint error, got {other:?}"),
    }
}

// ----- (h) a WR table carrying a DELETE trigger: the DELETE completes -------

#[test]
fn delete_on_wr_table_completes_with_a_trigger_attached() {
    // A WR table can carry DELETE triggers. This pins that the WR DELETE path COMPLETES with a
    // trigger ATTACHED — it does not fail closed: the plan carries one `AFTER DELETE` program
    // and the DELETE runs to completion, removing the PK-b-tree row. The trigger body is EMPTY,
    // whose observable effect is identical whether the trigger fires or is skipped, so this
    // test proves completion-with-a-trigger, NOT that the trigger actually fires. Real WR
    // trigger firing (width-N `OLD` binding, a non-empty body) is covered by
    // `crates/minisqlite/tests/conformance_wr_triggers.rs`.
    let (mut pager, cat) =
        db_with_table("CREATE TABLE t(a TEXT PRIMARY KEY, b INTEGER) WITHOUT ROWID");
    let mut seed_rt = Runtime::new();
    run_dml(
        &insert_plan(
            "t",
            2,
            vec![vec![lit(Value::Text("x".into())), lit(Value::Integer(1))]],
            affinities(&cat, "t"),
        ),
        &cat,
        &mut pager,
        &mut seed_rt,
    );
    assert_eq!(pk_entry_count(&pager, table_root(&cat, "t")), 1, "one row present before the delete");

    let mut rt = Runtime::new();
    run_dml(&delete_plan_with_trigger("t", 2, seqscan("t", 2)), &cat, &mut pager, &mut rt);
    assert_eq!(rt.changes(), 1, "the WR DELETE completed with a trigger attached and removed the row");

    // The row is gone from both the PK b-tree and the scan (no silent skip, no fail-closed).
    assert_eq!(pk_entry_count(&pager, table_root(&cat, "t")), 0, "the DELETE removed the row");
    assert!(scan_all(&cat, &mut pager, "t", 2).is_empty(), "the table is empty after the delete");
}

// ----- (i) FK: a WR PARENT cascades ON DELETE CASCADE into a ROWID child ----

#[test]
fn delete_on_wr_parent_cascades_to_rowid_child() {
    // The WR delete path calls `enforce_parent_delete` (parent side) exactly as the rowid
    // path does. With `PRAGMA foreign_keys` ON, deleting a WR parent row must CASCADE into a
    // ROWID child that references it: the child's matching rows are removed, the non-matching
    // one survives, and — since `sqlite3_changes()` excludes FK-action rows — only the parent
    // delete advances `changes()` (the cascade does not). This pins the parent-side FK
    // behavior the WR path adds; a wrong `logical` slice/arg would drop or mis-target it.
    let (mut pager, mut cat) =
        db_with_table("CREATE TABLE p(a TEXT PRIMARY KEY, b INTEGER) WITHOUT ROWID");
    create_table(
        &mut pager,
        &mut cat,
        "CREATE TABLE c(pref TEXT, note TEXT, FOREIGN KEY(pref) REFERENCES p(a) ON DELETE CASCADE)",
    );

    // Seed under the default FK-OFF runtime: the parent WR rows, then child rows referencing
    // them (two point at 'x', one at 'y').
    let mut seed_rt = Runtime::new();
    run_dml(
        &insert_plan(
            "p",
            2,
            vec![
                vec![lit(Value::Text("x".into())), lit(Value::Integer(1))],
                vec![lit(Value::Text("y".into())), lit(Value::Integer(2))],
            ],
            affinities(&cat, "p"),
        ),
        &cat,
        &mut pager,
        &mut seed_rt,
    );
    run_dml(
        &insert_plan(
            "c",
            2,
            vec![
                vec![lit(Value::Text("x".into())), lit(Value::Text("c1".into()))],
                vec![lit(Value::Text("x".into())), lit(Value::Text("c2".into()))],
                vec![lit(Value::Text("y".into())), lit(Value::Text("c3".into()))],
            ],
            affinities(&cat, "c"),
        ),
        &cat,
        &mut pager,
        &mut seed_rt,
    );

    // DELETE FROM p WHERE a = 'x', foreign_keys ON: the two pref='x' children cascade away.
    let mut del_rt = Runtime::new();
    del_rt.set_foreign_keys(true);
    let scan = filter(seqscan("p", 2), eq(col(0), lit(Value::Text("x".into()))));
    run_dml(&delete_plan("p", 2, scan, Vec::new()), &cat, &mut pager, &mut del_rt);
    assert_eq!(del_rt.changes(), 1, "changes() counts only the parent row, not cascaded children");

    // The parent 'x' row is gone; 'y' survives (WR width-2 rows, PK order).
    let prows = scan_all(&cat, &mut pager, "p", 2);
    let pgot: Vec<(&str, i64)> = prows.iter().map(|r| (text(&r[0]), int(&r[1]))).collect();
    assert_eq!(pgot, vec![("y", 2)], "only the p.a='y' WR row remains");
    assert_eq!(pk_entry_count(&pager, table_root(&cat, "p")), 1);

    // Exactly the two pref='x' child rows cascaded; only pref='y' survives (a ROWID child
    // scans back width-3 `[pref, note, rowid]`, so read columns 0 and 1).
    let crows = scan_all(&cat, &mut pager, "c", 2);
    let cgot: Vec<(&str, &str)> = crows.iter().map(|r| (text(&r[0]), text(&r[1]))).collect();
    assert_eq!(cgot, vec![("y", "c3")], "ON DELETE CASCADE removed exactly the pref='x' children");
}

// ----- (j) FK fail-closed: a WR parent whose FK child is itself WITHOUT ROWID

#[test]
fn delete_on_wr_parent_with_wr_child_fk_fails_loud() {
    // `enforce_parent_delete` cannot apply an IMMEDIATE ON DELETE action to a WITHOUT ROWID
    // child (its rows live in a PK-index b-tree the rowid child-scan cannot walk), so it fails
    // LOUD rather than silently orphaning it. Driven through the WR parent delete path with
    // foreign_keys ON, the DELETE must error and leave the parent row in place (fail-closed,
    // rolled back — not a silent partial). This pins the WR-child branch of the FK call.
    let (mut pager, mut cat) =
        db_with_table("CREATE TABLE p(a TEXT PRIMARY KEY, b INTEGER) WITHOUT ROWID");
    create_table(
        &mut pager,
        &mut cat,
        "CREATE TABLE c(k TEXT PRIMARY KEY, pref TEXT, \
         FOREIGN KEY(pref) REFERENCES p(a) ON DELETE CASCADE) WITHOUT ROWID",
    );

    let mut seed_rt = Runtime::new();
    run_dml(
        &insert_plan(
            "p",
            2,
            vec![vec![lit(Value::Text("x".into())), lit(Value::Integer(1))]],
            affinities(&cat, "p"),
        ),
        &cat,
        &mut pager,
        &mut seed_rt,
    );
    run_dml(
        &insert_plan(
            "c",
            2,
            vec![vec![lit(Value::Text("ck".into())), lit(Value::Text("x".into()))]],
            affinities(&cat, "c"),
        ),
        &cat,
        &mut pager,
        &mut seed_rt,
    );

    // DELETE FROM p WHERE a = 'x', foreign_keys ON: the WR child cannot be enforced -> loud.
    let mut del_rt = Runtime::new();
    del_rt.set_foreign_keys(true);
    let scan = filter(seqscan("p", 2), eq(col(0), lit(Value::Text("x".into()))));
    let err = run_dml_err_rt(&delete_plan("p", 2, scan, Vec::new()), &cat, &mut pager, &mut del_rt);
    match err {
        Error::Sql(m) => {
            assert!(m.contains("WITHOUT ROWID"), "the guard names WITHOUT ROWID, got {m:?}");
            assert!(
                m.contains("FOREIGN KEY") || m.contains("child"),
                "the guard names the FK/child limitation, got {m:?}"
            );
        }
        other => panic!("expected a fail-closed Sql error, got {other:?}"),
    }
    // Fail-closed + rolled back: the parent WR row is untouched.
    assert_eq!(pk_entry_count(&pager, table_root(&cat, "p")), 1, "the parent row survives");
    let prows = scan_all(&cat, &mut pager, "p", 2);
    assert_eq!(prows.len(), 1, "the parent WR row remains after the refused DELETE");
    assert_eq!((text(&prows[0][0]), int(&prows[0][1])), ("x", 1));
}
