//! Integration tests for `INSERT OR REPLACE` (SQLite's REPLACE conflict resolution),
//! run end to end: a hand-built [`Plan`] over a real [`MemPager`] + [`SchemaCatalog`]
//! through the [`Executor`]/[`RowCursor`] seam, then read back through the real
//! storage/catalog (a full scan and a walk of each index b-tree — not a private mock).
//!
//! The contract these pin, straight from `spec/sqlite-doc/lang_conflict.html`: when a
//! UNIQUE or PRIMARY KEY (rowid) violation occurs, REPLACE "deletes pre-existing rows
//! that are causing the constraint violation prior to inserting ... the current row" —
//! EVERY conflicting row, and each row's TABLE row AND all its index entries, so no
//! stale or missing index entry survives. "Nor does REPLACE increment the change
//! counter" for the rows it deletes, so a single-row REPLACE reports `changes() == 1`.
//!
//! Helpers are copied (not shared) because each integration-test file is its own crate;
//! `tests/insert.rs` holds the sibling copies for the plain-INSERT behaviors.

use minisqlite_btree::{init_database, IndexCursor};
use minisqlite_catalog::{Catalog, SchemaCatalog};
use minisqlite_exec::{Executor, Runtime, StreamingExecutor};
use minisqlite_expr::EvalExpr;
use minisqlite_fileformat::decode_record;
use minisqlite_pager::{MemPager, PageId, Pager};
use minisqlite_plan::{Insert, OnConflict, Plan, PlanNode, SeqScan};
use minisqlite_sql::{parse, Statement};
use minisqlite_types::{affinity_of_declared_type, Affinity, Error, Value};
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

/// Create an index through the real catalog path (persists a real b-tree).
fn create_index(pager: &mut MemPager, cat: &mut SchemaCatalog, sql: &str) {
    let ast = parse(sql).unwrap();
    let Statement::CreateIndex(stmt) = &ast.statements[0] else {
        panic!("not a CREATE INDEX: {sql}");
    };
    cat.create_index(pager, stmt, sql).unwrap();
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

fn lit(v: Value) -> EvalExpr {
    EvalExpr::Literal(v)
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

fn is_null(v: &Value) -> bool {
    matches!(v, Value::Null)
}

/// Build an `INSERT` plan over a literal `VALUES` source. `source_width` is inferred
/// from the first row (all `VALUES` rows are equal width).
fn insert_plan(
    table: &str,
    n: usize,
    rows: Vec<Vec<EvalExpr>>,
    column_affinities: Vec<Affinity>,
    on_conflict: OnConflict,
) -> Plan {
    let source_width = rows.first().map(|r| r.len()).unwrap_or(n);
    Plan {
        root: PlanNode::Insert(Insert { db: DbIndex::MAIN,
            table: table.to_string(),
            column_count: n,
            columns: None,
            source: Box::new(PlanNode::Values { rows }),
            source_width,
            column_affinities,
            on_conflict,
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

/// Apply an `INSERT` inside a write transaction, threading `rt` so the change counters
/// can be read afterward.
fn run_insert(plan: &Plan, cat: &SchemaCatalog, pager: &mut MemPager, rt: &mut Runtime) {
    pager.begin().unwrap();
    let mut exec = StreamingExecutor;
    {
        let mut cur = exec.execute(plan, cat, PagerSet::One { db: DbIndex::MAIN, pager: &mut *pager }).unwrap();
        while cur.next_row(rt).unwrap().is_some() {}
    }
    pager.commit().unwrap();
}

/// Like [`run_insert`] but for the error path: rolls the failed statement back (as the
/// engine would on a constraint violation) and returns the error.
fn run_insert_err(plan: &Plan, cat: &SchemaCatalog, pager: &mut MemPager) -> Error {
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
    err.expect("expected the INSERT to error")
}

/// Full table scan back through the executor, rows in base-table shape
/// `[c0, …, c_{N-1}, rowid]`, sorted by rowid so assertions are order-independent.
fn scan_sorted(cat: &SchemaCatalog, pager: &mut MemPager, table: &str, n: usize) -> Vec<Vec<Value>> {
    let plan = Plan {
        root: PlanNode::SeqScan(SeqScan { db: DbIndex::MAIN, table: table.to_string(), column_count: n }),
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
    out.sort_by_key(|r| int(&r[n]));
    out
}

/// Every entry in an index b-tree (decoded key records), in index order. Walking the
/// real b-tree is the point: it proves the index holds EXACTLY the surviving entries,
/// catching both a stale entry left behind and a live entry that went missing.
fn index_keys(pager: &MemPager, root: PageId) -> Vec<Vec<Value>> {
    let mut cur = IndexCursor::open(pager, root).unwrap();
    let mut out = Vec::new();
    if cur.first().unwrap() {
        loop {
            out.push(decode_record(&cur.key().unwrap()));
            if !cur.next().unwrap() {
                break;
            }
        }
    }
    out
}

fn index_root(cat: &SchemaCatalog, name: &str) -> PageId {
    cat.index(name).unwrap().unwrap().root_page
}

// ----- (1) UNIQUE conflict at a DIFFERENT rowid: delete the old row --------

#[test]
fn replace_on_unique_index_conflict_at_different_rowid_deletes_old_row() {
    // Row 1 owns b='x'. INSERT OR REPLACE at the fresh rowid 2 with b='x' collides on
    // the UNIQUE index with row 1 — REPLACE deletes row 1 (table row + index entry),
    // then inserts row 2. Result: only (2,'x') survives and the index points 'x' -> 2.
    let (mut pager, mut cat) = db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT)");
    create_index(&mut pager, &mut cat, "CREATE UNIQUE INDEX ub ON t(b)");
    let mut rt = Runtime::new();

    run_insert(
        &insert_plan(
            "t",
            2,
            vec![vec![lit(Value::Integer(1)), lit(Value::Text("x".into()))]],
            affinities(&cat, "t"),
            OnConflict::Abort,
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    run_insert(
        &insert_plan(
            "t",
            2,
            vec![vec![lit(Value::Integer(2)), lit(Value::Text("x".into()))]],
            affinities(&cat, "t"),
            OnConflict::Replace,
        ),
        &cat,
        &mut pager,
        &mut rt,
    );

    let rows = scan_sorted(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1, "old row 1 was deleted, only the replacement remains");
    assert_eq!((int(&rows[0][2]), text(&rows[0][1])), (2, "x"), "'x' now lives at rowid 2");

    let keys = index_keys(&pager, index_root(&cat, "ub"));
    assert_eq!(keys.len(), 1, "exactly one unique-index entry — the stale ['x',1] is gone");
    assert_eq!((text(&keys[0][0]), int(&keys[0][1])), ("x", 2));
}

// ----- (2) rowid conflict: the stale secondary-index entry is removed ------

#[test]
fn replace_on_rowid_conflict_updates_secondary_index() {
    // The rowid-conflict stale-index fix. Row 1 has b='x'. INSERT OR REPLACE at rowid 1
    // with b='y' must not merely overwrite the table row (which would leave the index
    // entry ['x',1] pointing at a row whose value is now 'y'): it deletes the old row's
    // index entry first, so the index ends up with only ['y',1].
    let (mut pager, mut cat) = db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT)");
    create_index(&mut pager, &mut cat, "CREATE INDEX ib ON t(b)");
    let mut rt = Runtime::new();

    run_insert(
        &insert_plan(
            "t",
            2,
            vec![vec![lit(Value::Integer(1)), lit(Value::Text("x".into()))]],
            affinities(&cat, "t"),
            OnConflict::Abort,
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    run_insert(
        &insert_plan(
            "t",
            2,
            vec![vec![lit(Value::Integer(1)), lit(Value::Text("y".into()))]],
            affinities(&cat, "t"),
            OnConflict::Replace,
        ),
        &cat,
        &mut pager,
        &mut rt,
    );

    let rows = scan_sorted(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1, "the row was replaced in place, not duplicated");
    assert_eq!((int(&rows[0][2]), text(&rows[0][1])), (1, "y"), "rowid 1 now holds 'y'");

    let keys = index_keys(&pager, index_root(&cat, "ib"));
    assert_eq!(keys.len(), 1, "no stale ['x',1] entry survives the replace");
    assert_eq!((text(&keys[0][0]), int(&keys[0][1])), ("y", 1), "index now points 'y' -> 1");
}

// ----- (3) one REPLACE can delete several rows (two unique indexes) --------

#[test]
fn replace_deletes_multiple_conflicting_rows_across_two_unique_indexes() {
    // The new row (3,'x','q') collides with row 1 on unique index ub (b='x') AND with
    // row 2 on unique index uc (c='q'). SQLite deletes BOTH conflicting rows, then
    // inserts. Result: only (3,'x','q') survives; each unique index has exactly one
    // entry, both pointing at rowid 3 (no stale ['x',1] / ['q',2]).
    let (mut pager, mut cat) = db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT, c TEXT)");
    create_index(&mut pager, &mut cat, "CREATE UNIQUE INDEX ub ON t(b)");
    create_index(&mut pager, &mut cat, "CREATE UNIQUE INDEX uc ON t(c)");
    let mut rt = Runtime::new();

    run_insert(
        &insert_plan(
            "t",
            3,
            vec![
                vec![lit(Value::Integer(1)), lit(Value::Text("x".into())), lit(Value::Text("p".into()))],
                vec![lit(Value::Integer(2)), lit(Value::Text("y".into())), lit(Value::Text("q".into()))],
            ],
            affinities(&cat, "t"),
            OnConflict::Abort,
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    let before = rt.changes();
    run_insert(
        &insert_plan(
            "t",
            3,
            vec![vec![
                lit(Value::Integer(3)),
                lit(Value::Text("x".into())),
                lit(Value::Text("q".into())),
            ]],
            affinities(&cat, "t"),
            OnConflict::Replace,
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    assert_eq!(
        rt.changes(),
        before + 1,
        "deleting two conflicting rows then inserting one counts as ONE change (the \
         implicit deletes are not counted)",
    );

    let rows = scan_sorted(&cat, &mut pager, "t", 3);
    assert_eq!(rows.len(), 1, "both conflicting rows (1 and 2) were deleted");
    // Row shape is [a, b, c, rowid]; the alias column a reads back as the rowid.
    assert_eq!(int(&rows[0][3]), 3, "the surviving rowid is 3");
    assert_eq!(text(&rows[0][1]), "x", "surviving row has b='x'");
    assert_eq!(text(&rows[0][2]), "q", "surviving row has c='q'");

    let ub = index_keys(&pager, index_root(&cat, "ub"));
    assert_eq!(ub.len(), 1, "ub holds exactly one entry (stale ['x',1] gone)");
    assert_eq!((text(&ub[0][0]), int(&ub[0][1])), ("x", 3));

    let uc = index_keys(&pager, index_root(&cat, "uc"));
    assert_eq!(uc.len(), 1, "uc holds exactly one entry (stale ['q',2] gone)");
    assert_eq!((text(&uc[0][0]), int(&uc[0][1])), ("q", 3));
}

// ----- (4) no conflict: a plain insert leaving prior rows intact -----------

#[test]
fn replace_with_no_conflict_is_a_plain_insert() {
    // An INSERT OR REPLACE whose key hits nothing existing must behave like a normal
    // insert: it adds the row and leaves every prior row (and index entry) untouched.
    let (mut pager, mut cat) = db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT)");
    create_index(&mut pager, &mut cat, "CREATE UNIQUE INDEX ub ON t(b)");
    let mut rt = Runtime::new();

    run_insert(
        &insert_plan(
            "t",
            2,
            vec![vec![lit(Value::Integer(1)), lit(Value::Text("x".into()))]],
            affinities(&cat, "t"),
            OnConflict::Abort,
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    run_insert(
        &insert_plan(
            "t",
            2,
            vec![vec![lit(Value::Integer(2)), lit(Value::Text("y".into()))]],
            affinities(&cat, "t"),
            OnConflict::Replace,
        ),
        &cat,
        &mut pager,
        &mut rt,
    );

    let rows = scan_sorted(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 2, "no conflict: both rows present");
    assert_eq!((int(&rows[0][2]), text(&rows[0][1])), (1, "x"), "prior row untouched");
    assert_eq!((int(&rows[1][2]), text(&rows[1][1])), (2, "y"), "new row inserted");

    let keys = index_keys(&pager, index_root(&cat, "ub"));
    assert_eq!(keys.len(), 2, "both index entries present");
    assert_eq!((text(&keys[0][0]), int(&keys[0][1])), ("x", 1));
    assert_eq!((text(&keys[1][0]), int(&keys[1][1])), ("y", 2));
}

// ----- (5) change counter: a single-row REPLACE reports changes() == 1 -----

#[test]
fn replace_changes_count_is_one_for_single_row() {
    // spec/sqlite-doc/lang_conflict.html: "Nor does REPLACE increment the change
    // counter" for the rows it deletes. So a single-row INSERT OR REPLACE that replaces
    // one existing row counts as ONE change (the insert), not two. A fresh Runtime
    // isolates this statement's count from the seed insert's.
    let (mut pager, mut cat) = db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT)");
    create_index(&mut pager, &mut cat, "CREATE UNIQUE INDEX ub ON t(b)");

    let mut seed_rt = Runtime::new();
    run_insert(
        &insert_plan(
            "t",
            2,
            vec![vec![lit(Value::Integer(1)), lit(Value::Text("x".into()))]],
            affinities(&cat, "t"),
            OnConflict::Abort,
        ),
        &cat,
        &mut pager,
        &mut seed_rt,
    );

    let mut rt = Runtime::new();
    run_insert(
        &insert_plan(
            "t",
            2,
            vec![vec![lit(Value::Integer(2)), lit(Value::Text("x".into()))]],
            affinities(&cat, "t"),
            OnConflict::Replace,
        ),
        &cat,
        &mut pager,
        &mut rt,
    );

    assert_eq!(rt.changes(), 1, "the implicit delete is not counted; only the insert is");
    assert_eq!(rt.total_changes(), 1, "total_changes likewise advances by one");
    assert_eq!(rt.last_insert_rowid(), 2, "last_insert_rowid is the inserted rowid");
}

// ----- (6) regression: Ignore / Abort are unchanged by the REPLACE work ----

#[test]
fn ignore_on_unique_conflict_skips() {
    // OnConflict::Ignore still drops the offending row silently and bumps no counter —
    // it must not be affected by the REPLACE gathering path.
    let (mut pager, mut cat) = db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT)");
    create_index(&mut pager, &mut cat, "CREATE UNIQUE INDEX ub ON t(b)");

    let mut rt = Runtime::new();
    run_insert(
        &insert_plan(
            "t",
            2,
            vec![vec![lit(Value::Integer(1)), lit(Value::Text("x".into()))]],
            affinities(&cat, "t"),
            OnConflict::Abort,
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    let before = rt.changes();
    run_insert(
        &insert_plan(
            "t",
            2,
            vec![vec![lit(Value::Integer(2)), lit(Value::Text("x".into()))]],
            affinities(&cat, "t"),
            OnConflict::Ignore,
        ),
        &cat,
        &mut pager,
        &mut rt,
    );

    let rows = scan_sorted(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1, "the conflicting row was skipped, original kept");
    assert_eq!((int(&rows[0][2]), text(&rows[0][1])), (1, "x"), "the original row is untouched");
    assert_eq!(rt.changes(), before, "a skipped row bumps no change counter");

    let keys = index_keys(&pager, index_root(&cat, "ub"));
    assert_eq!(keys.len(), 1, "the index still holds only the original ['x',1]");
    assert_eq!((text(&keys[0][0]), int(&keys[0][1])), ("x", 1));
}

#[test]
fn abort_on_unique_conflict_errors() {
    // OnConflict::Abort still raises a UNIQUE constraint error (and the rolled-back
    // statement leaves the original row and index entry intact).
    let (mut pager, mut cat) = db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT)");
    create_index(&mut pager, &mut cat, "CREATE UNIQUE INDEX ub ON t(b)");

    let mut rt = Runtime::new();
    run_insert(
        &insert_plan(
            "t",
            2,
            vec![vec![lit(Value::Integer(1)), lit(Value::Text("x".into()))]],
            affinities(&cat, "t"),
            OnConflict::Abort,
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    let err = run_insert_err(
        &insert_plan(
            "t",
            2,
            vec![vec![lit(Value::Integer(2)), lit(Value::Text("x".into()))]],
            affinities(&cat, "t"),
            OnConflict::Abort,
        ),
        &cat,
        &mut pager,
    );
    assert!(
        matches!(err, Error::Constraint(_)),
        "duplicate UNIQUE key under Abort -> Constraint, got {err:?}",
    );

    let rows = scan_sorted(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1, "the aborted statement was rolled back");
    assert_eq!((int(&rows[0][2]), text(&rows[0][1])), (1, "x"));
    let keys = index_keys(&pager, index_root(&cat, "ub"));
    assert_eq!(keys.len(), 1, "index unchanged after the aborted insert");
}

// ----- (7) dedup: two unique indexes naming the SAME victim row ------------

#[test]
fn replace_dedups_a_victim_reachable_via_two_unique_indexes() {
    // Row 1 owns BOTH b='x' (ub) and c='p' (uc). INSERT OR REPLACE (2,'x','p') collides
    // with row 1 on BOTH unique indexes, so both probes yield rowid 1. The victim set
    // must be deduped to a single 1 and deleted once: the write phase reads each victim
    // back before deleting and fails loud on a miss, so a NON-deduped [1,1] would delete
    // row 1, then re-read the now-gone row 1 and error. A clean pass proves the dedup.
    let (mut pager, mut cat) =
        db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT, c TEXT)");
    create_index(&mut pager, &mut cat, "CREATE UNIQUE INDEX ub ON t(b)");
    create_index(&mut pager, &mut cat, "CREATE UNIQUE INDEX uc ON t(c)");
    let mut rt = Runtime::new();

    run_insert(
        &insert_plan(
            "t",
            3,
            vec![vec![
                lit(Value::Integer(1)),
                lit(Value::Text("x".into())),
                lit(Value::Text("p".into())),
            ]],
            affinities(&cat, "t"),
            OnConflict::Abort,
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    run_insert(
        &insert_plan(
            "t",
            3,
            vec![vec![
                lit(Value::Integer(2)),
                lit(Value::Text("x".into())),
                lit(Value::Text("p".into())),
            ]],
            affinities(&cat, "t"),
            OnConflict::Replace,
        ),
        &cat,
        &mut pager,
        &mut rt,
    );

    let rows = scan_sorted(&cat, &mut pager, "t", 3);
    assert_eq!(rows.len(), 1, "the single shared victim (row 1) was deleted exactly once");
    assert_eq!(int(&rows[0][3]), 2, "only the replacement at rowid 2 survives");

    let ub = index_keys(&pager, index_root(&cat, "ub"));
    assert_eq!(ub.len(), 1, "ub has one entry, no stale ['x',1]");
    assert_eq!((text(&ub[0][0]), int(&ub[0][1])), ("x", 2));
    let uc = index_keys(&pager, index_root(&cat, "uc"));
    assert_eq!(uc.len(), 1, "uc has one entry, no stale ['p',1]");
    assert_eq!((text(&uc[0][0]), int(&uc[0][1])), ("p", 2));
}

// ----- (8) auto-assigned rowid replacing a UNIQUE victim at another rowid --

#[test]
fn replace_with_auto_rowid_deletes_the_unique_victim() {
    // The replacement supplies NULL for the INTEGER PRIMARY KEY, so its rowid is
    // auto-assigned (max existing + 1). It still collides on the UNIQUE index with a row
    // at a DIFFERENT rowid, which REPLACE must delete. Row 5 owns b='x'; the auto row
    // lands at rowid 6 and takes over 'x' — exercising the seed=max+1 path under REPLACE.
    let (mut pager, mut cat) = db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT)");
    create_index(&mut pager, &mut cat, "CREATE UNIQUE INDEX ub ON t(b)");
    let mut rt = Runtime::new();

    run_insert(
        &insert_plan(
            "t",
            2,
            vec![vec![lit(Value::Integer(5)), lit(Value::Text("x".into()))]],
            affinities(&cat, "t"),
            OnConflict::Abort,
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    run_insert(
        &insert_plan(
            "t",
            2,
            vec![vec![lit(Value::Null), lit(Value::Text("x".into()))]],
            affinities(&cat, "t"),
            OnConflict::Replace,
        ),
        &cat,
        &mut pager,
        &mut rt,
    );

    let rows = scan_sorted(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1, "the UNIQUE victim at rowid 5 was deleted");
    assert_eq!((int(&rows[0][2]), text(&rows[0][1])), (6, "x"), "auto rowid 6 now owns 'x'");
    assert_eq!(rt.last_insert_rowid(), 6, "the auto-assigned rowid is max(5)+1");

    let keys = index_keys(&pager, index_root(&cat, "ub"));
    assert_eq!(keys.len(), 1, "no stale ['x',5] left behind");
    assert_eq!((text(&keys[0][0]), int(&keys[0][1])), ("x", 6));
}

// ----- (9) NULLs never conflict, so REPLACE keeps the other NULL rows ------

#[test]
fn replace_with_null_unique_column_keeps_other_null_rows() {
    // NULLs are distinct in a UNIQUE index ("For the purposes of unique indices, all
    // NULL values are considered different"). A REPLACE whose unique-indexed column is
    // NULL therefore finds NO victim (the helper's NULL-distinctness branch) and must
    // ADD the row alongside the existing NULL rows, not delete any of them.
    let (mut pager, mut cat) = db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT)");
    create_index(&mut pager, &mut cat, "CREATE UNIQUE INDEX ub ON t(b)");
    let mut rt = Runtime::new();

    run_insert(
        &insert_plan(
            "t",
            2,
            vec![
                vec![lit(Value::Integer(1)), lit(Value::Null)],
                vec![lit(Value::Integer(2)), lit(Value::Null)],
            ],
            affinities(&cat, "t"),
            OnConflict::Abort,
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    run_insert(
        &insert_plan(
            "t",
            2,
            vec![vec![lit(Value::Integer(3)), lit(Value::Null)]],
            affinities(&cat, "t"),
            OnConflict::Replace,
        ),
        &cat,
        &mut pager,
        &mut rt,
    );

    let rows = scan_sorted(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 3, "all three NULL-keyed rows coexist — none replaced");
    assert!(rows.iter().all(|r| is_null(&r[1])), "every surviving row keeps its NULL b");

    let keys = index_keys(&pager, index_root(&cat, "ub"));
    assert_eq!(keys.len(), 3, "three distinct NULL index entries survive");
    assert!(keys.iter().all(|k| is_null(&k[0])), "each index key's column slot is NULL");
}

// ----- (10) a UNIQUE-conflict victim's NON-unique index entry is cleaned ---

#[test]
fn replace_via_unique_conflict_also_cleans_the_victims_nonunique_index() {
    // The victim is FOUND via the UNIQUE index ub, but it also has an entry in a
    // NON-unique index ic. delete_index_entries iterates ALL index plans, so deleting
    // the victim must remove its ic entry too — not only the ub entry that flagged it.
    let (mut pager, mut cat) =
        db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT, c TEXT)");
    create_index(&mut pager, &mut cat, "CREATE UNIQUE INDEX ub ON t(b)");
    create_index(&mut pager, &mut cat, "CREATE INDEX ic ON t(c)");
    let mut rt = Runtime::new();

    run_insert(
        &insert_plan(
            "t",
            3,
            vec![vec![
                lit(Value::Integer(1)),
                lit(Value::Text("x".into())),
                lit(Value::Text("p".into())),
            ]],
            affinities(&cat, "t"),
            OnConflict::Abort,
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    run_insert(
        &insert_plan(
            "t",
            3,
            vec![vec![
                lit(Value::Integer(2)),
                lit(Value::Text("x".into())),
                lit(Value::Text("p".into())),
            ]],
            affinities(&cat, "t"),
            OnConflict::Replace,
        ),
        &cat,
        &mut pager,
        &mut rt,
    );

    let rows = scan_sorted(&cat, &mut pager, "t", 3);
    assert_eq!(rows.len(), 1, "the ub victim (row 1) was deleted");
    assert_eq!(int(&rows[0][3]), 2, "only the replacement at rowid 2 survives");

    let ub = index_keys(&pager, index_root(&cat, "ub"));
    assert_eq!(ub.len(), 1);
    assert_eq!((text(&ub[0][0]), int(&ub[0][1])), ("x", 2));

    let ic = index_keys(&pager, index_root(&cat, "ic"));
    assert_eq!(ic.len(), 1, "the victim's stale non-unique ['p',1] was cleaned too");
    assert_eq!((text(&ic[0][0]), int(&ic[0][1])), ("p", 2));
}
