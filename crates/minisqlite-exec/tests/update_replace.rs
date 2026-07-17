//! Integration tests for `UPDATE ... OR REPLACE` conflict resolution (victim deletion),
//! run end to end: a hand-built [`Plan`] over a real [`MemPager`] + [`SchemaCatalog`]
//! through the [`Executor`]/[`RowCursor`] seam, then read back through the real
//! storage/catalog (a full scan AND a walk of each index b-tree — not a private mock).
//!
//! The contract these pin, from `spec/sqlite-doc/lang_conflict.html`: when an `UPDATE OR
//! REPLACE` causes a rowid (PRIMARY KEY) or UNIQUE violation, sqlite "deletes pre-existing
//! rows that are causing the constraint violation prior to inserting ... the current row"
//! — EVERY conflicting row, and each victim's TABLE row AND all its index entries, so no
//! stale or missing index entry survives. The row being updated is itself rewritten (not a
//! victim). "Nor does REPLACE increment the change counter" for the rows it deletes, so a
//! single-row `UPDATE OR REPLACE` that updates one row reports `changes() == 1`.
//!
//! Walking the real index b-tree is the point: it proves the index holds EXACTLY the
//! surviving entries, catching both a stale entry left behind AND a live entry gone missing.
//!
//! Helpers are copied (not shared) because each integration-test file is its own crate;
//! `tests/update.rs` / `tests/replace.rs` hold the sibling copies for the other operators.

use minisqlite_btree::{init_database, IndexCursor};
use minisqlite_catalog::{Catalog, SchemaCatalog};
use minisqlite_exec::{Executor, PagerSet, Runtime, StreamingExecutor};
use minisqlite_expr::{ArithOp, CmpOp, CompareMeta, EvalExpr};
use minisqlite_fileformat::decode_record;
use minisqlite_pager::{MemPager, PageId, Pager};
use minisqlite_plan::{Insert, OnConflict, Plan, Planner, PlanNode, QueryPlanner, SeqScan, Update};
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
            returning: Vec::new(),
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

/// Seed rows through the real `INSERT` operator (positional), so tests exercise the same
/// write path (and index maintenance) they later verify.
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

/// Apply a mutating plan inside a write transaction, draining any `RETURNING` rows and
/// threading `rt` so the change counters can be read afterward.
fn run(plan: &Plan, cat: &SchemaCatalog, pager: &mut MemPager, rt: &mut Runtime) -> Vec<Vec<Value>> {
    pager.begin().unwrap();
    let mut exec = StreamingExecutor;
    let mut out = Vec::new();
    {
        let mut cur =
            exec.execute(plan, cat, PagerSet::One { db: DbIndex::MAIN, pager: &mut *pager }).unwrap();
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

/// Full table scan back through the executor, rows in base-table shape
/// `[c0, …, c_{N-1}, rowid]`, sorted by rowid so assertions are order-independent.
fn scan_sorted(cat: &SchemaCatalog, pager: &mut MemPager, table: &str, n: usize) -> Vec<Vec<Value>> {
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
    out.sort_by_key(|r| int(&r[n]));
    out
}

/// Every entry in an index b-tree (decoded `[key.., rowid]` records), in index order.
/// Walking the real b-tree proves the index holds EXACTLY the surviving entries.
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

/// A full scan filtered to `rowid == r` (`Filter(rowid == r)` over the seq scan), the
/// `WHERE rowid = r` shape these single-row updates use. `col(n)` is the trailing rowid
/// register of a width-`n` (+rowid) scan row.
fn where_rowid(table: &str, n: usize, r: i64) -> PlanNode {
    PlanNode::Filter {
        input: Box::new(seqscan(table, n)),
        predicate: cmp(CmpOp::Eq, col(n), lit(Value::Integer(r))),
    }
}

// ----- (1) rowid move onto an occupied rowid: the victim is deleted --------

#[test]
fn update_replace_moving_rowid_onto_occupied_deletes_victim() {
    // t(id INTEGER PRIMARY KEY, b TEXT), index ib on b. Rows (1,'x'),(2,'y').
    // UPDATE OR REPLACE t SET id = 2 WHERE rowid = 1 moves row 1 onto the occupied rowid 2.
    // REPLACE deletes the victim row 2 ('y') — table row AND index entry — then lands the
    // updated row (id=2, b='x'). Final: one row (2,'x'); the index holds EXACTLY ['x',2]
    // (no stale victim ['y',2], no stale old ['x',1], and the live ['x',2] present).
    let (mut pager, mut cat) = db_with_table("CREATE TABLE t(id INTEGER PRIMARY KEY, b TEXT)");
    create_index(&mut pager, &mut cat, "CREATE INDEX ib ON t(b)");
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
    let plan = update_plan(
        "t",
        2,
        vec![(0, lit(Value::Integer(2)))],
        where_rowid("t", 2, 1),
        affinities(&cat, "t"),
        OnConflict::Replace,
    );
    run(&plan, &cat, &mut pager, &mut rt);
    assert_eq!(rt.changes(), 1, "one row updated; the implicit victim delete is not counted");

    let rows = scan_sorted(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1, "victim row 2 deleted, only the moved row remains");
    // Row shape [id, b, rowid]; the alias id reads back as the rowid.
    assert_eq!((int(&rows[0][2]), text(&rows[0][1])), (2, "x"), "updated row lands at rowid 2 with b='x'");

    let keys = index_keys(&pager, index_root(&cat, "ib"));
    assert_eq!(keys.len(), 1, "exactly one index entry survives");
    assert_eq!(
        (text(&keys[0][0]), int(&keys[0][1])),
        ("x", 2),
        "index points 'x' -> 2; stale ['y',2] and ['x',1] are gone",
    );
}

// ----- (2) UNIQUE conflict, rowid unchanged: the victim is deleted ---------

#[test]
fn update_replace_unique_conflict_rowid_unchanged_deletes_victim() {
    // t(a INTEGER, b TEXT) (implicit rowid), UNIQUE index ub on a. rowid1:(5,'first'),
    // rowid2:(9,'second'). UPDATE OR REPLACE t SET a = 9 WHERE rowid = 1 makes row 1's a
    // collide with row 2 on the unique index (rowid unchanged). REPLACE deletes the victim
    // row 2 (table row + index entry), leaving the updated row 1 (a=9, b='first'). The
    // unique index holds EXACTLY [9,1].
    let (mut pager, mut cat) = db_with_table("CREATE TABLE t(a INTEGER, b TEXT)");
    create_index(&mut pager, &mut cat, "CREATE UNIQUE INDEX ub ON t(a)");
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
    let plan = update_plan(
        "t",
        2,
        vec![(0, lit(Value::Integer(9)))],
        where_rowid("t", 2, 1),
        affinities(&cat, "t"),
        OnConflict::Replace,
    );
    run(&plan, &cat, &mut pager, &mut rt);
    assert_eq!(rt.changes(), 1, "one row updated; the victim delete is uncounted");

    let rows = scan_sorted(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1, "the unique-conflicting victim (row 2) was deleted");
    // rowid stays 1; a is now 9; b unchanged 'first'.
    assert_eq!((int(&rows[0][2]), int(&rows[0][0]), text(&rows[0][1])), (1, 9, "first"));

    let keys = index_keys(&pager, index_root(&cat, "ub"));
    assert_eq!(keys.len(), 1, "the unique index holds exactly the surviving row's entry");
    assert_eq!(
        (int(&keys[0][0]), int(&keys[0][1])),
        (9, 1),
        "index points 9 -> 1; stale [9,2] (victim) and [5,1] (old) are gone",
    );
}

// ----- (3) a victim reached via BOTH the rowid AND a unique index: dedup ---

#[test]
fn update_replace_victim_on_both_rowid_and_unique_deduped() {
    // t(id INTEGER PRIMARY KEY, b TEXT), UNIQUE index ub on b. Rows (1,'x'),(2,'y').
    // UPDATE OR REPLACE t SET id = 2, b = 'y' WHERE rowid = 1 moves row 1 onto rowid 2 (a
    // rowid conflict with row 2) AND sets b='y' (a unique conflict with row 2). Both probes
    // name rowid 2, so the victim set is {2,2} -> deduped to {2} and deleted ONCE. Without
    // dedup the write phase would delete row 2, then re-seek the now-gone row 2 and error —
    // so a clean pass proves the dedup. Final: one row (2,'y'); ub holds EXACTLY ['y',2].
    let (mut pager, mut cat) = db_with_table("CREATE TABLE t(id INTEGER PRIMARY KEY, b TEXT)");
    create_index(&mut pager, &mut cat, "CREATE UNIQUE INDEX ub ON t(b)");
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
    let plan = update_plan(
        "t",
        2,
        vec![(0, lit(Value::Integer(2))), (1, lit(Value::Text("y".into())))],
        where_rowid("t", 2, 1),
        affinities(&cat, "t"),
        OnConflict::Replace,
    );
    run(&plan, &cat, &mut pager, &mut rt);
    assert_eq!(rt.changes(), 1, "one row updated; the single shared victim deleted once, uncounted");

    let rows = scan_sorted(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1, "the shared victim (row 2) was deleted exactly once");
    assert_eq!((int(&rows[0][2]), text(&rows[0][1])), (2, "y"), "moved row lands at rowid 2 with b='y'");

    let keys = index_keys(&pager, index_root(&cat, "ub"));
    assert_eq!(keys.len(), 1, "no double-delete, no leftover: exactly one unique entry");
    assert_eq!((text(&keys[0][0]), int(&keys[0][1])), ("y", 2));
}

// ----- (4) no conflict: an OR REPLACE behaves exactly like a plain update --

#[test]
fn update_replace_with_no_conflict_is_a_plain_update() {
    // An UPDATE OR REPLACE whose new values collide with nothing behaves exactly like a
    // plain update: the matched row changes, every other row (and index entry) is untouched.
    let (mut pager, mut cat) = db_with_table("CREATE TABLE t(a INTEGER, b TEXT)");
    create_index(&mut pager, &mut cat, "CREATE UNIQUE INDEX ub ON t(a)");
    seed(
        &cat,
        &mut pager,
        "t",
        2,
        vec![
            vec![lit(Value::Integer(5)), lit(Value::Text("x".into()))],
            vec![lit(Value::Integer(6)), lit(Value::Text("y".into()))],
        ],
    );
    let mut rt = Runtime::new();
    // UPDATE OR REPLACE t SET a = 7 WHERE rowid = 1  — 7 collides with nothing.
    let plan = update_plan(
        "t",
        2,
        vec![(0, lit(Value::Integer(7)))],
        where_rowid("t", 2, 1),
        affinities(&cat, "t"),
        OnConflict::Replace,
    );
    run(&plan, &cat, &mut pager, &mut rt);
    assert_eq!(rt.changes(), 1, "exactly the matched row changed");

    let rows = scan_sorted(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 2, "no victim deleted: both rows present");
    assert_eq!((int(&rows[0][0]), text(&rows[0][1]), int(&rows[0][2])), (7, "x", 1), "row 1 updated in place");
    assert_eq!((int(&rows[1][0]), text(&rows[1][1]), int(&rows[1][2])), (6, "y", 2), "row 2 untouched");

    // Index holds both entries; assert as a rowid-sorted set (order-independent).
    let mut pairs: Vec<(i64, i64)> =
        index_keys(&pager, index_root(&cat, "ub")).iter().map(|k| (int(&k[0]), int(&k[1]))).collect();
    pairs.sort();
    assert_eq!(pairs, vec![(6, 2), (7, 1)], "both index entries present, old [5,1] moved to [7,1]");
}

// ----- (5) regression: OR ABORT / default still ERRORS on a conflict -------

#[test]
fn update_or_abort_still_errors_on_conflict() {
    // The narrowing was lifted ONLY for REPLACE: a UNIQUE conflict under OR ABORT (the
    // default) must still raise a constraint error, and the rolled-back statement leaves
    // both original rows (and their index entries) intact.
    let (mut pager, mut cat) = db_with_table("CREATE TABLE t(a INTEGER, b TEXT)");
    create_index(&mut pager, &mut cat, "CREATE UNIQUE INDEX ub ON t(a)");
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
    // UPDATE t SET a = 9 WHERE rowid = 1 (default Abort) -> UNIQUE constraint error.
    let plan = update_plan(
        "t",
        2,
        vec![(0, lit(Value::Integer(9)))],
        where_rowid("t", 2, 1),
        affinities(&cat, "t"),
        OnConflict::Abort,
    );
    let err = run_err(&plan, &cat, &mut pager);
    match &err {
        Error::Constraint(m) => {
            assert!(m.starts_with("UNIQUE constraint failed"), "expected UNIQUE kind, got {m:?}");
            assert!(m.contains("t.a"), "expected the ub index detail t.a, got {m:?}");
        }
        other => panic!("expected Error::Constraint, got {other:?}"),
    }
    let rows = scan_sorted(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 2, "the aborted statement rolled back");
    assert_eq!((int(&rows[0][0]), text(&rows[0][1])), (5, "first"));
    assert_eq!((int(&rows[1][0]), text(&rows[1][1])), (9, "second"));
    let keys = index_keys(&pager, index_root(&cat, "ub"));
    assert_eq!(keys.len(), 2, "the unique index is unchanged after the aborted update");
}

// ----- (6) a UNIQUE-conflict victim's NON-unique index entry is cleaned ----

#[test]
fn update_replace_cleans_victims_nonunique_index_entry() {
    // The victim is FOUND via the UNIQUE index ub, but it also has an entry in a NON-unique
    // index ic. delete_index_entries iterates ALL index plans, so deleting the victim must
    // remove its ic entry too — not only the ub entry that flagged it.
    let (mut pager, mut cat) = db_with_table("CREATE TABLE t(a INTEGER, b TEXT, c TEXT)");
    create_index(&mut pager, &mut cat, "CREATE UNIQUE INDEX ub ON t(b)");
    create_index(&mut pager, &mut cat, "CREATE INDEX ic ON t(c)");
    seed(
        &cat,
        &mut pager,
        "t",
        3,
        vec![
            vec![lit(Value::Integer(10)), lit(Value::Text("x".into())), lit(Value::Text("p".into()))],
            vec![lit(Value::Integer(20)), lit(Value::Text("y".into())), lit(Value::Text("q".into()))],
        ],
    );
    let mut rt = Runtime::new();
    // UPDATE OR REPLACE t SET b = 'y' WHERE rowid = 1  -> collides with row 2 on ub.
    let plan = update_plan(
        "t",
        3,
        vec![(1, lit(Value::Text("y".into())))],
        where_rowid("t", 3, 1),
        affinities(&cat, "t"),
        OnConflict::Replace,
    );
    run(&plan, &cat, &mut pager, &mut rt);
    assert_eq!(rt.changes(), 1, "one row updated; the victim delete is uncounted");

    let rows = scan_sorted(&cat, &mut pager, "t", 3);
    assert_eq!(rows.len(), 1, "the ub victim (row 2) was deleted");
    // The surviving row is the updated row 1: a=10, b='y', c='p', rowid 1.
    assert_eq!(int(&rows[0][3]), 1, "the updated row 1 survives");
    assert_eq!((int(&rows[0][0]), text(&rows[0][1]), text(&rows[0][2])), (10, "y", "p"));

    let ub = index_keys(&pager, index_root(&cat, "ub"));
    assert_eq!(ub.len(), 1, "ub holds only the surviving ['y',1]");
    assert_eq!((text(&ub[0][0]), int(&ub[0][1])), ("y", 1));

    let ic = index_keys(&pager, index_root(&cat, "ic"));
    assert_eq!(ic.len(), 1, "the victim's stale non-unique ['q',2] was cleaned; only ['p',1] remains");
    assert_eq!((text(&ic[0][0]), int(&ic[0][1])), ("p", 1));
}

// ----- (7) rowid MOVES but the UNIQUE column is unchanged and un-conflicting -
// This pins the single most load-bearing decision in the change: the unique-index victim
// probe excludes THIS row's own entry by `old_rowid` (its entries are still keyed by the
// OLD rowid), NOT `new_rowid`. When the rowid moves to a free slot while a UNIQUE column is
// unchanged, the row's own surviving entry [key, old_rowid] must NOT be seen as a foreign
// conflict — the update is an ordinary "renumber the PK", it must SUCCEED. Swapping
// `old_rowid` -> `new_rowid` at the probe turns this into a self-conflict: the row deletes
// itself as its own victim (debug: the `old_rowid`-not-a-victim debug_assert fires; release:
// the victim loop deletes rowid 1, then step 8 can't find the scanned row to rewrite).

#[test]
fn update_replace_moving_rowid_with_unchanged_unique_column_is_not_a_self_conflict() {
    let (mut pager, mut cat) = db_with_table("CREATE TABLE t(id INTEGER PRIMARY KEY, b TEXT)");
    create_index(&mut pager, &mut cat, "CREATE UNIQUE INDEX ub ON t(b)");
    seed(&cat, &mut pager, "t", 2, vec![vec![lit(Value::Integer(1)), lit(Value::Text("x".into()))]]);
    let mut rt = Runtime::new();
    // UPDATE OR REPLACE t SET id = 5 WHERE rowid = 1  -- rowid 1->5, b stays 'x', no conflict.
    let plan = update_plan(
        "t",
        2,
        vec![(0, lit(Value::Integer(5)))],
        where_rowid("t", 2, 1),
        affinities(&cat, "t"),
        OnConflict::Replace,
    );
    run(&plan, &cat, &mut pager, &mut rt);
    assert_eq!(rt.changes(), 1, "the row was renumbered, not deleted as its own victim");
    let rows = scan_sorted(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1, "the one row survives at its new rowid");
    assert_eq!((int(&rows[0][2]), text(&rows[0][1])), (5, "x"), "rowid moved 1->5, b unchanged");
    let keys = index_keys(&pager, index_root(&cat, "ub"));
    assert_eq!(keys.len(), 1, "the unique entry moved with the rowid, none stranded");
    assert_eq!((text(&keys[0][0]), int(&keys[0][1])), ("x", 5), "index now points 'x' -> 5");
}

// ----- (8) MULTI-ROW rowid shift: index stays consistent with the table -------
// `UPDATE OR REPLACE t SET id = id + 1` over ALL rows (ascending scan). Processing rowid 1
// (id 1->2) REPLACE-deletes the row at rowid 2 and moves row 1 onto that slot, so the
// buffered snapshot for the original row 2 is now STALE; the pass-2 refresh re-reads the row
// now AT rowid 2 live and reprocesses it. The LOAD-BEARING guarantee — and the exact round-2
// corruption this guards — is the universal invariant asserted first: every surviving index
// entry points at a live table row (the buggy pre-fix left table {3:(3,20)} but index
// {(10,2),(20,3)} — (10,2) dangling, the (2,10) row lost). The exact final collapse is also
// pinned below, but as THIS engine's deterministic pass-2 result (which mirrors sqlite's
// documented second-pass re-read model), NOT as an oracle-confirmed value — it has not been
// diffed against a real sqlite3, so a future differential that disagrees should UPDATE this
// expectation, not read a change here as a regression.

#[test]
fn update_replace_multirow_rowid_shift_keeps_index_consistent() {
    // t(id INTEGER PRIMARY KEY, b), non-unique index ib on b. Rows (1,10),(2,20).
    let (mut pager, mut cat) = db_with_table("CREATE TABLE t(id INTEGER PRIMARY KEY, b INTEGER)");
    create_index(&mut pager, &mut cat, "CREATE INDEX ib ON t(b)");
    seed(
        &cat,
        &mut pager,
        "t",
        2,
        vec![
            vec![lit(Value::Integer(1)), lit(Value::Integer(10))],
            vec![lit(Value::Integer(2)), lit(Value::Integer(20))],
        ],
    );
    let mut rt = Runtime::new();
    // UPDATE OR REPLACE t SET id = id + 1   (all rows; SeqScan is ascending)
    let add_one = EvalExpr::Arith {
        op: ArithOp::Add,
        left: Box::new(col(0)),
        right: Box::new(lit(Value::Integer(1))),
    };
    let plan = update_plan("t", 2, vec![(0, add_one)], seqscan("t", 2), affinities(&cat, "t"), OnConflict::Replace);
    run(&plan, &cat, &mut pager, &mut rt);

    // LOAD-BEARING INVARIANT (holds regardless of the exact cascade, and is exactly the
    // corruption the fix removes): every index entry points at a live table row, and there is
    // one index entry per row — index and table are in bijection, nothing stranded or missing.
    let rows = scan_sorted(&cat, &mut pager, "t", 2);
    let live: std::collections::HashSet<i64> = rows.iter().map(|r| int(&r[2])).collect();
    let keys = index_keys(&pager, index_root(&cat, "ib"));
    assert_eq!(keys.len(), rows.len(), "index holds exactly one entry per live row (no stale/missing entry)");
    for k in &keys {
        assert!(
            live.contains(&int(&k[1])),
            "DANGLING: index entry {:?} points at rowid {} with no table row",
            k,
            int(&k[1])
        );
    }

    // ENGINE-DETERMINISTIC cascade (see the header note: mirrors sqlite's second-pass model,
    // but not oracle-confirmed). Pass 2 re-reads the moved row live, so the block collapses to
    // a single survivor carrying b=10 (the row that ended up at rowid 2), NOT the stale b=20
    // buffered for the original row 2; each rewrite is counted once, so two rewrites for the
    // one survivor. Final: {3:(3,10)}, index {(10,3)}, changes()==2.
    assert_eq!(rt.changes(), 2, "both the original row 1 and the reoccupied rowid-2 slot were rewritten");
    assert_eq!(rows.len(), 1, "the contiguous block collapses to one surviving row");
    assert_eq!((int(&rows[0][2]), int(&rows[0][1])), (3, 10), "survivor is the re-read row at rowid 2, moved to 3");
    assert_eq!((int(&keys[0][0]), int(&keys[0][1])), (10, 3), "the single index entry: 10 -> 3");
}

// ----- (9) TWO DISTINCT victims in one statement (rowid AND unique) ----------
// Both victim sites fire for a SINGLE updated row and name DIFFERENT rows: the moved-to slot
// (rowid victim) and a different row colliding on the UNIQUE index. Both must be gathered and
// both deleted (tests 1 and 2 cover each site alone; test 3 is the same-row dedup case).

#[test]
fn update_replace_two_distinct_victims_in_one_statement() {
    // t(id INTEGER PRIMARY KEY, b TEXT), UNIQUE ub on b. Rows (1,'x'),(2,'y'),(3,'z').
    // UPDATE OR REPLACE t SET id = 2, b = 'z' WHERE rowid = 1: rowid 1->2 collides with row 2
    // (rowid victim), and b='z' collides with row 3 (unique victim). victims = {2, 3}.
    let (mut pager, mut cat) = db_with_table("CREATE TABLE t(id INTEGER PRIMARY KEY, b TEXT)");
    create_index(&mut pager, &mut cat, "CREATE UNIQUE INDEX ub ON t(b)");
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
    let plan = update_plan(
        "t",
        2,
        vec![(0, lit(Value::Integer(2))), (1, lit(Value::Text("z".into())))],
        where_rowid("t", 2, 1),
        affinities(&cat, "t"),
        OnConflict::Replace,
    );
    run(&plan, &cat, &mut pager, &mut rt);
    assert_eq!(rt.changes(), 1, "one row updated; two distinct victims deleted, both uncounted");

    let rows = scan_sorted(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1, "both the rowid victim (2) and the unique victim (3) were deleted");
    assert_eq!((int(&rows[0][2]), text(&rows[0][1])), (2, "z"), "the updated row lands at rowid 2 with b='z'");

    let keys = index_keys(&pager, index_root(&cat, "ub"));
    assert_eq!(keys.len(), 1, "only the surviving row's entry remains");
    assert_eq!((text(&keys[0][0]), int(&keys[0][1])), ("z", 2), "stale ['x',1],['y',2],['z',3] all gone");
}

// ----- (10) MULTI-ROW: a unique victim IS a later updated row -> it is skipped -
// `UPDATE OR REPLACE t SET b = 'y'` over ALL rows. Row 1 (b='x'->'y') collides on ub with
// row 2 (b='y') and REPLACE-deletes it. When the scan reaches the (now deleted) row 2, its
// buffered snapshot is stale: the live row is GONE, so — like sqlite's second pass — it is
// skipped, not resurrected. Exercises the "vanished -> skip" branch of the pass-2 refresh.

#[test]
fn update_replace_multirow_unique_victim_of_a_later_row_is_skipped() {
    let (mut pager, mut cat) = db_with_table("CREATE TABLE t(id INTEGER PRIMARY KEY, b TEXT)");
    create_index(&mut pager, &mut cat, "CREATE UNIQUE INDEX ub ON t(b)");
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
    let plan = update_plan(
        "t",
        2,
        vec![(1, lit(Value::Text("y".into())))],
        seqscan("t", 2),
        affinities(&cat, "t"),
        OnConflict::Replace,
    );
    run(&plan, &cat, &mut pager, &mut rt);
    assert_eq!(rt.changes(), 1, "row 1 updated; row 2 was deleted as a victim then skipped, uncounted");

    let rows = scan_sorted(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1, "row 2 (the deleted victim) is not resurrected by its stale snapshot");
    assert_eq!((int(&rows[0][2]), text(&rows[0][1])), (1, "y"), "only row 1 survives, b='y'");

    let keys = index_keys(&pager, index_root(&cat, "ub"));
    assert_eq!(keys.len(), 1, "the unique index agrees with the single surviving row");
    assert_eq!((text(&keys[0][0]), int(&keys[0][1])), ("y", 1), "index points 'y' -> 1; no dangling ['y',2]");
}

// ----- (11) VICTIM on a VIRTUAL generated-column index: gencol decode branch --
// `read_live_row` has a generated-column branch (`decode_table_row_skipping_virtual_enc` +
// `compute_generated`) that runs ONLY when an `UPDATE OR REPLACE` deletes a victim (or
// refreshes a slot) on a table WITH generated columns — a path the conformance suite does
// not currently reach (it has no `OR REPLACE` over a gencol) and one the positional
// `update_plan`/`seed` helpers cannot express (they leave `generated`/`index_key_exprs`
// empty, so every other test runs only the plain-decode arm). To exercise it FAITHFULLY —
// not with a hand-mis-wired plan, the very risk a wrong pin would introduce — this test
// drives the REAL compiler (`QueryPlanner`) end to end over a UNIQUE index on a VIRTUAL
// column whose value is COMPUTED, never stored: the victim's index key can be removed only
// if its `c` is RECOMPUTED on the live re-read. A broken gencol branch would key the victim
// wrong, miss the real `(20, rowid2)` entry, and strand it in the index.

/// Compile `sql` into a full [`Plan`] via the real [`QueryPlanner`], so `generated` and
/// `index_key_exprs` are populated exactly as in production (no hand-wiring) — for the
/// generated-column paths the positional `update_plan`/`seed` helpers cannot express.
fn plan_sql(cat: &SchemaCatalog, sql: &str) -> Plan {
    let ast = parse(sql).unwrap();
    QueryPlanner::new().plan(&ast.statements[0], cat).unwrap()
}

/// Plan `sql` with the real compiler and run it as a mutating statement (its own txn).
fn run_sql(cat: &SchemaCatalog, pager: &mut MemPager, rt: &mut Runtime, sql: &str) {
    let plan = plan_sql(cat, sql);
    run(&plan, cat, pager, rt);
}

/// Plan `sql` with the real compiler and collect its result rows (a read query).
fn query_sql(cat: &SchemaCatalog, pager: &mut MemPager, sql: &str) -> Vec<Vec<Value>> {
    let plan = plan_sql(cat, sql);
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

#[test]
fn update_replace_victim_on_virtual_generated_column_index_is_correctly_removed() {
    let (mut pager, mut cat) =
        db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b INTEGER, c INTEGER AS (b * 10) VIRTUAL)");
    create_index(&mut pager, &mut cat, "CREATE UNIQUE INDEX uc ON t(c)");
    // Seed via the real planner so the VIRTUAL-gencol index is keyed by the COMPUTED c.
    let mut seed_rt = Runtime::new();
    run_sql(&cat, &mut pager, &mut seed_rt, "INSERT INTO t(a, b) VALUES (1, 1), (2, 2)"); // c = 10, 20

    // UPDATE OR REPLACE t SET b = 2 WHERE a = 1  -> row a=1's computed c goes 10 -> 20, which
    // UNIQUE-conflicts with row a=2 (c = 20). Row a=2 is the victim; removing its index entry
    // requires recomputing its VIRTUAL c on the live re-read (read_live_row's gencol branch).
    let mut rt = Runtime::new();
    run_sql(&cat, &mut pager, &mut rt, "UPDATE OR REPLACE t SET b = 2 WHERE a = 1");
    assert_eq!(rt.changes(), 1, "one row updated; the gencol victim is deleted uncounted");

    // The table: exactly the surviving row, its VIRTUAL c recomputed on read as 20.
    let rows = query_sql(&cat, &mut pager, "SELECT a, b, c FROM t ORDER BY a");
    assert_eq!(rows.len(), 1, "row a=2 (the victim) is deleted, only a=1 survives");
    assert_eq!(
        (int(&rows[0][0]), int(&rows[0][1]), int(&rows[0][2])),
        (1, 2, 20),
        "survivor a=1 now has b=2 and computed c=20"
    );

    // The index: EXACTLY the surviving entry. A mis-decoded victim key would leave the stale
    // (20, rowid2) stranded here (2 entries) instead of the single (20, rowid1).
    let keys = index_keys(&pager, index_root(&cat, "uc"));
    assert_eq!(keys.len(), 1, "the victim's virtual-gencol index entry was recomputed and removed");
    assert_eq!((int(&keys[0][0]), int(&keys[0][1])), (20, 1), "uc holds only c=20 -> rowid 1");
}
