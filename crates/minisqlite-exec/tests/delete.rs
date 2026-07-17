//! Integration tests for the `DELETE` DML operator: hand-built [`Plan`] trees run
//! over a real [`MemPager`] + [`SchemaCatalog`] through the [`Executor`]/[`RowCursor`]
//! seam, then read back through the real storage/catalog (not a private mock). These
//! pin the operator end to end: delete-all, filtered delete, index-entry maintenance
//! (including duplicate indexed values disambiguated by rowid), `RETURNING` (single- and
//! multi-row streaming), the `Runtime` change counter (only rows actually present count),
//! and the fail-closed guards that turn a malformed plan into an `Error`, never a panic.
//!
//! Rows are seeded through the real `INSERT` operator (not raw `table_insert`) so that
//! index entries exist exactly as production writes them — otherwise the index-
//! maintenance test would have nothing to remove. DML assumes an open write
//! transaction (the engine opens one), so each apply is wrapped in `begin`/`commit`;
//! read-back scans need no transaction.

use minisqlite_btree::init_database;
use minisqlite_catalog::{Catalog, SchemaCatalog};
use minisqlite_exec::{Executor, Runtime, StreamingExecutor};
use minisqlite_expr::{CmpOp, CompareMeta, EvalExpr};
use minisqlite_pager::{MemPager, Pager};
use minisqlite_plan::{
    Delete, IndexOp, IndexScan, Insert, OnConflict, Plan, PlanNode, ScanDirection, SeqScan,
};
use minisqlite_sql::{parse, Statement};
use minisqlite_types::{affinity_of_declared_type, Affinity, Collation, Error, Value};
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

/// Create an index through the real catalog path (writes apply directly outside a
/// transaction, like `create_table` above).
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

// ----- expression / node helpers -------------------------------------------

fn col(i: usize) -> EvalExpr {
    EvalExpr::Column(i)
}

fn lit(v: Value) -> EvalExpr {
    EvalExpr::Literal(v)
}

fn binary_meta() -> CompareMeta {
    CompareMeta { apply_left: None, apply_right: None, collation: Collation::Binary }
}

/// `left = right` under BINARY collation — the WHERE predicate for the filtered tests.
fn eq(left: EvalExpr, right: EvalExpr) -> EvalExpr {
    EvalExpr::Compare {
        op: CmpOp::Eq,
        null_safe: false,
        left: Box::new(left),
        right: Box::new(right),
        meta: binary_meta(),
    }
}

fn seqscan(table: &str, n: usize) -> PlanNode {
    PlanNode::SeqScan(SeqScan { db: DbIndex::MAIN, table: table.to_string(), column_count: n })
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

/// An `INSERT` plan over a literal `VALUES` source, used to seed rows through the real
/// write path (so index entries are populated as production would).
fn insert_plan(table: &str, n: usize, rows: Vec<Vec<EvalExpr>>, affs: Vec<Affinity>) -> Plan {
    let source_width = rows.first().map(|r| r.len()).unwrap_or(n);
    Plan {
        root: PlanNode::Insert(Insert { db: DbIndex::MAIN,
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

/// A `DELETE` plan over an arbitrary `scan` subtree (`SeqScan`, a `Filter` over one,
/// or an `IndexScan`), optionally with `RETURNING`.
fn delete_plan(table: &str, n: usize, scan: PlanNode, returning: Vec<EvalExpr>) -> Plan {
    Plan {
        root: PlanNode::Delete(Delete { db: DbIndex::MAIN,
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

/// An `IndexScan` plan that seeks `idx_b` for `b = value`, emitting the matching table
/// rows `[a, b, rowid]`. Used to prove index entries survive / are removed by DELETE.
fn index_seek_plan(value: i64) -> Plan {
    let scan = PlanNode::IndexScan(IndexScan { db: DbIndex::MAIN,
        table: "t".to_string(),
        column_count: 2,
        index: "idx_b".to_string(),
        op: IndexOp::Seek { eq_prefix: vec![lit(Value::Integer(value))], low: None, high: None },
        direction: ScanDirection::Forward,
        covering: false,
    });
    Plan {
        root: scan,
        result_columns: Vec::new(),
        ctes: Vec::new(),
        subqueries: Vec::new(),
        mutates: false,
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
        let mut cur = exec.execute(plan, cat, PagerSet::One { db: DbIndex::MAIN, pager: &mut *pager }).unwrap();
        while let Some(row) = cur.next_row(rt).unwrap() {
            out.push(row);
        }
    }
    pager.commit().unwrap();
    out
}

/// Apply a DML plan expected to FAIL. Mirrors the engine's abort path: the statement is
/// rolled back and the error returned, so the operator's fail-closed guards (a malformed
/// plan) can be asserted instead of unwrapped into a panic.
fn run_dml_err(plan: &Plan, cat: &SchemaCatalog, pager: &mut MemPager) -> Error {
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
    err.expect("expected the DELETE to error")
}

/// Run a read plan to completion, returning the `Result` so an error path (e.g. a
/// dangling index entry) can be asserted rather than unwrapped.
fn run_read_result(
    plan: &Plan,
    cat: &SchemaCatalog,
    pager: &mut MemPager,
) -> minisqlite_types::Result<Vec<Vec<Value>>> {
    let mut rt = Runtime::new();
    let mut exec = StreamingExecutor;
    let mut cur = exec.execute(plan, cat, PagerSet::One { db: DbIndex::MAIN, pager: &mut *pager })?;
    let mut out = Vec::new();
    while let Some(row) = cur.next_row(&mut rt)? {
        out.push(row);
    }
    Ok(out)
}

/// Read helper that unwraps (for the happy paths).
fn run_read(plan: &Plan, cat: &SchemaCatalog, pager: &mut MemPager) -> Vec<Vec<Value>> {
    run_read_result(plan, cat, pager).unwrap()
}

/// Full table scan back through the executor: rows are `[c0, …, c_{N-1}, rowid]`.
fn scan_all(cat: &SchemaCatalog, pager: &mut MemPager, table: &str, n: usize) -> Vec<Vec<Value>> {
    let plan = Plan {
        root: seqscan(table, n),
        result_columns: Vec::new(),
        ctes: Vec::new(),
        subqueries: Vec::new(),
        mutates: false,
        generated: Vec::new(),
    };
    run_read(&plan, cat, pager)
}

/// Seed a two-column `(a, b)` table with the given integer rows via the INSERT path.
fn seed_ab(cat: &SchemaCatalog, pager: &mut MemPager, rows: &[(i64, i64)]) {
    let mut rt = Runtime::new();
    let vals: Vec<Vec<EvalExpr>> = rows
        .iter()
        .map(|(a, b)| vec![lit(Value::Integer(*a)), lit(Value::Integer(*b))])
        .collect();
    run_dml(&insert_plan("t", 2, vals, affinities(cat, "t")), cat, pager, &mut rt);
}

// ----- (1) delete-all ------------------------------------------------------

#[test]
fn delete_all_empties_the_table_and_counts_every_row() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER, b TEXT)");
    let mut seed_rt = Runtime::new();
    run_dml(
        &insert_plan(
            "t",
            2,
            vec![
                vec![lit(Value::Integer(10)), lit(Value::Text("x".into()))],
                vec![lit(Value::Integer(20)), lit(Value::Text("y".into()))],
                vec![lit(Value::Integer(30)), lit(Value::Text("z".into()))],
            ],
            affinities(&cat, "t"),
        ),
        &cat,
        &mut pager,
        &mut seed_rt,
    );
    assert_eq!(scan_all(&cat, &mut pager, "t", 2).len(), 3, "seeded three rows");

    // DELETE FROM t  (full SeqScan, no filter) over a fresh Runtime so changes() is
    // exactly this statement's.
    let mut del_rt = Runtime::new();
    let out = run_dml(&delete_plan("t", 2, seqscan("t", 2), Vec::new()), &cat, &mut pager, &mut del_rt);
    assert!(out.is_empty(), "no RETURNING clause yields no rows");
    assert_eq!(del_rt.changes(), 3, "all three rows deleted");
    assert!(scan_all(&cat, &mut pager, "t", 2).is_empty(), "the table is empty after delete-all");
}

// ----- (2) filtered delete -------------------------------------------------

#[test]
fn filtered_delete_removes_only_matching_rows() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER, b TEXT)");
    let mut seed_rt = Runtime::new();
    run_dml(
        &insert_plan(
            "t",
            2,
            vec![
                vec![lit(Value::Integer(10)), lit(Value::Text("x".into()))],
                vec![lit(Value::Integer(20)), lit(Value::Text("y".into()))],
                vec![lit(Value::Integer(30)), lit(Value::Text("z".into()))],
            ],
            affinities(&cat, "t"),
        ),
        &cat,
        &mut pager,
        &mut seed_rt,
    );

    // DELETE FROM t WHERE a = 20
    let scan = PlanNode::Filter {
        input: Box::new(seqscan("t", 2)),
        predicate: eq(col(0), lit(Value::Integer(20))),
    };
    let mut del_rt = Runtime::new();
    run_dml(&delete_plan("t", 2, scan, Vec::new()), &cat, &mut pager, &mut del_rt);
    assert_eq!(del_rt.changes(), 1, "only the a=20 row matched");

    let rows = scan_all(&cat, &mut pager, "t", 2);
    let avals: Vec<i64> = rows.iter().map(|r| int(&r[0])).collect();
    assert_eq!(avals, vec![10, 30], "the non-matching rows remain, in rowid order");
    assert_eq!(text(&rows[0][1]), "x");
    assert_eq!(text(&rows[1][1]), "z");
}

// ----- (3) index maintenance (the key correctness test) --------------------

#[test]
fn delete_removes_index_entries_not_just_table_rows() {
    // Seeding via the INSERT operator populates idx_b, so there is a real entry to
    // remove. Deleting the b=20 row must remove BOTH its table row and its index
    // entry: an IndexScan seek for b=20 then returns nothing. Crucially it must not
    // ERROR — a lingering index entry would make the seek fetch a now-missing table
    // row (the IndexScan fails loud on a dangling entry), so a clean empty result is
    // positive proof the index entry was removed, not just the table row.
    let (mut pager, mut cat) = db_with_table("CREATE TABLE t(a INTEGER, b INTEGER)");
    create_index(&mut pager, &mut cat, "CREATE INDEX idx_b ON t(b)");
    seed_ab(&cat, &mut pager, &[(1, 10), (2, 20), (3, 30)]);

    // Sanity: before the delete, the index seek finds the b=20 row.
    let before = run_read(&index_seek_plan(20), &cat, &mut pager);
    assert_eq!(before.len(), 1, "b=20 is reachable via idx_b before the delete");
    assert_eq!(int(&before[0][0]), 2, "and it is the (a=2, b=20) row");

    // DELETE FROM t WHERE b = 20
    let scan = PlanNode::Filter {
        input: Box::new(seqscan("t", 2)),
        predicate: eq(col(1), lit(Value::Integer(20))),
    };
    let mut del_rt = Runtime::new();
    run_dml(&delete_plan("t", 2, scan, Vec::new()), &cat, &mut pager, &mut del_rt);
    assert_eq!(del_rt.changes(), 1);

    // The deleted key's index entry is gone: the seek returns Ok(empty), never an
    // error from a dangling entry pointing at a removed table row.
    let after = run_read_result(&index_seek_plan(20), &cat, &mut pager);
    assert!(after.is_ok(), "seeking a deleted key must not error on a dangling entry: {after:?}");
    assert!(after.unwrap().is_empty(), "no idx_b entry remains for the deleted key b=20");

    // A surviving key still resolves through the index to its row.
    let surviving = run_read(&index_seek_plan(10), &cat, &mut pager);
    assert_eq!(surviving.len(), 1, "b=10 is still reachable via idx_b");
    assert_eq!((int(&surviving[0][0]), int(&surviving[0][1])), (1, 10));
}

// ----- (4) the scan is an IndexScan over the very index being modified ------

#[test]
fn delete_via_indexscan_over_the_modified_index_is_safe() {
    // The scan subtree reads idx_b while the delete modifies it. Buffering the whole
    // scan BEFORE any removal is what keeps this correct — the cursor never observes a
    // tree it is mutating. Delete the b=20 row by seeking it through idx_b itself.
    let (mut pager, mut cat) = db_with_table("CREATE TABLE t(a INTEGER, b INTEGER)");
    create_index(&mut pager, &mut cat, "CREATE INDEX idx_b ON t(b)");
    seed_ab(&cat, &mut pager, &[(1, 10), (2, 20), (3, 30)]);

    let scan = PlanNode::IndexScan(IndexScan { db: DbIndex::MAIN,
        table: "t".to_string(),
        column_count: 2,
        index: "idx_b".to_string(),
        op: IndexOp::Seek { eq_prefix: vec![lit(Value::Integer(20))], low: None, high: None },
        direction: ScanDirection::Forward,
        covering: false,
    });
    let mut del_rt = Runtime::new();
    run_dml(&delete_plan("t", 2, scan, Vec::new()), &cat, &mut pager, &mut del_rt);
    assert_eq!(del_rt.changes(), 1, "the single b=20 row was deleted");

    let rows = scan_all(&cat, &mut pager, "t", 2);
    let avals: Vec<i64> = rows.iter().map(|r| int(&r[0])).collect();
    assert_eq!(avals, vec![1, 3], "the b=20 row is removed; the others remain");
    assert!(
        run_read_result(&index_seek_plan(20), &cat, &mut pager).unwrap().is_empty(),
        "idx_b entry for b=20 is removed"
    );
    assert_eq!(run_read(&index_seek_plan(30), &cat, &mut pager).len(), 1, "b=30 still reachable");
}

// ----- (5) RETURNING -------------------------------------------------------

#[test]
fn delete_returning_streams_the_deleted_rows() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER, b TEXT)");
    let mut seed_rt = Runtime::new();
    run_dml(
        &insert_plan(
            "t",
            2,
            vec![
                vec![lit(Value::Integer(7)), lit(Value::Text("g".into()))],
                vec![lit(Value::Integer(8)), lit(Value::Text("h".into()))],
            ],
            affinities(&cat, "t"),
        ),
        &cat,
        &mut pager,
        &mut seed_rt,
    );

    // DELETE FROM t WHERE a = 7 RETURNING a, b, rowid  (rowid is register 2).
    let scan = PlanNode::Filter {
        input: Box::new(seqscan("t", 2)),
        predicate: eq(col(0), lit(Value::Integer(7))),
    };
    let mut del_rt = Runtime::new();
    let out = run_dml(
        &delete_plan("t", 2, scan, vec![col(0), col(1), col(2)]),
        &cat,
        &mut pager,
        &mut del_rt,
    );
    assert_eq!(out.len(), 1, "one row deleted yields one RETURNING row");
    // The a=7 row was seeded first, so its auto rowid is 1.
    assert_eq!((int(&out[0][0]), text(&out[0][1]), int(&out[0][2])), (7, "g", 1));
    assert_eq!(del_rt.changes(), 1);

    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1, "only the a=8 row remains");
    assert_eq!(int(&rows[0][0]), 8);
}

// ----- (6) changes() counts only rows actually present ---------------------

#[test]
fn delete_over_a_whole_miss_leaves_changes_at_zero() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER, b TEXT)");
    let mut seed_rt = Runtime::new();
    run_dml(
        &insert_plan(
            "t",
            2,
            vec![
                vec![lit(Value::Integer(10)), lit(Value::Text("x".into()))],
                vec![lit(Value::Integer(20)), lit(Value::Text("y".into()))],
            ],
            affinities(&cat, "t"),
        ),
        &cat,
        &mut pager,
        &mut seed_rt,
    );

    // DELETE FROM t WHERE a = 999  (matches nothing).
    let scan = PlanNode::Filter {
        input: Box::new(seqscan("t", 2)),
        predicate: eq(col(0), lit(Value::Integer(999))),
    };
    let mut del_rt = Runtime::new();
    run_dml(&delete_plan("t", 2, scan, Vec::new()), &cat, &mut pager, &mut del_rt);
    assert_eq!(del_rt.changes(), 0, "nothing matched, so no rows were deleted");
    assert_eq!(scan_all(&cat, &mut pager, "t", 2).len(), 2, "both rows remain untouched");
}

#[test]
fn delete_over_an_empty_table_is_a_noop() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER)");
    let mut del_rt = Runtime::new();
    let out = run_dml(&delete_plan("t", 1, seqscan("t", 1), Vec::new()), &cat, &mut pager, &mut del_rt);
    assert!(out.is_empty(), "no rows to return");
    assert_eq!(del_rt.changes(), 0, "an empty scan deletes nothing");
    assert!(scan_all(&cat, &mut pager, "t", 1).is_empty());
}

// ----- (7) fail-closed guards: a malformed plan errors, never panics --------

#[test]
fn delete_on_a_missing_table_is_an_error() {
    // The target table is resolved (in build) before anything is mutated; a missing
    // target fails closed with a `no such table` Sql error, never a panic.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER)");
    let err = run_dml_err(&delete_plan("ghost", 1, seqscan("ghost", 1), Vec::new()), &cat, &mut pager);
    assert!(matches!(err, Error::Sql(_)), "missing table -> Sql error, got {err:?}");
}

#[test]
fn delete_with_a_wrong_column_count_is_an_error() {
    // The plan claims 3 columns but `t` has 1; the mismatch is caught up front (before
    // any b-tree write), failing closed rather than corrupting storage.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER)");
    let err = run_dml_err(&delete_plan("t", 3, seqscan("t", 1), Vec::new()), &cat, &mut pager);
    assert!(matches!(err, Error::Sql(_)), "column_count mismatch -> Sql error, got {err:?}");
}

#[test]
fn delete_scan_row_without_an_integer_rowid_is_an_error() {
    // A malformed scan whose register N (the rowid slot) is not an Integer must fail
    // closed, never panic on the `&row[..n]` slice. A `Values` node supplies the bad row
    // directly: width 3 over a 2-column table, with Text where the rowid (register 2)
    // belongs.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER, b INTEGER)");
    let bad_scan = PlanNode::Values {
        rows: vec![vec![
            lit(Value::Integer(1)),
            lit(Value::Integer(2)),
            lit(Value::Text("not-a-rowid".into())),
        ]],
    };
    let err = run_dml_err(&delete_plan("t", 2, bad_scan, Vec::new()), &cat, &mut pager);
    assert!(matches!(err, Error::Sql(_)), "non-integer rowid -> Sql error, got {err:?}");
}

#[test]
fn delete_over_without_rowid_table_removes_the_matching_row() {
    // DELETE over a WITHOUT ROWID table is implemented (it was fail-closed here before):
    // `build` resolves the base table through the WR-aware `resolve_base_table` gate and
    // branches to the PRIMARY KEY index-b-tree delete path. Seed WR rows via the WR INSERT
    // path, delete one by its PRIMARY KEY, and confirm it is gone and counted while the
    // other survives (a width-N row, no rowid register). Comprehensive WR-delete coverage —
    // composite PK, RETURNING, and the fail-closed secondary-index / trigger guards — lives
    // in `tests/without_rowid_delete.rs`; this pins that the operator no longer rejects a WR
    // target and correctly routes through the WR path.
    let (mut pager, cat) =
        db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT) WITHOUT ROWID");
    let mut seed_rt = Runtime::new();
    run_dml(
        &insert_plan(
            "t",
            2,
            vec![
                vec![lit(Value::Integer(1)), lit(Value::Text("x".into()))],
                vec![lit(Value::Integer(2)), lit(Value::Text("y".into()))],
            ],
            affinities(&cat, "t"),
        ),
        &cat,
        &mut pager,
        &mut seed_rt,
    );

    // DELETE FROM t WHERE a = 1 (a is the WR PRIMARY KEY; the scan row is width-N [a, b]).
    let scan = PlanNode::Filter {
        input: Box::new(seqscan("t", 2)),
        predicate: eq(col(0), lit(Value::Integer(1))),
    };
    let mut del_rt = Runtime::new();
    run_dml(&delete_plan("t", 2, scan, Vec::new()), &cat, &mut pager, &mut del_rt);
    assert_eq!(del_rt.changes(), 1, "the WR row with a=1 was deleted and counted");

    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1, "one WR row remains after the delete");
    assert_eq!(rows[0].len(), 2, "the WR scan row is width N, no rowid register");
    assert_eq!((int(&rows[0][0]), text(&rows[0][1])), (2, "y"), "only the a=2 row survives");
}

// ----- (8) RETURNING streams every matched row, in order -------------------

#[test]
fn delete_returning_streams_multiple_rows_in_order() {
    // Drives the streaming half of the cursor (the built-gate + `ret_idx`) with MORE than
    // one row: each deleted row must stream out exactly once, in buffered (rowid) order.
    // A bug that emitted only the first row, all rows at once, or an off-by-one shows here.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER, b TEXT)");
    let mut seed_rt = Runtime::new();
    run_dml(
        &insert_plan(
            "t",
            2,
            vec![
                vec![lit(Value::Integer(10)), lit(Value::Text("x".into()))],
                vec![lit(Value::Integer(20)), lit(Value::Text("y".into()))],
                vec![lit(Value::Integer(30)), lit(Value::Text("z".into()))],
            ],
            affinities(&cat, "t"),
        ),
        &cat,
        &mut pager,
        &mut seed_rt,
    );

    // DELETE FROM t RETURNING a, rowid  (full scan -> all three; register 2 is the rowid).
    let mut del_rt = Runtime::new();
    let out = run_dml(&delete_plan("t", 2, seqscan("t", 2), vec![col(0), col(2)]), &cat, &mut pager, &mut del_rt);
    assert_eq!(out.len(), 3, "every deleted row streams its own RETURNING row");
    let got: Vec<(i64, i64)> = out.iter().map(|r| (int(&r[0]), int(&r[1]))).collect();
    assert_eq!(got, vec![(10, 1), (20, 2), (30, 3)], "rows stream once each, in rowid order");
    assert_eq!(del_rt.changes(), 3);
    assert!(scan_all(&cat, &mut pager, "t", 2).is_empty(), "the table is empty afterward");
}

// ----- (9) duplicate indexed value: the rowid disambiguates the key --------

#[test]
fn delete_one_of_two_rows_sharing_an_indexed_value_keeps_the_other() {
    // Two rows share b=10. The rowid is part of every index key, so deleting one must
    // remove ONLY its entry and leave the other still reachable through idx_b — this pins
    // the rowid-in-key disambiguation the distinct-key tests don't directly exercise.
    let (mut pager, mut cat) = db_with_table("CREATE TABLE t(a INTEGER, b INTEGER)");
    create_index(&mut pager, &mut cat, "CREATE INDEX idx_b ON t(b)");
    seed_ab(&cat, &mut pager, &[(1, 10), (2, 10)]);
    assert_eq!(run_read(&index_seek_plan(10), &cat, &mut pager).len(), 2, "both b=10 rows reachable before");

    // DELETE FROM t WHERE a = 1  (rowid 1, one of the two b=10 rows).
    let scan = PlanNode::Filter {
        input: Box::new(seqscan("t", 2)),
        predicate: eq(col(0), lit(Value::Integer(1))),
    };
    let mut del_rt = Runtime::new();
    run_dml(&delete_plan("t", 2, scan, Vec::new()), &cat, &mut pager, &mut del_rt);
    assert_eq!(del_rt.changes(), 1, "exactly one row deleted");

    // The untargeted b=10 row (rowid 2) is still reachable — exactly one entry survives,
    // and the seek does not error on a dangling entry.
    let survivors = run_read(&index_seek_plan(10), &cat, &mut pager);
    assert_eq!(survivors.len(), 1, "only the untargeted b=10 row remains in idx_b");
    assert_eq!((int(&survivors[0][0]), int(&survivors[0][1])), (2, 10), "and it is the (a=2, b=10) row");
}
