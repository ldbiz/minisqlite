//! Integration test for the UPSERT `DO UPDATE` fail-closed guard against a rowid /
//! INTEGER PRIMARY KEY change (`ops/insert.rs::do_upsert_update`, step 5).
//!
//! A `DO UPDATE` "behaves as an UPDATE" (`lang_upsert.html` §2), and an UPDATE MAY move an
//! INTEGER PRIMARY KEY (the rowid). The rowid UPSERT path does NOT yet support that move:
//! learning the moved-to rowid safely needs the auto-rowid high-water the caller owns, and
//! without it a later auto-assigned row in the SAME statement could be handed the vacated
//! `max+1` and — because a rowid collision REPLACES — silently overwrite the moved row (see
//! the step-5 comment in `do_upsert_update`). So the path fails LOUD rather than corrupting
//! data. No other exec/facade test pins this narrowing, so a regression that dropped the guard
//! (letting a rowid-changing DO UPDATE proceed) would go uncaught.
//!
//! This drives the REAL parse -> plan -> execute path: the statement is parsed
//! (`minisqlite_sql::parse`), compiled by the REAL planner (`QueryPlanner` — which ACCEPTS
//! `SET <ipk> = <int>`, so raising the guard is the executor's job), and run through the
//! `Executor`/`RowCursor` seam over a real `MemPager` + `SchemaCatalog`, then read back with a
//! real `SELECT` — no hand-built plan and no mock. DML assumes an open write transaction (the
//! engine opens one), so the apply is wrapped in `begin`/`commit`, rolling back on error as the
//! engine does for a failed statement; the read-back `SELECT` needs none.

use minisqlite_btree::init_database;
use minisqlite_catalog::{Catalog, SchemaCatalog};
use minisqlite_exec::{Executor, PagerSet, Runtime, StreamingExecutor};
use minisqlite_pager::{MemPager, Pager};
use minisqlite_plan::{Planner, QueryPlanner};
use minisqlite_sql::{parse, Statement};
use minisqlite_types::{DbIndex, Error, Value};

/// A fresh in-memory database with an initialized file header and an empty catalog.
fn db() -> (MemPager, SchemaCatalog) {
    let mut pager = MemPager::new(4096);
    init_database(&mut pager).unwrap();
    (pager, SchemaCatalog::new())
}

/// Create a table through the real catalog path (DDL applies directly, outside a transaction).
fn create(pager: &mut MemPager, cat: &mut SchemaCatalog, sql: &str) {
    let ast = parse(sql).unwrap();
    let Statement::CreateTable(stmt) = &ast.statements[0] else {
        panic!("not a CREATE TABLE: {sql}");
    };
    cat.create_table(pager, stmt, sql).unwrap();
}

/// Run one DML statement through the REAL planner + executor. Commits on success and returns
/// any RETURNING rows; rolls the statement back and returns the error on failure (mirroring
/// the engine's per-statement abort).
fn dml(cat: &SchemaCatalog, pager: &mut MemPager, sql: &str) -> Result<Vec<Vec<Value>>, Error> {
    let ast = parse(sql)?;
    let stmt = ast.statements.first().expect("one statement");
    let plan = QueryPlanner::new().plan(stmt, cat)?;
    pager.begin().unwrap();
    let mut exec = StreamingExecutor;
    let mut rt = Runtime::new();
    let mut out = Vec::new();
    let mut result: Result<(), Error> = Ok(());
    match exec.execute(&plan, cat, PagerSet::One { db: DbIndex::MAIN, pager: &mut *pager }) {
        Ok(mut cur) => loop {
            match cur.next_row(&mut rt) {
                Ok(Some(row)) => out.push(row),
                Ok(None) => break,
                Err(e) => {
                    result = Err(e);
                    break;
                }
            }
        },
        Err(e) => result = Err(e),
    }
    match result {
        Ok(()) => {
            pager.commit().unwrap();
            Ok(out)
        }
        Err(e) => {
            pager.rollback().unwrap();
            Err(e)
        }
    }
}

/// Read rows back through the REAL planner + executor (a `SELECT`); no transaction needed.
fn query(cat: &SchemaCatalog, pager: &mut MemPager, sql: &str) -> Vec<Vec<Value>> {
    let ast = parse(sql).unwrap();
    let stmt = ast.statements.first().expect("one statement");
    let plan = QueryPlanner::new().plan(stmt, cat).unwrap();
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

fn int(v: &Value) -> i64 {
    match v {
        Value::Integer(i) => *i,
        other => panic!("expected Integer, got {other:?}"),
    }
}

#[test]
fn do_update_changing_the_integer_primary_key_fails_closed() {
    // `t.id` is the INTEGER PRIMARY KEY (the rowid alias); `t.u` is a separate UNIQUE column
    // used only as the conflict target. Seed one row: id=1, u=100, v=1000.
    let (mut pager, mut cat) = db();
    create(&mut pager, &mut cat, "CREATE TABLE t(id INTEGER PRIMARY KEY, u INTEGER UNIQUE, v INTEGER)");
    dml(&cat, &mut pager, "INSERT INTO t VALUES (1, 100, 1000)").unwrap();

    // The candidate (auto rowid, u=100, v=2000) conflicts on the UNIQUE `u` with the existing
    // row (id=1); ON CONFLICT(u) routes to DO UPDATE, whose `SET id = 999` would MOVE that
    // row's rowid 1 -> 999. The rowid UPSERT path does not support a rowid move yet, so exec
    // must fail LOUD (step 5 of `do_upsert_update`) instead of silently corrupting the store.
    let err = dml(
        &cat,
        &mut pager,
        "INSERT INTO t(u, v) VALUES (100, 2000) ON CONFLICT(u) DO UPDATE SET id = 999",
    )
    .expect_err("a DO UPDATE that changes the INTEGER PRIMARY KEY must fail closed");

    // Assert the EXACT guard message (a substring match, not a bare is_err()), so a regression
    // that drops or reworks the guard — letting the rowid move proceed — is caught here.
    match &err {
        Error::Sql(m) => assert!(
            m.contains(
                "UPSERT DO UPDATE that changes the rowid / INTEGER PRIMARY KEY is not yet supported"
            ),
            "the guard must name the rowid/IPK-change limitation, got {m:?}"
        ),
        other => panic!("expected a fail-closed Error::Sql, got {other:?}"),
    }

    // Fail-closed + rolled back: the store is exactly as seeded — the original row unchanged,
    // no new row inserted, the rowid NOT moved to 999.
    let rows = query(&cat, &mut pager, "SELECT id, u, v FROM t");
    let got: Vec<(i64, i64, i64)> =
        rows.iter().map(|r| (int(&r[0]), int(&r[1]), int(&r[2]))).collect();
    assert_eq!(got, vec![(1, 100, 1000)], "the refused DO UPDATE left the store untouched");
}
