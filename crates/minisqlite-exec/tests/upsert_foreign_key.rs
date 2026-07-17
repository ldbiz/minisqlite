//! Integration tests for FOREIGN KEY enforcement on the UPSERT `DO UPDATE` in-place rewrite
//! (`ops/insert.rs::do_upsert_update`). A `DO UPDATE` "behaves as an UPDATE"
//! (`lang_upsert.html` §2) and is "OR ABORT" (§3), so — exactly like the rowid `UPDATE` path —
//! the rewritten row must (a) still satisfy CHILD-side FKs on its new key columns and (b) fire
//! the PARENT-side `ON UPDATE` action when it changes a referenced key. Without those the
//! rewrite silently orphaned children (dangling child FK) or skipped cascades (parent side),
//! diverging from real `sqlite3` on the DML-with-FK-ON behavior.
//!
//! These drive the REAL parse -> plan -> execute path: each statement is parsed
//! (`minisqlite_sql::parse`), compiled by the REAL planner (`QueryPlanner`), and run through
//! the `Executor`/`RowCursor` seam over a real `MemPager` + `SchemaCatalog`, then read back
//! with a real `SELECT` — no hand-built plan and no mock. The `foreign_keys` pragma is modeled
//! by `Runtime::set_foreign_keys` (what the real `PRAGMA` sets). DML assumes an open write
//! transaction (the engine opens one), so each apply is wrapped in `begin`/`commit`, rolling
//! back on error as the engine does for a failed statement; read-back `SELECT`s need none.

use minisqlite_btree::init_database;
use minisqlite_catalog::{Catalog, SchemaCatalog};
use minisqlite_exec::{Executor, PagerSet, Runtime, StreamingExecutor};
use minisqlite_pager::{MemPager, Pager};
use minisqlite_plan::{Planner, QueryPlanner};
use minisqlite_sql::{parse, Statement};
use minisqlite_types::{DbIndex, Error, Value};

// ----- fixtures ------------------------------------------------------------

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

/// Run one DML statement through the REAL planner + executor with the caller's `Runtime` (so
/// its `foreign_keys` state drives enforcement). Commits on success and returns any RETURNING
/// rows; rolls the statement back and returns the error on failure (as the engine does).
fn dml(
    cat: &SchemaCatalog,
    pager: &mut MemPager,
    rt: &mut Runtime,
    sql: &str,
) -> Result<Vec<Vec<Value>>, Error> {
    let ast = parse(sql)?;
    let stmt = ast.statements.first().expect("one statement");
    let plan = QueryPlanner::new().plan(stmt, cat)?;
    pager.begin().unwrap();
    let mut exec = StreamingExecutor;
    let mut out = Vec::new();
    let mut result: Result<(), Error> = Ok(());
    {
        match exec.execute(&plan, cat, PagerSet::One { db: DbIndex::MAIN, pager: &mut *pager }) {
            Ok(mut cur) => loop {
                match cur.next_row(rt) {
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

fn text(v: &Value) -> String {
    match v {
        Value::Text(s) => s.clone(),
        other => panic!("expected Text, got {other:?}"),
    }
}

fn assert_fk_error(err: &Error) {
    match err {
        Error::Constraint(m) => assert!(
            m.starts_with("FOREIGN KEY constraint failed"),
            "expected a FOREIGN KEY constraint error, got {m:?}"
        ),
        other => panic!("expected Error::Constraint(FOREIGN KEY ...), got {other:?}"),
    }
}

// ----- (a) child side: a DO UPDATE to a DANGLING reference aborts -----------

#[test]
fn do_update_dangling_child_fk_aborts() {
    // The witness: the INSERT candidate `(10, 1)` references a VALID parent, so
    // the candidate-row child-FK check passes; the `ON CONFLICT DO UPDATE` then rewrites the
    // existing row's `pid` to a DANGLING 999. Under `foreign_keys=ON` real sqlite raises
    // `FOREIGN KEY constraint failed` and leaves the row unchanged — the rewrite must be checked
    // like an UPDATE, not waved through because the candidate was fine.
    let (mut pager, mut cat) = db();
    create(&mut pager, &mut cat, "CREATE TABLE parent(id INTEGER PRIMARY KEY)");
    create(
        &mut pager,
        &mut cat,
        "CREATE TABLE child(id INTEGER PRIMARY KEY, pid INTEGER REFERENCES parent(id))",
    );

    let mut seed = Runtime::new();
    dml(&cat, &mut pager, &mut seed, "INSERT INTO parent VALUES (1)").unwrap();
    dml(&cat, &mut pager, &mut seed, "INSERT INTO child VALUES (10, 1)").unwrap();

    let mut fk = Runtime::new();
    fk.set_foreign_keys(true);
    let err = dml(
        &cat,
        &mut pager,
        &mut fk,
        "INSERT INTO child(id, pid) VALUES (10, 1) ON CONFLICT(id) DO UPDATE SET pid = 999",
    )
    .expect_err("a DO UPDATE that dangles the child FK must abort under foreign_keys=ON");
    assert_fk_error(&err);
    assert_eq!(fk.changes(), 0, "an aborted DO UPDATE records no change");

    let rows = query(&cat, &mut pager, "SELECT id, pid FROM child");
    let got: Vec<(i64, i64)> = rows.iter().map(|r| (int(&r[0]), int(&r[1]))).collect();
    assert_eq!(got, vec![(10, 1)], "the aborted rewrite left child.pid at its original 1");
}

// ----- (b) child side: a DO UPDATE to a VALID reference succeeds ------------

#[test]
fn do_update_valid_child_fk_succeeds() {
    // The check must not OVER-reject: a DO UPDATE whose new key references an EXISTING parent is
    // legal and must go through, updating the row.
    let (mut pager, mut cat) = db();
    create(&mut pager, &mut cat, "CREATE TABLE parent(id INTEGER PRIMARY KEY)");
    create(
        &mut pager,
        &mut cat,
        "CREATE TABLE child(id INTEGER PRIMARY KEY, pid INTEGER REFERENCES parent(id))",
    );

    let mut seed = Runtime::new();
    dml(&cat, &mut pager, &mut seed, "INSERT INTO parent VALUES (1)").unwrap();
    dml(&cat, &mut pager, &mut seed, "INSERT INTO parent VALUES (2)").unwrap();
    dml(&cat, &mut pager, &mut seed, "INSERT INTO child VALUES (10, 1)").unwrap();

    let mut fk = Runtime::new();
    fk.set_foreign_keys(true);
    dml(
        &cat,
        &mut pager,
        &mut fk,
        "INSERT INTO child(id, pid) VALUES (10, 1) ON CONFLICT(id) DO UPDATE SET pid = 2",
    )
    .expect("a DO UPDATE to an existing parent must succeed");
    assert_eq!(fk.changes(), 1, "one row updated");

    let rows = query(&cat, &mut pager, "SELECT pid FROM child WHERE id = 10");
    assert_eq!(rows.iter().map(|r| int(&r[0])).collect::<Vec<_>>(), vec![2]);
}

// ----- (c) child side: with the pragma OFF, the dangling rewrite is allowed -

#[test]
fn do_update_dangling_child_fk_allowed_when_pragma_off() {
    // The enforcement is pragma-gated exactly like every sibling path: with `foreign_keys` OFF
    // (the default) the same dangling DO UPDATE is allowed and writes 999.
    let (mut pager, mut cat) = db();
    create(&mut pager, &mut cat, "CREATE TABLE parent(id INTEGER PRIMARY KEY)");
    create(
        &mut pager,
        &mut cat,
        "CREATE TABLE child(id INTEGER PRIMARY KEY, pid INTEGER REFERENCES parent(id))",
    );

    let mut rt = Runtime::new(); // foreign_keys OFF (default)
    dml(&cat, &mut pager, &mut rt, "INSERT INTO parent VALUES (1)").unwrap();
    dml(&cat, &mut pager, &mut rt, "INSERT INTO child VALUES (10, 1)").unwrap();
    dml(
        &cat,
        &mut pager,
        &mut rt,
        "INSERT INTO child(id, pid) VALUES (10, 1) ON CONFLICT(id) DO UPDATE SET pid = 999",
    )
    .expect("with foreign_keys OFF the dangling rewrite is allowed");

    let rows = query(&cat, &mut pager, "SELECT pid FROM child WHERE id = 10");
    assert_eq!(rows.iter().map(|r| int(&r[0])).collect::<Vec<_>>(), vec![999]);
}

// ----- (d) parent side: a DO UPDATE fires ON UPDATE CASCADE into children ---

#[test]
fn do_update_on_parent_cascades_to_children() {
    // The second witness direction: the parent's referenced UNIQUE key changes in the rewrite
    // ('A' -> 'B'), so `ON UPDATE CASCADE` must rewrite every referencing child's key. The key
    // referenced is a NON-rowid UNIQUE column (an IPK/rowid change is rejected upstream, so the
    // cascade is only reachable via a non-rowid referenced key). `sqlite3_changes()` counts the
    // parent row ONLY — the cascade is a referential action, not a user change.
    let (mut pager, mut cat) = db();
    create(&mut pager, &mut cat, "CREATE TABLE parent(id INTEGER PRIMARY KEY, code TEXT UNIQUE)");
    create(
        &mut pager,
        &mut cat,
        "CREATE TABLE child(cid INTEGER PRIMARY KEY, pcode TEXT REFERENCES parent(code) ON UPDATE CASCADE)",
    );

    let mut seed = Runtime::new();
    dml(&cat, &mut pager, &mut seed, "INSERT INTO parent VALUES (1, 'A')").unwrap();
    dml(&cat, &mut pager, &mut seed, "INSERT INTO child VALUES (100, 'A')").unwrap();
    dml(&cat, &mut pager, &mut seed, "INSERT INTO child VALUES (101, 'A')").unwrap();

    let mut fk = Runtime::new();
    fk.set_foreign_keys(true);
    dml(
        &cat,
        &mut pager,
        &mut fk,
        "INSERT INTO parent(id, code) VALUES (1, 'A') ON CONFLICT(id) DO UPDATE SET code = 'B'",
    )
    .expect("a DO UPDATE changing a referenced key must cascade, not error");
    assert_eq!(fk.changes(), 1, "changes() counts the parent row only, not the cascaded children");

    let prow = query(&cat, &mut pager, "SELECT code FROM parent WHERE id = 1");
    assert_eq!(prow.iter().map(|r| text(&r[0])).collect::<Vec<_>>(), vec!["B".to_string()]);

    let crows = query(&cat, &mut pager, "SELECT cid, pcode FROM child");
    let got: Vec<(i64, String)> = crows.iter().map(|r| (int(&r[0]), text(&r[1]))).collect();
    assert_eq!(
        got,
        vec![(100, "B".to_string()), (101, "B".to_string())],
        "ON UPDATE CASCADE rewrote both children 'A' -> 'B'"
    );
}

// ----- (e) parent side: NO ACTION rejects a key change a child still uses ---

#[test]
fn do_update_on_parent_no_action_aborts_with_live_child() {
    // With the default NO ACTION (no `ON UPDATE` clause), a DO UPDATE that changes a referenced
    // key while a child still points at the OLD value must abort (`FOREIGN KEY constraint
    // failed`) and leave both tables unchanged — the parent-side RESTRICT/NO ACTION direction.
    let (mut pager, mut cat) = db();
    create(&mut pager, &mut cat, "CREATE TABLE parent(id INTEGER PRIMARY KEY, code TEXT UNIQUE)");
    create(
        &mut pager,
        &mut cat,
        "CREATE TABLE child(cid INTEGER PRIMARY KEY, pcode TEXT REFERENCES parent(code))",
    );

    let mut seed = Runtime::new();
    dml(&cat, &mut pager, &mut seed, "INSERT INTO parent VALUES (1, 'A')").unwrap();
    dml(&cat, &mut pager, &mut seed, "INSERT INTO child VALUES (100, 'A')").unwrap();

    let mut fk = Runtime::new();
    fk.set_foreign_keys(true);
    let err = dml(
        &cat,
        &mut pager,
        &mut fk,
        "INSERT INTO parent(id, code) VALUES (1, 'A') ON CONFLICT(id) DO UPDATE SET code = 'B'",
    )
    .expect_err("changing a referenced key under NO ACTION with a live child must abort");
    assert_fk_error(&err);

    let prow = query(&cat, &mut pager, "SELECT code FROM parent WHERE id = 1");
    assert_eq!(prow.iter().map(|r| text(&r[0])).collect::<Vec<_>>(), vec!["A".to_string()]);
    let crow = query(&cat, &mut pager, "SELECT pcode FROM child WHERE cid = 100");
    assert_eq!(crow.iter().map(|r| text(&r[0])).collect::<Vec<_>>(), vec!["A".to_string()]);
}
