//! Integration tests for FOREIGN KEY enforcement on the `OR REPLACE` VICTIM-deletion paths —
//! the delete-shaped half of "a DML path that removes/mutates a keyed row must run the FK
//! enforcement its standalone sibling runs." A REPLACE that deletes a conflicting parent row to
//! make room must fire that row's ON DELETE action exactly like a standalone `DELETE`
//! (`ops/delete.rs`): CASCADE removes the children, SET NULL/SET DEFAULT rewrite them, and
//! RESTRICT/NO ACTION abort — otherwise the REPLACE silently orphans children under
//! `PRAGMA foreign_keys=ON`, diverging from real `sqlite3` (`lang_conflict.html`: REPLACE
//! "deletes pre-existing rows that are causing the constraint violation").
//!
//! Four production sites share this: the rowid `INSERT OR REPLACE` victim loop
//! (`ops/insert.rs`), the rowid `UPDATE OR REPLACE` victim loop (`ops/update.rs`), and the WR
//! `INSERT`/`UPDATE OR REPLACE` victim removal (both via `dml_index::wr_delete_victim_by_pk`).
//!
//! These drive the REAL parse -> plan -> execute path (`QueryPlanner` + `StreamingExecutor`
//! over a real `MemPager` + `SchemaCatalog`), read back with a real `SELECT`, and model the
//! pragma with `Runtime::set_foreign_keys`. NO ACTION is asserted only on GENUINE key removals
//! (a different-rowid victim, so the parent key is truly gone at statement end); the WR victims
//! (keyed by PK, no secondary index possible) are same-key removals, so only their CASCADE
//! direction — which fires during the delete regardless of the re-insert — is asserted, to stay
//! in lockstep with real sqlite's end-of-statement NO ACTION timing.

use minisqlite_btree::init_database;
use minisqlite_catalog::{Catalog, SchemaCatalog};
use minisqlite_exec::{Executor, PagerSet, Runtime, StreamingExecutor};
use minisqlite_pager::{MemPager, Pager};
use minisqlite_plan::{Planner, QueryPlanner};
use minisqlite_sql::{parse, Statement};
use minisqlite_types::{DbIndex, Error, Value};

// ----- fixtures ------------------------------------------------------------

fn db() -> (MemPager, SchemaCatalog) {
    let mut pager = MemPager::new(4096);
    init_database(&mut pager).unwrap();
    (pager, SchemaCatalog::new())
}

fn create(pager: &mut MemPager, cat: &mut SchemaCatalog, sql: &str) {
    let ast = parse(sql).unwrap();
    let Statement::CreateTable(stmt) = &ast.statements[0] else {
        panic!("not a CREATE TABLE: {sql}");
    };
    cat.create_table(pager, stmt, sql).unwrap();
}

/// Run one DML statement through the REAL planner + executor with the caller's `Runtime`.
/// Commits on success (returning any RETURNING rows), rolls back + returns the error on failure.
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

/// Seed a rowid parent `p(a INTEGER PRIMARY KEY, u TEXT UNIQUE)` + a child of `p(a)` with the
/// given ON DELETE action, one parent row (1,'x') and one child (10 -> 1), FK OFF while seeding.
fn seed_rowid(on_delete: &str) -> (MemPager, SchemaCatalog) {
    let (mut pager, mut cat) = db();
    create(&mut pager, &mut cat, "CREATE TABLE p(a INTEGER PRIMARY KEY, u TEXT UNIQUE)");
    create(
        &mut pager,
        &mut cat,
        &format!("CREATE TABLE c(id INTEGER PRIMARY KEY, pa INTEGER REFERENCES p(a){on_delete})"),
    );
    let mut seed = Runtime::new();
    dml(&cat, &mut pager, &mut seed, "INSERT INTO p VALUES (1, 'x')").unwrap();
    dml(&cat, &mut pager, &mut seed, "INSERT INTO c VALUES (10, 1)").unwrap();
    (pager, cat)
}

// ----- (a) rowid INSERT OR REPLACE victim fires ON DELETE CASCADE ----------

#[test]
fn insert_or_replace_rowid_victim_cascades() {
    // The witness: the new row a=2 has no PK conflict but collides on UNIQUE u='x' with
    // the DIFFERENT row a=1, so it routes through detect_conflict -> Action::Replace -> the
    // victim loop, which deletes p(a=1). Under foreign_keys=ON that delete must fire ON DELETE
    // CASCADE, removing c(10). Real sqlite: SELECT count -> 0.
    let (mut pager, cat) = seed_rowid(" ON DELETE CASCADE");
    let mut fk = Runtime::new();
    fk.set_foreign_keys(true);
    dml(&cat, &mut pager, &mut fk, "INSERT OR REPLACE INTO p(a, u) VALUES (2, 'x')")
        .expect("INSERT OR REPLACE succeeds; the victim's cascade is not an error");

    let crows = query(&cat, &mut pager, "SELECT id FROM c");
    assert!(crows.is_empty(), "ON DELETE CASCADE on the REPLACE victim removed c(10)");
    // The victim a=1 is gone; the new row a=2 is present.
    let prows = query(&cat, &mut pager, "SELECT a FROM p");
    assert_eq!(prows.iter().map(|r| int(&r[0])).collect::<Vec<_>>(), vec![2]);
}

// ----- (b) rowid INSERT OR REPLACE victim under NO ACTION aborts ------------

#[test]
fn insert_or_replace_rowid_victim_no_action_aborts() {
    // Default NO ACTION: deleting the victim a=1 (a GENUINE key removal — the new row is a=2, so
    // no a=1 exists at statement end) while c(10) still references it must abort with
    // `FOREIGN KEY constraint failed`, leaving both tables unchanged. Real sqlite aborts too.
    let (mut pager, cat) = seed_rowid("");
    let mut fk = Runtime::new();
    fk.set_foreign_keys(true);
    let err = dml(&cat, &mut pager, &mut fk, "INSERT OR REPLACE INTO p(a, u) VALUES (2, 'x')")
        .expect_err("deleting a NO ACTION parent with a live child must abort");
    assert_fk_error(&err);

    // Rolled back: the original parent a=1 and the child both survive untouched.
    let prows = query(&cat, &mut pager, "SELECT a, u FROM p");
    let pgot: Vec<(i64, String)> = prows.iter().map(|r| (int(&r[0]), text(&r[1]))).collect();
    assert_eq!(pgot, vec![(1, "x".to_string())]);
    let crows = query(&cat, &mut pager, "SELECT id, pa FROM c");
    let cgot: Vec<(i64, i64)> = crows.iter().map(|r| (int(&r[0]), int(&r[1]))).collect();
    assert_eq!(cgot, vec![(10, 1)]);
}

// ----- (c) rowid INSERT OR REPLACE victim: FK OFF leaves the orphan ---------

#[test]
fn insert_or_replace_rowid_victim_pragma_off_orphans() {
    // The enforcement is pragma-gated: with foreign_keys OFF (the default) the same REPLACE
    // deletes a=1 and leaves c(10) dangling — no cascade, no abort — exactly as before the fix.
    let (mut pager, cat) = seed_rowid(" ON DELETE CASCADE");
    let mut rt = Runtime::new(); // FK OFF
    dml(&cat, &mut pager, &mut rt, "INSERT OR REPLACE INTO p(a, u) VALUES (2, 'x')").unwrap();
    let crows = query(&cat, &mut pager, "SELECT id, pa FROM c");
    let cgot: Vec<(i64, i64)> = crows.iter().map(|r| (int(&r[0]), int(&r[1]))).collect();
    assert_eq!(cgot, vec![(10, 1)], "FK off: the child survives as an orphan (no enforcement)");
}

// ----- (d) rowid UPDATE OR REPLACE victim fires ON DELETE CASCADE -----------

#[test]
fn update_or_replace_rowid_victim_cascades() {
    // UPDATE OR REPLACE: setting a=2's UNIQUE u to 'x' collides with the DIFFERENT row a=1, so
    // the update's conflict resolution deletes the victim a=1 (update.rs victim loop). Under
    // foreign_keys=ON that delete fires ON DELETE CASCADE, removing c(10).
    let (mut pager, mut cat) = db();
    create(&mut pager, &mut cat, "CREATE TABLE p(a INTEGER PRIMARY KEY, u TEXT UNIQUE)");
    create(
        &mut pager,
        &mut cat,
        "CREATE TABLE c(id INTEGER PRIMARY KEY, pa INTEGER REFERENCES p(a) ON DELETE CASCADE)",
    );
    let mut seed = Runtime::new();
    dml(&cat, &mut pager, &mut seed, "INSERT INTO p VALUES (1, 'x')").unwrap();
    dml(&cat, &mut pager, &mut seed, "INSERT INTO p VALUES (2, 'y')").unwrap();
    dml(&cat, &mut pager, &mut seed, "INSERT INTO c VALUES (10, 1)").unwrap();

    let mut fk = Runtime::new();
    fk.set_foreign_keys(true);
    dml(&cat, &mut pager, &mut fk, "UPDATE OR REPLACE p SET u = 'x' WHERE a = 2")
        .expect("UPDATE OR REPLACE succeeds; the victim's cascade is not an error");

    let crows = query(&cat, &mut pager, "SELECT id FROM c");
    assert!(crows.is_empty(), "ON DELETE CASCADE on the UPDATE-REPLACE victim removed c(10)");
    // a=1 was the deleted victim; a=2 survives, now carrying u='x'.
    let prows = query(&cat, &mut pager, "SELECT a, u FROM p");
    let pgot: Vec<(i64, String)> = prows.iter().map(|r| (int(&r[0]), text(&r[1]))).collect();
    assert_eq!(pgot, vec![(2, "x".to_string())]);
}

// ----- (e) WR INSERT OR REPLACE victim fires ON DELETE CASCADE --------------

#[test]
fn insert_or_replace_wr_victim_cascades() {
    // A WITHOUT ROWID parent: INSERT OR REPLACE on an existing PK 'x' deletes the stored 'x' row
    // (wr_delete_victim_by_pk) before writing the new one. That delete must fire ON DELETE
    // CASCADE — a REPLACE is delete-then-insert, and the cascade runs during the delete — so
    // c(10) is removed even though a new 'x' is then inserted (matches real sqlite's REPLACE).
    let (mut pager, mut cat) = db();
    create(&mut pager, &mut cat, "CREATE TABLE p(a TEXT PRIMARY KEY, note TEXT) WITHOUT ROWID");
    create(
        &mut pager,
        &mut cat,
        "CREATE TABLE c(id INTEGER PRIMARY KEY, pa TEXT REFERENCES p(a) ON DELETE CASCADE)",
    );
    let mut seed = Runtime::new();
    dml(&cat, &mut pager, &mut seed, "INSERT INTO p VALUES ('x', 'first')").unwrap();
    dml(&cat, &mut pager, &mut seed, "INSERT INTO c VALUES (10, 'x')").unwrap();

    let mut fk = Runtime::new();
    fk.set_foreign_keys(true);
    dml(&cat, &mut pager, &mut fk, "INSERT OR REPLACE INTO p VALUES ('x', 'second')")
        .expect("WR INSERT OR REPLACE succeeds; the victim's cascade is not an error");

    let crows = query(&cat, &mut pager, "SELECT id FROM c");
    assert!(crows.is_empty(), "ON DELETE CASCADE on the WR REPLACE victim removed c(10)");
    // The PK-'x' row was replaced in place (delete + insert), now carrying note='second'.
    let prows = query(&cat, &mut pager, "SELECT a, note FROM p");
    let pgot: Vec<(String, String)> = prows.iter().map(|r| (text(&r[0]), text(&r[1]))).collect();
    assert_eq!(pgot, vec![("x".to_string(), "second".to_string())]);
}

// ----- (f) WR UPDATE OR REPLACE victim fires ON DELETE CASCADE --------------

#[test]
fn update_or_replace_wr_victim_cascades() {
    // A WITHOUT ROWID parent: moving row 'y' to PK 'x' collides with the existing 'x', so the WR
    // update's REPLACE deletes the victim 'x' (wr_delete_victim_by_pk) before the move. That
    // delete must fire ON DELETE CASCADE, removing c(10) which referenced 'x'.
    let (mut pager, mut cat) = db();
    create(&mut pager, &mut cat, "CREATE TABLE p(a TEXT PRIMARY KEY, note TEXT) WITHOUT ROWID");
    create(
        &mut pager,
        &mut cat,
        "CREATE TABLE c(id INTEGER PRIMARY KEY, pa TEXT REFERENCES p(a) ON DELETE CASCADE)",
    );
    let mut seed = Runtime::new();
    dml(&cat, &mut pager, &mut seed, "INSERT INTO p VALUES ('x', 'first')").unwrap();
    dml(&cat, &mut pager, &mut seed, "INSERT INTO p VALUES ('y', 'second')").unwrap();
    dml(&cat, &mut pager, &mut seed, "INSERT INTO c VALUES (10, 'x')").unwrap();

    let mut fk = Runtime::new();
    fk.set_foreign_keys(true);
    dml(&cat, &mut pager, &mut fk, "UPDATE OR REPLACE p SET a = 'x' WHERE a = 'y'")
        .expect("WR UPDATE OR REPLACE succeeds; the victim's cascade is not an error");

    let crows = query(&cat, &mut pager, "SELECT id FROM c");
    assert!(crows.is_empty(), "ON DELETE CASCADE on the WR UPDATE-REPLACE victim removed c(10)");
    // The original 'x' was the deleted victim; 'y' moved onto 'x', carrying note='second'.
    let prows = query(&cat, &mut pager, "SELECT a, note FROM p");
    let pgot: Vec<(String, String)> = prows.iter().map(|r| (text(&r[0]), text(&r[1]))).collect();
    assert_eq!(pgot, vec![("x".to_string(), "second".to_string())]);
}
