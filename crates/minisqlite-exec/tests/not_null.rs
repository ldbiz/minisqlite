//! Integration tests for column `NOT NULL` enforcement in the `INSERT` and `UPDATE`
//! DML operators: hand-built [`Plan`] trees run over a real [`MemPager`] +
//! [`SchemaCatalog`] through the [`Executor`]/[`RowCursor`] seam, then read back through
//! the real storage/catalog. They pin the shared `constraints::enforce_not_null` helper
//! end to end via BOTH operators: the ABORT error and its column detail, `OR IGNORE`
//! skipping (and consuming no rowid), the `OR REPLACE` no-default / with-default cases
//! for INSERT and UPDATE (the with-default one a documented interim narrowing), an
//! omitted (unset) NOT NULL column, and the rowid-alias exemption — including the common
//! table that has BOTH a rowid alias AND a separate enforced NOT NULL column.
//!
//! The operators assume an open write transaction (the engine opens one for DML), so
//! each apply is wrapped in `begin`/`commit`; read-back scans need no transaction. The
//! fixtures mirror `tests/insert.rs` / `tests/update.rs` so this file is self-contained
//! (a separate test binary) and does not contend on those large files.

use minisqlite_btree::init_database;
use minisqlite_catalog::{Catalog, SchemaCatalog};
use minisqlite_exec::{Executor, Runtime, StreamingExecutor};
use minisqlite_expr::EvalExpr;
use minisqlite_pager::{MemPager, Pager};
use minisqlite_plan::{Insert, OnConflict, Plan, PlanNode, SeqScan, Update};
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

fn seqscan(table: &str, n: usize) -> PlanNode {
    PlanNode::SeqScan(SeqScan { db: DbIndex::MAIN, table: table.to_string(), column_count: n })
}

/// Build an `INSERT` plan over a literal `VALUES` source (source width inferred from
/// the first row).
fn insert_plan(
    table: &str,
    n: usize,
    columns: Option<Vec<usize>>,
    rows: Vec<Vec<EvalExpr>>,
    column_affinities: Vec<Affinity>,
    on_conflict: OnConflict,
    returning: Vec<EvalExpr>,
) -> Plan {
    let source_width = rows.first().map(|r| r.len()).unwrap_or(n);
    Plan {
        root: PlanNode::Insert(Insert { db: DbIndex::MAIN,
            table: table.to_string(),
            column_count: n,
            columns,
            source: Box::new(PlanNode::Values { rows }),
            source_width,
            column_affinities,
            on_conflict,
            returning,
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

/// Apply a mutating plan (INSERT or UPDATE) inside a write transaction, draining any
/// `RETURNING` rows and threading the caller's `Runtime` so counters can be read after.
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
    err.expect("expected the statement to error")
}

/// Full table scan back through the executor (no transaction needed for a read).
/// Rows are `[c0, …, c_{N-1}, rowid]`, ascending by rowid.
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

/// Assert `err` is a NOT NULL constraint error whose detail names `detail` (e.g.
/// `"t.a"`). The kind is folded into the message (`Error::Constraint` carries only a
/// string), so the prefix pins the kind — a wrong kind (UNIQUE / PRIMARY KEY) fails.
fn assert_not_null_err(err: &Error, detail: &str) {
    match err {
        Error::Constraint(m) => {
            assert!(m.starts_with("NOT NULL constraint failed"), "expected NOT NULL kind, got {m:?}");
            assert!(m.contains(detail), "expected the column detail {detail:?}, got {m:?}");
        }
        other => panic!("expected Error::Constraint, got {other:?}"),
    }
}

// ----- INSERT: ABORT (default) ---------------------------------------------

#[test]
fn insert_null_into_not_null_column_aborts() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER NOT NULL)");
    let err = run_err(
        &insert_plan(
            "t",
            1,
            None,
            vec![vec![lit(Value::Null)]],
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
    );
    assert_not_null_err(&err, "t.a");
    // The failed statement rolled back: no row was stored.
    assert!(scan_all(&cat, &mut pager, "t", 1).is_empty(), "the NULL row was not stored");
}

#[test]
fn non_null_value_into_not_null_column_inserts_fine() {
    // No false positive: a NOT NULL column given a real value inserts normally.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER NOT NULL, b TEXT)");
    let mut rt = Runtime::new();
    run(
        &insert_plan(
            "t",
            2,
            None,
            vec![vec![lit(Value::Integer(7)), lit(Value::Text("ok".into()))]],
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1);
    assert_eq!((int(&rows[0][0]), text(&rows[0][1]), int(&rows[0][2])), (7, "ok", 1));
    assert_eq!(rt.changes(), 1);
}

#[test]
fn not_null_error_names_the_correct_column() {
    // a is nullable, b is NOT NULL. Inserting (1, NULL) must name b, not a — proving the
    // helper reports the offending column index, not merely "some column".
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER, b INTEGER NOT NULL)");
    let err = run_err(
        &insert_plan(
            "t",
            2,
            None,
            vec![vec![lit(Value::Integer(1)), lit(Value::Null)]],
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
    );
    assert_not_null_err(&err, "t.b");
    if let Error::Constraint(m) = &err {
        assert!(!m.contains("t.a"), "must not name the nullable column a, got {m:?}");
    }
}

// ----- INSERT: OR IGNORE ----------------------------------------------------

#[test]
fn insert_or_ignore_skips_null_row_and_consumes_no_rowid() {
    // `a` is NOT NULL but NOT the rowid alias (plain INTEGER), so rowids auto-assign.
    // The first VALUES row NULLs the NOT NULL column (skipped); the second is valid and
    // must take rowid 1 — the skipped row consumes no rowid, exactly like an ignored
    // UNIQUE conflict (see insert.rs `auto_rowid_is_not_consumed_by_an_ignored_row`).
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER NOT NULL, b TEXT)");
    let mut rt = Runtime::new();
    run(
        &insert_plan(
            "t",
            2,
            None,
            vec![
                vec![lit(Value::Null), lit(Value::Text("skip".into()))],
                vec![lit(Value::Integer(5)), lit(Value::Text("keep".into()))],
            ],
            affinities(&cat, "t"),
            OnConflict::Ignore,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1, "only the valid row is stored");
    assert_eq!((int(&rows[0][0]), text(&rows[0][1]), int(&rows[0][2])), (5, "keep", 1));
    assert_eq!(rt.changes(), 1, "only the inserted row is counted");
    assert_eq!(rt.last_insert_rowid(), 1, "the skipped NULL row consumed no rowid");
}

// ----- INSERT: OR REPLACE (no default => correct ABORT fallback) ------------

#[test]
fn insert_or_replace_null_no_default_aborts() {
    // With no column DEFAULT, REPLACE falls back to ABORT — the CORRECT sqlite behavior
    // for a NOT NULL violation (lang_conflict.html), not a narrowing.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER NOT NULL)");
    assert!(cat.table("t").unwrap().unwrap().columns[0].default.is_none(), "column a has no default");
    let err = run_err(
        &insert_plan(
            "t",
            1,
            None,
            vec![vec![lit(Value::Null)]],
            affinities(&cat, "t"),
            OnConflict::Replace,
            Vec::new(),
        ),
        &cat,
        &mut pager,
    );
    assert_not_null_err(&err, "t.a");
    assert!(scan_all(&cat, &mut pager, "t", 1).is_empty(), "no row stored");
}

// ----- INSERT: OR REPLACE WITH a default (DOCUMENTED NARROWING) -------------

#[test]
fn insert_or_replace_null_with_default_is_documented_narrowing() {
    // DOCUMENTED NARROWING: real sqlite substitutes the column DEFAULT ('x') for a NULL
    // under REPLACE and stores that, succeeding. This executor cannot evaluate a column
    // DEFAULT yet (the plan carries no parsed default exprs — a separate known gap), so
    // it fails closed with the NOT NULL constraint error rather than storing NULL. When
    // defaults are threaded through the plan this test changes CONSCIOUSLY: the row would
    // then store b='x'. Until then it guards against silently regressing to a stored NULL.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER, b TEXT NOT NULL DEFAULT 'x')");
    {
        let tdef = cat.table("t").unwrap().unwrap();
        assert!(tdef.columns[1].not_null, "b is NOT NULL");
        assert_eq!(tdef.columns[1].default.as_deref(), Some("'x'"), "b carries a DEFAULT 'x'");
    }
    let err = run_err(
        &insert_plan(
            "t",
            2,
            None,
            vec![vec![lit(Value::Integer(1)), lit(Value::Null)]],
            affinities(&cat, "t"),
            OnConflict::Replace,
            Vec::new(),
        ),
        &cat,
        &mut pager,
    );
    assert_not_null_err(&err, "t.b");
    assert!(scan_all(&cat, &mut pager, "t", 2).is_empty(), "no NULL (nor default) stored yet");
}

// ----- INSERT: rowid-alias exemption ----------------------------------------

#[test]
fn rowid_alias_null_auto_assigns_despite_not_null() {
    // `a` is INTEGER PRIMARY KEY NOT NULL — the rowid alias. A NULL there means
    // "auto-assign a rowid", NOT a NOT NULL violation: the alias column is exempt even
    // though it is declared NOT NULL.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY NOT NULL, b TEXT)");
    {
        let tdef = cat.table("t").unwrap().unwrap();
        assert_eq!(tdef.rowid_alias, Some(0), "a is the rowid alias");
        assert!(tdef.columns[0].not_null, "a is declared NOT NULL (yet exempt)");
    }
    let mut rt = Runtime::new();
    run(
        &insert_plan(
            "t",
            2,
            None,
            vec![vec![lit(Value::Null), lit(Value::Text("hi".into()))]],
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1, "the row inserted with no NOT NULL error on the alias");
    // The alias column reads back AS the auto-assigned rowid 1.
    assert_eq!((int(&rows[0][0]), text(&rows[0][1]), int(&rows[0][2])), (1, "hi", 1));
}

#[test]
fn alias_is_exempt_while_a_not_null_neighbor_is_still_enforced() {
    // The CRITICAL edge, end to end: a table with BOTH a rowid alias (`a INTEGER PRIMARY
    // KEY`) AND a separate `NOT NULL` column (`b`). The exemption is per-column — only the
    // alias slot is skipped — so the neighbor is still enforced. This is the very common
    // `CREATE TABLE(id INTEGER PRIMARY KEY, name TEXT NOT NULL)` shape and it guards
    // against widening the exemption from `rowid_alias == Some(j)` to
    // `rowid_alias.is_some()` (which would silently disable NOT NULL for the whole table).
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT NOT NULL)");
    // (NULL, NULL): alias `a` is exempt (auto-assign), but `b` is NOT NULL and NULL, so the
    // error must name `t.b` — never the alias `t.a`.
    let err = run_err(
        &insert_plan(
            "t",
            2,
            None,
            vec![vec![lit(Value::Null), lit(Value::Null)]],
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
    );
    assert_not_null_err(&err, "t.b");
    if let Error::Constraint(m) = &err {
        assert!(!m.contains("t.a"), "the alias column a must not be reported, got {m:?}");
    }
    // (NULL, 'ok'): the alias NULL auto-assigns rowid 1 and `b` is non-null, so the row
    // stores fine — proving the alias exemption is real, not that NOT NULL is off entirely.
    let mut rt = Runtime::new();
    run(
        &insert_plan(
            "t",
            2,
            None,
            vec![vec![lit(Value::Null), lit(Value::Text("ok".into()))]],
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1, "the alias NULL auto-assigned while the non-null b stored");
    assert_eq!((int(&rows[0][0]), text(&rows[0][1]), int(&rows[0][2])), (1, "ok", 1));
}

#[test]
fn insert_omitting_a_not_null_column_without_default_aborts() {
    // An OMITTED NOT NULL column (no explicit NULL supplied) is filled with NULL by
    // `map_row`, so the same check catches it: `INSERT INTO t(a) VALUES(1)` with `b TEXT
    // NOT NULL` (no default) raises `NOT NULL constraint failed: t.b`, matching sqlite (a
    // NOT NULL column with no default and no supplied value is an error). Pins the
    // omitted-column path, not just the explicit-`Value::Null` path the other tests use.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER, b TEXT NOT NULL)");
    let err = run_err(
        &insert_plan(
            "t",
            2,
            Some(vec![0]), // supply only column a; b is omitted -> map_row leaves it NULL
            vec![vec![lit(Value::Integer(1))]],
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
    );
    assert_not_null_err(&err, "t.b");
    assert!(scan_all(&cat, &mut pager, "t", 2).is_empty(), "no row stored");
}

// ----- UPDATE: SET a NOT NULL column to NULL --------------------------------

#[test]
fn update_set_not_null_to_null_aborts_and_leaves_row() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER NOT NULL, b TEXT)");
    // Seed one valid row via the real INSERT path.
    let mut rt = Runtime::new();
    run(
        &insert_plan(
            "t",
            2,
            None,
            vec![vec![lit(Value::Integer(5)), lit(Value::Text("row".into()))]],
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    // UPDATE t SET a = NULL  (default Abort) -> NOT NULL error, row unchanged.
    let err = run_err(
        &update_plan(
            "t",
            2,
            vec![(0, lit(Value::Null))],
            seqscan("t", 2),
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
    );
    assert_not_null_err(&err, "t.a");
    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1);
    assert_eq!(
        (int(&rows[0][0]), text(&rows[0][1]), int(&rows[0][2])),
        (5, "row", 1),
        "the row is unchanged after the statement rolled back"
    );
}

#[test]
fn update_or_ignore_set_null_leaves_row_untouched() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER NOT NULL, b TEXT)");
    let mut rt = Runtime::new();
    run(
        &insert_plan(
            "t",
            2,
            None,
            vec![vec![lit(Value::Integer(5)), lit(Value::Text("row".into()))]],
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    // UPDATE OR IGNORE t SET a = NULL -> the row is skipped: no error, no change, row
    // left exactly as it was (fresh Runtime so `changes()` is this statement's count).
    let mut rt2 = Runtime::new();
    let out = run(
        &update_plan(
            "t",
            2,
            vec![(0, lit(Value::Null))],
            seqscan("t", 2),
            affinities(&cat, "t"),
            OnConflict::Ignore,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt2,
    );
    assert!(out.is_empty(), "no RETURNING rows");
    assert_eq!(rt2.changes(), 0, "the skipped row is not counted as a change");
    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1);
    assert_eq!(
        (int(&rows[0][0]), text(&rows[0][1]), int(&rows[0][2])),
        (5, "row", 1),
        "the row is left untouched"
    );
}

#[test]
fn update_or_replace_set_null_no_default_aborts_and_leaves_row() {
    // Symmetry with `insert_or_replace_null_no_default_aborts`: `UPDATE OR REPLACE` routes
    // through the SAME `enforce_not_null` Replace arm as INSERT, so with no column default
    // it fails closed with the NOT NULL error (never stores NULL) and leaves the row
    // unchanged. Closes the INSERT/UPDATE symmetry for the REPLACE policy.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER NOT NULL, b TEXT)");
    let mut rt = Runtime::new();
    run(
        &insert_plan(
            "t",
            2,
            None,
            vec![vec![lit(Value::Integer(5)), lit(Value::Text("row".into()))]],
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    let err = run_err(
        &update_plan(
            "t",
            2,
            vec![(0, lit(Value::Null))],
            seqscan("t", 2),
            affinities(&cat, "t"),
            OnConflict::Replace,
            Vec::new(),
        ),
        &cat,
        &mut pager,
    );
    assert_not_null_err(&err, "t.a");
    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1);
    assert_eq!(
        (int(&rows[0][0]), text(&rows[0][1]), int(&rows[0][2])),
        (5, "row", 1),
        "the row is unchanged after the statement rolled back"
    );
}
