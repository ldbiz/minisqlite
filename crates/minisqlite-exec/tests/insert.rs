//! Integration tests for the `INSERT` DML operator: hand-built [`Plan`] trees run
//! over a real [`MemPager`] + [`SchemaCatalog`] through the [`Executor`]/[`RowCursor`]
//! seam, then read back through the real storage/catalog (not a private mock). These
//! pin the operator's behavior end to end: rowid assignment, affinity on store,
//! `ON CONFLICT`, `RETURNING`, index maintenance, and the `Runtime` change counters.
//!
//! The operator assumes an open write transaction (the engine opens one for DML), so
//! each apply is wrapped in `begin`/`commit`; read-back scans need no transaction.

use minisqlite_btree::{init_database, IndexCursor, TableCursor};
use minisqlite_catalog::{Catalog, SchemaCatalog};
use minisqlite_exec::{Executor, Runtime, StreamingExecutor};
use minisqlite_expr::{CmpOp, CompareMeta, EvalExpr};
use minisqlite_fileformat::decode_record;
use minisqlite_pager::{MemPager, PageId, Pager};
use minisqlite_plan::{CheckConstraint, Insert, OnConflict, Plan, PlanNode, SeqScan};
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

fn col(i: usize) -> EvalExpr {
    EvalExpr::Column(i)
}

fn lit(v: Value) -> EvalExpr {
    EvalExpr::Literal(v)
}

/// `left <op> right` with plain binary collation and no operand affinity — enough to
/// build a real CHECK predicate (e.g. `x > 0`) that reads a base column register.
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

/// Build an `INSERT` plan over a literal `VALUES` source. `source_width` is inferred
/// from the first row (all `VALUES` rows are equal width).
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

/// Apply an `INSERT` inside a write transaction and collect any `RETURNING` rows,
/// threading the caller's `Runtime` so the change counters can be read afterward.
fn run_insert(
    plan: &Plan,
    cat: &SchemaCatalog,
    pager: &mut MemPager,
    rt: &mut Runtime,
) -> Vec<Vec<Value>> {
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

/// Full table scan back through the executor (no transaction needed for a read).
/// Returns rows in the base-table shape `[c0, …, c_{N-1}, rowid]`.
fn scan_all(cat: &SchemaCatalog, pager: &mut MemPager, table: &str, n: usize) -> Vec<Vec<Value>> {
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
    out
}

// ----- (1) VALUES source, positional map, auto rowids ----------------------

#[test]
fn insert_values_then_scan_back_with_auto_rowids() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER, b TEXT)");
    let mut rt = Runtime::new();
    let plan = insert_plan(
        "t",
        2,
        None,
        vec![
            vec![lit(Value::Integer(10)), lit(Value::Text("x".into()))],
            vec![lit(Value::Integer(20)), lit(Value::Text("y".into()))],
        ],
        affinities(&cat, "t"),
        OnConflict::Abort,
        Vec::new(),
    );
    let out = run_insert(&plan, &cat, &mut pager, &mut rt);
    assert!(out.is_empty(), "no RETURNING clause yields no rows");

    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 2);
    // Row shape is [a, b, rowid]; rowids auto-assign 1, 2 in insertion order.
    assert_eq!((int(&rows[0][0]), text(&rows[0][1]), int(&rows[0][2])), (10, "x", 1));
    assert_eq!((int(&rows[1][0]), text(&rows[1][1]), int(&rows[1][2])), (20, "y", 2));
}

// ----- explicit target column list -----------------------------------------

#[test]
fn insert_with_target_column_list_leaves_unnamed_null() {
    // INSERT INTO t(b) VALUES('only b'): column map [1], so the single source value
    // lands in column b and column a stays NULL (DEFAULT eval is a later enhancement).
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER, b TEXT)");
    let mut rt = Runtime::new();
    let plan = insert_plan(
        "t",
        2,
        Some(vec![1]),
        vec![vec![lit(Value::Text("only b".into()))]],
        affinities(&cat, "t"),
        OnConflict::Abort,
        Vec::new(),
    );
    run_insert(&plan, &cat, &mut pager, &mut rt);

    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1);
    assert!(rows[0][0].is_null(), "unnamed column a is NULL");
    assert_eq!(text(&rows[0][1]), "only b");
    assert_eq!(int(&rows[0][2]), 1);
}

// ----- (2) INTEGER PRIMARY KEY rowid alias ---------------------------------

#[test]
fn integer_primary_key_explicit_then_null_auto_and_readback() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT)");
    assert_eq!(cat.table("t").unwrap().unwrap().rowid_alias, Some(0), "a is the rowid alias");
    let mut rt = Runtime::new();
    let plan = insert_plan(
        "t",
        2,
        None,
        vec![
            vec![lit(Value::Integer(5)), lit(Value::Text("five".into()))],
            vec![lit(Value::Null), lit(Value::Text("auto".into()))],
        ],
        affinities(&cat, "t"),
        OnConflict::Abort,
        Vec::new(),
    );
    run_insert(&plan, &cat, &mut pager, &mut rt);

    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 2);
    // Explicit rowid 5 is honored; the alias column reads back AS the rowid (the
    // stored record holds NULL there, refilled on read).
    assert_eq!((int(&rows[0][0]), text(&rows[0][1]), int(&rows[0][2])), (5, "five", 5));
    // NULL PK auto-assigns max(rowid)+1 = 6, and likewise reads back as the rowid.
    assert_eq!((int(&rows[1][0]), text(&rows[1][1]), int(&rows[1][2])), (6, "auto", 6));
}

// ----- (2b) INTEGER PRIMARY KEY value coercion (lang_createtable.html §5) ----
// The rowid alias holds ONLY a signed 64-bit integer. A blob, a fractional or
// out-of-range real, or a non-numeric string cannot be losslessly converted and is a
// "datatype mismatch" (an Error::Sql — the facade has no MISMATCH variant) that aborts
// the insert writing nothing; an integral real or a numeric string converts to that
// rowid. Regression for the bug where ANY non-integer alias value silently fell through
// to an auto-assigned rowid (only NULL should auto-assign).

/// Assert an INSERT whose rowid-alias value (column 0) is `bad` errors with a datatype
/// mismatch and writes NO row.
fn assert_rowid_alias_mismatch(bad: Value) {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT)");
    let plan = insert_plan(
        "t",
        2,
        None,
        vec![vec![lit(bad.clone()), lit(Value::Text("x".into()))]],
        affinities(&cat, "t"),
        OnConflict::Abort,
        Vec::new(),
    );
    let err = run_insert_err(&plan, &cat, &mut pager);
    match err {
        Error::Sql(m) => assert_eq!(m, "datatype mismatch", "value {bad:?}"),
        other => panic!("expected Error::Sql(\"datatype mismatch\") for {bad:?}, got {other:?}"),
    }
    assert!(scan_all(&cat, &mut pager, "t", 2).is_empty(), "aborted insert wrote a row for {bad:?}");
}

#[test]
fn non_numeric_text_into_rowid_alias_is_datatype_mismatch() {
    assert_rowid_alias_mismatch(Value::Text("notint".into()));
}

#[test]
fn fractional_real_into_rowid_alias_is_datatype_mismatch() {
    assert_rowid_alias_mismatch(Value::Real(2.5));
}

#[test]
fn out_of_range_real_into_rowid_alias_is_datatype_mismatch() {
    // 1e30 is integral in value but far outside i64 range: a mismatch, never a clamp.
    assert_rowid_alias_mismatch(Value::Real(1e30));
}

#[test]
fn blob_into_rowid_alias_is_datatype_mismatch() {
    assert_rowid_alias_mismatch(Value::Blob(vec![1, 2, 3]));
}

#[test]
fn integral_real_into_rowid_alias_converts_to_that_rowid() {
    // datatype3.html §3 / lang_createtable.html §5: an exactly-integral real is
    // losslessly converted (2.0 -> rowid 2), neither rejected nor truncated.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT)");
    let mut rt = Runtime::new();
    let plan = insert_plan(
        "t",
        2,
        None,
        vec![vec![lit(Value::Real(2.0)), lit(Value::Text("x".into()))]],
        affinities(&cat, "t"),
        OnConflict::Abort,
        Vec::new(),
    );
    run_insert(&plan, &cat, &mut pager, &mut rt);
    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1);
    // The alias column and the rowid both read back as the converted integer 2.
    assert_eq!((int(&rows[0][0]), text(&rows[0][1]), int(&rows[0][2])), (2, "x", 2));
}

#[test]
fn numeric_text_into_rowid_alias_converts_to_that_rowid() {
    // lang_createtable.html §5: a string that losslessly converts to an integer IS the
    // rowid (real sqlite: INSERT INTO k VALUES('42', ...) -> rowid 42), the spec-correct
    // behavior beyond a blanket "text -> error".
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT)");
    let mut rt = Runtime::new();
    let plan = insert_plan(
        "t",
        2,
        None,
        vec![vec![lit(Value::Text("42".into())), lit(Value::Text("x".into()))]],
        affinities(&cat, "t"),
        OnConflict::Abort,
        Vec::new(),
    );
    run_insert(&plan, &cat, &mut pager, &mut rt);
    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1);
    assert_eq!((int(&rows[0][0]), text(&rows[0][1]), int(&rows[0][2])), (42, "x", 42));
}

#[test]
fn null_into_rowid_alias_auto_assigns() {
    // §5: a NULL alias slot is the one non-integer that is NOT a mismatch — it
    // auto-assigns max(rowid)+1 (1 for the first row).
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT)");
    let mut rt = Runtime::new();
    let plan = insert_plan(
        "t",
        2,
        None,
        vec![vec![lit(Value::Null), lit(Value::Text("x".into()))]],
        affinities(&cat, "t"),
        OnConflict::Abort,
        Vec::new(),
    );
    run_insert(&plan, &cat, &mut pager, &mut rt);
    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1);
    assert_eq!((int(&rows[0][0]), text(&rows[0][1]), int(&rows[0][2])), (1, "x", 1));
}

// ----- (3) affinity applied on store ---------------------------------------

#[test]
fn text_stored_into_integer_column_gets_integer_affinity() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER)");
    let mut rt = Runtime::new();
    let plan = insert_plan(
        "t",
        1,
        None,
        vec![vec![lit(Value::Text("123".into()))]],
        affinities(&cat, "t"),
        OnConflict::Abort,
        Vec::new(),
    );
    run_insert(&plan, &cat, &mut pager, &mut rt);

    let rows = scan_all(&cat, &mut pager, "t", 1);
    assert_eq!(rows.len(), 1);
    assert!(
        matches!(rows[0][0], Value::Integer(123)),
        "Text \"123\" is coerced to Integer(123) by INTEGER affinity, got {:?}",
        rows[0][0]
    );
}

// ----- (4) ON CONFLICT: Ignore / Replace / Abort ---------------------------

#[test]
fn on_conflict_ignore_skips_the_duplicate_rowid() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT)");
    let mut rt = Runtime::new();
    run_insert(
        &insert_plan(
            "t",
            2,
            None,
            vec![vec![lit(Value::Integer(1)), lit(Value::Text("first".into()))]],
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    let out = run_insert(
        &insert_plan(
            "t",
            2,
            None,
            vec![vec![lit(Value::Integer(1)), lit(Value::Text("second".into()))]],
            affinities(&cat, "t"),
            OnConflict::Ignore,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    assert!(out.is_empty());

    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1, "the duplicate was skipped, not inserted");
    assert_eq!(text(&rows[0][1]), "first", "the original row is untouched");
}

#[test]
fn on_conflict_replace_overwrites_the_duplicate_rowid() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT)");
    let mut rt = Runtime::new();
    run_insert(
        &insert_plan(
            "t",
            2,
            None,
            vec![vec![lit(Value::Integer(1)), lit(Value::Text("first".into()))]],
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    run_insert(
        &insert_plan(
            "t",
            2,
            None,
            vec![vec![lit(Value::Integer(1)), lit(Value::Text("second".into()))]],
            affinities(&cat, "t"),
            OnConflict::Replace,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );

    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1, "Replace overwrote in place, no second row");
    assert_eq!((int(&rows[0][2]), text(&rows[0][1])), (1, "second"), "new value at rowid 1");
}

#[test]
fn on_conflict_abort_errors_on_the_duplicate_rowid() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT)");
    let mut rt = Runtime::new();
    run_insert(
        &insert_plan(
            "t",
            2,
            None,
            vec![vec![lit(Value::Integer(1)), lit(Value::Text("first".into()))]],
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    let err = run_insert_err(
        &insert_plan(
            "t",
            2,
            None,
            vec![vec![lit(Value::Integer(1)), lit(Value::Text("dup".into()))]],
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
    );
    assert!(matches!(err, Error::Constraint(_)), "duplicate rowid under Abort -> Constraint, got {err:?}");

    // The rolled-back statement left the original row intact.
    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1);
    assert_eq!(text(&rows[0][1]), "first");
}

// ----- (5) RETURNING -------------------------------------------------------

#[test]
fn returning_yields_the_inserted_row() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER, b TEXT)");
    let mut rt = Runtime::new();
    // RETURNING a, b, rowid over the inserted row [c0, c1, rowid].
    let plan = insert_plan(
        "t",
        2,
        None,
        vec![vec![lit(Value::Integer(7)), lit(Value::Text("g".into()))]],
        affinities(&cat, "t"),
        OnConflict::Abort,
        vec![col(0), col(1), col(2)],
    );
    let out = run_insert(&plan, &cat, &mut pager, &mut rt);
    assert_eq!(out.len(), 1);
    assert_eq!((int(&out[0][0]), text(&out[0][1]), int(&out[0][2])), (7, "g", 1));
}

#[test]
fn returning_reflects_stored_affinity_and_alias_rowid() {
    // RETURNING returns the STORED (affinity-applied) values, and the INTEGER PRIMARY
    // KEY column reads back as its rowid — not the raw source value.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b INTEGER)");
    let mut rt = Runtime::new();
    let plan = insert_plan(
        "t",
        2,
        None,
        vec![vec![lit(Value::Null), lit(Value::Text("42".into()))]],
        affinities(&cat, "t"),
        OnConflict::Abort,
        vec![col(0), col(1)],
    );
    let out = run_insert(&plan, &cat, &mut pager, &mut rt);
    assert_eq!(out.len(), 1);
    assert_eq!(int(&out[0][0]), 1, "alias column RETURNS the auto rowid");
    assert!(matches!(out[0][1], Value::Integer(42)), "Text '42' RETURNS Integer(42) after affinity, got {:?}", out[0][1]);
}

// ----- (6) index maintenance -----------------------------------------------

#[test]
fn insert_writes_a_matching_index_entry() {
    let (mut pager, mut cat) = db_with_table("CREATE TABLE t(a INTEGER, b TEXT)");
    create_index(&mut pager, &mut cat, "CREATE INDEX idx_b ON t(b)");
    let mut rt = Runtime::new();
    run_insert(
        &insert_plan(
            "t",
            2,
            None,
            vec![vec![lit(Value::Integer(1)), lit(Value::Text("hello".into()))]],
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );

    let idx_root = cat.index("idx_b").unwrap().unwrap().root_page;
    let mut cur = IndexCursor::open(&pager, idx_root).unwrap();
    assert!(cur.first().unwrap(), "the index has an entry after the insert");
    // The index key is [b, rowid] = [Text("hello"), Integer(1)].
    let key = decode_record(&cur.key().unwrap());
    assert_eq!(key.len(), 2);
    assert_eq!(text(&key[0]), "hello");
    assert_eq!(int(&key[1]), 1);
    assert!(!cur.next().unwrap(), "exactly one index entry");
}

#[test]
fn unique_index_rejects_a_duplicate_key() {
    // A UNIQUE index enforces distinctness of the indexed value across rows (distinct
    // rowids); the second insert of the same key errors under the default Abort.
    let (mut pager, mut cat) = db_with_table("CREATE TABLE t(a INTEGER, b TEXT)");
    create_index(&mut pager, &mut cat, "CREATE UNIQUE INDEX uq_b ON t(b)");
    let mut rt = Runtime::new();
    run_insert(
        &insert_plan(
            "t",
            2,
            None,
            vec![vec![lit(Value::Integer(1)), lit(Value::Text("dup".into()))]],
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    let err = run_insert_err(
        &insert_plan(
            "t",
            2,
            None,
            vec![vec![lit(Value::Integer(2)), lit(Value::Text("dup".into()))]],
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
    );
    assert!(matches!(err, Error::Constraint(_)), "duplicate UNIQUE key -> Constraint, got {err:?}");

    // Distinct NULLs are allowed in a UNIQUE index (NULLs are never equal).
    run_insert(
        &insert_plan(
            "t",
            2,
            None,
            vec![
                vec![lit(Value::Integer(3)), lit(Value::Null)],
                vec![lit(Value::Integer(4)), lit(Value::Null)],
            ],
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 3, "the one 'dup' row plus two NULL-keyed rows");
}

// ----- (7) Runtime change counters -----------------------------------------

#[test]
fn runtime_counters_track_the_inserts() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER)");
    let mut rt = Runtime::new();
    let plan = insert_plan(
        "t",
        1,
        None,
        vec![
            vec![lit(Value::Integer(10))],
            vec![lit(Value::Integer(20))],
            vec![lit(Value::Integer(30))],
        ],
        affinities(&cat, "t"),
        OnConflict::Abort,
        Vec::new(),
    );
    run_insert(&plan, &cat, &mut pager, &mut rt);
    assert_eq!(rt.changes(), 3, "three rows inserted");
    assert_eq!(rt.total_changes(), 3);
    assert_eq!(rt.last_insert_rowid(), 3, "last auto-assigned rowid is 3");
}

#[test]
fn ignored_rows_do_not_bump_the_change_counter() {
    // OnConflict::Ignore rows are skipped entirely, including the change counters.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY)");
    let mut rt = Runtime::new();
    run_insert(
        &insert_plan(
            "t",
            1,
            None,
            vec![vec![lit(Value::Integer(1))]],
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    assert_eq!(rt.changes(), 1);
    // Second statement: a dup (ignored) and a fresh row (rowid 2) -> only +1 change.
    run_insert(
        &insert_plan(
            "t",
            1,
            None,
            vec![vec![lit(Value::Integer(1))], vec![lit(Value::Integer(2))]],
            affinities(&cat, "t"),
            OnConflict::Ignore,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    assert_eq!(rt.changes(), 2, "one ignored + one inserted since construction");
    assert_eq!(rt.last_insert_rowid(), 2);
}

// ----- (7.5) explicit rowid on a plain rowid table -------------------------

/// Build `INSERT INTO t(rowid, v) VALUES (rowid_val, v_val)` on a PLAIN rowid table
/// `t(v)` (N=1). The target list names the hidden rowid, so — exactly as the planner
/// encodes it — `columns` carries the SENTINEL N=1 at position 0 (skipped on store) and
/// `v` at position 1, with `rowid_source = Some(0)` pointing the executor at the supplied
/// rowid value. `insert_plan` can't build this (it hardcodes `rowid_source: None`).
fn explicit_rowid_insert_plan(cat: &SchemaCatalog, rowid_val: Value, v_val: Value) -> Plan {
    Plan {
        root: PlanNode::Insert(Insert { db: DbIndex::MAIN,
            table: "t".to_string(),
            column_count: 1,
            columns: Some(vec![1, 0]),
            source: Box::new(PlanNode::Values { rows: vec![vec![lit(rowid_val), lit(v_val)]] }),
            source_width: 2,
            column_affinities: affinities(cat, "t"),
            on_conflict: OnConflict::Abort,
            returning: Vec::new(),
            triggers: Vec::new(),
            checks: Vec::new(),
            rowid_source: Some(0),
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

/// Assert an explicit bare-`rowid` value supplied through the plain-table `rowid_source`
/// channel that is NOT losslessly integral errors with a datatype mismatch and writes NO
/// row — the SAME `must_be_int` coercion the INTEGER PRIMARY KEY alias path enforces
/// (mirrors `assert_rowid_alias_mismatch`, now pinned on the sentinel/`rowid_source` path).
fn assert_explicit_rowid_mismatch(bad: Value) {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(v TEXT)");
    let plan = explicit_rowid_insert_plan(&cat, bad.clone(), Value::Text("x".into()));
    let err = run_insert_err(&plan, &cat, &mut pager);
    match err {
        Error::Sql(m) => assert_eq!(m, "datatype mismatch", "value {bad:?}"),
        other => panic!("expected Error::Sql(\"datatype mismatch\") for {bad:?}, got {other:?}"),
    }
    assert!(scan_all(&cat, &mut pager, "t", 1).is_empty(), "aborted insert wrote a row for {bad:?}");
}

#[test]
fn explicit_rowid_on_plain_table_is_stored_then_auto_continues_from_max() {
    // A plain rowid table (no INTEGER PRIMARY KEY) whose target list names `rowid`: the
    // planner encodes that as the SENTINEL `N` (=1) in `columns` with `rowid_source`
    // pointing at the source value. The executor must store the supplied 50 AS the rowid
    // (not as a column), and a later auto-rowid insert must continue at max+1 = 51 —
    // matching real sqlite (the exec side of `conformance_rowid::explicit_rowid_*`).
    let (mut pager, cat) = db_with_table("CREATE TABLE t(v TEXT)");
    let mut rt = Runtime::new();

    // INSERT INTO t(rowid, v) VALUES (50, 'x') via the sentinel/`rowid_source` channel.
    let explicit = explicit_rowid_insert_plan(&cat, Value::Integer(50), Value::Text("x".into()));
    run_insert(&explicit, &cat, &mut pager, &mut rt);
    assert_eq!(rt.last_insert_rowid(), 50, "the explicit rowid 50 is used and reported");

    // INSERT INTO t(v) VALUES ('y') -> auto rowid is max(50)+1 = 51.
    let auto = insert_plan(
        "t",
        1,
        Some(vec![0]),
        vec![vec![lit(Value::Text("y".into()))]],
        affinities(&cat, "t"),
        OnConflict::Abort,
        Vec::new(),
    );
    run_insert(&auto, &cat, &mut pager, &mut rt);
    assert_eq!(rt.last_insert_rowid(), 51, "the next auto rowid continues from max+1");

    // Read back: scan_all yields [v, rowid]. rowid 50 holds 'x', rowid 51 holds 'y', and
    // the explicit-rowid value was NOT stored into the `v` column (it stayed 'x').
    let rows = scan_all(&cat, &mut pager, "t", 1);
    let mut by_rowid: Vec<(i64, String)> =
        rows.iter().map(|r| (int(&r[1]), text(&r[0]).to_string())).collect();
    by_rowid.sort_by_key(|(rid, _)| *rid);
    assert_eq!(by_rowid, vec![(50, "x".to_string()), (51, "y".to_string())]);
}

// The explicit bare-rowid channel (`rowid_source`) coerces the supplied value through the
// SAME `must_be_int` path as the INTEGER PRIMARY KEY alias (lang_createtable.html §5): an
// integer — or a losslessly-integral numeric string / real — IS the rowid; a NULL
// auto-assigns; a fractional or out-of-range real, a non-numeric string, or a blob is a
// "datatype mismatch" that aborts. These pin that parity on the plain-table path (the
// alias-path twins live in the (2b) section above), so a regression that made this channel
// diverge — silently auto-assigning a bad value, or rejecting a lossless one — fails loud.

#[test]
fn explicit_rowid_lossless_string_on_plain_table_converts_to_that_rowid() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(v TEXT)");
    let mut rt = Runtime::new();
    let plan = explicit_rowid_insert_plan(&cat, Value::Text("50".into()), Value::Text("x".into()));
    run_insert(&plan, &cat, &mut pager, &mut rt);
    assert_eq!(rt.last_insert_rowid(), 50, "numeric string '50' converts to rowid 50");
    // The rowid value must NOT leak into the data column: v is still 'x', rowid is 50.
    let rows = scan_all(&cat, &mut pager, "t", 1);
    assert_eq!((text(&rows[0][0]), int(&rows[0][1])), ("x", 50));
}

#[test]
fn explicit_null_rowid_on_plain_table_auto_assigns_not_mismatch() {
    // A NULL supplied for the explicit rowid folds to MustBeInt::Null and auto-assigns
    // max+1 (== 1 on an empty table), exactly like a NULL INTEGER PRIMARY KEY — it is the
    // one non-integer value that is NOT a datatype mismatch.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(v TEXT)");
    let mut rt = Runtime::new();
    let plan = explicit_rowid_insert_plan(&cat, Value::Null, Value::Text("x".into()));
    run_insert(&plan, &cat, &mut pager, &mut rt);
    assert_eq!(rt.last_insert_rowid(), 1, "NULL explicit rowid auto-assigns rowid 1");
}

#[test]
fn explicit_rowid_fractional_real_on_plain_table_is_datatype_mismatch() {
    assert_explicit_rowid_mismatch(Value::Real(2.5));
}

#[test]
fn explicit_rowid_non_numeric_text_on_plain_table_is_datatype_mismatch() {
    assert_explicit_rowid_mismatch(Value::Text("notint".into()));
}

#[test]
fn explicit_rowid_blob_on_plain_table_is_datatype_mismatch() {
    assert_explicit_rowid_mismatch(Value::Blob(vec![1, 2, 3]));
}

// ----- (8) OR REPLACE keeps table and indexes consistent -------------------

/// Collect every entry in an index b-tree (decoded key records), in index order.
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

#[test]
fn on_conflict_replace_cleans_up_stale_index_entries() {
    // REPLACE on a rowid conflict must remove the replaced row's OLD secondary-index
    // entry, not merely overwrite the table row — otherwise the index keeps a phantom
    // ['old', 1] pointing at a row whose value is now 'new' (a lookup returns a row
    // that no longer has that value). `index_delete` makes this sound; this pins that
    // the stale entry is gone.
    let (mut pager, mut cat) = db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT)");
    create_index(&mut pager, &mut cat, "CREATE INDEX idx_b ON t(b)");
    let mut rt = Runtime::new();
    run_insert(
        &insert_plan(
            "t",
            2,
            None,
            vec![vec![lit(Value::Integer(1)), lit(Value::Text("old".into()))]],
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    run_insert(
        &insert_plan(
            "t",
            2,
            None,
            vec![vec![lit(Value::Integer(1)), lit(Value::Text("new".into()))]],
            affinities(&cat, "t"),
            OnConflict::Replace,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );

    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1, "one live row after the replace");
    assert_eq!((int(&rows[0][2]), text(&rows[0][1])), (1, "new"));

    let idx_root = cat.index("idx_b").unwrap().unwrap().root_page;
    let keys = index_keys(&pager, idx_root);
    assert_eq!(keys.len(), 1, "exactly one index entry — the stale ['old',1] was deleted");
    assert_eq!((text(&keys[0][0]), int(&keys[0][1])), ("new", 1));
}

#[test]
fn on_conflict_replace_deletes_the_unique_conflicting_row() {
    // A fresh rowid whose UNIQUE key matches a DIFFERENT existing rowid: REPLACE deletes
    // that conflicting row (its table row AND index entry), then inserts the new one —
    // sqlite's "delete the conflicting rows, then insert".
    let (mut pager, mut cat) = db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT)");
    create_index(&mut pager, &mut cat, "CREATE UNIQUE INDEX uq_b ON t(b)");
    let mut rt = Runtime::new();
    run_insert(
        &insert_plan(
            "t",
            2,
            None,
            vec![
                vec![lit(Value::Integer(1)), lit(Value::Text("x".into()))],
                vec![lit(Value::Integer(2)), lit(Value::Text("y".into()))],
            ],
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    // rowid 3 is fresh, but b='y' collides with row 2 on the UNIQUE index.
    run_insert(
        &insert_plan(
            "t",
            2,
            None,
            vec![vec![lit(Value::Integer(3)), lit(Value::Text("y".into()))]],
            affinities(&cat, "t"),
            OnConflict::Replace,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );

    let mut rows = scan_all(&cat, &mut pager, "t", 2);
    rows.sort_by_key(|r| int(&r[2]));
    assert_eq!(rows.len(), 2, "row 2 deleted, row 3 inserted");
    assert_eq!((int(&rows[0][2]), text(&rows[0][1])), (1, "x"));
    assert_eq!((int(&rows[1][2]), text(&rows[1][1])), (3, "y"), "'y' now lives at rowid 3");

    // The UNIQUE index holds exactly ['x',1] and ['y',3] (index order) — no stale ['y',2].
    let idx_root = cat.index("uq_b").unwrap().unwrap().root_page;
    let keys = index_keys(&pager, idx_root);
    assert_eq!(keys.len(), 2, "no stale entry for the deleted row 2");
    assert_eq!((text(&keys[0][0]), int(&keys[0][1])), ("x", 1));
    assert_eq!((text(&keys[1][0]), int(&keys[1][1])), ("y", 3));
}

// ----- (9) auto-rowid is not consumed by a skipped row ---------------------

#[test]
fn auto_rowid_is_not_consumed_by_an_ignored_row() {
    // Within one INSERT OR IGNORE, an auto-rowid row skipped on a UNIQUE conflict must
    // NOT consume a rowid: the next auto row reuses max(rowid)+1, matching sqlite.
    // (Regression: the counter used to advance BEFORE the conflict check, so the
    // surviving row got rowid 3 instead of 2.)
    let (mut pager, mut cat) = db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT)");
    create_index(&mut pager, &mut cat, "CREATE UNIQUE INDEX uq_b ON t(b)");
    let mut rt = Runtime::new();
    // Seed a committed row: auto rowid 1, b='dup'.
    run_insert(
        &insert_plan(
            "t",
            2,
            None,
            vec![vec![lit(Value::Null), lit(Value::Text("dup".into()))]],
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    // OR IGNORE: the first auto row (b='dup') is skipped on the UNIQUE index; the second
    // (b='fresh') proceeds and must take rowid 2, not 3.
    run_insert(
        &insert_plan(
            "t",
            2,
            None,
            vec![
                vec![lit(Value::Null), lit(Value::Text("dup".into()))],
                vec![lit(Value::Null), lit(Value::Text("fresh".into()))],
            ],
            affinities(&cat, "t"),
            OnConflict::Ignore,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );

    let mut rows = scan_all(&cat, &mut pager, "t", 2);
    rows.sort_by_key(|r| int(&r[2]));
    assert_eq!(rows.len(), 2);
    assert_eq!((int(&rows[0][2]), text(&rows[0][1])), (1, "dup"));
    assert_eq!(
        (int(&rows[1][2]), text(&rows[1][1])),
        (2, "fresh"),
        "the skipped auto row did not consume rowid 2"
    );
    assert_eq!(rt.last_insert_rowid(), 2);
}

// ----- (10) INTEGER PRIMARY KEY stored as NULL bytes -----------------------

#[test]
fn integer_primary_key_is_stored_as_null_in_the_record() {
    // On-disk format: the INTEGER PRIMARY KEY value is NOT duplicated in the record —
    // its slot stores NULL and the rowid lives in the b-tree key. scan/RETURNING both
    // refill it on read, so pin the STORED bytes directly to catch a regression that
    // stored the rowid there (which would diverge from sqlite's on-disk format).
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT)");
    let mut rt = Runtime::new();
    run_insert(
        &insert_plan(
            "t",
            2,
            None,
            vec![vec![lit(Value::Integer(7)), lit(Value::Text("x".into()))]],
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );

    let root = cat.table("t").unwrap().unwrap().root_page;
    let mut tc = TableCursor::open(&pager, root).unwrap();
    assert!(tc.seek_exact(7).unwrap(), "row 7 exists");
    let rec = {
        let p = tc.payload().unwrap();
        decode_record(&p)
    };
    assert!(rec[0].is_null(), "the alias column is stored as NULL, got {:?}", rec[0]);
    assert_eq!(text(&rec[1]), "x");
}

// ----- (11) WITHOUT ROWID storage: insert into / scan the PK index b-tree ----
// A WITHOUT ROWID table keys on its PRIMARY KEY, storing each row in an index-style
// b-tree at `root_page` (withoutrowid.html). INSERT writes the full-row record there;
// the scan reads it back as a width-N row (no fabricated rowid). These pin the real
// storage path (they replaced the earlier fail-closed placeholders).

#[test]
fn insert_into_without_rowid_table_stores_row() {
    // INSERT into a WR table writes the row into its PRIMARY KEY index b-tree and counts
    // the change, but leaves last_insert_rowid() untouched (SQLite's WR rule — there is
    // no rowid to report).
    let (mut pager, cat) =
        db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT) WITHOUT ROWID");
    let mut rt = Runtime::new();
    let out = run_insert(
        &insert_plan(
            "t",
            2,
            None,
            vec![vec![lit(Value::Integer(1)), lit(Value::Text("x".into()))]],
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    assert!(out.is_empty(), "no RETURNING clause yields no rows");
    assert_eq!(rt.changes(), 1, "one WR row inserted");
    assert_eq!(rt.last_insert_rowid(), 0, "a WR insert leaves last_insert_rowid unchanged");

    // The stored record lives in the PK index b-tree (root_page is an index b-tree), so
    // an IndexCursor reads exactly one full-row key record `[a, b]` back — the row IS the
    // key, no rowid slot, and (unlike a rowid table's INTEGER-PK) the PK value is stored
    // in its column, not blanked.
    let root = cat.table("t").unwrap().unwrap().root_page;
    let mut ic = IndexCursor::open(&pager, root).unwrap();
    assert!(ic.first().unwrap(), "one WR row is present");
    let rec = {
        let k = ic.key().unwrap();
        decode_record(k.as_ref())
    };
    assert_eq!(rec.len(), 2, "the WR key record is the full row (width N), no rowid appended");
    assert_eq!(int(&rec[0]), 1, "the PK column value is stored (not blanked like a rowid alias)");
    assert_eq!(text(&rec[1]), "x");
    assert!(!ic.next().unwrap(), "exactly one row");
}

#[test]
fn duplicate_primary_key_in_without_rowid_is_a_constraint_error() {
    // The PRIMARY KEY of a WR table is UNIQUE: a second row with the same PK under the
    // default ABORT is an Error::Constraint (PRIMARY KEY), and the failed row is not
    // stored. Mirrors the rowid table's duplicate-rowid guarantee, keyed by PK instead.
    let (mut pager, cat) =
        db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT) WITHOUT ROWID");
    let mut rt = Runtime::new();
    run_insert(
        &insert_plan(
            "t",
            2,
            None,
            vec![vec![lit(Value::Integer(1)), lit(Value::Text("x".into()))]],
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    let err = run_insert_err(
        &insert_plan(
            "t",
            2,
            None,
            vec![vec![lit(Value::Integer(1)), lit(Value::Text("dup".into()))]],
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
    );
    assert!(
        matches!(err, Error::Constraint(_)),
        "a duplicate PRIMARY KEY is Error::Constraint(_), got {err:?}"
    );
}

#[test]
fn within_statement_duplicate_pk_in_without_rowid_aborts() {
    // The actual check-then-act: ONE multi-row INSERT whose two rows share a PK. The
    // second row's PK probe must see the FIRST row already staged in the transaction
    // overlay (read-your-writes) and raise a PRIMARY KEY Constraint under ABORT. The
    // two-statement dup test above runs the probe against a COMMITTED row; this pins the
    // within-statement overlay path, so a future pager change that made the probe read a
    // committed snapshot (missing the staged row) would fail loudly here instead of
    // silently storing two rows with the same PK. The raised error IS the proof: without
    // read-your-writes there would be no conflict and `run_insert_err` would panic
    // (it asserts the INSERT errors).
    let (mut pager, cat) =
        db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT) WITHOUT ROWID");
    let err = run_insert_err(
        &insert_plan(
            "t",
            2,
            None,
            vec![
                vec![lit(Value::Integer(7)), lit(Value::Text("first".into()))],
                vec![lit(Value::Integer(7)), lit(Value::Text("second".into()))],
            ],
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
    );
    assert!(
        matches!(err, Error::Constraint(_)),
        "a within-statement duplicate PK is Error::Constraint(_) (the probe saw the staged row), got {err:?}"
    );
}

#[test]
fn within_statement_or_ignore_keeps_first_duplicate_pk() {
    // OR IGNORE over a within-statement duplicate PK: the second row's probe sees the
    // first (staged) row and IGNORE SKIPS the second with no error — the first row
    // survives unchanged. Committed via `run_insert`, so the final stored state is
    // scanned: exactly one row, b = 'first'.
    let (mut pager, cat) =
        db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT) WITHOUT ROWID");
    let mut rt = Runtime::new();
    run_insert(
        &insert_plan(
            "t",
            2,
            None,
            vec![
                vec![lit(Value::Integer(7)), lit(Value::Text("first".into()))],
                vec![lit(Value::Integer(7)), lit(Value::Text("second".into()))],
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
    assert_eq!(rows.len(), 1, "OR IGNORE kept exactly one row");
    assert_eq!((int(&rows[0][0]), text(&rows[0][1])), (7, "first"), "the FIRST row survives");
}

#[test]
fn within_statement_or_replace_overwrites_first_duplicate_pk() {
    // OR REPLACE over a within-statement duplicate PK: the second row's probe sees the
    // first (staged) row and REPLACE deletes it, then inserts the second — so the final
    // row is the SECOND one. Pins that a REPLACE against a same-statement staged key
    // resolves correctly (delete-then-insert on the overlay), leaving exactly one row.
    let (mut pager, cat) =
        db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT) WITHOUT ROWID");
    let mut rt = Runtime::new();
    run_insert(
        &insert_plan(
            "t",
            2,
            None,
            vec![
                vec![lit(Value::Integer(7)), lit(Value::Text("first".into()))],
                vec![lit(Value::Integer(7)), lit(Value::Text("second".into()))],
            ],
            affinities(&cat, "t"),
            OnConflict::Replace,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );
    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 1, "OR REPLACE left exactly one row");
    assert_eq!((int(&rows[0][0]), text(&rows[0][1])), (7, "second"), "the SECOND row replaced the first");
}

#[test]
fn select_over_without_rowid_table_scans_rows_in_key_order() {
    // A SELECT (`SeqScan` plan) over a WR table iterates its PRIMARY KEY index b-tree and
    // yields width-N rows (NO trailing rowid), in PRIMARY KEY order regardless of insert
    // order. Rows are inserted 3,1,2 and must scan back 1,2,3.
    let (mut pager, cat) =
        db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT) WITHOUT ROWID");
    assert!(cat.table("t").unwrap().unwrap().without_rowid, "fixture is WITHOUT ROWID");
    let mut rt = Runtime::new();
    run_insert(
        &insert_plan(
            "t",
            2,
            None,
            vec![
                vec![lit(Value::Integer(3)), lit(Value::Text("c".into()))],
                vec![lit(Value::Integer(1)), lit(Value::Text("a".into()))],
                vec![lit(Value::Integer(2)), lit(Value::Text("b".into()))],
            ],
            affinities(&cat, "t"),
            OnConflict::Abort,
            Vec::new(),
        ),
        &cat,
        &mut pager,
        &mut rt,
    );

    let rows = scan_all(&cat, &mut pager, "t", 2);
    assert_eq!(rows.len(), 3);
    // The WR scan row is width N (no rowid register appended).
    assert!(rows.iter().all(|r| r.len() == 2), "WR scan rows are width N, got {rows:?}");
    assert_eq!((int(&rows[0][0]), text(&rows[0][1])), (1, "a"));
    assert_eq!((int(&rows[1][0]), text(&rows[1][1])), (2, "b"));
    assert_eq!((int(&rows[2][0]), text(&rows[2][1])), (3, "c"));
}

// ----- (12) CHECK on the INTEGER PRIMARY KEY alias reads the rowid register -
//
// A CHECK predicate is bound over the `[c0..c_{N-1}, rowid]` row, where an INTEGER PRIMARY
// KEY alias column (and a bare `rowid`) resolves to the TRAILING register N — NOT its
// column-i slot (minisqlite_plan::bind::scope maps `rowid_alias == Some(i)` to
// `base + columns.len()`). The INSERT operator must therefore evaluate checks over that
// N+1-wide row, not the width-N logical row. These pin it on `CREATE TABLE t(a INTEGER
// PRIMARY KEY)` (N == 1, so the rowid register is Column(1)) with a bare `CHECK(a)` bound
// to Column(1): an explicit rowid 0 violates (truth(0) == false) and a nonzero rowid passes.
// Evaluating over the width-1 logical row instead would read Column(1) out of range — a
// non-Constraint error, not this clean CHECK violation — so these fail without that layout.

/// Build an INSERT of one single-column row `[v]` into `t(a INTEGER PRIMARY KEY)` carrying a
/// single bare `CHECK(a)`, i.e. Column(1) — the trailing rowid register for a 1-column table.
fn insert_one_with_alias_check(cat: &SchemaCatalog, v: Value) -> Plan {
    Plan {
        root: PlanNode::Insert(Insert { db: DbIndex::MAIN,
            table: "t".to_string(),
            column_count: 1,
            columns: None,
            source: Box::new(PlanNode::Values { rows: vec![vec![lit(v)]] }),
            source_width: 1,
            column_affinities: affinities(cat, "t"),
            on_conflict: OnConflict::Abort,
            returning: Vec::new(),
            triggers: Vec::new(),
            checks: vec![CheckConstraint { expr: col(1), detail: "t".into() }],
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

#[test]
fn check_on_rowid_alias_violation_reads_rowid_register() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY)");
    assert_eq!(cat.table("t").unwrap().unwrap().rowid_alias, Some(0), "a is the rowid alias");
    // Explicit rowid 0: CHECK(a) reads Column(1) == Integer(0) -> truth false -> violation.
    let err = run_insert_err(&insert_one_with_alias_check(&cat, Value::Integer(0)), &cat, &mut pager);
    assert!(
        matches!(err, Error::Constraint(_)),
        "CHECK(a) on rowid 0 is a Constraint violation (it reads the rowid register), got {err:?}"
    );
    assert!(scan_all(&cat, &mut pager, "t", 1).is_empty(), "the aborted insert wrote nothing");
}

#[test]
fn check_on_rowid_alias_nonzero_passes() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER PRIMARY KEY)");
    let mut rt = Runtime::new();
    // Explicit rowid 5: CHECK(a) reads Column(1) == Integer(5) -> truth true -> passes.
    run_insert(&insert_one_with_alias_check(&cat, Value::Integer(5)), &cat, &mut pager, &mut rt);
    let rows = scan_all(&cat, &mut pager, "t", 1);
    assert_eq!(rows.len(), 1, "the satisfying row was stored");
    // A single-column IPK table's scan row is [a, rowid]; the alias reads back as rowid 5.
    assert_eq!((int(&rows[0][0]), int(&rows[0][1])), (5, 5));
}

// ----- (13) OR IGNORE skips a CHECK-violating row, keeps the rest ----------
//
// A statement-level `OR IGNORE` overrides the constraint-definition ABORT default for a
// CHECK exactly as it does for NOT NULL/UNIQUE (lang_conflict.html: the OR clause
// "overrides any algorithm specified in a CREATE TABLE"; its FAIL entry lists CHECK among
// the constraints the clause governs). So `INSERT OR IGNORE` past a CHECK-violating row
// SKIPS only that row and inserts the rest with NO error — it must NOT abort + roll back
// the whole statement. Reproduces the witness `CREATE TABLE t(x INTEGER
// CHECK(x>0)); INSERT OR IGNORE INTO t VALUES(5),(-1),(10)` -> rows 5 and 10 (the -1 row
// silently skipped). This is also the only operator-level test driving a CHECK over a
// NON-alias column ref (Column(0) == x), complementing the rowid-alias tests above.
//
// Regression guard: before enforce_checks took `on_conflict`, the -1 row raised
// SQLITE_CONSTRAINT mid-loop and the engine's implicit-txn rollback backed everything out
// (table empty + error). `run_insert` unwraps the execute/pull, so that pre-fix path would
// PANIC here — a genuine fail-without-fix / pass-with-fix.

#[test]
fn insert_or_ignore_skips_check_violating_row_keeps_the_rest() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(x INTEGER)");
    let mut rt = Runtime::new();
    // CHECK(x > 0): Column(0) is x; no rowid alias, so N == 1 and the check row is [x, rowid].
    let check = CheckConstraint {
        expr: cmp(CmpOp::Gt, col(0), lit(Value::Integer(0))),
        detail: "t.x".into(),
    };
    let plan = Plan {
        root: PlanNode::Insert(Insert { db: DbIndex::MAIN,
            table: "t".to_string(),
            column_count: 1,
            columns: None,
            source: Box::new(PlanNode::Values {
                rows: vec![
                    vec![lit(Value::Integer(5))],
                    vec![lit(Value::Integer(-1))], // violates x > 0 -> skipped under IGNORE
                    vec![lit(Value::Integer(10))],
                ],
            }),
            source_width: 1,
            column_affinities: affinities(&cat, "t"),
            on_conflict: OnConflict::Ignore,
            returning: Vec::new(),
            triggers: Vec::new(),
            checks: vec![check],
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
    // No error — the violating row is skipped, not an abort.
    let out = run_insert(&plan, &cat, &mut pager, &mut rt);
    assert!(out.is_empty(), "no RETURNING clause");
    assert_eq!(rt.changes(), 2, "two rows inserted (5 and 10); the -1 row was skipped, not counted");
    let mut rows = scan_all(&cat, &mut pager, "t", 1);
    rows.sort_by_key(|r| int(&r[0]));
    assert_eq!(rows.len(), 2, "the CHECK-violating -1 row was skipped, the surrounding rows inserted");
    assert_eq!(int(&rows[0][0]), 5);
    assert_eq!(int(&rows[1][0]), 10);
}
