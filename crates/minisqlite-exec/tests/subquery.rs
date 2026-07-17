//! Expression-subquery evaluation end to end: hand-built [`Plan`] trees whose
//! projections/predicates reference scalar / `EXISTS` / `IN`-select subqueries, run
//! over a real [`MemPager`] + [`SchemaCatalog`] through the [`Executor`] seam. These
//! pin the four [`minisqlite_exec`] subquery callbacks (not a private mock — the
//! actual `build_cursor` re-entrancy, storage, and catalog the engine uses):
//!
//! * scalar subquery — first row/column, NULL when empty;
//! * `EXISTS` / `NOT EXISTS` — empty vs non-empty;
//! * `x IN (subquery)` — three-valued (true / false / unknown), affinity, empty set;
//! * `(a,…) IN (subquery)` — tuple three-valued membership (rowvalue.html §2.2): the
//!   AND3 over elements folded by the OR3 across candidate rows, NULLs, empty set;
//! * correlation — a correlated subplan re-runs per outer row and reads the outer
//!   columns; a non-correlated one runs against an EMPTY outer;
//! * a subquery id past the plan's `subqueries` vector fails loud (no panic).

use minisqlite_btree::{init_database, table_insert};
use minisqlite_catalog::{Catalog, SchemaCatalog};
use minisqlite_exec::{Executor, Runtime, StreamingExecutor};
use minisqlite_expr::{CmpOp, CompareMeta, EvalExpr};
use minisqlite_fileformat::encode_record;
use minisqlite_pager::{MemPager, PageId, Pager};
use minisqlite_plan::{Plan, PlanNode, SeqScan, SubPlan};
use minisqlite_sql::{parse, Statement};
use minisqlite_types::{Affinity, Collation, Value};
use minisqlite_exec::PagerSet;
use minisqlite_types::DbIndex;

// ----- fixtures (mirrors tests/exec.rs) ------------------------------------

/// A fresh in-memory database with one table created through the real catalog path.
fn db_with_table(create_sql: &str) -> (MemPager, SchemaCatalog) {
    let mut pager = MemPager::new(4096);
    init_database(&mut pager).unwrap();
    let mut cat = SchemaCatalog::new();
    create_table(&mut pager, &mut cat, create_sql);
    (pager, cat)
}

/// Register another table in an existing database (a correlated subquery scans a
/// second table through the same catalog + pager as the outer).
fn create_table(pager: &mut MemPager, cat: &mut SchemaCatalog, create_sql: &str) {
    let ast = parse(create_sql).unwrap();
    let Statement::CreateTable(stmt) = &ast.statements[0] else {
        panic!("not a CREATE TABLE: {create_sql}");
    };
    cat.create_table(pager, stmt, create_sql).unwrap();
}

/// Insert `(rowid, values)` records into the table's b-tree, inside one transaction.
fn seed(pager: &mut MemPager, root: PageId, rows: &[(i64, Vec<Value>)]) {
    pager.begin().unwrap();
    for (rowid, vals) in rows {
        table_insert(pager, root, *rowid, &encode_record(vals)).unwrap();
    }
    pager.commit().unwrap();
}

/// The root page of a created table.
fn table_root(cat: &SchemaCatalog, name: &str) -> PageId {
    cat.table(name).unwrap().unwrap().root_page
}

/// Drain a plan to completion and collect the result rows.
fn run(plan: &Plan, cat: &SchemaCatalog, pager: &mut MemPager) -> Vec<Vec<Value>> {
    let mut rt = Runtime::new();
    let mut exec = StreamingExecutor;
    let mut cur = exec.execute(plan, cat, PagerSet::One { db: DbIndex::MAIN, pager: &mut *pager }).unwrap();
    let mut out = Vec::new();
    while let Some(row) = cur.next_row(&mut rt).unwrap() {
        out.push(row);
    }
    out
}

/// Like [`run`] but returns the `Result` so error paths (a loud failure, not a
/// silent empty result) can be asserted.
fn run_err(plan: &Plan, cat: &SchemaCatalog, pager: &mut MemPager) -> minisqlite_types::Result<()> {
    let mut rt = Runtime::new();
    let mut exec = StreamingExecutor;
    let mut cur = exec.execute(plan, cat, PagerSet::One { db: DbIndex::MAIN, pager: &mut *pager })?;
    while cur.next_row(&mut rt)?.is_some() {}
    Ok(())
}

// ----- plan-building helpers -----------------------------------------------

fn col(i: usize) -> EvalExpr {
    EvalExpr::Column(i)
}

fn lit(v: Value) -> EvalExpr {
    EvalExpr::Literal(v)
}

fn binary_meta() -> CompareMeta {
    CompareMeta { apply_left: None, apply_right: None, collation: Collation::Binary }
}

fn seqscan(table: &str, column_count: usize) -> PlanNode {
    PlanNode::SeqScan(SeqScan { db: DbIndex::MAIN, table: table.to_string(), column_count })
}

fn project(input: PlanNode, exprs: Vec<EvalExpr>) -> PlanNode {
    PlanNode::Project { input: Box::new(input), exprs }
}

/// The subplan `SELECT <first column> FROM <table>` (a one-column subquery result).
fn first_col(table: &str) -> PlanNode {
    project(seqscan(table, 1), vec![col(0)])
}

fn sub(plan: PlanNode, correlated: bool) -> SubPlan {
    // The correlated-cols / determinism / outer-width metadata is not exercised by these
    // executor tests (nothing reads it on the eval path yet), so the neutral defaults
    // suffice: no outer registers, deterministic, outer width 0.
    SubPlan { plan, correlated, correlated_cols: Vec::new(), deterministic: true, outer_width: 0 }
}

fn plan_subs(root: PlanNode, columns: &[&str], subqueries: Vec<SubPlan>) -> Plan {
    Plan {
        root,
        result_columns: columns.iter().map(|s| s.to_string()).collect(),
        ctes: Vec::new(),
        subqueries,
        mutates: false,
        generated: Vec::new(),
    }
}

/// `SELECT <expr>` — one expression over a single implicit row (`SingleRow`), with
/// the subplans the expression references. The result is one row, one column.
fn select_expr(expr: EvalExpr, subqueries: Vec<SubPlan>) -> Plan {
    plan_subs(project(PlanNode::SingleRow, vec![expr]), &["r"], subqueries)
}

fn int(v: &Value) -> i64 {
    match v {
        Value::Integer(i) => *i,
        other => panic!("expected Integer, got {other:?}"),
    }
}

// ----- (1) scalar subquery -------------------------------------------------

#[test]
fn scalar_subquery_returns_first_row_first_col() {
    // The scalar is the FIRST row's FIRST column. The subplan projects TWO columns
    // (a=5,b=6 at rowid 1; a=7,b=8 at rowid 2), so the result pins BOTH axes: first
    // ROW (5, not 7) and first COLUMN (5, not 6) — a `.last()` on the row or the wrong
    // row index would land on a different value.
    let (mut pager, cat) = db_with_table("CREATE TABLE t2(a INTEGER, b INTEGER)");
    let root = table_root(&cat, "t2");
    seed(
        &mut pager,
        root,
        &[
            (1, vec![Value::Integer(5), Value::Integer(6)]),
            (2, vec![Value::Integer(7), Value::Integer(8)]),
        ],
    );

    let subplan = project(seqscan("t2", 2), vec![col(0), col(1)]);
    let p = select_expr(EvalExpr::ScalarSubquery(0), vec![sub(subplan, false)]);
    let rows = run(&p, &cat, &mut pager);
    assert_eq!(rows.len(), 1);
    assert_eq!(int(&rows[0][0]), 5, "scalar subquery yields the first row's first column");
}

#[test]
fn scalar_subquery_empty_is_null() {
    // An empty subquery (no rows) evaluates to SQL NULL, not an error or 0.
    let (mut pager, cat) = db_with_table("CREATE TABLE t2(a INTEGER)"); // no rows
    let p = select_expr(EvalExpr::ScalarSubquery(0), vec![sub(first_col("t2"), false)]);
    let rows = run(&p, &cat, &mut pager);
    assert_eq!(rows.len(), 1);
    assert!(rows[0][0].is_null(), "an empty scalar subquery is NULL");
}

// ----- (2) EXISTS / NOT EXISTS ---------------------------------------------

#[test]
fn exists_and_not_exists_over_empty_and_non_empty() {
    // Non-empty subplan: EXISTS -> 1, NOT EXISTS -> 0.
    let (mut pager, cat) = db_with_table("CREATE TABLE t2(a INTEGER)");
    let root = table_root(&cat, "t2");
    seed(&mut pager, root, &[(1, vec![Value::Integer(1)])]);
    let exists = |negated| select_expr(EvalExpr::Exists { negated, id: 0 }, vec![sub(seqscan("t2", 1), false)]);
    assert_eq!(int(&run(&exists(false), &cat, &mut pager)[0][0]), 1, "EXISTS over a non-empty subplan");
    assert_eq!(int(&run(&exists(true), &cat, &mut pager)[0][0]), 0, "NOT EXISTS over a non-empty subplan");

    // Empty subplan: EXISTS -> 0, NOT EXISTS -> 1.
    let (mut pe, ce) = db_with_table("CREATE TABLE t3(a INTEGER)"); // no rows
    let exists_e = |negated| select_expr(EvalExpr::Exists { negated, id: 0 }, vec![sub(seqscan("t3", 1), false)]);
    assert_eq!(int(&run(&exists_e(false), &ce, &mut pe)[0][0]), 0, "EXISTS over an empty subplan");
    assert_eq!(int(&run(&exists_e(true), &ce, &mut pe)[0][0]), 1, "NOT EXISTS over an empty subplan");
}

// ----- (3) IN (subquery), three-valued -------------------------------------

/// `<probe> IN (SELECT v FROM <table>)`, no affinity, not negated.
fn in_select(probe: Value, table: &str) -> Plan {
    select_expr(
        EvalExpr::InSubquery {
            negated: false,
            subject: Box::new(lit(probe)),
            id: 0,
            meta: binary_meta(),
        },
        vec![sub(first_col(table), false)],
    )
}

#[test]
fn in_subquery_true_false_and_unknown() {
    // candidates {1,2,3}
    let (mut pager, cat) = db_with_table("CREATE TABLE t_in(v INTEGER)");
    let root = table_root(&cat, "t_in");
    seed(
        &mut pager,
        root,
        &[(1, vec![Value::Integer(1)]), (2, vec![Value::Integer(2)]), (3, vec![Value::Integer(3)])],
    );
    assert_eq!(int(&run(&in_select(Value::Integer(2), "t_in"), &cat, &mut pager)[0][0]), 1, "2 IN {{1,2,3}} -> true");
    assert_eq!(int(&run(&in_select(Value::Integer(9), "t_in"), &cat, &mut pager)[0][0]), 0, "9 IN {{1,2,3}} -> false");
    // A NULL probe against a non-empty set with no match is unknown (NULL).
    assert!(run(&in_select(Value::Null, "t_in"), &cat, &mut pager)[0][0].is_null(), "NULL IN {{1,2,3}} -> unknown");

    // candidates {1, NULL}: 9 not found, and the NULL candidate makes it unknown.
    let (mut pn, cn) = db_with_table("CREATE TABLE t_null(v INTEGER)");
    let rn = table_root(&cn, "t_null");
    seed(&mut pn, rn, &[(1, vec![Value::Integer(1)]), (2, vec![Value::Null])]);
    assert!(run(&in_select(Value::Integer(9), "t_null"), &cn, &mut pn)[0][0].is_null(), "9 IN {{1,NULL}} -> unknown");
    // A found value wins over the NULL candidate.
    assert_eq!(int(&run(&in_select(Value::Integer(1), "t_null"), &cn, &mut pn)[0][0]), 1, "1 IN {{1,NULL}} -> true");
}

#[test]
fn in_empty_subquery_is_false_even_for_null_probe() {
    // `x IN (empty)` is FALSE for any x, INCLUDING a NULL probe (it is not unknown).
    let (mut pager, cat) = db_with_table("CREATE TABLE t_empty(v INTEGER)"); // no rows
    assert_eq!(int(&run(&in_select(Value::Integer(1), "t_empty"), &cat, &mut pager)[0][0]), 0, "1 IN () -> false");
    assert_eq!(int(&run(&in_select(Value::Null, "t_empty"), &cat, &mut pager)[0][0]), 0, "NULL IN () -> false");
}

#[test]
fn in_subquery_applies_affinity() {
    // Candidates are TEXT '1','2','3'. With NUMERIC affinity on the right operand they
    // coerce to numbers, so integer 2 matches '2'; without affinity a number is never
    // equal to text (number < text), so 2 is not found.
    let (mut pager, cat) = db_with_table("CREATE TABLE t_txt(v TEXT)");
    let root = table_root(&cat, "t_txt");
    seed(
        &mut pager,
        root,
        &[
            (1, vec![Value::Text("1".into())]),
            (2, vec![Value::Text("2".into())]),
            (3, vec![Value::Text("3".into())]),
        ],
    );

    let with_aff = |apply_right| {
        select_expr(
            EvalExpr::InSubquery {
                negated: false,
                subject: Box::new(lit(Value::Integer(2))),
                id: 0,
                meta: CompareMeta { apply_left: None, apply_right, collation: Collation::Binary },
            },
            vec![sub(first_col("t_txt"), false)],
        )
    };
    assert_eq!(int(&run(&with_aff(Some(Affinity::Numeric)), &cat, &mut pager)[0][0]), 1, "NUMERIC affinity makes 2 match TEXT '2'");
    assert_eq!(int(&run(&with_aff(None), &cat, &mut pager)[0][0]), 0, "without affinity, 2 does not equal TEXT '2'");
}

#[test]
fn in_subquery_applies_probe_affinity() {
    // The probe side (`apply_left`) is applied too: a TEXT probe '2' with NUMERIC
    // affinity coerces to 2 and matches integer candidates; without it, TEXT '2' is
    // never equal to an integer (text sorts after numbers), so it is not found.
    let (mut pager, cat) = db_with_table("CREATE TABLE t_int(v INTEGER)");
    let root = table_root(&cat, "t_int");
    seed(
        &mut pager,
        root,
        &[(1, vec![Value::Integer(1)]), (2, vec![Value::Integer(2)]), (3, vec![Value::Integer(3)])],
    );

    let with_aff = |apply_left| {
        select_expr(
            EvalExpr::InSubquery {
                negated: false,
                subject: Box::new(lit(Value::Text("2".into()))),
                id: 0,
                meta: CompareMeta { apply_left, apply_right: None, collation: Collation::Binary },
            },
            vec![sub(first_col("t_int"), false)],
        )
    };
    assert_eq!(int(&run(&with_aff(Some(Affinity::Numeric)), &cat, &mut pager)[0][0]), 1, "NUMERIC probe affinity makes TEXT '2' match integer 2");
    assert_eq!(int(&run(&with_aff(None), &cat, &mut pager)[0][0]), 0, "without probe affinity, TEXT '2' does not equal integer 2");
}

#[test]
fn in_subquery_uses_collation() {
    // The comparison honors `meta.collation`: 'ABC' matches 'abc' under NOCASE, not
    // under BINARY.
    let (mut pager, cat) = db_with_table("CREATE TABLE t_txt(v TEXT)");
    let root = table_root(&cat, "t_txt");
    seed(&mut pager, root, &[(1, vec![Value::Text("abc".into())])]);

    let with_coll = |collation| {
        select_expr(
            EvalExpr::InSubquery {
                negated: false,
                subject: Box::new(lit(Value::Text("ABC".into()))),
                id: 0,
                meta: CompareMeta { apply_left: None, apply_right: None, collation },
            },
            vec![sub(first_col("t_txt"), false)],
        )
    };
    assert_eq!(int(&run(&with_coll(Collation::NoCase), &cat, &mut pager)[0][0]), 1, "'ABC' IN {{'abc'}} matches under NOCASE");
    assert_eq!(int(&run(&with_coll(Collation::Binary), &cat, &mut pager)[0][0]), 0, "'ABC' does not equal 'abc' under BINARY");
}

#[test]
fn not_in_subquery_inverts_true_but_not_unknown() {
    // NOT IN inverts a definite membership (Some) but leaves unknown (None) unknown —
    // the callback returns the raw Option<bool> and the evaluator applies the NOT.
    let (mut pager, cat) = db_with_table("CREATE TABLE t_in(v INTEGER)");
    let root = table_root(&cat, "t_in");
    seed(&mut pager, root, &[(1, vec![Value::Integer(1)]), (2, vec![Value::Integer(2)])]);
    let not_in = |probe, table: &str| {
        select_expr(
            EvalExpr::InSubquery { negated: true, subject: Box::new(lit(probe)), id: 0, meta: binary_meta() },
            vec![sub(first_col(table), false)],
        )
    };
    assert_eq!(int(&run(&not_in(Value::Integer(2), "t_in"), &cat, &mut pager)[0][0]), 0, "2 NOT IN {{1,2}} -> false");

    // 9 NOT IN {1, NULL} stays unknown (NULL), negation does not flip it.
    let (mut pn, cn) = db_with_table("CREATE TABLE t_null(v INTEGER)");
    let rn = table_root(&cn, "t_null");
    seed(&mut pn, rn, &[(1, vec![Value::Integer(1)]), (2, vec![Value::Null])]);
    assert!(run(&not_in(Value::Integer(9), "t_null"), &cn, &mut pn)[0][0].is_null(), "NOT IN with a NULL candidate stays unknown");
}

// ----- (3b) row-value (tuple) IN (subquery), three-valued ------------------
//
// `(a, b, …) IN (SELECT x, y, …)` (rowvalue.html §2.2): the tuple generalization of
// the scalar `IN`. A candidate row MATCHES iff every element compares equal; the result
// is TRUE if any candidate matches, else UNKNOWN (NULL) if some candidate was equal on
// all its non-NULL elements and blocked only by a NULL, else FALSE (an empty subquery is
// FALSE). These drive the real `EvalExpr::InSubqueryRow` -> exec `eval_in_subquery_row`
// probe (the same `build_cursor` re-entrancy and storage as the scalar `IN` above).

/// Per-element BINARY, no-affinity comparison metadata for an `n`-wide tuple.
fn binary_metas(n: usize) -> Vec<CompareMeta> {
    (0..n).map(|_| binary_meta()).collect()
}

/// `(probe…) [NOT] IN (<subplan>)` with per-element BINARY, no-affinity metadata. The
/// subplan (registered as subquery id 0) must project exactly `probe.len()` columns.
fn in_row(probe: Vec<Value>, negated: bool, subplan: PlanNode) -> Plan {
    let width = probe.len();
    select_expr(
        EvalExpr::InSubqueryRow {
            negated,
            subjects: probe.into_iter().map(lit).collect(),
            id: 0,
            metas: binary_metas(width),
        },
        vec![sub(subplan, false)],
    )
}

/// A one-row constant subplan projecting the given tuple (via `SingleRow`) — the
/// executor analogue of a `(SELECT v0, v1, …)` candidate with a single row.
fn const_row(vals: Vec<Value>) -> PlanNode {
    project(PlanNode::SingleRow, vals.into_iter().map(lit).collect())
}

/// A subplan projecting the first `ncols` columns of every row of `table` — a
/// multi-row, `ncols`-wide candidate set.
fn wide_scan(table: &str, ncols: usize) -> PlanNode {
    project(seqscan(table, ncols), (0..ncols).map(col).collect())
}

#[test]
fn in_subquery_row_match_is_true_and_no_match_is_false() {
    // rowvalue.html §2.2: the tuple is IN iff some candidate row equals it element-wise;
    // a partial (first-element-only) match is NOT a match.
    let (mut pager, cat) = db_with_table("CREATE TABLE t0(z INTEGER)");
    let two = |a, b| vec![Value::Integer(a), Value::Integer(b)];

    let m = in_row(two(1, 2), false, const_row(two(1, 2)));
    assert_eq!(int(&run(&m, &cat, &mut pager)[0][0]), 1, "(1,2) IN (SELECT 1,2) -> true");

    let nm = in_row(two(1, 2), false, const_row(two(3, 4)));
    assert_eq!(int(&run(&nm, &cat, &mut pager)[0][0]), 0, "(1,2) IN (SELECT 3,4) -> false");

    let partial = in_row(two(1, 2), false, const_row(two(1, 3)));
    assert_eq!(int(&run(&partial, &cat, &mut pager)[0][0]), 0, "(1,2) IN (SELECT 1,3) -> false (partial match is not a match)");
}

#[test]
fn in_subquery_row_matches_any_candidate_row() {
    // The matching row may be anywhere in the candidate set, not just the first.
    let (mut pager, cat) = db_with_table("CREATE TABLE tp(x INTEGER, y INTEGER)");
    let root = table_root(&cat, "tp");
    seed(
        &mut pager,
        root,
        &[(1, vec![Value::Integer(3), Value::Integer(4)]), (2, vec![Value::Integer(1), Value::Integer(2)])],
    );
    let hit = in_row(vec![Value::Integer(1), Value::Integer(2)], false, wide_scan("tp", 2));
    assert_eq!(int(&run(&hit, &cat, &mut pager)[0][0]), 1, "a match in a non-first candidate row still matches");
    let miss = in_row(vec![Value::Integer(9), Value::Integer(9)], false, wide_scan("tp", 2));
    assert_eq!(int(&run(&miss, &cat, &mut pager)[0][0]), 0, "no candidate row equals (9,9)");
}

#[test]
fn not_in_subquery_row_inverts_definite_membership() {
    // NOT IN flips a definite TRUE/FALSE (the evaluator applies `negated` to the Some).
    let (mut pager, cat) = db_with_table("CREATE TABLE t0(z INTEGER)");
    let two = |a, b| vec![Value::Integer(a), Value::Integer(b)];

    let member = in_row(two(1, 2), true, const_row(two(1, 2)));
    assert_eq!(int(&run(&member, &cat, &mut pager)[0][0]), 0, "(1,2) NOT IN (SELECT 1,2) -> false");

    let non_member = in_row(two(1, 2), true, const_row(two(3, 4)));
    assert_eq!(int(&run(&non_member, &cat, &mut pager)[0][0]), 1, "(1,2) NOT IN (SELECT 3,4) -> true");
}

#[test]
fn in_subquery_row_empty_is_false_and_not_in_empty_is_true() {
    // An empty candidate set is FALSE for any probe (so NOT IN empty is TRUE), INCLUDING
    // a probe with a NULL element — the empty set is definite, not unknown.
    let (mut pager, cat) = db_with_table("CREATE TABLE te(x INTEGER, y INTEGER)"); // no rows
    let in_e = in_row(vec![Value::Integer(1), Value::Integer(2)], false, wide_scan("te", 2));
    assert_eq!(int(&run(&in_e, &cat, &mut pager)[0][0]), 0, "(1,2) IN (empty) -> false");
    let not_in_e = in_row(vec![Value::Integer(1), Value::Integer(2)], true, wide_scan("te", 2));
    assert_eq!(int(&run(&not_in_e, &cat, &mut pager)[0][0]), 1, "(1,2) NOT IN (empty) -> true");
    let in_e_null = in_row(vec![Value::Integer(1), Value::Null], false, wide_scan("te", 2));
    assert_eq!(int(&run(&in_e_null, &cat, &mut pager)[0][0]), 0, "(1,NULL) IN (empty) -> false, not unknown");
}

#[test]
fn in_subquery_row_null_three_valued_logic() {
    // The NULL cases called out for row-value IN. A per-element NULL makes THAT element's
    // comparison unknown; the row-level AND3 then decides. A definite mismatch on ANY
    // element (even before a NULL element) settles the row FALSE.
    let (mut pager, cat) = db_with_table("CREATE TABLE t0(z INTEGER)");
    let probe = || vec![Value::Integer(1), Value::Null];

    // 1=1, then NULL vs 2 is unknown -> the whole row is unknown -> NULL.
    let a = in_row(probe(), false, const_row(vec![Value::Integer(1), Value::Integer(2)]));
    assert!(run(&a, &cat, &mut pager)[0][0].is_null(), "(1,NULL) IN (SELECT 1,2) -> unknown/NULL");

    // 1=1, then NULL vs NULL is unknown -> NULL.
    let b = in_row(probe(), false, const_row(vec![Value::Integer(1), Value::Null]));
    assert!(run(&b, &cat, &mut pager)[0][0].is_null(), "(1,NULL) IN (SELECT 1,NULL) -> unknown/NULL");

    // 1 vs 2 is DEFINITELY unequal -> the row is FALSE; the trailing NULL is never
    // reached (AND3 short-circuits on a definite false) -> 0.
    let c = in_row(probe(), false, const_row(vec![Value::Integer(2), Value::Integer(2)]));
    assert_eq!(int(&run(&c, &cat, &mut pager)[0][0]), 0, "(1,NULL) IN (SELECT 2,2) -> false");
}

#[test]
fn in_subquery_row_folds_unknown_and_false_across_rows() {
    // Across multiple candidate rows: an UNKNOWN row with no definite match anywhere
    // makes the whole result UNKNOWN; a set of only definite non-matches is FALSE. A
    // probe with a NULL element can never be TRUE (an element NULL is never "equal").
    let (mut pager, cat) = db_with_table("CREATE TABLE tm(x INTEGER, y INTEGER)");
    let root = table_root(&cat, "tm");
    // {(2,2),(1,3)}: (2,2)->1<>2 false; (1,3)->1=1 then NULL vs 3 unknown -> None; NULL.
    seed(
        &mut pager,
        root,
        &[(1, vec![Value::Integer(2), Value::Integer(2)]), (2, vec![Value::Integer(1), Value::Integer(3)])],
    );
    let unknown = in_row(vec![Value::Integer(1), Value::Null], false, wide_scan("tm", 2));
    assert!(run(&unknown, &cat, &mut pager)[0][0].is_null(), "an unknown row with no definite match -> NULL");

    // {(2,2),(3,3)}: both first elements differ -> both definite false -> FALSE. The NULL
    // probe element is never reached because each row short-circuits on the first.
    let (mut p2, c2) = db_with_table("CREATE TABLE tm2(x INTEGER, y INTEGER)");
    let r2 = table_root(&c2, "tm2");
    seed(
        &mut p2,
        r2,
        &[(1, vec![Value::Integer(2), Value::Integer(2)]), (2, vec![Value::Integer(3), Value::Integer(3)])],
    );
    let all_false = in_row(vec![Value::Integer(1), Value::Null], false, wide_scan("tm2", 2));
    assert_eq!(int(&run(&all_false, &c2, &mut p2)[0][0]), 0, "only definite non-matches -> false even with a NULL probe element");
}

#[test]
fn in_subquery_row_applies_per_element_metadata() {
    // Each element uses its OWN CompareMeta (affinity + collation). Element 0 gets
    // NUMERIC affinity so a TEXT probe '1' coerces and matches integer 1; element 1 stays
    // BINARY no-affinity. Dropping element-0 affinity makes '1' != 1, so no match — this
    // proves the metadata is per element, not shared.
    let (mut pager, cat) = db_with_table("CREATE TABLE t0(z INTEGER)");
    let numeric0 =
        CompareMeta { apply_left: Some(Affinity::Numeric), apply_right: None, collation: Collation::Binary };
    let make = |m0| {
        select_expr(
            EvalExpr::InSubqueryRow {
                negated: false,
                subjects: vec![lit(Value::Text("1".into())), lit(Value::Integer(2))],
                id: 0,
                metas: vec![m0, binary_meta()],
            },
            vec![sub(const_row(vec![Value::Integer(1), Value::Integer(2)]), false)],
        )
    };
    assert_eq!(int(&run(&make(numeric0), &cat, &mut pager)[0][0]), 1, "NUMERIC affinity on element 0 makes '1' match 1");
    assert_eq!(int(&run(&make(binary_meta()), &cat, &mut pager)[0][0]), 0, "without element-0 affinity, '1' != 1 -> no match");
}

// ----- (4) correlation -----------------------------------------------------

#[test]
fn correlated_exists_reruns_per_outer_row() {
    // SELECT a, EXISTS(SELECT 1 FROM t2 WHERE t2.b = t1.a) FROM t1.
    // t1.a in {1,2,3}; t2.b in {2,4} -> EXISTS is true only for a = 2.
    let (mut pager, mut cat) = db_with_table("CREATE TABLE t1(a INTEGER)");
    create_table(&mut pager, &mut cat, "CREATE TABLE t2(b INTEGER)");
    let (r1, r2) = (table_root(&cat, "t1"), table_root(&cat, "t2"));
    seed(
        &mut pager,
        r1,
        &[(1, vec![Value::Integer(1)]), (2, vec![Value::Integer(2)]), (3, vec![Value::Integer(3)])],
    );
    seed(&mut pager, r2, &[(1, vec![Value::Integer(2)]), (2, vec![Value::Integer(4)])]);

    // The outer SeqScan(t1) row is [a, rowid] (width 2); the correlated subplan's
    // SeqScan(t2) leaf PREPENDS that outer, so its row is [a, rowid, b, t2rowid] and
    // Column(0) = a (outer), Column(2) = b (t2's first column).
    let subplan = PlanNode::Filter {
        input: Box::new(seqscan("t2", 1)),
        predicate: EvalExpr::Compare {
            op: CmpOp::Eq,
            null_safe: false,
            left: Box::new(col(2)),  // t2.b
            right: Box::new(col(0)), // outer t1.a
            meta: binary_meta(),
        },
    };
    let root = project(seqscan("t1", 1), vec![col(0), EvalExpr::Exists { negated: false, id: 0 }]);
    let p = plan_subs(root, &["a", "e"], vec![sub(subplan, true)]);

    let rows = run(&p, &cat, &mut pager);
    assert_eq!(rows.len(), 3);
    assert_eq!((int(&rows[0][0]), int(&rows[0][1])), (1, 0), "a=1: no t2.b = 1");
    assert_eq!((int(&rows[1][0]), int(&rows[1][1])), (2, 1), "a=2: t2.b = 2 exists");
    assert_eq!((int(&rows[2][0]), int(&rows[2][1])), (3, 0), "a=3: no t2.b = 3");
}

#[test]
fn correlated_scalar_reads_outer_value_per_row() {
    // SELECT (SELECT t1.a) FROM t1 — the correlated scalar subquery reads the outer
    // column, so the projection reproduces each outer `a` (only reachable through the
    // subquery, whose SingleRow leaf prepends the outer row and reads Column(0) = a).
    let (mut pager, cat) = db_with_table("CREATE TABLE t1(a INTEGER)");
    let r1 = table_root(&cat, "t1");
    seed(
        &mut pager,
        r1,
        &[(1, vec![Value::Integer(1)]), (2, vec![Value::Integer(2)]), (3, vec![Value::Integer(3)])],
    );

    let subplan = project(PlanNode::SingleRow, vec![col(0)]);
    let root = project(seqscan("t1", 1), vec![EvalExpr::ScalarSubquery(0)]);
    let p = plan_subs(root, &["s"], vec![sub(subplan, true)]);

    let got: Vec<i64> = run(&p, &cat, &mut pager).iter().map(|r| int(&r[0])).collect();
    assert_eq!(got, vec![1, 2, 3], "a correlated subquery re-runs and reads the outer value per row");
}

#[test]
fn correlated_in_subquery_reruns_per_outer_row() {
    // SELECT a, a IN (SELECT b FROM t2 WHERE t2.b = t1.a) FROM t1.
    // The candidate set is correlated (depends on t1.a), so the membership is true
    // only where some t2.b equals a: t2.b in {2,4} -> true only for a=2.
    let (mut pager, mut cat) = db_with_table("CREATE TABLE t1(a INTEGER)");
    create_table(&mut pager, &mut cat, "CREATE TABLE t2(b INTEGER)");
    let (r1, r2) = (table_root(&cat, "t1"), table_root(&cat, "t2"));
    seed(
        &mut pager,
        r1,
        &[(1, vec![Value::Integer(1)]), (2, vec![Value::Integer(2)]), (3, vec![Value::Integer(3)])],
    );
    seed(&mut pager, r2, &[(1, vec![Value::Integer(2)]), (2, vec![Value::Integer(4)])]);

    // outer [a, rowid]; the correlated candidate subplan's SeqScan(t2) leaf prepends it
    // -> [a, rowid, b, t2rowid], so Column(0)=a, Column(2)=b. The subplan projects the
    // matching b values; the outer probe (subject) is Column(0)=a.
    let candidate = project(
        PlanNode::Filter {
            input: Box::new(seqscan("t2", 1)),
            predicate: EvalExpr::Compare {
                op: CmpOp::Eq,
                null_safe: false,
                left: Box::new(col(2)),
                right: Box::new(col(0)),
                meta: binary_meta(),
            },
        },
        vec![col(2)],
    );
    let root = project(
        seqscan("t1", 1),
        vec![
            col(0),
            EvalExpr::InSubquery { negated: false, subject: Box::new(col(0)), id: 0, meta: binary_meta() },
        ],
    );
    let p = plan_subs(root, &["a", "in"], vec![sub(candidate, true)]);

    let rows = run(&p, &cat, &mut pager);
    assert_eq!(rows.len(), 3);
    assert_eq!((int(&rows[0][0]), int(&rows[0][1])), (1, 0), "a=1: no t2.b = 1");
    assert_eq!((int(&rows[1][0]), int(&rows[1][1])), (2, 1), "a=2: t2.b = 2, so a IN (correlated set)");
    assert_eq!((int(&rows[2][0]), int(&rows[2][1])), (3, 0), "a=3: no t2.b = 3");
}

#[test]
fn correlated_in_subquery_row_reruns_per_outer_row() {
    // SELECT a, (a, 2) IN (SELECT b, c FROM t2 WHERE t2.b = t1.a) FROM t1.
    // The candidate tuple set is correlated (depends on t1.a). t2 rows {(2,2),(4,9)}, so
    // the probe (a, 2) matches only where a picks a candidate whose (b,c) == (a,2): a=2
    // selects {(2,2)}, which equals (2,2) -> true; a=1 and a=3 select {} -> false.
    let (mut pager, mut cat) = db_with_table("CREATE TABLE t1(a INTEGER)");
    create_table(&mut pager, &mut cat, "CREATE TABLE t2(b INTEGER, c INTEGER)");
    let (r1, r2) = (table_root(&cat, "t1"), table_root(&cat, "t2"));
    seed(
        &mut pager,
        r1,
        &[(1, vec![Value::Integer(1)]), (2, vec![Value::Integer(2)]), (3, vec![Value::Integer(3)])],
    );
    seed(&mut pager, r2, &[(1, vec![Value::Integer(2), Value::Integer(2)]), (2, vec![Value::Integer(4), Value::Integer(9)])]);

    // outer [a, rowid]; the correlated SeqScan(t2, 2) leaf prepends it ->
    // [a, rowid, b, c, t2rowid], so Column(0)=a, Column(2)=b, Column(3)=c. The candidate
    // subplan filters b=a and projects (b, c); the probe tuple is (Column(0)=a, 2).
    let candidate = project(
        PlanNode::Filter {
            input: Box::new(seqscan("t2", 2)),
            predicate: EvalExpr::Compare {
                op: CmpOp::Eq,
                null_safe: false,
                left: Box::new(col(2)),
                right: Box::new(col(0)),
                meta: binary_meta(),
            },
        },
        vec![col(2), col(3)],
    );
    let root = project(
        seqscan("t1", 1),
        vec![
            col(0),
            EvalExpr::InSubqueryRow {
                negated: false,
                subjects: vec![col(0), lit(Value::Integer(2))],
                id: 0,
                metas: binary_metas(2),
            },
        ],
    );
    let p = plan_subs(root, &["a", "in"], vec![sub(candidate, true)]);

    let rows = run(&p, &cat, &mut pager);
    assert_eq!(rows.len(), 3);
    assert_eq!((int(&rows[0][0]), int(&rows[0][1])), (1, 0), "a=1: correlated set empty -> (1,2) not in");
    assert_eq!((int(&rows[1][0]), int(&rows[1][1])), (2, 1), "a=2: candidate (2,2) == probe (2,2) -> in");
    assert_eq!((int(&rows[2][0]), int(&rows[2][1])), (3, 0), "a=3: correlated set empty -> not in");
}

#[test]
fn non_correlated_subquery_runs_with_empty_outer() {
    // The SAME subplan as the correlated scalar test, but marked non-correlated: the
    // correlation rule then runs it with an EMPTY outer, so SingleRow yields a
    // zero-width row and Column(0) is out of range -> a loud error. This pins that the
    // `correlated` flag (not the presence of an outer row) selects the outer slice.
    let (mut pager, cat) = db_with_table("CREATE TABLE t1(a INTEGER)");
    let r1 = table_root(&cat, "t1");
    seed(&mut pager, r1, &[(1, vec![Value::Integer(1)])]);

    let subplan = project(PlanNode::SingleRow, vec![col(0)]);
    let root = project(seqscan("t1", 1), vec![EvalExpr::ScalarSubquery(0)]);
    let p = plan_subs(root, &["s"], vec![sub(subplan, false)]);

    assert!(
        run_err(&p, &cat, &mut pager).is_err(),
        "a non-correlated subplan is run with an empty outer, so a Column(outer) ref is out of range"
    );
}

// ----- (5) id guard --------------------------------------------------------

#[test]
fn subquery_id_out_of_range_errors() {
    // Each callback guards the id (`.get(id)`): an index past the `subqueries` vector
    // is a loud Error, never a panic on `subqueries[id]`. One subplan (index 0) exists;
    // every reference uses id 5.
    let (mut pager, cat) = db_with_table("CREATE TABLE t2(a INTEGER)");
    let one = || vec![sub(seqscan("t2", 1), false)];

    let scalar = select_expr(EvalExpr::ScalarSubquery(5), one());
    assert!(run_err(&scalar, &cat, &mut pager).is_err(), "scalar subquery id out of range");

    let exists = select_expr(EvalExpr::Exists { negated: false, id: 5 }, one());
    assert!(run_err(&exists, &cat, &mut pager).is_err(), "EXISTS id out of range");

    let in_sub = select_expr(
        EvalExpr::InSubquery {
            negated: false,
            subject: Box::new(lit(Value::Integer(1))),
            id: 5,
            meta: binary_meta(),
        },
        one(),
    );
    assert!(run_err(&in_sub, &cat, &mut pager).is_err(), "IN subquery id out of range");
}
