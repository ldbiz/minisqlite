//! Per-statement caching of UNCORRELATED expression-subquery results, end to end
//! through the [`Executor`] seam (lang_expr.html §12: "An uncorrelated subquery is
//! evaluated only once and the result reused as necessary. A correlated subquery is
//! reevaluated each time its result is required.").
//!
//! These pin the behavior the cache exists to guarantee — not an implementation
//! detail:
//!
//! * EVALUATE-ONCE: an uncorrelated scalar / `EXISTS` / `IN` subquery whose result
//!   depends on the RNG is evaluated once — its value repeats across every outer row and
//!   the RNG advances exactly once for the whole scan (a per-row re-run would draw once
//!   per row). The `EXISTS` witness also covers its cache-HIT branch, which the isolation
//!   tests (one outer row each) never reach.
//! * CORRELATED still re-runs per outer row: its result tracks the outer row, and a
//!   volatile correlated subquery draws once PER outer row.
//! * `IN` three-valued logic is preserved through the cached HashSet probe (numeric
//!   fold, collation, empty set, NULLs) on both the first (build) row and later
//!   (cache-hit) rows.
//! * ROW-VALUE `(a,…) IN (subquery)` caches candidate ROWS (`CachedSubquery::InRows`)
//!   once and reuses them on later outer rows — the same evaluate-once + cache-HIT
//!   coverage as the scalar `IN`, but for the tuple probe (`probe_in_rows`).
//! * CROSS-STATEMENT ISOLATION: one `Runtime` reused across two statements whose
//!   `subqueries[0]` differ does not leak the first's cached value into the second —
//!   proven on both the read path (`build_cursor`) and the DML path (`build_dml`, via a
//!   `DELETE ... RETURNING (SELECT rng())`), since `execute()` wraps both in `StatementRoot`.

use std::sync::Arc;

use minisqlite_btree::{init_database, table_insert};
use minisqlite_catalog::{Catalog, SchemaCatalog};
use minisqlite_exec::{Executor, Runtime, StreamingExecutor};
use minisqlite_expr::{CompareMeta, EvalExpr, FnContext, ScalarFunction};
use minisqlite_fileformat::encode_record;
use minisqlite_pager::{MemPager, PageId, Pager};
use minisqlite_plan::{Delete, Plan, PlanNode, SeqScan, SubPlan};
use minisqlite_sql::{parse, Statement};
use minisqlite_types::{Collation, Result, Value};
use minisqlite_exec::PagerSet;
use minisqlite_types::DbIndex;

// ----- fixtures (mirror tests/subquery.rs) ---------------------------------

fn db_with_table(create_sql: &str) -> (MemPager, SchemaCatalog) {
    let mut pager = MemPager::new(4096);
    init_database(&mut pager).unwrap();
    let mut cat = SchemaCatalog::new();
    create_table(&mut pager, &mut cat, create_sql);
    (pager, cat)
}

fn create_table(pager: &mut MemPager, cat: &mut SchemaCatalog, create_sql: &str) {
    let ast = parse(create_sql).unwrap();
    let Statement::CreateTable(stmt) = &ast.statements[0] else {
        panic!("not a CREATE TABLE: {create_sql}");
    };
    cat.create_table(pager, stmt, create_sql).unwrap();
}

fn seed(pager: &mut MemPager, root: PageId, rows: &[(i64, Vec<Value>)]) {
    pager.begin().unwrap();
    for (rowid, vals) in rows {
        table_insert(pager, root, *rowid, &encode_record(vals)).unwrap();
    }
    pager.commit().unwrap();
}

fn table_root(cat: &SchemaCatalog, name: &str) -> PageId {
    cat.table(name).unwrap().unwrap().root_page
}

/// Drain a plan to completion with a caller-owned [`Runtime`], so the RNG state after
/// the run can be inspected (the evaluate-once / re-run witnesses count draws).
fn run_rt(plan: &Plan, cat: &SchemaCatalog, pager: &mut MemPager, rt: &mut Runtime) -> Vec<Vec<Value>> {
    let mut exec = StreamingExecutor;
    let mut cur = exec.execute(plan, cat, PagerSet::One { db: DbIndex::MAIN, pager: &mut *pager }).unwrap();
    let mut out = Vec::new();
    while let Some(row) = cur.next_row(rt).unwrap() {
        out.push(row);
    }
    out
}

/// Drain a plan with a fresh [`Runtime`] (when the RNG after the run is irrelevant).
fn run(plan: &Plan, cat: &SchemaCatalog, pager: &mut MemPager) -> Vec<Vec<Value>> {
    let mut rt = Runtime::new();
    run_rt(plan, cat, pager, &mut rt)
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

fn meta_coll(collation: Collation) -> CompareMeta {
    CompareMeta { apply_left: None, apply_right: None, collation }
}

fn seqscan(table: &str, column_count: usize) -> PlanNode {
    PlanNode::SeqScan(SeqScan { db: DbIndex::MAIN, table: table.to_string(), column_count })
}

fn project(input: PlanNode, exprs: Vec<EvalExpr>) -> PlanNode {
    PlanNode::Project { input: Box::new(input), exprs }
}

fn sub(plan: PlanNode, correlated: bool) -> SubPlan {
    // The new SubPlan analysis fields are irrelevant to these executor cache tests (the
    // eval path does not read them yet); neutral defaults keep the helper minimal.
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

/// `SELECT <expr>` — one expression over the single implicit row, with the subplans
/// the expression references. Result: one row, one column.
fn select_expr(expr: EvalExpr, subqueries: Vec<SubPlan>) -> Plan {
    plan_subs(project(PlanNode::SingleRow, vec![expr]), &["r"], subqueries)
}

/// A candidate subplan that yields one literal-valued row per inner `EvalExpr` (an
/// empty `Vec` is the empty candidate set). Each inner expr is one candidate value.
fn candidate_set(values: Vec<EvalExpr>) -> PlanNode {
    PlanNode::Values { rows: values.into_iter().map(|e| vec![e]).collect() }
}

/// `<probe> IN (<candidate subplan>)`, uncorrelated, not negated, with `meta`.
fn in_plan(probe: Value, candidates: PlanNode, meta: CompareMeta) -> Plan {
    select_expr(
        EvalExpr::InSubquery { negated: false, subject: Box::new(lit(probe)), id: 0, meta },
        vec![sub(candidates, false)],
    )
}

fn int(v: &Value) -> i64 {
    match v {
        Value::Integer(i) => *i,
        other => panic!("expected Integer, got {other:?}"),
    }
}

/// A test-only scalar function that draws one pseudo-random `i64` from the runtime
/// each time it is CALLED, so a re-evaluation is directly observable as an extra RNG
/// draw. Using a genuinely volatile function is what makes "evaluated once" and
/// "re-evaluated per row" distinguishable — a pure function would look identical
/// either way.
#[derive(Debug)]
struct RngDraw;

impl ScalarFunction for RngDraw {
    fn call(&self, _args: &[Value], ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(Value::Integer(ctx.random_i64()))
    }
}

fn rng_func() -> EvalExpr {
    EvalExpr::Func { func: Arc::new(RngDraw), args: Vec::new() }
}

// ----- (2) EVALUATE-ONCE witness (uncorrelated) ----------------------------

#[test]
fn uncorrelated_scalar_subquery_evaluated_once_across_outer_rows() {
    // SELECT (SELECT rng()) FROM t2, where t2 has 2 rows. The subquery is
    // uncorrelated (SingleRow leaf), so it must be evaluated ONCE: both output rows
    // carry the same drawn value, and the RNG advances exactly once for the scan.
    let (mut pager, cat) = db_with_table("CREATE TABLE t2(a INTEGER)");
    let root = table_root(&cat, "t2");
    seed(&mut pager, root, &[(1, vec![Value::Integer(10)]), (2, vec![Value::Integer(20)])]);

    let subplan = project(PlanNode::SingleRow, vec![rng_func()]);
    let p = plan_subs(
        project(seqscan("t2", 1), vec![EvalExpr::ScalarSubquery(0)]),
        &["s"],
        vec![sub(subplan, false)],
    );

    let mut rt = Runtime::new();
    let rows = run_rt(&p, &cat, &mut pager, &mut rt);
    assert_eq!(rows.len(), 2, "two outer rows");
    let (v0, v1) = (int(&rows[0][0]), int(&rows[1][0]));
    assert_eq!(v0, v1, "an uncorrelated subquery is evaluated once: same value on every row");

    // Draw-count witness: exactly ONE random draw occurred for the whole 2-row scan.
    let mut reference = Runtime::new();
    assert_eq!(v0, reference.random_i64(), "the reused value is the FIRST random draw");
    assert_eq!(rt.random_i64(), reference.random_i64(), "the RNG advanced exactly once (not per row)");
}

#[test]
fn uncorrelated_in_subquery_candidate_set_materialized_once() {
    // SELECT (col IN (SELECT rng())) FROM t2, t2 has 2 rows. The candidate set is
    // uncorrelated, so it is materialized ONCE (one RNG draw) and reused for both
    // outer rows — not rebuilt per row (which would draw twice).
    let (mut pager, cat) = db_with_table("CREATE TABLE t2(a INTEGER)");
    let root = table_root(&cat, "t2");
    seed(&mut pager, root, &[(1, vec![Value::Integer(10)]), (2, vec![Value::Integer(20)])]);

    let candidates = candidate_set(vec![rng_func()]);
    let p = plan_subs(
        project(
            seqscan("t2", 1),
            vec![EvalExpr::InSubquery {
                negated: false,
                subject: Box::new(col(0)),
                id: 0,
                meta: binary_meta(),
            }],
        ),
        &["m"],
        vec![sub(candidates, false)],
    );

    let mut rt = Runtime::new();
    let rows = run_rt(&p, &cat, &mut pager, &mut rt);
    assert_eq!(rows.len(), 2, "two outer rows");
    // Exactly ONE draw funded the candidate set for the whole scan.
    let mut reference = Runtime::new();
    let _ = reference.random_i64();
    assert_eq!(rt.random_i64(), reference.random_i64(), "IN set materialized once (one draw total)");
}

#[test]
fn uncorrelated_exists_evaluated_once_across_outer_rows() {
    // SELECT (EXISTS (SELECT rng())) FROM t2, t2 has 2 rows. EXISTS yields no value, so
    // the RNG-draw count is the observable that distinguishes once vs per-row: an
    // uncorrelated EXISTS is evaluated ONCE, so its subplan (whose one row draws from the
    // RNG) runs once for the whole scan — the RNG advances exactly once. This is the ONLY
    // test that reaches the EXISTS cache-HIT branch (`eval_exists`'s `Exists(b) => Ok(*b)`):
    // the isolation test below uses one outer row per statement, so it only exercises
    // miss+store. Without this, a broken hit arm (wrong variant, negated bool) stays green.
    let (mut pager, cat) = db_with_table("CREATE TABLE t2(a INTEGER)");
    let root = table_root(&cat, "t2");
    seed(&mut pager, root, &[(1, vec![Value::Integer(10)]), (2, vec![Value::Integer(20)])]);

    // The EXISTS subplan's single row draws from the RNG when pulled; EXISTS stops after
    // that first row, so each evaluation is exactly one draw.
    let subplan = project(PlanNode::SingleRow, vec![rng_func()]);
    let p = plan_subs(
        project(seqscan("t2", 1), vec![EvalExpr::Exists { negated: false, id: 0 }]),
        &["e"],
        vec![sub(subplan, false)],
    );

    let mut rt = Runtime::new();
    let rows = run_rt(&p, &cat, &mut pager, &mut rt);
    assert_eq!(rows.len(), 2, "two outer rows");
    assert_eq!(int(&rows[0][0]), 1, "EXISTS over a one-row subplan is true");
    assert_eq!(int(&rows[1][0]), 1, "and the cached hit reports the same truth");
    // The witness: exactly ONE draw funded EXISTS for the whole scan (the second outer
    // row is a cache hit that does NOT re-run the subplan, so it does not draw).
    let mut reference = Runtime::new();
    let _ = reference.random_i64();
    assert_eq!(
        rt.random_i64(),
        reference.random_i64(),
        "uncorrelated EXISTS evaluated once (one draw total, not one per outer row)"
    );
}

// ----- (3) CORRELATED still re-runs per outer row --------------------------

#[test]
fn correlated_scalar_subquery_reruns_and_tracks_outer_row() {
    // SELECT (SELECT t1.a) FROM t1 — a correlated subquery MUST re-run per outer row
    // and read the outer value, so the outputs track [1,2,3]. If caching wrongly
    // froze it, every row would show the first value (1,1,1).
    let (mut pager, cat) = db_with_table("CREATE TABLE t1(a INTEGER)");
    let r1 = table_root(&cat, "t1");
    seed(
        &mut pager,
        r1,
        &[(1, vec![Value::Integer(1)]), (2, vec![Value::Integer(2)]), (3, vec![Value::Integer(3)])],
    );

    let subplan = project(PlanNode::SingleRow, vec![col(0)]);
    let p = plan_subs(
        project(seqscan("t1", 1), vec![EvalExpr::ScalarSubquery(0)]),
        &["s"],
        vec![sub(subplan, true)],
    );
    let got: Vec<i64> = run(&p, &cat, &mut pager).iter().map(|r| int(&r[0])).collect();
    assert_eq!(got, vec![1, 2, 3], "correlated subquery re-runs per outer row and reads the outer value");
}

#[test]
fn correlated_volatile_subquery_reruns_per_outer_row() {
    // A correlated subquery marked correlated re-runs per outer row even when volatile:
    // over 3 outer rows the RNG advances exactly 3 times (once per re-run), and the
    // three outputs are the three successive draws. A wrongly-cached correlated subplan
    // would draw only once and repeat that value.
    let (mut pager, cat) = db_with_table("CREATE TABLE t1(a INTEGER)");
    let r1 = table_root(&cat, "t1");
    seed(
        &mut pager,
        r1,
        &[(1, vec![Value::Integer(1)]), (2, vec![Value::Integer(2)]), (3, vec![Value::Integer(3)])],
    );

    // Marked correlated, so it is NOT cached — re-run for each of the 3 outer rows.
    let subplan = project(PlanNode::SingleRow, vec![rng_func()]);
    let p = plan_subs(
        project(seqscan("t1", 1), vec![EvalExpr::ScalarSubquery(0)]),
        &["s"],
        vec![sub(subplan, true)],
    );

    let mut rt = Runtime::new();
    let rows = run_rt(&p, &cat, &mut pager, &mut rt);
    assert_eq!(rows.len(), 3);
    // The three outputs are the three successive draws (re-run each row).
    let mut ref_seq = Runtime::new();
    assert_eq!(int(&rows[0][0]), ref_seq.random_i64());
    assert_eq!(int(&rows[1][0]), ref_seq.random_i64());
    assert_eq!(int(&rows[2][0]), ref_seq.random_i64());
    // And exactly 3 draws happened (one per outer row).
    assert_eq!(rt.random_i64(), ref_seq.random_i64(), "correlated subquery drew once per outer row");
}

// ----- (4) IN three-valued logic preserved through the cache ---------------

#[test]
fn in_cache_numeric_fold_integer_matches_real() {
    // 2 IN (SELECT 2.0) -> true: Integer(2) and Real(2.0) fold to the same key, just
    // as compare_for_eq treats them as numerically equal.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(x INTEGER)"); // just to have a db
    let _ = table_root(&cat, "t");
    let p = in_plan(Value::Integer(2), candidate_set(vec![lit(Value::Real(2.0))]), binary_meta());
    assert_eq!(int(&run(&p, &cat, &mut pager)[0][0]), 1, "2 IN (SELECT 2.0) is true (numeric fold)");
}

#[test]
fn in_cache_collation_nocase_vs_binary() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(x INTEGER)");

    let nocase = in_plan(
        Value::Text("a".into()),
        candidate_set(vec![lit(Value::Text("A".into()))]),
        meta_coll(Collation::NoCase),
    );
    assert_eq!(int(&run(&nocase, &cat, &mut pager)[0][0]), 1, "'a' IN (SELECT 'A') is true under NOCASE");

    let binary = in_plan(
        Value::Text("a".into()),
        candidate_set(vec![lit(Value::Text("A".into()))]),
        binary_meta(),
    );
    assert_eq!(int(&run(&binary, &cat, &mut pager)[0][0]), 0, "'a' IN (SELECT 'A') is false under BINARY");
}

#[test]
fn in_cache_empty_set_is_false_even_for_null_probe() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(x INTEGER)");
    let empty = || candidate_set(Vec::new());
    assert_eq!(int(&run(&in_plan(Value::Integer(1), empty(), binary_meta()), &cat, &mut pager)[0][0]), 0, "1 IN () is false");
    assert_eq!(int(&run(&in_plan(Value::Null, empty(), binary_meta()), &cat, &mut pager)[0][0]), 0, "NULL IN () is false");
}

#[test]
fn in_cache_null_in_set_makes_nonmatching_probe_unknown() {
    // A NULL candidate + a non-matching non-NULL probe -> unknown (NULL).
    let (mut pager, cat) = db_with_table("CREATE TABLE t(x INTEGER)");
    let candidates = candidate_set(vec![lit(Value::Null), lit(Value::Integer(1))]);
    let p = in_plan(Value::Integer(9), candidates, binary_meta());
    assert!(run(&p, &cat, &mut pager)[0][0].is_null(), "9 IN (NULL, 1) is unknown");
}

#[test]
fn in_cache_null_probe_against_nonempty_set_is_unknown() {
    let (mut pager, cat) = db_with_table("CREATE TABLE t(x INTEGER)");
    let p = in_plan(Value::Null, candidate_set(vec![lit(Value::Integer(1))]), binary_meta());
    assert!(run(&p, &cat, &mut pager)[0][0].is_null(), "NULL IN (1) is unknown");
}

#[test]
fn in_cache_found_value_wins_over_null_candidate() {
    // A present match settles TRUE even with a NULL candidate around.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(x INTEGER)");
    let candidates = candidate_set(vec![lit(Value::Null), lit(Value::Integer(1))]);
    let p = in_plan(Value::Integer(1), candidates, binary_meta());
    assert_eq!(int(&run(&p, &cat, &mut pager)[0][0]), 1, "1 IN (NULL, 1) is true");
}

#[test]
fn in_cache_hit_path_probes_correctly_across_outer_rows() {
    // Exercise BOTH the build (first outer row) and the cache-hit (later rows) probe:
    // SELECT (col IN (SELECT 2)) FROM t2 over col in {2,9,2} yields [1,0,1]. The set
    // {2} is built on row 0 and reused (a HashSet::contains) on rows 1 and 2.
    let (mut pager, cat) = db_with_table("CREATE TABLE t2(a INTEGER)");
    let root = table_root(&cat, "t2");
    seed(
        &mut pager,
        root,
        &[(1, vec![Value::Integer(2)]), (2, vec![Value::Integer(9)]), (3, vec![Value::Integer(2)])],
    );
    let p = plan_subs(
        project(
            seqscan("t2", 1),
            vec![EvalExpr::InSubquery {
                negated: false,
                subject: Box::new(col(0)),
                id: 0,
                meta: binary_meta(),
            }],
        ),
        &["m"],
        vec![sub(candidate_set(vec![lit(Value::Integer(2))]), false)],
    );
    let got: Vec<i64> = run(&p, &cat, &mut pager).iter().map(|r| int(&r[0])).collect();
    assert_eq!(got, vec![1, 0, 1], "cached IN set probes correctly on the build row and cache-hit rows");
}

// ----- (4b) ROW-VALUE IN: the same cache coverage for the tuple probe -------

#[test]
fn uncorrelated_row_in_subquery_candidate_rows_materialized_once() {
    // The row-value analogue of `uncorrelated_in_subquery_candidate_set_materialized_once`.
    // SELECT ((a, 5) IN (SELECT rng(), 5)) FROM t2, t2 has 2 rows. The candidate row set
    // is uncorrelated, so it is materialized ONCE (one RNG draw) and reused for both outer
    // rows via `CachedSubquery::InRows` — not rebuilt per row (which would draw twice).
    // This is the "materialize once" witness for the tuple path; deleting the
    // `if !sub.correlated { … cache … }` block (re-materializing every outer row) would
    // draw twice and fail here.
    let (mut pager, cat) = db_with_table("CREATE TABLE t2(a INTEGER)");
    let root = table_root(&cat, "t2");
    seed(&mut pager, root, &[(1, vec![Value::Integer(10)]), (2, vec![Value::Integer(20)])]);

    // One-row, two-column candidate whose first column draws from the RNG when pulled.
    let candidate = project(PlanNode::SingleRow, vec![rng_func(), lit(Value::Integer(5))]);
    let p = plan_subs(
        project(
            seqscan("t2", 1),
            vec![EvalExpr::InSubqueryRow {
                negated: false,
                subjects: vec![col(0), lit(Value::Integer(5))],
                id: 0,
                metas: vec![binary_meta(), binary_meta()],
            }],
        ),
        &["m"],
        vec![sub(candidate, false)],
    );

    let mut rt = Runtime::new();
    let rows = run_rt(&p, &cat, &mut pager, &mut rt);
    assert_eq!(rows.len(), 2, "two outer rows");
    // The witness: exactly ONE draw funded the candidate rows for the whole scan (the
    // second outer row is a cache HIT that does not re-run the subplan, so it does not draw).
    let mut reference = Runtime::new();
    let _ = reference.random_i64();
    assert_eq!(
        rt.random_i64(),
        reference.random_i64(),
        "row-value IN candidate rows materialized once (one draw total, not one per outer row)"
    );
}

#[test]
fn row_in_cache_hit_path_probes_correctly_across_outer_rows() {
    // The row-value analogue of `in_cache_hit_path_probes_correctly_across_outer_rows`:
    // exercise BOTH the build (first outer row) and the cache-HIT (later rows) tuple probe.
    // SELECT ((a, 2) IN (SELECT 1, 2)) FROM t2 over a in {1,5,1} yields [1,0,1]: the
    // candidate rows {(1,2)} are materialized on row 0 (`CachedSubquery::InRows`) and
    // reused (`probe_in_rows` over the cached Vec) on rows 1 and 2. A broken hit arm — a
    // wrong variant, or a constant answer such as `Ok(Some(false))` — would corrupt the
    // cache-hit rows (giving [1,0,0]) while the build row stays correct, so this is the
    // test that fails on that mutation.
    let (mut pager, cat) = db_with_table("CREATE TABLE t2(a INTEGER)");
    let root = table_root(&cat, "t2");
    seed(
        &mut pager,
        root,
        &[(1, vec![Value::Integer(1)]), (2, vec![Value::Integer(5)]), (3, vec![Value::Integer(1)])],
    );
    // Two-column candidate set {(1,2)} (an uncorrelated `SELECT 1, 2`).
    let candidates =
        PlanNode::Values { rows: vec![vec![lit(Value::Integer(1)), lit(Value::Integer(2))]] };
    let p = plan_subs(
        project(
            seqscan("t2", 1),
            vec![EvalExpr::InSubqueryRow {
                negated: false,
                subjects: vec![col(0), lit(Value::Integer(2))],
                id: 0,
                metas: vec![binary_meta(), binary_meta()],
            }],
        ),
        &["m"],
        vec![sub(candidates, false)],
    );
    let got: Vec<i64> = run(&p, &cat, &mut pager).iter().map(|r| int(&r[0])).collect();
    assert_eq!(
        got,
        vec![1, 0, 1],
        "cached row-value IN probes correctly on the build row (a=1) and the cache-HIT rows (a=5, a=1)"
    );
}

// ----- (5) CROSS-STATEMENT ISOLATION ---------------------------------------

#[test]
fn cache_does_not_bleed_across_statements() {
    // One Runtime, two DIFFERENT statements whose subqueries[0] differ:
    //   A: SELECT (SELECT 1)  -> caches id 0 = Scalar(1)
    //   B: SELECT (SELECT 2)  -> must return 2, NOT A's stale cached 1.
    // This is the definitive proof the per-statement clear (StatementRoot) fires.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(x INTEGER)");
    let mut rt = Runtime::new();

    let plan_a = select_expr(
        EvalExpr::ScalarSubquery(0),
        vec![sub(project(PlanNode::SingleRow, vec![lit(Value::Integer(1))]), false)],
    );
    let a = run_rt(&plan_a, &cat, &mut pager, &mut rt);
    assert_eq!(int(&a[0][0]), 1, "statement A returns 1");

    let plan_b = select_expr(
        EvalExpr::ScalarSubquery(0),
        vec![sub(project(PlanNode::SingleRow, vec![lit(Value::Integer(2))]), false)],
    );
    let b = run_rt(&plan_b, &cat, &mut pager, &mut rt);
    assert_eq!(int(&b[0][0]), 2, "statement B returns 2, not A's cached 1 — no cross-statement bleed");
}

#[test]
fn exists_cache_does_not_bleed_across_statements() {
    // Same isolation proof for EXISTS: A over a non-empty set (EXISTS -> 1) then B over
    // an empty set (EXISTS -> 0), sharing one Runtime and id 0.
    let (mut pager, cat) = db_with_table("CREATE TABLE t2(a INTEGER)");
    let root = table_root(&cat, "t2");
    seed(&mut pager, root, &[(1, vec![Value::Integer(1)])]);
    let mut rt = Runtime::new();

    let plan_a = select_expr(
        EvalExpr::Exists { negated: false, id: 0 },
        vec![sub(seqscan("t2", 1), false)],
    );
    assert_eq!(int(&run_rt(&plan_a, &cat, &mut pager, &mut rt)[0][0]), 1, "A: EXISTS over non-empty is 1");

    let plan_b = select_expr(
        EvalExpr::Exists { negated: false, id: 0 },
        vec![sub(candidate_set(Vec::new()), false)],
    );
    assert_eq!(int(&run_rt(&plan_b, &cat, &mut pager, &mut rt)[0][0]), 0, "B: EXISTS over empty is 0, not A's cached 1");
}

#[test]
fn correlated_cache_does_not_bleed_across_statements() {
    // The CORRELATED analogue of `cache_does_not_bleed_across_statements`, and the isolation
    // guard the uncorrelated tests alone MISS: a regression that cleared only the
    // UNCORRELATED map (leaving the correlated one) would pass those two but be caught HERE.
    // One Runtime, two statements reusing subqueries[0] with the SAME correlation key
    // (outer g = 5) but different results:
    //   A: correlated (SELECT 1) over g=5  -> memoizes (id 0, key [Int(5)]) = Scalar(1)
    //   B: correlated (SELECT 2) over g=5  -> must return 2, NOT A's stale cached 1.
    // Both subplans are correlated on [0] + deterministic (memo-eligible) and would hit the
    // SAME (id, key); only the per-statement clear of the CORRELATED map (StatementRoot ->
    // clear_subquery_cache, which clears BOTH maps) keeps B from serving A's entry.
    let (mut pager, cat) = db_with_table("CREATE TABLE t1(g INTEGER)");
    let r1 = table_root(&cat, "t1");
    seed(&mut pager, r1, &[(1, vec![Value::Integer(5)])]);
    let mut rt = Runtime::new();

    let plan_a = select_scalar_over(
        "t1",
        1,
        EvalExpr::ScalarSubquery(0),
        sub_corr(project(PlanNode::SingleRow, vec![lit(Value::Integer(1))]), vec![0], true),
    );
    let a = run_rt(&plan_a, &cat, &mut pager, &mut rt);
    assert_eq!(int(&a[0][0]), 1, "statement A returns 1");
    assert_eq!(rt.correlated_cache_len(), 1, "A memoized one correlated entry under (id 0, key g=5)");

    let plan_b = select_scalar_over(
        "t1",
        1,
        EvalExpr::ScalarSubquery(0),
        sub_corr(project(PlanNode::SingleRow, vec![lit(Value::Integer(2))]), vec![0], true),
    );
    let b = run_rt(&plan_b, &cat, &mut pager, &mut rt);
    assert_eq!(
        int(&b[0][0]),
        2,
        "statement B returns 2, not A's cached 1 — the CORRELATED map cleared across statements"
    );
}

// ----- (6) DML path: execute() wraps build_dml in StatementRoot too --------

/// Apply a DML plan inside a write transaction, threading the caller's [`Runtime`] so
/// its RNG state can be inspected. Mirrors the engine's DML drive (`execute()` + drain).
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

/// `DELETE FROM <table> RETURNING <returning>` over a full scan, carrying the `subqueries`
/// the RETURNING expression references. Its root is a `Delete`, so `execute()` routes it
/// through the `build_dml` branch — the other branch `StatementRoot` must also wrap.
fn delete_returning(table: &str, returning: EvalExpr, subqueries: Vec<SubPlan>) -> Plan {
    Plan {
        root: PlanNode::Delete(Delete { db: DbIndex::MAIN,
            table: table.to_string(),
            column_count: 1,
            scan: Box::new(seqscan(table, 1)),
            returning: vec![returning],
            triggers: Vec::new(),
            index_key_exprs: Vec::new(),
            index_partial_predicates: Vec::new(),
        }),
        result_columns: Vec::new(),
        ctes: Vec::new(),
        subqueries,
        mutates: true,
        generated: Vec::new(),
    }
}

#[test]
fn dml_uncorrelated_returning_subquery_evaluated_once_and_isolated() {
    // Exercises the DML branch of execute()'s StatementRoot wrap (the read-path tests
    // above only cover build_cursor). A DELETE whose RETURNING carries an uncorrelated
    // (SELECT rng()) evaluates that subquery ONCE per statement — both deleted rows
    // return the same drawn value — and a SECOND DELETE on the same Runtime draws AGAIN,
    // proving StatementRoot cleared the cache on the DML drain's first pull (the RETURNING
    // subquery is evaluated inside DeleteCursor::build(), which runs on that first pull,
    // AFTER the clear). Without the clear, B would reuse A's cached Scalar and not draw.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER)");
    let root = table_root(&cat, "t");
    let mut rt = Runtime::new();

    // Statement A: delete 2 rows, RETURNING (SELECT rng()).
    seed(&mut pager, root, &[(1, vec![Value::Integer(10)]), (2, vec![Value::Integer(20)])]);
    let plan_a = delete_returning(
        "t",
        EvalExpr::ScalarSubquery(0),
        vec![sub(project(PlanNode::SingleRow, vec![rng_func()]), false)],
    );
    let a = run_dml(&plan_a, &cat, &mut pager, &mut rt);
    assert_eq!(a.len(), 2, "two rows deleted -> two RETURNING rows");
    assert_eq!(int(&a[0][0]), int(&a[1][0]), "uncorrelated RETURNING subquery evaluated once (same value)");

    let mut reference = Runtime::new();
    assert_eq!(int(&a[0][0]), reference.random_i64(), "the reused value is the first draw");

    // Statement B: a fresh DELETE on the same Runtime, re-seeded. If the cache bled, B
    // would reuse A's cached Scalar and draw nothing; StatementRoot's clear makes it draw.
    seed(&mut pager, root, &[(3, vec![Value::Integer(30)])]);
    let plan_b = delete_returning(
        "t",
        EvalExpr::ScalarSubquery(0),
        vec![sub(project(PlanNode::SingleRow, vec![rng_func()]), false)],
    );
    let b = run_dml(&plan_b, &cat, &mut pager, &mut rt);
    assert_eq!(b.len(), 1, "one row deleted -> one RETURNING row");
    assert_eq!(int(&b[0][0]), reference.random_i64(), "statement B drew a FRESH value (cache cleared on the DML drain)");
    assert_ne!(int(&a[0][0]), int(&b[0][0]), "B did not reuse A's cached subquery value");
}

// ===========================================================================
// (7) CORRELATED subquery MEMOIZATION
// ---------------------------------------------------------------------------
// The correlated memo (context::correlated_memo_eligible) collapses a DETERMINISTIC
// correlated subplan to ONE run per DISTINCT correlation key instead of one per outer row
// — the O(n*n) -> O(n) win — while preserving the exact per-row re-run answer. These pin
// the CONTRACT through the executor seam, hand-building a `SubPlan` so a test controls the
// `correlated_cols` / `deterministic` fields the planner computes for real SQL:
//
//   * a genuine cache HIT collapses same-key rows (an RNG-draw witness + `correlated_cache_len`),
//     and the toggle differential shows the memo answer is byte-identical to the re-run;
//   * a NULL correlation key is ONE key (two NULL rows share a run);
//   * a MULTI-COLUMN key does not collapse rows that differ in any key column;
//   * storage classes are DISTINCT keys (Int(2) != Real(2.0), the fold-free CorrCell) —
//     unlike the value-folding `IN` set;
//   * the `IN` memo shares the candidate SET/ROWS per key but probes with THIS row's
//     subject (a varying probe over a shared set stays correct);
//   * a VOLATILE correlated subplan (guard b) and a MUTATING statement (guard a) are
//     NEVER memoized — they re-run per row (these are the wrong-answer gates).

/// A CORRELATED subplan with explicit analysis fields, so a test can exercise the correlated
/// MEMO. (The `sub()` helper leaves `correlated_cols` EMPTY, which is ineligible by design,
/// so it cannot drive the memo.) `correlated_cols` are the outer registers the subplan's
/// result depends on — the memo key is built from exactly these — and `deterministic` gates
/// whether the memo engages at all (a volatile subplan must re-run per outer row).
fn sub_corr(plan: PlanNode, correlated_cols: Vec<usize>, deterministic: bool) -> SubPlan {
    let outer_width = correlated_cols.iter().copied().max().map_or(0, |m| m + 1);
    SubPlan { plan, correlated: true, correlated_cols, deterministic, outer_width }
}

/// A storage-class-EXACT string view of a result set, for value equality in assertions
/// (`Value` has no `PartialEq`). `Debug` distinguishes Integer(2) from Real(2.0) and NULL
/// from a value, so it is the right fold-free comparison for the differential and the
/// expected-result checks (mirrors how the executor's fold-free correlation key behaves).
fn rows_dbg(rows: &[Vec<Value>]) -> String {
    format!("{rows:?}")
}

/// Run `plan` twice on fresh runtimes with the SAME default RNG seed — once with the
/// correlated memo ON (the default) and once forced OFF via
/// [`Runtime::set_correlated_cache_disabled`] — and ASSERT the two answers are byte-
/// identical (the memo changes cost, never the answer) and that the OFF run cached nothing.
/// Returns `(memo_on_rows, memo_on_cache_len)` so a caller pins the expected result and the
/// hit count. Only valid for TRULY deterministic subplans; a volatile-observable test
/// compares RNG draw counts instead (on != off there is the whole point of the collapse).
fn run_memo_on_off(
    plan: &Plan,
    cat: &SchemaCatalog,
    pager: &mut MemPager,
) -> (Vec<Vec<Value>>, usize) {
    let mut rt_on = Runtime::new();
    let on = run_rt(plan, cat, pager, &mut rt_on);
    let on_len = rt_on.correlated_cache_len();
    let mut rt_off = Runtime::new();
    rt_off.set_correlated_cache_disabled(true);
    let off = run_rt(plan, cat, pager, &mut rt_off);
    assert_eq!(rt_off.correlated_cache_len(), 0, "memo disabled: nothing cached");
    assert_eq!(rows_dbg(&on), rows_dbg(&off), "correlated memo answer must equal the per-row re-run");
    (on, on_len)
}

/// A one-column `SELECT (SELECT 0)`-shaped read: `SELECT <subquery-expr> FROM <table>` with
/// the correlated subplan at id 0. The seqscan is the outer, so `regs` per eval is the outer
/// row and the subplan's `Column(i)` reads it.
fn select_scalar_over(table: &str, cols: usize, expr: EvalExpr, subplan: SubPlan) -> Plan {
    plan_subs(project(seqscan(table, cols), vec![expr]), &["s"], vec![subplan])
}

#[test]
fn correlated_scalar_memo_collapses_runs_for_shared_key() {
    // A DETERMINISTIC correlated subplan with a non-empty key is MEMOIZED: outer rows
    // sharing a key run the subplan ONCE. We make the collapse OBSERVABLE with a volatile
    // draw (marking it `deterministic:true` so the memo engages): over 5 rows whose
    // correlation column is [1,1,2,1,2] (2 distinct keys) the RNG advances exactly TWICE,
    // same-key rows carry identical values, and the memo holds exactly 2 entries. With the
    // memo DISABLED the same run draws 5 times (once per row) — the differential proving a
    // real hit, not merely a right value.
    let (mut pager, cat) = db_with_table("CREATE TABLE t1(g INTEGER)");
    let r1 = table_root(&cat, "t1");
    seed(
        &mut pager,
        r1,
        &[
            (1, vec![Value::Integer(1)]),
            (2, vec![Value::Integer(1)]),
            (3, vec![Value::Integer(2)]),
            (4, vec![Value::Integer(1)]),
            (5, vec![Value::Integer(2)]),
        ],
    );
    let plan = || {
        select_scalar_over(
            "t1",
            1,
            EvalExpr::ScalarSubquery(0),
            sub_corr(project(PlanNode::SingleRow, vec![rng_func()]), vec![0], true),
        )
    };

    // Memo ON (default): one draw per DISTINCT key.
    let mut rt = Runtime::new();
    let on = plan();
    let vals: Vec<i64> = run_rt(&on, &cat, &mut pager, &mut rt).iter().map(|r| int(&r[0])).collect();
    assert_eq!(vals[0], vals[1], "rows sharing key 1 reuse one memoized draw");
    assert_eq!(vals[1], vals[3], "the third key-1 row reuses the same memoized draw");
    assert_eq!(vals[2], vals[4], "rows sharing key 2 reuse one memoized draw");
    assert_ne!(vals[0], vals[2], "different keys get different runs");
    assert_eq!(rt.correlated_cache_len(), 2, "exactly two distinct correlation keys memoized");
    let mut reference = Runtime::new();
    let (d1, d2) = (reference.random_i64(), reference.random_i64());
    assert_eq!(vals, vec![d1, d1, d2, d1, d2], "same-key rows share the key's single draw");
    assert_eq!(rt.random_i64(), reference.random_i64(), "the RNG advanced exactly twice (once per key)");

    // Memo OFF via the toggle: the SAME plan re-runs per outer row -> 5 distinct draws.
    let mut rt_off = Runtime::new();
    rt_off.set_correlated_cache_disabled(true);
    let off = plan();
    let vals_off: Vec<i64> = run_rt(&off, &cat, &mut pager, &mut rt_off).iter().map(|r| int(&r[0])).collect();
    let mut ref_off = Runtime::new();
    let expected_off: Vec<i64> = (0..5).map(|_| ref_off.random_i64()).collect();
    assert_eq!(vals_off, expected_off, "memo disabled: one fresh draw per outer row");
    assert_eq!(rt_off.correlated_cache_len(), 0, "memo disabled: nothing cached");
}

#[test]
fn correlated_memo_null_key_is_one_key() {
    // Two outer rows with a NULL correlation key SHARE one memo entry (NULL == NULL for the
    // KEY, distinct from SQL's value NULL != NULL): the deterministic subquery yields the
    // same result for both, so collapsing them is correct. g = [NULL, NULL, 5] draws twice
    // (key NULL once, key 5 once), the two NULL rows carry the identical draw, and the memo
    // holds 2 entries — proving CorrCell::Null hashes/compares as a single key.
    let (mut pager, cat) = db_with_table("CREATE TABLE t1(g INTEGER)");
    let r1 = table_root(&cat, "t1");
    seed(
        &mut pager,
        r1,
        &[(1, vec![Value::Null]), (2, vec![Value::Null]), (3, vec![Value::Integer(5)])],
    );
    let mut rt = Runtime::new();
    let plan = select_scalar_over(
        "t1",
        1,
        EvalExpr::ScalarSubquery(0),
        sub_corr(project(PlanNode::SingleRow, vec![rng_func()]), vec![0], true),
    );
    let vals: Vec<i64> = run_rt(&plan, &cat, &mut pager, &mut rt).iter().map(|r| int(&r[0])).collect();
    let mut reference = Runtime::new();
    let (d1, d2) = (reference.random_i64(), reference.random_i64());
    assert_eq!(vals, vec![d1, d1, d2], "the two NULL-key rows share the NULL key's single draw");
    assert_eq!(rt.correlated_cache_len(), 2, "NULL and 5 are two distinct keys");
}

#[test]
fn correlated_memo_storage_class_keys_are_distinct() {
    // The correlation key is STORAGE-CLASS EXACT (fold-free CorrCell), UNLIKE the value-
    // folding `IN` set: an outer key Integer(2) and Real(2.0) are DIFFERENT keys, so each
    // gets its own subplan run (its own draw) and the memo holds 2 entries. A value-folding
    // key (2 == 2.0) would instead collapse them to one entry and serve the second row the
    // first's value — this test fails loudly on that regression.
    let (mut pager, cat) = db_with_table("CREATE TABLE t1(g INTEGER)");
    let r1 = table_root(&cat, "t1");
    // seed() writes raw records (no INSERT-time affinity), so a Real stored in `g` stays a
    // Real on read — exactly the storage-class distinction the correlation key must keep.
    seed(&mut pager, r1, &[(1, vec![Value::Integer(2)]), (2, vec![Value::Real(2.0)])]);
    let mut rt = Runtime::new();
    let plan = select_scalar_over(
        "t1",
        1,
        EvalExpr::ScalarSubquery(0),
        sub_corr(project(PlanNode::SingleRow, vec![rng_func()]), vec![0], true),
    );
    let vals: Vec<i64> = run_rt(&plan, &cat, &mut pager, &mut rt).iter().map(|r| int(&r[0])).collect();
    let mut reference = Runtime::new();
    let (d1, d2) = (reference.random_i64(), reference.random_i64());
    assert_eq!(vals, vec![d1, d2], "Integer(2) and Real(2.0) are distinct keys: two runs, two draws");
    assert_ne!(vals[0], vals[1], "the two storage-class keys did NOT collapse");
    assert_eq!(rt.correlated_cache_len(), 2, "int/real keys do not fold");
}

#[test]
fn correlated_memo_multicolumn_key() {
    // A MULTI-COLUMN correlation key (`correlated_cols == [0, 1]`) collapses only rows equal
    // in EVERY key column. Over (g,h) = (1,1),(1,1),(1,2): the first two share a key (one
    // draw), the third differs in h (a second draw). Under-keying (dropping h) would wrongly
    // collapse all three; this pins the full-tuple key.
    let (mut pager, cat) = db_with_table("CREATE TABLE t2(g INTEGER, h INTEGER)");
    let r = table_root(&cat, "t2");
    seed(
        &mut pager,
        r,
        &[
            (1, vec![Value::Integer(1), Value::Integer(1)]),
            (2, vec![Value::Integer(1), Value::Integer(1)]),
            (3, vec![Value::Integer(1), Value::Integer(2)]),
        ],
    );
    let mut rt = Runtime::new();
    let plan = select_scalar_over(
        "t2",
        2,
        EvalExpr::ScalarSubquery(0),
        sub_corr(project(PlanNode::SingleRow, vec![rng_func()]), vec![0, 1], true),
    );
    let vals: Vec<i64> = run_rt(&plan, &cat, &mut pager, &mut rt).iter().map(|r| int(&r[0])).collect();
    let mut reference = Runtime::new();
    let (d1, d2) = (reference.random_i64(), reference.random_i64());
    assert_eq!(vals, vec![d1, d1, d2], "(1,1) rows share; (1,2) is a distinct key");
    assert_eq!(rt.correlated_cache_len(), 2, "two distinct (g,h) keys");
}

#[test]
fn correlated_scalar_reads_outer_memo_matches_rerun() {
    // A genuinely DETERMINISTIC correlated scalar `SELECT outer.g` over repeated + NULL
    // keys: the memoized answer is byte-identical to the per-row re-run (cache-on == off),
    // each row reads its own outer value, and the memo holds one entry per distinct key.
    let (mut pager, cat) = db_with_table("CREATE TABLE t1(g INTEGER)");
    let r1 = table_root(&cat, "t1");
    seed(
        &mut pager,
        r1,
        &[
            (1, vec![Value::Integer(5)]),
            (2, vec![Value::Integer(5)]),
            (3, vec![Value::Integer(7)]),
            (4, vec![Value::Null]),
            (5, vec![Value::Null]),
        ],
    );
    let plan = select_scalar_over(
        "t1",
        1,
        EvalExpr::ScalarSubquery(0),
        sub_corr(project(PlanNode::SingleRow, vec![col(0)]), vec![0], true),
    );
    let (on, len) = run_memo_on_off(&plan, &cat, &mut pager);
    let expected =
        vec![vec![Value::Integer(5)], vec![Value::Integer(5)], vec![Value::Integer(7)], vec![Value::Null], vec![Value::Null]];
    assert_eq!(rows_dbg(&on), rows_dbg(&expected), "each row reads its own outer value (NULL-key rows both yield NULL)");
    assert_eq!(len, 3, "three distinct keys: 5, 7, NULL");
}

#[test]
fn correlated_scalar_column_memo_matches_rerun() {
    // The `ScalarSubqueryColumn` shape (a column-list UPDATE source, rowvalue.html §2.3):
    // one subplan run per outer row feeds N column reads. Correlated + deterministic, it is
    // memoized by the correlation key; each row reads column 1 of the memoized FirstRow.
    // Column 1 here is the OUTER KEY (not a constant), so the value pin CATCHES a wrong-key
    // serve: two same-key rows share the value and the differing-key row must differ — a
    // constant column (as this test once used) could not tell a mis-keyed serve from a hit.
    // Cache-on == off, and the memo holds one entry per distinct key.
    let (mut pager, cat) = db_with_table("CREATE TABLE t1(g INTEGER)");
    let r1 = table_root(&cat, "t1");
    seed(
        &mut pager,
        r1,
        &[(1, vec![Value::Integer(4)]), (2, vec![Value::Integer(4)]), (3, vec![Value::Integer(8)])],
    );
    // 2-column row (literal-tag, outer.g); reading column 1 picks the outer key, which VARIES
    // per correlation key. Deterministic (reads only the outer), so cache-on == off holds.
    let subplan = project(PlanNode::SingleRow, vec![lit(Value::Text("tag".into())), col(0)]);
    let plan = select_scalar_over(
        "t1",
        1,
        EvalExpr::ScalarSubqueryColumn { id: 0, col: 1 },
        sub_corr(subplan, vec![0], true),
    );
    let (on, len) = run_memo_on_off(&plan, &cat, &mut pager);
    let expected =
        vec![vec![Value::Integer(4)], vec![Value::Integer(4)], vec![Value::Integer(8)]];
    assert_eq!(
        rows_dbg(&on),
        rows_dbg(&expected),
        "column 1 (the outer key) of the ('tag', g) row; a wrong-key serve would change row 3"
    );
    assert_eq!(len, 2, "two distinct keys: 4, 8");
}

#[test]
fn correlated_scalar_column_memo_collapses_runs_for_shared_key() {
    // Genuine-HIT witness for the ScalarSubqueryColumn / FirstRow shape (the analogue of
    // `correlated_scalar_memo_collapses_runs_for_shared_key`): with a DETERMINISTIC subplan,
    // `len == 2` alone cannot tell a hit from a re-run (a re-run updates the SAME key, still
    // leaving 2 entries). So the subplan carries a volatile draw in column 1 (marked
    // deterministic so the memo engages), and same-key rows must read the SAME memoized draw
    // — only possible if row 2 was SERVED from the cached FirstRow, not re-run. Over
    // g = [4,4,8] the memo materializes the FirstRow twice (once per key), column 1 reads
    // [d1, d1, d2], the RNG advances exactly twice, and the memo holds two entries. A per-row
    // re-run would draw three times and give three distinct values.
    let (mut pager, cat) = db_with_table("CREATE TABLE t1(g INTEGER)");
    let r1 = table_root(&cat, "t1");
    seed(
        &mut pager,
        r1,
        &[(1, vec![Value::Integer(4)]), (2, vec![Value::Integer(4)]), (3, vec![Value::Integer(8)])],
    );
    // Column 0 is the outer key (keeps the FirstRow keyed), column 1 a fresh draw; we read
    // column 1 so a genuine hit shows up as a repeated draw within a key.
    let subplan = project(PlanNode::SingleRow, vec![col(0), rng_func()]);
    let plan = select_scalar_over(
        "t1",
        1,
        EvalExpr::ScalarSubqueryColumn { id: 0, col: 1 },
        sub_corr(subplan, vec![0], true),
    );
    let mut rt = Runtime::new();
    let vals: Vec<i64> =
        run_rt(&plan, &cat, &mut pager, &mut rt).iter().map(|r| int(&r[0])).collect();
    let mut reference = Runtime::new();
    let (d1, d2) = (reference.random_i64(), reference.random_i64());
    assert_eq!(vals, vec![d1, d1, d2], "same-key rows read the key's single memoized FirstRow draw");
    assert_ne!(vals[0], vals[2], "the differing key materialized its own FirstRow");
    assert_eq!(rt.correlated_cache_len(), 2, "two distinct keys: one FirstRow each");
    assert_eq!(rt.random_i64(), reference.random_i64(), "the subplan ran once per key (two draws)");
}

#[test]
fn correlated_exists_memo_collapses_runs_for_shared_key() {
    // EXISTS shape: the correlated subplan runs (draws once) per MISS. Over g = [1,1,2] the
    // memo runs it twice (once per key), the second key-1 row is a HIT that does not draw,
    // and all three EXISTS results are true. cache-off would draw three times.
    let (mut pager, cat) = db_with_table("CREATE TABLE t1(g INTEGER)");
    let r1 = table_root(&cat, "t1");
    seed(
        &mut pager,
        r1,
        &[(1, vec![Value::Integer(1)]), (2, vec![Value::Integer(1)]), (3, vec![Value::Integer(2)])],
    );
    let plan = || {
        select_scalar_over(
            "t1",
            1,
            EvalExpr::Exists { negated: false, id: 0 },
            sub_corr(project(PlanNode::SingleRow, vec![rng_func()]), vec![0], true),
        )
    };
    let mut rt = Runtime::new();
    let on = plan();
    let got: Vec<i64> = run_rt(&on, &cat, &mut pager, &mut rt).iter().map(|r| int(&r[0])).collect();
    assert_eq!(got, vec![1, 1, 1], "EXISTS over a one-row subplan is true on every row");
    assert_eq!(rt.correlated_cache_len(), 2, "two distinct keys memoized");
    let mut reference = Runtime::new();
    let _ = (reference.random_i64(), reference.random_i64());
    assert_eq!(rt.random_i64(), reference.random_i64(), "EXISTS subplan ran once per key (two draws)");

    let mut rt_off = Runtime::new();
    rt_off.set_correlated_cache_disabled(true);
    let off = plan();
    let _ = run_rt(&off, &cat, &mut pager, &mut rt_off);
    let mut ref_off = Runtime::new();
    let _ = (ref_off.random_i64(), ref_off.random_i64(), ref_off.random_i64());
    assert_eq!(rt_off.random_i64(), ref_off.random_i64(), "memo off: EXISTS drew once per outer row (three)");
}

#[test]
fn correlated_in_memo_shares_set_but_probes_fresh() {
    // Scalar `IN`: the candidate SET depends only on the correlation key, so same-key rows
    // reuse ONE materialization; the PROBE is this row's subject, applied fresh per row. The
    // candidate is `SELECT outer.g` (set = {g}); over (probe, g) = (1,5),(5,5),(9,5) — all
    // key 5 — the results are [false, true, false] (the probe varies within the one key),
    // cache-on == off, and the memo holds exactly one entry.
    let (mut pager, cat) = db_with_table("CREATE TABLE t2(p INTEGER, g INTEGER)");
    let r = table_root(&cat, "t2");
    seed(
        &mut pager,
        r,
        &[
            (1, vec![Value::Integer(1), Value::Integer(5)]),
            (2, vec![Value::Integer(5), Value::Integer(5)]),
            (3, vec![Value::Integer(9), Value::Integer(5)]),
        ],
    );
    // candidate `SELECT g` = SingleRow projecting outer column 1; correlated on [1].
    let candidate = project(PlanNode::SingleRow, vec![col(1)]);
    let plan = plan_subs(
        project(
            seqscan("t2", 2),
            vec![EvalExpr::InSubquery { negated: false, subject: Box::new(col(0)), id: 0, meta: binary_meta() }],
        ),
        &["m"],
        vec![sub_corr(candidate, vec![1], true)],
    );
    let (on, len) = run_memo_on_off(&plan, &cat, &mut pager);
    let got: Vec<i64> = on.iter().map(|r| int(&r[0])).collect();
    assert_eq!(got, vec![0, 1, 0], "shared set of just 5, fresh probe per row: 1 absent, 5 present, 9 absent");
    assert_eq!(len, 1, "one correlation key (g=5): the set is materialized once");
}

#[test]
fn correlated_in_memo_materializes_set_once_per_key() {
    // The build-once witness for the `IN` set memo: a volatile candidate `SELECT rng()`
    // marked deterministic, correlated on g. Over g = [7,7,9] the set is materialized ONCE
    // per key (two draws, not three), and the memo holds two entries. The IN result is
    // unpredictable (probe vs a random candidate), so we assert only the draw count + len —
    // the perf property, not the boolean.
    let (mut pager, cat) = db_with_table("CREATE TABLE t2(p INTEGER, g INTEGER)");
    let r = table_root(&cat, "t2");
    seed(
        &mut pager,
        r,
        &[
            (1, vec![Value::Integer(0), Value::Integer(7)]),
            (2, vec![Value::Integer(0), Value::Integer(7)]),
            (3, vec![Value::Integer(0), Value::Integer(9)]),
        ],
    );
    let candidate = project(PlanNode::SingleRow, vec![rng_func()]);
    let plan = plan_subs(
        project(
            seqscan("t2", 2),
            vec![EvalExpr::InSubquery { negated: false, subject: Box::new(col(0)), id: 0, meta: binary_meta() }],
        ),
        &["m"],
        vec![sub_corr(candidate, vec![1], true)],
    );
    let mut rt = Runtime::new();
    let _ = run_rt(&plan, &cat, &mut pager, &mut rt);
    assert_eq!(rt.correlated_cache_len(), 2, "two keys (7, 9): each set materialized once");
    let mut reference = Runtime::new();
    let _ = (reference.random_i64(), reference.random_i64());
    assert_eq!(rt.random_i64(), reference.random_i64(), "the candidate set drew once per key, not per row");
}

#[test]
fn correlated_in_row_memo_shares_rows_but_probes_fresh() {
    // Row-value `IN`: same "shared candidate ROWS, fresh probe" property as the scalar `IN`.
    // Candidate `SELECT g, 5` (rows = {(g,5)}), correlated on [1]; over (p, g) =
    // (5,5),(9,5),(5,5) — all key 5 — the tuple `(p, 5) IN {(5,5)}` is [true, false, true],
    // cache-on == off, one memo entry.
    let (mut pager, cat) = db_with_table("CREATE TABLE t2(p INTEGER, g INTEGER)");
    let r = table_root(&cat, "t2");
    seed(
        &mut pager,
        r,
        &[
            (1, vec![Value::Integer(5), Value::Integer(5)]),
            (2, vec![Value::Integer(9), Value::Integer(5)]),
            (3, vec![Value::Integer(5), Value::Integer(5)]),
        ],
    );
    // candidate rows {(g, 5)}: SingleRow projecting (outer col 1, literal 5).
    let candidate = project(PlanNode::SingleRow, vec![col(1), lit(Value::Integer(5))]);
    let plan = plan_subs(
        project(
            seqscan("t2", 2),
            vec![EvalExpr::InSubqueryRow {
                negated: false,
                subjects: vec![col(0), lit(Value::Integer(5))],
                id: 0,
                metas: vec![binary_meta(), binary_meta()],
            }],
        ),
        &["m"],
        vec![sub_corr(candidate, vec![1], true)],
    );
    let (on, len) = run_memo_on_off(&plan, &cat, &mut pager);
    let got: Vec<i64> = on.iter().map(|r| int(&r[0])).collect();
    assert_eq!(got, vec![1, 0, 1], "shared candidate row (5,5), fresh tuple probe per row");
    assert_eq!(len, 1, "one correlation key (g=5): the rows are materialized once");
}

#[test]
fn correlated_volatile_subquery_not_memoized() {
    // GUARD (b): a correlated subplan that is NOT deterministic must re-run per outer row,
    // NEVER be served from the memo. Marked `deterministic:false`, the rng subplan over
    // g = [1,1,2] draws THREE times (one per row, distinct values) and caches NOTHING —
    // even though two rows share a key. This is the volatile-staleness gate; without it the
    // two key-1 rows would wrongly repeat one draw.
    let (mut pager, cat) = db_with_table("CREATE TABLE t1(g INTEGER)");
    let r1 = table_root(&cat, "t1");
    seed(
        &mut pager,
        r1,
        &[(1, vec![Value::Integer(1)]), (2, vec![Value::Integer(1)]), (3, vec![Value::Integer(2)])],
    );
    let mut rt = Runtime::new();
    let plan = select_scalar_over(
        "t1",
        1,
        EvalExpr::ScalarSubquery(0),
        sub_corr(project(PlanNode::SingleRow, vec![rng_func()]), vec![0], false),
    );
    let vals: Vec<i64> = run_rt(&plan, &cat, &mut pager, &mut rt).iter().map(|r| int(&r[0])).collect();
    let mut reference = Runtime::new();
    let expected: Vec<i64> = (0..3).map(|_| reference.random_i64()).collect();
    assert_eq!(vals, expected, "volatile correlated subquery re-runs per outer row (three distinct draws)");
    assert_eq!(rt.correlated_cache_len(), 0, "a volatile subplan is never memoized");
}

#[test]
fn correlated_memo_disabled_when_statement_mutates() {
    // GUARD (a): a MUTATING statement never memoizes a correlated subquery, because the
    // subquery can read the table the statement writes and legitimately yield a different
    // result for the same key as rows change. Here a DELETE ... RETURNING carries a
    // correlated (deterministic) rng subplan over two deleted rows that SHARE a key: with
    // the guard the memo is off, so it draws TWICE (not once) and caches nothing. A missing
    // guard (a) would collapse them to one draw — the DML-staleness bug this pins.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER)");
    let root = table_root(&cat, "t");
    let mut rt = Runtime::new();
    seed(&mut pager, root, &[(1, vec![Value::Integer(10)]), (2, vec![Value::Integer(10)])]);
    let plan = delete_returning(
        "t",
        EvalExpr::ScalarSubquery(0),
        vec![sub_corr(project(PlanNode::SingleRow, vec![rng_func()]), vec![0], true)],
    );
    let out = run_dml(&plan, &cat, &mut pager, &mut rt);
    assert_eq!(out.len(), 2, "two rows deleted -> two RETURNING rows");
    assert_ne!(
        int(&out[0][0]),
        int(&out[1][0]),
        "mutating statement: subquery re-ran per row (two distinct draws), NOT memoized by key"
    );
    assert_eq!(rt.correlated_cache_len(), 0, "a mutating statement caches no correlated result");
}
