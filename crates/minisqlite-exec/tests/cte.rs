//! Integration tests for the CTE operators (`CteScan` / `RecursiveScan`), driven
//! through the real [`Executor`]/[`RowCursor`] seam. Inputs are hand-built
//! [`PlanNode::Values`] rows (so no table/storage is needed — the pager is never
//! touched), which lets these tests pin the OPERATORS' job — materialize-and-stream, the
//! recursive semi-naive fixpoint, `UNION ALL` vs `UNION` dedup, termination (empty
//! frontier and the hard row cap), and the malformed-plan error paths — independent of
//! the SQL front end or the planner.

use minisqlite_catalog::SchemaCatalog;
use minisqlite_exec::{Executor, Runtime, StreamingExecutor};
use minisqlite_expr::{ArithOp, CmpOp, CompareMeta, EvalExpr};
use minisqlite_pager::MemPager;
use minisqlite_plan::{CtePlan, Join, JoinStrategy, JoinType, Plan, PlanNode, SetOp, SetOpKind};
use minisqlite_types::{Collation, Value};
use minisqlite_exec::PagerSet;
use minisqlite_types::DbIndex;

// ----- harness -------------------------------------------------------------

/// A bare in-memory pager + empty catalog. The CTE tests feed rows from `Values`, so no
/// table b-tree is read; this only satisfies the `execute` signature.
fn empty_db() -> (MemPager, SchemaCatalog) {
    (MemPager::new(4096), SchemaCatalog::new())
}

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

/// Drive the plan expecting a failure somewhere on the pull path (build or a later
/// `next_row`). Returns the `Result` so a test can assert `is_err` — and, crucially,
/// that the engine does not panic.
fn run_err(plan: &Plan, cat: &SchemaCatalog, pager: &mut MemPager) -> minisqlite_types::Result<()> {
    let mut rt = Runtime::new();
    let mut exec = StreamingExecutor;
    let mut cur = exec.execute(plan, cat, PagerSet::One { db: DbIndex::MAIN, pager: &mut *pager })?;
    while cur.next_row(&mut rt)?.is_some() {}
    Ok(())
}

// ----- expr / plan builders ------------------------------------------------

fn col(i: usize) -> EvalExpr {
    EvalExpr::Column(i)
}

fn int(i: i64) -> Value {
    Value::Integer(i)
}

fn lit(i: i64) -> EvalExpr {
    EvalExpr::Literal(int(i))
}

fn as_int(v: &Value) -> i64 {
    match v {
        Value::Integer(i) => *i,
        other => panic!("expected Integer, got {other:?}"),
    }
}

/// The single-column integer contents of a result set, in row order.
fn col0_ints(rows: &[Vec<Value>]) -> Vec<i64> {
    rows.iter().map(|r| as_int(&r[0])).collect()
}

fn binary_meta() -> CompareMeta {
    CompareMeta { apply_left: None, apply_right: None, collation: Collation::Binary }
}

fn arith(op: ArithOp, left: EvalExpr, right: EvalExpr) -> EvalExpr {
    EvalExpr::Arith { op, left: Box::new(left), right: Box::new(right) }
}

/// `left < value` (three-valued), the stopping predicate of the counter recursion.
fn lt(left: EvalExpr, value: i64) -> EvalExpr {
    EvalExpr::Compare {
        op: CmpOp::Lt,
        null_safe: false,
        left: Box::new(left),
        right: Box::new(lit(value)),
        meta: binary_meta(),
    }
}

/// `left > value` (three-valued).
fn gt(left: EvalExpr, value: i64) -> EvalExpr {
    EvalExpr::Compare {
        op: CmpOp::Gt,
        null_safe: false,
        left: Box::new(left),
        right: Box::new(lit(value)),
        meta: binary_meta(),
    }
}

/// A `Values` node of single-column integer rows.
fn values_ints(vals: &[i64]) -> PlanNode {
    PlanNode::Values { rows: vals.iter().map(|&i| vec![lit(i)]).collect() }
}

fn project(input: PlanNode, exprs: Vec<EvalExpr>) -> PlanNode {
    PlanNode::Project { input: Box::new(input), exprs }
}

fn filter(input: PlanNode, predicate: EvalExpr) -> PlanNode {
    PlanNode::Filter { input: Box::new(input), predicate }
}

fn cte_scan(id: usize, column_count: usize) -> PlanNode {
    PlanNode::CteScan { id, column_count }
}

fn recursive_scan(column_count: usize) -> PlanNode {
    PlanNode::RecursiveScan { column_count }
}

fn materialized(name: &str, column_count: usize, body: PlanNode) -> CtePlan {
    CtePlan::Materialized { name: name.to_string(), column_count, body }
}

fn recursive(name: &str, column_count: usize, seed: PlanNode, step: PlanNode, union_all: bool) -> CtePlan {
    CtePlan::Recursive {
        name: name.to_string(),
        column_count,
        seed,
        step,
        union_all,
    }
}

/// A `LIMIT`/`OFFSET` node over `input` (integer-literal bounds), the outer-query shape
/// that must bound an otherwise-infinite recursive `CteScan` by stopping its pulls.
fn limit_node(input: PlanNode, limit: Option<i64>, offset: Option<i64>) -> PlanNode {
    PlanNode::Limit {
        input: Box::new(input),
        limit: limit.map(|n| EvalExpr::Literal(int(n))),
        offset: offset.map(|n| EvalExpr::Literal(int(n))),
    }
}

/// A `CROSS JOIN` run as a plain nested loop. The join rebuilds its RIGHT (inner)
/// subtree once per LEFT row, which is exactly the "RecursiveScan rebuilt within one
/// round" path PART D must keep stable. Combined row layout is `left ++ right`.
fn cross_nl_join(left: PlanNode, left_width: usize, right: PlanNode, right_width: usize) -> PlanNode {
    PlanNode::Join(Join {
        left: Box::new(left),
        left_width,
        right: Box::new(right),
        right_width,
        join_type: JoinType::Cross,
        on: None,
        strategy: JoinStrategy::NestedLoop,
    })
}

/// `left UNION ALL right` (single-column output). Lets a plan reference a CTE from two
/// scan sites, or a recursive step branch into two `RecursiveScan` reads.
fn union_all(left: PlanNode, right: PlanNode) -> PlanNode {
    PlanNode::SetOp(SetOp {
        op: SetOpKind::UnionAll,
        left: Box::new(left),
        right: Box::new(right),
        column_collations: vec![Collation::Binary],
    })
}

fn plan_with_ctes(root: PlanNode, ctes: Vec<CtePlan>) -> Plan {
    Plan { root, result_columns: Vec::new(), ctes, subqueries: Vec::new(), mutates: false, generated: Vec::new() }
}

// ----- (1) materialized CTE: run body once, stream the buffered rows -------

#[test]
fn materialized_scan_streams_body_rows() {
    // WITH c AS (VALUES (1),(2),(3)) SELECT * FROM c
    let (mut pager, cat) = empty_db();
    let ctes = vec![materialized("c", 1, values_ints(&[1, 2, 3]))];
    let rows = run(&plan_with_ctes(cte_scan(0, 1), ctes), &cat, &mut pager);

    assert!(rows.iter().all(|r| r.len() == 1), "output width is the CTE's column_count, no rowid");
    assert_eq!(col0_ints(&rows), vec![1, 2, 3]);
}

// ----- (2) materialized CTE feeding another operator (Project over Filter) --

#[test]
fn materialized_cte_feeds_project_and_filter() {
    // SELECT c.x + 100 FROM (VALUES(1),(2),(3)) c WHERE c.x > 1  ->  102, 103
    let (mut pager, cat) = empty_db();
    let ctes = vec![materialized("c", 1, values_ints(&[1, 2, 3]))];
    let root = project(filter(cte_scan(0, 1), gt(col(0), 1)), vec![arith(ArithOp::Add, col(0), lit(100))]);
    let rows = run(&plan_with_ctes(root, ctes), &cat, &mut pager);

    assert_eq!(col0_ints(&rows), vec![102, 103], "CteScan streams as an input to Filter/Project");
}

// ----- (3) recursive UNION ALL counter: terminates via the step predicate ---

#[test]
fn recursive_union_all_counter_terminates() {
    // WITH RECURSIVE c(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM c WHERE x<5)
    // SELECT x FROM c   ->  1,2,3,4,5 (and it must stop, not loop).
    let (mut pager, cat) = empty_db();
    let seed = values_ints(&[1]);
    let step = project(filter(recursive_scan(1), lt(col(0), 5)), vec![arith(ArithOp::Add, col(0), lit(1))]);
    let ctes = vec![recursive("c", 1, seed, step, true)];
    let rows = run(&plan_with_ctes(cte_scan(0, 1), ctes), &cat, &mut pager);

    assert_eq!(col0_ints(&rows), vec![1, 2, 3, 4, 5], "seed then each +1 up to 5, in order");
}

// ----- (4) recursive UNION: a revisited row is deduped, so it terminates ----

#[test]
fn recursive_union_dedups_and_terminates() {
    // WITH RECURSIVE c(x) AS (VALUES(1) UNION SELECT (x%3)+1 FROM c) SELECT x FROM c
    // Step maps 1->2->3->1; the revisit of 1 is dropped by `seen`, so the frontier
    // empties and the recursion halts with {1,2,3}. (The SAME shape under UNION ALL
    // would loop forever — see the cap test — which is exactly why dedup is required.)
    let (mut pager, cat) = empty_db();
    let seed = values_ints(&[1]);
    let step = project(recursive_scan(1), vec![arith(ArithOp::Add, arith(ArithOp::Mod, col(0), lit(3)), lit(1))]);
    let ctes = vec![recursive("c", 1, seed, step, false)];
    let rows = run(&plan_with_ctes(cte_scan(0, 1), ctes), &cat, &mut pager);

    assert_eq!(col0_ints(&rows), vec![1, 2, 3], "each new value once; the revisit of 1 is deduped");
}

// ----- (5) a bare RecursiveScan (no active frame) errors, never panics ------

#[test]
fn bare_recursive_scan_without_frame_errors() {
    // A RecursiveScan outside any recursive step has no working table to read; this is
    // a malformed plan and must surface an error rather than panic or return rows.
    let (mut pager, cat) = empty_db();
    let err = run_err(&plan_with_ctes(recursive_scan(1), Vec::new()), &cat, &mut pager);
    assert!(err.is_err(), "RecursiveScan with no active frame is an error");
}

// ----- (6) an out-of-range CteScan id errors -------------------------------

#[test]
fn cte_scan_out_of_range_id_errors() {
    // ctes is empty, so id 0 does not exist. The scan must error, not panic.
    let (mut pager, cat) = empty_db();
    let err = run_err(&plan_with_ctes(cte_scan(0, 1), Vec::new()), &cat, &mut pager);
    assert!(err.is_err(), "a CteScan whose id is out of range is an error");
}

// ----- (7) a non-terminating UNION ALL recursion trips the hard row cap -----

#[test]
fn recursive_union_all_hits_row_cap() {
    // An identity step under UNION ALL never empties the frontier: each round re-emits
    // the frontier and admits it all again, so the row count grows without bound. The
    // 1_000_000-row safety cap must turn this into a loud error (an unbounded result is
    // an incorrect result), proving the recursion is bounded / live. A wide-ish seed
    // (10_000 rows) reaches the cap in ~100 rounds so the test stays fast.
    let (mut pager, cat) = empty_db();
    let seed_vals: Vec<i64> = (0..10_000).collect();
    let seed = values_ints(&seed_vals);
    let step = project(recursive_scan(1), vec![col(0)]); // identity: re-emits the frontier
    let ctes = vec![recursive("c", 1, seed, step, true)];
    let err = run_err(&plan_with_ctes(cte_scan(0, 1), ctes), &cat, &mut pager);
    assert!(err.is_err(), "a non-terminating UNION ALL recursion must hit the row cap, not hang");
}

// ----- (8) PART D: a RecursiveScan rebuilt per outer row re-snapshots the frontier --

#[test]
fn recursive_scan_resnapshots_stable_frontier_when_rebuilt_per_outer_row() {
    // A nested-loop join rebuilds its inner (right) side once PER outer row. With
    // RecursiveScan as that inner side, each rebuild must re-snapshot the SAME frontier
    // (the PART D stability requirement). Two identical constant outer rows (9),(9) mean
    // the inner is rebuilt twice per round, so each round reads the frontier TWICE: under
    // UNION ALL the frontier value appears twice. If the second rebuild had seen an
    // empty/stale frame, that duplicate would be missing and the result would be shorter.
    let (mut pager, cat) = empty_db();
    let seed = values_ints(&[0]);
    // Join row layout is [left(=9), right(=frontier x)] -> the frontier value is col1.
    let inner = cross_nl_join(values_ints(&[9, 9]), 1, recursive_scan(1), 1);
    let step = project(filter(inner, lt(col(1), 1)), vec![arith(ArithOp::Add, col(1), lit(1))]);
    let ctes = vec![recursive("c", 1, seed, step, true)];
    let rows = run(&plan_with_ctes(cte_scan(0, 1), ctes), &cat, &mut pager);

    // seed 0 -> round1 reads {0} twice -> {1,1} -> round2 reads {1,1}, filter x<1 drops
    // all -> halt. The duplicate 1 witnesses that BOTH inner rebuilds saw frontier {0}.
    assert_eq!(
        col0_ints(&rows),
        vec![0, 1, 1],
        "both rebuilt inner RecursiveScans saw the same frontier (duplicate present)"
    );
}

// ----- (9) a CTE referenced by two scan sites materializes at each site --------------

#[test]
fn cte_referenced_twice_materializes_per_scan_site() {
    // Two CteScan sites over the same CTE (here the two arms of UNION ALL) each
    // materialize the body independently and yield the full rows. Pins the documented
    // per-scan re-materialization (the deliberately-skipped shared cache).
    let (mut pager, cat) = empty_db();
    let ctes = vec![materialized("c", 1, values_ints(&[1, 2, 3]))];
    let root = union_all(cte_scan(0, 1), cte_scan(0, 1));
    let rows = run(&plan_with_ctes(root, ctes), &cat, &mut pager);
    assert_eq!(col0_ints(&rows), vec![1, 2, 3, 1, 2, 3], "both scan sites yield the full CTE body");
}

// ----- (10) empty seed: zero rounds, no rows, no hang --------------------------------

#[test]
fn recursive_empty_seed_yields_no_rows() {
    // An empty initial-select is an empty frontier from the start: the fixpoint runs zero
    // rounds and produces nothing — and does not loop, even under UNION ALL.
    let (mut pager, cat) = empty_db();
    let step = project(recursive_scan(1), vec![arith(ArithOp::Add, col(0), lit(1))]);
    let ctes = vec![recursive("c", 1, PlanNode::Values { rows: Vec::new() }, step, true)];
    let rows = run(&plan_with_ctes(cte_scan(0, 1), ctes), &cat, &mut pager);
    assert!(rows.is_empty(), "an empty seed yields an empty recursive result");
}

// ----- (11) UNION dedups the SEED rows too -------------------------------------------

#[test]
fn recursive_union_dedups_the_seed_rows() {
    // `admit` runs the same `seen` gate on seed rows, so a duplicate seed collapses under
    // UNION. Seed (1),(1),(2) -> {1,2}; the identity step revisits 1,2 (both already
    // seen) so the frontier empties immediately and it halts with {1,2}.
    let (mut pager, cat) = empty_db();
    let seed = values_ints(&[1, 1, 2]);
    let step = project(recursive_scan(1), vec![col(0)]); // identity
    let ctes = vec![recursive("c", 1, seed, step, false)];
    let rows = run(&plan_with_ctes(cte_scan(0, 1), ctes), &cat, &mut pager);
    assert_eq!(col0_ints(&rows), vec![1, 2], "duplicate seed rows are deduped under UNION");
}

// ----- (12) branching (tree) recursion produces the full set -------------------------

#[test]
fn recursive_branching_produces_the_full_set() {
    // A branching recursion: each frontier value x spawns two children (2x, 2x+1) via two
    // RecursiveScan sites (the arms of UNION ALL in the step), bounded by x<4. Exercises
    // several RecursiveScan reads of one frontier in a round and a tree- (not linear-)
    // shaped recursion. Assert the SET: batch/breadth-first row order is unspecified by
    // SQL without ORDER BY, so pinning order would pin the mechanism, not the contract.
    let (mut pager, cat) = empty_db();
    let seed = values_ints(&[1]);
    let child = |off: i64| {
        project(
            filter(recursive_scan(1), lt(col(0), 4)),
            vec![arith(ArithOp::Add, arith(ArithOp::Mul, col(0), lit(2)), lit(off))],
        )
    };
    let step = union_all(child(0), child(1));
    let ctes = vec![recursive("c", 1, seed, step, true)];
    let mut got = col0_ints(&run(&plan_with_ctes(cte_scan(0, 1), ctes), &cat, &mut pager));
    got.sort();
    assert_eq!(got, vec![1, 2, 3, 4, 5, 6, 7], "binary-tree expansion: 1, then 2/3, then 4/5/6/7");
}

// ----- (13) an OUTER LIMIT bounds an infinite recursion via lazy streaming ------------

#[test]
fn outer_limit_bounds_infinite_recursion_via_lazy_streaming() {
    // The KEY laziness proof: an infinite +1 recursion with NO body limit, wrapped in an
    // OUTER `LIMIT 3`. Because the recursive CteScan streams (it computes a round only when
    // pulled), the Limit stops after 3 rows and the recursion never reaches the 1M cap. A
    // materializing CteScan would run to the cap and ERROR — so a clean [1,2,3] here is
    // exactly the streaming guarantee (the outer-limit idiom for bounding recursion).
    let (mut pager, cat) = empty_db();
    let seed = values_ints(&[1]);
    let step = project(recursive_scan(1), vec![arith(ArithOp::Add, col(0), lit(1))]);
    let ctes = vec![recursive("c", 1, seed, step, true)];
    let root = limit_node(cte_scan(0, 1), Some(3), None);
    let rows = run(&plan_with_ctes(root, ctes), &cat, &mut pager);
    assert_eq!(col0_ints(&rows), vec![1, 2, 3], "outer LIMIT 3 bounds the infinite recursion to 1,2,3");
}
