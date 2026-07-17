//! Integration tests for the RANKING window functions (`row_number`, `rank`,
//! `dense_rank`, `percent_rank`, `cume_dist`, `ntile`), driven through the real
//! [`Executor`]/[`RowCursor`] seam over a [`PlanNode::Values`] literal input (no
//! table/pager seeding — a window operator reads only its input rows).
//!
//! Self-contained on purpose (the aggregate/frame coverage lives in `tests/window.rs`).
//! The centerpiece is the spec's `windowfunctions.html` §3 worked oracle, transcribed
//! from real SQLite: input `a = ['a','a','a','b','c','c']` under `WINDOW (ORDER BY a)`.
//! The input rows are supplied in the SAME order as the `ORDER BY`, so the window
//! output (emitted in INPUT order) lines up position-for-position with the oracle.

use minisqlite_catalog::SchemaCatalog;
use minisqlite_exec::{Executor, Runtime, StreamingExecutor};
use minisqlite_expr::{EvalExpr, SortKey};
use minisqlite_pager::MemPager;
use minisqlite_plan::{Plan, PlanNode, Window, WindowFrame, WindowFunc, WindowFuncKind};
use minisqlite_types::{Collation, Error, Result, Value};
use minisqlite_exec::PagerSet;
use minisqlite_types::DbIndex;

// ----- value / plan helpers (mirroring tests/window.rs) ---------------------

fn vint(n: i64) -> Value {
    Value::Integer(n)
}
fn vtext(s: &str) -> Value {
    Value::Text(s.to_string())
}
fn col(i: usize) -> EvalExpr {
    EvalExpr::Column(i)
}
fn lit(v: Value) -> EvalExpr {
    EvalExpr::Literal(v)
}

/// A `VALUES` leaf from literal rows (each inner slice is one row of values).
fn values_node(rows: &[Vec<Value>]) -> PlanNode {
    PlanNode::Values {
        rows: rows.iter().map(|r| r.iter().cloned().map(lit).collect()).collect(),
    }
}

fn asc(col_idx: usize) -> SortKey {
    SortKey { expr: col(col_idx), desc: false, nulls_first: None, collation: Collation::Binary }
}

fn default_frame() -> WindowFrame {
    WindowFrame::default_frame()
}

fn wfunc(
    kind: WindowFuncKind,
    partition_by: Vec<EvalExpr>,
    order_by: Vec<SortKey>,
    frame: WindowFrame,
) -> WindowFunc {
    let partition_collations = vec![Collation::Binary; partition_by.len()];
    WindowFunc { kind, partition_by, partition_collations, order_by, frame }
}

fn window_plan(input: PlanNode, functions: Vec<WindowFunc>) -> Plan {
    Plan {
        root: PlanNode::Window(Window { input: Box::new(input), functions }),
        result_columns: Vec::new(),
        ctes: Vec::new(),
        subqueries: Vec::new(),
        mutates: false,
        generated: Vec::new(),
    }
}

/// Run a window plan over a fresh empty in-memory database (Values needs no storage).
fn run_result(plan: &Plan) -> Result<Vec<Vec<Value>>> {
    let mut pager = MemPager::new(4096);
    let cat = SchemaCatalog::new();
    let mut rt = Runtime::new();
    let mut exec = StreamingExecutor;
    let mut cur = exec.execute(plan, &cat, PagerSet::One { db: DbIndex::MAIN, pager: &mut pager })?;
    let mut out = Vec::new();
    while let Some(row) = cur.next_row(&mut rt)? {
        out.push(row);
    }
    Ok(out)
}

fn run(plan: &Plan) -> Vec<Vec<Value>> {
    run_result(plan).expect("window plan runs without error")
}

/// Run a plan expected to FAIL, returning the SQL error message.
fn run_err(plan: &Plan) -> String {
    match run_result(plan) {
        Ok(_) => panic!("expected a window error"),
        Err(Error::Sql(m)) => m,
        Err(other) => panic!("expected a SQL error, got {other:?}"),
    }
}

/// The appended (last) column of every output row as `i64`s — the value of a single
/// ranking function over a single-column input (output width = input(1) + function(1)).
fn appended_ints(input: PlanNode, kind: WindowFuncKind, order_by: Vec<SortKey>) -> Vec<i64> {
    let f = wfunc(kind, vec![], order_by, default_frame());
    run(&window_plan(input, vec![f]))
        .iter()
        .map(|r| match r.last().expect("row has an appended column") {
            Value::Integer(i) => *i,
            other => panic!("expected an Integer ranking value, got {other:?}"),
        })
        .collect()
}

/// The appended (last) column of every output row as `f64`s (for percent_rank/cume_dist).
fn appended_reals(input: PlanNode, kind: WindowFuncKind, order_by: Vec<SortKey>) -> Vec<f64> {
    let f = wfunc(kind, vec![], order_by, default_frame());
    run(&window_plan(input, vec![f]))
        .iter()
        .map(|r| match r.last().expect("row has an appended column") {
            Value::Real(r) => *r,
            other => panic!("expected a Real ranking value, got {other:?}"),
        })
        .collect()
}

/// windowfunctions.html §3 worked-example input: `a = ['a','a','a','b','c','c']`, given
/// in `ORDER BY a` order so the (input-order) output matches the oracle position-wise.
fn spec_input() -> PlanNode {
    values_node(&[
        vec![vtext("a")],
        vec![vtext("a")],
        vec![vtext("a")],
        vec![vtext("b")],
        vec![vtext("c")],
        vec![vtext("c")],
    ])
}

// ===========================================================================
// (1) The spec §3 ranking oracle over WINDOW (ORDER BY a)
// ===========================================================================

#[test]
fn row_number_is_dense_one_based_position() {
    let got = appended_ints(spec_input(), WindowFuncKind::RowNumber, vec![asc(0)]);
    assert_eq!(got, vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn rank_uses_first_peer_row_number_with_gaps() {
    let got = appended_ints(spec_input(), WindowFuncKind::Rank, vec![asc(0)]);
    assert_eq!(got, vec![1, 1, 1, 4, 5, 5]);
}

#[test]
fn dense_rank_counts_peer_groups_without_gaps() {
    let got = appended_ints(spec_input(), WindowFuncKind::DenseRank, vec![asc(0)]);
    assert_eq!(got, vec![1, 1, 1, 2, 3, 3]);
}

#[test]
fn percent_rank_matches_spec_oracle() {
    let got = appended_reals(spec_input(), WindowFuncKind::PercentRank, vec![asc(0)]);
    // (rank-1)/(n-1) with n=6: 0/5, 0/5, 0/5, 3/5, 4/5, 4/5. These decimals are the exact
    // f64 the divisions produce (IEEE division is correctly rounded), so `==` is safe.
    assert_eq!(got, vec![0.0, 0.0, 0.0, 0.6, 0.8, 0.8]);
}

#[test]
fn cume_dist_matches_spec_oracle() {
    let got = appended_reals(spec_input(), WindowFuncKind::CumeDist, vec![asc(0)]);
    // last-peer row_number / n: 3/6, 3/6, 3/6, 4/6, 6/6, 6/6.
    assert_eq!(got, vec![0.5, 0.5, 0.5, 0.6666666666666666, 1.0, 1.0]);
}

#[test]
fn ntile_two_splits_six_rows_into_even_halves() {
    let got = appended_ints(spec_input(), WindowFuncKind::Ntile(lit(vint(2))), vec![asc(0)]);
    assert_eq!(got, vec![1, 1, 1, 2, 2, 2]);
}

#[test]
fn ntile_four_gives_larger_buckets_first() {
    // 6 rows / 4 buckets: sizes 2,2,1,1 (larger first) ⇒ 1,1,2,2,3,4.
    let got = appended_ints(spec_input(), WindowFuncKind::Ntile(lit(vint(4))), vec![asc(0)]);
    assert_eq!(got, vec![1, 1, 2, 2, 3, 4]);
}

// ===========================================================================
// (2) no ORDER BY — all rows are one peer group
// ===========================================================================

#[test]
fn no_order_by_makes_rank_and_dense_rank_all_one() {
    // With no ORDER BY the whole partition is one peer group, so rank = dense_rank = 1
    // for every row, while row_number still counts 1..n in (arbitrary) input order.
    let input = values_node(&[vec![vtext("x")], vec![vtext("x")], vec![vtext("y")]]);

    let rn = appended_ints(input.clone(), WindowFuncKind::RowNumber, vec![]);
    assert_eq!(rn, vec![1, 2, 3]);

    let rank = appended_ints(input.clone(), WindowFuncKind::Rank, vec![]);
    assert_eq!(rank, vec![1, 1, 1]);

    let dense = appended_ints(input, WindowFuncKind::DenseRank, vec![]);
    assert_eq!(dense, vec![1, 1, 1]);
}

// ===========================================================================
// (3) single-row partition / PARTITION BY isolation / ntile edge cases
// ===========================================================================

#[test]
fn percent_rank_of_single_row_partition_is_zero() {
    // n == 1 ⇒ the (rank-1)/(n-1) formula's denominator is 0, defined as 0.0.
    let input = values_node(&[vec![vtext("a")]]);
    let got = appended_reals(input, WindowFuncKind::PercentRank, vec![asc(0)]);
    assert_eq!(got, vec![0.0]);
}

#[test]
fn partition_by_isolates_ranking_per_partition() {
    // rows [g, x], partitions interleaved in input order:
    //   idx0 (1,10) idx1 (2,5) idx2 (1,20) idx3 (2,7) idx4 (1,20)
    // g=1 ordered by x: 10,20,20 ⇒ row_number 1,2,3 ; rank 1,2,2 ; dense 1,2,2
    // g=2 ordered by x: 5,7      ⇒ row_number 1,2   ; rank 1,2   ; dense 1,2
    // Output is in INPUT order (idx0..idx4).
    let rows = [
        vec![vint(1), vint(10)],
        vec![vint(2), vint(5)],
        vec![vint(1), vint(20)],
        vec![vint(2), vint(7)],
        vec![vint(1), vint(20)],
    ];
    let make = || values_node(&rows);

    let per_partition = |kind: WindowFuncKind| {
        let f = wfunc(kind, vec![col(0)], vec![asc(1)], default_frame());
        run(&window_plan(make(), vec![f]))
            .iter()
            .map(|r| match r.last().unwrap() {
                Value::Integer(i) => *i,
                other => panic!("expected Integer, got {other:?}"),
            })
            .collect::<Vec<_>>()
    };

    assert_eq!(per_partition(WindowFuncKind::RowNumber), vec![1, 1, 2, 2, 3]);
    assert_eq!(per_partition(WindowFuncKind::Rank), vec![1, 1, 2, 2, 2]);
    assert_eq!(per_partition(WindowFuncKind::DenseRank), vec![1, 1, 2, 2, 2]);
}

#[test]
fn percent_rank_and_cume_dist_divide_by_the_partition_not_the_input() {
    // Two partitions of DIFFERENT sizes (g=1 has 3 rows, g=2 has 2), so each function's
    // denominator is `part.len()` (3 or 2), NOT the total row count (5). A regression that
    // divided by `all_rows.len()` would change every value here — the integer
    // partition-isolation test above cannot catch it because rank/dense_rank have no
    // partition-size denominator.
    // rows (g, x), input order idx0..idx4:
    //   idx0 (1,20) idx1 (2,15) idx2 (1,10) idx3 (2,5) idx4 (1,30)
    // g=1 ordered by x: 10(pos0),20(pos1),30(pos2) — len 3, all distinct ⇒ one peer each.
    // g=2 ordered by x: 5(pos0),15(pos1)           — len 2.
    let rows = [
        vec![vint(1), vint(20)],
        vec![vint(2), vint(15)],
        vec![vint(1), vint(10)],
        vec![vint(2), vint(5)],
        vec![vint(1), vint(30)],
    ];
    let make = || values_node(&rows);

    let reals = |kind: WindowFuncKind| {
        let f = wfunc(kind, vec![col(0)], vec![asc(1)], default_frame());
        run(&window_plan(make(), vec![f]))
            .iter()
            .map(|r| match r.last().expect("row has an appended column") {
                Value::Real(v) => *v,
                other => panic!("expected a Real ranking value, got {other:?}"),
            })
            .collect::<Vec<_>>()
    };

    // percent_rank = (rank-1)/(len-1): g1 pos0/1/2 = 0.0/0.5/1.0 ; g2 pos0/1 = 0.0/1.0.
    // (Denominator len-1 = 2 or 1 — a divide-by-4 bug would give 0.25 for g1 pos1.)
    assert_eq!(reals(WindowFuncKind::PercentRank), vec![0.5, 1.0, 0.0, 0.0, 1.0]);
    // cume_dist = (last-peer count)/len: g1 = 1/3,2/3,3/3 ; g2 = 1/2,2/2.
    // (Denominator len = 3 or 2 — a divide-by-5 bug would give 0.2 for g1 pos0.)
    assert_eq!(
        reals(WindowFuncKind::CumeDist),
        vec![0.6666666666666666, 1.0, 0.3333333333333333, 0.5, 1.0],
    );
}

#[test]
fn ntile_with_more_buckets_than_rows_gives_one_row_each() {
    // 3 rows, ntile(5): buckets 1,2,3 hold one row each; buckets 4,5 are empty.
    let input = values_node(&[vec![vtext("a")], vec![vtext("b")], vec![vtext("c")]]);
    let got = appended_ints(input, WindowFuncKind::Ntile(lit(vint(5))), vec![asc(0)]);
    assert_eq!(got, vec![1, 2, 3]);
}

#[test]
fn ntile_coerces_a_real_argument_by_truncating_toward_zero() {
    // N is coerced with the shared value→integer rule (`to_integer`), so a REAL argument
    // truncates toward zero: ntile(2.9) behaves as ntile(2) — 6 rows split into even halves.
    // Pins the coercion contract at the ntile layer (not just to_integer's own unit tests).
    let got = appended_ints(
        spec_input(),
        WindowFuncKind::Ntile(lit(Value::Real(2.9))),
        vec![asc(0)],
    );
    assert_eq!(got, vec![1, 1, 1, 2, 2, 2]);
}

#[test]
fn ntile_null_argument_is_a_loud_error() {
    // `to_integer(NULL) == 0`, which fails the `N >= 1` check as a loud SQL error — not a
    // panic and not a silent bucket 0. Pins the NULL→0→error path where the validation lives.
    let input = values_node(&[vec![vint(1)], vec![vint(2)]]);
    let f = wfunc(WindowFuncKind::Ntile(lit(Value::Null)), vec![], vec![asc(0)], default_frame());
    let msg = run_err(&window_plan(input, vec![f]));
    assert!(
        msg.contains("argument of ntile must be a positive integer"),
        "expected the ntile positive-integer error, got: {msg}"
    );
}

#[test]
fn ntile_zero_is_a_loud_error_not_a_panic() {
    let input = values_node(&[vec![vint(1)], vec![vint(2)]]);
    let f = wfunc(WindowFuncKind::Ntile(lit(vint(0))), vec![], vec![asc(0)], default_frame());
    let msg = run_err(&window_plan(input, vec![f]));
    assert!(
        msg.contains("argument of ntile must be a positive integer"),
        "expected the ntile positive-integer error, got: {msg}"
    );
}

#[test]
fn ntile_negative_is_a_loud_error() {
    let input = values_node(&[vec![vint(1)], vec![vint(2)]]);
    let f = wfunc(WindowFuncKind::Ntile(lit(vint(-3))), vec![], vec![asc(0)], default_frame());
    let msg = run_err(&window_plan(input, vec![f]));
    assert!(
        msg.contains("argument of ntile must be a positive integer"),
        "expected the ntile positive-integer error, got: {msg}"
    );
}

// ===========================================================================
// (4) input stored OUT of ORDER — ranking follows ORDER BY, output stays input-order
// ===========================================================================

#[test]
fn ranking_orders_by_key_then_emits_in_input_order() {
    // Stored out of order: x = 30, 10, 20, 10 (input indices 0..3). Sorted asc: 10,10,20,30.
    // row_number over sorted positions, mapped back to input rows:
    //   idx0 x=30 → sorted pos 3 → row_number 4
    //   idx1 x=10 → sorted pos 0 → row_number 1 (first of the 10-peers, stable)
    //   idx2 x=20 → sorted pos 2 → row_number 3
    //   idx3 x=10 → sorted pos 1 → row_number 2 (second of the 10-peers)
    let input = values_node(&[vec![vint(30)], vec![vint(10)], vec![vint(20)], vec![vint(10)]]);
    let rn = appended_ints(input, WindowFuncKind::RowNumber, vec![asc(0)]);
    assert_eq!(rn, vec![4, 1, 3, 2]);

    // rank groups the two x=10 peers: sorted 10(pos0),10(pos1),20(pos2),30(pos3) ⇒
    // ranks 1,1,3,4. Mapped to input: idx0→4, idx1→1, idx2→3, idx3→1.
    let input2 = values_node(&[vec![vint(30)], vec![vint(10)], vec![vint(20)], vec![vint(10)]]);
    let rank = appended_ints(input2, WindowFuncKind::Rank, vec![asc(0)]);
    assert_eq!(rank, vec![4, 1, 3, 1]);
}
