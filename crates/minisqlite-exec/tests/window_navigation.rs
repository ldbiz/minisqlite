//! Integration tests for the NAVIGATION window functions — `lag`/`lead` (offset) and
//! `first_value`/`last_value`/`nth_value` (value) — driven through the real
//! [`Executor`]/[`RowCursor`] seam over a [`PlanNode::Values`] input (a window operator
//! reads only its input rows, so no table/pager seeding is needed). The harness mirrors
//! `tests/window.rs`; `Value` has no `PartialEq`, so assertions project each result to a
//! small comparable [`Cell`].
//!
//! The primary oracles are the worked examples of windowfunctions.html §3 (`t1.b` =
//! A,B,C,D,E,F,G under `WINDOW (ORDER BY b ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT
//! ROW)`); the remaining tests pin the frame-ignored / frame-respected split, partition
//! isolation, edge/default behavior, and the error cases.

use minisqlite_catalog::SchemaCatalog;
use minisqlite_exec::{Executor, Runtime, StreamingExecutor};
use minisqlite_expr::{EvalExpr, SortKey};
use minisqlite_pager::MemPager;
use minisqlite_plan::{
    FrameBound, FrameExclude, FrameUnits, Plan, PlanNode, Window, WindowFrame, WindowFunc,
    WindowFuncKind,
};
use minisqlite_types::{Collation, Error, Result, Value};
use minisqlite_exec::PagerSet;
use minisqlite_types::DbIndex;

// ----- value / plan helpers (mirror tests/window.rs) ------------------------

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

fn values_node(rows: &[Vec<Value>]) -> PlanNode {
    PlanNode::Values { rows: rows.iter().map(|r| r.iter().cloned().map(lit).collect()).collect() }
}

fn asc(col_idx: usize) -> SortKey {
    SortKey { expr: col(col_idx), desc: false, nulls_first: None, collation: Collation::Binary }
}

fn default_frame() -> WindowFrame {
    WindowFrame::default_frame()
}
fn frame(units: FrameUnits, start: FrameBound, end: FrameBound, exclude: FrameExclude) -> WindowFrame {
    WindowFrame { units, start, end, exclude }
}
fn unbounded_preceding() -> FrameBound {
    FrameBound::UnboundedPreceding
}
fn current_row() -> FrameBound {
    FrameBound::CurrentRow
}
fn preceding(n: i64) -> FrameBound {
    FrameBound::Preceding(lit(vint(n)))
}
fn following(n: i64) -> FrameBound {
    FrameBound::Following(lit(vint(n)))
}

// navigation kind constructors
fn lag(expr: EvalExpr, offset: EvalExpr, default: EvalExpr) -> WindowFuncKind {
    WindowFuncKind::Lag { expr, offset, default }
}
fn lead(expr: EvalExpr, offset: EvalExpr, default: EvalExpr) -> WindowFuncKind {
    WindowFuncKind::Lead { expr, offset, default }
}
fn first_value(expr: EvalExpr) -> WindowFuncKind {
    WindowFuncKind::FirstValue(expr)
}
fn last_value(expr: EvalExpr) -> WindowFuncKind {
    WindowFuncKind::LastValue(expr)
}
fn nth_value(expr: EvalExpr, n: EvalExpr) -> WindowFuncKind {
    WindowFuncKind::NthValue { expr, n }
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

// ----- comparable projection (Value has no PartialEq) -----------------------

/// The subset of storage classes these tests produce, projected to a comparable form.
#[derive(Debug, Clone, PartialEq)]
enum Cell {
    Null,
    Int(i64),
    Text(String),
}

fn cell(v: &Value) -> Cell {
    match v {
        Value::Null => Cell::Null,
        Value::Integer(i) => Cell::Int(*i),
        Value::Text(s) => Cell::Text(s.clone()),
        other => panic!("unexpected navigation result value: {other:?}"),
    }
}

/// Column `j` of every output row, projected to [`Cell`] (output is in input order).
fn col_cells(rows: &[Vec<Value>], j: usize) -> Vec<Cell> {
    rows.iter().map(|r| cell(&r[j])).collect()
}

fn t(s: &str) -> Cell {
    Cell::Text(s.to_string())
}
fn n(i: i64) -> Cell {
    Cell::Int(i)
}

/// The 7 single-column rows `b` = A..G, already in sorted (== input) order.
fn b_rows() -> Vec<Vec<Value>> {
    ["A", "B", "C", "D", "E", "F", "G"].iter().map(|s| vec![vtext(s)]).collect()
}

/// The spec's window: `ORDER BY b ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`.
fn spec_frame() -> WindowFrame {
    frame(FrameUnits::Rows, unbounded_preceding(), current_row(), FrameExclude::NoOthers)
}

// ===========================================================================
// (1) the spec §3 worked oracles
// ===========================================================================

#[test]
fn lag_and_lead_match_spec_oracle() {
    // b=A..G: lag(b) = NULL,A,B,C,D,E,F ; lead(b,2,'n/a') = C,D,E,F,G,'n/a','n/a'.
    let f_lead = wfunc(lead(col(0), lit(vint(2)), lit(vtext("n/a"))), vec![], vec![asc(0)], spec_frame());
    let f_lag = wfunc(lag(col(0), lit(vint(1)), lit(Value::Null)), vec![], vec![asc(0)], spec_frame());
    let rows = run(&window_plan(values_node(&b_rows()), vec![f_lead, f_lag]));
    // columns: [b, lead, lag]
    assert_eq!(col_cells(&rows, 1), vec![t("C"), t("D"), t("E"), t("F"), t("G"), t("n/a"), t("n/a")]);
    assert_eq!(col_cells(&rows, 2), vec![Cell::Null, t("A"), t("B"), t("C"), t("D"), t("E"), t("F")]);
}

#[test]
fn first_last_nth_value_match_spec_oracle() {
    // Over ROWS UNBOUNDED PRECEDING .. CURRENT ROW: first=A*7 ; last=A..G ;
    // nth_value(b,3)=NULL,NULL,C,C,C,C,C (frame-relative, 1-based N).
    let f_first = wfunc(first_value(col(0)), vec![], vec![asc(0)], spec_frame());
    let f_last = wfunc(last_value(col(0)), vec![], vec![asc(0)], spec_frame());
    let f_nth3 = wfunc(nth_value(col(0), lit(vint(3))), vec![], vec![asc(0)], spec_frame());
    let rows = run(&window_plan(values_node(&b_rows()), vec![f_first, f_last, f_nth3]));
    // columns: [b, first, last, nth3]
    assert_eq!(col_cells(&rows, 1), vec![t("A"); 7]);
    assert_eq!(col_cells(&rows, 2), vec![t("A"), t("B"), t("C"), t("D"), t("E"), t("F"), t("G")]);
    assert_eq!(col_cells(&rows, 3), vec![Cell::Null, Cell::Null, t("C"), t("C"), t("C"), t("C"), t("C")]);
}

// ===========================================================================
// (2) lag/lead offset / default / edge behavior
// ===========================================================================

#[test]
fn lag_lead_offset_zero_is_current_row() {
    // offset 0 ⇒ target == pos ⇒ the current row (spec §3), for both directions.
    let input = values_node(&[vec![vint(10)], vec![vint(20)], vec![vint(30)]]);
    let f_lag0 = wfunc(lag(col(0), lit(vint(0)), lit(Value::Null)), vec![], vec![asc(0)], default_frame());
    let f_lead0 = wfunc(lead(col(0), lit(vint(0)), lit(Value::Null)), vec![], vec![asc(0)], default_frame());
    let rows = run(&window_plan(input, vec![f_lag0, f_lead0]));
    assert_eq!(col_cells(&rows, 1), vec![n(10), n(20), n(30)]);
    assert_eq!(col_cells(&rows, 2), vec![n(10), n(20), n(30)]);
}

#[test]
fn lag_lead_past_edge_uses_default() {
    let input = values_node(&[vec![vint(1)], vec![vint(2)], vec![vint(3)]]);
    // lag offset 1, default NULL: first row has no prior ⇒ NULL, then 1,2.
    let f_lag = wfunc(lag(col(0), lit(vint(1)), lit(Value::Null)), vec![], vec![asc(0)], default_frame());
    // lead offset 1, default NULL: 2,3 then last row has no next ⇒ NULL.
    let f_lead = wfunc(lead(col(0), lit(vint(1)), lit(Value::Null)), vec![], vec![asc(0)], default_frame());
    // lag offset 5, explicit default -1: always out of range ⇒ the default everywhere.
    let f_lag5 = wfunc(lag(col(0), lit(vint(5)), lit(vint(-1))), vec![], vec![asc(0)], default_frame());
    let rows = run(&window_plan(input, vec![f_lag, f_lead, f_lag5]));
    assert_eq!(col_cells(&rows, 1), vec![Cell::Null, n(1), n(2)]);
    assert_eq!(col_cells(&rows, 2), vec![n(2), n(3), Cell::Null]);
    assert_eq!(col_cells(&rows, 3), vec![n(-1), n(-1), n(-1)]);
}

// ---- overflow safety: a huge but VALID offset targets a row far outside the partition, so
// the result is the function `default` (NULL) — never a panic. These pin the arithmetic in
// `lag_lead`: without the saturating add the `lead` side overflows `pos + i64::MAX` and panics
// under the (default) overflow-checked test profile; the `lag` subtract is safe by construction
// (min `0 - i64::MAX = -i64::MAX > i64::MIN`) and is covered symmetrically so the deliberate
// asymmetry stays documented. (Mirrors frame.rs's overflow-safety tests.)

#[test]
fn lead_huge_offset_must_not_panic() {
    // lead(x, i64::MAX) over x=1,2: the target (pos + i64::MAX) is past the partition end for
    // every row, so real sqlite returns the default (NULL). Without saturating the add,
    // `1i64 + i64::MAX` overflows and panics under the (default) overflow-checked test profile.
    let f = wfunc(
        lead(col(0), lit(vint(i64::MAX)), lit(Value::Null)),
        vec![],
        vec![asc(0)],
        default_frame(),
    );
    let rows = run(&window_plan(values_node(&[vec![vint(1)], vec![vint(2)]]), vec![f]));
    assert_eq!(col_cells(&rows, 1), vec![Cell::Null, Cell::Null]);
}

#[test]
fn lag_huge_offset_returns_default() {
    // Symmetric companion to `lead_huge_offset_must_not_panic`: lag(x, i64::MAX) over x=1,2
    // targets `pos - i64::MAX`, far before the partition start for every row, so the result is
    // the default (NULL). The `lag` subtract is left as a plain `-` because it cannot underflow
    // (min `0 - i64::MAX = -i64::MAX > i64::MIN`); this pins that it was deliberately assessed.
    let f = wfunc(
        lag(col(0), lit(vint(i64::MAX)), lit(Value::Null)),
        vec![],
        vec![asc(0)],
        default_frame(),
    );
    let rows = run(&window_plan(values_node(&[vec![vint(1)], vec![vint(2)]]), vec![f]));
    assert_eq!(col_cells(&rows, 1), vec![Cell::Null, Cell::Null]);
}

#[test]
fn lag_default_is_evaluated_against_current_row() {
    // lag(x, 5, y): offset 5 is always out of range for 3 rows, so every result is the
    // `default` expression `y` (col 1) evaluated against the CURRENT row.
    let input = values_node(&[
        vec![vint(1), vint(100)],
        vec![vint(2), vint(200)],
        vec![vint(3), vint(300)],
    ]);
    let f = wfunc(lag(col(0), lit(vint(5)), col(1)), vec![], vec![asc(0)], default_frame());
    let rows = run(&window_plan(input, vec![f]));
    // columns: [x, y, lag]
    assert_eq!(col_cells(&rows, 2), vec![n(100), n(200), n(300)]);
}

#[test]
fn lag_follows_order_by_not_input_order() {
    // Stored out of order (x = 3,1,2): lag is by SORTED order, output re-emitted in input
    // order. Sorted 1,2,3 ⇒ lag NULL,1,2 at those positions; input rows (3),(1),(2) map
    // to lag 2, NULL, 1 respectively.
    let input = values_node(&[vec![vint(3)], vec![vint(1)], vec![vint(2)]]);
    let f = wfunc(lag(col(0), lit(vint(1)), lit(Value::Null)), vec![], vec![asc(0)], default_frame());
    let rows = run(&window_plan(input, vec![f]));
    assert_eq!(col_cells(&rows, 1), vec![n(2), Cell::Null, n(1)]);
}

#[test]
fn lag_lead_ignore_frame() {
    // lag/lead ignore the frame-spec (spec §3): three very different frames, one result.
    let data: Vec<Vec<Value>> = vec![vec![vint(10)], vec![vint(20)], vec![vint(30)]];
    let run_with = |fr: WindowFrame| {
        let f = wfunc(lag(col(0), lit(vint(1)), lit(Value::Null)), vec![], vec![asc(0)], fr);
        run(&window_plan(values_node(&data), vec![f]))
    };
    let a = run_with(default_frame());
    let b = run_with(frame(FrameUnits::Rows, preceding(1), following(1), FrameExclude::NoOthers));
    let c = run_with(frame(FrameUnits::Rows, current_row(), current_row(), FrameExclude::NoOthers));
    let expected = vec![Cell::Null, n(10), n(20)];
    assert_eq!(col_cells(&a, 1), expected);
    assert_eq!(col_cells(&b, 1), expected);
    assert_eq!(col_cells(&c, 1), expected);
}

// ===========================================================================
// (3) partitioning
// ===========================================================================

#[test]
fn partition_by_isolates_navigation() {
    // (g, x): g=1 ⇒ x 10,20 ; g=2 ⇒ x 100,200, interleaved in input order. Navigation
    // must stay within a partition.
    let input = values_node(&[
        vec![vint(1), vint(10)],
        vec![vint(2), vint(100)],
        vec![vint(1), vint(20)],
        vec![vint(2), vint(200)],
    ]);
    let f_lag = wfunc(lag(col(1), lit(vint(1)), lit(Value::Null)), vec![col(0)], vec![asc(1)], default_frame());
    let f_first = wfunc(first_value(col(1)), vec![col(0)], vec![asc(1)], default_frame());
    let rows = run(&window_plan(input, vec![f_lag, f_first]));
    // columns: [g, x, lag, first]; input order rows: (1,10),(2,100),(1,20),(2,200).
    assert_eq!(col_cells(&rows, 2), vec![Cell::Null, Cell::Null, n(10), n(100)]);
    assert_eq!(col_cells(&rows, 3), vec![n(10), n(100), n(10), n(100)]);
}

// ===========================================================================
// (4) value functions RESPECT the frame
// ===========================================================================

#[test]
fn nth_value_beyond_frame_is_null() {
    let input = values_node(&[vec![vint(10)], vec![vint(20)], vec![vint(30)]]);
    // N=10 never fits the growing frame ⇒ NULL; N=1 is always the first frame row.
    let f_nth10 = wfunc(nth_value(col(0), lit(vint(10))), vec![], vec![asc(0)], spec_frame());
    let f_nth1 = wfunc(nth_value(col(0), lit(vint(1))), vec![], vec![asc(0)], spec_frame());
    let rows = run(&window_plan(input, vec![f_nth10, f_nth1]));
    assert_eq!(col_cells(&rows, 1), vec![Cell::Null, Cell::Null, Cell::Null]);
    assert_eq!(col_cells(&rows, 2), vec![n(10), n(10), n(10)]);
}

#[test]
fn value_fns_respect_explicit_sliding_frame() {
    // x=1..5 over ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING. The frame slides, so
    // first/last/nth track the window edges rather than the whole partition.
    let input = values_node(&[vec![vint(1)], vec![vint(2)], vec![vint(3)], vec![vint(4)], vec![vint(5)]]);
    let fr = frame(FrameUnits::Rows, preceding(1), following(1), FrameExclude::NoOthers);
    let f_first = wfunc(first_value(col(0)), vec![], vec![asc(0)], fr.clone());
    let f_last = wfunc(last_value(col(0)), vec![], vec![asc(0)], fr.clone());
    let f_nth2 = wfunc(nth_value(col(0), lit(vint(2))), vec![], vec![asc(0)], fr);
    let rows = run(&window_plan(input, vec![f_first, f_last, f_nth2]));
    // frames: c0[0,1] c1[0,2] c2[1,3] c3[2,4] c4[3,4]
    assert_eq!(col_cells(&rows, 1), vec![n(1), n(1), n(2), n(3), n(4)]);
    assert_eq!(col_cells(&rows, 2), vec![n(2), n(3), n(4), n(5), n(5)]);
    assert_eq!(col_cells(&rows, 3), vec![n(2), n(2), n(3), n(4), n(5)]);
}

// ===========================================================================
// (5) error cases — every bad argument is an Error, never a panic
// ===========================================================================

#[test]
fn negative_offset_is_error() {
    for kind in [
        lag(col(0), lit(vint(-1)), lit(Value::Null)),
        lead(col(0), lit(vint(-2)), lit(Value::Null)),
    ] {
        let input = values_node(&[vec![vint(1)], vec![vint(2)]]);
        let f = wfunc(kind, vec![], vec![asc(0)], default_frame());
        let msg = run_err(&window_plan(input, vec![f]));
        assert!(msg.contains("must be a non-negative integer"), "got: {msg}");
    }
}

#[test]
fn nth_value_non_positive_is_error() {
    for bad in [0i64, -1] {
        let input = values_node(&[vec![vint(1)], vec![vint(2)]]);
        let f = wfunc(nth_value(col(0), lit(vint(bad))), vec![], vec![asc(0)], spec_frame());
        let msg = run_err(&window_plan(input, vec![f]));
        assert!(msg.contains("argument of nth_value must be a positive integer"), "N={bad}, got: {msg}");
    }
}

// ===========================================================================
// (6) value functions over empty and peer-extended frames (the O(1) first/last path,
//     exercised end-to-end through the executor rather than only the frame unit test)
// ===========================================================================

#[test]
fn value_fns_over_default_range_frame_track_peers() {
    // Default frame = RANGE UNBOUNDED PRECEDING .. CURRENT ROW: for RANGE, CURRENT ROW
    // extends to the current row's LAST peer, so the frame runs from the partition start to
    // the end of the current peer group. first_value is thus the partition's first row and
    // last_value is the current peer group's last row — NOT the current row. `(ord, val)`
    // grouped by `ord`; the value read is `val` (a different column than the ORDER BY key) so
    // last_value can't accidentally coincide with the current row. This is exactly the
    // `last_value(x) OVER (ORDER BY y)` shape the O(1) `last()` fix targets: the frame grows
    // to the whole peer run and must return its far edge in O(1).
    let input = values_node(&[
        vec![vint(1), vint(100)],
        vec![vint(1), vint(101)],
        vec![vint(2), vint(200)],
        vec![vint(2), vint(201)],
        vec![vint(2), vint(202)],
        vec![vint(3), vint(300)],
    ]);
    let f_first = wfunc(first_value(col(1)), vec![], vec![asc(0)], default_frame());
    let f_last = wfunc(last_value(col(1)), vec![], vec![asc(0)], default_frame());
    let rows = run(&window_plan(input, vec![f_first, f_last]));
    // columns: [ord, val, first, last]
    assert_eq!(col_cells(&rows, 2), vec![n(100); 6]); // first_value = partition start
    assert_eq!(col_cells(&rows, 3), vec![n(101), n(101), n(202), n(202), n(202), n(300)]);
}

#[test]
fn first_last_value_over_empty_frame_are_null() {
    // ROWS BETWEEN 5 FOLLOWING AND 9 FOLLOWING is entirely past a 3-row partition, so every
    // row's frame is empty ⇒ first_value/last_value/nth_value all yield NULL (fr.first()/
    // .last()/.nth() are None). Pins the None → NULL arm of the navigation helpers end-to-end.
    let input = values_node(&[vec![vint(10)], vec![vint(20)], vec![vint(30)]]);
    let fr = frame(FrameUnits::Rows, following(5), following(9), FrameExclude::NoOthers);
    let f_first = wfunc(first_value(col(0)), vec![], vec![asc(0)], fr.clone());
    let f_last = wfunc(last_value(col(0)), vec![], vec![asc(0)], fr.clone());
    let f_nth1 = wfunc(nth_value(col(0), lit(vint(1))), vec![], vec![asc(0)], fr);
    let rows = run(&window_plan(input, vec![f_first, f_last, f_nth1]));
    assert_eq!(col_cells(&rows, 1), vec![Cell::Null; 3]);
    assert_eq!(col_cells(&rows, 2), vec![Cell::Null; 3]);
    assert_eq!(col_cells(&rows, 3), vec![Cell::Null; 3]);
}

#[test]
fn value_fns_follow_order_by_within_a_single_partition() {
    // Unpartitioned, input NOT in sorted order. Value functions read the ORDER BY order and
    // write back to input positions, so a sorted≠input remap that returned the input-order row
    // would diverge. nth_value(x,1) = the sorted-FIRST value (1); last_value over the whole
    // partition = the sorted-LAST value (3) — the same for every input row.
    let input = values_node(&[vec![vint(3)], vec![vint(1)], vec![vint(2)]]);
    let full =
        frame(FrameUnits::Rows, unbounded_preceding(), FrameBound::UnboundedFollowing, FrameExclude::NoOthers);
    let f_nth1 = wfunc(nth_value(col(0), lit(vint(1))), vec![], vec![asc(0)], spec_frame());
    let f_last = wfunc(last_value(col(0)), vec![], vec![asc(0)], full);
    let rows = run(&window_plan(input, vec![f_nth1, f_last]));
    assert_eq!(col_cells(&rows, 1), vec![n(1), n(1), n(1)]);
    assert_eq!(col_cells(&rows, 2), vec![n(3), n(3), n(3)]);
}

#[test]
fn offset_and_n_coerce_non_integer_arguments() {
    // A non-integer offset/N is COERCED via to_integer (REAL truncates toward zero), not
    // rejected: lag(b, 2.9) behaves as offset 2 and nth_value(b, 2.9) as N=2. Every other
    // test uses integer literals, so this is the only pin on the to_integer coercion path.
    let f_lag = wfunc(lag(col(0), lit(Value::Real(2.9)), lit(Value::Null)), vec![], vec![asc(0)], spec_frame());
    let f_nth = wfunc(nth_value(col(0), lit(Value::Real(2.9))), vec![], vec![asc(0)], spec_frame());
    let rows = run(&window_plan(values_node(&b_rows()), vec![f_lag, f_nth]));
    // lag(b,2) ignores the frame: NULL,NULL,A,B,C,D,E.
    assert_eq!(col_cells(&rows, 1), vec![Cell::Null, Cell::Null, t("A"), t("B"), t("C"), t("D"), t("E")]);
    // nth_value(b,2) over the growing ROWS..CURRENT ROW frame: NULL then B once pos 1 exists.
    assert_eq!(col_cells(&rows, 2), vec![Cell::Null, t("B"), t("B"), t("B"), t("B"), t("B"), t("B")]);
}
