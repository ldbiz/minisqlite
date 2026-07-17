//! Integration tests for the `Window` operator, driven through the real
//! [`Executor`]/[`RowCursor`] seam. The input is a [`PlanNode::Values`] literal set (so
//! no table/pager seeding is needed — a window operator reads only its input rows), and
//! the window functions are tiny stub [`AggregateFunction`]s (`CountStar`, `SumAgg`, and
//! a `GroupConcatAgg` whose FRAME-ORDER concatenation lets the spec's worked examples be
//! used as exact oracles).
//!
//! These pin the OPERATOR's job over the new window vocab:
//!   * the DEFAULT frame — whole-partition aggregate (no `ORDER BY`) and running
//!     aggregate through peers (with `ORDER BY`);
//!   * EXPLICIT `ROWS`/`RANGE`/`GROUPS` frames with every boundary form and all four
//!     `EXCLUDE` modes, transcribed from windowfunctions.html §2.2;
//!   * per-function partitioning, `PARTITION BY` NULL grouping, `FILTER` (incl. the
//!     NULL-truth skip), the appended-column output shape, and input-order preservation;
//!   * boundedness on hostile-but-valid input — a huge (`i64::MAX`) `FOLLOWING` offset
//!     clamps rather than overflowing, a malformed (negative / non-numeric) offset is a
//!     loud `Error`, and an aggregate's own `ORDER BY` folds each frame in that order
//!     (the ordered-set window aggregate) rather than silently in frame order.
//!
//! The ranking (`row_number`/…) and navigation (`lag`/…) kinds now have real
//! implementations, tested in `tests/window_ranking.rs` and `tests/window_navigation.rs`.

use std::sync::Arc;

use minisqlite_catalog::SchemaCatalog;
use minisqlite_exec::{Executor, Runtime, StreamingExecutor};
use minisqlite_expr::{
    AggregateAccumulator, AggregateCall, AggregateFunction, CmpOp, CompareMeta, EvalExpr, FnContext,
    SortKey,
};
use minisqlite_pager::MemPager;
use minisqlite_plan::{
    FrameBound, FrameExclude, FrameUnits, Plan, PlanNode, Window, WindowFrame, WindowFunc,
    WindowFuncKind,
};
use minisqlite_types::{Collation, Error, Result, Value};
use minisqlite_exec::PagerSet;
use minisqlite_types::DbIndex;

// ----- stub aggregate functions (the reference model) -----------------------

/// `count(*)`: counts every `step` regardless of arguments.
#[derive(Debug)]
struct CountStar;

impl AggregateFunction for CountStar {
    fn new_accumulator(&self, _collation: Collation) -> Box<dyn AggregateAccumulator> {
        Box::new(CountAcc { n: 0 })
    }
}

struct CountAcc {
    n: i64,
}

impl AggregateAccumulator for CountAcc {
    fn step(&mut self, _args: &[Value], _ctx: &mut dyn FnContext) -> Result<()> {
        self.n += 1;
        Ok(())
    }
    fn finalize(&mut self, _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(Value::Integer(self.n))
    }
}

/// `sum(<arg>)`: folds numeric arguments (promoting to REAL once a real is seen); NULL if
/// it never folded a non-NULL value. `finalize` does NOT consume the accumulator, so the
/// operator can read a running snapshot per peer group.
#[derive(Debug)]
struct SumAgg;

impl AggregateFunction for SumAgg {
    fn new_accumulator(&self, _collation: Collation) -> Box<dyn AggregateAccumulator> {
        Box::new(SumAcc { stepped: false, is_real: false, int_sum: 0, real_sum: 0.0 })
    }
}

struct SumAcc {
    stepped: bool,
    is_real: bool,
    int_sum: i64,
    real_sum: f64,
}

impl AggregateAccumulator for SumAcc {
    fn step(&mut self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<()> {
        for v in args {
            match v {
                Value::Null => {}
                Value::Integer(i) => {
                    self.stepped = true;
                    if self.is_real {
                        self.real_sum += *i as f64;
                    } else {
                        self.int_sum += *i;
                    }
                }
                Value::Real(r) => {
                    self.stepped = true;
                    if !self.is_real {
                        self.is_real = true;
                        self.real_sum = self.int_sum as f64;
                    }
                    self.real_sum += *r;
                }
                Value::Text(_) | Value::Blob(_) => {
                    self.stepped = true;
                }
            }
        }
        Ok(())
    }
    fn finalize(&mut self, _ctx: &mut dyn FnContext) -> Result<Value> {
        if !self.stepped {
            Ok(Value::Null)
        } else if self.is_real {
            Ok(Value::Real(self.real_sum))
        } else {
            Ok(Value::Integer(self.int_sum))
        }
    }
}

/// `group_concat(<value>, <sep>)`: concatenates the first argument's text over the rows
/// it is stepped with, joined by the second argument (a text separator; default "."). It
/// steps in FRAME order, so the concatenation order is the window/frame order the SQLite
/// documentation's worked examples publish — which is why it works as an exact oracle.
/// `finalize` is non-consuming (running-snapshot friendly) and returns NULL over zero
/// stepped rows (an empty frame, e.g. `EXCLUDE GROUP` over a single peer group).
#[derive(Debug)]
struct GroupConcatAgg;

impl AggregateFunction for GroupConcatAgg {
    fn new_accumulator(&self, _collation: Collation) -> Box<dyn AggregateAccumulator> {
        Box::new(GcAcc { parts: Vec::new(), sep: ".".to_string() })
    }
}

struct GcAcc {
    parts: Vec<String>,
    sep: String,
}

impl AggregateAccumulator for GcAcc {
    fn step(&mut self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<()> {
        if let Some(v) = args.first() {
            if !v.is_null() {
                self.parts.push(value_text(v));
            }
        }
        if let Some(Value::Text(s)) = args.get(1) {
            self.sep = s.clone();
        }
        Ok(())
    }
    fn finalize(&mut self, _ctx: &mut dyn FnContext) -> Result<Value> {
        if self.parts.is_empty() {
            Ok(Value::Null)
        } else {
            Ok(Value::Text(self.parts.join(&self.sep)))
        }
    }
}

fn value_text(v: &Value) -> String {
    match v {
        Value::Text(s) => s.clone(),
        Value::Integer(i) => i.to_string(),
        Value::Real(r) => r.to_string(),
        Value::Blob(b) => String::from_utf8_lossy(b).into_owned(),
        Value::Null => String::new(),
    }
}

// ----- value / plan helpers -------------------------------------------------

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

fn count_star() -> AggregateCall {
    AggregateCall {
        func: Arc::new(CountStar),
        distinct: false,
        args: Vec::new(),
        filter: None,
        order_by: Vec::new(),
        arg_collations: Vec::new(),
    }
}
fn sum_of(arg: EvalExpr) -> AggregateCall {
    AggregateCall {
        func: Arc::new(SumAgg),
        distinct: false,
        args: vec![arg],
        filter: None,
        order_by: Vec::new(),
        arg_collations: Vec::new(),
    }
}
fn sum_filtered(arg: EvalExpr, filter: EvalExpr) -> AggregateCall {
    AggregateCall {
        func: Arc::new(SumAgg),
        distinct: false,
        args: vec![arg],
        filter: Some(filter),
        order_by: Vec::new(),
        arg_collations: Vec::new(),
    }
}
/// `group_concat(value, '.')` over the given value expression.
fn group_concat(value: EvalExpr) -> AggregateCall {
    AggregateCall {
        func: Arc::new(GroupConcatAgg),
        distinct: false,
        args: vec![value, lit(vtext("."))],
        filter: None,
        order_by: Vec::new(),
        arg_collations: Vec::new(),
    }
}

fn asc(col_idx: usize) -> SortKey {
    SortKey { expr: col(col_idx), desc: false, nulls_first: None, collation: Collation::Binary }
}
fn desc(col_idx: usize) -> SortKey {
    SortKey { expr: col(col_idx), desc: true, nulls_first: None, collation: Collation::Binary }
}

// Frame constructors mirroring the plan vocab.
fn default_frame() -> WindowFrame {
    WindowFrame::default_frame()
}
fn frame(units: FrameUnits, start: FrameBound, end: FrameBound, exclude: FrameExclude) -> WindowFrame {
    WindowFrame { units, start, end, exclude }
}
fn unbounded_preceding() -> FrameBound {
    FrameBound::UnboundedPreceding
}
fn unbounded_following() -> FrameBound {
    FrameBound::UnboundedFollowing
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

fn agg(call: AggregateCall) -> WindowFuncKind {
    WindowFuncKind::Aggregate(call)
}

fn wfunc(
    kind: WindowFuncKind,
    partition_by: Vec<EvalExpr>,
    order_by: Vec<SortKey>,
    frame: WindowFrame,
) -> WindowFunc {
    // These builders exercise BINARY partitioning; the collation-aware path is pinned by
    // the plan-level test and the facade conformance suite.
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

/// `col > value` (three-valued), used for `FILTER` predicates.
fn gt(left: EvalExpr, value: i64) -> EvalExpr {
    EvalExpr::Compare {
        op: CmpOp::Gt,
        null_safe: false,
        left: Box::new(left),
        right: Box::new(lit(vint(value))),
        meta: CompareMeta { apply_left: None, apply_right: None, collation: Collation::Binary },
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

fn int(v: &Value) -> i64 {
    match v {
        Value::Integer(i) => *i,
        other => panic!("expected Integer, got {other:?}"),
    }
}

/// The group_concat result as `Option<String>` (NULL over an empty frame → `None`).
fn gc(v: &Value) -> Option<String> {
    match v {
        Value::Null => None,
        Value::Text(s) => Some(s.clone()),
        other => panic!("expected Text or Null, got {other:?}"),
    }
}

// ===========================================================================
// (1) DEFAULT frame — whole partition (no ORDER BY) and running-through-peers
// ===========================================================================

#[test]
fn count_star_over_empty_window_appends_partition_count() {
    // COUNT(*) OVER () over 3 rows: every row gets 3. Width = input(1) + function(1).
    let input = values_node(&[vec![vint(10)], vec![vint(20)], vec![vint(30)]]);
    let rows = run(&window_plan(input, vec![wfunc(agg(count_star()), vec![], vec![], default_frame())]));

    assert_eq!(rows.len(), 3);
    for r in &rows {
        assert_eq!(r.len(), 2, "width = input(1) + functions(1)");
        assert_eq!(int(&r[1]), 3, "COUNT(*) OVER () = whole-partition count");
    }
    assert_eq!(int(&rows[0][0]), 10);
    assert_eq!(int(&rows[2][0]), 30);
}

#[test]
fn sum_over_partition_by_appends_group_total() {
    // rows (g=1,x=10),(g=1,x=20),(g=2,x=5) → 30,30,5, in input order.
    let input = values_node(&[
        vec![vint(1), vint(10)],
        vec![vint(1), vint(20)],
        vec![vint(2), vint(5)],
    ]);
    let f = wfunc(agg(sum_of(col(1))), vec![col(0)], vec![], default_frame());
    let rows = run(&window_plan(input, vec![f]));

    assert_eq!(int(&rows[0][2]), 30);
    assert_eq!(int(&rows[1][2]), 30);
    assert_eq!(int(&rows[2][2]), 5);
}

#[test]
fn running_sum_over_order_by_x() {
    // x = 1,2,3 → running sums 1,3,6 (unique key ⇒ no peers).
    let input = values_node(&[vec![vint(1)], vec![vint(2)], vec![vint(3)]]);
    let f = wfunc(agg(sum_of(col(0))), vec![], vec![asc(0)], default_frame());
    let rows = run(&window_plan(input, vec![f]));
    assert_eq!((int(&rows[0][1]), int(&rows[1][1]), int(&rows[2][1])), (1, 3, 6));
}

#[test]
fn running_sum_sorts_then_re_emits_in_input_order() {
    // Stored OUT of order (x = 3,1,2): running frame by SORTED x, output by INPUT order.
    let input = values_node(&[vec![vint(3)], vec![vint(1)], vec![vint(2)]]);
    let f = wfunc(agg(sum_of(col(0))), vec![], vec![asc(0)], default_frame());
    let rows = run(&window_plan(input, vec![f]));
    assert_eq!((int(&rows[0][0]), int(&rows[0][1])), (3, 6));
    assert_eq!((int(&rows[1][0]), int(&rows[1][1])), (1, 1));
    assert_eq!((int(&rows[2][0]), int(&rows[2][1])), (2, 3));
}

#[test]
fn running_sum_over_order_by_includes_peers() {
    // ORDER BY g with g=1(10),g=1(20),g=2(5): the two g=1 peers both get 30, g=2 → 35.
    let input = values_node(&[
        vec![vint(1), vint(10)],
        vec![vint(1), vint(20)],
        vec![vint(2), vint(5)],
    ]);
    let f = wfunc(agg(sum_of(col(1))), vec![], vec![asc(0)], default_frame());
    let rows = run(&window_plan(input, vec![f]));
    assert_eq!(int(&rows[0][2]), 30);
    assert_eq!(int(&rows[1][2]), 30);
    assert_eq!(int(&rows[2][2]), 35);
}

#[test]
fn running_sum_desc_order_by_with_peers() {
    // ORDER BY g DESC with g=1,2,2 (x=5,10,20): g=2 peers → 30 each, g=1 last → 35.
    let input = values_node(&[
        vec![vint(1), vint(5)],
        vec![vint(2), vint(10)],
        vec![vint(2), vint(20)],
    ]);
    let f = wfunc(agg(sum_of(col(1))), vec![], vec![desc(0)], default_frame());
    let rows = run(&window_plan(input, vec![f]));
    assert_eq!(int(&rows[0][2]), 35);
    assert_eq!(int(&rows[1][2]), 30);
    assert_eq!(int(&rows[2][2]), 30);
}

#[test]
fn running_sum_partition_by_with_order_by_is_isolated_per_partition() {
    // SUM(x) OVER (PARTITION BY g ORDER BY h): each partition its own running accumulator.
    let input = values_node(&[
        vec![vint(1), vint(1), vint(10)],
        vec![vint(2), vint(1), vint(100)],
        vec![vint(1), vint(2), vint(20)],
        vec![vint(2), vint(2), vint(200)],
    ]);
    let f = wfunc(agg(sum_of(col(2))), vec![col(0)], vec![asc(1)], default_frame());
    let rows = run(&window_plan(input, vec![f]));
    assert_eq!(int(&rows[0][3]), 10);
    assert_eq!(int(&rows[1][3]), 100);
    assert_eq!(int(&rows[2][3]), 30);
    assert_eq!(int(&rows[3][3]), 300);
}

// ===========================================================================
// (2) per-function partitioning / NULL partition / FILTER / empty input
// ===========================================================================

#[test]
fn multiple_functions_each_append_a_column_with_own_partitioning() {
    // f0 = SUM(x) OVER (PARTITION BY g); f1 = COUNT(*) OVER (). Input [g,x] scattered.
    let input = values_node(&[
        vec![vint(1), vint(10)],
        vec![vint(2), vint(20)],
        vec![vint(1), vint(30)],
    ]);
    let f0 = wfunc(agg(sum_of(col(1))), vec![col(0)], vec![], default_frame());
    let f1 = wfunc(agg(count_star()), vec![], vec![], default_frame());
    let rows = run(&window_plan(input, vec![f0, f1]));

    assert_eq!(rows[0].len(), 4, "width = input(2) + functions(2)");
    assert_eq!((int(&rows[0][2]), int(&rows[0][3])), (40, 3));
    assert_eq!((int(&rows[1][2]), int(&rows[1][3])), (20, 3));
    assert_eq!((int(&rows[2][2]), int(&rows[2][3])), (40, 3));
}

#[test]
fn partition_by_groups_null_keys_together() {
    // g = NULL, 1, NULL: the two NULL rows share a partition (40), g=1 its own (20).
    let input = values_node(&[
        vec![Value::Null, vint(10)],
        vec![vint(1), vint(20)],
        vec![Value::Null, vint(30)],
    ]);
    let f = wfunc(agg(sum_of(col(1))), vec![col(0)], vec![], default_frame());
    let rows = run(&window_plan(input, vec![f]));
    assert_eq!(int(&rows[0][2]), 40);
    assert_eq!(int(&rows[1][2]), 20);
    assert_eq!(int(&rows[2][2]), 40);
}

#[test]
fn filter_excludes_rows_from_aggregate_but_not_from_output() {
    // SUM(x) FILTER (WHERE x > 15) OVER (PARTITION BY g): only x>15 contribute; 20+30=50.
    let input = values_node(&[
        vec![vint(1), vint(10)],
        vec![vint(1), vint(20)],
        vec![vint(1), vint(30)],
    ]);
    let f = wfunc(agg(sum_filtered(col(1), gt(col(1), 15))), vec![col(0)], vec![], default_frame());
    let rows = run(&window_plan(input, vec![f]));
    assert_eq!(rows.len(), 3, "FILTER does not drop output rows");
    for r in &rows {
        assert_eq!(int(&r[2]), 50);
    }
}

#[test]
fn empty_input_yields_no_rows() {
    let input = values_node(&[]);
    let rows = run(&window_plan(input, vec![wfunc(agg(count_star()), vec![], vec![], default_frame())]));
    assert!(rows.is_empty());
}

// ===========================================================================
// (3) EXPLICIT ROWS frames — running total, sliding, current..unbounded following
// ===========================================================================

#[test]
fn rows_unbounded_preceding_to_current_row_running_total() {
    // ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW over unique x = 1,2,3 ⇒ 1,3,6.
    let input = values_node(&[vec![vint(1)], vec![vint(2)], vec![vint(3)]]);
    let fr = frame(FrameUnits::Rows, unbounded_preceding(), current_row(), FrameExclude::NoOthers);
    let f = wfunc(agg(sum_of(col(0))), vec![], vec![asc(0)], fr);
    let rows = run(&window_plan(input, vec![f]));
    assert_eq!((int(&rows[0][1]), int(&rows[1][1]), int(&rows[2][1])), (1, 3, 6));
}

#[test]
fn rows_does_not_group_peers() {
    // ROWS counts individual rows, so the two v=20 peers get DIFFERENT running sums.
    // Sorted by v: 5,10,20,20,30 ⇒ prefix sums 5,15,35,55,85. Emitted in input order;
    // ids 3,4 are the two v=20 rows and get 35 and 55 respectively (stable within peers).
    let input = values_node(&[
        vec![vint(10)], // id1
        vec![vint(30)], // id2
        vec![vint(20)], // id3
        vec![vint(20)], // id4
        vec![vint(5)],  // id5
    ]);
    let fr = frame(FrameUnits::Rows, unbounded_preceding(), current_row(), FrameExclude::NoOthers);
    let f = wfunc(agg(sum_of(col(0))), vec![], vec![asc(0)], fr);
    let rows = run(&window_plan(input, vec![f]));
    // sorted order 5(id5),10(id1),20(id3),20(id4),30(id2) ⇒ 5,15,35,55,85
    assert_eq!(int(&rows[0][1]), 15, "id1 v=10 is 2nd ⇒ 5+10");
    assert_eq!(int(&rows[1][1]), 85, "id2 v=30 is last ⇒ 85");
    assert_eq!(int(&rows[2][1]), 35, "id3 v=20 (first of the peers) ⇒ 5+10+20");
    assert_eq!(int(&rows[3][1]), 55, "id4 v=20 (second of the peers) ⇒ +20");
    assert_eq!(int(&rows[4][1]), 5, "id5 v=5 is first ⇒ 5");
}

#[test]
fn rows_sliding_one_preceding_one_following() {
    // ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING over ids 1..5 with v=10,30,20,20,5.
    // Clamped at edges: 40,60,70,45,25.
    let input = values_node(&[
        vec![vint(1), vint(10)],
        vec![vint(2), vint(30)],
        vec![vint(3), vint(20)],
        vec![vint(4), vint(20)],
        vec![vint(5), vint(5)],
    ]);
    let fr = frame(FrameUnits::Rows, preceding(1), following(1), FrameExclude::NoOthers);
    let f = wfunc(agg(sum_of(col(1))), vec![], vec![asc(0)], fr);
    let rows = run(&window_plan(input, vec![f]));
    assert_eq!(int(&rows[0][2]), 40);
    assert_eq!(int(&rows[1][2]), 60);
    assert_eq!(int(&rows[2][2]), 70);
    assert_eq!(int(&rows[3][2]), 45);
    assert_eq!(int(&rows[4][2]), 25);
}

#[test]
fn rows_following_only_is_null_at_partition_end() {
    // ROWS BETWEEN 1 FOLLOWING AND 2 FOLLOWING over x=1..5: the last row's frame is empty
    // ⇒ sum over zero rows ⇒ NULL. Emitted in input order.
    let input = values_node(&[vec![vint(1)], vec![vint(2)], vec![vint(3)], vec![vint(4)], vec![vint(5)]]);
    let fr = frame(FrameUnits::Rows, following(1), following(2), FrameExclude::NoOthers);
    let f = wfunc(agg(sum_of(col(0))), vec![], vec![asc(0)], fr);
    let rows = run(&window_plan(input, vec![f]));
    assert_eq!(int(&rows[0][1]), 5, "x=1: sum of x in [2,3]");
    assert_eq!(int(&rows[1][1]), 7, "x=2: sum of x in [3,4]");
    assert_eq!(int(&rows[2][1]), 9, "x=3: sum of x in [4,5]");
    assert_eq!(int(&rows[3][1]), 5, "x=4: sum of x in [5]");
    assert!(matches!(rows[4][1], Value::Null), "x=5: empty frame ⇒ NULL");
}

// ===========================================================================
// (4) group_concat over the SQLite doc's worked examples (§2.2) — exact oracles
// ===========================================================================

/// windowfunctions.html §2 table t1, columns [a, b, c] in a-order.
fn t1_input() -> PlanNode {
    values_node(&[
        vec![vint(1), vtext("A"), vtext("one")],
        vec![vint(2), vtext("B"), vtext("two")],
        vec![vint(3), vtext("C"), vtext("three")],
        vec![vint(4), vtext("D"), vtext("one")],
        vec![vint(5), vtext("E"), vtext("two")],
        vec![vint(6), vtext("F"), vtext("three")],
        vec![vint(7), vtext("G"), vtext("one")],
    ])
}

#[test]
fn default_frame_group_concat_over_order_by_c() {
    // group_concat(b, '.') OVER (ORDER BY c) — default RANGE frame includes peers.
    // Ordered by c ('one'<'three'<'two'): A,D,G | C,F | B,E. Output in input order (a=1..7).
    let f = wfunc(agg(group_concat(col(1))), vec![], vec![asc(2)], default_frame());
    let rows = run(&window_plan(t1_input(), vec![f]));
    let got: Vec<Option<String>> = rows.iter().map(|r| gc(&r[3])).collect();
    assert_eq!(
        got,
        vec![
            Some("A.D.G".into()),         // a1 (one)
            Some("A.D.G.C.F.B.E".into()), // a2 (two)
            Some("A.D.G.C.F".into()),     // a3 (three)
            Some("A.D.G".into()),         // a4 (one)
            Some("A.D.G.C.F.B.E".into()), // a5 (two)
            Some("A.D.G.C.F".into()),     // a6 (three)
            Some("A.D.G".into()),         // a7 (one)
        ]
    );
}

#[test]
fn rows_current_to_unbounded_following_group_concat() {
    // group_concat(b, '.') OVER (ORDER BY c, a ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING)
    // Ordered by (c,a): A,D,G,C,F,B,E. Output in input order (a=1..7).
    let fr = frame(FrameUnits::Rows, current_row(), unbounded_following(), FrameExclude::NoOthers);
    let f = wfunc(agg(group_concat(col(1))), vec![], vec![asc(2), asc(0)], fr);
    let rows = run(&window_plan(t1_input(), vec![f]));
    let got: Vec<Option<String>> = rows.iter().map(|r| gc(&r[3])).collect();
    assert_eq!(
        got,
        vec![
            Some("A.D.G.C.F.B.E".into()), // a1=A
            Some("B.E".into()),           // a2=B
            Some("C.F.B.E".into()),       // a3=C
            Some("D.G.C.F.B.E".into()),   // a4=D
            Some("E".into()),             // a5=E
            Some("F.B.E".into()),         // a6=F
            Some("G.C.F.B.E".into()),     // a7=G
        ]
    );
}

#[test]
fn rows_one_preceding_one_following_group_concat() {
    // §2 intro: group_concat(b, '.') OVER (ORDER BY a ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING).
    let fr = frame(FrameUnits::Rows, preceding(1), following(1), FrameExclude::NoOthers);
    let f = wfunc(agg(group_concat(col(1))), vec![], vec![asc(0)], fr);
    let rows = run(&window_plan(t1_input(), vec![f]));
    let got: Vec<Option<String>> = rows.iter().map(|r| gc(&r[3])).collect();
    assert_eq!(
        got,
        vec![
            Some("A.B".into()),
            Some("A.B.C".into()),
            Some("B.C.D".into()),
            Some("C.D.E".into()),
            Some("D.E.F".into()),
            Some("E.F.G".into()),
            Some("F.G".into()),
        ]
    );
}

/// windowfunctions.html §2.2.3 EXCLUDE table: four `group_concat(b,'.') OVER (ORDER BY c
/// GROUPS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE <mode>)` columns over t1.
/// Every expected value is transcribed from the doc, mapped to INPUT order (a=1..7).
#[test]
fn groups_frame_exclude_modes_match_spec_table() {
    let groups = |ex: FrameExclude| {
        wfunc(
            agg(group_concat(col(1))),
            vec![],
            vec![asc(2)],
            frame(FrameUnits::Groups, unbounded_preceding(), current_row(), ex),
        )
    };
    let funcs = vec![
        groups(FrameExclude::NoOthers),   // col 3
        groups(FrameExclude::CurrentRow), // col 4
        groups(FrameExclude::Group),      // col 5
        groups(FrameExclude::Ties),       // col 6
    ];
    let rows = run(&window_plan(t1_input(), funcs));

    // (a, b) → (no_others, current_row, group, ties), in input order.
    let expected: Vec<(Option<&str>, Option<&str>, Option<&str>, Option<&str>)> = vec![
        // a1=A (one)
        (Some("A.D.G"), Some("D.G"), None, Some("A")),
        // a2=B (two)
        (Some("A.D.G.C.F.B.E"), Some("A.D.G.C.F.E"), Some("A.D.G.C.F"), Some("A.D.G.C.F.B")),
        // a3=C (three)
        (Some("A.D.G.C.F"), Some("A.D.G.F"), Some("A.D.G"), Some("A.D.G.C")),
        // a4=D (one)
        (Some("A.D.G"), Some("A.G"), None, Some("D")),
        // a5=E (two)
        (Some("A.D.G.C.F.B.E"), Some("A.D.G.C.F.B"), Some("A.D.G.C.F"), Some("A.D.G.C.F.E")),
        // a6=F (three)
        (Some("A.D.G.C.F"), Some("A.D.G.C"), Some("A.D.G"), Some("A.D.G.F")),
        // a7=G (one)
        (Some("A.D.G"), Some("A.D"), None, Some("G")),
    ];

    for (r, (no_others, current, grp, ties)) in rows.iter().zip(expected) {
        assert_eq!(gc(&r[3]).as_deref(), no_others, "EXCLUDE NO OTHERS at a={:?}", int(&r[0]));
        assert_eq!(gc(&r[4]).as_deref(), current, "EXCLUDE CURRENT ROW at a={:?}", int(&r[0]));
        assert_eq!(gc(&r[5]).as_deref(), grp, "EXCLUDE GROUP at a={:?}", int(&r[0]));
        assert_eq!(gc(&r[6]).as_deref(), ties, "EXCLUDE TIES at a={:?}", int(&r[0]));
    }
}

// ===========================================================================
// (5) RANGE explicit frame
// ===========================================================================

#[test]
fn range_unbounded_preceding_current_row_matches_default() {
    // Explicit RANGE UNBOUNDED PRECEDING .. CURRENT ROW == the default frame: peers share.
    // v = 5,10,20,20,30 (input order) ⇒ 5,15,55,55,85 sorted; emitted in input order.
    let input =
        values_node(&[vec![vint(5)], vec![vint(10)], vec![vint(20)], vec![vint(20)], vec![vint(30)]]);
    let fr = frame(FrameUnits::Range, unbounded_preceding(), current_row(), FrameExclude::NoOthers);
    let f = wfunc(agg(sum_of(col(0))), vec![], vec![asc(0)], fr);
    let rows = run(&window_plan(input, vec![f]));
    assert_eq!(int(&rows[0][1]), 5);
    assert_eq!(int(&rows[1][1]), 15);
    assert_eq!(int(&rows[2][1]), 55, "peer group of the two 20s");
    assert_eq!(int(&rows[3][1]), 55);
    assert_eq!(int(&rows[4][1]), 85);
}

#[test]
fn range_numeric_band_one_preceding_one_following() {
    // RANGE BETWEEN 1 PRECEDING AND 1 FOLLOWING over v = 1,2,3,10.
    // bands: v1→{1,2}=3, v2→{1,2,3}=6, v3→{2,3}=5, v10→{10}=10.
    let input = values_node(&[vec![vint(1)], vec![vint(2)], vec![vint(3)], vec![vint(10)]]);
    let fr = frame(FrameUnits::Range, preceding(1), following(1), FrameExclude::NoOthers);
    let f = wfunc(agg(sum_of(col(0))), vec![], vec![asc(0)], fr);
    let rows = run(&window_plan(input, vec![f]));
    assert_eq!(int(&rows[0][1]), 3);
    assert_eq!(int(&rows[1][1]), 6);
    assert_eq!(int(&rows[2][1]), 5);
    assert_eq!(int(&rows[3][1]), 10);
}

// ===========================================================================
// (5b) BOUNDED edges — huge offsets clamp (no overflow panic), malformed offsets
//      and NULL-truth FILTER, and the ordered-set aggregate-internal-ORDER-BY fold
// ===========================================================================

#[test]
fn rows_huge_following_offset_must_not_panic() {
    // ROWS BETWEEN CURRENT ROW AND i64::MAX FOLLOWING over x=1,2. Real sqlite clamps the
    // offset to the partition end: row0 → {1,2}=3, row1 → {2}=2. Without saturating
    // arithmetic `c + i64::MAX` overflows and panics under the (default) test profile.
    let input = values_node(&[vec![vint(1)], vec![vint(2)]]);
    let fr = frame(FrameUnits::Rows, current_row(), following(i64::MAX), FrameExclude::NoOthers);
    let f = wfunc(agg(sum_of(col(0))), vec![], vec![asc(0)], fr);
    let rows = run(&window_plan(input, vec![f]));
    assert_eq!(int(&rows[0][1]), 3);
    assert_eq!(int(&rows[1][1]), 2);
}

#[test]
fn groups_huge_following_offset_must_not_panic() {
    // Same shape, GROUPS units: `group_index(c) + i64::MAX` must saturate then clamp to the
    // last group. Unique keys ⇒ one row per group ⇒ same answers as ROWS: 3, 2.
    let input = values_node(&[vec![vint(1)], vec![vint(2)]]);
    let fr = frame(FrameUnits::Groups, current_row(), following(i64::MAX), FrameExclude::NoOthers);
    let f = wfunc(agg(sum_of(col(0))), vec![], vec![asc(0)], fr);
    let rows = run(&window_plan(input, vec![f]));
    assert_eq!(int(&rows[0][1]), 3);
    assert_eq!(int(&rows[1][1]), 2);
}

#[test]
fn negative_rows_offset_is_loud_error() {
    // A negative ROWS offset is malformed: eval_offset returns an Error, never a wrong
    // frame and never a panic (a malformed plan must return an Error, never panic).
    let input = values_node(&[vec![vint(1)], vec![vint(2)]]);
    let fr = frame(FrameUnits::Rows, preceding(-1), current_row(), FrameExclude::NoOthers);
    let f = wfunc(agg(sum_of(col(0))), vec![], vec![asc(0)], fr);
    let msg = run_err(&window_plan(input, vec![f]));
    assert!(msg.contains("non-negative integer"), "got: {msg}");
}

#[test]
fn non_numeric_range_offset_is_loud_error() {
    // A RANGE offset must be numeric; a text offset is rejected loudly.
    let input = values_node(&[vec![vint(1)], vec![vint(2)]]);
    let fr = frame(
        FrameUnits::Range,
        current_row(),
        FrameBound::Following(lit(vtext("x"))),
        FrameExclude::NoOthers,
    );
    let f = wfunc(agg(sum_of(col(0))), vec![], vec![asc(0)], fr);
    let msg = run_err(&window_plan(input, vec![f]));
    assert!(msg.contains("non-negative number"), "got: {msg}");
}

#[test]
fn filter_null_predicate_skips_row_like_false() {
    // FILTER (WHERE f > 15) with f=NULL on the 3rd row: the predicate is NULL (three-valued,
    // not TRUE) so the row is skipped exactly like FALSE. The summed column is non-NULL, so
    // wrongly treating NULL-truth as TRUE would change the total (200 → 300).
    let input = values_node(&[
        vec![vint(1), vint(10), vint(100)],    // f=10 FALSE  → skip
        vec![vint(1), vint(20), vint(100)],    // f=20 TRUE   → +100
        vec![vint(1), Value::Null, vint(100)], // f=NULL NULL → skip
        vec![vint(1), vint(30), vint(100)],    // f=30 TRUE   → +100
    ]);
    let f = wfunc(agg(sum_filtered(col(2), gt(col(1), 15))), vec![col(0)], vec![], default_frame());
    let rows = run(&window_plan(input, vec![f]));
    for r in &rows {
        assert_eq!(int(&r[3]), 200, "NULL-truth FILTER row must be skipped like FALSE");
    }
}

#[test]
fn aggregate_internal_order_by_folds_frame_in_aggregate_order() {
    // A window aggregate carrying its OWN ORDER BY (group_concat(b, '.' ORDER BY a)) folds
    // each row's frame in the aggregate's ORDER BY order, NOT frame/window order — the
    // ordered-set window aggregate implemented in `ops/window.rs::eval_ordered_frame`. This
    // guards CORRECT ordering: it FAILS if `AggregateCall.order_by` is ever silently ignored
    // (which would fold in frame order) — the discriminating replacement for the old loud-
    // rejection guard, now that the feature is implemented.
    //
    // The input is stored (2,"B"),(1,"A") so frame order ("B.A") DIFFERS from the aggregate
    // ORDER BY a order ("A.B"). The default frame with no window ORDER BY is the WHOLE
    // partition, so every row's frame is both rows; sorting by col(0) ⇒ (1,"A"),(2,"B") ⇒
    // "A.B" for every row. If the ORDER BY were ignored the fold would read "B.A".
    let input = values_node(&[vec![vint(2), vtext("B")], vec![vint(1), vtext("A")]]);
    let call = AggregateCall {
        func: Arc::new(GroupConcatAgg),
        distinct: false,
        args: vec![col(1), lit(vtext("."))],
        filter: None,
        order_by: vec![asc(0)],
        arg_collations: Vec::new(),
    };
    let f = wfunc(agg(call), vec![], vec![], default_frame());
    let rows = run(&window_plan(input, vec![f]));

    assert_eq!(rows.len(), 2);
    for r in &rows {
        assert_eq!(r.len(), 3, "width = input(2) + functions(1)");
        assert_eq!(
            gc(&r[2]).as_deref(),
            Some("A.B"),
            "frame folded in aggregate ORDER BY a order (would be \"B.A\" if ignored)",
        );
    }
    // The leading columns are the untouched input columns, in INPUT order — both the
    // integer key (col 0) and the source text (col 1) pass through unchanged.
    assert_eq!(int(&rows[0][0]), 2);
    assert_eq!(int(&rows[1][0]), 1);
    assert!(matches!(&rows[0][1], Value::Text(s) if s == "B"), "col-1 untouched, got {:?}", rows[0][1]);
    assert!(matches!(&rows[1][1], Value::Text(s) if s == "A"), "col-1 untouched, got {:?}", rows[1][1]);
}

#[test]
fn aggregate_internal_order_by_desc_folds_frame_reversed() {
    // The DESC direction, kept INDEPENDENTLY discriminating by anti-sorting the input the
    // other way from the ASC test: with ASCENDING input (1,"A"),(2,"B") the frame/window
    // order is "A.B", but group_concat(b, '.' ORDER BY a DESC) folds the whole-partition
    // frame by col(0) DESCENDING ⇒ (2,"B"),(1,"A") ⇒ "B.A" for every row. On its OWN this
    // therefore FAILS both if `AggregateCall.order_by` is ignored (fold in frame order ⇒
    // "A.B") and if the DESC direction is dropped / always-ascending (also "A.B"). It mirrors
    // the ascending-input `ORDER BY b DESC` shape pinned end-to-end in the facade suite
    // (`crates/minisqlite/tests/conformance_window_ordered_aggregate.rs`).
    let input = values_node(&[vec![vint(1), vtext("A")], vec![vint(2), vtext("B")]]);
    let call = AggregateCall {
        func: Arc::new(GroupConcatAgg),
        distinct: false,
        args: vec![col(1), lit(vtext("."))],
        filter: None,
        order_by: vec![desc(0)],
        arg_collations: Vec::new(),
    };
    let f = wfunc(agg(call), vec![], vec![], default_frame());
    let rows = run(&window_plan(input, vec![f]));

    assert_eq!(rows.len(), 2);
    for r in &rows {
        assert_eq!(r.len(), 3, "width = input(2) + functions(1)");
        assert_eq!(
            gc(&r[2]).as_deref(),
            Some("B.A"),
            "frame folded in aggregate ORDER BY a DESC order (would be \"A.B\" if ignored)",
        );
    }
    // Leading columns untouched, in INPUT order.
    assert_eq!(int(&rows[0][0]), 1);
    assert_eq!(int(&rows[1][0]), 2);
    assert!(matches!(&rows[0][1], Value::Text(s) if s == "A"), "col-1 untouched, got {:?}", rows[0][1]);
    assert!(matches!(&rows[1][1], Value::Text(s) if s == "B"), "col-1 untouched, got {:?}", rows[1][1]);
}

#[test]
fn distinct_dedups_repeated_values_within_the_frame() {
    // DISTINCT window aggregates are rejected by the binder (SQLite disallows them), so this
    // drives the operator's DEFENSIVE dedup path directly: sum(DISTINCT x) folds each distinct
    // value ONCE. x = 2,2,3 over the whole partition (no ORDER BY) → 2+3 = 5, not 2+2+3 = 7.
    // Also guards the passed-up shared-agg-step refactor from silently dropping DISTINCT.
    let input = values_node(&[vec![vint(2)], vec![vint(2)], vec![vint(3)]]);
    let call = AggregateCall {
        func: Arc::new(SumAgg),
        distinct: true,
        args: vec![col(0)],
        filter: None,
        order_by: Vec::new(),
        arg_collations: Vec::new(),
    };
    let f = wfunc(agg(call), vec![], vec![], default_frame());
    let rows = run(&window_plan(input, vec![f]));
    for r in &rows {
        assert_eq!(int(&r[1]), 5, "DISTINCT must fold each value once (2+3), not 2+2+3");
    }
}

// The FILTER/DISTINCT tests above use default_frame(), which routes to the O(p) running
// `eval_default_frame`. The `eval_general_frame` rebuild path threads FILTER and the DISTINCT
// `seen` set SEPARATELY (a fresh accumulator + seen per row's frame), so these two variants
// force the general path with an EXPLICIT ROWS frame — a break confined to that branch (e.g. a
// mis-wired per-frame FILTER skip or dedup) would slip past the default-path tests. FILTER over
// an explicit frame is a genuinely reachable path now that the binder emits explicit frames.

#[test]
fn filter_null_predicate_skips_row_on_general_frame_path_too() {
    // Same NULL-truth FILTER skip as filter_null_predicate_skips_row_like_false, but through an
    // explicit ROWS whole-partition frame ⇒ eval_general_frame. f=NULL must be skipped (200,
    // not 300) on the general path too.
    let input = values_node(&[
        vec![vint(1), vint(10), vint(100)],    // f=10 FALSE  → skip
        vec![vint(1), vint(20), vint(100)],    // f=20 TRUE   → +100
        vec![vint(1), Value::Null, vint(100)], // f=NULL NULL → skip
        vec![vint(1), vint(30), vint(100)],    // f=30 TRUE   → +100
    ]);
    let fr = frame(FrameUnits::Rows, unbounded_preceding(), unbounded_following(), FrameExclude::NoOthers);
    let f = wfunc(agg(sum_filtered(col(2), gt(col(1), 15))), vec![col(0)], vec![], fr);
    let rows = run(&window_plan(input, vec![f]));
    for r in &rows {
        assert_eq!(int(&r[3]), 200, "NULL-truth FILTER row must be skipped on the general path too");
    }
}

#[test]
fn distinct_dedups_on_general_frame_path_too() {
    // Same DISTINCT dedup as distinct_dedups_repeated_values_within_the_frame, but through an
    // explicit ROWS whole-partition frame ⇒ eval_general_frame (fresh `seen` per frame).
    // sum(DISTINCT 2,2,3) over the whole frame = 5, not 7.
    let input = values_node(&[vec![vint(2)], vec![vint(2)], vec![vint(3)]]);
    let call = AggregateCall {
        func: Arc::new(SumAgg),
        distinct: true,
        args: vec![col(0)],
        filter: None,
        order_by: Vec::new(),
        arg_collations: Vec::new(),
    };
    let fr = frame(FrameUnits::Rows, unbounded_preceding(), unbounded_following(), FrameExclude::NoOthers);
    let f = wfunc(agg(call), vec![], vec![], fr);
    let rows = run(&window_plan(input, vec![f]));
    for r in &rows {
        assert_eq!(int(&r[1]), 5, "DISTINCT must dedup on the general path too (2+3), not 2+2+3");
    }
}

// ===========================================================================
// (6) the other window kinds live (and are tested) in their own modules
// ===========================================================================
//
// Every window kind is now implemented: the AGGREGATE kind here, the RANKING kinds
// (`row_number`/`rank`/…) in `ops/window/ranking.rs` (tests in `tests/window_ranking.rs`),
// and the NAVIGATION kinds (`lag`/`first_value`/…) in `ops/window/navigation.rs` (tests in
// `tests/window_navigation.rs`). This file pins the aggregate-over-frame kind and the shared
// frame/partition machinery, so there is no longer a "not yet implemented" stub to assert.
