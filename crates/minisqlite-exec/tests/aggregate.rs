//! Integration tests for the `Aggregate` (GROUP BY / aggregation) operator, driven
//! through the real [`Executor`]/[`RowCursor`] seam over a real [`MemPager`] +
//! [`SchemaCatalog`] seeded with real records.
//!
//! The aggregate *functions* live in a later crate, so these tests define TINY stub
//! [`AggregateFunction`]s inline — a `CountStar` (counts every step), a `SumAgg`
//! (folds numeric args; NULL if it never stepped), and a `ConcatAgg` (order-sensitive,
//! to exercise aggregate `ORDER BY`). The stubs are the reference model: they let the
//! tests pin the OPERATOR's job — grouping, first-appearance order, `DISTINCT`,
//! `FILTER`, aggregate `ORDER BY`, `HAVING`, and the implicit single group over an
//! empty input — independent of any real function library.

use std::sync::Arc;

use minisqlite_btree::{init_database, table_insert};
use minisqlite_catalog::{Catalog, SchemaCatalog};
use minisqlite_exec::{Executor, Runtime, StreamingExecutor};
use minisqlite_expr::{
    AggregateAccumulator, AggregateCall, AggregateFunction, CmpOp, CompareMeta, EvalExpr, FnContext,
    ScalarFunction, SortKey,
};
use minisqlite_fileformat::encode_record;
use minisqlite_pager::{MemPager, PageId, Pager};
use minisqlite_plan::{Aggregate, MinMaxBare, Plan, PlanNode, SeqScan};
use minisqlite_sql::{parse, Statement};
use minisqlite_types::{extremum_wins, Collation, Result, Value};
use minisqlite_exec::PagerSet;
use minisqlite_types::DbIndex;

// ----- stub aggregate functions (the reference model) ----------------------

/// `count(*)` / `count(<arg>)`: the accumulator counts every `step` it is handed,
/// regardless of the argument values. Combined with the operator's `DISTINCT` /
/// `FILTER` it becomes `count(DISTINCT x)` / `count(*) FILTER (WHERE …)`.
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

/// `sum(<arg>)`: folds numeric arguments, promoting to REAL once a real is seen;
/// finalizes to NULL if it never folded a non-NULL value (SQLite's `sum` over all
/// NULLs is NULL), else the Integer/Real sum.
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
                // A non-numeric input contributes 0 to the sum but still counts as a
                // non-NULL input (so finalize is not NULL).
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

/// An order-sensitive aggregate: appends each integer/text argument to a
/// comma-separated string in `step` order. With aggregate `ORDER BY` the operator
/// must feed the rows in sorted order, so the result witnesses that ordering.
#[derive(Debug)]
struct ConcatAgg;

impl AggregateFunction for ConcatAgg {
    fn new_accumulator(&self, _collation: Collation) -> Box<dyn AggregateAccumulator> {
        Box::new(ConcatAcc { s: String::new() })
    }
}

struct ConcatAcc {
    s: String,
}

impl AggregateAccumulator for ConcatAcc {
    fn step(&mut self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<()> {
        for v in args {
            // Render NULL as a literal token so aggregate ORDER BY NULL placement is
            // observable in the concatenated result.
            let piece = match v {
                Value::Integer(i) => i.to_string(),
                Value::Real(r) => r.to_string(),
                Value::Text(t) => t.clone(),
                Value::Null => "NULL".to_string(),
                Value::Blob(_) => continue,
            };
            if !self.s.is_empty() {
                self.s.push(',');
            }
            self.s.push_str(&piece);
        }
        Ok(())
    }
    fn finalize(&mut self, _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(Value::Text(self.s.clone()))
    }
}

/// `min(X)` / `max(X)`: the reference-model extremum accumulator, mirroring the real
/// aggregate (`minisqlite-functions` `agg/minmax.rs`) — a STRICT win under the argument's
/// collation (passed to `new_accumulator`), NULLs skipped, NULL result over an all-NULL or
/// empty group. Used to pair a real min/max result with the operator's single-min/max
/// bare-column capture, which tracks the SAME extremum independently under the SAME
/// collation; the two must agree on which row is the extremum.
#[derive(Debug)]
struct MinMaxAgg {
    is_max: bool,
}

impl AggregateFunction for MinMaxAgg {
    fn new_accumulator(&self, collation: Collation) -> Box<dyn AggregateAccumulator> {
        Box::new(MinMaxAcc { best: None, is_max: self.is_max, collation })
    }
}

struct MinMaxAcc {
    best: Option<Value>,
    is_max: bool,
    collation: Collation,
}

impl AggregateAccumulator for MinMaxAcc {
    fn step(&mut self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<()> {
        let v = &args[0];
        if v.is_null() {
            return Ok(());
        }
        // Through the SAME shared `extremum_wins` predicate (under the SAME collation) the
        // real accumulator and the operator's bare-column capture use, so this reference
        // model can never drift from the code under test.
        let replace = match &self.best {
            None => true,
            Some(cur) => extremum_wins(cur, v, self.is_max, self.collation),
        };
        if replace {
            self.best = Some(v.clone());
        }
        Ok(())
    }
    fn finalize(&mut self, _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(self.best.clone().unwrap_or(Value::Null))
    }
}

// ----- subtype-channel stubs (for the ephemeral JSON value-subtype, json1.html §3.4) ---
//
// The driver threads a per-argument value-subtype from arg evaluation into `step`
// (published via `FnContext::set_arg_subtypes`, read via `arg_subtype`). For SQL scalar
// JSON this is JSON_SUBTYPE=74; the driver treats it as an opaque `u8`, so these stubs use
// the same byte to stay faithful. They exist to pin that the driver publishes the subtype
// before EACH `step` — including the aggregate `ORDER BY` buffered-replay path, which
// carries the subtypes inside the buffered tuple and republishes them on replay. That path
// is UNREACHABLE from SQL today (the parser rejects `agg(expr ORDER BY …)`), so this
// driver-level test is the only place it can be exercised.
const JSON_SUBTYPE: u8 = 74;

/// A scalar that returns its argument unchanged but MARKS its result with `JSON_SUBTYPE` —
/// the stub analogue of `json(x)`, which is what makes an aggregate argument carry a
/// subtype for the driver to capture and thread.
#[derive(Debug)]
struct SubtypeMark;

impl ScalarFunction for SubtypeMark {
    fn call(&self, args: &[Value], ctx: &mut dyn FnContext) -> Result<Value> {
        ctx.set_result_subtype(JSON_SUBTYPE);
        Ok(args.first().cloned().unwrap_or(Value::Null))
    }
}

/// An order-sensitive, subtype-aware witness accumulator: for each `step` it appends the
/// argument to a comma-joined body, EMBEDDING it raw when the driver published
/// `arg_subtype(0) == JSON_SUBTYPE` and QUOTING it (`"…"`) otherwise, then wraps the body
/// in `[]` at finalize — the exact embed/quote decision `json_group_array` makes, but as a
/// reference model so the test does not depend on the functions crate. Because it is
/// order-sensitive AND subtype-sensitive, its result witnesses BOTH that the ORDER BY
/// buffered replay feeds rows in sorted order AND that each replayed `step` sees its own
/// row's republished subtype. Dropping the buffered `Vec<u8>` or the republish would flip
/// an embed to a quote and fail the test.
#[derive(Debug)]
struct SubtypeWitnessAgg;

impl AggregateFunction for SubtypeWitnessAgg {
    fn new_accumulator(&self, _collation: Collation) -> Box<dyn AggregateAccumulator> {
        Box::new(SubtypeWitnessAcc { body: String::new(), first: true })
    }
}

struct SubtypeWitnessAcc {
    body: String,
    first: bool,
}

impl AggregateAccumulator for SubtypeWitnessAcc {
    fn step(&mut self, args: &[Value], ctx: &mut dyn FnContext) -> Result<()> {
        let s = match &args[0] {
            Value::Text(s) => s.clone(),
            other => format!("{other:?}"),
        };
        if !self.first {
            self.body.push(',');
        }
        self.first = false;
        if ctx.arg_subtype(0) == JSON_SUBTYPE {
            self.body.push_str(&s); // embed the subtyped value's structure verbatim
        } else {
            self.body.push('"');
            self.body.push_str(&s);
            self.body.push('"'); // quote the plain (subtype-less) value
        }
        Ok(())
    }
    fn finalize(&mut self, _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(Value::Text(format!("[{}]", self.body)))
    }
}

// ----- fixtures / helpers (copied from tests/exec.rs) ----------------------

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

fn col(i: usize) -> EvalExpr {
    EvalExpr::Column(i)
}

fn lit(v: Value) -> EvalExpr {
    EvalExpr::Literal(v)
}

fn plan(root: PlanNode, columns: &[&str]) -> Plan {
    Plan {
        root,
        result_columns: columns.iter().map(|s| s.to_string()).collect(),
        ctes: Vec::new(),
        subqueries: Vec::new(),
        mutates: false,
        generated: Vec::new(),
    }
}

fn seqscan(table: &str, column_count: usize) -> PlanNode {
    PlanNode::SeqScan(SeqScan { db: DbIndex::MAIN, table: table.to_string(), column_count })
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

// ----- aggregate-call / plan builders --------------------------------------

fn binary_meta() -> CompareMeta {
    CompareMeta { apply_left: None, apply_right: None, collation: Collation::Binary }
}

/// `col > value` (three-valued), used for `HAVING` and `FILTER` predicates.
fn gt(left: EvalExpr, value: i64) -> EvalExpr {
    EvalExpr::Compare {
        op: CmpOp::Gt,
        null_safe: false,
        left: Box::new(left),
        right: Box::new(lit(Value::Integer(value))),
        meta: binary_meta(),
    }
}

fn count_star(distinct: bool, filter: Option<EvalExpr>) -> AggregateCall {
    AggregateCall {
        func: Arc::new(CountStar),
        distinct,
        args: Vec::new(),
        filter,
        order_by: Vec::new(),
        arg_collations: Vec::new(),
    }
}

fn count_of(arg: EvalExpr, distinct: bool) -> AggregateCall {
    AggregateCall {
        func: Arc::new(CountStar),
        distinct,
        args: vec![arg],
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

/// Build an [`Aggregate`] plan node from its parts.
fn agg(
    input: PlanNode,
    group_by: Vec<EvalExpr>,
    group_collations: Vec<Collation>,
    aggregates: Vec<AggregateCall>,
    having: Option<EvalExpr>,
) -> PlanNode {
    PlanNode::Aggregate(Aggregate {
        input: Box::new(input),
        group_by,
        group_collations,
        aggregates,
        having,
        minmax_bare: None,
        bare_arbitrary: None,
    })
}

/// A single-argument `min`/`max` aggregate call (the aggregate form the §2.5 bare-column
/// special case keys on).
fn minmax_of(arg: EvalExpr, is_max: bool) -> AggregateCall {
    AggregateCall {
        func: Arc::new(MinMaxAgg { is_max }),
        distinct: false,
        args: vec![arg],
        filter: None,
        order_by: Vec::new(),
        arg_collations: Vec::new(),
    }
}

/// Build an [`Aggregate`] with the single-min/max bare-column marker set: the operator
/// captures `captured_regs` from the extremum row and appends them after the aggregate
/// results, emitting `[group_keys.., agg_results.., captured..]`.
fn agg_minmax_bare(
    input: PlanNode,
    group_by: Vec<EvalExpr>,
    group_collations: Vec<Collation>,
    aggregate: AggregateCall,
    is_max: bool,
    captured_regs: Vec<usize>,
) -> PlanNode {
    PlanNode::Aggregate(Aggregate {
        input: Box::new(input),
        group_by,
        group_collations,
        aggregates: vec![aggregate],
        having: None,
        minmax_bare: Some(MinMaxBare { agg_index: 0, is_max, captured_regs }),
        bare_arbitrary: None,
    })
}

// ----- (1) GROUP BY: per-group count(*) and sum(a) -------------------------

#[test]
fn group_by_counts_and_sums_in_first_appearance_order() {
    // SELECT b, count(*), sum(a) FROM t GROUP BY b
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER, b INTEGER)");
    let root = table_root(&cat, "t");
    seed(
        &mut pager,
        root,
        &[
            (1, vec![Value::Integer(10), Value::Integer(1)]),
            (2, vec![Value::Integer(20), Value::Integer(1)]),
            (3, vec![Value::Integer(30), Value::Integer(2)]),
            (4, vec![Value::Integer(40), Value::Integer(2)]),
            (5, vec![Value::Integer(50), Value::Integer(3)]),
        ],
    );

    // seqscan emits [a, b, rowid]; group by b = col(1); sum over a = col(0).
    let tree = agg(
        seqscan("t", 2),
        vec![col(1)],
        vec![Collation::Binary],
        vec![count_star(false, None), sum_of(col(0))],
        None,
    );
    let rows = run(&plan(tree, &["b", "n", "s"]), &cat, &mut pager);

    // One row per group, in the order the groups first appeared (b = 1, 2, 3).
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0].len(), 3, "width = G(1) + A(2)");
    assert_eq!((int(&rows[0][0]), int(&rows[0][1]), int(&rows[0][2])), (1, 2, 30));
    assert_eq!((int(&rows[1][0]), int(&rows[1][1]), int(&rows[1][2])), (2, 2, 70));
    assert_eq!((int(&rows[2][0]), int(&rows[2][1]), int(&rows[2][2])), (3, 1, 50));
}

// ----- (2) implicit single group over an EMPTY input -----------------------

#[test]
fn no_group_by_over_empty_input_emits_one_row() {
    // SELECT count(*), sum(a) FROM t  -- t is empty
    // The implicit single group still emits exactly one row: count = 0, sum = NULL.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER)");

    let tree = agg(
        seqscan("t", 1),
        Vec::new(),
        Vec::new(),
        vec![count_star(false, None), sum_of(col(0))],
        None,
    );
    let rows = run(&plan(tree, &["n", "s"]), &cat, &mut pager);

    assert_eq!(rows.len(), 1, "empty input + empty GROUP BY still yields one row");
    assert_eq!(rows[0].len(), 2, "width = G(0) + A(2)");
    assert_eq!(int(&rows[0][0]), 0, "count over zero rows is 0");
    assert!(rows[0][1].is_null(), "sum over zero rows is NULL");
}

#[test]
fn group_by_over_empty_input_emits_no_rows() {
    // With a non-empty GROUP BY, an empty input yields ZERO rows (no group appears).
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER, b INTEGER)");
    let tree = agg(
        seqscan("t", 2),
        vec![col(1)],
        vec![Collation::Binary],
        vec![count_star(false, None)],
        None,
    );
    let rows = run(&plan(tree, &["b", "n"]), &cat, &mut pager);
    assert!(rows.is_empty(), "non-empty GROUP BY over empty input emits nothing");
}

// ----- (3) HAVING drops a group --------------------------------------------

#[test]
fn having_drops_a_group() {
    // SELECT b, count(*) FROM t GROUP BY b HAVING count(*) > 1
    // Groups b=1 (n=2) and b=2 (n=2) survive; b=3 (n=1) is dropped.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER, b INTEGER)");
    let root = table_root(&cat, "t");
    seed(
        &mut pager,
        root,
        &[
            (1, vec![Value::Integer(10), Value::Integer(1)]),
            (2, vec![Value::Integer(20), Value::Integer(1)]),
            (3, vec![Value::Integer(30), Value::Integer(2)]),
            (4, vec![Value::Integer(40), Value::Integer(2)]),
            (5, vec![Value::Integer(50), Value::Integer(3)]),
        ],
    );

    // HAVING binds against the emitted [b, n] row: n is col(1).
    let tree = agg(
        seqscan("t", 2),
        vec![col(1)],
        vec![Collation::Binary],
        vec![count_star(false, None)],
        Some(gt(col(1), 1)),
    );
    let rows = run(&plan(tree, &["b", "n"]), &cat, &mut pager);

    assert_eq!(rows.len(), 2, "the count=1 group is filtered out by HAVING");
    assert_eq!((int(&rows[0][0]), int(&rows[0][1])), (1, 2));
    assert_eq!((int(&rows[1][0]), int(&rows[1][1])), (2, 2));
}

// ----- (4) count(DISTINCT a) -----------------------------------------------

#[test]
fn count_distinct_dedups_arguments() {
    // SELECT count(*), count(DISTINCT a) FROM t
    // a = [1,1,2,2,3]: 5 rows total, 3 distinct values.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER)");
    let root = table_root(&cat, "t");
    seed(
        &mut pager,
        root,
        &[
            (1, vec![Value::Integer(1)]),
            (2, vec![Value::Integer(1)]),
            (3, vec![Value::Integer(2)]),
            (4, vec![Value::Integer(2)]),
            (5, vec![Value::Integer(3)]),
        ],
    );

    let tree = agg(
        seqscan("t", 1),
        Vec::new(),
        Vec::new(),
        vec![count_star(false, None), count_of(col(0), true)],
        None,
    );
    let rows = run(&plan(tree, &["n", "nd"]), &cat, &mut pager);

    assert_eq!(rows.len(), 1);
    assert_eq!(int(&rows[0][0]), 5, "count(*) sees every row");
    assert_eq!(int(&rows[0][1]), 3, "count(DISTINCT a) dedups to three values");
}

#[test]
fn count_distinct_folds_integer_and_real() {
    // count(DISTINCT a) over {2, 2.0, 3}: 2 and 2.0 fold to the same distinct value
    // (cell_key numeric folding), so the distinct count is 2, not 3.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a)");
    let root = table_root(&cat, "t");
    seed(
        &mut pager,
        root,
        &[
            (1, vec![Value::Integer(2)]),
            (2, vec![Value::Real(2.0)]),
            (3, vec![Value::Integer(3)]),
        ],
    );

    let tree = agg(seqscan("t", 1), Vec::new(), Vec::new(), vec![count_of(col(0), true)], None);
    let rows = run(&plan(tree, &["nd"]), &cat, &mut pager);
    assert_eq!(rows.len(), 1);
    assert_eq!(int(&rows[0][0]), 2, "2 and 2.0 are the same distinct value");
}

// ----- (5) count(*) FILTER (WHERE a > 0) -----------------------------------

#[test]
fn count_star_with_filter() {
    // SELECT count(*), count(*) FILTER (WHERE a > 0) FROM t
    // a = [-1, 0, 5, 10, -3]: 5 rows, 2 of them with a > 0.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER)");
    let root = table_root(&cat, "t");
    seed(
        &mut pager,
        root,
        &[
            (1, vec![Value::Integer(-1)]),
            (2, vec![Value::Integer(0)]),
            (3, vec![Value::Integer(5)]),
            (4, vec![Value::Integer(10)]),
            (5, vec![Value::Integer(-3)]),
        ],
    );

    // FILTER binds against the INPUT row [a, rowid]: a is col(0).
    let tree = agg(
        seqscan("t", 1),
        Vec::new(),
        Vec::new(),
        vec![count_star(false, None), count_star(false, Some(gt(col(0), 0)))],
        None,
    );
    let rows = run(&plan(tree, &["n", "nf"]), &cat, &mut pager);

    assert_eq!(rows.len(), 1);
    assert_eq!(int(&rows[0][0]), 5, "unfiltered count sees every row");
    assert_eq!(int(&rows[0][1]), 2, "FILTER keeps only a > 0 (5 and 10)");
}

// ----- (6) aggregate ORDER BY ----------------------------------------------

#[test]
fn aggregate_order_by_feeds_rows_in_sorted_order() {
    // concat(a ORDER BY a) and concat(a ORDER BY a DESC) over a stored out of order.
    // The order-sensitive stub witnesses that the operator sorts before stepping.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER)");
    let root = table_root(&cat, "t");
    seed(
        &mut pager,
        root,
        &[
            (1, vec![Value::Integer(3)]),
            (2, vec![Value::Integer(1)]),
            (3, vec![Value::Integer(2)]),
        ],
    );

    let ordered = |desc: bool| AggregateCall {
        func: Arc::new(ConcatAgg),
        distinct: false,
        args: vec![col(0)],
        filter: None,
        order_by: vec![SortKey { expr: col(0), desc, nulls_first: None, collation: Collation::Binary }],
        arg_collations: Vec::new(),
    };

    let tree = agg(
        seqscan("t", 1),
        Vec::new(),
        Vec::new(),
        vec![ordered(false), ordered(true)],
        None,
    );
    let rows = run(&plan(tree, &["asc", "desc"]), &cat, &mut pager);

    assert_eq!(rows.len(), 1);
    assert_eq!(text(&rows[0][0]), "1,2,3", "ORDER BY a ASC feeds 1,2,3");
    assert_eq!(text(&rows[0][1]), "3,2,1", "ORDER BY a DESC feeds 3,2,1");
}

#[test]
fn aggregate_order_by_places_nulls_by_direction() {
    // NULL order-key placement through the operator: ASC default -> NULL first,
    // DESC default -> NULL last. The stub renders NULL as a token so its position
    // shows in the result (this is the operator-level guard on the shared comparator's
    // NULL branch, which sortkey's unit tests pin directly).
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER)");
    let root = table_root(&cat, "t");
    seed(
        &mut pager,
        root,
        &[
            (1, vec![Value::Integer(3)]),
            (2, vec![Value::Null]),
            (3, vec![Value::Integer(1)]),
            (4, vec![Value::Integer(2)]),
        ],
    );

    let ordered = |desc: bool| AggregateCall {
        func: Arc::new(ConcatAgg),
        distinct: false,
        args: vec![col(0)],
        filter: None,
        order_by: vec![SortKey { expr: col(0), desc, nulls_first: None, collation: Collation::Binary }],
        arg_collations: Vec::new(),
    };

    let tree = agg(
        seqscan("t", 1),
        Vec::new(),
        Vec::new(),
        vec![ordered(false), ordered(true)],
        None,
    );
    let rows = run(&plan(tree, &["asc", "desc"]), &cat, &mut pager);

    assert_eq!(rows.len(), 1);
    assert_eq!(text(&rows[0][0]), "NULL,1,2,3", "ASC default: NULL sorts first");
    assert_eq!(text(&rows[0][1]), "3,2,1,NULL", "DESC default: NULL sorts last");
}

// ----- (7) group ordering / key composition --------------------------------

#[test]
fn group_by_preserves_first_appearance_order_out_of_key_order() {
    // Groups first appear (by rowid) as b = 2, 1, 3 — deliberately NOT ascending key
    // order — so a sorted or hash-iteration implementation would reorder them. The
    // output must be [2, 1, 3], pinning FIRST-APPEARANCE order specifically.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER, b INTEGER)");
    let root = table_root(&cat, "t");
    seed(
        &mut pager,
        root,
        &[
            (1, vec![Value::Integer(10), Value::Integer(2)]),
            (2, vec![Value::Integer(20), Value::Integer(1)]),
            (3, vec![Value::Integer(30), Value::Integer(3)]),
            (4, vec![Value::Integer(40), Value::Integer(1)]),
        ],
    );

    let tree = agg(
        seqscan("t", 2),
        vec![col(1)],
        vec![Collation::Binary],
        vec![count_star(false, None)],
        None,
    );
    let rows = run(&plan(tree, &["b", "n"]), &cat, &mut pager);

    assert_eq!(rows.len(), 3);
    let order: Vec<i64> = rows.iter().map(|r| int(&r[0])).collect();
    assert_eq!(order, vec![2, 1, 3], "groups emit in first-appearance (rowid) order, not sorted");
    assert_eq!(int(&rows[1][1]), 2, "b=1 (2nd group) counts its two rows");
}

#[test]
fn group_by_collapses_null_keys_into_one_group() {
    // NULL group keys group together (one NULL group), distinct from the non-NULL key.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER, b INTEGER)");
    let root = table_root(&cat, "t");
    seed(
        &mut pager,
        root,
        &[
            (1, vec![Value::Integer(10), Value::Null]),
            (2, vec![Value::Integer(20), Value::Integer(1)]),
            (3, vec![Value::Integer(30), Value::Null]),
            (4, vec![Value::Integer(40), Value::Integer(1)]),
        ],
    );

    let tree = agg(
        seqscan("t", 2),
        vec![col(1)],
        vec![Collation::Binary],
        vec![count_star(false, None), sum_of(col(0))],
        None,
    );
    let rows = run(&plan(tree, &["b", "n", "s"]), &cat, &mut pager);

    assert_eq!(rows.len(), 2, "the two NULL rows collapse into one group");
    assert!(rows[0][0].is_null(), "NULL group appears first (first-seen)");
    assert_eq!((int(&rows[0][1]), int(&rows[0][2])), (2, 40), "NULL group: 2 rows, sum 10+30");
    assert_eq!((int(&rows[1][0]), int(&rows[1][1]), int(&rows[1][2])), (1, 2, 60));
}

#[test]
fn multi_column_group_by_composes_the_key() {
    // GROUP BY b, c: rows sharing BOTH b and c collapse; differing on either split.
    // A key that used only the first column would wrongly merge (1,1) with (1,2).
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER, b INTEGER, c INTEGER)");
    let root = table_root(&cat, "t");
    seed(
        &mut pager,
        root,
        &[
            (1, vec![Value::Integer(10), Value::Integer(1), Value::Integer(1)]),
            (2, vec![Value::Integer(20), Value::Integer(1), Value::Integer(2)]),
            (3, vec![Value::Integer(30), Value::Integer(1), Value::Integer(1)]),
            (4, vec![Value::Integer(40), Value::Integer(2), Value::Integer(1)]),
        ],
    );

    // seqscan emits [a, b, c, rowid]; group by (b, c) = (col(1), col(2)).
    let tree = agg(
        seqscan("t", 3),
        vec![col(1), col(2)],
        vec![Collation::Binary, Collation::Binary],
        vec![count_star(false, None), sum_of(col(0))],
        None,
    );
    let rows = run(&plan(tree, &["b", "c", "n", "s"]), &cat, &mut pager);

    // Groups in first-appearance order: (1,1), (1,2), (2,1).
    assert_eq!(rows.len(), 3, "three distinct (b,c) pairs");
    assert_eq!(rows[0].len(), 4, "width = G(2) + A(2)");
    assert_eq!((int(&rows[0][0]), int(&rows[0][1]), int(&rows[0][2]), int(&rows[0][3])), (1, 1, 2, 40));
    assert_eq!((int(&rows[1][0]), int(&rows[1][1]), int(&rows[1][2]), int(&rows[1][3])), (1, 2, 1, 20));
    assert_eq!((int(&rows[2][0]), int(&rows[2][1]), int(&rows[2][2]), int(&rows[2][3])), (2, 1, 1, 40));
}

#[test]
fn having_drops_a_non_last_group() {
    // HAVING count(*) > 1 drops the FIRST group (b=1 has one row) while keeping later
    // ones — exercises the `continue` skip on a non-terminal group, not just the tail.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER, b INTEGER)");
    let root = table_root(&cat, "t");
    seed(
        &mut pager,
        root,
        &[
            (1, vec![Value::Integer(10), Value::Integer(1)]),
            (2, vec![Value::Integer(20), Value::Integer(2)]),
            (3, vec![Value::Integer(30), Value::Integer(2)]),
            (4, vec![Value::Integer(40), Value::Integer(3)]),
            (5, vec![Value::Integer(50), Value::Integer(3)]),
        ],
    );

    let tree = agg(
        seqscan("t", 2),
        vec![col(1)],
        vec![Collation::Binary],
        vec![count_star(false, None)],
        Some(gt(col(1), 1)),
    );
    let rows = run(&plan(tree, &["b", "n"]), &cat, &mut pager);

    assert_eq!(rows.len(), 2, "the first group (b=1, count 1) is dropped");
    assert_eq!((int(&rows[0][0]), int(&rows[0][1])), (2, 2));
    assert_eq!((int(&rows[1][0]), int(&rows[1][1])), (3, 2));
}

// ----- (8) aggregate ORDER BY corners (explicit NULLS, per-group, +DISTINCT) --

#[test]
fn aggregate_order_by_explicit_nulls_first_last_override_default() {
    // Operator-level guard that an explicit NULLS FIRST/LAST beats the direction
    // default (ASC would default NULL first; DESC would default NULL last). We flip
    // both, so a copy that ignored `nulls_first` would misplace the NULL.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER)");
    let root = table_root(&cat, "t");
    seed(
        &mut pager,
        root,
        &[
            (1, vec![Value::Integer(3)]),
            (2, vec![Value::Null]),
            (3, vec![Value::Integer(1)]),
            (4, vec![Value::Integer(2)]),
        ],
    );

    let with_nulls = |desc: bool, nulls_first: bool| AggregateCall {
        func: Arc::new(ConcatAgg),
        distinct: false,
        args: vec![col(0)],
        filter: None,
        order_by: vec![SortKey {
            expr: col(0),
            desc,
            nulls_first: Some(nulls_first),
            collation: Collation::Binary,
        }],
        arg_collations: Vec::new(),
    };

    let tree = agg(
        seqscan("t", 1),
        Vec::new(),
        Vec::new(),
        // ASC but NULLS LAST (opposite of the ASC default); DESC but NULLS FIRST.
        vec![with_nulls(false, false), with_nulls(true, true)],
        None,
    );
    let rows = run(&plan(tree, &["asc_nl", "desc_nf"]), &cat, &mut pager);

    assert_eq!(rows.len(), 1);
    assert_eq!(text(&rows[0][0]), "1,2,3,NULL", "ASC + explicit NULLS LAST puts NULL at the end");
    assert_eq!(text(&rows[0][1]), "NULL,3,2,1", "DESC + explicit NULLS FIRST puts NULL at the front");
}

#[test]
fn aggregate_order_by_sorts_each_group_buffer_independently() {
    // Under a real multi-group GROUP BY, each group's aggregate ORDER BY buffer is
    // sorted on its own: b=1 sees a in {30,10,20}, b=2 sees {10,20}, each concatenated
    // in its own ascending order. A shared/global buffer would cross-contaminate.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER, b INTEGER)");
    let root = table_root(&cat, "t");
    seed(
        &mut pager,
        root,
        &[
            (1, vec![Value::Integer(30), Value::Integer(1)]),
            (2, vec![Value::Integer(10), Value::Integer(2)]),
            (3, vec![Value::Integer(10), Value::Integer(1)]),
            (4, vec![Value::Integer(20), Value::Integer(2)]),
            (5, vec![Value::Integer(20), Value::Integer(1)]),
        ],
    );

    let concat_a = AggregateCall {
        func: Arc::new(ConcatAgg),
        distinct: false,
        args: vec![col(0)],
        filter: None,
        order_by: vec![SortKey { expr: col(0), desc: false, nulls_first: None, collation: Collation::Binary }],
        arg_collations: Vec::new(),
    };
    let tree = agg(
        seqscan("t", 2),
        vec![col(1)],
        vec![Collation::Binary],
        vec![concat_a],
        None,
    );
    let rows = run(&plan(tree, &["b", "cat"]), &cat, &mut pager);

    assert_eq!(rows.len(), 2);
    assert_eq!((int(&rows[0][0]), text(&rows[0][1])), (1, "10,20,30"), "b=1 sorts its own a's");
    assert_eq!((int(&rows[1][0]), text(&rows[1][1])), (2, "10,20"), "b=2 sorts its own a's");
}

#[test]
fn aggregate_distinct_then_order_by_dedups_before_sorting() {
    // DISTINCT keeps one copy of each argument tuple (first appearance), THEN aggregate
    // ORDER BY sorts the retained tuples: a = 3,1,3,2,1 -> distinct {3,1,2} -> sorted ->
    // "1,2,3". Skipping DISTINCT would give "1,1,2,3,3"; skipping the sort would leave
    // dedup order "3,1,2". So this pins the DISTINCT-then-ORDER-BY interaction.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(a INTEGER)");
    let root = table_root(&cat, "t");
    seed(
        &mut pager,
        root,
        &[
            (1, vec![Value::Integer(3)]),
            (2, vec![Value::Integer(1)]),
            (3, vec![Value::Integer(3)]),
            (4, vec![Value::Integer(2)]),
            (5, vec![Value::Integer(1)]),
        ],
    );

    let distinct_ordered = AggregateCall {
        func: Arc::new(ConcatAgg),
        distinct: true,
        args: vec![col(0)],
        filter: None,
        order_by: vec![SortKey { expr: col(0), desc: false, nulls_first: None, collation: Collation::Binary }],
        arg_collations: Vec::new(),
    };
    let tree = agg(
        seqscan("t", 1),
        Vec::new(),
        Vec::new(),
        vec![distinct_ordered],
        None,
    );
    let rows = run(&plan(tree, &["cat"]), &cat, &mut pager);

    assert_eq!(rows.len(), 1);
    assert_eq!(text(&rows[0][0]), "1,2,3", "DISTINCT dedups to three values, then ORDER BY a sorts them");
}

// ----- (9) aggregate ORDER BY threads the ephemeral JSON value-subtype ------
//
// The aggregate ORDER BY buffered-replay path carries a `Vec<u8>` of per-argument
// subtypes inside each buffered tuple and republishes it (`set_arg_subtypes`) before each
// replayed `step`, so a JSON aggregate embeds a subtyped operand in the ORDER BY path
// exactly as in the immediate fold. This is a DISTINCT code path from the immediate fold,
// and — because the SQL parser does not accept `agg(expr ORDER BY …)` today — it is
// unreachable from a query, so these driver-level tests are its only coverage. The witness
// accumulator embeds-or-quotes off the republished subtype, so a regression that dropped
// the buffered `Vec<u8>` (or the replay republish) would flip the embedded array to a
// quoted one and fail here while every SQL-level test stayed green.

#[test]
fn aggregate_order_by_replays_arg_subtypes_before_each_step() {
    // Arg is `SubtypeMark(x)` (a subtyped value, the stub analogue of `json(x)`), ORDER BY
    // x DESC over rows stored '[1]','[2]'. Sorted DESC feeds '[2]' then '[1]', and each is
    // EMBEDDED because its subtype rode the buffer and was republished before its step:
    // "[[2],[1]]". This is the driver-level mirror of `json_group_array(json(x) ORDER BY
    // x DESC)`, whose real-sqlite result is likewise `[[2],[1]]`.
    let (mut pager, cat) = db_with_table("CREATE TABLE t(x TEXT)");
    let root = table_root(&cat, "t");
    seed(
        &mut pager,
        root,
        &[(1, vec![Value::Text("[1]".into())]), (2, vec![Value::Text("[2]".into())])],
    );

    let call = AggregateCall {
        func: Arc::new(SubtypeWitnessAgg),
        distinct: false,
        args: vec![EvalExpr::Func { func: Arc::new(SubtypeMark), args: vec![col(0)] }],
        filter: None,
        order_by: vec![SortKey { expr: col(0), desc: true, nulls_first: None, collation: Collation::Binary }],
        arg_collations: Vec::new(),
    };
    let tree = agg(seqscan("t", 1), Vec::new(), Vec::new(), vec![call], None);
    let rows = run(&plan(tree, &["arr"]), &cat, &mut pager);

    assert_eq!(rows.len(), 1);
    assert_eq!(
        text(&rows[0][0]),
        "[[2],[1]]",
        "ORDER BY replay must both reorder AND embed each row's republished subtype",
    );
}

#[test]
fn aggregate_order_by_replay_quotes_subtype_less_args() {
    // Paired quote-guard: identical ORDER BY, but the arg is the RAW column `x` (no
    // SubtypeMark), so NO subtype is published — the buffered tuple carries an empty
    // `Vec<u8>`, republished as an empty slice, so `arg_subtype(0)` reads 0 and each
    // reordered element stays QUOTED: `["[2]","[1]"]`. Mirrors `json_group_array(x ORDER
    // BY x DESC)`. Together with the embed test above this defeats both a subtype-drop
    // (would quote the subtyped case) and a stale/forged-subtype leak (would embed here).
    let (mut pager, cat) = db_with_table("CREATE TABLE t(x TEXT)");
    let root = table_root(&cat, "t");
    seed(
        &mut pager,
        root,
        &[(1, vec![Value::Text("[1]".into())]), (2, vec![Value::Text("[2]".into())])],
    );

    let call = AggregateCall {
        func: Arc::new(SubtypeWitnessAgg),
        distinct: false,
        args: vec![col(0)],
        filter: None,
        order_by: vec![SortKey { expr: col(0), desc: true, nulls_first: None, collation: Collation::Binary }],
        arg_collations: Vec::new(),
    };
    let tree = agg(seqscan("t", 1), Vec::new(), Vec::new(), vec![call], None);
    let rows = run(&plan(tree, &["arr"]), &cat, &mut pager);

    assert_eq!(rows.len(), 1);
    assert_eq!(
        text(&rows[0][0]),
        "[\"[2]\",\"[1]\"]",
        "no subtype published => each reordered element stays quoted",
    );
}

#[test]
fn aggregate_distinct_order_by_replays_arg_subtypes() {
    // DISTINCT + ORDER BY routes through the SAME buffered-replay path: dedup first (on the
    // arg VALUE via `cell_key`, subtype-independent), then sort, then replay with each
    // survivor's subtype republished. Rows '[1]','[2]','[1]' dedup to {'[1]','[2]'}, ORDER
    // BY x DESC feeds '[2]' then '[1]', embedded: "[[2],[1]]". Pins that the subtype is
    // preserved through the DISTINCT skip (covers `json_group_array(DISTINCT json(x) ORDER
    // BY x)`).
    let (mut pager, cat) = db_with_table("CREATE TABLE t(x TEXT)");
    let root = table_root(&cat, "t");
    seed(
        &mut pager,
        root,
        &[
            (1, vec![Value::Text("[1]".into())]),
            (2, vec![Value::Text("[2]".into())]),
            (3, vec![Value::Text("[1]".into())]),
        ],
    );

    let call = AggregateCall {
        func: Arc::new(SubtypeWitnessAgg),
        distinct: true,
        args: vec![EvalExpr::Func { func: Arc::new(SubtypeMark), args: vec![col(0)] }],
        filter: None,
        order_by: vec![SortKey { expr: col(0), desc: true, nulls_first: None, collation: Collation::Binary }],
        arg_collations: Vec::new(),
    };
    let tree = agg(seqscan("t", 1), Vec::new(), Vec::new(), vec![call], None);
    let rows = run(&plan(tree, &["arr"]), &cat, &mut pager);

    assert_eq!(rows.len(), 1);
    assert_eq!(
        text(&rows[0][0]),
        "[[2],[1]]",
        "DISTINCT dedups on value, then ORDER BY replay embeds each survivor's subtype",
    );
}

// ----- single-min/max bare-column special case (lang_select.html §2.5) ------
//
// With EXACTLY ONE min()/max() aggregate, a bare (non-key, non-aggregate) column takes
// its value from the row that produced the extremum. The operator captures those columns
// (marked by `MinMaxBare`) and appends them after the aggregate results, so a driven
// Aggregate node emits `[group_keys.., agg_results.., captured..]`. These tests pin the
// OPERATOR contract directly — including the tie and all-NULL edges the SQL-level
// conformance tests (which use distinct extrema) never reach.
//
// `seqscan("mm", 3)` emits `[g, v, tag, rowid]`, so g=col(0), v=col(1), tag=reg 2.

fn mm_table() -> (MemPager, SchemaCatalog) {
    db_with_table("CREATE TABLE mm(g TEXT, v INTEGER, tag TEXT)")
}

fn mm_row(g: &str, v: Value, tag: &str) -> Vec<Value> {
    vec![Value::Text(g.into()), v, Value::Text(tag.into())]
}

#[test]
fn minmax_bare_max_captures_tag_from_max_row_per_group() {
    // SELECT g, max(v), tag FROM mm GROUP BY g — bare `tag` from each group's max-v row.
    let (mut pager, cat) = mm_table();
    let root = table_root(&cat, "mm");
    seed(
        &mut pager,
        root,
        &[
            (1, mm_row("a", Value::Integer(1), "lo")),
            (2, mm_row("a", Value::Integer(3), "hi")),
            (3, mm_row("b", Value::Integer(5), "x")),
            (4, mm_row("b", Value::Integer(2), "y")),
        ],
    );
    let tree = agg_minmax_bare(
        seqscan("mm", 3),
        vec![col(0)],
        vec![Collation::Binary],
        minmax_of(col(1), true),
        true,
        vec![2],
    );
    // Output layout: [g (key), max(v), captured tag].
    let rows = run(&plan(tree, &["g", "mx", "tag"]), &cat, &mut pager);
    assert_eq!(rows.len(), 2);
    assert_eq!((text(&rows[0][0]), int(&rows[0][1]), text(&rows[0][2])), ("a", 3, "hi"));
    assert_eq!((text(&rows[1][0]), int(&rows[1][1]), text(&rows[1][2])), ("b", 5, "x"));
}

#[test]
fn minmax_bare_min_captures_tag_from_min_row_per_group() {
    let (mut pager, cat) = mm_table();
    let root = table_root(&cat, "mm");
    seed(
        &mut pager,
        root,
        &[
            (1, mm_row("a", Value::Integer(1), "lo")),
            (2, mm_row("a", Value::Integer(3), "hi")),
            (3, mm_row("b", Value::Integer(5), "x")),
            (4, mm_row("b", Value::Integer(2), "y")),
        ],
    );
    let tree = agg_minmax_bare(
        seqscan("mm", 3),
        vec![col(0)],
        vec![Collation::Binary],
        minmax_of(col(1), false),
        false,
        vec![2],
    );
    let rows = run(&plan(tree, &["g", "mn", "tag"]), &cat, &mut pager);
    assert_eq!(rows.len(), 2);
    assert_eq!((text(&rows[0][0]), int(&rows[0][1]), text(&rows[0][2])), ("a", 1, "lo"));
    assert_eq!((text(&rows[1][0]), int(&rows[1][1]), text(&rows[1][2])), ("b", 2, "y"));
}

#[test]
fn minmax_bare_tie_keeps_first_seen_row() {
    // Two rows tie on the max value; §2.5 reports the bare value of ONE such row, and the
    // operator (strict-win, mirroring the accumulator) keeps the FIRST-seen extremum.
    let (mut pager, cat) = mm_table();
    let root = table_root(&cat, "mm");
    seed(
        &mut pager,
        root,
        &[
            (1, mm_row("a", Value::Integer(5), "first")),
            (2, mm_row("a", Value::Integer(5), "second")),
        ],
    );
    let tree = agg_minmax_bare(
        seqscan("mm", 3),
        vec![col(0)],
        vec![Collation::Binary],
        minmax_of(col(1), true),
        true,
        vec![2],
    );
    let rows = run(&plan(tree, &["g", "mx", "tag"]), &cat, &mut pager);
    assert_eq!(rows.len(), 1);
    assert_eq!((text(&rows[0][0]), int(&rows[0][1]), text(&rows[0][2])), ("a", 5, "first"));
}

#[test]
fn minmax_bare_all_null_extremum_column_yields_null_bare() {
    // A group whose extremum column is entirely NULL has no extremum row: max(v) is NULL
    // and the bare column emits NULL (no row was ever captured).
    let (mut pager, cat) = mm_table();
    let root = table_root(&cat, "mm");
    seed(
        &mut pager,
        root,
        &[(1, mm_row("a", Value::Null, "x")), (2, mm_row("a", Value::Null, "y"))],
    );
    let tree = agg_minmax_bare(
        seqscan("mm", 3),
        vec![col(0)],
        vec![Collation::Binary],
        minmax_of(col(1), true),
        true,
        vec![2],
    );
    let rows = run(&plan(tree, &["g", "mx", "tag"]), &cat, &mut pager);
    assert_eq!(rows.len(), 1);
    assert_eq!(text(&rows[0][0]), "a");
    assert!(matches!(rows[0][1], Value::Null), "max over an all-NULL group is NULL");
    assert!(matches!(rows[0][2], Value::Null), "bare column of an extremum-less group is NULL");
}

#[test]
fn minmax_bare_skips_nulls_and_captures_from_extremum() {
    // NULL extremum values are skipped and never captured; the bare column comes from the
    // non-NULL max row.
    let (mut pager, cat) = mm_table();
    let root = table_root(&cat, "mm");
    seed(
        &mut pager,
        root,
        &[
            (1, mm_row("a", Value::Null, "skip")),
            (2, mm_row("a", Value::Integer(7), "keep")),
            (3, mm_row("a", Value::Integer(3), "lo")),
        ],
    );
    let tree = agg_minmax_bare(
        seqscan("mm", 3),
        vec![col(0)],
        vec![Collation::Binary],
        minmax_of(col(1), true),
        true,
        vec![2],
    );
    let rows = run(&plan(tree, &["g", "mx", "tag"]), &cat, &mut pager);
    assert_eq!(rows.len(), 1);
    assert_eq!((text(&rows[0][0]), int(&rows[0][1]), text(&rows[0][2])), ("a", 7, "keep"));
}

#[test]
fn minmax_bare_no_group_by_over_whole_input() {
    // No GROUP BY: the special case applies over the whole set. Layout is
    // [agg_result, captured..] (no group keys), so the bare tag lands at reg 1.
    let (mut pager, cat) = mm_table();
    let root = table_root(&cat, "mm");
    seed(
        &mut pager,
        root,
        &[
            (1, mm_row("a", Value::Integer(1), "lo")),
            (2, mm_row("b", Value::Integer(9), "top")),
            (3, mm_row("c", Value::Integer(5), "mid")),
        ],
    );
    let tree = agg_minmax_bare(
        seqscan("mm", 3),
        Vec::new(),
        Vec::new(),
        minmax_of(col(1), true),
        true,
        vec![2],
    );
    let rows = run(&plan(tree, &["mx", "tag"]), &cat, &mut pager);
    assert_eq!(rows.len(), 1);
    assert_eq!((int(&rows[0][0]), text(&rows[0][1])), (9, "top"));
}

#[test]
fn minmax_bare_captures_when_minmax_is_not_the_first_aggregate() {
    // Widened §2.5 (lang_select.html): the sole min/max may sit ALONGSIDE other
    // aggregates — here `count(*), max(v)` — so the marker's `agg_index` is 1, not 0. The
    // operator keys the capture off `agg_index` and appends captured columns after ALL
    // results, so the bare tag lands at `num_keys(1) + num_aggregates(2) = 3`. Also
    // exercises the debug-only capture-vs-finalized-extremum assertion in `build`.
    let (mut pager, cat) = mm_table();
    let root = table_root(&cat, "mm");
    seed(
        &mut pager,
        root,
        &[
            (1, mm_row("a", Value::Integer(1), "lo")),
            (2, mm_row("a", Value::Integer(3), "hi")),
            (3, mm_row("b", Value::Integer(5), "x")),
            (4, mm_row("b", Value::Integer(2), "y")),
        ],
    );
    let tree = PlanNode::Aggregate(Aggregate {
        input: Box::new(seqscan("mm", 3)),
        group_by: vec![col(0)],
        group_collations: vec![Collation::Binary],
        aggregates: vec![count_star(false, None), minmax_of(col(1), true)],
        having: None,
        minmax_bare: Some(MinMaxBare { agg_index: 1, is_max: true, captured_regs: vec![2] }),
        bare_arbitrary: None,
    });
    // Output layout: [g (key), count(*), max(v), captured tag].
    let rows = run(&plan(tree, &["g", "n", "mx", "tag"]), &cat, &mut pager);
    assert_eq!(rows.len(), 2);
    assert_eq!(
        (text(&rows[0][0]), int(&rows[0][1]), int(&rows[0][2]), text(&rows[0][3])),
        ("a", 2, 3, "hi")
    );
    assert_eq!(
        (text(&rows[1][0]), int(&rows[1][1]), int(&rows[1][2]), text(&rows[1][3])),
        ("b", 2, 5, "x")
    );
}
