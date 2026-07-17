//! Integration tests for the compound `SetOp` operator (`UNION ALL` / `UNION` /
//! `INTERSECT` / `EXCEPT`), driven through the real [`Executor`]/[`RowCursor`] seam.
//!
//! The inputs are [`PlanNode::Values`] literal rows, so these pin the OPERATOR's job
//! — concatenation, duplicate elimination, the asymmetric right-set membership of
//! `INTERSECT`/`EXCEPT`, left-first output order, numeric folding, `NULL` dedup, and
//! per-column collation — with no table or storage behind them. A `VALUES` source
//! never reads the pager, so the fixture is just an (unused) empty in-memory database
//! to satisfy `execute`'s catalog+pager parameters.

use minisqlite_btree::init_database;
use minisqlite_catalog::SchemaCatalog;
use minisqlite_exec::{Executor, Runtime, StreamingExecutor};
use minisqlite_expr::EvalExpr;
use minisqlite_pager::MemPager;
use minisqlite_plan::{Plan, PlanNode, SetOp, SetOpKind};
use minisqlite_types::{Collation, Value};
use minisqlite_exec::PagerSet;
use minisqlite_types::DbIndex;

// ----- builders / harness --------------------------------------------------

fn vi(i: i64) -> Value {
    Value::Integer(i)
}

fn vt(s: &str) -> Value {
    Value::Text(s.into())
}

/// A `VALUES` leaf from literal rows (each inner `Vec` is one row of literals).
fn values(rows: Vec<Vec<Value>>) -> PlanNode {
    PlanNode::Values {
        rows: rows
            .into_iter()
            .map(|r| r.into_iter().map(EvalExpr::Literal).collect())
            .collect(),
    }
}

/// A `SetOp` node over two operand subtrees.
fn set_op(op: SetOpKind, left: PlanNode, right: PlanNode, collations: Vec<Collation>) -> PlanNode {
    PlanNode::SetOp(SetOp {
        op,
        left: Box::new(left),
        right: Box::new(right),
        column_collations: collations,
    })
}

/// Run a read plan through the real executor over an empty in-memory database and
/// collect its rows. (`VALUES` inputs never touch the pager, but `execute` still
/// requires a catalog + pager.)
fn run(root: PlanNode, result_columns: &[&str]) -> Vec<Vec<Value>> {
    let mut pager = MemPager::new(4096);
    init_database(&mut pager).unwrap();
    let cat = SchemaCatalog::new();
    let plan = Plan {
        root,
        result_columns: result_columns.iter().map(|s| s.to_string()).collect(),
        ctes: Vec::new(),
        subqueries: Vec::new(),
        mutates: false,
        generated: Vec::new(),
    };
    let mut rt = Runtime::new();
    let mut exec = StreamingExecutor;
    let mut cur = exec.execute(&plan, &cat, PagerSet::One { db: DbIndex::MAIN, pager: &mut pager }).unwrap();
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

fn text(v: &Value) -> &str {
    match v {
        Value::Text(s) => s,
        other => panic!("expected Text, got {other:?}"),
    }
}

/// The single-column integer projection of a result, for order-sensitive asserts.
fn ints(rows: &[Vec<Value>]) -> Vec<i64> {
    rows.iter().map(|r| int(&r[0])).collect()
}

// ----- (1) UNION ALL: concatenate, keep every duplicate --------------------

#[test]
fn union_all_concatenates_and_keeps_duplicates_in_order() {
    // [(1),(2)] UNION ALL [(2),(3)] -> (1),(2),(2),(3): all left rows, then all right,
    // no dedup.
    let root = set_op(
        SetOpKind::UnionAll,
        values(vec![vec![vi(1)], vec![vi(2)]]),
        values(vec![vec![vi(2)], vec![vi(3)]]),
        vec![Collation::Binary],
    );
    let rows = run(root, &["x"]);
    assert_eq!(rows[0].len(), 1, "output width = input width");
    assert_eq!(ints(&rows), vec![1, 2, 2, 3]);
}

#[test]
fn union_all_empty_left_streams_right() {
    // An empty left latches `left_done` on the first pull and falls straight through to
    // the right: [] UNION ALL [(1),(2)] -> (1),(2).
    let root = set_op(
        SetOpKind::UnionAll,
        values(vec![]),
        values(vec![vec![vi(1)], vec![vi(2)]]),
        vec![Collation::Binary],
    );
    assert_eq!(ints(&run(root, &["x"])), vec![1, 2]);
}

// ----- (2) UNION: distinct, left-first order -------------------------------

#[test]
fn union_dedups_keeping_left_first_appearance_order() {
    // [(1),(2),(2)] UNION [(2),(3)] -> (1),(2),(3): each distinct row once, in the
    // order it first appears scanning left then right.
    let root = set_op(
        SetOpKind::Union,
        values(vec![vec![vi(1)], vec![vi(2)], vec![vi(2)]]),
        values(vec![vec![vi(2)], vec![vi(3)]]),
        vec![Collation::Binary],
    );
    let rows = run(root, &["x"]);
    assert_eq!(ints(&rows), vec![1, 2, 3]);
}

// ----- (3) UNION folds numeric-equal values across storage classes ---------

#[test]
fn union_folds_integer_and_real() {
    // [(1)] UNION [(1.0)] -> one row: 1 and 1.0 share a dedup key. The surviving row
    // is the left operand's (Integer), since it appears first.
    let root = set_op(
        SetOpKind::Union,
        values(vec![vec![Value::Integer(1)]]),
        values(vec![vec![Value::Real(1.0)]]),
        vec![Collation::Binary],
    );
    let rows = run(root, &["x"]);
    assert_eq!(rows.len(), 1, "1 and 1.0 dedup to a single row");
    assert!(matches!(rows[0][0], Value::Integer(1)), "left-first Integer(1) survives, got {:?}", rows[0][0]);
}

// ----- (4) INTERSECT: distinct rows in BOTH, left order --------------------

#[test]
fn intersect_keeps_distinct_rows_in_both_left_order() {
    // left [(3),(1),(2),(2)] INTERSECT right [(2),(3),(4)] -> (3),(2): rows present in
    // BOTH, distinct, in LEFT order. The order is DISCRIMINATING — the right lists 2
    // before 3, so an implementation that emitted in right order (or right-set
    // iteration order) would give (2),(3). Also pins exclusion (1 is not in the right)
    // and dedup (the repeated left 2 is emitted once).
    let root = set_op(
        SetOpKind::Intersect,
        values(vec![vec![vi(3)], vec![vi(1)], vec![vi(2)], vec![vi(2)]]),
        values(vec![vec![vi(2)], vec![vi(3)], vec![vi(4)]]),
        vec![Collation::Binary],
    );
    let rows = run(root, &["x"]);
    assert_eq!(ints(&rows), vec![3, 2]);
}

#[test]
fn intersect_with_empty_right_is_empty() {
    // Nothing is in the (empty) right set, so INTERSECT yields no rows.
    let root = set_op(
        SetOpKind::Intersect,
        values(vec![vec![vi(1)], vec![vi(2)]]),
        values(vec![]),
        vec![Collation::Binary],
    );
    assert!(run(root, &["x"]).is_empty());
}

#[test]
fn intersect_with_empty_left_is_empty() {
    // The right is still drained on the first pull, then the empty left yields nothing.
    let root = set_op(
        SetOpKind::Intersect,
        values(vec![]),
        values(vec![vec![vi(1)], vec![vi(2)]]),
        vec![Collation::Binary],
    );
    assert!(run(root, &["x"]).is_empty());
}

#[test]
fn intersect_folds_integer_and_real_across_sides() {
    // [(2)] INTERSECT [(2.0)] -> one row: the numeric fold makes 2 (left) and 2.0
    // (right) the same row, so the left's Integer(2) is in the right key set.
    let root = set_op(
        SetOpKind::Intersect,
        values(vec![vec![Value::Integer(2)]]),
        values(vec![vec![Value::Real(2.0)]]),
        vec![Collation::Binary],
    );
    let rows = run(root, &["x"]);
    assert_eq!(rows.len(), 1);
    assert_eq!(int(&rows[0][0]), 2);
}

// ----- (5) EXCEPT: distinct left rows not in right, left order -------------

#[test]
fn except_keeps_distinct_left_rows_absent_from_right() {
    // [(1),(2),(3),(2)] EXCEPT [(2),(4)] -> (1),(3): left rows whose key is not in the
    // right, distinct, in left order.
    let root = set_op(
        SetOpKind::Except,
        values(vec![vec![vi(1)], vec![vi(2)], vec![vi(3)], vec![vi(2)]]),
        values(vec![vec![vi(2)], vec![vi(4)]]),
        vec![Collation::Binary],
    );
    let rows = run(root, &["x"]);
    assert_eq!(ints(&rows), vec![1, 3]);
}

#[test]
fn except_with_empty_right_returns_distinct_left() {
    // EXCEPT nothing still applies EXCEPT's DISTINCT: [(1),(1),(2)] -> (1),(2).
    let root = set_op(
        SetOpKind::Except,
        values(vec![vec![vi(1)], vec![vi(1)], vec![vi(2)]]),
        values(vec![]),
        vec![Collation::Binary],
    );
    let rows = run(root, &["x"]);
    assert_eq!(ints(&rows), vec![1, 2]);
}

#[test]
fn except_with_empty_left_is_empty() {
    // The right is drained on the first pull; the empty left then yields nothing.
    let root = set_op(
        SetOpKind::Except,
        values(vec![]),
        values(vec![vec![vi(1)]]),
        vec![Collation::Binary],
    );
    assert!(run(root, &["x"]).is_empty());
}

// ----- (6) NULL dedup (self-equal for compound-select) ---------------------

#[test]
fn union_dedups_null_rows() {
    // [(NULL),(1)] UNION [(NULL)] -> (NULL),(1): two NULL rows are the SAME row for
    // compound-select dedup (unlike SQL `=`), so the second NULL collapses.
    let root = set_op(
        SetOpKind::Union,
        values(vec![vec![Value::Null], vec![vi(1)]]),
        values(vec![vec![Value::Null]]),
        vec![Collation::Binary],
    );
    let rows = run(root, &["x"]);
    assert_eq!(rows.len(), 2, "the two NULL rows dedup to one");
    assert!(rows[0][0].is_null(), "the (left-first) NULL row survives");
    assert_eq!(int(&rows[1][0]), 1);
}

#[test]
fn intersect_treats_null_as_a_matchable_row() {
    // NULL is a normal self-equal key for compound-select, so a NULL row on both sides
    // intersects (this is NOT `NULL = NULL`, which SQL leaves unknown).
    let root = set_op(
        SetOpKind::Intersect,
        values(vec![vec![Value::Null], vec![vi(1)]]),
        values(vec![vec![Value::Null], vec![vi(2)]]),
        vec![Collation::Binary],
    );
    let rows = run(root, &["x"]);
    assert_eq!(rows.len(), 1, "only the NULL row is common");
    assert!(rows[0][0].is_null());
}

// ----- (7) per-column collation drives the dedup comparison ----------------

#[test]
fn union_dedup_respects_column_collation() {
    // 'A' UNION 'a' folds to one row under NOCASE but stays two under BINARY.
    let nocase = run(
        set_op(
            SetOpKind::Union,
            values(vec![vec![vt("A")]]),
            values(vec![vec![vt("a")]]),
            vec![Collation::NoCase],
        ),
        &["s"],
    );
    assert_eq!(nocase.len(), 1, "NOCASE folds 'A' and 'a'");
    assert_eq!(text(&nocase[0][0]), "A", "the left-first row survives");

    let binary = run(
        set_op(
            SetOpKind::Union,
            values(vec![vec![vt("A")]]),
            values(vec![vec![vt("a")]]),
            vec![Collation::Binary],
        ),
        &["s"],
    );
    assert_eq!(binary.len(), 2, "BINARY keeps 'A' and 'a' distinct");
}

#[test]
fn union_dedup_applies_collation_per_column() {
    // column_collations = [NOCASE, BINARY]: col 0 folds case, col 1 does NOT.
    //   left  = ('A','x'), ('B','y')
    //   right = ('a','x'), ('a','X'), ('a','y')
    // ('a','x') folds into ('A','x') (col0 NOCASE, col1 equal) and drops. ('a','X') is
    // the DISCRIMINATOR: col0 'a' folds to 'A' under NOCASE but col1 'X' != 'x' under
    // BINARY, so the row SURVIVES — were col1 wrongly folded NOCASE, 'X' would collapse
    // into 'x' and this row would vanish (leaving 3 rows). ('a','y') also survives.
    let root = set_op(
        SetOpKind::Union,
        values(vec![vec![vt("A"), vt("x")], vec![vt("B"), vt("y")]]),
        values(vec![vec![vt("a"), vt("x")], vec![vt("a"), vt("X")], vec![vt("a"), vt("y")]]),
        vec![Collation::NoCase, Collation::Binary],
    );
    let rows = run(root, &["c0", "c1"]);
    assert_eq!(rows.len(), 4, "col1 stays BINARY: 'X' does not fold into 'x'");
    assert_eq!((text(&rows[0][0]), text(&rows[0][1])), ("A", "x"));
    assert_eq!((text(&rows[1][0]), text(&rows[1][1])), ("B", "y"));
    assert_eq!((text(&rows[2][0]), text(&rows[2][1])), ("a", "X"), "col1 BINARY keeps 'X' distinct from 'x'");
    assert_eq!((text(&rows[3][0]), text(&rows[3][1])), ("a", "y"));
}

// ----- (8) multi-column rows dedup on the WHOLE row ------------------------

#[test]
fn union_dedups_on_the_whole_row() {
    // [(1,1),(1,2)] UNION [(1,1)] -> (1,1),(1,2): the right (1,1) matches the left
    // (1,1) on BOTH columns and drops; (1,2) shares only col0 and survives (a key that
    // used only the first column would wrongly merge them).
    let root = set_op(
        SetOpKind::Union,
        values(vec![vec![vi(1), vi(1)], vec![vi(1), vi(2)]]),
        values(vec![vec![vi(1), vi(1)]]),
        vec![Collation::Binary, Collation::Binary],
    );
    let rows = run(root, &["a", "b"]);
    assert_eq!(rows.len(), 2, "only the fully-equal (1,1) row dedups");
    assert_eq!((int(&rows[0][0]), int(&rows[0][1])), (1, 1));
    assert_eq!((int(&rows[1][0]), int(&rows[1][1])), (1, 2));
}
