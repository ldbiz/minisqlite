//! Integration tests for the `Join` operator: hand-built [`Plan`] trees run over a
//! real [`MemPager`] + [`SchemaCatalog`] seeded with real records, driven through the
//! [`Executor`]/[`RowCursor`] seam. They pin the `left ++ right` row layout, the outer
//! NULL-fill semantics, three-valued `on`, and that NestedLoop and Hash agree on the
//! same equijoin — against the actual storage the engine uses, not a mock.

use minisqlite_btree::{init_database, table_insert};
use minisqlite_catalog::{Catalog, SchemaCatalog};
use minisqlite_exec::{Executor, Runtime, StreamingExecutor};
use minisqlite_expr::{CmpOp, CompareMeta, EvalExpr};
use minisqlite_fileformat::encode_record;
use minisqlite_pager::{MemPager, PageId, Pager};
use minisqlite_plan::{
    Join, JoinStrategy, JoinType, Plan, PlanNode, RowidOp, RowidScan, ScanDirection, SeqScan,
};
use minisqlite_sql::{parse, Statement};
use minisqlite_types::{Affinity, Collation, Value};
use minisqlite_exec::PagerSet;
use minisqlite_types::DbIndex;

// ----- fixtures ------------------------------------------------------------

/// A fresh in-memory database with two tables created through the real catalog path.
fn db_with_two_tables(sql_a: &str, sql_b: &str) -> (MemPager, SchemaCatalog) {
    let mut pager = MemPager::new(4096);
    init_database(&mut pager).unwrap();
    let mut cat = SchemaCatalog::new();
    for sql in [sql_a, sql_b] {
        let ast = parse(sql).unwrap();
        let Statement::CreateTable(stmt) = &ast.statements[0] else {
            panic!("not a CREATE TABLE: {sql}");
        };
        cat.create_table(&mut pager, stmt, sql).unwrap();
    }
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

fn v_i(i: i64) -> Value {
    Value::Integer(i)
}

fn v_t(s: &str) -> Value {
    Value::Text(s.into())
}

fn binary_meta() -> CompareMeta {
    CompareMeta { apply_left: None, apply_right: None, collation: Collation::Binary }
}

/// `col(a) = col(b)` under BINARY (no affinity) — the equijoin `on` for the standard
/// tables, bound against the combined 6-wide row (`l.id` at 0, `r.id` at 3).
fn eq(a: usize, b: usize) -> EvalExpr {
    EvalExpr::Compare {
        op: CmpOp::Eq,
        null_safe: false,
        left: Box::new(col(a)),
        right: Box::new(col(b)),
        meta: binary_meta(),
    }
}

/// `col(a) <> col(b)` under BINARY — a residual predicate carried in `on` BEYOND the
/// equijoin key, so a bucket hit whose non-key columns fail it must be dropped.
fn ne(a: usize, b: usize) -> EvalExpr {
    EvalExpr::Compare {
        op: CmpOp::Ne,
        null_safe: false,
        left: Box::new(col(a)),
        right: Box::new(col(b)),
        meta: binary_meta(),
    }
}

/// `col(a) = col(b)` with NUMERIC affinity applied to BOTH operands — a COERCED equality
/// (`'5'` compares equal to `5`). The planner never turns a coerced `=` into a hash key
/// (raw hashing would MISS the coerced matches), so it must survive in the residual `on`;
/// this models exactly such a residual.
fn eq_numeric(a: usize, b: usize) -> EvalExpr {
    EvalExpr::Compare {
        op: CmpOp::Eq,
        null_safe: false,
        left: Box::new(col(a)),
        right: Box::new(col(b)),
        meta: CompareMeta {
            apply_left: Some(Affinity::Numeric),
            apply_right: Some(Affinity::Numeric),
            collation: Collation::Binary,
        },
    }
}

/// `col(a) IS col(b)` — a NULL-safe equality (`null_safe: true`), where `NULL IS NULL` is
/// TRUE. Never a hash key (the hash's NULL-excludes-match path can't reproduce that), so
/// it stays in the residual `on`.
fn is_eq(a: usize, b: usize) -> EvalExpr {
    EvalExpr::Compare {
        op: CmpOp::Eq,
        null_safe: true,
        left: Box::new(col(a)),
        right: Box::new(col(b)),
        meta: binary_meta(),
    }
}

fn seqscan(table: &str, column_count: usize) -> PlanNode {
    PlanNode::SeqScan(SeqScan { db: DbIndex::MAIN, table: table.to_string(), column_count })
}

#[allow(clippy::too_many_arguments)]
fn join_node(
    left: PlanNode,
    left_width: usize,
    right: PlanNode,
    right_width: usize,
    join_type: JoinType,
    on: Option<EvalExpr>,
    strategy: JoinStrategy,
) -> PlanNode {
    PlanNode::Join(Join {
        left: Box::new(left),
        left_width,
        right: Box::new(right),
        right_width,
        join_type,
        on,
        strategy,
    })
}

fn plan(root: PlanNode) -> Plan {
    Plan { root, result_columns: Vec::new(), ctes: Vec::new(), subqueries: Vec::new(), mutates: false, generated: Vec::new() }
}

/// The Hash strategy for the standard tables: key on column 0 of each side under
/// BINARY (`l.id` in the left row, `r.id` in the right row).
fn hash_strategy() -> JoinStrategy {
    JoinStrategy::Hash {
        left_keys: vec![col(0)],
        right_keys: vec![col(0)],
        key_collations: vec![Collation::Binary],
    }
}

/// A TWO-key Hash strategy: key on columns 0 and 1 of each side (both under BINARY). The
/// planner emits this for `a.k1=b.k1 AND a.k2=b.k2`, stripping BOTH equi-keys from `on`.
fn hash_strategy_2key() -> JoinStrategy {
    JoinStrategy::Hash {
        left_keys: vec![col(0), col(1)],
        right_keys: vec![col(0), col(1)],
        key_collations: vec![Collation::Binary, Collation::Binary],
    }
}

// ----- comparable normalization (order-independent set assertions) ---------

/// Render one value to a comparable, storage-class-tagged token so a `Vec<Value>`
/// result set can be sorted and compared as a set (`i2` != `t2` != `NULL`).
fn cell(v: &Value) -> String {
    match v {
        Value::Null => "NULL".to_string(),
        Value::Integer(i) => format!("i{i}"),
        Value::Real(r) => format!("r{r}"),
        Value::Text(s) => format!("t{s}"),
        Value::Blob(b) => format!("b{b:?}"),
    }
}

fn norm_sorted(rows: &[Vec<Value>]) -> Vec<Vec<String>> {
    let mut v: Vec<Vec<String>> = rows.iter().map(|r| r.iter().map(cell).collect()).collect();
    v.sort();
    v
}

/// Two tables `l(id,name)` and `r(id,val)` seeded so that `l.id ∈ {1,2,3}` and
/// `r.id ∈ {2,3,5}` — `l.id=1` has no right match, `r.id=5` no left match, and
/// `{2,3}` match. Base-table scans emit width 3 (`[col0, col1, rowid]`), so `WL=WR=3`.
fn standard_lr() -> (MemPager, SchemaCatalog) {
    let (mut pager, cat) = db_with_two_tables(
        "CREATE TABLE l(id INTEGER, name TEXT)",
        "CREATE TABLE r(id INTEGER, val TEXT)",
    );
    seed(
        &mut pager,
        table_root(&cat, "l"),
        &[
            (1, vec![v_i(1), v_t("a")]),
            (2, vec![v_i(2), v_t("b")]),
            (3, vec![v_i(3), v_t("c")]),
        ],
    );
    seed(
        &mut pager,
        table_root(&cat, "r"),
        &[
            (1, vec![v_i(2), v_t("x")]),
            (2, vec![v_i(3), v_t("y")]),
            (3, vec![v_i(5), v_t("z")]),
        ],
    );
    (pager, cat)
}

// ----- (1) INNER equijoin: NestedLoop and Hash agree -----------------------

#[test]
fn inner_equijoin_nested_loop_and_hash_agree() {
    let (mut pager, cat) = standard_lr();

    // Matches: l.id=2 -> r.id=2 (b,x); l.id=3 -> r.id=3 (c,y). Combined layout is
    // [l.id, l.name, l.rowid, r.id, r.val, r.rowid].
    let expected = vec![
        vec![v_i(2), v_t("b"), v_i(2), v_i(2), v_t("x"), v_i(1)],
        vec![v_i(3), v_t("c"), v_i(3), v_i(3), v_t("y"), v_i(2)],
    ];

    let nl = run(
        &plan(join_node(
            seqscan("l", 2),
            3,
            seqscan("r", 2),
            3,
            JoinType::Inner,
            Some(eq(0, 3)),
            JoinStrategy::NestedLoop,
        )),
        &cat,
        &mut pager,
    );
    let hash = run(
        &plan(join_node(
            seqscan("l", 2),
            3,
            seqscan("r", 2),
            3,
            JoinType::Inner,
            Some(eq(0, 3)),
            hash_strategy(),
        )),
        &cat,
        &mut pager,
    );

    assert_eq!(nl.len(), 2, "inner equijoin keeps only the two matched pairs");
    assert!(nl.iter().all(|r| r.len() == 6), "combined width is WL+WR = 6");
    assert_eq!(norm_sorted(&nl), norm_sorted(&expected));
    assert_eq!(norm_sorted(&hash), norm_sorted(&expected));
    assert_eq!(norm_sorted(&nl), norm_sorted(&hash), "NestedLoop and Hash must agree");
}

// ----- (2) LEFT join: unmatched left row gets a NULL-filled right half ------

#[test]
fn left_join_null_fills_unmatched_right_half() {
    let (mut pager, cat) = standard_lr();

    let rows = run(
        &plan(join_node(
            seqscan("l", 2),
            3,
            seqscan("r", 2),
            3,
            JoinType::Left,
            Some(eq(0, 3)),
            JoinStrategy::NestedLoop,
        )),
        &cat,
        &mut pager,
    );

    // Every left row survives; l.id=1 matched nothing so its right half is all NULL.
    let expected = vec![
        vec![v_i(1), v_t("a"), v_i(1), Value::Null, Value::Null, Value::Null],
        vec![v_i(2), v_t("b"), v_i(2), v_i(2), v_t("x"), v_i(1)],
        vec![v_i(3), v_t("c"), v_i(3), v_i(3), v_t("y"), v_i(2)],
    ];
    assert_eq!(rows.len(), 3);
    assert!(rows.iter().all(|r| r.len() == 6));
    assert_eq!(norm_sorted(&rows), norm_sorted(&expected));

    // The unmatched left row's right half (registers 3..6) is entirely NULL.
    let unmatched = rows.iter().find(|r| cell(&r[0]) == "i1").expect("l.id=1 row present");
    assert!(unmatched[3..6].iter().all(Value::is_null), "right half NULL-filled");
}

// ----- (3) CROSS join: full product, width WL+WR ---------------------------

#[test]
fn cross_join_is_the_full_product() {
    let (mut pager, cat) = standard_lr();

    let rows = run(
        &plan(join_node(
            seqscan("l", 2),
            3,
            seqscan("r", 2),
            3,
            JoinType::Cross,
            None, // a CROSS carries no ON and keeps every pair
            JoinStrategy::NestedLoop,
        )),
        &cat,
        &mut pager,
    );

    assert_eq!(rows.len(), 9, "3 left x 3 right = 9 pairs");
    assert!(rows.iter().all(|r| r.len() == 6), "each pair is width WL+WR = 6");
    // Every (left.id, right.id) combination appears exactly once.
    let mut pairs: Vec<(i64, i64)> = rows
        .iter()
        .map(|r| (as_int(&r[0]), as_int(&r[3])))
        .collect();
    pairs.sort();
    let mut expected: Vec<(i64, i64)> =
        [1, 2, 3].iter().flat_map(|&l| [2, 3, 5].iter().map(move |&r| (l, r))).collect();
    expected.sort();
    assert_eq!(pairs, expected);
}

// ----- (4) NULL join key never matches (Hash) ------------------------------

#[test]
fn hash_null_join_key_does_not_match() {
    // l has a NULL-id row (rowid 2); it must not equijoin to anything.
    let (mut pager, cat) = db_with_two_tables(
        "CREATE TABLE l(id INTEGER, name TEXT)",
        "CREATE TABLE r(id INTEGER, val TEXT)",
    );
    seed(
        &mut pager,
        table_root(&cat, "l"),
        &[
            (1, vec![v_i(1), v_t("a")]),
            (2, vec![Value::Null, v_t("n")]), // NULL key
            (3, vec![v_i(2), v_t("b")]),
        ],
    );
    seed(
        &mut pager,
        table_root(&cat, "r"),
        &[(1, vec![v_i(2), v_t("x")]), (2, vec![v_i(9), v_t("z")])],
    );

    let rows = run(
        &plan(join_node(
            seqscan("l", 2),
            3,
            seqscan("r", 2),
            3,
            JoinType::Inner,
            Some(eq(0, 3)),
            hash_strategy(),
        )),
        &cat,
        &mut pager,
    );

    // Only l.id=2 <-> r.id=2 matches; the NULL-id row and l.id=1 match nothing.
    let expected = vec![vec![v_i(2), v_t("b"), v_i(3), v_i(2), v_t("x"), v_i(1)]];
    assert_eq!(norm_sorted(&rows), norm_sorted(&expected));
    assert!(
        rows.iter().all(|r| cell(&r[1]) != "tn"),
        "the NULL-key left row must never appear in an inner join"
    );
}

// ----- (5) FULL join emits unmatched rows of BOTH sides, NULL-filled -------

#[test]
fn full_join_emits_unmatched_both_sides() {
    let (mut pager, cat) = standard_lr();

    let rows = run(
        &plan(join_node(
            seqscan("l", 2),
            3,
            seqscan("r", 2),
            3,
            JoinType::Full,
            Some(eq(0, 3)),
            JoinStrategy::NestedLoop,
        )),
        &cat,
        &mut pager,
    );

    let expected = vec![
        vec![v_i(2), v_t("b"), v_i(2), v_i(2), v_t("x"), v_i(1)], // match
        vec![v_i(3), v_t("c"), v_i(3), v_i(3), v_t("y"), v_i(2)], // match
        vec![v_i(1), v_t("a"), v_i(1), Value::Null, Value::Null, Value::Null], // left-only
        vec![Value::Null, Value::Null, Value::Null, v_i(5), v_t("z"), v_i(3)], // right-only
    ];
    assert_eq!(rows.len(), 4);
    assert!(rows.iter().all(|r| r.len() == 6));
    assert_eq!(norm_sorted(&rows), norm_sorted(&expected));
}

// ----- (5b) RIGHT join via Hash: unmatched right row NULL-filled -----------

#[test]
fn right_join_hash_emits_unmatched_right() {
    let (mut pager, cat) = standard_lr();

    let rows = run(
        &plan(join_node(
            seqscan("l", 2),
            3,
            seqscan("r", 2),
            3,
            JoinType::Right,
            Some(eq(0, 3)),
            hash_strategy(),
        )),
        &cat,
        &mut pager,
    );

    // Every right row survives; r.id=5 matched no left so its left half is NULL. The
    // unmatched LEFT row (l.id=1) is NOT emitted by a RIGHT join.
    let expected = vec![
        vec![v_i(2), v_t("b"), v_i(2), v_i(2), v_t("x"), v_i(1)],
        vec![v_i(3), v_t("c"), v_i(3), v_i(3), v_t("y"), v_i(2)],
        vec![Value::Null, Value::Null, Value::Null, v_i(5), v_t("z"), v_i(3)],
    ];
    assert_eq!(rows.len(), 3);
    assert_eq!(norm_sorted(&rows), norm_sorted(&expected));
}

// ----- (6) Hash FULL: NULL build-side key still emitted via outer path ------

#[test]
fn hash_full_join_null_build_key_still_emitted() {
    // r has a NULL-id row: excluded from the hash table (unmatchable) but a FULL join
    // must still emit it NULL-filled on the left.
    let (mut pager, cat) = db_with_two_tables(
        "CREATE TABLE l(id INTEGER, name TEXT)",
        "CREATE TABLE r(id INTEGER, val TEXT)",
    );
    seed(
        &mut pager,
        table_root(&cat, "l"),
        &[(1, vec![v_i(1), v_t("a")]), (2, vec![v_i(2), v_t("b")])],
    );
    seed(
        &mut pager,
        table_root(&cat, "r"),
        &[(1, vec![v_i(2), v_t("x")]), (2, vec![Value::Null, v_t("n")])],
    );

    let rows = run(
        &plan(join_node(
            seqscan("l", 2),
            3,
            seqscan("r", 2),
            3,
            JoinType::Full,
            Some(eq(0, 3)),
            hash_strategy(),
        )),
        &cat,
        &mut pager,
    );

    let expected = vec![
        vec![v_i(2), v_t("b"), v_i(2), v_i(2), v_t("x"), v_i(1)], // match
        vec![v_i(1), v_t("a"), v_i(1), Value::Null, Value::Null, Value::Null], // left-only
        vec![Value::Null, Value::Null, Value::Null, Value::Null, v_t("n"), v_i(2)], // right-only (NULL key)
    ];
    assert_eq!(rows.len(), 3);
    assert_eq!(norm_sorted(&rows), norm_sorted(&expected));
}

// ----- (7) IndexNestedLoop: per-left seek drives the right -----------------

/// Build `l(k INTEGER)` and `r(v TEXT)` where the right is reached by a rowid seek
/// keyed off the left's `k` (an index-nested-loop shape). The right leaf is built with
/// `outer = &L`, so each right row it yields is already `L ++ right_local`.
fn index_nl_db() -> (MemPager, SchemaCatalog) {
    let (mut pager, cat) =
        db_with_two_tables("CREATE TABLE l(k INTEGER)", "CREATE TABLE r(v TEXT)");
    seed(
        &mut pager,
        table_root(&cat, "l"),
        &[(1, vec![v_i(10)]), (2, vec![v_i(20)]), (3, vec![v_i(99)])],
    );
    seed(
        &mut pager,
        table_root(&cat, "r"),
        &[(10, vec![v_t("ten")]), (20, vec![v_t("twenty")]), (30, vec![v_t("thirty")])],
    );
    (pager, cat)
}

fn index_nl_right() -> PlanNode {
    // Seek r by rowid = col(0) of the outer (= the left row's k).
    PlanNode::RowidScan(RowidScan { db: DbIndex::MAIN,
        table: "r".into(),
        column_count: 1,
        op: RowidOp::Eq(col(0)),
        direction: ScanDirection::Forward,
    })
}

#[test]
fn index_nested_loop_inner_seeks_per_left() {
    let (mut pager, cat) = index_nl_db();
    // WL = 2 ([k, l.rowid]); WR = 2 ([v, r.rowid]); combined width 4.
    let rows = run(
        &plan(join_node(
            seqscan("l", 1),
            2,
            index_nl_right(),
            2,
            JoinType::Inner,
            None, // the seek IS the equality; no residual predicate
            JoinStrategy::IndexNestedLoop,
        )),
        &cat,
        &mut pager,
    );

    // k=10 -> r.rowid 10 ("ten"); k=20 -> 20 ("twenty"); k=99 -> absent (dropped).
    let expected = vec![
        vec![v_i(10), v_i(1), v_t("ten"), v_i(10)],
        vec![v_i(20), v_i(2), v_t("twenty"), v_i(20)],
    ];
    assert_eq!(rows.len(), 2);
    assert!(rows.iter().all(|r| r.len() == 4), "combined width WL+WR = 4");
    assert_eq!(norm_sorted(&rows), norm_sorted(&expected));
}

#[test]
fn index_nested_loop_left_fills_missing_seek() {
    let (mut pager, cat) = index_nl_db();
    let rows = run(
        &plan(join_node(
            seqscan("l", 1),
            2,
            index_nl_right(),
            2,
            JoinType::Left,
            None,
            JoinStrategy::IndexNestedLoop,
        )),
        &cat,
        &mut pager,
    );

    // k=99 finds no r row -> its right half is NULL-filled (Left join).
    let expected = vec![
        vec![v_i(10), v_i(1), v_t("ten"), v_i(10)],
        vec![v_i(20), v_i(2), v_t("twenty"), v_i(20)],
        vec![v_i(99), v_i(3), Value::Null, Value::Null],
    ];
    assert_eq!(rows.len(), 3);
    assert_eq!(norm_sorted(&rows), norm_sorted(&expected));
}

#[test]
fn index_nested_loop_right_full_is_a_loud_error() {
    // The right side is rebuilt per left row, so it cannot be materialized once for
    // unmatched-right tracking: RIGHT/FULL must error loudly, not silently drop rows.
    let (mut pager, cat) = index_nl_db();
    let err = run_err(
        &plan(join_node(
            seqscan("l", 1),
            2,
            index_nl_right(),
            2,
            JoinType::Full,
            None,
            JoinStrategy::IndexNestedLoop,
        )),
        &cat,
        &mut pager,
    );
    assert!(err.is_err(), "IndexNestedLoop + FULL is an unsupported combo and must error");
    // Assert the error identity, not merely that *some* error occurred — a different
    // failure (e.g. a missing table) would otherwise pass this test.
    let msg = err.unwrap_err().to_string();
    assert!(
        msg.contains("IndexNestedLoop") && msg.contains("RIGHT/FULL"),
        "error must name the unsupported IndexNestedLoop RIGHT/FULL combo, got: {msg}"
    );
}

// ----- (8) Hash fan-out: duplicate build-side keys emit every matched pair --

#[test]
fn hash_duplicate_build_keys_emit_all_matches() {
    // Two RIGHT rows share join key id=2, so that key's hash bucket holds TWO indices.
    // A probe on l.id=2 must emit BOTH pairs — this is the only test that exercises a
    // multi-index bucket, so a regression that kept one row per key would slip past the
    // others but fail here. NestedLoop must agree.
    let (mut pager, cat) = db_with_two_tables(
        "CREATE TABLE l(id INTEGER, name TEXT)",
        "CREATE TABLE r(id INTEGER, val TEXT)",
    );
    seed(
        &mut pager,
        table_root(&cat, "l"),
        &[(1, vec![v_i(2), v_t("b")]), (2, vec![v_i(7), v_t("c")])],
    );
    seed(
        &mut pager,
        table_root(&cat, "r"),
        &[
            (1, vec![v_i(2), v_t("x")]), // key 2
            (2, vec![v_i(2), v_t("y")]), // key 2 again -> same bucket, two indices
            (3, vec![v_i(9), v_t("z")]),
        ],
    );

    let expected = vec![
        vec![v_i(2), v_t("b"), v_i(1), v_i(2), v_t("x"), v_i(1)],
        vec![v_i(2), v_t("b"), v_i(1), v_i(2), v_t("y"), v_i(2)],
    ];

    let hash = run(
        &plan(join_node(
            seqscan("l", 2),
            3,
            seqscan("r", 2),
            3,
            JoinType::Inner,
            Some(eq(0, 3)),
            hash_strategy(),
        )),
        &cat,
        &mut pager,
    );
    let nl = run(
        &plan(join_node(
            seqscan("l", 2),
            3,
            seqscan("r", 2),
            3,
            JoinType::Inner,
            Some(eq(0, 3)),
            JoinStrategy::NestedLoop,
        )),
        &cat,
        &mut pager,
    );

    assert_eq!(hash.len(), 2, "both right rows sharing key 2 must be emitted");
    assert_eq!(norm_sorted(&hash), norm_sorted(&expected));
    assert_eq!(norm_sorted(&hash), norm_sorted(&nl), "NestedLoop and Hash agree on fan-out");
}

// ----- (9) Residual `on` beyond the equijoin key (Hash) --------------------

#[test]
fn hash_residual_on_beyond_the_equijoin_drops_nonmatching_bucket_hits() {
    // `on = (l.id = r.id) AND (l.name <> r.val)`: the key equality only gets a row into
    // the bucket; the residual then decides. A hash path that stopped re-checking the
    // full `on` on a bucket hit would wrongly emit the residual-failing pairs.
    let (mut pager, cat) = db_with_two_tables(
        "CREATE TABLE l(id INTEGER, name TEXT)",
        "CREATE TABLE r(id INTEGER, val TEXT)",
    );
    seed(
        &mut pager,
        table_root(&cat, "l"),
        &[
            (1, vec![v_i(2), v_t("b")]), // key 2: one bucket hit passes the residual
            (2, vec![v_i(5), v_t("m")]), // key 5: its only bucket hit FAILS the residual
        ],
    );
    seed(
        &mut pager,
        table_root(&cat, "r"),
        &[
            (1, vec![v_i(2), v_t("b")]), // key hit for l.id=2 but val == l.name -> FALSE
            (2, vec![v_i(2), v_t("q")]), // key hit for l.id=2 and val != l.name -> TRUE
            (3, vec![v_i(5), v_t("m")]), // key hit for l.id=5 but val == l.name -> FALSE
        ],
    );

    let on = EvalExpr::And(Box::new(eq(0, 3)), Box::new(ne(1, 4)));

    // INNER: only the (b,q) pair survives; both (b,b) and (m,m) bucket hits are dropped.
    let inner_expected = vec![vec![v_i(2), v_t("b"), v_i(1), v_i(2), v_t("q"), v_i(2)]];
    let hash_inner = run(
        &plan(join_node(
            seqscan("l", 2),
            3,
            seqscan("r", 2),
            3,
            JoinType::Inner,
            Some(on.clone()),
            hash_strategy(),
        )),
        &cat,
        &mut pager,
    );
    let nl_inner = run(
        &plan(join_node(
            seqscan("l", 2),
            3,
            seqscan("r", 2),
            3,
            JoinType::Inner,
            Some(on.clone()),
            JoinStrategy::NestedLoop,
        )),
        &cat,
        &mut pager,
    );
    assert_eq!(hash_inner.len(), 1, "residual drops the (b,b) and (m,m) bucket hits");
    assert_eq!(norm_sorted(&hash_inner), norm_sorted(&inner_expected));
    assert_eq!(norm_sorted(&hash_inner), norm_sorted(&nl_inner), "Hash and NestedLoop agree");

    // LEFT: l.id=2 has a residual-passing match (kept as the pair, not NULL-filled);
    // l.id=5's only bucket hit fails the residual, so it is NULL-filled — proving a
    // residual failure leaves the row unmatched rather than emitting the bucket hit.
    let left_expected = vec![
        vec![v_i(2), v_t("b"), v_i(1), v_i(2), v_t("q"), v_i(2)],
        vec![v_i(5), v_t("m"), v_i(2), Value::Null, Value::Null, Value::Null],
    ];
    let hash_left = run(
        &plan(join_node(
            seqscan("l", 2),
            3,
            seqscan("r", 2),
            3,
            JoinType::Left,
            Some(on),
            hash_strategy(),
        )),
        &cat,
        &mut pager,
    );
    assert_eq!(hash_left.len(), 2);
    assert_eq!(norm_sorted(&hash_left), norm_sorted(&left_expected));
}

// ----- (10) Empty left side: RIGHT/FULL still emit every right row ----------

#[test]
fn empty_left_right_and_full_still_emit_all_right_rows() {
    // With an EMPTY left, RIGHT/FULL must still emit every right row NULL-filled. This
    // pins the eager right-materialization: a regression to materializing lazily on the
    // first left row would emit nothing (the left loop never runs).
    let (mut pager, cat) = db_with_two_tables(
        "CREATE TABLE l(id INTEGER, name TEXT)",
        "CREATE TABLE r(id INTEGER, val TEXT)",
    );
    // `l` is intentionally left unseeded (empty); only `r` gets rows.
    seed(
        &mut pager,
        table_root(&cat, "r"),
        &[(1, vec![v_i(2), v_t("x")]), (2, vec![v_i(5), v_t("z")])],
    );

    let expected = vec![
        vec![Value::Null, Value::Null, Value::Null, v_i(2), v_t("x"), v_i(1)],
        vec![Value::Null, Value::Null, Value::Null, v_i(5), v_t("z"), v_i(2)],
    ];

    for (jt, strat) in [
        (JoinType::Right, JoinStrategy::NestedLoop),
        (JoinType::Full, JoinStrategy::NestedLoop),
        (JoinType::Right, hash_strategy()),
        (JoinType::Full, hash_strategy()),
    ] {
        let label = format!("{jt:?}");
        let rows = run(
            &plan(join_node(
                seqscan("l", 2),
                3,
                seqscan("r", 2),
                3,
                jt,
                Some(eq(0, 3)),
                strat,
            )),
            &cat,
            &mut pager,
        );
        assert_eq!(rows.len(), 2, "empty left still emits both right rows (jt={label})");
        assert_eq!(norm_sorted(&rows), norm_sorted(&expected), "jt={label}");
    }
}

// ----- (11) Planner equi-key STRIP is result-preserving --------------------
//
// The planner strips the conjuncts it consumed as hash keys from the `Join.on` it stores
// (`minisqlite-plan`'s `choose_join_strategy`): a hash-matched pair already satisfies a
// plain, un-coerced `=` (its collation folded into the key), so re-checking it per pair is
// pure O(output-pairs) waste. These tests are the EXECUTOR-side proof of that strip: for
// each shape the planner emits — a Hash carrying only the RESIDUAL — the result set is
// byte-identical to the semantic reference, a NestedLoop carrying the FULL predicate. They
// fail if the executor mishandles a stripped/None `on`, and (cases c/d) if a conjunct the
// planner must NOT strip were dropped.

/// Assert the strip is result-preserving for `jt`: a NestedLoop with the FULL `on` (the
/// reference) and a Hash carrying `strategy`'s keys with only `residual_on` (the planner's
/// post-strip output) must agree as sets. Both sides run over the standard 2-column `l`/`r`
/// tables (width 3), so the right columns are registers 3 and 4.
fn assert_strip_matches_reference(
    pager: &mut MemPager,
    cat: &SchemaCatalog,
    jt: JoinType,
    full_on: EvalExpr,
    strategy: JoinStrategy,
    residual_on: Option<EvalExpr>,
) {
    let reference = run(
        &plan(join_node(
            seqscan("l", 2), 3, seqscan("r", 2), 3, jt.clone(), Some(full_on), JoinStrategy::NestedLoop,
        )),
        cat,
        pager,
    );
    let stripped = run(
        &plan(join_node(
            seqscan("l", 2), 3, seqscan("r", 2), 3, jt.clone(), residual_on, strategy,
        )),
        cat,
        pager,
    );
    assert_eq!(
        norm_sorted(&stripped),
        norm_sorted(&reference),
        "Hash with the equi-key stripped ({jt:?}) must equal NestedLoop with the full ON",
    );
}

#[test]
fn strip_pure_equijoin_to_none_matches_full_predicate() {
    // (a) A pure equijoin `l.id = r.id` is fully consumed → the planner stores `on = None`
    // and the executor's `on_keeps` fast path keeps every hash-matched pair. Must equal the
    // full-predicate NestedLoop for INNER and every OUTER flavor (also covers case (f) with
    // no residual).
    let (mut pager, cat) = standard_lr();
    for jt in [JoinType::Inner, JoinType::Left, JoinType::Right, JoinType::Full] {
        assert_strip_matches_reference(&mut pager, &cat, jt, eq(0, 3), hash_strategy(), None);
    }
}

#[test]
fn strip_leaves_residual_matches_full_predicate() {
    // (b)+(f) `on = (l.id=r.id) AND (l.name <> r.val)`: the equi-key is stripped, the `<>`
    // residual is retained and must still filter — for INNER and every OUTER flavor. The
    // fixture is chosen so the residual ACTUALLY FIRES on a hash-matched pair: the id=4 rows
    // have name "z" == val "z", so a correct residual DROPS that pair (INNER) or leaves that
    // row null-extended (LEFT/RIGHT/FULL — the riskiest OUTER path: a bucket hit rejected by
    // the residual must NOT count as a match). `standard_lr` cannot test this: there every
    // matched pair passes `<>`, so the residual is a no-op and a dropped residual would be
    // indistinguishable from a kept one.
    let (mut pager, cat) = residual_filters_db();
    let full = EvalExpr::And(Box::new(eq(0, 3)), Box::new(ne(1, 4)));
    for jt in [JoinType::Inner, JoinType::Left, JoinType::Right, JoinType::Full] {
        assert_strip_matches_reference(&mut pager, &cat, jt, full.clone(), hash_strategy(), Some(ne(1, 4)));
    }

    // Load-bearing for every OUTER flavor: dropping the residual (on = None) diverges,
    // because the residual-rejected id=4 pair would then survive as a match instead of being
    // filtered (INNER) or null-extended (outer). `keep` is the correct (full-predicate) row
    // count; `drop` is the wrong (no-residual) one — pinned so a silently-empty or
    // over-emitting run can't pass the set comparison. For LEFT/RIGHT the counts coincide
    // (2 == 2) but the CONTENT differs (id=4 null-extended vs id=4 paired), which the
    // `assert_ne!` on the normalized sets catches.
    for (jt, keep, drop) in [
        (JoinType::Inner, 1usize, 2usize),
        (JoinType::Left, 2, 2),
        (JoinType::Right, 2, 2),
        (JoinType::Full, 3, 2),
    ] {
        let reference = run(
            &plan(join_node(seqscan("l", 2), 3, seqscan("r", 2), 3, jt.clone(), Some(full.clone()), JoinStrategy::NestedLoop)),
            &cat,
            &mut pager,
        );
        let wrongly_stripped = run(
            &plan(join_node(seqscan("l", 2), 3, seqscan("r", 2), 3, jt.clone(), None, hash_strategy())),
            &cat,
            &mut pager,
        );
        assert_eq!(reference.len(), keep, "residual filters the id=4 pair ({jt:?})");
        assert_eq!(wrongly_stripped.len(), drop, "without the residual the id=4 pair survives ({jt:?})");
        assert_ne!(
            norm_sorted(&wrongly_stripped),
            norm_sorted(&reference),
            "dropping the `<>` residual changes the {jt:?} result → it must be retained",
        );
    }
}

#[test]
fn strip_multi_key_equijoin_to_none_matches_full_predicate() {
    // (e) `l.k1=r.k1 AND l.k2=r.k2`: BOTH equalities are hash keys → `on = None`. The
    // two-key hash join must reproduce the full two-conjunct NestedLoop.
    let (mut pager, cat) = two_key_db();
    let full = EvalExpr::And(Box::new(eq(0, 3)), Box::new(eq(1, 4)));
    for jt in [JoinType::Inner, JoinType::Left, JoinType::Right, JoinType::Full] {
        assert_strip_matches_reference(&mut pager, &cat, jt, full.clone(), hash_strategy_2key(), None);
    }
    // Pin the concrete count so a silently-empty result can't pass the set comparison.
    let inner = run(
        &plan(join_node(seqscan("l", 2), 3, seqscan("r", 2), 3, JoinType::Inner, None, hash_strategy_2key())),
        &cat,
        &mut pager,
    );
    assert_eq!(inner.len(), 2, "two rows match on BOTH keys");
}

#[test]
fn coerced_equality_residual_is_kept_and_load_bearing() {
    // (c) A coerced `=` (NUMERIC affinity) is never a hash key — raw hashing would miss its
    // matches — so the planner keeps it in the residual `on`. Model that: Hash keyed on the
    // plain `l.id=r.id`, residual = the coerced `l.tag = r.num`.
    let (mut pager, cat) = coerced_residual_db();
    let full = EvalExpr::And(Box::new(eq(0, 3)), Box::new(eq_numeric(1, 4)));

    // Correct — the coerced residual is retained → equals the full-predicate reference.
    assert_strip_matches_reference(&mut pager, &cat, JoinType::Inner, full.clone(), hash_strategy(), Some(eq_numeric(1, 4)));
    assert_strip_matches_reference(&mut pager, &cat, JoinType::Left, full.clone(), hash_strategy(), Some(eq_numeric(1, 4)));

    // Load-bearing: had the planner ALSO stripped the coerced `=` (on = None), the Hash
    // join would over-emit the id=3 bucket hit the coercion rejects. This is why the strip
    // criterion must exclude a coerced equality.
    let reference = run(
        &plan(join_node(seqscan("l", 2), 3, seqscan("r", 2), 3, JoinType::Inner, Some(full), JoinStrategy::NestedLoop)),
        &cat,
        &mut pager,
    );
    let wrongly_stripped = run(
        &plan(join_node(seqscan("l", 2), 3, seqscan("r", 2), 3, JoinType::Inner, None, hash_strategy())),
        &cat,
        &mut pager,
    );
    assert_ne!(
        norm_sorted(&wrongly_stripped),
        norm_sorted(&reference),
        "wrongly stripping the coerced `=` changes the result → it must stay in the residual",
    );
    assert_eq!(reference.len(), 1, "the coerced residual keeps only the id=2 pair (tag '5' = num 5)");
    assert_eq!(wrongly_stripped.len(), 2, "without the residual, both id buckets emit");
}

#[test]
fn is_null_safe_residual_is_kept_and_load_bearing() {
    // (d) `IS` (null-safe `=`) is never a hash key (the hash's NULL-excludes-match path
    // can't yield `NULL IS NULL = true`), so it stays in the residual `on`. Model that:
    // Hash keyed on `l.id=r.id`, residual = `l.name IS r.val`.
    let (mut pager, cat) = is_residual_db();
    let full = EvalExpr::And(Box::new(eq(0, 3)), Box::new(is_eq(1, 4)));

    assert_strip_matches_reference(&mut pager, &cat, JoinType::Inner, full.clone(), hash_strategy(), Some(is_eq(1, 4)));
    assert_strip_matches_reference(&mut pager, &cat, JoinType::Left, full.clone(), hash_strategy(), Some(is_eq(1, 4)));

    // Load-bearing: the surviving pair is the NULL/NULL one (id=2), which ONLY `IS` keeps
    // (a plain `=` yields NULL there). Dropping the residual (on=None) would also emit id=3.
    let reference = run(
        &plan(join_node(seqscan("l", 2), 3, seqscan("r", 2), 3, JoinType::Inner, Some(full), JoinStrategy::NestedLoop)),
        &cat,
        &mut pager,
    );
    let wrongly_stripped = run(
        &plan(join_node(seqscan("l", 2), 3, seqscan("r", 2), 3, JoinType::Inner, None, hash_strategy())),
        &cat,
        &mut pager,
    );
    assert_ne!(
        norm_sorted(&wrongly_stripped),
        norm_sorted(&reference),
        "wrongly stripping the IS changes the result → it must stay in the residual",
    );
    assert_eq!(reference.len(), 1, "the IS residual keeps only the NULL/NULL id=2 pair");
    assert_eq!(wrongly_stripped.len(), 2, "without the residual, both id buckets emit");
}

/// `l(k1,k2)` and `r(k1,k2)` for the two-key equijoin: two rows match on BOTH keys,
/// while `l.(1,20)` and `r.(1,99)` share only `k1` and must NOT match.
fn two_key_db() -> (MemPager, SchemaCatalog) {
    let (mut pager, cat) = db_with_two_tables(
        "CREATE TABLE l(k1 INTEGER, k2 INTEGER)",
        "CREATE TABLE r(k1 INTEGER, k2 INTEGER)",
    );
    seed(
        &mut pager,
        table_root(&cat, "l"),
        &[(1, vec![v_i(1), v_i(10)]), (2, vec![v_i(1), v_i(20)]), (3, vec![v_i(2), v_i(10)])],
    );
    seed(
        &mut pager,
        table_root(&cat, "r"),
        &[(1, vec![v_i(1), v_i(10)]), (2, vec![v_i(1), v_i(99)]), (3, vec![v_i(2), v_i(10)])],
    );
    (pager, cat)
}

/// `l(id,tag TEXT)` and `r(id,num INTEGER)` for the coerced-residual case: under NUMERIC
/// affinity `tag '5'` equals `num 5` (id=2 keeps), while `tag 'xyz'` (non-numeric) does
/// not equal `num 7` (id=3 drops) — a distinction only the coercion makes.
fn coerced_residual_db() -> (MemPager, SchemaCatalog) {
    let (mut pager, cat) = db_with_two_tables(
        "CREATE TABLE l(id INTEGER, tag TEXT)",
        "CREATE TABLE r(id INTEGER, num INTEGER)",
    );
    seed(
        &mut pager,
        table_root(&cat, "l"),
        &[(1, vec![v_i(2), v_t("5")]), (2, vec![v_i(3), v_t("xyz")])],
    );
    seed(
        &mut pager,
        table_root(&cat, "r"),
        &[(1, vec![v_i(2), v_i(5)]), (2, vec![v_i(3), v_i(7)])],
    );
    (pager, cat)
}

/// `l(id,name TEXT)` and `r(id,val TEXT)` where the `l.name <> r.val` residual DISCRIMINATES
/// among hash-matched pairs: id=2 matches with name "b" <> val "x" (kept), while id=4 matches
/// with name "z" == val "z" (dropped). Unlike `standard_lr` — where every matched pair passes
/// `<>` and the residual is a no-op — this makes a run that ignored the residual over-emit the
/// id=4 pair, so a strip that wrongly discarded the residual is detectable, including on the
/// OUTER null-extension path (the residual-rejected id=4 row must be null-extended, not paired).
fn residual_filters_db() -> (MemPager, SchemaCatalog) {
    let (mut pager, cat) = db_with_two_tables(
        "CREATE TABLE l(id INTEGER, name TEXT)",
        "CREATE TABLE r(id INTEGER, val TEXT)",
    );
    seed(
        &mut pager,
        table_root(&cat, "l"),
        &[(1, vec![v_i(2), v_t("b")]), (2, vec![v_i(4), v_t("z")])],
    );
    seed(
        &mut pager,
        table_root(&cat, "r"),
        &[(1, vec![v_i(2), v_t("x")]), (2, vec![v_i(4), v_t("z")])],
    );
    (pager, cat)
}

/// `l(id,name TEXT)` and `r(id,val TEXT)` for the IS-residual case: the id=2 rows both
/// carry NULL (so `name IS val` is TRUE and the pair is kept), while id=3 carries `'c'`
/// vs `'d'` (dropped). A plain `=` would instead yield NULL for the id=2 pair and drop it.
fn is_residual_db() -> (MemPager, SchemaCatalog) {
    let (mut pager, cat) = db_with_two_tables(
        "CREATE TABLE l(id INTEGER, name TEXT)",
        "CREATE TABLE r(id INTEGER, val TEXT)",
    );
    seed(
        &mut pager,
        table_root(&cat, "l"),
        &[(1, vec![v_i(2), Value::Null]), (2, vec![v_i(3), v_t("c")])],
    );
    seed(
        &mut pager,
        table_root(&cat, "r"),
        &[(1, vec![v_i(2), Value::Null]), (2, vec![v_i(3), v_t("d")])],
    );
    (pager, cat)
}

// ----- helpers -------------------------------------------------------------

fn as_int(v: &Value) -> i64 {
    match v {
        Value::Integer(i) => *i,
        other => panic!("expected Integer, got {other:?}"),
    }
}
