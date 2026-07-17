//! The `count(*)` row-count fast path (`is_row_count_only` +
//! [`RowCursor::count_rows`]): a `count(*)` with no GROUP BY / HAVING / bare columns and
//! no per-row modifiers is computed by asking the input for its row count instead of
//! pulling and discarding a materialized row per input row. The hash join answers that
//! by summing its matched-bucket sizes rather than building a combined `Row` per pair.
//!
//! The optimization must be a PURE SPEED change: `count(*)` has to return EXACTLY the
//! count the row-materializing path would. The strongest check of that is differential —
//! `count(*)` over a source must equal the number of rows a plain `SELECT` over the SAME
//! source yields — so most tests here pit the two paths against each other across
//! inner / residual-`ON` / left / cross joins, filters, and empty inputs. The remaining
//! tests pin the queries that must NOT take the fast path (`count(X)`, `count(DISTINCT)`,
//! GROUP BY, HAVING, a mixed aggregate list) still return correct results.

use minisqlite::{Connection, Value};

/// The single integer a scalar `count(*)` query returns.
fn count_of(db: &mut Connection, sql: &str) -> i64 {
    let rows = db.query(sql).expect("count query runs");
    assert_eq!(rows.rows.len(), 1, "a bare count(*) yields exactly one row: {sql}");
    match rows.rows[0][0] {
        Value::Integer(n) => n,
        ref other => panic!("count(*) is INTEGER, got {other:?} for {sql}"),
    }
}

/// How many rows a query actually yields (the row-materializing path).
fn rowcount(db: &mut Connection, sql: &str) -> i64 {
    db.query(sql).expect("row query runs").rows.len() as i64
}

/// A table `t(id PK, k, v)` seeded with `rows` (a VALUES tail, or empty for no rows).
fn setup(rows: &str) -> Connection {
    let mut db = Connection::open_in_memory().unwrap();
    db.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, k INTEGER, v TEXT)").unwrap();
    if !rows.is_empty() {
        db.execute(&format!("INSERT INTO t(id,k,v) VALUES {rows}")).unwrap();
    }
    db
}

/// k multiplicities 2,1,3 over ids 1..=6 — a self-equijoin count of 2²+1²+3² = 14.
const ROWS: &str = "(1,1,'a'),(2,1,'b'),(3,2,'c'),(4,3,'d'),(5,3,'e'),(6,3,'f')";

#[test]
fn count_star_over_inner_equijoin_equals_materialized() {
    let mut db = setup(ROWS);
    let fast = count_of(&mut db, "SELECT count(*) FROM t a JOIN t b ON a.k=b.k");
    let materialized = rowcount(&mut db, "SELECT a.k FROM t a JOIN t b ON a.k=b.k");
    assert_eq!(fast, 14, "2*2 + 1*1 + 3*3");
    assert_eq!(fast, materialized, "fast count must equal the materialized row count");
}

#[test]
fn count_star_over_equijoin_with_index_equals_materialized() {
    // An index on the join key can change the plan (hash vs index-nested-loop); the
    // count must be identical regardless of which access path the planner picks.
    let mut db = setup(ROWS);
    db.execute("CREATE INDEX t_k ON t(k)").unwrap();
    let fast = count_of(&mut db, "SELECT count(*) FROM t a JOIN t b ON a.k=b.k");
    let materialized = rowcount(&mut db, "SELECT a.k FROM t a JOIN t b ON a.k=b.k");
    assert_eq!(fast, 14);
    assert_eq!(fast, materialized);
}

#[test]
fn count_star_with_residual_on_equals_materialized() {
    // `a.id < b.id` is a residual predicate the equijoin keys do not capture, so a hash
    // join carries it as `on` and the count path must evaluate it per pair (its
    // `Some(pred)` branch). Ordered pairs within each k group: k=1 -> (1,2); k=3 ->
    // (4,5),(4,6),(5,6); k=2 -> none. Total 4.
    let mut db = setup(ROWS);
    let sql = "FROM t a JOIN t b ON a.k=b.k AND a.id < b.id";
    let fast = count_of(&mut db, &format!("SELECT count(*) {sql}"));
    let materialized = rowcount(&mut db, &format!("SELECT a.id {sql}"));
    assert_eq!(fast, 4);
    assert_eq!(fast, materialized);
}

#[test]
fn count_star_over_left_join_with_matches_equals_materialized() {
    // Every left row matches itself on a.k=b.k, so a LEFT JOIN emits the same rows as the
    // inner join here (no null-fills). The join's count_rows override defers to the drain
    // for LEFT, and must still match the materialized count.
    let mut db = setup(ROWS);
    let fast = count_of(&mut db, "SELECT count(*) FROM t a LEFT JOIN t b ON a.k=b.k");
    let materialized = rowcount(&mut db, "SELECT a.k FROM t a LEFT JOIN t b ON a.k=b.k");
    assert_eq!(fast, 14);
    assert_eq!(fast, materialized);
}

#[test]
fn count_star_over_left_join_all_unmatched_counts_null_fills() {
    // `a.k = b.k + 100` can never hold (k in {1,2,3}), so every left row is unmatched and
    // a LEFT JOIN emits one null-filled row per left row (6). This exercises the
    // need_left_fill path where the count is NOT the matched-pair total.
    let mut db = setup(ROWS);
    let fast = count_of(&mut db, "SELECT count(*) FROM t a LEFT JOIN t b ON a.k=b.k+100");
    let materialized = rowcount(&mut db, "SELECT a.id FROM t a LEFT JOIN t b ON a.k=b.k+100");
    assert_eq!(fast, 6);
    assert_eq!(fast, materialized);
}

#[test]
fn count_star_over_cross_join_equals_materialized() {
    // A cross join is a nested loop (not a hash), so the aggregate uses the DEFAULT
    // count_rows drain over it. 6 * 6 = 36.
    let mut db = setup(ROWS);
    let fast = count_of(&mut db, "SELECT count(*) FROM t a, t b");
    let materialized = rowcount(&mut db, "SELECT a.id FROM t a, t b");
    assert_eq!(fast, 36);
    assert_eq!(fast, materialized);
}

#[test]
fn count_star_with_where_filter_equals_materialized() {
    let mut db = setup(ROWS);
    let fast = count_of(&mut db, "SELECT count(*) FROM t WHERE k=3");
    let materialized = rowcount(&mut db, "SELECT id FROM t WHERE k=3");
    assert_eq!(fast, 3);
    assert_eq!(fast, materialized);
}

#[test]
fn count_star_empty_table_is_zero() {
    let mut db = setup("");
    assert_eq!(count_of(&mut db, "SELECT count(*) FROM t"), 0);
}

#[test]
fn count_star_empty_inner_join_is_zero() {
    let mut db = setup(ROWS);
    // No pair satisfies a.k = b.k + 100, so the inner join is empty.
    assert_eq!(count_of(&mut db, "SELECT count(*) FROM t a JOIN t b ON a.k=b.k+100"), 0);
}

#[test]
fn two_count_stars_agree() {
    // Two zero-argument aggregates: both are replayed over the same row count.
    let mut db = setup(ROWS);
    let rows = db.query("SELECT count(*), count(*) FROM t").unwrap().rows;
    assert_eq!(rows.len(), 1);
    assert!(matches!(rows[0][0], Value::Integer(6)));
    assert!(matches!(rows[0][1], Value::Integer(6)));
}

#[test]
fn count_star_over_join_with_two_count_stars_agree() {
    let mut db = setup(ROWS);
    let rows = db
        .query("SELECT count(*), count(*) FROM t a JOIN t b ON a.k=b.k")
        .unwrap()
        .rows;
    assert_eq!(rows.len(), 1);
    assert!(matches!(rows[0][0], Value::Integer(14)));
    assert!(matches!(rows[0][1], Value::Integer(14)));
}

#[test]
fn group_by_count_is_not_fast_path_and_correct() {
    // GROUP BY disqualifies the fast path; per-group counts must still be right.
    let mut db = setup(ROWS);
    let rows = db.query("SELECT k, count(*) FROM t GROUP BY k ORDER BY k").unwrap().rows;
    let got: Vec<(i64, i64)> = rows
        .iter()
        .map(|r| match (&r[0], &r[1]) {
            (Value::Integer(k), Value::Integer(c)) => (*k, *c),
            other => panic!("unexpected group row {other:?}"),
        })
        .collect();
    assert_eq!(got, vec![(1, 2), (2, 1), (3, 3)]);
}

#[test]
fn count_of_column_skips_nulls_not_fast_path() {
    // count(X) has an argument, so it is NOT the fast path; it must skip NULLs while
    // count(*) counts every row.
    let mut db = setup("(1,1,'a'),(2,NULL,'b'),(3,3,NULL)");
    assert_eq!(count_of(&mut db, "SELECT count(*) FROM t"), 3);
    assert_eq!(count_of(&mut db, "SELECT count(k) FROM t"), 2);
    assert_eq!(count_of(&mut db, "SELECT count(v) FROM t"), 2);
}

#[test]
fn count_distinct_is_not_fast_path_and_correct() {
    let mut db = setup(ROWS);
    assert_eq!(count_of(&mut db, "SELECT count(DISTINCT k) FROM t"), 3);
}

#[test]
fn count_star_with_having_is_not_fast_path_and_correct() {
    let mut db = setup(ROWS);
    // The single implicit group has count 6: a satisfied HAVING keeps it, a failed one
    // drops it (zero rows). HAVING references the aggregate, so it is not the fast path.
    assert_eq!(count_of(&mut db, "SELECT count(*) FROM t HAVING count(*) >= 6"), 6);
    assert_eq!(rowcount(&mut db, "SELECT count(*) FROM t HAVING count(*) > 100"), 0);
}

#[test]
fn count_star_and_sum_mixed_is_not_fast_path_and_correct() {
    // A non-zero-arg aggregate in the list disqualifies the fast path; both results must
    // still be correct. sum(id) = 1+2+3+4+5+6 = 21.
    let mut db = setup(ROWS);
    let rows = db.query("SELECT count(*), sum(id) FROM t").unwrap().rows;
    assert_eq!(rows.len(), 1);
    assert!(matches!(rows[0][0], Value::Integer(6)));
    assert!(matches!(rows[0][1], Value::Integer(21)));
}
