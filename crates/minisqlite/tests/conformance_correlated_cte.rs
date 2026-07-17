//! Conformance: a **correlated `WITH` subquery** — a subquery that carries a leading
//! `WITH` clause AND whose (post-`WITH`) body references a column of the enclosing
//! query, e.g. `SELECT t.a, (WITH c(n) AS (SELECT 10) SELECT n + t.a FROM c) FROM t`.
//!
//! Per `spec/sqlite-doc/lang_with.html` a CTE "exists for the duration of a single SQL
//! statement" and acts like a temporary table/view: the CTE *definition* is materialized
//! standalone (it does not see the enclosing query's columns), but the SELECT the `WITH`
//! is attached to is an ordinary subquery, so its body may be correlated to the outer
//! query just like any other subquery. At runtime such a subquery is re-evaluated once
//! per outer row with that row prepended; the materialized CTE is scanned inside it.
//!
//! This previously returned a loud "a correlated WITH subquery … is not yet supported"
//! (`compile/cte.rs`) because the WITH compiler dropped the `base_offset` that places the
//! body's own sources after the outer row. The fix threads that seam through, so a
//! correlated WITH subquery binds exactly like a non-WITH one.
//!
//! Every expected value is hand-computed from documented SQL semantics, never read back
//! from the engine.

mod conformance;

use conformance::*;
use minisqlite::Connection;

/// `t(id, a)` — the outer table; `u(id, t_id, w)` — the correlated subquery's own data.
///
///   u grouped by t_id:
///     t_id=1 → w {100, 200}  (count 2, sum 300, max 200)
///     t_id=2 → w {50}        (count 1, sum  50, max  50)
///     t_id=3 → w {7, 8, 9}   (count 3, sum  24, max   9)
fn db() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER, a INTEGER)");
    exec(&mut db, "INSERT INTO t(id, a) VALUES (1, 10), (2, 20), (3, 30)");
    exec(&mut db, "CREATE TABLE u(id INTEGER, t_id INTEGER, w INTEGER)");
    exec(
        &mut db,
        "INSERT INTO u(id, t_id, w) VALUES \
         (1,1,100), (2,1,200), (3,2,50), (4,3,7), (5,3,8), (6,3,9)",
    );
    db
}

/// The body of a WITH-bearing scalar subquery references an outer column (`t.a`). The
/// CTE `c` is a single constant row; the subquery returns `10 + t.a` per outer row.
#[test]
fn correlated_with_scalar_in_select_list() {
    let mut d = db();
    assert_rows(
        &mut d,
        "SELECT t.id, (WITH c(n) AS (SELECT 10) SELECT n + t.a FROM c) FROM t ORDER BY t.id",
        &[
            vec![int(1), int(20)],
            vec![int(2), int(30)],
            vec![int(3), int(40)],
        ],
    );
}

/// The CTE materializes ALL of `u` (uncorrelated), and the body correlates to the outer
/// row through its own `WHERE` (`uu.t_id = t.id`). Per t.id the body sums the matching
/// `w`: 1→300, 2→50, 3→24.
#[test]
fn correlated_with_aggregate_body_over_materialized_cte() {
    let mut d = db();
    assert_rows(
        &mut d,
        "SELECT t.id, (WITH uu AS (SELECT t_id, w FROM u) \
                       SELECT sum(w) FROM uu WHERE uu.t_id = t.id) \
         FROM t ORDER BY t.id",
        &[
            vec![int(1), int(300)],
            vec![int(2), int(50)],
            vec![int(3), int(24)],
        ],
    );
}

/// A correlated WITH subquery inside `WHERE EXISTS`. The CTE is the full `u`
/// (uncorrelated); the body's `WHERE c.t_id = t.id AND c.w > 100` correlates. Only
/// t.id=1 has a matching row (w=200 > 100).
#[test]
fn correlated_with_in_where_exists() {
    let mut d = db();
    assert_rows(
        &mut d,
        "SELECT t.id FROM t \
         WHERE EXISTS (WITH c AS (SELECT t_id, w FROM u) \
                       SELECT 1 FROM c WHERE c.t_id = t.id AND c.w > 100) \
         ORDER BY t.id",
        &[vec![int(1)]],
    );
}

/// A `WITH RECURSIVE` CTE inside a correlated subquery. `seq` is the constant sequence
/// 1..5 (uncorrelated); the body counts how many of its rows are `<= t.id`, correlating
/// to the outer row: 1→1, 2→2, 3→3.
#[test]
fn correlated_recursive_with_subquery() {
    let mut d = db();
    assert_rows(
        &mut d,
        "SELECT t.id, \
                (WITH RECURSIVE seq(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM seq WHERE n < 5) \
                 SELECT count(*) FROM seq WHERE seq.n <= t.id) \
         FROM t ORDER BY t.id",
        &[
            vec![int(1), int(1)],
            vec![int(2), int(2)],
            vec![int(3), int(3)],
        ],
    );
}

/// The body references TWO distinct outer columns (`t.id` and `t.a`), exercising the
/// multi-column correlated-outer capture: `t.id * 100 + t.a + n` with n=1.
#[test]
fn correlated_with_captures_multiple_outer_columns() {
    let mut d = db();
    assert_rows(
        &mut d,
        "SELECT t.id, t.a, (WITH c(n) AS (SELECT 1) SELECT t.id * 100 + t.a + n FROM c) \
         FROM t ORDER BY t.id",
        &[
            vec![int(1), int(10), int(111)],
            vec![int(2), int(20), int(221)],
            vec![int(3), int(30), int(331)],
        ],
    );
}

/// A nested case: the correlated WITH subquery's body itself contains a further
/// (non-WITH) correlated subquery over `u`. Both correlate to the same outer `t.id`.
/// The CTE `c` holds the constant 1000; the body returns `1000 + (count of u for t.id)`:
/// 1→1002, 2→1001, 3→1003.
#[test]
fn correlated_with_body_nesting_a_further_correlated_subquery() {
    let mut d = db();
    assert_rows(
        &mut d,
        "SELECT t.id, \
                (WITH c(base) AS (SELECT 1000) \
                 SELECT base + (SELECT count(*) FROM u WHERE u.t_id = t.id) FROM c) \
         FROM t ORDER BY t.id",
        &[
            vec![int(1), int(1002)],
            vec![int(2), int(1001)],
            vec![int(3), int(1003)],
        ],
    );
}

/// Regression: an UNCORRELATED WITH subquery (its body references no outer column) still
/// compiles and returns the same constant for every outer row — the fix must not disturb
/// the common non-correlated path.
#[test]
fn uncorrelated_with_subquery_still_works() {
    let mut d = db();
    assert_rows(
        &mut d,
        "SELECT t.id, (WITH c(n) AS (SELECT 10) SELECT sum(n) FROM c) FROM t ORDER BY t.id",
        &[
            vec![int(1), int(10)],
            vec![int(2), int(10)],
            vec![int(3), int(10)],
        ],
    );
}

/// A correlated WITH subquery used as a value in the SELECT list of an AGGREGATE query,
/// correlating to a GROUP BY key. Grouping `u` by `t_id`, each group's row carries
/// `max(w)`; the correlated WITH subquery adds the group key `t_id` to the CTE constant.
/// Groups: t_id=1 (max 200), 2 (max 50), 3 (max 9); the subquery yields `t_id + 5`.
#[test]
fn correlated_with_subquery_in_aggregate_projection() {
    let mut d = db();
    assert_rows(
        &mut d,
        "SELECT u.t_id, max(u.w), \
                (WITH c(k) AS (SELECT 5) SELECT k + u.t_id FROM c) \
         FROM u GROUP BY u.t_id ORDER BY u.t_id",
        &[
            vec![int(1), int(200), int(6)],
            vec![int(2), int(50), int(7)],
            vec![int(3), int(9), int(8)],
        ],
    );
}
