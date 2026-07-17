//! Conformance: a CORRELATED subquery in the SELECT list / HAVING / ORDER BY of an
//! aggregate (GROUP BY) query, combined with the three shapes that WIDEN the
//! post-aggregate row the executor prepends to that subquery as its outer:
//!
//!   (a) a §2.5 "bare column" capture (lang_select.html §2.5) — a column neither
//!       inside an aggregate nor in GROUP BY, taken from the min()/max() row;
//!   (b) a post-aggregate WINDOW function (windowfunctions.html §1), which appends
//!       result columns after the aggregate row (Aggregate → Window → Project);
//!   (c) an outer ORDER BY that introduces a NEW aggregate not in the SELECT list.
//!
//! A correlated subquery here references a GROUP BY key of the enclosing aggregate
//! query (lang_select.html §2.5 / lang_expr.html "correlated subquery"): at runtime
//! the engine re-evaluates the subplan once per output group row with that group's
//! post-aggregate row `[group_keys.., agg_results.. (, captured / window)..]`
//! prepended as the outer, so the subquery's own FROM sources must be placed AFTER
//! that row. When (a)/(b)/(c) widen the row, the planner must re-bind the subquery
//! with the true (wider) outer width so its sources do not overlap the captured /
//! window columns — the behavior these tests pin.
//!
//! Every expected value is hand-computed from those documented semantics, never read
//! back from the engine. All comparisons go through the shared harness.

mod conformance;

use conformance::*;
use minisqlite::Connection;

/// Canonical fixture.
///
/// `t(g, v, tag)` — `tag` is a distinct label per row so a §2.5 bare-column capture
/// is observable (it must equal the tag of the min/max row):
///   * g='a': v = 1,3,7   → count 3, sum 11, min 1 ('a1'), max 7 ('a7')
///   * g='b': v = 5,2      → count 2, sum  7, min 2 ('b2'), max 5 ('b5')
///   * g='c': v = 4        → count 1, sum  4, min 4 ('c4'), max 4 ('c4')
///
/// `u(g, w)` — the correlated subquery's own table; per-g row count and sum(w):
///   * g='a': (10,20)      → count 2, sum 30
///   * g='b': (100)        → count 1, sum 100
///   * g='c': (7,8,9)      → count 3, sum 24
///   * TOTAL               → count 6
fn agg_db() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(g TEXT, v INTEGER, tag TEXT)");
    exec(
        &mut db,
        "INSERT INTO t(g, v, tag) VALUES \
         ('a',1,'a1'), \
         ('a',3,'a3'), \
         ('a',7,'a7'), \
         ('b',5,'b5'), \
         ('b',2,'b2'), \
         ('c',4,'c4')",
    );
    exec(&mut db, "CREATE TABLE u(g TEXT, w INTEGER)");
    exec(
        &mut db,
        "INSERT INTO u(g, w) VALUES ('a',10),('a',20),('b',100),('c',7),('c',8),('c',9)",
    );
    db
}

/// (a) §2.5 captured bare column + a correlated subquery in the SELECT list.
///
/// `tag` is a bare column; with a single `max(v)` aggregate it takes the value from
/// each group's maximum-`v` row (lang_select.html §2.5): a→'a7', b→'b5', c→'c4'.
/// The correlated `(SELECT count(*) FROM u WHERE u.g = t.g)` counts `u` rows per
/// group: a→2, b→1, c→3. The captured column widens the aggregate output row past
/// the provisional width, so the subquery is re-bound to sit AFTER it.
#[test]
fn captured_bare_column_with_correlated_subquery() {
    let mut db = agg_db();
    assert_rows(
        &mut db,
        "SELECT g, tag, max(v), (SELECT count(*) FROM u WHERE u.g = t.g) \
         FROM t GROUP BY g ORDER BY g",
        &[
            vec![text("a"), text("a7"), int(7), int(2)],
            vec![text("b"), text("b5"), int(5), int(1)],
            vec![text("c"), text("c4"), int(4), int(3)],
        ],
    );
}

/// (a′) Same capture, but the correlated subquery itself AGGREGATES over its own
/// (multi-row) source — so its own FROM base and its own aggregate must both be
/// correct past the widened outer row. `sum(w)` per group: a→30, b→100, c→24.
#[test]
fn captured_bare_column_with_correlated_aggregating_subquery() {
    let mut db = agg_db();
    assert_rows(
        &mut db,
        "SELECT g, tag, min(v), (SELECT sum(w) FROM u WHERE u.g = t.g) \
         FROM t GROUP BY g ORDER BY g",
        &[
            // min(v) row's tag: a→'a1', b→'b2', c→'c4'
            vec![text("a"), text("a1"), int(1), int(30)],
            vec![text("b"), text("b2"), int(2), int(100)],
            vec![text("c"), text("c4"), int(4), int(24)],
        ],
    );
}

/// (b) Post-aggregate WINDOW function + a correlated subquery in the SELECT list.
///
/// `count(*)` per group: a→3, b→2, c→1. `sum(count(*)) OVER ()` sums the group
/// counts over the whole result (no PARTITION/ORDER → default frame is every row):
/// 3+2+1 = 6 on every row. The correlated subquery counts `u` per group: a→2, b→1,
/// c→3. The window column widens the projection's outer row (Aggregate → Window →
/// Project), so the subquery is re-bound to sit AFTER the window result.
#[test]
fn window_function_with_correlated_subquery() {
    let mut db = agg_db();
    assert_rows(
        &mut db,
        "SELECT g, count(*), sum(count(*)) OVER (), \
                (SELECT count(*) FROM u WHERE u.g = t.g) \
         FROM t GROUP BY g ORDER BY g",
        &[
            vec![text("a"), int(3), int(6), int(2)],
            vec![text("b"), int(2), int(6), int(1)],
            vec![text("c"), int(1), int(6), int(3)],
        ],
    );
}

/// (c) An outer ORDER BY that introduces a NEW aggregate + a correlated subquery in
/// the SELECT list.
///
/// `ORDER BY sum(v)` adds `sum(v)` (a→11, b→7, c→4) as a hidden aggregate not in the
/// SELECT list, pushing the real aggregate row wider than the provisional width. The
/// rows come back in `sum(v)` order (c<b<a). `count(*)`: a→3, b→2, c→1; the
/// correlated subquery counts `u` per group: a→2, b→1, c→3.
#[test]
fn order_by_new_aggregate_with_correlated_subquery() {
    let mut db = agg_db();
    assert_rows(
        &mut db,
        "SELECT g, count(*), (SELECT count(*) FROM u WHERE u.g = t.g) \
         FROM t GROUP BY g ORDER BY sum(v)",
        &[
            vec![text("c"), int(1), int(3)],
            vec![text("b"), int(2), int(1)],
            vec![text("a"), int(3), int(2)],
        ],
    );
}

/// A correlated subquery in HAVING (no window) is fully supported: HAVING runs in the
/// aggregate operator over `[group_keys.., agg_results..]`, exactly the row the
/// subquery is bound against. Keep groups where `count(*) > (u count for g)`:
/// a 3>2 ✓, b 2>1 ✓, c 1>3 ✗.
#[test]
fn correlated_subquery_in_having_without_window() {
    let mut db = agg_db();
    assert_rows(
        &mut db,
        "SELECT g, count(*) FROM t GROUP BY g \
         HAVING count(*) > (SELECT count(*) FROM u WHERE u.g = t.g) ORDER BY g",
        &[vec![text("a"), int(3)], vec![text("b"), int(2)]],
    );
}

/// The witness: a correlated subquery in HAVING TOGETHER WITH a
/// post-aggregate WINDOW function. This is the exact combination that used to be a loud
/// "not yet supported" reject. HAVING runs INSIDE the aggregate operator over the
/// PRE-window row `[g, count(*)]` (width `num_keys + total_aggs`), while
/// `sum(count(*)) OVER ()` runs in the post-window Window stage over the WIDER row
/// `[g, count(*), window]` — two different outer widths. The planner now serves both by
/// re-pointing the correlated outer width per host clause (HAVING vs projection/ORDER BY)
/// instead of forcing one width for the whole query.
///
/// HAVING keeps a (3>2) and b (2>1), drops c (1>3). `sum(count(*)) OVER ()` sums the
/// SURVIVING groups' counts (the window runs after HAVING): 3 + 2 = 5 on every surviving
/// row. With no ORDER BY the groups come back in first-appearance (== g) order: a, b.
#[test]
fn correlated_subquery_in_having_with_window() {
    let mut db = agg_db();
    assert_rows(
        &mut db,
        "SELECT g, count(*), sum(count(*)) OVER () FROM t GROUP BY g \
         HAVING count(*) > (SELECT count(*) FROM u WHERE u.g = t.g)",
        &[vec![text("a"), int(3), int(5)], vec![text("b"), int(2), int(5)]],
    );
}

/// The same HAVING × window shape with the correlated subquery ALSO feeding a projection
/// window and an ORDER BY, to exercise BOTH outer widths in one query: the HAVING subquery
/// binds at the pre-window width and the projection subquery at the post-window width.
/// `avg(v)` per group: a→(1+3+7)/3, b→(5+2)/2=3, both kept by HAVING (count 3>2, 2>1); c
/// dropped. `row_number() OVER (ORDER BY g)` numbers the two survivors 1,2 in g order.
#[test]
fn correlated_having_and_projection_subqueries_use_their_own_widths() {
    let mut db = agg_db();
    assert_rows(
        &mut db,
        "SELECT g, count(*), (SELECT count(*) FROM u WHERE u.g = t.g), \
         row_number() OVER (ORDER BY g) \
         FROM t GROUP BY g \
         HAVING count(*) > (SELECT count(*) FROM u WHERE u.g = t.g) \
         ORDER BY g",
        &[
            vec![text("a"), int(3), int(2), int(1)],
            vec![text("b"), int(2), int(1), int(2)],
        ],
    );
}

/// SAFETY PROPERTY: an aggregate query with a §2.5 capture and a NON-correlated
/// subquery is unaffected by the correlated-subquery re-bind (the re-bind is gated on
/// a correlated subquery being present). `(SELECT count(*) FROM u)` is the whole-table
/// count 6 on every row; `tag` is still the max(v) row's tag.
#[test]
fn captured_bare_column_with_noncorrelated_subquery_unaffected() {
    let mut db = agg_db();
    assert_rows(
        &mut db,
        "SELECT g, tag, max(v), (SELECT count(*) FROM u) FROM t GROUP BY g ORDER BY g",
        &[
            vec![text("a"), text("a7"), int(7), int(6)],
            vec![text("b"), text("b5"), int(5), int(6)],
            vec![text("c"), text("c4"), int(4), int(6)],
        ],
    );
}

/// A correlated subquery in the SELECT list of a PLAIN aggregate query (nothing
/// widens the row) already worked; pin that it still does after the re-bind
/// refactor. `count(*)`: a→3, b→2, c→1; subquery counts `u`: a→2, b→1, c→3.
#[test]
fn plain_correlated_subquery_in_select_list_unchanged() {
    let mut db = agg_db();
    assert_rows(
        &mut db,
        "SELECT g, count(*), (SELECT count(*) FROM u WHERE u.g = t.g) \
         FROM t GROUP BY g ORDER BY g",
        &[
            vec![text("a"), int(3), int(2)],
            vec![text("b"), int(2), int(1)],
            vec![text("c"), int(1), int(3)],
        ],
    );
}

/// Combined stress: a §2.5 capture AND a correlated subquery AND a plain (unchanged)
/// column, to confirm the widened outer row and the captured column coexist. `min(v)`
/// single aggregate → `tag` from the min row (a→'a1', b→'b2', c→'c4'); subquery
/// `max(w)` per group: a→20, b→100, c→9.
#[test]
fn capture_and_correlated_subquery_coexist() {
    let mut db = agg_db();
    assert_rows(
        &mut db,
        "SELECT g, tag, min(v), (SELECT max(w) FROM u WHERE u.g = t.g) \
         FROM t GROUP BY g ORDER BY g",
        &[
            vec![text("a"), text("a1"), int(1), int(20)],
            vec![text("b"), text("b2"), int(2), int(100)],
            vec![text("c"), text("c4"), int(4), int(9)],
        ],
    );
}
