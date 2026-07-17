//! Conformance: WINDOW functions layered over a GROUP BY / aggregate query.
//!
//! A window function in the SELECT list of an aggregated query runs AFTER
//! `GROUP BY` + `HAVING`, ONCE per output group row, and BEFORE the outer
//! `ORDER BY` / `LIMIT` (lang_select.html: window functions are evaluated after
//! grouping; windowfunctions.html §1). Its arguments / `PARTITION BY` /
//! `ORDER BY` therefore see the aggregate's OUTPUT relation — a group KEY
//! resolves to that group's key value, and an AGGREGATE (`sum(x)`, `count(*)`)
//! to an aggregate result register holding that group's computed value (its own
//! non-deduped call, so an aggregate reused inside an OVER spec is computed again
//! at a second register with the same value) — never a bare-column error. The
//! engine plans this as `Aggregate → Window → Project → …`.
//!
//! Every expected value here is hand-computed from those documented semantics,
//! not read back from the engine. All comparisons go through the harness
//! (`assert_rows` / `assert_query_error`) because `Value` has no `PartialEq`.

mod conformance;

use conformance::*;
use minisqlite::Connection;

/// Canonical fixture: `emp(dept, region, sal)` where `region` is functionally
/// determined by `dept` (eng/sales → west; hr/mkt → east), so `GROUP BY dept`
/// yields one group per dept and `GROUP BY dept, region` yields the same groups
/// while making `region` a usable group key for `PARTITION BY`.
///
/// Per-dept aggregates (used throughout):
/// * eng   → count 2, sum 300
/// * sales → count 1, sum 150
/// * hr    → count 3, sum 600
/// * mkt   → count 1, sum 400
fn emp_db() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE emp(dept TEXT, region TEXT, sal INTEGER)");
    exec(
        &mut db,
        "INSERT INTO emp(dept, region, sal) VALUES \
         ('eng','west',100), \
         ('eng','west',200), \
         ('sales','west',150), \
         ('hr','east',300), \
         ('hr','east',100), \
         ('hr','east',200), \
         ('mkt','east',400)",
    );
    db
}

/// 1. `row_number()` OVER (ORDER BY an aggregate) on a GROUP BY query.
///
/// The window orders the four group rows by `sum(sal)` DESC and numbers them
/// 1..4. Sums are distinct (600 > 400 > 300 > 150), so the ordering — and thus
/// every row_number — is fully determined:
///   hr 600→1, mkt 400→2, eng 300→3, sales 150→4.
/// The outer `ORDER BY rn` then presents them in that order.
#[test]
fn row_number_over_aggregate_orders_groups() {
    let mut db = emp_db();
    assert_rows(
        &mut db,
        "SELECT dept, sum(sal) AS total, \
                row_number() OVER (ORDER BY sum(sal) DESC) AS rn \
         FROM emp GROUP BY dept ORDER BY rn",
        &[
            vec![text("hr"), int(600), int(1)],
            vec![text("mkt"), int(400), int(2)],
            vec![text("eng"), int(300), int(3)],
            vec![text("sales"), int(150), int(4)],
        ],
    );
}

/// 2. `rank()` vs `dense_rank()` OVER (ORDER BY an aggregate), WITH a tie.
///
/// Groups a/b/c/d have sums 30/20/20/10. Ordered by `sum(x)` DESC the two 20s
/// tie, so (windowfunctions.html §3): rank() gives 1,2,2,4 (a gap after the
/// tie) while dense_rank() gives 1,2,2,3 (no gap). The outer `ORDER BY rnk, g`
/// makes the tied pair deterministic (b before c).
#[test]
fn rank_and_dense_rank_ties_over_aggregate() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(g TEXT, x INTEGER)");
    exec(&mut db, "INSERT INTO t(g, x) VALUES ('a',30),('b',20),('c',20),('d',10)");
    assert_rows(
        &mut db,
        "SELECT g, sum(x) AS s, \
                rank()       OVER (ORDER BY sum(x) DESC) AS rnk, \
                dense_rank() OVER (ORDER BY sum(x) DESC) AS drnk \
         FROM t GROUP BY g ORDER BY rnk, g",
        &[
            vec![text("a"), int(30), int(1), int(1)],
            vec![text("b"), int(20), int(2), int(2)],
            vec![text("c"), int(20), int(2), int(2)],
            vec![text("d"), int(10), int(4), int(3)],
        ],
    );
}

/// 3. An aggregate WINDOW (`sum`, `count`) computed ACROSS the group rows.
///
/// `sum(sum(sal))` is a window `sum` over the per-group `sum(sal)` result: with
/// `ORDER BY dept ROWS UNBOUNDED PRECEDING..CURRENT ROW` it is the running total
/// of the group sums in dept order (eng 300, hr 600, mkt 400, sales 150):
///   300, 900, 1300, 1450.
/// `count(*) OVER ()` has no ORDER BY, so its default frame is the whole
/// partition — the group count, 4, on every row. This is the exact case the old
/// engine rejected with "misuse of window function".
#[test]
fn window_sum_and_count_across_group_rows() {
    let mut db = emp_db();
    assert_rows(
        &mut db,
        "SELECT dept, sum(sal) AS total, \
                sum(sum(sal)) OVER (ORDER BY dept ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running, \
                count(*) OVER () AS ngroups \
         FROM emp GROUP BY dept ORDER BY dept",
        &[
            vec![text("eng"), int(300), int(300), int(4)],
            vec![text("hr"), int(600), int(900), int(4)],
            vec![text("mkt"), int(400), int(1300), int(4)],
            vec![text("sales"), int(150), int(1450), int(4)],
        ],
    );
}

/// 4. PARTITION BY a group KEY, ORDER BY an aggregate — numbering RESETS per
/// partition.
///
/// `GROUP BY dept, region` gives one row per dept; `PARTITION BY region` then
/// splits those group rows by region and `row_number() OVER (... ORDER BY
/// sum(sal) DESC)` restarts at 1 in each region:
///   east: hr 600→1, mkt 400→2;  west: eng 300→1, sales 150→2.
#[test]
fn partition_by_group_key_resets_numbering() {
    let mut db = emp_db();
    assert_rows(
        &mut db,
        "SELECT region, dept, sum(sal) AS total, \
                row_number() OVER (PARTITION BY region ORDER BY sum(sal) DESC) AS rn \
         FROM emp GROUP BY dept, region ORDER BY region, rn",
        &[
            vec![text("east"), text("hr"), int(600), int(1)],
            vec![text("east"), text("mkt"), int(400), int(2)],
            vec![text("west"), text("eng"), int(300), int(1)],
            vec![text("west"), text("sales"), int(150), int(2)],
        ],
    );
}

/// 5. A window `ORDER BY` that references BOTH a group key AND an aggregate.
///
/// `row_number() OVER (ORDER BY region ASC, sum(sal) DESC)` orders the group
/// rows first by the group key `region`, then within a region by the aggregate
/// `sum(sal)` DESC — one global numbering across all groups:
///   east/hr 600→1, east/mkt 400→2, west/eng 300→3, west/sales 150→4.
#[test]
fn window_order_by_group_key_and_aggregate() {
    let mut db = emp_db();
    assert_rows(
        &mut db,
        "SELECT region, dept, sum(sal) AS total, \
                row_number() OVER (ORDER BY region ASC, sum(sal) DESC) AS rn \
         FROM emp GROUP BY dept, region ORDER BY rn",
        &[
            vec![text("east"), text("hr"), int(600), int(1)],
            vec![text("east"), text("mkt"), int(400), int(2)],
            vec![text("west"), text("eng"), int(300), int(3)],
            vec![text("west"), text("sales"), int(150), int(4)],
        ],
    );
}

/// 6. `lag`/`lead` (offset value functions) reading a neighbouring group's
/// aggregate.
///
/// Ordered by dept (eng 300, hr 600, mkt 400, sales 150), `lag(sum(sal))` is the
/// previous group's sum (NULL for the first) and `lead(sum(sal))` the next
/// group's (NULL for the last).
#[test]
fn lag_lead_over_aggregate_groups() {
    let mut db = emp_db();
    assert_rows(
        &mut db,
        "SELECT dept, sum(sal) AS total, \
                lag(sum(sal))  OVER (ORDER BY dept) AS prev, \
                lead(sum(sal)) OVER (ORDER BY dept) AS next \
         FROM emp GROUP BY dept ORDER BY dept",
        &[
            vec![text("eng"), int(300), null(), int(600)],
            vec![text("hr"), int(600), int(300), int(400)],
            vec![text("mkt"), int(400), int(600), int(150)],
            vec![text("sales"), int(150), int(400), null()],
        ],
    );
}

/// 7. A named `WINDOW` (trailing `WINDOW w AS (...)`) reused by two window
/// functions over the aggregate. `w` orders the group rows by `sum(sal)` DESC;
/// sums are distinct so `row_number` and `rank` agree (1..4).
#[test]
fn named_window_over_aggregate() {
    let mut db = emp_db();
    assert_rows(
        &mut db,
        "SELECT dept, sum(sal) AS total, \
                row_number() OVER w AS rn, \
                rank()       OVER w AS rnk \
         FROM emp GROUP BY dept \
         WINDOW w AS (ORDER BY sum(sal) DESC) \
         ORDER BY rn",
        &[
            vec![text("hr"), int(600), int(1), int(1)],
            vec![text("mkt"), int(400), int(2), int(2)],
            vec![text("eng"), int(300), int(3), int(3)],
            vec![text("sales"), int(150), int(4), int(4)],
        ],
    );
}

/// 8. A PARTITIONED aggregate window with an explicit ROWS frame: a per-region
/// running total of the group sums.
///
/// `GROUP BY dept, region`; within each `PARTITION BY region`, ordered by dept,
/// `sum(sum(sal))` with `ROWS UNBOUNDED PRECEDING..CURRENT ROW` accumulates and
/// RESETS at the partition boundary:
///   east: hr 600, mkt 600+400=1000;  west: eng 300, sales 300+150=450.
#[test]
fn partitioned_running_sum_of_aggregate_with_frame() {
    let mut db = emp_db();
    assert_rows(
        &mut db,
        "SELECT region, dept, sum(sal) AS total, \
                sum(sum(sal)) OVER (PARTITION BY region ORDER BY dept \
                                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS run \
         FROM emp GROUP BY dept, region ORDER BY region, dept",
        &[
            vec![text("east"), text("hr"), int(600), int(600)],
            vec![text("east"), text("mkt"), int(400), int(1000)],
            vec![text("west"), text("eng"), int(300), int(300)],
            vec![text("west"), text("sales"), int(150), int(450)],
        ],
    );
}

/// Regression guard A: a plain GROUP BY aggregate (NO window) is unchanged.
/// The post-aggregate windowing stage must be a strict no-op when the projection
/// carries zero window calls — same rows the engine produced before the feature.
#[test]
fn plain_group_by_aggregate_unchanged() {
    let mut db = emp_db();
    assert_rows(
        &mut db,
        "SELECT dept, sum(sal), count(*) FROM emp GROUP BY dept ORDER BY dept",
        &[
            vec![text("eng"), int(300), int(2)],
            vec![text("hr"), int(600), int(3)],
            vec![text("mkt"), int(400), int(1)],
            vec![text("sales"), int(150), int(1)],
        ],
    );
}

/// Regression guard B: a plain window query (NO GROUP BY / no aggregate) is
/// unchanged — it never enters the aggregate compile path. `row_number() OVER
/// (ORDER BY sal)` numbers the seven rows in ascending `sal`; projecting only
/// `sal` and `rn` and ordering by `rn` keeps the result deterministic despite
/// the tied sal values (100,100 and 200,200).
#[test]
fn plain_window_without_group_by_unchanged() {
    let mut db = emp_db();
    assert_rows(
        &mut db,
        "SELECT sal, row_number() OVER (ORDER BY sal) AS rn FROM emp ORDER BY rn",
        &[
            vec![int(100), int(1)],
            vec![int(100), int(2)],
            vec![int(150), int(3)],
            vec![int(200), int(4)],
            vec![int(200), int(5)],
            vec![int(300), int(6)],
            vec![int(400), int(7)],
        ],
    );
}

/// Residual loud error: a window function in HAVING is rejected. HAVING runs
/// BEFORE the window stage, so a window call there is a "misuse of window
/// function" error (matching WHERE/GROUP BY/ON), never a computed value.
#[test]
fn window_in_having_is_misuse_error() {
    let mut db = emp_db();
    let err = assert_query_error(
        &mut db,
        "SELECT dept, sum(sal) FROM emp GROUP BY dept HAVING row_number() OVER () > 0",
    );
    let msg = format!("{err}");
    assert!(
        msg.contains("misuse of window function"),
        "expected a 'misuse of window function' error for a window in HAVING, got: {msg}"
    );
}

/// Residual loud error: a window function in GROUP BY is rejected. Group keys
/// bind against the PRE-aggregate row with no windowing context, so a window
/// call there is the same "misuse of window function" error.
#[test]
fn window_in_group_by_is_misuse_error() {
    let mut db = emp_db();
    let err = assert_query_error(
        &mut db,
        "SELECT sal FROM emp GROUP BY row_number() OVER ()",
    );
    let msg = format!("{err}");
    assert!(
        msg.contains("misuse of window function"),
        "expected a 'misuse of window function' error for a window in GROUP BY, got: {msg}"
    );
}

/// Residual loud error: a window whose OVER spec references a NON-grouped bare
/// column (`ORDER BY sal`, sal neither grouped nor aggregated). This is a §2.5 bare
/// column (real SQLite accepts it, ordering the window by an arbitrary per-group
/// value), but combining a §2.5 capture with a post-aggregate window is a documented
/// engine limitation — rejected LOUDLY rather than risking a wrong register, never a
/// silent wrong value. (Was the "must appear in GROUP BY" error before general bare
/// columns; now the more honest window+bare-column limitation message.)
#[test]
fn window_over_non_grouped_bare_column_is_error() {
    let mut db = emp_db();
    let err = assert_query_error(
        &mut db,
        "SELECT dept, row_number() OVER (ORDER BY sal) FROM emp GROUP BY dept",
    );
    let msg = format!("{err}");
    assert!(
        msg.contains("window function together with a bare column"),
        "expected the window+bare-column limitation error, got: {msg}"
    );
}

/// A window function written DIRECTLY in an aggregate query's ORDER BY (matching no
/// output column) is a fresh post-aggregate expression: it binds against the grouping
/// context, is computed by the post-aggregate window stage, and orders the result — the
/// same machinery as a window in the SELECT list, just consumed by ORDER BY and then
/// dropped. Here `row_number() OVER (ORDER BY sum(sal))` numbers the groups by ascending
/// sum (sales 150, eng 300, mkt 400, hr 600 → 1,2,3,4), so ordering by it yields exactly
/// the ascending-sum order. (Was a loud "does not match any column" error before aggregate
/// ORDER BY resolved post-aggregate expressions.)
#[test]
fn fresh_window_in_aggregate_order_by_is_computed_and_orders_the_result() {
    let mut db = emp_db();
    assert_rows(
        &mut db,
        "SELECT dept, sum(sal) FROM emp GROUP BY dept ORDER BY row_number() OVER (ORDER BY sum(sal))",
        &[
            vec![text("sales"), int(150)],
            vec![text("eng"), int(300)],
            vec![text("mkt"), int(400)],
            vec![text("hr"), int(600)],
        ],
    );
}

/// The HEADLINE ordering semantic, pinned directly: a window runs AFTER
/// `GROUP BY` + `HAVING`, over the SURVIVING groups only.
///
/// `HAVING sum(sal) >= 300` drops the `sales` group (sum 150), leaving eng/hr/mkt.
/// `count(*) OVER ()` therefore sees THREE rows, not four — the discriminating
/// check that the window is layered on the post-HAVING relation. `sum(sum(sal))`
/// running over dept order accumulates only the survivors: 300, 900, 1300. (If a
/// refactor ever moved the window before HAVING, ngroups would read 4 and the
/// running total would include 150 — this test would catch it.)
#[test]
fn window_runs_after_having_over_surviving_groups() {
    let mut db = emp_db();
    assert_rows(
        &mut db,
        "SELECT dept, sum(sal) AS total, \
                sum(sum(sal)) OVER (ORDER BY dept ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running, \
                count(*) OVER () AS ngroups \
         FROM emp GROUP BY dept HAVING sum(sal) >= 300 ORDER BY dept",
        &[
            vec![text("eng"), int(300), int(300), int(3)],
            vec![text("hr"), int(600), int(900), int(3)],
            vec![text("mkt"), int(400), int(1300), int(3)],
        ],
    );
}

/// Window-family coverage part 1: `ntile` and the value functions
/// (`first_value` / `last_value` / `nth_value`) over an aggregate. These share the
/// same `bind_window_function` → exec Window path as the ranking/offset functions
/// tested above; asserting them keeps the whole class exhaustive.
///
/// Group sums ordered DESC: hr 600, mkt 400, eng 300, sales 150 (n=4).
/// * `ntile(2)` splits 4 rows into 2 even buckets → 1,1,2,2.
/// * `first_value(sum(sal))` under the default frame (partition start..current) is
///   always the partition's first row → 600.
/// * `last_value(sum(sal))` under a FULL frame is the partition's last row → 150.
/// * `nth_value(sum(sal), 2)` under a full frame is the 2nd row → 400.
#[test]
fn ntile_and_value_functions_over_aggregate() {
    let mut db = emp_db();
    assert_rows(
        &mut db,
        "SELECT dept, sum(sal) AS total, \
                ntile(2)              OVER (ORDER BY sum(sal) DESC) AS nt, \
                first_value(sum(sal)) OVER (ORDER BY sum(sal) DESC) AS fv, \
                last_value(sum(sal))  OVER (ORDER BY sum(sal) DESC \
                                            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS lv, \
                nth_value(sum(sal), 2) OVER (ORDER BY sum(sal) DESC \
                                            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS nv \
         FROM emp GROUP BY dept ORDER BY total DESC",
        &[
            vec![text("hr"), int(600), int(1), int(600), int(150), int(400)],
            vec![text("mkt"), int(400), int(1), int(600), int(150), int(400)],
            vec![text("eng"), int(300), int(2), int(600), int(150), int(400)],
            vec![text("sales"), int(150), int(2), int(600), int(150), int(400)],
        ],
    );
}

/// Window-family coverage part 2: `cume_dist` over an aggregate. Group sums are
/// distinct (no peers), so ordered ASC (sales 150, eng 300, mkt 400, hr 600) the
/// cumulative distribution is k/4: 0.25, 0.5, 0.75, 1.0 — all exactly
/// representable as f64, so the harness's bit-exact Real compare holds. (The outer
/// `ORDER BY total` references the output alias; a fresh post-aggregate expression in an
/// aggregate ORDER BY is also supported — see
/// `fresh_window_in_aggregate_order_by_is_computed_and_orders_the_result`.)
#[test]
fn cume_dist_over_aggregate() {
    let mut db = emp_db();
    assert_rows(
        &mut db,
        "SELECT dept, sum(sal) AS total, cume_dist() OVER (ORDER BY sum(sal)) AS cd \
         FROM emp GROUP BY dept ORDER BY total",
        &[
            vec![text("sales"), int(150), real(0.25)],
            vec![text("eng"), int(300), real(0.5)],
            vec![text("mkt"), int(400), real(0.75)],
            vec![text("hr"), int(600), real(1.0)],
        ],
    );
}

/// Window-family coverage part 3: `percent_rank` over an aggregate. A five-group
/// fixture makes the denominator `n-1 = 4`, so `(rank-1)/4` is 0, 0.25, 0.5, 0.75,
/// 1.0 — all exact. (Distinct sums, so rank equals ordinal position.)
#[test]
fn percent_rank_over_aggregate() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE g5(k TEXT, v INTEGER)");
    exec(&mut db, "INSERT INTO g5(k, v) VALUES ('a',10),('b',20),('c',30),('d',40),('e',50)");
    assert_rows(
        &mut db,
        "SELECT k, sum(v) AS s, percent_rank() OVER (ORDER BY sum(v)) AS pr \
         FROM g5 GROUP BY k ORDER BY s",
        &[
            vec![text("a"), int(10), real(0.0)],
            vec![text("b"), int(20), real(0.25)],
            vec![text("c"), int(30), real(0.5)],
            vec![text("d"), int(40), real(0.75)],
            vec![text("e"), int(50), real(1.0)],
        ],
    );
}

/// Residual loud error, pinned: the single-`min`/`max` §2.5 bare-column capture
/// COMBINED with a post-aggregate window call is rejected precisely.
///
/// First, the §2.5 case WITHOUT a window still works — `region` is a bare
/// (non-grouped) column, and with a single `max(sal)` it is captured from each
/// group's max-sal row (lang_select.html §2.5). Adding a window on top hits the
/// loud rejection: §2.5 applicability depends on the min/max count, which a window
/// OVER spec can change unseen, so the offsets are rejected rather than risked.
/// The ONLY difference between the two statements is the window call, so this pins
/// exactly the residual boundary.
#[test]
fn minmax_bare_capture_plus_window_is_loud_error() {
    let mut db = emp_db();
    // §2.5 capture alone: region taken from the max-sal row of each group.
    assert_rows(
        &mut db,
        "SELECT dept, region, max(sal) FROM emp GROUP BY dept ORDER BY dept",
        &[
            vec![text("eng"), text("west"), int(200)],
            vec![text("hr"), text("east"), int(300)],
            vec![text("mkt"), text("east"), int(400)],
            vec![text("sales"), text("west"), int(150)],
        ],
    );
    // Same query with a window added → precise loud error, not a wrong register.
    let err = assert_query_error(
        &mut db,
        "SELECT dept, region, max(sal), row_number() OVER () FROM emp GROUP BY dept",
    );
    let msg = format!("{err}");
    assert!(
        msg.contains("window function together with a bare column"),
        "expected the §2.5-plus-window loud error, got: {msg}"
    );
}
