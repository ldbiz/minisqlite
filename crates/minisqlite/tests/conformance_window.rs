//! Conformance battery — SQL window functions.
//!
//! Every expected value here is TRANSCRIBED FROM / DERIVED FROM the SQLite
//! documentation (`spec/sqlite-doc/windowfunctions.html`), NEVER from what the
//! engine returns. Binding sources:
//!
//!   * §1 (intro): `row_number()` "assigns consecutive integers to each row in
//!     order of the ORDER BY clause within the window-defn". The intro's t0
//!     example (`CREATE TABLE t0(x INTEGER PRIMARY KEY, y TEXT)` with rows
//!     (1,'aaa'),(2,'ccc'),(3,'bbb')) is transcribed verbatim in
//!     `row_number_spec_t0_intro_example`. §1 also defines the WINDOW clause:
//!     "Named window-defn clauses may also be added ... using a WINDOW clause
//!     and then referred to by name within window function invocations."
//!   * §2.1 "The PARTITION BY Clause": a partition "consists of all rows that
//!     have the same value for all terms of the PARTITION BY clause"; "If there
//!     is no PARTITION BY clause, then the entire result set ... is a single
//!     partition."
//!   * §2.2 "Frame Specifications": the default frame-spec is
//!     `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE NO OTHERS`,
//!     which "read[s] all rows from the beginning of the partition up to and
//!     including the current row AND ITS PEERS". Peers are rows with equal
//!     ORDER BY expressions (RANGE/GROUPS); with no ORDER BY, all rows are peers.
//!     Contrast: the ROWS frame type counts individual rows and does NOT pull in
//!     peers.
//!   * §3 "Built-in Window Functions":
//!       - `row_number()`: "The number of the row within the current partition.
//!         Rows are numbered starting from 1 in the order defined by the ORDER BY
//!         clause ..., or in arbitrary order otherwise."
//!       - `rank()`: "The row_number() of the first peer in each group - the rank
//!         of the current row with gaps. If there is no ORDER BY clause, then all
//!         rows are considered peers and this function always returns 1."
//!       - `dense_rank()`: "the rank of the current row without gaps ... If there
//!         is no ORDER BY clause ... always returns 1."
//!       - `lag(expr)` / `lead(expr, offset, default)`: `expr` evaluated against
//!         the previous / offset-ahead row in the partition, else NULL (or the
//!         supplied `default`). Both IGNORE the frame-spec.
//!       - `first_value(expr)` / `last_value(expr)` / `nth_value(expr, N)`:
//!         `expr` at the first / last / N-th row of the window FRAME (so under a
//!         `.. CURRENT ROW` frame `last_value` is the CURRENT row, not the
//!         partition's last row). These RESPECT the frame-spec.
//!     The §3 t2 ranking example is transcribed verbatim in
//!     `rank_dense_rank_spec_t2_example`; the §3 offset/value example over table
//!     `t1` (frame `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`) is
//!     transcribed verbatim in the `t1_*` tests.
//!
//! Window-function support in this engine is PROVISIONAL/incomplete. If a case
//! fails or errors because the engine disagrees with the documented behavior, the
//! assertion STAYS spec-correct (it is left to FAIL) and is NEVER weakened to match
//! the engine. Cases are split into many small `#[test]` fns so one failing (or
//! engine-rejected) case never masks the rest.
//!
//! Determinism note: `row_number()` numbers peers "in arbitrary order", and with
//! a ranking query the peer rows are interchangeable in the output. Where the
//! ORDER BY key of the query is not unique, such cases are asserted as a MULTISET
//! (`assert_rows_unordered`) so the expectation stays invariant to peer order.
//! `Value` has no `PartialEq`; all comparisons go through the shared harness.

mod conformance;

use conformance::*;
use minisqlite::Connection;

// ---------------------------------------------------------------------------
// Fixtures. The primary table `t` is the primary one here; `t0` and `t2`
// reproduce the doc's own example tables so their published output can be
// transcribed directly.
// ---------------------------------------------------------------------------

/// Task table:
/// | id | grp | v  |
/// |  1 | a   | 10 |
/// |  2 | a   | 30 |
/// |  3 | b   | 20 |
/// |  4 | b   | 20 |
/// |  5 | b   |  5 |
fn t_db() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER, grp TEXT, v INTEGER)");
    exec(
        &mut db,
        "INSERT INTO t VALUES (1,'a',10),(2,'a',30),(3,'b',20),(4,'b',20),(5,'b',5)",
    );
    db
}

/// `windowfunctions.html` §1 intro example table.
fn t0_db() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t0(x INTEGER PRIMARY KEY, y TEXT)");
    exec(&mut db, "INSERT INTO t0 VALUES (1, 'aaa'), (2, 'ccc'), (3, 'bbb')");
    db
}

/// `windowfunctions.html` §3 ranking example table.
fn t2_db() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t2(a, b)");
    exec(
        &mut db,
        "INSERT INTO t2 VALUES ('a','one'),('a','two'),('a','three'),('b','four'),('c','five'),('c','six')",
    );
    db
}

/// `windowfunctions.html` §2/§3 example table `t1` (used by the doc's
/// offset/value built-in example). `b` is unique (A..G), so `ORDER BY b` is a
/// total order and the derived cases are deterministic in ordered form.
fn t1_db() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t1(a INTEGER PRIMARY KEY, b, c)");
    exec(
        &mut db,
        "INSERT INTO t1 VALUES (1,'A','one'),(2,'B','two'),(3,'C','three'),\
         (4,'D','one'),(5,'E','two'),(6,'F','three'),(7,'G','one')",
    );
    db
}

// ===========================================================================
// row_number()  (§1 intro, §3)
// ===========================================================================

/// Direct transcription of the §1 intro example (line ~2610). ORDER BY y is a
/// unique key over {aaa,bbb,ccc}, so the assignment is fully deterministic:
/// aaa->1, bbb->2, ccc->3. The final output order is governed by the query's
/// `ORDER BY x`, not the window's ORDER BY. Also exercises the `AS` alias.
#[test]
fn row_number_spec_t0_intro_example() {
    let mut db = t0_db();
    assert_rows(
        &mut db,
        "SELECT x, y, row_number() OVER (ORDER BY y) AS rn FROM t0 ORDER BY x",
        &[
            vec![int(1), text("aaa"), int(1)],
            vec![int(2), text("ccc"), int(3)],
            vec![int(3), text("bbb"), int(2)],
        ],
    );
}

/// row_number over a two-term ORDER BY (v, id). Sorted by (v,id):
/// id5(v5)=1, id1(v10)=2, id3(v20)=3, id4(v20)=4, id2(v30)=5. Back by id.
#[test]
fn row_number_order_by_v_then_id() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT id, row_number() OVER (ORDER BY v, id) FROM t ORDER BY id",
        &[
            vec![int(1), int(2)],
            vec![int(2), int(5)],
            vec![int(3), int(3)],
            vec![int(4), int(4)],
            vec![int(5), int(1)],
        ],
    );
}

/// row_number restarts at 1 in every partition (§2.1), numbering within the
/// partition by the window ORDER BY. grp 'a' by (v,id): id1->1, id2->2.
/// grp 'b' by (v,id): id5(v5)->1, id3(v20)->2, id4(v20)->3.
#[test]
fn row_number_partition_by_grp_ordered() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT id, grp, row_number() OVER (PARTITION BY grp ORDER BY v, id) FROM t ORDER BY id",
        &[
            vec![int(1), text("a"), int(1)],
            vec![int(2), text("a"), int(2)],
            vec![int(3), text("b"), int(2)],
            vec![int(4), text("b"), int(3)],
            vec![int(5), text("b"), int(1)],
        ],
    );
}

/// Named window via the WINDOW clause (§1). `w AS (ORDER BY id)` numbers 1..5.
#[test]
fn row_number_named_window() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT id, row_number() OVER w FROM t WINDOW w AS (ORDER BY id) ORDER BY id",
        &[
            vec![int(1), int(1)],
            vec![int(2), int(2)],
            vec![int(3), int(3)],
            vec![int(4), int(4)],
            vec![int(5), int(5)],
        ],
    );
}

// ===========================================================================
// rank() / dense_rank()  (§3)
// ===========================================================================

/// rank() has gaps, dense_rank() does not. ORDER BY v => sorted {5,10,20,20,30}.
/// rank:  v5->1, v10->2, v20->3 (both), v30->5 (gap after the 2-way tie).
/// dense: v5->1, v10->2, v20->3 (both), v30->4 (no gap).
/// Mapped back by id — the outer ORDER BY id is unique so ordered comparison is
/// deterministic even though the two v=20 rows are peers (they share rank/dense).
#[test]
fn rank_and_dense_rank_with_ties() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT id, rank() OVER (ORDER BY v), dense_rank() OVER (ORDER BY v) FROM t ORDER BY id",
        &[
            vec![int(1), int(2), int(2)],
            vec![int(2), int(5), int(4)],
            vec![int(3), int(3), int(3)],
            vec![int(4), int(3), int(3)],
            vec![int(5), int(1), int(1)],
        ],
    );
}

/// §3: "If there is no ORDER BY clause, then all rows are considered peers and
/// this function always returns 1." Holds for both rank() and dense_rank().
#[test]
fn rank_and_dense_rank_no_order_by_all_ones() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT id, rank() OVER (), dense_rank() OVER () FROM t ORDER BY id",
        &[
            vec![int(1), int(1), int(1)],
            vec![int(2), int(1), int(1)],
            vec![int(3), int(1), int(1)],
            vec![int(4), int(1), int(1)],
            vec![int(5), int(1), int(1)],
        ],
    );
}

/// Direct transcription of the §3 ranking example (line ~7593), restricted to
/// the columns in scope (row_number/rank/dense_rank). The published output for
/// `WINDOW win AS (ORDER BY a)` over t2:
///   a: rn 1, rank 1, dense 1
///   a: rn 2, rank 1, dense 1
///   a: rn 3, rank 1, dense 1
///   b: rn 4, rank 4, dense 2
///   c: rn 5, rank 5, dense 3
///   c: rn 6, rank 5, dense 3
/// The query has no outer ORDER BY and row_number among peers is arbitrary, so
/// this is asserted as a MULTISET (each row is distinct via its row_number, so
/// the multiset itself is deterministic).
#[test]
fn rank_dense_rank_spec_t2_example() {
    let mut db = t2_db();
    assert_rows_unordered(
        &mut db,
        "SELECT a, row_number() OVER win, rank() OVER win, dense_rank() OVER win \
         FROM t2 WINDOW win AS (ORDER BY a)",
        &[
            vec![text("a"), int(1), int(1), int(1)],
            vec![text("a"), int(2), int(1), int(1)],
            vec![text("a"), int(3), int(1), int(1)],
            vec![text("b"), int(4), int(4), int(2)],
            vec![text("c"), int(5), int(5), int(3)],
            vec![text("c"), int(6), int(5), int(3)],
        ],
    );
}

// ===========================================================================
// percent_rank() / cume_dist() / ntile()  (§3)
// ===========================================================================
//
// End-to-end (parse -> bind -> execute) coverage for the three ranking-family
// built-ins that otherwise only meet the executor through its own unit tests, so
// a binding/dispatch/projection routing bug specific to these three would slip
// past the suite. Expected values are the §3 spec formulas applied to `t` under
// ORDER BY v (sorted 5,10,20,20,30 over ids 5,1,3,4,2), mapped back by the outer
// ORDER BY id. cume_dist's k/5 ratios are written as the exact spec fraction so
// the harness's bit-exact Real compare (total_cmp) is not defeated by decimal
// rounding of a written-out literal.

/// percent_rank() = (rank - 1) / (partition_rows - 1) (§3). ORDER BY v, N=5, so
/// the denominator is 4: v5->0/4, v10->1/4, v20->2/4 (a peer pair), v30->4/4. The
/// /4 values are all exactly representable, so plain literals compare bit-exact.
#[test]
fn percent_rank_over_order_by_v() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT id, percent_rank() OVER (ORDER BY v) FROM t ORDER BY id",
        &[
            vec![int(1), real(0.25)], // v10 -> rank 2 -> 1/4
            vec![int(2), real(1.0)],  // v30 -> rank 5 -> 4/4
            vec![int(3), real(0.5)],  // v20 -> rank 3 -> 2/4
            vec![int(4), real(0.5)],  // v20 -> rank 3 -> 2/4
            vec![int(5), real(0.0)],  // v5  -> rank 1 -> 0/4
        ],
    );
}

/// cume_dist() = (rows preceding or peer with the current row) / partition_rows
/// (§3). ORDER BY v, N=5: v5->1/5, v10->2/5, v20->4/5 (both peers counted), v30
/// ->5/5. Both v=20 rows share 4/5 because peers count toward each other.
#[test]
fn cume_dist_over_order_by_v() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT id, cume_dist() OVER (ORDER BY v) FROM t ORDER BY id",
        &[
            vec![int(1), real(2.0 / 5.0)], // v10 -> 2/5
            vec![int(2), real(1.0)],       // v30 -> 5/5
            vec![int(3), real(4.0 / 5.0)], // v20 -> 4/5
            vec![int(4), real(4.0 / 5.0)], // v20 -> 4/5
            vec![int(5), real(1.0 / 5.0)], // v5  -> 1/5
        ],
    );
}

/// ntile(2) divides the 5-row partition into 2 buckets as evenly as possible with
/// the LARGER bucket first (§3): 3 rows then 2 rows. Under ORDER BY id (unique, so
/// bucket assignment is deterministic) ids 1..3 -> bucket 1 and ids 4,5 -> bucket 2.
#[test]
fn ntile_splits_partition_larger_buckets_first() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT id, ntile(2) OVER (ORDER BY id) FROM t ORDER BY id",
        &[
            vec![int(1), int(1)],
            vec![int(2), int(1)],
            vec![int(3), int(1)],
            vec![int(4), int(2)],
            vec![int(5), int(2)],
        ],
    );
}

// ===========================================================================
// Aggregate window function: sum()  (§2)
// ===========================================================================

/// `sum(v) OVER ()`: no PARTITION BY and no ORDER BY => a single partition in
/// which all rows are peers, so the default frame spans the whole table. Every
/// row sees the grand total 10+30+20+20+5 = 85. sum() of integers is an integer.
#[test]
fn sum_over_empty_window_is_grand_total() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT id, sum(v) OVER () FROM t ORDER BY id",
        &[
            vec![int(1), int(85)],
            vec![int(2), int(85)],
            vec![int(3), int(85)],
            vec![int(4), int(85)],
            vec![int(5), int(85)],
        ],
    );
}

/// Running total: `sum(v) OVER (ORDER BY id)`. With the default RANGE frame
/// (UNBOUNDED PRECEDING .. CURRENT ROW) and a unique ORDER BY key there are no
/// peers, so this is a plain cumulative sum: 10, 40, 60, 80, 85.
#[test]
fn sum_running_total_order_by_id() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT id, sum(v) OVER (ORDER BY id) FROM t ORDER BY id",
        &[
            vec![int(1), int(10)],
            vec![int(2), int(40)],
            vec![int(3), int(60)],
            vec![int(4), int(80)],
            vec![int(5), int(85)],
        ],
    );
}

/// The default frame is RANGE, which includes the current row AND ITS PEERS
/// (§2.2). `sum(v) OVER (ORDER BY v)` sorts to {5,10,20,20,30}; the two v=20
/// rows are peers, so BOTH see the running sum through the end of that peer
/// group (5+10+20+20 = 55), not the individual-row cumulative sums.
///   id5(v5)  -> 5
///   id1(v10) -> 15
///   id3(v20) -> 55   (peer group total pulled in)
///   id4(v20) -> 55
///   id2(v30) -> 85
#[test]
fn sum_default_range_frame_includes_peers() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT id, sum(v) OVER (ORDER BY v) FROM t ORDER BY id",
        &[
            vec![int(1), int(15)],
            vec![int(2), int(85)],
            vec![int(3), int(55)],
            vec![int(4), int(55)],
            vec![int(5), int(5)],
        ],
    );
}

/// Explicit ROWS frame over the unique key `id` gives the same running total as
/// the default RANGE frame here (no peers): 10, 40, 60, 80, 85. This pins the
/// ROWS frame syntax `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`.
#[test]
fn sum_explicit_rows_frame_running_total() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT id, sum(v) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) \
         FROM t ORDER BY id",
        &[
            vec![int(1), int(10)],
            vec![int(2), int(40)],
            vec![int(3), int(60)],
            vec![int(4), int(80)],
            vec![int(5), int(85)],
        ],
    );
}

/// ROWS counts individual rows and does NOT pull in peers (§2.2), so unlike the
/// default RANGE frame the two v=20 rows get DIFFERENT running sums. Sorted by v
/// the prefix sums are 5, 15, 35, 55, 85. Which physical row is 35 vs 55 is
/// unspecified (peer order is arbitrary), but the MULTISET of results is fixed,
/// so this selects only the window value and compares as a multiset. Contrast
/// with `sum_default_range_frame_includes_peers` (both peers -> 55).
#[test]
fn sum_rows_frame_does_not_group_peers_multiset() {
    let mut db = t_db();
    assert_rows_unordered(
        &mut db,
        "SELECT sum(v) OVER (ORDER BY v ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM t",
        &[
            vec![int(5)],
            vec![int(15)],
            vec![int(35)],
            vec![int(55)],
            vec![int(85)],
        ],
    );
}

/// A bounded, symmetric ROWS frame — a sliding window of the row plus its
/// immediate neighbors: `ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING` (§2.2). Over
/// ids 1..5 with v = 10,30,20,20,5 the frame is clamped at the partition edges:
///   id1: {10,30}       = 40   (no preceding row)
///   id2: {10,30,20}    = 60
///   id3: {30,20,20}    = 70
///   id4: {20,20,5}     = 45
///   id5: {20,5}        = 25   (no following row)
#[test]
fn sum_explicit_rows_sliding_frame() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT id, sum(v) OVER (ORDER BY id ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) \
         FROM t ORDER BY id",
        &[
            vec![int(1), int(40)],
            vec![int(2), int(60)],
            vec![int(3), int(70)],
            vec![int(4), int(45)],
            vec![int(5), int(25)],
        ],
    );
}

/// PARTITION BY with no ORDER BY: each partition is a single peer group, so the
/// default frame spans the whole partition. grp 'a' total = 10+30 = 40 (ids
/// 1,2); grp 'b' total = 20+20+5 = 45 (ids 3,4,5).
#[test]
fn sum_over_partition_by_grp() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT id, sum(v) OVER (PARTITION BY grp) FROM t ORDER BY id",
        &[
            vec![int(1), int(40)],
            vec![int(2), int(40)],
            vec![int(3), int(45)],
            vec![int(4), int(45)],
            vec![int(5), int(45)],
        ],
    );
}

// ===========================================================================
// Aggregate window function: count()  (§2)
// ===========================================================================

// The `count()` case is spelled `count(*)` here: the spec documents only
// `count(X)` and `count(*)` (lang_aggfunc.html), so a bare `count()` is not a
// valid form; `count(*)` counts rows in the frame.

/// `count(*) OVER (PARTITION BY grp)`: per-partition row counts. grp 'a' has 2
/// rows (ids 1,2), grp 'b' has 3 rows (ids 3,4,5).
#[test]
fn count_star_over_partition_by_grp() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT id, count(*) OVER (PARTITION BY grp) FROM t ORDER BY id",
        &[
            vec![int(1), int(2)],
            vec![int(2), int(2)],
            vec![int(3), int(3)],
            vec![int(4), int(3)],
            vec![int(5), int(3)],
        ],
    );
}

/// `count(*) OVER ()`: single whole-table partition, so every row sees 5.
#[test]
fn count_star_over_empty_window_is_table_size() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT id, count(*) OVER () FROM t ORDER BY id",
        &[
            vec![int(1), int(5)],
            vec![int(2), int(5)],
            vec![int(3), int(5)],
            vec![int(4), int(5)],
            vec![int(5), int(5)],
        ],
    );
}

// ===========================================================================
// Other aggregates as window functions: avg() / min() / max()  (§2)
// ===========================================================================
//
// One function per test: once window aggregates land, a bug isolated to avg(),
// min(), or max() then pinpoints the exact function instead of surfacing as a
// single bundled failure.

/// avg() over the whole-table window ALWAYS yields a REAL: 85/5 = 17.0. (17.0 is
/// exactly representable, so the exact `value_eq` compare is safe here; a window
/// average over non-evenly-dividing data would need an approximate compare.)
#[test]
fn avg_over_empty_window_is_real() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT id, avg(v) OVER () FROM t ORDER BY id",
        &[
            vec![int(1), real(17.0)],
            vec![int(2), real(17.0)],
            vec![int(3), real(17.0)],
            vec![int(4), real(17.0)],
            vec![int(5), real(17.0)],
        ],
    );
}

/// min() over the whole-table window preserves the INTEGER class: 5.
#[test]
fn min_over_empty_window() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT id, min(v) OVER () FROM t ORDER BY id",
        &[
            vec![int(1), int(5)],
            vec![int(2), int(5)],
            vec![int(3), int(5)],
            vec![int(4), int(5)],
            vec![int(5), int(5)],
        ],
    );
}

/// max() over the whole-table window preserves the INTEGER class: 30.
#[test]
fn max_over_empty_window() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT id, max(v) OVER () FROM t ORDER BY id",
        &[
            vec![int(1), int(30)],
            vec![int(2), int(30)],
            vec![int(3), int(30)],
            vec![int(4), int(30)],
            vec![int(5), int(30)],
        ],
    );
}

// ===========================================================================
// Offset & value built-ins: lag() / lead() / first_value() / last_value() /
// nth_value()  (§3). Every expected value below is transcribed VERBATIM from the
// doc's offset/value example (windowfunctions.html §3, over table t1). The base
// window there is `ORDER BY b ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`;
// the WINDOW clause is inlined into each OVER since it is a separate feature.
// ===========================================================================

/// lag(b) is the previous row's b in ORDER BY b order, NULL for the first row.
/// lag() ignores the frame-spec, so a bare `OVER (ORDER BY b)` suffices.
#[test]
fn lag_default_offset() {
    let mut db = t1_db();
    assert_rows(
        &mut db,
        "SELECT b, lag(b) OVER (ORDER BY b) FROM t1 ORDER BY b",
        &[
            vec![text("A"), null()],
            vec![text("B"), text("A")],
            vec![text("C"), text("B")],
            vec![text("D"), text("C")],
            vec![text("E"), text("D")],
            vec![text("F"), text("E")],
            vec![text("G"), text("F")],
        ],
    );
}

/// lead(b, 2, 'n/a') is b two rows ahead; when there is no such row the supplied
/// default 'n/a' is returned (not NULL). lead() ignores the frame-spec.
#[test]
fn lead_with_offset_and_default() {
    let mut db = t1_db();
    assert_rows(
        &mut db,
        "SELECT b, lead(b, 2, 'n/a') OVER (ORDER BY b) FROM t1 ORDER BY b",
        &[
            vec![text("A"), text("C")],
            vec![text("B"), text("D")],
            vec![text("C"), text("E")],
            vec![text("D"), text("F")],
            vec![text("E"), text("G")],
            vec![text("F"), text("n/a")],
            vec![text("G"), text("n/a")],
        ],
    );
}

/// first_value / last_value RESPECT the frame. With `.. CURRENT ROW`, the frame
/// grows from the partition start to the current row, so first_value(b) is
/// always the first row ('A') and last_value(b) is the CURRENT row's b — NOT the
/// partition's last row (the classic last_value gotcha, §3).
#[test]
fn first_value_and_last_value_frame_sensitive() {
    let mut db = t1_db();
    assert_rows(
        &mut db,
        "SELECT b, \
                first_value(b) OVER (ORDER BY b ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), \
                last_value(b)  OVER (ORDER BY b ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) \
         FROM t1 ORDER BY b",
        &[
            vec![text("A"), text("A"), text("A")],
            vec![text("B"), text("A"), text("B")],
            vec![text("C"), text("A"), text("C")],
            vec![text("D"), text("A"), text("D")],
            vec![text("E"), text("A"), text("E")],
            vec![text("F"), text("A"), text("F")],
            vec![text("G"), text("A"), text("G")],
        ],
    );
}

/// nth_value(b, 3) is b at the 3rd row of the FRAME, or NULL when the frame has
/// fewer than 3 rows. With `.. CURRENT ROW` the frame reaches 3 rows only at the
/// 3rd row onward, so rows 1-2 are NULL and rows 3+ are 'C'.
#[test]
fn nth_value_frame_sensitive() {
    let mut db = t1_db();
    assert_rows(
        &mut db,
        "SELECT b, nth_value(b, 3) OVER (ORDER BY b ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) \
         FROM t1 ORDER BY b",
        &[
            vec![text("A"), null()],
            vec![text("B"), null()],
            vec![text("C"), text("C")],
            vec![text("D"), text("C")],
            vec![text("E"), text("C")],
            vec![text("F"), text("C")],
            vec![text("G"), text("C")],
        ],
    );
}

// ===========================================================================
// Column naming
// ===========================================================================

/// An `AS` alias names the window-expression result column.
#[test]
fn window_expression_alias_column_name() {
    let mut db = t_db();
    assert_columns(
        &mut db,
        "SELECT row_number() OVER (ORDER BY id) AS rn FROM t",
        &["rn"],
    );
}
