//! Conformance battery — ADVANCED window-function surface (frames, EXCLUDE,
//! FILTER, ranking formulas, chaining, NULL ordering, offset/navigation depth).
//!
//! This file DEEPENS `conformance_window.rs` (which covers the basics) and never
//! duplicates it. Every expected value is TRANSCRIBED FROM or DERIVED FROM the
//! SQLite documentation — `spec/sqlite-doc/windowfunctions.html` (and, for the
//! ORDER BY NULL rule, `spec/sqlite-doc/lang_select.html`) — NEVER from what the
//! engine currently returns. Each `#[test]` cites the doc section it encodes.
//!
//! Two provenance classes live here:
//!   * VERBATIM: the doc prints an exact result set for a query over its sample
//!     tables `t1`/`t2`; those outputs are transcribed cell-for-cell (the
//!     highest-value cases). The doc pins the *internal* order of group_concat
//!     over peer rows (e.g. "A.D.G") to input/rowid order; that ordering is
//!     transcribed as-is. If the engine frames peers in another order such a
//!     case may fail — that failure is a genuine, spec-anchored signal.
//!   * DERIVED: small deterministic integer tables (`win`, `winn`) where every
//!     expected value is hand-derived from the documented frame/ranking rules,
//!     shown in each test's comment. The tie at v=10 in `win` exercises peers.
//!
//! Window-frame support in this engine may be incomplete. A case that fails or
//! errors because the engine disagrees with the documented behavior is left LIVE
//! and spec-correct (it is left to fail) — it is never `#[ignore]`d, weakened, or
//! deleted. Cases are split into many small `#[test]` fns so one failing case never
//! masks the rest.
//!
//! `minisqlite::Value` has NO `PartialEq`/`Ord`; all comparisons go through the
//! shared harness helpers (`value_eq`, `real_approx`) — never `==`.

mod conformance;

use conformance::*;
use minisqlite::{Connection, Value};

// ===========================================================================
// Mixed exact/approx row assertion (local; the shared harness is pinned).
//
// Ranking functions percent_rank()/cume_dist() and avg() produce REAL values,
// some non-terminating (cume_dist 1/3, percent_rank 3/5). Such a value must be
// compared within an epsilon, while the INTEGER/TEXT columns in the SAME row
// stay exact. The harness offers exact `assert_rows` and scalar-only
// `assert_scalar_approx`, but no per-cell-mixed row compare — so this file adds
// one. It only reads `query()` output and the harness comparators; it does not
// touch the harness.
// ===========================================================================

/// Absolute tolerance for approximate REAL comparisons (the harness convention).
const APPROX_EPS: f64 = 1e-9;

/// A per-cell expectation for [`assert_rows_mixed`].
#[derive(Debug)]
enum Cell {
    /// Compared with the harness's exact `value_eq`.
    Exact(Value),
    /// The actual cell must be a `Value::Real` within `APPROX_EPS` of this.
    Approx(f64),
}

fn c_int(x: i64) -> Cell {
    Cell::Exact(int(x))
}
fn c_text(s: &str) -> Cell {
    Cell::Exact(text(s))
}
fn c_approx(x: f64) -> Cell {
    Cell::Approx(x)
}

/// ORDERED row comparison where designated cells may be approximate reals.
/// Mirrors the harness `assert_rows` (same shape checks) but each expected cell
/// is a [`Cell`], so a non-terminating REAL is checked within `APPROX_EPS` while
/// exact columns in the same row use `value_eq`. Panics with a full diff.
fn assert_rows_mixed(db: &mut Connection, sql: &str, expected: &[Vec<Cell>]) {
    let qr = query(db, sql);
    let ok = qr.rows.len() == expected.len()
        && qr
            .rows
            .iter()
            .zip(expected.iter())
            .all(|(got_row, exp_row)| {
                got_row.len() == exp_row.len()
                    && got_row.iter().zip(exp_row.iter()).all(|(got, exp)| match exp {
                        Cell::Exact(v) => value_eq(got, v),
                        Cell::Approx(target) => match got {
                            Value::Real(x) => real_approx(*x, *target, APPROX_EPS),
                            _ => false,
                        },
                    })
            });
    if !ok {
        panic!(
            "assert_rows_mixed mismatch\n  sql: {sql}\n  expected: {expected:#?}\n  actual: {:#?}",
            qr.rows
        );
    }
}

// ===========================================================================
// Fixtures.
// ===========================================================================

/// The base table. The tie at v=10 (ids 2,3) exercises peer handling in
/// RANGE/GROUPS frames, EXCLUDE, and the ranking peer-group rules.
/// | id | grp | v  |
/// |  1 | a   |  5 |
/// |  2 | a   | 10 |
/// |  3 | a   | 10 |
/// |  4 | b   | 20 |
/// |  5 | b   | 30 |
fn win_db() -> Connection {
    let mut db = mem();
    exec(
        &mut db,
        "CREATE TABLE win(id INTEGER PRIMARY KEY, grp TEXT, v INTEGER)",
    );
    exec(
        &mut db,
        "INSERT INTO win VALUES (1,'a',5),(2,'a',10),(3,'a',10),(4,'b',20),(5,'b',30)",
    );
    db
}

/// `win` plus a NULL v, for ORDER BY NULL-placement tests.
/// | id | v    |
/// |  1 |  5   |
/// |  2 | 10   |
/// |  3 | NULL |
/// |  4 | 20   |
fn winn_db() -> Connection {
    let mut db = mem();
    exec(
        &mut db,
        "CREATE TABLE winn(id INTEGER PRIMARY KEY, v INTEGER)",
    );
    exec(&mut db, "INSERT INTO winn VALUES (1,5),(2,10),(3,NULL),(4,20)");
    db
}

/// The documentation's sample table `t1` (windowfunctions.html §2 preamble).
/// Used to transcribe the doc's verbatim worked-example outputs.
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

/// The documentation's sample table `t2` (windowfunctions.html §3 preamble).
fn t2_db() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t2(a, b)");
    exec(
        &mut db,
        "INSERT INTO t2 VALUES ('a','one'),('a','two'),('a','three'),\
         ('b','four'),('c','five'),('c','six')",
    );
    db
}

// ===========================================================================
// §2 (aggregate window functions) — VERBATIM sliding-frame group_concat.
// The doc's first worked example: a ROWS frame of the previous, current, and
// next row. ORDER BY a is a unique key, so the output is fully deterministic.
// ===========================================================================

/// windowfunctions.html §2: group_concat over `ROWS BETWEEN 1 PRECEDING AND 1
/// FOLLOWING`, transcribed verbatim. Frame = {prev, current, next} clamped at
/// the partition edges. (The doc's `SELECT ... FROM t1` has no outer ORDER BY;
/// ORDER BY a added here so the transcription is order-stable — a is unique.)
#[test]
fn spec_2_rows_1preceding_1following_group_concat() {
    let mut db = t1_db();
    assert_rows(
        &mut db,
        "SELECT a, group_concat(b, '.') OVER (ORDER BY a ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) \
         FROM t1 ORDER BY a",
        &[
            vec![int(1), text("A.B")],
            vec![int(2), text("A.B.C")],
            vec![int(3), text("B.C.D")],
            vec![int(4), text("C.D.E")],
            vec![int(5), text("D.E.F")],
            vec![int(6), text("E.F.G")],
            vec![int(7), text("F.G")],
        ],
    );
}

// ===========================================================================
// §2.1 The PARTITION BY Clause — VERBATIM. Same window, two output orderings,
// showing that PARTITION BY frames each partition independently regardless of
// how the final result set is ordered.
// ===========================================================================

/// windowfunctions.html §2.1 (first example): `PARTITION BY c ORDER BY a RANGE
/// BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING`, output ordered by (c, a).
/// Within each c-partition (a unique => no peers) the frame runs from the
/// current row to the partition end. Transcribed verbatim.
#[test]
fn spec_2_1_partition_by_c_ordered_by_c_a() {
    let mut db = t1_db();
    assert_rows(
        &mut db,
        "SELECT c, a, b, group_concat(b, '.') OVER \
         (PARTITION BY c ORDER BY a RANGE BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) \
         FROM t1 ORDER BY c, a",
        &[
            vec![text("one"), int(1), text("A"), text("A.D.G")],
            vec![text("one"), int(4), text("D"), text("D.G")],
            vec![text("one"), int(7), text("G"), text("G")],
            vec![text("three"), int(3), text("C"), text("C.F")],
            vec![text("three"), int(6), text("F"), text("F")],
            vec![text("two"), int(2), text("B"), text("B.E")],
            vec![text("two"), int(5), text("E"), text("E")],
        ],
    );
}

/// windowfunctions.html §2.1 (second example): the SAME window as above, but the
/// query is ordered by `a`, so partitions are interleaved in the output while
/// each row's group_concat is unchanged. Transcribed verbatim.
#[test]
fn spec_2_1_partition_by_c_ordered_by_a_scattered() {
    let mut db = t1_db();
    assert_rows(
        &mut db,
        "SELECT c, a, b, group_concat(b, '.') OVER \
         (PARTITION BY c ORDER BY a RANGE BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) \
         FROM t1 ORDER BY a",
        &[
            vec![text("one"), int(1), text("A"), text("A.D.G")],
            vec![text("two"), int(2), text("B"), text("B.E")],
            vec![text("three"), int(3), text("C"), text("C.F")],
            vec![text("one"), int(4), text("D"), text("D.G")],
            vec![text("two"), int(5), text("E"), text("E")],
            vec![text("three"), int(6), text("F"), text("F")],
            vec![text("one"), int(7), text("G"), text("G")],
        ],
    );
}

// ===========================================================================
// §2.2 Frame Specifications — VERBATIM default frame. The default frame is
// `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`, which pulls in the
// current row AND ITS PEERS, so peers share one result.
// ===========================================================================

/// windowfunctions.html §2.2 (default-frame example): `group_concat(b,'.') OVER
/// (ORDER BY c)` with the implicit default RANGE frame. Peers by c => every row
/// in a c-group sees the same frame (start-of-partition .. end of its c-group).
/// Transcribed verbatim; output ordered by a (unique).
#[test]
fn spec_2_2_default_range_frame_pulls_in_peers() {
    let mut db = t1_db();
    assert_rows(
        &mut db,
        "SELECT a, b, c, group_concat(b, '.') OVER (ORDER BY c) FROM t1 ORDER BY a",
        &[
            vec![int(1), text("A"), text("one"), text("A.D.G")],
            vec![int(2), text("B"), text("two"), text("A.D.G.C.F.B.E")],
            vec![int(3), text("C"), text("three"), text("A.D.G.C.F")],
            vec![int(4), text("D"), text("one"), text("A.D.G")],
            vec![int(5), text("E"), text("two"), text("A.D.G.C.F.B.E")],
            vec![int(6), text("F"), text("three"), text("A.D.G.C.F")],
            vec![int(7), text("G"), text("one"), text("A.D.G")],
        ],
    );
}

// ===========================================================================
// §2.2.2 Frame Boundaries — VERBATIM future frame + DERIVED numeric offsets.
// ===========================================================================

/// windowfunctions.html §2.2.2 (worked example): `ROWS BETWEEN CURRENT ROW AND
/// UNBOUNDED FOLLOWING` over `ORDER BY c, a` (a total order). Each frame runs
/// from the current row to the end of the partition. Transcribed verbatim.
#[test]
fn spec_2_2_2_rows_current_to_unbounded_following() {
    let mut db = t1_db();
    assert_rows(
        &mut db,
        "SELECT c, a, b, group_concat(b, '.') OVER \
         (ORDER BY c, a ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) \
         FROM t1 ORDER BY c, a",
        &[
            vec![text("one"), int(1), text("A"), text("A.D.G.C.F.B.E")],
            vec![text("one"), int(4), text("D"), text("D.G.C.F.B.E")],
            vec![text("one"), int(7), text("G"), text("G.C.F.B.E")],
            vec![text("three"), int(3), text("C"), text("C.F.B.E")],
            vec![text("three"), int(6), text("F"), text("F.B.E")],
            vec![text("two"), int(2), text("B"), text("B.E")],
            vec![text("two"), int(5), text("E"), text("E")],
        ],
    );
}

/// §2.2.2 (`<expr> PRECEDING/FOLLOWING`, RANGE): a RANGE numeric offset frames
/// rows whose ORDER BY value is within [Xc-5, Xc+5]. DERIVED on `win` (v =
/// 5,10,10,20,30):
///   v5  -> [0,10]  = {5,10,10}         = 25
///   v10 -> [5,15]  = {5,10,10}         = 25
///   v10 -> [5,15]                       = 25
///   v20 -> [15,25] = {20}              = 20
///   v30 -> [25,35] = {30}              = 30
#[test]
fn range_numeric_offset_5_preceding_5_following() {
    let mut db = win_db();
    assert_rows(
        &mut db,
        "SELECT id, sum(v) OVER (ORDER BY v RANGE BETWEEN 5 PRECEDING AND 5 FOLLOWING) \
         FROM win ORDER BY id",
        &[
            vec![int(1), int(25)],
            vec![int(2), int(25)],
            vec![int(3), int(25)],
            vec![int(4), int(20)],
            vec![int(5), int(30)],
        ],
    );
}

/// §2.2.1/§2.2.2 contrast: the ROWS frame with the SAME literal offsets counts
/// rows, not values. With only 5 rows, `ROWS BETWEEN 5 PRECEDING AND 5
/// FOLLOWING` spans the whole partition for every row => 75 each. This differs
/// from the RANGE version above (25/25/25/20/30), proving RANGE != ROWS.
#[test]
fn rows_numeric_offset_differs_from_range() {
    let mut db = win_db();
    assert_rows(
        &mut db,
        "SELECT id, sum(v) OVER (ORDER BY v ROWS BETWEEN 5 PRECEDING AND 5 FOLLOWING) \
         FROM win ORDER BY id",
        &[
            vec![int(1), int(75)],
            vec![int(2), int(75)],
            vec![int(3), int(75)],
            vec![int(4), int(75)],
            vec![int(5), int(75)],
        ],
    );
}

/// §2.2.2 CURRENT ROW under RANGE includes the whole peer group, unlike ROWS.
/// `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` (== the default frame)
/// on `win` ORDER BY v: both v=10 peers see the sum through the end of their
/// peer group.
///   v5  -> 5
///   v10 -> 5+10+10        = 25   (peer pulled in)
///   v10 -> 25
///   v20 -> 5+10+10+20     = 45
///   v30 -> 75
#[test]
fn range_unbounded_to_current_includes_peer_group() {
    let mut db = win_db();
    assert_rows(
        &mut db,
        "SELECT id, sum(v) OVER (ORDER BY v RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) \
         FROM win ORDER BY id",
        &[
            vec![int(1), int(5)],
            vec![int(2), int(25)],
            vec![int(3), int(25)],
            vec![int(4), int(45)],
            vec![int(5), int(75)],
        ],
    );
}

/// §2.2.1/§2.2.2 contrast partner: the ROWS frame does NOT pull in peers, so the
/// two v=10 rows get different running sums. Prefix sums over v-sorted rows are
/// {5,15,25,45,75}; which physical row is 15 vs 25 is peer-order dependent, so
/// only the window value is selected and compared as a MULTISET.
#[test]
fn rows_unbounded_to_current_no_peer_grouping_multiset() {
    let mut db = win_db();
    assert_rows_unordered(
        &mut db,
        "SELECT sum(v) OVER (ORDER BY v ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM win",
        &[
            vec![int(5)],
            vec![int(15)],
            vec![int(25)],
            vec![int(45)],
            vec![int(75)],
        ],
    );
}

// ---- §2.2.2 future-only frames & empty frames -----------------------------

/// §2.2.2 future-only frame `ROWS BETWEEN 1 FOLLOWING AND 2 FOLLOWING` on `win`
/// ORDER BY id, exercising the EMPTY-frame tail. For each aggregate the frame is
/// the next two rows; the last rows run off the end:
///   id1 {10,10}  sum20 cnt2 min10 max10 avg10 fv10 nth2=10
///   id2 {10,20}  sum30 cnt2 min10 max20 avg15 fv10 nth2=20
///   id3 {20,30}  sum50 cnt2 min20 max30 avg25 fv20 nth2=30
///   id4 {30}     sum30 cnt1 min30 max30 avg30 fv30 nth2=NULL (only 1 row)
///   id5 {}       sum=NULL cnt0 min/max/avg=NULL fv=NULL nth2=NULL
/// Documents: empty aggregate -> NULL, empty count -> 0, empty navigation -> NULL.
#[test]
fn future_frame_and_empty_frame_semantics() {
    let mut db = win_db();
    assert_rows(
        &mut db,
        "SELECT id, sum(v) OVER w, count(*) OVER w, min(v) OVER w, max(v) OVER w, \
                avg(v) OVER w, first_value(v) OVER w, nth_value(v,2) OVER w \
         FROM win \
         WINDOW w AS (ORDER BY id ROWS BETWEEN 1 FOLLOWING AND 2 FOLLOWING) \
         ORDER BY id",
        &[
            vec![int(1), int(20), int(2), int(10), int(10), real(10.0), int(10), int(10)],
            vec![int(2), int(30), int(2), int(10), int(20), real(15.0), int(10), int(20)],
            vec![int(3), int(50), int(2), int(20), int(30), real(25.0), int(20), int(30)],
            vec![int(4), int(30), int(1), int(30), int(30), real(30.0), int(30), null()],
            vec![int(5), null(), int(0), null(), null(), null(), null(), null()],
        ],
    );
}

/// §2.2.2 `ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING` is never empty
/// (always includes the current row). On `win` ORDER BY id it is a reverse
/// running total: 75, 70, 60, 50, 30.
#[test]
fn current_to_unbounded_following_reverse_running_total() {
    let mut db = win_db();
    assert_rows(
        &mut db,
        "SELECT id, sum(v) OVER (ORDER BY id ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) \
         FROM win ORDER BY id",
        &[
            vec![int(1), int(75)],
            vec![int(2), int(70)],
            vec![int(3), int(60)],
            vec![int(4), int(50)],
            vec![int(5), int(30)],
        ],
    );
}

// ===========================================================================
// §2.2.1 GROUPS frame type — DERIVED. GROUPS counts PEER GROUPS, not rows.
// `win` ORDER BY v has groups G0={5}, G1={10,10}, G2={20}, G3={30}.
// ===========================================================================

/// §2.2.1 GROUPS `BETWEEN 1 PRECEDING AND 1 FOLLOWING` spans the neighbouring
/// peer groups (clamped at the partition edge):
///   id1 G0: G0..G1 = {5,10,10}        = 25
///   id2 G1: G0..G2 = {5,10,10,20}     = 45
///   id3 G1: same                       = 45
///   id4 G2: G1..G3 = {10,10,20,30}    = 70
///   id5 G3: G2..G3 = {20,30}          = 50
#[test]
fn groups_1_preceding_1_following_sum() {
    let mut db = win_db();
    assert_rows(
        &mut db,
        "SELECT id, sum(v) OVER (ORDER BY v GROUPS BETWEEN 1 PRECEDING AND 1 FOLLOWING) \
         FROM win ORDER BY id",
        &[
            vec![int(1), int(25)],
            vec![int(2), int(45)],
            vec![int(3), int(45)],
            vec![int(4), int(70)],
            vec![int(5), int(50)],
        ],
    );
}

/// §2.2.1 GROUPS `BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING`: CURRENT ROW in a
/// GROUPS frame is the whole current peer group, then everything after it.
/// count(*):
///   id1 G0: all 5 rows               = 5
///   id2 G1: {10,10,20,30}            = 4
///   id3 G1: same                      = 4
///   id4 G2: {20,30}                  = 2
///   id5 G3: {30}                     = 1
#[test]
fn groups_current_to_unbounded_following_count() {
    let mut db = win_db();
    assert_rows(
        &mut db,
        "SELECT id, count(*) OVER (ORDER BY v GROUPS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) \
         FROM win ORDER BY id",
        &[
            vec![int(1), int(5)],
            vec![int(2), int(4)],
            vec![int(3), int(4)],
            vec![int(4), int(2)],
            vec![int(5), int(1)],
        ],
    );
}

/// §2.2.1 GROUPS `BETWEEN CURRENT ROW AND CURRENT ROW` captures exactly the
/// current peer group (all peers), unlike a ROWS CURRENT..CURRENT frame which
/// would be just the one row:
///   id1 {5}      = 5
///   id2 {10,10}  = 20
///   id3 {10,10}  = 20
///   id4 {20}     = 20
///   id5 {30}     = 30
#[test]
fn groups_current_row_captures_whole_peer_group() {
    let mut db = win_db();
    assert_rows(
        &mut db,
        "SELECT id, sum(v) OVER (ORDER BY v GROUPS BETWEEN CURRENT ROW AND CURRENT ROW) \
         FROM win ORDER BY id",
        &[
            vec![int(1), int(5)],
            vec![int(2), int(20)],
            vec![int(3), int(20)],
            vec![int(4), int(20)],
            vec![int(5), int(30)],
        ],
    );
}

// ===========================================================================
// §2.2.3 The EXCLUDE Clause.
// ===========================================================================

/// windowfunctions.html §2.2.3 (worked example): all four EXCLUDE forms in one
/// query over `t1`, GROUPS frame `UNBOUNDED PRECEDING AND CURRENT ROW`.
/// Transcribed VERBATIM. Note the empty-frame cells (EXCLUDE GROUP for the
/// first peer group) render as group_concat over zero rows => NULL.
#[test]
fn spec_2_2_3_exclude_all_four_forms_verbatim() {
    let mut db = t1_db();
    assert_rows(
        &mut db,
        "SELECT c, a, b, \
           group_concat(b, '.') OVER (ORDER BY c GROUPS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE NO OTHERS), \
           group_concat(b, '.') OVER (ORDER BY c GROUPS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE CURRENT ROW), \
           group_concat(b, '.') OVER (ORDER BY c GROUPS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE GROUP), \
           group_concat(b, '.') OVER (ORDER BY c GROUPS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE TIES) \
         FROM t1 ORDER BY c, a",
        &[
            vec![text("one"), int(1), text("A"), text("A.D.G"), text("D.G"), null(), text("A")],
            vec![text("one"), int(4), text("D"), text("A.D.G"), text("A.G"), null(), text("D")],
            vec![text("one"), int(7), text("G"), text("A.D.G"), text("A.D"), null(), text("G")],
            vec![text("three"), int(3), text("C"), text("A.D.G.C.F"), text("A.D.G.F"), text("A.D.G"), text("A.D.G.C")],
            vec![text("three"), int(6), text("F"), text("A.D.G.C.F"), text("A.D.G.C"), text("A.D.G"), text("A.D.G.F")],
            vec![text("two"), int(2), text("B"), text("A.D.G.C.F.B.E"), text("A.D.G.C.F.E"), text("A.D.G.C.F"), text("A.D.G.C.F.B")],
            vec![text("two"), int(5), text("E"), text("A.D.G.C.F.B.E"), text("A.D.G.C.F.B"), text("A.D.G.C.F"), text("A.D.G.C.F.E")],
        ],
    );
}

// EXCLUDE forms isolated on `win`, frame `ROWS BETWEEN UNBOUNDED PRECEDING AND
// UNBOUNDED FOLLOWING` (whole partition) with ORDER BY v so the v=10 tie makes
// each variant distinct. group_concat(id) exposes WHICH rows survive (§2.2.3:
// "all rows with the same ORDER BY values ... are considered peers, even if the
// frame type is ROWS"). Base frame = {1,2,3,4,5}, sum 75, count 5.

/// §2.2.3 EXCLUDE NO OTHERS: nothing removed; every row sees the whole partition.
#[test]
fn exclude_no_others_is_full_frame() {
    let mut db = win_db();
    assert_rows(
        &mut db,
        "SELECT id, sum(v) OVER w, count(*) OVER w, group_concat(id) OVER w FROM win \
         WINDOW w AS (ORDER BY v ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING EXCLUDE NO OTHERS) \
         ORDER BY id",
        &[
            vec![int(1), int(75), int(5), text("1,2,3,4,5")],
            vec![int(2), int(75), int(5), text("1,2,3,4,5")],
            vec![int(3), int(75), int(5), text("1,2,3,4,5")],
            vec![int(4), int(75), int(5), text("1,2,3,4,5")],
            vec![int(5), int(75), int(5), text("1,2,3,4,5")],
        ],
    );
}

/// §2.2.3 EXCLUDE CURRENT ROW: drop only the current row (peers stay).
///   id1 -> {2,3,4,5} 70   id2 -> {1,3,4,5} 65   id3 -> {1,2,4,5} 65
///   id4 -> {1,2,3,5} 55   id5 -> {1,2,3,4} 45
#[test]
fn exclude_current_row_drops_only_current() {
    let mut db = win_db();
    assert_rows(
        &mut db,
        "SELECT id, sum(v) OVER w, count(*) OVER w, group_concat(id) OVER w FROM win \
         WINDOW w AS (ORDER BY v ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING EXCLUDE CURRENT ROW) \
         ORDER BY id",
        &[
            vec![int(1), int(70), int(4), text("2,3,4,5")],
            vec![int(2), int(65), int(4), text("1,3,4,5")],
            vec![int(3), int(65), int(4), text("1,2,4,5")],
            vec![int(4), int(55), int(4), text("1,2,3,5")],
            vec![int(5), int(45), int(4), text("1,2,3,4")],
        ],
    );
}

/// §2.2.3 EXCLUDE GROUP: drop the current row AND all its peers. The v=10 tie
/// (ids 2,3) drops both for either row.
///   id1 -> {2,3,4,5} 70   id2 -> {1,4,5} 55   id3 -> {1,4,5} 55
///   id4 -> {1,2,3,5} 55   id5 -> {1,2,3,4} 45
#[test]
fn exclude_group_drops_current_and_all_peers() {
    let mut db = win_db();
    assert_rows(
        &mut db,
        "SELECT id, sum(v) OVER w, count(*) OVER w, group_concat(id) OVER w FROM win \
         WINDOW w AS (ORDER BY v ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING EXCLUDE GROUP) \
         ORDER BY id",
        &[
            vec![int(1), int(70), int(4), text("2,3,4,5")],
            vec![int(2), int(55), int(3), text("1,4,5")],
            vec![int(3), int(55), int(3), text("1,4,5")],
            vec![int(4), int(55), int(4), text("1,2,3,5")],
            vec![int(5), int(45), int(4), text("1,2,3,4")],
        ],
    );
}

/// §2.2.3 EXCLUDE TIES: drop the current row's peers but KEEP the current row.
/// For the v=10 tie this is visibly different from EXCLUDE CURRENT ROW: id2
/// keeps itself and drops peer id3 ({1,2,4,5}), whereas EXCLUDE CURRENT ROW kept
/// id3 ({1,3,4,5}). Rows with no peers (ids 1,4,5) keep the full frame.
///   id1 -> {1,2,3,4,5} 75   id2 -> {1,2,4,5} 65   id3 -> {1,3,4,5} 65
///   id4 -> {1,2,3,4,5} 75   id5 -> {1,2,3,4,5} 75
#[test]
fn exclude_ties_drops_peers_keeps_current() {
    let mut db = win_db();
    assert_rows(
        &mut db,
        "SELECT id, sum(v) OVER w, count(*) OVER w, group_concat(id) OVER w FROM win \
         WINDOW w AS (ORDER BY v ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING EXCLUDE TIES) \
         ORDER BY id",
        &[
            vec![int(1), int(75), int(5), text("1,2,3,4,5")],
            vec![int(2), int(65), int(4), text("1,2,4,5")],
            vec![int(3), int(65), int(4), text("1,3,4,5")],
            vec![int(4), int(75), int(5), text("1,2,3,4,5")],
            vec![int(5), int(75), int(5), text("1,2,3,4,5")],
        ],
    );
}

// ===========================================================================
// §2.3 The FILTER Clause.
// ===========================================================================

/// windowfunctions.html §2.3 (worked example): `group_concat(b,'.') FILTER
/// (WHERE c!='two') OVER (ORDER BY a)`. FILTER removes non-matching rows from the
/// aggregate input BEFORE framing; the default frame then accumulates the
/// surviving rows up to the current row. Transcribed VERBATIM.
#[test]
fn spec_2_3_filter_clause_verbatim() {
    let mut db = t1_db();
    assert_rows(
        &mut db,
        "SELECT c, a, b, group_concat(b, '.') FILTER (WHERE c!='two') OVER (ORDER BY a) \
         FROM t1 ORDER BY a",
        &[
            vec![text("one"), int(1), text("A"), text("A")],
            vec![text("two"), int(2), text("B"), text("A")],
            vec![text("three"), int(3), text("C"), text("A.C")],
            vec![text("one"), int(4), text("D"), text("A.C.D")],
            vec![text("two"), int(5), text("E"), text("A.C.D")],
            vec![text("three"), int(6), text("F"), text("A.C.D.F")],
            vec![text("one"), int(7), text("G"), text("A.C.D.F.G")],
        ],
    );
}

/// §2.3 FILTER on `win`: `sum(v) FILTER (WHERE v <> 10)` over a running ROWS
/// frame. The two v=10 rows are removed from the aggregate input, so the running
/// sum only accumulates 5, 20, 30 as they enter the frame:
///   id1 {5}          = 5    id2 {5}          = 5    id3 {5}          = 5
///   id4 {5,20}       = 25   id5 {5,20,30}    = 55
#[test]
fn filter_excludes_rows_from_running_aggregate() {
    let mut db = win_db();
    assert_rows(
        &mut db,
        "SELECT id, sum(v) FILTER (WHERE v <> 10) OVER \
         (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) \
         FROM win ORDER BY id",
        &[
            vec![int(1), int(5)],
            vec![int(2), int(5)],
            vec![int(3), int(5)],
            vec![int(4), int(25)],
            vec![int(5), int(55)],
        ],
    );
}

/// §2.3 FILTER with `count(*) FILTER (WHERE v>=20) OVER ()`: the whole-partition
/// window counts only rows passing the filter (ids 4,5) => every row sees 2.
#[test]
fn filter_with_count_over_empty_window() {
    let mut db = win_db();
    assert_rows(
        &mut db,
        "SELECT id, count(*) FILTER (WHERE v>=20) OVER () FROM win ORDER BY id",
        &[
            vec![int(1), int(2)],
            vec![int(2), int(2)],
            vec![int(3), int(2)],
            vec![int(4), int(2)],
            vec![int(5), int(2)],
        ],
    );
}

/// §3: "It is a syntax error to specify a FILTER clause as part of a built-in
/// window function invocation." row_number() is a built-in window function, so
/// `row_number() FILTER (WHERE ...) OVER ()` must be REJECTED.
#[test]
fn filter_on_ranking_function_is_error() {
    let mut db = win_db();
    assert_query_error(
        &mut db,
        "SELECT row_number() FILTER (WHERE v>0) OVER () FROM win",
    );
}

/// §3 (deepening the case above to the whole class): FILTER rejection is a CLASS
/// invariant — "It is a syntax error to specify a FILTER clause as part of a
/// built-in window function invocation" — covering ALL 11 built-in window
/// functions, not just row_number(). Each `OVER (ORDER BY id)` is well-formed on
/// its own, so the ONLY reason to reject is the FILTER clause. An engine that
/// centralizes the check passes; one that forgot a specific built-in fails here
/// with the offending function named. (FILTER on a true AGGREGATE window
/// function — sum/count/… — is legal and is exercised by the passing FILTER tests
/// above, so this is specific to the non-aggregate built-ins.)
#[test]
fn filter_on_every_builtin_window_function_is_error() {
    let builtins = [
        "row_number()",
        "rank()",
        "dense_rank()",
        "percent_rank()",
        "cume_dist()",
        "ntile(2)",
        "lag(v)",
        "lead(v)",
        "first_value(v)",
        "last_value(v)",
        "nth_value(v, 1)",
    ];
    let mut wrongly_accepted = Vec::new();
    for call in builtins {
        let mut db = win_db();
        let sql = format!("SELECT {call} FILTER (WHERE v > 0) OVER (ORDER BY id) FROM win");
        if try_query(&mut db, &sql).is_ok() {
            wrongly_accepted.push(call);
        }
    }
    assert!(
        wrongly_accepted.is_empty(),
        "FILTER on a built-in window function must be a syntax error (§3), \
         but these were accepted: {wrongly_accepted:?}"
    );
}

// ===========================================================================
// §3 Built-in Window Functions — ranking-formula edge cases.
// ===========================================================================

/// windowfunctions.html §3 (five-ranking-functions example) over `t2`,
/// `WINDOW win AS (ORDER BY a)`. Peer-invariant columns transcribed VERBATIM;
/// row_number() is omitted because its value among peers is arbitrary. The
/// peer-invariant rows are identical within a group, so an a-ordered compare is
/// well-defined despite the ties. percent_rank = (rank-1)/(N-1), N=6;
/// cume_dist = (row_number of last peer)/N. Reals compared within eps (0.6, 0.8,
/// 4/6 are non-terminating; the doc's "0.66" is a display truncation of 4/6).
#[test]
fn spec_3_percent_rank_and_cume_dist_verbatim() {
    let mut db = t2_db();
    assert_rows_mixed(
        &mut db,
        "SELECT a, rank() OVER win, dense_rank() OVER win, percent_rank() OVER win, cume_dist() OVER win \
         FROM t2 WINDOW win AS (ORDER BY a) ORDER BY a",
        &[
            vec![c_text("a"), c_int(1), c_int(1), c_approx(0.0), c_approx(0.5)],
            vec![c_text("a"), c_int(1), c_int(1), c_approx(0.0), c_approx(0.5)],
            vec![c_text("a"), c_int(1), c_int(1), c_approx(0.0), c_approx(0.5)],
            vec![c_text("b"), c_int(4), c_int(2), c_approx(3.0 / 5.0), c_approx(4.0 / 6.0)],
            vec![c_text("c"), c_int(5), c_int(3), c_approx(4.0 / 5.0), c_approx(1.0)],
            vec![c_text("c"), c_int(5), c_int(3), c_approx(4.0 / 5.0), c_approx(1.0)],
        ],
    );
}

/// §3 percent_rank = (rank-1)/(partition_rows-1), partitioned. On `win`
/// PARTITION BY grp ORDER BY v: grp 'a' N=3 ranks {1,2,2} -> {0/2, 1/2, 1/2} =
/// {0.0,0.5,0.5}; grp 'b' N=2 ranks {1,2} -> {0/1, 1/1} = {0.0,1.0}. All values
/// are exactly representable, so compared exactly.
#[test]
fn percent_rank_partitioned_exact() {
    let mut db = win_db();
    assert_rows(
        &mut db,
        "SELECT id, percent_rank() OVER (PARTITION BY grp ORDER BY v) FROM win ORDER BY id",
        &[
            vec![int(1), real(0.0)],
            vec![int(2), real(0.5)],
            vec![int(3), real(0.5)],
            vec![int(4), real(0.0)],
            vec![int(5), real(1.0)],
        ],
    );
}

/// §3 percent_rank single-row-partition rule: "If the partition contains only
/// one row, this function returns 0.0." PARTITION BY id makes every partition a
/// singleton (N=1) => 0.0 for all rows.
#[test]
fn percent_rank_single_row_partition_is_zero() {
    let mut db = win_db();
    assert_rows(
        &mut db,
        "SELECT id, percent_rank() OVER (PARTITION BY id ORDER BY v) FROM win ORDER BY id",
        &[
            vec![int(1), real(0.0)],
            vec![int(2), real(0.0)],
            vec![int(3), real(0.0)],
            vec![int(4), real(0.0)],
            vec![int(5), real(0.0)],
        ],
    );
}

/// §3 cume_dist = (row_number of last peer in the group)/partition_rows,
/// partitioned. On `win` PARTITION BY grp ORDER BY v:
///   grp 'a' N=3: v5 last-peer-rn=1 -> 1/3 ; v10 peers last-peer-rn=3 -> 3/3=1.0 (both)
///   grp 'b' N=2: v20 last-peer-rn=1 -> 1/2=0.5 ; v30 last-peer-rn=2 -> 2/2=1.0
/// 1/3 is non-terminating -> approx compare.
#[test]
fn cume_dist_partitioned() {
    let mut db = win_db();
    assert_rows_mixed(
        &mut db,
        "SELECT id, cume_dist() OVER (PARTITION BY grp ORDER BY v) FROM win ORDER BY id",
        &[
            vec![c_int(1), c_approx(1.0 / 3.0)],
            vec![c_int(2), c_approx(1.0)],
            vec![c_int(3), c_approx(1.0)],
            vec![c_int(4), c_approx(0.5)],
            vec![c_int(5), c_approx(1.0)],
        ],
    );
}

/// §3 ntile(2) over the 5-row partition (ORDER BY id for determinism). 5 rows
/// into 2 buckets = 3+2 with the LARGER bucket first: ids 1,2,3 -> 1; ids 4,5 -> 2.
#[test]
fn ntile_2_larger_bucket_first() {
    let mut db = win_db();
    assert_rows(
        &mut db,
        "SELECT id, ntile(2) OVER (ORDER BY id) FROM win ORDER BY id",
        &[
            vec![int(1), int(1)],
            vec![int(2), int(1)],
            vec![int(3), int(1)],
            vec![int(4), int(2)],
            vec![int(5), int(2)],
        ],
    );
}

/// §3 ntile(3) over 5 rows = 2+2+1 (larger buckets first): ids 1,2 -> 1;
/// ids 3,4 -> 2; id 5 -> 3.
#[test]
fn ntile_3_uneven_buckets() {
    let mut db = win_db();
    assert_rows(
        &mut db,
        "SELECT id, ntile(3) OVER (ORDER BY id) FROM win ORDER BY id",
        &[
            vec![int(1), int(1)],
            vec![int(2), int(1)],
            vec![int(3), int(2)],
            vec![int(4), int(2)],
            vec![int(5), int(3)],
        ],
    );
}

/// §3 ntile(1) puts every row in bucket 1.
#[test]
fn ntile_1_single_bucket() {
    let mut db = win_db();
    assert_rows(
        &mut db,
        "SELECT id, ntile(1) OVER (ORDER BY id) FROM win ORDER BY id",
        &[
            vec![int(1), int(1)],
            vec![int(2), int(1)],
            vec![int(3), int(1)],
            vec![int(4), int(1)],
            vec![int(5), int(1)],
        ],
    );
}

/// §3 ntile(k) with k > rows: buckets fill one row each and the surplus buckets
/// are empty. PARTITION BY grp ORDER BY id, ntile(5): grp 'a' (3 rows) ->
/// buckets 1,2,3 (4,5 empty); grp 'b' (2 rows) -> buckets 1,2 (3,4,5 empty).
#[test]
fn ntile_5_more_buckets_than_rows() {
    let mut db = win_db();
    assert_rows(
        &mut db,
        "SELECT id, grp, ntile(5) OVER (PARTITION BY grp ORDER BY id) FROM win ORDER BY id",
        &[
            vec![int(1), text("a"), int(1)],
            vec![int(2), text("a"), int(2)],
            vec![int(3), text("a"), int(3)],
            vec![int(4), text("b"), int(1)],
            vec![int(5), text("b"), int(2)],
        ],
    );
}

/// §3 rank() (with gaps) vs dense_rank() (no gaps) on the v=10 tie in one query.
/// Over sorted v [5,10,10,20,30]: rank = [1,2,2,4,5] (gap after the 2-way tie);
/// dense_rank = [1,2,2,3,4] (no gap). Ordered by v; the two tied rows are
/// identical in (v,rank,dense_rank) so the ordered compare is well-defined.
#[test]
fn rank_gaps_vs_dense_rank_no_gaps_on_tie() {
    let mut db = win_db();
    assert_rows(
        &mut db,
        "SELECT v, rank() OVER w, dense_rank() OVER w FROM win WINDOW w AS (ORDER BY v) ORDER BY v",
        &[
            vec![int(5), int(1), int(1)],
            vec![int(10), int(2), int(2)],
            vec![int(10), int(2), int(2)],
            vec![int(20), int(4), int(3)],
            vec![int(30), int(5), int(4)],
        ],
    );
}

// ===========================================================================
// §3 ntile over the doc's t2 table — VERBATIM.
// ===========================================================================

/// windowfunctions.html §3 (ntile example) over `t2`, ntile(2): 6 rows into 2
/// buckets of 3. The 'a' peer group (3 rows) fills bucket 1; b and the two c's
/// are bucket 2. The per-(a) assignment is peer-invariant, so transcribed
/// verbatim, a-ordered.
#[test]
fn spec_3_ntile_2_verbatim() {
    let mut db = t2_db();
    assert_rows(
        &mut db,
        "SELECT a, ntile(2) OVER win FROM t2 WINDOW win AS (ORDER BY a) ORDER BY a",
        &[
            vec![text("a"), int(1)],
            vec![text("a"), int(1)],
            vec![text("a"), int(1)],
            vec![text("b"), int(2)],
            vec![text("c"), int(2)],
            vec![text("c"), int(2)],
        ],
    );
}

/// windowfunctions.html §3 (ntile example) over `t2`, ntile(4): 6 rows into
/// 2+2+1+1 (larger buckets first). Because the 'a' peer group straddles buckets
/// 1 and 2 and peer order is arbitrary, only the MULTISET of bucket labels is
/// well-defined: {1,1,2,2,3,4}.
#[test]
fn spec_3_ntile_4_multiset() {
    let mut db = t2_db();
    assert_rows_unordered(
        &mut db,
        "SELECT ntile(4) OVER win FROM t2 WINDOW win AS (ORDER BY a)",
        &[
            vec![int(1)],
            vec![int(1)],
            vec![int(2)],
            vec![int(2)],
            vec![int(3)],
            vec![int(4)],
        ],
    );
}

// ===========================================================================
// §3 lag()/lead() depth — offset, explicit default, offset 0, partition edges.
// PARTITION BY grp ORDER BY id: partitions 'a'={1,2,3}, 'b'={4,5}.
// ===========================================================================

/// §3 lag(v, 2, -1): value two rows back within the partition, else the explicit
/// default -1 (NOT NULL). grp 'a': id1,id2 have no row 2 back -> -1; id3 -> id1's
/// v=5. grp 'b': id4,id5 have no row 2 back -> -1.
#[test]
fn lag_offset_2_explicit_default() {
    let mut db = win_db();
    assert_rows(
        &mut db,
        "SELECT id, lag(v, 2, -1) OVER (PARTITION BY grp ORDER BY id) FROM win ORDER BY id",
        &[
            vec![int(1), int(-1)],
            vec![int(2), int(-1)],
            vec![int(3), int(5)],
            vec![int(4), int(-1)],
            vec![int(5), int(-1)],
        ],
    );
}

/// §3 lead(v, 1): next row within the partition, else NULL at the partition end.
/// grp 'a': id1->10, id2->10, id3->NULL. grp 'b': id4->30, id5->NULL.
#[test]
fn lead_offset_1_null_at_partition_end() {
    let mut db = win_db();
    assert_rows(
        &mut db,
        "SELECT id, lead(v, 1) OVER (PARTITION BY grp ORDER BY id) FROM win ORDER BY id",
        &[
            vec![int(1), int(10)],
            vec![int(2), int(10)],
            vec![int(3), null()],
            vec![int(4), int(30)],
            vec![int(5), null()],
        ],
    );
}

/// §3 lag(v, 0): "If offset is 0, then expr is evaluated against the current
/// row." So this just returns v.
#[test]
fn lag_offset_0_is_current_row() {
    let mut db = win_db();
    assert_rows(
        &mut db,
        "SELECT id, lag(v, 0) OVER (PARTITION BY grp ORDER BY id) FROM win ORDER BY id",
        &[
            vec![int(1), int(5)],
            vec![int(2), int(10)],
            vec![int(3), int(10)],
            vec![int(4), int(20)],
            vec![int(5), int(30)],
        ],
    );
}

// ===========================================================================
// §3 navigation functions with explicit frames & EXCLUDE.
// ===========================================================================

/// §3 nth_value(v, 2) with `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`:
/// NULL until the frame holds 2 rows, then the value of the frame's 2nd row (id2,
/// v=10) thereafter.
#[test]
fn nth_value_2_growing_frame() {
    let mut db = win_db();
    assert_rows(
        &mut db,
        "SELECT id, nth_value(v, 2) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) \
         FROM win ORDER BY id",
        &[
            vec![int(1), null()],
            vec![int(2), int(10)],
            vec![int(3), int(10)],
            vec![int(4), int(10)],
            vec![int(5), int(10)],
        ],
    );
}

/// §3 last_value gotcha under the DEFAULT frame (RANGE UNBOUNDED PRECEDING AND
/// CURRENT ROW): last_value is the CURRENT row's value, NOT the partition max.
/// ORDER BY id is unique (no peers), so last_value(v) == the row's own v.
#[test]
fn last_value_default_frame_is_current_row_not_max() {
    let mut db = win_db();
    assert_rows(
        &mut db,
        "SELECT id, last_value(v) OVER (ORDER BY id) FROM win ORDER BY id",
        &[
            vec![int(1), int(5)],
            vec![int(2), int(10)],
            vec![int(3), int(10)],
            vec![int(4), int(20)],
            vec![int(5), int(30)],
        ],
    );
}

/// §3 last_value with an explicit full frame `ROWS BETWEEN UNBOUNDED PRECEDING
/// AND UNBOUNDED FOLLOWING` DOES reach the partition's last row (v=30 for all).
/// Contrasts the default-frame gotcha above.
#[test]
fn last_value_full_frame_is_partition_last() {
    let mut db = win_db();
    assert_rows(
        &mut db,
        "SELECT id, last_value(v) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) \
         FROM win ORDER BY id",
        &[
            vec![int(1), int(30)],
            vec![int(2), int(30)],
            vec![int(3), int(30)],
            vec![int(4), int(30)],
            vec![int(5), int(30)],
        ],
    );
}

/// §2.2.3 + §3: first_value/last_value over a full frame with EXCLUDE CURRENT
/// ROW. The current row is removed, so first_value is the first OTHER row and
/// last_value the last OTHER row:
///   id1 frame {2,3,4,5}: fv=10(id2) lv=30(id5)
///   id2 frame {1,3,4,5}: fv=5       lv=30
///   id3 frame {1,2,4,5}: fv=5       lv=30
///   id4 frame {1,2,3,5}: fv=5       lv=30
///   id5 frame {1,2,3,4}: fv=5       lv=20(id4)
#[test]
fn first_last_value_with_exclude_current_row() {
    let mut db = win_db();
    assert_rows(
        &mut db,
        "SELECT id, first_value(v) OVER w, last_value(v) OVER w FROM win \
         WINDOW w AS (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING EXCLUDE CURRENT ROW) \
         ORDER BY id",
        &[
            vec![int(1), int(10), int(30)],
            vec![int(2), int(5), int(30)],
            vec![int(3), int(5), int(30)],
            vec![int(4), int(5), int(30)],
            vec![int(5), int(5), int(20)],
        ],
    );
}

/// §2.2.3 + §3: nth_value(v, 3) over a full frame with EXCLUDE CURRENT ROW —
/// the excluded current row shifts which row is "3rd":
///   id1 frame [2,3,4,5] 3rd=id4 -> 20
///   id2 frame [1,3,4,5] 3rd=id4 -> 20
///   id3 frame [1,2,4,5] 3rd=id4 -> 20
///   id4 frame [1,2,3,5] 3rd=id3 -> 10
///   id5 frame [1,2,3,4] 3rd=id3 -> 10
#[test]
fn nth_value_3_with_exclude_current_row() {
    let mut db = win_db();
    assert_rows(
        &mut db,
        "SELECT id, nth_value(v, 3) OVER w FROM win \
         WINDOW w AS (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING EXCLUDE CURRENT ROW) \
         ORDER BY id",
        &[
            vec![int(1), int(20)],
            vec![int(2), int(20)],
            vec![int(3), int(20)],
            vec![int(4), int(10)],
            vec![int(5), int(10)],
        ],
    );
}

// ===========================================================================
// Named WINDOW clause — reuse, chaining (§4), and multiple distinct windows.
// ===========================================================================

/// A single named window reused by MULTIPLE functions in one SELECT.
/// `WINDOW w AS (ORDER BY v)` (default RANGE frame) drives both sum(v) and
/// count(*); the v=10 peers share a frame:
///   id1 sum5 cnt1  id2 sum25 cnt3  id3 sum25 cnt3  id4 sum45 cnt4  id5 sum75 cnt5
#[test]
fn named_window_reused_by_multiple_functions() {
    let mut db = win_db();
    assert_rows(
        &mut db,
        "SELECT id, sum(v) OVER w, count(*) OVER w FROM win WINDOW w AS (ORDER BY v) ORDER BY id",
        &[
            vec![int(1), int(5), int(1)],
            vec![int(2), int(25), int(3)],
            vec![int(3), int(25), int(3)],
            vec![int(4), int(45), int(4)],
            vec![int(5), int(75), int(5)],
        ],
    );
}

/// windowfunctions.html §4 Window Chaining: `w AS (base ORDER BY v)` inherits the
/// PARTITION BY of `base`. Effective window = PARTITION BY grp ORDER BY v (default
/// RANGE frame). grp 'a': id1->5, id2/id3->25 (peers), grp 'b': id4->20, id5->50.
/// If chaining is unsupported this fails loudly = signal.
#[test]
fn spec_4_window_chaining_inherits_base_partition() {
    let mut db = win_db();
    assert_rows(
        &mut db,
        "SELECT id, grp, sum(v) OVER w FROM win \
         WINDOW base AS (PARTITION BY grp), w AS (base ORDER BY v) ORDER BY id",
        &[
            vec![int(1), text("a"), int(5)],
            vec![int(2), text("a"), int(25)],
            vec![int(3), text("a"), int(25)],
            vec![int(4), text("b"), int(20)],
            vec![int(5), text("b"), int(50)],
        ],
    );
}

/// §1/§2: two DIFFERENT inline windows in one SELECT, each with its own ordering.
/// row_number() OVER (ORDER BY id) counts 1..5 by id; rank() OVER (ORDER BY v)
/// ranks by v with the tie at v=10 (ids 2,3 both rank 2, gap to rank 4 at v=20).
#[test]
fn two_distinct_windows_each_use_own_ordering() {
    let mut db = win_db();
    assert_rows(
        &mut db,
        "SELECT id, row_number() OVER (ORDER BY id), rank() OVER (ORDER BY v) FROM win ORDER BY id",
        &[
            vec![int(1), int(1), int(1)],
            vec![int(2), int(2), int(2)],
            vec![int(3), int(3), int(2)],
            vec![int(4), int(4), int(4)],
            vec![int(5), int(5), int(5)],
        ],
    );
}

// ===========================================================================
// Window ORDER BY: DESC and NULL placement.
// lang_select.html (#nullslast): "SQLite considers NULL values to be smaller
// than any other values for sorting purposes. Hence, NULLs naturally appear at
// the beginning of an ASC order-by and at the end of a DESC order-by. This can
// be changed using the ASC NULLS LAST or DESC NULLS FIRST syntax."
// `winn` v = 5,10,NULL,20 (ids 1..4).
// ===========================================================================

/// Default ASC: NULL sorts FIRST. row_number() OVER (ORDER BY v):
/// v-order = NULL(id3), 5(id1), 10(id2), 20(id4) -> rn id3=1,id1=2,id2=3,id4=4.
#[test]
fn window_order_by_asc_nulls_first() {
    let mut db = winn_db();
    assert_rows(
        &mut db,
        "SELECT id, row_number() OVER (ORDER BY v) FROM winn ORDER BY id",
        &[
            vec![int(1), int(2)],
            vec![int(2), int(3)],
            vec![int(3), int(1)],
            vec![int(4), int(4)],
        ],
    );
}

/// Default DESC: NULL sorts LAST. row_number() OVER (ORDER BY v DESC):
/// v-order = 20(id4), 10(id2), 5(id1), NULL(id3) -> rn id4=1,id2=2,id1=3,id3=4.
#[test]
fn window_order_by_desc_nulls_last() {
    let mut db = winn_db();
    assert_rows(
        &mut db,
        "SELECT id, row_number() OVER (ORDER BY v DESC) FROM winn ORDER BY id",
        &[
            vec![int(1), int(3)],
            vec![int(2), int(2)],
            vec![int(3), int(4)],
            vec![int(4), int(1)],
        ],
    );
}

/// Explicit `ASC NULLS LAST` overrides the default placement:
/// v-order = 5(id1), 10(id2), 20(id4), NULL(id3) -> rn id1=1,id2=2,id4=3,id3=4.
#[test]
fn window_order_by_asc_nulls_last_override() {
    let mut db = winn_db();
    assert_rows(
        &mut db,
        "SELECT id, row_number() OVER (ORDER BY v ASC NULLS LAST) FROM winn ORDER BY id",
        &[
            vec![int(1), int(1)],
            vec![int(2), int(2)],
            vec![int(3), int(4)],
            vec![int(4), int(3)],
        ],
    );
}
