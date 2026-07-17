//! Conformance battery — ORDERED-SET aggregate WINDOW functions
//! (`f(x ORDER BY y) OVER (…)`), e.g. `group_concat(b, '.' ORDER BY b DESC) OVER (…)`.
//!
//! Semantics (spec, transcribed — never from engine output):
//!
//!   * `lang_aggfunc.html` (#aggorderby): "If an ORDER BY clause is provided, that clause
//!     determines the order in which the inputs to the aggregate are processed. For
//!     aggregate functions like max() and count(), the input order does not matter. But
//!     for things like group_concat()/json_group_object(), the ORDER BY clause will make a
//!     difference in the result."
//!   * `windowfunctions.html` §2: an aggregate window function's result at a row "is as if
//!     the corresponding aggregate were run over all rows in the window frame". So the
//!     "inputs to the aggregate" at each row ARE that row's window frame — the in-argument
//!     ORDER BY sorts the FRAME's rows before the fold. Order-insensitive aggregates
//!     (`sum`/`count`/`min`/`max`) are unaffected; `group_concat`/`json_group_array` are.
//!
//! All expected values below are computed BY HAND from those two rules over the doc's own
//! `t1` table (§2). The distinguishing choice throughout is an in-argument `ORDER BY b DESC`
//! against a window `ORDER BY a ASC`: if the engine ignored the in-arg ORDER BY and folded
//! in window order, every expectation here would differ (e.g. "B.A" would read "A.B").
//!
//! An assertion here stays spec-correct even if the engine disagrees — it is left to
//! FAIL and is never weakened to match the engine.

mod conformance;

use conformance::*;
use minisqlite::Connection;

/// The doc's §2 table `t1(a,b,c)`, seven rows a=1..7, b='A'..'G', c cycling one/two/three.
fn t1_db() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t1(a INTEGER PRIMARY KEY, b, c)");
    exec(
        &mut db,
        "INSERT INTO t1 VALUES \
         (1,'A','one'),(2,'B','two'),(3,'C','three'),\
         (4,'D','one'),(5,'E','two'),(6,'F','three'),(7,'G','one')",
    );
    db
}

// ---------------------------------------------------------------------------
// Default frame (RANGE UNBOUNDED PRECEDING .. CURRENT ROW) with in-arg ORDER BY.
// a is a unique PK, so each row is its own peer group and the frame for row a=k is
// exactly rows a=1..k; the in-arg ORDER BY b DESC reverses the fold order.
// ---------------------------------------------------------------------------

#[test]
fn default_frame_in_arg_order_desc_reverses_fold() {
    let mut db = t1_db();
    // Row a=k frame = {A..k-th}; folded b DESC ⇒ reversed concatenation.
    assert_rows(
        &mut db,
        "SELECT a, group_concat(b, '.' ORDER BY b DESC) OVER (ORDER BY a) AS gc \
         FROM t1 ORDER BY a",
        &[
            vec![int(1), text("A")],
            vec![int(2), text("B.A")],
            vec![int(3), text("C.B.A")],
            vec![int(4), text("D.C.B.A")],
            vec![int(5), text("E.D.C.B.A")],
            vec![int(6), text("F.E.D.C.B.A")],
            vec![int(7), text("G.F.E.D.C.B.A")],
        ],
    );
}

#[test]
fn default_frame_in_arg_order_asc_matches_window_order_here() {
    // With b ASC (same direction as window ORDER BY a), the fold order coincides with
    // window order: this is the plain running concatenation "A", "A.B", "A.B.C", …. It is
    // the control case for the DESC test above (same query, opposite in-arg direction).
    let mut db = t1_db();
    assert_rows(
        &mut db,
        "SELECT a, group_concat(b, '.' ORDER BY b ASC) OVER (ORDER BY a) AS gc \
         FROM t1 ORDER BY a",
        &[
            vec![int(1), text("A")],
            vec![int(2), text("A.B")],
            vec![int(3), text("A.B.C")],
            vec![int(4), text("A.B.C.D")],
            vec![int(5), text("A.B.C.D.E")],
            vec![int(6), text("A.B.C.D.E.F")],
            vec![int(7), text("A.B.C.D.E.F.G")],
        ],
    );
}

#[test]
fn default_separator_with_in_arg_order() {
    // One-argument group_concat defaults its separator to ",". The ORDER BY still applies.
    let mut db = t1_db();
    assert_rows(
        &mut db,
        "SELECT a, group_concat(b ORDER BY b DESC) OVER (ORDER BY a) AS gc \
         FROM t1 ORDER BY a",
        &[
            vec![int(1), text("A")],
            vec![int(2), text("B,A")],
            vec![int(3), text("C,B,A")],
            vec![int(4), text("D,C,B,A")],
            vec![int(5), text("E,D,C,B,A")],
            vec![int(6), text("F,E,D,C,B,A")],
            vec![int(7), text("G,F,E,D,C,B,A")],
        ],
    );
}

// ---------------------------------------------------------------------------
// Explicit frame × in-arg ORDER BY: the ORDER BY reorders WITHIN each frame.
// The doc's §2 example (ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING, no in-arg ORDER BY)
// gives 'A.B','A.B.C','B.C.D',… ; adding ORDER BY b DESC reverses each frame.
// ---------------------------------------------------------------------------

#[test]
fn rows_sliding_frame_reordered_by_in_arg_order() {
    let mut db = t1_db();
    assert_rows(
        &mut db,
        "SELECT a, group_concat(b, '.' ORDER BY b DESC) OVER \
           (ORDER BY a ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS gc \
         FROM t1 ORDER BY a",
        &[
            vec![int(1), text("B.A")],   // frame {A,B}
            vec![int(2), text("C.B.A")], // frame {A,B,C}
            vec![int(3), text("D.C.B")], // frame {B,C,D}
            vec![int(4), text("E.D.C")], // frame {C,D,E}
            vec![int(5), text("F.E.D")], // frame {D,E,F}
            vec![int(6), text("G.F.E")], // frame {E,F,G}
            vec![int(7), text("G.F")],   // frame {F,G}
        ],
    );
}

// ---------------------------------------------------------------------------
// PARTITION BY × in-arg ORDER BY.
// c splits into one={A,D,G}, two={B,E}, three={C,F} (in a order).
// ---------------------------------------------------------------------------

#[test]
fn partitioned_default_frame_with_in_arg_order() {
    // Default frame, ORDER BY a within each partition; frame for a=k = partition rows a≤k,
    // folded b DESC.
    let mut db = t1_db();
    assert_rows(
        &mut db,
        "SELECT a, group_concat(b, '.' ORDER BY b DESC) OVER \
           (PARTITION BY c ORDER BY a) AS gc \
         FROM t1 ORDER BY a",
        &[
            vec![int(1), text("A")],     // one: {A}
            vec![int(2), text("B")],     // two: {B}
            vec![int(3), text("C")],     // three: {C}
            vec![int(4), text("D.A")],   // one: {A,D} desc
            vec![int(5), text("E.B")],   // two: {B,E} desc
            vec![int(6), text("F.C")],   // three: {C,F} desc
            vec![int(7), text("G.D.A")], // one: {A,D,G} desc
        ],
    );
}

#[test]
fn partitioned_current_to_unbounded_following_with_in_arg_order() {
    // The doc's §2.1 frame (RANGE CURRENT ROW .. UNBOUNDED FOLLOWING) but folded b DESC.
    // Doc (no in-arg order): one a=1 'A.D.G'; here reversed to 'G.D.A', etc.
    let mut db = t1_db();
    assert_rows(
        &mut db,
        "SELECT a, group_concat(b, '.' ORDER BY b DESC) OVER \
           (PARTITION BY c ORDER BY a RANGE BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS gc \
         FROM t1 ORDER BY a",
        &[
            vec![int(1), text("G.D.A")], // one {A,D,G}
            vec![int(2), text("E.B")],   // two {B,E}
            vec![int(3), text("F.C")],   // three {C,F}
            vec![int(4), text("G.D")],   // one {D,G}
            vec![int(5), text("E")],     // two {E}
            vec![int(6), text("F")],     // three {F}
            vec![int(7), text("G")],     // one {G}
        ],
    );
}

// ---------------------------------------------------------------------------
// Order-INSENSITIVE aggregates are unaffected by the in-arg ORDER BY (spec: "for
// aggregate functions like max() and count(), the input order does not matter").
// ---------------------------------------------------------------------------

#[test]
fn order_insensitive_aggregate_unchanged_by_in_arg_order() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE s(id INTEGER, v INTEGER)");
    exec(&mut db, "INSERT INTO s VALUES (1,10),(2,30),(3,20)");
    // sum over the whole partition = 60 for every row, with or without the ORDER BY.
    assert_rows(
        &mut db,
        "SELECT id, sum(v ORDER BY v DESC) OVER () AS s FROM s ORDER BY id",
        &[vec![int(1), int(60)], vec![int(2), int(60)], vec![int(3), int(60)]],
    );
    // count with an argument + ORDER BY is likewise order-independent.
    assert_rows(
        &mut db,
        "SELECT id, count(v ORDER BY v) OVER () AS c FROM s ORDER BY id",
        &[vec![int(1), int(3)], vec![int(2), int(3)], vec![int(3), int(3)]],
    );
}

// ---------------------------------------------------------------------------
// Multi-key in-arg ORDER BY, and an ordering key that is NOT the aggregated argument.
// ---------------------------------------------------------------------------

#[test]
fn multi_key_in_arg_order_over_whole_partition() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE m(id INTEGER, x TEXT, y INTEGER)");
    exec(&mut db, "INSERT INTO m VALUES (1,'b',2),(2,'a',2),(3,'a',1)");
    // ORDER BY y ASC, x ASC over the whole partition:
    //   (y=1,x=a) id3, (y=2,x=a) id2, (y=2,x=b) id1  ⇒ x sequence "a.a.b".
    // Note y is NOT the aggregated argument — an ordering key may be any input column.
    assert_rows(
        &mut db,
        "SELECT id, group_concat(x, '.' ORDER BY y, x) OVER () AS gc FROM m ORDER BY id",
        &[
            vec![int(1), text("a.a.b")],
            vec![int(2), text("a.a.b")],
            vec![int(3), text("a.a.b")],
        ],
    );
}

// ---------------------------------------------------------------------------
// NULL handling: group_concat skips NULLs (value AND separator); a NULL is still placed
// by the ORDER BY (default NULLS LAST for DESC) but contributes nothing to the result.
// ---------------------------------------------------------------------------

#[test]
fn nulls_skipped_by_group_concat_under_in_arg_order() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE n(id INTEGER, v TEXT)");
    exec(&mut db, "INSERT INTO n VALUES (1,'x'),(2,NULL),(3,'y')");
    // Non-null v folded DESC ⇒ "y.x"; the NULL row is ordered but omitted.
    assert_rows(
        &mut db,
        "SELECT id, group_concat(v, '.' ORDER BY v DESC) OVER () AS gc FROM n ORDER BY id",
        &[
            vec![int(1), text("y.x")],
            vec![int(2), text("y.x")],
            vec![int(3), text("y.x")],
        ],
    );
}

// ---------------------------------------------------------------------------
// json_group_array is order-sensitive: the ORDER BY sets the array element order.
// ---------------------------------------------------------------------------

#[test]
fn json_group_array_honors_in_arg_order() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE j(id INTEGER, v INTEGER)");
    exec(&mut db, "INSERT INTO j VALUES (1,3),(2,1),(3,2)");
    // Whole-partition frame, elements ordered by v ASC ⇒ [1,2,3] for every row.
    assert_rows(
        &mut db,
        "SELECT id, json_group_array(v ORDER BY v) OVER () AS a FROM j ORDER BY id",
        &[
            vec![int(1), text("[1,2,3]")],
            vec![int(2), text("[1,2,3]")],
            vec![int(3), text("[1,2,3]")],
        ],
    );
    // Reversed direction ⇒ [3,2,1].
    assert_rows(
        &mut db,
        "SELECT id, json_group_array(v ORDER BY v DESC) OVER () AS a FROM j ORDER BY id",
        &[
            vec![int(1), text("[3,2,1]")],
            vec![int(2), text("[3,2,1]")],
            vec![int(3), text("[3,2,1]")],
        ],
    );
}

// ---------------------------------------------------------------------------
// A non-aggregate / built-in window function has no fold order, so an in-argument
// ORDER BY is a loud error (SQLite: "ORDER BY may not be used with the non-aggregate
// function …"), never silently dropped.
// ---------------------------------------------------------------------------

#[test]
fn in_arg_order_on_builtin_window_function_is_a_loud_error() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER, v INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1,10),(2,20)");
    // lead is a built-in window function (arity 1..3); an in-arg ORDER BY is invalid.
    let e = assert_query_error(
        &mut db,
        "SELECT lead(v, 1 ORDER BY v) OVER (ORDER BY id) FROM t",
    );
    let msg = format!("{e:?}");
    assert!(msg.contains("non-aggregate"), "expected a non-aggregate error, got: {msg}");
}
