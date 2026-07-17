//! Conformance battery: the aggregate `FILTER (WHERE ...)` clause on PLAIN
//! (non-window) aggregate functions — `aggregate(expr) FILTER (WHERE pred)`.
//!
//! Every expectation here is transcribed from the SQLite documentation, NOT
//! from what this engine currently returns. If the engine disagrees, the
//! spec-correct assertion stays and is left as a genuine failing assertion —
//! the same discrepancy real `sqlite3` would reveal — rather than weakened to
//! pass.
//!
//! Binding spec (anchors are the doc's `<a name=...>` targets):
//!   * FILTER clause — `spec/sqlite-doc/lang_aggfunc.html` (#aggfilter):
//!       "If a FILTER clause is provided, then only rows for which the expr is
//!        true are included in the aggregate."
//!     So the predicate is applied per candidate row BEFORE the aggregate sees
//!     it, and only a TRUE result includes the row — a FALSE *or* NULL result
//!     skips it. With GROUP BY the filter applies per group.
//!   * 3-valued truth of the predicate — `spec/sqlite-doc/lang_expr.html`
//!     (#booleanexpr, §14): a NULL predicate value "is still NULL" and is
//!     therefore not true (the doc lists NULL among the values "considered to
//!     be false"); and (operators section) "All operators generally evaluate to
//!     NULL when any operand is NULL", so e.g. `NULL = 20` is NULL and the row
//!     is excluded, exactly as a FALSE predicate would be.
//!   * Empty-input aggregate results — `lang_aggfunc.html` (#count, #sumunc,
//!     #avg, #min_agg, #max_agg, #group_concat): when the FILTER admits no rows
//!     the aggregate follows its normal empty-input rule — count(...) -> 0,
//!     sum(...) -> NULL, total(...) -> 0.0, avg/min/max/group_concat(...) -> NULL.
//!
//! This file deliberately fills the gap left by the siblings: DISTINCT + the
//! NULL rules of the aggregates themselves live in `conformance_aggregates.rs`,
//! and the *window*-aggregate FILTER (`... OVER (...)`) lives in
//! `conformance_window_frames.rs`. Here the aggregates are plain (no OVER).
//!
//! Real `sqlite3` has supported aggregate FILTER since 3.30 (2019); if this
//! engine has not implemented it, these cases fail loudly with the engine's own
//! error, which is the correct outcome to surface.

mod conformance;

use conformance::*;

// `Connection` is only needed for the private fixture-helper signature below;
// the harness imports it privately, so it is not in scope via `conformance::*`.
use minisqlite::Connection;

/// The one fixture every case reasons over. Keeping it in a single helper means
/// all the transcribed expected values below share exactly one table shape — a
/// divergent copy would silently break their mutual arithmetic.
///
/// Rows (id, g, v):
///   (1,'a',10) (2,'a',20) (3,'a',NULL) (4,'b',5) (5,'b',30)
/// Group 'a' = {10, 20, NULL}; group 'b' = {5, 30}.
fn table_f(db: &mut Connection) {
    exec(db, "CREATE TABLE f(id INTEGER, g TEXT, v INTEGER)");
    exec(
        db,
        "INSERT INTO f VALUES (1,'a',10),(2,'a',20),(3,'a',NULL),(4,'b',5),(5,'b',30)",
    );
}

// ---- FILTER includes only rows where the predicate is TRUE -------------------

#[test]
fn count_star_filter_only_includes_true_rows() {
    // #aggfilter: only TRUE-predicate rows feed the aggregate. v>10 is TRUE for
    // 20 and 30; 10>10 is FALSE, 5>10 is FALSE, and NULL>10 is NULL (a NULL
    // operand yields NULL, lang_expr operators section) — both skipped. => 2.
    let mut db = mem();
    table_f(&mut db);
    assert_scalar(&mut db, "SELECT count(*) FILTER (WHERE v > 10) FROM f", int(2));
}

#[test]
fn count_column_filter_ignores_null_arg() {
    // #aggfilter selects group 'a' = {id1:10, id2:20, id3:NULL}; then #count:
    // "count(X) ... returns a count of the number of times that X is not NULL",
    // so id3's NULL v is not counted. => 2 (distinct from count(*), which is 3).
    let mut db = mem();
    table_f(&mut db);
    assert_scalar(&mut db, "SELECT count(v) FILTER (WHERE g='a') FROM f", int(2));
}

#[test]
fn sum_filter_ge_threshold() {
    // #aggfilter: v>=20 admits 20 and 30 only. #sumunc: all-integer inputs =>
    // INTEGER result. 20 + 30 = 50.
    let mut db = mem();
    table_f(&mut db);
    assert_scalar(&mut db, "SELECT sum(v) FILTER (WHERE v >= 20) FROM f", int(50));
}

// ---- FILTER that admits no rows => each aggregate's empty-input rule ----------

#[test]
fn sum_filter_empty_input_is_null() {
    // #aggfilter admits nothing (no v>100). #sumunc: "If there are no non-NULL
    // input rows then sum() returns NULL".
    let mut db = mem();
    table_f(&mut db);
    assert_scalar(&mut db, "SELECT sum(v) FILTER (WHERE v > 100) FROM f", null());
}

#[test]
fn total_filter_empty_input_is_zero_real() {
    // Same empty FILTER input, but #sumunc: "total() returns 0.0" (a REAL zero),
    // and total() "is always a floating point value".
    let mut db = mem();
    table_f(&mut db);
    assert_scalar(
        &mut db,
        "SELECT total(v) FILTER (WHERE v > 100) FROM f",
        real(0.0),
    );
}

#[test]
fn count_star_filter_empty_input_is_zero() {
    // Same empty FILTER input; #count: count over no rows is 0 (an INTEGER),
    // never NULL.
    let mut db = mem();
    table_f(&mut db);
    assert_scalar(&mut db, "SELECT count(*) FILTER (WHERE v > 100) FROM f", int(0));
}

#[test]
fn avg_min_max_group_concat_empty_filter_is_null() {
    // Completes the empty-FILTER-input picture begun above (count->0, sum->NULL,
    // total->0.0): when the FILTER admits no rows, the remaining aggregates each
    // fall to their own empty-input rule and return NULL — #avg ("NULL if there
    // are no non-NULL inputs"), #min_agg / #max_agg ("NULL iff there are no
    // non-NULL values"), and #group_concat (no non-NULL X to concatenate). The
    // predicate v>100 admits nothing.
    let mut db = mem();
    table_f(&mut db);
    assert_scalar(&mut db, "SELECT avg(v) FILTER (WHERE v > 100) FROM f", null());
    assert_scalar(&mut db, "SELECT min(v) FILTER (WHERE v > 100) FROM f", null());
    assert_scalar(&mut db, "SELECT max(v) FILTER (WHERE v > 100) FROM f", null());
    assert_scalar(
        &mut db,
        "SELECT group_concat(v) FILTER (WHERE v > 100) FROM f",
        null(),
    );
}

// ---- avg / min / max under FILTER --------------------------------------------

#[test]
fn avg_filter_per_predicate() {
    // #aggfilter selects group 'b' = {5, 30}; #avg is total()/count() over the
    // non-NULL inputs = (5 + 30) / 2 = 17.5, always REAL. 17.5 is exact in f64.
    let mut db = mem();
    table_f(&mut db);
    assert_scalar(&mut db, "SELECT avg(v) FILTER (WHERE g='b') FROM f", real(17.5));
}

#[test]
fn max_filter_ignores_null() {
    // #aggfilter selects group 'a' = {10, 20, NULL}; #max_agg ignores NULL and
    // returns the largest value in its own class => integer 20.
    let mut db = mem();
    table_f(&mut db);
    assert_scalar(&mut db, "SELECT max(v) FILTER (WHERE g='a') FROM f", int(20));
}

#[test]
fn min_filter_ignores_null() {
    // #aggfilter selects group 'a' = {10, 20, NULL}; #min_agg ignores NULL and
    // returns the smallest value => integer 10.
    let mut db = mem();
    table_f(&mut db);
    assert_scalar(&mut db, "SELECT min(v) FILTER (WHERE g='a') FROM f", int(10));
}

// ---- FILTER applies PER GROUP under GROUP BY ---------------------------------

#[test]
fn grouped_count_star_filter_not_null() {
    // #aggfilter "with GROUP BY ... applies per group". `v IS NOT NULL` never
    // yields NULL (lang_expr #isisnot: IS/IS NOT can only be 0 or 1), so it is a
    // clean TRUE/FALSE gate. Group 'a' drops id3's NULL -> 2; group 'b' keeps
    // both -> 2. ORDER BY g pins the row order.
    let mut db = mem();
    table_f(&mut db);
    assert_rows(
        &mut db,
        "SELECT g, count(*) FILTER (WHERE v IS NOT NULL) FROM f GROUP BY g ORDER BY g",
        &[vec![text("a"), int(2)], vec![text("b"), int(2)]],
    );
}

#[test]
fn grouped_sum_filter_lt() {
    // Per-group #aggfilter with v<25. Group 'a': 10 and 20 pass, id3's NULL is
    // excluded (NULL<25 is NULL) => 30. Group 'b': 5 passes, 30 does not => 5.
    let mut db = mem();
    table_f(&mut db);
    assert_rows(
        &mut db,
        "SELECT g, sum(v) FILTER (WHERE v < 25) FROM f GROUP BY g ORDER BY g",
        &[vec![text("a"), int(30)], vec![text("b"), int(5)]],
    );
}

#[test]
fn grouped_count_star_filter_empty_per_group() {
    // A per-group FILTER that admits no row still yields a group (the group
    // exists because it HAS rows); #count over the empty filtered input is 0.
    let mut db = mem();
    table_f(&mut db);
    assert_rows(
        &mut db,
        "SELECT g, count(*) FILTER (WHERE v > 100) FROM f GROUP BY g ORDER BY g",
        &[vec![text("a"), int(0)], vec![text("b"), int(0)]],
    );
}

// ---- Several independent FILTERs, and FILTER beside a plain aggregate ---------

#[test]
fn multiple_independent_filters_in_one_select() {
    // #aggfilter: each aggregate carries its OWN filter, evaluated independently.
    // count(*) FILTER(v>10) = 2 (20, 30); count(*) FILTER(g='a') = 3 (all of
    // group 'a', including id3 — count(*) counts rows, not the NULL v).
    let mut db = mem();
    table_f(&mut db);
    assert_rows(
        &mut db,
        "SELECT count(*) FILTER (WHERE v>10), count(*) FILTER (WHERE g='a') FROM f",
        &[vec![int(2), int(3)]],
    );
}

#[test]
fn filtered_beside_unfiltered_aggregate() {
    // #aggfilter: a FILTER on one aggregate must not affect an unfiltered one in
    // the same SELECT. count(*) = 5 (every row); count(*) FILTER(v>=20) = 2 (20, 30).
    let mut db = mem();
    table_f(&mut db);
    assert_rows(
        &mut db,
        "SELECT count(*), count(*) FILTER (WHERE v>=20) FROM f",
        &[vec![int(5), int(2)]],
    );
}

// ---- FILTER combined with DISTINCT -------------------------------------------

#[test]
fn count_distinct_with_filter() {
    // FILTER admits rows first (#aggfilter), then DISTINCT collapses duplicates
    // in the aggregate's input (lang_aggfunc §1). This uses a LOCAL table (a
    // second table is used for DISTINCT) built so the FILTER is
    // actually DISCRIMINATING: g='c' occurs ONLY on the NULL-v row, so
    // `v IS NOT NULL` drops it. The filtered count(DISTINCT g) is {a,b} = 2,
    // while the SAME aggregate WITHOUT the filter is {a,b,c} = 3 — the pair pins
    // that the FILTER truly runs (an ignored FILTER would return 3, not 2).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE d(g TEXT, v INTEGER)");
    exec(
        &mut db,
        "INSERT INTO d VALUES ('a',10),('a',20),('b',5),('b',30),('c',NULL)",
    );
    assert_scalar(
        &mut db,
        "SELECT count(DISTINCT g) FILTER (WHERE v IS NOT NULL) FROM d",
        int(2),
    );
    assert_scalar(&mut db, "SELECT count(DISTINCT g) FROM d", int(3));
}

// ---- Predicate over a different column, and the 3-valued NULL rule -----------

#[test]
fn filter_predicate_references_other_column() {
    // The FILTER predicate may reference a column other than the aggregate's
    // argument (#aggfilter — it is an ordinary boolean expr over the row).
    // `v IS NULL` is TRUE only for id3, so sum(id) over the admitted rows = 3.
    let mut db = mem();
    table_f(&mut db);
    assert_scalar(&mut db, "SELECT sum(id) FILTER (WHERE v IS NULL) FROM f", int(3));
}

#[test]
fn filter_predicate_null_excludes_row() {
    // 3-valued truth (lang_expr #booleanexpr + operators): `v = 20` is TRUE for
    // id2, FALSE for the other non-NULL rows, and NULL for id3 (`NULL = 20` is
    // NULL). Only the TRUE row is admitted, so a NULL predicate behaves like
    // FALSE here. => 1.
    let mut db = mem();
    table_f(&mut db);
    assert_scalar(&mut db, "SELECT count(*) FILTER (WHERE v = 20) FROM f", int(1));
}

// ---- group_concat under FILTER (single admitted row keeps order pinned) ------

#[test]
fn group_concat_filter_single_row() {
    // #group_concat concatenates the non-NULL X of the admitted rows. Filtering
    // to the single row v=30 makes the (otherwise arbitrary, #aggorderby) input
    // order irrelevant; the integer 30 renders as the text "30".
    let mut db = mem();
    table_f(&mut db);
    assert_scalar(
        &mut db,
        "SELECT group_concat(v) FILTER (WHERE v = 30) FROM f",
        text("30"),
    );
}

// ---- Empty table under FILTER ------------------------------------------------

#[test]
fn empty_table_with_filter() {
    // With no rows at all, the FILTER predicate is never evaluated and every
    // aggregate hits its empty-input rule regardless of the predicate:
    // #count -> 0, #sumunc -> NULL.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE et(x)");
    assert_scalar(&mut db, "SELECT count(*) FILTER (WHERE 1) FROM et", int(0));
    assert_scalar(&mut db, "SELECT sum(x) FILTER (WHERE 1) FROM et", null());
}

// ---- FILTER on a non-aggregate is an error -----------------------------------

#[test]
fn filter_on_non_aggregate_is_error() {
    // The filter-clause grammar (lang_aggfunc.html "filter-clause") attaches only
    // to an aggregate (or window) function invocation. `abs()` is an ordinary
    // scalar function, so a FILTER on it is a semantic error. We assert only that
    // it errors (the exact message is not part of the binding spec); real sqlite3
    // rejects it ("FILTER clause may only be used with aggregate window functions").
    let mut db = mem();
    table_f(&mut db);
    assert_query_error(&mut db, "SELECT abs(v) FILTER (WHERE v>0) FROM f");
}

#[test]
fn filter_on_scalar_functions_is_error() {
    // Generalizes the case above from one scalar to the CLASS invariant: the
    // filter-clause grammar attaches only to an aggregate/window function, so a
    // FILTER on ANY ordinary scalar must error. The sharp, bug-prone edge is the
    // overloaded names — per lang_corefunc.html, min(X,Y,...) / max(X,Y,...) are
    // SIMPLE (scalar) functions with two or more arguments, while min(X) / max(X)
    // are the aggregates. So `max(v,0)` / `min(v,0)` are scalar calls whose FILTER
    // must be rejected DESPITE the aggregate-looking name; an engine that decided
    // "is this an aggregate?" by name alone would wrongly accept them. Each
    // spec-correct outcome is that query() errors; we report every scalar that
    // was wrongly accepted so a failure names the exact hole.
    let scalars = [
        "abs(v)",
        "length(g)",
        "upper(g)",
        "typeof(v)",
        "round(v)",
        "max(v, 0)",
        "min(v, 0)",
    ];
    let mut wrongly_accepted = Vec::new();
    for call in scalars {
        let mut db = mem();
        table_f(&mut db);
        let sql = format!("SELECT {call} FILTER (WHERE v > 0) FROM f");
        if try_query(&mut db, &sql).is_ok() {
            wrongly_accepted.push(call);
        }
    }
    assert!(
        wrongly_accepted.is_empty(),
        "FILTER may only follow an aggregate/window function, but these scalar \
         calls wrongly accepted a FILTER clause: {wrongly_accepted:?}"
    );
}

// ---- An always-true / always-false predicate bounds the behavior -------------

#[test]
fn filter_always_true_matches_unfiltered() {
    // A predicate that is always TRUE admits every row, so `FILTER (WHERE 1)` is
    // equivalent to no filter. sum(v) ignores id3's NULL either way:
    // 10 + 20 + 5 + 30 = 65 (#aggfilter + #sumunc).
    let mut db = mem();
    table_f(&mut db);
    assert_scalar(&mut db, "SELECT sum(v) FILTER (WHERE 1) FROM f", int(65));
    assert_scalar(&mut db, "SELECT sum(v) FROM f", int(65));
}

#[test]
fn filter_always_false_excludes_all_rows() {
    // The mirror of the always-true case: a constant-FALSE predicate (0 is
    // "considered to be false", lang_expr #booleanexpr) admits no rows, so each
    // aggregate falls to its empty-input rule: #count -> 0, #sumunc -> NULL.
    let mut db = mem();
    table_f(&mut db);
    assert_scalar(&mut db, "SELECT count(*) FILTER (WHERE 0) FROM f", int(0));
    assert_scalar(&mut db, "SELECT sum(v) FILTER (WHERE 0) FROM f", null());
}
