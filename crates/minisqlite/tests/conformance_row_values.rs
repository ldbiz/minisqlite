//! Conformance battery: SQLite ROW VALUE expressions — a parenthesized list of
//! two or more scalars, `(a, b, ...)`, used in comparisons and IN.
//!
//! SPEC (authoritative): the dedicated "Row Values" page
//! `spec/sqlite-doc/rowvalue.html` (which `lang_expr.html` links to for every
//! mention of a "row value"), plus `lang_expr.html` §8 (The IN and NOT IN
//! operators). Supporting comparison rules: `datatype3.html` §4 (comparison of
//! values under the default BINARY collation, and NULL handling) and
//! `lang_expr.html` §2 (three-valued AND / OR).
//!
//! BINDING RULE — every expected value here is TRANSCRIBED FROM THE SPEC by
//! expanding a row-value operation to its equivalent scalar form and applying
//! ordinary comparison plus three-valued NULL logic. It is NEVER taken from what
//! the engine returns:
//!   (a1,a2)  =  (b1,b2)  ≡  a1=b1  AND a2=b2
//!   (a1,a2)  <> (b1,b2)  ≡  a1<>b1 OR  a2<>b2
//!   (a1,a2)  <  (b1,b2)  ≡  a1<b1  OR  (a1=b1 AND a2<b2)     (<=, >, >= analogously)
//!   (a1,..,an) IN (rows) ≡  the row = ANY listed row         (each `=` as above)
//! `rowvalue.html` §2.1 states the NULL rule directly: a NULL means "unknown", and
//! a row-value comparison is NULL exactly when some substitution of the
//! constituent NULL(s) could make the result either true or false; otherwise it is
//! the definite 0 or 1.
//!
//! Expected values are transcribed from the SQLite documentation, not from what this
//! engine returns; a case that reveals an engine bug is left as a genuine failing
//! assertion rather than weakened to pass, and is never `#[ignore]`d. Each case is its
//! own one-assertion `#[test]` so a single failure never masks the rest. Row-value
//! comparison may be only partially implemented in this engine; this file's job is to
//! reveal exactly how much works.

mod conformance;

use conformance::*;
use minisqlite::Connection;

/// A fresh in-memory database holding `t(x, y)` with rows `(1, 2)` and `(3, 4)`.
/// Used by the IN-subquery and WHERE cases (`rowvalue.html` §2.2 and §3.3). Two
/// rows are the minimum that exercises both a matching and a non-matching row.
/// Rows are inserted with two single-row `INSERT`s so a possible multi-row
/// `VALUES` gap cannot masquerade as a row-value failure.
fn table_t() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x, y)");
    exec(&mut db, "INSERT INTO t(x, y) VALUES (1, 2)");
    exec(&mut db, "INSERT INTO t(x, y) VALUES (3, 4)");
    db
}

// ---- Equality / inequality, no NULLs (rowvalue.html §2.1) --------------------

#[test]
fn row_eq_identical_is_true() {
    // (1,2)=(1,2) ≡ (1=1 AND 2=2) ≡ 1 AND 1 ≡ 1
    eval_eq("(1,2) = (1,2)", int(1));
}

#[test]
fn row_eq_differing_element_is_false() {
    // (1,2)=(1,3) ≡ (1=1 AND 2=3) ≡ 1 AND 0 ≡ 0
    eval_eq("(1,2) = (1,3)", int(0));
}

#[test]
fn row_ne_differing_element_is_true() {
    // (1,2)<>(1,3) ≡ (1<>1 OR 2<>3) ≡ 0 OR 1 ≡ 1
    eval_eq("(1,2) <> (1,3)", int(1));
}

#[test]
fn row_ne_identical_is_false() {
    // (1,2)<>(1,2) ≡ (1<>1 OR 2<>2) ≡ 0 OR 0 ≡ 0
    eval_eq("(1,2) <> (1,2)", int(0));
}

// ---- Ordering: lexicographic left-to-right, no NULLs (rowvalue.html §2.1) ----

#[test]
fn row_lt_second_element_decides_true() {
    // (1,2)<(1,3) ≡ 1<1 OR (1=1 AND 2<3) ≡ 0 OR (1 AND 1) ≡ 1
    eval_eq("(1,2) < (1,3)", int(1));
}

#[test]
fn row_lt_equal_rows_is_false() {
    // (1,2)<(1,2) ≡ 1<1 OR (1=1 AND 2<2) ≡ 0 OR (1 AND 0) ≡ 0
    eval_eq("(1,2) < (1,2)", int(0));
}

#[test]
fn row_le_equal_rows_is_true() {
    // (1,2)<=(1,2) ≡ 1<1 OR (1=1 AND 2<=2) ≡ 0 OR (1 AND 1) ≡ 1
    eval_eq("(1,2) <= (1,2)", int(1));
}

#[test]
fn row_gt_first_element_decides_true() {
    // (2,0)>(1,9) ≡ 2>1 OR (2=1 AND 0>9) ≡ 1 OR ... ≡ 1 (first element decides)
    eval_eq("(2,0) > (1,9)", int(1));
}

#[test]
fn row_ge_equal_rows_is_true() {
    // (1,2)>=(1,2) ≡ 1>1 OR (1=1 AND 2>=2) ≡ 0 OR (1 AND 1) ≡ 1
    eval_eq("(1,2) >= (1,2)", int(1));
}

#[test]
fn row_lt_first_element_decides_true() {
    // (1,5)<(2,0) ≡ 1<2 OR (1=2 AND 5<0) ≡ 1 OR ... ≡ 1 (first element decides,
    // so the larger second element 5 vs 0 is irrelevant)
    eval_eq("(1,5) < (2,0)", int(1));
}

// ---- NULL semantics: 3-valued logic + the rowvalue.html §2.1 substitution rule
// A row-value comparison is NULL iff some substitution of the constituent NULL(s)
// could make the result either true or false; otherwise it is the definite 0/1.
// Each case shows both the boolean expansion and the substitution reasoning.

#[test]
fn row_eq_null_element_is_unknown() {
    // (1,NULL)=(1,2) ≡ (1=1 AND NULL=2) ≡ TRUE AND NULL ≡ NULL.
    // Substitution: NULL→2 makes it true, NULL→9 makes it false ⇒ NULL.
    eval_eq("(1,NULL) = (1,2)", null());
}

#[test]
fn row_eq_null_with_false_prefix_is_false() {
    // (1,NULL)=(2,3) ≡ (1=2 AND NULL=3) ≡ FALSE AND NULL ≡ FALSE ≡ 0.
    // 1=2 is definitely false, so no substitution of NULL can make it true ⇒ 0.
    eval_eq("(1,NULL) = (2,3)", int(0));
}

#[test]
fn row_lt_null_with_true_prefix_is_true() {
    // (1,NULL)<(2,3) ≡ 1<2 OR (1=2 AND NULL<3) ≡ TRUE OR ... ≡ TRUE ≡ 1.
    // 1<2 is definitely true regardless of the NULL ⇒ 1.
    eval_eq("(1,NULL) < (2,3)", int(1));
}

#[test]
fn row_lt_null_element_is_unknown() {
    // (1,NULL)<(1,3) ≡ 1<1 OR (1=1 AND NULL<3) ≡ FALSE OR (TRUE AND NULL) ≡ NULL.
    // Substitution: NULL→2 ⇒ true, NULL→9 ⇒ false ⇒ NULL.
    eval_eq("(1,NULL) < (1,3)", null());
}

#[test]
fn row_ne_null_with_true_prefix_is_true() {
    // (1,NULL)<>(2,3) ≡ 1<>2 OR NULL<>3 ≡ TRUE OR NULL ≡ TRUE ≡ 1.
    // 1<>2 is definitely true regardless of the NULL ⇒ 1.
    eval_eq("(1,NULL) <> (2,3)", int(1));
}

// ---- IN with a parenthesized value-list RHS is an error ----------------------
// A row-value IN requires a SUBQUERY right-hand side; a value-list RHS is the
// "row value misused" error family (the same rejection as the arity / scalar /
// arithmetic cases below), NOT a boolean:
//   * rowvalue.html §2.2: "the right-hand side (RHS) must be a subquery expression".
//   * lang_expr.html §8: the list form of IN requires the left expression AND each
//     list element to be scalars; a 2-tuple is not a scalar.
// (Standard SQL / PostgreSQL evaluate a value-list row IN to a boolean; SQLite
// does not — so the expected result is an error, transcribed from the spec above,
// never from the engine.) Both a "would-match" and a "would-not-match" list are
// checked to pin that the error rejects the FORM and is independent of membership.
// As with the arity/scalar family, `assert_query_error` pins only THAT an error
// occurs, not which: an engine that implements row values must keep REJECTING the
// value-list form here (never evaluate it to a boolean), whereas the subquery form
// below must instead return a value.

#[test]
fn row_in_value_list_would_match_is_error() {
    // (1,2) is present in the list, yet a value-list RHS for a row-value IN is
    // still "row value misused" (rowvalue.html §2.2; lang_expr.html §8).
    let mut db = mem();
    assert_query_error(&mut db, "SELECT (1,2) IN ((1,2),(3,4))");
}

#[test]
fn row_in_value_list_would_not_match_is_error() {
    // (1,2) is absent from the list; the rejection is of the FORM, not membership.
    let mut db = mem();
    assert_query_error(&mut db, "SELECT (1,2) IN ((3,4),(5,6))");
}

#[test]
fn row_not_in_value_list_is_error() {
    // NOT IN over a value-list RHS is the same "row value misused" error.
    let mut db = mem();
    assert_query_error(&mut db, "SELECT (1,2) NOT IN ((3,4),(5,6))");
}

// ---- IN with a subquery RHS (rowvalue.html §2.2 — the SUPPORTED form) --------

#[test]
fn row_in_subquery_match_is_true() {
    // (1,2) equals t's first row ⇒ 1. rowvalue.html §2.2 shows this exact form:
    // (a,b) IN (subquery) ≡ some candidate row equals (a,b) element-wise.
    let mut db = table_t();
    assert_scalar(&mut db, "SELECT (1,2) IN (SELECT x, y FROM t)", int(1));
}

#[test]
fn row_in_subquery_no_match_is_false() {
    // (9,9) equals neither (1,2) nor (3,4), and t has no NULLs ⇒ definite 0.
    let mut db = table_t();
    assert_scalar(&mut db, "SELECT (9,9) IN (SELECT x, y FROM t)", int(0));
}

#[test]
fn row_in_subquery_null_element_is_unknown() {
    // (1,NULL) IN (SELECT x,y FROM t) over t = {(1,2),(3,4)}. Per rowvalue.html §2.2 and
    // the §2.1 NULL rule, expand to OR over candidate rows of the element-wise AND:
    //   (1,NULL)=(1,2) ≡ 1=1 AND NULL=2 ≡ true AND unknown ≡ unknown
    //   (1,NULL)=(3,4) ≡ 1=3 AND NULL=4 ≡ false AND unknown ≡ false
    //   IN ≡ unknown OR false ≡ unknown ⇒ NULL (no candidate is a definite match, and one
    //   could become one were NULL substituted right).
    let mut db = table_t();
    assert_scalar(&mut db, "SELECT (1,NULL) IN (SELECT x, y FROM t)", null());
}

#[test]
fn row_in_empty_subquery_is_false() {
    // `x IN (empty)` is FALSE even for a tuple with NULLs (lang_expr.html §8): an empty
    // candidate set has no row to match, so the OR over zero candidates is definite 0.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE e(x, y)");
    assert_scalar(&mut db, "SELECT (1,2) IN (SELECT x, y FROM e)", int(0));
    assert_scalar(&mut db, "SELECT (1,NULL) IN (SELECT x, y FROM e)", int(0));
}

#[test]
fn row_not_in_empty_subquery_is_true() {
    // `NOT IN (empty)` is TRUE — the negation of the definite FALSE above (lang_expr.html
    // §8). This is the case a wrongly-NULL empty result would get wrong.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE e(x, y)");
    assert_scalar(&mut db, "SELECT (1,2) NOT IN (SELECT x, y FROM e)", int(1));
}

// ---- Row-value comparison in WHERE against table columns (rowvalue.html §3.3) -

#[test]
fn row_where_eq_selects_only_matching_row() {
    // (x,y)=(1,2) is true only for the (1,2) row ⇒ x = 1.
    let mut db = table_t();
    assert_rows(&mut db, "SELECT x FROM t WHERE (x, y) = (1, 2)", &[vec![int(1)]]);
}

#[test]
fn row_where_lt_selects_lexicographically_smaller_row() {
    // (x,y)<(3,4): (1,2)<(3,4) true; (3,4)<(3,4) false ⇒ only x = 1.
    let mut db = table_t();
    assert_rows(
        &mut db,
        "SELECT x FROM t WHERE (x, y) < (3, 4) ORDER BY x",
        &[vec![int(1)]],
    );
}

// ---- Arity / scalar mismatch is an error (rowvalue.html §2; lang_expr.html §8) -
// A row value is comparable only with a row value of the SAME size; a size
// mismatch or a row-value-vs-scalar comparison is the "row value misused" family.
// These cases PASS: the engine errors, which is the spec-required outcome.
// Caveat: `assert_query_error` pins only THAT an error occurs, not which (the
// harness exposes no error-kind matching). An engine that implements row-value
// comparison must still REJECT these — for the arity / scalar-context reason —
// never evaluate them, so the assertion stays an error, not a value.

#[test]
fn row_eq_arity_mismatch_is_error() {
    // Size 2 vs size 3 — not comparable.
    let mut db = mem();
    assert_query_error(&mut db, "SELECT (1,2) = (1,2,3)");
}

#[test]
fn row_eq_against_scalar_is_error() {
    // Row value (size 2) vs scalar (size 1) — not comparable.
    let mut db = mem();
    assert_query_error(&mut db, "SELECT (1,2) = 1");
}

// ---- Row value misused in a scalar-only context is an error -----------------

#[test]
fn row_value_in_arithmetic_is_error() {
    // `+` requires scalar operands; a row value cannot be an arithmetic operand.
    let mut db = mem();
    assert_query_error(&mut db, "SELECT (1,2) + 1");
}

// ---- Row-value BETWEEN (rowvalue.html §2, worked example §3.2) ---------------
// rowvalue.html §2 lists BETWEEN among the row-value contexts; §3.2 gives the
// worked form `... WHERE (year,month,day) BETWEEN (2015,9,12) AND (2016,9,12)`.
// Expected values are transcribed by expanding to the spec's scalar definition:
//   (s) BETWEEN (lo) AND (hi) ≡ (s) >= (lo) AND (s) <= (hi)
//   (s) NOT BETWEEN (lo) AND (hi) ≡ (s) < (lo) OR (s) > (hi)
// with each row-value comparison expanded lexicographically as in §2.1.

#[test]
fn row_between_inside_range_is_true() {
    // (2,2)>=(1,1) is 1 (2>1), (2,2)<=(3,3) is 1 (2<3) ⇒ 1 AND 1 ≡ 1.
    eval_eq("(2,2) BETWEEN (1,1) AND (3,3)", int(1));
}

#[test]
fn row_between_above_high_is_false() {
    // (4,4)>=(1,1) is 1, (4,4)<=(3,3) is 0 (4>3) ⇒ 1 AND 0 ≡ 0.
    eval_eq("(4,4) BETWEEN (1,1) AND (3,3)", int(0));
}

#[test]
fn row_between_below_low_is_false() {
    // (0,0)>=(1,1) is 0 (0<1) ⇒ 0 AND _ ≡ 0.
    eval_eq("(0,0) BETWEEN (1,1) AND (3,3)", int(0));
}

#[test]
fn row_between_is_inclusive_at_low_bound() {
    // (1,1)>=(1,1) is 1 (reflexive >=), (1,1)<=(3,3) is 1 ⇒ 1.
    eval_eq("(1,1) BETWEEN (1,1) AND (3,3)", int(1));
}

#[test]
fn row_between_is_inclusive_at_high_bound() {
    // (3,3)>=(1,1) is 1, (3,3)<=(3,3) is 1 (reflexive <=) ⇒ 1.
    eval_eq("(3,3) BETWEEN (1,1) AND (3,3)", int(1));
}

#[test]
fn row_not_between_above_high_is_true() {
    // (4,4)<(1,1) is 0, (4,4)>(3,3) is 1 (4>3) ⇒ 0 OR 1 ≡ 1.
    eval_eq("(4,4) NOT BETWEEN (1,1) AND (3,3)", int(1));
}

#[test]
fn row_not_between_inside_range_is_false() {
    // (2,2)<(1,1) is 0, (2,2)>(3,3) is 0 ⇒ 0 OR 0 ≡ 0.
    eval_eq("(2,2) NOT BETWEEN (1,1) AND (3,3)", int(0));
}

#[test]
fn row_between_null_that_could_swing_is_unknown() {
    // (1,NULL) BETWEEN (1,1) AND (1,3) ≡ (1,NULL)>=(1,1) AND (1,NULL)<=(1,3).
    // Both bounds have an equal first element, so each reduces to NULL>=1 / NULL<=3,
    // each NULL; NULL AND NULL ≡ NULL (a substitution could make it true or false).
    eval_eq("(1,NULL) BETWEEN (1,1) AND (1,3)", null());
}

#[test]
fn row_between_arity_mismatch_is_error() {
    // The subject, low, and high must be row values of the SAME size.
    let mut db = mem();
    assert_query_error(&mut db, "SELECT (2,2) BETWEEN (1,1,1) AND (3,3)");
}

#[test]
fn row_between_against_scalar_bound_is_error() {
    // A scalar bound against a row-value subject is "row value misused".
    let mut db = mem();
    assert_query_error(&mut db, "SELECT (2,2) BETWEEN 1 AND (3,3)");
}

// ---- Row-value simple CASE (rowvalue.html §2) -------------------------------
// rowvalue.html §2 lists CASE among the row-value contexts. A simple
// `CASE (operand) WHEN (w) THEN r … END` compares the operand to each WHEN by row
// equality, i.e. it behaves like a searched CASE whose conditions are `(operand)=(w)`
// (each `=` expanded as in §2.1). The first row-equal WHEN supplies the result;
// none matching takes the ELSE (or NULL when there is no ELSE).

#[test]
fn row_simple_case_matching_when_selects_its_result() {
    // (1,2)=(1,2) ≡ 1 ⇒ the first arm fires ⇒ 1.
    eval_eq("CASE (1,2) WHEN (1,2) THEN 1 ELSE 0 END", int(1));
}

#[test]
fn row_simple_case_no_match_takes_else() {
    // (1,2)=(3,4) ≡ 0 ⇒ no arm fires ⇒ ELSE ⇒ 0.
    eval_eq("CASE (1,2) WHEN (3,4) THEN 1 ELSE 0 END", int(0));
}

#[test]
fn row_simple_case_scans_whens_in_order() {
    // (1,2)=(9,9) ≡ 0, then (1,2)=(1,2) ≡ 1 ⇒ the second arm's result ⇒ 2.
    eval_eq("CASE (1,2) WHEN (9,9) THEN 1 WHEN (1,2) THEN 2 ELSE 0 END", int(2));
}

#[test]
fn row_simple_case_no_match_no_else_is_null() {
    // No arm matches and there is no ELSE ⇒ NULL (lang_expr.html §5 CASE).
    eval_eq("CASE (1,2) WHEN (3,4) THEN 1 END", null());
}

#[test]
fn row_simple_case_when_arity_mismatch_is_error() {
    // A WHEN row value must match the operand's size, else "row value misused".
    let mut db = mem();
    assert_query_error(&mut db, "SELECT CASE (1,2) WHEN (1,2,3) THEN 1 ELSE 0 END");
}

// ---- Per-element affinity/collation (datatype3 §7; rowvalue.html §2.1) -------
// rowvalue.html §2.1 compares the constituent scalars "from left to right"; each
// scalar comparison follows the ordinary rules (datatype3 §7), so element i uses the
// affinity + collation that a scalar `a_i <op> b_i` would use — INDEPENDENTLY of the
// other elements. With a NOCASE first column and a BINARY (default) second column and
// the single row ('foo','bar'), the two elements need DIFFERENT collations, so a
// (wrong) shared meta gives a different, detectable answer.

/// A fresh in-memory database holding `tc(a TEXT COLLATE NOCASE, b TEXT)` with the
/// single row `('foo', 'bar')`. The two columns carry DIFFERENT collations, which is
/// what makes per-element vs shared comparison metadata observable.
fn table_collated() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE tc(a TEXT COLLATE NOCASE, b TEXT)");
    exec(&mut db, "INSERT INTO tc(a, b) VALUES ('foo', 'bar')");
    db
}

#[test]
fn row_eq_uses_per_element_collation_first_matches_second_binary_fails() {
    // (a,b)=('FOO','BAR') ≡ ('foo'='FOO' NOCASE) AND ('bar'='BAR' BINARY) ≡ 1 AND 0 ≡ 0.
    // Element 1 must use its own BINARY collation; a shared NOCASE meta would give 1.
    let mut db = table_collated();
    assert_scalar(&mut db, "SELECT (a, b) = ('FOO', 'BAR') FROM tc", int(0));
}

#[test]
fn row_eq_uses_per_element_collation_first_really_nocase() {
    // (a,b)=('FOO','bar') ≡ ('foo'='FOO' NOCASE) AND ('bar'='bar' BINARY) ≡ 1 AND 1 ≡ 1.
    // Confirms element 0 truly uses NOCASE; a shared BINARY meta would give 0.
    let mut db = table_collated();
    assert_scalar(&mut db, "SELECT (a, b) = ('FOO', 'bar') FROM tc", int(1));
}

// The same per-element collation invariant, driven through BETWEEN and simple CASE
// (whose lowering computes its own per-element metadata). Integer-literal cases can't
// witness this: only differing per-column collations expose a shared-meta regression
// inside bind_row_between / bind_row_case. Row is ('foo','bar'); a=NOCASE, b=BINARY.

#[test]
fn row_between_uses_per_element_collation_high_bound_binary() {
    // (a,b) BETWEEN ('FOO','AAA') AND ('FOO','BAR') ≡ (a,b)>=('FOO','AAA') AND (a,b)<=('FOO','BAR').
    // Element0 'foo'='FOO' under NOCASE (equal), so each bound decides on element1 BINARY:
    // 'bar'>='AAA' is 1, but 'bar'<='BAR' is 0 (BINARY: 'bar' > 'BAR') ⇒ 1 AND 0 ≡ 0.
    // A shared NOCASE meta would make 'bar'<='BAR' true ⇒ 1, so this pins element1 BINARY.
    let mut db = table_collated();
    assert_scalar(&mut db, "SELECT (a, b) BETWEEN ('FOO','AAA') AND ('FOO','BAR') FROM tc", int(0));
}

#[test]
fn row_between_uses_per_element_collation_low_bound_nocase() {
    // (a,b) BETWEEN ('FOO','AAA') AND ('FOO','zzz'): element0 'foo'='FOO' NOCASE (equal),
    // element1 'bar' in ['AAA','zzz'] BINARY ⇒ 1. Confirms element0 uses NOCASE: a shared
    // BINARY meta would make 'foo' > 'FOO' (element0), failing the '<= ('FOO',…)' bound ⇒ 0.
    let mut db = table_collated();
    assert_scalar(&mut db, "SELECT (a, b) BETWEEN ('FOO','AAA') AND ('FOO','zzz') FROM tc", int(1));
}

#[test]
fn row_simple_case_uses_per_element_collation_when_binary() {
    // CASE (a,b) WHEN ('FOO','BAR'): ('foo'='FOO' NOCASE) AND ('bar'='BAR' BINARY) ≡ 1 AND 0
    // ⇒ no match ⇒ ELSE ⇒ 0. A shared NOCASE meta would match ('bar'='BAR' NOCASE) ⇒ 1.
    let mut db = table_collated();
    assert_scalar(&mut db, "SELECT CASE (a,b) WHEN ('FOO','BAR') THEN 1 ELSE 0 END FROM tc", int(0));
}

#[test]
fn row_simple_case_uses_per_element_collation_when_nocase() {
    // CASE (a,b) WHEN ('FOO','bar'): ('foo'='FOO' NOCASE) AND ('bar'='bar' BINARY) ≡ 1 AND 1
    // ⇒ match ⇒ 1. Confirms element0 uses NOCASE; a shared BINARY meta would fail element0 ⇒ 0.
    let mut db = table_collated();
    assert_scalar(&mut db, "SELECT CASE (a,b) WHEN ('FOO','bar') THEN 1 ELSE 0 END FROM tc", int(1));
}

// ---- NULL edges for the generalized contexts (rowvalue.html §2.1 substitution) ----

#[test]
fn row_not_between_null_that_could_swing_is_unknown() {
    // NOT BETWEEN is the negation of BETWEEN: (1,NULL) BETWEEN (1,1) AND (1,3) is NULL
    // (see row_between_null_that_could_swing_is_unknown), so its negation is also NULL.
    eval_eq("(1,NULL) NOT BETWEEN (1,1) AND (1,3)", null());
}

#[test]
fn row_simple_case_null_operand_element_falls_through_to_else() {
    // CASE (1,NULL) WHEN (1,2): (1=1 AND NULL=2) ≡ TRUE AND NULL ≡ NULL. A NULL searched-CASE
    // condition is not true, so the arm does not fire ⇒ ELSE ⇒ 0.
    eval_eq("CASE (1,NULL) WHEN (1,2) THEN 1 ELSE 0 END", int(0));
}

#[test]
fn row_simple_case_null_operand_no_else_is_null() {
    // Same NULL condition, but with no ELSE ⇒ no arm fires ⇒ NULL (lang_expr.html §5).
    eval_eq("CASE (1,NULL) WHEN (1,2) THEN 1 END", null());
}
