//! Conformance battery: the IN / NOT IN operators with a TABLE NAME or a
//! TABLE-VALUED FUNCTION on the right-hand side. Every expected value here is
//! DERIVED FROM THE SPEC in `spec/sqlite-doc/lang_expr.html` §8, never from what
//! the engine returns.
//!
//! Spec basis (`lang_expr.html` §8, "The IN and NOT IN operators"):
//!   - "The right-hand side of an IN or NOT IN operator can be a table name or
//!     table-valued function name in which case the right-hand side is understood
//!     to be a subquery of the form '(SELECT * FROM name)'." So `x IN t` is
//!     literally `x IN (SELECT * FROM t)`, and `x IN f(a)` is `x IN (SELECT * FROM f(a))`.
//!   - "The subquery on the right of an IN or NOT IN operator must be a scalar
//!     subquery if the left expression is not a row value expression." A scalar
//!     subquery returns exactly one column, so the documented rule "that table
//!     must have exactly one column" is exactly the one-column requirement on the
//!     synthesized `SELECT *`.
//!   - "When the right operand is an empty set, the result of IN is false and the
//!     result of NOT IN is true, regardless of the left operand and even if the
//!     left operand is NULL."
//!   - The IN / NOT IN result matrix (columns: left-NULL | RHS-has-NULL |
//!     RHS-empty | left-found-in-RHS -> IN | NOT IN):
//!       no  | no  | no  | no  -> false , true    (no match, no NULL anywhere)
//!       dnm | no  | yes | no  -> false , true    (empty set overrides all)
//!       no  | dnm | no  | yes -> true  , false   (a definite match wins)
//!       no  | yes | no  | no  -> NULL  , NULL    (no match, but a NULL is present)
//!       yes | dnm | no  | dnm -> NULL  , NULL    (left operand is NULL, RHS non-empty)
//!
//! Because a table / TVF RHS is literally `(SELECT * FROM name)`, the one-column
//! rule and the 3-valued NULL logic both come from the SAME scalar-IN-subquery
//! machinery that `x IN (SELECT ...)` uses — this file exists to prove that
//! equivalence end to end.
//!
//! TABLE-VALUED FUNCTION note (spec-correct, and it surprises people): the direct
//! form `x IN json_each('[...]')` is an ERROR, not a row match. `json_each`
//! exposes eight columns (`key, value, type, atom, id, parent, fullkey, path`;
//! json1.html §4.24), so its `SELECT *` returns eight columns and the scalar-IN
//! one-column rule rejects it ("sub-select returns 8 columns - expected 1"). The
//! value-column idiom that people actually want is `x IN (SELECT value FROM
//! json_each('[...]'))`, which is tested here to document the contrast.
//!
//! ROW-VALUE subject: the `(SELECT * FROM name)` rewrite is UNCONDITIONAL — lang_expr.html
//! §8 does not restrict it to a scalar left operand — and rowvalue.html §2.2 only requires
//! the RHS of a row-value IN to be a subquery, which a table name (so understood) is. So
//! `(a, b, c) IN t` is `(a, b, c) IN (SELECT * FROM t)`, evaluated by the tuple 3VL, and the
//! tuple width must equal the table's column count (the tuple analogue of the one-column
//! rule). Only a value-LIST RHS stays the "row value misused" error.
//!
//! Assertions are spec-derived and never weakened to pass. Cases are split into many small
//! `#[test]` fns so one failing (or engine-rejected) case never masks the rest.

mod conformance;

use conformance::*;
use minisqlite::Connection;

// ---- fixtures ----------------------------------------------------------------

/// A one-column table `t(c)` holding the integers 1, 2, 3 (NO NULLs). `SELECT *
/// FROM t` is therefore a single-column, NULL-free candidate set {1, 2, 3}.
fn one_col_t() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(c)");
    exec(&mut db, "INSERT INTO t VALUES (1), (2), (3)");
    db
}

/// A one-column table `tn(c)` holding 1, 2 and a NULL — the candidate set
/// {1, 2, NULL}, used to exercise the 3-valued NULL logic of IN / NOT IN.
fn null_col_tn() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE tn(c)");
    exec(&mut db, "INSERT INTO tn VALUES (1), (2), (NULL)");
    db
}

// ---- x IN t : one-column table, no NULLs (matrix rows no|no|no|{yes,no}) ------

#[test]
fn in_table_scalar_match_is_true() {
    // `2 IN t` == `2 IN (SELECT * FROM t)` == `2 IN {1,2,3}`. 2 = 2 holds and t
    // has no NULL, so IN is TRUE (1).  Matrix: no|no|no|yes -> true.
    let mut db = one_col_t();
    assert_scalar(&mut db, "SELECT 2 IN t", int(1));
}

#[test]
fn in_table_scalar_no_match_is_false() {
    // `4 IN t` == `4 IN {1,2,3}`. No equality holds and t has NO NULL, so IN is
    // FALSE (0).  Matrix: no|no|no|no -> false.
    let mut db = one_col_t();
    assert_scalar(&mut db, "SELECT 4 IN t", int(0));
}

#[test]
fn not_in_table_scalar_absent_is_true() {
    // NOT (4 IN {1,2,3}) = NOT 0 = 1.  Matrix: no|no|no|no -> NOT IN=true.
    let mut db = one_col_t();
    assert_scalar(&mut db, "SELECT 4 NOT IN t", int(1));
}

#[test]
fn not_in_table_scalar_present_is_false() {
    // NOT (2 IN {1,2,3}) = NOT 1 = 0.  Matrix: no|no|no|yes -> NOT IN=false.
    let mut db = one_col_t();
    assert_scalar(&mut db, "SELECT 2 NOT IN t", int(0));
}

// ---- WHERE-clause filter forms (drive the uncorrelated set over many rows) ----

#[test]
fn in_table_filters_where_clause() {
    // probe = {1,2,3,4,5}; t = {1,2,3}. `v IN t` keeps a row iff v is in t.
    //   v = 1,2,3 -> IN true  (kept)
    //   v = 4,5   -> IN false (dropped; t has no NULL, so it is a definite false)
    // -> {1,2,3}.  (ordered by v for a deterministic assertion)
    let mut db = one_col_t();
    exec(&mut db, "CREATE TABLE probe(v)");
    exec(&mut db, "INSERT INTO probe VALUES (1), (2), (3), (4), (5)");
    assert_rows(
        &mut db,
        "SELECT v FROM probe WHERE v IN t ORDER BY v",
        &[vec![int(1)], vec![int(2)], vec![int(3)]],
    );
}

#[test]
fn not_in_table_filters_where_clause() {
    // Mirror of `in_table_filters_where_clause`. `v NOT IN t` keeps a row iff v is
    // NOT in t. t has no NULL, so NOT IN is a definite true/false everywhere:
    //   v = 4,5   -> NOT IN true  (kept)
    //   v = 1,2,3 -> NOT IN false (dropped)
    // -> {4,5}.
    let mut db = one_col_t();
    exec(&mut db, "CREATE TABLE probe(v)");
    exec(&mut db, "INSERT INTO probe VALUES (1), (2), (3), (4), (5)");
    assert_rows(
        &mut db,
        "SELECT v FROM probe WHERE v NOT IN t ORDER BY v",
        &[vec![int(4)], vec![int(5)]],
    );
}

// ---- x IN t : text values -----------------------------------------------------

#[test]
fn in_table_text_values() {
    // A one-column text set {'a','b'}. 'b' matches (-> 1); 'c' does not and there
    // is no NULL (-> 0). Confirms the table path is value-class agnostic, not
    // integer-only.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE tt(c)");
    exec(&mut db, "INSERT INTO tt VALUES ('a'), ('b')");
    assert_scalar(&mut db, "SELECT 'b' IN tt", int(1));
    assert_scalar(&mut db, "SELECT 'c' IN tt", int(0));
}

// ---- 3-valued NULL logic: a NULL present in the column ------------------------

#[test]
fn in_table_match_wins_over_null_present() {
    // tn = {1,2,NULL}. `2 IN tn`: a DEFINITE match (2 = 2) makes IN true even
    // though the set contains a NULL.  Matrix: no|yes|no|yes -> true.
    let mut db = null_col_tn();
    assert_scalar(&mut db, "SELECT 2 IN tn", int(1));
}

#[test]
fn not_in_table_match_wins_over_null_present() {
    // Mirror: a definite match forces NOT IN false even with a NULL present.
    //   Matrix: no|yes|no|yes -> NOT IN=false.
    let mut db = null_col_tn();
    assert_scalar(&mut db, "SELECT 2 NOT IN tn", int(0));
}

#[test]
fn in_table_no_match_with_null_present_is_null() {
    // tn = {1,2,NULL}. `3 IN tn`: no equality holds, but the set CONTAINS a NULL,
    // so the result is NULL (unknown), not false.  Matrix: no|yes|no|no -> NULL.
    let mut db = null_col_tn();
    assert_scalar(&mut db, "SELECT 3 IN tn", null());
}

#[test]
fn not_in_table_no_match_with_null_present_is_null() {
    // Mirror: `3 NOT IN tn` is also NULL (NOT of unknown is unknown).
    //   Matrix: no|yes|no|no -> NOT IN=NULL.
    let mut db = null_col_tn();
    assert_scalar(&mut db, "SELECT 3 NOT IN tn", null());
}

#[test]
fn null_in_column_drops_row_from_both_in_and_not_in() {
    // The explicit NULL requirement. tn = {1,2,NULL}; probe2 = {3}, and 3
    // is NOT among {1,2}. Because a NULL is present and there is no match:
    //   `3 IN tn`     -> NULL -> WHERE drops the row (a NULL predicate is not TRUE)
    //   `3 NOT IN tn` -> NULL -> WHERE drops the row too
    // So BOTH filters return ZERO rows — the signature behavior of a column that
    // contains NULL under 3-valued IN (lang_expr §8).
    let mut db = null_col_tn();
    exec(&mut db, "CREATE TABLE probe2(v)");
    exec(&mut db, "INSERT INTO probe2 VALUES (3)");
    assert_rows(&mut db, "SELECT v FROM probe2 WHERE v IN tn", &[]);
    assert_rows(&mut db, "SELECT v FROM probe2 WHERE v NOT IN tn", &[]);
}

// ---- left operand NULL, RHS non-empty (matrix row yes|dnm|no|dnm) -------------

#[test]
fn in_table_null_left_operand_is_null() {
    // Left operand NULL against a NON-EMPTY set -> IN is NULL, regardless of the
    // set's contents.  Matrix: yes|dnm|no|dnm -> IN=NULL.
    let mut db = one_col_t();
    assert_scalar(&mut db, "SELECT NULL IN t", null());
}

#[test]
fn not_in_table_null_left_operand_is_null() {
    // Mirror: NOT IN with a NULL left operand against a non-empty set is NULL.
    let mut db = one_col_t();
    assert_scalar(&mut db, "SELECT NULL NOT IN t", null());
}

// ---- empty table = empty set (the empty-set rule overrides left-NULL) ---------

#[test]
fn in_empty_table_is_false_even_for_null_left() {
    // An empty table is an empty RHS set. §8: IN over an empty set is FALSE
    // "regardless of the left operand and even if the left operand is NULL". So
    // both `1 IN e` and `NULL IN e` are 0 — the empty-set rule overrides the
    // usual "left NULL -> NULL".  Matrix: dnm|no|yes|no -> false.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE e(c)");
    assert_scalar(&mut db, "SELECT 1 IN e", int(0));
    assert_scalar(&mut db, "SELECT NULL IN e", int(0));
}

#[test]
fn not_in_empty_table_is_true_even_for_null_left() {
    // Mirror: NOT IN over an empty set is TRUE, even for a NULL left operand.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE e(c)");
    assert_scalar(&mut db, "SELECT 1 NOT IN e", int(1));
    assert_scalar(&mut db, "SELECT NULL NOT IN e", int(1));
}

// ---- multi-column table => "exactly one column" rule (loud error) -------------

#[test]
fn in_two_column_table_errors() {
    // `1 IN t2` where t2 has TWO columns == `1 IN (SELECT * FROM t2)`, and
    // `SELECT * FROM t2` returns two columns. A scalar-left IN subquery must
    // return exactly one column (§8: "that table must have exactly one column"),
    // so this is a LOUD error, not a fall-through. The message is the shared
    // arity error, verbatim.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t2(a, b)");
    let e = assert_query_error(&mut db, "SELECT 1 IN t2");
    assert!(
        e.to_string().contains("sub-select returns 2 columns - expected 1"),
        "expected the one-column arity error, got: {e}"
    );
}

// ---- table-valued function on the RHS ----------------------------------------

#[test]
fn in_json_each_tvf_errors_because_select_star_is_eight_columns() {
    // SPEC-CORRECT and easy to get wrong: `2 IN json_each('[1,2,3]')` ==
    // `2 IN (SELECT * FROM json_each('[1,2,3]'))`. `json_each` exposes EIGHT
    // columns (key, value, type, atom, id, parent, fullkey, path), so its
    // `SELECT *` is eight columns and the scalar-IN one-column rule rejects it.
    // This is NOT a row match against the values — it is a loud arity error, the
    // same as any other multi-column relation on the RHS of a scalar IN.
    let mut db = mem();
    let e = assert_query_error(&mut db, "SELECT 2 IN json_each('[1,2,3]')");
    assert!(
        e.to_string().contains("sub-select returns 8 columns - expected 1"),
        "expected the eight-column arity error for json_each, got: {e}"
    );
}

#[test]
fn in_json_each_value_via_explicit_subquery_matches() {
    // The value-column idiom people actually want (contrast with the direct-TVF
    // error above): project the single `value` column, then it is a one-column
    // subquery and the ordinary scalar IN applies.
    //   json_each('[1,2,3]').value = {1,2,3}
    //   `2 IN {1,2,3}` -> 1 (match) ; `5 IN {1,2,3}` -> 0 (no match, no NULL)
    let mut db = mem();
    assert_scalar(&mut db, "SELECT 2 IN (SELECT value FROM json_each('[1,2,3]'))", int(1));
    assert_scalar(&mut db, "SELECT 5 IN (SELECT value FROM json_each('[1,2,3]'))", int(0));
}

// ---- row-value subject: (a,b,c) IN <table> / <tvf> ---------------------------
// lang_expr.html §8's `(SELECT * FROM name)` rewrite is UNCONDITIONAL (not restricted to a
// scalar left operand), and rowvalue.html §2.2 only requires a row-value IN's RHS to be a
// subquery — which a table name, so understood, is. So `(a,b,c) IN t` is the tuple membership
// `(a,b,c) IN (SELECT * FROM t)`, with per-element 3-valued logic.

/// A three-column table `t3(x, y, z)` with rows (1,2,3), (2,3,4) and (1,NULL,5) — the
/// row shape from rowvalue.html §2.2, including a NULL to exercise tuple 3VL.
fn t3_xyz() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t3(x, y, z)");
    exec(&mut db, "INSERT INTO t3 VALUES (1, 2, 3), (2, 3, 4), (1, NULL, 5)");
    db
}

#[test]
fn row_value_in_table_match_is_true() {
    // `(1,2,3) IN t3` == `(1,2,3) IN (SELECT * FROM t3)`. The tuple equals row (1,2,3)
    // element-wise (1=1, 2=2, 3=3), a definite tuple match -> TRUE (1).
    let mut db = t3_xyz();
    assert_scalar(&mut db, "SELECT (1, 2, 3) IN t3", int(1));
}

#[test]
fn row_value_in_table_no_match_is_false() {
    // `(7,8,9) IN t3`: every candidate row is definitely unequal at its FIRST element
    // (7!=1, 7!=2, 7!=1), so AND3 short-circuits on the definite inequality and the NULL
    // in (1,NULL,5) never participates. No match, no unknown -> FALSE (0).
    let mut db = t3_xyz();
    assert_scalar(&mut db, "SELECT (7, 8, 9) IN t3", int(0));
}

#[test]
fn row_value_in_table_unknown_with_null_is_null() {
    // `(1,3,5) IN t3`: no definite match (row (1,2,3) differs at y: 3!=2; row (2,3,4)
    // differs at x: 1!=2), BUT against (1,NULL,5) every NON-NULL element is equal (1=1,
    // 5=5) and only the y element is blocked by the column NULL -> that row is an UNKNOWN
    // match. No definite match + an unknown match -> NULL (rowvalue.html §2.2 tuple 3VL).
    let mut db = t3_xyz();
    assert_scalar(&mut db, "SELECT (1, 3, 5) IN t3", null());
}

#[test]
fn row_value_not_in_table_mirrors_in() {
    // NOT IN is the 3-valued negation of IN (the executor negates the tri-state):
    //   (1,2,3): IN true  -> NOT IN false (0)
    //   (7,8,9): IN false -> NOT IN true  (1)
    //   (1,3,5): IN NULL  -> NOT IN NULL
    let mut db = t3_xyz();
    assert_scalar(&mut db, "SELECT (1, 2, 3) NOT IN t3", int(0));
    assert_scalar(&mut db, "SELECT (7, 8, 9) NOT IN t3", int(1));
    assert_scalar(&mut db, "SELECT (1, 3, 5) NOT IN t3", null());
}

#[test]
fn row_value_in_table_column_count_mismatch_errors() {
    // The tuple width must equal the table's column count (the tuple analogue of the
    // scalar one-column rule): `SELECT * FROM t3` returns 3 columns, so a 2-wide or a
    // 4-wide tuple is the shared arity error, both directions.
    let mut db = t3_xyz();
    let e2 = assert_query_error(&mut db, "SELECT (1, 2) IN t3");
    assert!(
        e2.to_string().contains("sub-select returns 3 columns - expected 2"),
        "expected a 3-vs-2 arity error, got: {e2}"
    );
    let e4 = assert_query_error(&mut db, "SELECT (1, 2, 3, 4) IN t3");
    assert!(
        e4.to_string().contains("sub-select returns 3 columns - expected 4"),
        "expected a 3-vs-4 arity error, got: {e4}"
    );
}

#[test]
fn row_value_in_tvf_errors_because_select_star_is_eight_columns() {
    // Row-value subject against a table-valued function drives the TableFunction branch of
    // the row-value lowering: `(1,2) IN json_each('[1,2,3]')` ==
    // `(1,2) IN (SELECT * FROM json_each('[1,2,3]'))`, and json_each's `SELECT *` is 8
    // columns, so it is the shared arity error against a 2-wide tuple (the row-value
    // analogue of the scalar json_each rejection).
    let mut db = mem();
    let e = assert_query_error(&mut db, "SELECT (1, 2) IN json_each('[1,2,3]')");
    assert!(
        e.to_string().contains("sub-select returns 8 columns - expected 2"),
        "expected an 8-vs-2 arity error for json_each, got: {e}"
    );
}

#[test]
fn row_value_in_value_list_is_still_row_value_misused() {
    // Guard: only a subquery / table / TVF RHS is lowered for a row-value subject. A
    // value-LIST RHS stays the "row value misused" error (rowvalue.html §2.2) — SQLite does
    // not fold a row value against a value list to a boolean. Confirms the table lowering
    // did not loosen the value-list rejection.
    let mut db = mem();
    let e = assert_query_error(&mut db, "SELECT (1, 2) IN ((1, 2), (3, 4))");
    assert!(
        e.to_string().contains("row value misused"),
        "expected row-value-misused for a value-list RHS, got: {e}"
    );
}
