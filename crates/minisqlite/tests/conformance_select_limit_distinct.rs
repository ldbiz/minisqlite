//! Conformance battery: the SELECT `LIMIT` / `OFFSET` clause and `DISTINCT` /
//! `ALL` duplicate-row processing. Every expected value here is TRANSCRIBED FROM
//! THE SPEC in `spec/sqlite-doc/lang_select.html`, never from what the engine
//! returns:
//!
//! - LIMIT / OFFSET — `lang_select.html` §5 "The LIMIT clause":
//!     * "the SELECT returns the first N rows of its result set only, where N is
//!       the value that the LIMIT expression evaluates to."
//!     * "if the SELECT statement would return less than N rows without a LIMIT
//!       clause, then the entire result set is returned." (LIMIT above row count)
//!     * "If the LIMIT expression evaluates to a negative value, then there is no
//!       upper bound on the number of rows returned." (negative LIMIT = all rows)
//!     * "Any scalar expression may be used in the LIMIT clause" (e.g. `1+1`).
//!     * OFFSET: "the first M rows are omitted from the result set ... and the
//!       next N rows are returned", and "if the SELECT would return less than M+N
//!       rows ... the first M rows are skipped and the remaining rows (if any)
//!       are returned." (OFFSET past the end => empty; partial last page)
//!     * "If the OFFSET clause evaluates to a negative value, the results are the
//!       same as if it had evaluated to zero."
//!     * Comma form: "the LIMIT clause may specify two scalar expressions
//!       separated by a comma. In this case, the first expression is used as the
//!       OFFSET expression and the second as the LIMIT expression." So
//!       `LIMIT 1, 2` == `LIMIT 2 OFFSET 1`.
//!     * Error path: "If the expression evaluates to a NULL value or any other
//!       value that cannot be losslessly converted to an integer, an error is
//!       returned." This is SEPARATE from the negative-value clause: NULL and
//!       non-convertible values ERROR; only a NEGATIVE value means no-limit. So
//!       `LIMIT NULL` and `LIMIT 2.5` are errors; §5 also requires the OFFSET
//!       expression to "must also" meet the same rule, so `OFFSET NULL` /
//!       `OFFSET 2.5` likewise error. NOTE: these four cases currently FAIL
//!       against the engine (it returns rows instead of erroring); the assertions
//!       stay spec-correct and are left as genuine failing assertions, never
//!       weakened to pass.
//! - DISTINCT / ALL — `lang_select.html` §2.6 "Removal of duplicate rows
//!   (DISTINCT processing)":
//!     * "If the simple SELECT is a SELECT ALL, then the entire set of result
//!       rows are returned"; "If neither ALL or DISTINCT are present, then the
//!       behavior is as if ALL were specified." (ALL / no keyword keep dups)
//!     * "If the simple SELECT is a SELECT DISTINCT, then duplicate rows are
//!       removed from the set of result rows before it is returned."
//!     * Duplicate detection uses IS DISTINCT FROM: "Thus two NULL values are
//!       considered to be equal. An integer is equal to a floating point number
//!       if they represent the same quantity."
//! - ORDER BY of NULLs (for the ordered DISTINCT case) — `lang_select.html` §4:
//!   "SQLite considers NULL values to be smaller than any other values for
//!   sorting purposes. Hence, NULLs naturally appear at the beginning of an ASC
//!   order-by."
//! - count(DISTINCT X) — `lang_aggfunc.html`: "count(X) ... returns a count of
//!   the number of times that X is not NULL in a group" and
//!   "count(distinct X) will return the number of distinct values of column X
//!   instead of the total number of non-null values." So count(DISTINCT X)
//!   counts distinct NON-NULL values (a lone NULL group contributes nothing).
//!
//! If a case fails because the engine disagrees with the spec, the assertion here
//! STAYS spec-correct (it is allowed to FAIL) rather than being weakened to pass.
//! Cases are split into many small `#[test]` fns so one failing (or
//! engine-rejected) case never masks the rest.
//!
//! DETERMINISM: every LIMIT/OFFSET case pairs the clause with `ORDER BY a` and
//! asserts with the ORDERED `assert_rows`, so the selected slice is
//! well-defined. DISTINCT cases without an ORDER BY use the multiset
//! `assert_rows_unordered`; the one `ORDER BY x` DISTINCT case uses `assert_rows`.

mod conformance;

use conformance::*;

use minisqlite::Connection;

/// Fresh in-memory db with `t(a INTEGER, b TEXT)` holding the five rows
/// {(1,'a'),(2,'b'),(3,'c'),(4,'d'),(5,'e')}.
///
/// The rows are INSERTed in a DELIBERATELY SCRAMBLED order (physical scan order
/// != `a` order) so the LIMIT/OFFSET tests actually discriminate the ORDER OF
/// APPLICATION. A correct engine sorts by `ORDER BY a` BEFORE applying
/// OFFSET/LIMIT, so it still returns the documented slice; a buggy
/// "truncate-then-sort" engine (OFFSET/LIMIT applied to the raw scan first, sort
/// after) would truncate different rows and return the wrong answer. If the seed
/// were ascending, the scan order would already equal the sorted order and both
/// the correct and the buggy engine would return byte-identical results, hiding
/// the bug. The reordering changes neither the row SET nor any `ORDER BY a`
/// result, so every expected value below is unchanged and stays spec-faithful.
fn t_db() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
    exec(
        &mut db,
        "INSERT INTO t VALUES (3,'c'),(1,'a'),(5,'e'),(2,'b'),(4,'d')",
    );
    db
}

/// Fresh in-memory db with `d(x, y)` holding duplicate and NULL rows used by the
/// DISTINCT/ALL tests: (1,'a'),(1,'a'),(2,'b'),(2,'c'),(NULL,'z'),(NULL,'z').
/// Columns are declared WITHOUT a type so they take BLOB affinity and no
/// storage-class coercion occurs on insert (datatype3 §3.1) — the integers stay
/// integers and the NULLs stay NULL.
fn d_db() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE d(x, y)");
    exec(
        &mut db,
        "INSERT INTO d VALUES (1,'a'),(1,'a'),(2,'b'),(2,'c'),(NULL,'z'),(NULL,'z')",
    );
    db
}

// =============================================================================
// LIMIT / OFFSET  (lang_select.html §5)
// =============================================================================

#[test]
fn limit_returns_first_n_rows() {
    // "the SELECT returns the first N rows of its result set only" — first 2.
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT a, b FROM t ORDER BY a LIMIT 2",
        &[vec![int(1), text("a")], vec![int(2), text("b")]],
    );
}

#[test]
fn limit_with_offset_skips_then_takes() {
    // OFFSET 1 omits the first row; LIMIT 2 then takes the next two.
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT a, b FROM t ORDER BY a LIMIT 2 OFFSET 1",
        &[vec![int(2), text("b")], vec![int(3), text("c")]],
    );
}

#[test]
fn offset_past_end_returns_empty() {
    // OFFSET 10 skips past all five rows, so the result set is empty.
    let mut db = t_db();
    assert_rows(&mut db, "SELECT a, b FROM t ORDER BY a LIMIT 2 OFFSET 10", &[]);
}

#[test]
fn limit_comma_form_is_offset_then_limit() {
    // Comma form `LIMIT off, cnt`: first expr is OFFSET, second is LIMIT. So
    // `LIMIT 1, 2` == `LIMIT 2 OFFSET 1` -> rows 2 and 3.
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT a, b FROM t ORDER BY a LIMIT 1, 2",
        &[vec![int(2), text("b")], vec![int(3), text("c")]],
    );
}

#[test]
fn negative_limit_returns_all_rows() {
    // "If the LIMIT expression evaluates to a negative value, then there is no
    // upper bound on the number of rows returned." -> all five rows.
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT a, b FROM t ORDER BY a LIMIT -1",
        &[
            vec![int(1), text("a")],
            vec![int(2), text("b")],
            vec![int(3), text("c")],
            vec![int(4), text("d")],
            vec![int(5), text("e")],
        ],
    );
}

#[test]
fn limit_accepts_scalar_expression() {
    // "Any scalar expression may be used in the LIMIT clause" — 1+1 = 2 rows.
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT a, b FROM t ORDER BY a LIMIT 1+1",
        &[vec![int(1), text("a")], vec![int(2), text("b")]],
    );
}

#[test]
fn limit_zero_returns_empty() {
    // LIMIT 0 bounds the result to the first zero rows -> empty.
    let mut db = t_db();
    assert_rows(&mut db, "SELECT a, b FROM t ORDER BY a LIMIT 0", &[]);
}

#[test]
fn limit_exceeding_row_count_returns_all() {
    // "if the SELECT statement would return less than N rows without a LIMIT
    // clause, then the entire result set is returned." N=100 > 5 rows.
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT a, b FROM t ORDER BY a LIMIT 100",
        &[
            vec![int(1), text("a")],
            vec![int(2), text("b")],
            vec![int(3), text("c")],
            vec![int(4), text("d")],
            vec![int(5), text("e")],
        ],
    );
}

#[test]
fn negative_offset_treated_as_zero() {
    // "If the OFFSET clause evaluates to a negative value, the results are the
    // same as if it had evaluated to zero." -> LIMIT 2 OFFSET 0 -> rows 1 and 2.
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT a, b FROM t ORDER BY a LIMIT 2 OFFSET -3",
        &[vec![int(1), text("a")], vec![int(2), text("b")]],
    );
}

#[test]
fn offset_with_partial_final_page() {
    // "if the SELECT would return less than M+N rows ... the first M rows are
    // skipped and the remaining rows (if any) are returned." M=3, N=3, M+N=6 > 5,
    // so skip 3 and return the remaining two rows (only 2, not 3).
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT a, b FROM t ORDER BY a LIMIT 3 OFFSET 3",
        &[vec![int(4), text("d")], vec![int(5), text("e")]],
    );
}

#[test]
fn offset_accepts_scalar_expression() {
    // §5 attaches "The expression attached to the optional OFFSET clause" — like
    // LIMIT, OFFSET takes a scalar expression. `1+1` = 2, so this omits the first
    // two rows and takes the next two -> rows 3 and 4.
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT a, b FROM t ORDER BY a LIMIT 2 OFFSET 1+1",
        &[vec![int(3), text("c")], vec![int(4), text("d")]],
    );
}

// ---- LIMIT / OFFSET error path (§5): a NULL / non-convertible bound errors ----
// LIMIT: §5 keeps two clauses SEPARATE — "If the expression evaluates to a NULL
// value or any other value that cannot be losslessly converted to an integer, an
// error is returned. If the LIMIT expression evaluates to a negative value, then
// there is no upper bound..." So for LIMIT, NULL / non-convertible -> ERROR; only
// a NEGATIVE value -> no-limit (that clause is `negative_limit_returns_all_rows`).
// OFFSET: §5 says the OFFSET expression "must ALSO evaluate to an integer, or a
// value that can be losslessly converted to an integer" — the "also" binds it to
// the LIMIT error clause, so a NULL / non-convertible OFFSET is likewise an error
// (a NEGATIVE OFFSET is instead treated as 0 — `negative_offset_treated_as_zero`).
// The OFFSET paragraph does not repeat "an error is returned" verbatim, so that
// back-reference is its grounding and is marginally softer than the LIMIT case.
// All four tests below transcribe the ERROR requirement; they currently FAIL —
// this engine returns rows instead of erroring. The assertions stay spec-correct
// and are left failing — not weakened, not ignored (the queries return wrong
// result sets; they do not hang or panic).

#[test]
fn limit_null_is_an_error() {
    // §5: "If the expression evaluates to a NULL value ... an error is returned."
    // KNOWN DISCREPANCY: this engine accepts `LIMIT NULL` and returns
    // all 5 rows (treating it as no-limit) instead of raising an error.
    let mut db = t_db();
    assert_query_error(&mut db, "SELECT a, b FROM t ORDER BY a LIMIT NULL");
}

#[test]
fn limit_non_integer_real_is_an_error() {
    // §5: LIMIT must be "an integer or a value that can be losslessly converted
    // to an integer. If the expression evaluates to ... any other value that
    // cannot be losslessly converted to an integer, an error is returned." 2.5
    // is not losslessly convertible, so it is an error.
    // KNOWN DISCREPANCY: this engine truncates `LIMIT 2.5` to 2 and
    // returns 2 rows instead of raising an error.
    let mut db = t_db();
    assert_query_error(&mut db, "SELECT a, b FROM t ORDER BY a LIMIT 2.5");
}

#[test]
fn offset_null_is_an_error() {
    // §5: the OFFSET expression "must also evaluate to an integer, or a value
    // that can be losslessly converted to an integer" — the "also" ties it to the
    // LIMIT error clause, so a NULL OFFSET is an error (NULL is not an integer).
    // KNOWN DISCREPANCY: this engine accepts `OFFSET NULL` and returns
    // rows instead of raising an error.
    let mut db = t_db();
    assert_query_error(&mut db, "SELECT a, b FROM t ORDER BY a LIMIT 2 OFFSET NULL");
}

#[test]
fn offset_non_integer_real_is_an_error() {
    // §5: OFFSET "must also evaluate to an integer, or a value that can be
    // losslessly converted to an integer." 2.5 is not losslessly convertible, so
    // (by the same "must also" back-reference to the LIMIT error clause) it is an
    // error.
    // KNOWN DISCREPANCY: this engine accepts `OFFSET 2.5` and returns
    // rows instead of raising an error.
    let mut db = t_db();
    assert_query_error(&mut db, "SELECT a, b FROM t ORDER BY a LIMIT 2 OFFSET 2.5");
}

// =============================================================================
// DISTINCT / ALL  (lang_select.html §2.6)
// =============================================================================

#[test]
fn distinct_single_column_collapses_nulls() {
    // DISTINCT removes duplicate rows; two NULLs are considered equal, so the
    // two NULL rows collapse to a single NULL. Result set {1, 2, NULL}.
    let mut db = d_db();
    assert_rows_unordered(
        &mut db,
        "SELECT DISTINCT x FROM d",
        &[vec![int(1)], vec![int(2)], vec![null()]],
    );
}

#[test]
fn distinct_multi_column_dedups_whole_rows() {
    // DISTINCT compares the full row: (1,'a') appears twice -> once; (2,'b') and
    // (2,'c') differ in y -> both kept; (NULL,'z') twice -> once.
    let mut db = d_db();
    assert_rows_unordered(
        &mut db,
        "SELECT DISTINCT x, y FROM d",
        &[
            vec![int(1), text("a")],
            vec![int(2), text("b")],
            vec![int(2), text("c")],
            vec![null(), text("z")],
        ],
    );
}

#[test]
fn all_keyword_keeps_duplicates() {
    // "If the simple SELECT is a SELECT ALL, then the entire set of result rows
    // are returned." All six x values (with duplicates and both NULLs).
    let mut db = d_db();
    assert_rows_unordered(
        &mut db,
        "SELECT ALL x FROM d",
        &[
            vec![int(1)],
            vec![int(1)],
            vec![int(2)],
            vec![int(2)],
            vec![null()],
            vec![null()],
        ],
    );
}

#[test]
fn no_keyword_defaults_to_all() {
    // "If neither ALL or DISTINCT are present, then the behavior is as if ALL
    // were specified." Same six rows as SELECT ALL.
    let mut db = d_db();
    assert_rows_unordered(
        &mut db,
        "SELECT x FROM d",
        &[
            vec![int(1)],
            vec![int(1)],
            vec![int(2)],
            vec![int(2)],
            vec![null()],
            vec![null()],
        ],
    );
}

#[test]
fn distinct_ordered_nulls_first() {
    // DISTINCT collapses to {NULL, 1, 2}; ORDER BY x (ascending) puts NULL first
    // because "SQLite considers NULL values to be smaller than any other values
    // for sorting purposes." Ordered result [NULL, 1, 2].
    let mut db = d_db();
    assert_rows(
        &mut db,
        "SELECT DISTINCT x FROM d ORDER BY x",
        &[vec![null()], vec![int(1)], vec![int(2)]],
    );
}

#[test]
fn count_distinct_ignores_null() {
    // count(DISTINCT x) counts distinct NON-NULL values. Distinct non-null x is
    // {1, 2}, and the NULL group contributes nothing -> 2.
    let mut db = d_db();
    assert_scalar(&mut db, "SELECT count(DISTINCT x) FROM d", int(2));
}

#[test]
fn distinct_treats_integer_and_real_of_same_quantity_as_equal() {
    // "An integer is equal to a floating point number if they represent the same
    // quantity." With BLOB affinity the untyped column keeps 1 as INTEGER and 1.0
    // as REAL, yet DISTINCT deduplication treats them as one value. So distinct
    // {1==1.0, 2} -> count(DISTINCT val) == 2. Using count() keeps the check
    // deterministic (which storage class survives is not asserted).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE mixed(val)");
    exec(&mut db, "INSERT INTO mixed VALUES (1),(1.0),(2)");

    // Guard the premise the collapse test rests on: the column must actually hold
    // BOTH storage classes (INTEGER 1 and REAL 1.0). Without this, count(DISTINCT
    // val)==2 would ALSO hold if the engine wrongly coerced 1.0 to INTEGER 1 on
    // insert (yielding {1,1,2} -> 2) — a NUMERIC-affinity bug that would mask the
    // real question. A typeless column has BLOB affinity (datatype3 §3.1), so no
    // coercion happens and typeof() sees {'integer','real'} across the rows.
    assert_rows_unordered(
        &mut db,
        "SELECT DISTINCT typeof(val) FROM mixed",
        &[vec![text("integer")], vec![text("real")]],
    );

    // The documented int==real collapse: 1 and 1.0 are one distinct value.
    assert_scalar(&mut db, "SELECT count(DISTINCT val) FROM mixed", int(2));
}
