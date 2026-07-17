//! Conformance battery for the WHERE clause of `SELECT`.
//!
//! Every expected value here is TRANSCRIBED FROM THE SQLITE DOCS in `spec/`,
//! never from what the engine returns:
//!
//!   * `spec/sqlite-doc/lang_select.html` §2.3 "WHERE clause filtering": the
//!     WHERE expression is evaluated as a boolean for each row and the row is
//!     kept ONLY when it is TRUE — "Rows are excluded from the result if the
//!     WHERE clause evaluates to either false or NULL." This one rule drives
//!     every three-valued-logic case below (`val = 20`, `val <> 20`, and
//!     `val = NULL` all silently drop the NULL-`val` rows).
//!   * `spec/sqlite-doc/lang_expr.html` §2 "Operators": "All operators generally
//!     evaluate to NULL when any operand is NULL"; the `%` operator casts both
//!     operands to INTEGER and returns the remainder; when paired with NULL, AND
//!     is false if the other side is false and OR is true if the other side is
//!     true.
//!   * lang_expr.html §5 (LIKE: `%`/`_` are wildcards, any other character
//!     matches itself case-insensitively for ASCII, so a wildcard-free pattern
//!     is an equality-style match), §6 (`x BETWEEN y AND z` ≡ `x>=y AND x<=z`),
//!     §8 (the IN / NOT IN truth matrix), and §14 "Boolean Expressions" (a value
//!     is coerced to boolean by casting to NUMERIC: integer/real zero is false,
//!     NULL stays NULL, everything else is true — hence `WHERE 1` keeps every
//!     row and `WHERE 0` keeps none).
//!
//! Honesty rule: if a case fails because the engine disagrees with the spec, the
//! assertion STAYS spec-correct — a real discrepancy left as a genuine failing
//! assertion, never weakened to make the suite pass.

mod conformance;

use conformance::*;
use minisqlite::{Connection, Value};

/// The shared fixture used by every case: `t(id INTEGER, name TEXT, val INTEGER)`
/// with six rows. Ids 4 and 6 carry a NULL `val` (to exercise three-valued
/// logic) and `val = 20` is duplicated across rows 2 and 5 (to exercise
/// multiplicity). `id` is unique and dense, so `ORDER BY id` is a total order and
/// the ordered assertions below are fully deterministic.
///
/// Canonical rows (id, name, val):
///   (1,'a',10) (2,'b',20) (3,'a',30) (4,'c',NULL) (5,'b',20) (6,'a',NULL)
fn fixture() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER, name TEXT, val INTEGER)");
    exec(
        &mut db,
        "INSERT INTO t VALUES \
         (1,'a',10),(2,'b',20),(3,'a',30),(4,'c',NULL),(5,'b',20),(6,'a',NULL)",
    );
    db
}

// The six fixture rows as projected by `SELECT id, name, val`. Naming them once
// keeps each expectation below a readable list of the rows that must survive the
// filter; the NULL `val` on r4/r6 is spelled explicitly so the 3VL cases are
// obvious at the call site.
fn r1() -> Vec<Value> {
    vec![int(1), text("a"), int(10)]
}
fn r2() -> Vec<Value> {
    vec![int(2), text("b"), int(20)]
}
fn r3() -> Vec<Value> {
    vec![int(3), text("a"), int(30)]
}
fn r4() -> Vec<Value> {
    vec![int(4), text("c"), null()]
}
fn r5() -> Vec<Value> {
    vec![int(5), text("b"), int(20)]
}
fn r6() -> Vec<Value> {
    vec![int(6), text("a"), null()]
}

/// Assert that `SELECT id, name, val FROM t WHERE <pred> ORDER BY id` yields
/// exactly `expected`, in order. Only the predicate varies across cases; the
/// projection and the deterministic `ORDER BY id` are held constant here so each
/// test reads as "this predicate keeps exactly these rows".
fn where_rows(db: &mut Connection, pred: &str, expected: &[Vec<Value>]) {
    let sql = format!("SELECT id, name, val FROM t WHERE {pred} ORDER BY id");
    assert_rows(db, &sql, expected);
}

// ---- Equality / inequality (lang_expr.html §2) ------------------------------

#[test]
fn eq_on_integer_key() {
    let mut db = fixture();
    where_rows(&mut db, "id = 3", &[r3()]);
}

#[test]
fn eq_on_integer_value_matches_duplicates() {
    // `val = 20` holds for rows 2 and 5; the NULL-`val` rows 4/6 yield NULL and
    // are excluded, not matched.
    let mut db = fixture();
    where_rows(&mut db, "val = 20", &[r2(), r5()]);
}

#[test]
fn eq_on_text() {
    let mut db = fixture();
    where_rows(&mut db, "name = 'a'", &[r1(), r3(), r6()]);
}

#[test]
fn not_equal_on_non_null_column() {
    // `id` is never NULL, so `<>` simply drops the single matching row (id 2).
    let mut db = fixture();
    where_rows(&mut db, "id <> 2", &[r1(), r3(), r4(), r5(), r6()]);
}

// ---- Range: comparison and BETWEEN (lang_expr.html §2, §6) -------------------

#[test]
fn range_greater_or_equal() {
    let mut db = fixture();
    where_rows(&mut db, "val >= 20", &[r2(), r3(), r5()]);
}

#[test]
fn range_between_is_inclusive() {
    // §6: `id BETWEEN 2 AND 4` ≡ `id >= 2 AND id <= 4`, inclusive of both bounds.
    let mut db = fixture();
    where_rows(&mut db, "id BETWEEN 2 AND 4", &[r2(), r3(), r4()]);
}

// ---- Boolean connectives AND / OR (lang_expr.html §2) -----------------------

#[test]
fn conjunction_and() {
    // name='a' is {r1,r3,r6}; val=30 is {r3}. Their AND keeps only r3. Note r6
    // is name='a' AND (val=30 -> NULL): true AND NULL = NULL, which is excluded.
    let mut db = fixture();
    where_rows(&mut db, "name = 'a' AND val = 30", &[r3()]);
}

#[test]
fn disjunction_or() {
    let mut db = fixture();
    where_rows(&mut db, "name = 'a' OR name = 'c'", &[r1(), r3(), r4(), r6()]);
}

// ---- IN / NOT IN (lang_expr.html §8) ----------------------------------------

#[test]
fn in_value_list() {
    let mut db = fixture();
    where_rows(&mut db, "id IN (1, 3, 5)", &[r1(), r3(), r5()]);
}

#[test]
fn not_in_value_list() {
    // `name` has no NULLs, so NOT IN ('a') is the plain complement: everything
    // whose name is not 'a'.
    let mut db = fixture();
    where_rows(&mut db, "name NOT IN ('a')", &[r2(), r4(), r5()]);
}

#[test]
fn in_list_containing_null_still_matches_a_present_value() {
    // §8 matrix: a value found in the list makes IN true even when the list also
    // holds a NULL, so r1 (val=10) survives. Every other row is either "not found
    // with a NULL present" (val 20/30 -> NULL) or "LHS is NULL" (rows 4/6 ->
    // NULL); both are excluded by the WHERE.
    let mut db = fixture();
    where_rows(&mut db, "val IN (10, NULL)", &[r1()]);
}

#[test]
fn not_in_list_without_null_is_the_complement() {
    // With no NULL in the list, `val NOT IN (10)` keeps the non-10, non-NULL vals
    // (rows 2,3,5) and still drops the NULL-`val` rows (LHS NULL -> NULL). This
    // is the contrast case for the NULL-in-list gotcha below.
    let mut db = fixture();
    where_rows(&mut db, "val NOT IN (10)", &[r2(), r3(), r5()]);
}

#[test]
fn not_in_list_containing_null_is_always_empty() {
    // The classic NULL-in-NOT-IN gotcha (§8 matrix): once the list holds a NULL,
    // a value that is NOT in the list yields NULL rather than true, so NOT IN is
    // never true for any row. `val NOT IN (10, NULL)` therefore keeps NOTHING —
    // even the rows a naive set-complement (rows 2,3,5 above) would keep.
    let mut db = fixture();
    where_rows(&mut db, "val NOT IN (10, NULL)", &[]);
}

// ---- LIKE (lang_expr.html §5) -----------------------------------------------

#[test]
fn like_without_wildcards_matches_equal_text() {
    // A pattern with no `%`/`_` matches the string itself (case-insensitive for
    // ASCII, per §5); over the shared fixture that selects exactly the name='a'
    // rows. Case-insensitivity is pinned separately below, since every name in
    // the shared fixture is already lowercase.
    let mut db = fixture();
    where_rows(&mut db, "name LIKE 'a'", &[r1(), r3(), r6()]);
}

#[test]
fn like_is_case_insensitive_for_ascii() {
    // §5: "Any other character matches itself or its lower/upper case
    // equivalent" — the doc's own example is that `'a' LIKE 'A'` is TRUE. LIKE
    // yields the integer 1/0, so assert the truth values directly, then confirm
    // the pattern 'a' matches BOTH 'a' and 'A' through the WHERE path.
    eval_eq("'a' LIKE 'A'", int(1));
    eval_eq("'A' LIKE 'a'", int(1));
    eval_eq("'a' LIKE 'b'", int(0));

    let mut db = mem();
    exec(&mut db, "CREATE TABLE lk(id INTEGER, s TEXT)");
    exec(&mut db, "INSERT INTO lk VALUES (1,'a'),(2,'A'),(3,'b'),(4,'B')");
    assert_rows(
        &mut db,
        "SELECT id, s FROM lk WHERE s LIKE 'a' ORDER BY id",
        &[vec![int(1), text("a")], vec![int(2), text("A")]],
    );
}

#[test]
fn like_percent_matches_any_character_sequence() {
    // §5: `%` matches any sequence of zero or more characters.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE lk(id INTEGER, s TEXT)");
    exec(
        &mut db,
        "INSERT INTO lk VALUES (1,'apple'),(2,'application'),(3,'banana'),(4,'grape')",
    );
    // Leading anchor: 'a%' keeps the rows beginning with 'a'.
    assert_rows(
        &mut db,
        "SELECT id, s FROM lk WHERE s LIKE 'a%' ORDER BY id",
        &[vec![int(1), text("apple")], vec![int(2), text("application")]],
    );
    // Trailing anchor: '%e' keeps the rows ending in 'e'.
    assert_rows(
        &mut db,
        "SELECT id, s FROM lk WHERE s LIKE '%e' ORDER BY id",
        &[vec![int(1), text("apple")], vec![int(4), text("grape")]],
    );
}

#[test]
fn like_underscore_matches_exactly_one_character() {
    // §5: `_` matches any single character, so 'c_t' matches only the 3-char
    // strings c?t — not the 4-char 'coat' or the 2-char 'ct'.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE lk(id INTEGER, s TEXT)");
    exec(
        &mut db,
        "INSERT INTO lk VALUES (1,'cat'),(2,'cot'),(3,'coat'),(4,'ct')",
    );
    assert_rows(
        &mut db,
        "SELECT id, s FROM lk WHERE s LIKE 'c_t' ORDER BY id",
        &[vec![int(1), text("cat")], vec![int(2), text("cot")]],
    );
}

// ---- NULL handling / three-valued logic (lang_select.html §2.3, lang_expr §2)-

#[test]
fn is_null_selects_null_rows() {
    let mut db = fixture();
    where_rows(&mut db, "val IS NULL", &[r4(), r6()]);
}

#[test]
fn is_not_null_selects_non_null_rows() {
    let mut db = fixture();
    where_rows(&mut db, "val IS NOT NULL", &[r1(), r2(), r3(), r5()]);
}

#[test]
fn equality_to_null_is_never_true() {
    // `val = NULL` is NULL for every row (NULL is never equal to anything, even
    // NULL — use IS NULL for that), so the WHERE excludes all rows.
    let mut db = fixture();
    where_rows(&mut db, "val = NULL", &[]);
}

#[test]
fn not_equal_excludes_null_rows() {
    // `val <> 20`: rows 2/5 (=20) are false; rows 4/6 (NULL) evaluate to NULL and
    // are ALSO excluded. Only the non-NULL, non-20 rows 1 and 3 survive.
    let mut db = fixture();
    where_rows(&mut db, "val <> 20", &[r1(), r3()]);
}

// ---- Expressions in WHERE (lang_expr.html §2) -------------------------------

#[test]
fn arithmetic_expression_predicate() {
    // `val + 5 > 20` ⇔ `val > 15`; NULL+5 is NULL so rows 4/6 drop out.
    let mut db = fixture();
    where_rows(&mut db, "val + 5 > 20", &[r2(), r3(), r5()]);
}

#[test]
fn modulo_expression_selects_even_ids() {
    // §2: `%` takes the integer remainder, so `id % 2 = 0` selects the even ids.
    let mut db = fixture();
    where_rows(&mut db, "id % 2 = 0", &[r2(), r4(), r6()]);
}

// ---- Constant / computed WHERE (lang_expr.html §14) -------------------------

#[test]
fn constant_true_keeps_all_rows() {
    // §14: a non-zero numeric coerces to true, so `WHERE 1` filters nothing.
    // This case doubles as the fixture guard: it is the one assertion that pins
    // the full (id,name,val) contents, so an edit to the `INSERT` in `fixture()`
    // that drifts from the `r1()..r6()` builders fails HERE.
    let mut db = fixture();
    where_rows(&mut db, "1", &[r1(), r2(), r3(), r4(), r5(), r6()]);
}

#[test]
fn constant_false_keeps_no_rows() {
    // §14: numeric zero coerces to false, so `WHERE 0` removes everything.
    let mut db = fixture();
    where_rows(&mut db, "0", &[]);
}

#[test]
fn constant_nonnumeric_string_is_false() {
    // §14: a value is coerced to boolean by casting to NUMERIC; a string with no
    // leading number casts to 0 -> false. The docs list 'english' explicitly
    // among the false values, so `WHERE 'english'` drops every row.
    let mut db = fixture();
    where_rows(&mut db, "'english'", &[]);
}

#[test]
fn constant_leading_number_string_is_true() {
    // §14: a boolean context casts to NUMERIC "in the same way as a CAST
    // expression", which takes a string's LEADING numeric prefix. So '1english'
    // casts to 1 -> true (the docs list '1english' explicitly among the TRUE
    // values), and `WHERE '1english'` keeps every row rather than dropping them.
    let mut db = fixture();
    where_rows(&mut db, "'1english'", &[r1(), r2(), r3(), r4(), r5(), r6()]);
}

// ---- Projection column names ------------------------------------------------

#[test]
fn projection_reports_source_column_names() {
    let mut db = fixture();
    assert_columns(&mut db, "SELECT id, name FROM t", &["id", "name"]);
}

// ---- Multiset (order-independent) sanity ------------------------------------

#[test]
fn where_result_is_order_independent_as_multiset() {
    // No ORDER BY, asserted as a multiset: the filter must yield this exact set
    // with these multiplicities regardless of the order the engine emits rows in.
    // The predicate is an OR across two columns with a NULL branch (deliberately
    // distinct from every ordered case above so this broadens coverage rather
    // than re-checking one query): name='b' is {r2,r5}, val=30 is {r3}, and the
    // NULL-`val` rows evaluate `false OR NULL = NULL` and drop out. Expected rows
    // are supplied out of order to exercise the multiset canonicalization.
    let mut db = fixture();
    assert_rows_unordered(
        &mut db,
        "SELECT id, name, val FROM t WHERE name = 'b' OR val = 30",
        &[r3(), r5(), r2()],
    );
}
