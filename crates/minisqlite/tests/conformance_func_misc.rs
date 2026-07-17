//! Conformance tests for the miscellaneous & conditional core scalar functions.
//!
//! Spec: `spec/sqlite-doc/lang_corefunc.html` §3 (each `<a name="...">` anchor is
//! cited on the relevant test). Covered here:
//!   * `typeof(X)`
//!   * `coalesce(X,Y,...)`, `ifnull(X,Y)`, `nullif(X,Y)`
//!   * `iif(B1,V1,...)` and its alias `if(...)`
//!   * `unhex(X)` / `unhex(X,Y)`
//!   * `zeroblob(N)`
//!   * the optimizer-hint no-ops `likely(X)` / `unlikely(X)` / `likelihood(X,Y)`
//!   * the connection counters `last_insert_rowid()`, `changes()`, `total_changes()`
//!
//! Every expectation is TRANSCRIBED FROM THE SPEC, never from what the engine
//! returns. When the engine disagrees, the spec-correct assertion is KEPT (it
//! fails) rather than weakened to make a failing case pass. Cases are split into many small tests so
//! one failure does not mask the others, and the newer/risky forms (the 2-arg and
//! multi-arg `iif`, the two-argument `unhex`) live in their own tests.

mod conformance;

use conformance::*;

// ---------------------------------------------------------------------------
// typeof(X)  — lang_corefunc.html §3 "typeof"
// "returns a string that indicates the datatype of the expression X:
//  'null', 'integer', 'real', 'text', or 'blob'."
// ---------------------------------------------------------------------------

#[test]
fn typeof_reports_the_storage_class_name() {
    eval_eq("typeof(1)", text("integer"));
    eval_eq("typeof(1.0)", text("real"));
    eval_eq("typeof('a')", text("text"));
    eval_eq("typeof(x'00')", text("blob"));
    eval_eq("typeof(NULL)", text("null"));
}

// ---------------------------------------------------------------------------
// coalesce(X,Y,...)  — lang_corefunc.html §3 "coalesce"
// "returns a copy of its first non-NULL argument, or NULL if all arguments are
//  NULL. Coalesce() must have at least 2 arguments."
// ---------------------------------------------------------------------------

#[test]
fn coalesce_returns_first_non_null_argument() {
    eval_eq("coalesce(NULL, 2, 3)", int(2));
    eval_eq("coalesce(NULL, NULL, 'x')", text("x"));
    eval_eq("coalesce(1, 2)", int(1));
}

#[test]
fn coalesce_all_null_is_null() {
    eval_eq("coalesce(NULL, NULL)", null());
}

#[test]
fn coalesce_requires_at_least_two_arguments() {
    // "Coalesce() must have at least 2 arguments." A single-argument call is an
    // error, not a pass-through.
    let mut db = mem();
    assert_query_error(&mut db, "SELECT coalesce(1)");
}

// ---------------------------------------------------------------------------
// ifnull(X,Y)  — lang_corefunc.html §3 "ifnull"
// "returns a copy of its first non-NULL argument, or NULL if both arguments are
//  NULL. Ifnull() must have exactly 2 arguments. ... equivalent to coalesce()
//  with two arguments."
// ---------------------------------------------------------------------------

#[test]
fn ifnull_returns_first_non_null_argument() {
    eval_eq("ifnull(NULL, 5)", int(5));
    eval_eq("ifnull(3, 5)", int(3));
}

#[test]
fn ifnull_both_null_is_null() {
    eval_eq("ifnull(NULL, NULL)", null());
}

#[test]
fn ifnull_requires_exactly_two_arguments() {
    // "Ifnull() must have exactly 2 arguments." One or three arguments is an error.
    let mut db = mem();
    assert_query_error(&mut db, "SELECT ifnull(1)");
    assert_query_error(&mut db, "SELECT ifnull(1, 2, 3)");
}

// ---------------------------------------------------------------------------
// nullif(X,Y)  — lang_corefunc.html §3 "nullif"
// "returns its first argument if the arguments are different and NULL if the
//  arguments are the same." (i.e. CASE WHEN X=Y THEN NULL ELSE X END)
// ---------------------------------------------------------------------------

#[test]
fn nullif_is_null_when_equal_else_the_first_argument() {
    eval_eq("nullif(1, 1)", null());
    eval_eq("nullif(1, 2)", int(1));
    eval_eq("nullif('a', 'a')", null());
}

#[test]
fn nullif_with_null_first_argument_is_null() {
    // nullif(X,Y) == CASE WHEN X=Y THEN NULL ELSE X END. With X = NULL, the
    // comparison NULL=Y is NULL (not true), so the ELSE branch returns X = NULL.
    eval_eq("nullif(NULL, 1)", null());
}

#[test]
fn nullif_requires_exactly_two_arguments() {
    let mut db = mem();
    assert_query_error(&mut db, "SELECT nullif(1)");
    assert_query_error(&mut db, "SELECT nullif(1, 2, 3)");
}

// ---------------------------------------------------------------------------
// iif(B1,V1,...) / if(...)  — lang_corefunc.html §3 "iif"
// "iif(X,Y,Z) ... logically equivalent to ... CASE WHEN X THEN Y ELSE Z END".
// Arguments come in (Boolean, value) pairs; the first true Boolean wins; an odd
// trailing argument is the ELSE; an even count with all-false Booleans yields
// NULL. "requires at least two arguments." "if() is just an alternative
// spelling for iif()."
// ---------------------------------------------------------------------------

#[test]
fn iif_three_argument_selects_by_condition() {
    eval_eq("iif(1, 'y', 'n')", text("y"));
    eval_eq("iif(0, 'y', 'n')", text("n"));
    // A NULL condition is not true, so the ELSE value is returned.
    eval_eq("iif(NULL, 'y', 'n')", text("n"));
}

#[test]
fn if_is_an_alias_for_iif() {
    // "The if() function is just an alternative spelling for iif()."
    eval_eq("if(1, 'y', 'n')", text("y"));
    eval_eq("if(0, 'y', 'n')", text("n"));
}

#[test]
fn iif_two_argument_form_returns_value_or_null() {
    // Two arguments = one (Boolean, value) pair with no ELSE. True -> the value;
    // an even count with the Boolean false -> NULL. (The 2-argument form was
    // added in SQLite 3.48.0; the engine reports version 3.50.0.)
    eval_eq("iif(1, 'a')", text("a"));
    eval_eq("iif(0, 'a')", null());
}

#[test]
fn iif_multi_argument_odd_uses_trailing_else() {
    // (0,'a'),(1,'b') with trailing ELSE 'c': first true Boolean is the second
    // pair -> 'b'. All-false with an odd trailing argument -> the ELSE 'c'.
    // (Multi-argument iif was added in SQLite 3.49.0.)
    eval_eq("iif(0, 'a', 1, 'b', 'c')", text("b"));
    eval_eq("iif(0, 'a', 0, 'b', 'c')", text("c"));
}

#[test]
fn iif_multi_argument_even_all_false_is_null() {
    // (1,'a'),(1,'b'): first true Boolean is the first pair -> 'a'. An even count
    // with every Boolean false -> NULL.
    eval_eq("iif(1, 'a', 1, 'b')", text("a"));
    eval_eq("iif(0, 'a', 0, 'b')", null());
}

#[test]
fn iif_requires_at_least_two_arguments() {
    // "The iif() function requires at least two arguments."
    let mut db = mem();
    assert_query_error(&mut db, "SELECT iif(1)");
}

// ---------------------------------------------------------------------------
// unhex(X) / unhex(X,Y)  — lang_corefunc.html §3 "unhex"
// "returns a BLOB value which is the decoding of the hexadecimal string X." A
// non-hex character not in Y, a split or odd pair, or a NULL argument all yield
// NULL. If Y is omitted X must be pure hex. Characters in Y that are not hex
// digits are ignored in X.
// ---------------------------------------------------------------------------

#[test]
fn unhex_decodes_hex_string_to_blob() {
    // '414243' -> 0x41 0x42 0x43 (the ASCII bytes of "ABC").
    eval_eq("unhex('414243')", blob(&[0x41, 0x42, 0x43]));
}

#[test]
fn unhex_accepts_mixed_case_and_empty_input() {
    // "The X input may contain an arbitrary mix of upper and lower case."
    eval_eq("unhex('aB01')", blob(&[0xAB, 0x01]));
    // A pure hex string of zero digits decodes to the empty blob.
    eval_eq("unhex('')", blob(&[]));
}

#[test]
fn unhex_invalid_input_returns_null() {
    // A non-hex character with no ignore set -> NULL.
    eval_eq("unhex('zz')", null());
    // An odd number of hex digits (digits must occur in pairs) -> NULL.
    eval_eq("unhex('123')", null());
    // "If either parameter X or Y is NULL, then unhex(X,Y) returns NULL."
    eval_eq("unhex(NULL)", null());
}

#[test]
fn unhex_two_argument_ignores_non_hex_in_y() {
    // The space is not a hex digit and is present in Y, so it is ignored in X.
    eval_eq("unhex('41 42', ' ')", blob(&[0x41, 0x42]));
}

#[test]
fn unhex_two_argument_split_pair_is_null() {
    // "All hexadecimal digits in X must occur in pairs, with both digits of each
    // pair beginning immediately adjacent to one another." An ignored character
    // sitting between the two digits of a pair splits it -> NULL.
    eval_eq("unhex('4 1', ' ')", null());
}

#[test]
fn unhex_null_second_argument_is_null() {
    // "If either parameter X or Y is NULL, then unhex(X,Y) returns NULL."
    eval_eq("unhex('41', NULL)", null());
}

// ---------------------------------------------------------------------------
// zeroblob(N)  — lang_corefunc.html §3 "zeroblob"
// "returns a BLOB consisting of N bytes of 0x00."
// ---------------------------------------------------------------------------

#[test]
fn zeroblob_is_n_zero_bytes() {
    eval_eq("zeroblob(3)", blob(&[0, 0, 0]));
}

#[test]
fn zeroblob_result_type_is_blob() {
    eval_eq("typeof(zeroblob(2))", text("blob"));
}

#[test]
fn zeroblob_of_zero_is_the_empty_blob() {
    // N = 0 bytes of 0x00 is the empty blob.
    eval_eq("zeroblob(0)", blob(&[]));
}

// ---------------------------------------------------------------------------
// likely(X) / unlikely(X) / likelihood(X,Y)  — lang_corefunc.html §3
// "likely"/"unlikely"/"likelihood": planner hints that each "return the argument
// X unchanged". likelihood's Y "must be a floating point constant between 0.0
// and 1.0, inclusive."
// ---------------------------------------------------------------------------

#[test]
fn likely_returns_its_argument_unchanged() {
    eval_eq("likely(5)", int(5));
    eval_eq("likely('x')", text("x"));
    eval_eq("likely(NULL)", null());
}

#[test]
fn unlikely_returns_its_argument_unchanged() {
    eval_eq("unlikely('x')", text("x"));
    eval_eq("unlikely(2.5)", real(2.5));
}

#[test]
fn likelihood_returns_its_first_argument_unchanged() {
    eval_eq("likelihood(7, 0.5)", int(7));
    eval_eq("likelihood('a', 0.9375)", text("a"));
}

#[test]
fn likelihood_second_argument_out_of_range_is_an_error() {
    // "The value Y in likelihood(X,Y) must be a floating point constant between
    // 0.0 and 1.0, inclusive." Real sqlite rejects an out-of-range Y at prepare
    // time ("second argument to likelihood() must be a constant between 0.0 and
    // 1.0").
    let mut db = mem();
    assert_query_error(&mut db, "SELECT likelihood(7, 2.0)");
}

// ---------------------------------------------------------------------------
// Context functions (need a table + a prior DML statement) —
// last_insert_rowid(), changes(), total_changes()  — lang_corefunc.html §3.
//   * last_insert_rowid(): "the ROWID of the last row insert".
//   * changes(): "rows that were changed or inserted or deleted by the most
//     recently completed INSERT, DELETE, or UPDATE statement".
//   * total_changes(): "the number of row changes ... since the current database
//     connection was opened".
// A SELECT/DDL statement between DML does NOT reset changes().
// ---------------------------------------------------------------------------

#[test]
fn last_insert_rowid_and_changes_after_first_insert() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "INSERT INTO t VALUES (10)");
    // The first implicit rowid is 1.
    assert_scalar(&mut db, "SELECT last_insert_rowid()", int(1));
    // One row was inserted by the most recent INSERT.
    assert_scalar(&mut db, "SELECT changes()", int(1));
}

#[test]
fn last_insert_rowid_advances_on_the_next_insert() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "INSERT INTO t VALUES (10)");
    assert_scalar(&mut db, "SELECT last_insert_rowid()", int(1));
    exec(&mut db, "INSERT INTO t VALUES (20)");
    // The second implicit rowid is 2.
    assert_scalar(&mut db, "SELECT last_insert_rowid()", int(2));
    // Still one row changed by the most recent (single-row) INSERT.
    assert_scalar(&mut db, "SELECT changes()", int(1));
}

#[test]
fn changes_counts_all_rows_of_a_multi_row_insert() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1), (2), (3)");
    // Three rows inserted; the last assigned rowid is 3.
    assert_scalar(&mut db, "SELECT changes()", int(3));
    assert_scalar(&mut db, "SELECT last_insert_rowid()", int(3));
}

#[test]
fn changes_counts_delete_and_update_rows() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1), (2), (3)");
    exec(&mut db, "DELETE FROM t WHERE a = 2");
    assert_scalar(&mut db, "SELECT changes()", int(1));
    exec(&mut db, "UPDATE t SET a = 99 WHERE a = 1");
    assert_scalar(&mut db, "SELECT changes()", int(1));
    // An UPDATE/DELETE does not change last_insert_rowid: it stays at the last
    // INSERT's rowid (3).
    assert_scalar(&mut db, "SELECT last_insert_rowid()", int(3));
}

#[test]
fn total_changes_accumulates_across_statements() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1), (2), (3)"); // +3
    exec(&mut db, "INSERT INTO t VALUES (4)"); // +1
    exec(&mut db, "DELETE FROM t WHERE a = 1"); // +1
    // The lifetime counter is 3 + 1 + 1 = 5.
    assert_scalar(&mut db, "SELECT total_changes()", int(5));
    // changes() reflects only the most recent statement (the DELETE).
    assert_scalar(&mut db, "SELECT changes()", int(1));
}

#[test]
fn changes_and_rowid_are_zero_before_any_dml() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    // No INSERT/UPDATE/DELETE has completed, and CREATE TABLE is DDL (it does not
    // touch the counters), so all three counters read 0.
    assert_scalar(&mut db, "SELECT changes()", int(0));
    assert_scalar(&mut db, "SELECT last_insert_rowid()", int(0));
    assert_scalar(&mut db, "SELECT total_changes()", int(0));
}
