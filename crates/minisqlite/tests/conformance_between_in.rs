//! Conformance battery: the BETWEEN operator and the IN / NOT IN operators in
//! their list form. Every expected value here is TRANSCRIBED FROM THE SPEC in
//! `spec/sqlite-doc/`, never from what the engine returns:
//!
//! - BETWEEN: `lang_expr.html` §6 — "x BETWEEN y AND z" is equivalent to
//!   "x>=y AND x<=z" (with x evaluated only once); `datatype3.html` §4.2 repeats
//!   this as two separate comparisons "a>=b AND a<=c".
//! - IN list: `datatype3.html` §4.2 — "a IN (x, y, z, ...)" is equivalent to
//!   "a = +x OR a = +y OR a = +z OR ...".
//! - IN / NOT IN result matrix and the empty-set rule: `lang_expr.html` §8.
//! - Three-valued AND used to combine the two BETWEEN comparisons:
//!   `lang_expr.html` §2 (Operators, and Parse-Affecting Attributes) — operators
//!   generally yield NULL when any operand is NULL, EXCEPT that "AND evaluates to
//!   0 (false) when the other operand is false".
//!
//! If the engine disagrees with a documented value, the assertion here stays
//! spec-correct and is left as a genuine failing case rather than weakened to
//! pass. Cases are split into many
//! small `#[test]` fns so one failing (or engine-rejected) case never masks the
//! rest. Every case is a no-FROM expression evaluated via the shared harness's
//! `eval_eq`, which asserts the single result cell structurally (`Value` has no
//! `PartialEq`).

mod conformance;

use conformance::*;

// ---- BETWEEN: integer results 0/1 (lang_expr §6; datatype3 §4.2) -------------

#[test]
fn between_true_value_inside_range() {
    // 5 >= 1 AND 5 <= 10  ->  1 AND 1  ->  1
    eval_eq("5 BETWEEN 1 AND 10", int(1));
}

#[test]
fn between_true_on_equal_bounds() {
    // 5 >= 5 AND 5 <= 5  ->  1
    eval_eq("5 BETWEEN 5 AND 5", int(1));
}

#[test]
fn between_true_on_lower_endpoint() {
    // Lower endpoint is inclusive (>=): 1 >= 1 AND 1 <= 3 -> 1.
    eval_eq("1 BETWEEN 1 AND 3", int(1));
}

#[test]
fn between_true_on_upper_endpoint() {
    // Upper endpoint is inclusive (<=): 3 >= 1 AND 3 <= 3 -> 1.
    eval_eq("3 BETWEEN 1 AND 3", int(1));
}

#[test]
fn between_false_value_below_range() {
    // 0 >= 1 is false  ->  0
    eval_eq("0 BETWEEN 1 AND 10", int(0));
}

#[test]
fn between_false_value_above_range() {
    // 11 <= 10 is false  ->  0
    eval_eq("11 BETWEEN 1 AND 10", int(0));
}

#[test]
fn between_false_when_bounds_reversed() {
    // 5 >= 10 is false (and 5 <= 1 is false)  ->  0. BETWEEN does not reorder y,z.
    eval_eq("5 BETWEEN 10 AND 1", int(0));
}

#[test]
fn not_between_false_when_inside_range() {
    // NOT (5 BETWEEN 1 AND 10) -> NOT 1 -> 0
    eval_eq("5 NOT BETWEEN 1 AND 10", int(0));
}

#[test]
fn not_between_true_when_outside_range() {
    // 11 BETWEEN 1 AND 10 is 0; NOT 0 -> 1
    eval_eq("11 NOT BETWEEN 1 AND 10", int(1));
}

#[test]
fn between_text_true_within_range() {
    // TEXT comparison, default BINARY collation: 'a' <= 'b' <= 'c' (datatype3 §4.1).
    eval_eq("'b' BETWEEN 'a' AND 'c'", int(1));
}

#[test]
fn between_text_false_above_range() {
    // 'd' sorts after 'c' in BINARY order, so 'd' <= 'c' is false -> 0.
    eval_eq("'d' BETWEEN 'a' AND 'c'", int(0));
}

#[test]
fn not_between_text_false_within_range() {
    // NOT ('b' BETWEEN 'a' AND 'c') -> NOT 1 -> 0.
    eval_eq("'b' NOT BETWEEN 'a' AND 'c'", int(0));
}

// ---- BETWEEN with NULL: 3-valued AND (lang_expr §2; §6) ----------------------

#[test]
fn between_null_left_operand_is_null() {
    // (NULL >= 1) AND (NULL <= 10)  ->  NULL AND NULL  ->  NULL
    eval_eq("NULL BETWEEN 1 AND 10", null());
}

#[test]
fn between_null_lower_bound_is_null() {
    // (5 >= NULL) AND (5 <= 10)  ->  NULL AND true  ->  NULL
    eval_eq("5 BETWEEN NULL AND 10", null());
}

#[test]
fn between_null_upper_bound_is_null() {
    // (5 >= 1) AND (5 <= NULL)  ->  true AND NULL  ->  NULL
    eval_eq("5 BETWEEN 1 AND NULL", null());
}

#[test]
fn between_false_lower_comparison_with_null_upper() {
    // 5 BETWEEN 6 AND NULL: (5 >= 6) AND (5 <= NULL) -> false AND NULL -> 0.
    // The FALSE operand is the LOWER comparison (5 >= 6); the upper bound is NULL.
    // lang_expr §2: "AND evaluates to 0 (false) when the other operand is false".
    eval_eq("5 BETWEEN 6 AND NULL", int(0));
}

#[test]
fn between_false_upper_comparison_with_null_lower() {
    // 5 BETWEEN NULL AND 4: (5 >= NULL) AND (5 <= 4) -> NULL AND false -> 0.
    // Mirror of the case above — the FALSE operand is now the UPPER comparison
    // (5 <= 4) — proving the AND short-circuit is driven by the false operand,
    // not by which bound happens to be NULL (lang_expr §2).
    eval_eq("5 BETWEEN NULL AND 4", int(0));
}

// ---- IN (list form): integer/text results 0/1 (datatype3 §4.2; lang_expr §8) -

#[test]
fn in_list_integer_match() {
    // 3 = 1 OR 3 = 2 OR 3 = 3  ->  1
    eval_eq("3 IN (1, 2, 3)", int(1));
}

#[test]
fn in_list_integer_no_match() {
    // No equality holds, no NULL in list  ->  0
    eval_eq("4 IN (1, 2, 3)", int(0));
}

#[test]
fn in_list_text_match() {
    eval_eq("'b' IN ('a', 'b')", int(1));
}

#[test]
fn in_list_text_no_match() {
    eval_eq("'c' IN ('a', 'b')", int(0));
}

#[test]
fn not_in_list_true_when_absent() {
    // NOT (3 IN (1,2)) -> NOT 0 -> 1
    eval_eq("3 NOT IN (1, 2)", int(1));
}

#[test]
fn not_in_list_false_when_present() {
    // NOT (3 IN (1,2,3)) -> NOT 1 -> 0
    eval_eq("3 NOT IN (1, 2, 3)", int(0));
}

// ---- IN with NULL semantics — the result matrix in lang_expr §8 --------------
// Matrix rows used below (left-NULL | RHS-has-NULL | RHS-empty | found -> IN | NOT IN).
// Both the IN and the NOT IN column of each row is exercised.
//   no  | yes | no | yes -> true , false   (a match wins even if a NULL is present)
//   no  | yes | no | no  -> NULL , NULL     (no match, but a NULL is present)
//   yes | dnm | no | dnm -> NULL , NULL     (left operand is NULL)

#[test]
fn in_match_with_null_present_is_true() {
    // Row [no|yes|no|yes]: 1 is found in (1, NULL); a match yields 1 (IN=true)
    // even though the list has a NULL.
    eval_eq("1 IN (1, NULL)", int(1));
}

#[test]
fn not_in_match_with_null_present_is_false() {
    // Row [no|yes|no|yes], NOT IN side: a definite match forces NOT IN=false (0)
    // even though the list has a NULL. Mirror of `in_match_with_null_present_is_true`.
    eval_eq("1 NOT IN (1, NULL)", int(0));
}

#[test]
fn in_no_match_with_null_present_is_null() {
    // 3 not found in (1, 2, NULL) and a NULL is present  ->  NULL.
    eval_eq("3 IN (1, 2, NULL)", null());
}

#[test]
fn in_null_left_operand_is_null() {
    // Row [yes|dnm|no|dnm]: left operand NULL, non-empty list -> IN=NULL.
    eval_eq("NULL IN (1, 2)", null());
}

#[test]
fn not_in_null_left_operand_is_null() {
    // Row [yes|dnm|no|dnm], NOT IN side: left operand NULL, non-empty list ->
    // NOT IN=NULL. Mirror of `in_null_left_operand_is_null`.
    eval_eq("NULL NOT IN (1, 2)", null());
}

#[test]
fn not_in_no_match_with_null_present_is_null() {
    // NOT IN mirrors IN's NULL: no match with a NULL present  ->  NULL.
    eval_eq("3 NOT IN (1, 2, NULL)", null());
}

#[test]
fn not_in_no_match_with_single_null_is_null() {
    // 1 not equal to 2, and NULL present, no definite match  ->  NULL.
    eval_eq("1 NOT IN (2, NULL)", null());
}

#[test]
fn not_in_no_match_no_null_is_true() {
    // 1 not in (2,3), no NULL anywhere  ->  1.
    eval_eq("1 NOT IN (2, 3)", int(1));
}

// ---- Empty list — lang_expr §8: empty RHS => IN false / NOT IN true, ---------
// regardless of the left operand and even if the left operand is NULL. SQLite
// specifically permits the parenthesized empty list (§8 final note). Each case
// is isolated: if `IN ()` is rejected as a parse error by this engine, only
// these fail (the assertion is never weakened to match).

#[test]
fn in_empty_list_is_false() {
    eval_eq("5 IN ()", int(0));
}

#[test]
fn not_in_empty_list_is_true() {
    eval_eq("5 NOT IN ()", int(1));
}

#[test]
fn in_empty_list_with_null_left_is_false() {
    // Empty-set rule overrides the usual "left NULL -> NULL": result is 0.
    eval_eq("NULL IN ()", int(0));
}

#[test]
fn not_in_empty_list_with_null_left_is_true() {
    eval_eq("NULL NOT IN ()", int(1));
}
