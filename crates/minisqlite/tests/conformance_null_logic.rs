//! Conformance: three-valued (ternary) boolean logic.
//!
//! Every expected value below is transcribed from the SQLite documentation, NOT
//! from what this engine returns. The binding sources are:
//!
//!   * `spec/sqlite-doc/lang_expr.html` §2 (operators). "All operators generally
//!     evaluate to NULL when any operand is NULL, with specific exceptions." The
//!     two exceptions for the logical connectives: "When paired with NULL: AND
//!     evaluates to 0 (false) when the other operand is false; and OR evaluates
//!     to 1 (true) when the other operand is true."
//!   * `spec/sqlite-doc/lang_expr.html` (boolean coercion). "A numeric zero value
//!     (integer value 0 or real value 0.0) is considered to be false. A NULL
//!     value is still NULL. All other values are considered true." — so any
//!     nonzero number is TRUE, and a logical connective yields the integer 1 or 0
//!     (never the operand itself).
//!   * `spec/sqlite-doc/nulls.html` (cross-engine NULL chart, SQLite column):
//!     "null OR true" is true (Yes); "not (null AND false)" is true (Yes), which
//!     pins `NULL AND 0` = 0.
//!
//! In SQLite, booleans ARE integers: TRUE = 1, FALSE = 0, and UNKNOWN = NULL.
//! `Value` has no `PartialEq`, so every assertion goes through the shared harness
//! (`eval_eq` / `value_eq`); never compare a `Value` with `==`.

mod conformance;
use conformance::*;

// ---------------------------------------------------------------------------
// AND — full 3x3 truth table (lang_expr.html §2).
//
// Both operands non-NULL: ordinary boolean AND, result is the integer 1 or 0.
// One operand NULL: the result is NULL EXCEPT when the other operand is false
// (0), in which case AND short-circuits to 0 — the documented exception.
// ---------------------------------------------------------------------------

#[test]
fn and_true_true() {
    eval_eq("1 AND 1", int(1));
}

#[test]
fn and_true_false() {
    eval_eq("1 AND 0", int(0));
}

#[test]
fn and_false_true() {
    eval_eq("0 AND 1", int(0));
}

#[test]
fn and_false_false() {
    eval_eq("0 AND 0", int(0));
}

// The other operand is TRUE (not false), so the NULL is not absorbed → NULL.
#[test]
fn and_true_null() {
    eval_eq("1 AND NULL", null());
}

#[test]
fn and_null_true() {
    eval_eq("NULL AND 1", null());
}

// The other operand is FALSE, so AND is 0 regardless of the NULL (the exception).
#[test]
fn and_false_null() {
    eval_eq("0 AND NULL", int(0));
}

#[test]
fn and_null_false() {
    eval_eq("NULL AND 0", int(0));
}

// Neither operand is false, so no exception applies → NULL.
#[test]
fn and_null_null() {
    eval_eq("NULL AND NULL", null());
}

// ---------------------------------------------------------------------------
// OR — full 3x3 truth table (lang_expr.html §2).
//
// Both operands non-NULL: ordinary boolean OR, result is the integer 1 or 0.
// One operand NULL: the result is NULL EXCEPT when the other operand is true
// (nonzero), in which case OR short-circuits to 1 — the documented exception.
// ---------------------------------------------------------------------------

#[test]
fn or_true_true() {
    eval_eq("1 OR 1", int(1));
}

#[test]
fn or_true_false() {
    eval_eq("1 OR 0", int(1));
}

#[test]
fn or_false_true() {
    eval_eq("0 OR 1", int(1));
}

#[test]
fn or_false_false() {
    eval_eq("0 OR 0", int(0));
}

// The other operand is TRUE, so OR is 1 regardless of the NULL (the exception).
#[test]
fn or_true_null() {
    eval_eq("1 OR NULL", int(1));
}

#[test]
fn or_null_true() {
    eval_eq("NULL OR 1", int(1));
}

// The other operand is FALSE (not true), so the NULL is not absorbed → NULL.
#[test]
fn or_false_null() {
    eval_eq("0 OR NULL", null());
}

#[test]
fn or_null_false() {
    eval_eq("NULL OR 0", null());
}

// Neither operand is true, so no exception applies → NULL.
#[test]
fn or_null_null() {
    eval_eq("NULL OR NULL", null());
}

// ---------------------------------------------------------------------------
// NOT — unary (lang_expr.html §2). NOT flips a boolean; NOT of NULL is NULL by
// the general rule (any operand NULL → NULL).
// ---------------------------------------------------------------------------

#[test]
fn not_true() {
    eval_eq("NOT 1", int(0));
}

#[test]
fn not_false() {
    eval_eq("NOT 0", int(1));
}

#[test]
fn not_null() {
    eval_eq("NOT NULL", null());
}

// ---------------------------------------------------------------------------
// Nonzero truthiness (lang_expr.html boolean coercion). Any nonzero number is
// TRUE; the connective still yields the boolean integer 1/0, NOT the operand.
// ---------------------------------------------------------------------------

// Both nonzero → both true → 1 (not 3).
#[test]
fn and_two_three_is_one() {
    eval_eq("2 AND 3", int(1));
}

#[test]
fn and_two_zero_is_zero() {
    eval_eq("2 AND 0", int(0));
}

// Negative nonzero is still true.
#[test]
fn and_neg_one_one_is_one() {
    eval_eq("-1 AND 1", int(1));
}

#[test]
fn or_two_zero_is_one() {
    eval_eq("2 OR 0", int(1));
}

// ---------------------------------------------------------------------------
// IS NULL / IS NOT NULL predicates (lang_expr.html §2). These NEVER return
// NULL: the result is always 1 or 0.
// ---------------------------------------------------------------------------

#[test]
fn null_is_null_is_true() {
    eval_eq("NULL IS NULL", int(1));
}

#[test]
fn one_is_null_is_false() {
    eval_eq("1 IS NULL", int(0));
}

#[test]
fn null_is_not_null_is_false() {
    eval_eq("NULL IS NOT NULL", int(0));
}

#[test]
fn one_is_not_null_is_true() {
    eval_eq("1 IS NOT NULL", int(1));
}

// ---------------------------------------------------------------------------
// Combined: a connective's result feeding an IS NULL predicate. This confirms
// the NULL-vs-0 distinction through a second, independent operator path (the
// direct `eval_eq(.., null())` cases above already discriminate the two, since
// value_eq is class-sensitive) and exercises a parenthesized subexpression.
// ---------------------------------------------------------------------------

// (1 AND NULL) is NULL, so `... IS NULL` is 1 (true).
#[test]
fn and_1_null_is_null_is_true() {
    eval_eq("(1 AND NULL) IS NULL", int(1));
}

// (0 AND NULL) is 0 (the AND exception), so `... IS NULL` is 0 (false).
#[test]
fn and_0_null_is_null_is_false() {
    eval_eq("(0 AND NULL) IS NULL", int(0));
}
