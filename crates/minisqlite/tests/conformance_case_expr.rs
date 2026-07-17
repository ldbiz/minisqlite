//! Conformance battery for the SQL CASE expression, BOTH forms.
//!
//! Every expected value here is transcribed from the SQLite documentation —
//! `spec/sqlite-doc/lang_expr.html` §7 "The CASE expression" — never from what
//! the engine happens to return. The rules under test (§7):
//!
//! * Searched form `CASE WHEN c THEN r ... [ELSE e] END`: each WHEN is evaluated
//!   as a boolean left-to-right; the result is the THEN of the first WHEN that is
//!   TRUE, else the ELSE, else (no ELSE, no true WHEN) NULL. "A NULL result is
//!   considered untrue when evaluating WHEN terms" — so a WHEN of NULL (or one
//!   that evaluates to NULL) is not selected.
//! * Simple/base form `CASE x WHEN w THEN r ... [ELSE e] END`: `x` is evaluated
//!   once and compared to each WHEN "as if the base expression and WHEN
//!   expression are respectively the left- and right-hand operands of an `=`
//!   operator". Because `NULL = anything` is never true, a NULL base never
//!   matches any WHEN — "If the base expression is NULL then the result of the
//!   CASE is always the result of evaluating the ELSE expression if it exists,
//!   or NULL if it does not."
//! * Both forms use short-circuit evaluation and impose no single result type;
//!   each branch yields a value of its own storage class.
//!
//! If a case fails because the engine disagrees, the assertion STAYS
//! spec-correct (it fails) and is never weakened to pass.

mod conformance;

use conformance::*;

// ---------------------------------------------------------------------------
// Searched CASE (no base expression): first WHEN that is TRUE wins.
// ---------------------------------------------------------------------------

#[test]
fn searched_first_true_wins() {
    // The first (leftmost) TRUE WHEN selects its THEN; later WHENs are ignored.
    eval_eq("CASE WHEN 1 THEN 'a' WHEN 0 THEN 'b' END", text("a"));
}

#[test]
fn searched_first_true_wins_when_multiple_true() {
    // Two TRUE WHENs: the leftmost still wins ('a', not 'b').
    eval_eq("CASE WHEN 1 THEN 'a' WHEN 1 THEN 'b' END", text("a"));
}

#[test]
fn searched_skips_false_selects_later_true() {
    // A false leading WHEN is skipped; the next TRUE WHEN is selected.
    eval_eq("CASE WHEN 0 THEN 'a' WHEN 1 THEN 'b' ELSE 'c' END", text("b"));
}

#[test]
fn searched_else_taken_when_no_when_true() {
    // No WHEN is true -> the ELSE result.
    eval_eq("CASE WHEN 0 THEN 'a' ELSE 'b' END", text("b"));
}

#[test]
fn searched_no_match_no_else_is_null() {
    // No true WHEN and no ELSE -> the overall result is NULL (§7).
    eval_eq("CASE WHEN 0 THEN 'a' END", null());
}

// ---------------------------------------------------------------------------
// Searched CASE: "A NULL result is considered untrue when evaluating WHEN terms."
// ---------------------------------------------------------------------------

#[test]
fn searched_null_when_is_untrue_falls_to_else() {
    // A WHEN literal of NULL is not selected; the ELSE is taken.
    eval_eq("CASE WHEN NULL THEN 'a' ELSE 'b' END", text("b"));
}

#[test]
fn searched_null_when_no_else_is_null() {
    // NULL WHEN is untrue and there is no ELSE -> NULL.
    eval_eq("CASE WHEN NULL THEN 'a' END", null());
}

#[test]
fn searched_when_evaluating_to_null_is_untrue() {
    // A WHEN whose expression evaluates to NULL (e.g. `NULL = NULL`, `1 = NULL`)
    // is untrue, exactly like a literal NULL WHEN.
    eval_eq("CASE WHEN NULL = NULL THEN 'a' ELSE 'b' END", text("b"));
    eval_eq("CASE WHEN 1 = NULL THEN 'a' ELSE 'b' END", text("b"));
}

// ---------------------------------------------------------------------------
// Searched CASE: WHEN result is "treated as a boolean" — nonzero is TRUE, zero
// is FALSE (§7). Integer and real numeric truthiness.
// ---------------------------------------------------------------------------

#[test]
fn searched_nonzero_integer_is_true() {
    eval_eq("CASE WHEN 2 THEN 'a' ELSE 'b' END", text("a"));
    eval_eq("CASE WHEN -5 THEN 'a' ELSE 'b' END", text("a"));
}

#[test]
fn searched_zero_integer_is_false() {
    eval_eq("CASE WHEN 0 THEN 'a' ELSE 'b' END", text("b"));
}

#[test]
fn searched_nonzero_real_is_true_zero_real_is_false() {
    eval_eq("CASE WHEN 2.0 THEN 'a' ELSE 'b' END", text("a"));
    eval_eq("CASE WHEN 0.0 THEN 'a' ELSE 'b' END", text("b"));
}

// ---------------------------------------------------------------------------
// Searched CASE: WHEN may be an arbitrary boolean expression.
// ---------------------------------------------------------------------------

#[test]
fn searched_expression_conditions() {
    eval_eq("CASE WHEN 2>1 THEN 'y' ELSE 'n' END", text("y"));
    eval_eq("CASE WHEN 1=1 THEN 'y' ELSE 'n' END", text("y"));
    eval_eq("CASE WHEN 1<>1 THEN 'y' ELSE 'n' END", text("n"));
    eval_eq("CASE WHEN 'a'='a' THEN 'eq' ELSE 'ne' END", text("eq"));
}

// ---------------------------------------------------------------------------
// Simple/base CASE: base compared to each WHEN via `=` operator semantics.
// ---------------------------------------------------------------------------

#[test]
fn simple_selects_first_equal_when() {
    // base 2 equals the second WHEN (2) -> 'b'.
    eval_eq("CASE 2 WHEN 1 THEN 'a' WHEN 2 THEN 'b' END", text("b"));
}

#[test]
fn simple_basic_match() {
    eval_eq("CASE 5 WHEN 5 THEN 'hit' ELSE 'miss' END", text("hit"));
}

#[test]
fn simple_first_equal_when_wins() {
    // Two WHENs equal the base: the leftmost wins.
    eval_eq("CASE 1 WHEN 1 THEN 'a' WHEN 1 THEN 'b' END", text("a"));
}

#[test]
fn simple_when_expression_is_evaluated() {
    // The WHEN operand is itself an expression: base 6 equals 2*3 -> 'yes'.
    eval_eq("CASE 6 WHEN 2*3 THEN 'yes' ELSE 'no' END", text("yes"));
}

#[test]
fn simple_no_equal_when_takes_else() {
    // base 3 equals no WHEN -> the ELSE result.
    eval_eq("CASE 3 WHEN 1 THEN 'a' ELSE 'z' END", text("z"));
}

#[test]
fn simple_no_equal_when_no_else_is_null() {
    // base 3 equals no WHEN and there is no ELSE -> NULL.
    eval_eq("CASE 3 WHEN 1 THEN 'a' END", null());
}

// ---------------------------------------------------------------------------
// Simple/base CASE: NULL never matches via `=` (NULL = anything is not true).
// "If the base expression is NULL then the result of the CASE is always the
// result of evaluating the ELSE expression if it exists, or NULL if it does not."
// ---------------------------------------------------------------------------

#[test]
fn simple_null_base_does_not_match_null_when() {
    // NULL base vs NULL WHEN: `NULL = NULL` is not true, so the ELSE is taken.
    eval_eq("CASE NULL WHEN NULL THEN 'a' ELSE 'b' END", text("b"));
}

#[test]
fn simple_null_base_does_not_match_value_when() {
    // NULL base vs a value WHEN: not equal, so the ELSE is taken.
    eval_eq("CASE NULL WHEN 1 THEN 'a' ELSE 'b' END", text("b"));
}

#[test]
fn simple_null_base_no_else_is_null() {
    // NULL base, no ELSE -> NULL, regardless of whether a WHEN is NULL or a value.
    eval_eq("CASE NULL WHEN 1 THEN 'a' END", null());
    eval_eq("CASE NULL WHEN NULL THEN 'a' END", null());
}

#[test]
fn simple_value_base_does_not_match_null_when() {
    // A non-NULL base never equals a NULL WHEN (`1 = NULL` is not true): the NULL
    // WHEN is skipped and a later equal WHEN (or the ELSE) is chosen.
    eval_eq("CASE 1 WHEN NULL THEN 'a' ELSE 'b' END", text("b"));
    eval_eq("CASE 1 WHEN NULL THEN 'a' WHEN 1 THEN 'b' ELSE 'c' END", text("b"));
}

// ---------------------------------------------------------------------------
// Result value: each branch keeps its own storage class; CASE has no fixed type.
// ---------------------------------------------------------------------------

#[test]
fn result_types_differ_per_branch() {
    // The selected branch's value is returned verbatim: an INTEGER from one
    // branch, TEXT from another, in the same CASE expression.
    eval_eq("CASE WHEN 1 THEN 42 ELSE 'x' END", int(42));
    eval_eq("CASE WHEN 0 THEN 42 ELSE 'x' END", text("x"));
}

#[test]
fn result_branch_expression_is_evaluated() {
    // The THEN operand is an expression, returned as its computed value.
    eval_eq("CASE WHEN 1 THEN 3+4 END", int(7));
}

#[test]
fn result_typeof_reflects_selected_branch() {
    // typeof() of the CASE reports the storage class of the selected branch;
    // a NULL branch (or no-match, no-ELSE) reports 'null'.
    eval_eq("typeof(CASE WHEN 1 THEN NULL END)", text("null"));
    eval_eq("typeof(CASE WHEN 1 THEN 42 END)", text("integer"));
    eval_eq("typeof(CASE WHEN 1 THEN 2.5 END)", text("real"));
    eval_eq("typeof(CASE WHEN 1 THEN 'x' END)", text("text"));
    eval_eq("typeof(CASE WHEN 0 THEN 42 END)", text("null"));
}

// ---------------------------------------------------------------------------
// CASE is an expression: it nests inside THEN / ELSE of another CASE.
// ---------------------------------------------------------------------------

#[test]
fn nested_case_in_then() {
    eval_eq(
        "CASE WHEN 1 THEN CASE WHEN 0 THEN 'a' ELSE 'b' END ELSE 'c' END",
        text("b"),
    );
}

#[test]
fn nested_case_in_else() {
    eval_eq(
        "CASE WHEN 0 THEN 'a' ELSE CASE WHEN 1 THEN 'd' ELSE 'e' END END",
        text("d"),
    );
}
