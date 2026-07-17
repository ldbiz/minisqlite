//! Conformance: operators and operator precedence.
//!
//! Every expected result here is transcribed from the SQLite documentation, NOT
//! from whatever this engine currently returns:
//!   - operator set + precedence table: `spec/sqlite-doc/lang_expr.html` §2
//!   - operator type coercion:          `spec/sqlite-doc/datatype3.html`  §5
//!
//! Honesty rule: if the engine disagrees with the spec, KEEP the spec-correct
//! assertion and let it fail — do not weaken it to match the engine.
//!
//! Precedence table (highest -> lowest; operators in one cell share precedence
//! and, being binary, are left-associative):
//!   1. ~x  +x  -x (unary)      5. + -            9. = == <> != IS ... LIKE ...
//!   2. COLLATE                 6. & | << >>     10. NOT x
//!   3. || -> ->>               7. ESCAPE        11. AND
//!   4. * / %                   8. < > <= >=     12. OR
//! Key non-obvious consequences exercised below: `||` (3) binds tighter than
//! `*` `/` `%` (4) and `+` `-` (5); `+` `-` (5) bind tighter than the bitwise
//! `& | << >>` (6); unary `-`/`+`/`~` (1) bind tighter than everything binary.

mod conformance;
use conformance::*;

// ---------------------------------------------------------------------------
// Arithmetic — the basic binary and unary forms (lang_expr §2).
// ---------------------------------------------------------------------------

#[test]
fn add_integers() {
    eval_eq("2+3", int(5));
}

#[test]
fn subtract_integers_negative_result() {
    eval_eq("5-8", int(-3));
}

#[test]
fn multiply_integers() {
    eval_eq("4*3", int(12));
}

#[test]
fn unary_minus() {
    eval_eq("-5", int(-5));
}

#[test]
fn unary_plus_is_noop() {
    // lang_expr §2: the unary `+` operator is a no-op that returns its operand
    // unchanged (here, INTEGER 5).
    eval_eq("+5", int(5));
}

// ---------------------------------------------------------------------------
// Division — integer divide truncates toward zero; real if either operand is
// real (lang_expr §2, datatype3 §5).
// ---------------------------------------------------------------------------

#[test]
fn integer_division_truncates() {
    eval_eq("7/2", int(3));
}

#[test]
fn integer_division_negative_truncates_toward_zero() {
    // -7/2 = -3.5 truncated toward zero is -3 (NOT floored to -4).
    eval_eq("-7/2", int(-3));
}

#[test]
fn integer_division_below_one_is_zero() {
    eval_eq("1/2", int(0));
}

#[test]
fn division_real_left_operand_is_real() {
    eval_eq("7.0/2", real(3.5));
}

#[test]
fn division_real_right_operand_is_real() {
    eval_eq("7/2.0", real(3.5));
}

// ---------------------------------------------------------------------------
// Modulo — `%` casts both operands to INTEGER for the *computation* and the
// remainder's sign follows the dividend (lang_expr §2). But the RESULT storage
// class is not always INTEGER: datatype3 §5 singles `%` out — "the % operator
// returns either INTEGER or REAL (or NULL) depending on the type of its
// operands" — so an all-integer operand set gives an INTEGER result, while any
// REAL operand gives a REAL result (holding the same integer remainder value).
// This is unlike `<< >> & |`, which "always return an INTEGER".
// ---------------------------------------------------------------------------

#[test]
fn modulo_basic() {
    eval_eq("7%3", int(1));
}

#[test]
fn modulo_negative_dividend_sign_follows_dividend() {
    // (-7) % 3 -> sign follows the dividend (-7) -> -1.
    eval_eq("-7%3", int(-1));
}

#[test]
fn modulo_negative_divisor_sign_follows_dividend() {
    // 7 % (-3) -> sign follows the dividend (7) -> +1.
    eval_eq("7%-3", int(1));
}

#[test]
fn modulo_real_dividend_yields_real() {
    // Operand CAST to INTEGER only for the computation (5.5 -> 5, 5 % 2 = 1), but
    // the RESULT keeps storage class REAL because an operand is REAL -> Real(1.0).
    eval_eq("5.5%2", real(1.0));
}

#[test]
fn modulo_real_divisor_yields_real() {
    // Commuted: 2 % 5.5 -> 5.5 cast to 5, 2 % 5 = 2, REAL result -> Real(2.0).
    eval_eq("2%5.5", real(2.0));
}

#[test]
fn modulo_both_real_operands_yields_real() {
    // 5.5 -> 5, 2.0 -> 2, 5 % 2 = 1, REAL result -> Real(1.0).
    eval_eq("5.5%2.0", real(1.0));
}

// ---------------------------------------------------------------------------
// Division / modulo by zero -> NULL (datatype3 §5: "Division by zero gives a
// result of NULL"). Applies to integer, real, and modulo forms.
// ---------------------------------------------------------------------------

#[test]
fn integer_division_by_zero_is_null() {
    eval_eq("1/0", null());
}

#[test]
fn real_division_by_zero_is_null() {
    eval_eq("1.0/0", null());
}

#[test]
fn modulo_by_zero_is_null() {
    eval_eq("1%0", null());
}

// ---------------------------------------------------------------------------
// Concatenation `||` (lang_expr §2). Joins the text of both operands; result is
// TEXT. Any NULL operand makes the whole expression NULL (concat is NOT one of
// the AND/OR NULL exceptions).
// ---------------------------------------------------------------------------

#[test]
fn concat_two_strings() {
    eval_eq("'a'||'b'", text("ab"));
}

#[test]
fn concat_integers_yields_text() {
    // Each integer operand converts to its text form: '1' || '2' -> '12' (TEXT).
    eval_eq("1||2", text("12"));
}

#[test]
fn concat_with_null_is_null() {
    eval_eq("'x'||NULL", null());
}

// ---------------------------------------------------------------------------
// Bitwise operators — always INTEGER (or NULL) results (datatype3 §5).
// ---------------------------------------------------------------------------

#[test]
fn bitwise_and() {
    eval_eq("5&3", int(1));
}

#[test]
fn bitwise_or() {
    eval_eq("5|2", int(7));
}

#[test]
fn shift_left() {
    eval_eq("1<<4", int(16));
}

#[test]
fn shift_right() {
    eval_eq("256>>2", int(64));
}

#[test]
fn bitwise_not_of_zero_is_all_ones() {
    // ~0 sets every bit -> -1 in two's complement.
    eval_eq("~0", int(-1));
}

#[test]
fn bitwise_not_of_five() {
    // ~x == -x - 1, so ~5 == -6.
    eval_eq("~5", int(-6));
}

#[test]
fn bitwise_and_real_operand_stays_integer() {
    // Contrast with `%`: `& | << >>` cast a REAL operand to INTEGER for the
    // computation AND "always return an INTEGER" result (datatype3 §5). So
    // 5.5 & 2 -> 5.5 cast to 5, 5 & 2 = 0 -> Integer(0) (whereas 5.5 % 2 keeps
    // class REAL).
    eval_eq("5.5&2", int(0));
}

#[test]
fn shift_left_real_operand_stays_integer() {
    // 5.5 cast to 5, 5 << 1 = 10, INTEGER result -> Integer(10).
    eval_eq("5.5<<1", int(10));
}

// ---------------------------------------------------------------------------
// Type coercion in math (datatype3 §5): a non-NULL operand that "does not look
// in any way numeric" converts to 0/0.0; a numeric-looking STRING converts to
// INTEGER, or to REAL if it has a decimal point / exponent.
// ---------------------------------------------------------------------------

#[test]
fn non_numeric_string_is_zero_in_addition() {
    eval_eq("'abc'+1", int(1));
}

#[test]
fn non_numeric_string_is_zero_in_multiplication() {
    eval_eq("'abc'*2", int(0));
}

#[test]
fn integer_looking_string_converts_to_integer() {
    eval_eq("'10'+5", int(15));
}

#[test]
fn real_looking_string_converts_to_real() {
    // '2.5' has a decimal point, so it converts to REAL 2.5; 2.5 + 0 is REAL.
    eval_eq("'2.5'+0", real(2.5));
}

#[test]
fn integer_prefix_string_converts_to_integer_in_addition() {
    // datatype3 §5 / lang_expr §13: the coercion takes the LEADING numeric prefix,
    // not the whole string. '2abc' -> 2 (INTEGER form), so '2abc'+3 = 5. The
    // canonical '1english' likewise coerces to 1, exercising the arithmetic path
    // end-to-end (the boolean path is covered by conformance_select_where).
    eval_eq("'2abc'+3", int(5));
    eval_eq("'1english'+0", int(1));
}

#[test]
fn real_prefix_string_converts_to_real_in_addition() {
    // A leading prefix that carries a '.' is REAL: '2.5abc' -> 2.5, so '2.5abc'+0
    // is REAL 2.5 (not the whole-string-only rule, which would drop it to 0).
    eval_eq("'2.5abc'+0", real(2.5));
}

// ---------------------------------------------------------------------------
// NULL propagation (lang_expr §2: any operand NULL -> NULL, except AND/OR).
// ---------------------------------------------------------------------------

#[test]
fn addition_with_null_is_null() {
    eval_eq("1+NULL", null());
}

#[test]
fn multiplication_with_null_is_null() {
    eval_eq("NULL*5", null());
}

#[test]
fn bitwise_and_with_null_is_null() {
    eval_eq("NULL&1", null());
}

// ---------------------------------------------------------------------------
// Precedence — derived from the lang_expr §2 table.
// ---------------------------------------------------------------------------

#[test]
fn mul_binds_tighter_than_add() {
    // 2 + (3*4) = 14.
    eval_eq("2+3*4", int(14));
}

#[test]
fn parentheses_override_precedence() {
    eval_eq("(2+3)*4", int(20));
}

#[test]
fn mul_before_add_with_leading_product() {
    // (2*3) + 4 = 10.
    eval_eq("2*3+4", int(10));
}

#[test]
fn add_binds_tighter_than_bitwise_and() {
    // `+` (level 5) outranks `&` (level 6). Operands are chosen so the two parses
    // DIVERGE, so this genuinely pins the relationship (a `+`/`&`-swapped engine
    // would fail):
    //   correct (add first): (5+3) & 6 = 8 & 6 = 0
    //   broken  (and first): 5 + (3 & 6) = 5 + 2 = 7
    eval_eq("5+3&6", int(0));
}

#[test]
fn add_then_bitwise_and_value() {
    // Value check. NOTE: `1+2&3` does NOT by itself discriminate
    // precedence — both parses yield 3 ((1+2)&3 = 3&3 = 3; 1+(2&3) = 1+2 = 3).
    // The precedence relationship is guarded by `add_binds_tighter_than_bitwise_and`.
    eval_eq("1+2&3", int(3));
}

#[test]
fn add_binds_tighter_than_bitwise_or() {
    // `+` (5) outranks `|` (6), operands chosen to diverge:
    //   correct: (1+1) | 1 = 2 | 1 = 3 ;  broken: 1 + (1|1) = 1 + 1 = 2.
    eval_eq("1+1|1", int(3));
}

#[test]
fn add_binds_tighter_than_shift_left() {
    // `+` (5) outranks `<<` (6), operands chosen to diverge:
    //   correct: (1+1) << 1 = 2 << 1 = 4 ;  broken: 1 + (1<<1) = 1 + 2 = 3.
    eval_eq("1+1<<1", int(4));
}

#[test]
fn subtract_binds_tighter_than_shift_right() {
    // `-` (5) outranks `>>` (6), operands chosen to diverge:
    //   correct: (8-4) >> 1 = 4 >> 1 = 2 ;  broken: 8 - (4>>1) = 8 - 2 = 6.
    eval_eq("8-4>>1", int(2));
}

#[test]
fn unary_minus_binds_tighter_than_add() {
    // (-2) + 3 = 1.
    eval_eq("-2+3", int(1));
}

#[test]
fn unary_bitnot_binds_tighter_than_add() {
    // Unary `~` (level 1) outranks `+` (level 5): (~0) + 1 = -1 + 1 = 0.
    eval_eq("~0+1", int(0));
}

// ---------------------------------------------------------------------------
// Left-associativity — same-cell binary operators are left-associative
// (lang_expr §2 note 2).
// ---------------------------------------------------------------------------

#[test]
fn subtraction_is_left_associative() {
    // (10 - 3) - 2 = 5, not 10 - (3 - 2) = 9.
    eval_eq("10-3-2", int(5));
}

#[test]
fn division_is_left_associative() {
    // (16 / 4) / 2 = 2, not 16 / (4 / 2) = 8.
    eval_eq("16/4/2", int(2));
}

// ---------------------------------------------------------------------------
// The tricky `||` vs arithmetic precedence: `||` (level 3) outranks `*`/`/`/`%`
// (level 4) and `+`/`-` (level 5), so it binds TIGHTER than either. A common
// surprise, since concatenation reads like it should bind loosely.
// ---------------------------------------------------------------------------

#[test]
fn concat_vs_add_parenthesized_interpretations() {
    // The two candidate parses, pinned so the bare form below is unambiguous:
    eval_eq("(1||2)+3", int(15)); // '12' + 3 = 15 (INTEGER)
    eval_eq("1||(2+3)", text("15")); // '1' || 5 = '15' (TEXT)
}

#[test]
fn concat_binds_tighter_than_add() {
    // lang_expr §2 table: `||` (3) outranks `+` (5), so `||` binds first:
    //   1 || 2 + 3   =>   (1 || 2) + 3   =>   '12' + 3   =>   15 (INTEGER).
    // The competing "add first" parse would instead give '15' as TEXT.
    eval_eq("1||2+3", int(15));
}

#[test]
fn concat_binds_tighter_than_multiply() {
    // `||` (3) outranks `*` (4):
    //   2 || 3 * 4   =>   (2 || 3) * 4   =>   '23' * 4   =>   92 (INTEGER).
    eval_eq("2||3*4", int(92));
}

// ---------------------------------------------------------------------------
// Commuted operands — reversing operands must give the corresponding result
// (datatype3 §5 treats both operands symmetrically for coercion/NULL).
// ---------------------------------------------------------------------------

#[test]
fn addition_commutes() {
    eval_eq("3+2", int(5));
}

#[test]
fn null_addition_commuted_is_null() {
    eval_eq("NULL+1", null());
}

#[test]
fn multiplication_null_commuted_is_null() {
    eval_eq("5*NULL", null());
}

#[test]
fn non_numeric_string_addition_commuted() {
    eval_eq("1+'abc'", int(1));
}

#[test]
fn integer_looking_string_addition_commuted() {
    eval_eq("5+'10'", int(15));
}

#[test]
fn bitwise_and_commuted() {
    eval_eq("3&5", int(1));
}
