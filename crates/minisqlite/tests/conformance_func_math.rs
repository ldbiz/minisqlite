//! Conformance battery for the built-in scalar MATH functions.
//!
//! Every expectation here is transcribed from the SQLite documentation
//! (`spec/sqlite-doc/lang_mathfunc.html` and `lang_corefunc.html`), never from what
//! the engine happens to return. SQL is run through the pinned facade
//! `minisqlite::Connection` via the shared harness (`tests/conformance/mod.rs`); a
//! `Value` has no `PartialEq`, so results are compared only through the harness
//! (`eval_eq`, `assert_scalar_approx`, `assert_query_error`).
//!
//! Scope: `abs`, `round`, `sign`, scalar `min`/`max` (2+ args) from `lang_corefunc`;
//! `ceil`/`ceiling`, `floor`, `trunc`, `sqrt`, `pow`/`power`, `exp`, `ln`,
//! `log`/`log10`/`log2`, `mod`, `pi`, `degrees`, `radians`, and the trig/hyperbolic
//! family from `lang_mathfunc`. `lang_mathfunc` §1 fixes the cross-cutting rules
//! this file leans on: an argument that is NULL, a BLOB, or a non-numeric string
//! makes a math function return NULL, and a domain error (e.g. `sqrt` of a negative,
//! `asin` outside [-1,1]) also returns NULL.
//!
//! ## Exact (`eval_eq`) vs tolerant (`assert_scalar_approx`)
//!
//! The choice is not stylistic — it tracks whether the *reference* value is exactly
//! representable:
//!
//! * `eval_eq` (bit-exact) is used only when the true result is exactly representable
//!   in `f64` AND is produced by a single correctly-rounded operation: `sqrt(4)=2.0`,
//!   `pow(2,10)=1024.0`, `exp(0)=1.0`, `ln(1)=0.0`, `log10(1)=0.0`, the storage-class
//!   results of `abs`/`round`/`sign`/`ceil`/`floor`/`trunc`/`min`/`max`, and every
//!   NULL / domain-error case (which returns NULL, a discrete value).
//! * `assert_scalar_approx` is used for irrational results (`sqrt(2)`, `pi()`, the
//!   trig family) and for values computed by a *composition* of transcendental ops.
//!   The load-bearing example is `log10(1000)`: SQLite (and this engine, matching
//!   `log(x)/M_LN10`) computes base-b logs as `ln(x)/ln(b)`, so `log10(1000)` is
//!   `2.9999999999999996`, not the exact `3.0`. Asserting an exact `3.0` there would
//!   contradict the reference, so those cases are checked within a tolerance.
//!
//! Tolerances: `1e-12` when comparing to a `std::f64::consts` value or a clean value a
//! correctly-rounded libm reproduces; `1e-9` for hand-transcribed multi-digit decimals
//! (the hyperbolic reference values) and composed arguments.
//!
//! Signed zero: `value_eq` treats `-0.0` and `+0.0` as distinct (bit identity), so any
//! expression that can yield a signed zero (`ceil(-0.5)`, `sin(0)`, …) is checked with
//! `assert_scalar_approx`, whose absolute-difference test ignores the sign of zero.
//!
//! Expected values are transcribed from the SQLite documentation, not from what this
//! engine returns; a case that reveals an engine bug is left as a genuine failing
//! assertion rather than weakened to pass.

mod conformance;

use conformance::*;
use std::f64::consts;

// ===========================================================================
// abs(X) — lang_corefunc.html: absolute value of the numeric argument X.
// ===========================================================================

/// `abs` of an INTEGER stays an INTEGER (storage class preserved).
#[test]
fn abs_integer_input_stays_integer() {
    eval_eq("abs(-5)", int(5));
    eval_eq("abs(5)", int(5));
    eval_eq("abs(0)", int(0));
    eval_eq("abs(-9223372036854775807)", int(9223372036854775807));
}

/// `abs` of a REAL stays a REAL.
#[test]
fn abs_real_input_stays_real() {
    eval_eq("abs(-5.5)", real(5.5));
    eval_eq("abs(5.5)", real(5.5));
    eval_eq("abs(-0.25)", real(0.25));
}

/// `abs(X)` returns NULL if X is NULL (lang_corefunc abs).
#[test]
fn abs_null_is_null() {
    eval_eq("abs(NULL)", null());
}

/// "Abs(X) returns 0.0 if X is a string or blob that cannot be converted to a
/// numeric value." (lang_corefunc abs) — note the result is the REAL 0.0.
#[test]
fn abs_unconvertible_text_or_blob_is_zero_real() {
    eval_eq("abs('abc')", real(0.0));
    eval_eq("abs(x'6162')", real(0.0));
}

/// "If X is the integer -9223372036854775808 then abs(X) throws an integer overflow
/// error since there is no equivalent positive 64-bit two's complement value."
/// (lang_corefunc abs). The bare literal `9223372036854775808` overflows i64 and
/// tokenizes as a REAL, so the i64::MIN integer is built by arithmetic to actually
/// reach the integer path.
#[test]
fn abs_of_min_integer_is_overflow_error() {
    // The subtraction alone is a valid i64 (i64::MIN is representable), so the error
    // must come from abs, not the arithmetic — pin that first so the error test is
    // specific to the documented overflow rather than any failure of this expression.
    eval_eq("-9223372036854775807 - 1", int(i64::MIN));
    let mut db = mem();
    let e = assert_query_error(&mut db, "SELECT abs(-9223372036854775807 - 1)");
    // The spec names this specifically an "integer overflow error"; assert the error
    // mentions overflow (substring, not exact wording — robust to rephrasing) so an
    // unrelated failure on this expression cannot pass this test.
    assert!(
        format!("{e:?}").to_ascii_lowercase().contains("overflow"),
        "expected an integer-overflow error, got {e:?}"
    );
}

/// A string that IS a number is coerced through abs's double path, so a numeric
/// string yields a REAL (lang_corefunc abs: "absolute value of the numeric argument").
#[test]
fn abs_of_numeric_string_is_real() {
    eval_eq("abs('-5.5')", real(5.5));
    eval_eq("abs('5')", real(5.0));
}

// ===========================================================================
// round(X[,Y]) — lang_corefunc.html: floating-point value rounded to Y digits
// (default 0), half away from zero. ALWAYS returns a REAL.
// ===========================================================================

/// `round(X)` rounds half away from zero and always returns a REAL.
#[test]
fn round_is_half_away_from_zero() {
    eval_eq("round(2.5)", real(3.0));
    eval_eq("round(2.4)", real(2.0));
    eval_eq("round(2.6)", real(3.0));
    eval_eq("round(-2.5)", real(-3.0));
    eval_eq("round(0.5)", real(1.0));
    eval_eq("round(-0.5)", real(-1.0));
}

/// `round(X,Y)` rounds to Y digits right of the decimal point.
///
/// These use bit-exact `eval_eq` even though `3.14`/`2.57` are not exactly
/// representable in `f64`: a correctly-rounded `round(x, 2)` yields the nearest double
/// to the decimal, which is *exactly* the double the Rust literal `3.14` parses to, so
/// bit identity holds. A 1-ULP failure here would be a real engine rounding
/// discrepancy to record, not a reason to loosen to `assert_scalar_approx`.
#[test]
fn round_to_decimal_places() {
    eval_eq("round(3.14159, 2)", real(3.14));
    eval_eq("round(2.567, 2)", real(2.57));
    eval_eq("round(-2.567, 2)", real(-2.57));
    // "If the Y argument is omitted or negative, it is taken to be 0."
    eval_eq("round(2.567, -1)", real(3.0));
}

/// An INTEGER argument still yields a REAL (round always returns floating point).
#[test]
fn round_of_integer_is_real() {
    eval_eq("round(123)", real(123.0));
    eval_eq("round(0)", real(0.0));
}

/// A NULL X (or NULL Y) yields NULL.
#[test]
fn round_null_argument_is_null() {
    eval_eq("round(NULL)", null());
    eval_eq("round(2.5, NULL)", null());
}

// ===========================================================================
// sign(X) — lang_corefunc.html: -1/0/+1 (INTEGER) for numeric X, else NULL.
// ===========================================================================

/// `sign` of integers is -1, 0, or +1 as an INTEGER.
#[test]
fn sign_of_integers() {
    eval_eq("sign(-5)", int(-1));
    eval_eq("sign(5)", int(1));
    eval_eq("sign(0)", int(0));
}

/// `sign` of reals is also returned as an INTEGER (-1/0/+1).
#[test]
fn sign_of_reals() {
    eval_eq("sign(2.5)", int(1));
    eval_eq("sign(-2.5)", int(-1));
    eval_eq("sign(0.0)", int(0));
}

/// "If the argument to sign(X) is NULL or a BLOB or a string that cannot be
/// losslessly converted into a number, then sign(X) returns NULL."
#[test]
fn sign_null_blob_or_nonnumeric_is_null() {
    eval_eq("sign(NULL)", null());
    eval_eq("sign('abc')", null());
    eval_eq("sign(x'31')", null());
}

// ===========================================================================
// scalar min(X,Y,...) / max(X,Y,...) — lang_corefunc.html (2+ args).
// ===========================================================================

/// The multi-argument `min()` returns the argument with the minimum value,
/// preserving that argument's storage class.
#[test]
fn scalar_min_returns_minimum_argument() {
    eval_eq("min(3, 1, 2)", int(1));
    eval_eq("min(1, 2)", int(1));
    // 1 < 2.5, so the returned argument is the INTEGER 1.
    eval_eq("min(2.5, 1)", int(1));
}

/// The multi-argument `max()` returns the argument with the maximum value.
#[test]
fn scalar_max_returns_maximum_argument() {
    eval_eq("max(3, 1, 2)", int(3));
    eval_eq("max(1, 2, 3)", int(3));
    eval_eq("max(1.5, 2.5, 0.5)", real(2.5));
}

/// Text arguments compare under the BINARY collation (lang_corefunc min/max).
#[test]
fn scalar_min_max_text_uses_binary_collation() {
    eval_eq("max('a', 'b')", text("b"));
    eval_eq("min('cherry', 'apple', 'banana')", text("apple"));
}

/// "return NULL if any argument is NULL" (lang_corefunc max; min mirrors it).
#[test]
fn scalar_min_max_null_if_any_argument_null() {
    eval_eq("min(1, NULL)", null());
    eval_eq("max(1, NULL, 2)", null());
    eval_eq("min(NULL, NULL)", null());
    eval_eq("max(NULL, 1)", null());
}

/// Cross-class ordering (datatype3 sort order: numeric < text): a number is less
/// than any text, so `min` keeps the number and `max` keeps the text.
#[test]
fn scalar_min_max_cross_class_order() {
    eval_eq("min(1, 'a')", int(1));
    eval_eq("max(1, 'a')", text("a"));
}

// ===========================================================================
// ceil/ceiling/floor/trunc — lang_mathfunc.html. A REAL argument yields a REAL.
// ===========================================================================

/// "Return the first representable integer value greater than or equal to X.
/// For positive values of X, this routine rounds away from zero." `ceiling` is an
/// alias for `ceil`.
#[test]
fn ceil_rounds_toward_positive_infinity() {
    eval_eq("ceil(2.1)", real(3.0));
    eval_eq("ceiling(2.1)", real(3.0));
    eval_eq("ceil(2.9)", real(3.0));
    eval_eq("ceil(2.0)", real(2.0));
}

/// "For negative values of X, this routine rounds toward zero." (ceil)
#[test]
fn ceil_negative_rounds_toward_zero() {
    eval_eq("ceil(-2.1)", real(-2.0));
    eval_eq("ceil(-2.9)", real(-2.0));
}

/// "Return the first representable integer value less than or equal to X.
/// For positive numbers, this function rounds toward zero." (floor)
#[test]
fn floor_rounds_toward_negative_infinity() {
    eval_eq("floor(2.9)", real(2.0));
    eval_eq("floor(2.1)", real(2.0));
    eval_eq("floor(2.0)", real(2.0));
}

/// "For negative numbers, this function rounds away from zero." (floor)
#[test]
fn floor_negative_rounds_away_from_zero() {
    eval_eq("floor(-2.1)", real(-3.0));
    eval_eq("floor(-2.9)", real(-3.0));
}

/// "return the integer part of X, rounding toward zero." (trunc)
#[test]
fn trunc_rounds_toward_zero() {
    eval_eq("trunc(2.9)", real(2.0));
    eval_eq("trunc(2.1)", real(2.0));
    eval_eq("trunc(-2.9)", real(-2.0));
    eval_eq("trunc(-2.1)", real(-2.0));
}

/// NULL argument -> NULL (lang_mathfunc §1) for the step functions.
#[test]
fn ceil_floor_trunc_null_is_null() {
    eval_eq("ceil(NULL)", null());
    eval_eq("ceiling(NULL)", null());
    eval_eq("floor(NULL)", null());
    eval_eq("trunc(NULL)", null());
}

/// An INTEGER argument is returned unchanged as an INTEGER: the step functions
/// preserve the storage class of a numeric argument (integer in -> integer out), where
/// the REAL-argument cases above return REAL. lang_mathfunc's "first representable
/// integer value" is that same integer for an integer input; this is real sqlite's
/// observable behavior (the HTML does not spell out the class, so it is pinned here).
#[test]
fn ceil_floor_trunc_of_integer_stays_integer() {
    eval_eq("ceil(2)", int(2));
    eval_eq("ceiling(2)", int(2));
    eval_eq("floor(2)", int(2));
    eval_eq("trunc(2)", int(2));
    eval_eq("ceil(-3)", int(-3));
    eval_eq("floor(-3)", int(-3));
}

// ===========================================================================
// sqrt(X) — lang_mathfunc.html: square root; NULL if X is negative.
// ===========================================================================

/// Perfect squares are exact (single correctly-rounded op with representable result).
#[test]
fn sqrt_of_perfect_squares_is_exact() {
    eval_eq("sqrt(4)", real(2.0));
    eval_eq("sqrt(9)", real(3.0));
    eval_eq("sqrt(1)", real(1.0));
    eval_eq("sqrt(0)", real(0.0));
}

/// Non-square roots are irrational — checked within tolerance.
#[test]
fn sqrt_of_non_squares_is_approximate() {
    let mut db = mem();
    assert_scalar_approx(&mut db, "SELECT sqrt(2)", consts::SQRT_2, 1e-12);
    assert_scalar_approx(&mut db, "SELECT sqrt(3)", 1.7320508075688772, 1e-12);
}

/// "NULL is returned if X is negative." (sqrt)
#[test]
fn sqrt_of_negative_is_null() {
    eval_eq("sqrt(-1)", null());
    eval_eq("sqrt(-4)", null());
}

// ===========================================================================
// pow(X,Y) / power(X,Y) — lang_mathfunc.html: X raised to the power Y.
// ===========================================================================

/// Integer exponents with exactly-representable results are exact; `power` is an alias.
#[test]
fn pow_integer_exponent_is_exact() {
    eval_eq("pow(2, 10)", real(1024.0));
    eval_eq("power(2, 3)", real(8.0));
    eval_eq("pow(2, 0)", real(1.0));
    eval_eq("pow(2, -1)", real(0.5));
    // pow(0,0) == 1.0 (IEEE 754 pow / C pow special case).
    eval_eq("pow(0, 0)", real(1.0));
}

/// A fractional exponent gives an irrational result — checked within tolerance.
#[test]
fn pow_fractional_exponent_is_approximate() {
    let mut db = mem();
    assert_scalar_approx(&mut db, "SELECT pow(2, 0.5)", consts::SQRT_2, 1e-12);
    assert_scalar_approx(&mut db, "SELECT pow(9, 0.5)", 3.0, 1e-12);
}

/// A negative base with a fractional exponent is a domain error (would be complex),
/// so lang_mathfunc §1's domain-error rule returns NULL.
#[test]
fn pow_negative_base_fractional_exponent_is_null() {
    eval_eq("pow(-2, 0.5)", null());
    eval_eq("power(-4, 0.5)", null());
}

// ===========================================================================
// exp(X) — lang_mathfunc.html: e raised to the power X.
// ===========================================================================

/// `exp(0)` is exactly 1.0.
#[test]
fn exp_of_zero_is_one() {
    eval_eq("exp(0)", real(1.0));
}

/// `exp` of nonzero arguments is checked within tolerance.
#[test]
fn exp_of_nonzero_is_approximate() {
    let mut db = mem();
    assert_scalar_approx(&mut db, "SELECT exp(1)", consts::E, 1e-12);
    assert_scalar_approx(&mut db, "SELECT exp(2)", 7.38905609893065, 1e-9);
}

// ===========================================================================
// ln(X) — lang_mathfunc.html: natural logarithm.
// ===========================================================================

/// `ln(1)` is exactly 0.0.
#[test]
fn ln_of_one_is_zero() {
    eval_eq("ln(1)", real(0.0));
}

/// `ln` of other positive arguments is checked within tolerance.
#[test]
fn ln_of_positive_is_approximate() {
    let mut db = mem();
    assert_scalar_approx(&mut db, "SELECT ln(10)", consts::LN_10, 1e-12);
    assert_scalar_approx(&mut db, "SELECT ln(2)", consts::LN_2, 1e-12);
}

/// Domain edge: ln of a non-positive argument is out of domain. SQLite returns NULL
/// (lang_mathfunc §1 domain-error rule; the ln(0) = -inf pole is also NULL).
#[test]
fn ln_of_nonpositive_is_null() {
    eval_eq("ln(0)", null());
    eval_eq("ln(-1)", null());
}

// ===========================================================================
// log / log10 / log2 — lang_mathfunc.html: base-10 (log/log10), base-2 (log2),
// or base-B two-argument log(B,X). Composed via ln ratios -> approximate.
// ===========================================================================

/// "Return the base-10 logarithm for X." (log/log10). Clean powers of ten still
/// come back as `ln(x)/ln(10)`, i.e. not bit-exact, so checked within tolerance.
#[test]
fn log10_is_base_ten() {
    let mut db = mem();
    assert_scalar_approx(&mut db, "SELECT log10(1000)", 3.0, 1e-12);
    assert_scalar_approx(&mut db, "SELECT log10(100)", 2.0, 1e-12);
    assert_scalar_approx(&mut db, "SELECT log10(10)", 1.0, 1e-12);
}

/// `log10(1)` is exactly 0.0 (`ln(1)=0` makes the ratio exactly 0.0).
#[test]
fn log10_of_one_is_zero() {
    eval_eq("log10(1)", real(0.0));
}

/// "Return the logarithm base-2 for the number X." (log2)
#[test]
fn log2_is_base_two() {
    let mut db = mem();
    assert_scalar_approx(&mut db, "SELECT log2(8)", 3.0, 1e-12);
    assert_scalar_approx(&mut db, "SELECT log2(2)", 1.0, 1e-12);
    assert_scalar_approx(&mut db, "SELECT log2(1024)", 10.0, 1e-12);
}

/// `log2(1)` is exactly 0.0.
#[test]
fn log2_of_one_is_zero() {
    eval_eq("log2(1)", real(0.0));
}

/// "SQLite works like PostgreSQL in that the log() function computes a base-10
/// logarithm." (single-argument log)
#[test]
fn log_single_argument_is_base_ten() {
    let mut db = mem();
    assert_scalar_approx(&mut db, "SELECT log(100)", 2.0, 1e-12);
    assert_scalar_approx(&mut db, "SELECT log(1000)", 3.0, 1e-12);
}

/// "for the two-argument version, return the base-B logarithm of X ... the first
/// argument is the base and the second argument is the operand." (log(B,X))
#[test]
fn log_two_argument_is_base_b() {
    let mut db = mem();
    assert_scalar_approx(&mut db, "SELECT log(2, 8)", 3.0, 1e-12);
    assert_scalar_approx(&mut db, "SELECT log(10, 100)", 2.0, 1e-12);
    assert_scalar_approx(&mut db, "SELECT log(2, 16)", 4.0, 1e-12);
}

/// Domain edge: a logarithm of a non-positive operand (or base) is out of domain and
/// returns NULL (lang_mathfunc §1).
#[test]
fn log_of_nonpositive_is_null() {
    eval_eq("log10(0)", null());
    eval_eq("log2(0)", null());
    eval_eq("log(0)", null());
    eval_eq("log(2, 0)", null());
}

// ===========================================================================
// mod(X,Y) — lang_mathfunc.html: remainder after dividing X by Y (like fmod).
// A math function, so the result is a REAL.
// ===========================================================================

/// `mod` returns the floating-point remainder; the result class is REAL.
#[test]
fn mod_returns_real_remainder() {
    eval_eq("mod(10, 3)", real(1.0));
    eval_eq("mod(7, 4)", real(3.0));
    eval_eq("mod(10.5, 3)", real(1.5));
}

/// The remainder takes the sign of the dividend (fmod semantics).
#[test]
fn mod_sign_follows_dividend() {
    eval_eq("mod(-10, 3)", real(-1.0));
    eval_eq("mod(-10.5, 3)", real(-1.5));
}

/// Dividing by zero has no remainder (NaN) — a domain error returning NULL.
#[test]
fn mod_by_zero_is_null() {
    eval_eq("mod(10, 0)", null());
}

// ===========================================================================
// pi() / degrees(X) / radians(X) — lang_mathfunc.html.
// ===========================================================================

/// "Return an approximation for π." — the closest IEEE-754 double to π.
#[test]
fn pi_is_the_double_nearest_pi() {
    let mut db = mem();
    assert_scalar_approx(&mut db, "SELECT pi()", consts::PI, 1e-12);
}

/// "Convert value X from radians into degrees." degrees(π) = 180.
#[test]
fn degrees_converts_radians_to_degrees() {
    let mut db = mem();
    assert_scalar_approx(&mut db, "SELECT degrees(pi())", 180.0, 1e-12);
    assert_scalar_approx(&mut db, "SELECT degrees(pi() / 2)", 90.0, 1e-9);
    eval_eq("degrees(0)", real(0.0));
}

/// "Convert X from degrees into radians." radians(180) = π.
#[test]
fn radians_converts_degrees_to_radians() {
    let mut db = mem();
    assert_scalar_approx(&mut db, "SELECT radians(180)", consts::PI, 1e-12);
    assert_scalar_approx(&mut db, "SELECT radians(90)", consts::FRAC_PI_2, 1e-12);
    eval_eq("radians(0)", real(0.0));
}

// ===========================================================================
// Trigonometry (X in radians) — lang_mathfunc.html.
// ===========================================================================

/// sin/cos/tan at 0 (checked with approx to ignore signed-zero).
#[test]
fn sin_cos_tan_at_zero() {
    let mut db = mem();
    assert_scalar_approx(&mut db, "SELECT sin(0)", 0.0, 1e-12);
    assert_scalar_approx(&mut db, "SELECT cos(0)", 1.0, 1e-12);
    assert_scalar_approx(&mut db, "SELECT tan(0)", 0.0, 1e-12);
}

/// sin/cos at key angles built from `pi()`.
#[test]
fn sin_cos_at_key_angles() {
    let mut db = mem();
    assert_scalar_approx(&mut db, "SELECT sin(pi() / 2)", 1.0, 1e-9);
    assert_scalar_approx(&mut db, "SELECT cos(pi())", -1.0, 1e-9);
    assert_scalar_approx(&mut db, "SELECT sin(pi())", 0.0, 1e-9);
    assert_scalar_approx(&mut db, "SELECT cos(pi() / 2)", 0.0, 1e-9);
    // A nonzero pin so a constant-returning or mis-scaled tan can't slip past tan(0).
    assert_scalar_approx(&mut db, "SELECT tan(pi() / 4)", 1.0, 1e-9);
}

/// Inverse circular functions return radians: asin(1)=π/2, acos(0)=π/2,
/// acos(0.5)=π/3, atan(1)=π/4.
#[test]
fn inverse_trig_returns_radians() {
    let mut db = mem();
    assert_scalar_approx(&mut db, "SELECT asin(1)", consts::FRAC_PI_2, 1e-12);
    assert_scalar_approx(&mut db, "SELECT asin(0)", 0.0, 1e-12);
    assert_scalar_approx(&mut db, "SELECT acos(1)", 0.0, 1e-12);
    assert_scalar_approx(&mut db, "SELECT acos(0)", consts::FRAC_PI_2, 1e-12);
    assert_scalar_approx(&mut db, "SELECT acos(0.5)", consts::FRAC_PI_3, 1e-12);
    assert_scalar_approx(&mut db, "SELECT atan(0)", 0.0, 1e-12);
    assert_scalar_approx(&mut db, "SELECT atan(1)", consts::FRAC_PI_4, 1e-12);
}

/// "atan2(Y,X) ... placed into correct quadrant depending on the signs of X and Y."
/// The argument order pins Y first, X second.
#[test]
fn atan2_places_result_in_correct_quadrant() {
    let mut db = mem();
    assert_scalar_approx(&mut db, "SELECT atan2(1, 1)", consts::FRAC_PI_4, 1e-12);
    assert_scalar_approx(&mut db, "SELECT atan2(1, 0)", consts::FRAC_PI_2, 1e-12);
    assert_scalar_approx(&mut db, "SELECT atan2(0, 1)", 0.0, 1e-12);
    assert_scalar_approx(&mut db, "SELECT atan2(0, -1)", consts::PI, 1e-12);
    assert_scalar_approx(&mut db, "SELECT atan2(-1, 0)", -consts::FRAC_PI_2, 1e-12);
}

/// Domain edge: asin/acos of |X| > 1 is out of domain (lang_mathfunc §1) -> NULL.
#[test]
fn inverse_trig_out_of_domain_is_null() {
    eval_eq("asin(2)", null());
    eval_eq("asin(-2)", null());
    eval_eq("acos(2)", null());
    eval_eq("acos(-2)", null());
}

// ===========================================================================
// Hyperbolic functions — lang_mathfunc.html.
// ===========================================================================

/// Hyperbolic functions at 0 (checked with approx to ignore signed-zero).
#[test]
fn hyperbolic_at_zero() {
    let mut db = mem();
    assert_scalar_approx(&mut db, "SELECT sinh(0)", 0.0, 1e-12);
    assert_scalar_approx(&mut db, "SELECT cosh(0)", 1.0, 1e-12);
    assert_scalar_approx(&mut db, "SELECT tanh(0)", 0.0, 1e-12);
    assert_scalar_approx(&mut db, "SELECT asinh(0)", 0.0, 1e-12);
    assert_scalar_approx(&mut db, "SELECT atanh(0)", 0.0, 1e-12);
}

/// Hyperbolic functions at 1 against their exact libm reference values.
#[test]
fn hyperbolic_at_one() {
    let mut db = mem();
    assert_scalar_approx(&mut db, "SELECT sinh(1)", 1.1752011936438014, 1e-9);
    assert_scalar_approx(&mut db, "SELECT cosh(1)", 1.5430806348152437, 1e-9);
    assert_scalar_approx(&mut db, "SELECT tanh(1)", 0.7615941559557649, 1e-9);
    assert_scalar_approx(&mut db, "SELECT asinh(1)", 0.8813735870195430, 1e-9);
}

/// Inverse hyperbolic: acosh(1)=0, acosh(2), atanh(0.5).
#[test]
fn inverse_hyperbolic_values() {
    let mut db = mem();
    assert_scalar_approx(&mut db, "SELECT acosh(1)", 0.0, 1e-12);
    assert_scalar_approx(&mut db, "SELECT acosh(2)", 1.3169578969248166, 1e-9);
    assert_scalar_approx(&mut db, "SELECT atanh(0.5)", 0.5493061443340548, 1e-9);
}

/// Domain edge: acosh(X<1) and atanh(|X|>1) are out of domain -> NULL.
#[test]
fn hyperbolic_out_of_domain_is_null() {
    eval_eq("acosh(0)", null());
    eval_eq("acosh(0.5)", null());
    eval_eq("atanh(2)", null());
    eval_eq("atanh(-2)", null());
}

// ===========================================================================
// NULL propagation across the whole family (lang_mathfunc §1).
// ===========================================================================

/// "If any argument is NULL ... the function will return NULL." — unary math family.
#[test]
fn null_propagates_through_unary_math_functions() {
    for expr in [
        "sqrt(NULL)",
        "exp(NULL)",
        "ln(NULL)",
        "log(NULL)",
        "log10(NULL)",
        "log2(NULL)",
        "sin(NULL)",
        "cos(NULL)",
        "tan(NULL)",
        "asin(NULL)",
        "acos(NULL)",
        "atan(NULL)",
        "sinh(NULL)",
        "cosh(NULL)",
        "tanh(NULL)",
        "asinh(NULL)",
        "acosh(NULL)",
        "atanh(NULL)",
        "degrees(NULL)",
        "radians(NULL)",
    ] {
        eval_eq(expr, null());
    }
}

/// NULL in either position of a two-argument math function yields NULL.
#[test]
fn null_propagates_through_binary_math_functions() {
    for expr in [
        "pow(NULL, 2)",
        "pow(2, NULL)",
        "pow(NULL, NULL)",
        "power(NULL, 3)",
        "mod(NULL, 3)",
        "mod(3, NULL)",
        "atan2(NULL, 1)",
        "atan2(1, NULL)",
        "log(NULL, 8)",
        "log(2, NULL)",
    ] {
        eval_eq(expr, null());
    }
}

// ===========================================================================
// Argument coercion (lang_mathfunc §1): numeric strings are accepted; blobs and
// non-numeric strings return NULL. abs/round are excluded from the NULL half — they
// coerce with double semantics and return 0.0 for an unconvertible argument, not NULL.
// ===========================================================================

/// lang_mathfunc §1: arguments "can be integers, floating-point numbers, or strings
/// that look like integers or real numbers." A whole-string numeric literal is
/// accepted as the number it names (real-form text keeps REAL affinity).
#[test]
fn math_functions_accept_numeric_strings() {
    eval_eq("sqrt('4')", real(2.0));
    eval_eq("exp('0')", real(1.0));
    eval_eq("ceil('2.5')", real(3.0));
    eval_eq("sign('2.5')", int(1));
    let mut db = mem();
    assert_scalar_approx(&mut db, "SELECT sqrt('2')", consts::SQRT_2, 1e-12);
}

/// lang_mathfunc §1: "a blob or a string that is not readily converted into a number"
/// -> NULL. This is the higher-risk half of the coercion rule (a distinct code path
/// from NULL short-circuiting), exercised across the whole unary family that uses it.
#[test]
fn math_unary_rejects_blob_and_nonnumeric_string() {
    for f in [
        "ceil", "ceiling", "floor", "trunc", "sqrt", "exp", "ln", "log", "log10",
        "log2", "degrees", "radians", "sin", "cos", "tan", "asin", "acos", "atan",
        "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    ] {
        eval_eq(&format!("{f}('abc')"), null());
        eval_eq(&format!("{f}(x'6162')"), null());
    }
}

/// The same blob / non-numeric-string rejection when EITHER argument of a two-argument
/// math function is unconvertible (lang_mathfunc §1).
#[test]
fn math_binary_rejects_blob_and_nonnumeric_string() {
    for expr in [
        "pow('abc', 2)",
        "pow(2, 'abc')",
        "power('abc', 2)",
        "mod('abc', 3)",
        "mod(3, 'abc')",
        "atan2('abc', 1)",
        "atan2(1, 'abc')",
        "log(2, 'abc')",
        "log('abc', 8)",
        "pow(x'6162', 2)",
        "mod(3, x'6162')",
    ] {
        eval_eq(expr, null());
    }
}
