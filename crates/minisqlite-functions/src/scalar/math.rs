//! Math scalar functions: `abs`, `round`, `sign` (from `lang_corefunc.html`) plus
//! the `-DSQLITE_ENABLE_MATH_FUNCTIONS` family from `lang_mathfunc.html`
//! (`ceil`/`ceiling`, `floor`, `trunc`, `exp`, `ln`, `log`/`log10`/`log2`,
//! `pow`/`power`, `sqrt`, `mod`, `pi`, `degrees`, `radians`, the trig family) and
//! the two-or-more-argument scalar overloads of `min`/`max`.
//!
//! Each function is a zero-sized unit struct implementing
//! [`ScalarFunction`](minisqlite_expr::ScalarFunction). Every function decides its
//! own NULL behavior; the registry validates the argument count before dispatch,
//! so a fixed-arity impl may index `args[0]`/`args[1]` without a length check.
//!
//! ## Argument coercion — matching real SQLite, not `to_number_for_arith`
//!
//! There are two coercion rules, deliberately different, because SQLite uses two:
//!
//! * The `lang_mathfunc.html` family and `sign` use `sqlite3_value_numeric_type`:
//!   an argument is usable only if it is INTEGER/REAL or **whole-string** numeric
//!   TEXT. `lang_mathfunc.html` §1 is explicit: "If any argument is NULL or a blob
//!   or a string that is not readily converted into a number, then the function
//!   will return NULL." So NULL, any BLOB, and non-numeric / partial-numeric TEXT
//!   all yield NULL. This is [`numeric_value`], NOT `to_number_for_arith` (which
//!   would coerce `'abc'`/blobs to `0` and silently diverge from the reference).
//! * `abs` and `round` use `sqlite3_value_double`: TEXT/BLOB take the longest
//!   leading numeric prefix (`0.0` if none), never NULL for a bad string. This is
//!   [`value_double`].
//!
//! ## Domain and overflow handling
//!
//! `lang_mathfunc.html` §1: math functions "return NULL for domain errors, such as
//! trying to take the square root of a negative number, or compute the arccosine of
//! a value greater than 1.0 or less than -1.0." Out-of-domain inputs that make the
//! C library return NaN are folded to NULL by [`real_result`]. `real_result`
//! deliberately lets a *finite overflow to ±inf* through, matching SQLite's
//! `mathFuncFinish` (which only rejects NaN): e.g. `exp(1000)` is `+Inf`, and
//! `atanh(±1)` — the ±1 poles, which are *not* domain errors under the strict
//! "greater than 1.0 or less than -1.0" wording — return `±Inf`, verified against
//! real sqlite. The logarithms are the exception: SQLite's `logFunc` NULLs a
//! non-positive argument explicitly (so `ln(0)`/`log(0)` are NULL, not `-Inf`),
//! which we match with an explicit `x > 0` guard; `sqrt`/`asin`/`acos`/`acosh` also
//! guard their domains explicitly, all producing NaN (hence NULL) otherwise.

use std::f64::consts;
use std::sync::Arc;

use minisqlite_expr::{to_integer, FnContext, ScalarFunction};
use minisqlite_types::{
    compare_values, looks_like_integer, looks_like_real, parse_real_prefix, text_to_numeric,
    Collation, Error, Result, Value,
};

use crate::registry::{Arity, FunctionRegistry};

/// Register the math scalar family.
pub(crate) fn register(reg: &mut FunctionRegistry) {
    // lang_corefunc.html
    reg.add_scalar("abs", Arity::Exact(1), Arc::new(Abs));
    reg.add_scalar("round", Arity::Range(1, 2), Arc::new(Round));
    reg.add_scalar("sign", Arity::Exact(1), Arc::new(Sign));
    // The multi-argument (>=2) scalar min()/max(). The single-argument forms are
    // the aggregate min()/max(), owned by the aggregate family in a separate
    // namespace; a 1-arg call never resolves here.
    reg.add_scalar("min", Arity::AtLeast(2), Arc::new(Min::default()));
    reg.add_scalar("max", Arity::AtLeast(2), Arc::new(Max::default()));

    // lang_mathfunc.html — rounding/step
    reg.add_scalar("ceil", Arity::Exact(1), Arc::new(Ceil));
    reg.add_scalar("ceiling", Arity::Exact(1), Arc::new(Ceil));
    reg.add_scalar("floor", Arity::Exact(1), Arc::new(Floor));
    reg.add_scalar("trunc", Arity::Exact(1), Arc::new(Trunc));

    // lang_mathfunc.html — exp/log
    reg.add_scalar("exp", Arity::Exact(1), Arc::new(Exp));
    reg.add_scalar("ln", Arity::Exact(1), Arc::new(Ln));
    reg.add_scalar("log", Arity::Range(1, 2), Arc::new(Log));
    reg.add_scalar("log10", Arity::Exact(1), Arc::new(Log10));
    reg.add_scalar("log2", Arity::Exact(1), Arc::new(Log2));
    reg.add_scalar("pow", Arity::Exact(2), Arc::new(Pow));
    reg.add_scalar("power", Arity::Exact(2), Arc::new(Pow));
    reg.add_scalar("sqrt", Arity::Exact(1), Arc::new(Sqrt));
    reg.add_scalar("mod", Arity::Exact(2), Arc::new(Mod));

    // lang_mathfunc.html — constants / conversions
    reg.add_scalar("pi", Arity::Exact(0), Arc::new(Pi));
    reg.add_scalar("degrees", Arity::Exact(1), Arc::new(Degrees));
    reg.add_scalar("radians", Arity::Exact(1), Arc::new(Radians));

    // lang_mathfunc.html — trigonometry
    reg.add_scalar("sin", Arity::Exact(1), Arc::new(Sin));
    reg.add_scalar("cos", Arity::Exact(1), Arc::new(Cos));
    reg.add_scalar("tan", Arity::Exact(1), Arc::new(Tan));
    reg.add_scalar("asin", Arity::Exact(1), Arc::new(Asin));
    reg.add_scalar("acos", Arity::Exact(1), Arc::new(Acos));
    reg.add_scalar("atan", Arity::Exact(1), Arc::new(Atan));
    reg.add_scalar("atan2", Arity::Exact(2), Arc::new(Atan2));
    reg.add_scalar("sinh", Arity::Exact(1), Arc::new(Sinh));
    reg.add_scalar("cosh", Arity::Exact(1), Arc::new(Cosh));
    reg.add_scalar("tanh", Arity::Exact(1), Arc::new(Tanh));
    reg.add_scalar("asinh", Arity::Exact(1), Arc::new(Asinh));
    reg.add_scalar("acosh", Arity::Exact(1), Arc::new(Acosh));
    reg.add_scalar("atanh", Arity::Exact(1), Arc::new(Atanh));
}

// ---------------------------------------------------------------------------
// Coercion helpers
// ---------------------------------------------------------------------------

/// A numeric argument after `sqlite3_value_numeric_type` coercion: exactly an
/// integer or a real. Keeping int vs real distinct lets `ceil`/`floor`/`trunc`
/// preserve the storage class the way SQLite does (integer in → integer out).
#[derive(Debug, Clone, Copy)]
enum Numeric {
    Int(i64),
    Real(f64),
}

impl Numeric {
    fn to_f64(self) -> f64 {
        match self {
            Numeric::Int(i) => i as f64,
            Numeric::Real(r) => r,
        }
    }
}

/// `sqlite3_value_numeric_type` restricted to the numeric classes the math family
/// accepts. `None` means the value is NULL, a BLOB, or a TEXT that is not a
/// whole-string numeric literal — every one of which makes a math function (and
/// `sign`) return NULL.
///
/// TEXT is classified by numeric *affinity* (`applyNumericAffinity`, `bTryForInt`
/// off): a whole integer literal that fits `i64` is INTEGER, one that overflows is
/// REAL, and a real-form literal stays REAL — crucially `'5.0'` stays REAL (so
/// `ceil('5.0')` is `5.0`), unlike CAST-to-NUMERIC which would demote it to `5`.
fn numeric_value(v: &Value) -> Option<Numeric> {
    match v {
        Value::Integer(i) => Some(Numeric::Int(*i)),
        Value::Real(r) => Some(Numeric::Real(*r)),
        Value::Text(s) => {
            if looks_like_integer(s) {
                // Whole integer literal: `text_to_numeric` yields INTEGER when it
                // fits i64 and REAL on overflow (the affinity rule for integers).
                match text_to_numeric(s) {
                    Value::Integer(i) => Some(Numeric::Int(i)),
                    Value::Real(r) => Some(Numeric::Real(r)),
                    _ => None,
                }
            } else if looks_like_real(s) {
                // Real-form literal: keep it REAL, do not demote to an integer.
                Some(Numeric::Real(parse_real_prefix(s)))
            } else {
                None
            }
        }
        Value::Null | Value::Blob(_) => None,
    }
}

/// `sqlite3_value_double`: the longest leading numeric prefix of TEXT/BLOB (BLOB
/// bytes read as text), `0.0` when there is no numeric prefix. Used by `abs` and
/// `round`, which coerce with double semantics rather than the stricter
/// numeric-type rule. NULL is handled by callers before this point.
fn value_double(v: &Value) -> f64 {
    match v {
        Value::Null => 0.0,
        Value::Integer(i) => *i as f64,
        Value::Real(r) => *r,
        Value::Text(s) => parse_real_prefix(s),
        Value::Blob(b) => parse_real_prefix(&String::from_utf8_lossy(b)),
    }
}

/// SQLite's `mathFuncFinish`: a NaN result (an unguarded domain error) becomes
/// NULL; any finite value — and a finite overflow to ±infinity — is returned as a
/// REAL. The only poles the docs require to be NULL are the logarithms' (`ln(0)`,
/// `log(_, 0)`, …), which produce -inf; those are rejected by explicit domain
/// checks at the call site, not here. `atanh(±1)` is deliberately NOT such a case —
/// its ±inf poles pass through as REAL, matching sqlite.
fn real_result(x: f64) -> Value {
    if x.is_nan() {
        Value::Null
    } else {
        Value::Real(x)
    }
}

/// Apply a one-argument math function under numeric-type coercion: a
/// non-coercible argument (NULL/blob/non-numeric text) yields NULL.
fn unary(arg: &Value, f: impl Fn(f64) -> Value) -> Value {
    match numeric_value(arg) {
        Some(n) => f(n.to_f64()),
        None => Value::Null,
    }
}

/// Apply a two-argument math function under numeric-type coercion: if *either*
/// argument is non-coercible the result is NULL.
fn binary(a: &Value, b: &Value, f: impl Fn(f64, f64) -> Value) -> Value {
    match (numeric_value(a), numeric_value(b)) {
        (Some(x), Some(y)) => f(x.to_f64(), y.to_f64()),
        _ => Value::Null,
    }
}

// ---------------------------------------------------------------------------
// abs / round / sign (lang_corefunc.html)
// ---------------------------------------------------------------------------

/// `abs(X)` — absolute value. Switches on the *storage* class (like SQLite):
/// INTEGER stays INTEGER (with the `i64::MIN` overflow error), everything else
/// that is not NULL goes through the double path and returns a REAL.
#[derive(Debug)]
struct Abs;
impl ScalarFunction for Abs {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        match &args[0] {
            Value::Null => Ok(Value::Null),
            // abs(-9223372036854775808) has no positive i64, so it is an error
            // rather than a silent wrap (lang_corefunc.html abs).
            Value::Integer(i) => match i.checked_abs() {
                Some(a) => Ok(Value::Integer(a)),
                None => Err(Error::sql("integer overflow")),
            },
            Value::Real(r) => Ok(Value::Real(r.abs())),
            // TEXT/BLOB: abs of the parsed double, always REAL (abs('5')=5.0,
            // abs('abc')=0.0), matching SQLite's default branch.
            other => Ok(Value::Real(value_double(other).abs())),
        }
    }
}

/// `round(X)` / `round(X, Y)` — X rounded to Y decimal digits (default 0),
/// half away from zero, ALWAYS returning a REAL. A NULL X (or NULL Y) yields
/// NULL; Y is clamped to `[0, 30]` (SQLite caps at 30, and negative Y is 0).
#[derive(Debug)]
struct Round;
impl ScalarFunction for Round {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        if args[0].is_null() {
            return Ok(Value::Null);
        }
        let y = match args.get(1) {
            None => 0,
            Some(v) if v.is_null() => return Ok(Value::Null),
            Some(v) => to_integer(v),
        };
        Ok(Value::Real(round_half_away(value_double(&args[0]), y)))
    }
}

/// `2^52`: at or above this magnitude a double is an integer with no fractional
/// part, so rounding to any number of decimal places is a no-op (SQLite's
/// `roundFunc` short-circuits the same boundary).
const NO_FRACTION: f64 = 4_503_599_627_370_496.0; // 2^52

/// Round `r` to `y` decimal places, half away from zero, reproducing SQLite's
/// `round()` — the `n == 0` integer-cast path and, for `n >= 1`, its
/// `%!.*f`-then-`strtod` path — bit-for-bit on the values that matter.
///
/// This does NOT scale-round-unscale (`(r * 10^n).round() / 10^n`): that multiply
/// rounds `r` to a double *before* the decimal rounding, so a value stored just
/// below `x.d5` (e.g. `0.15` = `0.1499999999999999944…`) is inflated to exactly
/// `x.5` and then rounded up — `round(0.15,1)` would give `0.2` where SQLite gives
/// `0.1`. Instead we round the *exact* decimal value: `format!` is correctly
/// rounded to `n` places on the true value of `r` (so the near-tie cases match),
/// and we fix the one remaining discrepancy — `format!` breaks an exact tie to
/// even while SQLite breaks it away from zero — by detecting an exact tie and
/// bumping the magnitude up when the even choice rounded it down.
fn round_half_away(r: f64, y: i64) -> f64 {
    if !r.is_finite() || r.abs() > NO_FRACTION {
        return r;
    }
    let n = y.clamp(0, 30);
    if n == 0 {
        // SQLite's integer-cast path: add/subtract 0.5, then truncate toward zero.
        // Truncation makes any value that rounds to zero come out as +0.0 (so
        // `round(-0.4)` is `0.0`, not `-0.0`), matching SQLite; a value that rounds
        // to a nonzero integer keeps its sign.
        let shifted = if r < 0.0 { r - 0.5 } else { r + 0.5 };
        return (shifted as i64) as f64;
    }
    let neg = r.is_sign_negative();
    let a = r.abs();
    // `format!("{:.n}", a)` is correctly rounded to `n` places on the exact value
    // of `a`, ties to even. That matches SQLite (which rounds the exact value too)
    // everywhere except an exact tie, where SQLite rounds away from zero.
    let mut s = format!("{:.*}", n as usize, a);
    if is_exact_tie(a, n as i32) && s.parse::<f64>().expect("finite decimal") < a {
        // Exact tie that `format!` rounded down to even: bump the magnitude up so
        // the tie goes away from zero, as `%!.*f` does.
        s = bump_last_digit(&s);
    }
    let mag: f64 = s.parse().expect("finite decimal");
    // Reapply the sign, but a value that rounds to zero is always +0.0 (sqlite
    // returns +0.0 for round-to-zero at every precision, not just n == 0 — so
    // `round(-0.001, 2)` is `0.0`, not `-0.0`).
    if neg && mag != 0.0 {
        -mag
    } else {
        mag
    }
}

/// Whether `a` (`>= 0`, finite, `< 2^52`) is an *exact* tie at `n` decimal places —
/// i.e. its true value is exactly halfway between two `n`-place decimals. That
/// happens iff `a == odd / 2^(n+1)`, equivalently `a * 2^(n+1)` is an odd integer.
/// Multiplying by a power of two is lossless, so the test is exact (unlike a
/// `* 10^n` scale). Only these ties distinguish SQLite's round-half-away from
/// Rust's round-half-to-even.
fn is_exact_tie(a: f64, n: i32) -> bool {
    let scaled = a * 2f64.powi(n + 1);
    scaled.is_finite() && scaled.fract() == 0.0 && (scaled % 2.0) == 1.0
}

/// Add one unit to the last decimal digit of the non-negative decimal string `s`
/// (which contains a `.`), propagating the carry and skipping the point. Used to
/// turn a rounded-down tie into its round-up (away-from-zero) neighbor, e.g.
/// `"0.2" -> "0.3"`, `"0.9" -> "1.0"`, `"9.99" -> "10.00"`.
fn bump_last_digit(s: &str) -> String {
    let mut digits = s.as_bytes().to_vec();
    let mut i = digits.len();
    loop {
        if i == 0 {
            digits.insert(0, b'1'); // carry past the most-significant digit
            break;
        }
        i -= 1;
        match digits[i] {
            b'.' => continue,
            b'9' => digits[i] = b'0', // carry into the next digit up
            d => {
                digits[i] = d + 1;
                break;
            }
        }
    }
    String::from_utf8(digits).expect("ascii digits stay valid utf-8")
}

/// `sign(X)` — `-1`, `0`, or `+1` (as INTEGER) for a numeric X; NULL if X is
/// NULL, a BLOB, or a string that is not losslessly a number.
#[derive(Debug)]
struct Sign;
impl ScalarFunction for Sign {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(match numeric_value(&args[0]) {
            None => Value::Null,
            Some(n) => {
                let x = n.to_f64();
                Value::Integer(if x < 0.0 {
                    -1
                } else if x > 0.0 {
                    1
                } else {
                    0
                })
            }
        })
    }
}

// ---------------------------------------------------------------------------
// scalar min / max (lang_corefunc.html, >= 2 args)
// ---------------------------------------------------------------------------

/// Fold the arguments to the extreme one under SQLite's total comparison order.
/// If any argument is NULL the whole result is NULL. On a tie the algorithm keeps
/// the *first* argument for `max` and the *last* for `min` — a faithful
/// replication of SQLite's `minmaxFunc`, which matters only for the returned
/// storage class of equal values (e.g. `max(1, 1.0)` is the INTEGER `1`).
///
/// `collation` is the text collating sequence chosen by the multi-arg min/max rule
/// (lang_corefunc.html): the binder scans the argument expressions left→right for the first
/// that defines a collating function and passes it here (else BINARY). It affects only
/// TEXT-vs-TEXT comparisons; numeric / BLOB / cross-storage-class ordering and the
/// NULL-if-any-NULL and tie rules are collation-independent (all handled by
/// [`compare_values`]).
fn extremum(args: &[Value], want_max: bool, collation: Collation) -> Value {
    if args.iter().any(Value::is_null) {
        return Value::Null;
    }
    let mut best = 0;
    for i in 1..args.len() {
        let ord = compare_values(&args[best], &args[i], collation);
        // max: move to `i` when the current best is strictly less than it.
        // min: move to `i` when the current best is greater-or-equal (>= keeps
        // the later element on a tie, as SQLite does).
        let replace = if want_max { ord.is_lt() } else { !ord.is_lt() };
        if replace {
            best = i;
        }
    }
    args[best].clone()
}

/// `max(X, Y, ...)` — the maximum argument, or NULL if any argument is NULL.
///
/// `collation` is the text collating sequence for the string comparisons, chosen by the
/// multi-arg rule (lang_corefunc.html) and baked in at bind time via
/// [`ScalarFunction::specialize_collation`]; the registry's shared handle defaults to
/// BINARY. See [`extremum`].
#[derive(Debug, Default)]
struct Max {
    collation: Collation,
}
impl ScalarFunction for Max {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(extremum(args, true, self.collation))
    }
    fn specialize_collation(&self, collation: Collation) -> Option<Arc<dyn ScalarFunction>> {
        Some(Arc::new(Max { collation }))
    }
}

/// `min(X, Y, ...)` — the minimum argument, or NULL if any argument is NULL.
///
/// See [`Max`] for the `collation` field.
#[derive(Debug, Default)]
struct Min {
    collation: Collation,
}
impl ScalarFunction for Min {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(extremum(args, false, self.collation))
    }
    fn specialize_collation(&self, collation: Collation) -> Option<Arc<dyn ScalarFunction>> {
        Some(Arc::new(Min { collation }))
    }
}

// ---------------------------------------------------------------------------
// ceil / floor / trunc (lang_mathfunc.html)
// ---------------------------------------------------------------------------

/// Apply a step function that maps a real to an integer-valued real, preserving
/// SQLite's storage-class rule: an INTEGER argument (or whole-integer text) is
/// returned unchanged as an INTEGER; a REAL argument yields a REAL.
fn step(arg: &Value, f: fn(f64) -> f64) -> Value {
    match numeric_value(arg) {
        None => Value::Null,
        Some(Numeric::Int(i)) => Value::Integer(i),
        Some(Numeric::Real(r)) => Value::Real(f(r)),
    }
}

/// `ceil(X)` / `ceiling(X)` — least integer `>= X`.
#[derive(Debug)]
struct Ceil;
impl ScalarFunction for Ceil {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(step(&args[0], f64::ceil))
    }
}

/// `floor(X)` — greatest integer `<= X`.
#[derive(Debug)]
struct Floor;
impl ScalarFunction for Floor {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(step(&args[0], f64::floor))
    }
}

/// `trunc(X)` — the integer part of X (round toward zero).
#[derive(Debug)]
struct Trunc;
impl ScalarFunction for Trunc {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(step(&args[0], f64::trunc))
    }
}

// ---------------------------------------------------------------------------
// exp / logarithms (lang_mathfunc.html)
// ---------------------------------------------------------------------------

/// `exp(X)` — e raised to X.
#[derive(Debug)]
struct Exp;
impl ScalarFunction for Exp {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(unary(&args[0], |x| real_result(x.exp())))
    }
}

/// `ln(X)` — natural logarithm; NULL for X <= 0.
#[derive(Debug)]
struct Ln;
impl ScalarFunction for Ln {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(unary(&args[0], |x| ln_domain(x, |v| v.ln())))
    }
}

/// `log(X)` (base 10) or `log(B, X)` (base B). SQLite's `log` is base-10, and the
/// two-argument form takes the base first, the operand second.
#[derive(Debug)]
struct Log;
impl ScalarFunction for Log {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(match args.len() {
            1 => unary(&args[0], log10),
            // Registered as Range(1,2), so the only other accepted argc is 2.
            _ => log_base(&args[0], &args[1]),
        })
    }
}

/// `log10(X)` — base-10 logarithm; NULL for X <= 0.
#[derive(Debug)]
struct Log10;
impl ScalarFunction for Log10 {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(unary(&args[0], log10))
    }
}

/// `log2(X)` — base-2 logarithm; NULL for X <= 0.
#[derive(Debug)]
struct Log2;
impl ScalarFunction for Log2 {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(unary(&args[0], |x| ln_domain(x, |v| v.ln() / consts::LN_2)))
    }
}

/// Evaluate a logarithm on the natural-log domain, returning NULL for every
/// `x <= 0`. This is the sole domain guard for `ln`/`log10`/`log2` (no caller
/// pre-filters), covering both the `ln(0) = -inf` pole and negative `x` (whose
/// `f(x)` would be NaN); matches SQLite's `logFunc`, which NULLs a non-positive
/// argument rather than letting `-inf` through. The `!(x > 0.0)` form also treats
/// `NaN` as out of domain.
fn ln_domain(x: f64, f: impl Fn(f64) -> f64) -> Value {
    if !(x > 0.0) {
        Value::Null
    } else {
        real_result(f(x))
    }
}

/// base-10 log via `ln(x)/ln(10)`, matching SQLite's `log(x)/M_LN10`.
fn log10(x: f64) -> Value {
    ln_domain(x, |v| v.ln() / consts::LN_10)
}

/// `log(B, X)` = `ln(X) / ln(B)`. SQLite's `logFunc` rejects only an out-of-range
/// base (`B <= 0`) or the divide-by-zero pole (`B == 1`, where `ln(B) == 0`), and
/// an out-of-range operand (`X <= 0`) — NOT `0 < B < 1`, whose `ln(B)` is finite
/// and negative, so e.g. `log(0.5, 8) = -3.0` and `log(0.5, 0.25) = 2.0`. Anything
/// else is NULL. (`!(_ > 0.0)` also rejects a NaN base/operand.)
fn log_base(base: &Value, x: &Value) -> Value {
    match (numeric_value(base), numeric_value(x)) {
        (Some(b), Some(v)) => {
            let bb = b.to_f64();
            let vx = v.to_f64();
            if !(bb > 0.0) || bb == 1.0 || !(vx > 0.0) {
                Value::Null
            } else {
                real_result(vx.ln() / bb.ln())
            }
        }
        _ => Value::Null,
    }
}

// ---------------------------------------------------------------------------
// pow / sqrt / mod (lang_mathfunc.html)
// ---------------------------------------------------------------------------

/// `pow(X, Y)` / `power(X, Y)` — X raised to Y (NULL for a NaN result, e.g. a
/// negative base with a fractional exponent).
#[derive(Debug)]
struct Pow;
impl ScalarFunction for Pow {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(binary(&args[0], &args[1], |x, y| real_result(x.powf(y))))
    }
}

/// `sqrt(X)` — square root; NULL for X < 0.
#[derive(Debug)]
struct Sqrt;
impl ScalarFunction for Sqrt {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(unary(&args[0], |x| {
            if x < 0.0 {
                Value::Null
            } else {
                real_result(x.sqrt())
            }
        }))
    }
}

/// `mod(X, Y)` — floating-point remainder (like `fmod`); NULL when Y == 0 (the
/// remainder is NaN).
#[derive(Debug)]
struct Mod;
impl ScalarFunction for Mod {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(binary(&args[0], &args[1], |x, y| real_result(x % y)))
    }
}

// ---------------------------------------------------------------------------
// pi / degrees / radians (lang_mathfunc.html)
// ---------------------------------------------------------------------------

/// `pi()` — the closest IEEE-754 double to π.
#[derive(Debug)]
struct Pi;
impl ScalarFunction for Pi {
    fn call(&self, _args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(Value::Real(consts::PI))
    }
}

/// `degrees(X)` — radians to degrees.
#[derive(Debug)]
struct Degrees;
impl ScalarFunction for Degrees {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(unary(&args[0], |x| real_result(x.to_degrees())))
    }
}

/// `radians(X)` — degrees to radians.
#[derive(Debug)]
struct Radians;
impl ScalarFunction for Radians {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(unary(&args[0], |x| real_result(x.to_radians())))
    }
}

// ---------------------------------------------------------------------------
// Trigonometry (lang_mathfunc.html)
// ---------------------------------------------------------------------------

/// `sin(X)` — X in radians.
#[derive(Debug)]
struct Sin;
impl ScalarFunction for Sin {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(unary(&args[0], |x| real_result(x.sin())))
    }
}

/// `cos(X)` — X in radians.
#[derive(Debug)]
struct Cos;
impl ScalarFunction for Cos {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(unary(&args[0], |x| real_result(x.cos())))
    }
}

/// `tan(X)` — X in radians.
#[derive(Debug)]
struct Tan;
impl ScalarFunction for Tan {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(unary(&args[0], |x| real_result(x.tan())))
    }
}

/// `asin(X)` — arcsine (radians); NULL for |X| > 1.
#[derive(Debug)]
struct Asin;
impl ScalarFunction for Asin {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(unary(&args[0], |x| {
            if !(-1.0..=1.0).contains(&x) {
                Value::Null
            } else {
                real_result(x.asin())
            }
        }))
    }
}

/// `acos(X)` — arccosine (radians); NULL for |X| > 1.
#[derive(Debug)]
struct Acos;
impl ScalarFunction for Acos {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(unary(&args[0], |x| {
            if !(-1.0..=1.0).contains(&x) {
                Value::Null
            } else {
                real_result(x.acos())
            }
        }))
    }
}

/// `atan(X)` — arctangent (radians).
#[derive(Debug)]
struct Atan;
impl ScalarFunction for Atan {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(unary(&args[0], |x| real_result(x.atan())))
    }
}

/// `atan2(Y, X)` — arctangent of Y/X, placed in the correct quadrant.
#[derive(Debug)]
struct Atan2;
impl ScalarFunction for Atan2 {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(binary(&args[0], &args[1], |y, x| real_result(y.atan2(x))))
    }
}

/// `sinh(X)` — hyperbolic sine.
#[derive(Debug)]
struct Sinh;
impl ScalarFunction for Sinh {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(unary(&args[0], |x| real_result(x.sinh())))
    }
}

/// `cosh(X)` — hyperbolic cosine.
#[derive(Debug)]
struct Cosh;
impl ScalarFunction for Cosh {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(unary(&args[0], |x| real_result(x.cosh())))
    }
}

/// `tanh(X)` — hyperbolic tangent.
#[derive(Debug)]
struct Tanh;
impl ScalarFunction for Tanh {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(unary(&args[0], |x| real_result(x.tanh())))
    }
}

/// `asinh(X)` — hyperbolic arcsine (defined for all reals).
#[derive(Debug)]
struct Asinh;
impl ScalarFunction for Asinh {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(unary(&args[0], |x| real_result(x.asinh())))
    }
}

/// `acosh(X)` — hyperbolic arccosine; NULL for X < 1.
#[derive(Debug)]
struct Acosh;
impl ScalarFunction for Acosh {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(unary(&args[0], |x| {
            if x < 1.0 {
                Value::Null
            } else {
                real_result(x.acosh())
            }
        }))
    }
}

/// `atanh(X)` — hyperbolic arctangent. `|X| > 1` is a domain error (NaN → NULL),
/// but the poles at exactly ±1 are `±Inf` and pass through as REAL, matching real
/// sqlite (they are not "greater than 1.0 or less than -1.0"). Both cases fall out
/// of `real_result` with no explicit guard: `atanh(±1) = ±inf`, `atanh(|x|>1) = NaN`.
#[derive(Debug)]
struct Atanh;
impl ScalarFunction for Atanh {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(unary(&args[0], |x| real_result(x.atanh())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trivial [`FnContext`]; the math functions never consult it.
    struct TestCtx;
    impl FnContext for TestCtx {
        fn now_unix_millis(&self) -> i64 {
            0
        }
        fn random_i64(&mut self) -> i64 {
            0
        }
        fn fill_random(&mut self, _buf: &mut [u8]) {}
        fn last_insert_rowid(&self) -> i64 {
            0
        }
        fn changes(&self) -> i64 {
            0
        }
        fn total_changes(&self) -> i64 {
            0
        }
    }

    fn call(f: &dyn ScalarFunction, args: &[Value]) -> Value {
        let mut ctx = TestCtx;
        f.call(args, &mut ctx).expect("scalar call should succeed")
    }

    fn call_res(f: &dyn ScalarFunction, args: &[Value]) -> Result<Value> {
        let mut ctx = TestCtx;
        f.call(args, &mut ctx)
    }

    fn real(v: &Value) -> f64 {
        match v {
            Value::Real(r) => *r,
            other => panic!("expected Real, got {other:?}"),
        }
    }
    fn int(v: &Value) -> i64 {
        match v {
            Value::Integer(i) => *i,
            other => panic!("expected Integer, got {other:?}"),
        }
    }
    fn is_null(v: &Value) -> bool {
        matches!(v, Value::Null)
    }
    fn approx(v: &Value, want: f64) {
        let got = real(v);
        assert!((got - want).abs() < 1e-9, "expected ~{want}, got {got}");
    }

    // -----------------------------------------------------------------------
    // abs
    // -----------------------------------------------------------------------

    #[test]
    fn abs_preserves_type_and_handles_overflow() {
        assert_eq!(int(&call(&Abs, &[Value::Integer(-5)])), 5);
        assert_eq!(int(&call(&Abs, &[Value::Integer(5)])), 5);
        assert_eq!(real(&call(&Abs, &[Value::Real(-5.0)])), 5.0);
        assert_eq!(real(&call(&Abs, &[Value::Real(5.5)])), 5.5);
        assert!(is_null(&call(&Abs, &[Value::Null])));
        // TEXT/BLOB go through the double path and always return a REAL.
        assert_eq!(real(&call(&Abs, &[Value::Text("-3".into())])), 3.0);
        assert_eq!(real(&call(&Abs, &[Value::Text("5".into())])), 5.0);
        // Non-numeric text/blob -> 0.0 (REAL), never NULL, never an error.
        assert_eq!(real(&call(&Abs, &[Value::Text("abc".into())])), 0.0);
        assert_eq!(real(&call(&Abs, &[Value::Blob(vec![0xff, 0x00])])), 0.0);
        // i64::MIN has no positive counterpart -> integer overflow error.
        match call_res(&Abs, &[Value::Integer(i64::MIN)]) {
            Err(Error::Sql(m)) => assert_eq!(m, "integer overflow"),
            other => panic!("expected integer overflow error, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // round
    // -----------------------------------------------------------------------

    /// `round(X[,Y])` as a free helper for the many round cases below.
    fn round(x: f64, y: i64) -> f64 {
        real(&call(&Round, &[Value::Real(x), Value::Integer(y)]))
    }

    #[test]
    fn round_is_half_away_from_zero_and_always_real() {
        assert_eq!(real(&call(&Round, &[Value::Real(2.5)])), 3.0);
        assert_eq!(real(&call(&Round, &[Value::Real(2.4)])), 2.0);
        assert_eq!(real(&call(&Round, &[Value::Real(-2.5)])), -3.0);
        assert_eq!(round(2.567, 2), 2.57);
        assert_eq!(round(-2.567, 2), -2.57);
        // Integer input still yields a REAL.
        assert_eq!(real(&call(&Round, &[Value::Integer(5)])), 5.0);
        // Negative Y is treated as 0.
        assert_eq!(round(2.567, -1), 3.0);
        // NULL X or NULL Y -> NULL.
        assert!(is_null(&call(&Round, &[Value::Null])));
        assert!(is_null(&call(&Round, &[Value::Real(2.5), Value::Null])));
        // A huge Y is clamped and does not blow up (rounding beyond precision is a no-op).
        assert_eq!(round(1.25, 1_000_000_000), 1.25);
    }

    #[test]
    fn round_result_type_is_real() {
        assert!(matches!(call(&Round, &[Value::Integer(7)]), Value::Real(_)));
        assert!(matches!(call(&Round, &[Value::Real(1.0)]), Value::Real(_)));
        // TEXT X routes through value_double (prefix parse), still REAL.
        assert!(matches!(call(&Round, &[Value::Text("2.567".into()), Value::Integer(2)]), Value::Real(_)));
        assert_eq!(real(&call(&Round, &[Value::Text("2.567".into()), Value::Integer(2)])), 2.57);
    }

    #[test]
    fn round_matches_sqlite_on_near_half_values() {
        // These doubles are stored just BELOW x.d5, so rounding the exact value
        // goes DOWN — a naive (x*10^n).round() inflates them to x.5 and rounds up,
        // diverging from real sqlite's %!.*f (derived from lang_mathfunc.html /
        // lang_corefunc.html round()).
        assert_eq!(round(0.15, 1), 0.1);
        assert_eq!(round(2.05, 1), 2.0);
        assert_eq!(round(1.45, 1), 1.4);
        assert_eq!(round(2.15, 1), 2.1);
        assert_eq!(round(9.45, 1), 9.4);
        assert_eq!(round(1.15, 1), 1.1);
        assert_eq!(round(35.855, 2), 35.85);
    }

    #[test]
    fn round_exact_ties_go_away_from_zero() {
        // Exactly-representable ties (odd / 2^(n+1)) must round away from zero, not
        // to even (std's default), per round()'s half-away contract.
        assert_eq!(round(0.25, 1), 0.3);
        assert_eq!(round(0.75, 1), 0.8);
        assert_eq!(round(1.25, 1), 1.3);
        assert_eq!(round(1.75, 1), 1.8);
        assert_eq!(round(0.125, 2), 0.13);
        assert_eq!(round(0.375, 2), 0.38);
        // Away-from-zero is symmetric about the sign: a negative tie rounds down.
        assert_eq!(round(-0.25, 1), -0.3);
        assert_eq!(round(-1.25, 1), -1.3);
        assert_eq!(round(-0.125, 2), -0.13);
        // n == 0 integer path is also half-away.
        assert_eq!(real(&call(&Round, &[Value::Real(2.5)])), 3.0);
        assert_eq!(real(&call(&Round, &[Value::Real(-2.5)])), -3.0);
        assert_eq!(real(&call(&Round, &[Value::Real(0.5)])), 1.0);
        assert_eq!(real(&call(&Round, &[Value::Real(-0.5)])), -1.0);
    }

    #[test]
    fn round_to_zero_is_always_positive_zero() {
        // A value that rounds to zero is +0.0 at EVERY precision (sqlite returns
        // +0.0 for round-to-zero regardless of n; there is no -0.0 case). The n == 0
        // integer-cast path truncates toward zero, and the n >= 1 path normalizes a
        // zero magnitude before reapplying the sign.
        for &(x, y) in &[(-0.4, 0), (-0.001, 2), (-0.012, 1), (-0.0004, 3), (-0.0, 1)] {
            let v = round(x, y);
            assert_eq!(v, 0.0);
            assert!(v.is_sign_positive(), "round({x}, {y}) must be +0.0, got {v}");
        }
    }

    #[test]
    fn bump_last_digit_carries() {
        // No-carry (the only case the round tie-path reaches, since format!'s
        // round-to-even leaves an even last digit that never bumps to a carry).
        assert_eq!(bump_last_digit("0.2"), "0.3");
        assert_eq!(bump_last_digit("0.38"), "0.39");
        // Carry propagation, including across the point and past the MSD — exercised
        // directly here because the round tie-path can't produce a trailing 9.
        assert_eq!(bump_last_digit("0.9"), "1.0");
        assert_eq!(bump_last_digit("0.29"), "0.30");
        assert_eq!(bump_last_digit("9.99"), "10.00");
    }

    // -----------------------------------------------------------------------
    // sign
    // -----------------------------------------------------------------------

    #[test]
    fn sign_values_and_non_numeric() {
        assert_eq!(int(&call(&Sign, &[Value::Integer(-9)])), -1);
        assert_eq!(int(&call(&Sign, &[Value::Integer(0)])), 0);
        assert_eq!(int(&call(&Sign, &[Value::Integer(9)])), 1);
        assert_eq!(int(&call(&Sign, &[Value::Real(-0.5)])), -1);
        assert_eq!(int(&call(&Sign, &[Value::Real(0.0)])), 0);
        // Numeric text is accepted.
        assert_eq!(int(&call(&Sign, &[Value::Text("2.5".into())])), 1);
        assert_eq!(int(&call(&Sign, &[Value::Text(" -3 ".into())])), -1);
        // NULL, blob, and non-numeric text -> NULL.
        assert!(is_null(&call(&Sign, &[Value::Null])));
        assert!(is_null(&call(&Sign, &[Value::Text("abc".into())])));
        assert!(is_null(&call(&Sign, &[Value::Blob(vec![0x31])])));
        // Partial-numeric text is not "readily converted" -> NULL.
        assert!(is_null(&call(&Sign, &[Value::Text("2abc".into())])));
    }

    // -----------------------------------------------------------------------
    // ceil / floor / trunc
    // -----------------------------------------------------------------------

    #[test]
    fn ceil_floor_trunc_type_and_values() {
        // INTEGER (or whole-integer text) in -> INTEGER out.
        assert_eq!(int(&call(&Ceil, &[Value::Integer(4)])), 4);
        assert_eq!(int(&call(&Floor, &[Value::Integer(4)])), 4);
        assert_eq!(int(&call(&Trunc, &[Value::Integer(4)])), 4);
        assert_eq!(int(&call(&Ceil, &[Value::Text("7".into())])), 7);
        // REAL (or real-form text, including '5.0') in -> REAL out.
        assert_eq!(real(&call(&Ceil, &[Value::Real(4.2)])), 5.0);
        assert_eq!(real(&call(&Floor, &[Value::Real(4.8)])), 4.0);
        assert_eq!(real(&call(&Ceil, &[Value::Real(-4.2)])), -4.0);
        assert_eq!(real(&call(&Floor, &[Value::Real(-4.2)])), -5.0);
        assert_eq!(real(&call(&Trunc, &[Value::Real(4.8)])), 4.0);
        assert_eq!(real(&call(&Trunc, &[Value::Real(-4.8)])), -4.0);
        // '5.0' keeps REAL affinity (not demoted to integer), so REAL out.
        assert_eq!(real(&call(&Ceil, &[Value::Text("5.0".into())])), 5.0);
        // NULL / blob / non-numeric -> NULL.
        assert!(is_null(&call(&Ceil, &[Value::Null])));
        assert!(is_null(&call(&Floor, &[Value::Blob(vec![1])])));
        assert!(is_null(&call(&Trunc, &[Value::Text("x".into())])));
    }

    // -----------------------------------------------------------------------
    // exp / logarithms
    // -----------------------------------------------------------------------

    #[test]
    fn exp_and_natural_log() {
        approx(&call(&Exp, &[Value::Integer(0)]), 1.0);
        approx(&call(&Exp, &[Value::Integer(1)]), consts::E);
        approx(&call(&Ln, &[Value::Real(consts::E)]), 1.0);
        approx(&call(&Ln, &[Value::Integer(1)]), 0.0);
        // Domain: ln(x <= 0) -> NULL (including the ln(0) = -inf pole).
        assert!(is_null(&call(&Ln, &[Value::Integer(0)])));
        assert!(is_null(&call(&Ln, &[Value::Real(-1.0)])));
    }

    #[test]
    fn log_base_variants() {
        // log/log10 is base 10.
        approx(&call(&Log10, &[Value::Integer(1000)]), 3.0);
        approx(&call(&Log, &[Value::Integer(1000)]), 3.0);
        approx(&call(&Log2, &[Value::Integer(8)]), 3.0);
        // Two-argument log(B, X): base first, operand second.
        approx(&call(&Log, &[Value::Integer(2), Value::Integer(8)]), 3.0);
        approx(&call(&Log, &[Value::Integer(10), Value::Integer(100)]), 2.0);
        // A fractional base 0 < B < 1 is valid (ln(B) is finite and negative), not
        // a domain error: log(0.5, 8) = -3, log(0.5, 0.25) = 2 (matches sqlite).
        approx(&call(&Log, &[Value::Real(0.5), Value::Integer(8)]), -3.0);
        approx(&call(&Log, &[Value::Real(0.5), Value::Real(0.25)]), 2.0);
        // Domain: x <= 0 -> NULL for the single-arg forms.
        assert!(is_null(&call(&Log10, &[Value::Integer(0)])));
        assert!(is_null(&call(&Log2, &[Value::Real(-2.0)])));
        assert!(is_null(&call(&Log, &[Value::Integer(0)])));
        // Only base <= 0, base == 1 (divide-by-zero), or operand <= 0 -> NULL.
        assert!(is_null(&call(&Log, &[Value::Integer(1), Value::Integer(8)])));
        assert!(is_null(&call(&Log, &[Value::Integer(0), Value::Integer(8)])));
        assert!(is_null(&call(&Log, &[Value::Real(-2.0), Value::Integer(8)])));
        assert!(is_null(&call(&Log, &[Value::Integer(2), Value::Integer(0)])));
    }

    // -----------------------------------------------------------------------
    // pow / sqrt / mod
    // -----------------------------------------------------------------------

    #[test]
    fn pow_sqrt_mod() {
        approx(&call(&Pow, &[Value::Integer(2), Value::Integer(10)]), 1024.0);
        approx(&call(&Pow, &[Value::Integer(2), Value::Real(0.5)]), consts::SQRT_2);
        // Negative base with a fractional exponent is NaN -> NULL.
        assert!(is_null(&call(&Pow, &[Value::Integer(-2), Value::Real(0.5)])));
        approx(&call(&Sqrt, &[Value::Integer(9)]), 3.0);
        approx(&call(&Sqrt, &[Value::Integer(0)]), 0.0);
        assert!(is_null(&call(&Sqrt, &[Value::Integer(-1)])));
        approx(&call(&Mod, &[Value::Integer(10), Value::Integer(3)]), 1.0);
        approx(&call(&Mod, &[Value::Real(5.5), Value::Integer(2)]), 1.5);
        approx(&call(&Mod, &[Value::Integer(-10), Value::Integer(3)]), -1.0);
        // Division by zero -> NULL.
        assert!(is_null(&call(&Mod, &[Value::Integer(10), Value::Integer(0)])));
    }

    // -----------------------------------------------------------------------
    // pi / degrees / radians / trig
    // -----------------------------------------------------------------------

    #[test]
    fn pi_constant() {
        assert_eq!(real(&call(&Pi, &[])), std::f64::consts::PI);
        assert_eq!(real(&call(&Pi, &[])), 3.141592653589793);
    }

    #[test]
    fn degrees_radians_round_trip() {
        approx(&call(&Degrees, &[Value::Real(consts::PI)]), 180.0);
        approx(&call(&Radians, &[Value::Integer(180)]), consts::PI);
    }

    /// Each trig/hyperbolic function is pinned at an argument whose value is
    /// DISTINCT from every plausible sibling substitution (cos↔cosh, tan↔tanh,
    /// atan↔atanh, radians-vs-degrees, …), so a wrong-libm-call among the ~26
    /// near-identical wrappers is caught rather than passing on a coincidental
    /// shared point (all sibling values at 0 coincide). Reference values are the
    /// exact libm results; `approx` compares within 1e-9.
    #[test]
    fn trig_family_values_are_distinct_from_siblings() {
        // Circular: cos(π/3)=0.5 (cosh(π/3)≈1.600); tan(π/4)=1.0 (sin≈0.707, tanh≈0.656).
        approx(&call(&Sin, &[Value::Real(consts::FRAC_PI_2)]), 1.0);
        approx(&call(&Cos, &[Value::Real(consts::FRAC_PI_3)]), 0.5);
        approx(&call(&Tan, &[Value::Real(consts::FRAC_PI_4)]), 1.0);
        // Inverse circular: asin(1)=π/2; acos(0.5)=π/3 (acosh out of domain there);
        // atan(1)=π/4=0.785 (tan(1)≈1.557).
        approx(&call(&Asin, &[Value::Integer(1)]), consts::FRAC_PI_2);
        approx(&call(&Acos, &[Value::Real(0.5)]), consts::FRAC_PI_3);
        approx(&call(&Atan, &[Value::Integer(1)]), consts::FRAC_PI_4);
        // atan2 argument order (Y, X): atan2(1,0)=π/2 and atan2(0,1)=0 pin it, so a
        // y.atan2(x) ↔ x.atan2(y) swap is caught (the symmetric (1,1) point cannot).
        approx(&call(&Atan2, &[Value::Integer(1), Value::Integer(0)]), consts::FRAC_PI_2);
        approx(&call(&Atan2, &[Value::Integer(0), Value::Integer(1)]), 0.0);
        // Hyperbolic at 1, where sinh/cosh/tanh/asinh are all distinct.
        approx(&call(&Sinh, &[Value::Integer(1)]), 1.175_201_193_643_801_4);
        approx(&call(&Cosh, &[Value::Integer(1)]), 1.543_080_634_815_243_7);
        approx(&call(&Tanh, &[Value::Integer(1)]), 0.761_594_155_955_764_9);
        approx(&call(&Asinh, &[Value::Integer(1)]), 0.881_373_587_019_543_0);
        // Inverse hyperbolic: acosh(2)=1.317; atanh(0.5)=0.549 (atan(0.5)≈0.464, tanh(0.5)≈0.462).
        approx(&call(&Acosh, &[Value::Integer(2)]), 1.316_957_896_924_816_6);
        approx(&call(&Atanh, &[Value::Real(0.5)]), 0.549_306_144_334_054_8);
    }

    #[test]
    fn trig_out_of_domain_is_null() {
        assert!(is_null(&call(&Asin, &[Value::Real(1.5)])));
        assert!(is_null(&call(&Asin, &[Value::Real(-1.5)])));
        assert!(is_null(&call(&Acos, &[Value::Real(2.0)])));
        assert!(is_null(&call(&Acosh, &[Value::Real(0.5)])));
        // atanh: only |x| > 1 is a domain error (NaN -> NULL).
        assert!(is_null(&call(&Atanh, &[Value::Real(2.0)])));
        assert!(is_null(&call(&Atanh, &[Value::Real(-2.0)])));
    }

    #[test]
    fn atanh_poles_are_infinite_not_null() {
        // atanh(±1) are the ±inf poles, NOT domain errors under lang_mathfunc.html
        // §1's strict "greater than 1.0 or less than -1.0" wording, so they pass
        // through as ±Inf (mathFuncFinish only nulls NaN) — consistent with the
        // finite-overflow pass-through policy (see real_result / exp below).
        let pos = real(&call(&Atanh, &[Value::Real(1.0)]));
        assert!(pos.is_infinite() && pos > 0.0, "atanh(1.0) must be +Inf, got {pos}");
        let neg = real(&call(&Atanh, &[Value::Real(-1.0)]));
        assert!(neg.is_infinite() && neg < 0.0, "atanh(-1.0) must be -Inf, got {neg}");
    }

    #[test]
    fn finite_overflow_passes_through_as_infinite_real() {
        // Committed policy: real_result maps NaN -> NULL but lets an in-domain
        // finite overflow to ±inf through as REAL (matches sqlite's mathFuncFinish,
        // which only rejects NaN). e.g. exp(1000) = +Inf, cosh(1000) = +Inf.
        let e = real(&call(&Exp, &[Value::Integer(1000)]));
        assert!(e.is_infinite() && e > 0.0, "exp(1000) must be +Inf, got {e}");
        let c = real(&call(&Cosh, &[Value::Integer(1000)]));
        assert!(c.is_infinite() && c > 0.0, "cosh(1000) must be +Inf, got {c}");
        // pow overflow likewise; a NaN result (e.g. pow(-2, 0.5)) stays NULL.
        let p = real(&call(&Pow, &[Value::Real(10.0), Value::Integer(400)]));
        assert!(p.is_infinite() && p > 0.0, "pow(10, 400) must be +Inf, got {p}");
    }

    // -----------------------------------------------------------------------
    // scalar min / max
    // -----------------------------------------------------------------------

    #[test]
    fn scalar_min_max_comparison_order_across_classes() {
        // Numeric ordering.
        assert_eq!(int(&call(&Max::default(), &[Value::Integer(1), Value::Integer(2), Value::Integer(3)])), 3);
        assert_eq!(int(&call(&Min::default(), &[Value::Integer(3), Value::Integer(1), Value::Integer(2)])), 1);
        // Storage-class order: NULL < numeric < text < blob.
        let mixed = [Value::Integer(1), Value::Real(2.5), Value::Text("a".into()), Value::Blob(vec![0])];
        assert!(matches!(call(&Max::default(), &mixed), Value::Blob(_)));
        assert_eq!(int(&call(&Min::default(), &mixed)), 1);
        // A number is less than text, so max picks the text.
        assert!(matches!(call(&Max::default(), &[Value::Integer(9), Value::Text("a".into())]), Value::Text(_)));
        assert_eq!(int(&call(&Min::default(), &[Value::Integer(9), Value::Text("a".into())])), 9);
    }

    #[test]
    fn scalar_min_max_null_propagates() {
        assert!(is_null(&call(&Min::default(), &[Value::Integer(1), Value::Null, Value::Integer(2)])));
        assert!(is_null(&call(&Max::default(), &[Value::Integer(1), Value::Null])));
        assert!(is_null(&call(&Max::default(), &[Value::Null, Value::Null])));
    }

    #[test]
    fn scalar_min_max_tie_keeps_sqlite_storage_class() {
        // On an equal-value tie SQLite's max keeps the first argument and min the
        // last, which is observable through the returned storage class.
        assert!(matches!(call(&Max::default(), &[Value::Integer(1), Value::Real(1.0)]), Value::Integer(1)));
        assert!(matches!(call(&Min::default(), &[Value::Integer(1), Value::Real(1.0)]), Value::Real(_)));
    }

    #[test]
    fn scalar_min_max_fold_text_under_collation() {
        // The multi-arg rule: string comparisons use the chosen collation. Under NOCASE,
        // 'B' > 'a' (case-folded), so max('B','a')='B' and min='a'; the BINARY handle
        // orders bytewise ('B'=0x42 < 'a'=0x61) so max='a', min='B'. Same values, opposite
        // extremes — the discriminator that a wrong (BINARY-always) handle fails.
        let nocase_max = Max { collation: Collation::NoCase };
        let nocase_min = Min { collation: Collation::NoCase };
        let ba = [Value::Text("B".into()), Value::Text("a".into())];
        assert!(matches!(call(&nocase_max, &ba), Value::Text(t) if t == "B"));
        assert!(matches!(call(&nocase_min, &ba), Value::Text(t) if t == "a"));
        assert!(matches!(call(&Max::default(), &ba), Value::Text(t) if t == "a"));
        assert!(matches!(call(&Min::default(), &ba), Value::Text(t) if t == "B"));
        // specialize_collation bakes the collation into the handle the binder stores.
        let specialized = Max::default().specialize_collation(Collation::NoCase).expect("min/max specialize");
        assert!(matches!(call(specialized.as_ref(), &ba), Value::Text(t) if t == "B"));
        // A collation-insensitive scalar does not specialize (keeps its shared handle).
        assert!(Abs.specialize_collation(Collation::NoCase).is_none());
    }

    /// The standing enumeration guard over the collation-SENSITIVE scalar set — the analogue
    /// of `scalar/misc.rs`'s `deterministic()` guard, and the test the `ScalarFunction::
    /// specialize_collation` doc points at. `specialize_collation` defaults to `None` (a
    /// BINARY-defaulting opt-in), so like `deterministic()` it is a hand-maintained
    /// convention: the multi-arg `min`/`max` MUST override it (else the binder cannot bake in
    /// the argument collation and they silently compare under BINARY — lang_corefunc.html
    /// max_scalar/min_scalar), and a representative sample of collation-INSENSITIVE scalars
    /// MUST keep the default. If the min/max wiring regresses (override removed) or a new
    /// pure scalar wrongly overrides, this fails loudly.
    #[test]
    fn collation_sensitive_scalars_specialize_and_others_do_not() {
        // The complete collation-sensitive scalar set today: multi-arg min/max.
        let sensitive: &[&dyn ScalarFunction] = &[&Max::default(), &Min::default()];
        for f in sensitive {
            assert!(
                f.specialize_collation(Collation::NoCase).is_some(),
                "{f:?} must specialize on collation"
            );
        }
        // Representative collation-insensitive scalars keep the trait default (`None`).
        let insensitive: &[&dyn ScalarFunction] = &[&Abs, &Sign, &Round, &Ceil, &Sqrt];
        for f in insensitive {
            assert!(
                f.specialize_collation(Collation::NoCase).is_none(),
                "{f:?} must not specialize (collation-insensitive)"
            );
        }
    }

    // -----------------------------------------------------------------------
    // NULL propagation across the whole family, via the real registry
    // -----------------------------------------------------------------------

    #[test]
    fn null_argument_yields_null_for_every_function() {
        let reg = FunctionRegistry::builtins();
        // (name, argc): a NULL in any position makes each of these return NULL.
        let cases: &[(&str, usize)] = &[
            ("abs", 1),
            ("round", 1),
            ("round", 2),
            ("sign", 1),
            ("ceil", 1),
            ("ceiling", 1),
            ("floor", 1),
            ("trunc", 1),
            ("exp", 1),
            ("ln", 1),
            ("log", 1),
            ("log", 2),
            ("log10", 1),
            ("log2", 1),
            ("pow", 2),
            ("power", 2),
            ("sqrt", 1),
            ("mod", 2),
            ("degrees", 1),
            ("radians", 1),
            ("sin", 1),
            ("cos", 1),
            ("tan", 1),
            ("asin", 1),
            ("acos", 1),
            ("atan", 1),
            ("atan2", 2),
            ("sinh", 1),
            ("cosh", 1),
            ("tanh", 1),
            ("asinh", 1),
            ("acosh", 1),
            ("atanh", 1),
            ("min", 2),
            ("max", 2),
        ];
        for &(name, argc) in cases {
            let f = reg.resolve_scalar(name, argc).unwrap_or_else(|e| panic!("{name}/{argc}: {e:?}"));
            let args = vec![Value::Null; argc];
            assert!(
                is_null(&call(f.as_ref(), &args)),
                "{name}/{argc} with NULL args should be NULL"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Registration spot-check
    // -----------------------------------------------------------------------

    #[test]
    fn registered_names_resolve_at_their_arities() {
        let reg = FunctionRegistry::builtins();
        let exact: &[(&str, usize)] = &[
            ("abs", 1),
            ("sign", 1),
            ("ceil", 1),
            ("ceiling", 1),
            ("floor", 1),
            ("trunc", 1),
            ("exp", 1),
            ("ln", 1),
            ("log10", 1),
            ("log2", 1),
            ("pow", 2),
            ("power", 2),
            ("sqrt", 1),
            ("mod", 2),
            ("pi", 0),
            ("degrees", 1),
            ("radians", 1),
            ("sin", 1),
            ("cos", 1),
            ("tan", 1),
            ("asin", 1),
            ("acos", 1),
            ("atan", 1),
            ("atan2", 2),
            ("sinh", 1),
            ("cosh", 1),
            ("tanh", 1),
            ("asinh", 1),
            ("acosh", 1),
            ("atanh", 1),
        ];
        for &(name, argc) in exact {
            assert!(reg.resolve_scalar(name, argc).is_ok(), "{name}/{argc} should resolve");
        }
        // round and log accept 1 or 2 arguments.
        assert!(reg.resolve_scalar("round", 1).is_ok());
        assert!(reg.resolve_scalar("round", 2).is_ok());
        assert!(reg.resolve_scalar("log", 1).is_ok());
        assert!(reg.resolve_scalar("log", 2).is_ok());
        // Case-insensitive.
        assert!(reg.resolve_scalar("SQRT", 1).is_ok());
        // Scalar min/max resolve at argc >= 2 (1-arg is the aggregate, elsewhere).
        for argc in [2usize, 3, 5] {
            assert!(reg.resolve_scalar("min", argc).is_ok(), "min/{argc} should resolve");
            assert!(reg.resolve_scalar("max", argc).is_ok(), "max/{argc} should resolve");
        }
        // pi() takes no arguments; a 1-arg call is a wrong-arity error.
        assert!(reg.resolve_scalar("pi", 0).is_ok());
        assert!(reg.resolve_scalar("pi", 1).is_err());
    }
}
