//! Text <-> number rules shared by column affinity and CAST.
//!
//! Two families live here and they are deliberately different, matching SQLite:
//!
//! * **Affinity predicates** ([`looks_like_integer`], [`looks_like_real`]) ask
//!   whether the *whole* trimmed string is a well-formed integer / real literal.
//!   Column affinity only converts text when the entire value is a clean literal.
//! * **CAST-style prefix parsers** ([`parse_int_prefix`], [`parse_real_prefix`],
//!   [`text_to_numeric`]) take the *longest leading prefix* that parses and ignore
//!   the rest, per lang_expr.html §13.
//!
//! Neither family recognizes hexadecimal integers or the `_` digit separator:
//! those are understood only by the SQL tokenizer (lang_expr.html §3, "the 0x
//! notation is only understood by the SQL language parser, not by the type
//! conversion routines"), which is `minisqlite-sql`'s concern, not this one.
//! Leading and trailing whitespace is tolerated (SQLite's `sqlite3AtoF` skips
//! both), using SQLite's whitespace set (space, tab, LF, VT, FF, CR).

use crate::value::Value;

/// SQLite's `sqlite3Isspace`: ASCII space, tab, LF, vertical-tab, form-feed, CR.
/// Note this includes `0x0b` (vertical tab), which Rust's `is_ascii_whitespace`
/// excludes — matching SQLite here avoids an off-by-one-class affinity mismatch.
#[inline]
fn is_sqlite_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

/// Trim leading and trailing SQLite whitespace. All such bytes are ASCII, so the
/// slice stays on UTF-8 boundaries.
fn trim_ws(s: &str) -> &str {
    let b = s.as_bytes();
    let mut start = 0;
    let mut end = b.len();
    while start < end && is_sqlite_space(b[start]) {
        start += 1;
    }
    while end > start && is_sqlite_space(b[end - 1]) {
        end -= 1;
    }
    &s[start..end]
}

/// `2^51`, the exact-integer margin CAST-to-NUMERIC uses to decide whether a
/// float-form text describes an integer (one bit below the 52-bit f64 mantissa,
/// per lang_expr.html §13 NUMERIC).
const NUMERIC_INT_MARGIN: f64 = 2_251_799_813_685_248.0; // 2^51

/// `2^63`: one past `i64::MAX`. `i64::MAX as f64` rounds up to this, so it is the
/// exclusive upper bound for "this f64 fits in an i64". Shared with `compare` so the
/// i64/f64 boundary is defined once (a mistyped copy would silently disagree).
pub(crate) const TWO_POW_63: f64 = 9_223_372_036_854_775_808.0;

// ---------------------------------------------------------------------------
// Whole-string literal predicates (column affinity)
// ---------------------------------------------------------------------------

/// True if `s` (after trimming whitespace) is a well-formed **decimal integer
/// literal**: an optional sign followed by one or more ASCII digits and nothing
/// else. No decimal point, no exponent, no hex, no `_` separators.
pub fn looks_like_integer(s: &str) -> bool {
    is_integer_literal(trim_ws(s))
}

/// True if `s` (after trimming whitespace) is a well-formed **floating-point
/// literal**: it parses fully and has a decimal point and/or an exponent (a pure
/// integer is *not* a real literal — SQLite classifies by the presence of `.`/`e`,
/// lang_expr.html §3).
pub fn looks_like_real(s: &str) -> bool {
    is_real_literal(trim_ws(s))
}

/// Whole-slice integer-literal check (slice already trimmed).
fn is_integer_literal(t: &str) -> bool {
    let b = t.as_bytes();
    let n = b.len();
    let mut i = 0;
    if i < n && (b[i] == b'+' || b[i] == b'-') {
        i += 1;
    }
    let first_digit = i;
    while i < n && b[i].is_ascii_digit() {
        i += 1;
    }
    i == n && i > first_digit
}

/// Whole-slice real-literal check (slice already trimmed). Requires a `.` or an
/// exponent so that integers fall to [`is_integer_literal`] instead.
fn is_real_literal(t: &str) -> bool {
    let b = t.as_bytes();
    let n = b.len();
    let mut i = 0;
    if i < n && (b[i] == b'+' || b[i] == b'-') {
        i += 1;
    }
    let mut digits = 0;
    let mut has_dot = false;
    let mut has_exp = false;
    while i < n && b[i].is_ascii_digit() {
        i += 1;
        digits += 1;
    }
    if i < n && b[i] == b'.' {
        has_dot = true;
        i += 1;
        while i < n && b[i].is_ascii_digit() {
            i += 1;
            digits += 1;
        }
    }
    if digits == 0 {
        return false; // no mantissa digits ("." or "+.e5" etc.)
    }
    if i < n && (b[i] == b'e' || b[i] == b'E') {
        has_exp = true;
        i += 1;
        if i < n && (b[i] == b'+' || b[i] == b'-') {
            i += 1;
        }
        let exp_start = i;
        while i < n && b[i].is_ascii_digit() {
            i += 1;
        }
        if i == exp_start {
            return false; // "1e" with no exponent digits
        }
    }
    i == n && (has_dot || has_exp)
}

/// Parse a fully-validated integer literal (optional sign + digits) exactly, or
/// `None` on i64 overflow. `i64::MIN` (`-9223372036854775808`) parses exactly.
fn parse_exact_i64(t: &str) -> Option<i64> {
    // `i64::from_str` rejects a leading '+', which SQLite accepts, so strip it.
    let t = t.strip_prefix('+').unwrap_or(t);
    t.parse::<i64>().ok()
}

/// If `r` is finite, has no fractional part, and fits exactly in an `i64`, return
/// that integer. This is column-affinity's "float that can be represented exactly
/// as an integer is converted to an integer" rule (datatype3.html §3).
pub fn real_to_int_if_exact(r: f64) -> Option<i64> {
    if r.is_finite() && r.fract() == 0.0 && r >= -TWO_POW_63 && r < TWO_POW_63 {
        let i = r as i64;
        // Guard the top of the range where `as i64` could disagree with the value.
        if i as f64 == r {
            return Some(i);
        }
    }
    None
}

/// Column-affinity conversion of a TEXT value under NUMERIC / INTEGER affinity:
/// returns `Some(Integer|Real)` when the whole text is a numeric literal, else
/// `None` (leave it TEXT). See datatype3.html §3:
/// * integer literal that fits i64 -> INTEGER, too large -> REAL;
/// * real literal -> INTEGER if the value is exactly integral, else REAL.
pub(crate) fn numeric_affinity_text(s: &str) -> Option<Value> {
    let t = trim_ws(s);
    if is_integer_literal(t) {
        match parse_exact_i64(t) {
            Some(i) => Some(Value::Integer(i)),
            // Well-formed but overflows i64: datatype3 says store as REAL.
            None => Some(Value::Real(scan_real(t).0)),
        }
    } else if is_real_literal(t) {
        let r = scan_real(t).0;
        Some(real_to_int_if_exact(r).map(Value::Integer).unwrap_or(Value::Real(r)))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// CAST-style prefix parsers
// ---------------------------------------------------------------------------

/// CAST-to-REAL of text (lang_expr.html §13): the longest leading prefix that
/// parses as a real number; leading whitespace ignored; `0.0` if there is no such
/// prefix. Overflowing magnitudes yield ±inf (as SQLite does).
pub fn parse_real_prefix(s: &str) -> f64 {
    scan_real(s).0
}

/// CAST-to-INTEGER of text (lang_expr.html §13): the longest leading integer
/// prefix (stops at `.`, `e`, `x`, or any non-digit), clamped to the i64 range;
/// `0` if there is no such prefix. The exponent of a float-looking string is *not*
/// part of the integer prefix, so `"123e5"` casts to `123`.
pub fn parse_int_prefix(s: &str) -> i64 {
    scan_int(s).0
}

/// The kind of the longest leading numeric prefix of a TEXT/BLOB value, for the
/// operator-level implicit coercions (lang_expr.html §13, which the boolean rule
/// §14 defers to). Unlike the whole-string affinity predicates
/// ([`looks_like_integer`]/[`looks_like_real`]), only a *leading* run must parse:
/// `"2abc"` has an integer prefix, `"2.5x"` a real prefix, while `"abc"`, `""`,
/// `"-"`, and `"."` have no numeric prefix at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericPrefix {
    /// No numeric prefix: the value coerces to integer 0.
    None,
    /// An integer-form prefix — an optional sign then digits, with no `.` or
    /// exponent consumed (`"2abc"`, `"-3x"`, and `"1e"` where the `e` has no
    /// following exponent digits). Take the value with [`parse_int_prefix`].
    Integer,
    /// A real-form prefix — a `.` or an exponent was part of the consumed number
    /// (`"2.5x"`, `"2e3x"`, `"1."`). Take the value with [`parse_real_prefix`].
    Real,
}

/// Classify the longest leading numeric prefix of `s` (see [`NumericPrefix`]).
///
/// Reuses [`scan_real`]'s scan so the classification and the value extracted by
/// [`parse_real_prefix`] agree on exactly where the number ends. A returned
/// [`NumericPrefix::Integer`] always covers precisely the run [`parse_int_prefix`]
/// consumes: an integer-form scan has no `.`/exponent, and both scanners take the
/// same optional-sign-then-digits run.
pub fn numeric_prefix(s: &str) -> NumericPrefix {
    let (_value, consumed, is_float) = scan_real(s);
    if consumed == 0 {
        NumericPrefix::None
    } else if is_float {
        NumericPrefix::Real
    } else {
        NumericPrefix::Integer
    }
}

/// CAST-to-NUMERIC of TEXT/BLOB (lang_expr.html §13): yields INTEGER or REAL.
/// Integer-form text -> INTEGER if it fits i64, else REAL. Float-form text ->
/// INTEGER iff the value is integral and within the 51-bit lossless margin, else
/// REAL. Unparseable text -> INTEGER 0. (Casting a REAL/INTEGER value to NUMERIC
/// is a no-op and is handled in `cast`, not here.)
pub fn text_to_numeric(s: &str) -> Value {
    let (r, consumed, is_float) = scan_real(s);
    if consumed == 0 {
        return Value::Integer(0);
    }
    if is_float {
        if r.is_finite() && r.fract() == 0.0 && r.abs() <= NUMERIC_INT_MARGIN {
            Value::Integer(r as i64)
        } else {
            Value::Real(r)
        }
    } else {
        let (i, _c, overflow) = scan_int(s);
        if overflow {
            Value::Real(r)
        } else {
            Value::Integer(i)
        }
    }
}

/// CAST-to-INTEGER of a REAL (lang_expr.html §13): truncate toward zero and clamp
/// to the i64 range. A value at or past `+2^63` clamps to `i64::MAX`; at or past
/// `-2^63` clamps to `i64::MIN`.
pub fn real_to_int_trunc(r: f64) -> i64 {
    if r.is_nan() {
        return 0;
    }
    let t = r.trunc();
    if t >= TWO_POW_63 {
        i64::MAX
    } else if t <= -TWO_POW_63 {
        i64::MIN
    } else {
        t as i64
    }
}

/// Scan the longest real-number prefix. Returns `(value, bytes_consumed,
/// is_float_form)` where `is_float_form` is true iff a `.` or exponent was part of
/// the consumed number. `bytes_consumed == 0` means no numeric prefix was found.
///
/// The value is parsed by slicing the scanned bytes in place and handing that slice
/// to Rust's correctly-rounded `f64::from_str` (no intermediate `String`), so the
/// rounding matches a real `strtod` rather than an ad-hoc mantissa/exponent multiply.
fn scan_real(s: &str) -> (f64, usize, bool) {
    let b = s.as_bytes();
    let n = b.len();
    let mut i = 0;
    while i < n && is_sqlite_space(b[i]) {
        i += 1;
    }
    // Remember where the number itself starts (at the optional sign) so we can hand
    // the exact byte slice to `f64::from_str` with zero intermediate allocation —
    // this runs per row on TEXT->numeric coercion and CAST, so avoid String churn.
    let num_start = i;
    if i < n && (b[i] == b'+' || b[i] == b'-') {
        i += 1;
    }
    let mut saw_digit = false;
    while i < n && b[i].is_ascii_digit() {
        saw_digit = true;
        i += 1;
    }
    let mut is_float = false;
    if i < n && b[i] == b'.' {
        is_float = true;
        i += 1;
        while i < n && b[i].is_ascii_digit() {
            saw_digit = true;
            i += 1;
        }
    }
    if !saw_digit {
        return (0.0, 0, false); // no mantissa digits: not a number
    }
    if i < n && (b[i] == b'e' || b[i] == b'E') {
        let mut k = i + 1;
        if k < n && (b[k] == b'+' || b[k] == b'-') {
            k += 1;
        }
        let exp_start = k;
        while k < n && b[k].is_ascii_digit() {
            k += 1;
        }
        if k > exp_start {
            is_float = true;
            i = k; // include the exponent only when it has digits
        }
    }
    // `s[num_start..i]` is a canonical f64 literal: an optional sign, decimal digits,
    // an optional '.', and an optional exponent. Rust's correctly-rounded
    // `f64::from_str` accepts every form the scan can emit (incl. "1.", ".5",
    // "1.e3", "+.5e2"), so the parse is total and matches a real `strtod` rounding.
    let value = s[num_start..i]
        .parse::<f64>()
        .expect("scanned numeric prefix is always a valid f64 literal");
    (value, i, is_float)
}

/// Scan the longest integer prefix. Returns `(clamped_value, bytes_consumed,
/// overflowed)`. `bytes_consumed == 0` means no digits were found.
fn scan_int(s: &str) -> (i64, usize, bool) {
    let b = s.as_bytes();
    let n = b.len();
    let mut i = 0;
    while i < n && is_sqlite_space(b[i]) {
        i += 1;
    }
    let neg = i < n && b[i] == b'-';
    if i < n && (b[i] == b'+' || b[i] == b'-') {
        i += 1;
    }
    let digit_start = i;
    // Accumulate magnitude in u64 so `i64::MIN`'s magnitude (2^63) is exact.
    let mut mag: u64 = 0;
    let mut overflow = false;
    while i < n && b[i].is_ascii_digit() {
        let d = (b[i] - b'0') as u64;
        match mag.checked_mul(10).and_then(|m| m.checked_add(d)) {
            Some(m) => mag = m,
            None => overflow = true,
        }
        i += 1;
    }
    if i == digit_start {
        return (0, 0, false);
    }
    let limit_pos = i64::MAX as u64; // 9223372036854775807
    let limit_neg = (i64::MAX as u64) + 1; // 9223372036854775808 == |i64::MIN|
    if neg {
        // `overflow` here means the u64 accumulation itself wrapped (value far out
        // of range); either way clamp to i64::MIN. `limit_neg` is the exact MIN.
        if overflow || mag > limit_neg {
            (i64::MIN, i, true)
        } else if mag == limit_neg {
            (i64::MIN, i, false)
        } else {
            (-(mag as i64), i, false)
        }
    } else if overflow || mag > limit_pos {
        (i64::MAX, i, true)
    } else {
        (mag as i64, i, false)
    }
}

// ---------------------------------------------------------------------------
// Number -> text
// ---------------------------------------------------------------------------

/// Render an integer as SQLite renders it in text form: plain decimal.
pub fn integer_to_text(i: i64) -> String {
    i.to_string()
}

/// Render a real as SQLite renders it in text form (CAST-to-TEXT / TEXT affinity).
///
/// Digit selection follows SQLite's rounding rule from `floatingpoint.html`
/// §1.3.1 (the default since 3.52.0, 2026-03-06): round to 15 significant digits;
/// if that decimal does not parse back to the *exact* same binary-64 value, round
/// to 17 instead; strip trailing zeros. This is deliberately NOT Rust's
/// shortest-round-trip (`{:e}`), which can emit a 16-digit form that SQLite never
/// produces (e.g. `1.0/3.0` -> `0.33333333333333331`, not `0.3333333333333333`).
///
/// Layout is `%g`-style: fixed notation when the decimal exponent is in `-4..15`,
/// otherwise scientific (`d.ddde±NN`, exponent signed and zero-padded to two
/// digits). A real always shows a decimal point or exponent so it reads back as a
/// real, never an integer.
///
/// KNOWN RESIDUAL DIVERGENCE (a known limitation, not a spec error): the
/// fall-to-17 step uses Rust's correctly-rounded (round-half-to-even)
/// 17-digit form. Real SQLite's dtoa is NOT plain correctly-rounded-to-17, so its
/// last digit can differ on the narrow class where 15 digits fail to round-trip.
/// Witness: `real_to_text(1.23 * 2.34)` yields `"2.8781999999999996"`, whereas
/// `floatingpoint.html` §1.3 documents real SQLite rendering that same f64
/// (`6481130223748880 * 2^-51`) as `"2.8781999999999997"` — both round-trip, but
/// they differ. This is a leaf concern: we match SQLite on the common 17-digit
/// cases (1/3, 2/3, 0.1+0.2 are verified exact); faithfully reproducing SQLite's
/// integer-arithmetic dtoa last-digit choice is a separate sub-project and cannot
/// be derived from the docs alone. Re-check here first when a float→text
/// mismatch surfaces.
pub fn real_to_text(r: f64) -> String {
    if r.is_nan() {
        // A stored REAL is never NaN (SQLite turns NaN into NULL on the way in),
        // so this is unreachable in practice; keep it total.
        return "NULL".to_string();
    }
    if r.is_infinite() {
        return if r < 0.0 { "-Inf".to_string() } else { "Inf".to_string() };
    }
    let neg = r.is_sign_negative();
    let a = r.abs();
    if a == 0.0 {
        return if neg { "-0.0".to_string() } else { "0.0".to_string() };
    }

    let (digits, exp) = select_digits(a);
    let ndigits = digits.len() as i32;
    let core = if exp < -4 || exp >= 15 {
        format_scientific(&digits, exp)
    } else {
        format_fixed(&digits, exp, ndigits)
    };
    if neg { format!("-{core}") } else { core }
}

/// Pick the significant digits and decimal exponent for `a` (finite, `> 0`) using
/// SQLite's 15-or-17 rule (`floatingpoint.html` §1.3.1): prefer 15 significant
/// digits, but fall back to 17 when the 15-digit decimal does not round-trip to the
/// same `f64`. Trailing zeros are stripped. Returns `(digits_without_point,
/// decimal_exponent_of_the_leading_digit)`.
fn select_digits(a: f64) -> (String, i32) {
    // `{:.14e}` = 15 significant digits (1 before the point + 14 after); `{:.16e}`
    // = 17. Rust's `{:e}` family is correctly rounded (round-half-to-even); this
    // reproduces SQLite's 15-vs-17 *decision* exactly, though SQLite's dtoa can pick
    // a different last digit in the 17-digit case (see `real_to_text` doc).
    let s15 = format!("{a:.14e}");
    let s = if s15.parse::<f64>().map(|v| v == a).unwrap_or(false) {
        s15
    } else {
        format!("{a:.16e}")
    };
    let (mant, exp_str) = s.split_once('e').expect("{:e} always contains 'e'");
    let exp: i32 = exp_str.parse().expect("{:e} exponent is an integer");
    let mut digits: String = mant.chars().filter(|c| *c != '.').collect();
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }
    (digits, exp)
}

/// `d.ddde±NN` from normalized digits and decimal exponent.
fn format_scientific(digits: &str, exp: i32) -> String {
    let mantissa = if digits.len() == 1 {
        format!("{digits}.0")
    } else {
        format!("{}.{}", &digits[..1], &digits[1..])
    };
    let (esign, emag) = if exp < 0 { ('-', -exp) } else { ('+', exp) };
    format!("{mantissa}e{esign}{emag:02}")
}

/// Fixed-point placement of `digits` given decimal exponent `exp` (the power of
/// ten of the leading digit). Always leaves a fractional part so the text reads as
/// a real.
fn format_fixed(digits: &str, exp: i32, ndigits: i32) -> String {
    if exp >= 0 {
        let int_len = (exp + 1) as usize;
        if int_len >= ndigits as usize {
            let zeros = int_len - ndigits as usize;
            format!("{digits}{}.0", "0".repeat(zeros))
        } else {
            format!("{}.{}", &digits[..int_len], &digits[int_len..])
        }
    } else {
        let zeros = (-exp - 1) as usize;
        format!("0.{}{digits}", "0".repeat(zeros))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_literal_predicate() {
        for s in ["0", "123", "+123", "-123", "-0", "  42  ", "\t-7\n"] {
            assert!(looks_like_integer(s), "{s:?} should be an integer literal");
        }
        for s in ["", "+", "-", "1.0", "1e5", "1.", ".5", "0x10", "1_000", "12a", "abc", " "] {
            assert!(!looks_like_integer(s), "{s:?} should NOT be an integer literal");
        }
    }

    #[test]
    fn real_literal_predicate() {
        for s in ["1.0", "1.", ".5", "1e5", "1E+5", "3.0e+5", "+.5e2", "  -2.5  ", "0.0"] {
            assert!(looks_like_real(s), "{s:?} should be a real literal");
        }
        for s in ["123", "+123", "", ".", "1e", "1.2.3", "e5", "0x1p4", "1_0.0", "abc"] {
            assert!(!looks_like_real(s), "{s:?} should NOT be a real literal");
        }
    }

    #[test]
    fn numeric_affinity_of_text() {
        // Whole well-formed integer literal -> INTEGER.
        assert!(matches!(numeric_affinity_text("123"), Some(Value::Integer(123))));
        assert!(matches!(numeric_affinity_text("  -7 "), Some(Value::Integer(-7))));
        // Real literal that is exactly integral -> INTEGER (datatype3: '3.0e+5' -> 300000).
        assert!(matches!(numeric_affinity_text("3.0e+5"), Some(Value::Integer(300000))));
        assert!(matches!(numeric_affinity_text("500.0"), Some(Value::Integer(500))));
        // Real literal with a fraction -> REAL.
        assert!(matches!(numeric_affinity_text("1.5"), Some(Value::Real(r)) if r == 1.5));
        // Integer literal too large for i64 -> REAL.
        match numeric_affinity_text("99999999999999999999") {
            Some(Value::Real(r)) => assert!(r > 9e19),
            other => panic!("expected REAL, got {other:?}"),
        }
        // Not numeric -> None (stays TEXT).
        assert!(numeric_affinity_text("abc").is_none());
        assert!(numeric_affinity_text("0x10").is_none());
        assert!(numeric_affinity_text("12a").is_none());
    }

    #[test]
    fn i64_bounds_parse_exactly() {
        assert!(matches!(numeric_affinity_text("9223372036854775807"), Some(Value::Integer(i64::MAX))));
        assert!(matches!(numeric_affinity_text("-9223372036854775808"), Some(Value::Integer(i64::MIN))));
        // One past MAX overflows to REAL.
        assert!(matches!(numeric_affinity_text("9223372036854775808"), Some(Value::Real(_))));
    }

    #[test]
    fn cast_integer_prefix() {
        assert_eq!(parse_int_prefix("123abc"), 123);
        assert_eq!(parse_int_prefix("  -45xyz"), -45);
        assert_eq!(parse_int_prefix("123e5"), 123); // exponent not part of int prefix
        assert_eq!(parse_int_prefix("0x12"), 0); // stops at 'x'
        assert_eq!(parse_int_prefix("abc"), 0);
        assert_eq!(parse_int_prefix(""), 0);
        assert_eq!(parse_int_prefix("+7"), 7);
        // Clamping.
        assert_eq!(parse_int_prefix("99999999999999999999"), i64::MAX);
        assert_eq!(parse_int_prefix("-99999999999999999999"), i64::MIN);
        assert_eq!(parse_int_prefix("9223372036854775807"), i64::MAX);
        assert_eq!(parse_int_prefix("-9223372036854775808"), i64::MIN);
    }

    #[test]
    fn numeric_prefix_classification() {
        use NumericPrefix::{Integer, None, Real};
        // No numeric prefix at all -> None (coerces to 0). A lone sign or dot is
        // NOT a number.
        for s in ["", "abc", "-", "+", ".", "e5", "x12", " ", "+.e5"] {
            assert_eq!(numeric_prefix(s), None, "{s:?} should have no numeric prefix");
        }
        // Integer-form prefix: sign + digits, stopping before any `.`/exponent. A
        // dangling `e` with no exponent digits stays outside the number.
        for s in ["2abc", "-3x", "42", "1english", "1e", "  5x", "+7", "0abc", "123e"] {
            assert_eq!(numeric_prefix(s), Integer, "{s:?} should be an integer prefix");
        }
        // Real-form prefix: a `.` or a real exponent is consumed.
        for s in ["2.5x", "2e3abc", "1.", ".5", "1.5", "1e5", "-2.5", "3.0e+5xyz"] {
            assert_eq!(numeric_prefix(s), Real, "{s:?} should be a real prefix");
        }
    }

    #[test]
    fn numeric_prefix_agrees_with_value_parsers() {
        // Where the classification says Integer, parse_int_prefix reads the value;
        // where it says Real, parse_real_prefix does. This pins the invariant the
        // coercion path relies on (same scan boundary, right extractor).
        assert_eq!(numeric_prefix("2abc"), NumericPrefix::Integer);
        assert_eq!(parse_int_prefix("2abc"), 2);
        assert_eq!(numeric_prefix("-3x"), NumericPrefix::Integer);
        assert_eq!(parse_int_prefix("-3x"), -3);
        assert_eq!(numeric_prefix("1e"), NumericPrefix::Integer); // dangling e
        assert_eq!(parse_int_prefix("1e"), 1);
        assert_eq!(numeric_prefix("2.5x"), NumericPrefix::Real);
        assert_eq!(parse_real_prefix("2.5x"), 2.5);
        assert_eq!(numeric_prefix("2e3abc"), NumericPrefix::Real);
        assert_eq!(parse_real_prefix("2e3abc"), 2000.0);
    }

    #[test]
    fn cast_real_prefix() {
        assert_eq!(parse_real_prefix("1.5abc"), 1.5);
        assert_eq!(parse_real_prefix("  -2.5"), -2.5);
        assert_eq!(parse_real_prefix("1."), 1.0);
        assert_eq!(parse_real_prefix(".5"), 0.5);
        assert_eq!(parse_real_prefix("1.e3"), 1000.0);
        assert_eq!(parse_real_prefix("123e5"), 12300000.0);
        assert_eq!(parse_real_prefix("abc"), 0.0);
        assert_eq!(parse_real_prefix(""), 0.0);
        assert!(parse_real_prefix("1e400").is_infinite());
    }

    #[test]
    fn cast_real_to_int_truncates_and_clamps() {
        assert_eq!(real_to_int_trunc(3.9), 3);
        assert_eq!(real_to_int_trunc(-3.9), -3); // toward zero
        assert_eq!(real_to_int_trunc(9e18), 9000000000000000000);
        assert_eq!(real_to_int_trunc(1e30), i64::MAX);
        assert_eq!(real_to_int_trunc(-1e30), i64::MIN);
    }

    #[test]
    fn cast_text_to_numeric_rules() {
        assert!(matches!(text_to_numeric("123"), Value::Integer(123)));
        assert!(matches!(text_to_numeric("123abc"), Value::Integer(123)));
        assert!(matches!(text_to_numeric("1.5"), Value::Real(r) if r == 1.5));
        assert!(matches!(text_to_numeric("1e3"), Value::Integer(1000))); // float-form but integral
        assert!(matches!(text_to_numeric("2.0"), Value::Integer(2)));
        assert!(matches!(text_to_numeric("abc"), Value::Integer(0))); // unparseable -> 0
        assert!(matches!(text_to_numeric(""), Value::Integer(0)));
        // Integer-form too large for i64 -> REAL.
        assert!(matches!(text_to_numeric("99999999999999999999"), Value::Real(_)));
        // NUMERIC_INT_MARGIN (2^51) straddle: an integral float-form below the margin
        // demotes to INTEGER, above it stays REAL. (Both are exactly representable, so
        // this pins the deliberate 51-bit bound rather than i64/f64 exactness.)
        assert!(matches!(text_to_numeric("1e15"), Value::Integer(1_000_000_000_000_000)));
        assert!(matches!(text_to_numeric("1e18"), Value::Real(r) if r == 1e18));
    }

    #[test]
    fn real_to_text_fixed_forms() {
        assert_eq!(real_to_text(1.0), "1.0");
        assert_eq!(real_to_text(500.0), "500.0");
        assert_eq!(real_to_text(3.14), "3.14");
        assert_eq!(real_to_text(0.1), "0.1");
        assert_eq!(real_to_text(12.5), "12.5");
        assert_eq!(real_to_text(0.0001), "0.0001");
        assert_eq!(real_to_text(100.0), "100.0");
        assert_eq!(real_to_text(-2.5), "-2.5");
    }

    #[test]
    fn real_to_text_scientific_forms() {
        assert_eq!(real_to_text(1e20), "1.0e+20");
        assert_eq!(real_to_text(1e-20), "1.0e-20");
        assert_eq!(real_to_text(1e-5), "1.0e-05");
        assert_eq!(real_to_text(1e15), "1.0e+15");
        // Just below the scientific threshold stays fixed.
        assert_eq!(real_to_text(1e14), "100000000000000.0");
    }

    #[test]
    fn real_to_text_specials() {
        assert_eq!(real_to_text(f64::INFINITY), "Inf");
        assert_eq!(real_to_text(f64::NEG_INFINITY), "-Inf");
        assert_eq!(real_to_text(0.0), "0.0");
    }

    #[test]
    fn real_to_text_round_trips() {
        // The rendered text must parse back to the same bits for a spread of values.
        for v in [1.0, 3.14159265358979, 2.5, 0.1, 1234.5678, 6.022e23, 1e-12, -7.25] {
            let s = real_to_text(v);
            let back: f64 = s.parse().unwrap_or_else(|_| panic!("cannot reparse {s:?}"));
            assert_eq!(back, v, "round-trip failed for {v} via {s:?}");
        }
    }

    #[test]
    fn real_to_text_matches_sqlite_15_or_17_rule() {
        // floatingpoint.html §1.3.1: the shortest form of 1/3 is the 16-digit
        // "0.3333333333333333", which SQLite NEVER emits. 15 digits does not
        // round-trip, so SQLite falls to 17 digits. (These are the
        // exact witnesses.)
        assert_eq!(real_to_text(1.0 / 3.0), "0.33333333333333331");
        assert_eq!(real_to_text(2.0 / 3.0), "0.66666666666666663");
        // The canonical 0.1+0.2 case: 15 digits ("0.3") does not round-trip -> 17.
        assert_eq!(real_to_text(0.1 + 0.2), "0.30000000000000004");
        // 15 digits suffices here, so it is used and trailing zeros stripped.
        assert_eq!(real_to_text(0.2), "0.2");
        assert_eq!(real_to_text(1.0 / 8.0), "0.125");
    }

    #[test]
    fn real_to_text_avoids_shortest_and_round_trips() {
        // Count significant digits in a rendered decimal (sign / point / exponent
        // stripped, then leading & trailing zeros removed). Never called on zero.
        fn sig_digits(s: &str) -> usize {
            let mant = s.trim_start_matches('-').split(['e', 'E']).next().unwrap();
            let ds: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
            ds.trim_start_matches('0').trim_end_matches('0').len()
        }

        // These arithmetic results have a 16-significant-digit shortest form, which
        // SQLite never emits. Through the PUBLIC api, real_to_text must never
        // reproduce that 16-digit form, yet must still round-trip to the same bits.
        let vals: [f64; 6] = [1.0 / 3.0, 2.0 / 3.0, 0.1 + 0.2, 1.23 * 2.34, 10.0 / 9.0, 1.0 / 7.0];
        let mut exercised = 0;
        for v in vals {
            let rendered = real_to_text(v);
            if sig_digits(&format!("{v:e}")) == 16 {
                exercised += 1;
                assert_ne!(
                    sig_digits(&rendered),
                    16,
                    "{v} rendered as the forbidden 16-digit shortest form {rendered:?}"
                );
            }
            let back: f64 = rendered.parse().expect("re-parse");
            assert_eq!(back, v, "round-trip failed for {v} via {rendered:?}");
        }
        // The anti-16 guard must actually fire, else the check above is vacuous.
        assert!(exercised > 0, "no 16-digit-shortest value exercised the guard");
    }
}
