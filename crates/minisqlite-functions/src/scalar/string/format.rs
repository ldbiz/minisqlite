//! `format(FORMAT, ...)` / `printf(FORMAT, ...)` — SQLite's C-`printf`-style string
//! formatter (`spec/sqlite-doc/printf.html`, `lang_corefunc.html#format`).
//!
//! A conversion is `%[flags][width][.precision][length]type`. The format string and
//! all substitutions are assembled into a byte buffer and converted to TEXT at the
//! end, because SQLite works on bytes: a byte-measured precision may cut a multibyte
//! UTF-8 character, which SQLite emits verbatim but a `Value::Text` cannot hold, so
//! the final conversion is lossy in that (rare) case.
//!
//! Argument coercion follows SQLite: integer conversions coerce like
//! `sqlite3_value_int64` ([`to_integer`]), float conversions like
//! `sqlite3_value_double` ([`value_to_f64`]), string conversions take the text view.
//! A missing argument (too few supplied) is a NULL, i.e. 0 / 0.0 / "".
//!
//! Known residuals vs real SQLite (documented, not spec errors): float conversions
//! use Rust's correctly-rounded formatter, whereas SQLite caps a rendering at 16
//! significant digits (26 with `!`), so a very high precision or very large
//! magnitude can differ in trailing digits; and exact-halfway rounding follows
//! Rust/C round-half-to-even. Common precisions and magnitudes match.

use minisqlite_expr::{to_integer, FnContext, ScalarFunction};
use minisqlite_types::{parse_real_prefix, Result, Value};

use super::text_view;

/// Caps every field-driven allocation (width padding, precision-driven zero fill /
/// float digits, `%c` repetition) to about one megabyte, so a hostile
/// `format('%2000000000d', 1)` cannot ask the allocator for gigabytes and take the
/// shared host down. Real SQLite bounds output by `SQLITE_MAX_LENGTH` (1e9); a field
/// between this cap and that bound is clamped here rather than honored, which only
/// affects pathological inputs no realistic query produces.
const SAFE_FIELD: i64 = 1_000_000;

/// `format(FORMAT, ...)` / `printf(FORMAT, ...)`.
#[derive(Debug)]
pub(super) struct Format;

impl ScalarFunction for Format {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        debug_assert!(!args.is_empty(), "format/printf needs the format string");
        Ok(format_impl(args))
    }
}

/// The value -> f64 coercion SQLite's `sqlite3_value_double` performs: NULL is 0.0,
/// numbers pass through, text/blob take the leading real-number prefix.
fn value_to_f64(v: &Value) -> f64 {
    match v {
        Value::Null => 0.0,
        Value::Integer(i) => *i as f64,
        Value::Real(r) => *r,
        Value::Text(s) => parse_real_prefix(s),
        Value::Blob(b) => parse_real_prefix(&String::from_utf8_lossy(b)),
    }
}

/// The parsed elements of one `%…` conversion.
#[derive(Default)]
struct Spec {
    minus: bool,          // '-'  left-justify
    plus: bool,           // '+'  force sign on positives
    space: bool,          // ' '  space before positives
    zero: bool,           // '0'  zero-pad to width
    alt: bool,            // '#'  alternate form 1
    comma: bool,          // ','  thousands separators (base-10 numerics)
    alt2: bool,           // '!'  alternate form 2 (char-measured width/precision)
    width: usize,             // minimum field width, clamped to [0, SAFE_FIELD]
    precision: Option<usize>, // None = absent; normalized to >= 0 once at parse time
}

fn format_impl(args: &[Value]) -> Value {
    // args[0] is the format string (AtLeast(1) guarantees it exists).
    if args[0].is_null() {
        return Value::Null;
    }
    let fmt = text_view(&args[0]);
    let bytes = fmt.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len() + 16);
    let mut argi = 1usize;
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] != b'%' {
            // Copy the whole literal run up to the next '%' in one shot rather than
            // byte-by-byte.
            let start = i;
            while i < bytes.len() && bytes[i] != b'%' {
                i += 1;
            }
            out.extend_from_slice(&bytes[start..i]);
            continue;
        }
        match parse_and_render(bytes, i, args, &mut argi, &mut out) {
            Some(next) => i = next,
            // An unrecognized conversion (SQLite's etINVALID) stops processing and
            // returns what has been produced so far.
            None => break,
        }
    }
    super::text_from_bytes(out)
}

/// The next argument, or a NULL if the format asks for more arguments than were
/// supplied (SQLite treats a missing argument as NULL). Always advances the index.
fn next_arg<'a>(args: &'a [Value], idx: &mut usize) -> &'a Value {
    const NULL: Value = Value::Null;
    let v = args.get(*idx).unwrap_or(&NULL);
    *idx += 1;
    v
}

/// Scan a run of ASCII digits into an `i64`, saturating rather than overflowing.
/// Returns `(value, index_after_digits)`; no digits yields `(0, start)`.
fn parse_uint(bytes: &[u8], mut i: usize) -> (i64, usize) {
    let mut v: i64 = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        v = v.saturating_mul(10).saturating_add((bytes[i] - b'0') as i64);
        i += 1;
    }
    (v, i)
}

/// Parse the conversion beginning at `bytes[start] == b'%'`, render it into `out`,
/// and return the index just past it. Returns `None` for an unknown type (caller
/// stops), matching SQLite's behavior of aborting the format on an invalid field.
fn parse_and_render(
    bytes: &[u8],
    start: usize,
    args: &[Value],
    argi: &mut usize,
    out: &mut Vec<u8>,
) -> Option<usize> {
    let n = bytes.len();
    let mut i = start + 1;
    let mut spec = Spec::default();

    // Flags (any order, repeats allowed).
    while i < n {
        match bytes[i] {
            b'-' => spec.minus = true,
            b'+' => spec.plus = true,
            b' ' => spec.space = true,
            b'0' => spec.zero = true,
            b'#' => spec.alt = true,
            b',' => spec.comma = true,
            b'!' => spec.alt2 = true,
            _ => break,
        }
        i += 1;
    }

    // Width: a '*' reads the width from the next argument (negative => left-justify
    // with the absolute value); otherwise a literal digit run.
    if i < n && bytes[i] == b'*' {
        let w = to_integer(next_arg(args, argi));
        if w < 0 {
            spec.minus = true;
            spec.width = w.unsigned_abs().min(SAFE_FIELD as u64) as usize;
        } else {
            spec.width = w.min(SAFE_FIELD) as usize;
        }
        i += 1;
    } else {
        let (w, ni) = parse_uint(bytes, i);
        spec.width = w.min(SAFE_FIELD) as usize;
        i = ni;
    }

    // Precision: '.' then a '*' (from the next argument; negative => absent) or a
    // digit run ('.' with no digits means precision 0).
    if i < n && bytes[i] == b'.' {
        i += 1;
        if i < n && bytes[i] == b'*' {
            let p = to_integer(next_arg(args, argi));
            // A negative '*' precision means "absent"; otherwise normalize to a
            // non-negative `usize` here so no downstream consumer re-clamps a signed
            // value (the invariant "precision >= 0 when Some" is type-enforced).
            spec.precision = if p < 0 { None } else { Some(p as usize) };
            i += 1;
        } else {
            let (p, ni) = parse_uint(bytes, i);
            spec.precision = Some(p as usize);
            i = ni;
        }
    }

    // Length modifiers are ignored: format() always uses 64-bit values.
    while i < n && bytes[i] == b'l' {
        i += 1;
    }

    // Type.
    if i >= n {
        return None; // trailing '%' with no type: abort like an invalid field
    }
    let ty = bytes[i];
    i += 1;

    if render_conversion(ty, &spec, args, argi, out) {
        Some(i)
    } else {
        None
    }
}

/// Render one conversion of type `ty`. Returns `false` for an unrecognized type.
fn render_conversion(
    ty: u8,
    spec: &Spec,
    args: &[Value],
    argi: &mut usize,
    out: &mut Vec<u8>,
) -> bool {
    match ty {
        b'%' => push_padded_string(out, spec, b"%"),
        b'd' | b'i' => {
            let n = to_integer(next_arg(args, argi));
            render_signed(out, spec, n);
        }
        b'u' => {
            let u = to_integer(next_arg(args, argi)) as u64;
            render_unsigned(out, spec, u, Radix::Dec, b'u');
        }
        b'x' => {
            let u = to_integer(next_arg(args, argi)) as u64;
            render_unsigned(out, spec, u, Radix::Hex, b'x');
        }
        b'X' => {
            let u = to_integer(next_arg(args, argi)) as u64;
            render_unsigned(out, spec, u, Radix::HexUpper, b'X');
        }
        b'o' => {
            let u = to_integer(next_arg(args, argi)) as u64;
            render_unsigned(out, spec, u, Radix::Oct, b'o');
        }
        // %p works like %x for the SQL function (there are no SQL pointers).
        b'p' => {
            let u = to_integer(next_arg(args, argi)) as u64;
            render_unsigned(out, spec, u, Radix::Hex, b'p');
        }
        b'f' | b'e' | b'E' | b'g' | b'G' => {
            let v = value_to_f64(next_arg(args, argi));
            render_float(out, spec, v, ty);
        }
        // %z is interchangeable with %s for the SQL function.
        b's' | b'z' => {
            let v = next_arg(args, argi);
            let s = text_view(v);
            let taken = precision_take(s.as_bytes(), spec);
            push_padded_string(out, spec, taken);
        }
        b'c' => render_char(out, spec, next_arg(args, argi)),
        b'q' => render_sql_escape(out, spec, next_arg(args, argi), b'\'', false),
        b'Q' => render_sql_escape(out, spec, next_arg(args, argi), b'\'', true),
        b'w' => render_sql_escape(out, spec, next_arg(args, argi), b'"', false),
        // %n is silently ignored and consumes no argument.
        b'n' => {}
        _ => return false,
    }
    true
}

// ---------------------------------------------------------------------------
// Padding
// ---------------------------------------------------------------------------

fn push_repeat(out: &mut Vec<u8>, b: u8, n: usize) {
    out.resize(out.len() + n, b);
}

/// Emit `prefix` + `digits` right- or left-justified within `spec.width`. When
/// zero-padding is permitted the fill goes between the prefix (sign / radix marker)
/// and the digits; otherwise spaces pad outside the whole value.
fn push_padded_numeric(out: &mut Vec<u8>, spec: &Spec, prefix: &str, digits: &str, allow_zero: bool) {
    let content = prefix.len() + digits.len();
    if spec.width <= content {
        out.extend_from_slice(prefix.as_bytes());
        out.extend_from_slice(digits.as_bytes());
        return;
    }
    let pad = spec.width - content;
    if spec.minus {
        out.extend_from_slice(prefix.as_bytes());
        out.extend_from_slice(digits.as_bytes());
        push_repeat(out, b' ', pad);
    } else if spec.zero && allow_zero {
        out.extend_from_slice(prefix.as_bytes());
        push_repeat(out, b'0', pad);
        out.extend_from_slice(digits.as_bytes());
    } else {
        push_repeat(out, b' ', pad);
        out.extend_from_slice(prefix.as_bytes());
        out.extend_from_slice(digits.as_bytes());
    }
}

/// Emit an already-built string body space-padded to `spec.width`. Width is measured
/// in bytes, or in characters when the `!` (alternate-form-2) flag is present.
fn push_padded_string(out: &mut Vec<u8>, spec: &Spec, body: &[u8]) {
    let content = if spec.alt2 { count_chars(body) } else { body.len() };
    if spec.width <= content {
        out.extend_from_slice(body);
        return;
    }
    let pad = spec.width - content;
    if spec.minus {
        out.extend_from_slice(body);
        push_repeat(out, b' ', pad);
    } else {
        push_repeat(out, b' ', pad);
        out.extend_from_slice(body);
    }
}

/// Count UTF-8 characters in a byte slice (every non-continuation byte).
fn count_chars(s: &[u8]) -> usize {
    s.iter().filter(|&&b| (b & 0xC0) != 0x80).count()
}

/// Apply a string precision: the leading `precision` bytes (default) or characters
/// (with `!`) of `s`. No precision returns all of `s`. A precision that runs past
/// the end returns the whole slice, so this never allocates.
fn precision_take<'a>(s: &'a [u8], spec: &Spec) -> &'a [u8] {
    let p = match spec.precision {
        None => return s,
        Some(p) => p,
    };
    if spec.alt2 {
        let mut count = 0;
        for (i, &b) in s.iter().enumerate() {
            if (b & 0xC0) != 0x80 {
                if count == p {
                    return &s[..i];
                }
                count += 1;
            }
        }
        s
    } else {
        &s[..p.min(s.len())]
    }
}

// ---------------------------------------------------------------------------
// Integer conversions
// ---------------------------------------------------------------------------

enum Radix {
    Dec,
    Oct,
    Hex,
    HexUpper,
}

/// The sign prefix for a numeric conversion: "-" when `neg`, else the "+"/" "/""
/// dictated by the `+` and ` ` flags. Shared by the signed-integer and float paths.
fn sign_str(neg: bool, spec: &Spec) -> &'static str {
    if neg {
        "-"
    } else if spec.plus {
        "+"
    } else if spec.space {
        " "
    } else {
        ""
    }
}

/// The integer/float precision, clamped so a huge precision cannot drive an
/// unbounded allocation. `None` stays `None` (absent).
fn clamped_precision(spec: &Spec) -> Option<usize> {
    spec.precision.map(|p| p.min(SAFE_FIELD as usize))
}

/// Zero-pad `digits` on the left to at least `prec` digits. A precision of 0 applied
/// to a zero value produces no digits at all (the C `%.0d` of 0 rule).
fn apply_int_precision(digits: &mut String, prec: Option<usize>, is_zero: bool) {
    if let Some(p) = prec {
        if p == 0 && is_zero {
            digits.clear();
        } else if digits.len() < p {
            let mut d = String::with_capacity(p);
            for _ in 0..(p - digits.len()) {
                d.push('0');
            }
            d.push_str(digits);
            *digits = d;
        }
    }
}

fn render_signed(out: &mut Vec<u8>, spec: &Spec, n: i64) {
    let neg = n < 0;
    let mut digits = n.unsigned_abs().to_string();
    let prec = clamped_precision(spec);
    apply_int_precision(&mut digits, prec, n == 0);
    if spec.comma {
        digits = group_commas(&digits);
    }
    let sign = sign_str(neg, spec);
    // A precision disables the 0 flag for integer conversions.
    let allow_zero = spec.zero && prec.is_none();
    push_padded_numeric(out, spec, sign, &digits, allow_zero);
}

fn render_unsigned(out: &mut Vec<u8>, spec: &Spec, u: u64, radix: Radix, ty: u8) {
    let mut digits = match radix {
        Radix::Dec => u.to_string(),
        Radix::Oct => format!("{u:o}"),
        Radix::Hex => format!("{u:x}"),
        Radix::HexUpper => format!("{u:X}"),
    };
    let prec = clamped_precision(spec);
    apply_int_precision(&mut digits, prec, u == 0);
    if spec.comma && matches!(radix, Radix::Dec) {
        digits = group_commas(&digits);
    }
    // Alternate form: 0x / 0X before hex, a leading 0 before octal (never for 0).
    let prefix = if spec.alt {
        match ty {
            b'x' | b'p' if u != 0 => "0x",
            b'X' if u != 0 => "0X",
            b'o' if !digits.starts_with('0') => "0",
            _ => "",
        }
    } else {
        ""
    };
    let allow_zero = spec.zero && prec.is_none();
    push_padded_numeric(out, spec, prefix, &digits, allow_zero);
}

/// Insert `,` every three digits from the right of an all-digit string.
fn group_commas(digits: &str) -> String {
    let b = digits.as_bytes();
    let len = b.len();
    let mut out = String::with_capacity(len + len / 3);
    for (i, &d) in b.iter().enumerate() {
        if i > 0 && (len - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(d as char);
    }
    out
}

// ---------------------------------------------------------------------------
// Floating-point conversions
// ---------------------------------------------------------------------------

fn render_float(out: &mut Vec<u8>, spec: &Spec, value: f64, ty: u8) {
    let neg = value.is_sign_negative();
    if !value.is_finite() {
        // Inf/NaN render as words; the 0 flag switches them to SQL/JSON literals.
        let body: &[u8] = if value.is_nan() {
            if spec.zero {
                b"null"
            } else {
                b"NaN"
            }
        } else if spec.zero {
            b"9.0e+999"
        } else {
            b"Inf"
        };
        // NaN carries no sign; Inf follows the usual sign flags.
        let sign = if value.is_nan() { "" } else { sign_str(neg, spec) };
        // Non-finite values are never zero-padded.
        let bodystr = std::str::from_utf8(body).expect("ascii");
        push_padded_numeric(out, spec, sign, bodystr, false);
        return;
    }

    // `#` (alt) and `!` (alt2) both force a decimal point; `!` additionally forces at
    // least one fractional digit (SQLite's alternate-form-2 for floats).
    let force_point = spec.alt || spec.alt2;
    let min_frac = spec.alt2;
    let mag = value.abs();
    let mut suppress_sign = false;
    let body = match ty {
        b'f' => {
            let prec = float_prec(spec, 6);
            let mut s = format!("{:.*}", prec, mag);
            apply_float_altforms(&mut s, force_point, min_frac);
            // Alternate form suppresses a negative sign when every digit is 0.
            if spec.alt && !spec.plus && neg && all_zero_digits(&s) {
                suppress_sign = true;
            }
            if spec.comma {
                s = comma_float(&s);
            }
            s
        }
        b'e' | b'E' => {
            let prec = float_prec(spec, 6);
            format_exp(mag, prec, ty == b'E', force_point, min_frac)
        }
        // b'g' | b'G'
        _ => format_general(mag, spec, ty == b'G'),
    };

    let sign = sign_str(neg && !suppress_sign, spec);
    push_padded_numeric(out, spec, sign, &body, spec.zero);
}

/// Float precision: the requested value clamped to a safe upper bound, or `default`
/// when absent.
fn float_prec(spec: &Spec, default: usize) -> usize {
    match spec.precision {
        None => default,
        Some(p) => p.min(SAFE_FIELD as usize),
    }
}

fn all_zero_digits(s: &str) -> bool {
    s.bytes().all(|b| !b.is_ascii_digit() || b == b'0')
}

/// Apply the float alternate forms to a number body (the mantissa for `%e`/`%g`, the
/// whole value for `%f`): `force_point` (`#` or `!`) inserts a decimal point when one
/// is absent, and `min_frac` (`!`) then guarantees at least one digit after it.
fn apply_float_altforms(s: &mut String, force_point: bool, min_frac: bool) {
    if force_point && !s.contains('.') {
        s.push('.');
    }
    // `!` guarantees a digit after the point: a number body has at most one '.', so a
    // trailing '.' (the only way to have no fractional digit) gets a '0'.
    if min_frac && s.ends_with('.') {
        s.push('0');
    }
}

/// Append a C/SQLite-style exponent suffix (`e±dd`, at least two exponent digits, the
/// sign always present) to a mantissa already built in `s`.
fn push_exponent(s: &mut String, exp_num: i32, upper: bool) {
    s.push(if upper { 'E' } else { 'e' });
    s.push(if exp_num < 0 { '-' } else { '+' });
    s.push_str(&format!("{:02}", exp_num.unsigned_abs()));
}

/// Split Rust's `{:e}` output (`<mant>e<exp>`) into its mantissa and integer exponent.
/// Only ever called on the formatter's own finite-float output, so the shape is
/// guaranteed and the `expect`s are invariants, not failure points.
fn split_sci(s: &str) -> (&str, i32) {
    let (mant, exp) = s.split_once('e').expect("{:e} contains 'e'");
    (mant, exp.parse().expect("{:e} exponent is an integer"))
}

/// `%e`/`%E`: `d.dddde±dd` with a signed, >=2-digit exponent (C/SQLite style, which
/// Rust's `{:e}` does not produce).
fn format_exp(mag: f64, prec: usize, upper: bool, force_point: bool, min_frac: bool) -> String {
    let s = format!("{:.*e}", prec, mag);
    let (mant, exp_num) = split_sci(&s);
    let mut mantissa = mant.to_string();
    apply_float_altforms(&mut mantissa, force_point, min_frac);
    push_exponent(&mut mantissa, exp_num, upper);
    mantissa
}

/// `%g`/`%G`: choose fixed or exponential per the C rule (fixed when the exponent is
/// in `-4..P`), then strip trailing zeros unless the `#` flag keeps them. The `!`
/// flag strips like the plain form but then forces a point and one fractional digit.
fn format_general(mag: f64, spec: &Spec, upper: bool) -> String {
    let p_req = spec.precision.unwrap_or(6);
    // Precision 0 is treated as 1; clamp for allocation safety. The parens make the
    // `.min()` apply to the whole `if` (both arms are already <= the cap, so this is
    // clarity, not a behavior change).
    let p = (if p_req == 0 { 1 } else { p_req }).min(SAFE_FIELD as usize);
    let es = format!("{:.*e}", p - 1, mag);
    let (mant, exp_num) = split_sci(&es);

    // `#` keeps trailing zeros; the plain and `!` forms strip them.
    let strip = !spec.alt;
    let force_point = spec.alt || spec.alt2;
    let min_frac = spec.alt2;

    if exp_num < -4 || exp_num >= p as i32 {
        let mut mantissa = mant.to_string();
        if strip {
            mantissa = strip_trailing_zeros(&mantissa).to_string();
        }
        apply_float_altforms(&mut mantissa, force_point, min_frac);
        push_exponent(&mut mantissa, exp_num, upper);
        mantissa
    } else {
        let fprec = (p as i32 - 1 - exp_num).max(0) as usize;
        let mut body = format!("{:.*}", fprec, mag);
        if strip {
            body = strip_trailing_zeros(&body).to_string();
        }
        apply_float_altforms(&mut body, force_point, min_frac);
        body
    }
}

/// Strip trailing fractional zeros (and a now-bare decimal point) from a number that
/// has no exponent suffix.
fn strip_trailing_zeros(s: &str) -> &str {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.')
    } else {
        s
    }
}

/// Insert thousands separators into the integer part of a formatted float.
fn comma_float(body: &str) -> String {
    match body.split_once('.') {
        Some((int_part, frac)) => format!("{}.{}", group_commas(int_part), frac),
        None => group_commas(body),
    }
}

// ---------------------------------------------------------------------------
// %c and the SQL-escape conversions (%q, %Q, %w)
// ---------------------------------------------------------------------------

/// `%c` — the first *character* of the argument's text, repeated `precision` times
/// when the precision exceeds 1 (a SQLite extension), then width-padded. The whole
/// leading UTF-8 scalar is emitted, so a multibyte first character (e.g. `é`) is
/// preserved rather than truncated to a lone lead byte. An empty/NULL argument yields
/// a single NUL byte, matching SQLite's `c = z ? z[0] : 0`.
fn render_char(out: &mut Vec<u8>, spec: &Spec, v: &Value) {
    let s = text_view(v);
    let mut buf = [0u8; 4];
    let ch_bytes: &[u8] = match s.chars().next() {
        Some(c) => c.encode_utf8(&mut buf).as_bytes(),
        None => &[0],
    };
    let reps = match spec.precision {
        Some(p) if p > 1 => p.min(SAFE_FIELD as usize),
        _ => 1,
    };
    let mut body = Vec::with_capacity(reps * ch_bytes.len());
    for _ in 0..reps {
        body.extend_from_slice(ch_bytes);
    }
    push_padded_string(out, spec, &body);
}

/// `%q`/`%Q`/`%w` — SQL-literal escaping. `q` (the quote byte) is doubled throughout;
/// when `wrap` is set (`%Q`) the whole thing is surrounded by `q` and a NULL argument
/// renders as the bare word `NULL`. Precision limits how much of the argument is
/// taken (before escaping); width pads the final output.
fn render_sql_escape(out: &mut Vec<u8>, spec: &Spec, v: &Value, q: u8, wrap: bool) {
    if wrap && v.is_null() {
        push_padded_string(out, spec, b"NULL");
        return;
    }
    let s = text_view(v);
    let taken = precision_take(s.as_bytes(), spec);
    let mut body: Vec<u8> = Vec::with_capacity(taken.len() + 2);
    if wrap {
        body.push(q);
    }
    for &b in taken {
        if b == q {
            body.push(q);
        }
        body.push(b);
    }
    if wrap {
        body.push(q);
    }
    push_padded_string(out, spec, &body);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(fmt: &str, args: &[Value]) -> String {
        let mut all = Vec::with_capacity(args.len() + 1);
        all.push(Value::Text(fmt.to_string()));
        all.extend_from_slice(args);
        match format_impl(&all) {
            Value::Text(s) => s,
            other => panic!("expected Text, got {other:?}"),
        }
    }
    fn i(n: i64) -> Value {
        Value::Integer(n)
    }
    fn r(x: f64) -> Value {
        Value::Real(x)
    }
    fn t(s: &str) -> Value {
        Value::Text(s.to_string())
    }

    #[test]
    fn pinned_examples() {
        assert_eq!(f("%d-%s", &[i(5), t("x")]), "5-x");
        assert_eq!(f("%.2f", &[r(3.14159)]), "3.14");
        assert_eq!(f("%05d", &[i(42)]), "00042");
        assert_eq!(f("%x", &[i(255)]), "ff");
        assert_eq!(f("%q", &[t("a'b")]), "a''b");
        assert_eq!(f("%Q", &[Value::Null]), "NULL");
    }

    #[test]
    fn null_format_is_null() {
        assert!(matches!(format_impl(&[Value::Null]), Value::Null));
        assert!(matches!(format_impl(&[Value::Null, i(1)]), Value::Null));
    }

    #[test]
    fn literal_and_percent() {
        assert_eq!(f("no substitutions", &[]), "no substitutions");
        assert_eq!(f("100%%", &[]), "100%");
        assert_eq!(f("%d%%", &[i(50)]), "50%");
    }

    #[test]
    fn integer_specifiers() {
        assert_eq!(f("%d", &[i(-7)]), "-7");
        assert_eq!(f("%i", &[i(42)]), "42");
        assert_eq!(f("%+d", &[i(5)]), "+5");
        assert_eq!(f("% d", &[i(5)]), " 5");
        assert_eq!(f("%+d", &[i(-5)]), "-5");
        // Unsigned reinterpretation of a negative.
        assert_eq!(f("%u", &[i(-1)]), "18446744073709551615");
        // Coercion: text/real coerce like sqlite3_value_int64.
        assert_eq!(f("%d", &[t("42abc")]), "42");
        assert_eq!(f("%d", &[r(3.9)]), "3");
        // Missing argument is 0.
        assert_eq!(f("%d", &[]), "0");
    }

    #[test]
    fn integer_width_precision_flags() {
        assert_eq!(f("%5d", &[i(42)]), "   42");
        assert_eq!(f("%-5d", &[i(42)]), "42   ");
        assert_eq!(f("%05d", &[i(42)]), "00042");
        assert_eq!(f("%05d", &[i(-42)]), "-0042");
        assert_eq!(f("%.4d", &[i(42)]), "0042");
        // Precision disables the 0 flag; sign then space-pad.
        assert_eq!(f("%8.4d", &[i(-42)]), "   -0042");
        // %.0d of 0 is empty.
        assert_eq!(f("%.0d", &[i(0)]), "");
        assert_eq!(f("[%.0d]", &[i(0)]), "[]");
        // width from '*', negative => left-justify.
        assert_eq!(f("%*d", &[i(5), i(42)]), "   42");
        assert_eq!(f("%*d", &[i(-5), i(42)]), "42   ");
        // precision from '*'.
        assert_eq!(f("%.*d", &[i(4), i(42)]), "0042");
    }

    #[test]
    fn hex_octal() {
        assert_eq!(f("%x", &[i(255)]), "ff");
        assert_eq!(f("%X", &[i(255)]), "FF");
        assert_eq!(f("%#x", &[i(255)]), "0xff");
        assert_eq!(f("%#X", &[i(255)]), "0XFF");
        assert_eq!(f("%08x", &[i(255)]), "000000ff");
        assert_eq!(f("%#010x", &[i(255)]), "0x000000ff");
        assert_eq!(f("%o", &[i(8)]), "10");
        assert_eq!(f("%#o", &[i(8)]), "010");
        // No 0x/0 prefix for a zero value.
        assert_eq!(f("%#x", &[i(0)]), "0");
    }

    #[test]
    fn float_f() {
        assert_eq!(f("%f", &[r(3.5)]), "3.500000");
        assert_eq!(f("%.2f", &[r(3.14159)]), "3.14");
        assert_eq!(f("%.0f", &[r(3.7)]), "4");
        assert_eq!(f("%8.2f", &[r(3.14159)]), "    3.14");
        assert_eq!(f("%-8.2f", &[r(3.14159)]), "3.14    ");
        assert_eq!(f("%08.2f", &[r(3.14159)]), "00003.14");
        assert_eq!(f("%+.2f", &[r(3.14)]), "+3.14");
        assert_eq!(f("%.2f", &[r(-3.14159)]), "-3.14");
        // Alternate form: force the point, and suppress a negative that rounds to 0.
        assert_eq!(f("%#.0f", &[r(3.0)]), "3.");
        assert_eq!(f("%#.2f", &[r(-0.004)]), "0.00");
        // Missing arg -> 0.0.
        assert_eq!(f("%.1f", &[]), "0.0");
    }

    #[test]
    fn float_e_and_g() {
        assert_eq!(f("%e", &[r(1000.0)]), "1.000000e+03");
        assert_eq!(f("%E", &[r(1000.0)]), "1.000000E+03");
        assert_eq!(f("%.2e", &[r(0.5)]), "5.00e-01");
        assert_eq!(f("%e", &[r(1.5e300)]), "1.500000e+300");
        assert_eq!(f("%g", &[r(100000.0)]), "100000");
        assert_eq!(f("%g", &[r(1000000.0)]), "1e+06");
        assert_eq!(f("%g", &[r(0.0001)]), "0.0001");
        assert_eq!(f("%g", &[r(0.00001)]), "1e-05");
        assert_eq!(f("%.3g", &[r(3.14159)]), "3.14");
        assert_eq!(f("%g", &[r(0.0)]), "0");
        // Alternate form keeps trailing zeros.
        assert_eq!(f("%#.3g", &[r(2.0)]), "2.00");
    }

    #[test]
    fn float_non_finite() {
        assert_eq!(f("%f", &[r(f64::INFINITY)]), "Inf");
        assert_eq!(f("%f", &[r(f64::NEG_INFINITY)]), "-Inf");
        // The 0 flag switches to SQL/JSON literals.
        assert_eq!(f("%05f", &[r(f64::INFINITY)]), "9.0e+999");
    }

    #[test]
    fn strings() {
        assert_eq!(f("%s", &[t("hello")]), "hello");
        assert_eq!(f("%.2s", &[t("hello")]), "he");
        assert_eq!(f("%5s", &[t("hi")]), "   hi");
        assert_eq!(f("%-5s", &[t("hi")]), "hi   ");
        // NULL string arg is empty.
        assert_eq!(f("[%s]", &[Value::Null]), "[]");
        // Numbers coerce to text.
        assert_eq!(f("%s", &[i(42)]), "42");
        assert_eq!(f("%s", &[r(1.5)]), "1.5");
        // %z is the same as %s.
        assert_eq!(f("%z", &[t("zed")]), "zed");
        // Byte precision (default) counts bytes; the ! flag counts characters.
        assert_eq!(f("%.2s", &[t("éq")]), "é"); // first 2 bytes = the 2-byte 'é'
        assert_eq!(f("%!.2s", &[t("éq")]), "éq"); // first 2 characters
    }

    #[test]
    fn char_specifier() {
        assert_eq!(f("%c", &[t("hello")]), "h");
        assert_eq!(f("%c", &[t("")]), "\0");
        // NULL argument behaves like the empty string: a single NUL byte.
        assert_eq!(f("%c", &[Value::Null]), "\0");
        // Precision > 1 repeats.
        assert_eq!(f("%.3c", &[t("x")]), "xxx");
    }

    #[test]
    fn c_extracts_first_character_not_first_byte() {
        // printf.html: for format(), %c displays the first CHARACTER of the argument.
        // 'é' == U+00E9 == bytes C3 A9; its first character is 'é', not a lone byte.
        assert_eq!(f("%c", &[t("é")]), "é");
        // Precision > 1 repeats that whole character, not the lead byte.
        assert_eq!(f("%.3c", &[t("é")]), "ééé");
        // A leading multibyte char with trailing text still yields just that char.
        assert_eq!(f("%c", &[t("日本")]), "日");
    }

    #[test]
    fn float_alt2_forces_decimal_point_and_digit() {
        // '!' (alternate-form-2) on floats forces a decimal point AND at least one
        // fractional digit, even where the plain form would show neither.
        assert_eq!(f("%!g", &[r(3.0)]), "3.0");
        assert_eq!(f("%!.0f", &[r(3.0)]), "3.0");
        assert_eq!(f("%!.0e", &[r(3.0)]), "3.0e+00");
        // A value that already has fractional digits is unchanged by '!'.
        assert_eq!(f("%!g", &[r(3.5)]), "3.5");
        assert_eq!(f("%!f", &[r(3.5)]), "3.500000");
        // '!' still strips %g trailing zeros (unlike '#'), then re-adds one.
        assert_eq!(f("%!g", &[r(2.0)]), "2.0");
    }

    #[test]
    fn sql_escapes() {
        assert_eq!(f("%q", &[t("a'b'c")]), "a''b''c");
        assert_eq!(f("%Q", &[t("a'b")]), "'a''b'");
        assert_eq!(f("%Q", &[Value::Null]), "NULL");
        assert_eq!(f("%q", &[Value::Null]), ""); // %q NULL -> empty
        assert_eq!(f("%w", &[t("a\"b")]), "a\"\"b");
        assert_eq!(f("%Q", &[i(5)]), "'5'");
    }

    #[test]
    fn ignored_and_alias_conversions() {
        // %n is ignored and consumes no argument, so the following %d still sees 7.
        assert_eq!(f("a%nb%d", &[i(7)]), "ab7");
        // %p works like %x.
        assert_eq!(f("%p", &[i(255)]), "ff");
    }

    #[test]
    fn comma_grouping() {
        assert_eq!(f("%,d", &[i(2147483647)]), "2,147,483,647");
        assert_eq!(f("%,d", &[i(1000)]), "1,000");
        assert_eq!(f("%,d", &[i(999)]), "999");
    }

    #[test]
    fn too_few_arguments_are_null() {
        // Every missing argument is a NULL: 0 / 0.0 / "".
        assert_eq!(f("%d %s %f", &[]), "0  0.000000");
    }

    #[test]
    fn multiple_conversions_track_arguments() {
        assert_eq!(f("%d/%d/%d", &[i(2024), i(1), i(9)]), "2024/1/9");
        assert_eq!(f("%s=%d", &[t("k"), i(3)]), "k=3");
        // '*' width/precision consume arguments in order before the value.
        assert_eq!(f("%*.*f", &[i(10), i(2), r(3.14159)]), "      3.14");
    }
}
