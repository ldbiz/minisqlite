//! String scalar functions from `spec/sqlite-doc/lang_corefunc.html`:
//! `length`, `octet_length`, `substr`/`substring`, `upper`, `lower`,
//! `trim`/`ltrim`/`rtrim`, `replace`, `instr`, `concat`/`concat_ws`, and
//! `format`/`printf` (the last two live in the [`format`] submodule).
//!
//! Each function is a zero-sized unit struct implementing
//! [`ScalarFunction`](minisqlite_expr::ScalarFunction) that delegates to a pure
//! `*_impl` free function. Keeping the logic in ctx-free free functions makes it a
//! functional core: exhaustively testable with plain `Value`s and no `FnContext`.
//!
//! NULL handling is per-function (the evaluator does not pre-check), and the
//! registry validates argument count before dispatch, so a fixed-arity function may
//! index `args[0]`/`args[1]` without a bounds check.
//!
//! `hex`/`unhex`/`quote`/`char`/`unicode` are intentionally NOT here — they live in
//! `scalar::misc`.

mod format;

use std::borrow::Cow;
use std::sync::Arc;

use minisqlite_expr::{to_integer, FnContext, ScalarFunction};
use minisqlite_types::{integer_to_text, real_to_text, Error, Result, Value};

use crate::registry::{Arity, FunctionRegistry};

/// SQLite's default `SQLITE_MAX_LENGTH`: the largest string/blob the engine will
/// build. `replace` refuses to construct a longer result (SQLite raises
/// "string or blob too big"), so a pathological `replace` of a small needle by a
/// large replacement cannot ask the allocator for terabytes and take the shared
/// host down.
const MAX_LEN: i64 = 1_000_000_000;

/// Register the string-family scalar functions.
pub(crate) fn register(reg: &mut FunctionRegistry) {
    reg.add_scalar("length", Arity::Exact(1), Arc::new(Length));
    reg.add_scalar("octet_length", Arity::Exact(1), Arc::new(OctetLength));
    // One implementation serves both spellings and both the 2- and 3-arg forms.
    reg.add_scalar("substr", Arity::Range(2, 3), Arc::new(Substr));
    reg.add_scalar("substring", Arity::Range(2, 3), Arc::new(Substr));
    reg.add_scalar("upper", Arity::Exact(1), Arc::new(Upper));
    reg.add_scalar("lower", Arity::Exact(1), Arc::new(Lower));
    reg.add_scalar("trim", Arity::Range(1, 2), Arc::new(Trim));
    reg.add_scalar("ltrim", Arity::Range(1, 2), Arc::new(Ltrim));
    reg.add_scalar("rtrim", Arity::Range(1, 2), Arc::new(Rtrim));
    reg.add_scalar("replace", Arity::Exact(3), Arc::new(Replace));
    reg.add_scalar("instr", Arity::Exact(2), Arc::new(Instr));
    reg.add_scalar("concat", Arity::AtLeast(1), Arc::new(Concat));
    reg.add_scalar("concat_ws", Arity::AtLeast(1), Arc::new(ConcatWs));
    // `format` and its legacy alias `printf` share one implementation.
    reg.add_scalar("format", Arity::AtLeast(1), Arc::new(format::Format));
    reg.add_scalar("printf", Arity::AtLeast(1), Arc::new(format::Format));
}

// ---------------------------------------------------------------------------
// Shared value -> text views (also used by the `format` submodule)
// ---------------------------------------------------------------------------

/// The text view SQLite's `sqlite3_value_text` yields: TEXT borrows; INTEGER/REAL
/// render to their canonical text form; a BLOB's bytes are read as UTF-8 (lossily,
/// since a `Value::Text` must be valid UTF-8 whereas a SQLite blob need not be);
/// NULL yields an empty string. Callers that must distinguish NULL check it first.
pub(super) fn text_view(v: &Value) -> Cow<'_, str> {
    match v {
        Value::Text(s) => Cow::Borrowed(s.as_str()),
        Value::Integer(i) => Cow::Owned(integer_to_text(*i)),
        Value::Real(r) => Cow::Owned(real_to_text(*r)),
        Value::Blob(b) => String::from_utf8_lossy(b),
        Value::Null => Cow::Borrowed(""),
    }
}

/// The raw text-bytes view: like [`text_view`] but keeps a BLOB's bytes exactly
/// (no lossy UTF-8 fixup), for the byte-exact matching `replace` needs.
fn text_bytes(v: &Value) -> Cow<'_, [u8]> {
    match v {
        Value::Text(s) => Cow::Borrowed(s.as_bytes()),
        Value::Blob(b) => Cow::Borrowed(b.as_slice()),
        Value::Integer(i) => Cow::Owned(integer_to_text(*i).into_bytes()),
        Value::Real(r) => Cow::Owned(real_to_text(*r).into_bytes()),
        Value::Null => Cow::Borrowed(&[]),
    }
}

/// Wrap bytes as a TEXT value. The bytes come from text/number inputs (always
/// valid UTF-8) on the common path, so this is a zero-copy move; a blob input with
/// invalid UTF-8 falls back to a lossy conversion (a `Value::Text` cannot hold
/// invalid UTF-8, unlike a raw SQLite text value).
fn text_from_bytes(bytes: Vec<u8>) -> Value {
    match String::from_utf8(bytes) {
        Ok(s) => Value::Text(s),
        Err(e) => Value::Text(String::from_utf8_lossy(&e.into_bytes()).into_owned()),
    }
}

/// The byte offset of the `n`-th character of `s` (0-based), or `s.len()` if `n` is
/// at or past the end. Always lands on a UTF-8 boundary, so slicing at the result
/// never panics. `nth` short-circuits when the iterator is exhausted, so a huge `n`
/// on a short string is O(len), not O(n).
fn char_byte_index(s: &str, n: usize) -> usize {
    s.char_indices().nth(n).map_or(s.len(), |(i, _)| i)
}

// ---------------------------------------------------------------------------
// length / octet_length
// ---------------------------------------------------------------------------

/// The number of characters in `s` before the first U+0000, or the whole character
/// count if there is no NUL. This is `length()`'s rule for TEXT: SQLite counts code
/// points up to the first NUL (`lang_corefunc.html`).
fn char_len_to_nul(s: &str) -> usize {
    match s.find('\0') {
        Some(i) => s[..i].chars().count(),
        None => s.chars().count(),
    }
}

/// `length(X)` — TEXT: code points before the first NUL. BLOB: byte count.
/// INTEGER/REAL: character length of the text rendering. NULL: NULL.
fn length_impl(v: &Value) -> Value {
    let n = match v {
        Value::Null => return Value::Null,
        Value::Blob(b) => b.len(),
        Value::Text(s) => char_len_to_nul(s),
        Value::Integer(i) => integer_to_text(*i).chars().count(),
        Value::Real(r) => real_to_text(*r).chars().count(),
    };
    Value::Integer(n as i64)
}

/// `octet_length(X)` — the byte count of the value's representation: TEXT is its
/// full UTF-8 byte length (NOT truncated at a NUL, unlike `length`), BLOB its byte
/// count, INTEGER/REAL the byte length of the text rendering. NULL: NULL.
fn octet_length_impl(v: &Value) -> Value {
    let n = match v {
        Value::Null => return Value::Null,
        Value::Blob(b) => b.len(),
        Value::Text(s) => s.len(),
        Value::Integer(i) => integer_to_text(*i).len(),
        Value::Real(r) => real_to_text(*r).len(),
    };
    Value::Integer(n as i64)
}

// ---------------------------------------------------------------------------
// substr / substring
// ---------------------------------------------------------------------------

/// `substr(X, Y[, Z])` — a port of SQLite's `substrFunc`. For a BLOB, indices count
/// bytes and the result is a BLOB; otherwise X is taken as text (numbers render to
/// text) and indices count UTF-8 code points, yielding TEXT. Y is 1-based; Y<0
/// counts from the right; Z omitted means "to the end"; Z<0 means the |Z| units to
/// the LEFT of position Y. Any NULL argument yields NULL.
///
/// Y and Z are coerced through 32-bit truncation because SQLite reads them with
/// `sqlite3_value_int` (a 32-bit `int`), not `sqlite3_value_int64`. This only
/// affects magnitudes >= 2^31; every ordinary index is unchanged. (ASSUMPTION: the
/// 32-bit width matches current SQLite; verify against real sqlite if an
/// extreme-index case ever diverges.)
fn substr_impl(x: &Value, y: &Value, z: Option<&Value>) -> Value {
    if x.is_null() || y.is_null() {
        return Value::Null;
    }
    if matches!(z, Some(v) if v.is_null()) {
        return Value::Null;
    }

    let mut p1 = to_integer(y) as i32 as i64;
    let (mut p2, neg_p2) = match z {
        Some(zv) => {
            let v = to_integer(zv) as i32 as i64;
            if v < 0 {
                (-v, true) // -(i32) fits in i64, so no overflow
            } else {
                (v, false)
            }
        }
        // Z omitted: SQLite uses SQLITE_LIMIT_LENGTH as an effectively-unbounded Z.
        None => (MAX_LEN, false),
    };

    // SQLite computes the total unit count only when it is needed: always for a
    // BLOB (used in the final clamp), but for TEXT only when Y<0 (to resolve the
    // from-the-right index). Match that so a positive-index substr never scans the
    // whole string to count characters.
    let len: i64 = match x {
        Value::Blob(b) => b.len() as i64,
        _ if p1 < 0 => text_view(x).chars().count() as i64,
        _ => 0,
    };

    if p1 < 0 {
        p1 += len;
        if p1 < 0 {
            p2 += p1;
            if p2 < 0 {
                p2 = 0;
            }
            p1 = 0;
        }
    } else if p1 > 0 {
        p1 -= 1; // 1-based to 0-based
    } else if p2 > 0 {
        // Position 0 is "before the first character": it consumes one length unit.
        p2 -= 1;
    }

    if neg_p2 {
        p1 -= p2;
        if p1 < 0 {
            p2 += p1;
            if p2 < 0 {
                p2 = 0;
            }
            p1 = 0;
        }
    }
    // Invariant: p1 >= 0 && p2 >= 0.

    if let Value::Blob(b) = x {
        let mut take = p2;
        if p1 + take > len {
            take = (len - p1).max(0);
        }
        let start = p1.min(len) as usize;
        Value::Blob(b[start..start + take as usize].to_vec())
    } else {
        let s = text_view(x);
        let start = char_byte_index(&s, p1 as usize);
        let end = start + char_byte_index(&s[start..], p2 as usize);
        Value::Text(s[start..end].to_string())
    }
}

// ---------------------------------------------------------------------------
// upper / lower
// ---------------------------------------------------------------------------

/// `upper(X)` — ASCII-only upper-casing (the default build has no Unicode case
/// folding); non-ASCII bytes are unchanged. Numbers render to text first. NULL:
/// NULL. `str::to_ascii_uppercase` maps only `a`-`z`, which is exactly SQLite's
/// default `upper()`.
fn upper_impl(v: &Value) -> Value {
    if v.is_null() {
        return Value::Null;
    }
    Value::Text(text_view(v).to_ascii_uppercase())
}

/// `lower(X)` — ASCII-only lower-casing; the mirror of [`upper_impl`].
fn lower_impl(v: &Value) -> Value {
    if v.is_null() {
        return Value::Null;
    }
    Value::Text(text_view(v).to_ascii_lowercase())
}

// ---------------------------------------------------------------------------
// trim / ltrim / rtrim
// ---------------------------------------------------------------------------

/// Shared engine for `trim`/`ltrim`/`rtrim`. Removes, greedily, any leading
/// (`left`) and/or trailing (`right`) character that is a member of the SET of
/// characters in `Y` (default a single space). NULL X or NULL Y yields NULL; an
/// empty Y trims nothing (X is returned unchanged).
fn trim_impl(x: &Value, y: Option<&Value>, left: bool, right: bool) -> Value {
    if x.is_null() {
        return Value::Null;
    }
    // The default trim set (a single space) is a stack constant, so `trim(col)` over
    // millions of rows never heap-allocates a set; only a custom Y collects a Vec.
    const DEFAULT_SET: &[char] = &[' '];
    let custom: Vec<char>;
    let set: &[char] = match y {
        None => DEFAULT_SET,
        Some(v) if v.is_null() => return Value::Null,
        Some(v) => {
            custom = text_view(v).chars().collect();
            &custom
        }
    };
    let input = text_view(x);
    if set.is_empty() {
        return Value::Text(input.into_owned());
    }
    let s: &str = &input;
    let mut start = 0usize;
    let mut end = s.len();
    if left {
        while start < end {
            match s[start..end].chars().next() {
                Some(c) if set.contains(&c) => start += c.len_utf8(),
                _ => break,
            }
        }
    }
    if right {
        while end > start {
            match s[start..end].chars().next_back() {
                Some(c) if set.contains(&c) => end -= c.len_utf8(),
                _ => break,
            }
        }
    }
    Value::Text(s[start..end].to_string())
}

// ---------------------------------------------------------------------------
// replace
// ---------------------------------------------------------------------------

/// `replace(X, Y, Z)` — replace every non-overlapping occurrence of Y in X with Z,
/// matching bytes exactly (BINARY collation). Returns TEXT.
///
/// The NULL/empty ordering mirrors SQLite's `replaceFunc` exactly: X NULL -> NULL;
/// else Y NULL -> NULL; else Y empty -> X returned UNCHANGED (Z is never examined,
/// so `replace(x,'',NULL)` is `x`, not NULL); else Z NULL -> NULL; else the
/// substitution. A result that would exceed `MAX_LEN` is refused (SQLite's
/// "string or blob too big").
fn replace_impl(x: &Value, y: &Value, z: &Value) -> Result<Value> {
    if x.is_null() {
        return Ok(Value::Null);
    }
    if y.is_null() {
        return Ok(Value::Null);
    }
    let ny = text_bytes(y);
    if ny.is_empty() {
        // Empty needle: X unchanged, preserving its original storage class.
        return Ok(x.clone());
    }
    if z.is_null() {
        return Ok(Value::Null);
    }
    let hx = text_bytes(x);
    let nz = text_bytes(z);

    // First pass: count non-overlapping matches and size the result, so a huge
    // output is rejected before a single allocation (rather than after growing a
    // multi-gigabyte buffer).
    let mut count: i64 = 0;
    let mut i = 0usize;
    while i + ny.len() <= hx.len() {
        if hx[i..i + ny.len()] == *ny {
            count += 1;
            i += ny.len();
        } else {
            i += 1;
        }
    }
    if count == 0 {
        return Ok(text_from_bytes(hx.into_owned()));
    }
    let new_len =
        hx.len() as i128 + count as i128 * (nz.len() as i128 - ny.len() as i128);
    if new_len > MAX_LEN as i128 {
        return Err(Error::sql("string or blob too big"));
    }

    let mut out = Vec::with_capacity(new_len as usize);
    let mut i = 0usize;
    while i < hx.len() {
        if i + ny.len() <= hx.len() && hx[i..i + ny.len()] == *ny {
            out.extend_from_slice(&nz);
            i += ny.len();
        } else {
            out.push(hx[i]);
            i += 1;
        }
    }
    Ok(text_from_bytes(out))
}

// ---------------------------------------------------------------------------
// instr
// ---------------------------------------------------------------------------

/// The byte offset of the first occurrence of `needle` in `hay`, or `None`.
/// `needle` is assumed non-empty.
fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.len() > hay.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

/// `instr(X, Y)` — 1-based position of the first occurrence of Y in X, or 0 if
/// absent. If X AND Y are both BLOBs the search is byte-wise and the offset counts
/// bytes; otherwise both are taken as text and the offset counts characters. An
/// empty needle matches at position 1. NULL in either argument yields NULL.
fn instr_impl(x: &Value, y: &Value) -> Value {
    if x.is_null() || y.is_null() {
        return Value::Null;
    }
    if let (Value::Blob(hay), Value::Blob(needle)) = (x, y) {
        if needle.is_empty() {
            return Value::Integer(1);
        }
        return match find_subslice(hay, needle) {
            Some(pos) => Value::Integer(pos as i64 + 1),
            None => Value::Integer(0),
        };
    }
    let hay = text_view(x);
    let needle = text_view(y);
    if needle.is_empty() {
        return Value::Integer(1);
    }
    match hay.find(&*needle) {
        Some(byte_pos) => Value::Integer(hay[..byte_pos].chars().count() as i64 + 1),
        None => Value::Integer(0),
    }
}

// ---------------------------------------------------------------------------
// concat / concat_ws
// ---------------------------------------------------------------------------

/// `concat(...)` — concatenate the text rendering of every non-NULL argument
/// (NULL contributes nothing; all-NULL yields the empty string).
fn concat_impl(args: &[Value]) -> Value {
    let parts: Vec<Cow<str>> = args.iter().filter(|a| !a.is_null()).map(text_view).collect();
    let cap: usize = parts.iter().map(|p| p.len()).sum();
    let mut out = String::with_capacity(cap);
    for p in &parts {
        out.push_str(p);
    }
    Value::Text(out)
}

/// `concat_ws(SEP, ...)` — join the text renderings of the arguments after SEP with
/// SEP between them, skipping NULL arguments entirely (no separator is emitted
/// around a skipped NULL; an empty-string argument is kept and still separated). A
/// NULL separator makes the whole result NULL.
fn concat_ws_impl(args: &[Value]) -> Value {
    if args[0].is_null() {
        return Value::Null;
    }
    let sep = text_view(&args[0]);
    let parts: Vec<Cow<str>> =
        args[1..].iter().filter(|a| !a.is_null()).map(text_view).collect();
    let cap: usize =
        parts.iter().map(|p| p.len()).sum::<usize>() + sep.len() * parts.len().saturating_sub(1);
    let mut out = String::with_capacity(cap);
    for (i, p) in parts.iter().enumerate() {
        if i > 0 {
            out.push_str(&sep);
        }
        out.push_str(p);
    }
    Value::Text(out)
}

// ---------------------------------------------------------------------------
// ScalarFunction wiring — each unit struct delegates to its pure `*_impl`.
// ---------------------------------------------------------------------------

/// `length(X)`.
#[derive(Debug)]
struct Length;
impl ScalarFunction for Length {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        debug_assert!(args.len() == 1, "length/1 arity precondition");
        Ok(length_impl(&args[0]))
    }
}

/// `octet_length(X)`.
#[derive(Debug)]
struct OctetLength;
impl ScalarFunction for OctetLength {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        debug_assert!(args.len() == 1, "octet_length/1 arity precondition");
        Ok(octet_length_impl(&args[0]))
    }
}

/// `substr(X,Y[,Z])` / `substring(X,Y[,Z])`.
#[derive(Debug)]
struct Substr;
impl ScalarFunction for Substr {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        debug_assert!(matches!(args.len(), 2 | 3), "substr/2..3 arity precondition");
        Ok(substr_impl(&args[0], &args[1], args.get(2)))
    }
}

/// `upper(X)`.
#[derive(Debug)]
struct Upper;
impl ScalarFunction for Upper {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        debug_assert!(args.len() == 1, "upper/1 arity precondition");
        Ok(upper_impl(&args[0]))
    }
}

/// `lower(X)`.
#[derive(Debug)]
struct Lower;
impl ScalarFunction for Lower {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        debug_assert!(args.len() == 1, "lower/1 arity precondition");
        Ok(lower_impl(&args[0]))
    }
}

/// `trim(X[,Y])`.
#[derive(Debug)]
struct Trim;
impl ScalarFunction for Trim {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        debug_assert!(matches!(args.len(), 1 | 2), "trim/1..2 arity precondition");
        Ok(trim_impl(&args[0], args.get(1), true, true))
    }
}

/// `ltrim(X[,Y])`.
#[derive(Debug)]
struct Ltrim;
impl ScalarFunction for Ltrim {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        debug_assert!(matches!(args.len(), 1 | 2), "ltrim/1..2 arity precondition");
        Ok(trim_impl(&args[0], args.get(1), true, false))
    }
}

/// `rtrim(X[,Y])`.
#[derive(Debug)]
struct Rtrim;
impl ScalarFunction for Rtrim {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        debug_assert!(matches!(args.len(), 1 | 2), "rtrim/1..2 arity precondition");
        Ok(trim_impl(&args[0], args.get(1), false, true))
    }
}

/// `replace(X,Y,Z)`.
#[derive(Debug)]
struct Replace;
impl ScalarFunction for Replace {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        debug_assert!(args.len() == 3, "replace/3 arity precondition");
        replace_impl(&args[0], &args[1], &args[2])
    }
}

/// `instr(X,Y)`.
#[derive(Debug)]
struct Instr;
impl ScalarFunction for Instr {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        debug_assert!(args.len() == 2, "instr/2 arity precondition");
        Ok(instr_impl(&args[0], &args[1]))
    }
}

/// `concat(...)`.
#[derive(Debug)]
struct Concat;
impl ScalarFunction for Concat {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        debug_assert!(!args.is_empty(), "concat needs at least one argument");
        Ok(concat_impl(args))
    }
}

/// `concat_ws(SEP,...)`.
#[derive(Debug)]
struct ConcatWs;
impl ScalarFunction for ConcatWs {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        debug_assert!(!args.is_empty(), "concat_ws needs the separator argument");
        Ok(concat_ws_impl(args))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(v: &Value) -> &str {
        match v {
            Value::Text(s) => s,
            other => panic!("expected Text, got {other:?}"),
        }
    }
    fn blob(v: &Value) -> &[u8] {
        match v {
            Value::Blob(b) => b,
            other => panic!("expected Blob, got {other:?}"),
        }
    }
    fn int(v: &Value) -> i64 {
        match v {
            Value::Integer(i) => *i,
            other => panic!("expected Integer, got {other:?}"),
        }
    }
    fn t(s: &str) -> Value {
        Value::Text(s.to_string())
    }
    fn i(n: i64) -> Value {
        Value::Integer(n)
    }

    // ---- length / octet_length ----

    #[test]
    fn length_char_vs_byte_and_types() {
        assert!(matches!(length_impl(&Value::Null), Value::Null));
        // TEXT: characters, not bytes ("é" is 2 UTF-8 bytes but one character).
        assert_eq!(int(&length_impl(&t("abc"))), 3);
        assert_eq!(int(&length_impl(&t("café"))), 4);
        // BLOB: bytes.
        assert_eq!(int(&length_impl(&Value::Blob(vec![1, 2, 3, 4, 5]))), 5);
        // Numbers: length of the text rendering.
        assert_eq!(int(&length_impl(&Value::Integer(123))), 3);
        assert_eq!(int(&length_impl(&Value::Integer(-123))), 4);
        assert_eq!(int(&length_impl(&Value::Real(-4.5))), 4);
    }

    #[test]
    fn length_stops_at_first_nul_but_octet_length_does_not() {
        // "a\0b": length counts characters before the NUL (1); octet_length counts
        // all bytes (3). This is the documented length/octet_length distinction.
        let s = t("a\0b");
        assert_eq!(int(&length_impl(&s)), 1);
        assert_eq!(int(&octet_length_impl(&s)), 3);
    }

    #[test]
    fn octet_length_byte_counts() {
        assert!(matches!(octet_length_impl(&Value::Null), Value::Null));
        assert_eq!(int(&octet_length_impl(&t("abc"))), 3);
        assert_eq!(int(&octet_length_impl(&t("café"))), 5); // é is 2 bytes
        assert_eq!(int(&octet_length_impl(&Value::Blob(vec![0; 7]))), 7);
        assert_eq!(int(&octet_length_impl(&Value::Integer(1000))), 4);
    }

    // ---- substr ----

    fn substr2(x: &str, y: i64) -> Value {
        substr_impl(&t(x), &Value::Integer(y), None)
    }
    fn substr3(x: &str, y: i64, z: i64) -> Value {
        substr_impl(&t(x), &Value::Integer(y), Some(&Value::Integer(z)))
    }

    #[test]
    fn substr_pinned_examples() {
        assert_eq!(text(&substr3("abcde", 2, 3)), "bcd");
        assert_eq!(text(&substr2("abcde", -2)), "de");
        assert_eq!(text(&substr3("abcde", -2, 1)), "d");
        assert_eq!(text(&substr3("abcde", 2, -1)), "a");
        assert_eq!(text(&substr3("abcde", 0, 2)), "a");
    }

    #[test]
    fn substr_more_boundaries() {
        assert_eq!(text(&substr2("abcde", 0)), "abcde");
        assert_eq!(text(&substr2("abcde", 1)), "abcde");
        assert_eq!(text(&substr3("abcde", 3, 100)), "cde");
        assert_eq!(text(&substr3("abcde", -2, 5)), "de");
        // Start 10 from the right of a 5-char string, length 5 -> off the left end.
        assert_eq!(text(&substr3("abcde", -10, 5)), "");
        // Preceding characters (negative Z) that run off the front are clipped.
        assert_eq!(text(&substr3("abcde", 3, -5)), "ab");
        // NULL propagation.
        assert!(matches!(substr_impl(&Value::Null, &Value::Integer(1), None), Value::Null));
        assert!(matches!(substr_impl(&t("x"), &Value::Null, None), Value::Null));
        assert!(matches!(
            substr_impl(&t("x"), &Value::Integer(1), Some(&Value::Null)),
            Value::Null
        ));
    }

    #[test]
    fn substr_counts_characters_for_text() {
        // Multi-byte characters are indexed by code point, not byte.
        assert_eq!(text(&substr3("áéíóú", 2, 2)), "éí");
        assert_eq!(text(&substr2("áéíóú", -1)), "ú");
    }

    #[test]
    fn substr_blob_is_byte_indexed() {
        let b = Value::Blob(vec![10, 20, 30, 40, 50]);
        let r = substr_impl(&b, &Value::Integer(2), Some(&Value::Integer(3)));
        assert_eq!(blob(&r), &[20, 30, 40]);
        // Negative start counts bytes from the end.
        let r2 = substr_impl(&b, &Value::Integer(-2), None);
        assert_eq!(blob(&r2), &[40, 50]);
        // Out-of-range start yields an empty blob, never a panic.
        let r3 = substr_impl(&b, &Value::Integer(99), Some(&Value::Integer(3)));
        assert_eq!(blob(&r3), &[] as &[u8]);
    }

    // ---- upper / lower ----

    #[test]
    fn upper_lower_ascii_only() {
        assert_eq!(text(&upper_impl(&t("abcñ"))), "ABCñ");
        assert_eq!(text(&lower_impl(&t("ABCÑ"))), "abcÑ");
        assert_eq!(text(&upper_impl(&t("MixedCase123"))), "MIXEDCASE123");
        // Numbers render then fold (a no-op for digits).
        assert_eq!(text(&upper_impl(&Value::Integer(42))), "42");
        assert!(matches!(upper_impl(&Value::Null), Value::Null));
        assert!(matches!(lower_impl(&Value::Null), Value::Null));
    }

    #[test]
    fn text_functions_render_numbers_and_blobs() {
        // The "non-text X renders to text" path for the text-oriented functions.
        // substr on a numeric X: rendered to text, then character-indexed -> TEXT.
        assert_eq!(
            text(&substr_impl(&Value::Integer(12345), &Value::Integer(2), Some(&Value::Integer(3)))),
            "234"
        );
        // substr on a BLOB X stays byte-indexed and returns a BLOB.
        assert_eq!(
            blob(&substr_impl(
                &Value::Blob(vec![b'a', b'b', b'c', b'd']),
                &Value::Integer(2),
                Some(&Value::Integer(2))
            )),
            b"bc"
        );
        // upper/lower render non-text inputs (numbers, blobs) to text first.
        assert_eq!(text(&upper_impl(&Value::Real(1.5))), "1.5");
        assert_eq!(text(&lower_impl(&Value::Integer(42))), "42");
        assert_eq!(text(&upper_impl(&Value::Blob(vec![b'a', b'b', b'c']))), "ABC");
        // concat renders every non-NULL argument (numbers, blobs) to text.
        assert_eq!(
            text(&concat_impl(&[Value::Integer(1), Value::Blob(vec![b'x']), Value::Real(2.5)])),
            "1x2.5"
        );
    }

    // ---- trim family ----

    fn trim2(x: &str, y: &str) -> Value {
        trim_impl(&t(x), Some(&t(y)), true, true)
    }

    #[test]
    fn trim_pinned_and_set_semantics() {
        assert_eq!(text(&trim_impl(&t("  hi  "), None, true, true)), "hi");
        assert_eq!(text(&trim2("xyhixy", "xy")), "hi");
        assert_eq!(text(&trim_impl(&t("xxhi"), Some(&t("x")), true, false)), "hi");
        // rtrim removes only from the right.
        assert_eq!(text(&trim_impl(&t("xxhixx"), Some(&t("x")), false, true)), "xxhi");
        // Y is a SET of characters, applied greedily.
        assert_eq!(text(&trim2("abacaba", "ab")), "c");
    }

    #[test]
    fn trim_null_and_empty_set() {
        assert!(matches!(trim_impl(&Value::Null, None, true, true), Value::Null));
        // NULL Y -> NULL result.
        assert!(matches!(trim_impl(&t("hi"), Some(&Value::Null), true, true), Value::Null));
        // Empty Y trims nothing.
        assert_eq!(text(&trim2("  hi  ", "")), "  hi  ");
    }

    // ---- replace ----

    fn rep(x: &str, y: &str, z: &str) -> Value {
        replace_impl(&t(x), &t(y), &t(z)).expect("replace ok")
    }

    #[test]
    fn replace_pinned_and_edge_cases() {
        assert_eq!(text(&rep("abcabc", "bc", "X")), "aXaX");
        assert_eq!(text(&rep("abc", "", "Z")), "abc"); // empty needle -> unchanged
        // Overlap is non-overlapping, left to right.
        assert_eq!(text(&rep("aaaa", "aa", "b")), "bb");
        // No match -> the (text) input unchanged.
        assert_eq!(text(&rep("abc", "x", "y")), "abc");
        // Replacement longer than needle.
        assert_eq!(text(&rep("a.b.c", ".", "--")), "a--b--c");
    }

    #[test]
    fn replace_null_ordering() {
        // Any real NULL propagates, EXCEPT the empty-needle short-circuit which
        // returns X before Z is ever inspected.
        assert!(matches!(replace_impl(&Value::Null, &t("a"), &t("b")), Ok(Value::Null)));
        assert!(matches!(replace_impl(&t("a"), &Value::Null, &t("b")), Ok(Value::Null)));
        assert!(matches!(replace_impl(&t("abc"), &t("b"), &Value::Null), Ok(Value::Null)));
        // Empty needle + NULL Z -> X unchanged, NOT NULL.
        assert_eq!(text(&replace_impl(&t("abc"), &t(""), &Value::Null).unwrap()), "abc");
        // Empty needle preserves the original storage class.
        assert!(matches!(
            replace_impl(&Value::Integer(7), &t(""), &t("z")),
            Ok(Value::Integer(7))
        ));
    }

    #[test]
    fn replace_result_too_big_errors() {
        // ~1M single-char matches each replaced by a 1000-byte string would build a
        // ~1e9-byte result; replace must refuse it ("string or blob too big") and do
        // so from the size estimate, BEFORE allocating the oversized output.
        let hay = "a".repeat(1_001_000);
        let repl = "b".repeat(1000);
        let err = replace_impl(&t(&hay), &t("a"), &t(&repl));
        assert!(err.is_err(), "oversized replace should error, got {err:?}");
    }

    #[test]
    fn replace_on_blob_matches_bytes() {
        // A non-empty needle against a BLOB: byte-exact match and substitution (this
        // path only saw TEXT before). {1,2,3} -> {b'X'} in {1,2,3,1,2,3} = "XX".
        let x = Value::Blob(vec![1, 2, 3, 1, 2, 3]);
        let y = Value::Blob(vec![1, 2, 3]);
        let z = Value::Blob(vec![b'X']);
        assert_eq!(text(&replace_impl(&x, &y, &z).unwrap()), "XX");
    }

    // ---- instr ----

    #[test]
    fn instr_pinned_and_char_offsets() {
        assert_eq!(int(&instr_impl(&t("abcbc"), &t("bc"))), 2);
        assert_eq!(int(&instr_impl(&t("abc"), &t("x"))), 0);
        // Empty needle matches at 1.
        assert_eq!(int(&instr_impl(&t("abc"), &t(""))), 1);
        assert_eq!(int(&instr_impl(&t(""), &t(""))), 1);
        assert_eq!(int(&instr_impl(&t(""), &t("a"))), 0);
        // Character offset (not byte) for multi-byte text.
        assert_eq!(int(&instr_impl(&t("áéí"), &t("í"))), 3);
        // NULL in either -> NULL.
        assert!(matches!(instr_impl(&Value::Null, &t("a")), Value::Null));
        assert!(matches!(instr_impl(&t("a"), &Value::Null), Value::Null));
    }

    #[test]
    fn instr_blob_is_byte_offset() {
        let hay = Value::Blob(vec![0, 1, 2, 3, 2, 3]);
        let needle = Value::Blob(vec![2, 3]);
        assert_eq!(int(&instr_impl(&hay, &needle)), 3);
        let missing = Value::Blob(vec![9, 9]);
        assert_eq!(int(&instr_impl(&hay, &missing)), 0);
    }

    #[test]
    fn instr_mixed_blob_and_text_uses_text_path() {
        // Exactly one BLOB argument: NOT the both-blob byte path — both coerce to text
        // and the offset is character-based. Pins the `(Blob, Blob)` branch boundary.
        let blob_abc = Value::Blob(vec![b'a', b'b', b'c']);
        assert_eq!(int(&instr_impl(&blob_abc, &t("bc"))), 2);
        let blob_bc = Value::Blob(vec![b'b', b'c']);
        assert_eq!(int(&instr_impl(&t("abc"), &blob_bc)), 2);
    }

    // ---- concat / concat_ws ----

    #[test]
    fn concat_skips_null() {
        assert_eq!(text(&concat_impl(&[t("a"), Value::Null, Value::Integer(2)])), "a2");
        // All-NULL -> empty string.
        assert_eq!(text(&concat_impl(&[Value::Null, Value::Null])), "");
        assert_eq!(text(&concat_impl(&[Value::Real(1.5), t("x")])), "1.5x");
    }

    #[test]
    fn concat_ws_separator_rules() {
        let sep = t("-");
        assert_eq!(
            text(&concat_ws_impl(&[sep.clone(), Value::Integer(1), Value::Null, Value::Integer(2)])),
            "1-2"
        );
        // Empty-string argument is kept (still separated); only NULL is skipped.
        assert_eq!(text(&concat_ws_impl(&[t("-"), t("a"), t(""), t("b")])), "a--b");
        // NULL separator -> NULL result.
        assert!(matches!(
            concat_ws_impl(&[Value::Null, t("a"), t("b")]),
            Value::Null
        ));
    }

    // ---- end-to-end dispatch + registration ----

    /// A minimal deterministic context; the string family never reads it, but the
    /// trait method requires one.
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

    /// Resolve `name`/arity through the real builtins registry and invoke the
    /// returned handle, so a test drives the exact product path
    /// (name -> registered struct -> `ScalarFunction::call`). A mis-registration OR a
    /// mis-delegation inside `call()` turns the assertion red.
    fn call_by_name(name: &str, args: &[Value]) -> Value {
        let reg = FunctionRegistry::builtins();
        let handle = reg
            .resolve_scalar(name, args.len())
            .unwrap_or_else(|_| panic!("{name}/{} should resolve", args.len()));
        let mut ctx = TestCtx;
        handle.call(args, &mut ctx).expect("scalar call should succeed")
    }

    #[test]
    fn trim_family_names_map_to_the_right_direction() {
        // Ltrim/Rtrim/Trim differ ONLY in the (left,right) flags each `call()` passes,
        // so the delegation itself is what distinguishes them. Drive all three through
        // the registry on the SAME input and pin three distinct, direction-correct
        // results: a copy-paste flag swap in these near-identical structs, or a
        // mis-registered name, fails here (nothing else pins name -> direction).
        assert_eq!(text(&call_by_name("ltrim", &[t("xxhix"), t("x")])), "hix");
        assert_eq!(text(&call_by_name("rtrim", &[t("xxhix"), t("x")])), "xxhi");
        assert_eq!(text(&call_by_name("trim", &[t("xxhix"), t("x")])), "hi");
    }

    #[test]
    fn scalar_names_dispatch_to_the_right_impl() {
        // Spot-check every remaining name end-to-end through the registry so a
        // mis-delegation in `call()` (calling the wrong `*_impl`) cannot stay green.
        assert_eq!(int(&call_by_name("length", &[t("café")])), 4);
        assert_eq!(int(&call_by_name("octet_length", &[t("café")])), 5);
        assert_eq!(text(&call_by_name("upper", &[t("abc")])), "ABC");
        assert_eq!(text(&call_by_name("lower", &[t("ABC")])), "abc");
        assert_eq!(text(&call_by_name("substr", &[t("hello"), i(2), i(3)])), "ell");
        assert_eq!(text(&call_by_name("substring", &[t("hello"), i(2)])), "ello");
        assert_eq!(text(&call_by_name("replace", &[t("abcabc"), t("bc"), t("X")])), "aXaX");
        assert_eq!(int(&call_by_name("instr", &[t("abcbc"), t("bc")])), 2);
        assert_eq!(text(&call_by_name("concat", &[t("a"), Value::Null, i(2)])), "a2");
        assert_eq!(text(&call_by_name("concat_ws", &[t("-"), i(1), Value::Null, i(2)])), "1-2");
        assert_eq!(text(&call_by_name("format", &[t("%d-%s"), i(5), t("x")])), "5-x");
        assert_eq!(text(&call_by_name("printf", &[t("%05d"), i(42)])), "00042");
    }

    #[test]
    fn registered_names_resolve_through_the_registry() {
        let reg = FunctionRegistry::builtins();
        for (name, argc) in [
            ("length", 1usize),
            ("OCTET_LENGTH", 1),
            ("substr", 2),
            ("substr", 3),
            ("substring", 2),
            ("upper", 1),
            ("lower", 1),
            ("trim", 1),
            ("trim", 2),
            ("ltrim", 1),
            ("rtrim", 2),
            ("replace", 3),
            ("instr", 2),
            ("concat", 1),
            ("concat", 5),
            ("concat_ws", 1),
            ("concat_ws", 4),
            ("format", 1),
            ("format", 3),
            ("printf", 2),
        ] {
            assert!(reg.resolve_scalar(name, argc).is_ok(), "{name}/{argc} should resolve");
        }
        // A wrong arity for a fixed-arity name is rejected.
        assert!(reg.resolve_scalar("length", 2).is_err());
        assert!(reg.resolve_scalar("substr", 1).is_err());
    }
}
