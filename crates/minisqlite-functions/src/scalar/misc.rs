//! Miscellaneous / utility core scalar functions from
//! `spec/sqlite-doc/lang_corefunc.html`: `typeof`, `quote`, `hex`, `unhex`,
//! `char`, `unicode`, `random`, `randomblob`, `zeroblob`, the optimizer-hint
//! no-ops (`likelihood`/`likely`/`unlikely`), the connection counters
//! (`last_insert_rowid`/`changes`/`total_changes`), and the version constants
//! (`sqlite_version`/`sqlite_source_id`).
//!
//! Each function is a zero-sized unit struct implementing
//! [`ScalarFunction`](minisqlite_expr::ScalarFunction). Every function decides its
//! own NULL behavior — the evaluator does not pre-check arguments for NULL. The
//! registry validates argument count before dispatch, so a fixed-arity function
//! may index `args[0]` without a length check; that invariant is the whole reason
//! `resolve_scalar` exists. A `debug_assert!` on `args.len()` at each such call
//! records that precondition and turns a future mis-wired dispatch into a loud
//! test failure rather than an index panic in production.

use std::borrow::Cow;
use std::sync::Arc;

use minisqlite_expr::{to_integer, FnContext, ScalarFunction};
use minisqlite_types::{integer_to_text, real_to_text, Error, Result, Value};

use crate::registry::{Arity, FunctionRegistry};

/// `randomblob`/`zeroblob` refuse a request past this many bytes with SQLite's
/// "string or blob too big" error, so a hostile `zeroblob(1e18)` cannot ask the
/// allocator for exabytes and take the process (and the shared host) down. SQLite
/// caps blob length at `SQLITE_MAX_LENGTH`, one billion bytes by default.
const MAX_BLOB_LEN: i64 = 1_000_000_000;

// NOTE: `sqlite_version()` reports a fixed, recent released version string. This
// is a compatibility approximation — real SQLite reports the exact version of the
// library that was built — chosen so version-gated queries see a plausible,
// well-formed value. Update when the engine intends to claim a different version.
const SQLITE_VERSION: &str = "3.50.0";

// NOTE: `sqlite_source_id()` reports a fixed, plausible source-id string (the
// "<UTC check-in timestamp> <SHA3-256 hex>" shape real SQLite uses). The exact
// hash is build-specific and cannot be derived from the docs, so this is a
// compatibility approximation, like `sqlite_version()` above.
const SQLITE_SOURCE_ID: &str =
    "2025-05-29 14:07:00 f7d3f8d5a4b3c2e1d0f9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b9c8d7";

/// Register the misc/util core scalar functions.
pub(crate) fn register(reg: &mut FunctionRegistry) {
    reg.add_scalar("typeof", Arity::Exact(1), Arc::new(Typeof));
    reg.add_scalar("quote", Arity::Exact(1), Arc::new(Quote));
    reg.add_scalar("hex", Arity::Exact(1), Arc::new(Hex));
    // One implementation handles both `unhex(X)` and `unhex(X,Y)`.
    reg.add_scalar("unhex", Arity::Range(1, 2), Arc::new(Unhex));
    reg.add_scalar("char", Arity::Any, Arc::new(Char));
    reg.add_scalar("unicode", Arity::Exact(1), Arc::new(Unicode));
    reg.add_scalar("random", Arity::Exact(0), Arc::new(Random));
    reg.add_scalar("randomblob", Arity::Exact(1), Arc::new(RandomBlob));
    reg.add_scalar("zeroblob", Arity::Exact(1), Arc::new(ZeroBlob));
    // The three optimizer hints all return their first argument unchanged.
    reg.add_scalar("likelihood", Arity::Exact(2), Arc::new(Identity));
    reg.add_scalar("likely", Arity::Exact(1), Arc::new(Identity));
    reg.add_scalar("unlikely", Arity::Exact(1), Arc::new(Identity));
    reg.add_scalar("last_insert_rowid", Arity::Exact(0), Arc::new(LastInsertRowid));
    reg.add_scalar("changes", Arity::Exact(0), Arc::new(Changes));
    reg.add_scalar("total_changes", Arity::Exact(0), Arc::new(TotalChanges));
    reg.add_scalar("sqlite_version", Arity::Exact(0), Arc::new(SqliteVersion));
    reg.add_scalar("sqlite_source_id", Arity::Exact(0), Arc::new(SqliteSourceId));
}

// ---------------------------------------------------------------------------
// Byte / hex helpers
// ---------------------------------------------------------------------------

/// Append the uppercase hex of `bytes` to `out` with a table lookup (no per-byte
/// `format!`), so `hex()` on a multi-megabyte blob stays linear and
/// allocation-light. Each nibble maps to an ASCII digit, so `as char` is always a
/// valid, non-multibyte push. Callers that render into an existing buffer (e.g.
/// `quote`'s `X'..'` form) use this directly to avoid a throwaway intermediate.
fn push_upper_hex(out: &mut String, bytes: &[u8]) {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    out.reserve(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
}

/// Uppercase-hex-encode a byte slice into a fresh, right-sized `String`.
fn to_upper_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    push_upper_hex(&mut out, bytes);
    out
}

/// The nibble value of one hex digit, or `None` if `b` is not `[0-9A-Fa-f]`.
fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// The text-byte view SQLite's `sqlite3_value_text` yields: TEXT/BLOB borrow their
/// bytes; INTEGER/REAL render to their text form; NULL yields an empty slice. The
/// NULL handling differs by caller: `unhex` short-circuits NULL to a NULL result
/// before ever calling this, whereas `hex` relies on the empty slice here so that
/// `hex(NULL)` is `""` (matching SQLite).
fn value_text_bytes(v: &Value) -> Cow<'_, [u8]> {
    match v {
        Value::Text(s) => Cow::Borrowed(s.as_bytes()),
        Value::Blob(b) => Cow::Borrowed(b.as_slice()),
        Value::Integer(i) => Cow::Owned(integer_to_text(*i).into_bytes()),
        Value::Real(r) => Cow::Owned(real_to_text(*r).into_bytes()),
        Value::Null => Cow::Borrowed(b"".as_slice()),
    }
}

/// Decode hex text `hex` to bytes, treating any character present in `ignore` (and
/// not itself a hex digit) as a skippable separator. Returns `None` — SQLite's
/// NULL result — when a non-hex, non-ignored character appears, or when a hex pair
/// is split (its two digits are not immediately adjacent) or left odd.
///
/// The adjacency rule is why this scans rather than "strip ignores then decode":
/// `unhex('4 1', ' ')` must be NULL (the pair `4 1` is split), not `0x41`. A hex
/// digit is tested for hex-ness *before* the ignore set, so a hex digit listed in
/// `Y` still counts as part of a pair — matching the spec's "hexadecimal digits in
/// Y have no affect on the translation of X" (e.g. `unhex('41', '1')` is `[0x41]`,
/// not a stripped `'4'`).
fn unhex_decode(hex: &[u8], ignore: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(hex.len() / 2);
    let mut i = 0;
    while i < hex.len() {
        match hex_nibble(hex[i]) {
            Some(hi) => {
                // A hex digit must be immediately followed by a second hex digit.
                let lo = hex.get(i + 1).copied().and_then(hex_nibble)?;
                out.push((hi << 4) | lo);
                i += 2;
            }
            // Not a hex digit: skip it only if it is an explicit ignore char.
            None if ignore.contains(&hex[i]) => i += 1,
            None => return None,
        }
    }
    Some(out)
}

/// Single-quote `s` for an SQL string literal, doubling every interior `'`. SQL
/// string literals cannot hold a NUL, so `s` is truncated at the first NUL first
/// (matching SQLite's `quote()`).
fn quote_text(s: &str) -> String {
    let truncated = match s.find('\0') {
        Some(nul) => &s[..nul],
        None => s,
    };
    let mut out = String::with_capacity(truncated.len() + 2);
    out.push('\'');
    for ch in truncated.chars() {
        if ch == '\'' {
            out.push('\'');
        }
        out.push(ch);
    }
    out.push('\'');
    out
}

/// Map a (possibly out-of-range) integer code point to a `char`, folding any
/// negative, above-`U+10FFFF`, or surrogate value to the replacement character —
/// SQLite's lenient `char()` behavior. `char::from_u32` already rejects surrogates
/// and out-of-range values; the explicit `< 0` guard covers negatives that would
/// otherwise wrap when cast to `u32`.
fn codepoint_to_char(cp: i64) -> char {
    if cp < 0 || cp > 0x10FFFF {
        return char::REPLACEMENT_CHARACTER;
    }
    char::from_u32(cp as u32).unwrap_or(char::REPLACEMENT_CHARACTER)
}

/// The code point of the first character of `v` cast to text, or `None` for NULL
/// and for the empty string. Used by `unicode()`.
fn first_char_codepoint(v: &Value) -> Option<i64> {
    let cp = |s: &str| s.chars().next().map(|c| c as u32 as i64);
    match v {
        Value::Null => None,
        Value::Text(s) => cp(s),
        Value::Blob(b) => cp(&String::from_utf8_lossy(b)),
        Value::Integer(i) => cp(&integer_to_text(*i)),
        Value::Real(r) => cp(&real_to_text(*r)),
    }
}

// ---------------------------------------------------------------------------
// typeof / quote / hex / unhex
// ---------------------------------------------------------------------------

/// `typeof(X)` — the storage-class name of `X` ("null"/"integer"/…).
#[derive(Debug)]
struct Typeof;
impl ScalarFunction for Typeof {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        debug_assert!(args.len() == 1, "typeof/1 arity precondition");
        Ok(Value::Text(args[0].type_name().to_string()))
    }
}

/// `quote(X)` — the SQL literal text of `X`.
#[derive(Debug)]
struct Quote;
impl ScalarFunction for Quote {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        debug_assert!(args.len() == 1, "quote/1 arity precondition");
        let out = match &args[0] {
            Value::Null => "NULL".to_string(),
            Value::Integer(i) => integer_to_text(*i),
            Value::Real(r) => real_to_text(*r),
            Value::Text(s) => quote_text(s),
            Value::Blob(b) => {
                // Render `X'HEX'` directly into one buffer (no throwaway hex String).
                let mut q = String::with_capacity(b.len() * 2 + 3);
                q.push('X');
                q.push('\'');
                push_upper_hex(&mut q, b);
                q.push('\'');
                q
            }
        };
        Ok(Value::Text(out))
    }
}

/// `hex(X)` — uppercase hex of `X` interpreted as bytes (INTEGER/REAL first render
/// to text, then those text bytes are hexed). NULL renders as the empty string.
#[derive(Debug)]
struct Hex;
impl ScalarFunction for Hex {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        debug_assert!(args.len() == 1, "hex/1 arity precondition");
        Ok(Value::Text(to_upper_hex(&value_text_bytes(&args[0]))))
    }
}

/// `unhex(X)` / `unhex(X, Y)` — decode hex text `X` to a BLOB, ignoring the
/// non-hex characters in `Y`. Any stray non-hex char, a split pair, or an odd
/// digit count yields NULL; a NULL in either argument yields NULL.
#[derive(Debug)]
struct Unhex;
impl ScalarFunction for Unhex {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        debug_assert!(!args.is_empty(), "unhex needs at least one argument");
        if args[0].is_null() {
            return Ok(Value::Null);
        }
        let ignore = match args.get(1) {
            Some(Value::Null) => return Ok(Value::Null),
            Some(y) => value_text_bytes(y),
            None => Cow::Borrowed(b"".as_slice()),
        };
        let hex = value_text_bytes(&args[0]);
        Ok(match unhex_decode(&hex, &ignore) {
            Some(bytes) => Value::Blob(bytes),
            None => Value::Null,
        })
    }
}

// ---------------------------------------------------------------------------
// char / unicode
// ---------------------------------------------------------------------------

/// `char(X1, …, XN)` — a string of the Unicode characters named by the integer
/// code points `X1…XN`. Zero arguments yield the empty string.
#[derive(Debug)]
struct Char;
impl ScalarFunction for Char {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        let mut s = String::with_capacity(args.len());
        for a in args {
            s.push(codepoint_to_char(to_integer(a)));
        }
        Ok(Value::Text(s))
    }
}

/// `unicode(X)` — the code point of the first character of `X` (as text). NULL and
/// the empty string yield NULL.
#[derive(Debug)]
struct Unicode;
impl ScalarFunction for Unicode {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        debug_assert!(args.len() == 1, "unicode/1 arity precondition");
        Ok(match first_char_codepoint(&args[0]) {
            Some(cp) => Value::Integer(cp),
            None => Value::Null,
        })
    }
}

// ---------------------------------------------------------------------------
// random / randomblob / zeroblob
// ---------------------------------------------------------------------------

/// `random()` — a pseudo-random integer from the connection RNG.
///
/// NOTE: the spec has `random()` avoid `i64::MIN` so the result is always safe to
/// pass to `abs()`. That guarantee belongs to the RNG behind `FnContext` (the
/// connection owns and seeds it), not to this call site, which just surfaces
/// whatever `random_i64` yields.
#[derive(Debug)]
struct Random;
impl ScalarFunction for Random {
    fn call(&self, _args: &[Value], ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(Value::Integer(ctx.random_i64()))
    }
    /// Non-deterministic: each call draws a fresh value from the connection RNG, so a
    /// correlated subquery containing `random()` must not be memoized.
    fn deterministic(&self) -> bool {
        false
    }
}

/// `randomblob(N)` — an `N`-byte blob of random bytes (`N < 1` yields 1 byte).
#[derive(Debug)]
struct RandomBlob;
impl ScalarFunction for RandomBlob {
    fn call(&self, args: &[Value], ctx: &mut dyn FnContext) -> Result<Value> {
        debug_assert!(args.len() == 1, "randomblob/1 arity precondition");
        let n = to_integer(&args[0]);
        if n > MAX_BLOB_LEN {
            return Err(Error::sql("string or blob too big"));
        }
        let len = if n < 1 { 1 } else { n as usize };
        let mut buf = vec![0u8; len];
        ctx.fill_random(&mut buf);
        Ok(Value::Blob(buf))
    }
    /// Non-deterministic: fills the blob from the RNG, a fresh draw each call.
    fn deterministic(&self) -> bool {
        false
    }
}

/// `zeroblob(N)` — a blob of `max(0, N)` zero bytes.
#[derive(Debug)]
struct ZeroBlob;
impl ScalarFunction for ZeroBlob {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        debug_assert!(args.len() == 1, "zeroblob/1 arity precondition");
        let n = to_integer(&args[0]);
        if n > MAX_BLOB_LEN {
            return Err(Error::sql("string or blob too big"));
        }
        Ok(Value::Blob(vec![0u8; n.max(0) as usize]))
    }
}

// ---------------------------------------------------------------------------
// Optimizer hints — all return their first argument unchanged
// ---------------------------------------------------------------------------

/// `likelihood(X, Y)` / `likely(X)` / `unlikely(X)` — planner hints with no
/// run-time effect; each returns `X` unchanged.
#[derive(Debug)]
struct Identity;
impl ScalarFunction for Identity {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        debug_assert!(!args.is_empty(), "optimizer hint needs at least one argument");
        Ok(args[0].clone())
    }
}

// ---------------------------------------------------------------------------
// Connection counters
// ---------------------------------------------------------------------------

/// `last_insert_rowid()` — the rowid of the most recent successful INSERT.
#[derive(Debug)]
struct LastInsertRowid;
impl ScalarFunction for LastInsertRowid {
    fn call(&self, _args: &[Value], ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(Value::Integer(ctx.last_insert_rowid()))
    }
    /// Non-deterministic: it reads connection state that MUTATES mid-statement during DML
    /// (each inserted row advances `last_insert_rowid`), so two calls with identical
    /// arguments within one statement can disagree. Memoizing a correlated subquery
    /// containing it — e.g. `UPDATE a SET x=(SELECT last_insert_rowid() ...)` — would freeze
    /// a stale value across outer rows (a wrong answer). Deterministic within a pure read,
    /// but the flag must hold for the worst case.
    fn deterministic(&self) -> bool {
        false
    }
}

/// `changes()` — rows changed by the most recent DML statement.
#[derive(Debug)]
struct Changes;
impl ScalarFunction for Changes {
    fn call(&self, _args: &[Value], ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(Value::Integer(ctx.changes()))
    }
    /// Non-deterministic for the same reason as `last_insert_rowid()`: the change counter
    /// advances per row during DML, so it is not a pure function of its (no) arguments
    /// within a statement.
    fn deterministic(&self) -> bool {
        false
    }
}

/// `total_changes()` — rows changed since the connection was opened.
#[derive(Debug)]
struct TotalChanges;
impl ScalarFunction for TotalChanges {
    fn call(&self, _args: &[Value], ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(Value::Integer(ctx.total_changes()))
    }
    /// Non-deterministic: the lifetime change counter advances per row during DML, so like
    /// `changes()`/`last_insert_rowid()` it can differ between two calls in one statement.
    fn deterministic(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Version constants
// ---------------------------------------------------------------------------

/// `sqlite_version()` — the (approximate) library version string.
#[derive(Debug)]
struct SqliteVersion;
impl ScalarFunction for SqliteVersion {
    fn call(&self, _args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(Value::Text(SQLITE_VERSION.to_string()))
    }
}

/// `sqlite_source_id()` — the (approximate) source-id string.
#[derive(Debug)]
struct SqliteSourceId;
impl ScalarFunction for SqliteSourceId {
    fn call(&self, _args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        Ok(Value::Text(SQLITE_SOURCE_ID.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic [`FnContext`] for tests: a fixed clock, an xorshift RNG with
    /// a fixed seed, and fixed counter values so assertions are stable.
    struct TestCtx {
        rng: u64,
        last_rowid: i64,
        changes: i64,
        total: i64,
    }
    impl TestCtx {
        fn new() -> Self {
            TestCtx { rng: 0x2545_F491_4F6C_DD1D, last_rowid: 42, changes: 7, total: 100 }
        }
    }
    impl FnContext for TestCtx {
        fn now_unix_millis(&self) -> i64 {
            1_700_000_000_000
        }
        fn random_i64(&mut self) -> i64 {
            // xorshift64* — deterministic, and never stuck at zero for our seed.
            let mut x = self.rng;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.rng = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D) as i64
        }
        fn fill_random(&mut self, buf: &mut [u8]) {
            for b in buf.iter_mut() {
                *b = self.random_i64() as u8;
            }
        }
        fn last_insert_rowid(&self) -> i64 {
            self.last_rowid
        }
        fn changes(&self) -> i64 {
            self.changes
        }
        fn total_changes(&self) -> i64 {
            self.total
        }
    }

    /// Call `f` with `args` against a fresh deterministic context, expecting `Ok`.
    fn call(f: &dyn ScalarFunction, args: &[Value]) -> Value {
        let mut ctx = TestCtx::new();
        f.call(args, &mut ctx).expect("scalar call should succeed")
    }

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

    #[test]
    fn typeof_reports_storage_class() {
        assert_eq!(text(&call(&Typeof, &[Value::Null])), "null");
        assert_eq!(text(&call(&Typeof, &[Value::Integer(1)])), "integer");
        assert_eq!(text(&call(&Typeof, &[Value::Real(1.0)])), "real");
        assert_eq!(text(&call(&Typeof, &[Value::Text("x".into())])), "text");
        assert_eq!(text(&call(&Typeof, &[Value::Blob(vec![0])])), "blob");
    }

    #[test]
    fn quote_all_storage_classes() {
        assert_eq!(text(&call(&Quote, &[Value::Null])), "NULL");
        assert_eq!(text(&call(&Quote, &[Value::Integer(-17)])), "-17");
        assert_eq!(text(&call(&Quote, &[Value::Real(1.5)])), "1.5");
        // A REAL reuses `real_to_text`, so its literal round-trips (harder value).
        assert_eq!(text(&call(&Quote, &[Value::Real(0.1)])), "0.1");
        // A plain string is single-quoted; interior quotes are doubled.
        assert_eq!(text(&call(&Quote, &[Value::Text("abc".into())])), "'abc'");
        assert_eq!(text(&call(&Quote, &[Value::Text("ab'c".into())])), "'ab''c'");
        // Blob -> X'..' uppercase hex.
        assert_eq!(text(&call(&Quote, &[Value::Blob(vec![0xAB, 0x01])])), "X'AB01'");
        assert_eq!(text(&call(&Quote, &[Value::Blob(vec![])])), "X''");
    }

    #[test]
    fn quote_truncates_text_at_first_nul() {
        // "ab\0cd" cannot be an SQL string literal past the NUL, so quote stops there.
        assert_eq!(text(&call(&Quote, &[Value::Text("ab\0cd".into())])), "'ab'");
    }

    #[test]
    fn hex_of_bytes_text_and_numbers() {
        assert_eq!(text(&call(&Hex, &[Value::Text("abc".into())])), "616263");
        assert_eq!(text(&call(&Hex, &[Value::Blob(vec![0xAB, 0x01])])), "AB01");
        // NULL renders as the empty string (matches sqlite).
        assert_eq!(text(&call(&Hex, &[Value::Null])), "");
        // An integer is rendered to text first, then hexed (spec example).
        assert_eq!(text(&call(&Hex, &[Value::Integer(12345678)])), "3132333435363738");
        // A REAL likewise renders via `real_to_text` before hexing: "2.5" -> 32 2E 35.
        assert_eq!(text(&call(&Hex, &[Value::Real(2.5)])), "322E35");
    }

    #[test]
    fn unhex_pure_and_ignore_set() {
        assert_eq!(blob(&call(&Unhex, &[Value::Text("616263".into())])), &[0x61, 0x62, 0x63]);
        // Mixed case is accepted.
        assert_eq!(blob(&call(&Unhex, &[Value::Text("aB01".into())])), &[0xAB, 0x01]);
        // Empty input -> empty blob.
        assert_eq!(blob(&call(&Unhex, &[Value::Text("".into())])), &[] as &[u8]);
        // With Y, its non-hex chars are ignored between pairs.
        let with_sep = call(&Unhex, &[Value::Text("41 42".into()), Value::Text(" ".into())]);
        assert_eq!(blob(&with_sep), &[0x41, 0x42]);
        // "hex digits in Y have no affect": a hex char listed in Y is still decoded
        // as hex in X, NOT stripped. unhex('41','1') == [0x41] (not '4' -> odd -> NULL).
        // This pins the hex-tested-before-ignore ordering against a "strip then
        // decode" refactor that every other unhex test would still pass.
        let y_has_hex = call(&Unhex, &[Value::Text("41".into()), Value::Text("1".into())]);
        assert_eq!(blob(&y_has_hex), &[0x41]);
    }

    #[test]
    fn unhex_null_on_bad_input() {
        // Odd number of hex digits.
        assert!(matches!(call(&Unhex, &[Value::Text("6".into())]), Value::Null));
        assert!(matches!(call(&Unhex, &[Value::Text("616".into())]), Value::Null));
        // Non-hex char with no ignore set.
        assert!(matches!(call(&Unhex, &[Value::Text("zz".into())]), Value::Null));
        // A separator that splits a pair is NULL even when it is in Y (adjacency).
        let split = call(&Unhex, &[Value::Text("4 1".into()), Value::Text(" ".into())]);
        assert!(matches!(split, Value::Null));
        // NULL in either argument -> NULL.
        assert!(matches!(call(&Unhex, &[Value::Null]), Value::Null));
        assert!(matches!(
            call(&Unhex, &[Value::Text("41".into()), Value::Null]),
            Value::Null
        ));
    }

    #[test]
    fn char_builds_string_from_code_points() {
        assert_eq!(text(&call(&Char, &[Value::Integer(0x41), Value::Integer(0x42)])), "AB");
        // Zero arguments -> empty string.
        assert_eq!(text(&call(&Char, &[])), "");
        // A multibyte code point (U+00E9 'é').
        assert_eq!(text(&call(&Char, &[Value::Integer(0x00E9)])), "é");
        // Invalid code points fold to U+FFFD: negative, surrogate, above U+10FFFF.
        assert_eq!(text(&call(&Char, &[Value::Integer(-1)])), "\u{FFFD}");
        assert_eq!(text(&call(&Char, &[Value::Integer(0xD800)])), "\u{FFFD}");
        assert_eq!(text(&call(&Char, &[Value::Integer(0x110000)])), "\u{FFFD}");
    }

    #[test]
    fn unicode_first_char_code_point() {
        assert_eq!(int(&call(&Unicode, &[Value::Text("A".into())])), 65);
        // First character only.
        assert_eq!(int(&call(&Unicode, &[Value::Text("Abc".into())])), 65);
        // Multibyte first char.
        assert_eq!(int(&call(&Unicode, &[Value::Text("é".into())])), 0x00E9);
        // Empty and NULL -> NULL.
        assert!(matches!(call(&Unicode, &[Value::Text("".into())]), Value::Null));
        assert!(matches!(call(&Unicode, &[Value::Null]), Value::Null));
    }

    #[test]
    fn random_reads_the_rng_not_a_constant() {
        // One shared context so successive calls advance the same RNG. The values
        // are deterministic but must differ call-to-call — a bug returning a
        // constant (or the wrong ctx field, e.g. a counter) would be caught.
        let mut ctx = TestCtx::new();
        let a = int(&Random.call(&[], &mut ctx).expect("ok"));
        let b = int(&Random.call(&[], &mut ctx).expect("ok"));
        assert_ne!(a, b, "random() must advance the RNG, not return a constant");
        // Not silently reading a fixed counter (42/7/100).
        assert!(a != 42 && a != 7 && a != 100, "random() returned a counter value: {a}");
    }

    #[test]
    fn randomblob_length_rules() {
        assert_eq!(blob(&call(&RandomBlob, &[Value::Integer(3)])).len(), 3);
        // N < 1 yields a 1-byte blob.
        assert_eq!(blob(&call(&RandomBlob, &[Value::Integer(0)])).len(), 1);
        assert_eq!(blob(&call(&RandomBlob, &[Value::Integer(-5)])).len(), 1);
    }

    #[test]
    fn nondeterministic_builtins_report_false_pure_ones_report_true() {
        // The single standing gate over this module's non-pure surface: every builtin whose
        // result can differ between two calls with identical arguments within one statement
        // MUST report `deterministic() == false`, else a correlated subquery containing it is
        // wrongly memoized (a stale-cache wrong answer). `random`/`randomblob` draw a fresh
        // value each call; the connection counters (last_insert_rowid/changes/total_changes)
        // MUTATE per row during DML (ops/{update,insert,delete}.rs advance them). If a future
        // non-pure builtin (or a re-added one) forgets the override, this fails — a loud
        // reminder that the determinism convention is hand-maintained (function.rs default is
        // `true`).
        let nondeterministic: &[&dyn ScalarFunction] =
            &[&Random, &RandomBlob, &LastInsertRowid, &Changes, &TotalChanges];
        for f in nondeterministic {
            assert!(!f.deterministic(), "{f:?} must be non-deterministic");
        }
        // Representative pure builtins in this module keep the trait default `true` (a
        // regression here would mean the default flipped or a pure fn wrongly overrode). A
        // pure function of its arguments (here `zeroblob`) and a constant (`sqlite_version`).
        let deterministic: &[&dyn ScalarFunction] = &[&ZeroBlob, &SqliteVersion];
        for f in deterministic {
            assert!(f.deterministic(), "{f:?} must be deterministic");
        }
    }

    #[test]
    fn zeroblob_is_n_zero_bytes() {
        assert_eq!(blob(&call(&ZeroBlob, &[Value::Integer(3)])), &[0, 0, 0]);
        // Negative N -> empty blob.
        assert_eq!(blob(&call(&ZeroBlob, &[Value::Integer(-2)])), &[] as &[u8]);
        assert_eq!(blob(&call(&ZeroBlob, &[Value::Integer(0)])), &[] as &[u8]);
    }

    #[test]
    fn blob_builders_reject_huge_sizes() {
        let mut ctx = TestCtx::new();
        let too_big = Value::Integer(MAX_BLOB_LEN + 1);
        match ZeroBlob.call(&[too_big.clone()], &mut ctx) {
            Err(Error::Sql(m)) => assert_eq!(m, "string or blob too big"),
            other => panic!("expected 'too big' error, got {other:?}"),
        }
        match RandomBlob.call(&[too_big], &mut ctx) {
            Err(Error::Sql(m)) => assert_eq!(m, "string or blob too big"),
            other => panic!("expected 'too big' error, got {other:?}"),
        }
    }

    #[test]
    fn optimizer_hints_return_first_argument() {
        // likely/unlikely (1 arg) and likelihood (2 args) all return arg[0].
        assert_eq!(int(&call(&Identity, &[Value::Integer(9)])), 9);
        assert_eq!(int(&call(&Identity, &[Value::Integer(9), Value::Real(0.5)])), 9);
        assert_eq!(text(&call(&Identity, &[Value::Text("hi".into())])), "hi");
    }

    #[test]
    fn connection_counters_read_context() {
        assert_eq!(int(&call(&LastInsertRowid, &[])), 42);
        assert_eq!(int(&call(&Changes, &[])), 7);
        assert_eq!(int(&call(&TotalChanges, &[])), 100);
    }

    #[test]
    fn version_constants_are_present() {
        assert_eq!(text(&call(&SqliteVersion, &[])), "3.50.0");
        // Source id is a fixed, non-empty "<timestamp> <hash>"-shaped constant.
        let sid = call(&SqliteSourceId, &[]);
        assert!(!text(&sid).is_empty());
    }

    #[test]
    fn registered_names_resolve_through_the_registry() {
        // The family registers exactly under these names/arities; a spot-check that
        // registration is wired (case-insensitive) through the public resolver.
        let reg = FunctionRegistry::builtins();
        for (name, argc) in [
            ("typeof", 1usize),
            ("QUOTE", 1),
            ("hex", 1),
            ("unhex", 2),
            ("char", 3),
            ("unicode", 1),
            ("random", 0),
            ("randomblob", 1),
            ("zeroblob", 1),
            ("likelihood", 2),
            ("likely", 1),
            ("unlikely", 1),
            ("last_insert_rowid", 0),
            ("changes", 0),
            ("total_changes", 0),
            ("sqlite_version", 0),
            ("sqlite_source_id", 0),
        ] {
            assert!(reg.resolve_scalar(name, argc).is_ok(), "{name}/{argc} should resolve");
        }
    }
}
