//! A recursive-descent parser for JSON text, understanding RFC-8259 plus the JSON5
//! extensions SQLite accepts (`spec/sqlite-doc/json1.html` §3.1, §3.6).
//!
//! The parser does three jobs the JSON functions need:
//!
//! * **Parse** text into a [`Json`] tree (`json`, `json_extract`, …).
//! * **Track canonicity** — whether any JSON5-only construct was used — so
//!   `json_valid(X,1)` (canonical only) and `json_valid(X,2)` (JSON5) can be
//!   distinguished from one parse (json1.html §4.21).
//! * **Report the first error position** for `json_error_position` (json1.html §4.7).
//!
//! Nesting is capped at 1000 levels (json1.html §3.5): deeper input is *invalid*,
//! and the cap also bounds recursion so hostile input cannot overflow the stack.
//!
//! Everything here is total: any byte sequence either parses or returns a
//! [`ParseError`] with a byte offset — never a panic (the crate's parser-must-not-
//! panic invariant). Unchecked indexing is avoided; bytes are read through `peek`.

use minisqlite_types::{integer_to_text, Error, Result, Value};

use super::value::{render_real, Json, NumLit, NumValue};

/// Maximum JSON nesting depth. Input nested deeper than this is invalid, matching
/// SQLite's recursive-descent limit (json1.html §3.5). The cap doubles as the
/// recursion bound that keeps a hostile `[[[[…` from overflowing the stack.
const MAX_DEPTH: usize = 1000;

/// A successful parse: the value plus whether any JSON5-only extension was used
/// (so `false` means the input was strictly canonical RFC-8259 JSON).
pub(crate) struct Parsed {
    pub(crate) json: Json,
    pub(crate) json5: bool,
}

/// A parse failure, carrying the 0-based byte offset of the first error. Callers
/// that surface a character position (`json_error_position`) convert it with
/// [`char_pos_of_byte`].
#[derive(Debug)]
pub(crate) struct ParseError {
    pub(crate) byte_pos: usize,
}

/// Parse `s` as a complete JSON (or JSON5) document. Leading and trailing
/// whitespace/comments are allowed; any non-whitespace after the top-level value is
/// an error.
pub(crate) fn parse_text(s: &str) -> std::result::Result<Parsed, ParseError> {
    let mut p = Parser::new(s);
    let json = p.parse_value()?;
    p.skip_ws()?;
    if p.pos != p.bytes.len() {
        return Err(ParseError { byte_pos: p.pos });
    }
    Ok(Parsed { json, json5: p.json5 })
}

/// Decode a double-quoted JSON string beginning at byte offset `at` in `s` (the byte
/// at `at` must be `"`). Returns the decoded content and the offset just past the
/// closing quote, or `None` if the quoted string is unterminated or malformed.
///
/// This is the SINGLE JSON string decoder — the same one `json`/`json_extract` use
/// for object keys — reused by the JSONPath label parser so a quoted label decodes
/// *identically* to the key it targets (surrogate pairs, `\v`, `\xHH`, and every
/// other escape included). Sharing one decoder is what keeps a label and its key
/// comparable; a second, drifting copy is exactly the "path silently misses the key"
/// bug this design removes.
pub(crate) fn decode_quoted(s: &str, at: usize) -> Option<(String, usize)> {
    let tail = s.get(at..)?;
    if tail.as_bytes().first() != Some(&b'"') {
        return None;
    }
    let mut p = Parser::new(tail);
    // parse_string consumes the opening quote at the cursor and stops past the close.
    let decoded = p.parse_string(b'"').ok()?;
    Some((decoded, at + p.pos))
}

/// Parse a JSON *argument* (the "json" parameter of a function) into a [`Json`].
/// SQL numbers are JSON numbers; TEXT is parsed as JSON (JSON5 accepted); a BLOB is
/// interpreted as text JSON (the documented legacy "JSON BLOB input" behavior,
/// json1.html §3.8 — real JSONB is deferred). Invalid text is `"malformed JSON"`.
///
/// A NULL value must be handled by the caller *before* calling this (most JSON
/// functions return SQL NULL for a NULL json argument); the `Null` arm maps to JSON
/// null only to keep the function total.
pub(crate) fn parse_json_arg(v: &Value) -> Result<Json> {
    match v {
        Value::Null => Ok(Json::Null),
        Value::Integer(i) => Ok(Json::Integer(*i)),
        Value::Real(r) => Ok(Json::Real(*r)),
        Value::Text(s) => parse_text(s).map(|p| p.json).map_err(|_| malformed()),
        Value::Blob(b) => {
            let s = String::from_utf8_lossy(b);
            parse_text(&s).map(|p| p.json).map_err(|_| malformed())
        }
    }
}

/// The SQLite error raised when a function that requires valid JSON is given text
/// that is not well-formed JSON.
pub(crate) fn malformed() -> Error {
    Error::sql("malformed JSON")
}

/// Convert a 0-based byte offset into the 1-based character position
/// `json_error_position` reports (json1.html §4.7: the left-most character is 1).
/// The offset is clamped to a char boundary so a partially-decoded position can
/// never panic.
pub(crate) fn char_pos_of_byte(s: &str, byte_pos: usize) -> usize {
    let mut bp = byte_pos.min(s.len());
    while bp > 0 && !s.is_char_boundary(bp) {
        bp -= 1;
    }
    s[..bp].chars().count() + 1
}

/// The mutable parse cursor over the input string. `bytes` is `s.as_bytes()` for
/// O(1) lookahead; multibyte scalars are decoded from `s` when a real character is
/// needed (string bodies, unquoted keys, identity escapes).
struct Parser<'a> {
    s: &'a str,
    bytes: &'a [u8],
    pos: usize,
    depth: usize,
    json5: bool,
}

impl<'a> Parser<'a> {
    fn new(s: &'a str) -> Self {
        Parser { s, bytes: s.as_bytes(), pos: 0, depth: 0, json5: false }
    }

    #[inline]
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    #[inline]
    fn peek_at(&self, off: usize) -> Option<u8> {
        self.bytes.get(self.pos + off).copied()
    }

    /// An error anchored at the current position.
    fn err<T>(&self) -> std::result::Result<T, ParseError> {
        Err(ParseError { byte_pos: self.pos })
    }

    /// Skip insignificant whitespace and JSON5 comments. The four RFC-8259
    /// whitespace bytes are canonical; vertical tab / form feed and `//`, `/* */`
    /// comments are JSON5 (they set `json5`). An unterminated block comment is an
    /// error. (Unicode whitespace beyond ASCII is not recognized — a known gap.)
    fn skip_ws(&mut self) -> std::result::Result<(), ParseError> {
        loop {
            match self.peek() {
                Some(b' ') | Some(b'\t') | Some(b'\n') | Some(b'\r') => self.pos += 1,
                Some(0x0b) | Some(0x0c) => {
                    self.json5 = true;
                    self.pos += 1;
                }
                Some(b'/') => match self.peek_at(1) {
                    Some(b'/') => {
                        self.json5 = true;
                        self.pos += 2;
                        while let Some(c) = self.peek() {
                            if c == b'\n' {
                                break;
                            }
                            self.pos += 1;
                        }
                    }
                    Some(b'*') => {
                        self.json5 = true;
                        self.pos += 2;
                        loop {
                            match self.peek() {
                                None => return self.err(),
                                Some(b'*') if self.peek_at(1) == Some(b'/') => {
                                    self.pos += 2;
                                    break;
                                }
                                Some(_) => self.pos += 1,
                            }
                        }
                    }
                    _ => break,
                },
                _ => break,
            }
        }
        Ok(())
    }

    /// Parse one JSON value (skipping leading whitespace/comments).
    fn parse_value(&mut self) -> std::result::Result<Json, ParseError> {
        self.skip_ws()?;
        match self.peek() {
            None => self.err(),
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b'"') => Ok(Json::Str(self.parse_string(b'"')?)),
            Some(b'\'') => {
                self.json5 = true;
                Ok(Json::Str(self.parse_string(b'\'')?))
            }
            Some(b't') => {
                if self.match_exact("true") {
                    Ok(Json::Bool(true))
                } else {
                    self.err()
                }
            }
            Some(b'f') => {
                if self.match_exact("false") {
                    Ok(Json::Bool(false))
                } else {
                    self.err()
                }
            }
            Some(b'n') => {
                if self.match_exact("null") {
                    Ok(Json::Null)
                } else if self.match_ci("nan") {
                    self.json5 = true;
                    Ok(Json::Null)
                } else {
                    self.err()
                }
            }
            Some(b'N') => {
                if self.match_ci("nan") {
                    self.json5 = true;
                    Ok(Json::Null)
                } else {
                    self.err()
                }
            }
            Some(b'I') | Some(b'i') => {
                if self.match_ci("infinity") || self.match_ci("inf") {
                    self.json5 = true;
                    Ok(real_number(render_real(f64::INFINITY), f64::INFINITY))
                } else {
                    self.err()
                }
            }
            Some(b'Q') | Some(b'q') | Some(b'S') | Some(b's') => {
                if self.match_ci("qnan") || self.match_ci("snan") {
                    self.json5 = true;
                    Ok(Json::Null)
                } else {
                    self.err()
                }
            }
            Some(b'-') | Some(b'+') | Some(b'.') | Some(b'0'..=b'9') => self.parse_number(),
            _ => self.err(),
        }
    }

    /// Match `kw` exactly (case-sensitive) at the cursor, advancing on success. Used
    /// for the canonical literals `true`/`false`/`null`, which are lower-case only.
    fn match_exact(&mut self, kw: &str) -> bool {
        let end = self.pos + kw.len();
        if self.bytes.len() >= end && &self.bytes[self.pos..end] == kw.as_bytes() {
            self.pos = end;
            true
        } else {
            false
        }
    }

    /// Match `kw` case-insensitively at the cursor, advancing on success. Used for
    /// the JSON5 float keywords (`Infinity`/`Inf`/`NaN`/`QNaN`/`SNaN`), which SQLite
    /// accepts in any case (json1.html §3.6).
    fn match_ci(&mut self, kw: &str) -> bool {
        let end = self.pos + kw.len();
        if self.bytes.len() >= end && self.bytes[self.pos..end].eq_ignore_ascii_case(kw.as_bytes()) {
            self.pos = end;
            true
        } else {
            false
        }
    }

    /// Parse an object: `{` members `}` with `"key": value` (or JSON5 single-quoted
    /// / unquoted keys), comma-separated, with an optional single trailing comma
    /// (JSON5). Members keep insertion order and duplicates are preserved.
    fn parse_object(&mut self) -> std::result::Result<Json, ParseError> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return self.err();
        }
        self.pos += 1; // consume '{'
        let mut members: Vec<(String, Json)> = Vec::new();
        self.skip_ws()?;
        if self.peek() == Some(b'}') {
            self.pos += 1;
            self.depth -= 1;
            return Ok(Json::Object(members));
        }
        loop {
            self.skip_ws()?;
            let key = self.parse_object_key()?;
            self.skip_ws()?;
            if self.peek() != Some(b':') {
                return self.err();
            }
            self.pos += 1;
            let val = self.parse_value()?;
            members.push((key, val));
            self.skip_ws()?;
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                    self.skip_ws()?;
                    if self.peek() == Some(b'}') {
                        self.json5 = true; // trailing comma
                        self.pos += 1;
                        break;
                    }
                }
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                _ => return self.err(),
            }
        }
        self.depth -= 1;
        Ok(Json::Object(members))
    }

    /// Parse an array: `[` values `]`, comma-separated, with an optional single
    /// trailing comma (JSON5).
    fn parse_array(&mut self) -> std::result::Result<Json, ParseError> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return self.err();
        }
        self.pos += 1; // consume '['
        let mut items: Vec<Json> = Vec::new();
        self.skip_ws()?;
        if self.peek() == Some(b']') {
            self.pos += 1;
            self.depth -= 1;
            return Ok(Json::Array(items));
        }
        loop {
            let val = self.parse_value()?;
            items.push(val);
            self.skip_ws()?;
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                    self.skip_ws()?;
                    if self.peek() == Some(b']') {
                        self.json5 = true; // trailing comma
                        self.pos += 1;
                        break;
                    }
                }
                Some(b']') => {
                    self.pos += 1;
                    break;
                }
                _ => return self.err(),
            }
        }
        self.depth -= 1;
        Ok(Json::Array(items))
    }

    /// Parse an object key: a quoted string, a JSON5 single-quoted string, or a
    /// JSON5 unquoted identifier.
    fn parse_object_key(&mut self) -> std::result::Result<String, ParseError> {
        match self.peek() {
            Some(b'"') => self.parse_string(b'"'),
            Some(b'\'') => {
                self.json5 = true;
                self.parse_string(b'\'')
            }
            Some(_) => self.parse_identifier(),
            None => self.err(),
        }
    }

    /// Parse a JSON5 unquoted object key. SQLite accepts ASCII identifier
    /// characters plus any non-whitespace character above U+007F (json1.html §3.6).
    fn parse_identifier(&mut self) -> std::result::Result<String, ParseError> {
        let rest = &self.s[self.pos..];
        let mut end = 0;
        for (i, ch) in rest.char_indices() {
            let ok = if i == 0 { is_ident_start(ch) } else { is_ident_part(ch) };
            if !ok {
                break;
            }
            end = i + ch.len_utf8();
        }
        if end == 0 {
            return self.err();
        }
        self.json5 = true;
        let key = rest[..end].to_string();
        self.pos += end;
        Ok(key)
    }

    /// Parse a string body (the opening quote is at the cursor) into decoded text.
    /// Handles RFC-8259 escapes and the JSON5 escapes SQLite accepts; a raw control
    /// character (`< 0x20`, including a bare newline) is invalid.
    fn parse_string(&mut self, quote: u8) -> std::result::Result<String, ParseError> {
        self.pos += 1; // opening quote
        let mut out = String::new();
        loop {
            match self.peek() {
                None => return self.err(), // unterminated
                Some(c) if c == quote => {
                    self.pos += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.pos += 1;
                    self.parse_escape(&mut out)?;
                }
                Some(c) if c < 0x20 => return self.err(),
                Some(_) => {
                    // Decode one whole UTF-8 scalar. `.get()` keeps this total by
                    // construction — no direct slice that could panic if `pos` were
                    // ever left off a char boundary.
                    let ch = match self.s.get(self.pos..).and_then(|s| s.chars().next()) {
                        Some(ch) => ch,
                        None => return self.err(),
                    };
                    self.pos += ch.len_utf8();
                    out.push(ch);
                }
            }
        }
    }

    /// Parse one escape sequence (the backslash is already consumed) and append the
    /// decoded character(s) to `out`. Standard JSON escapes are canonical; single
    /// quote, `\v`, `\0`, `\xHH`, a line continuation, and an identity escape of any
    /// other character are JSON5.
    fn parse_escape(&mut self, out: &mut String) -> std::result::Result<(), ParseError> {
        let ch = match self.s.get(self.pos..).and_then(|s| s.chars().next()) {
            Some(ch) => ch,
            None => return self.err(), // trailing backslash
        };
        match ch {
            '"' => {
                self.pos += 1;
                out.push('"');
            }
            '\\' => {
                self.pos += 1;
                out.push('\\');
            }
            '/' => {
                self.pos += 1;
                out.push('/');
            }
            'b' => {
                self.pos += 1;
                out.push('\u{08}');
            }
            'f' => {
                self.pos += 1;
                out.push('\u{0c}');
            }
            'n' => {
                self.pos += 1;
                out.push('\n');
            }
            'r' => {
                self.pos += 1;
                out.push('\r');
            }
            't' => {
                self.pos += 1;
                out.push('\t');
            }
            'u' => {
                self.pos += 1;
                let c = self.parse_unicode_escape()?;
                out.push(c);
            }
            '\'' => {
                self.pos += 1;
                self.json5 = true;
                out.push('\'');
            }
            'v' => {
                self.pos += 1;
                self.json5 = true;
                out.push('\u{0b}');
            }
            '0' => {
                self.pos += 1;
                if matches!(self.peek(), Some(d) if d.is_ascii_digit()) {
                    return self.err();
                }
                self.json5 = true;
                out.push('\0');
            }
            'x' => {
                self.pos += 1;
                self.json5 = true;
                let c = self.parse_hex2()?;
                out.push(c);
            }
            '\n' => {
                self.pos += 1;
                self.json5 = true; // line continuation: the newline is elided
            }
            '\r' => {
                self.pos += 1;
                self.json5 = true;
                if self.peek() == Some(b'\n') {
                    self.pos += 1; // CRLF continuation
                }
            }
            d if d.is_ascii_digit() => return self.err(), // \1..\9 are not valid escapes
            other => {
                // JSON5 identity escape of any other character (e.g. `\A` -> `A`).
                self.pos += other.len_utf8();
                self.json5 = true;
                out.push(other);
            }
        }
        Ok(())
    }

    /// Parse a `\uXXXX` escape (the `u` is already consumed), combining a
    /// high+low surrogate pair into one scalar. A lone or malformed surrogate
    /// decodes to U+FFFD rather than erroring, keeping the parser total on the
    /// (technically malformed) input real parsers tolerate.
    fn parse_unicode_escape(&mut self) -> std::result::Result<char, ParseError> {
        let hi = self.read_hex(4)?;
        if (0xD800..=0xDBFF).contains(&hi) {
            let save = self.pos;
            if self.peek() == Some(b'\\') && self.peek_at(1) == Some(b'u') {
                self.pos += 2;
                if let Ok(lo) = self.read_hex(4) {
                    if (0xDC00..=0xDFFF).contains(&lo) {
                        let c = 0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00);
                        return Ok(char::from_u32(c).unwrap_or('\u{FFFD}'));
                    }
                }
                self.pos = save; // not a low surrogate: leave it for the next iteration
            }
            return Ok('\u{FFFD}');
        }
        if (0xDC00..=0xDFFF).contains(&hi) {
            return Ok('\u{FFFD}'); // lone low surrogate
        }
        Ok(char::from_u32(hi).unwrap_or('\u{FFFD}'))
    }

    /// Read exactly `n` hex digits as a `u32`, advancing past them. The error is
    /// anchored at the first non-hex digit (or at the cursor if input ends early) so
    /// `json_error_position` points at the offending byte. Shared by `\uXXXX`
    /// (`n = 4`) and `\xHH` (`n = 2`).
    fn read_hex(&mut self, n: usize) -> std::result::Result<u32, ParseError> {
        if self.pos + n > self.bytes.len() {
            return self.err();
        }
        let mut v = 0u32;
        for i in 0..n {
            match (self.bytes[self.pos + i] as char).to_digit(16) {
                Some(d) => v = v * 16 + d,
                None => return Err(ParseError { byte_pos: self.pos + i }),
            }
        }
        self.pos += n;
        Ok(v)
    }

    /// Read a `\xHH` Latin-1 code point (two hex digits, JSON5).
    fn parse_hex2(&mut self) -> std::result::Result<char, ParseError> {
        let v = self.read_hex(2)?;
        Ok(char::from_u32(v).unwrap_or('\u{FFFD}'))
    }

    /// Parse a number: an optional sign, then a decimal integer/real, a JSON5 hex
    /// integer, or a JSON5 `Infinity`/`NaN` keyword. Integer-vs-real is decided
    /// structurally (a `.` or exponent means real). A NaN keyword parses to JSON
    /// null (json1.html §3.6).
    ///
    /// **Number tokens are preserved verbatim for canonical (RFC-8259) input** and
    /// only canonicalized for JSON5 spellings (json1.html §4.1: `json()` merely
    /// strips whitespace from canonical JSON). So `1e3`, `2.00`, and a >i64 integer
    /// all round-trip through a minify unchanged; a JSON5 form (`0x1F`, `+5`, `.5`,
    /// `Infinity`) is rewritten to canonical text. Canonicity is tracked *per number*
    /// via `num_json5`, not via the document-wide `self.json5`, so a canonical number
    /// beside a JSON5 construct still renders verbatim.
    fn parse_number(&mut self) -> std::result::Result<Json, ParseError> {
        let start = self.pos;
        // Whether THIS number used a JSON5-only spelling; drives verbatim-vs-
        // canonical rendering, independently of the document-wide `self.json5`.
        let mut num_json5 = false;
        let mut sign_neg = false;
        match self.peek() {
            Some(b'+') => {
                self.mark_json5(&mut num_json5);
                self.pos += 1;
            }
            Some(b'-') => {
                sign_neg = true;
                self.pos += 1;
            }
            _ => {}
        }
        // Infinity / NaN after an optional sign (JSON5): rendered canonically.
        match self.peek() {
            Some(b'I') | Some(b'i') => {
                if self.match_ci("infinity") || self.match_ci("inf") {
                    self.json5 = true;
                    let v = if sign_neg { f64::NEG_INFINITY } else { f64::INFINITY };
                    return Ok(real_number(render_real(v), v));
                }
                return Err(ParseError { byte_pos: start });
            }
            Some(b'N') | Some(b'n') | Some(b'Q') | Some(b'q') | Some(b'S') | Some(b's') => {
                if self.match_ci("nan") || self.match_ci("qnan") || self.match_ci("snan") {
                    self.json5 = true;
                    return Ok(Json::Null);
                }
                return Err(ParseError { byte_pos: start });
            }
            _ => {}
        }
        // Hex integer (JSON5): 0x… — canonicalized to a decimal integer token.
        if self.peek() == Some(b'0') && matches!(self.peek_at(1), Some(b'x') | Some(b'X')) {
            self.json5 = true;
            self.pos += 2;
            let hstart = self.pos;
            while matches!(self.peek(), Some(c) if c.is_ascii_hexdigit()) {
                self.pos += 1;
            }
            if self.pos == hstart {
                return Err(ParseError { byte_pos: start });
            }
            let hexstr = &self.s[hstart..self.pos];
            let mag =
                i128::from_str_radix(hexstr, 16).map_err(|_| ParseError { byte_pos: start })?;
            let val = if sign_neg { -mag } else { mag };
            return Ok(number_from_i128(val));
        }
        // Decimal integer part.
        let int_start = self.pos;
        let first_int = self.peek();
        let mut has_int_digits = false;
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            has_int_digits = true;
            self.pos += 1;
        }
        // Leading-zero rule: "0" immediately followed by another digit is invalid
        // (RFC-8259, and JSON5 for decimals).
        if first_int == Some(b'0') && self.pos - int_start > 1 {
            return Err(ParseError { byte_pos: int_start });
        }
        let mut has_dot = false;
        let mut has_frac_digits = false;
        if self.peek() == Some(b'.') {
            has_dot = true;
            self.pos += 1;
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                has_frac_digits = true;
                self.pos += 1;
            }
        }
        if !has_int_digits && !has_frac_digits {
            return Err(ParseError { byte_pos: start });
        }
        // A leading `.5` or trailing `5.` decimal point is JSON5.
        if has_dot && (!has_int_digits || !has_frac_digits) {
            self.mark_json5(&mut num_json5);
        }
        let mut has_exp = false;
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            let esave = self.pos;
            self.pos += 1;
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.pos += 1;
            }
            let exp_start = self.pos;
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.pos += 1;
            }
            if self.pos == exp_start {
                return Err(ParseError { byte_pos: esave });
            }
            has_exp = true;
        }
        let raw = &self.s[start..self.pos];
        if !has_dot && !has_exp {
            // Integer token. `digits` (a JSON5 '+' stripped) is the canonical
            // spelling and the source of the typed value; canonical input renders
            // the raw slice verbatim.
            let digits = raw.strip_prefix('+').unwrap_or(raw);
            let text = if num_json5 { digits.to_string() } else { raw.to_string() };
            Ok(int_number(text, digits))
        } else {
            // Real token. Canonical input renders verbatim; a JSON5 spelling
            // (`.5`, `5.`, `+1.0`) is canonicalized through the float renderer.
            let f = parse_decimal_f64(raw);
            let text = if num_json5 { render_real(f) } else { raw.to_string() };
            Ok(real_number(text, f))
        }
    }

    /// Mark that a JSON5-only construct was used: set both the document-wide
    /// `json5` flag (which `json_valid` reads) and the caller's per-number flag
    /// (which decides verbatim-vs-canonical rendering of the number in hand).
    fn mark_json5(&mut self, num_json5: &mut bool) {
        self.json5 = true;
        *num_json5 = true;
    }
}

/// Build an integer-token [`Json::Number`]: `text` is the exact rendering (verbatim
/// source for canonical, canonicalized for JSON5) and `digits` is the sign-normalized
/// integer spelling used to derive the SQL value. A token that overflows `i64`
/// carries a REAL value but keeps `json_type` 'integer' and its text intact — so
/// `json('9223372036854775808')` round-trips even though the value can't be an i64.
fn int_number(text: String, digits: &str) -> Json {
    let value = match digits.parse::<i64>() {
        Ok(i) => NumValue::Int(i),
        Err(_) => NumValue::Real(parse_decimal_f64(digits)),
    };
    Json::Number(NumLit { text, is_integer: true, value })
}

/// Build a real-token [`Json::Number`] with rendering `text` and value `f`.
fn real_number(text: String, f: f64) -> Json {
    Json::Number(NumLit { text, is_integer: false, value: NumValue::Real(f) })
}

/// Build a [`Json::Number`] from a JSON5 hex magnitude (already sign-applied),
/// canonicalized to decimal text: an `i64`-fitting value is an integer token, and a
/// larger magnitude keeps 'integer' type with a REAL value and its decimal spelling.
fn number_from_i128(v: i128) -> Json {
    match i64::try_from(v) {
        Ok(i) => Json::Number(NumLit {
            text: integer_to_text(i),
            is_integer: true,
            value: NumValue::Int(i),
        }),
        Err(_) => Json::Number(NumLit {
            text: v.to_string(),
            is_integer: true,
            value: NumValue::Real(v as f64),
        }),
    }
}

/// Whether `ch` may start a JSON5 unquoted identifier.
fn is_ident_start(ch: char) -> bool {
    ch.is_ascii_alphabetic() || ch == '_' || ch == '$' || (ch as u32 > 0x7f && !ch.is_whitespace())
}

/// Whether `ch` may continue a JSON5 unquoted identifier.
fn is_ident_part(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_' || ch == '$' || (ch as u32 > 0x7f && !ch.is_whitespace())
}

/// Parse a scanned decimal real token to `f64`. The token is already structurally
/// valid; a leading `+` is stripped and bare-dot forms (`.5`, `5.`, `5.e3`) are
/// normalized so `f64::from_str` accepts them regardless of platform leniency.
fn parse_decimal_f64(raw: &str) -> f64 {
    let raw = raw.strip_prefix('+').unwrap_or(raw);
    if let Ok(v) = raw.parse::<f64>() {
        return v;
    }
    let (sign, body) = match raw.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", raw),
    };
    let (mantissa, exp) = match body.find(['e', 'E']) {
        Some(i) => (&body[..i], &body[i..]),
        None => (body, ""),
    };
    let mantissa = if mantissa.is_empty() { "0" } else { mantissa };
    let norm_mant = if let Some(dot) = mantissa.find('.') {
        let ip = &mantissa[..dot];
        let fp = &mantissa[dot + 1..];
        let ip = if ip.is_empty() { "0" } else { ip };
        let fp = if fp.is_empty() { "0" } else { fp };
        format!("{ip}.{fp}")
    } else {
        mantissa.to_string()
    };
    match format!("{sign}{norm_mant}{exp}").parse::<f64>() {
        Ok(v) => v,
        Err(_) => {
            // The scanner only calls this on a structurally valid real token (a huge
            // magnitude parses to ±inf, a tiny one to 0.0 — neither errors), so a
            // parse failure means the scanner and this normalizer have drifted. Catch
            // that in debug; stay total (no panic) in release per the parser invariant.
            debug_assert!(false, "scanned real token failed to parse: {raw:?}");
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse `s`, expecting success, returning the value.
    fn ok(s: &str) -> Json {
        parse_text(s).unwrap_or_else(|_| panic!("{s:?} should parse")).json
    }

    /// Parse `s` and render it back to minified JSON — exactly what `json()` does.
    fn mini(s: &str) -> String {
        ok(s).to_text()
    }

    /// Whether `s` parses at all.
    fn is_ok(s: &str) -> bool {
        parse_text(s).is_ok()
    }

    /// Whether `s` parses AND used only canonical (non-JSON5) syntax.
    fn is_canonical(s: &str) -> bool {
        matches!(parse_text(s), Ok(p) if !p.json5)
    }

    #[test]
    fn parses_scalars() {
        assert_eq!(ok("null"), Json::Null);
        assert_eq!(ok("true"), Json::Bool(true));
        assert_eq!(ok("false"), Json::Bool(false));
        assert_eq!(ok("\"hi\""), Json::Str("hi".into()));
        // Canonical number tokens render back verbatim (only whitespace is stripped).
        assert_eq!(mini("123"), "123");
        assert_eq!(mini("-7"), "-7");
        assert_eq!(mini("3.5"), "3.5");
        assert_eq!(mini("1e3"), "1e3");
        assert_eq!(mini("  42  "), "42");
        // Typed access still classifies and coerces correctly.
        assert_eq!(ok("123").type_name(), "integer");
        assert_eq!(ok("3.5").type_name(), "real");
        assert!(matches!(ok("-7").to_sql_scalar(), Value::Integer(-7)));
        assert!(matches!(ok("1e3").to_sql_scalar(), Value::Real(r) if r == 1000.0));
    }

    #[test]
    fn parses_containers_preserving_order_and_duplicates() {
        // Container round-trips preserve every canonical number token verbatim.
        assert_eq!(mini(r#"[1,2,"3",4]"#), r#"[1,2,"3",4]"#);
        // Object keeps insertion order and does not de-duplicate.
        assert_eq!(mini(r#"{"a":1,"a":2}"#), r#"{"a":1,"a":2}"#);
        match ok(r#"{"a":1,"a":2}"#) {
            Json::Object(m) => assert_eq!(m.len(), 2, "duplicate keys must be preserved"),
            other => panic!("expected object, got {other:?}"),
        }
    }

    #[test]
    fn canonical_number_tokens_are_preserved_by_minify() {
        // json1.html §4.1: json() only strips whitespace for canonical RFC-8259
        // input, so a canonical number token must survive a minify verbatim
        // (regression: these were reformatted through f64/i64).
        assert_eq!(parse_text("1e3").unwrap().json.to_text(), "1e3");
        assert_eq!(parse_text("[1.50]").unwrap().json.to_text(), "[1.50]");
        assert_eq!(parse_text("2.00").unwrap().json.to_text(), "2.00");
        assert_eq!(parse_text("-0").unwrap().json.to_text(), "-0");
        // A valid RFC-8259 integer larger than i64 keeps its digits and 'integer'
        // type, even though the extracted SQL value must fall back to REAL.
        let big = parse_text("9223372036854775808").unwrap().json;
        assert_eq!(big.to_text(), "9223372036854775808");
        assert_eq!(big.type_name(), "integer");
        assert!(matches!(big.to_sql_scalar(), Value::Real(_)));
    }

    #[test]
    fn decodes_string_escapes() {
        assert_eq!(ok(r#""a\tb\nc""#), Json::Str("a\tb\nc".into()));
        assert_eq!(ok(r#""\u0041""#), Json::Str("A".into()));
        assert_eq!(ok(r#""quote:\"""#), Json::Str("quote:\"".into()));
        // Surrogate pair -> astral character (U+1F600).
        assert_eq!(ok(r#""\ud83d\ude00""#), Json::Str("\u{1F600}".into()));
    }

    #[test]
    fn canonical_vs_json5_classification() {
        // Canonical inputs.
        assert!(is_canonical(r#"{"x":35}"#));
        assert!(is_canonical(r#"[1,2,3]"#));
        assert!(is_canonical("3.14"));
        // JSON5-only constructs parse but are NOT canonical.
        assert!(is_ok("{x:35}") && !is_canonical("{x:35}")); // unquoted key
        assert!(is_ok("[1,2,3,]") && !is_canonical("[1,2,3,]")); // trailing comma
        assert!(is_ok("'single'") && !is_canonical("'single'")); // single quotes
        assert!(is_ok("0x1F") && !is_canonical("0x1F")); // hex
        assert!(is_ok(".5") && !is_canonical(".5")); // leading dot
        assert!(is_ok("+5") && !is_canonical("+5")); // leading plus
        assert!(is_ok("Infinity") && !is_canonical("Infinity"));
        assert!(is_ok("{a:1}//comment\n") && !is_canonical("{a:1}//comment"));
    }

    #[test]
    fn json5_number_forms() {
        // JSON5 spellings ARE canonicalized on render (unlike canonical tokens).
        assert_eq!(mini("0x1F"), "31");
        assert_eq!(mini(".5"), "0.5");
        assert_eq!(mini("5."), "5.0");
        assert_eq!(mini("+5"), "5");
        assert_eq!(mini("Infinity"), "9.0e+999");
        assert_eq!(mini("-Infinity"), "-9.0e+999");
        assert_eq!(mini("Inf"), "9.0e+999");
        // The typed values behind the canonicalized spellings.
        assert_eq!(ok("0x1F").type_name(), "integer");
        assert!(matches!(ok("0x1F").to_sql_scalar(), Value::Integer(31)));
        assert!(matches!(ok(".5").to_sql_scalar(), Value::Real(r) if r == 0.5));
        // NaN in any spelling is treated as JSON null.
        assert_eq!(ok("NaN"), Json::Null);
        assert_eq!(ok("QNaN"), Json::Null);
    }

    #[test]
    fn json5_number_beside_canonical_number_keeps_each_form() {
        // A JSON5 number in the same document must not make a *canonical* sibling
        // render canonically — canonicity is tracked per number, not per document.
        assert_eq!(mini("[+5, 1e3, 2.00]"), "[5,1e3,2.00]");
    }

    #[test]
    fn rejects_malformed_input() {
        for bad in [
            "", "{", "[1,2", "{\"x\":}", "tru", "nul", "01", "1.2.3", "\"unterminated", "{,}",
            "[,]", "1e", "--1", "\"a\tb\"", // raw tab in string
        ] {
            assert!(!is_ok(bad), "{bad:?} should be rejected");
        }
    }

    #[test]
    fn nesting_depth_is_capped() {
        // 1000 levels parse; 1001 do not (json1.html §3.5).
        let ok_deep = format!("{}{}", "[".repeat(1000), "]".repeat(1000));
        assert!(is_ok(&ok_deep), "1000 levels should parse");
        let too_deep = format!("{}{}", "[".repeat(1001), "]".repeat(1001));
        assert!(!is_ok(&too_deep), "1001 levels should be rejected");
    }

    #[test]
    fn error_positions_are_one_based_chars() {
        // "[1,2" — the error is at end of input (position 5).
        match parse_text("[1,2") {
            Err(e) => assert_eq!(char_pos_of_byte("[1,2", e.byte_pos), 5),
            Ok(_) => panic!("should fail"),
        }
        // Leading multibyte char then an error keeps character (not byte) counting.
        let s = "é@"; // 'é' is 2 bytes; '@' at char position 2 is the first error
        match parse_text(s) {
            Err(e) => assert_eq!(char_pos_of_byte(s, e.byte_pos), 1),
            Ok(_) => panic!("should fail"),
        }
    }

    #[test]
    fn parse_json_arg_handles_sql_values() {
        // A SQL numeric value becomes a formatted number (Integer/Real provenance).
        assert_eq!(parse_json_arg(&Value::Integer(5)).unwrap(), Json::Integer(5));
        assert_eq!(parse_json_arg(&Value::Real(2.5)).unwrap(), Json::Real(2.5));
        // TEXT is parsed as JSON; numbers inside keep their canonical token.
        assert_eq!(parse_json_arg(&Value::Text("[1,2]".into())).unwrap().to_text(), "[1,2]");
        // A BLOB is interpreted as text JSON (legacy bug, json1.html §3.8).
        assert_eq!(parse_json_arg(&Value::Blob(b"[1]".to_vec())).unwrap().to_text(), "[1]");
        // Malformed text is an error.
        assert!(matches!(parse_json_arg(&Value::Text("{".into())), Err(Error::Sql(_))));
    }

    #[test]
    fn does_not_panic_on_hostile_bytes() {
        // A spread of tricky inputs must return (Ok or Err), never panic.
        for s in ["\\", "\"\\u", "\"\\uD800\"", "\"\\x\"", "/*", "0x", "\"\\", "'\\", "{\"a\""] {
            let _ = parse_text(s);
        }
    }
}
