//! Tokenizer — SQLite lexical rules (`spec/sqlite-doc/lang_keywords.html` plus the
//! lexical notes scattered through the language reference).
//!
//! A [`Token`] carries only its kind and byte span. Unquoted identifiers and
//! keywords keep no owned text: the parser recovers their spelling by slicing the
//! original SQL with the span (preserving original case, allocating nothing).
//! Quoted identifiers, string literals and blobs carry their *decoded* payload,
//! because the decoded form differs from the raw source slice.

use minisqlite_types::{Error, Result};

use crate::ast_expr::BindParam;
use crate::keyword::Keyword;

/// How an identifier was quoted. Only `Double` participates in SQLite's
/// double-quote fallback (identifier if resolvable, else string literal).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuoteKind {
    /// `"name"`
    Double,
    /// `[name]`
    Bracket,
    /// `` `name` ``
    Backtick,
}

/// The lexical category of a token, with any decoded payload.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    /// Decimal or `0x` hex integer that fits in `i64`.
    Integer(i64),
    /// Real literal, or an integer literal too large for `i64` (SQLite promotes
    /// an overflowing integer literal to a real).
    Real(f64),
    /// `'...'` string literal, decoded (`''` collapsed to `'`).
    Str(String),
    /// `x'..'` / `X'..'` blob literal, decoded to raw bytes.
    Blob(Vec<u8>),
    /// Unquoted identifier; spelling is the source slice `[start, end)`.
    Ident,
    /// Quoted identifier, decoded (delimiter removed, doubled delimiter collapsed).
    QuotedIdent { value: String, quote: QuoteKind },
    /// A keyword; original spelling is the source slice `[start, end)`.
    Keyword(Keyword),
    /// A bind parameter: `?`, `?NNN`, `:name`, `@name`, `$name`.
    Param(BindParam),

    // Operators and punctuation.
    Plus,       // +
    Minus,      // -
    Star,       // *
    Slash,      // /
    Percent,    // %
    Concat,     // ||
    Arrow,      // ->
    Arrow2,     // ->>
    Amp,        // &
    Pipe,       // |
    Shl,        // <<
    Shr,        // >>
    Tilde,      // ~
    Eq,         // = or ==
    Ne,         // != or <>
    Lt,         // <
    LtEq,       // <=
    Gt,         // >
    GtEq,       // >=
    LParen,     // (
    RParen,     // )
    Comma,      // ,
    Dot,        // .
    Semicolon,  // ;
    /// End of input sentinel (always the final token).
    Eof,
}

/// A lexical token: its kind and the byte range it spans in the source.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub start: usize,
    pub end: usize,
}

impl Token {
    /// This token's raw source lexeme.
    pub fn text<'a>(&self, src: &'a str) -> &'a str {
        &src[self.start..self.end]
    }

    /// The keyword this token is, if any.
    pub fn keyword(&self) -> Option<Keyword> {
        match self.kind {
            TokenKind::Keyword(k) => Some(k),
            _ => None,
        }
    }
}

/// Tokenize an entire SQL string into a token stream terminated by [`TokenKind::Eof`].
///
/// Skips whitespace and both comment forms (`-- ...` to end of line, `/* ... */`
/// with an unterminated trailing block allowed, matching SQLite). Byte offsets are
/// tracked for error reporting.
pub fn tokenize(sql: &str) -> Result<Vec<Token>> {
    Lexer::new(sql).run()
}

struct Lexer<'a> {
    src: &'a str,
    bytes: &'a [u8],
    pos: usize,
}

fn is_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0c)
}

/// A byte that may *start* an unquoted identifier: ASCII letter, `_`, or any
/// non-ASCII byte (SQLite treats bytes >= 0x80 as identifier characters).
fn is_ident_start(b: u8) -> bool {
    b == b'_' || b.is_ascii_alphabetic() || b >= 0x80
}

/// A byte that may *continue* an unquoted identifier (adds digits and `$`).
fn is_ident_continue(b: u8) -> bool {
    is_ident_start(b) || b.is_ascii_digit() || b == b'$'
}

/// Drop the readability `_` separators from a numeric literal before it is parsed.
/// Borrows the common no-underscore case so the hot path stays allocation-free.
fn strip_underscores(s: &str) -> std::borrow::Cow<'_, str> {
    if s.as_bytes().contains(&b'_') {
        std::borrow::Cow::Owned(s.chars().filter(|&c| c != '_').collect())
    } else {
        std::borrow::Cow::Borrowed(s)
    }
}

impl<'a> Lexer<'a> {
    fn new(src: &'a str) -> Self {
        Lexer { src, bytes: src.as_bytes(), pos: 0 }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn peek_at(&self, off: usize) -> Option<u8> {
        self.bytes.get(self.pos + off).copied()
    }

    fn err<T>(&self, msg: impl Into<String>) -> Result<T> {
        Err(Error::Sql(msg.into()))
    }

    fn run(mut self) -> Result<Vec<Token>> {
        let mut out = Vec::new();
        loop {
            self.skip_trivia()?;
            let start = self.pos;
            let Some(b) = self.peek() else {
                out.push(Token { kind: TokenKind::Eof, start, end: start });
                return Ok(out);
            };
            let kind = self.next_kind(b)?;
            out.push(Token { kind, start, end: self.pos });
        }
    }

    /// Skip whitespace and comments. A `/* ... */` left unterminated at EOF is
    /// accepted (SQLite does the same).
    fn skip_trivia(&mut self) -> Result<()> {
        loop {
            match self.peek() {
                Some(b) if is_space(b) => {
                    self.pos += 1;
                }
                Some(b'-') if self.peek_at(1) == Some(b'-') => {
                    // Line comment to end of line (or EOF).
                    self.pos += 2;
                    while let Some(b) = self.peek() {
                        self.pos += 1;
                        if b == b'\n' {
                            break;
                        }
                    }
                }
                Some(b'/') if self.peek_at(1) == Some(b'*') => {
                    self.pos += 2;
                    loop {
                        match self.peek() {
                            Some(b'*') if self.peek_at(1) == Some(b'/') => {
                                self.pos += 2;
                                break;
                            }
                            Some(_) => self.pos += 1,
                            // Unterminated block comment at EOF: allowed.
                            None => break,
                        }
                    }
                }
                _ => return Ok(()),
            }
        }
    }

    fn next_kind(&mut self, b: u8) -> Result<TokenKind> {
        match b {
            b'\'' => self.string_literal(),
            b'"' => self.quoted_ident(b'"', QuoteKind::Double),
            b'[' => self.bracket_ident(),
            b'`' => self.quoted_ident(b'`', QuoteKind::Backtick),
            b'?' | b':' | b'@' | b'$' => self.bind_param(b),
            b'0'..=b'9' => self.number(),
            b'.' if self.peek_at(1).is_some_and(|c| c.is_ascii_digit()) => self.number(),
            _ if (b == b'x' || b == b'X') && self.peek_at(1) == Some(b'\'') => self.blob_literal(),
            _ if is_ident_start(b) => Ok(self.ident_or_keyword()),
            _ => self.operator(b),
        }
    }

    /// `'...'` string literal with `''` escape.
    fn string_literal(&mut self) -> Result<TokenKind> {
        self.pos += 1; // opening quote
        let mut s = String::new();
        loop {
            match self.peek() {
                None => return self.err("unterminated string literal"),
                Some(b'\'') => {
                    if self.peek_at(1) == Some(b'\'') {
                        s.push('\'');
                        self.pos += 2;
                    } else {
                        self.pos += 1; // closing quote
                        return Ok(TokenKind::Str(s));
                    }
                }
                Some(_) => {
                    let ch = self.take_char();
                    s.push(ch);
                }
            }
        }
    }

    /// A quoted identifier delimited by `delim` (`"` or `` ` ``) where a doubled
    /// delimiter escapes one.
    fn quoted_ident(&mut self, delim: u8, quote: QuoteKind) -> Result<TokenKind> {
        self.pos += 1; // opening delimiter
        let mut value = String::new();
        loop {
            match self.peek() {
                None => return self.err("unterminated quoted identifier"),
                Some(c) if c == delim => {
                    if self.peek_at(1) == Some(delim) {
                        value.push(delim as char);
                        self.pos += 2;
                    } else {
                        self.pos += 1;
                        return Ok(TokenKind::QuotedIdent { value, quote });
                    }
                }
                Some(_) => {
                    let ch = self.take_char();
                    value.push(ch);
                }
            }
        }
    }

    /// `[...]` identifier: everything up to the first `]`, no escape mechanism.
    fn bracket_ident(&mut self) -> Result<TokenKind> {
        self.pos += 1; // '['
        let mut value = String::new();
        loop {
            match self.peek() {
                None => return self.err("unterminated [ ] identifier"),
                Some(b']') => {
                    self.pos += 1;
                    return Ok(TokenKind::QuotedIdent { value, quote: QuoteKind::Bracket });
                }
                Some(_) => {
                    let ch = self.take_char();
                    value.push(ch);
                }
            }
        }
    }

    /// `x'..'` / `X'..'` blob literal — an even number of hex digits.
    fn blob_literal(&mut self) -> Result<TokenKind> {
        self.pos += 2; // x'
        let start = self.pos;
        while let Some(b) = self.peek() {
            if b == b'\'' {
                break;
            }
            if !b.is_ascii_hexdigit() {
                return self.err("malformed blob literal: non-hex digit");
            }
            self.pos += 1;
        }
        if self.peek() != Some(b'\'') {
            return self.err("unterminated blob literal");
        }
        let hex = &self.bytes[start..self.pos];
        self.pos += 1; // closing quote
        if hex.len() % 2 != 0 {
            return self.err("malformed blob literal: odd number of hex digits");
        }
        let mut out = Vec::with_capacity(hex.len() / 2);
        for pair in hex.chunks_exact(2) {
            let hi = hex_val(pair[0]);
            let lo = hex_val(pair[1]);
            out.push((hi << 4) | lo);
        }
        Ok(TokenKind::Blob(out))
    }

    /// `?`, `?NNN`, `:name`, `@name`, `$name`.
    fn bind_param(&mut self, sigil: u8) -> Result<TokenKind> {
        if sigil == b'?' {
            self.pos += 1;
            if self.peek().is_some_and(|c| c.is_ascii_digit()) {
                let start = self.pos;
                while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                    self.pos += 1;
                }
                let digits = &self.src[start..self.pos];
                let n: u32 = digits
                    .parse()
                    .map_err(|_| Error::Sql(format!("parameter number too large: ?{digits}")))?;
                return Ok(TokenKind::Param(BindParam::Numbered(n)));
            }
            return Ok(TokenKind::Param(BindParam::Anonymous));
        }
        // :name / @name / $name — sigil kept in the stored name so `:x` and `$x`
        // remain distinct parameters.
        let start = self.pos;
        self.pos += 1; // sigil
        let name_start = self.pos;
        while self.peek().is_some_and(|c| c == b'_' || c.is_ascii_alphanumeric() || c >= 0x80) {
            self.pos += 1;
        }
        if self.pos == name_start {
            return self.err(format!("bind parameter '{}' has no name", sigil as char));
        }
        Ok(TokenKind::Param(BindParam::Named(self.src[start..self.pos].to_string())))
    }

    /// A numeric literal: decimal integer, `0x` hex integer, or a real with a
    /// fraction and/or exponent. Overflowing decimal integers promote to real.
    fn number(&mut self) -> Result<TokenKind> {
        let start = self.pos;

        // Hex integer: 0x / 0X followed by at least one hex digit.
        if self.peek() == Some(b'0')
            && matches!(self.peek_at(1), Some(b'x') | Some(b'X'))
            && self.peek_at(2).is_some_and(|c| c.is_ascii_hexdigit())
        {
            self.pos += 2;
            let digits_start = self.pos;
            // A single `_` between two hex digits is allowed for readability (SQLite
            // 3.46.0) and stripped before parsing; `0x_1`, `0x1_`, `0x1__2` do not
            // match (the `_` is left to be rejected as a trailing ident char).
            self.consume_digit_run(u8::is_ascii_hexdigit);
            self.reject_trailing_ident_char()?;
            let raw = &self.src[digits_start..self.pos];
            let digits = strip_underscores(raw);
            // SQLite reads a hex literal as a 64-bit value (wraps past i64::MAX).
            let v = u64::from_str_radix(&digits, 16)
                .map(|u| u as i64)
                .map_err(|_| Error::Sql(format!("hex literal out of range: 0x{raw}")))?;
            return Ok(TokenKind::Integer(v));
        }

        // A single `_` between two decimal digits is allowed for readability (SQLite
        // 3.46.0). It is only consumed when both neighbours are digits, so it never
        // crosses a `.`/`e` boundary (`1_.0`, `1e_1` stay invalid) and is stripped
        // before the value is parsed.
        let mut is_real = false;
        self.consume_digit_run(u8::is_ascii_digit);
        if self.peek() == Some(b'.') {
            is_real = true;
            self.pos += 1;
            self.consume_digit_run(u8::is_ascii_digit);
        }
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            // Only an exponent if a sign/digit follows; otherwise `e` starts an ident.
            let after = self.peek_at(1);
            let exp_ok = match after {
                Some(b'+') | Some(b'-') => self.peek_at(2).is_some_and(|c| c.is_ascii_digit()),
                Some(c) => c.is_ascii_digit(),
                None => false,
            };
            if exp_ok {
                is_real = true;
                self.pos += 1; // e
                if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                    self.pos += 1;
                }
                self.consume_digit_run(u8::is_ascii_digit);
            }
        }

        self.reject_trailing_ident_char()?;
        let raw = &self.src[start..self.pos];
        let text = strip_underscores(raw);
        if is_real {
            let v: f64 = text
                .parse()
                .map_err(|_| Error::Sql(format!("malformed real literal: {raw}")))?;
            Ok(TokenKind::Real(v))
        } else {
            match text.parse::<i64>() {
                Ok(v) => Ok(TokenKind::Integer(v)),
                // Integer literal too big for i64: SQLite promotes to real.
                Err(_) => {
                    let v: f64 = text
                        .parse()
                        .map_err(|_| Error::Sql(format!("malformed integer literal: {raw}")))?;
                    Ok(TokenKind::Real(v))
                }
            }
        }
    }

    /// Consume a run of digits (per `is_digit`) with optional single `_` separators,
    /// each of which must sit BETWEEN two digits. An `_` is only stepped over when the
    /// following byte is itself a digit, and it is only reached after a digit has been
    /// consumed — so `1_2` advances past all three, while `1_`, `1__2`, and a leading
    /// `_` leave the `_` in place (to be rejected as a stray token).
    fn consume_digit_run(&mut self, is_digit: fn(&u8) -> bool) {
        while self.peek().is_some_and(|c| is_digit(&c)) {
            self.pos += 1;
            if self.peek() == Some(b'_') && self.peek_at(1).is_some_and(|c| is_digit(&c)) {
                self.pos += 1;
            }
        }
    }

    /// After a numeric literal, an identifier character with no separating space is
    /// an "unrecognized token" in SQLite (e.g. `123abc`).
    fn reject_trailing_ident_char(&mut self) -> Result<()> {
        if self.peek().is_some_and(is_ident_start) {
            return self.err("unrecognized token: number followed by identifier character");
        }
        Ok(())
    }

    fn ident_or_keyword(&mut self) -> TokenKind {
        let start = self.pos;
        while self.peek().is_some_and(is_ident_continue) {
            self.pos += 1;
        }
        let text = &self.src[start..self.pos];
        match Keyword::lookup(text) {
            Some(kw) => TokenKind::Keyword(kw),
            None => TokenKind::Ident,
        }
    }

    fn operator(&mut self, b: u8) -> Result<TokenKind> {
        let two = self.peek_at(1);
        let kind = match b {
            b'+' => {
                self.pos += 1;
                TokenKind::Plus
            }
            b'-' => {
                if two == Some(b'>') {
                    if self.peek_at(2) == Some(b'>') {
                        self.pos += 3;
                        TokenKind::Arrow2
                    } else {
                        self.pos += 2;
                        TokenKind::Arrow
                    }
                } else {
                    self.pos += 1;
                    TokenKind::Minus
                }
            }
            b'*' => {
                self.pos += 1;
                TokenKind::Star
            }
            b'/' => {
                self.pos += 1;
                TokenKind::Slash
            }
            b'%' => {
                self.pos += 1;
                TokenKind::Percent
            }
            b'|' => {
                if two == Some(b'|') {
                    self.pos += 2;
                    TokenKind::Concat
                } else {
                    self.pos += 1;
                    TokenKind::Pipe
                }
            }
            b'&' => {
                self.pos += 1;
                TokenKind::Amp
            }
            b'~' => {
                self.pos += 1;
                TokenKind::Tilde
            }
            b'<' => match two {
                Some(b'<') => {
                    self.pos += 2;
                    TokenKind::Shl
                }
                Some(b'=') => {
                    self.pos += 2;
                    TokenKind::LtEq
                }
                Some(b'>') => {
                    self.pos += 2;
                    TokenKind::Ne
                }
                _ => {
                    self.pos += 1;
                    TokenKind::Lt
                }
            },
            b'>' => match two {
                Some(b'>') => {
                    self.pos += 2;
                    TokenKind::Shr
                }
                Some(b'=') => {
                    self.pos += 2;
                    TokenKind::GtEq
                }
                _ => {
                    self.pos += 1;
                    TokenKind::Gt
                }
            },
            b'=' => {
                if two == Some(b'=') {
                    self.pos += 2;
                } else {
                    self.pos += 1;
                }
                TokenKind::Eq
            }
            b'!' => {
                if two == Some(b'=') {
                    self.pos += 2;
                    TokenKind::Ne
                } else {
                    return self.err("unrecognized token: '!' (did you mean '!=' ?)");
                }
            }
            b'(' => {
                self.pos += 1;
                TokenKind::LParen
            }
            b')' => {
                self.pos += 1;
                TokenKind::RParen
            }
            b',' => {
                self.pos += 1;
                TokenKind::Comma
            }
            b'.' => {
                self.pos += 1;
                TokenKind::Dot
            }
            b';' => {
                self.pos += 1;
                TokenKind::Semicolon
            }
            other => {
                return self.err(format!("unrecognized token: {:?}", other as char));
            }
        };
        Ok(kind)
    }

    /// Consume one UTF-8 char starting at `pos`, advancing past its whole encoding.
    fn take_char(&mut self) -> char {
        let rest = &self.src[self.pos..];
        let ch = rest.chars().next().expect("take_char at valid position");
        self.pos += ch.len_utf8();
        ch
    }
}

fn hex_val(b: u8) -> u8 {
    match b {
        b'0'..=b'9' => b - b'0',
        b'a'..=b'f' => b - b'a' + 10,
        b'A'..=b'F' => b - b'A' + 10,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(sql: &str) -> Vec<TokenKind> {
        tokenize(sql).unwrap().into_iter().map(|t| t.kind).collect()
    }

    #[test]
    fn integers_and_reals() {
        assert_eq!(kinds("42"), vec![TokenKind::Integer(42), TokenKind::Eof]);
        assert_eq!(kinds("0xff"), vec![TokenKind::Integer(255), TokenKind::Eof]);
        assert_eq!(kinds("3.14"), vec![TokenKind::Real(3.14), TokenKind::Eof]);
        assert_eq!(kinds("1e3"), vec![TokenKind::Real(1000.0), TokenKind::Eof]);
        assert_eq!(kinds(".5"), vec![TokenKind::Real(0.5), TokenKind::Eof]);
        assert_eq!(kinds("2.5e-1"), vec![TokenKind::Real(0.25), TokenKind::Eof]);
    }

    #[test]
    fn strings_blobs_idents() {
        assert_eq!(kinds("'it''s'"), vec![TokenKind::Str("it's".into()), TokenKind::Eof]);
        assert_eq!(kinds("x'41ff'"), vec![TokenKind::Blob(vec![0x41, 0xff]), TokenKind::Eof]);
        assert_eq!(
            kinds("\"a b\""),
            vec![
                TokenKind::QuotedIdent { value: "a b".into(), quote: QuoteKind::Double },
                TokenKind::Eof
            ]
        );
        assert_eq!(
            kinds("[a b]"),
            vec![
                TokenKind::QuotedIdent { value: "a b".into(), quote: QuoteKind::Bracket },
                TokenKind::Eof
            ]
        );
        assert_eq!(kinds("foo"), vec![TokenKind::Ident, TokenKind::Eof]);
        assert_eq!(kinds("select"), vec![TokenKind::Keyword(Keyword::Select), TokenKind::Eof]);
    }

    #[test]
    fn bind_params() {
        assert_eq!(kinds("?"), vec![TokenKind::Param(BindParam::Anonymous), TokenKind::Eof]);
        assert_eq!(kinds("?12"), vec![TokenKind::Param(BindParam::Numbered(12)), TokenKind::Eof]);
        assert_eq!(
            kinds(":name"),
            vec![TokenKind::Param(BindParam::Named(":name".into())), TokenKind::Eof]
        );
        assert_eq!(
            kinds("@x $y"),
            vec![
                TokenKind::Param(BindParam::Named("@x".into())),
                TokenKind::Param(BindParam::Named("$y".into())),
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn operators_and_comments() {
        assert_eq!(
            kinds("a || b -- trailing\n"),
            vec![TokenKind::Ident, TokenKind::Concat, TokenKind::Ident, TokenKind::Eof]
        );
        assert_eq!(
            kinds("1 /* block */ + /* eof-unterminated"),
            vec![TokenKind::Integer(1), TokenKind::Plus, TokenKind::Eof]
        );
        assert_eq!(
            kinds("<= >= << >> <> != == ->>"),
            vec![
                TokenKind::LtEq,
                TokenKind::GtEq,
                TokenKind::Shl,
                TokenKind::Shr,
                TokenKind::Ne,
                TokenKind::Ne,
                TokenKind::Eq,
                TokenKind::Arrow2,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn errors_are_loud() {
        assert!(tokenize("'unterminated").is_err());
        assert!(tokenize("x'abc'").is_err()); // odd hex digits
        assert!(tokenize("123abc").is_err());
        assert!(tokenize("!").is_err());
    }

    #[test]
    fn spans_recover_original_text() {
        let sql = "SeLeCt Foo";
        let toks = tokenize(sql).unwrap();
        assert_eq!(toks[0].text(sql), "SeLeCt");
        assert_eq!(toks[1].text(sql), "Foo");
    }

    #[test]
    fn numeric_underscore_separators() {
        // SQLite 3.46.0 allows a single `_` BETWEEN two digits of a numeric literal for
        // readability; it is stripped before the value is parsed. This pins both the
        // happy cases and the "only between two digits" rule — a coverage the happy-path
        // `integer_literal_underscore_separator` conformance test (which only checks
        // `1_000`) cannot provide (it would stay green if `1_` were wrongly accepted as
        // `1`). Accepted: `_` flanked by digits across decimal / hex / fraction / exponent.
        assert_eq!(kinds("1_000"), vec![TokenKind::Integer(1000), TokenKind::Eof]);
        assert_eq!(kinds("0xFF_FF"), vec![TokenKind::Integer(0xFFFF), TokenKind::Eof]);
        assert_eq!(kinds("1_000.5"), vec![TokenKind::Real(1000.5), TokenKind::Eof]);
        assert_eq!(kinds("1.5_5"), vec![TokenKind::Real(1.55), TokenKind::Eof]);
        assert_eq!(kinds("1e1_0"), vec![TokenKind::Real(1e10), TokenKind::Eof]);
        // Rejected: a `_` NOT between two digits is a stray identifier char after the
        // number → "unrecognized token" (never silently accepted as if it were absent).
        // A trailing `_`, a doubled `__`, and a hex `_` with no following digit all error
        // rather than tokenizing as the bare number.
        assert!(tokenize("1_").is_err(), "trailing _ must error");
        assert!(tokenize("1__2").is_err(), "doubled __ must error");
        assert!(tokenize("0x1_").is_err(), "trailing _ after hex digits must error");
    }
}
