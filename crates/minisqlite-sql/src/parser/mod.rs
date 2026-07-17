//! Recursive-descent / Pratt parser. `Parser` holds the token stream and a cursor;
//! its `impl` blocks are split across sibling files by concern:
//! - `mod.rs` — cursor + token helpers, the statement dispatcher, and the small
//!   utility statements (transactions, PRAGMA, VACUUM, ATTACH, ...).
//! - `expr.rs` — the Pratt expression parser.
//! - `select.rs` — SELECT / compound / VALUES / FROM / WITH.
//! - `ddl.rs` — CREATE {TABLE,INDEX,VIEW,TRIGGER}, DROP, ALTER TABLE.
//! - `dml.rs` — INSERT, UPDATE, DELETE (with UPSERT and RETURNING).

mod ddl;
mod dml;
mod expr;
mod select;
#[cfg(test)]
mod tests;

use crate::ast_expr::Literal;
use crate::ast_stmt::{
    Ast, PragmaArg, PragmaValue, QualifiedName, Statement, TransactionMode,
};
use crate::keyword::Keyword;
use crate::token::{tokenize, Token, TokenKind};
use minisqlite_types::{Error, Result};

/// Parse a whole SQL program into an [`Ast`]. Entry point behind the crate's
/// `pub fn parse`.
pub fn parse_program(sql: &str) -> Result<Ast> {
    let mut p = Parser::new(sql)?;
    p.parse_program()
}

/// Maximum nesting depth of the recursive-descent parser.
///
/// SQLite bounds parse nesting the same way (`SQLITE_MAX_EXPR_DEPTH`,
/// `spec/sqlite-doc/limits.html`) and returns an error past the limit rather than
/// letting a pathologically deep statement — `((((...))))`, nested subqueries or
/// parenthesized joins, a long `NOT`/unary-minus chain, repeated `EXPLAIN` —
/// overflow the native call stack and abort the whole process. Legitimate SQL
/// nests only a handful of levels, so the only inputs this rejects are the
/// hostile ones real SQLite also rejects.
///
/// The value is calibrated for the *heaviest* recursion construct on the
/// *tightest* stack: on a 2 MiB debug thread, nested subqueries (each carrying a
/// full `parse_select_core` frame) overflow at ~50 levels — ~depth 100, since a
/// subquery nests through both `parse_select` and `parse_table_or_subquery` — so
/// 64 keeps even that case (≤32 subquery levels) well clear of the cliff, with
/// head-room for frame-size drift. Optimized builds have smaller frames and far
/// more room, so this is a conservative floor, not a tight fit. Real SQL nests
/// only a handful of levels, so nothing legitimate is rejected.
pub(crate) const MAX_PARSE_DEPTH: u32 = 64;

// Width limits on the loop-built, left-nested AST chains (compound SELECT terms,
// join tables, expression folds). MAX_PARSE_DEPTH above bounds parse *recursion*
// (nesting); these bound sibling *width*, which the recursion counter never sees.
// Each matches the corresponding real-SQLite limit *per chain*, so an oversized flat
// chain errors like SQLite (`spec/sqlite-doc/limits.html`) instead of building a giant
// tree. Caveat: the expr limit is a per-`parse_expr_bp_inner` fold counter, not
// SQLite's bottom-up total `nHeight`, so a *composed* expr (chains joined across
// nesting) can still exceed 1000 total height — a known residual that diverges from
// SQLite in the accept-more direction, and that a recursive consumer (`Clone`/`Eq`/
// `Debug`, or a downstream tree walk) could still overflow on. The proper fix is a
// bottom-up total-`nHeight` cap here at parse time. Crash-safety of *teardown* does
// not rely on the caps being exact: the iterative `Drop` impls on these three AST
// types make dropping safe at any height.
/// Max terms (SELECT cores) in one compound SELECT (`SQLITE_MAX_COMPOUND_SELECT`).
pub(crate) const MAX_COMPOUND_SELECT_TERMS: u32 = 500;
/// Max tables in one join — SQLite's fixed 64-table (join-bitmask) limit.
pub(crate) const MAX_JOIN_TABLES: u32 = 64;
/// Max expression-fold height per `parse_expr_bp_inner` level (`SQLITE_MAX_EXPR_DEPTH`,
/// default 1000). Per-level, not total tree height — see the caveat above.
pub(crate) const MAX_EXPR_DEPTH: u32 = 1000;

pub(crate) struct Parser<'a> {
    pub(crate) src: &'a str,
    pub(crate) tokens: Vec<Token>,
    pub(crate) pos: usize,
    /// Current recursion depth of the descent (see [`MAX_PARSE_DEPTH`]).
    pub(crate) depth: u32,
}

impl<'a> Parser<'a> {
    pub(crate) fn new(src: &'a str) -> Result<Self> {
        let tokens = tokenize(src)?;
        Ok(Parser { src, tokens, pos: 0, depth: 0 })
    }

    /// Enter one level of recursive descent, failing loudly if the nesting limit
    /// would be exceeded. Checks *before* incrementing so the depth counter never
    /// runs past the limit, and every successful `enter` must be balanced by a
    /// [`Parser::leave`] on unwind — the guarded recursion heads
    /// (`parse_statement`, `parse_expr_bp`, `parse_select`,
    /// `parse_table_or_subquery`) do this via a thin wrapper. Balancing matters:
    /// only *nesting* must count toward the limit, never the *width* of a sibling
    /// list (a 10 000-element `IN (...)` or a wide `SELECT` column list recurses
    /// one level at a time and must not accumulate).
    pub(crate) fn enter(&mut self) -> Result<()> {
        if self.depth >= MAX_PARSE_DEPTH {
            return self
                .error("parser stack overflow: statement or expression nests too deeply");
        }
        self.depth += 1;
        Ok(())
    }

    /// Leave one level of recursive descent, balancing an [`Parser::enter`].
    pub(crate) fn leave(&mut self) {
        self.depth -= 1;
    }

    /// Fail (like the matching SQLite limit) when a loop-built, left-nested chain
    /// exceeds `max` items. Shared by the three width-bounded productions —
    /// compound SELECT terms, join tables, and expression folds — so an oversized
    /// chain errors fast instead of building a tree whose teardown could be deep.
    /// Centralizing the check keeps the three loops from drifting and stops a
    /// future loop-built chain from silently reintroducing the unbounded-width
    /// class. `count` is the running item count; `msg` is SQLite's error text.
    pub(crate) fn enforce_max(&self, count: u32, max: u32, msg: &str) -> Result<()> {
        if count > max { self.error(msg) } else { Ok(()) }
    }

    // --- cursor / token access ------------------------------------------------

    /// The kind of the current token (always valid: the stream ends in `Eof`).
    pub(crate) fn kind(&self) -> &TokenKind {
        &self.tokens[self.pos].kind
    }

    /// The kind `off` tokens ahead (saturates at the trailing `Eof`).
    pub(crate) fn kind_at(&self, off: usize) -> &TokenKind {
        let i = (self.pos + off).min(self.tokens.len() - 1);
        &self.tokens[i].kind
    }

    pub(crate) fn at_eof(&self) -> bool {
        matches!(self.kind(), TokenKind::Eof)
    }

    /// Advance one token, returning the position that was current.
    pub(crate) fn bump(&mut self) -> usize {
        let p = self.pos;
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
        p
    }

    /// The keyword of the current token, if it is one.
    pub(crate) fn kw(&self) -> Option<Keyword> {
        match self.kind() {
            TokenKind::Keyword(k) => Some(*k),
            _ => None,
        }
    }

    pub(crate) fn kw_at(&self, off: usize) -> Option<Keyword> {
        match self.kind_at(off) {
            TokenKind::Keyword(k) => Some(*k),
            _ => None,
        }
    }

    // --- punctuation / operator matching -------------------------------------

    pub(crate) fn check(&self, k: &TokenKind) -> bool {
        self.kind() == k
    }

    pub(crate) fn eat(&mut self, k: &TokenKind) -> bool {
        if self.check(k) {
            self.bump();
            true
        } else {
            false
        }
    }

    pub(crate) fn expect(&mut self, k: &TokenKind, what: &str) -> Result<()> {
        if self.eat(k) {
            Ok(())
        } else {
            self.error(&format!("expected {what}"))
        }
    }

    // --- keyword matching -----------------------------------------------------

    pub(crate) fn check_kw(&self, kw: Keyword) -> bool {
        self.kw() == Some(kw)
    }

    pub(crate) fn eat_kw(&mut self, kw: Keyword) -> bool {
        if self.check_kw(kw) {
            self.bump();
            true
        } else {
            false
        }
    }

    pub(crate) fn expect_kw(&mut self, kw: Keyword) -> Result<()> {
        if self.eat_kw(kw) {
            Ok(())
        } else {
            self.error(&format!("expected keyword {}", kw.as_str()))
        }
    }

    /// Match a two-keyword sequence without consuming unless both are present.
    pub(crate) fn eat_kw2(&mut self, a: Keyword, b: Keyword) -> bool {
        if self.kw() == Some(a) && self.kw_at(1) == Some(b) {
            self.bump();
            self.bump();
            true
        } else {
            false
        }
    }

    // --- identifiers ----------------------------------------------------------

    /// Parse one identifier: an unquoted identifier, a quoted identifier, or a
    /// non-reserved keyword used as a name. Reserved keywords are rejected (they
    /// would have to be quoted).
    pub(crate) fn parse_ident(&mut self) -> Result<String> {
        let pos = self.pos;
        let name = match &self.tokens[pos].kind {
            TokenKind::Ident => self.tokens[pos].text(self.src).to_string(),
            TokenKind::QuotedIdent { value, .. } => value.clone(),
            TokenKind::Keyword(kw) if kw.can_be_identifier() => {
                self.tokens[pos].text(self.src).to_string()
            }
            _ => return self.error("expected an identifier"),
        };
        self.pos += 1;
        Ok(name)
    }

    /// Whether the current token is an *unquoted* identifier equal (case-insensitively)
    /// to `s`. Used for the pseudo-keywords SQLite tokenizes as identifiers, e.g.
    /// `STRICT`, `STORED`, `TRUE`, `FALSE`.
    pub(crate) fn at_ident_eq(&self, s: &str) -> bool {
        matches!(self.kind(), TokenKind::Ident)
            && self.tokens[self.pos].text(self.src).eq_ignore_ascii_case(s)
    }

    /// Whether the current token can start an identifier (for optional aliases).
    pub(crate) fn at_ident(&self) -> bool {
        match self.kind() {
            TokenKind::Ident | TokenKind::QuotedIdent { .. } => true,
            TokenKind::Keyword(kw) => kw.can_be_identifier(),
            _ => false,
        }
    }

    /// A possibly schema-qualified name: `name` or `schema.name`.
    pub(crate) fn parse_qualified_name(&mut self) -> Result<QualifiedName> {
        let first = self.parse_ident()?;
        if self.eat(&TokenKind::Dot) {
            let name = self.parse_ident()?;
            Ok(QualifiedName { schema: Some(first), name })
        } else {
            Ok(QualifiedName { schema: None, name: first })
        }
    }

    // --- errors ---------------------------------------------------------------

    pub(crate) fn error<T>(&self, msg: &str) -> Result<T> {
        self.error_at(self.pos, msg)
    }

    pub(crate) fn error_at<T>(&self, pos: usize, msg: &str) -> Result<T> {
        let tok = &self.tokens[pos.min(self.tokens.len() - 1)];
        let near = if matches!(tok.kind, TokenKind::Eof) {
            "end of input".to_string()
        } else {
            format!("'{}'", tok.text(self.src))
        };
        Err(Error::Sql(format!("{msg}, near {near} (byte offset {})", tok.start)))
    }

    pub(crate) fn unsupported<T>(&self, what: &str) -> Result<T> {
        Err(Error::Sql(format!("unsupported: {what}")))
    }

    // --- program / statement list --------------------------------------------

    fn parse_program(&mut self) -> Result<Ast> {
        let mut statements = Vec::new();
        let mut statement_sources = Vec::new();
        loop {
            // Skip empty statements (`;;`) and a trailing separator.
            while self.eat(&TokenKind::Semicolon) {}
            if self.at_eof() {
                break;
            }
            // The statement's source span runs from its first token's start byte to
            // its last token's end byte. `parse_statement` consumes exactly the
            // statement and stops on the following `;`/`Eof`, so the last token is at
            // `pos - 1`. Slicing token-to-token (not raw text) is what makes a `;`
            // inside a string/quoted-identifier/blob/comment a non-issue — those are
            // single tokens, never the `Semicolon` separator. Token starts are past
            // `skip_trivia`, so leading/trailing whitespace and comments fall outside
            // the span; the terminating `;` is a separate later token, also outside.
            let first = self.pos;
            let stmt = self.parse_statement()?;
            // Every statement parser consumes at least its leading keyword, so the
            // statement's last token is at `pos - 1`. Guard that invariant explicitly
            // (not only via `debug_assert`, which is compiled out) so a future statement
            // parser that could return Ok without advancing fails loud here instead of
            // underflowing `pos - 1` into a reversed/out-of-bounds slice in release.
            if self.pos <= first {
                return self.error("internal parser error: statement consumed no tokens");
            }
            let span = self.tokens[first].start..self.tokens[self.pos - 1].end;
            statements.push(stmt);
            statement_sources.push(self.src[span].to_string());
            // A statement is followed by `;` or end of input.
            if self.at_eof() {
                break;
            }
            if !self.check(&TokenKind::Semicolon) {
                return self.error("expected ';' or end of input after statement");
            }
        }
        // The two vectors are filled in lockstep above; pin the parallel-length
        // invariant so a future edit that pushes one without the other fails a test.
        debug_assert_eq!(statements.len(), statement_sources.len());
        Ok(Ast { statements, statement_sources })
    }

    /// Dispatch on the leading keyword to the right statement parser. Guards
    /// recursion depth (a repeated `EXPLAIN` prefix recurses here); see
    /// [`MAX_PARSE_DEPTH`].
    pub(crate) fn parse_statement(&mut self) -> Result<Statement> {
        self.enter()?;
        let r = self.parse_statement_inner();
        self.leave();
        r
    }

    fn parse_statement_inner(&mut self) -> Result<Statement> {
        // EXPLAIN [QUERY PLAN] <stmt>
        if self.eat_kw(Keyword::Explain) {
            if self.eat_kw(Keyword::Query) {
                self.expect_kw(Keyword::Plan)?;
                let inner = self.parse_statement()?;
                return Ok(Statement::ExplainQueryPlan(Box::new(inner)));
            }
            let inner = self.parse_statement()?;
            return Ok(Statement::Explain(Box::new(inner)));
        }

        let Some(kw) = self.kw() else {
            return self.error("expected a statement");
        };

        match kw {
            Keyword::With | Keyword::Select | Keyword::Values => self.parse_dml_or_select_stmt(),
            Keyword::Insert | Keyword::Replace => {
                let ins = self.parse_insert(None)?;
                Ok(Statement::Insert(Box::new(ins)))
            }
            Keyword::Update => {
                let upd = self.parse_update(None)?;
                Ok(Statement::Update(Box::new(upd)))
            }
            Keyword::Delete => {
                let del = self.parse_delete(None)?;
                Ok(Statement::Delete(Box::new(del)))
            }
            Keyword::Create => self.parse_create(),
            Keyword::Drop => self.parse_drop(),
            Keyword::Alter => self.parse_alter_table(),
            Keyword::Begin => self.parse_begin(),
            Keyword::Commit | Keyword::End => {
                self.bump();
                self.eat_kw(Keyword::Transaction);
                Ok(Statement::Commit)
            }
            Keyword::Rollback => self.parse_rollback(),
            Keyword::Savepoint => {
                self.bump();
                let name = self.parse_ident()?;
                Ok(Statement::Savepoint(name))
            }
            Keyword::Release => {
                self.bump();
                self.eat_kw(Keyword::Savepoint);
                let name = self.parse_ident()?;
                Ok(Statement::Release(name))
            }
            Keyword::Pragma => self.parse_pragma(),
            Keyword::Vacuum => self.parse_vacuum(),
            Keyword::Analyze => {
                self.bump();
                let target = if self.at_ident() { Some(self.parse_qualified_name()?) } else { None };
                Ok(Statement::Analyze { target })
            }
            Keyword::Reindex => {
                self.bump();
                let target = if self.at_ident() { Some(self.parse_qualified_name()?) } else { None };
                Ok(Statement::Reindex { target })
            }
            Keyword::Attach => self.parse_attach(),
            Keyword::Detach => self.parse_detach(),
            _ => self.error("expected a statement"),
        }
    }

    /// A statement that begins with `WITH`, `SELECT`, or `VALUES`: parse the optional
    /// `WITH`, then dispatch to SELECT / INSERT / UPDATE / DELETE.
    fn parse_dml_or_select_stmt(&mut self) -> Result<Statement> {
        let with = if self.check_kw(Keyword::With) { Some(self.parse_with()?) } else { None };
        match self.kw() {
            Some(Keyword::Select) | Some(Keyword::Values) => {
                let sel = self.parse_select_after_with(with)?;
                Ok(Statement::Select(Box::new(sel)))
            }
            Some(Keyword::Insert) | Some(Keyword::Replace) => {
                let ins = self.parse_insert(with)?;
                Ok(Statement::Insert(Box::new(ins)))
            }
            Some(Keyword::Update) => {
                let upd = self.parse_update(with)?;
                Ok(Statement::Update(Box::new(upd)))
            }
            Some(Keyword::Delete) => {
                let del = self.parse_delete(with)?;
                Ok(Statement::Delete(Box::new(del)))
            }
            _ => self.error("expected SELECT, VALUES, INSERT, UPDATE or DELETE after WITH"),
        }
    }

    // --- transactions ---------------------------------------------------------

    fn parse_begin(&mut self) -> Result<Statement> {
        self.expect_kw(Keyword::Begin)?;
        let mode = if self.eat_kw(Keyword::Deferred) {
            TransactionMode::Deferred
        } else if self.eat_kw(Keyword::Immediate) {
            TransactionMode::Immediate
        } else if self.eat_kw(Keyword::Exclusive) {
            TransactionMode::Exclusive
        } else {
            TransactionMode::Deferred
        };
        self.eat_kw(Keyword::Transaction);
        Ok(Statement::Begin { mode })
    }

    fn parse_rollback(&mut self) -> Result<Statement> {
        self.expect_kw(Keyword::Rollback)?;
        self.eat_kw(Keyword::Transaction);
        let to_savepoint = if self.eat_kw(Keyword::To) {
            self.eat_kw(Keyword::Savepoint);
            Some(self.parse_ident()?)
        } else {
            None
        };
        Ok(Statement::Rollback { to_savepoint })
    }

    // --- utility statements ---------------------------------------------------

    fn parse_pragma(&mut self) -> Result<Statement> {
        self.expect_kw(Keyword::Pragma)?;
        let name = self.parse_qualified_name()?;
        let arg = if self.eat(&TokenKind::Eq) {
            Some(PragmaArg::Equals(self.parse_pragma_value()?))
        } else if self.eat(&TokenKind::LParen) {
            let v = self.parse_pragma_value()?;
            self.expect(&TokenKind::RParen, "')' after PRAGMA argument")?;
            Some(PragmaArg::Call(v))
        } else {
            None
        };
        Ok(Statement::Pragma { name, arg })
    }

    /// A pragma value: an optionally-signed number, a string literal, or a bare
    /// name/keyword (`ON`, `OFF`, `WAL`, `FULL`, ...).
    fn parse_pragma_value(&mut self) -> Result<PragmaValue> {
        // Signed number.
        let neg = if self.eat(&TokenKind::Minus) {
            true
        } else {
            self.eat(&TokenKind::Plus);
            false
        };
        match self.kind().clone() {
            TokenKind::Integer(v) => {
                self.bump();
                // `wrapping_neg`, not `-v`: `-0x8000000000000000` tokenizes to
                // `i64::MIN`, and `-(i64::MIN)` overflows (a debug-build panic).
                // Wrapping folds it back to `i64::MIN`, matching SQLite.
                Ok(PragmaValue::Literal(Literal::Integer(if neg { v.wrapping_neg() } else { v })))
            }
            TokenKind::Real(v) => {
                self.bump();
                Ok(PragmaValue::Literal(Literal::Real(if neg { -v } else { v })))
            }
            _ if neg => self.error("expected a number after sign in PRAGMA value"),
            TokenKind::Str(s) => {
                self.bump();
                Ok(PragmaValue::Literal(Literal::Text(s)))
            }
            _ if self.at_ident() => Ok(PragmaValue::Name(self.parse_ident()?)),
            // A pragma value may be a bare RESERVED keyword used as a name — `DELETE`,
            // `OFF`, `FULL`, ... — which is not a valid identifier elsewhere and so is
            // rejected by `at_ident`/`parse_ident`. SQLite accepts any keyword in this
            // position (e.g. `PRAGMA journal_mode=DELETE`), so read the keyword's own
            // text as the value name.
            TokenKind::Keyword(_) => {
                let name = self.tokens[self.pos].text(self.src).to_string();
                self.bump();
                Ok(PragmaValue::Name(name))
            }
            _ => self.error("expected a PRAGMA value"),
        }
    }

    fn parse_vacuum(&mut self) -> Result<Statement> {
        self.expect_kw(Keyword::Vacuum)?;
        let schema = if self.at_ident() { Some(self.parse_ident()?) } else { None };
        let into = if self.eat_kw(Keyword::Into) { Some(self.parse_expr()?) } else { None };
        Ok(Statement::Vacuum { schema, into })
    }

    fn parse_attach(&mut self) -> Result<Statement> {
        self.expect_kw(Keyword::Attach)?;
        self.eat_kw(Keyword::Database);
        let file = self.parse_expr()?;
        self.expect_kw(Keyword::As)?;
        let schema = self.parse_expr()?;
        let key = if self.eat_kw(Keyword::Key) { Some(self.parse_expr()?) } else { None };
        Ok(Statement::Attach { file, schema, key })
    }

    fn parse_detach(&mut self) -> Result<Statement> {
        self.expect_kw(Keyword::Detach)?;
        self.eat_kw(Keyword::Database);
        let schema = self.parse_expr()?;
        Ok(Statement::Detach { schema })
    }
}

/// A convenience used by tests and callers that want just the first statement.
#[cfg(test)]
pub(crate) fn parse_one(sql: &str) -> Result<Statement> {
    let ast = parse_program(sql)?;
    match ast.statements.into_iter().next() {
        Some(s) => Ok(s),
        None => Err(Error::Sql("no statement".into())),
    }
}

/// Parse a single standalone expression (used by the precedence tests).
#[cfg(test)]
pub(crate) fn parse_expr_str(sql: &str) -> Result<crate::ast_expr::Expr> {
    let mut p = Parser::new(sql)?;
    let e = p.parse_expr()?;
    if !p.at_eof() {
        return p.error("trailing tokens after expression");
    }
    Ok(e)
}
