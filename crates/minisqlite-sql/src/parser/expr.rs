//! The Pratt / precedence-climbing expression parser.
//!
//! Precedence (tightest first), taken verbatim from the operator table in
//! `spec/sqlite-doc/lang_expr.html`:
//! 1. unary `~ + -`  2. `COLLATE`  3. `|| -> ->>`  4. `* / %`  5. `+ -`
//! 6. `& | << >>`  7. `ESCAPE`  8. `< > <= >=`
//! 9. `= == <> != IS IS-NOT IS-DISTINCT-FROM  BETWEEN IN MATCH LIKE REGEXP GLOB
//!    ISNULL NOTNULL "NOT NULL"`  10. `NOT`  11. `AND`  12. `OR`.
//!
//! The numeric binding powers below reproduce that order (bigger binds tighter).
//! Left-associative binary operators parse their right operand at their own binding
//! power, so equal-precedence operators nest to the left.

use crate::ast_expr::{
    BinaryOp, Expr, FrameBound, FrameExclude, FrameUnits, FunctionArgs, LikeKind, Literal, InRhs,
    OverClause, RaiseAction, UnaryOp, WindowFrame, WindowSpec,
};
use crate::keyword::Keyword;
use crate::parser::{Parser, MAX_EXPR_DEPTH};
use crate::token::{QuoteKind, TokenKind};
use minisqlite_types::Result;

const PREC_OR: u8 = 10;
const PREC_AND: u8 = 20;
const PREC_NOT: u8 = 30;
const PREC_EQ: u8 = 40; // = == != <> IS IN LIKE GLOB MATCH REGEXP BETWEEN ISNULL NOTNULL
const PREC_CMP: u8 = 50; // < <= > >=
const PREC_BIT: u8 = 60; // & | << >>
const PREC_ADD: u8 = 70; // + -
const PREC_MUL: u8 = 80; // * / %
const PREC_CONCAT: u8 = 90; // || -> ->>
const PREC_COLLATE: u8 = 100; // postfix COLLATE
const PREC_UNARY: u8 = 110; // prefix ~ + -

fn binop(op: BinaryOp, left: Expr, right: Expr) -> Expr {
    Expr::Binary { op, left: Box::new(left), right: Box::new(right) }
}

fn unary(op: UnaryOp, expr: Expr) -> Expr {
    Expr::Unary { op, expr: Box::new(expr) }
}

impl<'a> Parser<'a> {
    /// Parse a full expression.
    pub(crate) fn parse_expr(&mut self) -> Result<Expr> {
        self.parse_expr_bp(0)
    }

    /// Precedence-climbing core: parse an expression whose operators all bind
    /// tighter than `min_bp`. This is the expression-recursion head, so it guards
    /// nesting depth (see `MAX_PARSE_DEPTH`) before descending.
    fn parse_expr_bp(&mut self, min_bp: u8) -> Result<Expr> {
        self.enter()?;
        let r = self.parse_expr_bp_inner(min_bp);
        self.leave();
        r
    }

    fn parse_expr_bp_inner(&mut self, min_bp: u8) -> Result<Expr> {
        let mut lhs = self.parse_prefix()?;
        // Each iteration folds one left-associative operator onto `lhs`, growing a
        // loop-built left-nested chain that the parse-recursion depth guard never
        // sees (its enter/leave is balanced by the RHS parse). Bound the chain like
        // SQLite (`SQLITE_MAX_EXPR_DEPTH`) so a long flat `a + a + a + …` errors
        // instead of building a tree too tall to walk or drop.
        let mut folds: u32 = 0;
        loop {
            let Some(lbp) = self.peek_infix_bp() else { break };
            if lbp <= min_bp {
                break;
            }
            folds += 1;
            self.enforce_max(folds, MAX_EXPR_DEPTH, "expression tree is too large")?;
            lhs = self.parse_infix(lhs, lbp)?;
        }
        Ok(lhs)
    }

    /// A prefix operator (`~ + - NOT`) or an atom.
    fn parse_prefix(&mut self) -> Result<Expr> {
        match self.kind() {
            TokenKind::Minus => {
                self.bump();
                // SQLite special case: unary minus applied DIRECTLY to the integer
                // literal 9223372036854775808 (2^63) yields i64::MIN as an INTEGER, not
                // a real. The magnitude overflows i64 as a positive value, so the lexer
                // tokenized it as a real; but `-2^63` fits exactly, so fold it here. This
                // fires only for a bare integer literal (no `.`/exponent) immediately
                // after the `-`. `-9223372036854775808.0` (real lexeme), `-(9223...808)`
                // (parenthesized), and `-9223372036854775809` (true overflow) stay real,
                // matching sqlite3DecOrHexToI64 + codeInteger's negFlag handling.
                //
                // The magnitude MUST be read from the decimal lexeme, not the token's
                // f64: 9223372036854775809 rounds to the same f64 as 2^63 (the ULP there
                // is 2048), so only the digit string distinguishes them.
                let fold_int_min = match self.kind() {
                    TokenKind::Real(_) => {
                        let lex = self.tokens[self.pos].text(self.src);
                        if lex.as_bytes().iter().any(|b| matches!(b, b'.' | b'e' | b'E')) {
                            false // a real literal (fraction/exponent), never i64::MIN
                        } else {
                            // Pure decimal integer literal (underscores are separators):
                            // fold iff its magnitude is exactly 2^63. A larger magnitude
                            // (…809, or one past u64) fails the equality / parse and stays
                            // real — the true-overflow case.
                            let digits: String = lex.chars().filter(|&c| c != '_').collect();
                            digits.parse::<u64>() == Ok(9223372036854775808)
                        }
                    }
                    _ => false,
                };
                if fold_int_min {
                    self.bump();
                    return Ok(Expr::Literal(Literal::Integer(i64::MIN)));
                }
                Ok(unary(UnaryOp::Negative, self.parse_expr_bp(PREC_UNARY)?))
            }
            TokenKind::Plus => {
                self.bump();
                Ok(unary(UnaryOp::Positive, self.parse_expr_bp(PREC_UNARY)?))
            }
            TokenKind::Tilde => {
                self.bump();
                Ok(unary(UnaryOp::BitNot, self.parse_expr_bp(PREC_UNARY)?))
            }
            TokenKind::Keyword(Keyword::Not) => {
                self.bump();
                Ok(unary(UnaryOp::Not, self.parse_expr_bp(PREC_NOT)?))
            }
            _ => self.parse_atom(),
        }
    }

    /// The binding power of the current token *if* it acts as an infix/postfix
    /// operator, else `None`.
    fn peek_infix_bp(&self) -> Option<u8> {
        match self.kind() {
            TokenKind::Concat | TokenKind::Arrow | TokenKind::Arrow2 => Some(PREC_CONCAT),
            TokenKind::Star | TokenKind::Slash | TokenKind::Percent => Some(PREC_MUL),
            TokenKind::Plus | TokenKind::Minus => Some(PREC_ADD),
            TokenKind::Amp | TokenKind::Pipe | TokenKind::Shl | TokenKind::Shr => Some(PREC_BIT),
            TokenKind::Lt | TokenKind::LtEq | TokenKind::Gt | TokenKind::GtEq => Some(PREC_CMP),
            TokenKind::Eq | TokenKind::Ne => Some(PREC_EQ),
            TokenKind::Keyword(k) => match k {
                Keyword::Or => Some(PREC_OR),
                Keyword::And => Some(PREC_AND),
                Keyword::Collate => Some(PREC_COLLATE),
                Keyword::Is
                | Keyword::In
                | Keyword::Like
                | Keyword::Glob
                | Keyword::Match
                | Keyword::Regexp
                | Keyword::Between
                | Keyword::Isnull
                | Keyword::Notnull
                | Keyword::Not => Some(PREC_EQ),
                _ => None,
            },
            _ => None,
        }
    }

    /// Consume the current infix/postfix operator and fold it with `lhs`.
    fn parse_infix(&mut self, lhs: Expr, lbp: u8) -> Result<Expr> {
        if let Some(kw) = self.kw() {
            match kw {
                Keyword::Collate => {
                    self.bump();
                    let collation = self.parse_ident()?;
                    return Ok(Expr::Collate { expr: Box::new(lhs), collation });
                }
                Keyword::Isnull => {
                    self.bump();
                    return Ok(Expr::IsNull(Box::new(lhs)));
                }
                Keyword::Notnull => {
                    self.bump();
                    return Ok(Expr::NotNull(Box::new(lhs)));
                }
                Keyword::Is => return self.parse_is(lhs),
                Keyword::Not => return self.parse_not_infix(lhs),
                Keyword::In => return self.parse_in(lhs, false),
                Keyword::Like => return self.parse_like(lhs, false, LikeKind::Like),
                Keyword::Glob => return self.parse_like(lhs, false, LikeKind::Glob),
                Keyword::Regexp => return self.parse_like(lhs, false, LikeKind::Regexp),
                Keyword::Match => return self.parse_like(lhs, false, LikeKind::Match),
                Keyword::Between => return self.parse_between(lhs, false),
                Keyword::And => {
                    self.bump();
                    let rhs = self.parse_expr_bp(lbp)?;
                    return Ok(binop(BinaryOp::And, lhs, rhs));
                }
                Keyword::Or => {
                    self.bump();
                    let rhs = self.parse_expr_bp(lbp)?;
                    return Ok(binop(BinaryOp::Or, lhs, rhs));
                }
                _ => {}
            }
        }
        let op = match self.kind() {
            TokenKind::Concat => BinaryOp::Concat,
            TokenKind::Arrow => BinaryOp::JsonArrow,
            TokenKind::Arrow2 => BinaryOp::JsonArrow2,
            TokenKind::Star => BinaryOp::Mul,
            TokenKind::Slash => BinaryOp::Div,
            TokenKind::Percent => BinaryOp::Mod,
            TokenKind::Plus => BinaryOp::Add,
            TokenKind::Minus => BinaryOp::Sub,
            TokenKind::Amp => BinaryOp::BitAnd,
            TokenKind::Pipe => BinaryOp::BitOr,
            TokenKind::Shl => BinaryOp::LShift,
            TokenKind::Shr => BinaryOp::RShift,
            TokenKind::Lt => BinaryOp::Lt,
            TokenKind::LtEq => BinaryOp::Le,
            TokenKind::Gt => BinaryOp::Gt,
            TokenKind::GtEq => BinaryOp::Ge,
            TokenKind::Eq => BinaryOp::Eq,
            TokenKind::Ne => BinaryOp::Ne,
            _ => return self.error("expected a binary operator"),
        };
        self.bump();
        let rhs = self.parse_expr_bp(lbp)?;
        Ok(binop(op, lhs, rhs))
    }

    /// `expr IS [NOT] [DISTINCT FROM] expr`.
    fn parse_is(&mut self, lhs: Expr) -> Result<Expr> {
        self.expect_kw(Keyword::Is)?;
        // `IS NOT DISTINCT FROM` == `IS`; `IS DISTINCT FROM` == `IS NOT`.
        if self.eat_kw(Keyword::Not) {
            if self.eat_kw(Keyword::Distinct) {
                self.expect_kw(Keyword::From)?;
                let rhs = self.parse_expr_bp(PREC_EQ)?;
                return Ok(binop(BinaryOp::Is, lhs, rhs));
            }
            let rhs = self.parse_expr_bp(PREC_EQ)?;
            return Ok(binop(BinaryOp::IsNot, lhs, rhs));
        }
        if self.eat_kw(Keyword::Distinct) {
            self.expect_kw(Keyword::From)?;
            let rhs = self.parse_expr_bp(PREC_EQ)?;
            return Ok(binop(BinaryOp::IsNot, lhs, rhs));
        }
        let rhs = self.parse_expr_bp(PREC_EQ)?;
        Ok(binop(BinaryOp::Is, lhs, rhs))
    }

    /// The infix `NOT ...` forms: `NOT NULL` / `NOT IN` / `NOT LIKE` / etc.
    fn parse_not_infix(&mut self, lhs: Expr) -> Result<Expr> {
        self.expect_kw(Keyword::Not)?;
        match self.kw() {
            Some(Keyword::Null) => {
                self.bump();
                Ok(Expr::NotNull(Box::new(lhs)))
            }
            Some(Keyword::In) => self.parse_in(lhs, true),
            Some(Keyword::Like) => self.parse_like(lhs, true, LikeKind::Like),
            Some(Keyword::Glob) => self.parse_like(lhs, true, LikeKind::Glob),
            Some(Keyword::Regexp) => self.parse_like(lhs, true, LikeKind::Regexp),
            Some(Keyword::Match) => self.parse_like(lhs, true, LikeKind::Match),
            Some(Keyword::Between) => self.parse_between(lhs, true),
            _ => self.error("expected NULL, IN, LIKE, GLOB, MATCH, REGEXP or BETWEEN after NOT"),
        }
    }

    /// `expr [NOT] LIKE|GLOB|REGEXP|MATCH expr [ESCAPE expr]`.
    fn parse_like(&mut self, lhs: Expr, negated: bool, kind: LikeKind) -> Result<Expr> {
        self.bump(); // the LIKE / GLOB / REGEXP / MATCH keyword
        let rhs = self.parse_expr_bp(PREC_EQ)?;
        let escape = if self.eat_kw(Keyword::Escape) {
            Some(Box::new(self.parse_expr_bp(PREC_EQ)?))
        } else {
            None
        };
        Ok(Expr::Like { negated, kind, lhs: Box::new(lhs), rhs: Box::new(rhs), escape })
    }

    /// `expr [NOT] BETWEEN low AND high`. `low`/`high` parse at the operator's own
    /// precedence so a trailing boolean `AND` is left for the outer expression.
    fn parse_between(&mut self, lhs: Expr, negated: bool) -> Result<Expr> {
        self.expect_kw(Keyword::Between)?;
        let low = self.parse_expr_bp(PREC_EQ)?;
        self.expect_kw(Keyword::And)?;
        let high = self.parse_expr_bp(PREC_EQ)?;
        Ok(Expr::Between {
            negated,
            expr: Box::new(lhs),
            low: Box::new(low),
            high: Box::new(high),
        })
    }

    /// `expr [NOT] IN (...)` — a list, a subquery, or a table / table-function.
    fn parse_in(&mut self, lhs: Expr, negated: bool) -> Result<Expr> {
        self.expect_kw(Keyword::In)?;
        if self.eat(&TokenKind::LParen) {
            if self.eat(&TokenKind::RParen) {
                return Ok(Expr::In { negated, expr: Box::new(lhs), rhs: InRhs::List(Vec::new()) });
            }
            if self.at_select_start() {
                let sel = self.parse_select()?;
                self.expect(&TokenKind::RParen, "')' after IN subquery")?;
                return Ok(Expr::In {
                    negated,
                    expr: Box::new(lhs),
                    rhs: InRhs::Select(Box::new(sel)),
                });
            }
            let mut list = vec![self.parse_expr()?];
            while self.eat(&TokenKind::Comma) {
                list.push(self.parse_expr()?);
            }
            self.expect(&TokenKind::RParen, "')' after IN list")?;
            return Ok(Expr::In { negated, expr: Box::new(lhs), rhs: InRhs::List(list) });
        }
        // `IN table` / `IN schema.table` / `IN func(args)`.
        let name = self.parse_qualified_name()?;
        let args = if self.eat(&TokenKind::LParen) {
            let mut a = Vec::new();
            if !self.check(&TokenKind::RParen) {
                a.push(self.parse_expr()?);
                while self.eat(&TokenKind::Comma) {
                    a.push(self.parse_expr()?);
                }
            }
            self.expect(&TokenKind::RParen, "')' after table-function arguments")?;
            a
        } else {
            Vec::new()
        };
        Ok(Expr::In { negated, expr: Box::new(lhs), rhs: InRhs::Table { name, args } })
    }

    // --- atoms ---------------------------------------------------------------

    fn parse_atom(&mut self) -> Result<Expr> {
        match self.kind().clone() {
            TokenKind::Integer(v) => {
                self.bump();
                Ok(Expr::Literal(Literal::Integer(v)))
            }
            TokenKind::Real(v) => {
                self.bump();
                Ok(Expr::Literal(Literal::Real(v)))
            }
            TokenKind::Str(s) => {
                self.bump();
                Ok(Expr::Literal(Literal::Text(s)))
            }
            TokenKind::Blob(b) => {
                self.bump();
                Ok(Expr::Literal(Literal::Blob(b)))
            }
            TokenKind::Param(p) => {
                self.bump();
                Ok(Expr::BindParam(p))
            }
            TokenKind::LParen => self.parse_paren_expr(),
            TokenKind::Ident | TokenKind::QuotedIdent { .. } => self.parse_identifier_expr(),
            TokenKind::Keyword(kw) => self.parse_keyword_atom(kw),
            _ => self.error("expected an expression"),
        }
    }

    fn parse_keyword_atom(&mut self, kw: Keyword) -> Result<Expr> {
        match kw {
            Keyword::Null => {
                self.bump();
                Ok(Expr::Literal(Literal::Null))
            }
            Keyword::CurrentDate => {
                self.bump();
                Ok(Expr::Literal(Literal::CurrentDate))
            }
            Keyword::CurrentTime => {
                self.bump();
                Ok(Expr::Literal(Literal::CurrentTime))
            }
            Keyword::CurrentTimestamp => {
                self.bump();
                Ok(Expr::Literal(Literal::CurrentTimestamp))
            }
            Keyword::Case => self.parse_case(),
            Keyword::Cast => self.parse_cast(),
            Keyword::Exists => {
                self.bump();
                self.expect(&TokenKind::LParen, "'(' after EXISTS")?;
                let sel = self.parse_select()?;
                self.expect(&TokenKind::RParen, "')' after EXISTS subquery")?;
                Ok(Expr::Exists { negated: false, select: Box::new(sel) })
            }
            Keyword::Raise => self.parse_raise(),
            _ if kw.can_be_identifier() => self.parse_identifier_expr(),
            _ => self.error("expected an expression"),
        }
    }

    /// A name in expression position: a column reference (possibly qualified), a
    /// function call, or the bare boolean literals `TRUE` / `FALSE`.
    fn parse_identifier_expr(&mut self) -> Result<Expr> {
        // The boolean literals are *unquoted* identifiers only (a quoted "true" is a
        // column name). SQLite also lets a real column named true/false shadow the
        // literal at resolution time; that rare case is left to the executor.
        let was_unquoted = matches!(self.kind(), TokenKind::Ident);
        // Capture the double-quote flag on the FIRST token before `parse_ident`
        // bumps it. It only matters for the bare (unqualified, non-call) case below:
        // a dotted reference or a function name is never a DQS string literal.
        let from_dqs =
            matches!(self.kind(), TokenKind::QuotedIdent { quote: QuoteKind::Double, .. });
        let first = self.parse_ident()?;

        if self.check(&TokenKind::LParen) {
            return self.parse_function_call(first);
        }
        if self.eat(&TokenKind::Dot) {
            let second = self.parse_ident()?;
            if self.eat(&TokenKind::Dot) {
                let third = self.parse_ident()?;
                return Ok(Expr::Column {
                    schema: Some(first),
                    table: Some(second),
                    name: third,
                    from_dqs: false,
                });
            }
            return Ok(Expr::Column {
                schema: None,
                table: Some(first),
                name: second,
                from_dqs: false,
            });
        }
        if was_unquoted {
            if first.eq_ignore_ascii_case("true") {
                return Ok(Expr::Literal(Literal::True));
            }
            if first.eq_ignore_ascii_case("false") {
                return Ok(Expr::Literal(Literal::False));
            }
        }
        Ok(Expr::Column { schema: None, table: None, name: first, from_dqs })
    }

    fn parse_function_call(&mut self, name: String) -> Result<Expr> {
        self.expect(&TokenKind::LParen, "'(' in function call")?;
        let (distinct, args, order_by) = if self.eat(&TokenKind::Star) {
            (false, FunctionArgs::Star, Vec::new())
        } else if self.check(&TokenKind::RParen) {
            (false, FunctionArgs::Empty, Vec::new())
        } else {
            let distinct = self.eat_kw(Keyword::Distinct);
            if !distinct {
                self.eat_kw(Keyword::All);
            }
            let mut list = vec![self.parse_expr()?];
            while self.eat(&TokenKind::Comma) {
                list.push(self.parse_expr()?);
            }
            // Aggregate `ORDER BY` inside the argument list: `group_concat(x ORDER BY y)`
            // (lang_aggfunc.html). It follows the last argument and orders the values fed
            // to the aggregate. Parsed for any function; the binder enforces that it is
            // only valid on an aggregate call.
            let order_by = if self.eat_kw(Keyword::Order) {
                self.expect_kw(Keyword::By)?;
                self.parse_ordering_terms()?
            } else {
                Vec::new()
            };
            (distinct, FunctionArgs::List(list), order_by)
        };
        self.expect(&TokenKind::RParen, "')' after function arguments")?;

        let filter = if self.eat_kw(Keyword::Filter) {
            self.expect(&TokenKind::LParen, "'(' after FILTER")?;
            self.expect_kw(Keyword::Where)?;
            let e = self.parse_expr()?;
            self.expect(&TokenKind::RParen, "')' after FILTER clause")?;
            Some(Box::new(e))
        } else {
            None
        };
        let over = if self.eat_kw(Keyword::Over) { Some(self.parse_over_clause()?) } else { None };
        Ok(Expr::Function { name, distinct, args, filter, over, order_by })
    }

    fn parse_over_clause(&mut self) -> Result<OverClause> {
        if self.check(&TokenKind::LParen) {
            Ok(OverClause::Spec(self.parse_window_spec()?))
        } else {
            Ok(OverClause::WindowName(self.parse_ident()?))
        }
    }

    /// `( [base-window] [PARTITION BY ...] [ORDER BY ...] [frame] )`.
    pub(crate) fn parse_window_spec(&mut self) -> Result<WindowSpec> {
        self.expect(&TokenKind::LParen, "'(' for window definition")?;
        // A leading base-window name, unless the next token starts a known clause.
        let base = if self.at_ident()
            && !matches!(
                self.kw(),
                Some(Keyword::Partition | Keyword::Order | Keyword::Range | Keyword::Rows | Keyword::Groups)
            ) {
            Some(self.parse_ident()?)
        } else {
            None
        };
        let partition_by = if self.eat_kw(Keyword::Partition) {
            self.expect_kw(Keyword::By)?;
            let mut v = vec![self.parse_expr()?];
            while self.eat(&TokenKind::Comma) {
                v.push(self.parse_expr()?);
            }
            v
        } else {
            Vec::new()
        };
        let order_by = if self.eat_kw(Keyword::Order) {
            self.expect_kw(Keyword::By)?;
            self.parse_ordering_terms()?
        } else {
            Vec::new()
        };
        let frame = if matches!(
            self.kw(),
            Some(Keyword::Range | Keyword::Rows | Keyword::Groups)
        ) {
            Some(self.parse_window_frame()?)
        } else {
            None
        };
        self.expect(&TokenKind::RParen, "')' after window definition")?;
        Ok(WindowSpec { base, partition_by, order_by, frame })
    }

    fn parse_window_frame(&mut self) -> Result<WindowFrame> {
        let units = if self.eat_kw(Keyword::Range) {
            FrameUnits::Range
        } else if self.eat_kw(Keyword::Rows) {
            FrameUnits::Rows
        } else if self.eat_kw(Keyword::Groups) {
            FrameUnits::Groups
        } else {
            return self.error("expected RANGE, ROWS or GROUPS in window frame");
        };
        let (start, end) = if self.eat_kw(Keyword::Between) {
            let s = self.parse_frame_bound()?;
            self.expect_kw(Keyword::And)?;
            let e = self.parse_frame_bound()?;
            (s, Some(e))
        } else {
            (self.parse_frame_bound()?, None)
        };
        let exclude = if self.eat_kw(Keyword::Exclude) {
            Some(self.parse_frame_exclude()?)
        } else {
            None
        };
        Ok(WindowFrame { units, start, end, exclude })
    }

    fn parse_frame_bound(&mut self) -> Result<FrameBound> {
        if self.eat_kw(Keyword::Unbounded) {
            if self.eat_kw(Keyword::Preceding) {
                return Ok(FrameBound::UnboundedPreceding);
            }
            if self.eat_kw(Keyword::Following) {
                return Ok(FrameBound::UnboundedFollowing);
            }
            return self.error("expected PRECEDING or FOLLOWING after UNBOUNDED");
        }
        if self.eat_kw(Keyword::Current) {
            self.expect_kw(Keyword::Row)?;
            return Ok(FrameBound::CurrentRow);
        }
        let e = self.parse_expr()?;
        if self.eat_kw(Keyword::Preceding) {
            return Ok(FrameBound::Preceding(Box::new(e)));
        }
        if self.eat_kw(Keyword::Following) {
            return Ok(FrameBound::Following(Box::new(e)));
        }
        self.error("expected PRECEDING or FOLLOWING in window frame bound")
    }

    fn parse_frame_exclude(&mut self) -> Result<FrameExclude> {
        if self.eat_kw(Keyword::No) {
            self.expect_kw(Keyword::Others)?;
            return Ok(FrameExclude::NoOthers);
        }
        if self.eat_kw(Keyword::Current) {
            self.expect_kw(Keyword::Row)?;
            return Ok(FrameExclude::CurrentRow);
        }
        if self.eat_kw(Keyword::Group) {
            return Ok(FrameExclude::Group);
        }
        if self.eat_kw(Keyword::Ties) {
            return Ok(FrameExclude::Ties);
        }
        self.error("expected NO OTHERS, CURRENT ROW, GROUP or TIES after EXCLUDE")
    }

    fn parse_case(&mut self) -> Result<Expr> {
        self.expect_kw(Keyword::Case)?;
        let operand =
            if !self.check_kw(Keyword::When) { Some(Box::new(self.parse_expr()?)) } else { None };
        let mut whens = Vec::new();
        while self.eat_kw(Keyword::When) {
            let cond = self.parse_expr()?;
            self.expect_kw(Keyword::Then)?;
            let res = self.parse_expr()?;
            whens.push((cond, res));
        }
        if whens.is_empty() {
            return self.error("CASE requires at least one WHEN branch");
        }
        let else_expr =
            if self.eat_kw(Keyword::Else) { Some(Box::new(self.parse_expr()?)) } else { None };
        self.expect_kw(Keyword::End)?;
        Ok(Expr::Case { operand, whens, else_expr })
    }

    fn parse_cast(&mut self) -> Result<Expr> {
        self.expect_kw(Keyword::Cast)?;
        self.expect(&TokenKind::LParen, "'(' after CAST")?;
        let e = self.parse_expr()?;
        self.expect_kw(Keyword::As)?;
        let type_name = self.parse_type_name()?;
        self.expect(&TokenKind::RParen, "')' after CAST(...)")?;
        Ok(Expr::Cast { expr: Box::new(e), type_name })
    }

    fn parse_raise(&mut self) -> Result<Expr> {
        self.expect_kw(Keyword::Raise)?;
        self.expect(&TokenKind::LParen, "'(' after RAISE")?;
        let action = if self.eat_kw(Keyword::Ignore) {
            RaiseAction::Ignore
        } else {
            let ctor: fn(String) -> RaiseAction = match self.kw() {
                Some(Keyword::Rollback) => RaiseAction::Rollback,
                Some(Keyword::Abort) => RaiseAction::Abort,
                Some(Keyword::Fail) => RaiseAction::Fail,
                _ => return self.error("expected IGNORE, ROLLBACK, ABORT or FAIL in RAISE"),
            };
            self.bump();
            self.expect(&TokenKind::Comma, "',' after RAISE action")?;
            let msg = self.parse_string_literal()?;
            ctor(msg)
        };
        self.expect(&TokenKind::RParen, "')' after RAISE(...)")?;
        Ok(Expr::Raise(action))
    }

    /// `(subquery)` or a parenthesized expression / row value `(a, b, ...)`.
    fn parse_paren_expr(&mut self) -> Result<Expr> {
        self.expect(&TokenKind::LParen, "'('")?;
        if self.at_select_start() {
            let sel = self.parse_select()?;
            self.expect(&TokenKind::RParen, "')' after subquery")?;
            return Ok(Expr::Subquery(Box::new(sel)));
        }
        let mut list = vec![self.parse_expr()?];
        while self.eat(&TokenKind::Comma) {
            list.push(self.parse_expr()?);
        }
        self.expect(&TokenKind::RParen, "')'")?;
        if list.len() == 1 {
            Ok(list.into_iter().next().expect("len checked"))
        } else {
            Ok(Expr::Parenthesized(list))
        }
    }

    // --- small shared helpers ------------------------------------------------

    /// Consume a string-literal token, returning its decoded text.
    pub(crate) fn parse_string_literal(&mut self) -> Result<String> {
        match self.kind().clone() {
            TokenKind::Str(s) => {
                self.bump();
                Ok(s)
            }
            _ => self.error("expected a string literal"),
        }
    }

    /// Whether the current keyword begins a query (`SELECT` / `VALUES` / `WITH`).
    pub(crate) fn at_select_start(&self) -> bool {
        matches!(self.kw(), Some(Keyword::Select | Keyword::Values | Keyword::With))
    }

    /// A type name: one or more identifier words plus an optional `(n[,m])` suffix.
    /// Stored verbatim (affinity is derived downstream).
    pub(crate) fn parse_type_name(&mut self) -> Result<String> {
        let mut parts: Vec<String> = Vec::new();
        while self.at_type_word() {
            parts.push(self.parse_ident()?);
        }
        let mut name = parts.join(" ");
        if self.check(&TokenKind::LParen) {
            self.bump();
            name.push('(');
            name.push_str(&self.parse_signed_number_str()?);
            if self.eat(&TokenKind::Comma) {
                name.push(',');
                name.push_str(&self.parse_signed_number_str()?);
            }
            self.expect(&TokenKind::RParen, "')' in type name")?;
            name.push(')');
        }
        if name.is_empty() {
            return self.error("expected a type name");
        }
        Ok(name)
    }

    /// A token that can be part of a type name (identifier or non-reserved keyword).
    pub(crate) fn at_type_word(&self) -> bool {
        match self.kind() {
            TokenKind::Ident | TokenKind::QuotedIdent { .. } => true,
            TokenKind::Keyword(kw) => kw.can_be_identifier(),
            _ => false,
        }
    }

    fn parse_signed_number_str(&mut self) -> Result<String> {
        let mut s = String::new();
        if self.eat(&TokenKind::Minus) {
            s.push('-');
        } else {
            self.eat(&TokenKind::Plus);
        }
        match self.kind().clone() {
            TokenKind::Integer(v) => {
                self.bump();
                s.push_str(&v.to_string());
                Ok(s)
            }
            TokenKind::Real(v) => {
                self.bump();
                s.push_str(&v.to_string());
                Ok(s)
            }
            _ => self.error("expected a number"),
        }
    }
}
