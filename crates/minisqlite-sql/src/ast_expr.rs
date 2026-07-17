//! Expression AST — the whole SQLite expression surface
//! (`spec/sqlite-doc/lang_expr.html`).
//!
//! High-fan-in contract: the planner and executor pattern-match these variants, so
//! the enum is comprehensive even where the parser does not yet build every variant.
//! Keep additions localized to this file.

use crate::ast_select::{OrderingTerm, Select};
use crate::ast_stmt::QualifiedName;

/// A scalar SQL expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A constant: number, string, blob, NULL, boolean, or a `CURRENT_*` value.
    Literal(Literal),
    /// A column reference, optionally qualified by table and schema.
    ///
    /// `from_dqs` is `true` only for a bare, unqualified double-quoted name
    /// (`"foo"`). SQLite's legacy DQS misfeature (`quirks.html` §8, the library
    /// default) makes such a token fall back to a TEXT string literal when it
    /// resolves to no column. Every other construction — unquoted, bracketed,
    /// backtick, or any qualified reference (`t."foo"`) — sets it `false` and never
    /// falls back.
    Column { schema: Option<String>, table: Option<String>, name: String, from_dqs: bool },
    /// A bind parameter (`?`, `?NNN`, `:name`, `@name`, `$name`).
    BindParam(BindParam),
    /// A prefix unary operator applied to an operand.
    Unary { op: UnaryOp, expr: Box<Expr> },
    /// A binary operator applied to two operands.
    Binary { op: BinaryOp, left: Box<Expr>, right: Box<Expr> },
    /// A function call, possibly aggregate (`DISTINCT`, `FILTER`, aggregate `ORDER BY`)
    /// or window (`OVER`).
    Function {
        name: String,
        distinct: bool,
        args: FunctionArgs,
        filter: Option<Box<Expr>>,
        over: Option<OverClause>,
        /// Aggregate `ORDER BY` written inside the argument list —
        /// `group_concat(x ORDER BY y)` (lang_aggfunc.html): the values are ordered by
        /// these terms before the aggregate folds them. Empty for the common call with no
        /// in-argument ORDER BY, and only meaningful for an aggregate in an aggregate query
        /// (the binder rejects it on a scalar function or an OVER window aggregate).
        order_by: Vec<OrderingTerm>,
    },
    /// `CAST(expr AS type-name)`.
    Cast { expr: Box<Expr>, type_name: String },
    /// `expr COLLATE collation-name`.
    Collate { expr: Box<Expr>, collation: String },
    /// `expr [NOT] LIKE|GLOB|REGEXP|MATCH expr [ESCAPE expr]`.
    Like { negated: bool, kind: LikeKind, lhs: Box<Expr>, rhs: Box<Expr>, escape: Option<Box<Expr>> },
    /// `expr [NOT] BETWEEN low AND high`.
    Between { negated: bool, expr: Box<Expr>, low: Box<Expr>, high: Box<Expr> },
    /// `expr [NOT] IN (...)`.
    In { negated: bool, expr: Box<Expr>, rhs: InRhs },
    /// `[NOT] EXISTS (select)`.
    Exists { negated: bool, select: Box<Select> },
    /// A scalar subquery `(SELECT ...)`.
    Subquery(Box<Select>),
    /// `CASE [operand] WHEN ... THEN ... [ELSE ...] END`.
    Case { operand: Option<Box<Expr>>, whens: Vec<(Expr, Expr)>, else_expr: Option<Box<Expr>> },
    /// `expr ISNULL`.
    IsNull(Box<Expr>),
    /// `expr NOTNULL` / `expr NOT NULL`.
    NotNull(Box<Expr>),
    /// `RAISE(...)` — only valid inside a trigger body.
    Raise(RaiseAction),
    /// A parenthesized expression list / row value: `(a, b, c)`. A single-element
    /// list is just a grouped expression.
    Parenthesized(Vec<Expr>),
}

impl Drop for Expr {
    /// Tear the expression down iteratively instead of via the compiler-derived
    /// recursion.
    ///
    /// The parser folds left-associative operators in a *loop* (`a + a + a + …`),
    /// so an `Expr::Binary` chain — and, composed through subqueries, the other
    /// recursive variants — can grow far taller than the parse-recursion depth
    /// guard bounds. A derived recursive `Drop` of such a chain would overflow the
    /// native stack (a SIGSEGV/SIGABRT, not a recoverable `Error`). Instead, move
    /// each node's owned child `Expr`s onto an explicit worklist (leaving a trivial
    /// `NULL` behind) and drop them one at a time, so teardown uses O(1) call-stack
    /// depth for any tree height. Subquery bodies (`Box<Select>`) are left to
    /// `Select`'s own drop, bounded by the parser's nesting-depth guard.
    fn drop(&mut self) {
        let mut stack: Vec<Expr> = Vec::new();
        take_expr_children(self, &mut stack);
        while let Some(mut e) = stack.pop() {
            take_expr_children(&mut e, &mut stack);
        }
    }
}

impl Expr {
    /// Whether this node owns any child `Expr` worth moving onto the teardown
    /// worklist. The trivial leaves (and the `NULL` left behind by
    /// [`take_expr_children`]) own none, so skipping them keeps a re-entrant drop
    /// of an already-emptied node from allocating a worklist at all.
    fn has_expr_children(&self) -> bool {
        !matches!(
            self,
            Expr::Literal(_)
                | Expr::Column { .. }
                | Expr::BindParam(_)
                | Expr::Raise(_)
                | Expr::Exists { .. }
                | Expr::Subquery(_)
        )
    }
}

/// Move every owned child `Expr` of `e` onto `out`, replacing each in place with a
/// trivial `NULL`. Exhaustive on purpose: adding a recursive `Expr` variant without
/// handling it here is a compile error, so the iterative teardown can't silently
/// regress to recursion for a new shape.
fn take_expr_children(e: &mut Expr, out: &mut Vec<Expr>) {
    fn take(slot: &mut Box<Expr>, out: &mut Vec<Expr>) {
        let child = std::mem::replace(slot.as_mut(), Expr::Literal(Literal::Null));
        if child.has_expr_children() {
            out.push(child);
        }
    }
    fn take_opt(slot: &mut Option<Box<Expr>>, out: &mut Vec<Expr>) {
        if let Some(b) = slot.as_mut() {
            take(b, out);
        }
    }
    fn take_vec(v: &mut Vec<Expr>, out: &mut Vec<Expr>) {
        for child in std::mem::take(v) {
            if child.has_expr_children() {
                out.push(child);
            }
        }
    }
    match e {
        // Leaves and cross-type children: `Exists`/`Subquery` hold only a
        // `Box<Select>`, which drops via `Select` (nesting-bounded), not here.
        Expr::Literal(_)
        | Expr::Column { .. }
        | Expr::BindParam(_)
        | Expr::Raise(_)
        | Expr::Exists { .. }
        | Expr::Subquery(_) => {}
        Expr::Unary { expr, .. } => take(expr, out),
        Expr::Binary { left, right, .. } => {
            take(left, out);
            take(right, out);
        }
        Expr::Function { args, filter, order_by, .. } => {
            if let FunctionArgs::List(v) = args {
                take_vec(v, out);
            }
            take_opt(filter, out);
            // Aggregate ORDER BY terms own an `Expr` each; move them onto the worklist so a
            // deeply-nested ordering key tears down iteratively like every other child.
            for term in std::mem::take(order_by) {
                if term.expr.has_expr_children() {
                    out.push(term.expr);
                }
            }
        }
        Expr::Cast { expr, .. } | Expr::Collate { expr, .. } => take(expr, out),
        Expr::Like { lhs, rhs, escape, .. } => {
            take(lhs, out);
            take(rhs, out);
            take_opt(escape, out);
        }
        Expr::Between { expr, low, high, .. } => {
            take(expr, out);
            take(low, out);
            take(high, out);
        }
        Expr::In { expr, rhs, .. } => {
            take(expr, out);
            match rhs {
                InRhs::List(v) | InRhs::Table { args: v, .. } => take_vec(v, out),
                InRhs::Select(_) => {}
            }
        }
        Expr::Case { operand, whens, else_expr } => {
            take_opt(operand, out);
            for (w, t) in std::mem::take(whens) {
                if w.has_expr_children() {
                    out.push(w);
                }
                if t.has_expr_children() {
                    out.push(t);
                }
            }
            take_opt(else_expr, out);
        }
        Expr::IsNull(e) | Expr::NotNull(e) => take(e, out),
        Expr::Parenthesized(v) => take_vec(v, out),
    }
}

/// A literal constant value.
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Null,
    Integer(i64),
    Real(f64),
    Text(String),
    Blob(Vec<u8>),
    /// `CURRENT_TIME`
    CurrentTime,
    /// `CURRENT_DATE`
    CurrentDate,
    /// `CURRENT_TIMESTAMP`
    CurrentTimestamp,
    /// `TRUE` (SQLite recognizes the keyword literal, value 1).
    True,
    /// `FALSE` (value 0).
    False,
}

/// A prefix unary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// `-expr`
    Negative,
    /// `+expr` (a no-op that SQLite still parses)
    Positive,
    /// `NOT expr`
    Not,
    /// `~expr`
    BitNot,
}

/// A binary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    /// `||`
    Concat,
    /// `->` — JSON: select a subcomponent and return its JSON representation
    /// (json1.html §4.10). Same precedence as `||` (PREC_CONCAT).
    JsonArrow,
    /// `->>` — JSON: select a subcomponent and return its SQL representation
    /// (json1.html §4.10). Same precedence as `||` (PREC_CONCAT).
    JsonArrow2,
    /// `*`
    Mul,
    /// `/`
    Div,
    /// `%`
    Mod,
    /// `+`
    Add,
    /// `-`
    Sub,
    /// `<<`
    LShift,
    /// `>>`
    RShift,
    /// `&`
    BitAnd,
    /// `|`
    BitOr,
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
    /// `=` / `==`
    Eq,
    /// `!=` / `<>`
    Ne,
    /// `IS` (also `IS NOT DISTINCT FROM`) — NULL-safe equality.
    Is,
    /// `IS NOT` (also `IS DISTINCT FROM`) — NULL-safe inequality.
    IsNot,
    /// `AND`
    And,
    /// `OR`
    Or,
}

/// The argument list of a function call.
#[derive(Debug, Clone, PartialEq)]
pub enum FunctionArgs {
    /// `f(a, b, c)` — an explicit list (possibly empty inside the parens if the
    /// caller uses [`FunctionArgs::Empty`]).
    List(Vec<Expr>),
    /// `f(*)` — e.g. `count(*)`.
    Star,
    /// `f()` — no arguments.
    Empty,
}

/// The pattern operator family in a `LIKE`-style expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LikeKind {
    Like,
    Glob,
    Regexp,
    Match,
}

/// The right-hand side of an `IN` expression.
#[derive(Debug, Clone, PartialEq)]
pub enum InRhs {
    /// `IN (a, b, c)` — an explicit value list (empty list allowed: `IN ()`).
    List(Vec<Expr>),
    /// `IN (SELECT ...)`.
    Select(Box<Select>),
    /// `IN table` / `IN schema.table` / `IN table(args)` — a table or table-valued
    /// function reference. Table-function arguments are carried in `args`.
    Table { name: QualifiedName, args: Vec<Expr> },
}

/// The `OVER` clause of a window function: either a named window or an inline spec.
#[derive(Debug, Clone, PartialEq)]
pub enum OverClause {
    /// `OVER window-name`.
    WindowName(String),
    /// `OVER (window-defn)`.
    Spec(WindowSpec),
}

/// A window definition (`PARTITION BY`, `ORDER BY`, frame). Also used by the
/// `WINDOW` clause of a `SELECT`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct WindowSpec {
    /// A base window name this spec extends (`OVER (base PARTITION BY ...)`).
    pub base: Option<String>,
    pub partition_by: Vec<Expr>,
    pub order_by: Vec<OrderingTerm>,
    pub frame: Option<WindowFrame>,
}

/// A window frame specification.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowFrame {
    pub units: FrameUnits,
    pub start: FrameBound,
    /// `None` for the implicit `AND CURRENT ROW` short form.
    pub end: Option<FrameBound>,
    pub exclude: Option<FrameExclude>,
}

/// The unit a window frame counts in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameUnits {
    Range,
    Rows,
    Groups,
}

/// One bound of a window frame.
#[derive(Debug, Clone, PartialEq)]
pub enum FrameBound {
    UnboundedPreceding,
    Preceding(Box<Expr>),
    CurrentRow,
    Following(Box<Expr>),
    UnboundedFollowing,
}

/// A window frame `EXCLUDE` option.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameExclude {
    NoOthers,
    CurrentRow,
    Group,
    Ties,
}

/// The action of a `RAISE(...)` expression (trigger bodies only).
#[derive(Debug, Clone, PartialEq)]
pub enum RaiseAction {
    /// `RAISE(IGNORE)`
    Ignore,
    /// `RAISE(ROLLBACK, message)`
    Rollback(String),
    /// `RAISE(ABORT, message)`
    Abort(String),
    /// `RAISE(FAIL, message)`
    Fail(String),
}

/// A bind parameter. Also produced by the tokenizer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindParam {
    /// `?` — anonymous positional; the engine assigns its number by position.
    Anonymous,
    /// `?NNN` — explicitly numbered.
    Numbered(u32),
    /// `:name`, `@name`, `$name` — the sigil is retained so the three forms stay
    /// distinct parameters.
    Named(String),
}
