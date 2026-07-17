//! SELECT statement AST (`spec/sqlite-doc/lang_select.html`): the query body
//! (single core or a compound tree), result columns, the FROM join tree,
//! ORDER BY / LIMIT, and the VALUES form.

use crate::ast_expr::{Expr, WindowSpec};
use crate::ast_stmt::{QualifiedName, With};

/// A complete SELECT statement: an optional `WITH`, the query body (a single core
/// or a compound-operator tree), and trailing `ORDER BY` / `LIMIT`.
#[derive(Debug, Clone, PartialEq)]
pub struct Select {
    pub with: Option<With>,
    pub body: SelectBody,
    pub order_by: Vec<OrderingTerm>,
    pub limit: Option<Limit>,
}

/// The body of a SELECT: either a single core, or two bodies joined by a compound
/// set operator. Left-associative: `a UNION b UNION c` nests as `(a UNION b) UNION c`.
#[derive(Debug, Clone, PartialEq)]
pub enum SelectBody {
    Select(SelectCore),
    Compound { op: CompoundOp, left: Box<SelectBody>, right: SelectCore },
}

/// A set operator joining two select cores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompoundOp {
    Union,
    UnionAll,
    Intersect,
    Except,
}

/// A single select core: a `SELECT ...` query or a `VALUES` row list.
#[derive(Debug, Clone, PartialEq)]
pub enum SelectCore {
    /// `SELECT [DISTINCT|ALL] columns [FROM] [WHERE] [GROUP BY] [HAVING] [WINDOW]`.
    Query {
        distinct: Distinct,
        columns: Vec<ResultColumn>,
        from: Option<FromClause>,
        where_clause: Option<Expr>,
        group_by: Vec<Expr>,
        having: Option<Expr>,
        /// Named windows from a `WINDOW name AS (...)` clause.
        windows: Vec<(String, WindowSpec)>,
    },
    /// `VALUES (row), (row), ...` — each inner vec is one row of expressions.
    Values(Vec<Vec<Expr>>),
}

/// `ALL` (the default) or `DISTINCT` row duplicate handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Distinct {
    #[default]
    All,
    Distinct,
}

/// One entry in a SELECT result-column list.
#[derive(Debug, Clone, PartialEq)]
pub enum ResultColumn {
    /// `expr [[AS] alias]`.
    Expr { expr: Expr, alias: Option<String> },
    /// `*` — all columns of all sources.
    Star,
    /// `table.*` — all columns of one source.
    TableStar(String),
}

/// The FROM clause is a join tree; see [`JoinTree`].
pub type FromClause = JoinTree;

/// A FROM-clause join tree. A leaf is one table/subquery; an interior node joins a
/// subtree to another table/subquery with an operator and optional constraint.
/// Comma-separated tables use [`JoinKind::Comma`].
#[derive(Debug, Clone, PartialEq)]
pub enum JoinTree {
    Table(TableOrSubquery),
    Join {
        left: Box<JoinTree>,
        op: JoinOperator,
        right: TableOrSubquery,
        constraint: Option<JoinConstraint>,
    },
}

/// One source in a FROM clause.
#[derive(Debug, Clone, PartialEq)]
pub enum TableOrSubquery {
    /// `[schema.]table [[AS] alias] [INDEXED BY x | NOT INDEXED]`.
    Table { name: QualifiedName, alias: Option<String>, indexed: Option<IndexedClause> },
    /// `(select) [[AS] alias]`.
    Subquery { select: Box<Select>, alias: Option<String> },
    /// `[schema.]func(args) [[AS] alias]` — a table-valued function.
    TableFunction { name: QualifiedName, args: Vec<Expr>, alias: Option<String> },
    /// A parenthesized join tree: `( join-clause )`.
    Join(Box<JoinTree>),
}

/// The `INDEXED BY` / `NOT INDEXED` hint on a table reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexedClause {
    NotIndexed,
    IndexedBy(String),
}

/// A join operator: its kind plus whether it was `NATURAL`. `OUTER` is folded into
/// `Left`/`Right`/`Full` (it is noise in SQLite: `LEFT` == `LEFT OUTER`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JoinOperator {
    pub natural: bool,
    pub kind: JoinKind,
}

/// The kind of join.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinKind {
    /// `,` — an implicit cross join.
    Comma,
    /// `[INNER] JOIN`.
    Inner,
    /// `LEFT [OUTER] JOIN`.
    Left,
    /// `RIGHT [OUTER] JOIN`.
    Right,
    /// `FULL [OUTER] JOIN`.
    Full,
    /// `CROSS JOIN`.
    Cross,
}

/// A join constraint: `ON expr` or `USING (col, ...)`.
#[derive(Debug, Clone, PartialEq)]
pub enum JoinConstraint {
    On(Expr),
    Using(Vec<String>),
}

/// One `ORDER BY` term.
#[derive(Debug, Clone, PartialEq)]
pub struct OrderingTerm {
    pub expr: Expr,
    pub collation: Option<String>,
    /// `ASC` / `DESC`; `None` means unspecified (defaults to ASC).
    pub order: Option<SortOrder>,
    /// `NULLS FIRST` / `NULLS LAST`; `None` means unspecified.
    pub nulls: Option<NullsOrder>,
}

/// Sort direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortOrder {
    Asc,
    Desc,
}

/// NULL ordering within a sort.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NullsOrder {
    First,
    Last,
}

/// A `LIMIT expr [OFFSET expr]` (or `LIMIT expr, expr`) clause.
#[derive(Debug, Clone, PartialEq)]
pub struct Limit {
    pub limit: Expr,
    pub offset: Option<Expr>,
}

// --- Iterative teardown -----------------------------------------------------
//
// `SelectBody::Compound` and `JoinTree::Join` are built by *loops* in the parser
// (`a UNION b UNION …`, `t1, t2, …`), so their left spines grow with input width —
// taller than the parse-recursion depth guard bounds. A compiler-derived recursive
// `Drop` of a long spine would overflow the native stack, so each peels its
// recursive `left` link iteratively onto a worklist (mirroring `Expr`'s `Drop`).
// Only a genuinely recursive link is moved; a terminal leaf and each node's
// non-spine field (e.g. `Compound.right: SelectCore`) drop in place at
// nesting-bounded depth, which also keeps a re-entrant drop allocation-free.

impl Drop for SelectBody {
    fn drop(&mut self) {
        let mut stack: Vec<Box<SelectBody>> = Vec::new();
        push_compound_spine(self, &mut stack);
        while let Some(mut node) = stack.pop() {
            push_compound_spine(&mut node, &mut stack);
        }
    }
}

/// Move `b`'s `Compound.left` onto `out` iff that link is itself a `Compound` (the
/// recursive spine). A terminal `Select` left is left in place to drop normally.
fn push_compound_spine(b: &mut SelectBody, out: &mut Vec<Box<SelectBody>>) {
    if let SelectBody::Compound { left, .. } = b {
        if matches!(left.as_ref(), SelectBody::Compound { .. }) {
            out.push(std::mem::replace(
                left,
                Box::new(SelectBody::Select(SelectCore::Values(Vec::new()))),
            ));
        }
    }
}

impl Drop for JoinTree {
    fn drop(&mut self) {
        let mut stack: Vec<Box<JoinTree>> = Vec::new();
        push_join_spine(self, &mut stack);
        while let Some(mut node) = stack.pop() {
            push_join_spine(&mut node, &mut stack);
        }
    }
}

/// Move `t`'s `Join.left` onto `out` iff that link is itself a `Join` (the
/// recursive spine); a terminal `Table`/subquery left drops in place.
fn push_join_spine(t: &mut JoinTree, out: &mut Vec<Box<JoinTree>>) {
    if let JoinTree::Join { left, .. } = t {
        if matches!(left.as_ref(), JoinTree::Join { .. }) {
            out.push(std::mem::replace(
                left,
                Box::new(JoinTree::Table(TableOrSubquery::Table {
                    name: QualifiedName { schema: None, name: String::new() },
                    alias: None,
                    indexed: None,
                })),
            ));
        }
    }
}
