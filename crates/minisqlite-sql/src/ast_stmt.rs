//! Top-level statement AST and the small cross-cutting nodes shared by many
//! statements (`QualifiedName`, `With`/`Cte`, `ConflictClause`, transaction and
//! utility statements). Statement index: `spec/sqlite-doc/lang.html`.

use crate::ast_ddl::{AlterTable, CreateIndex, CreateTable, CreateTrigger, CreateView, Drop};
use crate::ast_dml::{Delete, Insert, Update};
use crate::ast_expr::{Expr, Literal};
use crate::ast_select::Select;

/// A parsed SQL program: the statements of the input, in order.
///
/// `statement_sources` runs parallel to `statements`: `statement_sources[i]` is the
/// exact verbatim source text of `statements[i]` — the substring from the first token
/// of the statement through its last token, so it excludes any surrounding whitespace
/// and comments and the terminating `;`, and preserves everything between (internal
/// whitespace, newlines, comments, and case) byte-for-byte. SQLite records this text in
/// `sqlite_schema.sql` for `CREATE` statements and it must match what the user typed, so
/// it is stored owned (not a byte range) to keep the `Ast` self-contained: a caller need
/// not retain the original input buffer to recover a statement's source.
///
/// Invariant: `statement_sources.len() == statements.len()`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Ast {
    pub statements: Vec<Statement>,
    pub statement_sources: Vec<String>,
}

/// One top-level SQL statement. Every documented statement kind has a variant;
/// the parser fully builds the core ones and errors loudly on the long tail.
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Select(Box<Select>),
    Insert(Box<Insert>),
    Update(Box<Update>),
    Delete(Box<Delete>),
    CreateTable(Box<CreateTable>),
    CreateIndex(Box<CreateIndex>),
    CreateView(Box<CreateView>),
    CreateTrigger(Box<CreateTrigger>),
    Drop(Drop),
    AlterTable(Box<AlterTable>),
    /// `BEGIN [DEFERRED|IMMEDIATE|EXCLUSIVE] [TRANSACTION]`.
    Begin { mode: TransactionMode },
    /// `COMMIT` / `END`.
    Commit,
    /// `ROLLBACK [TO [SAVEPOINT] name]`.
    Rollback { to_savepoint: Option<String> },
    /// `SAVEPOINT name`.
    Savepoint(String),
    /// `RELEASE [SAVEPOINT] name`.
    Release(String),
    /// `PRAGMA name [= value | (value)]`.
    Pragma { name: QualifiedName, arg: Option<PragmaArg> },
    /// `VACUUM [schema] [INTO expr]`.
    Vacuum { schema: Option<String>, into: Option<Expr> },
    /// `ANALYZE [name]`.
    Analyze { target: Option<QualifiedName> },
    /// `REINDEX [name]`.
    Reindex { target: Option<QualifiedName> },
    /// `ATTACH [DATABASE] expr AS schema [KEY expr]`.
    Attach { file: Expr, schema: Expr, key: Option<Expr> },
    /// `DETACH [DATABASE] schema`.
    Detach { schema: Expr },
    /// `EXPLAIN stmt`.
    Explain(Box<Statement>),
    /// `EXPLAIN QUERY PLAN stmt`.
    ExplainQueryPlan(Box<Statement>),
}

/// The isolation mode of `BEGIN`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TransactionMode {
    #[default]
    Deferred,
    Immediate,
    Exclusive,
}

/// A possibly schema-qualified object name (`schema.name` or just `name`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualifiedName {
    pub schema: Option<String>,
    pub name: String,
}

impl QualifiedName {
    pub fn bare(name: impl Into<String>) -> Self {
        QualifiedName { schema: None, name: name.into() }
    }
}

/// A leading `WITH [RECURSIVE] ...` clause.
#[derive(Debug, Clone, PartialEq)]
pub struct With {
    pub recursive: bool,
    pub ctes: Vec<Cte>,
}

/// One common table expression within a `WITH` clause.
#[derive(Debug, Clone, PartialEq)]
pub struct Cte {
    pub name: String,
    pub columns: Option<Vec<String>>,
    /// `MATERIALIZED` (`Some(true)`) / `NOT MATERIALIZED` (`Some(false)`) hint.
    pub materialized: Option<bool>,
    pub select: Box<Select>,
}

/// The conflict-resolution algorithm named by `ON CONFLICT` / `OR <algorithm>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictClause {
    Rollback,
    Abort,
    Fail,
    Ignore,
    Replace,
}

/// The argument form of a `PRAGMA`.
#[derive(Debug, Clone, PartialEq)]
pub enum PragmaArg {
    /// `PRAGMA name = value`.
    Equals(PragmaValue),
    /// `PRAGMA name(value)`.
    Call(PragmaValue),
}

/// A pragma value: a literal, or a bare name/keyword (`ON`, `WAL`, `FULL`, ...).
#[derive(Debug, Clone, PartialEq)]
pub enum PragmaValue {
    Literal(Literal),
    Name(String),
}
