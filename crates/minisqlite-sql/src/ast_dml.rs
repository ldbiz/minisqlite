//! DML statement AST: INSERT, UPDATE, DELETE (with UPSERT and RETURNING).
//! Specs: `lang_insert.html`, `lang_update.html`, `lang_delete.html`,
//! `lang_upsert.html`, `lang_returning.html`.

use crate::ast_ddl::IndexedColumn;
use crate::ast_expr::Expr;
use crate::ast_select::{FromClause, IndexedClause, ResultColumn, Select};
use crate::ast_stmt::{ConflictClause, QualifiedName, With};

/// `INSERT [OR conflict] INTO table [(cols)] <source> [upsert...] [RETURNING ...]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Insert {
    pub with: Option<With>,
    /// `INSERT OR REPLACE/IGNORE/ABORT/FAIL/ROLLBACK` (also plain `REPLACE INTO`,
    /// which is `INSERT OR REPLACE`).
    pub or_conflict: Option<ConflictClause>,
    pub table: QualifiedName,
    pub alias: Option<String>,
    pub columns: Option<Vec<String>>,
    pub source: InsertSource,
    /// Zero or more chained `ON CONFLICT ...` upsert clauses (SQLite allows more
    /// than one). Empty means no upsert.
    pub upsert: Vec<Upsert>,
    pub returning: Vec<ResultColumn>,
}

/// The row source of an INSERT.
#[derive(Debug, Clone, PartialEq)]
pub enum InsertSource {
    /// `VALUES (row), (row), ...`.
    Values(Vec<Vec<Expr>>),
    /// `SELECT ...`.
    Select(Box<Select>),
    /// `DEFAULT VALUES`.
    DefaultValues,
}

/// One `ON CONFLICT [target] DO ...` upsert clause.
#[derive(Debug, Clone, PartialEq)]
pub struct Upsert {
    /// The conflict target `(indexed-columns) [WHERE expr]`; `None` for a bare
    /// `ON CONFLICT DO NOTHING`.
    pub target: Option<UpsertTarget>,
    pub action: UpsertAction,
}

/// The conflict target of an upsert.
#[derive(Debug, Clone, PartialEq)]
pub struct UpsertTarget {
    pub columns: Vec<IndexedColumn>,
    pub where_clause: Option<Expr>,
}

/// What an upsert does on conflict.
#[derive(Debug, Clone, PartialEq)]
pub enum UpsertAction {
    /// `DO NOTHING`.
    Nothing,
    /// `DO UPDATE SET ... [WHERE expr]`.
    Update { set: Vec<SetClause>, where_clause: Option<Expr> },
}

/// `UPDATE [OR conflict] table [indexed] SET ... [FROM ...] [WHERE ...] [RETURNING]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Update {
    pub with: Option<With>,
    pub or_conflict: Option<ConflictClause>,
    pub table: QualifiedName,
    pub alias: Option<String>,
    pub indexed: Option<IndexedClause>,
    pub set: Vec<SetClause>,
    pub from: Option<FromClause>,
    pub where_clause: Option<Expr>,
    pub returning: Vec<ResultColumn>,
}

/// One assignment in a `SET` list.
#[derive(Debug, Clone, PartialEq)]
pub enum SetClause {
    /// `column = expr`.
    Column { name: String, value: Expr },
    /// `(a, b, ...) = expr` — a column-list assignment (RHS is a row value or
    /// a scalar subquery returning that many columns).
    Columns { names: Vec<String>, value: Expr },
}

/// `DELETE FROM table [indexed] [WHERE ...] [RETURNING]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Delete {
    pub with: Option<With>,
    pub table: QualifiedName,
    pub alias: Option<String>,
    pub indexed: Option<IndexedClause>,
    pub where_clause: Option<Expr>,
    pub returning: Vec<ResultColumn>,
}
