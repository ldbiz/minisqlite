//! SQL front end: turns SQL text into the engine's parsed statement AST.
//!
//! A separate crate so it compiles and tests independently of storage and
//! execution. This `lib.rs` is a **thin hub**: it only declares the submodules and
//! re-exports their public types plus the single [`parse`] entrypoint (the SQL
//! seam — there must be exactly one `pub fn parse` in this crate). All real code
//! lives in the submodule files so features land in their own file and parallel
//! edits don't contend on the crate root.
//!
//! Layout:
//! - [`token`] / [`keyword`] — the tokenizer and keyword table.
//! - `ast_*` — the comprehensive AST for the whole SQLite SQL surface.
//! - `parser` — the recursive-descent / Pratt parser that builds the AST.

pub mod ast_ddl;
pub mod ast_dml;
pub mod ast_expr;
pub mod ast_select;
pub mod ast_stmt;
pub mod keyword;
mod parser;
pub mod token;

pub use ast_ddl::*;
pub use ast_dml::*;
pub use ast_expr::*;
pub use ast_select::*;
pub use ast_stmt::*;
pub use keyword::Keyword;
pub use token::{tokenize, QuoteKind, Token, TokenKind};

use minisqlite_types::Result;

/// Parse a SQL program (one or more `;`-separated statements) into an [`Ast`].
///
/// This is the SQL seam the engine's front end routes through. It tokenizes the
/// whole input and parses every statement; a trailing `;`, trailing whitespace or
/// comments, and empty statements between `;;` are all allowed. Parse failures are
/// returned as [`minisqlite_types::Error::Sql`].
pub fn parse(sql: &str) -> Result<Ast> {
    parser::parse_program(sql)
}
