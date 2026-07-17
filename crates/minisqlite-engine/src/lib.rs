//! The engine behind the pinned API — it answers SQL against the on-disk (or
//! in-memory) database by routing each statement through the owned component seams:
//! parse (`minisqlite-sql`) → plan (`minisqlite-plan`) → execute (`minisqlite-exec`)
//! over the schema (`minisqlite-catalog`) and storage (`minisqlite-pager`). Below
//! these seams the architecture is yours; split internals into smaller crates as
//! they grow so parallel work stays independently buildable.
//!
//! SEAM (stability invariant, enforced by `minisqlite/tests/seams.rs`). The engine
//! route is a fixed set of one-per-crate traits and named entrypoints: `Engine`
//! here (the trait the facade routes every statement through) with `open` /
//! `open_in_memory` (the one pair of constructors the facade names); `parse` in
//! `minisqlite-sql`; `Pager` (storage), `Catalog` (schema), `Planner` (planning),
//! and `Executor` (execution) each alone in their crate. Add implementations and
//! split internals freely, but do not introduce a competing engine or seam route in
//! another crate: a route change is expand-then-contract behind these traits, so
//! drift is a compile or guard error rather than a hand-maintained "which path is
//! active" contract.

use minisqlite_types::{QueryResult, Result};
use std::path::Path;

// The concrete engine and its per-kind statement handling live in submodules so a
// feature lands in its own file and this hub stays thin. `lib.rs` keeps only the
// `Engine` seam trait and the two named constructors the facade routes through.
mod analyze;
mod attach;
mod ctas;
mod deferred_fk;
mod dispatch;
mod engine;
mod namespace;
mod pragma;
mod txn;
mod vacuum;

// The owned component seams the engine routes through, re-exported so the whole
// route is nameable from one place. A concrete engine constructed in `open` /
// `open_in_memory` wires concrete implementations of these behind the `Engine`
// trait without changing the public surface.
pub use minisqlite_catalog::Catalog;
pub use minisqlite_exec::{Executor, RowCursor};
pub use minisqlite_pager::Pager;
pub use minisqlite_plan::{Plan, Planner};
pub use minisqlite_sql::parse;

/// The single engine interface the facade (`minisqlite::Connection`) routes
/// through. Object-safe so the facade can hold `Box<dyn Engine>` and swap the
/// implementation behind this one stable seam without changing the public surface.
pub trait Engine {
    /// Run a statement that returns no rows (DDL, `INSERT`/`UPDATE`/`DELETE`,
    /// transactions, `PRAGMA`). Multiple `;`-separated statements are allowed.
    fn execute(&mut self, sql: &str) -> Result<()>;

    /// Run a query and return its result set (column names + rows).
    fn query(&mut self, sql: &str) -> Result<QueryResult>;
}

/// Open (or create) the on-disk database at `path` (official SQLite format, read +
/// write). The one on-disk constructor the facade names; it builds the concrete
/// [`engine::SqlEngine`] over a disk-backed `Pager` and returns it boxed as
/// `dyn Engine`.
pub fn open(path: &Path) -> Result<Box<dyn Engine>> {
    Ok(Box::new(engine::SqlEngine::open(path)?))
}

/// Open a transient in-memory database. The one in-memory constructor the facade
/// names; it builds the concrete [`engine::SqlEngine`] over an in-memory `Pager`.
pub fn open_in_memory() -> Result<Box<dyn Engine>> {
    Ok(Box::new(engine::SqlEngine::open_in_memory()?))
}
