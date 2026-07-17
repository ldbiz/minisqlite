//! minisqlite — the public facade. It re-exports the shared value types and
//! delegates to the engine; the public surface is `Connection` plus the
//! re-exported `Value`/`Row`/`Error`, with the real work living in the component
//! crates (`minisqlite-sql`, `minisqlite-fileformat`, `minisqlite-engine`, and
//! the smaller crates they are split into).
//!
//! SQL behavior follows the SQLite spec; the on-disk file is the official SQLite
//! format in both directions (read what sqlite wrote, write what sqlite reads
//! back). This is a library only — no CLI/REPL, no other-language
//! binding.

pub use minisqlite_types::{Error, QueryResult, Result, Row, Value};

// `Engine` is the one engine-route trait; the facade holds it as
// `Box<dyn Engine>` and names exactly the `open`/`open_in_memory` constructors.
use minisqlite_engine::Engine;
use std::path::Path;

/// A connection to a single database (on-disk file or in-memory).
pub struct Connection {
    engine: Box<dyn Engine>,
}

impl Connection {
    /// Open (or create) the on-disk database at `path`, including a file produced
    /// by real sqlite3 (official format, read + write).
    pub fn open(path: &Path) -> Result<Connection> {
        Ok(Connection {
            engine: minisqlite_engine::open(path)?,
        })
    }

    /// Open a transient in-memory database.
    pub fn open_in_memory() -> Result<Connection> {
        Ok(Connection {
            engine: minisqlite_engine::open_in_memory()?,
        })
    }

    /// Run statements that return no rows (DDL, INSERT/UPDATE/DELETE,
    /// transactions, PRAGMA). Multiple `;`-separated statements are allowed.
    pub fn execute(&mut self, sql: &str) -> Result<()> {
        self.engine.execute(sql)
    }

    /// Run a query and return its result set (column names + rows). Column order
    /// follows the query; row order follows `ORDER BY` (otherwise unspecified, as
    /// in SQLite).
    pub fn query(&mut self, sql: &str) -> Result<QueryResult> {
        self.engine.query(sql)
    }
}
