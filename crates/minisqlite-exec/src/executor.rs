//! The execution seam: [`Executor`] and its [`RowCursor`]. This is the one place
//! rows are produced — the executor runs a [`Plan`] over the catalog and pager and
//! streams the result.
//!
//! Load-bearing performance invariant: execution STREAMS. An operator yields rows
//! through a [`RowCursor`], one row at a time, so a scan, join, or subquery over a
//! large input does not first materialize its result into a `Vec<Row>`. Operators
//! compose by pulling from their inputs, so memory stays proportional to the rows in
//! flight rather than the size of the intermediates. The facade collects only the
//! final result set the caller asked for; everything before it flows row by row. Do
//! not buffer an intermediate into a vector unless an operator provably must (for
//! example a sort or a hash-join build side), and bound it even then.
//!
//! Two seam facts pinned here (both traits stay singular in this crate — the seam
//! guard enforces it):
//!
//! * The catalog is borrowed `&dyn Catalog` — READ-ONLY. Execution only *reads*
//!   schema and *mutates data* through the pager; DDL / schema mutation
//!   (`CREATE TABLE`/`CREATE INDEX`) is the engine's job via the [`Catalog`] seam,
//!   never the executor's.
//! * [`RowCursor::next_row`] threads a `&mut `[`Runtime`] on every pull. `Runtime`
//!   is the per-connection state (a deterministic RNG, the change counters, the
//!   bind parameters) that expression evaluation needs. Passing it per pull rather
//!   than storing it in each operator means several operators can evaluate
//!   expressions (a `random()` call mutates the RNG, an `INSERT` bumps the counters)
//!   without each holding a `&mut` to it. After draining a DML cursor the engine
//!   reads `rt.changes()` / `rt.last_insert_rowid()` from the same `Runtime`.

use minisqlite_catalog::Catalog;
use minisqlite_plan::Plan;
use minisqlite_types::{Result, Row};

use crate::env::PagerSet;
use crate::runtime::Runtime;

/// A pull-based row stream (the volcano model): each call returns the next row, or
/// `None` at end of input. Composing cursors keeps memory at one row per operator
/// rather than a full materialized intermediate.
///
/// `rt` is the connection [`Runtime`] threaded through every pull so an operator can
/// evaluate expressions (parameters, `random()`, the change counters) without owning
/// a `&mut` to that shared state for its whole life.
pub trait RowCursor {
    fn next_row(&mut self, rt: &mut Runtime) -> Result<Option<Row>>;

    /// Consume this cursor and return how many rows it WOULD yield, without requiring
    /// each row to be materialized.
    ///
    /// This is an OPTIMIZATION HOOK, not a new route: the default just pulls and
    /// discards via [`next_row`](RowCursor::next_row), so it is always correct and every
    /// cursor keeps working unchanged whether or not it overrides it. A cursor MAY
    /// override it to compute the SAME count more cheaply — the hash join sums its
    /// matched-bucket sizes instead of building (and immediately dropping) a combined
    /// `Row` for every pair, which is what makes `count(*)` over a high-cardinality join
    /// stream in bounded memory and time. An override changes cost, never the count: it
    /// MUST return exactly what the default drain would. A consumer that needs only the
    /// row count (e.g. `count(*)` with no other column use — see the aggregate operator)
    /// calls this instead of looping `next_row`, letting the producer skip materializing
    /// rows the consumer would throw away.
    fn count_rows(&mut self, rt: &mut Runtime) -> Result<usize> {
        let mut n = 0usize;
        while self.next_row(rt)?.is_some() {
            n += 1;
        }
        Ok(n)
    }
}

/// The single execution seam. Declared exactly once, in this crate (enforced by
/// `minisqlite/tests/seams.rs`), so the execution route cannot fork.
///
/// It runs a plan and returns a streaming cursor, borrowing the catalog and the pager
/// SET rather than copying them. The borrows last as long as the cursor, so the cursor
/// can pull from the live schema and storage without owning a snapshot. The catalog is
/// borrowed `&dyn Catalog` (read-only): the executor reads schema but never mutates it —
/// schema mutation belongs to the engine via the `Catalog` seam.
///
/// Storage is passed as a [`PagerSet`] — the connection's live stores indexed by
/// [`DbIndex`](minisqlite_types::DbIndex) (`main` = 0, `temp` = 1, attached = 2..). A read
/// derives a SHARED view over the set (many cursors read at once, possibly across
/// namespaces); DML reborrows one element MUTABLY for its write. The engine hands over its
/// whole store slice ([`PagerSet::Set`]); a single-store caller can hand over one store
/// ([`PagerSet::One`]), which for a `main`-only plan is byte-identical to the former
/// single-`&mut dyn Pager` shape.
pub trait Executor {
    fn execute<'a>(
        &'a mut self,
        plan: &'a Plan,
        catalog: &'a dyn Catalog,
        pagers: PagerSet<'a>,
    ) -> Result<Box<dyn RowCursor + 'a>>;
}
