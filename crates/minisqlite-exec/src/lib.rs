//! Executor — the execution layer of the engine. It runs a compiled [`Plan`] over
//! the catalog and pager and produces rows, streaming them one at a time so memory
//! stays proportional to the rows in flight rather than the size of intermediates.
//!
//! This root is a thin re-export hub; the seam traits live in the [`executor`]
//! submodule and the real code in the operator/support submodules, so a feature
//! lands in its own file rather than contending on `lib.rs`.
//!
//! Layout:
//! - [`executor`] — the two seam traits ([`Executor`], [`RowCursor`]).
//! - [`runtime`] — [`Runtime`], the per-connection state threaded through each pull.
//! - `env` / `context` — the shared read context and the expression eval context.
//! - `row` / `keys` / `corr_key` — table-row decoding, the canonical dedup key, and the
//!   fold-free correlation-cache key (deliberately distinct from `keys::CellKey`).
//! - `runner` — [`StreamingExecutor`] and the `build_cursor` / `build_dml` dispatch.
//! - `index_build` — [`build_index`], the `CREATE INDEX` backfill the engine calls
//!   (after `create_index`, in the same write txn) to populate a new index from the
//!   table's existing rows, reusing the DML index-maintenance path.
//! - `statement` — [`StatementRoot`], the per-statement wrapper that resets the
//!   `Runtime`'s subquery caches (uncorrelated and correlated) at the start of each
//!   statement's drain.
//! - `ops` — one streaming operator per file.
//!
//! [`Plan`]: minisqlite_plan::Plan

mod context;
mod corr_key;
mod env;
mod executor;
mod index_build;
mod keys;
mod ops;
mod row;
mod runner;
mod runtime;
mod statement;

pub use env::PagerSet;
pub use executor::{Executor, RowCursor};
pub use index_build::build_index;
pub use ops::foreign_key::check_deferred_foreign_keys;
pub use runner::StreamingExecutor;
pub use runtime::Runtime;
