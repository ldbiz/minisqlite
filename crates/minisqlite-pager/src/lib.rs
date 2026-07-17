//! Pager — the storage seam. It hands out database pages and runs transactions
//! over either an in-memory store or the on-disk file (through
//! `minisqlite-fileformat`). It is the one owned storage seam the rest of the
//! engine builds on; the page-cache, b-tree, freelist, and overflow design beneath
//! it are yours.
//!
//! Load-bearing performance and correctness invariant: reads BORROW pages from the
//! cache and never copy the whole database, and writes go through a copy-on-write
//! transaction — a page is cloned lazily on its first write within the transaction
//! and its pre-image is journalled — so a statement touches only the pages it
//! actually changes, never a deep copy of the database. Work stays proportional to
//! the data touched, and durable commit, rollback, and crash recovery all fall out
//! of the same journalled copy-on-write path. Do not add a "snapshot the whole
//! database" entry point; it defeats both the performance and the recovery model.
//!
//! This crate root is a thin hub: it declares the shared `PageId`, re-exports the
//! on-disk `codec`, and re-exports the `Pager` seam (`pager`) with its in-memory
//! backing (`mempager`). Real code lives in those submodules so a feature lands in
//! its own file instead of contending on the root:
//!
//! - `pager` — the `Pager` seam trait (the one storage seam).
//! - `dbinfo` — page-1 header reads through the seam (text encoding §1.3.13, vacuum flag).
//! - `store` — `PageStore`, the committed-image backing seam, and its `MemStore`.
//! - `cow` — `Cow<S>`, the copy-on-write transaction layer shared by every backing.
//! - `av_commit` — commit-time auto_vacuum finalize + FULL compaction as ONE durable txn.
//! - `alloc` — the freelist allocate/free policy over `Cow`.
//! - `ptrmap_build` — the auto_vacuum WRITE side: rebuild ptrmap pages + offset 52 at commit.
//! - `reclaim` — the auto_vacuum RECLAMATION side: relocate + truncate free pages (FULL /
//!   `incremental_vacuum`).
//! - `roots_first` — the auto_vacuum §1.8 ROOTS-FIRST side: relocate b-tree roots to the
//!   front of the file at commit (invoked by `ptrmap_build`, reusing `reclaim`'s rewriters).
//! - `mempager` — `MemPager`, the in-memory `Pager` (a thin wrapper over `Cow`).
//! - `diskstore` — `DiskStore`, the on-disk committed store with a rollback journal.
//! - `diskpager` — `DiskPager`, the disk-backed `Pager` (rollback OR WAL, one `Cow`).
//! - `page1` — shared page-1 header maintenance for every durable commit path.
//! - `walstore` — `WalStore`, the WAL-mode committed store (`-wal` writer + reads).
//! - `walindex` — the incrementally-maintained WAL frame index (hot-path resolve).
//! - `walshared` — the cross-connection WAL coordinator and process-global registry.
//! - `busy` — BUSY signalling for the WAL write lock (`is_busy`).
//! - `checkpoint` — the `PRAGMA wal_checkpoint` mode/report vocabulary threaded through the seam.

/// The on-disk codec the disk-backed pager reads and writes through (page layout,
/// varints, record serial types). Re-exported so the storage seam names the format
/// layer it builds on from one place.
pub use minisqlite_fileformat as codec;

/// A page number, 1-based as in the file format. Page 1 carries the database
/// header in its first 100 bytes.
pub type PageId = u32;

mod alloc;
mod av_commit;
mod busy;
mod checkpoint;
mod cow;
mod dbinfo;
mod diskpager;
mod diskstore;
mod mempager;
mod page1;
mod pager;
mod ptrmap_build;
mod reclaim;
mod roots_first;
mod store;
mod walindex;
mod walshared;
mod walstore;

pub use busy::is_busy;
pub use checkpoint::{CheckpointMode, CheckpointReport};
pub use dbinfo::{read_database_header, text_encoding_of};
pub use diskpager::DiskPager;
pub use mempager::MemPager;
pub use pager::Pager;
pub use reclaim::VacuumOutcome;
