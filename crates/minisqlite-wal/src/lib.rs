//! SQLite write-ahead log (`-wal`) codec and algorithms — a STANDALONE layer
//! (see `spec/sqlite-doc/fileformat2.html` §4 and `walformat.html`). This crate is
//! the codec + in-memory algorithm layer only: the WAL byte format, the cumulative
//! Fibonacci-weighted frame checksum, the frame index that answers the reader's
//! `FindFrame(P, mxFrame)` snapshot query, and the checkpoint copy + WAL-reset
//! algorithms. It does **no** file I/O and holds **no** pager/connection state —
//! it operates on `(page_number: u32, bytes: &[u8])` and a small `DbSink` write
//! abstraction, so the pager and the two-connection concurrency layer (a separate,
//! later component) can wire it in without a dependency cycle.
//!
//! It depends only on [`minisqlite_types`] (for [`Error`]/[`Result`]) and
//! [`minisqlite_fileformat`]. It deliberately does NOT depend on the pager: the
//! pager will depend on this crate.
//!
//! This root is a thin re-export hub; the real code lives in the submodules:
//!
//! - [`checksum`] — the cumulative WAL checksum ([`wal_checksum`]).
//! - [`codec`] — the 32-byte [`WalHeader`], the 24-byte [`FrameHeader`], byte
//!   offsets, and frame writing ([`WalBuilder`]).
//! - [`index`] — [`scan`] a WAL into a [`WalIndex`] and answer [`WalIndex::resolve`].
//! - [`checkpoint`] — the [`DbSink`] sink, [`checkpoint`], and [`reset_header`].

pub mod checkpoint;
pub mod checksum;
pub mod codec;
pub mod index;

pub use checksum::wal_checksum;

pub use codec::{
    frame_checksum, frame_offset, frame_page_data_offset, frame_stride, is_commit_marker,
    write_frame, FrameHeader, WalBuilder, WalHeader, FRAME_HEADER_SIZE, WAL_FILE_FORMAT,
    WAL_HEADER_SIZE, WAL_MAGIC_BE, WAL_MAGIC_LE,
};

pub use index::{scan, FrameMeta, WalIndex};

pub use checkpoint::{checkpoint, reset_header, CheckpointOptions, CheckpointResult, DbSink};
