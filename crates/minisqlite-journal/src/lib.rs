//! SQLite's rollback journal: the on-disk `-journal` file codec plus crash-recovery
//! replay. Together these give the engine single-connection durability — a
//! committed transaction survives a reopen, and a transaction interrupted by a
//! crash leaves no trace because its pre-images are rolled back on the next open.
//!
//! The design follows `spec/sqlite-doc/fileformat2.html` §3 (the journal byte
//! format) and `spec/sqlite-doc/atomiccommit.html` §4 (hot-journal playback). This
//! crate is a pure `(page_number, bytes)` layer: it never sees a pager type, so the
//! pager can depend on it without a cycle.
//!
//! Root is a thin re-export hub; the real code lives in the submodules:
//!
//! - [`codec`] — header + page-record byte format, the checksum, and validation.
//! - [`writer`] — the [`Journal`] a pager writes pre-images into during a commit.
//! - [`recover`] — [`recover`], the hot-journal playback run on open.

mod codec;
mod recover;
mod util;
mod writer;

pub use codec::{
    decode_page_record, encode_page_record, has_valid_magic, is_valid_page_size,
    is_valid_sector_size, journal_checksum, page_record_len, JournalHeader, PageRecord,
    DEFAULT_SECTOR_SIZE, HEADER_PREFIX_LEN, MAGIC, PAGE_COUNT_TO_EOF,
};

pub use recover::recover;

pub use writer::{Journal, JournalMode};
