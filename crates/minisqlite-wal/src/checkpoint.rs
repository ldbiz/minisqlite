//! The checkpoint copy algorithm (fileformat2 §4.3) and WAL reset (§4.4).
//!
//! A checkpoint transfers the newest committed version of every page in the WAL
//! back into the database file. The durability ordering is a write barrier before
//! and after (§4.3): the WAL is fsync'd first (so the frames we copy are durable),
//! the pages are copied, then the database is fsync'd. This layer is pager-free, so
//! the two barriers are modeled explicitly: a `wal_sync` closure (called before the
//! copy) and [`DbSink::sync`] (called after).
//!
//! A checkpoint need not run to completion. Readers holding an older snapshot still
//! read some pages from the database file, so a checkpoint must not overwrite the
//! database past the oldest reader's mark (§2.3.1). That bound is passed as
//! [`CheckpointOptions::reader_limit`]; the checkpoint clamps it *down* to a commit
//! boundary (you cannot apply half a transaction) and reports whether it drained
//! all the way to `mxFrame`.

use minisqlite_types::{Error, Result};

use crate::codec::WalHeader;
use crate::index::WalIndex;

/// Where checkpointed pages are written: the database file. Kept deliberately tiny
/// (three methods) so the pager can implement it over a real file and tests can
/// implement it over an in-memory image, with no pager type leaking into this crate.
pub trait DbSink {
    /// Write `data` (exactly one page) as page `page_no` (1-based) of the database.
    fn write_page(&mut self, page_no: u32, data: &[u8]) -> Result<()>;

    /// Set the database file length to `n_pages` pages (truncating a database that
    /// shrank across the checkpointed transactions, or extending one that grew).
    fn set_size_pages(&mut self, n_pages: u32) -> Result<()>;

    /// Flush the database file to durable storage (the xSync after the copy).
    fn sync(&mut self) -> Result<()>;
}

/// Inputs that bound a checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointOptions {
    /// Frames already copied into the database by prior checkpoints (nBackfill).
    /// Pages whose newest frame is ≤ this are already in the database and are
    /// skipped. Use 0 for a fresh checkpoint.
    pub nbackfill: u32,
    /// The highest frame the checkpoint may drain, from the oldest active reader's
    /// read-mark (the minimum over readers holding `WAL_READ_LOCK(N)` for N>0).
    /// `None` means no reader constrains it — a full drain up to `mxFrame`. The
    /// checkpoint additionally clamps this to a commit boundary and to `mxFrame`.
    pub reader_limit: Option<u32>,
}

impl Default for CheckpointOptions {
    /// A full, fresh checkpoint: start at nBackfill 0 with no reader constraint.
    fn default() -> Self {
        CheckpointOptions { nbackfill: 0, reader_limit: None }
    }
}

impl CheckpointOptions {
    /// A full, fresh checkpoint (nBackfill 0, no reader limit).
    pub fn full() -> Self {
        CheckpointOptions::default()
    }
}

/// The outcome of a checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointResult {
    /// The new nBackfill: the highest frame now guaranteed present in the database
    /// (the commit boundary the checkpoint drained to). Unchanged from the input
    /// nBackfill when nothing could be copied.
    pub frames_backfilled: u32,
    /// How many pages were written to the database this call.
    pub pages_written: u32,
    /// The database size in pages after the copy (the db-size of the commit the
    /// checkpoint drained to), or 0 if nothing was drained.
    pub db_size_pages: u32,
    /// True if the checkpoint drained all the way to `mxFrame` (a complete
    /// checkpoint, after which the WAL may be reset). False for a partial drain
    /// bounded by a reader.
    pub complete: bool,
}

/// Run a checkpoint, copying the newest committed version of every page in
/// `(nbackfill, safe]` from the WAL into `sink`, where `safe` is the last commit
/// frame at or before `min(mxFrame, reader_limit)`.
///
/// `wal_sync` is invoked once, before any page is written, to make the WAL durable
/// (the pre-copy write barrier). [`DbSink::sync`] is invoked once after the copy
/// (the post-copy barrier). Both are skipped when there is nothing to copy.
///
/// Returns a [`CheckpointResult`]; `complete` is true iff the drain reached
/// `mxFrame`.
pub fn checkpoint<S, F>(
    wal: &[u8],
    index: &WalIndex,
    sink: &mut S,
    opts: CheckpointOptions,
    mut wal_sync: F,
) -> Result<CheckpointResult>
where
    S: DbSink,
    F: FnMut() -> Result<()>,
{
    let mx = index.mx_frame();
    // The reader limit caps the drain; absent one, drain everything committed.
    let target = opts.reader_limit.unwrap_or(mx).min(mx);
    // Clamp down to a commit boundary: never apply a partial transaction.
    let safe = index.last_commit_at_or_before(target);
    let start = opts.nbackfill;

    // Nothing new to drain: either the WAL holds no reachable commit past what is
    // already backfilled, or a reader pins us at/behind nBackfill. When `safe == mx`
    // here, `start >= safe == mx`, so the database already holds everything committed
    // — that is a complete checkpoint even though this call copied nothing.
    if safe <= start {
        return Ok(CheckpointResult {
            frames_backfilled: start,
            pages_written: 0,
            db_size_pages: index.db_size_at(start),
            complete: mx != 0 && safe == mx,
        });
    }

    // Pre-copy barrier: the frames we are about to copy must be durable first.
    wal_sync()?;

    // Copy each page's newest committed version in (start, safe]. Iterating frames
    // in ascending order and writing a frame only when it is that page's newest
    // frame ≤ safe writes every touched page exactly once, deterministically.
    let mut pages_written = 0u32;
    for f in (start + 1)..=safe {
        // `f` is within `1..=safe <= mx_frame <= n_valid_frames`, so this is always
        // present; go through the checked accessor and fail closed rather than index
        // raw, so a future bound slip surfaces as an error instead of a panic.
        let meta = index.frame(f).ok_or_else(|| {
            Error::format(format!("checkpoint: frame {f} missing from a consistent index"))
        })?;
        if index.resolve(meta.page_no, safe) == Some(f) {
            let data = index.frame_page_data(wal, f)?;
            sink.write_page(meta.page_no, data)?;
            pages_written += 1;
        }
    }

    let db_size_pages = index.db_size_at(safe);
    sink.set_size_pages(db_size_pages)?;

    // Post-copy barrier: the database must be durable before the WAL can be reset.
    sink.sync()?;

    Ok(CheckpointResult {
        frames_backfilled: safe,
        pages_written,
        db_size_pages,
        complete: safe == mx,
    })
}

/// The WAL header for a reset (§4.4): after a complete checkpoint the next write can
/// overwrite the WAL from the beginning. Salt-1 is incremented and salt-2 is
/// replaced by `new_salt2` (the caller supplies fresh randomness — kept out of this
/// pure function so it stays deterministic and testable), which invalidates every
/// already-checkpointed frame still in the file (their salts no longer match). The
/// checkpoint sequence number is incremented; the page size and checksum byte order
/// are preserved. The returned header carries a freshly computed checksum.
///
/// Truncating the WAL file on reset is optional (§4.4) and left to the caller; the
/// salt change alone makes the leftover frames fail validation on the next scan.
pub fn reset_header(prev: &WalHeader, new_salt2: u32) -> WalHeader {
    WalHeader::new(
        prev.page_size,
        prev.salt1.wrapping_add(1),
        new_salt2,
        prev.checkpoint_seq.wrapping_add(1),
        prev.big_endian,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::WalBuilder;
    use crate::index::scan;
    use std::collections::BTreeMap;

    /// An in-memory database image standing in for the real file.
    #[derive(Default)]
    struct MemDb {
        pages: BTreeMap<u32, Vec<u8>>,
        size_pages: u32,
        syncs: u32,
    }
    impl DbSink for MemDb {
        fn write_page(&mut self, page_no: u32, data: &[u8]) -> Result<()> {
            self.pages.insert(page_no, data.to_vec());
            Ok(())
        }
        fn set_size_pages(&mut self, n_pages: u32) -> Result<()> {
            self.size_pages = n_pages;
            // Model a real file truncation: pages beyond the new size are gone.
            self.pages.retain(|&pgno, _| pgno <= n_pages);
            Ok(())
        }
        fn sync(&mut self) -> Result<()> {
            self.syncs += 1;
            Ok(())
        }
    }

    const PS: u32 = 512;

    fn page(byte: u8) -> Vec<u8> {
        vec![byte; PS as usize]
    }

    #[test]
    fn full_checkpoint_copies_newest_committed_versions() {
        let mut b = WalBuilder::new(WalHeader::new(PS, 100, 200, 0, false));
        // Txn1: page 1 = 0xa1, page 2 = 0xa2, commit (db size 2).
        b.append_frame(1, 0, &page(0xa1)).unwrap();
        b.append_frame(2, 2, &page(0xa2)).unwrap();
        // Txn2: page 1 updated to 0xb1, commit (db size 2).
        b.append_frame(1, 2, &page(0xb1)).unwrap();
        let wal = b.into_bytes();
        let idx = scan(&wal);
        assert_eq!(idx.mx_frame(), 3);

        let mut db = MemDb::default();
        let mut synced = false;
        let res = checkpoint(&wal, &idx, &mut db, CheckpointOptions::full(), || {
            synced = true;
            Ok(())
        })
        .unwrap();

        assert!(synced, "WAL must be synced before copying");
        assert_eq!(db.syncs, 1, "db must be synced after copying");
        assert!(res.complete);
        assert_eq!(res.frames_backfilled, 3);
        assert_eq!(res.db_size_pages, 2);
        // Page 1 newest is the txn2 version; page 2 is the txn1 version.
        assert_eq!(db.pages.get(&1).unwrap(), &page(0xb1));
        assert_eq!(db.pages.get(&2).unwrap(), &page(0xa2));
        assert_eq!(db.size_pages, 2);
        // Page 1 written once (only its newest frame), plus page 2 = 2 writes.
        assert_eq!(res.pages_written, 2);
    }

    #[test]
    fn partial_checkpoint_stops_at_reader_mark() {
        let mut b = WalBuilder::new(WalHeader::new(PS, 1, 2, 0, false));
        b.append_frame(1, 0, &page(0xa1)).unwrap();
        b.append_frame(2, 2, &page(0xa2)).unwrap(); // commit #1 at frame 2, db size 2
        b.append_frame(1, 3, &page(0xb1)).unwrap(); // commit #2 at frame 3, db size 3
        let wal = b.into_bytes();
        let idx = scan(&wal);
        assert_eq!(idx.mx_frame(), 3);

        // A reader is pinned at the first commit (mark = frame 2): the checkpoint may
        // not drain past it.
        let mut db = MemDb::default();
        let res = checkpoint(
            &wal,
            &idx,
            &mut db,
            CheckpointOptions { nbackfill: 0, reader_limit: Some(2) },
            || Ok(()),
        )
        .unwrap();

        assert!(!res.complete, "reader mark below mxFrame ⇒ incomplete");
        assert_eq!(res.frames_backfilled, 2);
        assert_eq!(res.db_size_pages, 2);
        // Only the first transaction's versions are copied; page 1 is still 0xa1.
        assert_eq!(db.pages.get(&1).unwrap(), &page(0xa1));
        assert_eq!(db.pages.get(&2).unwrap(), &page(0xa2));
        assert_eq!(db.size_pages, 2);
    }

    #[test]
    fn reader_limit_between_commits_clamps_to_earlier_commit() {
        let mut b = WalBuilder::new(WalHeader::new(PS, 1, 2, 0, false));
        b.append_frame(1, 1, &page(0xa1)).unwrap(); // commit at frame 1, db size 1
        b.append_frame(2, 0, &page(0xa2)).unwrap(); // non-commit
        b.append_frame(2, 2, &page(0xb2)).unwrap(); // commit at frame 3, db size 2
        let wal = b.into_bytes();
        let idx = scan(&wal);

        // A limit of 2 is not a commit boundary; it must clamp down to commit at 1.
        let mut db = MemDb::default();
        let res = checkpoint(
            &wal,
            &idx,
            &mut db,
            CheckpointOptions { nbackfill: 0, reader_limit: Some(2) },
            || Ok(()),
        )
        .unwrap();

        assert!(!res.complete, "clamped below mxFrame ⇒ incomplete");
        assert_eq!(res.frames_backfilled, 1);
        assert_eq!(res.db_size_pages, 1);
        assert_eq!(db.pages.len(), 1);
        assert_eq!(db.pages.get(&1).unwrap(), &page(0xa1));
        assert!(!db.pages.contains_key(&2));
    }

    #[test]
    fn incremental_checkpoint_skips_already_backfilled() {
        let mut b = WalBuilder::new(WalHeader::new(PS, 1, 2, 0, false));
        b.append_frame(1, 0, &page(0xa1)).unwrap();
        b.append_frame(2, 2, &page(0xa2)).unwrap(); // commit at 2
        b.append_frame(3, 3, &page(0xa3)).unwrap(); // commit at 3
        let wal = b.into_bytes();
        let idx = scan(&wal);

        // Resume from nBackfill = 2: only frame 3 (page 3) should be copied.
        let mut db = MemDb::default();
        let res = checkpoint(
            &wal,
            &idx,
            &mut db,
            CheckpointOptions { nbackfill: 2, reader_limit: None },
            || Ok(()),
        )
        .unwrap();
        assert!(res.complete);
        assert_eq!(res.frames_backfilled, 3);
        assert_eq!(res.pages_written, 1);
        assert_eq!(db.pages.get(&3).unwrap(), &page(0xa3));
        assert!(!db.pages.contains_key(&1), "page 1 was already backfilled, skip");
    }

    #[test]
    fn checkpoint_of_empty_wal_does_nothing() {
        let wal = WalBuilder::new(WalHeader::new(PS, 1, 2, 0, false)).into_bytes();
        let idx = scan(&wal);
        assert_eq!(idx.mx_frame(), 0);
        let mut db = MemDb::default();
        let mut synced = false;
        let res = checkpoint(&wal, &idx, &mut db, CheckpointOptions::full(), || {
            synced = true;
            Ok(())
        })
        .unwrap();
        assert_eq!(res.pages_written, 0);
        assert!(!res.complete, "an empty WAL (mxFrame 0) is not a complete drain");
        assert!(!synced, "no barrier when nothing to copy");
        assert_eq!(db.syncs, 0);
        assert!(db.pages.is_empty());
    }

    #[test]
    fn recheckpoint_when_fully_backfilled_is_a_complete_noop() {
        let mut b = WalBuilder::new(WalHeader::new(PS, 1, 2, 0, false));
        b.append_frame(1, 1, &page(0xa1)).unwrap(); // commit at 1
        b.append_frame(2, 2, &page(0xa2)).unwrap(); // commit at 2 (mxFrame)
        let wal = b.into_bytes();
        let idx = scan(&wal);
        assert_eq!(idx.mx_frame(), 2);

        // nBackfill already == mxFrame: nothing to copy, no barriers fire, but the
        // checkpoint reports complete (the WAL may be reset). This exercises the
        // no-copy branch's `complete = mx != 0 && safe == mx`.
        let mut db = MemDb::default();
        let mut synced = false;
        let res = checkpoint(
            &wal,
            &idx,
            &mut db,
            CheckpointOptions { nbackfill: 2, reader_limit: None },
            || {
                synced = true;
                Ok(())
            },
        )
        .unwrap();
        assert!(res.complete);
        assert_eq!(res.pages_written, 0);
        assert_eq!(res.frames_backfilled, 2);
        assert!(!synced);
        assert_eq!(db.syncs, 0);
        assert!(db.pages.is_empty());
    }

    #[test]
    fn checkpoint_truncates_a_shrunk_database() {
        let mut b = WalBuilder::new(WalHeader::new(PS, 1, 2, 0, false));
        b.append_frame(1, 0, &page(0xa1)).unwrap();
        b.append_frame(2, 0, &page(0xa2)).unwrap();
        b.append_frame(3, 3, &page(0xa3)).unwrap(); // commit at 3: db grows to 3 pages
        b.append_frame(1, 2, &page(0xb1)).unwrap(); // commit at 4: db SHRINKS to 2 pages
        let wal = b.into_bytes();
        let idx = scan(&wal);
        assert_eq!(idx.mx_frame(), 4);
        assert_eq!(idx.db_size_pages(), 2);

        let mut db = MemDb::default();
        let res = checkpoint(&wal, &idx, &mut db, CheckpointOptions::full(), || Ok(())).unwrap();
        assert!(res.complete);
        assert_eq!(res.db_size_pages, 2);
        assert_eq!(db.size_pages, 2);
        // Page 3 was copied from frame 3 but truncated away by the shrink to 2 pages.
        assert!(!db.pages.contains_key(&3), "page beyond the final db size is truncated");
        assert_eq!(db.pages.get(&1).unwrap(), &page(0xb1));
        assert_eq!(db.pages.get(&2).unwrap(), &page(0xa2));
    }

    #[test]
    fn full_checkpoint_big_endian_end_to_end() {
        // Exercise the whole build→scan→checkpoint path under big-endian checksums.
        let mut b = WalBuilder::new(WalHeader::new(PS, 3, 4, 0, true));
        b.append_frame(1, 0, &page(0xa1)).unwrap();
        b.append_frame(2, 2, &page(0xa2)).unwrap();
        b.append_frame(1, 2, &page(0xb1)).unwrap();
        let wal = b.into_bytes();
        let idx = scan(&wal);
        assert!(idx.big_endian());
        assert_eq!(idx.mx_frame(), 3);

        let mut db = MemDb::default();
        let res = checkpoint(&wal, &idx, &mut db, CheckpointOptions::full(), || Ok(())).unwrap();
        assert!(res.complete);
        assert_eq!(res.db_size_pages, 2);
        assert_eq!(db.pages.get(&1).unwrap(), &page(0xb1));
        assert_eq!(db.pages.get(&2).unwrap(), &page(0xa2));
    }

    #[test]
    fn reset_header_bumps_salt1_randomizes_salt2_and_seq() {
        let prev = WalHeader::new(PS, 10, 20, 5, false);
        let next = reset_header(&prev, 0xfeed_face);
        assert_eq!(next.salt1, 11);
        assert_eq!(next.salt2, 0xfeed_face);
        assert_ne!(next.salt2, prev.salt2);
        assert_eq!(next.checkpoint_seq, 6);
        assert_eq!(next.page_size, prev.page_size);
        assert_eq!(next.big_endian, prev.big_endian);
        assert!(next.verify_checksum());
    }
}
