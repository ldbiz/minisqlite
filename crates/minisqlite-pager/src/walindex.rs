//! `FrameIndex` — an INCREMENTALLY-maintained frame index over the committed prefix
//! of a WAL: the pager's hot-path counterpart to [`minisqlite_wal::index::WalIndex`].
//!
//! ## Why this exists (it is not gratuitous duplication of the wal crate)
//! [`minisqlite_wal::index::scan`] is the ONLY way to build a `WalIndex`, and it is
//! O(total WAL bytes): a single pass re-decodes and re-validates the cumulative
//! checksum over every frame's page data. Calling it to refresh a connection's
//! snapshot on every read/write transaction, or to re-index after every commit's
//! append, makes each such operation cost proportional to the WHOLE log — i.e.
//! O(N^2) as the WAL grows between checkpoints — which violates the crate's budget
//! ("frame append is O(pages touched)", "no O(n^2) scans on hot paths").
//!
//! The wal crate is a separately-owned pure codec and offers no incremental index
//! builder, and it cannot be edited from here. So the pager keeps its own index that:
//! - is built ONCE from a `scan` at open and after a reset (cheap then: the WAL is
//!   empty or small), via [`FrameIndex::from_wal_index`];
//! - is EXTENDED in place by [`FrameIndex::extend_commit`] using the frame metadata a
//!   commit already computed while writing frames — no re-scan, O(pages touched);
//! - is cheap to clone / `Arc`-share for a per-connection snapshot (a memcpy of the
//!   frame table, never a re-checksum).
//!
//! ## Invariant it relies on
//! The shared WAL image always ends exactly at a commit frame (uncommitted crash
//! leftovers are trimmed — see [`crate::walshared`]), and a commit appends a whole
//! transaction ending in its commit frame. So this index only ever holds COMMITTED
//! frames: `frames.len() == mx_frame`, and every stored frame is resolvable. Its
//! `resolve` / `db_size_at` / `running_checksum_for_append` mirror `WalIndex`
//! semantics exactly, so the resolve path is unchanged.

use std::collections::HashMap;
use std::sync::Arc;

use minisqlite_types::{Error, Result};
use minisqlite_wal::codec::frame_page_data_offset;
use minisqlite_wal::index::{scan, FrameMeta, WalIndex};

/// An incrementally-maintained index over a WAL's committed frame prefix. See the
/// module docs for why this exists alongside [`WalIndex`].
#[derive(Debug, Clone, Default)]
pub(crate) struct FrameIndex {
    /// Page size in bytes (needed to locate a frame's page data in the WAL bytes).
    page_size: u32,
    /// The WAL header checksum — the running-checksum seed for the first appended
    /// frame (and the append seed whenever the WAL holds no frame yet).
    header_checksum: (u32, u32),
    /// Every committed frame in order; frame number `n` (1-based) is `frames[n - 1]`.
    frames: Vec<FrameMeta>,
    /// `page_no -> ascending committed frame numbers carrying that page`. Ascending
    /// push order keeps each list sorted for the binary search in [`Self::resolve`].
    page_map: HashMap<u32, Vec<u32>>,
}

impl FrameIndex {
    /// Build a fresh `Arc<FrameIndex>` by scanning a WAL byte image. This is the ONE
    /// sanctioned place the O(WAL) `scan` is used to (re)build the index — at open, a
    /// reset, and header-init, all cold or trivially-small paths — so hot paths clone
    /// or extend the result instead. Centralizes the rebuild idiom.
    pub(crate) fn rebuilt(wal_bytes: &[u8], page_size: u32) -> Arc<FrameIndex> {
        Arc::new(FrameIndex::from_wal_index(&scan(wal_bytes), page_size))
    }

    /// Snapshot the committed prefix (`1..=mx_frame`) of a freshly-`scan`ned
    /// [`WalIndex`] into an owned, extendable index. Used at open and after a reset,
    /// where the underlying `scan` is cheap (empty or small WAL).
    pub(crate) fn from_wal_index(wi: &WalIndex, page_size: u32) -> FrameIndex {
        let mx = wi.mx_frame();
        let mut frames = Vec::with_capacity(mx as usize);
        let mut page_map: HashMap<u32, Vec<u32>> = HashMap::new();
        for f in 1..=mx {
            // `f <= mx_frame`, so the frame is present and committed.
            let fm = *wi.frame(f).expect("frame within mx_frame is present");
            frames.push(fm);
            page_map.entry(fm.page_no).or_default().push(f);
        }
        FrameIndex { page_size, header_checksum: wi.header_checksum(), frames, page_map }
    }

    /// Append one committed transaction's frames (in WAL order, the last one its
    /// commit frame). Every frame becomes part of the committed prefix immediately,
    /// so all are indexed and `mx_frame` advances to the new tail. O(frames appended).
    pub(crate) fn extend_commit(&mut self, new_frames: &[FrameMeta]) {
        self.frames.reserve(new_frames.len());
        for fm in new_frames {
            self.frames.push(*fm);
            let frame_no = self.frames.len() as u32;
            self.page_map.entry(fm.page_no).or_default().push(frame_no);
        }
    }

    /// The committed ceiling: the number of the last committed frame (== the number
    /// of frames, since only committed frames are stored).
    #[inline]
    pub(crate) fn mx_frame(&self) -> u32 {
        self.frames.len() as u32
    }

    /// The running checksum to continue from when appending the next frame: the last
    /// frame's cumulative checksum, or the WAL header checksum if the WAL is empty.
    #[inline]
    pub(crate) fn running_checksum_for_append(&self) -> (u32, u32) {
        match self.frames.last() {
            Some(f) => f.checksum,
            None => self.header_checksum,
        }
    }

    /// `FindFrame(P, snapshot_mx)`: the largest frame number carrying page `page_no`
    /// that does not exceed `snapshot_mx`, or `None` (⇒ read the page from the db
    /// file). Binary search over the page's ascending frame list.
    pub(crate) fn resolve(&self, page_no: u32, snapshot_mx: u32) -> Option<u32> {
        let list = self.page_map.get(&page_no)?;
        let count = list.partition_point(|&f| f <= snapshot_mx);
        if count == 0 {
            None
        } else {
            Some(list[count - 1])
        }
    }

    /// The database size in pages a reader at `snapshot_mx` observes: the db-size of
    /// the last commit frame at or before `snapshot_mx`, or 0 if none. (A pinned
    /// snapshot is always a commit boundary, so this returns on the first step.)
    pub(crate) fn db_size_at(&self, snapshot_mx: u32) -> u32 {
        let hi = snapshot_mx.min(self.mx_frame());
        for f in (1..=hi).rev() {
            let m = &self.frames[(f - 1) as usize];
            if m.db_size != 0 {
                return m.db_size;
            }
        }
        0
    }

    /// Borrow the page data of frame `frame_no` (1-based) out of `wal` (a WAL byte
    /// image consistent with this index). Fails closed on an out-of-range frame or a
    /// truncated buffer rather than panicking.
    pub(crate) fn frame_page_data<'a>(&self, wal: &'a [u8], frame_no: u32) -> Result<&'a [u8]> {
        if frame_no == 0 || frame_no > self.mx_frame() {
            return Err(Error::Format(format!(
                "WAL frame {frame_no} out of range (valid 1..={})",
                self.mx_frame()
            )));
        }
        let start = frame_page_data_offset(frame_no, self.page_size);
        let end = start + self.page_size as usize;
        wal.get(start..end).ok_or_else(|| {
            Error::Format(format!("WAL truncated reading page data of frame {frame_no}"))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minisqlite_wal::codec::{frame_page_data_offset, WalBuilder, WalHeader};
    use minisqlite_wal::index::scan;

    const PS: u32 = 512;

    /// Build a WAL byte image from `(page_no, db_size)` frames and return both the
    /// bytes and a `scan`ned reference `WalIndex` to compare our incremental index to.
    fn build(frames: &[(u32, u32)]) -> (Vec<u8>, WalIndex) {
        let mut b = WalBuilder::new(WalHeader::new(PS, 7, 9, 0, false));
        for &(page_no, db_size) in frames {
            b.append_frame(page_no, db_size, &vec![page_no as u8; PS as usize]).unwrap();
        }
        let bytes = b.into_bytes();
        let wi = scan(&bytes);
        (bytes, wi)
    }

    #[test]
    fn from_wal_index_matches_scan_for_resolve_and_size() {
        // One committed transaction: pages 1,2 then a commit frame for page 1.
        let (_bytes, wi) = build(&[(1, 0), (2, 0), (1, 3)]);
        let fi = FrameIndex::from_wal_index(&wi, PS);
        assert_eq!(fi.mx_frame(), wi.mx_frame());
        for p in 1..=2u32 {
            assert_eq!(fi.resolve(p, fi.mx_frame()), wi.resolve(p, wi.mx_frame()), "page {p}");
        }
        assert_eq!(fi.db_size_at(fi.mx_frame()), wi.db_size_at(wi.mx_frame()));
        assert_eq!(fi.running_checksum_for_append(), wi.running_checksum_for_append());
    }

    #[test]
    fn extend_commit_equals_a_full_rescan() {
        // The property that makes the O(pages) commit path correct: extending the
        // index with a second transaction's frame metadata yields the SAME index a
        // full re-scan of the whole WAL would produce.
        //
        // Base index = the first transaction (frames 1..=2).
        let (_base_bytes, wi0) = build(&[(1, 0), (2, 2)]);
        let mut fi = FrameIndex::from_wal_index(&wi0, PS);

        // Reference: build the full 2-transaction WAL in one pass and scan it.
        let mut full = WalBuilder::new(WalHeader::new(PS, 7, 9, 0, false));
        for &(p, s) in &[(1u32, 0u32), (2, 2), (3, 0), (2, 4)] {
            full.append_frame(p, s, &vec![p as u8; PS as usize]).unwrap();
        }
        let full_bytes = full.into_bytes();
        let wi_full = scan(&full_bytes);

        // Feed the two NEW frames' metadata (frames 3 and 4) into the incremental idx.
        let new_frames: Vec<FrameMeta> = (3..=4).map(|f| *wi_full.frame(f).unwrap()).collect();
        fi.extend_commit(&new_frames);

        assert_eq!(fi.mx_frame(), wi_full.mx_frame());
        for p in 1..=3u32 {
            assert_eq!(
                fi.resolve(p, fi.mx_frame()),
                wi_full.resolve(p, wi_full.mx_frame()),
                "resolve(page {p}) must match a full re-scan"
            );
        }
        assert_eq!(fi.db_size_at(fi.mx_frame()), wi_full.db_size_at(wi_full.mx_frame()));
        assert_eq!(fi.running_checksum_for_append(), wi_full.running_checksum_for_append());
        // Page-data borrow lands at the same offset a scan would compute.
        let f = fi.resolve(2, fi.mx_frame()).unwrap();
        assert_eq!(
            fi.frame_page_data(&full_bytes, f).unwrap(),
            &full_bytes[frame_page_data_offset(f, PS)..frame_page_data_offset(f, PS) + PS as usize]
        );
    }

    #[test]
    fn resolve_respects_the_snapshot_ceiling() {
        // page 1 at frames 1 and 3; a snapshot at frame 1 must not see frame 3.
        let (_b, wi) = build(&[(1, 1), (2, 0), (1, 3)]);
        let fi = FrameIndex::from_wal_index(&wi, PS);
        assert_eq!(fi.resolve(1, 1), Some(1), "snapshot at 1 sees the older frame");
        assert_eq!(fi.resolve(1, 3), Some(3), "snapshot at 3 sees the newer frame");
        assert_eq!(fi.resolve(9, 3), None, "absent page falls through to the db file");
    }

    #[test]
    fn frame_page_data_out_of_range_fails_closed() {
        let (bytes, wi) = build(&[(1, 1)]);
        let fi = FrameIndex::from_wal_index(&wi, PS);
        assert!(fi.frame_page_data(&bytes, 0).is_err());
        assert!(fi.frame_page_data(&bytes, 2).is_err());
        assert!(fi.frame_page_data(&bytes, 1).is_ok());
    }
}
