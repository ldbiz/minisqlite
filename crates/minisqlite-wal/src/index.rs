//! Scanning a WAL into an in-memory frame index and answering the reader's snapshot
//! query (fileformat2 §4.5 and §4.6).
//!
//! [`scan`] does a single forward pass over the WAL, validating each frame's salts
//! and cumulative checksum, and stops at the first invalid frame or end of file —
//! exactly the recovery pass SQLite runs. The result is a [`WalIndex`] that answers
//! `FindFrame(P, mxFrame)`: the largest frame number ≤ `mxFrame` that carries page
//! `P`, or `None` (⇒ read that page from the database file).
//!
//! Only *committed* content is resolvable. `mxFrame` is the last valid commit frame
//! (§4.5): frames written after it (an interrupted transaction with no commit
//! marker) are counted as valid frames but are not part of any snapshot, so the
//! index resolves against `1..=mxFrame` only. A reader records `mxFrame` at the
//! start of its transaction and passes it to [`WalIndex::resolve`] for every read,
//! giving it a stable snapshot even as later transactions append more frames.

use std::collections::HashMap;

use minisqlite_types::{Error, Result};

use crate::codec::{
    frame_checksum, frame_page_data_offset, frame_stride, FrameHeader, WalHeader, FRAME_HEADER_SIZE,
    WAL_FILE_FORMAT, WAL_HEADER_SIZE,
};

/// The retained per-frame metadata after validation. Page data itself is not
/// copied — read it from the WAL bytes via [`WalIndex::frame_page_data`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameMeta {
    /// Database page number this frame carries.
    pub page_no: u32,
    /// Post-commit database size in pages if this is a commit frame, else 0.
    pub db_size: u32,
    /// The cumulative checksum through this frame (the seed for the next frame).
    pub checksum: (u32, u32),
}

impl FrameMeta {
    /// A commit frame carries a non-zero db-size and marks the end of a transaction.
    #[inline]
    pub fn is_commit(&self) -> bool {
        crate::codec::is_commit_marker(self.db_size)
    }
}

/// An in-memory index over the valid prefix of a WAL, built by [`scan`].
///
/// Invariants (upheld by [`scan`]):
/// - `frames` holds every valid frame in order; frame number `n` (1-based) is
///   `frames[n - 1]`.
/// - `mx_frame` is the number of the last valid *commit* frame, or 0 if none.
/// - `page_map[p]` is the ascending list of frame numbers ≤ `mx_frame` carrying
///   page `p`. Frames after `mx_frame` (uncommitted) are deliberately excluded, so
///   a resolve can never expose uncommitted data.
#[derive(Debug, Clone)]
pub struct WalIndex {
    page_size: u32,
    salt1: u32,
    salt2: u32,
    big_endian: bool,
    header_checksum: (u32, u32),
    frames: Vec<FrameMeta>,
    mx_frame: u32,
    page_map: HashMap<u32, Vec<u32>>,
}

impl WalIndex {
    /// The index for a WAL with no usable header (missing, truncated, wrong magic,
    /// bad header checksum, or an invalid page size). It resolves nothing, so every
    /// read falls through to the database file.
    fn empty() -> WalIndex {
        WalIndex {
            page_size: 0,
            salt1: 0,
            salt2: 0,
            big_endian: false,
            header_checksum: (0, 0),
            frames: Vec::new(),
            mx_frame: 0,
            page_map: HashMap::new(),
        }
    }

    /// The database page size recorded in the WAL header, or 0 if the header was
    /// unusable (see [`WalIndex::has_valid_header`]).
    #[inline]
    pub fn page_size(&self) -> u32 {
        self.page_size
    }

    /// The WAL header salt-1 (every valid frame carries this salt).
    #[inline]
    pub fn salt1(&self) -> u32 {
        self.salt1
    }

    /// The WAL header salt-2.
    #[inline]
    pub fn salt2(&self) -> u32 {
        self.salt2
    }

    /// The checksum byte order in effect (true ⇒ big-endian magic `0x377f0683`).
    #[inline]
    pub fn big_endian(&self) -> bool {
        self.big_endian
    }

    /// The WAL header checksum (the seed for frame 1, and the append seed when the
    /// WAL holds no commit).
    #[inline]
    pub fn header_checksum(&self) -> (u32, u32) {
        self.header_checksum
    }

    /// True if the WAL had a valid header (so `page_size`/salts are meaningful and a
    /// writer could append to it). A WAL with a valid header but zero frames is
    /// valid-but-empty (`mx_frame == 0`).
    #[inline]
    pub fn has_valid_header(&self) -> bool {
        self.page_size != 0
    }

    /// The mxFrame: the number of the last valid commit frame, or 0 if the WAL holds
    /// no committed transaction. This is the snapshot ceiling a fresh reader records.
    #[inline]
    pub fn mx_frame(&self) -> u32 {
        self.mx_frame
    }

    /// True if the WAL exposes no committed content (all reads go to the database).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.mx_frame == 0
    }

    /// The total number of valid frames found, including any valid-but-uncommitted
    /// frames after the last commit. Always `>= mx_frame`.
    #[inline]
    pub fn n_valid_frames(&self) -> u32 {
        self.frames.len() as u32
    }

    /// All validated frames in order (frame `n` is `frames()[n - 1]`).
    #[inline]
    pub fn frames(&self) -> &[FrameMeta] {
        &self.frames
    }

    /// Metadata for frame `frame_no` (1-based), or `None` if out of range.
    #[inline]
    pub fn frame(&self, frame_no: u32) -> Option<&FrameMeta> {
        if frame_no == 0 {
            return None;
        }
        self.frames.get((frame_no - 1) as usize)
    }

    /// The database size in pages implied by the committed snapshot at `mx_frame`
    /// (the db-size of the last commit frame), or 0 if the WAL holds no commit.
    #[inline]
    pub fn db_size_pages(&self) -> u32 {
        match self.frame(self.mx_frame) {
            Some(f) => f.db_size,
            None => 0,
        }
    }

    /// The frame number of the last commit frame at or before `frame_no`, or 0 if
    /// there is no commit in `1..=min(frame_no, mx_frame)`. A checkpoint bounded by
    /// a reader mark can only drain up to a commit boundary (never a partial
    /// transaction), so it clamps its limit through this.
    pub fn last_commit_at_or_before(&self, frame_no: u32) -> u32 {
        let hi = frame_no.min(self.mx_frame);
        for f in (1..=hi).rev() {
            if self.frame(f).is_some_and(FrameMeta::is_commit) {
                return f;
            }
        }
        0
    }

    /// The database size in pages a reader snapshotting at `snapshot_mx` observes:
    /// the db-size recorded by the last commit at or before `snapshot_mx`, or 0 if
    /// none.
    #[inline]
    pub fn db_size_at(&self, snapshot_mx: u32) -> u32 {
        match self.last_commit_at_or_before(snapshot_mx) {
            0 => 0,
            c => self.frame(c).map_or(0, |m| m.db_size),
        }
    }

    /// The running checksum a writer must continue from to append the next frame:
    /// the checksum of the last commit frame, or the header checksum if the WAL
    /// holds no commit (the next write overwrites any uncommitted leftovers and
    /// starts the frame chain again right after `mx_frame`).
    #[inline]
    pub fn running_checksum_for_append(&self) -> (u32, u32) {
        match self.frame(self.mx_frame) {
            Some(f) => f.checksum,
            None => self.header_checksum,
        }
    }

    /// `FindFrame(P, mxFrame)`: the largest frame number carrying page `page_no`
    /// that does not exceed `snapshot_mx`, or `None` if the WAL holds no committed
    /// frame for that page within the snapshot (⇒ read page `P` from the database
    /// file).
    ///
    /// `snapshot_mx` is the reader's recorded mxFrame; entries above `self.mx_frame`
    /// never exist in the index, so passing a larger value simply resolves against
    /// the full committed history.
    pub fn resolve(&self, page_no: u32, snapshot_mx: u32) -> Option<u32> {
        let list = self.page_map.get(&page_no)?;
        // `list` is ascending; count entries ≤ snapshot_mx and take the last.
        let count = list.partition_point(|&f| f <= snapshot_mx);
        if count == 0 {
            None
        } else {
            Some(list[count - 1])
        }
    }

    /// Borrow the page data of frame `frame_no` (1-based) from the WAL bytes. Fails
    /// closed if `frame_no` is out of the valid range or the slice is truncated.
    pub fn frame_page_data<'a>(&self, wal: &'a [u8], frame_no: u32) -> Result<&'a [u8]> {
        if frame_no == 0 || frame_no > self.n_valid_frames() {
            return Err(Error::format(format!(
                "WAL frame {frame_no} out of range (valid 1..={})",
                self.n_valid_frames()
            )));
        }
        let start = frame_page_data_offset(frame_no, self.page_size);
        let end = start + self.page_size as usize;
        wal.get(start..end).ok_or_else(|| {
            Error::format(format!("WAL truncated reading page data of frame {frame_no}"))
        })
    }

    /// Resolve `page_no` at `snapshot_mx` and, if a frame carries it, borrow that
    /// frame's page data. Returns `Ok(None)` when the page is not in the snapshot
    /// (the caller reads it from the database file).
    pub fn resolve_page_data<'a>(
        &self,
        wal: &'a [u8],
        page_no: u32,
        snapshot_mx: u32,
    ) -> Result<Option<&'a [u8]>> {
        match self.resolve(page_no, snapshot_mx) {
            Some(frame_no) => Ok(Some(self.frame_page_data(wal, frame_no)?)),
            None => Ok(None),
        }
    }
}

/// Scan a WAL image into a [`WalIndex`].
///
/// A single forward pass validates each frame's salts and cumulative checksum,
/// stopping at the first invalid frame or the end of the buffer. This never errors:
/// a missing/corrupt header or a bad frame simply yields fewer valid frames (that is
/// normal WAL behavior — leftover bytes from prior checkpoints are expected).
pub fn scan(wal: &[u8]) -> WalIndex {
    let header = match WalHeader::decode(wal) {
        Ok(h) => h,
        Err(_) => return WalIndex::empty(),
    };
    // A header that fails its own checksum, carries an unrecognized file-format
    // version, or names an impossible page size means the WAL cannot be interpreted —
    // treat it as empty (all reads fall through to the db), matching SQLite recovery,
    // which rejects a version != WAL_FILE_FORMAT even when the header checksum passes.
    if !header.verify_checksum()
        || header.file_format != WAL_FILE_FORMAT
        || !header.page_size_is_valid()
    {
        return WalIndex::empty();
    }

    let page_size = header.page_size;
    let stride = frame_stride(page_size);

    let mut frames: Vec<FrameMeta> = Vec::new();
    let mut running = header.checksum;
    let mut off = WAL_HEADER_SIZE;

    // Stop unless a whole frame (24-byte header + page data) remains. `checked_add`
    // guards the (practically unreachable) offset overflow on a colossal buffer.
    while let Some(end) = off.checked_add(stride) {
        if end > wal.len() {
            break;
        }
        let fh = match FrameHeader::decode(&wal[off..off + FRAME_HEADER_SIZE]) {
            Ok(fh) => fh,
            Err(_) => break,
        };
        // Leftover frames from a prior checkpoint carry stale salts.
        if fh.salt1 != header.salt1 || fh.salt2 != header.salt2 {
            break;
        }
        let page_data = &wal[off + FRAME_HEADER_SIZE..end];
        let expected = frame_checksum(running, fh.page_no, fh.db_size, page_data, header.big_endian);
        if expected != fh.checksum {
            break;
        }
        running = expected;
        frames.push(FrameMeta { page_no: fh.page_no, db_size: fh.db_size, checksum: expected });
        off = end;
    }

    // mxFrame is the last valid *commit* frame (non-zero db-size). Frames after it
    // are valid but uncommitted and belong to no snapshot.
    let mut mx_frame = 0u32;
    for (i, f) in frames.iter().enumerate() {
        if f.is_commit() {
            mx_frame = (i as u32) + 1;
        }
    }

    // Index only committed frames (1..=mx_frame). Ascending push order keeps each
    // page's frame list sorted for binary search in resolve().
    let mut page_map: HashMap<u32, Vec<u32>> = HashMap::new();
    for (i, f) in frames.iter().enumerate().take(mx_frame as usize) {
        page_map.entry(f.page_no).or_default().push((i as u32) + 1);
    }

    WalIndex {
        page_size,
        salt1: header.salt1,
        salt2: header.salt2,
        big_endian: header.big_endian,
        header_checksum: header.checksum,
        frames,
        mx_frame,
        page_map,
    }
}
