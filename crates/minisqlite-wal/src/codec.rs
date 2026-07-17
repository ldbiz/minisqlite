//! The WAL byte codec (fileformat2 §4.1): the 32-byte WAL header and the 24-byte
//! frame header, plus frame byte offsets and frame writing. Every field is stored
//! big-endian on disk. Decoding fails closed ([`Error::Format`]) on a truncated or
//! unrecognized header; a *checksum* or *salt* mismatch is not a codec error — it
//! is "this frame is not valid", which the [`crate::index`] scan uses to mark the
//! end of the valid frame prefix.

use minisqlite_types::{Error, Result};

use crate::checksum::wal_checksum;

/// The WAL header is exactly 32 bytes (eight big-endian u32).
pub const WAL_HEADER_SIZE: usize = 32;

/// Each frame begins with a 24-byte frame header (six big-endian u32).
pub const FRAME_HEADER_SIZE: usize = 24;

/// WAL magic with little-endian checksum computation (offset 0).
pub const WAL_MAGIC_LE: u32 = 0x377f_0682;

/// WAL magic with big-endian checksum computation (offset 0).
pub const WAL_MAGIC_BE: u32 = 0x377f_0683;

/// The WAL file-format version stored at offset 4. Fixed at 3007000.
pub const WAL_FILE_FORMAT: u32 = 3_007_000;

#[inline]
fn be32(buf: &[u8], off: usize) -> u32 {
    u32::from_be_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

/// A frame is a commit frame iff its db-size field (frame-header bytes 4..8) is
/// non-zero (§4.1): it ends a transaction and records the post-commit page count.
/// The single definition both [`FrameHeader::is_commit`] and
/// [`crate::index::FrameMeta::is_commit`] delegate to.
#[inline]
pub const fn is_commit_marker(db_size: u32) -> bool {
    db_size != 0
}

/// The stride from the start of one frame to the start of the next: a 24-byte
/// frame header plus `page_size` bytes of page data.
#[inline]
pub const fn frame_stride(page_size: u32) -> usize {
    FRAME_HEADER_SIZE + page_size as usize
}

/// Byte offset of the start of frame `frame_no` (1-based) in the WAL file.
/// `frame_no` must be ≥ 1 — frame 0 does not exist, and passing it underflows
/// (a debug assert fires; release would wrap to a bogus offset).
#[inline]
pub const fn frame_offset(frame_no: u32, page_size: u32) -> usize {
    debug_assert!(frame_no >= 1, "frame_no is 1-based and must be >= 1");
    WAL_HEADER_SIZE + (frame_no as usize - 1) * frame_stride(page_size)
}

/// Byte offset of the page data (after the 24-byte frame header) of frame
/// `frame_no` (1-based). `frame_no` must be ≥ 1.
#[inline]
pub const fn frame_page_data_offset(frame_no: u32, page_size: u32) -> usize {
    frame_offset(frame_no, page_size) + FRAME_HEADER_SIZE
}

/// A decoded 32-byte WAL header.
///
/// `checksum` is the pair stored at bytes 24..32 (the checksum over the first 24
/// bytes). Build a fresh, self-consistent header with [`WalHeader::new`] (which
/// computes the checksum); [`WalHeader::decode`] preserves whatever checksum was on
/// disk so [`WalHeader::verify_checksum`] can validate it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WalHeader {
    /// Checksum byte order: `true` ⇒ magic `0x377f0683` (big-endian words),
    /// `false` ⇒ magic `0x377f0682` (little-endian words).
    pub big_endian: bool,
    /// File-format version (offset 4); [`WAL_FILE_FORMAT`] for a valid WAL.
    pub file_format: u32,
    /// Database page size in bytes (offset 8).
    pub page_size: u32,
    /// Checkpoint sequence number (offset 12), incremented on each WAL reset.
    pub checkpoint_seq: u32,
    /// Salt-1 (offset 16), incremented on each reset; copied into every frame.
    pub salt1: u32,
    /// Salt-2 (offset 20), randomized on each reset; copied into every frame.
    pub salt2: u32,
    /// Stored checksum over the first 24 bytes (offsets 24 and 28).
    pub checksum: (u32, u32),
}

impl WalHeader {
    /// Build a self-consistent header, computing the checksum over the first 24
    /// bytes. `file_format` is set to [`WAL_FILE_FORMAT`].
    pub fn new(page_size: u32, salt1: u32, salt2: u32, checkpoint_seq: u32, big_endian: bool) -> Self {
        let mut h = WalHeader {
            big_endian,
            file_format: WAL_FILE_FORMAT,
            page_size,
            checkpoint_seq,
            salt1,
            salt2,
            checksum: (0, 0),
        };
        h.checksum = h.compute_checksum();
        h
    }

    /// The magic value this header serializes with, derived from `big_endian`.
    #[inline]
    pub fn magic(&self) -> u32 {
        if self.big_endian {
            WAL_MAGIC_BE
        } else {
            WAL_MAGIC_LE
        }
    }

    fn write_first_24(&self, buf: &mut [u8]) {
        buf[0..4].copy_from_slice(&self.magic().to_be_bytes());
        buf[4..8].copy_from_slice(&self.file_format.to_be_bytes());
        buf[8..12].copy_from_slice(&self.page_size.to_be_bytes());
        buf[12..16].copy_from_slice(&self.checkpoint_seq.to_be_bytes());
        buf[16..20].copy_from_slice(&self.salt1.to_be_bytes());
        buf[20..24].copy_from_slice(&self.salt2.to_be_bytes());
    }

    /// The checksum this header *should* carry given its other fields.
    pub fn compute_checksum(&self) -> (u32, u32) {
        let mut first24 = [0u8; 24];
        self.write_first_24(&mut first24);
        wal_checksum((0, 0), &first24, self.big_endian)
    }

    /// True if the stored checksum matches the checksum computed over the first 24
    /// bytes. A header with a bad checksum means the WAL is treated as empty.
    #[inline]
    pub fn verify_checksum(&self) -> bool {
        self.compute_checksum() == self.checksum
    }

    /// True if `page_size` is a power of two in the range 512..=65536, the sizes
    /// SQLite permits. (65536 fits in the WAL header's full u32 page-size field, so
    /// the `1 ⇒ 65536` shorthand of the database header does not apply here.)
    #[inline]
    pub fn page_size_is_valid(&self) -> bool {
        self.page_size >= 512 && self.page_size <= 65536 && self.page_size.is_power_of_two()
    }

    /// Serialize to the 32 on-disk bytes.
    pub fn serialize(&self) -> [u8; WAL_HEADER_SIZE] {
        let mut buf = [0u8; WAL_HEADER_SIZE];
        self.write_first_24(&mut buf);
        buf[24..28].copy_from_slice(&self.checksum.0.to_be_bytes());
        buf[28..32].copy_from_slice(&self.checksum.1.to_be_bytes());
        buf
    }

    /// Decode the 32-byte header. Fails closed if the slice is shorter than 32 bytes
    /// or the magic is neither recognized value. The stored checksum is preserved
    /// (verify separately via [`WalHeader::verify_checksum`]).
    pub fn decode(bytes: &[u8]) -> Result<WalHeader> {
        if bytes.len() < WAL_HEADER_SIZE {
            return Err(Error::format(format!(
                "WAL header truncated: {} bytes, need {WAL_HEADER_SIZE}",
                bytes.len()
            )));
        }
        let magic = be32(bytes, 0);
        let big_endian = match magic {
            WAL_MAGIC_LE => false,
            WAL_MAGIC_BE => true,
            _ => {
                return Err(Error::format(format!("unrecognized WAL magic {magic:#010x}")));
            }
        };
        Ok(WalHeader {
            big_endian,
            file_format: be32(bytes, 4),
            page_size: be32(bytes, 8),
            checkpoint_seq: be32(bytes, 12),
            salt1: be32(bytes, 16),
            salt2: be32(bytes, 20),
            checksum: (be32(bytes, 24), be32(bytes, 28)),
        })
    }
}

/// A decoded 24-byte WAL frame header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeader {
    /// Page number this frame carries (offset 0).
    pub page_no: u32,
    /// Post-commit database size in pages for a commit frame; 0 otherwise
    /// (offset 4).
    pub db_size: u32,
    /// Salt-1 copied from the WAL header (offset 8).
    pub salt1: u32,
    /// Salt-2 copied from the WAL header (offset 12).
    pub salt2: u32,
    /// The cumulative checksum through this frame (offsets 16 and 20).
    pub checksum: (u32, u32),
}

impl FrameHeader {
    /// A commit frame is one whose db-size field is non-zero; it marks the end of a
    /// transaction.
    #[inline]
    pub fn is_commit(&self) -> bool {
        is_commit_marker(self.db_size)
    }

    /// Serialize to the 24 on-disk bytes.
    pub fn serialize(&self) -> [u8; FRAME_HEADER_SIZE] {
        let mut buf = [0u8; FRAME_HEADER_SIZE];
        buf[0..4].copy_from_slice(&self.page_no.to_be_bytes());
        buf[4..8].copy_from_slice(&self.db_size.to_be_bytes());
        buf[8..12].copy_from_slice(&self.salt1.to_be_bytes());
        buf[12..16].copy_from_slice(&self.salt2.to_be_bytes());
        buf[16..20].copy_from_slice(&self.checksum.0.to_be_bytes());
        buf[20..24].copy_from_slice(&self.checksum.1.to_be_bytes());
        buf
    }

    /// Decode the 24-byte frame header. Fails closed if the slice is short.
    pub fn decode(bytes: &[u8]) -> Result<FrameHeader> {
        if bytes.len() < FRAME_HEADER_SIZE {
            return Err(Error::format(format!(
                "WAL frame header truncated: {} bytes, need {FRAME_HEADER_SIZE}",
                bytes.len()
            )));
        }
        Ok(FrameHeader {
            page_no: be32(bytes, 0),
            db_size: be32(bytes, 4),
            salt1: be32(bytes, 8),
            salt2: be32(bytes, 12),
            checksum: (be32(bytes, 16), be32(bytes, 20)),
        })
    }
}

/// The cumulative checksum through a frame, continuing from `prev` (the previous
/// frame's checksum, or the WAL header checksum for the first frame). It runs over
/// the frame's first 8 header bytes (page number then db-size, big-endian) followed
/// by the full page data — exactly the bytes the validity rule (§4.1) specifies.
pub fn frame_checksum(
    prev: (u32, u32),
    page_no: u32,
    db_size: u32,
    page_data: &[u8],
    big_endian: bool,
) -> (u32, u32) {
    let mut first8 = [0u8; 8];
    first8[0..4].copy_from_slice(&page_no.to_be_bytes());
    first8[4..8].copy_from_slice(&db_size.to_be_bytes());
    // Splitting the checksum at the 8-byte boundary is exact: each call consumes
    // whole 32-bit word pairs (first8 is one pair; page_size is a multiple of 8)
    // and threads (s0, s1), so it equals one call over the concatenation.
    let after_hdr = wal_checksum(prev, &first8, big_endian);
    wal_checksum(after_hdr, page_data, big_endian)
}

/// Serialize one full frame (24-byte header + page data) onto `out`, given the
/// running cumulative checksum from the prior frame (or the WAL header checksum for
/// the first frame) and the WAL header `salts` (`(salt1, salt2)`, stamped into every
/// frame). Returns the new running checksum to thread into the next frame.
/// `page_data.len()` must equal the WAL header page size (a multiple of 8); this
/// low-level helper does not check it — use [`WalBuilder::append_frame`] for a
/// checked, stateful append.
pub fn write_frame(
    out: &mut Vec<u8>,
    running: (u32, u32),
    page_no: u32,
    db_size: u32,
    salts: (u32, u32),
    page_data: &[u8],
    big_endian: bool,
) -> (u32, u32) {
    let checksum = frame_checksum(running, page_no, db_size, page_data, big_endian);
    let fh = FrameHeader { page_no, db_size, salt1: salts.0, salt2: salts.1, checksum };
    out.extend_from_slice(&fh.serialize());
    out.extend_from_slice(page_data);
    checksum
}

/// A stateful WAL writer that accumulates the header and frames into a byte buffer,
/// threading the cumulative checksum and stamping the header's salts into each
/// frame. Convenient for tests and for the pager integration to build WAL content
/// before writing it to a file. It appends only (a WAL always grows toward the end
/// within a transaction); a reset starts a new builder from [`crate::reset_header`].
#[derive(Debug, Clone)]
pub struct WalBuilder {
    header: WalHeader,
    bytes: Vec<u8>,
    running: (u32, u32),
    frame_count: u32,
}

impl WalBuilder {
    /// Start a builder from `header`, emitting the 32 header bytes and seeding the
    /// running checksum from the header checksum.
    pub fn new(header: WalHeader) -> Self {
        let mut bytes = Vec::with_capacity(WAL_HEADER_SIZE);
        bytes.extend_from_slice(&header.serialize());
        let running = header.checksum;
        WalBuilder { header, bytes, running, frame_count: 0 }
    }

    /// Append one frame. `db_size` is 0 for a non-commit frame, or the post-commit
    /// database size in pages for a commit frame. `page_data.len()` must equal the
    /// header page size. Returns the 1-based frame number just written.
    pub fn append_frame(&mut self, page_no: u32, db_size: u32, page_data: &[u8]) -> Result<u32> {
        if page_data.len() != self.header.page_size as usize {
            return Err(Error::format(format!(
                "WAL frame page data is {} bytes, expected page size {}",
                page_data.len(),
                self.header.page_size
            )));
        }
        self.running = write_frame(
            &mut self.bytes,
            self.running,
            page_no,
            db_size,
            (self.header.salt1, self.header.salt2),
            page_data,
            self.header.big_endian,
        );
        self.frame_count += 1;
        Ok(self.frame_count)
    }

    /// The header this builder writes.
    #[inline]
    pub fn header(&self) -> &WalHeader {
        &self.header
    }

    /// The running cumulative checksum after the frames written so far (the header
    /// checksum if none). This is the seed for the next appended frame.
    #[inline]
    pub fn running_checksum(&self) -> (u32, u32) {
        self.running
    }

    /// Number of frames appended so far.
    #[inline]
    pub fn frame_count(&self) -> u32 {
        self.frame_count
    }

    /// The accumulated WAL bytes (header + all frames).
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consume the builder and return the accumulated WAL bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trip_and_checksum_valid() {
        let h = WalHeader::new(4096, 0xdead_beef, 0x0102_0304, 7, false);
        let bytes = h.serialize();
        assert_eq!(bytes.len(), WAL_HEADER_SIZE);
        let parsed = WalHeader::decode(&bytes).unwrap();
        assert_eq!(parsed, h);
        assert!(parsed.verify_checksum());
        assert_eq!(parsed.file_format, WAL_FILE_FORMAT);
        assert_eq!(parsed.magic(), WAL_MAGIC_LE);
    }

    // Pin the exact on-disk byte offset of every header field with distinct values,
    // independent of decode(). Round-trip alone can't catch a symmetric field swap
    // inside the checksummed region (e.g. salt1<->salt2): decode would mirror
    // serialize and the checksum would still validate, yet real sqlite3 would read
    // the wrong bytes.
    #[test]
    fn header_field_byte_offsets_are_exact() {
        let h = WalHeader::new(4096, 0x1111_1111, 0x2222_2222, 0x3333_3333, false);
        let b = h.serialize();
        assert_eq!(be32(&b, 0), WAL_MAGIC_LE, "magic@0");
        assert_eq!(be32(&b, 4), WAL_FILE_FORMAT, "file_format@4");
        assert_eq!(be32(&b, 8), 4096, "page_size@8");
        assert_eq!(be32(&b, 12), 0x3333_3333, "checkpoint_seq@12");
        assert_eq!(be32(&b, 16), 0x1111_1111, "salt1@16");
        assert_eq!(be32(&b, 20), 0x2222_2222, "salt2@20");
        assert_eq!((be32(&b, 24), be32(&b, 28)), h.checksum, "checksum@24");
    }

    // A hand-constructed golden frame header pins the six field offsets exactly.
    #[test]
    fn frame_header_golden_bytes() {
        let fh = FrameHeader {
            page_no: 0x0102_0304,
            db_size: 0x0506_0708,
            salt1: 0x090a_0b0c,
            salt2: 0x0d0e_0f10,
            checksum: (0x1112_1314, 0x1516_1718),
        };
        let expected: [u8; FRAME_HEADER_SIZE] = [
            0x01, 0x02, 0x03, 0x04, // page_no
            0x05, 0x06, 0x07, 0x08, // db_size
            0x09, 0x0a, 0x0b, 0x0c, // salt1
            0x0d, 0x0e, 0x0f, 0x10, // salt2
            0x11, 0x12, 0x13, 0x14, // checksum-1
            0x15, 0x16, 0x17, 0x18, // checksum-2
        ];
        assert_eq!(fh.serialize(), expected);
    }

    #[test]
    fn header_big_endian_magic_round_trips() {
        let h = WalHeader::new(1024, 1, 2, 0, true);
        let bytes = h.serialize();
        assert_eq!(be32(&bytes, 0), WAL_MAGIC_BE);
        let parsed = WalHeader::decode(&bytes).unwrap();
        assert!(parsed.big_endian);
        assert!(parsed.verify_checksum());
    }

    #[test]
    fn header_checksum_detects_tampering() {
        let h = WalHeader::new(4096, 5, 6, 0, false);
        let mut bytes = h.serialize();
        bytes[16] ^= 0x01; // flip a salt-1 bit without fixing the checksum
        let parsed = WalHeader::decode(&bytes).unwrap();
        assert!(!parsed.verify_checksum());
    }

    #[test]
    fn header_decode_rejects_bad_magic_and_short() {
        let mut bytes = WalHeader::new(4096, 0, 0, 0, false).serialize();
        bytes[0] = 0xff;
        assert!(WalHeader::decode(&bytes).is_err());
        assert!(WalHeader::decode(&[0u8; 8]).is_err());
    }

    #[test]
    fn page_size_validity() {
        assert!(WalHeader::new(512, 0, 0, 0, false).page_size_is_valid());
        assert!(WalHeader::new(65536, 0, 0, 0, false).page_size_is_valid());
        assert!(!WalHeader::new(1000, 0, 0, 0, false).page_size_is_valid());
        assert!(!WalHeader::new(256, 0, 0, 0, false).page_size_is_valid());
    }

    #[test]
    fn frame_header_round_trip() {
        let fh = FrameHeader { page_no: 42, db_size: 100, salt1: 7, salt2: 8, checksum: (11, 22) };
        let bytes = fh.serialize();
        assert_eq!(bytes.len(), FRAME_HEADER_SIZE);
        assert_eq!(FrameHeader::decode(&bytes).unwrap(), fh);
        assert!(fh.is_commit());
        let non_commit = FrameHeader { db_size: 0, ..fh };
        assert!(!non_commit.is_commit());
    }

    #[test]
    fn frame_checksum_split_equals_concatenation() {
        let page = [0xabu8; 512];
        let split = frame_checksum((3, 5), 9, 100, &page, false);
        // Reference: one checksum call over first8 ++ page.
        let mut all = Vec::new();
        all.extend_from_slice(&9u32.to_be_bytes());
        all.extend_from_slice(&100u32.to_be_bytes());
        all.extend_from_slice(&page);
        assert_eq!(split, wal_checksum((3, 5), &all, false));
    }

    #[test]
    fn offsets_are_contiguous() {
        let ps = 512u32;
        assert_eq!(frame_offset(1, ps), WAL_HEADER_SIZE);
        assert_eq!(frame_page_data_offset(1, ps), WAL_HEADER_SIZE + FRAME_HEADER_SIZE);
        assert_eq!(frame_offset(2, ps), WAL_HEADER_SIZE + frame_stride(ps));
        assert_eq!(frame_stride(ps), FRAME_HEADER_SIZE + 512);
    }

    #[test]
    fn builder_emits_header_then_frames() {
        let h = WalHeader::new(512, 1, 2, 0, false);
        let mut b = WalBuilder::new(h);
        assert_eq!(b.running_checksum(), h.checksum);
        let n1 = b.append_frame(1, 0, &[0u8; 512]).unwrap();
        let n2 = b.append_frame(2, 2, &[1u8; 512]).unwrap();
        assert_eq!((n1, n2), (1, 2));
        assert_eq!(b.frame_count(), 2);
        let bytes = b.as_bytes();
        assert_eq!(bytes.len(), WAL_HEADER_SIZE + 2 * frame_stride(512));
        // First frame header's checksum equals the running checksum after frame 1.
        let fh1 = FrameHeader::decode(&bytes[frame_offset(1, 512)..]).unwrap();
        let expected1 = frame_checksum(h.checksum, 1, 0, &[0u8; 512], false);
        assert_eq!(fh1.checksum, expected1);
    }

    #[test]
    fn builder_rejects_wrong_page_size() {
        let mut b = WalBuilder::new(WalHeader::new(512, 1, 2, 0, false));
        assert!(b.append_frame(1, 0, &[0u8; 256]).is_err());
    }
}
