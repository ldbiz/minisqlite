//! The rollback-journal byte format (fileformat2 §3): the header, the per-page
//! record, and the record checksum. A pure bytes<->data layer with no I/O — the
//! writer and recovery paths do the file work and call these to encode/decode.
//!
//! Every multi-byte integer is big-endian. Decoders fail closed with
//! [`Error::Format`] on a bad magic, a truncated buffer, or an out-of-range page or
//! sector size, so a corrupt journal is rejected rather than half-interpreted.

use minisqlite_types::{Error, Result};

use crate::util::{be32, write_be32};

/// The 8-byte magic that opens a valid rollback journal (fileformat2 §3).
pub const MAGIC: [u8; 8] = [0xd9, 0xd5, 0x05, 0xf9, 0x20, 0xa1, 0x63, 0xd7];

/// The fixed, information-bearing part of a journal header: magic (8) + page count
/// (4) + nonce (4) + initial size (4) + sector size (4) + page size (4). The header
/// is then zero-padded out to the sector size so records start on a sector boundary.
pub const HEADER_PREFIX_LEN: usize = 28;

/// Sector size a freshly created journal assumes (SQLite's historical default). The
/// header is padded to this, so page records begin at byte `DEFAULT_SECTOR_SIZE`.
pub const DEFAULT_SECTOR_SIZE: u32 = 512;

/// The page-count sentinel meaning "-1": records run to the end of the file rather
/// than to a fixed count. Stored in the header at offset 8.
pub const PAGE_COUNT_TO_EOF: u32 = 0xffff_ffff;

// Header field offsets within the prefix (fileformat2 §3). `OFF_PAGE_COUNT` is
// `pub(crate)` so the writer can backfill just that field in place on sync.
const OFF_MAGIC: usize = 0;
pub(crate) const OFF_PAGE_COUNT: usize = 8;
const OFF_NONCE: usize = 12;
const OFF_INITIAL_PAGES: usize = 16;
const OFF_SECTOR_SIZE: usize = 20;
const OFF_PAGE_SIZE: usize = 24;

/// A valid SQLite page size: a power of two in `[512, 65536]` (fileformat2 §1.3.2).
/// The journal's page size must be one, since it stores whole database pages.
pub fn is_valid_page_size(size: u32) -> bool {
    (512..=65536).contains(&size) && size.is_power_of_two()
}

/// A plausible sector size: a power of two in `[32, 65536]`. The lower bound keeps
/// the [`HEADER_PREFIX_LEN`]-byte header inside its own sector; the upper bound and
/// power-of-two rule reject the absurd values a corrupt/zeroed header would carry.
pub fn is_valid_sector_size(size: u32) -> bool {
    (32..=65536).contains(&size) && size.is_power_of_two()
}

/// The decoded 28-byte journal header. One header opens each segment of the journal;
/// a multi-segment journal repeats the layout, and every header shares the same page
/// and sector size (but carries its own nonce and page count).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JournalHeader {
    /// Number of page records in this segment, or [`PAGE_COUNT_TO_EOF`] for "as many
    /// as fit before the end of the file".
    pub page_count: u32,
    /// Random per-transaction (per-segment) checksum seed. Varying it each time is
    /// what lets a stale or zeroed sector be detected: its bytes were checksummed
    /// under a different nonce, so they will not validate under this one.
    pub nonce: u32,
    /// Database size in pages at the start of the transaction. Recovery truncates
    /// the database back to this, undoing any growth the aborted transaction caused.
    pub initial_db_pages: u32,
    /// Sector size the writer assumed; the header is padded to this length.
    pub sector_size: u32,
    /// Page size `N` used throughout this journal.
    pub page_size: u32,
}

impl JournalHeader {
    /// Decode a header from the first [`HEADER_PREFIX_LEN`] bytes of a buffer. Fails
    /// closed on a short buffer, a bad magic, or an out-of-range page/sector size —
    /// each marks the bytes as not a trustworthy journal header.
    pub fn decode(buf: &[u8]) -> Result<JournalHeader> {
        if buf.len() < HEADER_PREFIX_LEN {
            return Err(Error::format(format!(
                "journal header truncated: need {HEADER_PREFIX_LEN} bytes, got {}",
                buf.len()
            )));
        }
        if buf[OFF_MAGIC..OFF_MAGIC + 8] != MAGIC {
            return Err(Error::format("journal header has bad magic".to_string()));
        }
        let page_size = be32(buf, OFF_PAGE_SIZE);
        let sector_size = be32(buf, OFF_SECTOR_SIZE);
        if !is_valid_page_size(page_size) {
            return Err(Error::format(format!("journal header invalid page size {page_size}")));
        }
        if !is_valid_sector_size(sector_size) {
            return Err(Error::format(format!(
                "journal header invalid sector size {sector_size}"
            )));
        }
        Ok(JournalHeader {
            page_count: be32(buf, OFF_PAGE_COUNT),
            nonce: be32(buf, OFF_NONCE),
            initial_db_pages: be32(buf, OFF_INITIAL_PAGES),
            sector_size,
            page_size,
        })
    }

    /// Encode this header into a fresh, zero-padded sector-sized buffer ready to
    /// write at the start of a journal (segment). The prefix fields occupy the first
    /// [`HEADER_PREFIX_LEN`] bytes; the rest is zero padding out to `sector_size`.
    pub fn to_padded_header(&self) -> Vec<u8> {
        // `sector_size` is validated to be >= 32 > HEADER_PREFIX_LEN by every
        // constructor, so the prefix always fits inside the padded buffer.
        let mut buf = vec![0u8; self.sector_size as usize];
        buf[OFF_MAGIC..OFF_MAGIC + 8].copy_from_slice(&MAGIC);
        write_be32(&mut buf, OFF_PAGE_COUNT, self.page_count);
        write_be32(&mut buf, OFF_NONCE, self.nonce);
        write_be32(&mut buf, OFF_INITIAL_PAGES, self.initial_db_pages);
        write_be32(&mut buf, OFF_SECTOR_SIZE, self.sector_size);
        write_be32(&mut buf, OFF_PAGE_SIZE, self.page_size);
        buf
    }

    /// Whether the page count is the "-1" sentinel (records run to end of file).
    pub fn records_to_eof(&self) -> bool {
        self.page_count == PAGE_COUNT_TO_EOF
    }
}

/// Whether `buf` begins with the journal magic. Recovery uses this to decide, cheaply
/// and without erroring, whether a file even looks like a journal: a zeroed header
/// (a PERSIST commit) or an absent one simply is not hot.
pub fn has_valid_magic(buf: &[u8]) -> bool {
    buf.len() >= 8 && buf[0..8] == MAGIC
}

/// The rollback-journal page-record checksum (fileformat2 §3): seed with the header
/// nonce, then add a sparse sample of bytes at offsets `N-200, N-400, …` down to the
/// first non-negative index, as an unsigned 32-bit *wrapping* add.
///
/// It is deliberately a sparse sample (not the whole page) for speed, and seeded
/// with a per-transaction nonce so a torn write — or a stale sector left by an
/// earlier journal — is caught: those bytes will not reproduce this nonce's sum.
pub fn journal_checksum(nonce: u32, page: &[u8]) -> u32 {
    let mut cksum = nonce;
    // X starts at N-200 and steps down by 200; it can go negative, so it must be
    // signed. Each sampled byte is a plain u8 zero-extended to u32.
    let mut x: isize = page.len() as isize - 200;
    while x >= 0 {
        cksum = cksum.wrapping_add(page[x as usize] as u32);
        x -= 200;
    }
    cksum
}

/// The on-disk length of one page record: a 4-byte page number, the `page_size`-byte
/// original page content, and a 4-byte checksum.
pub const fn page_record_len(page_size: usize) -> usize {
    4 + page_size + 4
}

/// A page record borrowed from a decode buffer. `content` is a sub-slice of the input
/// (no copy); `checksum_ok` reports whether the stored checksum matched — recovery
/// treats a mismatch as the end of trustworthy data rather than as an error, so the
/// verdict is surfaced here instead of thrown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageRecord<'a> {
    /// 1-based page number in the database file. Zero is invalid.
    pub page_no: u32,
    /// The original page content, exactly `page_size` bytes.
    pub content: &'a [u8],
    /// Whether the stored checksum matched the recomputed one under the segment nonce.
    pub checksum_ok: bool,
}

/// Serialize one page record: `[page_no][content][checksum]`, with the checksum
/// computed over `content` under `nonce`. The record length tracks the content
/// length, which the caller sizes to the journal's page size.
pub fn encode_page_record(page_no: u32, content: &[u8], nonce: u32) -> Vec<u8> {
    let mut buf = vec![0u8; page_record_len(content.len())];
    write_be32(&mut buf, 0, page_no);
    buf[4..4 + content.len()].copy_from_slice(content);
    let cksum = journal_checksum(nonce, content);
    write_be32(&mut buf, 4 + content.len(), cksum);
    buf
}

/// Decode one page record of `page_size` content bytes from `buf`, verifying its
/// checksum against `nonce`. Fails closed only when the buffer is too short to hold a
/// whole record (a truncated read); a *checksum* mismatch is reported via
/// [`PageRecord::checksum_ok`], not as an error.
pub fn decode_page_record<'a>(buf: &'a [u8], page_size: usize, nonce: u32) -> Result<PageRecord<'a>> {
    let need = page_record_len(page_size);
    if buf.len() < need {
        return Err(Error::format(format!(
            "journal page record truncated: need {need} bytes, got {}",
            buf.len()
        )));
    }
    let page_no = be32(buf, 0);
    let content = &buf[4..4 + page_size];
    let stored = be32(buf, 4 + page_size);
    let checksum_ok = journal_checksum(nonce, content) == stored;
    Ok(PageRecord { page_no, content, checksum_ok })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checksum_known_vector() {
        // N = 512 samples at X = 312 and X = 112 (512-200, then 312-200; 112-200 < 0).
        let mut page = vec![0u8; 512];
        page[312] = 5;
        page[112] = 7;
        // Also set a NON-sampled byte to prove it is ignored.
        page[113] = 99;
        let nonce = 1000u32;
        assert_eq!(journal_checksum(nonce, &page), 1000 + 5 + 7);
    }

    #[test]
    fn checksum_only_samples_stride_200_from_end() {
        let n = 512usize;
        let sampled = [312usize, 112usize];
        let nonce = 0u32;
        for i in 0..n {
            let mut page = vec![0u8; n];
            page[i] = 1;
            let expected = if sampled.contains(&i) { 1 } else { 0 };
            assert_eq!(
                journal_checksum(nonce, &page),
                expected,
                "byte {i} should {} affect the checksum",
                if expected == 1 { "" } else { "not" }
            );
        }
    }

    #[test]
    fn checksum_depends_on_nonce() {
        let page = vec![3u8; 4096];
        assert_ne!(journal_checksum(1, &page), journal_checksum(2, &page));
        // With the same page the difference is exactly the nonce difference.
        assert_eq!(
            journal_checksum(7, &page).wrapping_sub(journal_checksum(0, &page)),
            7
        );
    }

    #[test]
    fn checksum_wraps_u32() {
        // A single sampled byte of 0xFF added to a near-max nonce must wrap, not panic.
        let mut page = vec![0u8; 512];
        page[312] = 0xFF;
        page[112] = 0x02;
        let nonce = u32::MAX;
        assert_eq!(journal_checksum(nonce, &page), u32::MAX.wrapping_add(0xFF + 0x02));
    }

    #[test]
    fn checksum_small_page_below_stride_is_just_nonce() {
        // If N < 200 there are no sampled bytes, so the checksum is the nonce alone.
        let page = vec![0xABu8; 199];
        assert_eq!(journal_checksum(42, &page), 42);
    }

    #[test]
    fn header_roundtrip_recovers_all_fields() {
        let hdr = JournalHeader {
            page_count: 7,
            nonce: 0xDEAD_BEEF,
            initial_db_pages: 12,
            sector_size: 512,
            page_size: 4096,
        };
        let buf = hdr.to_padded_header();
        assert_eq!(buf.len(), 512, "header padded to the sector size");
        // Padding after the prefix is zero.
        assert!(buf[HEADER_PREFIX_LEN..].iter().all(|&b| b == 0));
        let back = JournalHeader::decode(&buf).unwrap();
        assert_eq!(back, hdr);
    }

    #[test]
    fn header_field_offsets_pinned() {
        // Distinct, non-palindromic values so a byte-order regression is caught even
        // if it flips BOTH the writer and reader consistently: the raw on-disk bytes
        // are asserted big-endian (MSB first) directly, independent of this crate's
        // be32/write_be32 accessors.
        let hdr = JournalHeader {
            page_count: 0x0102_0304,
            nonce: 0x0506_0708,
            initial_db_pages: 0x090A_0B0C,
            sector_size: 4096,
            page_size: 8192,
        };
        let buf = hdr.to_padded_header();
        assert_eq!(&buf[0..8], &MAGIC);
        assert_eq!(&buf[8..12], &[0x01, 0x02, 0x03, 0x04], "page_count @8 big-endian");
        assert_eq!(&buf[12..16], &[0x05, 0x06, 0x07, 0x08], "nonce @12 big-endian");
        assert_eq!(&buf[16..20], &[0x09, 0x0A, 0x0B, 0x0C], "initial_db_pages @16 big-endian");
        // 4096 = 0x0000_1000, 8192 = 0x0000_2000.
        assert_eq!(&buf[20..24], &[0x00, 0x00, 0x10, 0x00], "sector_size @20 big-endian");
        assert_eq!(&buf[24..28], &[0x00, 0x00, 0x20, 0x00], "page_size @24 big-endian");
        // And the offsets read back through the accessors and full decode.
        assert_eq!(be32(&buf, 8), 0x0102_0304);
        assert_eq!(be32(&buf, 12), 0x0506_0708);
        assert_eq!(be32(&buf, 16), 0x090A_0B0C);
        assert_eq!(JournalHeader::decode(&buf).unwrap(), hdr);
    }

    #[test]
    fn header_bad_magic_rejected() {
        let hdr = JournalHeader {
            page_count: 1,
            nonce: 1,
            initial_db_pages: 1,
            sector_size: 512,
            page_size: 512,
        };
        let mut buf = hdr.to_padded_header();
        buf[0] ^= 0xFF;
        assert!(JournalHeader::decode(&buf).is_err());
        assert!(!has_valid_magic(&buf));
    }

    #[test]
    fn header_truncated_rejected() {
        let buf = [0u8; HEADER_PREFIX_LEN - 1];
        assert!(JournalHeader::decode(&buf).is_err());
    }

    #[test]
    fn header_bad_page_size_rejected() {
        let mut buf = JournalHeader {
            page_count: 1,
            nonce: 1,
            initial_db_pages: 1,
            sector_size: 512,
            page_size: 512,
        }
        .to_padded_header();
        // 1000 is not a power of two.
        write_be32(&mut buf, OFF_PAGE_SIZE, 1000);
        assert!(JournalHeader::decode(&buf).is_err());
    }

    #[test]
    fn header_bad_sector_size_rejected() {
        let mut buf = JournalHeader {
            page_count: 1,
            nonce: 1,
            initial_db_pages: 1,
            sector_size: 512,
            page_size: 512,
        }
        .to_padded_header();
        write_be32(&mut buf, OFF_SECTOR_SIZE, 7); // not a power of two, too small
        assert!(JournalHeader::decode(&buf).is_err());
    }

    #[test]
    fn zeroed_header_has_no_valid_magic() {
        let buf = vec![0u8; 512];
        assert!(!has_valid_magic(&buf));
        assert!(JournalHeader::decode(&buf).is_err());
    }

    #[test]
    fn page_size_validation_bounds() {
        assert!(is_valid_page_size(512));
        assert!(is_valid_page_size(65536));
        assert!(is_valid_page_size(4096));
        assert!(!is_valid_page_size(256));
        assert!(!is_valid_page_size(1000));
        assert!(!is_valid_page_size(0));
    }

    #[test]
    fn page_record_roundtrip() {
        let content: Vec<u8> = (0..512u32).map(|i| (i % 256) as u8).collect();
        let nonce = 0x0BAD_F00D;
        let rec = encode_page_record(42, &content, nonce);
        assert_eq!(rec.len(), page_record_len(512));
        let decoded = decode_page_record(&rec, 512, nonce).unwrap();
        assert_eq!(decoded.page_no, 42);
        assert_eq!(decoded.content, &content[..]);
        assert!(decoded.checksum_ok);
    }

    #[test]
    fn page_record_bad_checksum_flagged_not_errored() {
        let content = vec![9u8; 512];
        let nonce = 1;
        let mut rec = encode_page_record(1, &content, nonce);
        // Corrupt the last checksum byte.
        let last = rec.len() - 1;
        rec[last] ^= 0xFF;
        let decoded = decode_page_record(&rec, 512, nonce).unwrap();
        assert!(!decoded.checksum_ok, "corrupted checksum must be reported as not-ok");
    }

    #[test]
    fn page_record_wrong_nonce_fails_checksum() {
        let content = vec![9u8; 512];
        let rec = encode_page_record(1, &content, 1);
        let decoded = decode_page_record(&rec, 512, 2).unwrap();
        assert!(!decoded.checksum_ok);
    }

    #[test]
    fn page_record_truncated_rejected() {
        let content = vec![0u8; 512];
        let rec = encode_page_record(1, &content, 1);
        assert!(decode_page_record(&rec[..rec.len() - 1], 512, 1).is_err());
    }
}
