//! The 100-byte database header that opens page 1 (fileformat2 §1.3). Every
//! multibyte field is big-endian. The struct stores decoded, ready-to-use values
//! (e.g. `page_size` is the true size in bytes, not the on-disk `1 => 65536`
//! shorthand) so callers never repeat the field-by-field decoding.

use crate::bytes::{be16, be32, write_be16, write_be32};
use minisqlite_types::{Error, Result};

/// The database header is exactly 100 bytes.
pub const HEADER_SIZE: usize = 100;

/// The 16-byte magic string every SQLite database begins with, including the
/// trailing NUL: `"SQLite format 3\0"`.
pub const MAGIC: &[u8; 16] = b"SQLite format 3\0";

/// Fixed payload-fraction bytes required by the format (§1.3.5): maximum embedded
/// 64, minimum embedded 32, leaf 32. They were once tunable and are now constant.
pub const MAX_EMBEDDED_PAYLOAD_FRACTION: u8 = 64;
pub const MIN_EMBEDDED_PAYLOAD_FRACTION: u8 = 32;
pub const LEAF_PAYLOAD_FRACTION: u8 = 32;

/// Text encoding codes stored at offset 56.
pub const TEXT_ENCODING_UTF8: u32 = 1;
pub const TEXT_ENCODING_UTF16LE: u32 = 2;
pub const TEXT_ENCODING_UTF16BE: u32 = 3;

/// Default page size for a freshly created database.
pub const DEFAULT_PAGE_SIZE: u32 = 4096;

/// A decoded copy of the 100-byte database header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatabaseHeader {
    /// True page size in bytes (512..=65536, a power of two). On disk the value
    /// 65536 is stored as the u16 `1`.
    pub page_size: u32,
    /// File format write version: 1 = rollback journal, 2 = WAL.
    pub write_version: u8,
    /// File format read version: 1 = rollback journal, 2 = WAL.
    pub read_version: u8,
    /// Bytes of reserved space at the end of every page (usually 0).
    pub reserved_space: u8,
    pub max_embedded_payload_fraction: u8,
    pub min_embedded_payload_fraction: u8,
    pub leaf_payload_fraction: u8,
    pub file_change_counter: u32,
    /// In-header database size in pages (valid only when it equals
    /// `version_valid_for`; otherwise the true size is the file length / page).
    pub database_size_pages: u32,
    pub first_freelist_trunk: u32,
    pub freelist_count: u32,
    pub schema_cookie: u32,
    /// Schema format number (1..=4); new databases use 4.
    pub schema_format: u32,
    pub default_cache_size: u32,
    /// Largest root b-tree page in auto/incremental-vacuum mode, else 0.
    pub largest_root_btree: u32,
    /// Text encoding: 1 = UTF-8, 2 = UTF-16le, 3 = UTF-16be.
    pub text_encoding: u32,
    pub user_version: u32,
    /// Non-zero for incremental-vacuum mode; zero otherwise.
    pub incremental_vacuum: u32,
    pub application_id: u32,
    pub version_valid_for: u32,
    pub sqlite_version_number: u32,
}

impl Default for DatabaseHeader {
    /// A sane header for a newly created, single-page database: 4096-byte pages,
    /// UTF-8, legacy (rollback-journal) versions, schema format 4, and the fixed
    /// payload fractions. `file_change_counter == version_valid_for == 0`, so the
    /// in-header database size is valid.
    fn default() -> Self {
        DatabaseHeader {
            page_size: DEFAULT_PAGE_SIZE,
            write_version: 1,
            read_version: 1,
            reserved_space: 0,
            max_embedded_payload_fraction: MAX_EMBEDDED_PAYLOAD_FRACTION,
            min_embedded_payload_fraction: MIN_EMBEDDED_PAYLOAD_FRACTION,
            leaf_payload_fraction: LEAF_PAYLOAD_FRACTION,
            file_change_counter: 0,
            database_size_pages: 1,
            first_freelist_trunk: 0,
            freelist_count: 0,
            schema_cookie: 0,
            schema_format: 4,
            default_cache_size: 0,
            largest_root_btree: 0,
            text_encoding: TEXT_ENCODING_UTF8,
            user_version: 0,
            incremental_vacuum: 0,
            application_id: 0,
            version_valid_for: 0,
            sqlite_version_number: 0,
        }
    }
}

impl DatabaseHeader {
    /// Decode a header from the first 100 bytes of a database file. Fails closed
    /// on a bad magic string or an out-of-spec page size, which mark the input as
    /// not a valid SQLite database rather than a header to trust.
    pub fn read(buf: &[u8; HEADER_SIZE]) -> Result<DatabaseHeader> {
        if &buf[0..16] != MAGIC {
            return Err(Error::Format("not a SQLite database (bad magic header)".into()));
        }
        let page_size = decode_page_size(be16(buf, 16))?;
        let reserved_space = buf[20];
        // The usable size (page size minus reserved bytes) is never allowed below
        // 480 (§1.3.4). A value below it is a corrupt / non-SQLite header, and it
        // would break the b-tree and overflow arithmetic that assume U >= 480
        // (e.g. the `payload_split` floor). Fail closed here rather than panicking
        // downstream. This rejects nothing a real SQLite file could contain.
        if (page_size as usize).saturating_sub(reserved_space as usize) < 480 {
            return Err(Error::Format(format!(
                "usable page size below 480-byte minimum (page {page_size}, reserved {reserved_space})"
            )));
        }
        Ok(DatabaseHeader {
            page_size,
            write_version: buf[18],
            read_version: buf[19],
            reserved_space,
            max_embedded_payload_fraction: buf[21],
            min_embedded_payload_fraction: buf[22],
            leaf_payload_fraction: buf[23],
            file_change_counter: be32(buf, 24),
            database_size_pages: be32(buf, 28),
            first_freelist_trunk: be32(buf, 32),
            freelist_count: be32(buf, 36),
            schema_cookie: be32(buf, 40),
            schema_format: be32(buf, 44),
            default_cache_size: be32(buf, 48),
            largest_root_btree: be32(buf, 52),
            text_encoding: be32(buf, 56),
            user_version: be32(buf, 60),
            incremental_vacuum: be32(buf, 64),
            application_id: be32(buf, 68),
            version_valid_for: be32(buf, 92),
            sqlite_version_number: be32(buf, 96),
        })
    }

    /// Encode this header into a 100-byte buffer. Bytes 72..92 are the
    /// reserved-for-expansion region and are written as zero.
    pub fn write(&self, buf: &mut [u8; HEADER_SIZE]) {
        buf[0..16].copy_from_slice(MAGIC);
        write_be16(buf, 16, encode_page_size(self.page_size));
        buf[18] = self.write_version;
        buf[19] = self.read_version;
        buf[20] = self.reserved_space;
        buf[21] = self.max_embedded_payload_fraction;
        buf[22] = self.min_embedded_payload_fraction;
        buf[23] = self.leaf_payload_fraction;
        write_be32(buf, 24, self.file_change_counter);
        write_be32(buf, 28, self.database_size_pages);
        write_be32(buf, 32, self.first_freelist_trunk);
        write_be32(buf, 36, self.freelist_count);
        write_be32(buf, 40, self.schema_cookie);
        write_be32(buf, 44, self.schema_format);
        write_be32(buf, 48, self.default_cache_size);
        write_be32(buf, 52, self.largest_root_btree);
        write_be32(buf, 56, self.text_encoding);
        write_be32(buf, 60, self.user_version);
        write_be32(buf, 64, self.incremental_vacuum);
        write_be32(buf, 68, self.application_id);
        for b in &mut buf[72..92] {
            *b = 0;
        }
        write_be32(buf, 92, self.version_valid_for);
        write_be32(buf, 96, self.sqlite_version_number);
    }

    /// Encode this header into a fresh 100-byte array.
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        self.write(&mut buf);
        buf
    }

    /// Usable bytes per page: page size minus the reserved region (§1.3.4). This
    /// is the `U` used throughout the b-tree and overflow computations. Uses a
    /// saturating subtraction so a directly-constructed header with an
    /// out-of-range `reserved_space` yields 0 rather than underflow-panicking
    /// (`read()` already rejects `reserved_space` that drops usable below 480).
    pub fn usable_size(&self) -> usize {
        (self.page_size as usize).saturating_sub(self.reserved_space as usize)
    }

    /// Whether the in-header database size (offset 28) may be trusted: it is valid
    /// only when the change counter matches the version-valid-for number (§1.3.7).
    pub fn in_header_size_valid(&self) -> bool {
        self.database_size_pages != 0 && self.file_change_counter == self.version_valid_for
    }

    /// Whether this header selects WAL mode: both the file-format write and read
    /// versions are 2 (fileformat2 §1.3). The single home for the "version 2 = WAL"
    /// rule — the pager's backing selection at open and the engine's `journal_mode` /
    /// `wal_checkpoint` reporting all read it here, so the invariant cannot drift
    /// (a mistyped `== 1` or a future version value changes exactly one place).
    pub fn is_wal(&self) -> bool {
        self.write_version == 2 && self.read_version == 2
    }
}

/// Decode the on-disk page-size field: the u16 `1` means 65536; otherwise the
/// value must be a power of two in 512..=32768.
fn decode_page_size(field: u16) -> Result<u32> {
    let size: u32 = if field == 1 { 65_536 } else { field as u32 };
    if size < 512 || size > 65_536 || !size.is_power_of_two() {
        return Err(Error::Format(format!("invalid page size {size}")));
    }
    Ok(size)
}

/// Encode a page size back into the on-disk u16 field (65536 becomes `1`).
fn encode_page_size(size: u32) -> u16 {
    if size == 65_536 { 1 } else { size as u16 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_header_roundtrips_and_starts_with_magic() {
        let hdr = DatabaseHeader::default();
        let bytes = hdr.to_bytes();
        assert_eq!(&bytes[0..16], MAGIC.as_slice());
        assert_eq!(&bytes[0..16], b"SQLite format 3\0");
        assert_eq!(bytes[21], 64);
        assert_eq!(bytes[22], 32);
        assert_eq!(bytes[23], 32);
        let back = DatabaseHeader::read(&bytes).unwrap();
        assert_eq!(back, hdr);
        assert_eq!(back.page_size, 4096);
        assert_eq!(back.text_encoding, TEXT_ENCODING_UTF8);
        assert!(back.in_header_size_valid());
        assert_eq!(back.usable_size(), 4096);
    }

    #[test]
    fn page_size_65536_uses_sentinel_one() {
        let mut hdr = DatabaseHeader::default();
        hdr.page_size = 65_536;
        let bytes = hdr.to_bytes();
        // On disk the field is the big-endian u16 value 1 (bytes 0x00 0x01).
        assert_eq!(&bytes[16..18], &[0x00, 0x01]);
        let back = DatabaseHeader::read(&bytes).unwrap();
        assert_eq!(back.page_size, 65_536);
    }

    #[test]
    fn all_valid_page_sizes_roundtrip() {
        let mut size = 512u32;
        while size <= 65_536 {
            let mut hdr = DatabaseHeader::default();
            hdr.page_size = size;
            let bytes = hdr.to_bytes();
            assert_eq!(DatabaseHeader::read(&bytes).unwrap().page_size, size);
            size *= 2;
        }
    }

    #[test]
    fn reserved_space_shrinks_usable_size() {
        let mut hdr = DatabaseHeader::default();
        hdr.reserved_space = 32;
        assert_eq!(hdr.usable_size(), 4096 - 32);
    }

    #[test]
    fn usable_size_saturates_on_bogus_reserved_space() {
        // A directly-constructed header (bypassing read()'s validation) whose
        // reserved_space exceeds page_size must yield 0, not underflow-panic.
        // reserved_space is a u8 (<=255), so this requires an out-of-range
        // page_size that read() would reject but a struct literal still allows;
        // without saturating_sub the `page_size - reserved_space` here panics.
        let hdr = DatabaseHeader { page_size: 100, reserved_space: 200, ..Default::default() };
        assert_eq!(hdr.usable_size(), 0);
    }

    #[test]
    fn every_field_survives_roundtrip() {
        let hdr = DatabaseHeader {
            page_size: 8192,
            write_version: 2,
            read_version: 2,
            reserved_space: 16,
            max_embedded_payload_fraction: 64,
            min_embedded_payload_fraction: 32,
            leaf_payload_fraction: 32,
            file_change_counter: 0x1122_3344,
            database_size_pages: 0x00AB_CDEF,
            first_freelist_trunk: 7,
            freelist_count: 3,
            schema_cookie: 42,
            schema_format: 4,
            default_cache_size: 0xFFFF_0000,
            largest_root_btree: 9,
            text_encoding: TEXT_ENCODING_UTF16LE,
            user_version: 0x7EED_BEEF,
            incremental_vacuum: 1,
            application_id: 0xCAFE_F00D,
            version_valid_for: 0x1122_3344,
            sqlite_version_number: 3_045_000,
        };
        let bytes = hdr.to_bytes();
        assert_eq!(DatabaseHeader::read(&bytes).unwrap(), hdr);
        // The expansion region is zeroed.
        assert!(bytes[72..92].iter().all(|&b| b == 0));
    }

    #[test]
    fn field_offsets_pinned_at_documented_positions() {
        // The round-trip tests only prove read == write⁻¹, which holds for ANY
        // consistent offset choice. This pins each field at its LITERAL documented
        // offset (§1.3) with an independent read of the written bytes, so a
        // transposition of two same-width fields (even applied to both read AND
        // write) fails here. Every field carries a distinct value, so a read-side
        // offset swap is also caught by the round-trip at the end.
        let hdr = DatabaseHeader {
            page_size: 8192,
            write_version: 0x11,
            read_version: 0x22,
            reserved_space: 0x33,
            max_embedded_payload_fraction: 64,
            min_embedded_payload_fraction: 32,
            leaf_payload_fraction: 32,
            file_change_counter: 0xF000_0018, // 0x18 = 24
            database_size_pages: 0xF000_001C, // 0x1C = 28
            first_freelist_trunk: 0xF000_0020, // 0x20 = 32
            freelist_count: 0xF000_0024,      // 0x24 = 36
            schema_cookie: 0xF000_0028,       // 0x28 = 40
            schema_format: 4,                 // valid + distinct from the 0xF0.. values
            default_cache_size: 0xF000_0030,  // 0x30 = 48
            largest_root_btree: 0xF000_0034,  // 0x34 = 52
            text_encoding: 2,                 // valid + distinct
            user_version: 0xF000_003C,        // 0x3C = 60
            incremental_vacuum: 0xF000_0040,  // 0x40 = 64
            application_id: 0xF000_0044,      // 0x44 = 68
            version_valid_for: 0xF000_005C,   // 0x5C = 92
            sqlite_version_number: 0xF000_0060, // 0x60 = 96
        };
        let bytes = hdr.to_bytes();

        // Single-byte fields.
        assert_eq!(bytes[18], 0x11, "write_version @18");
        assert_eq!(bytes[19], 0x22, "read_version @19");
        assert_eq!(bytes[20], 0x33, "reserved_space @20");
        // Page size u16 @16 (big-endian, non-sentinel value).
        assert_eq!(be16(&bytes, 16), 8192, "page_size @16");
        // Every u32 at its exact big-endian offset.
        assert_eq!(be32(&bytes, 24), hdr.file_change_counter, "file_change_counter @24");
        assert_eq!(be32(&bytes, 28), hdr.database_size_pages, "database_size_pages @28");
        assert_eq!(be32(&bytes, 32), hdr.first_freelist_trunk, "first_freelist_trunk @32");
        assert_eq!(be32(&bytes, 36), hdr.freelist_count, "freelist_count @36");
        assert_eq!(be32(&bytes, 40), hdr.schema_cookie, "schema_cookie @40");
        assert_eq!(be32(&bytes, 44), hdr.schema_format, "schema_format @44");
        assert_eq!(be32(&bytes, 48), hdr.default_cache_size, "default_cache_size @48");
        assert_eq!(be32(&bytes, 52), hdr.largest_root_btree, "largest_root_btree @52");
        assert_eq!(be32(&bytes, 56), hdr.text_encoding, "text_encoding @56");
        assert_eq!(be32(&bytes, 60), hdr.user_version, "user_version @60");
        assert_eq!(be32(&bytes, 64), hdr.incremental_vacuum, "incremental_vacuum @64");
        assert_eq!(be32(&bytes, 68), hdr.application_id, "application_id @68");
        assert_eq!(be32(&bytes, 92), hdr.version_valid_for, "version_valid_for @92");
        assert_eq!(be32(&bytes, 96), hdr.sqlite_version_number, "sqlite_version_number @96");
        // With all-distinct values, the read side must use the same offsets to
        // recover every field — this catches a read-only offset transposition.
        assert_eq!(DatabaseHeader::read(&bytes).unwrap(), hdr);
    }

    #[test]
    fn usable_size_below_480_is_rejected() {
        // §1.3.4: the usable size (page - reserved) may not drop below 480. For a
        // 512-byte page that means reserved <= 32.
        let mut hdr = DatabaseHeader::default();
        hdr.page_size = 512;
        hdr.reserved_space = 32; // usable = 480 — the exact minimum, accepted.
        assert!(DatabaseHeader::read(&hdr.to_bytes()).is_ok());
        hdr.reserved_space = 33; // usable = 479 — rejected.
        assert!(DatabaseHeader::read(&hdr.to_bytes()).is_err());
    }

    #[test]
    fn bad_magic_is_rejected() {
        let mut bytes = DatabaseHeader::default().to_bytes();
        bytes[3] = b'X';
        assert!(DatabaseHeader::read(&bytes).is_err());
    }

    #[test]
    fn invalid_page_size_is_rejected() {
        let mut bytes = DatabaseHeader::default().to_bytes();
        // 1000 is not a power of two.
        write_be16(&mut bytes, 16, 1000);
        assert!(DatabaseHeader::read(&bytes).is_err());
        // 256 is below the 512 minimum.
        write_be16(&mut bytes, 16, 256);
        assert!(DatabaseHeader::read(&bytes).is_err());
    }

    #[test]
    fn in_header_size_invalid_when_counters_differ() {
        let mut hdr = DatabaseHeader::default();
        hdr.file_change_counter = 5;
        hdr.version_valid_for = 4;
        hdr.database_size_pages = 10;
        assert!(!hdr.in_header_size_valid());
        hdr.version_valid_for = 5;
        assert!(hdr.in_header_size_valid());
    }
}
