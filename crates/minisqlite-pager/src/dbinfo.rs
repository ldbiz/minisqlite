//! Whole-database metadata read from the page-1 header through the `Pager` seam.
//!
//! The 100-byte database header lives at the start of page 1 (fileformat2 §1.3). A few
//! READ paths need a field from it — the TEXT encoding (§1.3.13) to decode UTF-16 text
//! into the engine's UTF-8 strings, and the auto/incremental-vacuum flag (§1.3.12,
//! whose non-zero value implies the ptrmap layout of §1.8). The `Pager` trait
//! deliberately hands out PAGES, not parsed header fields, so this helper parses the
//! header on demand from the page the pager already caches: no extra I/O and no change
//! to the storage seam. It is the single place the read path (catalog schema load,
//! executor row/index decode) learns the database's text encoding, so that knowledge
//! is derived one way rather than re-parsed inconsistently at each decode site.

use minisqlite_fileformat::{DatabaseHeader, HEADER_SIZE, TextEncoding};

use crate::Pager;

/// Parse the 100-byte database header (§1.3) from page 1, or `None` when there is no
/// valid header yet: an empty / just-created in-memory database, or a page-1 read that
/// errors or is somehow shorter than the header. Callers treat `None` as "use the
/// documented default" (UTF-8 text, non-vacuum) rather than failing a read over a
/// missing header.
pub fn read_database_header(pager: &dyn Pager) -> Option<DatabaseHeader> {
    let page = pager.read_page(1).ok()?;
    let bytes: &[u8; HEADER_SIZE] = page.get(..HEADER_SIZE)?.try_into().ok()?;
    DatabaseHeader::read(bytes).ok()
}

/// This database's TEXT encoding (§1.3.13), read from the page-1 header and defaulting
/// to UTF-8 when no valid header is present. A UTF-16 database (encoding 2 or 3) stores
/// every TEXT value — including schema names and SQL — as UTF-16, so the read path
/// threads this into the record decoder to transcode those bytes to the engine's
/// internal UTF-8 `String`.
pub fn text_encoding_of(pager: &dyn Pager) -> TextEncoding {
    read_database_header(pager).map(|h| h.text_encoding_kind()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemPager;
    use minisqlite_fileformat::{TEXT_ENCODING_UTF16BE, TEXT_ENCODING_UTF16LE};

    /// Build an in-memory pager whose page 1 begins with `header` (the rest of the page
    /// zero-filled), so the header helpers can be exercised without an on-disk file.
    /// Page 1 must be allocated (growing the db to one page) before it can be written.
    fn pager_with_header(header: &DatabaseHeader) -> MemPager {
        let page_size = header.page_size as usize;
        let mut page1 = vec![0u8; page_size];
        page1[..HEADER_SIZE].copy_from_slice(&header.to_bytes());
        let mut pager = MemPager::new(header.page_size);
        pager.begin().unwrap();
        assert_eq!(pager.allocate_page().unwrap(), 1, "first allocated page is page 1");
        pager.write_page(1, &page1).unwrap();
        pager.commit().unwrap();
        pager
    }

    #[test]
    fn reads_encoding_from_header() {
        for (code, want) in [
            (1u32, TextEncoding::Utf8),
            (TEXT_ENCODING_UTF16LE, TextEncoding::Utf16le),
            (TEXT_ENCODING_UTF16BE, TextEncoding::Utf16be),
        ] {
            let mut h = DatabaseHeader::default();
            h.text_encoding = code;
            let pager = pager_with_header(&h);
            assert_eq!(text_encoding_of(&pager), want, "encoding code {code}");
        }
    }

    #[test]
    fn reads_full_header_including_vacuum_flag() {
        let mut h = DatabaseHeader::default();
        h.largest_root_btree = 3; // auto_vacuum (§1.3.12)
        let pager = pager_with_header(&h);
        let got = read_database_header(&pager).expect("valid header");
        assert!(got.uses_ptrmap(), "offset-52 non-zero ⇒ ptrmap pages present");
        assert!(got.is_ptrmap_page(2), "page 2 is the first ptrmap page");
    }

    #[test]
    fn missing_header_defaults_to_utf8() {
        // A page-1 image with no valid SQLite magic must not fail the read; the helper
        // falls back to the UTF-8 default so an unformatted/empty db still decodes.
        // A freshly allocated page is zero-filled (no magic), which is exactly this case.
        let mut pager = MemPager::new(4096);
        pager.begin().unwrap();
        assert_eq!(pager.allocate_page().unwrap(), 1);
        pager.commit().unwrap();
        assert!(read_database_header(&pager).is_none());
        assert_eq!(text_encoding_of(&pager), TextEncoding::Utf8);
    }
}
