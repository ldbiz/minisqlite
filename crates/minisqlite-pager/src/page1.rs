//! Page-1 header maintenance shared by every durable commit path (the rollback
//! [`DiskStore`](crate::diskstore) and the WAL [`WalStore`](crate::walstore)).
//!
//! A committing transaction must land a page-1 header that records the new page
//! count and a bumped change counter, so a later reader (or a checkpoint that
//! copies the page-1 frame into the database file) sees a self-consistent header.
//! Both durable backings need exactly this transformation, so it lives here once as
//! a pure function over bytes rather than being duplicated per backing.

use std::collections::BTreeMap;

use minisqlite_fileformat::{DatabaseHeader, HEADER_SIZE};
use minisqlite_types::Result;

/// Fold the page-1 header maintenance into `write_set` for a commit that grows the
/// database to `new_count` pages.
///
/// The page-1 bytes to patch are the version being committed (`write_set[1]`) when
/// this transaction already changed page 1 — so freelist/b-tree edits are preserved
/// — otherwise `current_page1` (the currently committed page 1, or `None` when the
/// database has no page 1 yet). Only the first 100 bytes are rewritten; the region
/// beyond byte 100 and every header field other than the change counter,
/// version-valid-for, and size are left exactly as they were. If page 1 is not a
/// formatted SQLite header there is nothing to maintain and `write_set` is
/// unchanged.
///
/// Keeping `version_valid_for == file_change_counter` keeps
/// [`DatabaseHeader::in_header_size_valid`] true so a real `sqlite3` trusts the
/// in-header size (offset 28). The counter wraps (matching SQLite's 32-bit counter)
/// and never panics.
pub(crate) fn maintain_header(
    write_set: &mut BTreeMap<u32, Box<[u8]>>,
    new_count: u32,
    current_page1: Option<&[u8]>,
) -> Result<()> {
    let mut page1: Box<[u8]> = match write_set.get(&1) {
        Some(buf) => buf.clone(),
        None => match current_page1 {
            Some(bytes) => Box::from(bytes),
            None => return Ok(()),
        },
    };
    // A page shorter than the header cannot carry one; leave it untouched rather
    // than slicing out of bounds (real pages are >= 512 bytes, so this is defense).
    if page1.len() < HEADER_SIZE {
        return Ok(());
    }

    let mut head = [0u8; HEADER_SIZE];
    head.copy_from_slice(&page1[..HEADER_SIZE]);
    let mut header = match DatabaseHeader::read(&head) {
        Ok(h) => h,
        Err(_) => return Ok(()),
    };

    header.file_change_counter = header.file_change_counter.wrapping_add(1);
    header.version_valid_for = header.file_change_counter;
    header.database_size_pages = new_count;
    header.write(&mut head);
    page1[..HEADER_SIZE].copy_from_slice(&head);
    write_set.insert(1, page1);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PS: usize = 512;

    fn header_page(page_size: u32, tail: u8) -> Box<[u8]> {
        let hdr = DatabaseHeader { page_size, ..DatabaseHeader::default() };
        let mut page1 = vec![tail; page_size as usize];
        page1[..HEADER_SIZE].copy_from_slice(&hdr.to_bytes());
        page1.into_boxed_slice()
    }

    fn read_header(page1: &[u8]) -> DatabaseHeader {
        let mut buf = [0u8; HEADER_SIZE];
        buf.copy_from_slice(&page1[..HEADER_SIZE]);
        DatabaseHeader::read(&buf).unwrap()
    }

    #[test]
    fn patches_page1_from_write_set_and_bumps_counter() {
        let mut ws: BTreeMap<u32, Box<[u8]>> = BTreeMap::new();
        ws.insert(1, header_page(PS as u32, 0));
        maintain_header(&mut ws, 7, None).unwrap();
        let h = read_header(&ws[&1]);
        assert_eq!(h.database_size_pages, 7, "size recorded");
        assert_eq!(h.file_change_counter, 1, "counter bumped from 0");
        assert_eq!(h.version_valid_for, 1, "version-valid-for tracks the counter");
        assert!(h.in_header_size_valid());
    }

    #[test]
    fn uses_current_page1_when_write_set_lacks_it() {
        let mut ws: BTreeMap<u32, Box<[u8]>> = BTreeMap::new();
        ws.insert(2, vec![0u8; PS].into_boxed_slice());
        let current = header_page(PS as u32, 0x5A);
        maintain_header(&mut ws, 3, Some(&current)).unwrap();
        // Page 1 is now in the write set, patched from the committed image, and its
        // tail bytes (past the header) are preserved.
        let page1 = &ws[&1];
        assert_eq!(read_header(page1).database_size_pages, 3);
        assert_eq!(&page1[HEADER_SIZE..], &vec![0x5A; PS - HEADER_SIZE][..]);
    }

    #[test]
    fn write_set_page1_wins_over_current() {
        // When the transaction already staged page 1 (e.g. a freelist edit), that
        // version is patched, NOT the committed one — its tail must survive.
        let mut ws: BTreeMap<u32, Box<[u8]>> = BTreeMap::new();
        ws.insert(1, header_page(PS as u32, 0xEE));
        let current = header_page(PS as u32, 0x11);
        maintain_header(&mut ws, 2, Some(&current)).unwrap();
        assert_eq!(&ws[&1][HEADER_SIZE..], &vec![0xEE; PS - HEADER_SIZE][..], "staged tail kept");
    }

    #[test]
    fn no_page1_anywhere_is_a_noop() {
        let mut ws: BTreeMap<u32, Box<[u8]>> = BTreeMap::new();
        ws.insert(2, vec![0u8; PS].into_boxed_slice());
        maintain_header(&mut ws, 5, None).unwrap();
        assert!(!ws.contains_key(&1), "nothing to maintain without a page 1");
    }

    #[test]
    fn unformatted_page1_left_untouched() {
        // Page 1 that is not a valid SQLite header (no magic) is not patched.
        let mut ws: BTreeMap<u32, Box<[u8]>> = BTreeMap::new();
        ws.insert(1, vec![0x22u8; PS].into_boxed_slice());
        maintain_header(&mut ws, 9, None).unwrap();
        assert_eq!(&ws[&1][..], &vec![0x22u8; PS][..], "garbage page 1 is not rewritten");
    }

    #[test]
    fn preserves_wal_version_bytes() {
        // A version-2 (WAL) header must stay version 2 after maintenance — the WAL
        // path relies on this so a checkpointed page-1 keeps the file in WAL mode.
        let mut hdr = DatabaseHeader::default();
        hdr.page_size = PS as u32;
        hdr.write_version = 2;
        hdr.read_version = 2;
        let mut page1 = vec![0u8; PS];
        page1[..HEADER_SIZE].copy_from_slice(&hdr.to_bytes());
        let mut ws: BTreeMap<u32, Box<[u8]>> = BTreeMap::new();
        ws.insert(1, page1.into_boxed_slice());
        maintain_header(&mut ws, 4, None).unwrap();
        let h = read_header(&ws[&1]);
        assert_eq!(h.write_version, 2);
        assert_eq!(h.read_version, 2);
        assert_eq!(h.database_size_pages, 4);
    }
}
