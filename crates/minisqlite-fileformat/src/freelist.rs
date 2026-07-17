//! Freelist trunk pages (fileformat2 §1.5). Pages that fall out of use are kept
//! on a freelist so they can be reused instead of growing the file. The freelist
//! is a singly linked list of "trunk" pages; each trunk page is an array of 4-byte
//! big-endian integers filling the usable page area:
//!
//! - `int[0]` — the page number of the next trunk page, or 0 if this is the last.
//! - `int[1]` — L, the number of freelist *leaf* page pointers that follow.
//! - `int[2 .. 2+L]` — the freelist leaf page numbers.
//!
//! Freelist leaf pages themselves carry no information (SQLite never reads or
//! writes their contents), so this module only models trunk pages.
//!
//! This is the trunk-page BYTE LAYOUT only. Which pages are free, the order they
//! are handed back out, and the header's freelist head/count (offsets 32 and 36)
//! are the pager's allocation policy, not this codec's concern.

use crate::bytes::{be32, try_be32, write_be32};
use minisqlite_types::{Error, Result};

/// The fixed prefix of a trunk page: `int[0]` (next trunk) and `int[1]` (leaf
/// count), 8 bytes, before the leaf-pointer array begins.
pub const TRUNK_HEADER_LEN: usize = 8;

/// Number of leaf-pointer slots a trunk page of usable size `usable_size` can
/// physically hold: the usable area is `usable_size / 4` four-byte integers, less
/// the two-integer header (`next` and `L`).
///
/// Note for the allocator (not enforced here): real SQLite additionally leaves the
/// last six leaf slots unused so that files it writes stay readable by pre-3.6.0
/// versions. That is a writer policy on top of this physical capacity; a trunk
/// page written by another tool may legitimately fill all `trunk_leaf_capacity`
/// slots, and this codec reads either.
pub fn trunk_leaf_capacity(usable_size: usize) -> usize {
    (usable_size / 4).saturating_sub(2)
}

/// The next trunk page in the freelist (0 if this is the last trunk), read from
/// `int[0]`. Fails closed if `page` is shorter than the 4-byte field.
pub fn trunk_next(page: &[u8]) -> Result<u32> {
    try_be32(page, 0).ok_or_else(|| Error::Format("freelist trunk page too small for next pointer".into()))
}

/// The declared leaf-pointer count L from `int[1]`. Fails closed if `page` is
/// shorter than the header. This is the raw stored count; [`parse_trunk`] and
/// [`trunk_leaves`] additionally validate it against the page's capacity before
/// reading that many pointers.
pub fn trunk_leaf_count(page: &[u8]) -> Result<u32> {
    try_be32(page, 4).ok_or_else(|| Error::Format("freelist trunk page too small for leaf count".into()))
}

/// Validate the declared leaf count against both the physical capacity for
/// `usable_size` and the actual length of `page`, returning it as `usize`. A
/// corrupt count (one that would read past the usable area or the slice) is
/// rejected rather than followed out of bounds.
fn checked_leaf_count(page: &[u8], usable_size: usize) -> Result<usize> {
    let count = trunk_leaf_count(page)? as usize;
    let capacity = trunk_leaf_capacity(usable_size);
    if count > capacity || TRUNK_HEADER_LEN + count * 4 > page.len() {
        return Err(Error::Format(format!(
            "freelist trunk leaf count {count} exceeds capacity {capacity} (usable {usable_size})"
        )));
    }
    Ok(count)
}

/// Iterate the freelist leaf page numbers on a trunk page without allocating.
/// `usable_size` bounds the declared count; a count beyond the page's capacity is
/// rejected. Yields exactly L page numbers in stored order.
pub fn trunk_leaves(page: &[u8], usable_size: usize) -> Result<impl Iterator<Item = u32> + '_> {
    let count = checked_leaf_count(page, usable_size)?;
    // Bounds proven by `checked_leaf_count`, so the `be32` reads cannot panic.
    Ok((0..count).map(move |i| be32(page, TRUNK_HEADER_LEN + i * 4)))
}

/// Parse a freelist trunk page into its next-trunk pointer and the list of
/// freelist leaf page numbers it carries. `usable_size` (page size minus reserved
/// bytes) bounds how many leaf pointers the page can physically hold.
pub fn parse_trunk(page: &[u8], usable_size: usize) -> Result<(u32, Vec<u32>)> {
    let next = trunk_next(page)?;
    let leaves = trunk_leaves(page, usable_size)?.collect();
    Ok((next, leaves))
}

/// Write a freelist trunk page into `page` (which must cover at least the usable
/// region): `next` as `int[0]`, the leaf count as `int[1]`, then the leaf page
/// numbers. Fails closed if the leaves do not fit the usable area or the buffer.
/// Bytes past the written integers are left untouched (the caller supplies a
/// zeroed page buffer).
pub fn write_trunk(page: &mut [u8], next: u32, leaves: &[u32], usable_size: usize) -> Result<()> {
    if usable_size > page.len() {
        return Err(Error::Format("freelist trunk page buffer smaller than usable size".into()));
    }
    let capacity = trunk_leaf_capacity(usable_size);
    if leaves.len() > capacity {
        return Err(Error::Format(format!(
            "freelist trunk cannot hold {} leaves (capacity {capacity})",
            leaves.len()
        )));
    }
    write_be32(page, 0, next);
    write_be32(page, 4, leaves.len() as u32);
    for (i, &leaf) in leaves.iter().enumerate() {
        write_be32(page, TRUNK_HEADER_LEN + i * 4, leaf);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_math() {
        // usable/4 integers, minus the 2-integer header.
        assert_eq!(trunk_leaf_capacity(512), 128 - 2);
        assert_eq!(trunk_leaf_capacity(4096), 1024 - 2);
        assert_eq!(trunk_leaf_capacity(65536), 16384 - 2);
        // Degenerate tiny sizes never underflow.
        assert_eq!(trunk_leaf_capacity(8), 0);
        assert_eq!(trunk_leaf_capacity(4), 0);
        assert_eq!(trunk_leaf_capacity(0), 0);
    }

    #[test]
    fn roundtrip_with_leaves() {
        let usable = 512usize;
        let mut page = vec![0u8; usable];
        let leaves = [7u32, 9, 100, 4_294_967_294];
        write_trunk(&mut page, 5, &leaves, usable).unwrap();

        let (next, got) = parse_trunk(&page, usable).unwrap();
        assert_eq!(next, 5);
        assert_eq!(got, leaves);

        // The zero-copy accessors agree with the parsed form.
        assert_eq!(trunk_next(&page).unwrap(), 5);
        assert_eq!(trunk_leaf_count(&page).unwrap(), leaves.len() as u32);
        let iter: Vec<u32> = trunk_leaves(&page, usable).unwrap().collect();
        assert_eq!(iter, leaves);
    }

    #[test]
    fn byte_exact_layout() {
        // Lock the on-disk bytes: four big-endian ints, next then L then the leaves.
        let usable = 512usize;
        let mut page = vec![0u8; usable];
        write_trunk(&mut page, 5, &[7, 9, 100], usable).unwrap();
        assert_eq!(&page[0..4], &[0x00, 0x00, 0x00, 0x05]); // next = 5
        assert_eq!(&page[4..8], &[0x00, 0x00, 0x00, 0x03]); // L = 3
        assert_eq!(&page[8..12], &[0x00, 0x00, 0x00, 0x07]); // leaf 7
        assert_eq!(&page[12..16], &[0x00, 0x00, 0x00, 0x09]); // leaf 9
        assert_eq!(&page[16..20], &[0x00, 0x00, 0x00, 0x64]); // leaf 100
    }

    #[test]
    fn last_trunk_and_empty_trunk() {
        let usable = 4096usize;
        let mut page = vec![0u8; usable];
        // next = 0 marks the final trunk; L = 0 means it lists no leaves.
        write_trunk(&mut page, 0, &[], usable).unwrap();
        let (next, leaves) = parse_trunk(&page, usable).unwrap();
        assert_eq!(next, 0);
        assert!(leaves.is_empty());
        assert_eq!(trunk_leaf_count(&page).unwrap(), 0);
    }

    #[test]
    fn full_capacity_roundtrip() {
        let usable = 512usize;
        let capacity = trunk_leaf_capacity(usable); // 126
        let leaves: Vec<u32> = (0..capacity as u32).map(|i| i + 2).collect();
        let mut page = vec![0u8; usable];
        write_trunk(&mut page, 0, &leaves, usable).unwrap();
        // The last leaf ends exactly at the usable boundary.
        assert_eq!(TRUNK_HEADER_LEN + capacity * 4, usable);
        let (next, got) = parse_trunk(&page, usable).unwrap();
        assert_eq!(next, 0);
        assert_eq!(got, leaves);
    }

    #[test]
    fn write_rejects_too_many_leaves() {
        let usable = 512usize;
        let capacity = trunk_leaf_capacity(usable);
        let leaves: Vec<u32> = (0..capacity as u32 + 1).collect();
        let mut page = vec![0u8; usable];
        assert!(write_trunk(&mut page, 0, &leaves, usable).is_err());
    }

    #[test]
    fn write_rejects_buffer_smaller_than_usable() {
        let mut page = vec![0u8; 100];
        assert!(write_trunk(&mut page, 0, &[], 512).is_err());
    }

    #[test]
    fn parse_rejects_corrupt_leaf_count() {
        // Page buffer larger than the usable region so the length check passes and
        // the capacity check is what rejects the bogus count.
        let usable = 512usize; // capacity 126
        let mut page = vec![0u8; 1024];
        write_be32(&mut page, 0, 3); // next
        write_be32(&mut page, 4, 200); // L far beyond capacity 126
        assert!(parse_trunk(&page, usable).is_err());
        assert!(trunk_leaves(&page, usable).is_err());
    }

    #[test]
    fn parse_rejects_count_past_slice() {
        // L fits the usable capacity by claim, but the actual slice is too short to
        // hold the pointers, so reading them would overrun — reject instead.
        let usable = 512usize;
        let mut page = vec![0u8; 20]; // room for header + 3 leaves only
        write_be32(&mut page, 0, 0);
        write_be32(&mut page, 4, 5); // claims 5 leaves, slice holds 3
        assert!(parse_trunk(&page, usable).is_err());
    }

    #[test]
    fn accessors_fail_closed_on_short_page() {
        assert!(trunk_next(&[]).is_err());
        assert!(trunk_next(&[0, 0, 0]).is_err());
        assert!(trunk_leaf_count(&[0, 0, 0, 0]).is_err());
        assert!(trunk_leaf_count(&[0, 0, 0, 0, 0, 0, 0]).is_err());
        assert!(trunk_leaf_count(&[0, 0, 0, 0, 0, 0, 0, 0]).is_ok());
    }
}
