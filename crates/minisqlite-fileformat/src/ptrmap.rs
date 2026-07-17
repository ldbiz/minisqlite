//! Pointer-map (ptrmap) page positions for auto_vacuum / incremental_vacuum databases
//! (fileformat2 §1.8, gated by the "largest root b-tree page" field at offset 52,
//! §1.3.12).
//!
//! A database whose offset-52 field is non-zero carries extra *ptrmap* pages that hold
//! child→parent back-pointers (5-byte entries) for the vacuum machinery — they are NOT
//! b-tree pages and hold no row data (§1.8). Crucially, **no b-tree pointer ever points
//! at a ptrmap page**, so a correct root→child traversal (which is how this engine reads
//! the schema on page 1 and every table/index from its `rootpage`) already skips them:
//! reading an auto_vacuum file needs no special-casing on the read path.
//!
//! This module is the single source of the ptrmap POSITION formula. Because the read
//! path is entirely pointer-driven, these helpers have NO read-path caller today and
//! guard nothing at runtime yet — a LATENT invariant, not a permanent fact. Any code that
//! ever walks pages by NUMBER rather than by pointer — a future VACUUM, `integrity_check`,
//! freelist reclamation, a page-by-page dump, or the auto_vacuum WRITE path (all out of
//! scope for the read path this landed with) — MUST consult
//! [`DatabaseHeader::is_ptrmap_page`] and skip a ptrmap page, or it will read 5-byte
//! back-pointer entries as b-tree/cell bytes and corrupt the result. Ask here rather than
//! re-deriving §1.8 arithmetic and drifting.
//!
//! Layout (§1.8): the first ptrmap page is page 2. With `J = U / 5` entries per page (U =
//! usable page size), page 2 covers pages `3 ..= J+2`, the next ptrmap page is `J+3` and
//! covers `J+4 ..= 2J+3`, and so on — i.e. ptrmap pages sit every `J+1` pages starting at
//! page 2.
//!
//! Each ptrmap page holds `J` fixed 5-byte entries back to back, one per data page it
//! covers: `[1-byte type][4-byte big-endian parent page number]` (§1.8). This module
//! also owns that entry codec ([`encode_ptrmap_entry`] / [`decode_ptrmap_entry`] and the
//! in-page [`put_ptrmap_entry`] / [`get_ptrmap_entry`]) and the position→entry mapping
//! ([`ptrmap_page_of`] gives the carrying ptrmap page, [`ptrmap_offset_of`] the byte
//! offset of the entry within it), so the whole §1.8 layout lives in one place.

use crate::header::DatabaseHeader;

/// The fixed width of a single ptrmap entry — `[1-byte type][4-byte big-endian parent
/// page number]` = 5 bytes (§1.8). This is the SINGLE source of that width: the
/// entries-per-page divisor, the in-page entry offset stride, and the codec's array
/// width all derive from it, so they cannot drift apart.
pub const PTRMAP_ENTRY_SIZE: usize = 5;

/// The number of [`PTRMAP_ENTRY_SIZE`]-byte back-pointer entries that fit in a ptrmap
/// page: `J = U / 5` (§1.8), where `U` is the usable page size.
#[inline]
pub fn ptrmap_entries_per_page(usable_size: usize) -> u64 {
    (usable_size / PTRMAP_ENTRY_SIZE) as u64
}

/// The spacing between consecutive ptrmap pages: `J + 1` (§1.8). Each ptrmap page at
/// page `B` provides back-links for the `J` pages `B+1 ..= B+J`, so the next ptrmap page
/// is `J+1` pages later.
#[inline]
pub fn ptrmap_stride(usable_size: usize) -> u64 {
    ptrmap_entries_per_page(usable_size) + 1
}

/// Whether page `page_no` is at a ptrmap POSITION for a database with usable page size
/// `usable_size` (§1.8): page 2, then every `J+1` pages. This is the pure position test
/// and does NOT check whether the database uses ptrmap pages at all — callers must gate
/// it on that (see [`DatabaseHeader::is_ptrmap_page`]), because in a non-vacuum database
/// those same page numbers are ordinary b-tree/overflow/freelist pages.
///
/// Deferred edge (§1.8): a ptrmap page that would collide with the *lock-byte page* (the
/// page holding the byte at offset 2^30) is moved to the following page. That only occurs
/// in databases larger than 1 GiB, so it is documented and not handled here; typical
/// files and every hand-built fixture stay far below it. `usable_size >= 480` is enforced
/// by the header reader, so `J >= 96` and the stride is never zero.
#[inline]
pub fn is_ptrmap_page(usable_size: usize, page_no: u32) -> bool {
    if page_no < 2 {
        return false;
    }
    let stride = ptrmap_stride(usable_size);
    debug_assert!(stride >= 1, "ptrmap stride must be positive (usable_size {usable_size})");
    (page_no as u64 - 2) % stride == 0
}

/// The ptrmap page that carries `page_no`'s 5-byte back-pointer entry (§1.8). A ptrmap
/// page `B` provides the back-links for the `J` data pages `B+1 ..= B+J`, so this rounds
/// `page_no` down to the start of its stride window: `((page_no - 2) / stride) * stride
/// + 2`. Combine with [`ptrmap_offset_of`] to locate the exact entry.
///
/// Precondition: `page_no` is a real DATA page — `page_no >= 3` AND
/// `!is_ptrmap_page(usable_size, page_no)` (never page 0/1, page 2, or a ptrmap page
/// itself). Callers hold this; it is `debug_assert!`ed here (a `page_no < 2` would also
/// underflow the `- 2`). The same lock-byte-page edge (>1 GiB files) deferred by
/// [`is_ptrmap_page`] is deferred here too.
#[inline]
pub fn ptrmap_page_of(usable_size: usize, page_no: u32) -> u32 {
    debug_assert!(page_no >= 3, "ptrmap_page_of expects a data page (>= 3), got {page_no}");
    debug_assert!(
        !is_ptrmap_page(usable_size, page_no),
        "ptrmap_page_of expects a data page, not ptrmap page {page_no}"
    );
    let stride = ptrmap_stride(usable_size);
    (((page_no as u64 - 2) / stride) * stride + 2) as u32
}

/// The byte offset of `page_no`'s 5-byte entry WITHIN its ptrmap page (the one returned
/// by [`ptrmap_page_of`]): `5 * (page_no - ptrmap_page_of(..) - 1)` (§1.8). The first
/// data page after a ptrmap page lands at offset 0, the next at 5, up to `5 * (J - 1)`
/// for the last (`J`-th) page the ptrmap page covers.
///
/// Same data-page precondition as [`ptrmap_page_of`] (asserted there), and the same
/// lock-byte-page edge is deferred.
#[inline]
pub fn ptrmap_offset_of(usable_size: usize, page_no: u32) -> usize {
    PTRMAP_ENTRY_SIZE * ((page_no - ptrmap_page_of(usable_size, page_no) - 1) as usize)
}

impl DatabaseHeader {
    /// Whether this database uses pointer-map pages, i.e. it is in auto_vacuum OR
    /// incremental_vacuum mode: the "largest root b-tree page" field (offset 52,
    /// §1.3.12) is non-zero exactly in that case, and both modes keep ptrmap pages
    /// (§1.8). A plain (non-vacuum) database has the field at 0 and contains no ptrmap
    /// pages.
    pub fn uses_ptrmap(&self) -> bool {
        self.largest_root_btree != 0
    }

    /// Whether page `page_no` is a ptrmap page IN THIS database: false unless the
    /// database uses ptrmap pages, otherwise the §1.8 position test at this header's
    /// usable page size. This is the safe, gated form the read/verify paths use so they
    /// cannot mistake a non-vacuum file's ordinary page for a ptrmap page.
    pub fn is_ptrmap_page(&self, page_no: u32) -> bool {
        self.uses_ptrmap() && is_ptrmap_page(self.usable_size(), page_no)
    }
}

// ---- 5-byte ptrmap entry codec (§1.8) ---------------------------------------------
//
// Every entry is `[1-byte type][4-byte big-endian parent page number]`. The parent's
// meaning is set by the type; a parent of 0 means "no parent" (root and freelist pages).

/// Entry type 1 — a b-tree ROOT page (a table's or index's `rootpage`). Parent is 0: a
/// root has no parent (§1.8).
pub const PTRMAP_ROOTPAGE: u8 = 1;

/// Entry type 2 — a FREELIST page (trunk or leaf). Parent is 0 (§1.8).
pub const PTRMAP_FREEPAGE: u8 = 2;

/// Entry type 3 — the FIRST overflow page of a cell's overflow chain. Parent is the
/// b-tree page holding the cell whose payload spilled (§1.8).
pub const PTRMAP_OVERFLOW1: u8 = 3;

/// Entry type 4 — a LATER overflow page in a chain. Parent is the previous overflow page
/// in that same chain (§1.8).
pub const PTRMAP_OVERFLOW2: u8 = 4;

/// Entry type 5 — a NON-root b-tree page. Parent is its parent b-tree page (§1.8).
pub const PTRMAP_BTREE: u8 = 5;

/// Encode a ptrmap entry into a [`PTRMAP_ENTRY_SIZE`]-byte array: `[entry_type, parent
/// as 4 big-endian bytes]` (§1.8). Inverse of [`decode_ptrmap_entry`].
#[inline]
pub fn encode_ptrmap_entry(entry_type: u8, parent: u32) -> [u8; PTRMAP_ENTRY_SIZE] {
    let mut out = [0u8; PTRMAP_ENTRY_SIZE];
    out[0] = entry_type;
    crate::bytes::write_be32(&mut out, 1, parent);
    out
}

/// Decode a ptrmap entry into `(entry_type, parent)`. Reads EXACTLY [`PTRMAP_ENTRY_SIZE`]
/// bytes starting at index 0 of `bytes` — the caller slices its ptrmap page down to the
/// entry first (e.g. via [`ptrmap_offset_of`]); panics on a shorter slice.
#[inline]
pub fn decode_ptrmap_entry(bytes: &[u8]) -> (u8, u32) {
    (bytes[0], crate::bytes::be32(bytes, 1))
}

/// Write an encoded entry into `ptrmap_page[offset .. offset + PTRMAP_ENTRY_SIZE]`.
/// `offset` is normally [`ptrmap_offset_of`] for the data page whose back-pointer this
/// is; panics if the page is shorter than `offset + PTRMAP_ENTRY_SIZE`.
#[inline]
pub fn put_ptrmap_entry(ptrmap_page: &mut [u8], offset: usize, entry_type: u8, parent: u32) {
    let entry = encode_ptrmap_entry(entry_type, parent);
    ptrmap_page[offset..offset + PTRMAP_ENTRY_SIZE].copy_from_slice(&entry);
}

/// Read the entry at `ptrmap_page[offset .. offset + PTRMAP_ENTRY_SIZE]` as `(entry_type,
/// parent)` — the in-page inverse of [`put_ptrmap_entry`].
#[inline]
pub fn get_ptrmap_entry(ptrmap_page: &[u8], offset: usize) -> (u8, u32) {
    decode_ptrmap_entry(&ptrmap_page[offset..offset + PTRMAP_ENTRY_SIZE])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_and_stride_follow_the_j_formula() {
        // §1.8: J = U/5 entries per ptrmap page; stride between ptrmap pages is J+1.
        assert_eq!(ptrmap_entries_per_page(512), 102);
        assert_eq!(ptrmap_stride(512), 103);
        assert_eq!(ptrmap_entries_per_page(4096), 819);
        assert_eq!(ptrmap_stride(4096), 820);
        assert_eq!(ptrmap_entries_per_page(65536), 13107);
        assert_eq!(ptrmap_stride(65536), 13108);
    }

    #[test]
    fn first_ptrmap_is_page_2_then_every_stride() {
        // Page 1 (the header/schema root) and page 0 are never ptrmap pages; page 2 is
        // the first, then every J+1 pages. Checked against the §1.8 worked layout for a
        // 512-byte page (J=102, stride=103): ptrmap pages at 2, 105, 208, ...
        let u = 512;
        assert!(!is_ptrmap_page(u, 0));
        assert!(!is_ptrmap_page(u, 1));
        assert!(is_ptrmap_page(u, 2), "first ptrmap page is page 2 (§1.8)");
        assert!(!is_ptrmap_page(u, 3), "page 3 is a data page covered by ptrmap page 2");
        // Pages 3..=104 (= J+2 = 104) are covered by ptrmap page 2 and are NOT ptrmap.
        for p in 3..=104 {
            assert!(!is_ptrmap_page(u, p), "page {p} is data, not ptrmap");
        }
        assert!(is_ptrmap_page(u, 105), "second ptrmap page is J+3 = 105 (§1.8)");
        assert!(is_ptrmap_page(u, 208), "third ptrmap page is 2*(J+1)+2 = 208");
        assert!(!is_ptrmap_page(u, 106));
        assert!(!is_ptrmap_page(u, 207));
    }

    #[test]
    fn header_gate_requires_offset_52_nonzero() {
        let mut h = DatabaseHeader::default();
        // A plain database (offset 52 == 0) has NO ptrmap pages, so page 2 is an
        // ordinary page here — the gated test must say false even at a ptrmap position.
        assert!(!h.uses_ptrmap());
        assert!(!h.is_ptrmap_page(2), "non-vacuum db: page 2 is not a ptrmap page");
        // An auto_vacuum database (offset 52 != 0) does have them.
        h.largest_root_btree = 1;
        assert!(h.uses_ptrmap());
        assert!(h.is_ptrmap_page(2), "auto_vacuum db: page 2 is the first ptrmap page");
        assert!(!h.is_ptrmap_page(1), "page 1 is the header/schema root, never ptrmap");
        assert!(!h.is_ptrmap_page(3), "page 3 is the first data page after the ptrmap");
    }

    #[test]
    fn stride_scales_with_usable_size() {
        // At 4096-byte pages J=819, stride=820: ptrmap pages at 2, 822, 1642, ...
        let u = 4096;
        assert!(h_is(u, 2));
        assert!(!h_is(u, 3));
        assert!(!h_is(u, 821));
        assert!(h_is(u, 822));
        assert!(h_is(u, 1642));
    }

    #[test]
    fn page_and_offset_worked_examples_512() {
        // §1.8 worked layout for a 512-byte usable size: J=102, stride=103, so ptrmap
        // page 2 covers data pages 3..=104 and ptrmap page 105 covers 106..=207.
        let u = 512;
        // First data page after ptrmap page 2 → first entry (offset 0).
        assert_eq!(ptrmap_page_of(u, 3), 2);
        assert_eq!(ptrmap_offset_of(u, 3), 0);
        // Page 104 is the LAST page ptrmap page 2 covers → the 102nd (last) entry at
        // offset 5*101 = 505.
        assert_eq!(ptrmap_page_of(u, 104), 2);
        assert_eq!(ptrmap_offset_of(u, 104), 505);
        // Page 106 is the first data page after ptrmap page 105 → first entry there.
        assert_eq!(ptrmap_page_of(u, 106), 105);
        assert_eq!(ptrmap_offset_of(u, 106), 0);
        // Page 207 is the last page ptrmap page 105 covers → last entry.
        assert_eq!(ptrmap_page_of(u, 207), 105);
        assert_eq!(ptrmap_offset_of(u, 207), 505);
    }

    #[test]
    fn page_and_offset_worked_examples_4096() {
        // A larger usable size, checked across the SECOND ptrmap window (past where the
        // consistency sweep below reaches): 4096B → J=819, stride=820, ptrmap pages at
        // 2, 822, 1642. Page 823 is the first data page after ptrmap page 822; page 1641
        // is the last page 822 covers (the 819th entry at offset 5*818 = 4090).
        let u = 4096;
        assert_eq!(ptrmap_page_of(u, 823), 822);
        assert_eq!(ptrmap_offset_of(u, 823), 0);
        assert_eq!(ptrmap_page_of(u, 1641), 822);
        assert_eq!(ptrmap_offset_of(u, 1641), 4090);
    }

    #[test]
    fn page_of_consistent_with_is_ptrmap_page() {
        // For every real data page p: its carrying ptrmap page must itself BE a ptrmap
        // page, sit strictly before p and within one stride window (B < p <= B+J), and
        // the computed entry offset must land inside that page's J-entry region.
        for u in [512usize, 4096] {
            let j = ptrmap_entries_per_page(u);
            for p in 3u32..=400 {
                if is_ptrmap_page(u, p) {
                    continue;
                }
                let b = ptrmap_page_of(u, p);
                assert!(
                    is_ptrmap_page(u, b),
                    "carrying page {b} of data page {p} (u={u}) must be a ptrmap page"
                );
                assert!(
                    (b as u64) < p as u64 && p as u64 <= b as u64 + j,
                    "data page {p} must fall in ({b}, {b}+{j}] (u={u})"
                );
                assert!(
                    ptrmap_offset_of(u, p) < j as usize * PTRMAP_ENTRY_SIZE,
                    "entry for page {p} must land within the J-entry region (u={u})"
                );
            }
        }
    }

    #[test]
    fn entry_type_constants_match_spec() {
        // §1.8 entry-type codes — pinned so a typo cannot silently change the on-disk
        // meaning that downstream (pager) code writes into real files.
        assert_eq!(
            (PTRMAP_ROOTPAGE, PTRMAP_FREEPAGE, PTRMAP_OVERFLOW1, PTRMAP_OVERFLOW2, PTRMAP_BTREE),
            (1, 2, 3, 4, 5)
        );
    }

    #[test]
    fn entry_codec_round_trips() {
        // Every entry type across a spread of parents (0 = "no parent", 1, 3, and the
        // max u32) survives encode→decode unchanged.
        let types =
            [PTRMAP_ROOTPAGE, PTRMAP_FREEPAGE, PTRMAP_OVERFLOW1, PTRMAP_OVERFLOW2, PTRMAP_BTREE];
        for t in types {
            for parent in [0u32, 1, 3, 0xFFFF_FFFF] {
                assert_eq!(decode_ptrmap_entry(&encode_ptrmap_entry(t, parent)), (t, parent));
            }
        }
    }

    #[test]
    fn put_get_round_trip_touches_only_five_bytes() {
        // put/get is the in-page form of encode/decode: writing an entry must change
        // exactly its PTRMAP_ENTRY_SIZE bytes and leave the rest untouched. Start from a
        // non-zero sentinel so a stray zeroing of a byte adjacent to the entry is caught
        // (an all-zero page would hide it).
        for offset in [0usize, 37] {
            let mut page = [0xFFu8; 64];
            put_ptrmap_entry(&mut page, offset, PTRMAP_BTREE, 0x0A0B_0C0D);
            assert_eq!(get_ptrmap_entry(&page, offset), (PTRMAP_BTREE, 0x0A0B_0C0D));
            for (i, &byte) in page.iter().enumerate() {
                if (offset..offset + PTRMAP_ENTRY_SIZE).contains(&i) {
                    continue;
                }
                assert_eq!(byte, 0xFF, "byte {i} outside the entry must be untouched");
            }
        }
    }

    #[test]
    fn entry_byte_layout_is_type_then_big_endian_parent() {
        // Pin the exact wire bytes: type first, then the 4-byte big-endian parent.
        assert_eq!(encode_ptrmap_entry(PTRMAP_ROOTPAGE, 0), [1, 0, 0, 0, 0]);
        assert_eq!(encode_ptrmap_entry(PTRMAP_BTREE, 0x0102_0304), [5, 1, 2, 3, 4]);
        // Pin decode DIRECTLY (not just transitively via round-trip), so a decoder bug
        // cannot hide behind a matching encoder bug.
        assert_eq!(decode_ptrmap_entry(&[5, 1, 2, 3, 4]), (PTRMAP_BTREE, 0x0102_0304));
        assert_eq!(decode_ptrmap_entry(&[1, 0, 0, 0, 0]), (PTRMAP_ROOTPAGE, 0));
    }

    // Gated helper: an auto_vacuum header at the given usable size (reserved 0, so
    // usable == page_size for a power-of-two size).
    fn h_is(page_size: u32, page_no: u32) -> bool {
        let h = DatabaseHeader { page_size, largest_root_btree: 1, ..DatabaseHeader::default() };
        h.is_ptrmap_page(page_no)
    }
}
