//! Cell payload overflow: the spill-threshold arithmetic (fileformat2 §1.6) and
//! the overflow-page chain (§1.7). When a cell's payload is larger than what fits
//! on its b-tree page, a prefix stays on the page and the remainder spills onto a
//! linked list of overflow pages. This module is pure arithmetic plus zero-copy
//! chain walking; the actual page I/O belongs to the pager, which supplies page
//! bytes through [`PageSource`] or drives the chain itself with
//! [`parse_overflow_page`].

use minisqlite_types::{Error, Result};

/// Which spill formula applies. Table-interior cells carry no payload and so never
/// spill; only table-leaf cells and index cells (leaf or interior, same formula)
/// do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellKind {
    /// Table b-tree leaf cell (page type 0x0d).
    TableLeaf,
    /// Index b-tree cell, leaf (0x0a) or interior (0x02).
    Index,
}

/// How a payload of a given size divides between the b-tree page and the overflow
/// chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PayloadSplit {
    /// Bytes stored inline on the b-tree page.
    pub local: usize,
    /// Bytes stored on overflow pages (`total - local`).
    pub overflow: usize,
    /// Whether any overflow pages are needed (i.e. `overflow > 0`).
    pub has_overflow: bool,
}

/// Compute how many payload bytes of a `payload_len`-byte cell stay on the b-tree
/// page for usable page size `usable_size` (page size minus reserved bytes).
///
/// The formula is fixed by the file format (§1.6), quirks and all:
///   X = U-35 (table leaf) or ((U-12)*64/255)-23 (index);
///   M = ((U-12)*32/255)-23;  K = M + ((P-M) % (U-4)).
///   P<=X: all inline.  P>X && K<=X: inline K.  P>X && K>X: inline M.
pub fn payload_split(usable_size: usize, payload_len: u64, kind: CellKind) -> PayloadSplit {
    let u = usable_size;
    debug_assert!(u >= 480, "usable page size is never below 480 (§1.3.4)");
    let p = payload_len as usize;

    let x = match kind {
        CellKind::TableLeaf => u - 35,
        CellKind::Index => (u - 12) * 64 / 255 - 23,
    };
    if p <= x {
        return PayloadSplit { local: p, overflow: 0, has_overflow: false };
    }
    let m = (u - 12) * 32 / 255 - 23;
    let k = m + (p - m) % (u - 4);
    let local = if k <= x { k } else { m };
    PayloadSplit { local, overflow: p - local, has_overflow: true }
}

/// Usable payload bytes per overflow page: the page minus the 4-byte next-page
/// pointer at its head (§1.7).
pub fn overflow_page_capacity(usable_size: usize) -> usize {
    usable_size - 4
}

/// Split a full payload into its inline prefix and its overflow tail for writing.
/// The overflow slice is empty when the payload fits inline.
pub fn split_payload_for_write(payload: &[u8], usable_size: usize, kind: CellKind) -> (&[u8], &[u8]) {
    let split = payload_split(usable_size, payload.len() as u64, kind);
    payload.split_at(split.local)
}

/// Chunk the overflow tail into per-page payload slices (each up to
/// `overflow_page_capacity(usable_size)` bytes). The pager assigns a page number
/// to each chunk and writes `[next_page: u32][chunk]`, linking them in order with
/// a final next-page of 0.
pub fn overflow_chunks(overflow: &[u8], usable_size: usize) -> impl Iterator<Item = &[u8]> {
    overflow.chunks(overflow_page_capacity(usable_size).max(1))
}

/// A parsed overflow page: the next page in the chain (0 = last) and the payload
/// bytes it carries (borrowed from the page, the `usable_size - 4` bytes after the
/// pointer).
#[derive(Debug, Clone, Copy)]
pub struct OverflowPage<'a> {
    pub next: u32,
    pub payload: &'a [u8],
}

/// Parse an overflow page's header and payload region, borrowing from `page`.
/// `page` must be at least `usable_size` bytes.
pub fn parse_overflow_page(page: &[u8], usable_size: usize) -> Result<OverflowPage<'_>> {
    if page.len() < usable_size || usable_size < 5 {
        return Err(Error::Format("overflow page shorter than usable size".into()));
    }
    let next = u32::from_be_bytes([page[0], page[1], page[2], page[3]]);
    Ok(OverflowPage { next, payload: &page[4..usable_size] })
}

/// Write the next-page pointer (the leading 4 bytes) of an overflow page. A
/// `next` of 0 marks the last page in the chain. The write side of
/// [`parse_overflow_page`]'s `next`; keeps the pager from hard-coding the 4-byte
/// header layout.
pub fn set_overflow_next(page: &mut [u8], next: u32) -> Result<()> {
    if page.len() < 4 {
        return Err(Error::Format("overflow page too small for next pointer".into()));
    }
    page[0..4].copy_from_slice(&next.to_be_bytes());
    Ok(())
}

/// The writable content region of an overflow page: the `usable_size - 4` bytes
/// after the next-page pointer. The mutable counterpart of
/// [`parse_overflow_page`]'s `payload`, letting the pager fill overflow content in
/// place without copying through this crate.
pub fn overflow_content_mut(page: &mut [u8], usable_size: usize) -> Result<&mut [u8]> {
    if page.len() < usable_size || usable_size < 5 {
        return Err(Error::Format("overflow page shorter than usable size".into()));
    }
    Ok(&mut page[4..usable_size])
}

/// Write one complete overflow page into `page`: the next-page pointer followed by
/// `content`, which must fit the `usable_size - 4` capacity. Content shorter than
/// the capacity leaves the trailing bytes of the usable region as-is (the caller
/// supplies a zeroed buffer). The exact inverse of [`parse_overflow_page`].
pub fn write_overflow_page(
    page: &mut [u8],
    next: u32,
    content: &[u8],
    usable_size: usize,
) -> Result<()> {
    let region = overflow_content_mut(page, usable_size)?;
    if content.len() > region.len() {
        return Err(Error::Format(format!(
            "overflow content {} exceeds page capacity {}",
            content.len(),
            region.len()
        )));
    }
    region[..content.len()].copy_from_slice(content);
    set_overflow_next(page, next)
}

/// Supplies raw page bytes by page number. The pager implements this over its
/// cache/file so payload assembly can borrow pages without copying whole pages
/// into this crate.
pub trait PageSource {
    /// Borrow the full bytes of page `page_no` (1-based).
    fn page(&self, page_no: u32) -> Result<&[u8]>;
}

/// Reassemble a complete `total_len`-byte payload from its inline prefix `local`
/// plus the overflow chain beginning at `first_overflow`. Pages are fetched
/// through `src` and copied out exactly once into the result (overflow is the
/// rare large-payload path). Fails closed if the chain ends early or a page is
/// missing.
///
/// Termination: each overflow page contributes at least one byte, so the walk
/// strictly approaches `total_len` and is additionally bounded to the number of
/// pages a well-formed chain needs (`ceil(overflow / capacity)`) — a corrupt
/// chain cannot make it loop. The caller is responsible for validating
/// `total_len` against the database size before calling; the up-front reservation
/// is capped so a corrupt (huge) `total_len` cannot trigger a giant allocation.
pub fn assemble_payload<S: PageSource>(
    local: &[u8],
    total_len: usize,
    first_overflow: u32,
    usable_size: usize,
    src: &S,
) -> Result<Vec<u8>> {
    if local.len() > total_len {
        return Err(Error::Format("inline payload exceeds declared total".into()));
    }
    // Cap the pre-reservation; the vector still grows for genuinely large payloads.
    const MAX_PREALLOC: usize = 1 << 20;
    let mut result = Vec::with_capacity(total_len.min(MAX_PREALLOC));
    result.extend_from_slice(local);

    let cap = overflow_page_capacity(usable_size).max(1);
    let max_pages = (total_len - local.len()).div_ceil(cap);
    let mut pages_read = 0usize;
    let mut next = first_overflow;
    while result.len() < total_len {
        if next == 0 {
            return Err(Error::Format("overflow chain shorter than declared payload".into()));
        }
        if pages_read >= max_pages {
            return Err(Error::Format("overflow chain longer than declared payload".into()));
        }
        pages_read += 1;
        let page = src.page(next)?;
        let ov = parse_overflow_page(page, usable_size)?;
        let remaining = total_len - result.len();
        let take = remaining.min(ov.payload.len());
        result.extend_from_slice(&ov.payload[..take]);
        next = ov.next;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    struct MapSource(HashMap<u32, Vec<u8>>);
    impl PageSource for MapSource {
        fn page(&self, page_no: u32) -> Result<&[u8]> {
            self.0
                .get(&page_no)
                .map(|v| v.as_slice())
                .ok_or_else(|| Error::Format(format!("missing page {page_no}")))
        }
    }

    #[test]
    fn small_payload_stays_inline() {
        // Worked from the §1.6 formula for U=4096: table-leaf X = U-35 = 4061.
        let s = payload_split(4096, 100, CellKind::TableLeaf);
        assert_eq!(s.local, 100);
        assert_eq!(s.overflow, 0);
        assert!(!s.has_overflow);

        let boundary = payload_split(4096, 4061, CellKind::TableLeaf);
        assert_eq!(boundary.local, 4061, "P == X stays fully inline");
        assert!(!boundary.has_overflow);
    }

    #[test]
    fn table_leaf_threshold_math_4096() {
        // U=4096: X=4061, M=((4084)*32/255)-23=489, U-4=4092.
        // P=5000 (> X): K = 489 + (5000-489)%4092 = 489 + 419 = 908 (<= X) -> local 908.
        let s = payload_split(4096, 5000, CellKind::TableLeaf);
        assert_eq!(s.local, 908);
        assert_eq!(s.overflow, 5000 - 908);
        assert!(s.has_overflow);
    }

    #[test]
    fn index_threshold_math_4096() {
        // U=4096: index X = ((4084)*64/255)-23 = 1002.
        let x = (4096 - 12) * 64 / 255 - 23;
        assert_eq!(x, 1002);
        let s = payload_split(4096, 1002, CellKind::Index);
        assert!(!s.has_overflow, "P == X inline");
        assert_eq!(s.local, 1002);
        // P = X+1 = 1003 crosses into overflow. M = ((4084)*32/255)-23 = 489, and
        // K = 489 + (1003-489)%4092 = 1003 > X, so this is the K>X branch: the
        // inline part is exactly M. Pinning M here is load-bearing — see
        // `m_branch_local_is_exactly_m` for why the K-branch pins cannot catch an
        // off-by-one M.
        let s2 = payload_split(4096, 1003, CellKind::Index);
        assert!(s2.has_overflow, "P == X+1 spills");
        assert_eq!(s2.local, 489, "K>X branch stores exactly M inline");
        assert_eq!(s2.overflow, 1003 - 489);
    }

    #[test]
    fn m_branch_local_is_exactly_m() {
        // The K>X branch stores exactly M bytes inline. M MUST be pinned to an
        // exact value here, because no other test can catch an off-by-one in M:
        //   * the P<=X and K<=X branches never reference M as the result;
        //   * the K<=X pin (table_leaf_threshold_math_4096, local==908) is INVARIANT
        //     to M±1, since K = M + (P-M)%(U-4) — shift M by +1 and (P-M) drops by 1,
        //     leaving K unchanged;
        //   * the `local >= m` bracket in split_never_below_m_and_local_le_x only
        //     bounds, it does not uniquely determine M.
        // So an M that is too large by 1 would slip through every other overflow
        // test while producing a wrong split real sqlite rejects. These two exact
        // pins are the mutation guard for the M constant. (Mutating M's `-23` to
        // `-22` makes exactly these assertions fail.)

        // Table leaf, U=512: X = 512-35 = 477; M = (500*32/255)-23 = 62-23 = 39.
        // P = 512 > X, K = 39 + (512-39)%508 = 512 > X, so local = M = 39.
        let tl = payload_split(512, 512, CellKind::TableLeaf);
        assert!(tl.has_overflow);
        assert_eq!(tl.local, 39, "table-leaf K>X branch stores exactly M=39");
        assert_eq!(tl.overflow, 512 - 39);

        // Index, U=512: X = (500*64/255)-23 = 125-23 = 102; M = 39 as above.
        // P = 512 > X, K = 39 + (512-39)%508 = 512 > X, so local = M = 39.
        let ix = payload_split(512, 512, CellKind::Index);
        assert!(ix.has_overflow);
        assert_eq!(ix.local, 39, "index K>X branch stores exactly M=39");
        assert_eq!(ix.overflow, 512 - 39);
    }

    #[test]
    fn spill_boundary_pinned_for_both_kinds_and_page_sizes() {
        // Close the whole "spill-formula constant off-by-one stays green" class by
        // pinning BOTH sides of the inline/overflow boundary for EACH cell kind:
        // P==X must stay fully inline, and P==X+1 must spill. X is derived here
        // straight from the §1.6 constants (U-35 for table leaf; (U-12)*64/255-23
        // for index), independent of the production path, so a shifted X is caught
        // in EITHER direction:
        //   * X too SMALL (e.g. `u-36`) makes P==X spill    -> the inline pin fails;
        //   * X too LARGE (e.g. `u-34`) makes P==X+1 inline -> the spill pin fails.
        // Without this, the table-leaf X upper bound was unpinned: `small_payload_
        // stays_inline` only pins P==X (catches X too small), and every other
        // table-leaf test lands on K<=X or the M branch regardless of X±1.
        // At P==X+1 the remainder (P-M) < U-4, so K = M+(P-M) = P > X: always the
        // K>X branch, whose inline part is exactly M — so the spill pin also
        // re-guards M across three page sizes for both kinds.
        for &u in &[512usize, 4096, 65536] {
            let m = (u - 12) * 32 / 255 - 23;
            for kind in [CellKind::TableLeaf, CellKind::Index] {
                let x = match kind {
                    CellKind::TableLeaf => u - 35,
                    CellKind::Index => (u - 12) * 64 / 255 - 23,
                };

                // P == X: the largest payload that stays fully inline.
                let inline = payload_split(u, x as u64, kind);
                assert!(!inline.has_overflow, "{kind:?} U={u}: P==X must stay inline");
                assert_eq!(inline.local, x, "{kind:?} U={u}: inline local == X");
                assert_eq!(inline.overflow, 0);

                // P == X+1: the smallest payload that spills; inline part is M.
                let spill = payload_split(u, x as u64 + 1, kind);
                assert!(spill.has_overflow, "{kind:?} U={u}: P==X+1 must spill");
                assert_eq!(spill.local, m, "{kind:?} U={u}: spill inline == M");
                assert_eq!(spill.overflow, x + 1 - m, "{kind:?} U={u}: overflow == P-M");
            }
        }
    }

    #[test]
    fn k_equals_x_boundary_selects_k_not_m() {
        // The branch selector is `local = if k <= x { k } else { m }`. Pin the
        // K==X boundary so a `<=`->`<` mutation (which would wrongly fall through
        // to M) is caught. Table leaf U=4096: X=4061, M=489, U-4=4092. For P=8153
        // (>X): K = 489 + (8153-489)%4092 = 489 + 3572 = 4061 == X, so local = K =
        // 4061 (the maximal inline of a K-branch split), NOT M.
        let s = payload_split(4096, 8153, CellKind::TableLeaf);
        assert!(s.has_overflow);
        assert_eq!(s.local, 4061, "K == X takes the K branch (<=), inlining K not M");
        assert_eq!(s.overflow, 8153 - 4061);
    }

    #[test]
    fn split_never_below_m_and_local_le_x() {
        // Across many payload sizes and page sizes, the inline part obeys the
        // invariants the format guarantees: local <= X, local >= M when spilling,
        // and local + overflow == P.
        for &u in &[512usize, 1024, 4096, 65536] {
            let x_tl = u - 35;
            let m = (u - 12) * 32 / 255 - 23;
            for p in [
                1u64,
                u as u64,
                (u as u64) * 2,
                (u as u64) * 10 + 7,
                (u as u64) * 100 + 3,
            ] {
                let s = payload_split(u, p, CellKind::TableLeaf);
                assert_eq!(s.local + s.overflow, p as usize);
                assert!(s.local <= x_tl);
                if s.has_overflow {
                    assert!(s.local >= m, "local {} >= M {}", s.local, m);
                }
            }
        }
    }

    #[test]
    fn chunk_and_reassemble_roundtrip() {
        let usable = 512usize;
        let cap = overflow_page_capacity(usable); // 508
        // A payload that needs several overflow pages on a table-leaf cell.
        let payload: Vec<u8> = (0..5000u32).map(|i| (i % 251) as u8).collect();
        let (local, overflow) = split_payload_for_write(&payload, usable, CellKind::TableLeaf);
        assert!(!overflow.is_empty());
        assert_eq!(local.len() + overflow.len(), payload.len());

        // Lay the overflow chunks out into pages, linked in order.
        let chunks: Vec<&[u8]> = overflow_chunks(overflow, usable).collect();
        let mut pages = HashMap::new();
        let first_page = 10u32;
        for (i, chunk) in chunks.iter().enumerate() {
            let page_no = first_page + i as u32;
            let next = if i + 1 < chunks.len() { first_page + i as u32 + 1 } else { 0 };
            let mut page = vec![0u8; usable];
            page[0..4].copy_from_slice(&next.to_be_bytes());
            page[4..4 + chunk.len()].copy_from_slice(chunk);
            pages.insert(page_no, page);
        }
        assert!(chunks.len() >= 2, "payload should span multiple overflow pages");
        assert_eq!(chunks.iter().map(|c| c.len()).sum::<usize>(), overflow.len());
        assert!(chunks[..chunks.len() - 1].iter().all(|c| c.len() == cap));

        let src = MapSource(pages);
        let assembled =
            assemble_payload(local, payload.len(), first_page, usable, &src).unwrap();
        assert_eq!(assembled, payload);
    }

    #[test]
    fn inline_only_needs_no_chain() {
        let usable = 4096usize;
        let payload = vec![7u8; 100];
        let (local, overflow) = split_payload_for_write(&payload, usable, CellKind::TableLeaf);
        assert_eq!(local.len(), 100);
        assert!(overflow.is_empty());
        let src = MapSource(HashMap::new());
        // No overflow: first_overflow = 0, chain not walked.
        let assembled = assemble_payload(local, payload.len(), 0, usable, &src).unwrap();
        assert_eq!(assembled, payload);
    }

    #[test]
    fn missing_page_fails_closed() {
        let usable = 512usize;
        let payload = vec![1u8; 5000];
        let (local, _overflow) = split_payload_for_write(&payload, usable, CellKind::TableLeaf);
        // Point at an overflow page that does not exist.
        let src = MapSource(HashMap::new());
        assert!(assemble_payload(local, payload.len(), 99, usable, &src).is_err());
    }

    #[test]
    fn chain_ending_early_fails_closed() {
        let usable = 512usize;
        let cap = overflow_page_capacity(usable);
        // The cell claims 3 overflow pages' worth of bytes, but the single page's
        // next pointer is 0, so the chain ends before the payload is complete.
        let mut page = vec![0u8; usable];
        page[0..4].copy_from_slice(&0u32.to_be_bytes()); // next = last
        let mut pages = HashMap::new();
        pages.insert(7u32, page);
        let src = MapSource(pages);
        let err = assemble_payload(&[], cap * 3, 7, usable, &src);
        assert!(err.is_err(), "a chain that ends before total_len must error");
    }

    #[test]
    fn inline_longer_than_total_is_rejected() {
        let src = MapSource(HashMap::new());
        assert!(assemble_payload(&[0u8; 10], 4, 0, 512, &src).is_err());
    }

    #[test]
    fn write_then_parse_overflow_page_roundtrips() {
        let usable = 512usize;
        let content: Vec<u8> = (0..overflow_page_capacity(usable) as u32).map(|i| i as u8).collect();
        let mut page = vec![0u8; usable];
        write_overflow_page(&mut page, 88, &content, usable).unwrap();
        let parsed = parse_overflow_page(&page, usable).unwrap();
        assert_eq!(parsed.next, 88);
        assert_eq!(parsed.payload, content.as_slice());
        // A last-in-chain page uses next = 0.
        write_overflow_page(&mut page, 0, &content, usable).unwrap();
        assert_eq!(parse_overflow_page(&page, usable).unwrap().next, 0);
    }

    #[test]
    fn write_overflow_page_rejects_oversized_content() {
        let usable = 512usize;
        let too_big = vec![0u8; overflow_page_capacity(usable) + 1];
        let mut page = vec![0u8; usable];
        assert!(write_overflow_page(&mut page, 0, &too_big, usable).is_err());
    }

    #[test]
    fn set_next_and_content_mut_compose() {
        let usable = 512usize;
        let mut page = vec![0u8; usable];
        set_overflow_next(&mut page, 12345).unwrap();
        {
            let region = overflow_content_mut(&mut page, usable).unwrap();
            assert_eq!(region.len(), overflow_page_capacity(usable));
            region[0] = 0xEE;
            region[region.len() - 1] = 0xFF;
        }
        let parsed = parse_overflow_page(&page, usable).unwrap();
        assert_eq!(parsed.next, 12345);
        assert_eq!(parsed.payload[0], 0xEE);
        assert_eq!(*parsed.payload.last().unwrap(), 0xFF);
    }

    #[test]
    fn overflow_write_helpers_fail_closed_on_short_page() {
        assert!(set_overflow_next(&mut [0u8; 3], 1).is_err());
        assert!(overflow_content_mut(&mut [0u8; 100], 512).is_err());
        assert!(write_overflow_page(&mut [0u8; 100], 0, &[], 512).is_err());
    }

    #[test]
    fn write_helpers_build_a_reassemblable_chain() {
        // Build an overflow chain with the write helpers, then reassemble it with
        // assemble_payload — the write and read sides agree end to end.
        let usable = 512usize;
        let payload: Vec<u8> = (0..3000u32).map(|i| (i % 253) as u8).collect();
        let (local, overflow) = split_payload_for_write(&payload, usable, CellKind::TableLeaf);
        let chunks: Vec<&[u8]> = overflow_chunks(overflow, usable).collect();
        let first_page = 20u32;
        let mut pages = HashMap::new();
        for (i, chunk) in chunks.iter().enumerate() {
            let next = if i + 1 < chunks.len() { first_page + i as u32 + 1 } else { 0 };
            let mut page = vec![0u8; usable];
            write_overflow_page(&mut page, next, chunk, usable).unwrap();
            pages.insert(first_page + i as u32, page);
        }
        let src = MapSource(pages);
        let assembled = assemble_payload(local, payload.len(), first_page, usable, &src).unwrap();
        assert_eq!(assembled, payload);
    }
}
