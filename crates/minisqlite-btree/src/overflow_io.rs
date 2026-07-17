//! Overflow-page I/O: the bridge between the pure overflow codec in
//! `minisqlite-fileformat` (spill math, page layout, chain assembly) and the
//! `Pager` that allocates and stores pages. It writes a payload's spilled tail as
//! a forward-linked chain of overflow pages and reads a spilled payload back by
//! walking that chain.
//!
//! fileformat2 §1.7: each overflow page begins with a 4-byte big-endian next-page
//! number (0 on the last page); the remaining `usable - 4` bytes carry content.
//! All of that byte layout and the split arithmetic live in the codec — this
//! module only drives the pager around it, so the table b-tree and the sibling
//! index b-tree share one correct overflow path.

use std::borrow::Cow;

use minisqlite_fileformat::{
    assemble_payload, overflow_chunks, parse_overflow_page, split_payload_for_write,
    write_overflow_page, CellKind, PageSource,
};
use minisqlite_pager::Pager;
use minisqlite_types::{Error, Result};

/// Adapts a `Pager` to the codec's [`PageSource`] so `assemble_payload` can borrow
/// overflow pages straight from the pager cache without copying whole pages through
/// this crate.
pub(crate) struct PagerPageSource<'a>(pub &'a dyn Pager);

impl PageSource for PagerPageSource<'_> {
    fn page(&self, page_no: u32) -> Result<&[u8]> {
        self.0.read_page(page_no)
    }
}

/// Write `payload`'s overflow tail as a chain of overflow pages and report how the
/// b-tree cell should reference it.
///
/// Returns `(local_len, overflow_page)`: `local_len` is how many leading payload
/// bytes stay inline on the b-tree page — the caller slices `&payload[..local_len]`
/// for the cell — and `overflow_page` is the first page of the chain, or `None`
/// when the payload fits inline (in which case nothing is allocated).
///
/// The chain is forward-linked in allocation order with the final page's next
/// pointer set to 0 (§1.7). Overflow pages carry raw payload bytes (never a b-tree
/// header and never page 1), so they are written straight through `write_page`.
pub(crate) fn write_overflow_chain(
    pager: &mut dyn Pager,
    payload: &[u8],
    kind: CellKind,
    usable: usize,
) -> Result<(usize, Option<u32>)> {
    let (inline, tail) = split_payload_for_write(payload, usable, kind);
    if tail.is_empty() {
        return Ok((inline.len(), None));
    }

    // One page per chunk, linked as we go. Each page must name its successor, so we
    // allocate the next page's id *before* writing the current page; the last chunk
    // terminates the chain with next = 0. Allocation ids increase, so the chain runs
    // forward through the file. O(1) bookkeeping — no list of ids or chunks is held.
    let page_size = pager.page_size() as usize;
    let mut chunks = overflow_chunks(tail, usable).peekable();
    let first = pager.allocate_page()?;
    let mut current = first;
    // Reuse one page buffer across the whole chain, re-zeroing it each iteration so
    // every page starts blank — byte-identical to a fresh `vec![0u8; page_size]`, but
    // a large-blob spill costs O(1) allocations instead of one per overflow page.
    let mut page = vec![0u8; page_size];
    while let Some(chunk) = chunks.next() {
        let next = if chunks.peek().is_some() { pager.allocate_page()? } else { 0 };
        page.fill(0);
        write_overflow_page(&mut page, next, chunk, usable)?;
        pager.write_page(current, &page)?;
        current = next;
    }
    Ok((inline.len(), Some(first)))
}

/// Free every page of an overflow chain back to the pager's freelist.
///
/// Walks the forward-linked chain from `first` (fileformat2 §1.7: each overflow
/// page's leading 4 bytes are the big-endian next-page number, 0 on the last page)
/// and releases each page with `Pager::free_page`, so a later `allocate_page` reuses
/// it before the file grows. Runs inside whatever transaction the caller has open
/// (it does not begin/commit); the frees are copy-on-write like every other write.
///
/// `first == 0` is an empty chain and a no-op. This is the counterpart of
/// [`write_overflow_chain`]: a row whose payload spilled is deleted or replaced by
/// freeing the chain its cell pointed at.
///
/// Termination is bounded: a well-formed chain has at most `page_count` pages, so a
/// walk that would free more than that has hit a corrupt next-pointer cycle. Rather
/// than loop forever (a result that never arrives is a wrong result) it stops and
/// returns a format error. The next pointer is copied out of the page before the
/// read borrow is dropped, because `free_page` needs `&mut` on the pager.
///
/// The cap guards *termination*, not freelist integrity: this reclaim assumes a
/// well-formed chain. On a corrupt one it may `free_page` the same id more than once
/// before the cap trips, so it leans on the cap only to fail closed — the caller's
/// transaction then rolls back on the returned `Err`, discarding the staged frees.
/// Detecting a re-visited page would need an O(n) visited-set; the O(1) walk is the
/// deliberate choice (the perf/space budget prefers it, and a corrupt chain is a
/// should-never-happen state whose only requirement is that we not hang).
pub(crate) fn free_overflow_chain(pager: &mut dyn Pager, first: u32, usable: usize) -> Result<()> {
    if first == 0 {
        return Ok(());
    }
    // Upper bound on how many pages a legitimate chain can hold: every page in the
    // database, which strictly exceeds any real chain (page 1 is the header, never an
    // overflow page). Freeing never shrinks the page count, so this bound stays valid
    // for the whole walk. A chain that would exceed it is a cycle — fail closed.
    let max_pages = pager.page_count()?;
    let mut freed: u32 = 0;
    let mut current = first;
    while current != 0 {
        if freed >= max_pages {
            return Err(Error::format(
                "overflow chain longer than the database: corrupt next-pointer cycle",
            ));
        }
        // Read the next pointer, then drop the borrow before the &mut free_page call.
        let next = {
            let page = pager.read_page(current)?;
            parse_overflow_page(page, usable)?.next
        };
        pager.free_page(current)?;
        freed += 1;
        current = next;
    }
    Ok(())
}

/// Read a payload back, transparently assembling any overflow chain.
///
/// An inline row (`overflow_page` is `None`) returns `Cow::Borrowed(local)` with no
/// copy. A spilled row reassembles its full `payload_len` bytes from `local` plus
/// the chain beginning at `overflow_page` and returns them `Cow::Owned`.
pub(crate) fn read_payload<'p>(
    pager: &'p dyn Pager,
    local: &'p [u8],
    payload_len: u64,
    overflow_page: Option<u32>,
    usable: usize,
) -> Result<Cow<'p, [u8]>> {
    match overflow_page {
        None => Ok(Cow::Borrowed(local)),
        Some(first) => {
            let assembled = assemble_payload(
                local,
                payload_len as usize,
                first,
                usable,
                &PagerPageSource(pager),
            )?;
            Ok(Cow::Owned(assembled))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{free_overflow_chain, read_payload, write_overflow_chain};
    use minisqlite_fileformat::CellKind;
    use minisqlite_pager::{MemPager, PageId, Pager};
    use minisqlite_types::{Error, Result};
    use std::borrow::Cow;

    /// Deterministic, index- and seed-dependent bytes so distinct rows differ and a
    /// round-trip can check every byte.
    fn payload(len: usize, seed: u64) -> Vec<u8> {
        (0..len)
            .map(|i| (((i as u64).wrapping_mul(31).wrapping_add(seed)) % 251) as u8)
            .collect()
    }

    /// A pager with page 1 reserved, so overflow pages get realistic ids (>= 2).
    fn pager_with_page1(page_size: u32) -> MemPager {
        let mut p = MemPager::new(page_size);
        p.allocate_page().unwrap(); // page 1 placeholder
        p
    }

    /// A pager whose page 1 is a *formatted* database header (as `init_database`
    /// writes). `free_page` (and thus `free_overflow_chain`) needs that header to
    /// locate the freelist, so the zero-page-1 helper above cannot exercise frees.
    fn formatted_pager(page_size: u32) -> MemPager {
        let mut p = MemPager::new(page_size);
        crate::init_database(&mut p).unwrap();
        p
    }

    #[test]
    fn chain_roundtrips_across_several_pages() {
        // 5000 bytes on a 512-byte page spills across many overflow pages
        // (capacity 508 each), so this exercises a multi-page forward chain.
        let usable = 512usize;
        let mut pager = pager_with_page1(usable as u32);
        let data = payload(5000, 7);

        let before = pager.page_count().unwrap();
        let (local_len, first) =
            write_overflow_chain(&mut pager, &data, CellKind::TableLeaf, usable).unwrap();
        let first = first.expect("a 5000-byte row on a 512 page must overflow");
        assert!(local_len < data.len(), "some bytes must spill");
        assert!(
            pager.page_count().unwrap() > before + 1,
            "a 5000-byte payload needs multiple overflow pages"
        );

        let got = read_payload(&pager, &data[..local_len], data.len() as u64, Some(first), usable)
            .unwrap();
        assert!(matches!(got, Cow::Owned(_)), "a spilled row assembles into an owned buffer");
        assert_eq!(got.as_ref(), data.as_slice(), "overflow chain must round-trip byte-for-byte");
    }

    #[test]
    fn no_overflow_allocates_nothing_and_borrows() {
        let usable = 4096usize;
        let mut pager = pager_with_page1(usable as u32);
        let data = payload(100, 1);

        let before = pager.page_count().unwrap();
        let (local_len, first) =
            write_overflow_chain(&mut pager, &data, CellKind::TableLeaf, usable).unwrap();
        assert_eq!(local_len, data.len(), "a small payload stays fully inline");
        assert_eq!(first, None);
        assert_eq!(pager.page_count().unwrap(), before, "an inline payload allocates no pages");

        let got = read_payload(&pager, &data, data.len() as u64, None, usable).unwrap();
        assert!(matches!(got, Cow::Borrowed(_)), "an inline row is borrowed with no copy");
        assert_eq!(got.as_ref(), data.as_slice());
    }

    #[test]
    fn index_kind_chain_roundtrips() {
        // The sibling index b-tree reuses this path with CellKind::Index; confirm the
        // kind is threaded through the split math and the chain reassembles.
        let usable = 512usize;
        let mut pager = pager_with_page1(usable as u32);
        let data = payload(4000, 99);

        let (local_len, first) =
            write_overflow_chain(&mut pager, &data, CellKind::Index, usable).unwrap();
        let first = first.expect("a 4000-byte index key on a 512 page overflows");
        let got = read_payload(&pager, &data[..local_len], data.len() as u64, Some(first), usable)
            .unwrap();
        assert_eq!(got.as_ref(), data.as_slice());
    }

    #[test]
    fn single_overflow_page_boundary() {
        // A payload that spills onto exactly one overflow page (tail == capacity).
        let usable = 4096usize;
        let mut pager = pager_with_page1(usable as u32);
        let data = payload(5000, 3); // local 908, tail 4092 == capacity -> 1 page

        let before = pager.page_count().unwrap();
        let (local_len, first) =
            write_overflow_chain(&mut pager, &data, CellKind::TableLeaf, usable).unwrap();
        let first = first.expect("5000 > 4096 must overflow");
        assert_eq!(pager.page_count().unwrap(), before + 1, "exactly one overflow page");

        let got = read_payload(&pager, &data[..local_len], data.len() as u64, Some(first), usable)
            .unwrap();
        assert_eq!(got.as_ref(), data.as_slice());
    }

    #[test]
    fn free_overflow_chain_returns_pages_to_the_freelist() {
        // Build a multi-page chain, free the whole chain, then build a same-size
        // chain again: every freed page must be reused so the file does not grow by
        // a second chain. This is the leak fix's core invariant, at the codec level.
        let usable = 512usize;
        let mut pager = formatted_pager(usable as u32);
        let data = payload(6000, 5); // ~12 overflow pages on a 512-byte page

        let (_, first) =
            write_overflow_chain(&mut pager, &data, CellKind::TableLeaf, usable).unwrap();
        let first = first.expect("a 6000-byte payload on a 512 page overflows");
        let after_build = pager.page_count().unwrap();
        let chain_len = after_build - 1; // pages past page 1 are the chain
        assert!(chain_len >= 3, "the payload must span several overflow pages (got {chain_len})");

        // Free the chain and rebuild an identical one inside one transaction.
        pager.begin().unwrap();
        free_overflow_chain(&mut pager, first, usable).unwrap();
        let (local_len2, second) =
            write_overflow_chain(&mut pager, &data, CellKind::TableLeaf, usable).unwrap();
        let second = second.expect("the second chain also overflows");
        pager.commit().unwrap();

        assert_eq!(
            pager.page_count().unwrap(),
            after_build,
            "freeing then rebuilding a same-size chain reuses every page (no growth)"
        );
        // The reused chain still round-trips byte-for-byte.
        let got =
            read_payload(&pager, &data[..local_len2], data.len() as u64, Some(second), usable)
                .unwrap();
        assert_eq!(got.as_ref(), data.as_slice(), "the rebuilt chain reads back correctly");
    }

    #[test]
    fn free_overflow_chain_empty_is_a_noop() {
        // first == 0 means "no chain" (an inline row); freeing it must touch nothing.
        let mut pager = formatted_pager(512);
        let before = pager.page_count().unwrap();
        free_overflow_chain(&mut pager, 0, 512).unwrap();
        assert_eq!(pager.page_count().unwrap(), before, "an empty chain frees nothing");
    }

    #[test]
    fn free_overflow_chain_bounds_a_corrupt_cycle() {
        // A pager whose every page points to itself/another page and whose free_page
        // never rewrites the page: a naive walk would loop forever. The page_count cap
        // must stop it and surface the corruption as an error rather than hang. This
        // proves termination independently of how a real freelist happens to rewrite
        // freed pages.
        struct CyclePager {
            page: Vec<u8>,
            count: PageId,
        }
        impl Pager for CyclePager {
            fn read_page(&self, _id: PageId) -> Result<&[u8]> {
                Ok(&self.page)
            }
            fn page_count(&self) -> Result<PageId> {
                Ok(self.count)
            }
            fn page_size(&self) -> u32 {
                self.page.len() as u32
            }
            fn begin(&mut self) -> Result<()> {
                Ok(())
            }
            fn write_page(&mut self, _id: PageId, _bytes: &[u8]) -> Result<()> {
                Ok(())
            }
            fn allocate_page(&mut self) -> Result<PageId> {
                Ok(0)
            }
            fn free_page(&mut self, _id: PageId) -> Result<()> {
                Ok(()) // no-op: never breaks the cycle, so only the cap can
            }
            fn commit(&mut self) -> Result<()> {
                Ok(())
            }
            fn rollback(&mut self) -> Result<()> {
                Ok(())
            }
        }
        let usable = 512usize;
        // next-pointer (first 4 bytes) points to page 2; read_page returns this same
        // page for every id, so the walk keeps seeing next == 2 forever.
        let mut page = vec![0u8; usable];
        page[0..4].copy_from_slice(&2u32.to_be_bytes());
        let mut pager = CyclePager { page, count: 4 };

        let err = free_overflow_chain(&mut pager, 2, usable).unwrap_err();
        assert!(matches!(err, Error::Format(_)), "a cycle surfaces as a format error, not a hang");
    }

    #[test]
    fn final_overflow_page_zero_pads_past_the_tail() {
        // The chain reuses one page buffer across chunks. Every non-final chunk fills the
        // whole content region, but the final chunk is short — and `write_overflow_page`
        // copies only `content.len()` bytes without zeroing the rest — so WITHOUT the
        // per-iteration `page.fill(0)` the final page's padding would carry the previous
        // full chunk's tail bytes. `assemble_payload` never reads past the real length, so
        // a read round-trip cannot observe this; only a direct page inspection can. Clean
        // zero padding matters for `.db` bytes and for not leaking prior page contents.
        use minisqlite_fileformat::parse_overflow_page;
        let usable = 512usize;
        let capacity = usable - 4;
        let mut pager = pager_with_page1(usable as u32);
        let data = payload(1500, 5);
        let (local_len, first) =
            write_overflow_chain(&mut pager, &data, CellKind::TableLeaf, usable).unwrap();
        let first = first.expect("1500 bytes on a 512 page overflows");
        let tail_len = data.len() - local_len;
        assert!(tail_len > capacity, "setup: need >1 overflow page so a full chunk precedes the last");
        let last_chunk_len = match tail_len % capacity {
            0 => capacity,
            r => r,
        };
        assert!(last_chunk_len < capacity, "setup: final chunk must be partial so padding exists");

        // Walk to the final page (next == 0) and assert its content past the tail is zero.
        let mut page_no = first;
        loop {
            let raw = pager.read_page(page_no).unwrap();
            let ov = parse_overflow_page(raw, usable).unwrap();
            if ov.next == 0 {
                assert!(
                    ov.payload[last_chunk_len..].iter().all(|&b| b == 0),
                    "final overflow page must zero-pad the region after the {last_chunk_len}-byte tail"
                );
                break;
            }
            page_no = ov.next;
        }
    }
}
