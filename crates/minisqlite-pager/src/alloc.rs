//! Freelist allocation policy (fileformat2 §1.5), written once over the generic
//! copy-on-write layer so both the in-memory and disk-backed pagers share it.
//!
//! The freelist is a singly linked list of *trunk* pages rooted at the page-1
//! header: offset 32 holds the first trunk page number (0 if empty) and offset 36
//! holds the total number of freelist pages (trunks + leaves, matching SQLite's
//! definition). Each trunk page lists zero or more *leaf* (free) page numbers plus
//! a pointer to the next trunk; the byte layout lives in
//! `minisqlite_fileformat::freelist`, so this module only decides *policy*: which
//! page to hand back, and where a freed page goes.
//!
//! ## Why the header is parsed, not read raw
//! The freelist head/count are read by parsing page 1 as a full [`DatabaseHeader`]
//! (magic-checked). An unformatted page store — one whose page 1 is not a valid
//! SQLite header (a freshly grown page, or arbitrary bytes) — has NO freelist by
//! definition: `allocate_page` simply grows, and `free_page` fails closed because
//! there is nowhere to record the freelist. Reading offsets 32/36 as raw bytes
//! instead would misread arbitrary page-1 content as a bogus freelist head, so the
//! parse (and its magic check) is the load-bearing discriminator. In a real
//! database page 1 is always a formatted header before any page is freed.
//!
//! ## Cost
//! Allocate and free touch only the page-1 header and a single trunk page — O(1)
//! amortized, never a scan of the whole freelist.

use minisqlite_fileformat::freelist::{parse_trunk, trunk_leaf_capacity, write_trunk};
use minisqlite_fileformat::{DatabaseHeader, HEADER_SIZE};
use minisqlite_types::{Error, Result};

use crate::PageId;
use crate::cow::Cow;
use crate::store::PageStore;

/// Allocate a page: reuse one from the freelist when the database has a formatted
/// header with a non-empty freelist, otherwise grow the file by one page. A reused
/// page is handed back zero-filled, preserving the "fresh zero page" contract that
/// callers rely on. Runs inside the caller's transaction.
pub(crate) fn alloc_in_txn<S: PageStore>(cow: &mut Cow<S>) -> Result<PageId> {
    let count = cow.page_count()?;
    if count == 0 {
        // Empty database: no page 1, header, or freelist to consult yet.
        return cow.grow_one();
    }
    let header = match read_header(cow)? {
        Some(h) => h,
        // Unformatted page store: no freelist, so grow.
        None => return cow.grow_one(),
    };
    if header.first_freelist_trunk == 0 {
        // Formatted, but the freelist is empty: grow (skipping ptrmap positions in an
        // auto_vacuum database — see `grow_data_page`).
        return grow_data_page(cow, &header);
    }
    // A reused freelist page is always a former DATA page, never a ptrmap position (a
    // ptrmap page is never freed), so the reuse path needs no ptrmap skip.
    reuse_from_freelist(cow, header)
}

/// Grow the file by one DATA page, skipping any pointer-map (ptrmap) page positions in
/// an auto_vacuum database (offset 52 non-zero, §1.8). A ptrmap position is reserved
/// with a zero-filled placeholder — its real 5-byte back-pointer entries are written at
/// commit by [`crate::ptrmap_build::finalize`] — and is never handed back to the caller,
/// so no b-tree pointer ever targets a ptrmap page. For a plain (non-vacuum) database
/// this is exactly `grow_one`, keeping that path byte-identical.
fn grow_data_page<S: PageStore>(cow: &mut Cow<S>, header: &DatabaseHeader) -> Result<PageId> {
    if !header.uses_ptrmap() {
        return cow.grow_one();
    }
    let usable = header.usable_size();
    loop {
        let next = cow
            .page_count()?
            .checked_add(1)
            .ok_or_else(|| Error::Io("page count would overflow PageId (u32)".into()))?;
        if minisqlite_fileformat::is_ptrmap_page(usable, next) {
            // Reserve the ptrmap page position; its entries are filled at commit.
            let reserved = cow.grow_one()?;
            debug_assert_eq!(reserved, next, "grow_one returns the next sequential page id");
        } else {
            return cow.grow_one();
        }
    }
}

/// Release a page to the freelist so a later allocation reuses it before the file
/// grows. Rejects page 0, page 1 (it holds the header), and any id past the current
/// page count. Requires a formatted page-1 header (the freelist lives there); a
/// pager with no valid header fails closed. Runs inside the caller's transaction.
pub(crate) fn free_in_txn<S: PageStore>(cow: &mut Cow<S>, id: PageId) -> Result<()> {
    let count = cow.page_count()?;
    if id == 0 || id == 1 || id > count {
        return Err(Error::Io(format!(
            "free_page: id {id} is not freeable (page_count = {count}; page 1 holds the header)"
        )));
    }
    let mut header = read_header(cow)?.ok_or_else(|| {
        Error::Format("free_page: page 1 is not a valid database header".into())
    })?;
    let usable = usable_size_for(cow, &header)?;
    // A pointer-map page must NEVER be freed: it holds no data, is never handed out by
    // the allocator (`grow_data_page` skips its position), and a §1.8 reader computes
    // its position rather than following a pointer — so putting one on the freelist
    // would corrupt the auto_vacuum structure. The invariant holds today (nothing frees
    // one); this guard fails closed so a future compaction bug that tried to relocate a
    // ptrmap page declines rather than corrupts. Costs one arithmetic check, and only
    // in an auto_vacuum database.
    if header.uses_ptrmap() && minisqlite_fileformat::is_ptrmap_page(usable, id) {
        return Err(Error::Io(format!(
            "free_page: id {id} is a pointer-map page and must never be freed"
        )));
    }
    let page_size = cow.page_size() as usize;
    let first_trunk = header.first_freelist_trunk;

    if first_trunk == 0 {
        // No trunk yet: the freed page becomes the first trunk (empty, next = 0).
        write_new_trunk(cow, id, 0, page_size, usable)?;
        header.first_freelist_trunk = id;
        header.freelist_count = header.freelist_count.saturating_add(1);
        return write_header(cow, &header);
    }

    // Add the freed page as a leaf on the first trunk if it has room; otherwise the
    // freed page becomes a new first trunk chained to the old one.
    let (next, mut leaves) = read_trunk(cow, first_trunk, usable)?;
    if leaves.len() < trunk_leaf_capacity(usable) {
        leaves.push(id);
        write_existing_trunk(cow, first_trunk, next, &leaves, page_size, usable)?;
        header.freelist_count = header.freelist_count.saturating_add(1);
        write_header(cow, &header)
    } else {
        write_new_trunk(cow, id, first_trunk, page_size, usable)?;
        header.first_freelist_trunk = id;
        header.freelist_count = header.freelist_count.saturating_add(1);
        write_header(cow, &header)
    }
}

/// Reuse a page from a non-empty freelist. If the first trunk lists any leaves,
/// hand back one leaf (shrinking the trunk); otherwise the trunk itself has no more
/// use, so reuse the trunk page and advance the header to the next trunk. The
/// returned page is zero-filled for the caller.
fn reuse_from_freelist<S: PageStore>(cow: &mut Cow<S>, mut header: DatabaseHeader) -> Result<PageId> {
    let usable = usable_size_for(cow, &header)?;
    let page_size = cow.page_size() as usize;
    let first_trunk = header.first_freelist_trunk;
    let (next, mut leaves) = read_trunk(cow, first_trunk, usable)?;

    if let Some(leaf) = leaves.pop() {
        // Rewrite the trunk from a fresh zero page so the popped slot leaves no
        // stale bytes, drop one from the count, then zero the reused leaf.
        write_existing_trunk(cow, first_trunk, next, &leaves, page_size, usable)?;
        header.freelist_count = header.freelist_count.saturating_sub(1);
        write_header(cow, &header)?;
        zero_fill(cow, leaf, page_size)?;
        Ok(leaf)
    } else {
        // The trunk has no leaves: reuse the trunk page itself and point the header
        // at the next trunk in the chain.
        header.first_freelist_trunk = next;
        header.freelist_count = header.freelist_count.saturating_sub(1);
        write_header(cow, &header)?;
        zero_fill(cow, first_trunk, page_size)?;
        Ok(first_trunk)
    }
}

/// Parse page 1 as a database header. Returns `None` (not an error) when page 1 is
/// not a valid SQLite header, which the callers treat as "no freelist".
fn read_header<S: PageStore>(cow: &Cow<S>) -> Result<Option<DatabaseHeader>> {
    let page1 = cow.read_page(1)?;
    let mut buf = [0u8; HEADER_SIZE];
    // Page 1 is at least 512 bytes (the minimum page size), so the 100-byte header
    // slice is always in bounds.
    buf.copy_from_slice(&page1[..HEADER_SIZE]);
    Ok(DatabaseHeader::read(&buf).ok())
}

/// Write the header back into page 1, preserving the page's bytes beyond the
/// 100-byte header (the b-tree content of page 1). Copies the page out first so no
/// read borrow is held across the write.
fn write_header<S: PageStore>(cow: &mut Cow<S>, header: &DatabaseHeader) -> Result<()> {
    let mut page1 = cow.read_page(1)?.to_vec();
    let mut head = [0u8; HEADER_SIZE];
    header.write(&mut head);
    page1[..HEADER_SIZE].copy_from_slice(&head);
    cow.stage_write(1, &page1)
}

/// The usable page size from the header, validated against the pager's page size.
/// A header whose page size disagrees with the pager's is inconsistent (a corrupt
/// or mis-opened database) and fails closed rather than producing a trunk sized for
/// the wrong page.
fn usable_size_for<S: PageStore>(cow: &Cow<S>, header: &DatabaseHeader) -> Result<usize> {
    if header.page_size != cow.page_size() {
        return Err(Error::Format(format!(
            "page-1 header page_size {} disagrees with pager page_size {}",
            header.page_size,
            cow.page_size()
        )));
    }
    Ok(header.usable_size())
}

/// Read and parse a trunk page into `(next_trunk, leaves)`, copying out so no read
/// borrow is held across a later write.
fn read_trunk<S: PageStore>(cow: &Cow<S>, trunk: PageId, usable: usize) -> Result<(PageId, Vec<PageId>)> {
    let page = cow.read_page(trunk)?;
    parse_trunk(page, usable)
}

/// Stage a freshly-built trunk page at `id` with the given `next` pointer and no
/// leaves — used when a freed page becomes a trunk (either the first trunk of a new
/// freelist, or a new head chained to a full trunk).
fn write_new_trunk<S: PageStore>(
    cow: &mut Cow<S>,
    id: PageId,
    next: PageId,
    page_size: usize,
    usable: usize,
) -> Result<()> {
    let mut buf = vec![0u8; page_size];
    write_trunk(&mut buf, next, &[], usable)?;
    cow.stage_write(id, &buf)
}

/// Stage a rewritten existing trunk page from a fresh zero buffer, so a shrunk
/// leaf list leaves no stale slots behind.
fn write_existing_trunk<S: PageStore>(
    cow: &mut Cow<S>,
    trunk: PageId,
    next: PageId,
    leaves: &[PageId],
    page_size: usize,
    usable: usize,
) -> Result<()> {
    let mut buf = vec![0u8; page_size];
    write_trunk(&mut buf, next, leaves, usable)?;
    cow.stage_write(trunk, &buf)
}

/// Zero-fill a page (the reuse contract: an allocated page is handed back zeroed so
/// the caller can overwrite it).
fn zero_fill<S: PageStore>(cow: &mut Cow<S>, id: PageId, page_size: usize) -> Result<()> {
    let zero = vec![0u8; page_size];
    cow.stage_write(id, &zero)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use minisqlite_fileformat::freelist::{parse_trunk, trunk_leaf_capacity};
    use minisqlite_fileformat::{DatabaseHeader, HEADER_SIZE};

    use crate::{MemPager, PageId, Pager};

    const PS: u32 = 512;

    /// Build a pager with a valid page-1 header and `extra` additional pages, so
    /// pages `1..=extra+1` all exist and the freelist is empty. Page 1 is a real
    /// SQLite header (as `init_database` would write), which the freelist requires.
    fn formatted_pager(page_size: u32, extra: u32) -> MemPager {
        let mut p = MemPager::new(page_size);
        p.begin().unwrap();
        let one = p.allocate_page().unwrap();
        assert_eq!(one, 1);
        let mut hdr = DatabaseHeader::default();
        hdr.page_size = page_size;
        let mut page1 = vec![0u8; page_size as usize];
        page1[..HEADER_SIZE].copy_from_slice(&hdr.to_bytes());
        p.write_page(1, &page1).unwrap();
        for _ in 0..extra {
            p.allocate_page().unwrap();
        }
        p.commit().unwrap();
        p
    }

    fn page1_header(p: &MemPager) -> DatabaseHeader {
        let page1 = p.read_page(1).unwrap();
        let mut buf = [0u8; HEADER_SIZE];
        buf.copy_from_slice(&page1[..HEADER_SIZE]);
        DatabaseHeader::read(&buf).unwrap()
    }

    /// Walk the trunk chain and collect every freelist page (trunks + leaves).
    fn freelist_pages(p: &MemPager) -> BTreeSet<PageId> {
        let hdr = page1_header(p);
        let usable = hdr.usable_size();
        let mut pages = BTreeSet::new();
        let mut trunk = hdr.first_freelist_trunk;
        while trunk != 0 {
            pages.insert(trunk);
            let page = p.read_page(trunk).unwrap();
            let (next, leaves) = parse_trunk(page, usable).unwrap();
            for leaf in leaves {
                pages.insert(leaf);
            }
            trunk = next;
        }
        pages
    }

    #[test]
    fn free_then_allocate_reuses_before_growing() {
        let mut p = formatted_pager(PS, 4); // pages 1..=5
        assert_eq!(p.page_count().unwrap(), 5);
        p.begin().unwrap();
        p.free_page(3).unwrap();
        let reused = p.allocate_page().unwrap();
        assert_eq!(reused, 3, "the freed page must be reused before growing");
        assert_eq!(p.page_count().unwrap(), 5, "reuse must not grow the file");
        p.commit().unwrap();
    }

    #[test]
    fn free_two_then_allocate_two_then_growth_resumes() {
        let mut p = formatted_pager(PS, 4); // pages 1..=5
        p.begin().unwrap();
        p.free_page(3).unwrap();
        p.free_page(4).unwrap();
        let a = p.allocate_page().unwrap();
        let b = p.allocate_page().unwrap();
        let mut got = [a, b];
        got.sort_unstable();
        assert_eq!(got, [3, 4], "both freed ids reused in some order");
        assert_eq!(p.page_count().unwrap(), 5);
        // Freelist now empty → growth resumes at the next id.
        let c = p.allocate_page().unwrap();
        assert_eq!(c, 6);
        assert_eq!(p.page_count().unwrap(), 6);
        p.commit().unwrap();
    }

    #[test]
    fn freelist_survives_commit() {
        let mut p = formatted_pager(PS, 4); // pages 1..=5
        p.begin().unwrap();
        p.free_page(2).unwrap();
        p.free_page(5).unwrap();
        p.commit().unwrap();
        // A later transaction reuses pages freed in an earlier committed one.
        p.begin().unwrap();
        let a = p.allocate_page().unwrap();
        let b = p.allocate_page().unwrap();
        let mut got = [a, b];
        got.sort_unstable();
        assert_eq!(got, [2, 5]);
        assert_eq!(p.page_count().unwrap(), 5);
        p.commit().unwrap();
    }

    #[test]
    fn rollback_undoes_frees_and_allocations() {
        let mut p = formatted_pager(PS, 4); // pages 1..=5
        p.begin().unwrap();
        p.free_page(3).unwrap();
        let reused = p.allocate_page().unwrap();
        assert_eq!(reused, 3);
        p.rollback().unwrap();

        // Committed state is untouched: no freelist, count still 5.
        let hdr = page1_header(&p);
        assert_eq!(hdr.first_freelist_trunk, 0, "rolled-back free left no trunk");
        assert_eq!(hdr.freelist_count, 0);
        assert_eq!(p.page_count().unwrap(), 5);

        // A fresh allocation grows (page 3 is NOT reusable after the rollback).
        p.begin().unwrap();
        let next = p.allocate_page().unwrap();
        assert_eq!(next, 6, "a rolled-back free must not leave the page reusable");
        p.commit().unwrap();
    }

    #[test]
    fn header_and_trunk_reflect_freed_pages() {
        let mut p = formatted_pager(PS, 5); // pages 1..=6
        p.begin().unwrap();
        p.free_page(3).unwrap();
        p.free_page(5).unwrap();
        p.free_page(6).unwrap();
        p.commit().unwrap();

        let hdr = page1_header(&p);
        assert_ne!(hdr.first_freelist_trunk, 0, "the header points at a trunk");
        assert_eq!(hdr.freelist_count, 3, "count = trunk + leaves = freed pages");
        assert_eq!(
            freelist_pages(&p),
            [3, 5, 6].into_iter().collect(),
            "the freed ids are exactly the freelist pages"
        );
    }

    #[test]
    fn freelist_spans_multiple_trunks_and_count_includes_trunks() {
        // Free more pages than one trunk can hold so a second trunk is created, and
        // confirm the header count includes trunk pages (SQLite's definition).
        let usable = PS as usize; // reserved 0
        let capacity = trunk_leaf_capacity(usable) as u32;
        let to_free = capacity + 2; // fills the first trunk, then spills to a new one
        let extra = to_free + 3;
        let mut p = formatted_pager(PS, extra); // pages 1..=(extra+1)
        assert!(p.page_count().unwrap() >= 1 + to_free);

        let freed: Vec<PageId> = (2..2 + to_free).collect();
        p.begin().unwrap();
        for &id in &freed {
            p.free_page(id).unwrap();
        }
        p.commit().unwrap();

        let hdr = page1_header(&p);
        assert_eq!(hdr.freelist_count, to_free, "count = trunks + leaves = freed pages");

        // Two trunks in the chain, and every freed id appears exactly once.
        let mut trunks = 0u32;
        let mut all = BTreeSet::new();
        let mut trunk = hdr.first_freelist_trunk;
        while trunk != 0 {
            trunks += 1;
            assert!(all.insert(trunk), "no page appears twice on the freelist");
            let page = p.read_page(trunk).unwrap();
            let (next, leaves) = parse_trunk(page, usable).unwrap();
            for leaf in leaves {
                assert!(all.insert(leaf), "no page appears twice on the freelist");
            }
            trunk = next;
        }
        assert_eq!(trunks, 2, "capacity+2 frees span two trunk pages");
        assert_eq!(all, freed.iter().copied().collect());
    }

    #[test]
    fn reallocate_across_full_trunk_boundary_reuses_all_freed_pages() {
        // The inverse of the multi-trunk test: after building a two-trunk freelist,
        // draining it by allocation must hand back every freed page and no other,
        // then resume growth.
        let usable = PS as usize;
        let capacity = trunk_leaf_capacity(usable) as u32;
        let to_free = capacity + 2;
        let extra = to_free + 3;
        let mut p = formatted_pager(PS, extra);
        let grown_count = p.page_count().unwrap();

        let freed: BTreeSet<PageId> = (2..2 + to_free).collect();
        p.begin().unwrap();
        for &id in &freed {
            p.free_page(id).unwrap();
        }
        // Drain the whole freelist by re-allocating exactly `to_free` pages.
        let mut reused = BTreeSet::new();
        for _ in 0..to_free {
            let id = p.allocate_page().unwrap();
            assert!(reused.insert(id), "each reused id is distinct");
        }
        assert_eq!(reused, freed, "draining the freelist returns exactly the freed pages");
        assert_eq!(p.page_count().unwrap(), grown_count, "reuse never grew the file");
        // Freelist empty now: the next allocation grows.
        let after = p.allocate_page().unwrap();
        assert_eq!(after, grown_count + 1);
        p.commit().unwrap();
        assert_eq!(page1_header(&p).freelist_count, 0);
    }

    #[test]
    fn free_page_rejects_invalid_ids_without_panicking() {
        let mut p = formatted_pager(PS, 2); // pages 1..=3
        p.begin().unwrap();
        assert!(p.free_page(0).is_err(), "id 0 is never valid");
        assert!(p.free_page(1).is_err(), "page 1 holds the header");
        assert!(p.free_page(4).is_err(), "id past page_count");
        // A valid free still works afterward — the rejects did not corrupt state.
        assert!(p.free_page(2).is_ok());
        p.commit().unwrap();
    }

    #[test]
    fn free_and_allocate_outside_a_txn_autocommit() {
        let mut p = formatted_pager(PS, 3); // pages 1..=4
        // Free with no active transaction: auto-commits.
        p.free_page(3).unwrap();
        assert_eq!(p.page_count().unwrap(), 4);
        let hdr = page1_header(&p);
        assert_eq!(hdr.first_freelist_trunk, 3);
        assert_eq!(hdr.freelist_count, 1);
        // Allocate with no active transaction: reuses page 3 and auto-commits.
        let reused = p.allocate_page().unwrap();
        assert_eq!(reused, 3);
        let hdr = page1_header(&p);
        assert_eq!(hdr.first_freelist_trunk, 0);
        assert_eq!(hdr.freelist_count, 0);
        assert_eq!(p.page_count().unwrap(), 4);
    }

    #[test]
    fn free_requires_a_formatted_header() {
        // Page 1 left as zeros is not a valid SQLite header, so there is nowhere to
        // record the freelist: free must fail closed rather than corrupt page 1.
        let mut p = MemPager::new(PS);
        p.begin().unwrap();
        p.allocate_page().unwrap(); // page 1, all zeros
        p.allocate_page().unwrap(); // page 2
        assert!(p.free_page(2).is_err(), "freeing without a header fails closed");
        p.commit().unwrap();
    }

    #[test]
    fn free_rejects_header_pager_page_size_mismatch() {
        // Defense-in-depth (`usable_size_for`): if page 1's header claims a different
        // page size than the pager was opened with — a corrupt or mis-opened database
        // — the freelist code must fail closed rather than size a trunk for the wrong
        // page. The header below is otherwise valid (magic + a legal 1024 page size),
        // so only the header-vs-pager disagreement can reject it.
        let mut p = MemPager::new(PS); // pager page size 512
        p.begin().unwrap();
        p.allocate_page().unwrap(); // page 1
        p.allocate_page().unwrap(); // page 2 (a freeable page)
        let mut hdr = DatabaseHeader::default();
        hdr.page_size = 1024; // lies: != the pager's 512
        let mut page1 = vec![0u8; PS as usize];
        page1[..HEADER_SIZE].copy_from_slice(&hdr.to_bytes());
        p.write_page(1, &page1).unwrap();
        assert!(p.free_page(2).is_err(), "header/pager page_size mismatch must fail closed");
        p.rollback().unwrap();
    }

    #[test]
    fn allocate_does_not_treat_garbage_page1_as_a_freelist() {
        // Regression guard for the core design decision: the freelist head is found
        // by PARSING a real header, never by reading offset 32 raw. Page 1 filled
        // with 0x22 would look like a huge freelist head (0x22222222) under a raw
        // read; a real allocate must instead see "not a header" and grow.
        let mut p = MemPager::new(PS);
        p.begin().unwrap();
        let one = p.allocate_page().unwrap();
        p.write_page(one, &vec![0x22u8; PS as usize]).unwrap();
        let two = p.allocate_page().unwrap();
        assert_eq!(two, 2, "garbage page 1 must not be mistaken for a freelist head");
        p.commit().unwrap();
    }

    #[test]
    fn trunk_reused_page_is_handed_back_zero_filled() {
        // The FIRST free routes through the `first_trunk == 0` branch, so page 3
        // becomes a trunk (write_new_trunk with next=0, L=0). Re-allocation then
        // takes the trunk-consumption path. This pins reuse-of-the-trunk-page; the
        // LEAF path (the dominant one) is pinned separately below, and a chained
        // trunk whose `next` pointer must be scrubbed by `reuse_from_freelist`'s
        // trunk-consumption path is pinned by
        // `trunk_reused_from_chain_scrubs_stale_next_pointer`.
        let mut p = formatted_pager(PS, 3); // pages 1..=4
        p.begin().unwrap();
        p.write_page(3, &vec![0x7Eu8; PS as usize]).unwrap();
        p.free_page(3).unwrap();
        let reused = p.allocate_page().unwrap();
        assert_eq!(reused, 3);
        assert_eq!(p.read_page(3).unwrap(), &vec![0u8; PS as usize][..], "reused page is zeroed");
        p.commit().unwrap();
    }

    #[test]
    fn leaf_reused_page_is_handed_back_zero_filled() {
        // The DOMINANT reuse path, and the one the trunk-only test above cannot
        // reach: a page freed as a LEAF is never touched by the free (its trunk
        // just records the id), so allocation's explicit zero-fill is the ONLY
        // thing that clears its pre-free bytes. Free >=2 pages so the checked page
        // comes back as a leaf on an existing trunk rather than becoming a trunk
        // itself. Removing the leaf-path `zero_fill` in `reuse_from_freelist`
        // makes this fail (the reused page still reads 0x7E).
        let mut p = formatted_pager(PS, 4); // pages 1..=5
        p.begin().unwrap();
        p.write_page(4, &vec![0x7Eu8; PS as usize]).unwrap(); // dirty the future leaf
        p.free_page(3).unwrap(); // 3 -> first trunk (empty)
        p.free_page(4).unwrap(); // 4 -> LEAF on trunk 3; page-4 bytes left as 0x7E
        let reused = p.allocate_page().unwrap(); // pops leaf 4 before consuming trunk 3
        assert_eq!(reused, 4, "the leaf is reused before the trunk page");
        assert_eq!(
            p.read_page(4).unwrap(),
            &vec![0u8; PS as usize][..],
            "a leaf-reused page must be handed back zero-filled, not with its pre-free bytes"
        );
        p.commit().unwrap();
    }

    #[test]
    fn trunk_reused_from_chain_scrubs_stale_next_pointer() {
        // Pins the trunk-consumption `zero_fill` for a CHAINED trunk (unlike the
        // first-trunk case, whose page is coincidentally all-zero because next=0).
        // Fill trunk T1 to capacity, then free one more so a new head trunk T2 is
        // created with next=T1 (non-zero bytes at offset 0). Draining then consumes
        // T2 first; without the zero-fill it would be handed back still carrying the
        // `next=T1` pointer in its first bytes.
        let usable = PS as usize; // reserved 0 for the default header
        let capacity = trunk_leaf_capacity(usable) as u32;
        // id 2 becomes T1 and the next `capacity` frees fill it; that is capacity+1
        // frees to fill T1 exactly, so capacity+2 is the first count that spills a
        // new head trunk T2 chained to T1.
        let to_free = capacity + 2;
        let extra = to_free + 2;
        let mut p = formatted_pager(PS, extra);

        // Free ids 2..=last: id 2 -> T1, next `capacity` ids -> leaves of T1,
        // the last id -> new head trunk T2 chained to T1.
        let last = 2 + to_free - 1;
        p.begin().unwrap();
        for id in 2..=last {
            p.free_page(id).unwrap();
        }
        // The head trunk is the last-freed id, and it carries next = T1 (= 2).
        let head_trunk = last;
        {
            let page = p.read_page(head_trunk).unwrap();
            let (next, leaves) = parse_trunk(page, usable).unwrap();
            assert_eq!(next, 2, "head trunk chains to T1");
            assert!(leaves.is_empty(), "the spilled head trunk has no leaves yet");
        }
        // First allocation consumes the head trunk (it has no leaves) and hands the
        // page back; it must be zero-filled despite having carried a next pointer.
        let reused = p.allocate_page().unwrap();
        assert_eq!(reused, head_trunk, "the empty head trunk page is reused");
        assert_eq!(
            p.read_page(head_trunk).unwrap(),
            &vec![0u8; PS as usize][..],
            "a reused trunk page must be scrubbed of its stale next pointer"
        );
        p.commit().unwrap();
    }

    #[test]
    fn rollback_undoes_a_free_without_reallocation() {
        // Distinguishes rollback from commit for the freelist: free WITHOUT
        // re-allocating in the same txn (so the freelist does not net back to
        // empty), roll back, and confirm the free is gone. A `rollback == commit`
        // mutant fails here — under commit, page 3 would be on the freelist and the
        // next allocate would reuse it (returning 3, not a grown 6).
        let mut p = formatted_pager(PS, 4); // pages 1..=5
        p.begin().unwrap();
        p.free_page(3).unwrap();
        // Inside the txn the freelist reflects the free.
        assert_ne!(page1_header(&p).first_freelist_trunk, 0, "free is visible mid-txn");
        p.rollback().unwrap();

        // Committed state: the free was undone, so there is no freelist at all.
        let hdr = page1_header(&p);
        assert_eq!(hdr.first_freelist_trunk, 0, "rolled-back free left no trunk");
        assert_eq!(hdr.freelist_count, 0, "rolled-back free left no freelist count");
        assert_eq!(p.page_count().unwrap(), 5);

        // A fresh allocation must GROW — the rolled-back page 3 is not reusable.
        p.begin().unwrap();
        let next = p.allocate_page().unwrap();
        assert_eq!(next, 6, "a rolled-back free must not leave the page reusable");
        p.commit().unwrap();
    }

    #[test]
    fn freelist_reuse_and_multitrunk_at_a_larger_page_size() {
        // Every other freelist test uses 512-byte pages; this exercises the
        // usable-size / trunk-capacity math at a different page size so an error in
        // `usable_size`/`trunk_leaf_capacity` scaling would be caught. 4096-byte
        // pages give a much larger leaf capacity, so spanning two trunks frees
        // proportionally more pages.
        const BIG: u32 = 4096;
        let usable = BIG as usize; // default header => reserved 0
        let capacity = trunk_leaf_capacity(usable) as u32;
        let to_free = capacity + 2; // fill one trunk, spill to a second
        let extra = to_free + 3;
        let mut p = formatted_pager(BIG, extra);

        let freed: BTreeSet<PageId> = (2..2 + to_free).collect();
        p.begin().unwrap();
        for &id in &freed {
            p.free_page(id).unwrap();
        }
        p.commit().unwrap();

        // Header count includes trunks + leaves, and the freed set is exactly the
        // freelist pages across a two-trunk chain.
        let hdr = page1_header(&p);
        assert_eq!(hdr.freelist_count, to_free, "count = trunks + leaves at 4096-byte pages");
        assert_eq!(freelist_pages(&p), freed, "freed ids are exactly the freelist pages");

        // Draining reuses every freed page before the file grows again.
        p.begin().unwrap();
        let mut reused = BTreeSet::new();
        for _ in 0..to_free {
            assert!(reused.insert(p.allocate_page().unwrap()), "each reused id is distinct");
        }
        assert_eq!(reused, freed, "draining returns exactly the freed pages at 4096-byte pages");
        assert_eq!(page1_header(&p).freelist_count, 0);
        p.commit().unwrap();
    }
}
