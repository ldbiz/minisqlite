//! `MemPager` — the fully-resident, in-memory implementation of the `Pager` seam.
//!
//! It is a thin wrapper over the shared copy-on-write layer ([`Cow`]) parameterized
//! by an in-memory committed store ([`MemStore`]): every `Pager` method delegates
//! straight to `Cow`. The transaction, read-through, allocate/free, and freelist
//! logic all live in `Cow`/`alloc` so this backing and the future disk-backed one
//! share ONE implementation — the only thing specific to "in memory" is the store
//! (`MemStore`, a `Vec<Box<[u8]>>`), not the transaction machinery.
//!
//! The COW and borrow model (reads borrow tied to `&self`, writes are per-page
//! copy-on-write into an overlay, the committed store is untouched until `commit`)
//! is documented on [`Cow`](crate::cow) and [`MemStore`](crate::store::MemStore).

use minisqlite_types::Result;

use crate::cow::Cow;
use crate::store::MemStore;
use crate::{PageId, Pager, VacuumOutcome};

/// Smallest and largest page size SQLite permits; every legal size is a power of
/// two in this inclusive range (512, 1024, …, 32768, 65536).
const MIN_PAGE_SIZE: u32 = 512;
const MAX_PAGE_SIZE: u32 = 65536;

/// A resident, in-memory pager: the shared copy-on-write layer over an in-memory
/// committed store.
pub struct MemPager(Cow<MemStore>);

impl MemPager {
    /// Create an empty in-memory pager (0 pages). Page 1 is NOT formatted here — a
    /// higher layer allocates and formats it. `page_size` must be a power of two in
    /// `[512, 65536]`; anything else is a construction-time programming error, so it
    /// fails loud rather than silently clamping.
    pub fn new(page_size: u32) -> MemPager {
        assert!(
            (MIN_PAGE_SIZE..=MAX_PAGE_SIZE).contains(&page_size) && page_size.is_power_of_two(),
            "page_size must be a power of two in [{MIN_PAGE_SIZE}, {MAX_PAGE_SIZE}], got {page_size}"
        );
        MemPager(Cow::new(MemStore::new(page_size)))
    }
}

impl Pager for MemPager {
    fn read_page(&self, id: PageId) -> Result<&[u8]> {
        self.0.read_page(id)
    }

    fn page_count(&self) -> Result<PageId> {
        self.0.page_count()
    }

    fn page_size(&self) -> u32 {
        self.0.page_size()
    }

    fn begin(&mut self) -> Result<()> {
        self.0.begin()
    }

    fn write_page(&mut self, id: PageId, bytes: &[u8]) -> Result<()> {
        self.0.write_page(id, bytes)
    }

    fn page_mut(&mut self, id: PageId) -> Result<&mut [u8]> {
        self.0.page_mut(id)
    }

    fn allocate_page(&mut self) -> Result<PageId> {
        self.0.allocate_page()
    }

    fn free_page(&mut self, id: PageId) -> Result<()> {
        self.0.free_page(id)
    }

    fn commit(&mut self) -> Result<()> {
        self.0.commit()
    }

    fn rollback(&mut self) -> Result<()> {
        self.0.rollback()
    }

    fn savepoint(&mut self) -> Result<usize> {
        self.0.savepoint()
    }

    fn release_savepoint(&mut self, depth: usize) -> Result<()> {
        self.0.release_savepoint(depth)
    }

    fn rollback_to_savepoint(&mut self, depth: usize) -> Result<()> {
        self.0.rollback_to_savepoint(depth)
    }

    fn incremental_vacuum(&mut self, max: Option<PageId>) -> Result<VacuumOutcome> {
        self.0.incremental_vacuum(max)
    }

    fn take_root_moved(&mut self) -> bool {
        self.0.take_root_moved()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PS: u32 = 4096;

    fn page_of(byte: u8) -> Vec<u8> {
        vec![byte; PS as usize]
    }

    #[test]
    fn allocate_write_read_within_txn_then_commit_persists() {
        let mut p = MemPager::new(PS);
        p.begin().unwrap();
        let (a, b, c) =
            (p.allocate_page().unwrap(), p.allocate_page().unwrap(), p.allocate_page().unwrap());
        assert_eq!((a, b, c), (1, 2, 3));

        p.write_page(a, &page_of(0xAA)).unwrap();
        p.write_page(b, &page_of(0xBB)).unwrap();
        p.write_page(c, &page_of(0xCC)).unwrap();

        // Reads within the transaction see the staged bytes.
        assert_eq!(p.read_page(a).unwrap(), &page_of(0xAA)[..]);
        assert_eq!(p.read_page(b).unwrap(), &page_of(0xBB)[..]);
        assert_eq!(p.read_page(c).unwrap(), &page_of(0xCC)[..]);
        assert_eq!(p.page_count().unwrap(), 3);

        p.commit().unwrap();

        // After commit (no active transaction) reads return the committed image.
        assert_eq!(p.read_page(a).unwrap(), &page_of(0xAA)[..]);
        assert_eq!(p.read_page(b).unwrap(), &page_of(0xBB)[..]);
        assert_eq!(p.read_page(c).unwrap(), &page_of(0xCC)[..]);
        assert_eq!(p.page_count().unwrap(), 3);
    }

    #[test]
    fn rollback_restores_committed_pages_and_page_count() {
        let mut p = MemPager::new(PS);
        // Commit an initial page 1.
        p.begin().unwrap();
        let one = p.allocate_page().unwrap();
        p.write_page(one, &page_of(0x11)).unwrap();
        p.commit().unwrap();
        assert_eq!(p.page_count().unwrap(), 1);

        // In a new transaction, overwrite page 1 and allocate a new page 2.
        p.begin().unwrap();
        p.write_page(one, &page_of(0x22)).unwrap();
        let two = p.allocate_page().unwrap();
        assert_eq!(two, 2);
        p.write_page(two, &page_of(0x33)).unwrap();
        assert_eq!(p.read_page(one).unwrap(), &page_of(0x22)[..]);
        assert_eq!(p.page_count().unwrap(), 2);

        p.rollback().unwrap();

        // Page 1 reverts to its committed bytes; the allocated page 2 is gone.
        assert_eq!(p.read_page(one).unwrap(), &page_of(0x11)[..]);
        assert_eq!(p.page_count().unwrap(), 1);
        assert!(p.read_page(two).is_err());
    }

    #[test]
    fn unwritten_page_falls_through_to_base_while_sibling_is_overlaid() {
        let mut p = MemPager::new(PS);
        // Commit two distinct pages.
        p.begin().unwrap();
        let one = p.allocate_page().unwrap();
        let two = p.allocate_page().unwrap();
        p.write_page(one, &page_of(0x11)).unwrap();
        p.write_page(two, &page_of(0x22)).unwrap();
        p.commit().unwrap();

        // New transaction writing only page 2: page 1 must miss the overlay and read
        // its committed bytes, while page 2 reads the staged bytes.
        p.begin().unwrap();
        p.write_page(two, &page_of(0x33)).unwrap();
        assert_eq!(p.read_page(one).unwrap(), &page_of(0x11)[..], "unwritten page falls through to base");
        assert_eq!(p.read_page(two).unwrap(), &page_of(0x33)[..], "written page reads from overlay");
        p.commit().unwrap();
        assert_eq!(p.read_page(one).unwrap(), &page_of(0x11)[..]);
        assert_eq!(p.read_page(two).unwrap(), &page_of(0x33)[..]);
    }

    #[test]
    fn repeated_writes_to_same_page_in_txn_keep_latest() {
        // Pins the overlay buffer-reuse path: rewriting one id many times in a txn
        // keeps the last write (and does not corrupt on the in-place copy).
        let mut p = MemPager::new(PS);
        p.begin().unwrap();
        let id = p.allocate_page().unwrap();
        p.write_page(id, &page_of(0x01)).unwrap();
        p.write_page(id, &page_of(0x02)).unwrap();
        p.write_page(id, &page_of(0x03)).unwrap();
        assert_eq!(p.read_page(id).unwrap(), &page_of(0x03)[..]);
        p.commit().unwrap();
        assert_eq!(p.read_page(id).unwrap(), &page_of(0x03)[..]);
    }

    #[test]
    fn page_size_returns_constructor_value() {
        assert_eq!(MemPager::new(512).page_size(), 512);
        assert_eq!(MemPager::new(4096).page_size(), 4096);
        assert_eq!(MemPager::new(65536).page_size(), 65536);
    }

    #[test]
    fn allocate_returns_strictly_increasing_ids_from_one() {
        let mut p = MemPager::new(PS);
        let mut prev = 0;
        for expected in 1..=5 {
            let id = p.allocate_page().unwrap();
            assert_eq!(id, expected);
            assert!(id > prev, "ids must strictly increase");
            prev = id;
        }
        assert_eq!(p.page_count().unwrap(), 5);
    }

    #[test]
    fn write_wrong_length_errors_without_panicking() {
        let mut p = MemPager::new(PS);
        p.begin().unwrap();
        let id = p.allocate_page().unwrap();
        // Too short and too long both error rather than panic or truncate.
        assert!(p.write_page(id, &[0u8; 10]).is_err());
        assert!(p.write_page(id, &vec![0u8; PS as usize + 1]).is_err());
        // A correct-length write still succeeds afterward (state not corrupted).
        p.write_page(id, &page_of(0x7E)).unwrap();
        assert_eq!(p.read_page(id).unwrap(), &page_of(0x7E)[..]);
    }

    #[test]
    fn write_page_rejects_out_of_range_id() {
        let mut p = MemPager::new(PS);
        p.begin().unwrap();
        p.allocate_page().unwrap(); // only page 1 exists
        // A well-formed page written to a non-existent id must error, not land
        // silently in the overlay — inside a txn, at id 0, and outside a txn.
        assert!(p.write_page(2, &page_of(0x01)).is_err(), "id past page_count");
        assert!(p.write_page(0, &page_of(0x01)).is_err(), "id 0 is never valid");
        p.commit().unwrap();
        assert!(p.write_page(2, &page_of(0x01)).is_err(), "still rejected outside a txn");
    }

    #[test]
    fn read_nonexistent_or_zero_page_errors() {
        let p = MemPager::new(PS);
        assert!(p.read_page(1).is_err(), "no pages exist yet");
        assert!(p.read_page(0).is_err(), "id 0 is never valid (1-based)");
    }

    #[test]
    fn writes_and_allocations_outside_a_txn_apply_directly() {
        let mut p = MemPager::new(PS);
        let id = p.allocate_page().unwrap();
        assert_eq!(id, 1);
        p.write_page(id, &page_of(0x5A)).unwrap();
        assert_eq!(p.read_page(id).unwrap(), &page_of(0x5A)[..]);
        assert_eq!(p.page_count().unwrap(), 1);
    }

    #[test]
    fn transaction_state_misuse_is_reported_not_panicked() {
        let mut p = MemPager::new(PS);
        assert!(p.commit().is_err(), "commit with no transaction");
        assert!(p.rollback().is_err(), "rollback with no transaction");
        p.begin().unwrap();
        assert!(p.begin().is_err(), "nested begin");
        p.rollback().unwrap();
    }

    #[test]
    fn page_mut_edit_is_visible_and_commits() {
        // A page_mut edit is visible to reads in the same transaction and survives
        // commit — the in-place counterpart of write_page.
        let mut p = MemPager::new(PS);
        p.begin().unwrap();
        let id = p.allocate_page().unwrap();
        p.write_page(id, &page_of(0x00)).unwrap();
        {
            let buf = p.page_mut(id).unwrap();
            assert_eq!(buf.len(), PS as usize, "page_mut hands out the whole page");
            buf[10] = 0xAB;
            buf[PS as usize - 1] = 0xCD;
        }
        assert_eq!(p.read_page(id).unwrap()[10], 0xAB, "edit visible in-txn");
        assert_eq!(p.read_page(id).unwrap()[PS as usize - 1], 0xCD);
        p.commit().unwrap();
        assert_eq!(p.read_page(id).unwrap()[10], 0xAB, "edit survives commit");
        assert_eq!(p.read_page(id).unwrap()[PS as usize - 1], 0xCD);
    }

    #[test]
    fn page_mut_requires_active_transaction() {
        // A returned &mut borrow cannot auto-commit, so page_mut fails closed outside a
        // transaction (the b-tree falls back to write_page there).
        let mut p = MemPager::new(PS);
        p.begin().unwrap();
        p.allocate_page().unwrap();
        p.commit().unwrap();
        assert!(p.page_mut(1).is_err(), "page_mut with no active transaction is an error");
        // In a transaction it works.
        p.begin().unwrap();
        assert!(p.page_mut(1).is_ok());
        p.rollback().unwrap();
    }

    #[test]
    fn page_mut_rejects_out_of_range_and_zero_id() {
        let mut p = MemPager::new(PS);
        p.begin().unwrap();
        p.allocate_page().unwrap(); // only page 1 exists
        assert!(p.page_mut(2).is_err(), "id past page_count");
        assert!(p.page_mut(0).is_err(), "id 0 is never valid");
        p.rollback().unwrap();
    }

    #[test]
    fn page_mut_first_touch_reads_committed_image() {
        // On its first touch page_mut copies the COMMITTED image into the dirty buffer,
        // so the caller edits the current page content — not zeros.
        let mut p = MemPager::new(PS);
        p.begin().unwrap();
        let id = p.allocate_page().unwrap();
        p.write_page(id, &page_of(0x77)).unwrap();
        p.commit().unwrap();
        p.begin().unwrap();
        let buf = p.page_mut(id).unwrap();
        assert!(buf.iter().all(|&b| b == 0x77), "page_mut sees the committed bytes");
        p.rollback().unwrap();
    }

    #[test]
    fn page_mut_matches_read_modify_write_page_byte_for_byte() {
        // The core equivalence: editing a page in place through page_mut produces the
        // EXACT committed bytes the read-modify-write_page cycle would. Two pagers,
        // same committed base, same logical edit, must end identical.
        let base = |seed: u8| {
            (0..PS as usize).map(|i| (i as u8).wrapping_add(seed)).collect::<Vec<u8>>()
        };
        let edits: &[(usize, u8)] = &[(0, 0x01), (5, 0x02), (100, 0x03), (PS as usize - 1, 0x04)];

        let mut a = MemPager::new(PS);
        a.begin().unwrap();
        let id = a.allocate_page().unwrap();
        a.write_page(id, &base(0)).unwrap();
        a.commit().unwrap();

        let mut b = MemPager::new(PS);
        b.begin().unwrap();
        b.allocate_page().unwrap();
        b.write_page(id, &base(0)).unwrap();
        b.commit().unwrap();

        // A: read-modify-write_page.
        a.begin().unwrap();
        let mut buf = a.read_page(id).unwrap().to_vec();
        for &(off, val) in edits {
            buf[off] = val;
        }
        a.write_page(id, &buf).unwrap();
        a.commit().unwrap();

        // B: page_mut in place.
        b.begin().unwrap();
        {
            let m = b.page_mut(id).unwrap();
            for &(off, val) in edits {
                m[off] = val;
            }
        }
        b.commit().unwrap();

        assert_eq!(a.read_page(id).unwrap(), b.read_page(id).unwrap(), "page_mut == read-modify-write");
    }

    #[test]
    fn page_mut_rollback_reverts_to_committed() {
        let mut p = MemPager::new(PS);
        p.begin().unwrap();
        let id = p.allocate_page().unwrap();
        p.write_page(id, &page_of(0x11)).unwrap();
        p.commit().unwrap();

        p.begin().unwrap();
        p.page_mut(id).unwrap()[0] = 0x22;
        assert_eq!(p.read_page(id).unwrap()[0], 0x22);
        p.rollback().unwrap();
        assert_eq!(p.read_page(id).unwrap()[0], 0x11, "rollback reverts a page_mut edit");
    }

    #[test]
    fn page_mut_savepoint_rollback_reverts_edit() {
        // A savepoint captures a page_mut page's pre-image exactly like write_page:
        // rollback-to restores the bytes as of the mark, whether the page was first
        // touched before or after the savepoint.
        let mut p = MemPager::new(PS);
        p.begin().unwrap();
        let id = p.allocate_page().unwrap();
        p.write_page(id, &page_of(0x00)).unwrap();

        // Edit to A before the savepoint (first touch was write_page above).
        p.page_mut(id).unwrap()[0] = 0xAA;
        let sp = p.savepoint().unwrap();
        // Edit to B under the savepoint.
        p.page_mut(id).unwrap()[0] = 0xBB;
        assert_eq!(p.read_page(id).unwrap()[0], 0xBB);
        p.rollback_to_savepoint(sp).unwrap();
        assert_eq!(p.read_page(id).unwrap()[0], 0xAA, "rollback-to restores the pre-savepoint edit");
        p.commit().unwrap();
        assert_eq!(p.read_page(id).unwrap()[0], 0xAA);
    }

    #[test]
    fn page_mut_savepoint_first_touch_under_savepoint_reverts_to_committed() {
        // First touch of a page happens UNDER the savepoint via page_mut: the captured
        // pre-image is the "was-absent" marker, so rollback-to drops the overlay entry
        // and the page reverts to its committed image.
        let mut p = MemPager::new(PS);
        p.begin().unwrap();
        let id = p.allocate_page().unwrap();
        p.write_page(id, &page_of(0x33)).unwrap();
        p.commit().unwrap();

        p.begin().unwrap();
        let sp = p.savepoint().unwrap();
        p.page_mut(id).unwrap()[0] = 0x99; // first touch this txn, under the savepoint
        assert_eq!(p.read_page(id).unwrap()[0], 0x99);
        p.rollback_to_savepoint(sp).unwrap();
        assert_eq!(p.read_page(id).unwrap()[0], 0x33, "reverts to committed (was-absent marker)");
        p.rollback().unwrap();
    }

    #[test]
    fn page_mut_and_write_page_share_one_dirty_buffer() {
        // page_mut after write_page (and vice-versa) edit the SAME dirty buffer: no
        // second copy, and the latest edit wins whichever primitive made it.
        let mut p = MemPager::new(PS);
        p.begin().unwrap();
        let id = p.allocate_page().unwrap();
        p.write_page(id, &page_of(0x00)).unwrap();
        p.page_mut(id).unwrap()[0] = 0x01;
        // write_page overwrites the whole page (including the page_mut edit).
        p.write_page(id, &page_of(0x02)).unwrap();
        assert_eq!(p.read_page(id).unwrap()[0], 0x02);
        // A subsequent page_mut edits that same buffer.
        p.page_mut(id).unwrap()[0] = 0x03;
        assert_eq!(p.read_page(id).unwrap()[0], 0x03);
        assert_eq!(p.read_page(id).unwrap()[1], 0x02, "rest of the write_page bytes preserved");
        p.commit().unwrap();
        assert_eq!(p.read_page(id).unwrap()[0], 0x03);
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn new_rejects_non_power_of_two_page_size() {
        let _ = MemPager::new(1000);
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn new_rejects_below_range_page_size() {
        let _ = MemPager::new(256);
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn new_rejects_power_of_two_above_range() {
        // A clean power of two, but past the upper bound — pins the `..=MAX` edge.
        let _ = MemPager::new(131_072);
    }
}
