//! Auto_vacuum §1.8 ROOTS-FIRST placement: in any file that contains ptrmap pages, "all
//! b-tree root pages must come before any non-root b-tree page, cell payload overflow
//! page, or freelist page" (fileformat2 §1.8, lines 1094-1103). SQLite relies on this so
//! auto-vacuum never has to move a root (it cannot update `sqlite_schema.rootpage` during a
//! vacuum); a file that violates it is read fine by real sqlite but CORRUPTS the moment
//! that file is auto-vacuumed or `PRAGMA integrity_check`ed.
//!
//! ## Why this module exists
//! Roots are allocated with a plain `allocate_page` ([`crate::alloc`]), which for an
//! auto_vacuum database grows to the TAIL (or reuses a freelist page). So a root created
//! AFTER data exists — the ubiquitous `CREATE INDEX` on a populated table, or a second
//! `CREATE TABLE` after inserts — lands ABOVE existing non-root/overflow/freelist pages,
//! exactly what §1.8 forbids. This module restores the invariant at commit, which is the
//! spec's "root pages are moved to the beginning of the database file by the CREATE
//! TABLE, CREATE INDEX, DROP TABLE, and DROP INDEX operations."
//!
//! ## What [`enforce`] does (and why at commit, from the fresh forest)
//! Called from [`crate::ptrmap_build::finalize`] with the freshly-walked forest map, so it
//! never reads a stale on-disk ptrmap: it relocates roots so they occupy the lowest data
//! slots. For each root sitting too high, it takes a low "desired" slot and, if a non-root
//! page occupies it, SWAPS the two — the root's bytes go to the low slot and the blocker's
//! bytes go to the (now vacated) high slot — rewriting the ONE pointer that named the
//! blocker (its parent b-tree pointer / overflow link, from the fresh `(kind, parent)`
//! entry) and the moved root's `sqlite_schema.rootpage`. A SWAP keeps the page count fixed
//! (no growth, no holes). If the desired slot is a FREE page instead, the root simply takes
//! it and the vacated root slot becomes free (the freelist is rebuilt).
//!
//! The pointer rewrites reuse [`crate::reclaim`]'s primitives (the same ones compaction
//! uses to relocate a page). The ptrmap itself is NOT touched here — [`finalize`] re-derives
//! the whole ptrmap from the relocated forest, so this module only has to leave the FOREST
//! (pointers + schema rootpages + freelist) consistent, just like [`crate::reclaim`].
//!
//! ## Fail closed, never corrupt
//! [`finalize`] re-derives and re-checks after a relocation; if the forest is left
//! inconsistent (a missed pointer, an unclassifiable page) the re-derivation errors and the
//! whole commit rolls back to the pre-relocation, fully-valid state (a relocation bug can
//! only fail the CREATE statement loudly, never persist a corrupt file). A no-op — a single
//! set comparison — on the overwhelmingly common commit whose roots are already first
//! (every INSERT/UPDATE/DELETE, and any DDL that did not create a misplaced root).

use std::collections::{BTreeMap, BTreeSet, HashMap};

use minisqlite_fileformat::ptrmap::{
    PTRMAP_BTREE, PTRMAP_FREEPAGE, PTRMAP_OVERFLOW1, PTRMAP_OVERFLOW2,
};
use minisqlite_types::{Error, Result};

use crate::cow::Cow;
use crate::reclaim;
use crate::store::PageStore;
use crate::PageId;

/// A ptrmap back-pointer `(type, parent)`, matching [`crate::ptrmap_build`]'s `Entry`.
type Entry = (u8, u32);

/// Relocate b-tree roots to the front so every root precedes every non-root / overflow /
/// freelist page (§1.8). `roots` are the table/index root pages (>= 3) and `entries` is the
/// freshly-derived `(kind, parent)` for EVERY non-ptrmap page in `2..=count` (live pages
/// from the forest walk, plus freelist pages) — exactly what [`crate::ptrmap_build`] built.
///
/// Returns `true` if it moved any page (so the caller re-derives the forest for the new
/// placement), `false` if the roots were already first (the common case: one set compare).
/// The page count never changes (roots swap with blockers or take free low slots).
pub(crate) fn enforce<S: PageStore>(
    cow: &mut Cow<S>,
    roots: &[PageId],
    entries: &BTreeMap<PageId, Entry>,
    count: PageId,
    usable: usize,
) -> Result<bool> {
    if roots.is_empty() {
        return Ok(false);
    }
    let desired = desired_root_slots(roots.len(), count, usable)?;
    let desired_set: BTreeSet<PageId> = desired.iter().copied().collect();
    let root_set: BTreeSet<PageId> = roots.iter().copied().collect();
    if root_set == desired_set {
        // Already roots-first: nothing to do (the hot path for every non-DDL commit).
        return Ok(false);
    }

    // A root not among the lowest |roots| slots is "misplaced" (it sits too high); a
    // desired slot not currently holding a root is an open "target" it can move into.
    // These two lists have equal length (the roots fill exactly |roots| slots), and every
    // misplaced root is strictly greater than every target (a target is <= max(desired)
    // and a misplaced root is > max(desired)), so `dest < from` holds for every move —
    // which is what lets `rewrite_schema_rootpage` overwrite the rootpage in place.
    let mut misplaced: Vec<PageId> =
        roots.iter().copied().filter(|r| !desired_set.contains(r)).collect();
    misplaced.sort_unstable();
    let mut targets: Vec<PageId> =
        desired.iter().copied().filter(|s| !root_set.contains(s)).collect();
    targets.sort_unstable();
    if misplaced.len() != targets.len() {
        return Err(Error::Format(format!(
            "roots_first: {} misplaced roots but {} open target slots (forest inconsistent)",
            misplaced.len(),
            targets.len()
        )));
    }

    // Track relocations so a blocker whose parent ALSO moved this pass is repointed at the
    // parent's current home, and track freelist membership changes for free-slot targets.
    let mut moved: HashMap<PageId, PageId> = HashMap::new();
    let mut freelist_changed = false;
    let mut free_set: BTreeSet<PageId> =
        entries.iter().filter(|&(_, &(k, _))| k == PTRMAP_FREEPAGE).map(|(&p, _)| p).collect();

    for (&root, &slot) in misplaced.iter().zip(targets.iter()) {
        let &(blocker_kind, blocker_parent) = entries.get(&slot).ok_or_else(|| {
            Error::Format(format!(
                "roots_first: target slot {slot} has no forest entry (not live or free)"
            ))
        })?;

        if blocker_kind == PTRMAP_FREEPAGE {
            // The desired slot is a FREE page: the root simply takes it, and the vacated
            // root slot becomes free. Only the schema rootpage moves; the freelist is
            // rebuilt below.
            let root_bytes = cow.read_page(root)?.to_vec();
            cow.stage_write(slot, &root_bytes)?;
            moved.insert(root, slot);
            reclaim::rewrite_schema_rootpage(cow, root, slot, usable)?;
            free_set.remove(&slot);
            free_set.insert(root);
            freelist_changed = true;
            continue;
        }

        // The desired slot holds a LIVE non-root page (blocker). SWAP it with the root:
        // read both, then cross-write, so neither read borrows across the writes.
        let root_bytes = cow.read_page(root)?.to_vec();
        let blocker_bytes = cow.read_page(slot)?.to_vec();
        cow.stage_write(slot, &root_bytes)?; // root content -> low slot
        cow.stage_write(root, &blocker_bytes)?; // blocker content -> vacated high slot

        // Record BOTH moves before rewriting any pointer: when the blocker is a direct
        // child of THIS root, its child pointer now lives in the root's bytes (at `slot`),
        // so resolving the blocker's parent must already see the root at `slot`.
        moved.insert(root, slot);
        moved.insert(slot, root);

        // Repoint the ONE pointer that named the blocker (now at `root`), following its
        // parent to the parent's current home if the parent itself moved this pass.
        let parent = current(&moved, blocker_parent);
        match blocker_kind {
            PTRMAP_BTREE => reclaim::rewrite_child_pointer(cow, parent, slot, root, usable)?,
            PTRMAP_OVERFLOW1 => reclaim::rewrite_overflow_head(cow, parent, slot, root, usable)?,
            PTRMAP_OVERFLOW2 => reclaim::rewrite_overflow_next(cow, parent, slot, root)?,
            other => {
                return Err(Error::Format(format!(
                    "roots_first: page {slot} has ptrmap type {other}, expected a non-root \
                     b-tree/overflow blocker; declining"
                )))
            }
        }
        // The forest now names the blocker at `root`; move the root's schema rootpage
        // (schema b-tree is consistent — a schema-page blocker was already repointed above).
        reclaim::rewrite_schema_rootpage(cow, root, slot, usable)?;
    }

    if freelist_changed {
        reclaim::rebuild_freelist(cow, &free_set)?;
    }
    Ok(true)
}

/// The lowest `n` non-ptrmap page numbers `>= 3` — the slots b-tree roots must occupy so
/// every root precedes every non-root page (§1.8). Page 1 is `sqlite_schema` (special, its
/// b-tree root, never a table/index root slot) and page 2 is always the first ptrmap page,
/// so root slots start at page 3.
fn desired_root_slots(n: usize, count: PageId, usable: usize) -> Result<Vec<PageId>> {
    let mut slots = Vec::with_capacity(n);
    let mut p: PageId = 3;
    while slots.len() < n {
        if !minisqlite_fileformat::is_ptrmap_page(usable, p) {
            slots.push(p);
        }
        p = p
            .checked_add(1)
            .ok_or_else(|| Error::Format("roots_first: page id overflow computing root slots".into()))?;
    }
    // The database already holds `n` roots, so at least `n` data slots exist; a desired
    // slot past the file means the forest is inconsistent — fail closed.
    if let Some(&max) = slots.last() {
        if max > count {
            return Err(Error::Format(format!(
                "roots_first: desired root slot {max} is past the page count {count}"
            )));
        }
    }
    Ok(slots)
}

/// A page's CURRENT location after this pass's moves — a SINGLE lookup, not an iterating
/// chase. A swap records both directions (`moved[root]=slot` and `moved[slot]=root`), a
/// 2-cycle an iterating resolver (like [`reclaim`]'s `resolve_moved`) would loop on; but
/// every original slot maps DIRECTLY to its final slot in this pass (each page moves at
/// most once, to a distinct destination), so exactly one lookup is correct.
fn current(moved: &HashMap<PageId, PageId>, page: PageId) -> PageId {
    moved.get(&page).copied().unwrap_or(page)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::store::MemStore;
    use minisqlite_fileformat::ptrmap::{ptrmap_offset_of, PTRMAP_ENTRY_SIZE, PTRMAP_ROOTPAGE};
    use minisqlite_fileformat::{
        decode_ptrmap_entry, decode_record, encode_record, encode_table_interior_cell,
        encode_table_leaf_cell, DatabaseHeader, PageBuilder, PageType, PageView, HEADER_SIZE,
    };
    use minisqlite_types::Value;

    // usable = 512 (reserved 0) -> ptrmap stride 102, ptrmap pages at 2, 105, 208, ...
    const USABLE_512: usize = 512;

    // Page size for the hand-built forest fixtures below; with reserved 0 it is also the
    // usable size, and at 512 (the §1.3.4 minimum) only page 2 is a ptrmap page in these
    // tiny files, so the §1.8 offset math is trivial to reason about by hand.
    const PS: usize = 512;

    #[test]
    fn desired_slots_skip_ptrmap_positions() {
        // One root: the lowest data slot is page 3 (page 2 is the first ptrmap page).
        assert_eq!(desired_root_slots(1, 300, USABLE_512).unwrap(), vec![3]);
        // Three roots: pages 3,4,5 (none of which is a ptrmap page).
        assert_eq!(desired_root_slots(3, 300, USABLE_512).unwrap(), vec![3, 4, 5]);
    }

    #[test]
    fn desired_slots_span_a_ptrmap_page() {
        // Enough roots to run past page 105 (a ptrmap position at usable 512): the run
        // 3..=104 is 102 slots, then 105 is skipped, so slot #103 is page 106.
        let slots = desired_root_slots(103, 300, USABLE_512).unwrap();
        assert_eq!(slots.len(), 103);
        assert_eq!(slots[101], 104, "102nd data slot is page 104");
        assert_eq!(slots[102], 106, "103rd data slot skips the ptrmap page 105");
        assert!(!slots.contains(&105), "ptrmap page 105 is never a root slot");
        assert!(!slots.contains(&2), "ptrmap page 2 is never a root slot");
    }

    #[test]
    fn desired_slots_past_page_count_fail_closed() {
        // Asking for more root slots than the file has data pages is a forest
        // inconsistency; fail closed rather than pointing a root past the file.
        assert!(desired_root_slots(5, 4, USABLE_512).is_err());
    }

    #[test]
    fn current_is_a_single_lookup_over_a_swap_cycle() {
        // A swap records both directions; `current` must NOT chase the 2-cycle.
        let mut moved = HashMap::new();
        moved.insert(10u32, 3u32); // root 10 -> slot 3
        moved.insert(3u32, 10u32); // blocker at 3 -> slot 10
        assert_eq!(current(&moved, 10), 3);
        assert_eq!(current(&moved, 3), 10);
        assert_eq!(current(&moved, 7), 7, "an unmoved page maps to itself");
    }

    // ---- enforce: end-to-end MULTI-ROOT relocation incl. the cross-move case ----------
    //
    // The tests above cover `desired_root_slots`/`current` in isolation and the crate's
    // facade tests cover a SINGLE misplaced root. These drive `enforce` directly on a
    // hand-built forest with TWO misplaced roots in one pass — including a blocker whose
    // parent is itself a root that moved this same pass (the cross-move `current` handles)
    // — and then re-derive the ptrmap with the real [`crate::ptrmap_build::finalize`] to
    // prove the whole forest is left consistent (fail-closed if not).

    fn mem_cow() -> Cow<MemStore> {
        Cow::new(MemStore::new(PS as u32))
    }

    /// Build a database from explicit page images (`pages[0]` is page 1, …) into an OPEN
    /// transaction and return the `Cow` with it still active. Every image must be exactly
    /// `PS` bytes. Mirrors `reclaim.rs`'s `staged`: the caller runs the code under test,
    /// then rolls back, so nothing commits except what the test explicitly finalizes.
    fn staged(pages: Vec<Vec<u8>>) -> Cow<MemStore> {
        let mut cow = mem_cow();
        cow.begin().expect("begin");
        for (i, bytes) in pages.iter().enumerate() {
            assert_eq!(bytes.len(), PS, "page {} image must be exactly {PS} bytes", i + 1);
            let id = cow.grow_one().expect("grow_one");
            cow.stage_write(id, bytes).expect("stage_write");
        }
        cow
    }

    fn zeros() -> Vec<u8> {
        vec![0u8; PS]
    }

    /// A page-1 `sqlite_schema` leaf naming one `(name, rootpage)` table per row (record
    /// columns `type, name, tbl_name, rootpage, sql`), with a valid auto_vacuum database
    /// header so `uses_ptrmap()` is true (offset 52 = `largest_root`) and `finalize` runs.
    fn schema_page(rows: &[(&str, i64)], largest_root: u32) -> Vec<u8> {
        let mut b = PageBuilder::new(PS, PS, 1, PageType::LeafTable);
        for (i, (name, rootpage)) in rows.iter().enumerate() {
            let rec = encode_record(&[
                Value::Text("table".into()),
                Value::Text((*name).into()),
                Value::Text((*name).into()),
                Value::Integer(*rootpage),
                Value::Text(format!("CREATE TABLE {name}(a)")),
            ]);
            let mut cell = Vec::new();
            encode_table_leaf_cell(rec.len() as u64, (i + 1) as i64, &rec, None, &mut cell);
            assert!(b.add_cell(&cell), "schema row {name} must fit on page 1");
        }
        let mut page = b.finish();
        let header = DatabaseHeader {
            page_size: PS as u32,
            reserved_space: 0,
            largest_root_btree: largest_root,
            ..DatabaseHeader::default()
        };
        let head: &mut [u8; HEADER_SIZE] =
            (&mut page[..HEADER_SIZE]).try_into().expect("page 1 has at least HEADER_SIZE bytes");
        header.write(head);
        page
    }

    /// An interior table page with exactly two children: one cell (`left_child`) plus the
    /// right-most pointer. This is the shape a root needs to parent two blocker pages.
    fn interior_two_children(page_number: u32, left_child: u32, right_most: u32) -> Vec<u8> {
        let mut b = PageBuilder::new(PS, PS, page_number, PageType::InteriorTable);
        b.set_right_most_pointer(right_most);
        let mut cell = Vec::new();
        encode_table_interior_cell(left_child, 1, &mut cell);
        assert!(b.add_cell(&cell), "the single interior cell must fit");
        b.finish()
    }

    /// A one-row table-leaf page holding `Integer(value)` at `rowid`.
    fn leaf_one_row(page_number: u32, rowid: i64, value: i64) -> Vec<u8> {
        let mut b = PageBuilder::new(PS, PS, page_number, PageType::LeafTable);
        let rec = encode_record(&[Value::Integer(value)]);
        let mut cell = Vec::new();
        encode_table_leaf_cell(rec.len() as u64, rowid, &rec, None, &mut cell);
        assert!(b.add_cell(&cell), "row must fit on a 512-byte leaf");
        b.finish()
    }

    fn empty_leaf(page_number: u32) -> Vec<u8> {
        PageBuilder::new(PS, PS, page_number, PageType::LeafTable).finish()
    }

    /// The `rootpage` column (index 3) of every `sqlite_schema` leaf cell on page 1.
    fn schema_rootpages(cow: &Cow<MemStore>) -> Vec<i64> {
        let p1 = cow.read_page(1).unwrap().to_vec();
        let view = PageView::new(&p1, 1, PS).unwrap();
        let mut out = Vec::new();
        for i in 0..view.cell_count() as usize {
            let cell = view.table_leaf_cell(i).unwrap();
            let vals = decode_record(cell.local_payload);
            match vals[3] {
                Value::Integer(r) => out.push(r),
                ref other => panic!("schema rootpage column must be an integer, got {other:?}"),
            }
        }
        out
    }

    /// Decode the on-disk ptrmap entry for data page `p` from a ptrmap page image.
    fn ptrmap_entry(ptrmap_page: &[u8], p: PageId) -> (u8, u32) {
        let off = ptrmap_offset_of(PS, p);
        decode_ptrmap_entry(&ptrmap_page[off..off + PTRMAP_ENTRY_SIZE])
    }

    #[test]
    fn enforce_relocates_two_roots_across_a_moved_parent() {
        // Forest laid out so BOTH cross-move shapes fire in one `enforce` pass:
        //
        //   page 1: schema (t1 root = 5, t2 root = 6)   page 2: ptrmap (finalize rebuilds)
        //   page 3: B3 leaf  (a child of R1)            page 4: B4 leaf  (a child of R1)
        //   page 5: R1 interior root (cell child = 3, right-most = 4) -> children B3, B4
        //   page 6: R2 empty leaf root
        //
        // Two roots (5, 6) are misplaced; their desired low slots are (3, 4). `enforce`
        // pairs them as (5 -> 3) then (6 -> 4):
        //   * 5 -> 3 SWAPs R1 with its own child B3. B3's parent is R1, which just moved to
        //     slot 3, so `current(&moved, 5) == 3` — the child-pointer rewrite must land in
        //     R1's NEW home (page 3), retargeting its cell child 3 -> 5 (SELF-REF variant).
        //   * 6 -> 4 SWAPs R2 with B4. B4's parent is R1 (= 5), which ALREADY moved to 3 in
        //     the first step, so `current(&moved, 5) == 3` again — the rewrite lands in page
        //     3, retargeting its right-most 4 -> 6 (PARENT-ALREADY-MOVED variant).
        // If `current` chased the swap 2-cycle instead of a single lookup, or if both moves
        // were not recorded before the first rewrite, one pointer would aim at the wrong
        // page and `finalize`'s fail-closed re-derivation would reject the forest.
        let mut cow = staged(vec![
            schema_page(&[("t1", 5), ("t2", 6)], 6),
            zeros(),
            leaf_one_row(3, 1, 300),                // B3 (child of R1)
            leaf_one_row(4, 1, 400),                // B4 (child of R1)
            interior_two_children(5, 3, 4),         // R1 root: children B3, B4
            empty_leaf(6),                          // R2 root
        ]);

        let mut entries: BTreeMap<PageId, Entry> = BTreeMap::new();
        entries.insert(3, (PTRMAP_BTREE, 5)); // B3 -> parent R1 (=5)
        entries.insert(4, (PTRMAP_BTREE, 5)); // B4 -> parent R1 (=5)
        entries.insert(5, (PTRMAP_ROOTPAGE, 0)); // R1 root
        entries.insert(6, (PTRMAP_ROOTPAGE, 0)); // R2 root
        let roots = [5u32, 6u32];

        // Byte images that must survive the relocation verbatim (only their slot changes).
        let b3_orig = cow.read_page(3).unwrap().to_vec();
        let b4_orig = cow.read_page(4).unwrap().to_vec();
        let r2_orig = cow.read_page(6).unwrap().to_vec();

        let moved = enforce(&mut cow, &roots, &entries, 6, PS).unwrap();
        assert!(moved, "enforce relocated the two misplaced roots, so it returns true");

        // Both schema rootpages were rewritten to the roots' new low homes (3 and 4).
        let mut after = schema_rootpages(&cow);
        after.sort_unstable();
        assert_eq!(after, vec![3, 4], "schema rootpages moved to the two lowest data slots");

        // R1's bytes now live at slot 3 with BOTH child pointers repointed at the blockers'
        // new (high) homes: the self-ref child 3 -> 5 and the parent-moved right-most 4 -> 6.
        let p3 = cow.read_page(3).unwrap().to_vec();
        let v3 = PageView::new(&p3, 3, PS).unwrap();
        assert_eq!(v3.page_type(), PageType::InteriorTable, "R1 (interior) relocated to slot 3");
        assert_eq!(
            v3.table_interior_cell(0).unwrap().left_child,
            5,
            "self-ref child pointer repointed 3 -> 5 (blocker B3's new home)"
        );
        assert_eq!(
            v3.right_most_pointer(),
            Some(6),
            "parent-already-moved child pointer repointed 4 -> 6 (blocker B4's new home)"
        );

        // The swapped-out pages are byte-identical to their originals at the vacated slots.
        assert_eq!(cow.read_page(4).unwrap(), &r2_orig[..], "R2 root took vacated slot 4 verbatim");
        assert_eq!(cow.read_page(5).unwrap(), &b3_orig[..], "B3 moved to vacated slot 5 verbatim");
        assert_eq!(cow.read_page(6).unwrap(), &b4_orig[..], "B4 moved to vacated slot 6 verbatim");

        // Re-derive the ptrmap from the relocated forest via the REAL commit path. It must
        // see the forest already roots-first (relocating nothing) AND not fail closed —
        // proof the forest `enforce` left is fully consistent.
        let moved_again = crate::ptrmap_build::finalize(&mut cow).unwrap();
        assert!(!moved_again, "finalize sees a forest already in roots-first order");

        // The on-disk ptrmap (page 2) now describes the relocated forest exactly.
        let ptrmap = cow.read_page(2).unwrap().to_vec();
        assert_eq!(ptrmap_entry(&ptrmap, 3), (PTRMAP_ROOTPAGE, 0), "page 3 is now root R1");
        assert_eq!(ptrmap_entry(&ptrmap, 4), (PTRMAP_ROOTPAGE, 0), "page 4 is now root R2");
        assert_eq!(ptrmap_entry(&ptrmap, 5), (PTRMAP_BTREE, 3), "B3 is a child of R1 at 3");
        assert_eq!(ptrmap_entry(&ptrmap, 6), (PTRMAP_BTREE, 3), "B4 is a child of R1 at 3");

        // Offset 52 records the largest root (4) after relocation (§1.3.12).
        let p1 = cow.read_page(1).unwrap().to_vec();
        let hdr = DatabaseHeader::read((&p1[..HEADER_SIZE]).try_into().unwrap()).unwrap();
        assert_eq!(hdr.largest_root_btree, 4, "offset 52 == the largest root after relocation");

        cow.rollback().unwrap();
    }

    #[test]
    fn enforce_mixes_a_free_slot_target_with_a_swap_in_one_pass() {
        // Two misplaced roots where one desired slot is FREE and the other holds a live
        // blocker, so a single pass exercises BOTH `enforce` branches together plus the
        // freelist rebuild:
        //
        //   page 1: schema (t1 root = 5, t2 root = 6)   page 2: ptrmap (finalize rebuilds)
        //   page 3: FREE (a lone empty freelist trunk: next = 0, 0 leaves)
        //   page 4: B4 leaf  (R1's cell child)
        //   page 5: R1 interior root (cell child = 4, right-most = 7) -> children B4, B7
        //   page 6: R2 empty leaf root
        //   page 7: B7 leaf  (R1's right-most child)                          count = 7
        //
        // Desired low slots (3, 4) for roots (5, 6): pairing (5 -> 3) then (6 -> 4).
        //   * 5 -> 3 is a FREE target: R1's bytes take page 3, the vacated page 5 becomes
        //     free, and only R1's schema rootpage moves (no pointer rewrite for a free slot).
        //   * 6 -> 4 SWAPs R2 with the live blocker B4, whose parent R1 (= 5) already moved
        //     to page 3 this pass — so the child-pointer rewrite must follow `current` to
        //     R1's new home (page 3) and retarget its cell child 4 -> 6 (parent-moved
        //     cross-move). R1's unmoved right-most child (B7 at page 7) is left untouched.
        // After the pass the freelist must hold exactly the newly-vacated page 5 (page 3 is
        // no longer free), which the ptrmap re-derivation confirms as a FREEPAGE.
        let mut cow = staged(vec![
            schema_page(&[("t1", 5), ("t2", 6)], 6),
            zeros(),
            free_trunk_page(),                      // page 3: free (lone trunk)
            leaf_one_row(4, 1, 400),                // page 4: B4 (R1's cell child)
            interior_two_children(5, 4, 7),         // page 5: R1 (child 4, right-most 7)
            empty_leaf(6),                          // page 6: R2 root
            leaf_one_row(7, 1, 700),                // page 7: B7 (R1's right-most child)
        ]);
        // Page 1 header must name the freelist: one trunk at page 3, count 1.
        set_freelist_head(&mut cow, 3, 1);

        let mut entries: BTreeMap<PageId, Entry> = BTreeMap::new();
        entries.insert(3, (PTRMAP_FREEPAGE, 0)); // free target for R1
        entries.insert(4, (PTRMAP_BTREE, 5)); // B4 -> parent R1 (=5)
        entries.insert(5, (PTRMAP_ROOTPAGE, 0)); // R1 root
        entries.insert(6, (PTRMAP_ROOTPAGE, 0)); // R2 root
        entries.insert(7, (PTRMAP_BTREE, 5)); // B7 -> parent R1 (=5)
        let roots = [5u32, 6u32];

        let r2_orig = cow.read_page(6).unwrap().to_vec();
        let b4_orig = cow.read_page(4).unwrap().to_vec();

        let moved = enforce(&mut cow, &roots, &entries, 7, PS).unwrap();
        assert!(moved, "enforce relocated two roots (a free-slot take + a swap)");

        let mut after = schema_rootpages(&cow);
        after.sort_unstable();
        assert_eq!(after, vec![3, 4], "roots moved to the two lowest data slots");

        // R1 now at page 3: its cell child (was 4, the swapped blocker) points at B4's new
        // home 6; its right-most child (B7 at page 7, unmoved) is unchanged.
        let p3 = cow.read_page(3).unwrap().to_vec();
        let v3 = PageView::new(&p3, 3, PS).unwrap();
        assert_eq!(v3.page_type(), PageType::InteriorTable, "R1 relocated into the free slot 3");
        assert_eq!(
            v3.table_interior_cell(0).unwrap().left_child,
            6,
            "R1's cell child (blocker B4) repointed 4 -> 6 via the moved-parent lookup"
        );
        assert_eq!(v3.right_most_pointer(), Some(7), "R1's unmoved right-most child stays at 7");

        // R2 took the swapped slot 4; B4 took R2's vacated slot 6.
        assert_eq!(cow.read_page(4).unwrap(), &r2_orig[..], "R2 took vacated slot 4 verbatim");
        assert_eq!(cow.read_page(6).unwrap(), &b4_orig[..], "B4 moved to vacated slot 6 verbatim");

        // Re-derive: consistent forest, and the freelist now holds exactly page 5 (R1's old
        // slot), since page 3 stopped being free when R1 took it.
        let moved_again = crate::ptrmap_build::finalize(&mut cow).unwrap();
        assert!(!moved_again, "finalize sees a forest already in roots-first order");

        let ptrmap = cow.read_page(2).unwrap().to_vec();
        assert_eq!(ptrmap_entry(&ptrmap, 3), (PTRMAP_ROOTPAGE, 0), "page 3 is now root R1");
        assert_eq!(ptrmap_entry(&ptrmap, 4), (PTRMAP_ROOTPAGE, 0), "page 4 is now root R2");
        assert_eq!(ptrmap_entry(&ptrmap, 5), (PTRMAP_FREEPAGE, 0), "page 5 is the newly-freed slot");
        assert_eq!(ptrmap_entry(&ptrmap, 6), (PTRMAP_BTREE, 3), "B4 now a child of R1 at 3");
        assert_eq!(ptrmap_entry(&ptrmap, 7), (PTRMAP_BTREE, 3), "B7 now a child of R1 at 3");

        let p1 = cow.read_page(1).unwrap().to_vec();
        let hdr = DatabaseHeader::read((&p1[..HEADER_SIZE]).try_into().unwrap()).unwrap();
        assert_eq!(hdr.largest_root_btree, 4, "offset 52 == the largest root after relocation");
        assert_eq!(hdr.freelist_count, 1, "exactly one page (the old root slot 5) is free");
        assert_eq!(hdr.first_freelist_trunk, 5, "the freelist head is the newly-freed page 5");

        cow.rollback().unwrap();
    }

    #[test]
    fn enforce_rewrites_a_blocker_pointer_before_its_parent_root_moves() {
        // The THIRD cross-move ordering: a blocker's parent root is relocated in a LATER
        // pair than the blocker itself. At the blocker's rewrite `current(&moved, parent)`
        // returns IDENTITY (the parent has NOT moved yet), so the pointer is rewritten in
        // the parent's ORIGINAL page — and correctness then rests on the parent's later swap
        // carrying that just-rewritten pointer along in its bytes. Tests 1/2 above only cover
        // the parent-moves-THIS-step (self-ref) and parent-moved-EARLIER orderings, where the
        // rewrite lands in the parent's final home; this pins the "rewrite page P, THEN move
        // P" sequence that neither does.
        //
        //   page 1: schema (t1 root = 5, t2 root = 6)   page 2: ptrmap (finalize rebuilds)
        //   page 3: B  leaf  (R_hi's right-most child)  page 4: B2 leaf (R_hi's cell child)
        //   page 5: R_lo empty leaf root
        //   page 6: R_hi interior root (cell child = 4, right-most = 3) -> children B2, B
        //
        // desired [3,4] for roots [5,6]; pairs (5->3) then (6->4):
        //   * 5 -> 3 swaps R_lo with B; B's parent is R_hi (= 6), which is processed in the
        //     NEXT pair and so is NOT yet in `moved` -> `current(6) == 6` (identity). The
        //     right-most rewrite 3 -> 5 lands in page 6 (R_hi's ORIGINAL home).
        //   * 6 -> 4 swaps R_hi with B2. R_hi's bytes (already carrying right-most = 5) move
        //     to page 4; then B2's cell-child pointer is rewritten with `current(6) == 4`
        //     (R_hi's NEW home). Final R_hi at page 4 must read cell child = 6, right-most = 5.
        let mut cow = staged(vec![
            schema_page(&[("t1", 5), ("t2", 6)], 6),
            zeros(),
            leaf_one_row(3, 1, 300),                // B  (R_hi's right-most child)
            leaf_one_row(4, 1, 400),                // B2 (R_hi's cell child)
            empty_leaf(5),                          // R_lo root
            interior_two_children(6, 4, 3),         // R_hi root: cell child 4 (B2), rm 3 (B)
        ]);

        let mut entries: BTreeMap<PageId, Entry> = BTreeMap::new();
        entries.insert(3, (PTRMAP_BTREE, 6)); // B  -> parent R_hi (=6)
        entries.insert(4, (PTRMAP_BTREE, 6)); // B2 -> parent R_hi (=6)
        entries.insert(5, (PTRMAP_ROOTPAGE, 0)); // R_lo root
        entries.insert(6, (PTRMAP_ROOTPAGE, 0)); // R_hi root
        let roots = [5u32, 6u32];

        let b_orig = cow.read_page(3).unwrap().to_vec();
        let b2_orig = cow.read_page(4).unwrap().to_vec();
        let rlo_orig = cow.read_page(5).unwrap().to_vec();

        let moved = enforce(&mut cow, &roots, &entries, 6, PS).unwrap();
        assert!(moved, "enforce relocated the two misplaced roots");

        let mut after = schema_rootpages(&cow);
        after.sort_unstable();
        assert_eq!(after, vec![3, 4], "schema rootpages moved to the two lowest data slots");

        // R_hi's bytes now live at slot 4; the right-most rewrite (done while R_hi was still
        // at page 6, then carried along by its swap into page 4) points at B's new home 5,
        // and the cell child (rewritten in R_hi's new home) points at B2's new home 6.
        let p4 = cow.read_page(4).unwrap().to_vec();
        let v4 = PageView::new(&p4, 4, PS).unwrap();
        assert_eq!(v4.page_type(), PageType::InteriorTable, "R_hi (interior) relocated to slot 4");
        assert_eq!(
            v4.table_interior_cell(0).unwrap().left_child,
            6,
            "cell child repointed 4 -> 6 in R_hi's NEW home (parent-already-moved arm)"
        );
        assert_eq!(
            v4.right_most_pointer(),
            Some(5),
            "right-most rewrite 3 -> 5 was done in R_hi's OLD home and carried along by its swap"
        );

        // Swapped-out pages are byte-identical at their vacated slots.
        assert_eq!(cow.read_page(3).unwrap(), &rlo_orig[..], "R_lo took vacated slot 3 verbatim");
        assert_eq!(cow.read_page(5).unwrap(), &b_orig[..], "B moved to vacated slot 5 verbatim");
        assert_eq!(cow.read_page(6).unwrap(), &b2_orig[..], "B2 moved to vacated slot 6 verbatim");

        // Re-derive via the REAL finalize: the forest is consistent (fail-closed otherwise)
        // and already roots-first, with the exact ptrmap the relocated forest implies.
        let moved_again = crate::ptrmap_build::finalize(&mut cow).unwrap();
        assert!(!moved_again, "finalize sees a forest already in roots-first order");

        let ptrmap = cow.read_page(2).unwrap().to_vec();
        assert_eq!(ptrmap_entry(&ptrmap, 3), (PTRMAP_ROOTPAGE, 0), "page 3 is now root R_lo");
        assert_eq!(ptrmap_entry(&ptrmap, 4), (PTRMAP_ROOTPAGE, 0), "page 4 is now root R_hi");
        assert_eq!(ptrmap_entry(&ptrmap, 5), (PTRMAP_BTREE, 4), "B is a child of R_hi at 4");
        assert_eq!(ptrmap_entry(&ptrmap, 6), (PTRMAP_BTREE, 4), "B2 is a child of R_hi at 4");

        let p1 = cow.read_page(1).unwrap().to_vec();
        let hdr = DatabaseHeader::read((&p1[..HEADER_SIZE]).try_into().unwrap()).unwrap();
        assert_eq!(hdr.largest_root_btree, 4, "offset 52 == the largest root after relocation");

        cow.rollback().unwrap();
    }

    /// A lone freelist trunk page: `next = 0`, `leaf_count = 0` (the two leading big-endian
    /// u32s a trunk page holds), the rest zero — a valid empty trunk (`parse_trunk` reads 0
    /// leaves). Used as the pre-existing free target in the mixed pass.
    fn free_trunk_page() -> Vec<u8> {
        zeros()
    }

    /// Point page 1's header freelist at trunk `head` with `count` free pages, preserving
    /// every other header field.
    fn set_freelist_head(cow: &mut Cow<MemStore>, head: u32, count: u32) {
        let mut page1 = cow.read_page(1).unwrap().to_vec();
        let head_bytes: &mut [u8; HEADER_SIZE] = (&mut page1[..HEADER_SIZE]).try_into().unwrap();
        let mut header = DatabaseHeader::read(head_bytes).unwrap();
        header.first_freelist_trunk = head;
        header.freelist_count = count;
        header.write(head_bytes);
        cow.stage_write(1, &page1).unwrap();
    }
}
