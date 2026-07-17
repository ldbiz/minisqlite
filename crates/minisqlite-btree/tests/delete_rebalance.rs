//! End-to-end tests that table-b-tree DELETE **rebalances an under-min interior** rather
//! than leaving it below the on-disk minimum. fileformat2 §1.6: every NON-ROOT interior
//! b-tree page must hold `K >= 2` keys (`>= 3` child pointers); "the only exception is
//! when page 1 is an interior b-tree page". `table_delete` must therefore, after every
//! delete, rotate or merge an interior that a removed child pushed below `K >= 2` — never
//! leave a `K < 2` non-root pass-through (which real sqlite never writes and which wastes
//! a level of indirection, deepening the tree under delete-heavy workloads).
//!
//! The `walk` validator here adds the assertion the other `tests/*.rs` walkers omit: a
//! non-root interior page has `cell_count() >= 2`. The ROOT is EXEMPT — a two-child
//! (`K == 1`) interior root is valid per §1.6 and must be accepted, not forced to `K >= 2`
//! nor wrongly collapsed — so the check keys off `id != root`.
//!
//! Each `tests/*.rs` is its own crate, so the structural validator and builder are copied
//! here (as in `tests/table_fill.rs`) rather than shared — a small duplication that keeps
//! the file self-contained and lets `walk` assert exactly these post-delete invariants.
//! Small (512-byte) pages force small fan-out and real interior splits, so the tree reaches
//! `height >= 3` and the `K >= 2` check runs on interiors BELOW the root.

use std::collections::BTreeSet;

use minisqlite_btree::{create_table_btree, init_database, table_delete, table_insert, TableCursor};
use minisqlite_fileformat::{PageType, PageView};
use minisqlite_pager::{MemPager, PageId, Pager};

/// Multiplier/increment of the 64-bit LCG (Knuth MMIX) all deterministic generators here
/// step with, so the payload/shuffle/property-mix PRNGs share one definition instead of
/// re-spelling the two magic constants at each call site.
const LCG_MUL: u64 = 6364136223846793005;
const LCG_INC: u64 = 1442695040888963407;

/// A deterministic ~50-byte payload for `rowid`, so a scan can check bytes exactly and a
/// surviving row proves it kept its own payload (not a neighbour's) across page rebuilds.
fn payload_for(rowid: i64) -> Vec<u8> {
    let mut v = format!("row-{rowid}-").into_bytes();
    let fill = (rowid as u8).wrapping_mul(31).wrapping_add(7);
    while v.len() < 50 {
        v.push(fill);
    }
    v
}

/// A deterministic payload of `len` bytes seeded by `rowid`, so the delete/insert mix can
/// vary leaf fill (short rows pack many per leaf, long rows few) and still read back
/// byte-for-byte. Stays inline at 512-byte pages (well under the overflow threshold).
fn var_payload(rowid: i64, len: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(len);
    let mut state = (rowid as u64).wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for _ in 0..len {
        state = state.wrapping_mul(LCG_MUL).wrapping_add(LCG_INC);
        v.push((state >> 40) as u8);
    }
    v
}

/// A deterministic shuffle of `1..=n` (a fixed LCG permutation) so tests exercise
/// out-of-order build and delete without a PRNG dependency.
fn shuffled(n: i64) -> Vec<i64> {
    let mut ids: Vec<i64> = (1..=n).collect();
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut next = || {
        state = state.wrapping_mul(LCG_MUL).wrapping_add(LCG_INC);
        (state >> 33) as usize
    };
    for i in (1..ids.len()).rev() {
        let j = next() % (i + 1);
        ids.swap(i, j);
    }
    ids
}

/// Scan the whole tree via `first` + `next`, returning the rowids in order.
fn scan_rowids(pager: &dyn Pager, root: PageId) -> Vec<i64> {
    let mut cur = TableCursor::open(pager, root).unwrap();
    let mut out = Vec::new();
    if !cur.first().unwrap() {
        return out;
    }
    loop {
        out.push(cur.rowid());
        if !cur.next().unwrap() {
            break;
        }
    }
    out
}

/// Scan the whole tree BACKWARDS via `last` + `prev`, returning the rowids in descending
/// order. A retarget/split that corrupted a right-link or a separator would desync the
/// forward and reverse scans, so checking both pins the post-delete tree from both ends.
fn scan_rowids_reverse(pager: &dyn Pager, root: PageId) -> Vec<i64> {
    let mut cur = TableCursor::open(pager, root).unwrap();
    let mut out = Vec::new();
    if !cur.last().unwrap() {
        return out;
    }
    loop {
        out.push(cur.rowid());
        if !cur.prev().unwrap() {
            break;
        }
    }
    out
}

/// Read a single row's payload by rowid through `TableCursor::seek_exact`, or `None` if
/// absent. Proves a row survives byte-for-byte without walking the whole table.
fn read_row(pager: &dyn Pager, root: PageId, rowid: i64) -> Option<Vec<u8>> {
    let mut cur = TableCursor::open(pager, root).unwrap();
    if cur.seek_exact(rowid).unwrap() {
        Some(cur.payload().unwrap().into_owned())
    } else {
        None
    }
}

/// `first_freelist_trunk` from page 1's header (offset 32, big-endian u32): zero when the
/// freelist is empty, non-zero once any page has been freed. Lets a test assert a delete
/// that MERGED interiors actually reclaimed the merged-away pages rather than leaking them.
fn freelist_trunk(pager: &MemPager) -> u32 {
    let page = pager.read_page(1).unwrap();
    u32::from_be_bytes([page[32], page[33], page[34], page[35]])
}

/// `freelist_count` from page 1's header (offset 36, big-endian u32): the total number of
/// pages (trunks + leaves) currently on the freelist.
fn freelist_count(pager: &MemPager) -> u32 {
    let page = pager.read_page(1).unwrap();
    u32::from_be_bytes([page[36], page[37], page[38], page[39]])
}

/// Recursively validate the table b-tree rooted at `root` and return its in-order rowids
/// and height (leaf = 1). `id` is the page being visited; `root` is threaded unchanged so
/// the `K >= 2` check can exempt the root. Asserts the full table-b-tree contract DELETE
/// must preserve:
///   - **§1.6 `K >= 2` for every NON-ROOT interior page** (`cell_count() >= 2`) — the core
///     invariant DELETE must maintain; the root is exempt (a two-child root has
///     `K == 1`, which §1.6 permits and which cannot be forced to `>= 2`).
///   - every interior separator equals the largest rowid in its left child's subtree;
///   - keys are strictly ascending within a page and across children in order;
///   - all child subtrees have equal height (balanced);
///   - no interior points at an empty child.
fn walk(pager: &dyn Pager, id: PageId, root: PageId, usable: usize) -> (Vec<i64>, u32) {
    let (is_leaf, leaf_rowids, children, keys) = {
        let view = PageView::new(pager.read_page(id).unwrap(), id, usable).unwrap();
        assert!(view.page_type().is_table(), "page {id} is not a table page");
        if view.page_type().is_leaf() {
            let count = view.cell_count() as usize;
            let mut rowids = Vec::with_capacity(count);
            for i in 0..count {
                rowids.push(view.table_leaf_cell(i).unwrap().rowid);
            }
            (true, rowids, Vec::new(), Vec::new())
        } else {
            let count = view.cell_count() as usize;
            // §1.6: a NON-ROOT interior page must hold K >= 2 keys (>= 3 children). The root
            // is exempt — a shallow tree's root can be a K==1 (2-child) interior, which real
            // sqlite also writes and which cannot be forced to >= 2.
            if id != root {
                assert!(
                    count >= 2,
                    "non-root interior page {id} has K={count} (< 2): violates fileformat2 §1.6 \
                     (DELETE must rotate/merge an under-min interior, not leave a pass-through)"
                );
            }
            let mut children = Vec::with_capacity(count + 1);
            let mut keys = Vec::with_capacity(count);
            for i in 0..count {
                let c = view.table_interior_cell(i).unwrap();
                children.push(c.left_child);
                keys.push(c.rowid);
            }
            children.push(view.right_most_pointer().expect("interior must have a right-most pointer"));
            (false, Vec::new(), children, keys)
        }
    };

    if is_leaf {
        for w in leaf_rowids.windows(2) {
            assert!(w[0] < w[1], "leaf {id} rowids not strictly ascending: {w:?}");
        }
        return (leaf_rowids, 1);
    }

    for w in keys.windows(2) {
        assert!(w[0] < w[1], "interior {id} separators not strictly ascending: {w:?}");
    }
    let mut all = Vec::new();
    let mut child_height: Option<u32> = None;
    for j in 0..keys.len() {
        let (sub, h) = walk(pager, children[j], root, usable);
        assert!(!sub.is_empty(), "child {} of interior {id} is empty (interior->empty child)", children[j]);
        assert_eq!(
            *sub.last().unwrap(),
            keys[j],
            "interior {id} cell {j}: separator must equal left subtree max"
        );
        if j > 0 {
            assert!(
                *sub.first().unwrap() > keys[j - 1],
                "interior {id} cell {j}: subtree must be > previous separator"
            );
        }
        match child_height {
            None => child_height = Some(h),
            Some(ph) => assert_eq!(ph, h, "interior {id} children heights differ (unbalanced)"),
        }
        all.extend(sub);
    }
    let (sub, h) = walk(pager, *children.last().unwrap(), root, usable);
    assert!(!sub.is_empty(), "right child of interior {id} is empty (interior->empty child)");
    if let Some(&last_key) = keys.last() {
        assert!(
            *sub.first().unwrap() > last_key,
            "interior {id}: right subtree must be > last separator"
        );
    }
    match child_height {
        None => child_height = Some(h),
        Some(ph) => assert_eq!(ph, h, "interior {id} children heights differ (unbalanced)"),
    }
    all.extend(sub);

    for w in all.windows(2) {
        assert!(w[0] < w[1], "interior {id} in-order rowids not ascending across children");
    }
    (all, child_height.unwrap() + 1)
}

/// Build a table (root allocated by `create_table_btree`, i.e. NOT page 1), insert `ids`
/// in the given order with `payload_for`, and return `(pager, root)`.
fn build_table(page_size: u32, ids: &[i64]) -> (MemPager, PageId) {
    let mut pager = MemPager::new(page_size);
    init_database(&mut pager).unwrap();
    let root = create_table_btree(&mut pager).unwrap();
    for &id in ids {
        table_insert(&mut pager, root, id, &payload_for(id)).unwrap();
    }
    (pager, root)
}

/// Validate structure (via `walk`, which fires the non-root `K >= 2` assertion) and prove
/// the tree scans to exactly `expected` (sorted here — `expected` need not be sorted).
fn assert_valid(pager: &MemPager, root: PageId, expected: &[i64]) {
    let usable = pager.page_size() as usize;
    let (in_order, _height) = walk(pager, root, root, usable);
    let mut sorted = expected.to_vec();
    sorted.sort_unstable();
    assert_eq!(in_order, sorted, "structural walk must equal the surviving rowids");
    assert_eq!(scan_rowids(pager, root), sorted, "cursor scan must equal the surviving rowids");
}

// -------------------------------------------------------------------------------------

#[test]
fn delete_down_to_a_few_keeps_every_non_root_interior_valid() {
    // The primary regression. Build a height>=3 tree, then delete rows in a deterministic
    // pseudo-random order down to a handful, validating (walk + scan) after each delete
    // batch. Under the OLD code, at some batch a removed child leaves a K<2 non-root
    // interior pass-through and `walk`'s §1.6 assertion fails; the rebalance keeps every
    // non-root interior at K>=2 through the whole sequence.
    let n = 4000i64;
    let (mut pager, root) = build_table(512, &(1..=n).collect::<Vec<_>>());
    let usable = pager.page_size() as usize;
    let (_o, h0) = walk(&pager, root, root, usable);
    assert!(h0 >= 3, "want height>=3 so the invariant runs on interiors below the root, got {h0}");

    let order = shuffled(n);
    let mut remaining: BTreeSet<i64> = (1..=n).collect();
    let batch = 200usize;
    for (i, &id) in order.iter().enumerate() {
        assert!(table_delete(&mut pager, root, id).unwrap(), "present row {id} must delete");
        remaining.remove(&id);
        // Validate at each batch boundary and on every step once down to a handful.
        if (i + 1) % batch == 0 || remaining.len() <= 5 {
            let expect: Vec<i64> = remaining.iter().copied().collect();
            assert_valid(&pager, root, &expect);
        }
        if remaining.len() <= 3 {
            break; // "down to a handful"
        }
    }
    // Whatever handful remains is still a valid, scannable tree.
    let expect: Vec<i64> = remaining.iter().copied().collect();
    assert_valid(&pager, root, &expect);
}

#[test]
fn delete_contiguous_range_forces_interior_merges() {
    // Deleting a large CONTIGUOUS middle range empties whole subtrees and forces adjacent
    // interiors to underflow together — the case rotations cannot always absorb, so real
    // interior MERGES fire. Beyond staying valid, this pins the merge path's *freeing*: a
    // merge must orphan its dead sibling page onto the freelist (a forgotten `orphans.push`
    // leaks silently — `walk` still passes and `MemPager` just grows), and a later insert
    // must REUSE those pages instead of ballooning the file. (The §1.6 K>=2 regression
    // itself is witnessed by tests 1/4/5, whose `walk` goes red on the old pass-through code.)
    let n = 4000i64;
    let (mut pager, root) = build_table(512, &(1..=n).collect::<Vec<_>>());
    let usable = pager.page_size() as usize;
    let (_o, h0) = walk(&pager, root, root, usable);
    assert!(h0 >= 3, "want a height>=3 tree so interior merges fire below the root, got {h0}");
    let pc_full = pager.page_count().unwrap();

    let (lo, hi) = (1000i64, 3000i64);
    for id in lo..=hi {
        assert!(table_delete(&mut pager, root, id).unwrap(), "row {id} must delete");
    }
    let expect: Vec<i64> = (1..lo).chain((hi + 1)..=n).collect();
    assert_valid(&pager, root, &expect);

    // The height must have shrunk from all those merges/collapses (a delete-heavy workload
    // must not leave a needlessly deep tree of pass-throughs).
    let (_o2, h1) = walk(&pager, root, root, usable);
    assert!(h1 <= h0, "a large contiguous delete must not deepen the tree ({h1} vs {h0})");

    // The merged-away / emptied pages went onto the freelist (not leaked); freeing does not
    // shrink the file (pages move to the freelist in place), so page_count is unchanged.
    assert_ne!(freelist_trunk(&pager), 0, "merging/emptying interiors must free pages onto the freelist");
    let freed = freelist_count(&pager);
    assert!(freed >= 5, "a 2000-row contiguous delete should free many pages, freed {freed}");
    assert_eq!(pager.page_count().unwrap(), pc_full, "free_page must not shrink the file");

    // Re-insert the deleted range: allocations must consume the freelist before growing the
    // file. Reuse is not byte-perfect for a partial delete (half-full edge leaves are not
    // repacked as densely as the original sequential build), so the file may grow by a few
    // pages; the proof of reclamation is that it grew by FAR less than `freed` (a leak would
    // grow by ~`freed`) and that the reinsert drained the freelist it reused.
    for id in lo..=hi {
        table_insert(&mut pager, root, id, &payload_for(id)).unwrap();
    }
    assert_valid(&pager, root, &(1..=n).collect::<Vec<_>>());
    let growth = pager.page_count().unwrap().saturating_sub(pc_full);
    assert!(
        growth * 2 < freed,
        "re-inserting the merged-away range must reuse freed pages, not leak \
         (grew {growth} pages, but {freed} had been freed; a leak would grow by ~{freed})"
    );
    let remaining_free = freelist_count(&pager);
    assert!(
        remaining_free < freed,
        "the reinsert must drain the freed pages it reused (freelist now {remaining_free}, was {freed})"
    );
}

#[test]
fn mass_delete_all_then_reinsert() {
    // Delete every rowid (shuffled): the tree collapses back to a single EMPTY LEAF at the
    // same root id, and a scan yields nothing. Then re-insert a subset and require it reads
    // back and validates — proving the collapsed root is a clean, reusable empty table.
    let n = 2000i64;
    let (mut pager, root) = build_table(512, &shuffled(n));
    let usable = pager.page_size() as usize;
    let (_o, h0) = walk(&pager, root, root, usable);
    assert!(h0 >= 3, "want a multi-level tree to collapse, got height {h0}");

    for id in shuffled(n) {
        assert!(table_delete(&mut pager, root, id).unwrap(), "row {id} must delete");
    }
    {
        let view = PageView::new(pager.read_page(root).unwrap(), root, usable).unwrap();
        assert_eq!(view.page_type(), PageType::LeafTable, "collapsed root must be a table leaf");
        assert_eq!(view.cell_count(), 0, "collapsed root leaf must be empty");
    }
    assert!(scan_rowids(&pager, root).is_empty(), "empty tree scans to nothing");
    assert!(!table_delete(&mut pager, root, 1).unwrap(), "delete on empty tree is false");

    // Re-insert a subset (every 7th rowid) and require it validates and reads back.
    let reinsert: Vec<i64> = (1..=n).filter(|r| r % 7 == 0).collect();
    for &id in &reinsert {
        table_insert(&mut pager, root, id, &payload_for(id)).unwrap();
    }
    assert_valid(&pager, root, &reinsert);
    for &id in &reinsert {
        assert_eq!(
            read_row(&pager, root, id).as_deref(),
            Some(payload_for(id).as_slice()),
            "re-inserted row {id} must read back byte-for-byte"
        );
    }
}

#[test]
fn randomized_delete_insert_mix_stays_valid() {
    // A long, deterministic mix of inserts (some REPLACE) and deletes on 512-byte pages,
    // with VARIED payload sizes so leaves have variable fill (short rows pack many, long
    // rows few) — a broad distribution of split/underflow/rotate/merge shapes. A reference
    // BTreeSet of the live rowids is the oracle: after each periodic checkpoint the walk
    // (K>=2 invariant) and a full scan must both equal it. This is the property test that
    // proves no row is ever lost, duplicated, or misordered across the churn.
    let mut pager = MemPager::new(512);
    init_database(&mut pager).unwrap();
    let root = create_table_btree(&mut pager).unwrap();
    let usable = pager.page_size() as usize;

    let mut live: BTreeSet<i64> = BTreeSet::new();
    let key_space = 3000i64;
    let steps = 20_000u64;
    let mut state: u64 = 0xDEAD_BEEF_CAFE_F00D;
    let mut next = || {
        state = state.wrapping_mul(LCG_MUL).wrapping_add(LCG_INC);
        state
    };
    let mut max_height = 1u32;

    for step in 0..steps {
        let r = next();
        let id = (r % key_space as u64) as i64 + 1;
        // ~55% insert / 45% delete, so the tree grows to a multi-level equilibrium then
        // churns there (where rotate/merge fire), rather than staying tiny.
        let do_insert = (r >> 20) % 100 < 55;
        if do_insert {
            let len = 8 + ((r >> 30) % 300) as usize; // 8..=307 bytes: variable leaf fill, inline
            table_insert(&mut pager, root, id, &var_payload(id, len)).unwrap();
            live.insert(id);
        } else if live.remove(&id) {
            assert!(table_delete(&mut pager, root, id).unwrap(), "present {id} must delete");
        } else {
            assert!(!table_delete(&mut pager, root, id).unwrap(), "absent {id} must delete to false");
        }

        if step % 500 == 0 {
            let expect: Vec<i64> = live.iter().copied().collect();
            let (in_order, h) = walk(&pager, root, root, usable);
            assert_eq!(in_order, expect, "walk must equal the live set at step {step}");
            assert_eq!(scan_rowids(&pager, root), expect, "scan must equal the live set at step {step}");
            max_height = max_height.max(h);
        }
    }

    // Final full validation, and a check that the churn actually built multi-level trees
    // (so the interior rotate/merge paths really ran, not just leaf-only edits).
    let expect: Vec<i64> = live.iter().copied().collect();
    assert_valid(&pager, root, &expect);
    assert!(
        max_height >= 3,
        "the mix must reach a height>=3 tree so interior rebalancing is exercised, saw {max_height}"
    );
}

/// A deterministic LCG shuffle of an arbitrary `i64` slice in place (the `shuffled(n)`
/// permutation generalised to any values, so the negative+positive mix builds out of
/// order without a PRNG dependency). Same MMIX constants and seed as `shuffled`.
fn shuffle_in_place(ids: &mut [i64]) {
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut next = || {
        state = state.wrapping_mul(LCG_MUL).wrapping_add(LCG_INC);
        (state >> 33) as usize
    };
    for i in (1..ids.len()).rev() {
        let j = next() % (i + 1);
        ids.swap(i, j);
    }
}

#[test]
fn delete_max_with_negative_predecessor_splits_interior_not_errors() {
    // PRIMARY differential regression (table side; mirrors the index-side
    // `delete_interior_divider_with_larger_predecessor_must_not_error`).
    //
    // Deleting a subtree's MAX rowid retargets the bordering interior separator to the new
    // max — the deleted row's in-order predecessor. A table separator stores the rowid as a
    // varint: 1 byte for a small positive, 9 bytes for ANY negative (the i64 cast to u64
    // sets the high bit). The predecessor of a value is <= it, so among positives a retarget
    // only ever NARROWS the separator — EXCEPT at the single sign boundary: rowid `1`'s
    // in-order predecessor (with `0` absent) is `-1`, so retargeting `1 -> -1` GROWS the
    // separator by 8 bytes. On an interior already packed to within 8 bytes of full, the
    // rebuilt page no longer fits. Real sqlite SPLITS and the DELETE succeeds; the pre-fix
    // engine returned `Err("delete: interior overflow retargeting a separator")`, aborting a
    // valid DELETE — a differential divergence (Err vs success).
    //
    // Build rowids `-n..=-1` and `1..=n` (0 absent) shuffled into a height>=3 tree on
    // 512-byte pages, then delete rowid 1. Whether the boundary interior is packed to
    // within 8 bytes of full — so the +8 retarget overflows — depends on how the mixed
    // 9-byte (negative) and small (positive) separators land, i.e. on (n, payload width);
    // there is exactly ONE sign boundary, so unlike the index side we cannot scatter many
    // triggers. So SWEEP (payload width, n) and take the FIRST config whose delete forces
    // the split, detected by the file high-water page count GROWING across the delete: a
    // pure DELETE never allocates EXCEPT through this overflow->split branch (rotate/merge
    // only free pages), so a grown page count PROVES the split fired — the same dodge-proof
    // signal the index test uses. The break-on-first-hit keeps the common case fast (the
    // first swept width triggers within tens of iterations); the wider tail only runs if
    // packing drift moved the trigger. Under the PRE-FIX code the FIRST triggering config's
    // `table_delete(..).unwrap()` panics with the interior-overflow message (the pre-fix
    // failure); after the fix it returns Ok(true) and the tree stays valid (the post-fix pass).
    let mut saw_split = false;
    'sweep: for &plen in &[32usize, 36, 40, 44, 48] {
        for n in 740i64..=1400 {
            let mut rowids: Vec<i64> = (-n..=-1).chain(1..=n).collect();
            shuffle_in_place(&mut rowids);

            let mut pager = MemPager::new(512);
            init_database(&mut pager).unwrap();
            let root = create_table_btree(&mut pager).unwrap();
            for &id in &rowids {
                table_insert(&mut pager, root, id, &var_payload(id, plen)).unwrap();
            }
            let usable = pager.page_size() as usize;
            let (_o, h) = walk(&pager, root, root, usable);
            assert!(h >= 3, "want height>=3 so the retarget hits a real non-root interior (n={n}), got {h}");

            let pages_before = pager.page_count().unwrap();
            // Pre-fix: on the first triggering (plen, n) this unwrap PANICS with
            // "delete: interior overflow retargeting a separator". Post-fix: Ok(true).
            assert!(
                table_delete(&mut pager, root, 1).unwrap(),
                "deleting the present boundary rowid 1 must return true (n={n}, plen={plen})"
            );
            if pager.page_count().unwrap() <= pages_before {
                continue; // this config's retarget still fit one page — not the witness
            }

            // Witness config: the retarget overflowed and the split branch allocated a page.
            // Validate the whole post-delete tree from both ends.
            saw_split = true;
            let expected: Vec<i64> = (-n..=-1).chain(2..=n).collect(); // sorted survivors (no 1)
            assert_valid(&pager, root, &expected); // walk (non-root K>=2) + forward scan
            let mut desc = expected.clone();
            desc.reverse();
            assert_eq!(
                scan_rowids_reverse(&pager, root),
                desc,
                "reverse cursor scan must equal the survivors in descending order (n={n}, plen={plen})"
            );
            // Re-deleting rowid 1 is now a no-op (it is gone), and a neighbour still deletes,
            // proving the split left a coherent, mutable tree.
            assert!(!table_delete(&mut pager, root, 1).unwrap(), "re-deleting the gone rowid 1 is false");
            break 'sweep;
        }
    }
    assert!(
        saw_split,
        "no swept (payload width, n) packed the boundary interior to within 8 bytes of full, \
         so the +8 retarget never overflowed; widen the sweep until a delete forces the split"
    );
}

#[test]
fn delete_rotate_widen_parent_overflow_splits_not_errors() {
    // SECONDARY hardening regression (same class as the primary, one level up). When an
    // under-min interior is rebalanced by a ROTATE, the rotate installs a sibling's max as
    // a PARENT separator. If that separator re-encodes WIDER than the one it replaces (the
    // rotated-in max crosses a varint-width boundary: a small positive giving way to a
    // negative 9-byte one, or to a huge >= 2^56 9-byte positive) it can push a near-full
    // parent past one page. The pre-fix rotate paths wrote the parent through
    // `write_interior`, which fail-closed with `Err("delete: rebalanced interior overflowed
    // one page")` — aborting a valid DELETE, exactly the differential divergence the primary
    // fix removed. `rebalance` now routes that parent overflow through `write_parent_or_split`
    // (the same split machinery), so the DELETE splits and succeeds.
    //
    // The trigger needs a coincidence — a near-full boundary parent AND an adjacent interior
    // that underflows next to a still-lendable (>= 4-child) sibling straddling the boundary —
    // that a single delete cannot stage; it emerges mid-sequence in a delete-heavy workload.
    // This exact `(payload width, counts, delete order)` was isolated by a pattern search as
    // a deterministic witness: it drives the rotate-widen parent overflow (verified: pre-fix
    // it panics here with the `write_interior` overflow error; post-fix every delete
    // succeeds). Rowids span BOTH varint-width jumps — negatives (9 B) | small positives
    // 1..=100 (1-2 B) | huge positives >= 2^60 (9 B) — so both the left-rotate (small→neg)
    // and right-rotate (small→huge) widenings are in play.
    //
    // COVERAGE CAVEAT — READ BEFORE CHANGING INSERT/SPLIT PACKING. Unlike the primary
    // single-delete test (which has an always-on dodge-proof: a grown high-water page count),
    // this delete-heavy test has NO in-test signal that the rotate-widen -> split branch was
    // actually taken. A grown page count is unusable here because the split reuses pages the
    // sequence has already freed. The ONLY proof that the fix is exercised is the recorded
    // fail->pass transition: with the fix reverted, this test PANICS on the delete below with
    // `Err("delete: rebalanced interior overflowed one page")`; with the fix it succeeds.
    // Consequence: if a future change to insert/build packing or rotate/merge thresholds
    // shifts this hand-found witness OFF the rotate-widen path, the test keeps passing as a
    // plain delete-everything test while silently NO LONGER covering `write_parent_or_split`'s
    // split arm. If you touch that packing, RE-VERIFY coverage by temporarily reverting the
    // secondary fix and confirming this test still fails with that exact error; if it does
    // not, re-derive the witness counts/order. The always-on invariant this test does pin is
    // that every delete returns Ok and the tree stays a valid table b-tree (non-root interiors
    // K>=2) throughout, ending empty — a correctness check, not a branch-fired check.
    let huge_base: i64 = 1 << 60; // 9-byte varint positives, above the 2^56 varint jump
    let (nneg, nsmall, nhuge) = (1000i64, 100i64, 1000i64);
    let plen = 12usize;
    let all: Vec<i64> =
        (-nneg..=-1).chain(1..=nsmall).chain(huge_base..huge_base + nhuge).collect();

    // Two witness delete orders (both isolated by the search as driving the overflow):
    // "smalls, then negatives, then huges" and "smalls, then huges desc, then negatives desc".
    let orders: [Vec<i64>; 2] = [
        (1..=nsmall).chain(-nneg..=-1).chain(huge_base..huge_base + nhuge).collect(),
        (1..=nsmall)
            .chain((huge_base..huge_base + nhuge).rev())
            .chain((-nneg..=-1).rev())
            .collect(),
    ];

    for order in &orders {
        let mut pager = MemPager::new(512);
        init_database(&mut pager).unwrap();
        let root = create_table_btree(&mut pager).unwrap();
        let mut build = all.clone();
        shuffle_in_place(&mut build);
        for &id in &build {
            table_insert(&mut pager, root, id, &var_payload(id, plen)).unwrap();
        }
        let usable = pager.page_size() as usize;
        let (_o, h0) = walk(&pager, root, root, usable);
        assert!(h0 >= 3, "want a height>=3 tree so interior rotates fire below the root, got {h0}");

        let mut live: BTreeSet<i64> = all.iter().copied().collect();
        for (di, &id) in order.iter().enumerate() {
            // Pre-fix, the rotate-widen parent overflow makes this unwrap panic with
            // "delete: rebalanced interior overflowed one page". Post-fix it splits and
            // returns true.
            assert!(table_delete(&mut pager, root, id).unwrap(), "present {id} must delete");
            live.remove(&id);
            // Periodic structural + scan validation (every delete is too slow for 2100 rows).
            if di % 200 == 0 {
                let expect: Vec<i64> = live.iter().copied().collect();
                assert_valid(&pager, root, &expect);
            }
        }
        assert!(live.is_empty(), "the order must delete every row");
        assert!(scan_rowids(&pager, root).is_empty(), "the fully-deleted tree scans to nothing");
        let view = PageView::new(pager.read_page(root).unwrap(), root, usable).unwrap();
        assert_eq!(view.page_type(), PageType::LeafTable, "collapsed root must be a table leaf");
        assert_eq!(view.cell_count(), 0, "collapsed root leaf must be empty");
    }
}

#[test]
fn k1_interior_root_after_deletes_is_accepted() {
    // Drive a tree DOWN (via deletes) to the shape where the ROOT is a K==1 (2-child)
    // interior, and require `walk` (root-exempt) to accept it with a correct scan. As the
    // tree collapses it must pass through a two-child root: a node loses children one at a
    // time, so the root goes ...->3->2->collapse, and it cannot skip 2. This pins that the
    // root exemption is load-bearing — the code neither forces the root to K>=2 nor wrongly
    // collapses a valid two-child root.
    let n = 1500i64;
    let (mut pager, root) = build_table(512, &shuffled(n));
    let usable = pager.page_size() as usize;
    let (_o, h0) = walk(&pager, root, root, usable);
    assert!(h0 >= 3, "want a multi-level tree to shrink through a K==1 root, got {h0}");

    let mut remaining: BTreeSet<i64> = (1..=n).collect();
    let mut proved = false;
    for &id in &shuffled(n) {
        assert!(table_delete(&mut pager, root, id).unwrap(), "row {id} must delete");
        remaining.remove(&id);
        let (is_interior, kcount) = {
            let view = PageView::new(pager.read_page(root).unwrap(), root, usable).unwrap();
            (view.page_type() == PageType::InteriorTable, view.cell_count())
        };
        if is_interior && kcount == 1 {
            // A genuine K==1 (2-child) interior root reached via deletes: valid, exempt from
            // the >=2 rule, and scans exactly to the survivors.
            let expect: Vec<i64> = remaining.iter().copied().collect();
            assert_valid(&pager, root, &expect);
            proved = true;
            break;
        }
    }
    assert!(proved, "a full shuffled delete must pass through a K==1 (2-child) interior root");
}
