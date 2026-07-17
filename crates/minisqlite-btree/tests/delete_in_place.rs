//! Tests for the FAST in-place DELETE route (`table_delete` under an open transaction,
//! which reaches `Pager::page_mut` + the fileformat in-place cell editor `delete_cell`).
//! Every OTHER btree test calls `table_delete` in auto-commit, where a returned mutable
//! page borrow cannot commit, so `page_mut` fails closed and the path falls back to the
//! whole-page `PageBuilder` rebuild — meaning those tests only cover the FALLBACK route.
//! These wrap the deletes in `begin()`/`commit()` so the in-place drop actually runs, and
//! prove:
//!
//! 1. **Logical equivalence** — the same delete sequence run in-transaction (fast route)
//!    and in auto-commit (fallback route) yields identical scans, identical re-parsed
//!    tree structure (rowids + separator invariant), and identical page counts. The fast
//!    route must not change a single query result or the tree's shape.
//! 2. **The fast route really ran** — an in-place middle delete leaves a FREEBLOCK behind
//!    (offset 1 of the page header is non-zero), which the defragmenting `PageBuilder`
//!    rebuild NEVER produces. Same workload, different fingerprint ⇒ the routes differ.
//! 3. **Free-byte accounting** — after in-place deletes (and delete+reinsert) the leaf's
//!    total reclaimable free space still satisfies the spec identity
//!    `total == usable - header - cell_pointer_array - Σ cell_len`, so a delete-heavy file
//!    stays consistent for an integrity check.
//! 4. **Overflow chains** — a deleted SPILLED row's overflow chain is freed by the fast
//!    route (which touches only the leaf cell) and reused on reinsert, never leaked.

use minisqlite_btree::{create_table_btree, init_database, table_delete, table_insert, TableCursor};
use minisqlite_fileformat::{leaf_cell_len, leaf_free_space, PageType, PageView};
use minisqlite_pager::{MemPager, PageId, Pager};

/// A deterministic ~50-byte payload for `rowid` (mirrors `insert_in_place.rs`).
fn payload_for(rowid: i64) -> Vec<u8> {
    let mut v = format!("row-{rowid}-").into_bytes();
    let fill = (rowid as u8).wrapping_mul(31).wrapping_add(7);
    while v.len() < 50 {
        v.push(fill);
    }
    v
}

/// A deterministic pseudo-random payload of `len` bytes, distinct per `seed`. Large `len`
/// spills onto an overflow chain, so a round-trip checks every byte.
fn big_payload(len: usize, seed: u64) -> Vec<u8> {
    let mut v = Vec::with_capacity(len);
    let mut state = seed.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for _ in 0..len {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        v.push((state >> 40) as u8);
    }
    v
}

/// A deterministic shuffle of `1..=n` (fixed LCG permutation), no PRNG dependency.
fn shuffled(n: i64) -> Vec<i64> {
    let mut ids: Vec<i64> = (1..=n).collect();
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    for i in (1..ids.len()).rev() {
        let j = next() % (i + 1);
        ids.swap(i, j);
    }
    ids
}

/// Scan the whole tree via `first` + `next`, returning `(rowid, payload)` in order.
fn scan_all(pager: &dyn Pager, root: PageId) -> Vec<(i64, Vec<u8>)> {
    let mut cur = TableCursor::open(pager, root).unwrap();
    let mut out = Vec::new();
    if !cur.first().unwrap() {
        return out;
    }
    loop {
        out.push((cur.rowid(), cur.payload().unwrap().into_owned()));
        if !cur.next().unwrap() {
            break;
        }
    }
    out
}

/// Recursively re-parse every page as a table b-tree, asserting key order and the
/// separator invariant, and return the in-order rowids. Re-parsing through `PageView`
/// (which fails closed on a malformed page) is the spec-validity check: an in-place
/// delete that corrupted the header, cell-pointer array, or a freeblock would be caught
/// here or by the cursor scan.
fn valid_rowids(pager: &dyn Pager, id: PageId, usable: usize) -> Vec<i64> {
    let (is_leaf, leaf, children, keys) = {
        let view = PageView::new(pager.read_page(id).unwrap(), id, usable).unwrap();
        assert!(view.page_type().is_table(), "page {id} is not a table page");
        let count = view.cell_count() as usize;
        if view.page_type().is_leaf() {
            let rowids: Vec<i64> =
                (0..count).map(|i| view.table_leaf_cell(i).unwrap().rowid).collect();
            (true, rowids, Vec::new(), Vec::new())
        } else {
            let mut children = Vec::with_capacity(count + 1);
            let mut keys = Vec::with_capacity(count);
            for i in 0..count {
                let c = view.table_interior_cell(i).unwrap();
                children.push(c.left_child);
                keys.push(c.rowid);
            }
            children.push(view.right_most_pointer().unwrap());
            (false, Vec::new(), children, keys)
        }
    };
    if is_leaf {
        for w in leaf.windows(2) {
            assert!(w[0] < w[1], "leaf {id} rowids not strictly ascending");
        }
        return leaf;
    }
    for w in keys.windows(2) {
        assert!(w[0] < w[1], "interior {id} separators not strictly ascending");
    }
    let mut all = Vec::new();
    for j in 0..keys.len() {
        let sub = valid_rowids(pager, children[j], usable);
        assert_eq!(*sub.last().unwrap(), keys[j], "interior {id} sep must equal left subtree max");
        all.extend(sub);
    }
    all.extend(valid_rowids(pager, *children.last().unwrap(), usable));
    for w in all.windows(2) {
        assert!(w[0] < w[1], "interior {id} in-order rowids not ascending");
    }
    all
}

/// The page header's first-freeblock field (offset 1 of the b-tree header; page 1's
/// header starts at byte 100). Non-zero means the page carries a freeblock chain — the
/// fingerprint of an in-place delete that the `PageBuilder` rebuild never leaves.
fn first_freeblock(pager: &dyn Pager, id: PageId) -> u16 {
    let page = pager.read_page(id).unwrap();
    let ho = if id == 1 { 100 } else { 0 };
    ((page[ho + 1] as u16) << 8) | page[ho + 2] as u16
}

/// Assert the spec free-byte identity on a leaf table page: the total reclaimable free
/// space (`leaf_free_space` total = contiguous gap + freeblock chain + fragmented bytes)
/// equals `usable - header - cell_pointer_array - Σ cell_len`. This holds for ANY
/// spec-valid leaf regardless of physical layout, so it is exactly the accounting an
/// integrity check on a delete-heavy file relies on.
fn assert_free_accounting(pager: &dyn Pager, id: PageId, usable: usize) {
    let page = pager.read_page(id).unwrap();
    let view = PageView::new(page, id, usable).unwrap();
    assert!(view.page_type().is_leaf() && view.page_type().is_table(), "page {id} is a leaf table page");
    let n = view.cell_count() as usize;
    let h = if id == 1 { 100 } else { 0 };
    let mut sum_cell_len = 0usize;
    for i in 0..n {
        let off = view.cell_pointer(i);
        sum_cell_len += leaf_cell_len(page, id, usable, off).unwrap();
    }
    let total = leaf_free_space(page, id, usable).unwrap().total;
    assert_eq!(
        total,
        usable - h - 8 - 2 * n - sum_cell_len,
        "free-byte accounting on page {id}: total == usable - header - CPA(2n) - Σ cell_len",
    );
}

/// Build a table by inserting `insert_ids` (auto-commit), then delete `delete_ids` INSIDE
/// one transaction so the fast in-place route (`page_mut` + `delete_cell`) runs, then
/// commit. Inserts are auto-commit in BOTH build helpers, so only the DELETE route differs.
fn build_then_delete_in_txn(ps: u32, insert_ids: &[i64], delete_ids: &[i64]) -> (MemPager, PageId) {
    let mut pager = MemPager::new(ps);
    init_database(&mut pager).unwrap();
    let root = create_table_btree(&mut pager).unwrap();
    for &id in insert_ids {
        table_insert(&mut pager, root, id, &payload_for(id)).unwrap();
    }
    pager.begin().unwrap();
    for &id in delete_ids {
        assert!(table_delete(&mut pager, root, id).unwrap(), "rowid {id} must exist to delete");
    }
    pager.commit().unwrap();
    (pager, root)
}

/// Same as [`build_then_delete_in_txn`] but the deletes run in auto-commit, so `page_mut`
/// fails closed and every delete takes the FALLBACK whole-page rebuild route.
fn build_then_delete_autocommit(ps: u32, insert_ids: &[i64], delete_ids: &[i64]) -> (MemPager, PageId) {
    let mut pager = MemPager::new(ps);
    init_database(&mut pager).unwrap();
    let root = create_table_btree(&mut pager).unwrap();
    for &id in insert_ids {
        table_insert(&mut pager, root, id, &payload_for(id)).unwrap();
    }
    for &id in delete_ids {
        assert!(table_delete(&mut pager, root, id).unwrap(), "rowid {id} must exist to delete");
    }
    (pager, root)
}

/// The two delete routes must agree on every scanned row, the re-parsed tree structure,
/// and the page count. Physical layout differs (freeblocks vs defragmented) but nothing
/// observable does.
fn assert_delete_routes_agree(ps: u32, insert_ids: &[i64], delete_ids: &[i64]) {
    let (fast, froot) = build_then_delete_in_txn(ps, insert_ids, delete_ids);
    let (slow, sroot) = build_then_delete_autocommit(ps, insert_ids, delete_ids);
    let usable = ps as usize;
    assert_eq!(scan_all(&fast, froot), scan_all(&slow, sroot), "fast vs fallback delete scans disagree (ps {ps})");
    assert_eq!(
        valid_rowids(&fast, froot, usable),
        valid_rowids(&slow, sroot, usable),
        "fast vs fallback tree structure/rowids disagree (ps {ps})",
    );
    assert_eq!(
        fast.page_count().unwrap(),
        slow.page_count().unwrap(),
        "fast vs fallback page counts disagree (ps {ps})",
    );
}

#[test]
fn delete_middle_subset_in_place_matches_fallback() {
    // Delete every even rowid from a multi-page tree, in shuffled order. Hits middle
    // cells, some leaf maxima (separator retarget up the path), and empties some leaves
    // (interior rebalance) — all of which must match the fallback rebuild exactly.
    for &ps in &[512u32, 1024, 4096] {
        let insert_ids: Vec<i64> = (1..=1500).collect();
        let delete_ids: Vec<i64> = shuffled(1500).into_iter().filter(|&id| id % 2 == 0).collect();
        assert_delete_routes_agree(ps, &insert_ids, &delete_ids);
    }
}

#[test]
fn delete_max_of_each_leaf_retargets_matches_fallback() {
    // Delete rowids in DESCENDING order: each delete removes the current maximum of its
    // leaf, so `new_max` retargeting up the interior path runs on nearly every step.
    // A wrong `new_max` would break the separator invariant `valid_rowids` checks, or
    // desync the scans from the fallback.
    for &ps in &[512u32, 4096] {
        let insert_ids: Vec<i64> = (1..=1200).collect();
        let delete_ids: Vec<i64> = (1..=1200).rev().filter(|&id| id % 3 != 0).collect();
        assert_delete_routes_agree(ps, &insert_ids, &delete_ids);
    }
}

#[test]
fn delete_most_rows_triggers_collapse_matches_fallback() {
    // Delete all but a handful, in shuffled order → many empty leaves, interior merges,
    // and root height collapse. The fast in-place leaf drops and the (unchanged)
    // structural machinery must together reach the same logical state (scans, tree
    // structure) and page count as the fallback — not byte-for-byte (the fast route
    // leaves freeblocks the defragmenting rebuild does not).
    for &ps in &[512u32, 4096] {
        let insert_ids: Vec<i64> = (1..=2000).collect();
        let keep = [7i64, 999, 1500];
        let delete_ids: Vec<i64> =
            shuffled(2000).into_iter().filter(|id| !keep.contains(id)).collect();
        assert_delete_routes_agree(ps, &insert_ids, &delete_ids);
    }
}

#[test]
fn in_place_delete_leaves_a_freeblock_that_rebuild_never_does() {
    // PROOF the fast route ran (not a silent fallback): a mid-page in-place delete frees
    // the removed cell's slot as a FREEBLOCK (it is not at the content-area top). The
    // auto-commit rebuild always defragments, so its first-freeblock stays zero. Same
    // workload, different fingerprint ⇒ the routes are genuinely different.
    let ps = 4096u32;

    let mut fast = MemPager::new(ps);
    init_database(&mut fast).unwrap();
    let root = create_table_btree(&mut fast).unwrap();
    for id in 1..=5 {
        table_insert(&mut fast, root, id, &payload_for(id)).unwrap();
    }
    fast.begin().unwrap();
    assert!(table_delete(&mut fast, root, 3).unwrap()); // a middle cell
    fast.commit().unwrap();
    assert_ne!(
        first_freeblock(&fast, root),
        0,
        "in-place middle delete must leave a freeblock (the fast route ran)"
    );

    let mut slow = MemPager::new(ps);
    init_database(&mut slow).unwrap();
    let sroot = create_table_btree(&mut slow).unwrap();
    for id in 1..=5 {
        table_insert(&mut slow, sroot, id, &payload_for(id)).unwrap();
    }
    assert!(table_delete(&mut slow, sroot, 3).unwrap());
    assert_eq!(
        first_freeblock(&slow, sroot),
        0,
        "the rebuild route always defragments (no freeblock)"
    );

    // Despite the different physical layout, both read back the same rows, and the fast
    // page re-parses as a spec-valid leaf.
    assert_eq!(scan_all(&fast, root), scan_all(&slow, sroot));
    let view = PageView::new(fast.read_page(root).unwrap(), root, ps as usize).unwrap();
    assert_eq!(view.page_type(), PageType::LeafTable);
    assert_eq!(view.cell_count(), 4);
}

#[test]
fn free_byte_accounting_holds_after_delete_and_reinsert() {
    // The fast in-place route must keep the leaf's free-byte accounting consistent (the
    // spec identity), both while a freeblock is present (after a delete) and after a
    // reinsert that may defragment-on-demand.
    let ps = 4096u32;
    let usable = ps as usize;
    let mut pager = MemPager::new(ps);
    init_database(&mut pager).unwrap();
    let root = create_table_btree(&mut pager).unwrap();
    // A modest number of small rows keeps this a single-leaf tree (root IS the leaf).
    for id in 1..=30 {
        table_insert(&mut pager, root, id, &payload_for(id)).unwrap();
    }
    assert_free_accounting(&pager, root, usable);

    pager.begin().unwrap();
    for id in [5i64, 11, 17, 23] {
        assert!(table_delete(&mut pager, root, id).unwrap());
    }
    assert_free_accounting(&pager, root, usable); // freeblocks present
    // Reinsert two (in the same txn → still the fast route; may reclaim freeblocks).
    for id in [11i64, 23] {
        table_insert(&mut pager, root, id, &payload_for(id)).unwrap();
    }
    assert_free_accounting(&pager, root, usable);
    pager.commit().unwrap();
    assert_free_accounting(&pager, root, usable);

    // The surviving rows are exactly 1..=30 minus {5, 17}.
    let got: Vec<i64> = scan_all(&pager, root).into_iter().map(|(r, _)| r).collect();
    let expect: Vec<i64> = (1..=30).filter(|id| *id != 5 && *id != 17).collect();
    assert_eq!(got, expect);
}

#[test]
fn delete_all_rows_in_place_leaves_empty_root_leaf() {
    // Deleting the last cell of the root leaf resets it to the canonical empty page (a
    // valid empty table). A follow-up reinsert must work on the emptied root.
    let ps = 4096u32;
    let usable = ps as usize;
    let mut pager = MemPager::new(ps);
    init_database(&mut pager).unwrap();
    let root = create_table_btree(&mut pager).unwrap();
    for id in 1..=6 {
        table_insert(&mut pager, root, id, &payload_for(id)).unwrap();
    }
    pager.begin().unwrap();
    for id in 1..=6 {
        assert!(table_delete(&mut pager, root, id).unwrap());
    }
    pager.commit().unwrap();

    assert!(scan_all(&pager, root).is_empty(), "all rows deleted");
    let view = PageView::new(pager.read_page(root).unwrap(), root, usable).unwrap();
    assert_eq!(view.page_type(), PageType::LeafTable);
    assert_eq!(view.cell_count(), 0, "empty root leaf");
    // A delete of an absent rowid is now a no-op.
    assert!(!table_delete(&mut pager, root, 1).unwrap());
    // The emptied root accepts new rows again.
    table_insert(&mut pager, root, 42, &payload_for(42)).unwrap();
    assert_eq!(scan_all(&pager, root), vec![(42, payload_for(42))]);
}

#[test]
fn delete_spilled_row_in_place_frees_and_reuses_overflow_chain() {
    // A deleted SPILLED row's overflow chain must be freed by the fast route (which
    // touches only the leaf cell) and reused on reinsert. Repeated in-place
    // delete+reinsert of a same-size spilled row must NOT grow the file (no leak), and
    // must reach the same state as the fallback.
    let ps = 512u32;

    let build = |in_txn: bool| -> (MemPager, PageId, Vec<PageId>) {
        let mut pager = MemPager::new(ps);
        init_database(&mut pager).unwrap();
        let root = create_table_btree(&mut pager).unwrap();
        // Two small anchors so the leaf never empties, plus one spilled row.
        table_insert(&mut pager, root, 1, &payload_for(1)).unwrap();
        table_insert(&mut pager, root, 3, &payload_for(3)).unwrap();
        table_insert(&mut pager, root, 2, &big_payload(4000, 0)).unwrap();
        if in_txn {
            pager.begin().unwrap();
        }
        let mut counts = Vec::new();
        for cycle in 0..6u64 {
            assert!(table_delete(&mut pager, root, 2).unwrap());
            table_insert(&mut pager, root, 2, &big_payload(4000, cycle + 1)).unwrap();
            counts.push(pager.page_count().unwrap());
        }
        if in_txn {
            pager.commit().unwrap();
        }
        (pager, root, counts)
    };

    let (fast, froot, fast_counts) = build(true);
    let (slow, sroot, _slow_counts) = build(false);

    // Reuse, not leak: page count is stable across the delete+reinsert cycles.
    for w in fast_counts.windows(2) {
        assert_eq!(w[0], w[1], "repeated spilled delete+reinsert must reuse freed pages, not grow the file");
    }
    // Correctness: the newest spilled payload reads back byte-exact.
    let scan = scan_all(&fast, froot);
    assert_eq!(scan.len(), 3, "two anchors + one spilled row");
    assert_eq!(
        scan.iter().find(|(r, _)| *r == 2).unwrap().1,
        big_payload(4000, 6),
        "spilled row reads back the newest payload",
    );
    // Equivalence with the fallback route, including the final page count.
    assert_eq!(scan_all(&slow, sroot), scan, "fast and fallback agree after spilled delete+reinsert cycles");
    assert_eq!(
        fast.page_count().unwrap(),
        slow.page_count().unwrap(),
        "fast and fallback reach the same page count after a spilled delete+reinsert",
    );
}
