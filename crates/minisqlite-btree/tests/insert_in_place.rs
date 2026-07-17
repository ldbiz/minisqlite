//! Tests for the FAST in-place insert route (`table_insert` under an open
//! transaction, which reaches `Pager::page_mut` + the fileformat in-place cell
//! editor). Every OTHER btree test calls `table_insert` in auto-commit, where a
//! returned mutable page borrow cannot commit, so `page_mut` fails closed and the
//! path falls back to the whole-page `PageBuilder` rebuild — meaning those tests only
//! cover the FALLBACK route. These tests wrap the inserts in `begin()`/`commit()` so
//! the in-place splice actually runs, and prove three things:
//!
//! 1. **Logical equivalence** — the same workload run in-transaction (fast route) and
//!    in auto-commit (fallback route) yields byte-for-byte identical scans, and every
//!    page re-parses as a structurally valid table b-tree. The fast route must not
//!    change a single query result.
//! 2. **The fast route really ran** — an in-place REPLACE leaves a freeblock behind
//!    (offset 1 of the page header is non-zero), which the defragmenting `PageBuilder`
//!    rebuild NEVER produces. This is the observable fingerprint of the in-place edit,
//!    so the test fails if the fast path silently fell back.
//! 3. **Durability of in-place edits** — inserts spliced in place survive a commit and
//!    a fresh cursor over the committed image (via the on-disk pager in `ondisk.rs`
//!    territory; here the in-memory pager's commit path).

use minisqlite_btree::{create_table_btree, init_database, table_insert, TableCursor};
use minisqlite_fileformat::{PageType, PageView};
use minisqlite_pager::{MemPager, PageId, Pager};

/// A deterministic ~50-byte payload for `rowid` (mirrors `btree.rs`).
fn payload_for(rowid: i64) -> Vec<u8> {
    let mut v = format!("row-{rowid}-").into_bytes();
    let fill = (rowid as u8).wrapping_mul(31).wrapping_add(7);
    while v.len() < 50 {
        v.push(fill);
    }
    v
}

/// A same-size (50-byte) replacement payload, distinguishable from `payload_for`.
fn replaced_payload(rowid: i64) -> Vec<u8> {
    let mut v = format!("new-{rowid}-").into_bytes();
    let fill = (rowid as u8).wrapping_mul(53).wrapping_add(11);
    while v.len() < 50 {
        v.push(fill);
    }
    v
}

/// A deterministic pseudo-random payload of `len` bytes, distinct per `seed`, so a
/// round-trip checks every byte and cross-row contamination is caught. Large `len`
/// spills onto an overflow chain.
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
/// edit that corrupted the header, cell-pointer array, or a freeblock would be caught
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
/// header starts at byte 100). Non-zero means the page carries a freeblock chain —
/// the fingerprint of an in-place delete that the `PageBuilder` rebuild never leaves.
fn first_freeblock(pager: &dyn Pager, id: PageId) -> u16 {
    let page = pager.read_page(id).unwrap();
    let ho = if id == 1 { 100 } else { 0 };
    ((page[ho + 1] as u16) << 8) | page[ho + 2] as u16
}

/// Build a table by inserting `(rowid, payload)` pairs INSIDE one transaction, so the
/// fast in-place route (`page_mut`) is exercised, then commit.
fn build_in_txn(page_size: u32, rows: &[(i64, Vec<u8>)]) -> (MemPager, PageId) {
    let mut pager = MemPager::new(page_size);
    init_database(&mut pager).unwrap();
    let root = create_table_btree(&mut pager).unwrap();
    pager.begin().unwrap();
    for (id, payload) in rows {
        table_insert(&mut pager, root, *id, payload).unwrap();
    }
    pager.commit().unwrap();
    (pager, root)
}

/// Build the same table in auto-commit (each insert its own implicit transaction), so
/// `page_mut` fails closed and every insert takes the FALLBACK rebuild route.
fn build_autocommit(page_size: u32, rows: &[(i64, Vec<u8>)]) -> (MemPager, PageId) {
    let mut pager = MemPager::new(page_size);
    init_database(&mut pager).unwrap();
    let root = create_table_btree(&mut pager).unwrap();
    for (id, payload) in rows {
        table_insert(&mut pager, root, *id, payload).unwrap();
    }
    (pager, root)
}

/// The two routes must agree on every scanned row and both stay structurally valid.
fn assert_routes_agree(page_size: u32, rows: &[(i64, Vec<u8>)]) {
    let (fast, froot) = build_in_txn(page_size, rows);
    let (slow, sroot) = build_autocommit(page_size, rows);
    let usable = page_size as usize;
    let fast_scan = scan_all(&fast, froot);
    let slow_scan = scan_all(&slow, sroot);
    assert_eq!(fast_scan, slow_scan, "fast in-place route and fallback route disagree on scan");
    // Both trees re-parse as valid table b-trees with the identical rowid set.
    assert_eq!(valid_rowids(&fast, froot, usable), valid_rowids(&slow, sroot, usable));
}

fn rows_for(ids: &[i64]) -> Vec<(i64, Vec<u8>)> {
    ids.iter().map(|&id| (id, payload_for(id))).collect()
}

#[test]
fn sequential_bulk_in_place_matches_fallback() {
    // Sequential rowids all land on the growing right-most leaf; the hot bulk-load
    // case. In-place append and PageBuilder produce byte-identical layouts here, so the
    // scans must match exactly across many splits.
    for &ps in &[512u32, 1024, 4096] {
        let rows = rows_for(&(1..=2000).collect::<Vec<_>>());
        assert_routes_agree(ps, &rows);
    }
}

#[test]
fn shuffled_bulk_in_place_matches_fallback() {
    // Out-of-order rowids force inserts BEFORE existing cells (pointer-array shifts) and
    // heterogeneous physical layouts, yet the logical content must be identical to the
    // rebuild route.
    for &ps in &[512u32, 4096] {
        let rows = rows_for(&shuffled(2000));
        assert_routes_agree(ps, &rows);
    }
}

#[test]
fn in_place_replace_matches_fallback() {
    // Insert 1..=800 then REPLACE every row with a same-size payload, all in one
    // transaction: each REPLACE is an in-place delete-then-insert. The result must
    // equal the auto-commit rebuild of the same workload.
    let ids: Vec<i64> = (1..=800).collect();
    let mut rows = rows_for(&ids);
    for id in &ids {
        rows.push((*id, replaced_payload(*id)));
    }
    for &ps in &[512u32, 4096] {
        assert_routes_agree(ps, &rows);
    }
    // And the final payloads are the REPLACE-ments, not the originals.
    let (fast, root) = build_in_txn(4096, &rows);
    let scan = scan_all(&fast, root);
    assert_eq!(scan.len(), 800);
    for (i, (rowid, payload)) in scan.iter().enumerate() {
        assert_eq!(*rowid, (i + 1) as i64);
        assert_eq!(payload, &replaced_payload(*rowid));
    }
}

#[test]
fn in_place_edit_leaves_a_freeblock_that_rebuild_never_does() {
    // PROOF the fast route ran (not a silent fallback): a mid-page in-place REPLACE
    // deletes the old cell, and because free space is plentiful the new cell goes into
    // the gap, leaving the deleted cell's slot as a FREEBLOCK. The PageBuilder rebuild
    // that the auto-commit route uses always defragments, so its first-freeblock stays
    // zero. Same workload, different fingerprint ⇒ the routes are genuinely different.
    let ps = 4096u32;

    // Fast route: three small rows on the single leaf root (page 1), then REPLACE the
    // middle one in place.
    let mut fast = MemPager::new(ps);
    init_database(&mut fast).unwrap();
    let root = create_table_btree(&mut fast).unwrap();
    fast.begin().unwrap();
    for id in 1..=3 {
        table_insert(&mut fast, root, id, &payload_for(id)).unwrap();
    }
    table_insert(&mut fast, root, 2, &replaced_payload(2)).unwrap();
    fast.commit().unwrap();
    assert_ne!(
        first_freeblock(&fast, root),
        0,
        "in-place REPLACE must leave a freeblock behind (the fast route ran)"
    );

    // Fallback route: identical operations in auto-commit → fully defragmented page.
    let mut slow = MemPager::new(ps);
    init_database(&mut slow).unwrap();
    let sroot = create_table_btree(&mut slow).unwrap();
    for id in 1..=3 {
        table_insert(&mut slow, sroot, id, &payload_for(id)).unwrap();
    }
    table_insert(&mut slow, sroot, 2, &replaced_payload(2)).unwrap();
    assert_eq!(
        first_freeblock(&slow, sroot),
        0,
        "the rebuild route always defragments (no freeblock)"
    );

    // Despite the different physical layout, both read back the same rows, and the fast
    // page is a spec-valid leaf (re-parses; the freeblock-bearing page decodes cleanly).
    assert_eq!(scan_all(&fast, root), scan_all(&slow, sroot));
    let usable = ps as usize;
    let view = PageView::new(fast.read_page(root).unwrap(), root, usable).unwrap();
    assert_eq!(view.page_type(), PageType::LeafTable);
    assert_eq!(view.cell_count(), 3);
}

#[test]
fn in_place_replace_that_grows_still_splits_correctly() {
    // A REPLACE whose new payload no longer fits the leaf must, from inside a
    // transaction, take the fit-check's Split branch — not corrupt the page. Fill a
    // small page near-full with tiny rows, then REPLACE one with a large (still inline)
    // payload that forces a split. The result must equal the auto-commit rebuild.
    let ps = 512u32;
    let ids: Vec<i64> = (1..=40).collect();
    let mut rows = rows_for(&ids);
    // Grow rowid 20 to a large inline payload (fits a 512B leaf alone, forces a split).
    let mut big = b"big-20-".to_vec();
    while big.len() < 300 {
        big.push(0xAB);
    }
    rows.push((20, big.clone()));

    assert_routes_agree(ps, &rows);

    let (fast, root) = build_in_txn(ps, &rows);
    let scan = scan_all(&fast, root);
    assert_eq!(scan.len(), 40, "REPLACE keeps the row count");
    let row20 = scan.iter().find(|(r, _)| *r == 20).unwrap();
    assert_eq!(row20.1, big, "rowid 20 reads back the large replacement payload");
}

#[test]
fn in_place_inserts_survive_commit_and_fresh_cursor() {
    // Inserts spliced in place must be durable across the commit: a cursor opened AFTER
    // commit (over the committed image, no active transaction) sees exactly them.
    let ps = 1024u32;
    let rows = rows_for(&shuffled(500));
    let (pager, root) = build_in_txn(ps, &rows);
    // No transaction is open now; the scan reads the committed image.
    let scan = scan_all(&pager, root);
    let mut expect: Vec<(i64, Vec<u8>)> = rows.clone();
    expect.sort_by_key(|(id, _)| *id);
    assert_eq!(scan, expect, "committed in-place inserts read back exactly");
}

#[test]
fn in_place_replace_of_spilled_row_frees_old_chain_via_fast_route() {
    // The one gap the equivalence tests do not cover directly: an in-transaction REPLACE
    // of a SPILLED (overflow) row with another spilled row, through the fast route. The
    // old overflow chain must be freed (before the in-place splice) and its pages reused,
    // and the newest payload must read back byte-exact. Chain freeing runs on the shared
    // path before the fast/fallback branch, so the two routes must also agree.
    let ps = 512u32;
    let big_a = big_payload(4000, 1);
    let big_b = big_payload(4000, 2);
    let big_c = big_payload(4000, 3);

    // Fast route: all three writes in ONE transaction, so `page_mut` (and thus the
    // in-place splice) runs for both REPLACEs.
    let mut fast = MemPager::new(ps);
    init_database(&mut fast).unwrap();
    let root = create_table_btree(&mut fast).unwrap();
    fast.begin().unwrap();
    table_insert(&mut fast, root, 1, &big_a).unwrap();
    table_insert(&mut fast, root, 1, &big_b).unwrap(); // REPLACE spilled -> spilled
    let pc_after_first_replace = fast.page_count().unwrap();
    table_insert(&mut fast, root, 1, &big_c).unwrap(); // REPLACE again, reuses freed chain
    let pc_after_second_replace = fast.page_count().unwrap();
    fast.commit().unwrap();

    // Correctness: the newest spilled payload reads back byte-for-byte.
    let scan = scan_all(&fast, root);
    assert_eq!(scan, vec![(1i64, big_c.clone())], "fast REPLACE reads the newest spilled payload");
    // Reuse, not leak: the second replace reused the first's freed chain (same-size
    // payload → same chain length), so the file did not grow.
    assert_eq!(
        pc_after_second_replace, pc_after_first_replace,
        "the freed overflow chain is reused, not leaked, across repeated in-place REPLACEs",
    );

    // Equivalence: the auto-commit fallback (each write its own txn) reaches the same
    // logical and physical state.
    let mut slow = MemPager::new(ps);
    init_database(&mut slow).unwrap();
    let sroot = create_table_btree(&mut slow).unwrap();
    table_insert(&mut slow, sroot, 1, &big_a).unwrap();
    table_insert(&mut slow, sroot, 1, &big_b).unwrap();
    table_insert(&mut slow, sroot, 1, &big_c).unwrap();
    assert_eq!(scan_all(&slow, sroot), scan, "fast and fallback REPLACE agree on the payload");
    assert_eq!(
        fast.page_count().unwrap(),
        slow.page_count().unwrap(),
        "fast and fallback reach the same page count after a spilled REPLACE",
    );
}
