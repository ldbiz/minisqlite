//! End-to-end INSERT-only tests that the table b-tree never writes a malformed interior
//! page: every NON-ROOT interior b-tree page holds `K >= 2` keys (`>= 3` child pointers),
//! the on-disk-format invariant of fileformat2 §1.6.
//!
//! §1.6: "The number of keys on an interior b-tree page, K, is almost always at least 2 ...
//! The only exception is when page 1 is an interior b-tree page ... In all other cases, K is
//! 2 or more." Real sqlite maintains this, so we must too. The regression it guards: the
//! interior split planner (`build::plan_interior_groups`) used to leave the LAST group of an
//! ascending-insert split with exactly one key (`K == 1`), producing a non-root interior page
//! that violates §1.6 — a file real sqlite would consider malformed. The other `tests/*.rs`
//! `walk` validators never assert `K >= 2`, so the break went unseen; this file's `walk` adds
//! that assertion and drives it on real interior pages built by many-way splits.
//!
//! The ROOT is exempt from the `K >= 2` assertion — not because §1.6 blesses a `K == 1` root
//! (§1.6 strictly exempts only page 1), but because this fix does not govern the root: the two
//! `plan_interior_groups` callers (`split_interior_in_place`, `balance_deeper_interior`) only
//! ever build NON-root pages — the root is written separately by balance-deeper — so exempting
//! the root hides none of the fix's own output. A freshly balance-deepened `create_table_btree`
//! root can currently be a 2-child (`K == 1`) interior, or a leaf for a tiny tree; whether a
//! non-page-1 `K == 1` root is itself §1.6-conformant is a separate balance-deeper concern,
//! out of scope here. `walk` therefore asserts `cell_count() >= 2` only for interior pages
//! whose id differs from the root — every page the fix produces still gets the check.
//!
//! Each `tests/*.rs` is its own crate, so the structural validator and builder are copied here
//! (as in `tests/delete.rs`) rather than shared — a small duplication that keeps the file
//! self-contained and lets `walk` assert exactly these post-insert invariants.

use minisqlite_btree::{create_table_btree, init_database, table_insert, TableCursor};
use minisqlite_fileformat::{PageType, PageView};
use minisqlite_pager::{MemPager, PageId, Pager};

/// A deterministic ~50-byte payload for `rowid`, so a scan can check bytes exactly and each
/// row proves it kept its own payload (not a neighbour's) across page rebuilds.
fn payload_for(rowid: i64) -> Vec<u8> {
    let mut v = format!("row-{rowid}-").into_bytes();
    let fill = (rowid as u8).wrapping_mul(31).wrapping_add(7);
    while v.len() < 50 {
        v.push(fill);
    }
    v
}

/// A deterministic shuffle of `1..=n` (a fixed LCG permutation) so tests exercise out-of-order
/// build without a PRNG dependency.
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

/// Recursively validate the table b-tree rooted at `root` and return its in-order rowids and
/// height (leaf = 1). `id` is the page being visited; `root` is threaded unchanged so the
/// `K >= 2` check can exempt the root. Asserts the full table-b-tree contract INSERT must
/// preserve:
///   - **§1.6 `K >= 2` for every NON-ROOT interior page** (`cell_count() >= 2`) — the new
///     invariant; the root is exempt (a 2-child root has `K == 1`, which §1.6 permits).
///   - every interior separator equals the largest rowid in its left child's subtree;
///   - keys are strictly ascending within a page and across children in order;
///   - all child subtrees have equal height (balanced);
///   - no interior points at an empty child.
/// A single-child interior (0 separators, only a right-most pointer) is a valid pass-through:
/// `walk` follows its right-most pointer like the read cursor does.
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
            // sqlite also writes, and which cannot be forced to >= 2.
            if id != root {
                assert!(
                    count >= 2,
                    "non-root interior page {id} has K={count} (< 2): violates fileformat2 §1.6"
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

/// Build a table (root allocated by `create_table_btree`, i.e. NOT page 1), insert `ids` in the
/// given order with `payload_for`, and return `(pager, root)`.
fn build_table(page_size: u32, ids: &[i64]) -> (MemPager, PageId) {
    let mut pager = MemPager::new(page_size);
    init_database(&mut pager).unwrap();
    let root = create_table_btree(&mut pager).unwrap();
    for &id in ids {
        table_insert(&mut pager, root, id, &payload_for(id)).unwrap();
    }
    (pager, root)
}

/// Validate structure (via `walk`, which fires the non-root `K >= 2` assertion) and prove the
/// tree scans to exactly `1..=n` with byte-identical payloads. Returns the tree height so a
/// caller can assert it is deep enough to have forced real interior splits.
fn assert_valid_full(pager: &MemPager, root: PageId, n: i64) -> u32 {
    let usable = pager.page_size() as usize;
    let (in_order, height) = walk(pager, root, root, usable);
    let expected: Vec<i64> = (1..=n).collect();
    assert_eq!(in_order, expected, "structural walk must equal the inserted rowids in order");
    let scanned = scan_all(pager, root);
    let got: Vec<i64> = scanned.iter().map(|(r, _)| *r).collect();
    assert_eq!(got, expected, "cursor scan must equal the inserted rowids in order");
    for (rowid, payload) in scanned {
        assert_eq!(payload, payload_for(rowid), "payload for rowid {rowid} must round-trip");
    }
    height
}

#[test]
fn ascending_inserts_keep_every_non_root_interior_at_least_two_keys_512() {
    // Ascending rowids on 512-byte pages build a height>=3 interior-of-interiors, exercising
    // many real non-root interior splits. COVERAGE NOTE (do not trim to this test): on PURE
    // ascending inserts a transient K=1 right-edge interior — the tail the planner could once
    // emit — is filled by the next right-edge insert before the tree finalizes, so this
    // final-state walk does NOT by itself witness the K=1 regression (it stays green even
    // against the old buggy tail). The regression is witnessed by `shuffled_inserts_...` (a
    // K=1 non-root page survives out-of-order inserts) and, at the planner level over ascending
    // keys, by `build::tests::interior_plan_every_group_two_keys_across_all_tail_residues`.
    // This test still guards the broader on-disk contract on the ascending shape: every
    // non-root interior is K>=2 in the final tree, separators/heights/order hold, and the full
    // 6000-row scan round-trips.
    let n = 6000i64;
    let ids: Vec<i64> = (1..=n).collect();
    let (pager, root) = build_table(512, &ids);
    let height = assert_valid_full(&pager, root, n);
    assert!(
        height >= 3,
        "ascending insert of {n} rows at 512B must build an interior-of-interiors (height>=3), got {height}"
    );
}

#[test]
fn shuffled_inserts_keep_every_non_root_interior_at_least_two_keys_512() {
    // Out-of-order inserts drive splits at interior positions other than the right edge
    // (interior cells inserted in the middle, propagating splits up a mixed path), a
    // different distribution of split shapes than the ascending case — the K>=2 invariant must
    // hold for all of them.
    let n = 6000i64;
    let ids = shuffled(n);
    let (pager, root) = build_table(512, &ids);
    let height = assert_valid_full(&pager, root, n);
    assert!(
        height >= 3,
        "shuffled insert of {n} rows at 512B must build a height>=3 tree, got {height}"
    );
}

#[test]
fn leaf_root_of_a_tiny_tree_is_valid() {
    // A tiny table stays a single LEAF root (no interior page at all). `walk` must accept it —
    // the K>=2 rule concerns interior pages only, and this pins that a leaf root is never
    // subjected to it.
    let (pager, root) = build_table(512, &[1, 2, 3]);
    {
        let view = PageView::new(pager.read_page(root).unwrap(), root, 512).unwrap();
        assert!(view.page_type().is_leaf(), "3 rows must fit a single leaf root");
    }
    assert_valid_full(&pager, root, 3);
}

#[test]
fn k1_interior_root_is_exempt_and_accepted() {
    // The root exemption in `walk` must be LOAD-BEARING. `plan_interior_groups` never produces
    // the root, so the K>=2 fix does not govern the root's key count — balance-deeper (out of
    // scope) does, and right after the first balance-deeper the root is a 2-child (K==1)
    // interior. `walk` exempts the root, so it must accept that K==1 root; without the
    // `id != root` guard this same page would (wrongly) trip the non-root assertion — so this
    // test goes red if the exemption is removed, proving it does real work. (Whether a
    // non-page-1 K==1 root is itself §1.6-conformant is a separate balance-deeper concern,
    // not asserted here.) Find the smallest ascending tree whose ROOT
    // is a K==1 interior and require `walk` to accept it.
    let mut proved = false;
    for n in 2i64..=80 {
        let (pager, root) = build_table(512, &(1..=n).collect::<Vec<_>>());
        let (is_interior, kcount) = {
            let view = PageView::new(pager.read_page(root).unwrap(), root, 512).unwrap();
            (view.page_type() == PageType::InteriorTable, view.cell_count())
        };
        if is_interior && kcount == 1 {
            // A genuine K==1 root: the tree is valid and scans to 1..=n, and `walk` accepts the
            // K==1 root via the exemption (every NON-root interior it visits is still >=2).
            assert_valid_full(&pager, root, n);
            proved = true;
            break;
        }
    }
    assert!(proved, "expected some small ascending tree at 512B to have a K==1 (2-child) interior root");
}
