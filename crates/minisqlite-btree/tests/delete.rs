//! End-to-end tests for table-b-tree DELETE (`table_delete`): remove rows by rowid and
//! prove the tree stays structurally valid, balanced, and cursor-scannable exactly like
//! real sqlite, through single deletes, mass deletes that collapse the tree height back to
//! a single leaf, delete-then-reinsert, and the page-1 (sqlite_schema) header-preserving
//! path. The bar these tests hold: after *every* delete the on-disk shape re-parses as a
//! valid table b-tree (`walk`), the separator invariant still holds, no interior points at
//! an empty leaf, the root page id never moves, and a full scan equals the surviving set.
//!
//! Each `tests/*.rs` is its own crate, so the structural validator and builders are copied
//! here rather than shared with `tests/btree.rs` — a small duplication that keeps this file
//! self-contained and lets `walk` assert the exact post-delete invariants.

use std::collections::BTreeSet;

use minisqlite_btree::{create_table_btree, init_database, table_delete, table_insert, TableCursor};
use minisqlite_fileformat::freelist::parse_trunk;
use minisqlite_fileformat::{DatabaseHeader, PageType, PageView, HEADER_SIZE};
use minisqlite_pager::{MemPager, PageId, Pager};

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

/// A deterministic payload of `len` bytes seeded by `seed` (same generator as
/// `tests/overflow.rs`), so a big row that spills onto an overflow chain can be
/// checked back byte-for-byte after unrelated deletes and reinserts.
fn big_payload(len: usize, seed: u64) -> Vec<u8> {
    let mut v = Vec::with_capacity(len);
    let mut state = seed.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for _ in 0..len {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        v.push((state >> 40) as u8);
    }
    v
}

/// Read a single row's payload by rowid through `TableCursor::seek_exact`, or
/// `None` if the rowid is absent. Reassembles any overflow chain, so it proves a
/// (possibly large) row survives byte-for-byte without walking the whole table.
fn read_row(pager: &dyn Pager, root: PageId, rowid: i64) -> Option<Vec<u8>> {
    let mut cur = TableCursor::open(pager, root).unwrap();
    if cur.seek_exact(rowid).unwrap() {
        Some(cur.payload().unwrap().into_owned())
    } else {
        None
    }
}

/// A deterministic shuffle of `1..=n` (a fixed LCG permutation) so tests exercise
/// out-of-order build and delete without a PRNG dependency.
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

/// Recursively validate the b-tree rooted at `id` and return its in-order rowids and height
/// (leaf = 1). Asserts the full table-b-tree contract that DELETE must preserve: every
/// interior separator equals the largest rowid in its left child's subtree, keys are
/// strictly ascending, no child subtree is empty (i.e. no interior points at an empty
/// leaf), and every level is balanced (all children of an interior have equal height). A
/// single-child interior (0 separators, only a right-most pointer) is a valid pass-through:
/// `walk` follows its right-most pointer just like the read cursor does.
fn walk(pager: &dyn Pager, id: PageId, usable: usize) -> (Vec<i64>, u32) {
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
        let (sub, h) = walk(pager, children[j], usable);
        assert!(!sub.is_empty(), "child {} of interior {id} is empty (interior->empty leaf)", children[j]);
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
    let (sub, h) = walk(pager, *children.last().unwrap(), usable);
    assert!(!sub.is_empty(), "right child of interior {id} is empty (interior->empty leaf)");
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

/// Build a table (root allocated by `create_table_btree`, i.e. NOT page 1), insert `ids` in
/// the given order with `payload_for`, and return `(pager, root)`.
fn build_table(page_size: u32, ids: &[i64]) -> (MemPager, PageId) {
    let mut pager = MemPager::new(page_size);
    init_database(&mut pager).unwrap();
    let root = create_table_btree(&mut pager).unwrap();
    for &id in ids {
        table_insert(&mut pager, root, id, &payload_for(id)).unwrap();
    }
    (pager, root)
}

/// Assert the tree structurally validates AND scans (payloads byte-identical) to exactly
/// `expected` (which need not be sorted; it is sorted here).
fn assert_valid(pager: &MemPager, root: PageId, expected: &[i64]) {
    let usable = pager.page_size() as usize;
    let (in_order, _height) = walk(pager, root, usable);
    let mut sorted = expected.to_vec();
    sorted.sort_unstable();
    assert_eq!(in_order, sorted, "structural walk must equal the surviving rowids");
    let scanned = scan_all(pager, root);
    let got: Vec<i64> = scanned.iter().map(|(r, _)| *r).collect();
    assert_eq!(got, sorted, "cursor scan must equal the surviving rowids");
    for (rowid, payload) in scanned {
        assert_eq!(payload, payload_for(rowid), "payload for rowid {rowid} must survive intact");
    }
}

/// Read the root interior page's separator keys (the maxes of its non-right-most subtrees).
/// Empty if the root is a leaf.
fn root_separators(pager: &MemPager, root: PageId) -> Vec<i64> {
    let usable = pager.page_size() as usize;
    let view = PageView::new(pager.read_page(root).unwrap(), root, usable).unwrap();
    if view.page_type().is_leaf() {
        return Vec::new();
    }
    (0..view.cell_count() as usize)
        .map(|i| view.table_interior_cell(i).unwrap().rowid)
        .collect()
}

/// Assert the root page id still holds a valid, empty table LEAF page (the end state after
/// deleting every row: the tree collapsed all the way back into a single empty leaf at the
/// original root id).
fn assert_empty_root_leaf(pager: &MemPager, root: PageId) {
    let usable = pager.page_size() as usize;
    let view = PageView::new(pager.read_page(root).unwrap(), root, usable).unwrap();
    assert_eq!(view.page_type(), PageType::LeafTable, "collapsed root must be a table leaf");
    assert_eq!(view.cell_count(), 0, "collapsed root leaf must be empty");
    let mut cur = TableCursor::open(pager, root).unwrap();
    assert!(!cur.first().unwrap(), "empty tree: first() must be false");
    assert!(!cur.seek_exact(1).unwrap(), "empty tree: seek_exact must be false");
}

/// Whether the leaf cell for `rowid` records an overflow-chain pointer, found by descending
/// the tree the same way a search does (first interior separator `>= rowid`, else the
/// right-most pointer; separators are left-subtree maxima). Panics if `rowid` is absent, so
/// a test that expects a spilled row proves the row is actually there and actually spilled.
fn row_overflows(pager: &dyn Pager, id: PageId, rowid: i64, usable: usize) -> bool {
    let view = PageView::new(pager.read_page(id).unwrap(), id, usable).unwrap();
    if view.page_type().is_leaf() {
        for i in 0..view.cell_count() as usize {
            let cell = view.table_leaf_cell(i).unwrap();
            if cell.rowid == rowid {
                return cell.overflow_page.is_some();
            }
        }
        panic!("rowid {rowid} not present in leaf {id}");
    }
    for i in 0..view.cell_count() as usize {
        let cell = view.table_interior_cell(i).unwrap();
        if rowid <= cell.rowid {
            return row_overflows(pager, cell.left_child, rowid, usable);
        }
    }
    let right = view.right_most_pointer().expect("interior must have a right-most pointer");
    row_overflows(pager, right, rowid, usable)
}

/// Read and decode page 1's 100-byte database header. Page 1 always carries a valid
/// header in these tests (written by `init_database` and preserved by every page-1
/// rebuild), so `DatabaseHeader::read` returns `Ok`.
fn read_page1_header(pager: &MemPager) -> DatabaseHeader {
    let page1 = pager.read_page(1).unwrap();
    let buf: [u8; HEADER_SIZE] = page1[0..HEADER_SIZE].try_into().unwrap();
    DatabaseHeader::read(&buf).expect("page 1 must carry a valid database header")
}

/// Assert page 1's database header kept its IMMUTABLE identity across a mutating
/// b-tree operation, and that the freelist it now describes is structurally sound.
///
/// A delete that frees pages legitimately updates four bookkeeping fields —
/// `first_freelist_trunk` and `freelist_count` (the freelist head/count, §1.5), plus
/// `file_change_counter` and the in-header `database_size_pages` (§1.3.2/§1.3.7) — so
/// those are NOT pinned to `header_before`. Every OTHER field is identity the b-tree
/// must never touch, so each is asserted field-for-field against `header_before`;
/// this fails loudly if a page-1 rebuild clobbered, say, `page_size`, `schema_cookie`,
/// `schema_format`, or `text_encoding`. Comparing decoded fields (not raw bytes) makes
/// the check robust to the bookkeeping fields evolving as reclamation/durability grow.
fn assert_header_identity_preserved(pager: &MemPager, header_before: &DatabaseHeader) {
    let now = read_page1_header(pager);

    assert_eq!(now.page_size, header_before.page_size, "page_size must not change");
    assert_eq!(now.write_version, header_before.write_version, "write_version must not change");
    assert_eq!(now.read_version, header_before.read_version, "read_version must not change");
    assert_eq!(now.reserved_space, header_before.reserved_space, "reserved_space must not change");
    assert_eq!(
        now.max_embedded_payload_fraction, header_before.max_embedded_payload_fraction,
        "max_embedded_payload_fraction must not change"
    );
    assert_eq!(
        now.min_embedded_payload_fraction, header_before.min_embedded_payload_fraction,
        "min_embedded_payload_fraction must not change"
    );
    assert_eq!(
        now.leaf_payload_fraction, header_before.leaf_payload_fraction,
        "leaf_payload_fraction must not change"
    );
    assert_eq!(now.schema_cookie, header_before.schema_cookie, "schema_cookie must not change");
    assert_eq!(now.schema_format, header_before.schema_format, "schema_format must not change");
    assert_eq!(
        now.default_cache_size, header_before.default_cache_size,
        "default_cache_size must not change"
    );
    assert_eq!(
        now.largest_root_btree, header_before.largest_root_btree,
        "largest_root_btree must not change"
    );
    assert_eq!(now.text_encoding, header_before.text_encoding, "text_encoding must not change");
    assert_eq!(now.user_version, header_before.user_version, "user_version must not change");
    assert_eq!(
        now.incremental_vacuum, header_before.incremental_vacuum,
        "incremental_vacuum must not change"
    );
    assert_eq!(now.application_id, header_before.application_id, "application_id must not change");
    assert_eq!(
        now.version_valid_for, header_before.version_valid_for,
        "version_valid_for must not change"
    );
    assert_eq!(
        now.sqlite_version_number, header_before.sqlite_version_number,
        "sqlite_version_number must not change"
    );

    assert_freelist_valid(pager, &now);
}

/// Walk the freelist trunk chain described by `header` and prove it is structurally
/// sound (fileformat2 §1.5). Collects every trunk page and every leaf page into a set
/// (the same trunk walk `minisqlite-pager`'s `alloc` tests use) and asserts:
///   (a) the collected page count equals the header's `freelist_count`
///       (SQLite counts trunks AND leaves);
///   (b) every id is a real freeable page, `2..=page_count` — never 0 or 1;
///   (c) no id appears twice (a double-free would list a page twice);
///   (d) page 1 (the root, holding the header) is never on the freelist.
/// So a future bug that freed page 1 or double-freed a page fails here.
fn assert_freelist_valid(pager: &MemPager, header: &DatabaseHeader) {
    let usable = header.usable_size();
    let page_count = pager.page_count().unwrap();
    let mut seen: BTreeSet<PageId> = BTreeSet::new();
    let mut trunk = header.first_freelist_trunk;
    while trunk != 0 {
        assert!(
            (2..=page_count).contains(&trunk),
            "freelist trunk page {trunk} out of range 2..={page_count} (never 0 or 1)"
        );
        assert!(seen.insert(trunk), "freelist page {trunk} appears twice (double-free)");
        let (next, leaves) = parse_trunk(pager.read_page(trunk).unwrap(), usable).unwrap();
        for leaf in leaves {
            assert!(
                (2..=page_count).contains(&leaf),
                "freelist leaf page {leaf} out of range 2..={page_count} (never 0 or 1)"
            );
            assert!(seen.insert(leaf), "freelist page {leaf} appears twice (double-free)");
        }
        trunk = next;
    }
    // (b)'s range already excludes page 1 from `seen`; this pins invariant (d) directly
    // so it still holds if that range bound is ever loosened.
    assert!(!seen.contains(&1), "page 1 (the root/header page) must never be on the freelist");
    assert_eq!(
        seen.len() as u32,
        header.freelist_count,
        "freelist chain length {} must equal header freelist_count {}",
        seen.len(),
        header.freelist_count
    );
}

/// `first_freelist_trunk` from page 1's header (offset 32, big-endian u32). Zero
/// when the freelist is empty; non-zero once any page has been freed — so a test
/// can assert a delete actually reclaimed pages rather than leaking them.
fn freelist_trunk(pager: &MemPager) -> u32 {
    let page = pager.read_page(1).unwrap();
    u32::from_be_bytes([page[32], page[33], page[34], page[35]])
}

/// `freelist_count` from page 1's header (offset 36, big-endian u32): the total
/// number of pages (trunks + leaves) currently on the freelist.
fn freelist_count(pager: &MemPager) -> u32 {
    let page = pager.read_page(1).unwrap();
    u32::from_be_bytes([page[36], page[37], page[38], page[39]])
}

// -------------------------------------------------------------------------------------

#[test]
fn delete_present_returns_true_absent_returns_false_and_scan_tracks() {
    // Even rowids only, so odd rowids are genuine gaps (absent-in-the-middle).
    let ids: Vec<i64> = (1..=1000).map(|i| i * 2).collect(); // 2,4,...,2000
    let (mut pager, root) = build_table(512, &ids);
    assert!(pager.page_count().unwrap() > 1, "tree must be multi-page for this to be interesting");
    assert_valid(&pager, root, &ids);

    // Absent rowids leave the tree byte-for-byte unchanged and return false: gaps (odd),
    // below-min, above-max.
    let before = scan_all(&pager, root);
    for &absent in &[1i64, 3, 999, 1001, 1999, 0, -5, 2001, 100_000] {
        assert!(!table_delete(&mut pager, root, absent).unwrap(), "absent {absent} must return false");
    }
    assert_eq!(scan_all(&pager, root), before, "absent deletes must not change the tree");

    // A spread of present rowids: min, an interior separator-ish spot, the max, some middles.
    let mut remaining: BTreeSet<i64> = ids.iter().copied().collect();
    for &present in &[2i64, 4, 500, 1000, 1498, 2000, 1000, 12, 1974] {
        // The second `1000` in the list is already gone -> exercises the re-delete=false case.
        let existed = remaining.remove(&present);
        assert_eq!(
            table_delete(&mut pager, root, present).unwrap(),
            existed,
            "delete {present}: return value must match presence"
        );
        let expect: Vec<i64> = remaining.iter().copied().collect();
        assert_valid(&pager, root, &expect);
    }
}

/// Delete every row of a freshly built (shuffled) tree in `order`, validating structure and
/// scan after EVERY delete, then assert the tree collapsed to an empty leaf at the same root
/// id. This is the exhaustive correctness core: over a full delete sequence it necessarily
/// exercises leaf removal at the left edge, the right edge, and the middle; separator
/// retargeting when a subtree's max is deleted; interior unlinking; and root collapse.
fn delete_all_in_order(order: &[i64]) {
    let n = order.len() as i64;
    let (mut pager, root) = build_table(512, &shuffled(n));
    let mut remaining: BTreeSet<i64> = (1..=n).collect();

    for &id in order {
        assert!(table_delete(&mut pager, root, id).unwrap(), "present {id} must delete");
        assert!(remaining.remove(&id));
        // Re-deleting the same rowid is now a no-op false.
        assert!(!table_delete(&mut pager, root, id).unwrap(), "re-delete {id} must be false");
        let expect: Vec<i64> = remaining.iter().copied().collect();
        assert_valid(&pager, root, &expect);
    }

    assert_empty_root_leaf(&pager, root);
}

#[test]
fn delete_every_row_forward_order() {
    let order: Vec<i64> = (1..=300).collect();
    delete_all_in_order(&order);
}

#[test]
fn delete_every_row_reverse_order() {
    let order: Vec<i64> = (1..=300).rev().collect();
    delete_all_in_order(&order);
}

#[test]
fn delete_every_row_shuffled_order() {
    let order = shuffled(300);
    delete_all_in_order(&order);
}

#[test]
fn mass_delete_collapses_height_and_keeps_root_id_512() {
    // A deep tree (height >= 3) at 512-byte pages, then delete everything in shuffled order.
    // Interior pages must empty and unlink, and the root must collapse height back down —
    // all at the SAME root page id. Orphaned pages are now freed to the freelist (not
    // leaked): freeing does not shrink the file, so page_count does NOT shrink here, but the
    // pages are reused by later inserts (proved by the reclamation tests below). Correctness
    // in this test is walk + scan + the stable, collapsing root.
    let n = 3000i64;
    let (mut pager, root) = build_table(512, &shuffled(n));
    let usable = pager.page_size() as usize;
    let (_in_order, h0) = walk(&pager, root, usable);
    assert!(h0 >= 3, "expected a height>=3 tree to force interior unlinks, got {h0}");

    let order = shuffled(n);
    let mut remaining: BTreeSet<i64> = (1..=n).collect();
    let mut deleted = 0i64;
    let mut saw_smaller_height = false;
    for &id in &order {
        assert!(table_delete(&mut pager, root, id).unwrap());
        remaining.remove(&id);
        deleted += 1;
        // Full walk+scan is O(n); doing it every delete is O(n^2). Validate structure every
        // 128 deletes (and always near the very end) to keep the test bounded but still catch
        // a mid-sequence invariant break to within a small window.
        if deleted % 128 == 0 || remaining.len() <= 4 {
            let expect: Vec<i64> = remaining.iter().copied().collect();
            assert_valid(&pager, root, &expect);
            // Root id must never move as the tree collapses.
            let ty = PageView::new(pager.read_page(root).unwrap(), root, usable).unwrap().page_type();
            assert!(ty.is_table(), "root {root} must stay a table page while collapsing");
            let (_ord, h) = walk(&pager, root, usable);
            if h < h0 {
                saw_smaller_height = true;
            }
        }
    }
    assert!(saw_smaller_height, "mass delete must reduce tree height (root collapse) at a stable id");
    assert_empty_root_leaf(&pager, root);
}

#[test]
fn delete_then_reinsert_roundtrips() {
    let n = 800i64;
    let (mut pager, root) = build_table(512, &shuffled(n));

    // Delete a spread (every 3rd rowid).
    let removed: Vec<i64> = (1..=n).filter(|r| r % 3 == 0).collect();
    for &id in &removed {
        assert!(table_delete(&mut pager, root, id).unwrap());
    }
    let survivors: Vec<i64> = (1..=n).filter(|r| r % 3 != 0).collect();
    assert_valid(&pager, root, &survivors);

    // Reinsert them (in shuffled order) and the full set must be back, valid and scannable.
    let mut reinsert = removed.clone();
    reinsert.reverse();
    for &id in &reinsert {
        table_insert(&mut pager, root, id, &payload_for(id)).unwrap();
    }
    let full: Vec<i64> = (1..=n).collect();
    assert_valid(&pager, root, &full);
}

#[test]
fn delete_root_separator_keys_retargets_separators() {
    // The root's separator keys are exactly the maxes of its non-right-most subtrees.
    // Deleting each one is the case that goes stale unless a separator is retargeted (or the
    // borderline leaf empties and unlinks). Walk after each delete enforces the invariant.
    let n = 2000i64;
    let (mut pager, root) = build_table(512, &shuffled(n));
    let (_o, h) = walk(&pager, root, pager.page_size() as usize);
    assert!(h >= 3, "want a tall tree so separators sit above real subtrees, got {h}");

    let seps = root_separators(&pager, root);
    assert!(seps.len() >= 2, "root should have several separators, got {}", seps.len());

    let mut remaining: BTreeSet<i64> = (1..=n).collect();
    for &key in &seps {
        assert!(table_delete(&mut pager, root, key).unwrap(), "separator rowid {key} must exist");
        remaining.remove(&key);
        let expect: Vec<i64> = remaining.iter().copied().collect();
        assert_valid(&pager, root, &expect);
    }
}

#[test]
fn single_leaf_delete_down_to_empty_root_leaf() {
    // A table small enough to live in one leaf: page 1 (schema) + page 2 (the single leaf).
    let (mut pager, root) = build_table(4096, &[1, 2, 3]);
    assert_eq!(pager.page_count().unwrap(), 2, "one schema page + one table leaf");

    // Delete the middle row: leaf rebuilt in place, no structural change, root id stable.
    assert!(table_delete(&mut pager, root, 2).unwrap());
    assert!(!table_delete(&mut pager, root, 2).unwrap(), "re-delete is false");
    assert_valid(&pager, root, &[1, 3]);

    // Delete the rest down to zero: an empty ROOT leaf is a valid empty table (not an error).
    assert!(table_delete(&mut pager, root, 1).unwrap());
    assert!(table_delete(&mut pager, root, 3).unwrap());
    assert_valid(&pager, root, &[]);
    assert_empty_root_leaf(&pager, root);

    // Deleting from the now-empty tree stays false and harmless.
    assert!(!table_delete(&mut pager, root, 1).unwrap());
    assert!(!table_delete(&mut pager, root, 42).unwrap());
    assert_empty_root_leaf(&pager, root);
}

#[test]
fn mass_delete_on_page1_root_collapses_to_empty_leaf_preserving_header() {
    // Combines the two page-1-specific hazards: the root IS page 1 (the sqlite_schema root,
    // carrying the 100-byte database header) AND the tree is tall enough that deleting every
    // row must collapse height all the way back into page 1. This is the ONLY test that drives
    // `collapse_root`/`try_copy_leaf` on the reduced-capacity page 1 (its 100-byte header
    // leaves less room than a full-size child page, so a full child leaf may not fit and the
    // root is kept as a single-child interior instead). Through the whole collapse the header's
    // IMMUTABLE identity fields must be preserved (only the free/alloc/commit bookkeeping —
    // `file_change_counter`, `database_size_pages`, and the freelist head/count — may change),
    // the root must remain page 1, and the tree must stay valid — and the collapse must
    // actually free pages onto the freelist rather than leaking them (so the freelist is
    // validated structurally: chain length == count, ids in range, no page 1, no double-free).
    let mut pager = MemPager::new(512);
    init_database(&mut pager).unwrap();
    let header_before = read_page1_header(&pager);

    let n = 3000i64;
    for id in shuffled(n) {
        table_insert(&mut pager, 1, id, &payload_for(id)).unwrap();
    }
    let usable = pager.page_size() as usize;
    let (_o, h0) = walk(&pager, 1, usable);
    assert!(h0 >= 3, "want a tall page-1 tree to force a real collapse, got height {h0}");

    let order = shuffled(n);
    let mut remaining: BTreeSet<i64> = (1..=n).collect();
    let mut deleted = 0i64;
    let mut saw_smaller_height = false;
    for &id in &order {
        assert!(table_delete(&mut pager, 1, id).unwrap());
        remaining.remove(&id);
        deleted += 1;
        if deleted % 128 == 0 || remaining.len() <= 4 {
            // Header identity survives every page-1 rebuild along the way, not just at
            // the end. The freelist bookkeeping fields legitimately change as pages are
            // freed, so they are excluded from the identity check and validated
            // structurally instead (chain length == count, ids valid, no page 1).
            assert_header_identity_preserved(&pager, &header_before);
            let expect: Vec<i64> = remaining.iter().copied().collect();
            assert_valid(&pager, 1, &expect);
            let (_ord, h) = walk(&pager, 1, usable);
            if h < h0 {
                saw_smaller_height = true;
            }
        }
    }
    assert!(saw_smaller_height, "deleting every row must collapse the page-1 root's height");
    assert_empty_root_leaf(&pager, 1);
    // The header's identity survived the full collapse to an empty leaf (the freelist
    // bookkeeping fields differ; validated structurally).
    assert_header_identity_preserved(&pager, &header_before);
    // The collapse must have reclaimed the emptied pages onto the freelist, not leaked
    // them: the header now points at a trunk and counts the freed pages.
    assert_ne!(freelist_trunk(&pager), 0, "collapsing the page-1 tree must free pages onto the freelist");
    assert!(freelist_count(&pager) >= 3, "the full page-1 collapse frees many pages");
}

#[test]
fn delete_from_empty_tree_is_false() {
    let mut pager = MemPager::new(4096);
    init_database(&mut pager).unwrap();
    let root = create_table_btree(&mut pager).unwrap();
    for &id in &[-1i64, 0, 1, 5, 1_000_000] {
        assert!(!table_delete(&mut pager, root, id).unwrap(), "delete {id} on empty tree is false");
    }
    assert_empty_root_leaf(&pager, root);
}

#[test]
fn delete_on_page1_root_preserves_database_header() {
    // Page 1 is the sqlite_schema root and carries the 100-byte database header. A delete
    // that rebuilds page 1 MUST preserve the header's identity fields. Grow page 1 into an
    // interior root, then delete a spread and confirm the immutable header fields survive
    // (bookkeeping fields may differ, and any freelist is validated) and the tree stays valid.
    let mut pager = MemPager::new(4096);
    init_database(&mut pager).unwrap();
    let header_before = read_page1_header(&pager);

    let n = 500i64;
    for id in 1..=n {
        table_insert(&mut pager, 1, id, &payload_for(id)).unwrap();
    }
    let view = PageView::new(pager.read_page(1).unwrap(), 1, 4096).unwrap();
    assert_eq!(view.page_type(), PageType::InteriorTable, "page 1 should have grown to interior");

    let mut remaining: BTreeSet<i64> = (1..=n).collect();
    for id in (1..=n).step_by(7) {
        assert!(table_delete(&mut pager, 1, id).unwrap());
        remaining.remove(&id);
    }
    // This sparse pattern (every 7th of 500 rows on 4096-byte pages) empties no leaf, so it
    // frees no page. Assert that "nothing freed" property explicitly (the freelist stays
    // empty), then check the header's identity field-wise — which also validates the (here
    // empty) freelist — so a future denser variant that DOES free pages still passes on the
    // legitimately-updated bookkeeping fields instead of spuriously failing.
    assert_eq!(freelist_trunk(&pager), 0, "sparse deletes empty no leaf, so nothing is freed");
    assert_eq!(freelist_count(&pager), 0, "...and the freelist count stays zero");
    assert_header_identity_preserved(&pager, &header_before);

    // Tree still valid and scans to the survivors (root is page 1).
    let expect: Vec<i64> = remaining.iter().copied().collect();
    let usable = pager.page_size() as usize;
    let (in_order, _h) = walk(&pager, 1, usable);
    assert_eq!(in_order, expect);
    let scanned = scan_all(&pager, 1);
    let got: Vec<i64> = scanned.iter().map(|(r, _)| *r).collect();
    assert_eq!(got, expect);
    for (rowid, payload) in scanned {
        assert_eq!(payload, payload_for(rowid));
    }
}

// -------------------------------------------------------------------------------------
// Freelist reclamation: deleting rows must FREE the b-tree pages and overflow chains it
// orphans (onto the page-1 freelist), and later inserts must REUSE those freed pages
// instead of growing the file. Each test is self-calibrating: it measures the page_count
// of the built structure, confirms the delete put pages on the freelist (header trunk /
// count), and asserts the reinsert does not grow past that baseline. Freeing never shrinks
// the file (pages move to the freelist in place), so page_count is compared, not expected
// to drop. These fail loudly if the free_page / free_overflow_chain wiring is removed:
// without it the freelist stays empty and the reinsert balloons the file.

#[test]
fn delete_range_frees_leaf_and_interior_pages_reused_by_reinsert() {
    // Build a multi-level tree, delete a large contiguous prefix so whole leaves (and the
    // interiors above them) empty and unlink, then re-insert the same range. With
    // reclamation the freed pages are reused and page_count does not grow past the
    // pre-delete baseline; a leak would force the reinsert to extend the file instead.
    let n = 2000i64;
    let ids: Vec<i64> = (1..=n).collect();
    let (mut pager, root) = build_table(512, &ids);
    let (_o, h0) = walk(&pager, root, pager.page_size() as usize);
    assert!(h0 >= 3, "want a multi-level tree so interiors are freed too, got height {h0}");
    let pc_full = pager.page_count().unwrap();

    // Delete the lower half: a long contiguous run that empties whole leaves and cascades
    // interior unlinks.
    let k = n / 2;
    for id in 1..=k {
        assert!(table_delete(&mut pager, root, id).unwrap(), "row {id} must delete");
    }
    let survivors: Vec<i64> = (k + 1..=n).collect();
    assert_valid(&pager, root, &survivors);
    // The emptied pages went onto the freelist (not leaked), and freeing did not shrink
    // the file.
    assert_ne!(freelist_trunk(&pager), 0, "deleting whole leaves/interiors must free pages");
    let freed = freelist_count(&pager);
    assert!(freed >= 5, "a half-tree delete should free several pages, freed {freed}");
    assert_eq!(pager.page_count().unwrap(), pc_full, "free_page must not shrink the file");

    // Re-insert the deleted range: allocations must consume the freelist before growing.
    for id in 1..=k {
        table_insert(&mut pager, root, id, &payload_for(id)).unwrap();
    }
    assert_valid(&pager, root, &ids);
    let pc_reinsert = pager.page_count().unwrap();
    // Reuse is not byte-perfect for a partial delete: the delete leaves some half-full
    // leaves at the survivors' edge that the reinsert does not repack as densely as the
    // original sequential build, so the file may grow by a few pages. The proof of
    // reclamation is that it grew by FAR less than the freed count (a leak, reusing
    // nothing, would grow by ~`freed`) AND that the reinsert drained the freelist it reused.
    let growth = pc_reinsert.saturating_sub(pc_full);
    assert!(
        growth * 2 < freed,
        "re-inserting the freed range must reuse freed pages, not grow the file \
         (grew {growth} pages, but {freed} pages had been freed; a leak would grow by ~{freed})"
    );
    let remaining_free = freelist_count(&pager);
    assert!(
        remaining_free < freed,
        "the reinsert must consume (drain) the freed pages it reused, not leave them all on \
         the freelist (freelist now {remaining_free}, was {freed})"
    );
}

#[test]
fn delete_overflowed_row_reclaims_chain_for_reuse() {
    // A row far larger than a 512-byte page spills onto a forward-linked overflow chain.
    // Deleting the row must free that whole chain; inserting another big row of the same
    // size must then reuse the freed overflow pages instead of growing the file by a
    // second full chain. Small neighbour rows prove the delete leaves them untouched.
    let mut pager = MemPager::new(512);
    init_database(&mut pager).unwrap();
    let root = create_table_btree(&mut pager).unwrap();

    for id in 1..=5i64 {
        table_insert(&mut pager, root, id, &payload_for(id)).unwrap();
    }
    let big_a = big_payload(40_000, 0xA11CE);
    table_insert(&mut pager, root, 6, &big_a).unwrap();
    assert_eq!(read_row(&pager, root, 6).as_deref(), Some(big_a.as_slice()), "big row A must round-trip");
    let pc_with_chain = pager.page_count().unwrap();
    assert!(pc_with_chain > 20, "40KB on 512-byte pages must span many overflow pages, got {pc_with_chain}");

    // Delete the big row: its overflow chain must be freed, not leaked.
    assert!(table_delete(&mut pager, root, 6).unwrap(), "big row must delete");
    assert_eq!(read_row(&pager, root, 6), None, "deleted big row must be gone");
    for id in 1..=5i64 {
        assert_eq!(
            read_row(&pager, root, id).as_deref(),
            Some(payload_for(id).as_slice()),
            "neighbour {id} must survive the big-row delete byte-for-byte"
        );
    }
    assert_ne!(freelist_trunk(&pager), 0, "deleting an overflowed row must free its chain");
    let freed = freelist_count(&pager);
    assert!(freed >= 20, "a 40KB chain on 512-byte pages is many pages, freed {freed}");
    assert_eq!(pager.page_count().unwrap(), pc_with_chain, "free must not shrink the file");

    // Insert another equally-large row: its chain must reuse the freed pages.
    let big_b = big_payload(40_000, 0xB0B);
    table_insert(&mut pager, root, 7, &big_b).unwrap();
    assert_eq!(read_row(&pager, root, 7).as_deref(), Some(big_b.as_slice()), "big row B must round-trip");
    let pc_reused = pager.page_count().unwrap();
    assert!(
        pc_reused <= pc_with_chain,
        "the second big row must reuse the freed overflow chain, not grow the file \
         (page_count {pc_reused} > {pc_with_chain}; {freed} pages had been freed)"
    );
    // Neighbours and the new big row all still read back correctly.
    for id in 1..=5i64 {
        assert_eq!(read_row(&pager, root, id).as_deref(), Some(payload_for(id).as_slice()));
    }
    assert_eq!(read_row(&pager, root, 7).as_deref(), Some(big_b.as_slice()));
}

#[test]
fn delete_all_then_reinsert_reuses_freed_pages() {
    // Deleting every row collapses the tree to an empty root leaf (root id unchanged) and
    // frees every other page. Re-inserting a fresh full set must reuse those freed pages
    // rather than roughly doubling the file.
    let n = 1500i64;
    let (mut pager, root) = build_table(512, &shuffled(n));
    let (_o, h0) = walk(&pager, root, pager.page_size() as usize);
    assert!(h0 >= 3, "want a multi-level tree, got height {h0}");
    let pc_full = pager.page_count().unwrap();

    for id in shuffled(n) {
        assert!(table_delete(&mut pager, root, id).unwrap(), "row {id} must delete");
    }
    assert_empty_root_leaf(&pager, root);
    assert_ne!(freelist_trunk(&pager), 0, "deleting all rows must free the emptied pages");
    let freed = freelist_count(&pager);
    assert!(freed >= 5, "a collapsed multi-level tree frees several pages, freed {freed}");
    assert_eq!(pager.page_count().unwrap(), pc_full, "free must not shrink the file");

    // Re-insert a fresh full set (same shuffled order): allocations reuse the freelist.
    for id in shuffled(n) {
        table_insert(&mut pager, root, id, &payload_for(id)).unwrap();
    }
    assert_valid(&pager, root, &(1..=n).collect::<Vec<_>>());
    let pc_reinsert = pager.page_count().unwrap();
    assert!(
        pc_reinsert <= pc_full,
        "re-inserting after delete-all must reuse freed pages, not balloon the file \
         (page_count {pc_reinsert} > original {pc_full}; {freed} pages had been freed)"
    );
}

// -------------------------------------------------------------------------------------
// Overflow-pointer PRESERVATION on the two leaf re-encode paths. Distinct from the
// reclamation tests above (which delete a spilled row and free its chain): here a spilled
// row SURVIVES a delete and must keep its overflow pointer through `rebuild_leaf_dropping`
// (a neighbour is dropped) and `try_copy_leaf` (a root collapse pulls its leaf up). Every
// other delete test uses <=50-byte payloads that never spill, so a mutant that re-encodes a
// survivor with `None` for the overflow pointer would go unnoticed — these close that gap.

#[test]
fn delete_preserves_overflow_pointer_of_surviving_rows_in_a_leaf() {
    // `rebuild_leaf_dropping` re-encodes every SURVIVING cell, and the delete contract says
    // it must carry each survivor's overflow pointer through the rebuild (a spilled row's
    // `payload_len` records the FULL length, so losing its chain head corrupts the row). Put a
    // spilled row between two small rows in ONE leaf, delete the small neighbours, and require
    // the overflow survivor to read back byte-exact. Then delete the spilled row itself and
    // confirm the tree is a valid empty leaf (its chain is reclaimed; nothing dangles).
    let usable = 512usize;
    let (mut pager, root) = build_table(512, &[]);
    let big = big_payload(1000, 42); // > usable-35 (477) spills; K>X branch keeps ~39B inline

    table_insert(&mut pager, root, 1, &payload_for(1)).unwrap();
    table_insert(&mut pager, root, 2, &big).unwrap();
    table_insert(&mut pager, root, 3, &payload_for(3)).unwrap();

    // Precondition the test depends on: one leaf holds all three, and rowid 2 truly spilled.
    {
        let view = PageView::new(pager.read_page(root).unwrap(), root, usable).unwrap();
        assert!(view.page_type().is_leaf(), "the small table stays a single leaf");
        assert_eq!(view.cell_count(), 3, "all three rows share one leaf (the big row's inline part is tiny)");
    }
    assert!(row_overflows(&pager, root, 2, usable), "rowid 2 must have spilled to an overflow chain");

    // Delete a small neighbour: the overflow survivor is re-encoded. A dropped pointer here
    // corrupts its readback.
    assert!(table_delete(&mut pager, root, 1).unwrap());
    assert_eq!(
        read_row(&pager, root, 2).as_deref(),
        Some(big.as_slice()),
        "overflow survivor byte-exact after deleting a neighbour"
    );
    assert!(row_overflows(&pager, root, 2, usable), "overflow pointer intact after the leaf rebuild");
    assert_eq!(read_row(&pager, root, 3).as_deref(), Some(payload_for(3).as_slice()));

    // Delete the other neighbour: rowid 2 is re-encoded again, still byte-exact.
    assert!(table_delete(&mut pager, root, 3).unwrap());
    assert_eq!(
        read_row(&pager, root, 2).as_deref(),
        Some(big.as_slice()),
        "overflow survivor byte-exact after deleting the second neighbour"
    );

    // Delete the spilled row itself: its chain is reclaimed and the tree is a valid empty root
    // leaf, with no dangling reference to the freed chain.
    assert!(table_delete(&mut pager, root, 2).unwrap());
    assert!(!table_delete(&mut pager, root, 2).unwrap(), "re-delete of the spilled row is false");
    assert_valid(&pager, root, &[]);
    assert_empty_root_leaf(&pager, root);
}

#[test]
fn root_collapse_pulls_up_an_overflow_row_byte_exact() {
    // The collapse path re-encodes cells too: `try_copy_leaf` pulls a leaf's cells up into the
    // root, and must carry their overflow pointers. Build a tall tree of small rows plus ONE
    // spilled row at the maximum rowid (so it lives in the right-most leaf), delete every small
    // row so the tree collapses, and require the pulled-up spilled row to read back byte-exact.
    let usable = 512usize;
    let (mut pager, root) = build_table(512, &[]);
    let big_id = 5000i64; // above every small rowid -> the right-most leaf
    let big = big_payload(1000, 99);

    for id in 1..=400i64 {
        table_insert(&mut pager, root, id, &payload_for(id)).unwrap();
    }
    table_insert(&mut pager, root, big_id, &big).unwrap();

    let (_o, h0) = walk(&pager, root, usable);
    assert!(h0 >= 2, "need a multi-level tree so a real root collapse occurs, got height {h0}");
    assert!(row_overflows(&pager, root, big_id, usable), "the big row must have spilled before collapse");

    // Delete every small row; only the spilled row remains and the tree collapses into the root.
    for id in 1..=400i64 {
        assert!(table_delete(&mut pager, root, id).unwrap());
    }

    // The collapse brought the root back to a single leaf holding only the spilled row (root id
    // never moved).
    {
        let view = PageView::new(pager.read_page(root).unwrap(), root, usable).unwrap();
        assert!(view.page_type().is_leaf(), "collapse brought the root back to a leaf");
        assert_eq!(view.cell_count(), 1, "only the spilled row remains");
    }
    // The pulled-up spilled row survived byte-exact: `try_copy_leaf` preserved its chain pointer.
    assert_eq!(
        read_row(&pager, root, big_id).as_deref(),
        Some(big.as_slice()),
        "overflow row pulled up by the root collapse must read back byte-exact"
    );
    assert!(row_overflows(&pager, root, big_id, usable), "pulled-up row keeps its overflow pointer");
    assert_eq!(walk(&pager, root, usable).0, vec![big_id], "the collapsed tree holds exactly the spilled row");

    // Deleting it now reclaims the chain and empties the tree.
    assert!(table_delete(&mut pager, root, big_id).unwrap());
    assert_empty_root_leaf(&pager, root);
}
