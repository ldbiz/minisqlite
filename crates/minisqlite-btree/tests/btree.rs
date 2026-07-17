//! End-to-end tests for the table b-tree: build up trees large enough to force
//! multiple leaf splits and interior splits, then verify scans, seeks, REPLACE, the
//! empty tree, and — the real-sqlite bar — that every page re-parses as a
//! structurally valid table b-tree (the separator invariant holds and every level
//! is in key order).

use std::collections::BTreeSet;

use minisqlite_btree::{create_table_btree, init_database, table_insert, TableCursor};
use minisqlite_fileformat::freelist::parse_trunk;
use minisqlite_fileformat::{DatabaseHeader, PageType, PageView, HEADER_SIZE};
use minisqlite_pager::{MemPager, PageId, Pager};

/// A deterministic ~50-byte payload for `rowid`, so a scan can check bytes exactly.
fn payload_for(rowid: i64) -> Vec<u8> {
    let mut v = format!("row-{rowid}-").into_bytes();
    // Pad to a stable length with a rowid-dependent byte so different rows differ
    // beyond the prefix too.
    let fill = (rowid as u8).wrapping_mul(31).wrapping_add(7);
    while v.len() < 50 {
        v.push(fill);
    }
    v
}

/// A distinguishable replacement payload the SAME size (50 bytes) as `payload_for`, so
/// REPLACE-ing a row leaves the leaf byte-for-byte the same size — the tree shape (and
/// therefore its separator set) does not change across a batch of replacements.
fn replaced_payload(rowid: i64) -> Vec<u8> {
    let mut v = format!("new-{rowid}-").into_bytes();
    let fill = (rowid as u8).wrapping_mul(53).wrapping_add(11);
    while v.len() < 50 {
        v.push(fill);
    }
    v
}

/// A deterministic, per-rowid payload whose size varies a lot: most rows are tiny,
/// some are medium, and every 13th is a large-but-still-inline (no overflow) row. This
/// makes leaves heterogeneous, which is what forces N-way (not just 2-way) splits.
/// Sizes stay under X (max inline) for the page sizes the tests use.
fn hetero_payload(rowid: i64) -> Vec<u8> {
    let size = if rowid % 13 == 0 {
        300
    } else if rowid % 5 == 0 {
        120
    } else {
        16
    };
    let mut v = format!("h{rowid}:").into_bytes();
    let fill = (rowid as u8).wrapping_mul(37).wrapping_add(3);
    while v.len() < size {
        v.push(fill);
    }
    v
}

/// A deterministic shuffle of `1..=n` (a fixed LCG permutation over indices) so the
/// tests exercise out-of-order insertion without a PRNG dependency.
fn shuffled(n: i64) -> Vec<i64> {
    let mut ids: Vec<i64> = (1..=n).collect();
    // Fisher-Yates with a simple deterministic LCG.
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

/// Recursively validate the b-tree rooted at `id` and return its in-order rowids and
/// its height (leaf = 1). Asserts the table-b-tree separator invariant: every
/// interior separator equals the largest rowid in its left child's subtree, keys are
/// ascending, and every level is in key order and balanced.
fn walk(pager: &dyn Pager, id: PageId, usable: usize) -> (Vec<i64>, u32) {
    // Read the page into owned lists, then drop the borrow before recursing.
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
            children.push(view.right_most_pointer().unwrap());
            (false, Vec::new(), children, keys)
        }
    };

    if is_leaf {
        for w in leaf_rowids.windows(2) {
            assert!(w[0] < w[1], "leaf {id} rowids not strictly ascending: {w:?}");
        }
        return (leaf_rowids, 1);
    }

    // Interior: keys strictly ascending.
    for w in keys.windows(2) {
        assert!(w[0] < w[1], "interior {id} separators not strictly ascending: {w:?}");
    }
    let mut all = Vec::new();
    let mut child_height: Option<u32> = None;
    for j in 0..keys.len() {
        let (sub, h) = walk(pager, children[j], usable);
        assert!(!sub.is_empty(), "child {} of interior {id} is empty", children[j]);
        // Separator equals the largest rowid in the left child's subtree.
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
    // Right-most child holds everything greater than the last separator.
    let (sub, h) = walk(pager, *children.last().unwrap(), usable);
    assert!(!sub.is_empty(), "right child of interior {id} is empty");
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

/// Collect every interior separator rowid in the tree rooted at `id` (each interior
/// cell's key). These are exactly the rowids whose descent hits `choose_child`'s
/// equality boundary (`rowid == separator`): a separator equals its left subtree's max,
/// so a seek/REPLACE of one must resolve into the LEFT child. A test that probes these
/// specific rowids is what distinguishes `rowid <= key` from `rowid < key`.
fn collect_separators(pager: &dyn Pager, id: PageId, usable: usize, out: &mut Vec<i64>) {
    let (children, keys) = {
        let view = PageView::new(pager.read_page(id).unwrap(), id, usable).unwrap();
        if view.page_type().is_leaf() {
            return;
        }
        let count = view.cell_count() as usize;
        let mut children = Vec::with_capacity(count + 1);
        let mut keys = Vec::with_capacity(count);
        for i in 0..count {
            let c = view.table_interior_cell(i).unwrap();
            children.push(c.left_child);
            keys.push(c.rowid);
        }
        children.push(view.right_most_pointer().unwrap());
        (children, keys)
    };
    out.extend_from_slice(&keys);
    for &child in &children {
        collect_separators(pager, child, usable, out);
    }
}

/// Build a table, insert `ids` (in the given order) with `payload_for`, and return
/// `(pager, root)`. The returned `root` is the page id `create_table_btree` handed
/// back; every caller then reads the tree back through that same id, which only
/// succeeds if `table_insert` kept the root anchored there across all splits and
/// balance-deepers.
fn build_table(page_size: u32, ids: &[i64]) -> (MemPager, PageId) {
    let mut pager = MemPager::new(page_size);
    init_database(&mut pager).unwrap();
    let root = create_table_btree(&mut pager).unwrap();
    for &id in ids {
        table_insert(&mut pager, root, id, &payload_for(id)).unwrap();
    }
    (pager, root)
}

fn assert_full_scan(pager: &MemPager, root: PageId, expected_ids: &[i64]) {
    let usable = pager.page_size() as usize;
    // Structural validation of the on-disk shape.
    let (in_order, _height) = walk(pager, root, usable);
    let mut sorted = expected_ids.to_vec();
    sorted.sort_unstable();
    assert_eq!(in_order, sorted, "structural in-order walk must equal the sorted rowids");
    // Cursor scan returns the rows with byte-identical payloads.
    let scanned = scan_all(pager, root);
    assert_eq!(scanned.len(), expected_ids.len());
    for (rowid, payload) in scanned {
        assert_eq!(payload, payload_for(rowid), "payload for rowid {rowid} must round-trip");
    }
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
/// A mutating b-tree operation that allocates or frees pages legitimately updates four
/// bookkeeping fields — `first_freelist_trunk` and `freelist_count` (the freelist
/// head/count, §1.5), plus `file_change_counter` and the in-header `database_size_pages`
/// (§1.3.2/§1.3.7) — so those are NOT pinned to `header_before`. Every OTHER field is
/// identity the b-tree must never touch, so each is asserted field-for-field against
/// `header_before`; this fails loudly if a page-1 rebuild clobbered, say, `page_size`,
/// `schema_cookie`, `schema_format`, or `text_encoding`. Comparing decoded fields (not
/// raw bytes) makes the check robust to the bookkeeping fields evolving as
/// reclamation/durability grow.
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

#[test]
fn ascending_inserts_scale_and_scan_4096() {
    let n = 5000;
    let ids: Vec<i64> = (1..=n).collect();
    let (pager, root) = build_table(4096, &ids);
    assert!(pager.page_count().unwrap() > 1, "the tree must have grown past one page");
    assert_full_scan(&pager, root, &ids);
}

#[test]
fn shuffled_inserts_scale_and_scan_4096() {
    let n = 5000;
    let ids = shuffled(n);
    let (pager, root) = build_table(4096, &ids);
    assert!(pager.page_count().unwrap() > 1);
    let expected: Vec<i64> = (1..=n).collect();
    assert_full_scan(&pager, root, &expected);
}

#[test]
fn deep_tree_forces_interior_splits_512() {
    // A 512-byte page gives a small interior fanout, so a few thousand rows force
    // interior pages to split (tree height >= 3), not just a single balance-deeper.
    let n = 2000;
    let ids = shuffled(n);
    let (pager, root) = build_table(512, &ids);
    let usable = 512usize;
    let (in_order, height) = walk(&pager, root, usable);
    assert!(height >= 3, "expected an interior-of-interiors (height>=3), got {height}");
    let expected: Vec<i64> = (1..=n).collect();
    assert_eq!(in_order, expected);
    assert_full_scan(&pager, root, &expected);
}

#[test]
fn large_pages_65536_roundtrip() {
    // Big pages: fewer, larger pages. Enough rows to trigger at least one
    // balance-deeper so the root becomes interior even at this page size.
    let n = 4000;
    let ids = shuffled(n);
    let (pager, root) = build_table(65536, &ids);
    let expected: Vec<i64> = (1..=n).collect();
    assert_full_scan(&pager, root, &expected);
}

#[test]
fn seek_exact_present_and_absent() {
    let ids: Vec<i64> = (1..=1000).map(|i| i * 2).collect(); // even rowids 2..=2000
    let (pager, root) = build_table(512, &ids);
    let mut cur = TableCursor::open(&pager, root).unwrap();

    // Present even rowids are found with the right payload.
    for &id in &[2, 4, 100, 1000, 1998, 2000] {
        assert!(cur.seek_exact(id).unwrap(), "rowid {id} should be present");
        assert_eq!(cur.rowid(), id);
        assert_eq!(cur.payload().unwrap().into_owned(), payload_for(id));
    }
    // Absent: odd rowids (gaps), below-min, above-max.
    for &id in &[1, 3, 999, 1001, 1999, 0, -5, 2001, 100_000] {
        assert!(!cur.seek_exact(id).unwrap(), "rowid {id} should be absent");
    }
}

#[test]
fn seek_ge_lands_on_next_present_rowid() {
    let ids: Vec<i64> = (1..=500).map(|i| i * 10).collect(); // 10,20,...,5000
    let (pager, root) = build_table(512, &ids);
    let mut cur = TableCursor::open(&pager, root).unwrap();

    // Below the minimum lands on the minimum.
    assert!(cur.seek_ge(0).unwrap());
    assert_eq!(cur.rowid(), 10);
    assert!(cur.seek_ge(1).unwrap());
    assert_eq!(cur.rowid(), 10);
    // Exact hits stay put.
    assert!(cur.seek_ge(20).unwrap());
    assert_eq!(cur.rowid(), 20);
    // Across a gap: 15 -> 20, 4999 -> 5000.
    assert!(cur.seek_ge(15).unwrap());
    assert_eq!(cur.rowid(), 20);
    assert!(cur.seek_ge(4999).unwrap());
    assert_eq!(cur.rowid(), 5000);
    // At the max boundary.
    assert!(cur.seek_ge(5000).unwrap());
    assert_eq!(cur.rowid(), 5000);
    // Past the max: nothing.
    assert!(!cur.seek_ge(5001).unwrap());
    assert!(!cur.seek_ge(1_000_000).unwrap());
}

#[test]
fn seek_ge_then_next_streams_a_range() {
    let ids: Vec<i64> = (1..=1000).collect();
    let (pager, root) = build_table(512, &ids);
    let mut cur = TableCursor::open(&pager, root).unwrap();

    // A range scan [250, 260]: seek_ge then next until past the upper bound.
    assert!(cur.seek_ge(250).unwrap());
    let mut got = Vec::new();
    while cur.rowid() <= 260 {
        got.push(cur.rowid());
        if !cur.next().unwrap() {
            break;
        }
    }
    assert_eq!(got, (250..=260).collect::<Vec<_>>());
}

#[test]
fn seek_sweep_finds_every_rowid_including_interior_separators_512() {
    // The descent boundary in `nav::choose_child` is `rowid <= separator`: a rowid
    // EQUAL to an interior separator goes into that separator's LEFT child (the
    // separator is that subtree's max). The other seek tests only probe a handful of
    // hand-picked rowids, none of which happens to be an interior separator, so
    // mutating that boundary to `rowid < separator` — which misroutes a separator-valued
    // rowid into the WRONG (right) subtree — goes undetected by them.
    //
    // A full sweep closes that gap by construction: separators ARE inserted rowids, so
    // seeking every r in 1..=N necessarily hits `choose_child` at the equality boundary
    // for every separator at every interior level of a height>=3 tree. Under the `<`
    // mutant, `seek_exact(sep)` returns false for a present rowid and `seek_ge(sep)`
    // skips it — both caught here.
    let n = 2000i64;
    let ids = shuffled(n);
    let (pager, root) = build_table(512, &ids);
    let (_in_order, height) = walk(&pager, root, 512);
    assert!(height >= 3, "need an interior-of-interiors so separators span 2 levels, got {height}");

    let mut cur = TableCursor::open(&pager, root).unwrap();
    for r in 1..=n {
        assert!(cur.seek_exact(r).unwrap(), "seek_exact({r}): a present rowid must be found");
        assert_eq!(cur.rowid(), r, "seek_exact({r}) positioned on the wrong rowid");
        assert_eq!(
            cur.payload().unwrap().into_owned(),
            payload_for(r),
            "seek_exact({r}) returned the wrong payload"
        );
        assert!(cur.seek_ge(r).unwrap(), "seek_ge({r}): a present rowid must be reachable");
        assert_eq!(cur.rowid(), r, "seek_ge({r}) landed past a present rowid");
    }
}

#[test]
fn reverse_scan_last_and_prev_mirror_the_forward_scan_512() {
    // `last`/`prev` are part of the TableCursor API, but the non-overflow reverse
    // walk — `descend_right_spine` plus `prev`'s across-leaf ancestor-climb — is otherwise
    // only exercised by the peer overflow suite. Pin it on a plain table tree: on a
    // shuffled height>=3 512B tree, `last()` lands on the max rowid and repeated `prev()`
    // must visit every rowid exactly once in strict descending order with byte-identical
    // payloads (the exact mirror of `first()`+`next()`), then return false at the start.
    let n = 2000i64;
    let ids = shuffled(n);
    let (pager, root) = build_table(512, &ids);
    let (_in_order, height) = walk(&pager, root, 512);
    assert!(height >= 3, "need a multi-level tree to exercise the prev ancestor-climb, got {height}");

    let mut cur = TableCursor::open(&pager, root).unwrap();
    assert!(cur.last().unwrap(), "last() must position on a non-empty tree");
    assert_eq!(cur.rowid(), n, "last() must land on the max rowid");
    assert_eq!(cur.payload().unwrap().into_owned(), payload_for(n), "last() payload mismatch");

    let mut got: Vec<(i64, Vec<u8>)> = vec![(cur.rowid(), cur.payload().unwrap().into_owned())];
    while cur.prev().unwrap() {
        got.push((cur.rowid(), cur.payload().unwrap().into_owned()));
    }
    // prev at the first row leaves the cursor unpositioned; a further prev stays false.
    assert!(!cur.prev().unwrap(), "prev at the start returns false");

    let expected: Vec<(i64, Vec<u8>)> = (1..=n).rev().map(|r| (r, payload_for(r))).collect();
    assert_eq!(got, expected, "reverse walk must be the exact descending mirror of the forward scan");
}

#[test]
fn negative_and_zero_rowids_order_and_seek_as_signed_i64() {
    // Rowids are signed i64 — an explicit INTEGER PRIMARY KEY can be negative or zero, and
    // real sqlite allows such values. Every end-to-end test above uses only
    // positive rowids, so nothing pins that `choose_child`/`leaf_search` compare rowids as
    // SIGNED i64 (not unsigned) through the tree. Insert a range spanning negatives, zero,
    // and positives in shuffled order (deep enough to force interior separators that are
    // themselves negative), then assert the scan is in ascending signed order across the
    // sign boundary and seeks land correctly on both sides of zero.
    let lo = -1500i64;
    let hi = 1500i64;
    let ordered: Vec<i64> = (lo..=hi).collect(); // negatives, then 0, then positives

    // A deterministic shuffle over the signed range (the shared `shuffled` helper only
    // covers 1..=n, so shuffle in place here).
    let mut ids = ordered.clone();
    let mut state: u64 = 0xD1B54A32D192ED03;
    for i in (1..ids.len()).rev() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (state >> 33) as usize % (i + 1);
        ids.swap(i, j);
    }
    let (mut pager, root) = build_table(512, &ids);
    let (in_order, height) = walk(&pager, root, 512);
    assert!(height >= 3, "range must force interior separators (incl. negative ones), got height {height}");
    assert_eq!(in_order, ordered, "structural walk must be ascending signed-i64 order");

    // Forward scan is ascending signed order across the sign boundary, bytes exact.
    let scanned = scan_all(&pager, root);
    let got_ids: Vec<i64> = scanned.iter().map(|(r, _)| *r).collect();
    assert_eq!(got_ids, ordered, "scan must be ascending signed-i64 order (negatives < 0 < positives)");
    for (rowid, payload) in &scanned {
        assert_eq!(*payload, payload_for(*rowid), "payload mismatch at rowid {rowid}");
    }

    // Exact seeks on both sides of zero, at the extremes, and at zero itself.
    let mut cur = TableCursor::open(&pager, root).unwrap();
    for &r in &[lo, -1000, -1, 0, 1, 1000, hi] {
        assert!(cur.seek_exact(r).unwrap(), "seek_exact({r}) must find a present signed rowid");
        assert_eq!(cur.rowid(), r);
        assert_eq!(cur.payload().unwrap().into_owned(), payload_for(r));
    }
    // seek_ge below the (negative) minimum lands on the minimum; past the max is false.
    assert!(cur.seek_ge(lo - 100).unwrap());
    assert_eq!(cur.rowid(), lo, "seek_ge below min lands on the min (a negative rowid)");
    assert!(!cur.seek_ge(hi + 1).unwrap(), "seek_ge past the max is false");
}

#[test]
fn replace_overwrites_payload_keeps_count() {
    let ids: Vec<i64> = (1..=300).collect();
    let (mut pager, root) = build_table(512, &ids);

    // Overwrite a spread of existing rowids with a new, distinguishable payload.
    let touched = [1i64, 5, 150, 299, 300];
    for &id in &touched {
        let mut new_payload = format!("REPLACED-{id}-").into_bytes();
        while new_payload.len() < 40 {
            new_payload.push(0xEE);
        }
        table_insert(&mut pager, root, id, &new_payload).unwrap();
    }

    let scanned = scan_all(&pager, root);
    assert_eq!(scanned.len(), ids.len(), "REPLACE must not change the row count");
    for (rowid, payload) in scanned {
        if touched.contains(&rowid) {
            let mut expect = format!("REPLACED-{rowid}-").into_bytes();
            while expect.len() < 40 {
                expect.push(0xEE);
            }
            assert_eq!(payload, expect, "rowid {rowid} should show the replaced payload");
        } else {
            assert_eq!(payload, payload_for(rowid));
        }
    }
    // Structure still valid after replacements.
    walk(&pager, root, pager.page_size() as usize);
}

#[test]
fn replace_that_overflows_the_leaf_splits_and_keeps_count() {
    // Fill a single 512-byte leaf, then REPLACE an existing row with a much larger
    // payload so the page overflows — exercising the replace branch on the split
    // path (collect_leaf_cells + balance-deeper), not just the in-place rebuild.
    let ids: Vec<i64> = (1..=9).collect();
    let (mut pager, root) = build_table(512, &ids);
    // Sanity: page 1 (sqlite_schema) + page 2 (this table's single leaf), no split yet.
    assert_eq!(pager.page_count().unwrap(), 2, "page 1 (schema) + one table leaf");

    let big: Vec<u8> = {
        let mut v = b"BIG-5-".to_vec();
        while v.len() < 220 {
            v.push(0xA5);
        }
        v
    };
    table_insert(&mut pager, root, 5, &big).unwrap();
    assert!(pager.page_count().unwrap() > 2, "the oversized replace must have split the leaf");

    let scanned = scan_all(&pager, root);
    assert_eq!(scanned.len(), 9, "REPLACE keeps the row count even when it splits");
    for (rowid, payload) in scanned {
        if rowid == 5 {
            assert_eq!(payload, big);
        } else {
            assert_eq!(payload, payload_for(rowid));
        }
    }
    walk(&pager, root, 512);
}

#[test]
fn replace_at_interior_separators_overwrites_not_duplicates() {
    // REPLACE a rowid that IS an interior separator. Because a separator equals its left
    // subtree's max, the correct `rowid <= separator` descent goes LEFT and finds the
    // row to overwrite. The `rowid < separator` mutant descends RIGHT, does not find the
    // row in that subtree, and INSERTS a duplicate — growing the row count. Every
    // separator is replaced with a SAME-size payload, so no split reshapes the tree
    // mid-batch and each collected separator stays a separator; then the count must be
    // unchanged and each separator must show the new bytes.
    let n = 2000i64;
    let ids = shuffled(n);
    let (mut pager, root) = build_table(512, &ids);
    let usable = 512usize;

    let mut separators = Vec::new();
    collect_separators(&pager, root, usable, &mut separators);
    separators.sort_unstable();
    separators.dedup();
    assert!(
        separators.len() >= 2,
        "a 2000-row 512B tree must have multiple interior separators, got {}",
        separators.len()
    );

    for &sep in &separators {
        table_insert(&mut pager, root, sep, &replaced_payload(sep)).unwrap();
    }

    let scanned = scan_all(&pager, root);
    assert_eq!(
        scanned.len(),
        n as usize,
        "REPLACE at a separator must overwrite in place, not insert a duplicate"
    );
    for (rowid, payload) in scanned {
        if separators.binary_search(&rowid).is_ok() {
            assert_eq!(payload, replaced_payload(rowid), "separator rowid {rowid} must show the replaced bytes");
        } else {
            assert_eq!(payload, payload_for(rowid), "untouched rowid {rowid} must be unchanged");
        }
    }
    // The tree is still a structurally valid table b-tree after the replacements.
    walk(&pager, root, usable);
}

#[test]
fn heterogeneous_row_sizes_insert_must_not_fail() {
    // 512-byte page: leaf capacity = 512 - 8 = 504 bytes for cells + pointers. Eleven
    // 40-byte rows (cell cost 44) fill it to 484. rowid 6 with a 300-byte payload is
    // inline (300 <= X = 512 - 35 = 477, no overflow) but forces a split whose two-way
    // greedy distribution would strand the big cell's right neighbours — the big cell
    // fits beside neither its full left run nor its full right run, so the leaf must
    // split into >= 3 pages. Regression test for the N-way split: before the fix this
    // returned Err("leaf split right page overflowed unexpectedly").
    let mut pager = MemPager::new(512);
    init_database(&mut pager).unwrap();
    let root = create_table_btree(&mut pager).unwrap();
    let small = |rowid: i64| {
        let mut v = format!("r{rowid}").into_bytes();
        while v.len() < 40 {
            v.push(0x11);
        }
        v
    };
    for id in [1i64, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12] {
        table_insert(&mut pager, root, id, &small(id)).unwrap();
    }
    let big = vec![0xA5u8; 300]; // inline: 300 <= 477 = X, no overflow
    table_insert(&mut pager, root, 6, &big).unwrap(); // must NOT error

    // The big row round-trips, and the whole set is present in ascending order.
    let mut cur = TableCursor::open(&pager, root).unwrap();
    assert!(cur.seek_exact(6).unwrap());
    assert_eq!(cur.payload().unwrap().into_owned(), big);
    let scanned = scan_all(&pager, root);
    let got: Vec<i64> = scanned.iter().map(|(r, _)| *r).collect();
    assert_eq!(got, (1..=12).collect::<Vec<_>>());
    for (rowid, payload) in scanned {
        if rowid == 6 {
            assert_eq!(payload, big);
        } else {
            assert_eq!(payload, small(rowid));
        }
    }
    // Every page re-parses as a structurally valid table b-tree.
    walk(&pager, root, 512);
}

#[test]
fn heterogeneous_shuffled_stress_forces_nway_splits_512() {
    // Mixed payload sizes (many tiny, some medium, every 13th large-but-inline)
    // inserted in shuffled order at 512-byte pages, deep enough to force interior
    // splits (height >= 3) AND N-way leaf splits below the root (the propagate path),
    // not only at the root's balance-deeper. A single spurious split error, a dropped
    // row, or a malformed page fails the full scan or the structural walk.
    let n = 4000i64;
    let ids = shuffled(n);
    let mut pager = MemPager::new(512);
    init_database(&mut pager).unwrap();
    let root = create_table_btree(&mut pager).unwrap();
    for &id in &ids {
        table_insert(&mut pager, root, id, &hetero_payload(id)).unwrap();
    }
    let usable = 512usize;
    let (in_order, height) = walk(&pager, root, usable);
    assert!(height >= 3, "expected interior splits (height>=3), got {height}");
    assert_eq!(in_order, (1..=n).collect::<Vec<_>>());
    let scanned = scan_all(&pager, root);
    assert_eq!(scanned.len(), n as usize);
    for (rowid, payload) in scanned {
        assert_eq!(payload, hetero_payload(rowid), "payload mismatch at rowid {rowid}");
    }
}

#[test]
fn empty_tree_first_and_seek_return_false() {
    let mut pager = MemPager::new(4096);
    init_database(&mut pager).unwrap();
    let root = create_table_btree(&mut pager).unwrap();
    let mut cur = TableCursor::open(&pager, root).unwrap();
    assert!(!cur.first().unwrap(), "first on an empty tree is false");
    assert!(!cur.seek_exact(1).unwrap(), "seek_exact on an empty tree is false");
    assert!(!cur.seek_ge(1).unwrap(), "seek_ge on an empty tree is false");
    // next after a failed positioning stays false.
    assert!(!cur.next().unwrap());
}

#[test]
fn single_row_roundtrip() {
    let (pager, root) = build_table(4096, &[42]);
    let mut cur = TableCursor::open(&pager, root).unwrap();
    assert!(cur.first().unwrap());
    assert_eq!(cur.rowid(), 42);
    // Zero-copy contract: a non-overflow row's payload borrows the leaf bytes, so a
    // regression to `Cow::Owned` (an extra copy per row) fails here loudly.
    assert!(
        matches!(cur.payload().unwrap(), std::borrow::Cow::Borrowed(_)),
        "payload() must return Cow::Borrowed (zero-copy) for an inline row"
    );
    assert_eq!(cur.payload().unwrap().into_owned(), payload_for(42));
    assert!(!cur.next().unwrap(), "only one row");
}

#[test]
fn sqlite_schema_root_page1_balance_deeper_keeps_page1() {
    // Insert enough rows directly into the sqlite_schema tree (root = page 1) to
    // force page 1 to balance-deeper. Page 1 must stay the root, become an interior
    // page, and keep its 100-byte database header; the scan must be complete.
    let mut pager = MemPager::new(4096);
    init_database(&mut pager).unwrap();

    let header_before = read_page1_header(&pager);
    let n = 500;
    for id in 1..=n {
        table_insert(&mut pager, 1, id, &payload_for(id)).unwrap();
    }

    // Page 1 is now an interior table page (root grew) but still page 1.
    let view = PageView::new(pager.read_page(1).unwrap(), 1, 4096).unwrap();
    assert_eq!(view.page_type(), PageType::InteriorTable, "page 1 must have grown to interior");
    // The database header survived every rebuild of page 1: every IMMUTABLE identity
    // field still matches the pre-insert snapshot. Whole-header byte-identity would be
    // the wrong check — a spec-correct pager updates the bookkeeping fields
    // (database_size_pages @28, file_change_counter @24) as pages are allocated, and
    // those are not part of the b-tree's header identity.
    assert_header_identity_preserved(&pager, &header_before);
    // Inserts allocate but never free, so the freelist stays empty throughout.
    let after = read_page1_header(&pager);
    assert_eq!(after.first_freelist_trunk, 0, "inserts free nothing: freelist head must stay 0");
    assert_eq!(after.freelist_count, 0, "inserts free nothing: freelist count must stay 0");

    let expected: Vec<i64> = (1..=n).collect();
    let (in_order, _h) = walk(&pager, 1, 4096);
    assert_eq!(in_order, expected);
    let scanned = scan_all(&pager, 1);
    assert_eq!(scanned.len(), n as usize);
    for (rowid, payload) in scanned {
        assert_eq!(payload, payload_for(rowid));
    }
}

#[test]
fn sqlite_schema_root_page1_deepens_to_interior_of_interiors_512() {
    // The page-1 special case driven PAST a single balance-deeper. On 512-byte pages,
    // insert enough rows directly into the sqlite_schema tree (root = page 1) to force
    // page 1 to become an interior-of-interiors (height >= 3). Page 1 must stay the
    // root through BOTH balance-deepers it hits — the first turns the overflowing leaf
    // root into an interior (balance_deeper_leaf), the second turns the overflowing
    // interior root into an interior-over-interiors (balance_deeper_interior) — while
    // keeping its 100-byte database header intact across every rebuild. This covers a
    // distinct code path from the height-2 case: `write_root_interior` rebuilding page
    // 1 from balance_deeper_interior (not just balance_deeper_leaf).
    let mut pager = MemPager::new(512);
    init_database(&mut pager).unwrap();
    let header_before = read_page1_header(&pager);

    let n = 2000;
    let ids = shuffled(n);
    for &id in &ids {
        table_insert(&mut pager, 1, id, &payload_for(id)).unwrap();
        // The root must never move off page 1, whatever b-tree type it currently is.
        let ty = PageView::new(pager.read_page(1).unwrap(), 1, 512).unwrap().page_type();
        assert!(ty.is_table(), "page 1 must remain a table b-tree page");
    }

    // Page 1 is now an interior root, and the tree deepened to at least 3 levels — so
    // page 1's interior content was itself moved down by a balance_deeper_interior.
    let view = PageView::new(pager.read_page(1).unwrap(), 1, 512).unwrap();
    assert_eq!(view.page_type(), PageType::InteriorTable, "page 1 must be an interior root");
    let (in_order, height) = walk(&pager, 1, 512);
    assert!(height >= 3, "page-1 root must be interior-of-interiors (height>=3), got {height}");

    // The database header survived every rebuild of page 1 (leaf and interior): every
    // IMMUTABLE identity field still matches the pre-insert snapshot. Whole-header
    // byte-identity would be the wrong check — a spec-correct pager updates the
    // bookkeeping fields (database_size_pages @28, file_change_counter @24) as pages are
    // allocated, and those are not part of the b-tree's header identity.
    assert_header_identity_preserved(&pager, &header_before);
    // Inserts allocate but never free, so the freelist stays empty throughout.
    let after = read_page1_header(&pager);
    assert_eq!(after.first_freelist_trunk, 0, "inserts free nothing: freelist head must stay 0");
    assert_eq!(after.freelist_count, 0, "inserts free nothing: freelist count must stay 0");

    let expected: Vec<i64> = (1..=n).collect();
    assert_eq!(in_order, expected);
    let scanned = scan_all(&pager, 1);
    assert_eq!(scanned.len(), n as usize);
    for (rowid, payload) in scanned {
        assert_eq!(payload, payload_for(rowid));
    }
}

#[test]
fn interior_split_preserves_root_id_and_grows_pages() {
    // Confirm the root page id is stable even as the tree deepens through interior
    // splits, and that pages were actually allocated.
    let ids = shuffled(3000);
    let mut pager = MemPager::new(512);
    init_database(&mut pager).unwrap();
    let root = create_table_btree(&mut pager).unwrap();
    for &id in &ids {
        table_insert(&mut pager, root, id, &payload_for(id)).unwrap();
        // The root page's b-tree header may change type, but its id must not move.
        let ty = PageView::new(pager.read_page(root).unwrap(), root, 512).unwrap().page_type();
        assert!(ty.is_table());
    }
    assert!(pager.page_count().unwrap() > 10, "many pages should exist for 3000 rows on 512B pages");
    let (_in_order, height) = walk(&pager, root, 512);
    assert!(height >= 3);
}
