//! End-to-end tests for index-b-tree DELETE (`index_delete`): remove keys — leaf
//! keys AND interior divider keys — and prove the tree stays a structurally valid,
//! balanced CLASSIC B-tree that real sqlite could read, through single deletes,
//! delete-all (forward/reverse/shuffled) collapsing to an empty root leaf at the
//! same id, mass contiguous deletes, delete-then-reinsert, big overflow-key deletes
//! whose chains are reclaimed, and page-count boundedness (freed pages reused).
//!
//! Ordering oracle: most tests use `[first_col, rowid]` integer records, whose index
//! order equals native `(i64, i64)` tuple order (integers compare by value; the
//! trailing rowid breaks ties), decoded by `decode_ab`. The two variable-width tests
//! (`delete_interior_divider_with_larger_predecessor_must_not_error`,
//! `delete_all_spilled_text_dividers_validates_and_reclaims_chains`) use `[Text, rowid]`
//! keys that spill onto overflow pages, ordered by `text_payload_prefix` (the text
//! payload after the record header, which works on a spilled cell's inline prefix).
//! Every oracle is computed independently of the engine's own comparator, so a
//! comparator bug cannot hide behind a matching bug in the check.
//!
//! Each `tests/*.rs` is its own crate, so the structural validator is written here
//! rather than shared — a small duplication that keeps this file self-contained and
//! lets `walk` assert the exact post-delete invariants.
//!
//! Required-scenario coverage map (the delete rebalance spec names each required
//! scenario; several are covered by a test under a behaviorally-descriptive name rather
//! than the spec's literal name — grep here to find the test for a spec name):
//!   - delete_random_subset_keeps_classic_b_and_k2 ....... `delete_random_subset_keeps_classic_b_and_k2`
//!   - delete_interior_divider_keys_then_validate ........ `delete_interior_divider_keys_pulls_up_predecessor`
//!   - delete_contiguous_range_forces_merges ............. `mass_delete_contiguous_range_keeps_tree_valid`
//!   - delete_all_then_reinsert .......................... `delete_every_key_shuffled_order` + `delete_then_reinsert_roundtrip`
//!   - delete_overflow_keys_frees_chains_and_stays_valid . `delete_all_spilled_text_dividers_validates_and_reclaims_chains` + `delete_big_overflow_key_frees_its_chain`
//!   - randomized_insert_delete_mix_matches_reference ..... `randomized_insert_delete_mix_matches_reference`
//!   - k1_interior_root_after_deletes_is_accepted ........ `k1_interior_root_after_deletes_is_accepted`
//! Density (a delete keeps pages dense, not just valid): `delete_scatter_reclaims_pages_via_leaf_merges`.

use minisqlite_btree::{
    create_index_btree, index_delete, index_insert, init_database, IndexCursor,
};
use minisqlite_fileformat::{decode_record, encode_record, PageType, PageView};
use minisqlite_pager::{MemPager, PageId, Pager};
use minisqlite_types::Value;

/// An index key record `[Integer(first), Integer(rowid)]`.
fn key(first: i64, rowid: i64) -> Vec<u8> {
    encode_record(&[Value::Integer(first), Value::Integer(rowid)])
}

/// Decode a 2-column integer key record into `(first, rowid)`.
fn decode_ab(rec: &[u8]) -> (i64, i64) {
    let vals = decode_record(rec);
    let a = match vals.first() {
        Some(Value::Integer(i)) => *i,
        other => panic!("index key first column not an integer: {other:?}"),
    };
    let b = match vals.get(1) {
        Some(Value::Integer(i)) => *i,
        other => panic!("index key rowid column not an integer: {other:?}"),
    };
    (a, b)
}

/// A deterministic Fisher-Yates shuffle (fixed LCG) so tests exercise out-of-order
/// build/delete without a PRNG dependency.
fn shuffled(mut values: Vec<(i64, i64)>) -> Vec<(i64, i64)> {
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    for i in (1..values.len()).rev() {
        let j = next() % (i + 1);
        values.swap(i, j);
    }
    values
}

/// Build an index over `pairs` (inserted in order) and return `(pager, root)`.
fn build_index(page_size: u32, pairs: &[(i64, i64)]) -> (MemPager, PageId) {
    let mut pager = MemPager::new(page_size);
    init_database(&mut pager).unwrap();
    let root = create_index_btree(&mut pager).unwrap();
    for &(a, b) in pairs {
        index_insert(&mut pager, root, &key(a, b)).unwrap();
    }
    (pager, root)
}

/// Scan the whole index forward via `first` + `next`.
fn scan_all(pager: &dyn Pager, root: PageId) -> Vec<(i64, i64)> {
    let mut cur = IndexCursor::open(pager, root).unwrap();
    let mut out = Vec::new();
    if !cur.first().unwrap() {
        return out;
    }
    loop {
        out.push(decode_ab(&cur.key().unwrap()));
        if !cur.next().unwrap() {
            break;
        }
    }
    out
}

/// Scan the whole index backward via `last` + `prev` — the result is in DESCENDING key
/// order (the exact reverse of `scan_all`). A reverse scan re-walks the classic-B
/// interior dividers in the opposite direction, so it independently proves no key was
/// lost or duplicated by a delete rebalance.
fn scan_all_reverse(pager: &dyn Pager, root: PageId) -> Vec<(i64, i64)> {
    let mut cur = IndexCursor::open(pager, root).unwrap();
    let mut out = Vec::new();
    if !cur.last().unwrap() {
        return out;
    }
    loop {
        out.push(decode_ab(&cur.key().unwrap()));
        if !cur.prev().unwrap() {
            break;
        }
    }
    out
}

/// A variable-width TEXT index key `[Text(text), Integer(rowid)]` — the shape of a
/// real index on a text column (indexed value + rowid tie-breaker). Long text spills
/// onto overflow pages exactly as sqlite does.
fn tkey(text: &str, rowid: i64) -> Vec<u8> {
    encode_record(&[Value::Text(text.to_string()), Value::Integer(rowid)])
}

/// Read a SQLite varint, returning `(value, bytes_consumed)`.
fn read_varint(buf: &[u8]) -> (u64, usize) {
    let mut result: u64 = 0;
    for i in 0..9 {
        let byte = buf[i];
        if i == 8 {
            return ((result << 8) | byte as u64, 9);
        }
        result = (result << 7) | (byte & 0x7f) as u64;
        if byte & 0x80 == 0 {
            return (result, i + 1);
        }
    }
    (result, 9)
}

/// Independent ordering oracle for `[Text, rowid]` keys that works on a SPILLED cell's
/// inline prefix: return the text-column payload bytes, which start right after the
/// record header. For every key in these tests the text begins with a fixed-width
/// value that totally orders the keys, so the (possibly truncated) prefix is enough to
/// order them — and it deliberately does NOT decode the whole record (a spilled key's
/// inline bytes are only a prefix) nor use the engine's own comparator. It also skips
/// the record header, whose rowid serial-type byte precedes the payload and would
/// otherwise misorder raw bytes.
fn text_payload_prefix(rec: &[u8]) -> Vec<u8> {
    let (hdr_len, _) = read_varint(rec);
    let start = (hdr_len as usize).min(rec.len());
    rec[start..].to_vec()
}

/// Scan the whole index forward, returning each full (reassembled) key record. Used by
/// the variable-width TEXT tests, whose spilled keys cannot be decoded to a tuple.
fn scan_all_raw(pager: &dyn Pager, root: PageId) -> Vec<Vec<u8>> {
    let mut cur = IndexCursor::open(pager, root).unwrap();
    let mut out = Vec::new();
    if !cur.first().unwrap() {
        return out;
    }
    loop {
        out.push(cur.key().unwrap().as_ref().to_vec());
        if !cur.next().unwrap() {
            break;
        }
    }
    out
}

/// Scan the whole index backward, returning each full (reassembled) key record in
/// DESCENDING order — the raw-key analogue of `scan_all_reverse`, for the variable-width
/// TEXT tests whose spilled keys cannot be decoded to a tuple. Re-walking the classic-B
/// dividers in the opposite direction is an independent witness that no key was lost or
/// duplicated by a delete rebalance (including an overflow-driven interior split).
fn scan_all_raw_reverse(pager: &dyn Pager, root: PageId) -> Vec<Vec<u8>> {
    let mut cur = IndexCursor::open(pager, root).unwrap();
    let mut out = Vec::new();
    if !cur.last().unwrap() {
        return out;
    }
    loop {
        out.push(cur.key().unwrap().as_ref().to_vec());
        if !cur.prev().unwrap() {
            break;
        }
    }
    out
}

/// Recursively validate the classic index b-tree rooted at `id` and return its full
/// in-order key records and height (leaf = 1). Asserts: an interior divider is a REAL
/// entry that the in-order walk INTERLEAVES between its child subtrees
/// (`walk(c0) ++ [d0] ++ walk(c1) ++ ... ++ [d_{m-1}] ++ walk(cm)`); dividers strictly
/// ascending; each subtree strictly separated by its bordering dividers; all children
/// the same height (balanced); the whole interleaved sequence strictly ascending (no
/// key twice). Re-parsing every page through `PageView::new` is itself the structural
/// re-parse the real-sqlite bar demands.
///
/// `decode` maps a raw key record to a comparable, independently-computed ordering key
/// (native `(i64,i64)` tuples for integer tests, `(String,i64)` for the variable-width
/// TEXT test), so the ordering checks never lean on the engine's own comparator. A
/// spilled key's `local_payload` is only its inline prefix, so `decode` must not need
/// the whole record; the TEXT test embeds the ordering-relevant prefix early in the key.
fn walk<K: Ord + std::fmt::Debug>(
    pager: &dyn Pager,
    id: PageId,
    root: PageId,
    usable: usize,
    decode: &dyn Fn(&[u8]) -> K,
) -> (Vec<Vec<u8>>, u32) {
    let (is_leaf, leaf_keys, children, dividers) = {
        let view = PageView::new(pager.read_page(id).unwrap(), id, usable).unwrap();
        assert!(view.page_type().is_index(), "page {id} is not an index page");
        if view.page_type().is_leaf() {
            let count = view.cell_count() as usize;
            let mut keys = Vec::with_capacity(count);
            for i in 0..count {
                let c = view.index_leaf_cell(i).unwrap();
                keys.push(c.local_payload.to_vec());
            }
            (true, keys, Vec::new(), Vec::new())
        } else {
            let count = view.cell_count() as usize;
            let mut children = Vec::with_capacity(count + 1);
            let mut dividers = Vec::with_capacity(count);
            for i in 0..count {
                let c = view.index_interior_cell(i).unwrap();
                children.push(c.left_child);
                dividers.push(c.local_payload.to_vec());
            }
            children.push(view.right_most_pointer().unwrap());
            (false, Vec::new(), children, dividers)
        }
    };

    if is_leaf {
        for w in leaf_keys.windows(2) {
            assert!(decode(&w[0]) < decode(&w[1]), "leaf {id} keys not strictly ascending");
        }
        return (leaf_keys, 1);
    }

    for w in dividers.windows(2) {
        assert!(decode(&w[0]) < decode(&w[1]), "interior {id} dividers not strictly ascending");
    }
    assert_eq!(children.len(), dividers.len() + 1, "interior {id}: children must be dividers + 1");
    // fileformat2 §1.6: every non-root interior page must have K>=2 dividers (>=3
    // children); "In all other cases, K is 2 or more." The ROOT is the sole exception
    // (a shallow 2-child root has K==1 and cannot be K>=2, so it is never forced; the
    // root may even be a leaf). So require K>=2 for every interior that is not the root
    // — the invariant a correct classic-B delete rebalance must re-establish. A delete
    // that removed a child slot without rebalancing would leave a K<2 non-root interior,
    // which this assertion rejects.
    if id != root {
        assert!(
            dividers.len() >= 2,
            "interior {id}: §1.6 requires K>=2 (>=3 children) for a non-root interior, got K={}",
            dividers.len()
        );
    }

    let mut all: Vec<Vec<u8>> = Vec::new();
    let mut child_height: Option<u32> = None;
    for j in 0..children.len() {
        let (sub, h) = walk(pager, children[j], root, usable, decode);
        assert!(!sub.is_empty(), "child {} of interior {id} is empty", children[j]);
        match child_height {
            None => child_height = Some(h),
            Some(ph) => assert_eq!(ph, h, "interior {id} children heights differ (unbalanced)"),
        }
        if j > 0 {
            assert!(
                decode(&dividers[j - 1]) < decode(sub.first().unwrap()),
                "interior {id}: subtree {j} min must be > divider {}",
                j - 1
            );
        }
        if j < dividers.len() {
            assert!(
                decode(sub.last().unwrap()) < decode(&dividers[j]),
                "interior {id} cell {j}: subtree max must be < divider"
            );
        }
        all.extend(sub);
        if j < dividers.len() {
            all.push(dividers[j].clone());
        }
    }

    for w in all.windows(2) {
        assert!(
            decode(&w[0]) < decode(&w[1]),
            "interior {id} in-order keys not strictly ascending (a duplicate would tie here)"
        );
    }
    (all, child_height.unwrap() + 1)
}

/// Structural validation + a forward cursor scan must both equal the sorted expected
/// pairs, each key exactly once. Integer-key variant (small integers never overflow,
/// so the walk sees whole inline keys).
fn assert_scan_and_structure(pager: &MemPager, root: PageId, expected: &[(i64, i64)]) {
    let usable = pager.page_size() as usize;
    let (in_order, _height) = walk(pager, root, root, usable, &decode_ab);
    let mut sorted = expected.to_vec();
    sorted.sort_unstable();
    let in_order_pairs: Vec<(i64, i64)> = in_order.iter().map(|k| decode_ab(k)).collect();
    assert_eq!(in_order_pairs, sorted, "structural in-order walk must equal the sorted keys");
    let scanned = scan_all(pager, root);
    assert_eq!(scanned, sorted, "forward cursor scan must equal the sorted keys, each once");
}

/// The root's divider key records (inline; integer keys never overflow). Empty if the
/// root is a leaf.
fn root_divider_keys(pager: &MemPager, root: PageId) -> Vec<Vec<u8>> {
    let usable = pager.page_size() as usize;
    let view = PageView::new(pager.read_page(root).unwrap(), root, usable).unwrap();
    if view.page_type().is_leaf() {
        return Vec::new();
    }
    (0..view.cell_count() as usize)
        .map(|i| view.index_interior_cell(i).unwrap().local_payload.to_vec())
        .collect()
}

/// Assert the root id still holds a valid, empty index LEAF page — the end state
/// after deleting every key (the tree collapsed all the way back to an empty leaf at
/// the original root id).
fn assert_empty_root_leaf(pager: &MemPager, root: PageId) {
    let usable = pager.page_size() as usize;
    let view = PageView::new(pager.read_page(root).unwrap(), root, usable).unwrap();
    assert_eq!(view.page_type(), PageType::LeafIndex, "collapsed root must be an index leaf");
    assert_eq!(view.cell_count(), 0, "collapsed root leaf must be empty");
    let mut cur = IndexCursor::open(pager, root).unwrap();
    assert!(!cur.first().unwrap(), "empty index: first() must be false");
}

/// Count the b-tree pages actually REACHABLE from `root` (interiors + leaves), the
/// index's LIVE page footprint. This excludes freelist pages, so — unlike the pager's
/// high-water `page_count()` — it shrinks when a delete merges nodes, making it the
/// right density measure: a scattered-delete index that failed to reclaim under-full
/// leaves shows a large live footprint even though `page_count()` (monotonic) would not.
/// Integer keys never spill, so no overflow pages are involved.
fn reachable_pages(pager: &MemPager, root: PageId, usable: usize) -> usize {
    fn rec(pager: &MemPager, id: PageId, usable: usize, seen: &mut usize) {
        *seen += 1;
        let view = PageView::new(pager.read_page(id).unwrap(), id, usable).unwrap();
        if view.page_type().is_interior() {
            let n = view.cell_count() as usize;
            for i in 0..n {
                rec(pager, view.index_interior_cell(i).unwrap().left_child, usable, seen);
            }
            rec(pager, view.right_most_pointer().unwrap(), usable, seen);
        }
    }
    let mut seen = 0;
    rec(pager, root, usable, &mut seen);
    seen
}

// -----------------------------------------------------------------------------------

#[test]
fn delete_present_leaf_key_and_absent_key() {
    // Even first-columns only, so odd probes are genuine gaps.
    let pairs: Vec<(i64, i64)> = (1..=200).map(|i| (i * 2, i)).collect();
    let (mut pager, root) = build_index(512, &shuffled(pairs.clone()));
    assert!(pager.page_count().unwrap() > 1, "tree must be multi-page for this to be interesting");
    assert_scan_and_structure(&pager, root, &pairs);

    // Absent keys return false and leave the tree byte-for-byte unchanged.
    let before = scan_all(&pager, root);
    for &(a, b) in &[(1i64, 1), (3, 3), (401, 200), (0, 0), (-4, 1), (1000, 1)] {
        assert!(!index_delete(&mut pager, root, &key(a, b)).unwrap(), "absent ({a},{b}) => false");
    }
    assert_eq!(scan_all(&pager, root), before, "absent deletes must not change the tree");

    // A present leaf key: removed, gone, tree still valid.
    let (a, b) = (50, 25);
    assert!(index_delete(&mut pager, root, &key(a, b)).unwrap(), "present key must delete");
    assert!(!index_delete(&mut pager, root, &key(a, b)).unwrap(), "re-delete is false");
    let expect: Vec<(i64, i64)> = pairs.iter().copied().filter(|&p| p != (a, b)).collect();
    assert_scan_and_structure(&pager, root, &expect);
}

#[test]
fn delete_interior_divider_keys_pulls_up_predecessor() {
    // A tall tree (height >= 3) so interior pages carry real divider entries. Deleting
    // a divider must reduce to a predecessor swap and keep the tree a valid classic
    // B-tree with the deleted key gone and every survivor exactly once.
    let pairs: Vec<(i64, i64)> = (1..=4000).map(|i| (i, i)).collect();
    let (mut pager, root) = build_index(512, &shuffled(pairs.clone()));
    let usable = 512usize;
    let (_o, h) = walk(&pager, root, root, usable, &decode_ab);
    assert!(h >= 3, "want height >= 3 so root dividers sit above real subtrees, got {h}");

    let dividers = root_divider_keys(&pager, root);
    assert!(dividers.len() >= 2, "root should have several dividers, got {}", dividers.len());

    let mut remaining: std::collections::BTreeSet<(i64, i64)> = pairs.iter().copied().collect();
    // Delete each root divider (they are real, unique keys living in the interior).
    for d in &dividers {
        let (a, b) = decode_ab(d);
        assert!(remaining.contains(&(a, b)), "root divider ({a},{b}) must be a present key");
        assert!(index_delete(&mut pager, root, d).unwrap(), "divider ({a},{b}) must delete");
        assert!(remaining.remove(&(a, b)));
        assert!(!index_delete(&mut pager, root, d).unwrap(), "re-delete divider is false");
        let expect: Vec<(i64, i64)> = remaining.iter().copied().collect();
        assert_scan_and_structure(&pager, root, &expect);
    }
}

/// Delete every key of a freshly built (shuffled) tree in `order`, validating after
/// each delete, then assert the tree collapsed to an empty leaf at the same root id.
/// Over a full sequence this necessarily exercises leaf removal at both edges and the
/// middle, sibling merge / redistribute as leaves empty, and root collapse.
fn delete_all_in_order(order: &[(i64, i64)]) {
    let n = order.len() as i64;
    let pairs: Vec<(i64, i64)> = (1..=n).map(|i| (i, i)).collect();
    let (mut pager, root) = build_index(512, &shuffled(pairs));
    let mut remaining: std::collections::BTreeSet<(i64, i64)> = (1..=n).map(|i| (i, i)).collect();

    for &(a, b) in order {
        assert!(index_delete(&mut pager, root, &key(a, b)).unwrap(), "present ({a},{b}) must delete");
        assert!(remaining.remove(&(a, b)));
        assert!(!index_delete(&mut pager, root, &key(a, b)).unwrap(), "re-delete ({a},{b}) is false");
        let expect: Vec<(i64, i64)> = remaining.iter().copied().collect();
        assert_scan_and_structure(&pager, root, &expect);
    }
    assert_empty_root_leaf(&pager, root);

    // Re-insert everything and the full set must be back, valid, at the same root id.
    for i in 1..=n {
        index_insert(&mut pager, root, &key(i, i)).unwrap();
    }
    let full: Vec<(i64, i64)> = (1..=n).map(|i| (i, i)).collect();
    assert_scan_and_structure(&pager, root, &full);
}

#[test]
fn delete_every_key_forward_order() {
    let order: Vec<(i64, i64)> = (1..=300).map(|i| (i, i)).collect();
    delete_all_in_order(&order);
}

#[test]
fn delete_every_key_reverse_order() {
    let order: Vec<(i64, i64)> = (1..=300).rev().map(|i| (i, i)).collect();
    delete_all_in_order(&order);
}

#[test]
fn delete_every_key_shuffled_order() {
    let order = shuffled((1..=300).map(|i| (i, i)).collect());
    delete_all_in_order(&order);
}

#[test]
fn mass_delete_multilevel_collapses_and_reclaims_pages() {
    // A tall tree (height >= 3), delete every key in shuffled order. Underfull leaves
    // and interiors must merge with a sibling (the divider pulled down / promoted), the
    // root must collapse height back down to an empty leaf at the SAME id, and freed
    // pages must be reused so page_count stays bounded across the delete-all + a full
    // re-insert (not doubled by leaks).
    let n = 4000i64;
    let pairs: Vec<(i64, i64)> = (1..=n).map(|i| (i, i)).collect();
    let (mut pager, root) = build_index(512, &shuffled(pairs.clone()));
    let usable = 512usize;
    let (_o, h0) = walk(&pager, root, root, usable, &decode_ab);
    assert!(h0 >= 3, "expected a height>=3 tree to force sibling merges and root collapse, got {h0}");
    let built_pages = pager.page_count().unwrap();

    let order = shuffled(pairs.clone());
    let mut remaining: std::collections::BTreeSet<(i64, i64)> = pairs.iter().copied().collect();
    let mut deleted = 0i64;
    let mut saw_smaller_height = false;
    for &(a, b) in &order {
        assert!(index_delete(&mut pager, root, &key(a, b)).unwrap());
        remaining.remove(&(a, b));
        deleted += 1;
        // Full walk+scan is O(n); every delete would be O(n^2). Validate every 256
        // deletes and near the very end to stay bounded but still localize a break.
        if deleted % 256 == 0 || remaining.len() <= 4 {
            let expect: Vec<(i64, i64)> = remaining.iter().copied().collect();
            assert_scan_and_structure(&pager, root, &expect);
            let ty = PageView::new(pager.read_page(root).unwrap(), root, usable).unwrap().page_type();
            assert!(ty.is_index(), "root {root} must stay an index page while collapsing");
            let (_ord, h) = walk(&pager, root, root, usable, &decode_ab);
            if h < h0 {
                saw_smaller_height = true;
            }
        }
    }
    assert!(saw_smaller_height, "mass delete must reduce tree height (root collapse) at a stable id");
    assert_empty_root_leaf(&pager, root);

    // delete never allocates, so the high-water page_count is unchanged by the delete-all.
    assert_eq!(
        pager.page_count().unwrap(),
        built_pages,
        "delete-all must not grow the file (it only frees)"
    );
    // Re-insert everything: allocations must reuse the freed pages before growing, so
    // the file does not balloon. A leak would make this ~2x built_pages.
    for i in 1..=n {
        index_insert(&mut pager, root, &key(i, i)).unwrap();
    }
    assert_scan_and_structure(&pager, root, &pairs);
    assert!(
        pager.page_count().unwrap() <= built_pages + built_pages / 10 + 4,
        "re-insert after delete-all must reuse freed pages (bounded), got {} vs built {}",
        pager.page_count().unwrap(),
        built_pages
    );
}

#[test]
fn mass_delete_contiguous_range_keeps_tree_valid() {
    // Delete a large contiguous middle-to-front range from a multi-level tree; the
    // survivors must scan correctly and the tree must stay a valid classic B-tree.
    let n = 4000i64;
    let pairs: Vec<(i64, i64)> = (1..=n).map(|i| (i, i)).collect();
    let (mut pager, root) = build_index(512, &shuffled(pairs.clone()));
    let usable = 512usize;
    assert!(walk(&pager, root, root, usable, &decode_ab).1 >= 3, "want a tall tree");

    for i in 1..=3000i64 {
        assert!(index_delete(&mut pager, root, &key(i, i)).unwrap(), "contiguous delete {i}");
    }
    let survivors: Vec<(i64, i64)> = (3001..=n).map(|i| (i, i)).collect();
    assert_scan_and_structure(&pager, root, &survivors);
}

#[test]
fn delete_then_reinsert_roundtrip() {
    let n = 800i64;
    let pairs: Vec<(i64, i64)> = (1..=n).map(|i| (i, i)).collect();
    let (mut pager, root) = build_index(512, &shuffled(pairs.clone()));

    let removed: Vec<(i64, i64)> = (1..=n).filter(|r| r % 3 == 0).map(|i| (i, i)).collect();
    for &(a, b) in &removed {
        assert!(index_delete(&mut pager, root, &key(a, b)).unwrap());
    }
    let survivors: Vec<(i64, i64)> = (1..=n).filter(|r| r % 3 != 0).map(|i| (i, i)).collect();
    assert_scan_and_structure(&pager, root, &survivors);

    let mut reinsert = removed.clone();
    reinsert.reverse();
    for &(a, b) in &reinsert {
        index_insert(&mut pager, root, &key(a, b)).unwrap();
    }
    assert_scan_and_structure(&pager, root, &pairs);
}

#[test]
fn delete_same_prefix_keys_by_rowid() {
    // Every key shares first column 7 with a distinct rowid, so all are unique full
    // keys distinguished only by the trailing rowid. Deleting a spread must leave the
    // rest correctly ordered by rowid.
    let rowids: Vec<i64> = (0..600).collect();
    let pairs: Vec<(i64, i64)> = rowids.iter().map(|&r| (7, r)).collect();
    let (mut pager, root) = build_index(512, &shuffled(pairs.clone()));
    assert_scan_and_structure(&pager, root, &pairs);

    let mut remaining: std::collections::BTreeSet<(i64, i64)> = pairs.iter().copied().collect();
    for &r in &[0i64, 599, 300, 1, 598, 250, 251] {
        assert!(index_delete(&mut pager, root, &key(7, r)).unwrap(), "(7,{r}) must delete");
        remaining.remove(&(7, r));
    }
    let expect: Vec<(i64, i64)> = remaining.iter().copied().collect();
    assert_scan_and_structure(&pager, root, &expect);
}

#[test]
fn delete_from_empty_index_is_false() {
    let mut pager = MemPager::new(4096);
    init_database(&mut pager).unwrap();
    let root = create_index_btree(&mut pager).unwrap();
    for &(a, b) in &[(-1i64, 0), (0, 0), (1, 1), (1_000_000, 5)] {
        assert!(!index_delete(&mut pager, root, &key(a, b)).unwrap(), "delete on empty index is false");
    }
    assert_empty_root_leaf(&pager, root);
}

#[test]
fn single_key_delete_to_empty_root_leaf() {
    let (mut pager, root) = build_index(4096, &[(42, 99)]);
    assert!(index_delete(&mut pager, root, &key(42, 99)).unwrap(), "the one key must delete");
    assert!(!index_delete(&mut pager, root, &key(42, 99)).unwrap(), "re-delete is false");
    assert_empty_root_leaf(&pager, root);
    // Re-insert works and lands at the same root id.
    index_insert(&mut pager, root, &key(7, 7)).unwrap();
    assert_scan_and_structure(&pager, root, &[(7, 7)]);
}

#[test]
fn delete_big_overflow_key_frees_its_chain() {
    // On a 512-byte page a key larger than one page spills onto a multi-page overflow
    // chain (index.rs::large_overflow_index_key). Deleting it must free that chain, so
    // inserting another big key afterward reuses the freed pages rather than growing
    // the file by a whole second chain.
    let mut pager = MemPager::new(512);
    init_database(&mut pager).unwrap();
    let root = create_index_btree(&mut pager).unwrap();

    let big1 = encode_record(&[Value::Text("x".repeat(800)), Value::Integer(1)]);
    assert!(big1.len() > 512, "key must exceed a page to force a multi-page chain");
    index_insert(&mut pager, root, &big1).unwrap();
    // Also insert a couple of small keys so the tree is not a single cell.
    for i in 0..3i64 {
        index_insert(&mut pager, root, &key(i, i)).unwrap();
    }
    let pages_with_chain = pager.page_count().unwrap();

    // Delete the big key: present -> true, gone from the scan, its chain reclaimed.
    assert!(index_delete(&mut pager, root, &big1).unwrap(), "big key must delete");
    assert!(!index_delete(&mut pager, root, &big1).unwrap(), "re-delete big key is false");
    {
        let mut cur = IndexCursor::open(&pager, root).unwrap();
        let mut found_big = false;
        if cur.first().unwrap() {
            loop {
                if cur.key().unwrap().as_ref() == big1.as_slice() {
                    found_big = true;
                }
                if !cur.next().unwrap() {
                    break;
                }
            }
        }
        assert!(!found_big, "the deleted big key must be gone from the scan");
    }

    // Insert a different big key of the same size: it reuses the freed chain pages, so
    // page_count must not grow past the earlier high-water mark by another full chain.
    let big2 = encode_record(&[Value::Text("y".repeat(800)), Value::Integer(2)]);
    index_insert(&mut pager, root, &big2).unwrap();
    assert!(
        pager.page_count().unwrap() <= pages_with_chain,
        "deleting a big key must free its chain so a same-size insert reuses it (got {} vs {})",
        pager.page_count().unwrap(),
        pages_with_chain
    );
    // The new big key round-trips byte-exact and the small keys survive.
    let mut cur = IndexCursor::open(&pager, root).unwrap();
    assert!(cur.first().unwrap());
    // First three are the small (0,0),(1,1),(2,2); the big "y…" key sorts last (text > numeric).
    for i in 0..3i64 {
        assert_eq!(decode_ab(&cur.key().unwrap()), (i, i));
        assert!(cur.next().unwrap());
    }
    assert_eq!(cur.key().unwrap().as_ref(), big2.as_slice(), "the surviving big key round-trips");
    assert!(!cur.next().unwrap(), "exactly four keys remain");
}

#[test]
fn delete_interior_divider_with_larger_predecessor_must_not_error() {
    // Regression for the divider-replacement overflow: deleting an interior divider
    // swaps its in-order predecessor up over it, and the predecessor can encode WIDER
    // than the divider (a long spilled TEXT key replacing a tiny one) — which overflows
    // a near-full interior page. That must SPLIT and propagate (as an insert does), not
    // return Err. Real sqlite deletes these fine; a build that errors is a differential
    // failure on any non-integer index.
    //
    // 512-byte pages fill and split interior pages quickly, so a near-full interior is
    // common. For each group g, two keys ADJACENT in index order:
    //   huge(g)  = [Text("g{g:04}/" + "a"*600), 2g]   -> spills onto overflow pages
    //   small(g) = [Text("g{g:04}0"),           2g+1] -> tiny inline cell
    // '/'(0x2f) < '0'(0x30) at byte 6, so huge(g) is the immediate in-order predecessor
    // of small(g): whenever small(g) is an interior divider, its left subtree's max
    // (its predecessor) is the big spilled huge(g).
    let mut pager = MemPager::new(512);
    init_database(&mut pager).unwrap();
    let root = create_index_btree(&mut pager).unwrap();

    let groups = 800i64;
    let huge = |g: i64| tkey(&format!("g{:04}/{}", g, "a".repeat(600)), 2 * g);
    let small = |g: i64| tkey(&format!("g{:04}0", g), 2 * g + 1);
    for g in 0..groups {
        index_insert(&mut pager, root, &huge(g)).unwrap();
        index_insert(&mut pager, root, &small(g)).unwrap();
    }

    // High-water page count before the deletes: a wide-predecessor divider swap that
    // overflows SPLITS the interior (allocating a fresh `right_id`), so the high-water
    // mark must strictly grow during the delete loop. This pins that the overflow→split
    // branch is actually taken — without it, a future change could silently degrade to
    // passing the Ok(true) asserts on a tree that never exercised the split path.
    let pages_before_small_deletes = pager.page_count().unwrap();

    // Delete every small key (all present). Any that is an interior divider triggers the
    // wide-predecessor swap; each MUST return Ok(true), never error.
    for g in 0..groups {
        assert!(
            index_delete(&mut pager, root, &small(g)).unwrap(),
            "deleting present key small({g}) must return Ok(true), not error on a divider overflow"
        );
        assert!(!index_delete(&mut pager, root, &small(g)).unwrap(), "re-delete small({g}) is false");
    }
    assert!(
        pager.page_count().unwrap() > pages_before_small_deletes,
        "wide-predecessor divider deletes must allocate via an interior split: \
         high-water page count {} did not grow past {}",
        pager.page_count().unwrap(),
        pages_before_small_deletes
    );

    // Exactly the huge keys survive, in group order, as a valid classic B-tree.
    let usable = 512usize;
    let expected: Vec<Vec<u8>> = (0..groups).map(huge).collect();
    assert_eq!(scan_all_raw(&pager, root), expected, "survivors must be the huge keys in group order");
    let (in_order, _h) = walk(&pager, root, root, usable, &text_payload_prefix);
    assert_eq!(in_order.len() as i64, groups, "one surviving key per group after deleting the smalls");

    // And the huge keys are still individually deletable to a clean empty root.
    for g in 0..groups {
        assert!(index_delete(&mut pager, root, &huge(g)).unwrap(), "huge({g}) must delete");
    }
    assert_empty_root_leaf(&pager, root);
}

#[test]
fn delete_leaf_underflow_redistribute_wider_divider_must_not_error() {
    // Regression for the rebalance-parent overflow in `index_delete::rebalance_child`.
    // This is the REDISTRIBUTE sibling of the predecessor-swap case above
    // (`delete_interior_divider_with_larger_predecessor_must_not_error`): when a leaf
    // underflows, `index_delete` GATHERS it with a sibling and REDISTRIBUTES into two
    // pages, PROMOTING a fresh boundary divider back up to the parent. That promoted
    // divider is a real index key of ARBITRARY width, so it can encode WIDER than the
    // SHORT window-internal divider it replaces — and on a near-full interior packed
    // with short dividers, rewriting the parent then OVERFLOWS one page. Real sqlite
    // SPLITS the interior and propagates up (the DELETE succeeds); a build that instead
    // fails closed returns Err where sqlite returns success — an Err-vs-success
    // differential divergence that aborts the enclosing statement/transaction.
    //
    // Construction: 1500 keys ordered by an 8-digit zero-padded prefix (so index order ==
    // i order via `text_payload_prefix`, independent of width). Every 6th key is a WIDE
    // spilled TEXT key; the rest are tiny. Inserting/deleting in a fixed shuffled order
    // packs a height-3 tree whose non-root leaf-parent interiors are near-full of SHORT
    // dividers, and the 7th delete drives a leaf-underflow redistribute (win_count=2,
    // g=2) whose promoted boundary divider is one of the WIDE keys.
    //
    // Red->green proof: PRE-FIX the 7th delete returns
    //   Err("index_delete: rebalance parent overflowed a page")
    // (the exact `write_interior(..., "rebalance parent")` fail-closed at the parent
    // rewrite); the fix converts that into a `split_index_interior` + `propagate_split`,
    // so the delete returns Ok(true). The red->green transition IS the proof the split
    // path is exercised, because a delete-heavy sequence frees pages the split can reuse,
    // so a grown high-water page count is NOT a reliable dodge-proof here.
    //
    // COVERAGE CAVEAT: the exact overflow depends on the interior packing (512-byte
    // pages, key widths, and the shuffled insert order). If you change any of those,
    // re-verify this regression still bites by reverting the fix and confirming the 7th
    // delete errors with "index_delete: rebalance parent overflowed a page" — mirroring
    // how `delete_all_spilled_text_dividers_validates_and_reclaims_chains` documents its
    // own packing dependence. The `order[6] == 526` assert below pins the witness so a
    // drift in the shared shuffle is caught loudly rather than silently skipping the path.
    let n = 1500i64;
    let vkey = |i: i64| {
        if i % 6 == 0 {
            tkey(&format!("{:08}/{}", i, "a".repeat(300)), i) // wide: spills to overflow
        } else {
            tkey(&format!("{:08}0", i), i) // narrow: tiny inline cell
        }
    };

    let mut pager = MemPager::new(512);
    init_database(&mut pager).unwrap();
    let root = create_index_btree(&mut pager).unwrap();

    // One deterministic order for both insert and delete (the packing this needs). The
    // shared `shuffled` permutes by position only, so shuffling `(i, 0)` pairs yields the
    // same i-permutation the config was derived from.
    let order: Vec<i64> =
        shuffled((0..n).map(|i| (i, 0)).collect()).into_iter().map(|(i, _)| i).collect();
    for &i in &order {
        index_insert(&mut pager, root, &vkey(i)).unwrap();
    }
    let usable = 512usize;
    let (_o, h) = walk(&pager, root, root, usable, &text_payload_prefix);
    assert!(
        h >= 3,
        "want height>=3 so the overflowing leaf-parent is a NON-root interior (the \
         propagate_split branch of the fix), got {h}"
    );
    assert_eq!(order[6], 526, "witness delete drifted; re-derive the config (see COVERAGE CAVEAT)");

    // Delete through the witness. PRE-FIX, the 7th delete (`order[6]`) `.unwrap()`s an
    // Err and this panics (RED); POST-FIX every delete returns Ok(true) (GREEN).
    let mut remaining: std::collections::BTreeSet<i64> = (0..n).collect();
    for &i in &order[..=6] {
        assert!(
            index_delete(&mut pager, root, &vkey(i)).unwrap(),
            "deleting present key {i} must return Ok(true), not error on the rebalance-parent overflow"
        );
        assert!(remaining.remove(&i));
    }

    // After the overflow-driven split: still a valid classic B-tree holding exactly the
    // survivors (structural `walk` via the independent `text_payload_prefix` oracle), and
    // BOTH cursor directions agree with the sorted reference — an independent witness that
    // the split lost/duplicated no key.
    walk(&pager, root, root, usable, &text_payload_prefix);
    let expected_fwd: Vec<Vec<u8>> = remaining.iter().map(|&i| vkey(i)).collect();
    assert_eq!(scan_all_raw(&pager, root), expected_fwd, "forward scan must equal the surviving keys in order");
    let mut expected_rev = expected_fwd.clone();
    expected_rev.reverse();
    assert_eq!(
        scan_all_raw_reverse(&pager, root),
        expected_rev,
        "reverse scan must equal the survivors in descending order"
    );

    // The post-split tree stays fully healthy: delete every remaining key (more merges,
    // redistributes, and possibly further overflow-splits) down to a clean empty root.
    for &i in &order[7..] {
        assert!(index_delete(&mut pager, root, &vkey(i)).unwrap(), "surviving key {i} must delete");
    }
    assert_empty_root_leaf(&pager, root);
}

#[test]
fn delete_all_spilled_text_dividers_validates_and_reclaims_chains() {
    // Uniform-width TEXT keys large enough to SPILL onto an overflow page, so interior
    // dividers are themselves spilled keys. Deleting them (including deep, non-root
    // dividers) exercises the spilled-divider paths the integer tests can't: installing
    // a spilled predecessor over a divider + freeing the old divider's chain (the
    // predecessor-swap path), and MOVING spilled dividers (with their chains intact)
    // between pages as the rebalance merges/redistributes — a divider is a live key, so
    // its chain is preserved, never leaked or double-freed. Per-delete walk+scan
    // localizes a break; delete-all + reinsert proves every reclaimed page/chain (the
    // deleted keys' and the merged-away nodes') is reused (no leak).
    //
    // Ordering oracle: same-shape keys whose text starts with a zero-padded index, via
    // `text_payload_prefix` (independent of the engine's comparator, works on the
    // spilled inline prefix).
    let m = 160i64;
    let tk = |i: i64| tkey(&format!("{:06}{}", i, "z".repeat(200)), i);

    let mut pager = MemPager::new(512);
    init_database(&mut pager).unwrap();
    let root = create_index_btree(&mut pager).unwrap();
    for &(i, _) in &shuffled((0..m).map(|i| (i, i)).collect()) {
        index_insert(&mut pager, root, &tk(i)).unwrap();
    }
    let usable = 512usize;
    let (_o, h) = walk(&pager, root, root, usable, &text_payload_prefix);
    assert!(h >= 3, "want a height>=3 tree with real spilled interior dividers, got {h}");
    // Confirm interior dividers really spilled (overflow chains to reclaim exist).
    {
        let view = PageView::new(pager.read_page(root).unwrap(), root, usable).unwrap();
        assert!(view.cell_count() >= 1, "root must carry dividers");
        assert!(
            view.index_interior_cell(0).unwrap().overflow_page.is_some(),
            "a root divider must be a spilled key (its chain is what delete must reclaim)"
        );
    }
    let built_pages = pager.page_count().unwrap();

    // Delete every key in shuffled order, validating structure + scan after each.
    let mut remaining: std::collections::BTreeSet<i64> = (0..m).collect();
    for &(i, _) in &shuffled((0..m).map(|i| (i, i)).collect()) {
        assert!(index_delete(&mut pager, root, &tk(i)).unwrap(), "present spilled key {i} must delete");
        remaining.remove(&i);
        assert!(!index_delete(&mut pager, root, &tk(i)).unwrap(), "re-delete {i} is false");
        // `walk` re-parses every page and asserts the classic-B invariants (dividers
        // ascending, subtree separation, balance, strictly-ascending global in-order)
        // through the independent `text_payload_prefix` oracle — the per-delete
        // structural check. Then the cursor scan must equal the exact remaining set.
        walk(&pager, root, root, usable, &text_payload_prefix);
        let expected: Vec<Vec<u8>> = remaining.iter().map(|&x| tk(x)).collect();
        assert_eq!(scan_all_raw(&pager, root), expected, "scan mismatch after deleting {i}");
    }
    assert_empty_root_leaf(&pager, root);

    // Reinsert everything: allocations must reuse the freed leaf/interior/overflow pages
    // (the divider chains were reclaimed), so the file does not balloon (~2x = a leak).
    for i in 0..m {
        index_insert(&mut pager, root, &tk(i)).unwrap();
    }
    let expected: Vec<Vec<u8>> = (0..m).map(tk).collect();
    assert_eq!(scan_all_raw(&pager, root), expected, "reinserted spilled set must round-trip");
    assert!(
        pager.page_count().unwrap() <= built_pages + built_pages / 10 + 8,
        "reinsert after delete-all must reuse reclaimed pages/chains (got {} vs built {})",
        pager.page_count().unwrap(),
        built_pages
    );
}

#[test]
fn delete_random_subset_keeps_classic_b_and_k2() {
    // PRIMARY §1.6 regression. Build a height>=3 tree and delete a deterministic
    // pseudo-random ~70% of its keys IN BATCHES; after EACH batch assert the
    // strengthened `walk` (every non-root interior K>=2, root-exempt) AND that a full
    // forward scan AND a full reverse scan each equal the exact remaining sorted key
    // set (each surviving key exactly once). Scattered deletes are the case where a
    // delete that removes a child slot from a K==2 attach-point without rebalancing
    // would leave a malformed K<2 non-root interior — `walk` rejects that, so this pins
    // that the classic-B merge/redistribute keeps every non-root interior at K>=2. The
    // reverse scan re-walks the interior dividers in the opposite direction, an
    // independent witness that no key is lost/duplicated.
    let n = 4000i64;
    let pairs: Vec<(i64, i64)> = (1..=n).map(|i| (i, i)).collect();
    let (mut pager, root) = build_index(512, &shuffled(pairs.clone()));
    let usable = 512usize;
    let (_o, h) = walk(&pager, root, root, usable, &decode_ab);
    assert!(h >= 3, "want height >= 3 so K>=2 is checked on interiors below the root, got {h}");

    // Deterministic pseudo-random order; delete the first ~70%.
    let order = shuffled(pairs.clone());
    let cut = (n as usize * 7) / 10;
    let to_delete = &order[..cut];

    let mut remaining: std::collections::BTreeSet<(i64, i64)> = pairs.iter().copied().collect();
    for batch in to_delete.chunks(137) {
        for &(a, b) in batch {
            assert!(index_delete(&mut pager, root, &key(a, b)).unwrap(), "present ({a},{b}) must delete");
            assert!(remaining.remove(&(a, b)), "({a},{b}) must have still been present");
        }
        // Structural K>=2 walk + forward scan (both via assert_scan_and_structure), then
        // the reverse scan must equal the same set in descending order.
        let expect: Vec<(i64, i64)> = remaining.iter().copied().collect();
        assert_scan_and_structure(&pager, root, &expect);
        let mut rev = expect.clone();
        rev.reverse();
        assert_eq!(
            scan_all_reverse(&pager, root),
            rev,
            "reverse scan must equal the remaining keys in descending order"
        );
    }
    assert!(remaining.len() < pairs.len(), "the subset delete must have removed keys");
}

#[test]
fn randomized_insert_delete_mix_matches_reference() {
    // The correctness proof that nothing is ever lost, duplicated, or misordered: a long
    // deterministic mix (fixed LCG, no RNG crate) of inserts and deletes over a bounded
    // key domain (so deletes hit present keys and inserts re-add absent ones — real
    // churn that interleaves merge / redistribute / split / collapse), checked against a
    // reference `BTreeSet` of the live keys. Every delete's returned present/absent bit
    // must match the reference, and periodically `walk` (K>=2) plus a forward scan must
    // equal the reference in sorted order.
    let mut pager = MemPager::new(512);
    init_database(&mut pager).unwrap();
    let root = create_index_btree(&mut pager).unwrap();
    let usable = 512usize;

    let mut reference: std::collections::BTreeSet<(i64, i64)> = std::collections::BTreeSet::new();
    let mut state: u64 = 0xDEAD_BEEF_CAFE_F00D;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        state
    };
    let domain = 1200i64; // small enough that inserts and deletes collide often

    for step in 0..20_000usize {
        let a = (next() % domain as u64) as i64;
        let pair = (a, a);
        // Always consume one RNG draw for the op choice; bias to inserts until the tree
        // has grown, then mix evenly, so the tree spends most of the run multi-level.
        let choose_insert = (next() & 1) == 0;
        if choose_insert || reference.len() < 250 {
            index_insert(&mut pager, root, &key(a, a)).unwrap();
            reference.insert(pair);
        } else {
            let deleted = index_delete(&mut pager, root, &key(a, a)).unwrap();
            let was_present = reference.remove(&pair);
            assert_eq!(deleted, was_present, "step {step}: delete({pair:?}) return must match reference membership");
        }
        if step % 500 == 0 {
            walk(&pager, root, root, usable, &decode_ab);
            let expect: Vec<(i64, i64)> = reference.iter().copied().collect();
            assert_eq!(scan_all(&pager, root), expect, "step {step}: scan must equal the live reference set");
        }
    }

    // Final full validation of structure + both scan directions against the reference.
    walk(&pager, root, root, usable, &decode_ab);
    let expect: Vec<(i64, i64)> = reference.iter().copied().collect();
    assert_eq!(scan_all(&pager, root), expect, "final forward scan must equal the live reference set");
    let mut rev = expect.clone();
    rev.reverse();
    assert_eq!(scan_all_reverse(&pager, root), rev, "final reverse scan must equal the live set reversed");
}

#[test]
fn k1_interior_root_after_deletes_is_accepted() {
    // A shallow 2-child (K==1) INTERIOR root is VALID per §1.6 (the root exemption) — a
    // 2-child root cannot be K>=2, so the rebalance must neither force it up to K>=2 nor
    // wrongly collapse it while it still separates two real subtrees. Drive a tree down
    // by deletes until its root is a K==1 interior, and assert `walk` (root-exempt)
    // accepts it and the scan is exactly the remaining keys. Pins that the root
    // exemption is load-bearing (a K>=2-everywhere validator would reject this).
    let n = 300i64;
    let pairs: Vec<(i64, i64)> = (1..=n).map(|i| (i, i)).collect();
    let (mut pager, root) = build_index(512, &shuffled(pairs.clone()));
    let usable = 512usize;
    assert!(!root_divider_keys(&pager, root).is_empty(), "root must start as an interior");

    let order = shuffled(pairs.clone());
    let mut remaining: std::collections::BTreeSet<(i64, i64)> = pairs.iter().copied().collect();
    let mut saw_k1_interior_root = false;
    for &(a, b) in &order {
        assert!(index_delete(&mut pager, root, &key(a, b)).unwrap());
        remaining.remove(&(a, b));

        // A K==1 interior root is an interior page (not a leaf) with exactly one divider
        // (== two children). Merges shrink the root's fan-out one child at a time, so it
        // necessarily passes through this state before a final 2->1 collapse to a leaf.
        let root_is_k1_interior = {
            let view = PageView::new(pager.read_page(root).unwrap(), root, usable).unwrap();
            view.page_type().is_index() && !view.page_type().is_leaf() && view.cell_count() == 1
        };
        if root_is_k1_interior {
            saw_k1_interior_root = true;
            // walk exempts the root, so a K==1 root is accepted; the scan must still be
            // exactly the remaining keys, each once, in order.
            let expect: Vec<(i64, i64)> = remaining.iter().copied().collect();
            assert_scan_and_structure(&pager, root, &expect);
            let mut rev = expect.clone();
            rev.reverse();
            assert_eq!(scan_all_reverse(&pager, root), rev, "reverse scan on a K==1 root must be correct");
            break; // proven while the root is still a valid K==1 interior
        }
    }
    assert!(
        saw_k1_interior_root,
        "deletes must pass the root through a valid K==1 (2-child) interior before collapsing to a leaf"
    );
}

#[test]
fn delete_scatter_reclaims_pages_via_leaf_merges() {
    // DENSITY (perf). Deleting a scattered ~70% of the keys must keep the index
    // COMPACT, not merely valid: its LIVE (reachable) page footprint must stay within a
    // bounded factor (1.5x) of a freshly built index holding exactly the survivors. The
    // classic-B underflow trigger fires when a leaf falls under ~1/3 full (not only when
    // it empties), so adjacent shrunken leaves MERGE and the tree shrinks with the data.
    //
    // Measured (deterministic shuffle): widened trigger => 52 live pages, a fresh build of
    // the same 1200 survivors => 119, an "empties only" trigger => 243. So the merge keeps
    // the tree DENSER than even a fresh insert-built one (which leaves leaves ~half full),
    // while "empties only" bloats to ~2x fresh. Verified load-bearing: reverting
    // `leaf_underflows` to `Ok(n == 0)` pushes `live_after` to 243 and this goes red.
    let n = 4000i64;
    let pairs: Vec<(i64, i64)> = (1..=n).map(|i| (i, i)).collect();
    let usable = 512usize;
    let (mut pager, root) = build_index(512, &shuffled(pairs.clone()));
    let (_o, h) = walk(&pager, root, root, usable, &decode_ab);
    assert!(h >= 3, "want height >= 3 so merges cascade through interior levels, got {h}");

    let order = shuffled(pairs.clone());
    let cut = (n as usize * 7) / 10;
    let survivors: Vec<(i64, i64)> = order[cut..].to_vec();
    for &(a, b) in &order[..cut] {
        assert!(index_delete(&mut pager, root, &key(a, b)).unwrap(), "present ({a},{b}) must delete");
    }

    // Still a valid classic B-tree holding exactly the survivors...
    let mut expect = survivors.clone();
    expect.sort_unstable();
    assert_scan_and_structure(&pager, root, &expect);

    // ...and dense: within 2x a fresh build of the same survivor set.
    let live_after = reachable_pages(&pager, root, usable);
    let (fresh_pager, fresh_root) = build_index(512, &shuffled(survivors.clone()));
    let live_fresh = reachable_pages(&fresh_pager, fresh_root, usable);
    assert!(
        live_after <= live_fresh * 3 / 2,
        "scattered-deleted index must stay dense: {live_after} live pages holding {} keys vs \
         {live_fresh} for a fresh build of the same keys (exceeded the 1.5x bounded factor — \
         under-full leaves were not reclaimed)",
        survivors.len(),
    );
}
