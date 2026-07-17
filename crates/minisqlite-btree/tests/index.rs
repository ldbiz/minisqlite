//! End-to-end tests for the index b-tree: build trees large enough to force
//! multiple leaf and interior splits, then verify ordered scans (forward and
//! reverse), point/prefix/`<=` seeks, duplicate indexed prefixes distinguished by
//! rowid, the empty tree, large keys that spill onto overflow pages (round-tripping
//! byte-exact through `key()`, including comparison on reassembled spilled tails and
//! spilled interior dividers), and — the real-sqlite bar — that every page re-parses
//! as a structurally valid index b-tree (the separator invariant holds and every
//! level is in key order and balanced).
//!
//! Ordering oracle: index keys here are `[first_col, rowid]` records of integers, so
//! their index order equals Rust's native `(i64, i64)` tuple order (integers compare
//! by value; the trailing rowid breaks ties). The tests sort with that native order
//! rather than calling the engine's own comparator, so a bug in `compare_index_keys`
//! cannot hide behind a matching bug in the check.

use minisqlite_btree::{create_index_btree, index_insert, init_database, IndexCursor};
use minisqlite_fileformat::{decode_record, encode_record, PageView};
use minisqlite_pager::{MemPager, PageId, Pager};
use minisqlite_types::Value;

/// An index key record `[Integer(first), Integer(rowid)]`.
fn key(first: i64, rowid: i64) -> Vec<u8> {
    encode_record(&[Value::Integer(first), Value::Integer(rowid)])
}

/// A prefix search key of just the first indexed column (no trailing rowid).
fn prefix(first: i64) -> Vec<u8> {
    encode_record(&[Value::Integer(first)])
}

/// Decode a 2-column integer key record back into `(first, rowid)`.
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

/// A deterministic shuffle of `values` (a fixed LCG Fisher-Yates) so tests exercise
/// out-of-order insertion without a PRNG dependency.
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

/// Build an index over `pairs` (inserted in the given order) and return
/// `(pager, root)`. The `root` is the page id `create_index_btree` handed back; every
/// caller then reads the tree back through that same id, which only succeeds if
/// `index_insert` kept the root anchored there across all splits and balance-deepers.
fn build_index(page_size: u32, pairs: &[(i64, i64)]) -> (MemPager, PageId) {
    let mut pager = MemPager::new(page_size);
    init_database(&mut pager).unwrap();
    let root = create_index_btree(&mut pager).unwrap();
    for &(a, b) in pairs {
        index_insert(&mut pager, root, &key(a, b)).unwrap();
    }
    (pager, root)
}

/// Scan the whole index forward via `first` + `next`, returning keys in order.
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

/// Scan the whole index backward via `last` + `prev`, returning keys in order.
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

/// Recursively validate the index b-tree page `id` inside the tree rooted at `root`,
/// returning its full in-order key records and its height (leaf = 1). Asserts the
/// CLASSIC B-tree invariant: an interior divider is a REAL index entry (present
/// exactly once) that sits strictly between its two child subtrees, so the in-order
/// sequence INTERLEAVES each divider between the subtrees around it:
///   `walk(child_0) ++ [div_0] ++ walk(child_1) ++ ... ++ [div_{m-1}] ++ walk(child_m)`.
/// Checks: dividers strictly ascending; every key under `child_j` strictly `<` `div_j`
/// and every key under `child_{j+1}` strictly `>` `div_j`; all children the same
/// height (balanced); the whole interleaved list strictly ascending (so no key appears
/// twice — a B+-style duplicated divider would show up as an equal neighbour); and,
/// per fileformat2 §1.6, that every NON-ROOT interior holds K >= 2 dividers (>= 3
/// children). The `root` is exempt from the K >= 2 check: §1.6 forbids K < 2 only off
/// page 1, but a shallow index root may legitimately be a 2-child (K = 1) interior —
/// real sqlite produces those, and a 2-child root cannot have >= 2 dividers.
/// Small integer keys never overflow, so `overflow_page` must be `None` throughout.
fn walk(pager: &dyn Pager, id: PageId, root: PageId, usable: usize) -> (Vec<Vec<u8>>, u32) {
    // Read the page into owned lists, then drop the borrow before recursing.
    let (is_leaf, leaf_keys, children, dividers) = {
        let view = PageView::new(pager.read_page(id).unwrap(), id, usable).unwrap();
        assert!(view.page_type().is_index(), "page {id} is not an index page");
        if view.page_type().is_leaf() {
            let count = view.cell_count() as usize;
            let mut keys = Vec::with_capacity(count);
            for i in 0..count {
                let c = view.index_leaf_cell(i).unwrap();
                assert!(c.overflow_page.is_none(), "leaf {id} cell {i} unexpectedly overflowed");
                keys.push(c.local_payload.to_vec());
            }
            (true, keys, Vec::new(), Vec::new())
        } else {
            let count = view.cell_count() as usize;
            let mut children = Vec::with_capacity(count + 1);
            let mut dividers = Vec::with_capacity(count);
            for i in 0..count {
                let c = view.index_interior_cell(i).unwrap();
                assert!(c.overflow_page.is_none(), "interior {id} cell {i} unexpectedly overflowed");
                children.push(c.left_child);
                dividers.push(c.local_payload.to_vec());
            }
            children.push(view.right_most_pointer().unwrap());
            (false, Vec::new(), children, dividers)
        }
    };

    if is_leaf {
        for w in leaf_keys.windows(2) {
            assert!(decode_ab(&w[0]) < decode_ab(&w[1]), "leaf {id} keys not strictly ascending");
        }
        return (leaf_keys, 1);
    }

    for w in dividers.windows(2) {
        assert!(decode_ab(&w[0]) < decode_ab(&w[1]), "interior {id} dividers not strictly ascending");
    }
    assert_eq!(children.len(), dividers.len() + 1, "interior {id}: children must be dividers + 1");

    // fileformat2 §1.6: a NON-ROOT interior b-tree page must hold K >= 2 dividers
    // (>= 3 children). A K < 2 interior (e.g. the lone-child K=0 right page a greedy
    // interior split strands) is malformed — it wastes a level of pointer indirection
    // and is not a shape real sqlite writes. The root is exempt (a shallow root may be
    // a valid 2-child K=1 interior); this is the invariant the validator must enforce
    // to catch a split that emits an underfull interior half.
    if id != root {
        assert!(
            dividers.len() >= 2,
            "non-root interior {id} has K={} dividers; §1.6 requires K >= 2 (>= 3 children)",
            dividers.len()
        );
    }

    // Interleave each divider between its surrounding child subtrees (classic B-tree).
    let mut all: Vec<Vec<u8>> = Vec::new();
    let mut child_height: Option<u32> = None;
    for j in 0..children.len() {
        let (sub, h) = walk(pager, children[j], root, usable);
        assert!(!sub.is_empty(), "child {} of interior {id} is empty", children[j]);
        match child_height {
            None => child_height = Some(h),
            Some(ph) => assert_eq!(ph, h, "interior {id} children heights differ (unbalanced)"),
        }
        // The divider before this subtree (if any) is strictly below its minimum.
        if j > 0 {
            assert!(
                decode_ab(&dividers[j - 1]) < decode_ab(sub.first().unwrap()),
                "interior {id}: subtree {j} min must be > divider {}",
                j - 1
            );
        }
        // The divider after this subtree (if any) is strictly above its maximum.
        if j < dividers.len() {
            assert!(
                decode_ab(sub.last().unwrap()) < decode_ab(&dividers[j]),
                "interior {id} cell {j}: subtree max must be < divider (divider is a real entry between subtrees)"
            );
        }
        all.extend(sub);
        if j < dividers.len() {
            all.push(dividers[j].clone());
        }
    }

    for w in all.windows(2) {
        assert!(
            decode_ab(&w[0]) < decode_ab(&w[1]),
            "interior {id} in-order keys not strictly ascending (a duplicated key would tie here)"
        );
    }
    (all, child_height.unwrap() + 1)
}

/// Structural validation + forward scan must equal the sorted expected pairs, each
/// key exactly once. In this classic B-tree an interior divider is a REAL entry that
/// the in-order `walk` interleaves between its child subtrees and the forward cursor
/// scan yields in turn — so both must reproduce the full sorted key list with no key
/// dropped (a B+ leaf-only scan would miss the promoted interior entries) and none
/// doubled (a B+ copied divider would appear twice).
fn assert_scan_and_structure(pager: &MemPager, root: PageId, expected: &[(i64, i64)]) {
    let usable = pager.page_size() as usize;
    let (in_order, _height) = walk(pager, root, root, usable);
    let mut sorted = expected.to_vec();
    sorted.sort_unstable();
    let in_order_pairs: Vec<(i64, i64)> = in_order.iter().map(|k| decode_ab(k)).collect();
    assert_eq!(in_order_pairs, sorted, "structural in-order walk must equal the sorted keys");
    let scanned = scan_all(pager, root);
    assert_eq!(scanned, sorted, "forward cursor scan must equal the sorted keys, each once");
}

#[test]
fn many_keys_force_multilevel_splits_and_scan_in_order_512() {
    // A 512-byte page gives small fanout, so a few thousand SHUFFLED keys force
    // interior pages to split (tree height >= 3), not just a single balance-deeper.
    // This is the PRIMARY regression guard for the interior-split fix: a greedy split
    // that overflows a non-rightmost interior by one divider strands a lone-child (K=0)
    // right page, and — unlike the ascending build below — a shuffled build leaves that
    // strand in the FINAL tree (no later key descends into it to heal it), so the
    // strengthened walk() below catches the malformed K<2 interior.
    let pairs: Vec<(i64, i64)> = (1..=4000).map(|i| (i, i)).collect();
    let (pager, root) = build_index(512, &shuffled(pairs.clone()));
    let usable = 512usize;
    let (_in_order, height) = walk(&pager, root, root, usable);
    assert!(height >= 3, "expected an interior-of-interiors (height>=3), got {height}");
    assert!(pager.page_count().unwrap() > 1, "the index must have grown past one page");
    assert_scan_and_structure(&pager, root, &pairs);
}

#[test]
fn ascending_keys_force_multilevel_interior_splits_512() {
    // Strict-ascending inserts drive many interior splits, all on the right spine, so
    // this stresses the balanced-split path under monotonic load and confirms a purely
    // ascending build stays well-formed (height >= 3, every non-root interior K >= 2).
    // NB: ascending is NOT a standalone guard for the lone-child bug — a greedy split's
    // K=0 right half sits on the active right spine and is HEALED by the next larger
    // keys descending into it, so a pure-ascending FINAL tree strands no K<2 page. The
    // shuffled companion above is what pins a surviving K<2; here we prove the fix keeps
    // the monotonic (worst case for greedy) build balanced and valid.
    let pairs: Vec<(i64, i64)> = (1..=4000).map(|i| (i, i)).collect();
    let (pager, root) = build_index(512, &pairs);
    let usable = 512usize;
    let (_in_order, height) = walk(&pager, root, root, usable);
    assert!(height >= 3, "expected height >= 3 to force interior splits, got {height}");
    assert!(pager.page_count().unwrap() > 1, "the index must have grown past one page");
    assert_scan_and_structure(&pager, root, &pairs);
}

#[test]
fn multilevel_tree_forward_and_reverse_scan_visit_every_key_once() {
    // Direct regression for the classic-B cursor: build a tree tall enough (height >= 3)
    // that interior pages carry real entries, then confirm BOTH a forward scan and a
    // reverse scan visit EVERY key exactly once in order. In a classic B-tree those
    // promoted keys live only in interior pages, so a B+-style leaf-only cursor would
    // silently drop them — the count check below would fail. Interior dividers moved up
    // (not copied) also means none is double-counted.
    let pairs: Vec<(i64, i64)> = (1..=4000).map(|i| (i, i)).collect();
    let (pager, root) = build_index(512, &shuffled(pairs.clone()));
    let usable = 512usize;
    let (in_order, height) = walk(&pager, root, root, usable);
    assert!(height >= 3, "expected height >= 3 so interior pages carry entries, got {height}");

    let mut sorted = pairs.clone();
    sorted.sort_unstable();

    // The structural in-order walk (interleaving interior dividers with leaf keys) is
    // itself the full set, exactly once each.
    let in_order_pairs: Vec<(i64, i64)> = in_order.iter().map(|k| decode_ab(k)).collect();
    assert_eq!(in_order_pairs, sorted, "in-order walk must contain every key exactly once");

    // Forward: first()+next() yields every key exactly once, ascending.
    let forward = scan_all(&pager, root);
    assert_eq!(forward.len(), pairs.len(), "forward scan must yield every key (none dropped)");
    assert_eq!(forward, sorted, "forward scan must visit every key exactly once in order");

    // Reverse: last()+prev() yields every key exactly once, and reversed equals sorted.
    let mut reverse = scan_all_reverse(&pager, root);
    assert_eq!(reverse.len(), pairs.len(), "reverse scan must yield every key (none dropped)");
    reverse.reverse();
    assert_eq!(reverse, sorted, "reverse scan must visit every key exactly once in order");
}

#[test]
fn reinsert_every_key_including_interior_dividers_is_noop() {
    // Classic-B: many keys are promoted into interior pages as dividers. Re-inserting
    // ANY existing key — including one that now lives in an interior page — must be an
    // idempotent no-op (index keys are unique). Without the descent-time divider
    // duplicate check, re-inserting an interior divider would fail to find it in a leaf
    // (it is not there) and wrongly splice a second copy into a leaf, duplicating a
    // "unique" key. Re-insert EVERY key and assert nothing duplicates.
    let pairs: Vec<(i64, i64)> = (1..=4000).map(|i| (i, i)).collect();
    let (mut pager, root) = build_index(512, &shuffled(pairs.clone()));
    let usable = 512usize;
    let (_in_order, height) = walk(&pager, root, root, usable);
    assert!(height >= 3, "expected height >= 3 so interior dividers exist to re-insert, got {height}");

    let before = scan_all(&pager, root);
    assert_eq!(before.len(), pairs.len(), "sanity: every key present before re-insertion");
    // Re-insert every key (shuffled), covering leaf keys and interior dividers alike.
    for &(a, b) in &shuffled(pairs.clone()) {
        index_insert(&mut pager, root, &key(a, b)).unwrap();
    }
    let after = scan_all(&pager, root);
    assert_eq!(after.len(), pairs.len(), "the key count must be unchanged after re-insertion");
    assert_eq!(before, after, "re-inserting keys (incl. interior dividers) must not duplicate any");
    // The tree is still a structurally valid classic B-tree with each key exactly once.
    assert_scan_and_structure(&pager, root, &pairs);
}

#[test]
fn ascending_and_shuffled_agree_4096() {
    let pairs: Vec<(i64, i64)> = (1..=3000).map(|i| (i * 2, i)).collect();
    let (asc_pager, asc_root) = build_index(4096, &pairs);
    let (shuf_pager, shuf_root) = build_index(4096, &shuffled(pairs.clone()));
    assert_scan_and_structure(&asc_pager, asc_root, &pairs);
    assert_scan_and_structure(&shuf_pager, shuf_root, &pairs);
    // Insertion order must not change the final logical contents.
    assert_eq!(scan_all(&asc_pager, asc_root), scan_all(&shuf_pager, shuf_root));
}

#[test]
fn duplicate_indexed_prefix_distinct_rowids_all_coexist() {
    // Every key shares first column 7 but has a distinct rowid, so all are unique
    // full keys and must coexist, ordered by the trailing rowid.
    let rowids: Vec<i64> = (0..600).collect();
    let pairs: Vec<(i64, i64)> = rowids.iter().map(|&r| (7, r)).collect();
    let (pager, root) = build_index(512, &shuffled(pairs.clone()));
    let scanned = scan_all(&pager, root);
    let expected: Vec<(i64, i64)> = rowids.iter().map(|&r| (7, r)).collect();
    assert_eq!(scanned, expected, "same-prefix keys must all coexist, ordered by rowid");
    assert_scan_and_structure(&pager, root, &pairs);
}

#[test]
fn reinsert_identical_key_is_idempotent_noop() {
    let pairs: Vec<(i64, i64)> = (1..=200).map(|i| (i, i)).collect();
    let (mut pager, root) = build_index(512, &pairs);
    let before = scan_all(&pager, root);
    // Re-insert a spread of existing keys; index keys are unique, so nothing changes.
    for &(a, b) in &[(1, 1), (50, 50), (137, 137), (200, 200)] {
        index_insert(&mut pager, root, &key(a, b)).unwrap();
    }
    let after = scan_all(&pager, root);
    assert_eq!(before, after, "re-inserting an identical key must not duplicate it");
    assert_scan_and_structure(&pager, root, &pairs);
}

#[test]
fn prefix_seek_ge_lands_on_first_key_of_group() {
    // Keys (c, r) for c in 0..10 and r in 0..50: 500 keys across many pages.
    let pairs: Vec<(i64, i64)> = (0..10).flat_map(|c| (0..50).map(move |r| (c, r))).collect();
    let (pager, root) = build_index(512, &shuffled(pairs));
    let mut cur = IndexCursor::open(&pager, root).unwrap();

    // A prefix seek on just the first column lands on the smallest key whose first
    // column is >= the prefix — here the first (7, *) entry.
    assert!(cur.seek_ge(&prefix(7)).unwrap());
    assert_eq!(decode_ab(&cur.key().unwrap()), (7, 0));

    // Iterating while the first column stays 7 reads the whole point-lookup group.
    let mut group = Vec::new();
    loop {
        let (c, r) = decode_ab(&cur.key().unwrap());
        if c != 7 {
            break;
        }
        group.push((c, r));
        if !cur.next().unwrap() {
            break;
        }
    }
    assert_eq!(group, (0..50).map(|r| (7, r)).collect::<Vec<_>>());

    // A prefix past the last group finds nothing.
    assert!(!cur.seek_ge(&prefix(10)).unwrap());
    // A prefix below the first group lands on the very first key.
    assert!(cur.seek_ge(&prefix(-1)).unwrap());
    assert_eq!(decode_ab(&cur.key().unwrap()), (0, 0));
}

#[test]
fn seek_ge_full_key_exact_and_gap() {
    // Even first columns only (0,2,4,...), one rowid each, so odd probes fall in gaps.
    let pairs: Vec<(i64, i64)> = (0..500).map(|i| (i * 2, 1)).collect();
    let (pager, root) = build_index(512, &pairs);
    let mut cur = IndexCursor::open(&pager, root).unwrap();

    // Exact hit stays put.
    assert!(cur.seek_ge(&key(20, 1)).unwrap());
    assert_eq!(decode_ab(&cur.key().unwrap()), (20, 1));
    // A gap rounds up to the next present key: (21,1) -> (22,1).
    assert!(cur.seek_ge(&key(21, 1)).unwrap());
    assert_eq!(decode_ab(&cur.key().unwrap()), (22, 1));
    // Below the minimum lands on the minimum.
    assert!(cur.seek_ge(&key(-5, 0)).unwrap());
    assert_eq!(decode_ab(&cur.key().unwrap()), (0, 1));
    // At and past the maximum.
    assert!(cur.seek_ge(&key(998, 1)).unwrap());
    assert_eq!(decode_ab(&cur.key().unwrap()), (998, 1));
    assert!(!cur.seek_ge(&key(999, 0)).unwrap());
    assert!(!cur.seek_ge(&key(1_000_000, 0)).unwrap());
}

#[test]
fn reverse_scan_is_exact_reverse_of_forward() {
    let pairs: Vec<(i64, i64)> = (1..=1500).map(|i| (i, i * 3)).collect();
    let (pager, root) = build_index(512, &shuffled(pairs.clone()));
    let forward = scan_all(&pager, root);
    // Pin the forward order against an independent sorted oracle too, so a bug that
    // breaks BOTH directions identically cannot pass on reverse==forward alone.
    let mut sorted = pairs;
    sorted.sort_unstable();
    assert_eq!(forward, sorted, "forward scan must equal the sorted keys (independent oracle)");
    let mut reverse = scan_all_reverse(&pager, root);
    reverse.reverse();
    assert_eq!(reverse, forward, "last()+prev() must yield the exact reverse of first()+next()");
}

#[test]
fn seek_le_finds_greatest_key_at_or_below_probe() {
    // Multiples of 10 as first column: 10,20,...,5000 (one rowid each).
    let pairs: Vec<(i64, i64)> = (1..=500).map(|i| (i * 10, 7)).collect();
    let (pager, root) = build_index(512, &pairs);
    let mut cur = IndexCursor::open(&pager, root).unwrap();
    let sorted = {
        let mut s = pairs.clone();
        s.sort_unstable();
        s
    };
    let greatest_le = |probe_first: i64, probe_rowid: i64| -> Option<(i64, i64)> {
        sorted.iter().rev().find(|&&(a, b)| (a, b) <= (probe_first, probe_rowid)).copied()
    };

    // Exact hit.
    assert!(cur.seek_le(&key(200, 7)).unwrap());
    assert_eq!(Some(decode_ab(&cur.key().unwrap())), greatest_le(200, 7));
    // Between entries rounds DOWN: probe (205,*) -> (200,7).
    assert!(cur.seek_le(&key(205, 7)).unwrap());
    assert_eq!(decode_ab(&cur.key().unwrap()), (200, 7));
    assert_eq!(Some((200, 7)), greatest_le(205, 7));
    // At the maximum and above.
    assert!(cur.seek_le(&key(5000, 7)).unwrap());
    assert_eq!(decode_ab(&cur.key().unwrap()), (5000, 7));
    assert!(cur.seek_le(&key(999_999, 0)).unwrap());
    assert_eq!(decode_ab(&cur.key().unwrap()), (5000, 7));
    // Below the minimum: nothing is <= the probe.
    assert!(!cur.seek_le(&key(5, 0)).unwrap());
    assert!(!cur.seek_le(&key(9, 999)).unwrap());
    // A prefix probe rounds down to the last key strictly below that prefix group.
    assert!(cur.seek_le(&prefix(205)).unwrap());
    assert_eq!(decode_ab(&cur.key().unwrap()), (200, 7));
}

#[test]
fn seek_le_and_seek_ge_land_exactly_on_interior_divider_keys() {
    // Regression for the classic-B `seek_le` bug: a key promoted into an INTERIOR page
    // as a divider lives ONLY there (never copied to a leaf). Seeking such a key must
    // land ON it, not on its leaf-level neighbour. The old leaf-only `seek_le`
    // descended into the divider's LEFT subtree (all keys strictly below it) and
    // returned the PREDECESSOR; `seek_ge` was already correct (it climbs to the
    // divider). Build a tall tree (height >= 3) so real dividers exist, then probe an
    // actual root divider.
    let pairs: Vec<(i64, i64)> = (1..=4000).map(|i| (i, i)).collect();
    let (pager, root) = build_index(512, &shuffled(pairs.clone()));
    let usable = 512usize;
    let (_in_order, height) = walk(&pager, root, root, usable);
    assert!(height >= 3, "expected height >= 3 so interior pages carry dividers, got {height}");

    // A ROOT divider is guaranteed to be a real interior entry (present exactly once,
    // only in the interior). Take its first divider cell's key bytes as the probe `D`.
    // Small integer keys never overflow, so `local_payload` is the whole key record.
    let divider = {
        let view = PageView::new(pager.read_page(root).unwrap(), root, usable).unwrap();
        assert!(view.page_type().is_interior(), "a height>=3 tree's root must be interior");
        let cell = view.index_interior_cell(0).unwrap();
        assert!(cell.overflow_page.is_none(), "small integer dividers never overflow");
        cell.local_payload.to_vec()
    };
    let (a, b) = decode_ab(&divider);

    // Name D's immediate predecessor and successor from an independent sorted oracle.
    let sorted = {
        let mut s = pairs.clone();
        s.sort_unstable();
        s
    };
    let pos = sorted.iter().position(|&p| p == (a, b)).expect("the divider must be a real key");
    assert!(pos > 0 && pos + 1 < sorted.len(), "a root divider is neither the global min nor max");
    let predecessor = sorted[pos - 1];
    let successor = sorted[pos + 1];

    let mut cur = IndexCursor::open(&pager, root).unwrap();

    // seek_le(D) must land EXACTLY on D, not on its predecessor (the pre-fix result).
    assert!(cur.seek_le(&divider).unwrap(), "seek_le of an existing key returns true");
    assert_eq!(
        decode_ab(&cur.key().unwrap()),
        (a, b),
        "seek_le must land ON the interior divider, not its predecessor"
    );
    // Stepping back one from that position must yield D's immediate predecessor, which
    // confirms the cursor was truly ON D (not already on the predecessor).
    assert!(cur.prev().unwrap(), "an interior divider has a predecessor");
    assert_eq!(decode_ab(&cur.key().unwrap()), predecessor, "prev() from D is D's immediate predecessor");

    // seek_ge(D) also lands exactly on D (this direction was already correct).
    assert!(cur.seek_ge(&divider).unwrap(), "seek_ge of an existing key returns true");
    assert_eq!(decode_ab(&cur.key().unwrap()), (a, b), "seek_ge must land ON the interior divider");
    // Stepping forward one from that position must yield D's immediate successor.
    assert!(cur.next().unwrap(), "an interior divider has a successor");
    assert_eq!(decode_ab(&cur.key().unwrap()), successor, "next() from D is D's immediate successor");
}

#[test]
fn seek_le_and_seek_ge_hit_every_present_key_including_interior_dividers() {
    // Exhaustive companion to the root-divider regression above: a point seek to EVERY
    // present key in a tall tree (height >= 3) must land exactly on that key, whether it
    // lives in a leaf or was promoted into an interior page as a divider. This catches a
    // seek_le/seek_ge that mishandles interior-resident keys at ANY level, not just the
    // root's first divider. (The pre-fix seek_le returned an interior divider's
    // predecessor; seek_ge was already correct — both are pinned here for symmetry.)
    let pairs: Vec<(i64, i64)> = (1..=4000).map(|i| (i, i)).collect();
    let (pager, root) = build_index(512, &shuffled(pairs.clone()));
    let usable = 512usize;
    let (_in_order, height) = walk(&pager, root, root, usable);
    assert!(height >= 3, "expected height >= 3 so interior-resident keys exist, got {height}");

    let mut cur = IndexCursor::open(&pager, root).unwrap();
    for &(a, b) in &pairs {
        assert!(cur.seek_le(&key(a, b)).unwrap(), "seek_le must find present key ({a},{b})");
        assert_eq!(
            decode_ab(&cur.key().unwrap()),
            (a, b),
            "seek_le on present key ({a},{b}) must return it, not its predecessor"
        );
        assert!(cur.seek_ge(&key(a, b)).unwrap(), "seek_ge must find present key ({a},{b})");
        assert_eq!(
            decode_ab(&cur.key().unwrap()),
            (a, b),
            "seek_ge on present key ({a},{b}) must return it"
        );
    }
}

#[test]
fn empty_index_first_last_and_seeks_return_false() {
    let mut pager = MemPager::new(4096);
    init_database(&mut pager).unwrap();
    let root = create_index_btree(&mut pager).unwrap();
    let mut cur = IndexCursor::open(&pager, root).unwrap();
    assert!(!cur.first().unwrap(), "first on an empty index is false");
    assert!(!cur.last().unwrap(), "last on an empty index is false");
    assert!(!cur.seek_ge(&key(1, 1)).unwrap(), "seek_ge on an empty index is false");
    assert!(!cur.seek_le(&key(1, 1)).unwrap(), "seek_le on an empty index is false");
    // A step after a failed positioning stays false.
    assert!(!cur.next().unwrap());
    assert!(!cur.prev().unwrap());
}

#[test]
fn single_key_roundtrip() {
    let (pager, root) = build_index(4096, &[(42, 99)]);
    let mut cur = IndexCursor::open(&pager, root).unwrap();
    assert!(cur.first().unwrap());
    assert_eq!(decode_ab(&cur.key().unwrap()), (42, 99));
    assert!(!cur.next().unwrap(), "only one key");
    assert!(cur.last().unwrap());
    assert_eq!(decode_ab(&cur.key().unwrap()), (42, 99));
    assert!(!cur.prev().unwrap(), "only one key");
}

#[test]
fn large_overflow_index_key_roundtrips_byte_exact() {
    // On a 512-byte page the index spill threshold X = ((512-12)*64/255)-23 = 102, so
    // a key larger than one whole page must spill onto a multi-page overflow chain. It
    // must insert (not fail closed), its leaf cell must carry an overflow pointer, and
    // key() must reassemble the full record byte-exact.
    let mut pager = MemPager::new(512);
    init_database(&mut pager).unwrap();
    let root = create_index_btree(&mut pager).unwrap();
    let big = encode_record(&[Value::Text("x".repeat(800)), Value::Integer(1)]);
    assert!(big.len() > 512, "the key must exceed one page to force a multi-page chain");
    index_insert(&mut pager, root, &big).unwrap();

    // The single leaf (the root) holds one cell, and it references an overflow chain.
    {
        let view = PageView::new(pager.read_page(root).unwrap(), root, 512).unwrap();
        assert!(view.page_type().is_leaf(), "one key is a single leaf (the root)");
        assert_eq!(view.cell_count(), 1);
        assert!(
            view.index_leaf_cell(0).unwrap().overflow_page.is_some(),
            "the oversized key's leaf cell must reference an overflow chain"
        );
    }

    // key() reassembles the full record byte-exact, and a seek to it finds it.
    let mut cur = IndexCursor::open(&pager, root).unwrap();
    assert!(cur.first().unwrap());
    assert_eq!(cur.key().unwrap().as_ref(), big.as_slice(), "the full spilled key must round-trip");
    assert!(!cur.next().unwrap(), "only one key");
    assert!(cur.seek_ge(&big).unwrap());
    assert_eq!(cur.key().unwrap().as_ref(), big.as_slice());
}

#[test]
fn overflow_keys_sharing_inline_prefix_compare_on_full_bytes() {
    // Two keys whose inline prefixes are byte-identical (a long common TEXT head that
    // fills well past the ~102-byte inline threshold) but whose spilled tails differ.
    // If comparison used only the inline prefix they would tie and the second insert
    // would be dropped as a duplicate; reassembling the full key distinguishes them,
    // so BOTH must coexist. This guards the pager threading in index_leaf_search.
    let mut pager = MemPager::new(512);
    init_database(&mut pager).unwrap();
    let root = create_index_btree(&mut pager).unwrap();
    let common = "A".repeat(300);
    let k1 = encode_record(&[Value::Text(format!("{common}B")), Value::Integer(1)]);
    let k2 = encode_record(&[Value::Text(format!("{common}C")), Value::Integer(2)]);
    index_insert(&mut pager, root, &k1).unwrap();
    index_insert(&mut pager, root, &k2).unwrap();

    let mut cur = IndexCursor::open(&pager, root).unwrap();
    assert!(cur.first().unwrap());
    assert_eq!(cur.key().unwrap().as_ref(), k1.as_slice(), "smaller spilled tail (…B) sorts first");
    assert!(cur.next().unwrap(), "the two keys must be distinct, not collapsed into one");
    assert_eq!(cur.key().unwrap().as_ref(), k2.as_slice(), "larger spilled tail (…C) sorts second");
    assert!(!cur.next().unwrap(), "exactly two distinct keys");

    // Re-inserting k1 is still a no-op (its full bytes match an existing key), proving
    // the duplicate check also compares reassembled spilled keys.
    index_insert(&mut pager, root, &k1).unwrap();
    let mut count = 0;
    let mut cur = IndexCursor::open(&pager, root).unwrap();
    if cur.first().unwrap() {
        loop {
            count += 1;
            if !cur.next().unwrap() {
                break;
            }
        }
    }
    assert_eq!(count, 2, "re-inserting an identical spilled key must not duplicate it");
}

#[test]
fn many_overflow_keys_build_multilevel_tree_with_spilled_dividers() {
    // Many keys that ALL share a long common inline prefix (so every inline prefix is
    // byte-identical and an inline-only comparison would collapse them) but differ in
    // their spilled tails. Enough of them force interior pages whose dividers are
    // themselves spilled keys promoted up from leaves (classic-B: a divider is the real
    // entry moved up, still spilled), so correct descent REQUIRES reassembling divider
    // keys — proving the pager threading in index_choose_child. A full forward scan must
    // then return every key byte-exact in ascending full-key order.
    let mut pager = MemPager::new(512);
    init_database(&mut pager).unwrap();
    let root = create_index_btree(&mut pager).unwrap();

    let common = "K".repeat(220);
    let n = 300i64;
    let make = |i: i64| encode_record(&[Value::Text(format!("{common}{i:06}")), Value::Integer(i)]);
    // Insert shuffled so the tree is built by real routing, not in-order append.
    let order: Vec<i64> =
        shuffled((0..n).map(|i| (i, i)).collect()).into_iter().map(|(a, _)| a).collect();
    for &i in &order {
        index_insert(&mut pager, root, &make(i)).unwrap();
    }

    // The tree grew past one page into an interior root, and at least one root divider
    // is itself spilled (else reassembly-in-descent would not be exercised).
    assert!(pager.page_count().unwrap() > 1, "many overflow keys must grow the tree past one page");
    {
        let view = PageView::new(pager.read_page(root).unwrap(), root, 512).unwrap();
        assert!(view.page_type().is_interior(), "the root must have split into an interior page");
        let spilled = (0..view.cell_count() as usize)
            .filter(|&i| view.index_interior_cell(i).unwrap().overflow_page.is_some())
            .count();
        assert!(spilled > 0, "dividers promoted from spilled leaf keys must themselves stay spilled");
    }

    // Forward scan yields every key byte-exact. The fixed-width numeric suffix orders
    // lexicographically by i, so ascending full-key order is simply i = 0..n.
    let mut cur = IndexCursor::open(&pager, root).unwrap();
    let mut scanned: Vec<Vec<u8>> = Vec::new();
    if cur.first().unwrap() {
        loop {
            scanned.push(cur.key().unwrap().into_owned());
            if !cur.next().unwrap() {
                break;
            }
        }
    }
    let mut expected: Vec<Vec<u8>> = Vec::with_capacity(n as usize);
    for i in 0..n {
        expected.push(make(i));
    }
    assert_eq!(scanned.len(), n as usize, "every overflow key present exactly once");
    assert_eq!(scanned, expected, "overflow keys must scan in full-key sorted order");

    // A point seek on a specific full key must find it (descent reassembles dividers).
    let probe = make(n / 2);
    assert!(cur.seek_ge(&probe).unwrap());
    assert_eq!(cur.key().unwrap().as_ref(), probe.as_slice());
}
