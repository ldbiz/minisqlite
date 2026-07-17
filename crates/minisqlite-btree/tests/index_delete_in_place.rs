//! Tests for the FAST in-place index DELETE route (`index_delete` under an open
//! transaction, which reaches `Pager::page_mut` + the fileformat in-place cell editor
//! `delete_cell`). Every OTHER index test calls `index_delete` in auto-commit, where a
//! returned mutable page borrow cannot commit, so `page_mut` fails closed and the path
//! falls back to the whole-page `PageBuilder` rebuild — so those cover only the FALLBACK
//! route. These wrap the deletes in `begin()`/`commit()` so the in-place drop runs, and
//! prove:
//!
//! 1. **Logical equivalence** — the same delete sequence run in-transaction (fast) and in
//!    auto-commit (fallback) yields identical ordered scans, identical re-parsed tree
//!    structure, and identical page counts, across leaf removals, interior-divider
//!    deletes (the predecessor swap), and underflow rebalances/collapse. The critical
//!    parity point: the underflow decision keys off TOTAL free space, which is identical
//!    for a freeblock-bearing page (fast) and a defragmented rebuild (fallback), so both
//!    routes rebalance at the same fill.
//! 2. **The fast route really ran** — an in-place middle delete leaves a FREEBLOCK behind
//!    (offset 1 of the page header is non-zero), which the defragmenting rebuild never
//!    produces.
//! 3. **Free-byte accounting** — after in-place deletes the leaf still satisfies the spec
//!    identity `total == usable - header - cell_pointer_array - Σ cell_len`.
//! 4. **Overflow chains** — a deleted SPILLED key's overflow chain is freed and reused,
//!    never leaked.

use minisqlite_btree::{create_index_btree, index_delete, index_insert, init_database, IndexCursor};
use minisqlite_fileformat::{decode_record, encode_record, leaf_cell_len, leaf_free_space, PageView};
use minisqlite_pager::{MemPager, PageId, Pager};
use minisqlite_types::Value;

/// An index key record `[Integer(first), Integer(rowid)]` (mirrors `index.rs`).
fn key(first: i64, rowid: i64) -> Vec<u8> {
    encode_record(&[Value::Integer(first), Value::Integer(rowid)])
}

/// A large index key whose tail spills onto overflow pages: a long blob column plus the
/// rowid, deterministic per `first` so a delete+reinsert of the identical key round-trips.
fn big_key(first: i64, rowid: i64) -> Vec<u8> {
    let blob = vec![(first as u8).wrapping_mul(7).wrapping_add(1); 900];
    encode_record(&[Value::Integer(first), Value::Blob(blob), Value::Integer(rowid)])
}

fn decode_ab(rec: &[u8]) -> (i64, i64) {
    let vals = decode_record(rec);
    let a = match vals.first() {
        Some(Value::Integer(i)) => *i,
        other => panic!("index key first column not an integer: {other:?}"),
    };
    let b = match vals.last() {
        Some(Value::Integer(i)) => *i,
        other => panic!("index key rowid column not an integer: {other:?}"),
    };
    (a, b)
}

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

/// Scan the whole index forward, returning `(first, rowid)` keys in order.
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

/// Recursively re-parse every page as an index b-tree (fails closed on a malformed page)
/// and return the in-order leaf keys. The spec-validity check for the in-place-edited
/// pages: a corrupted header, cell-pointer array, or freeblock would be caught here.
fn valid_keys(pager: &dyn Pager, id: PageId, usable: usize) -> Vec<Vec<u8>> {
    let (is_leaf, leaf, children) = {
        let view = PageView::new(pager.read_page(id).unwrap(), id, usable).unwrap();
        assert!(view.page_type().is_index(), "page {id} is not an index page");
        let count = view.cell_count() as usize;
        if view.page_type().is_leaf() {
            let keys: Vec<Vec<u8>> =
                (0..count).map(|i| view.index_leaf_cell(i).unwrap().local_payload.to_vec()).collect();
            (true, keys, Vec::new())
        } else {
            let mut children = Vec::with_capacity(count + 1);
            for i in 0..count {
                children.push(view.index_interior_cell(i).unwrap().left_child);
            }
            children.push(view.right_most_pointer().unwrap());
            (false, Vec::new(), children)
        }
    };
    if is_leaf {
        return leaf;
    }
    let mut all = Vec::new();
    for &child in &children {
        all.extend(valid_keys(pager, child, usable));
    }
    all
}

/// The page header's first-freeblock field (offset 1 of the b-tree header; index leaves
/// are never page 1, so the header offset is 0). Non-zero means the page carries a
/// freeblock chain — the fingerprint of an in-place delete the rebuild never leaves.
fn first_freeblock(pager: &dyn Pager, id: PageId) -> u16 {
    let page = pager.read_page(id).unwrap();
    ((page[1] as u16) << 8) | page[2] as u16
}

/// Assert the spec free-byte identity on an index leaf page: total reclaimable free space
/// (`leaf_free_space` total) equals `usable - header(8) - cell_pointer_array(2n) - Σ cell_len`.
fn assert_free_accounting(pager: &dyn Pager, id: PageId, usable: usize) {
    let page = pager.read_page(id).unwrap();
    let view = PageView::new(page, id, usable).unwrap();
    assert!(view.page_type().is_leaf() && view.page_type().is_index(), "page {id} is a leaf index page");
    let n = view.cell_count() as usize;
    let mut sum_cell_len = 0usize;
    for i in 0..n {
        sum_cell_len += leaf_cell_len(page, id, usable, view.cell_pointer(i)).unwrap();
    }
    let total = leaf_free_space(page, id, usable).unwrap().total;
    assert_eq!(
        total,
        usable - 8 - 2 * n - sum_cell_len,
        "index free-byte accounting on page {id}: total == usable - 8 - CPA(2n) - Σ cell_len",
    );
}

fn keys_for(pairs: &[(i64, i64)]) -> Vec<Vec<u8>> {
    pairs.iter().map(|&(a, b)| key(a, b)).collect()
}

/// Insert `insert_keys` (auto-commit), then delete `delete_keys` INSIDE one transaction so
/// the fast in-place route runs, then commit. Inserts are auto-commit in BOTH helpers, so
/// only the DELETE route differs.
fn build_then_delete_in_txn(ps: u32, insert_keys: &[Vec<u8>], delete_keys: &[Vec<u8>]) -> (MemPager, PageId) {
    let mut pager = MemPager::new(ps);
    init_database(&mut pager).unwrap();
    let root = create_index_btree(&mut pager).unwrap();
    for k in insert_keys {
        index_insert(&mut pager, root, k).unwrap();
    }
    pager.begin().unwrap();
    for k in delete_keys {
        assert!(index_delete(&mut pager, root, k).unwrap(), "key must exist to delete");
    }
    pager.commit().unwrap();
    (pager, root)
}

/// Same but the deletes run in auto-commit, so every delete takes the FALLBACK rebuild.
fn build_then_delete_autocommit(ps: u32, insert_keys: &[Vec<u8>], delete_keys: &[Vec<u8>]) -> (MemPager, PageId) {
    let mut pager = MemPager::new(ps);
    init_database(&mut pager).unwrap();
    let root = create_index_btree(&mut pager).unwrap();
    for k in insert_keys {
        index_insert(&mut pager, root, k).unwrap();
    }
    for k in delete_keys {
        assert!(index_delete(&mut pager, root, k).unwrap(), "key must exist to delete");
    }
    (pager, root)
}

fn assert_delete_routes_agree(ps: u32, insert_keys: &[Vec<u8>], delete_keys: &[Vec<u8>]) {
    let (fast, froot) = build_then_delete_in_txn(ps, insert_keys, delete_keys);
    let (slow, sroot) = build_then_delete_autocommit(ps, insert_keys, delete_keys);
    let usable = ps as usize;
    assert_eq!(
        scan_all(&fast, froot),
        scan_all(&slow, sroot),
        "fast vs fallback index delete scans disagree (ps {ps})",
    );
    assert_eq!(
        valid_keys(&fast, froot, usable),
        valid_keys(&slow, sroot, usable),
        "fast vs fallback index structure disagree (ps {ps})",
    );
    assert_eq!(
        fast.page_count().unwrap(),
        slow.page_count().unwrap(),
        "fast vs fallback index page counts disagree (ps {ps})",
    );
}

#[test]
fn delete_subset_index_in_place_matches_fallback() {
    // Delete every even-`first` key from a multi-level index, in shuffled order. Hits
    // leaf cells, interior dividers (the predecessor swap), and leaf underflow rebalances
    // — all of which must reach the identical logical state, tree shape, and page count as
    // the fallback rebuild. This is the load-bearing parity test for the underflow
    // decision now keying off TOTAL free space.
    for &ps in &[512u32, 4096] {
        let pairs: Vec<(i64, i64)> = (1..=800).map(|i| (i, i)).collect();
        let insert_keys = keys_for(&pairs);
        let delete_keys: Vec<Vec<u8>> = shuffled(pairs.clone())
            .into_iter()
            .filter(|&(a, _)| a % 2 == 0)
            .map(|(a, b)| key(a, b))
            .collect();
        assert_delete_routes_agree(ps, &insert_keys, &delete_keys);
    }
}

#[test]
fn delete_all_index_keys_in_place_matches_fallback() {
    // Delete every key (shuffled) from a multi-level index, exercising the full collapse:
    // interior-divider predecessor swaps, cascading merges, and root height-decrease. Both
    // routes must end at the same empty index.
    for &ps in &[512u32, 4096] {
        let pairs: Vec<(i64, i64)> = (1..=600).map(|i| (i, i)).collect();
        let insert_keys = keys_for(&pairs);
        let delete_keys: Vec<Vec<u8>> =
            shuffled(pairs.clone()).into_iter().map(|(a, b)| key(a, b)).collect();
        assert_delete_routes_agree(ps, &insert_keys, &delete_keys);
        // Both routes truly emptied the index.
        let (fast, froot) = build_then_delete_in_txn(ps, &insert_keys, &delete_keys);
        assert!(scan_all(&fast, froot).is_empty(), "index emptied by the fast route");
    }
}

#[test]
fn in_place_index_delete_leaves_a_freeblock_that_rebuild_never_does() {
    // PROOF the fast route ran: a middle in-place delete on the single-leaf root (which
    // never rebalances) frees the removed cell's slot as a FREEBLOCK; the auto-commit
    // rebuild defragments, so its first-freeblock stays zero.
    let ps = 4096u32;
    let pairs: Vec<(i64, i64)> = (1..=6).map(|i| (i, i)).collect();
    let keys = keys_for(&pairs);

    let mut fast = MemPager::new(ps);
    init_database(&mut fast).unwrap();
    let root = create_index_btree(&mut fast).unwrap();
    for k in &keys {
        index_insert(&mut fast, root, k).unwrap();
    }
    fast.begin().unwrap();
    assert!(index_delete(&mut fast, root, &key(3, 3)).unwrap()); // a middle key
    fast.commit().unwrap();
    assert_ne!(
        first_freeblock(&fast, root),
        0,
        "in-place middle index delete must leave a freeblock (the fast route ran)"
    );

    let mut slow = MemPager::new(ps);
    init_database(&mut slow).unwrap();
    let sroot = create_index_btree(&mut slow).unwrap();
    for k in &keys {
        index_insert(&mut slow, sroot, k).unwrap();
    }
    assert!(index_delete(&mut slow, sroot, &key(3, 3)).unwrap());
    assert_eq!(first_freeblock(&slow, sroot), 0, "the rebuild route defragments (no freeblock)");

    assert_eq!(scan_all(&fast, root), scan_all(&slow, sroot));
    assert_eq!(scan_all(&fast, root), vec![(1, 1), (2, 2), (4, 4), (5, 5), (6, 6)]);
}

#[test]
fn free_byte_accounting_holds_after_index_delete() {
    // The fast in-place index delete must keep the leaf's free-byte accounting consistent
    // (the spec identity), with freeblocks present after the deletes.
    let ps = 4096u32;
    let usable = ps as usize;
    let mut pager = MemPager::new(ps);
    init_database(&mut pager).unwrap();
    let root = create_index_btree(&mut pager).unwrap();
    // A modest number of small keys keeps this a single-leaf root (no rebalance).
    let keys = keys_for(&(1..=20).map(|i| (i, i)).collect::<Vec<_>>());
    for k in &keys {
        index_insert(&mut pager, root, k).unwrap();
    }
    assert_free_accounting(&pager, root, usable);

    pager.begin().unwrap();
    for (a, b) in [(4i64, 4i64), (9, 9), (14, 14), (18, 18)] {
        assert!(index_delete(&mut pager, root, &key(a, b)).unwrap());
    }
    assert_free_accounting(&pager, root, usable); // freeblocks present
    pager.commit().unwrap();
    assert_free_accounting(&pager, root, usable);

    let got = scan_all(&pager, root);
    let expect: Vec<(i64, i64)> =
        (1..=20).map(|i| (i, i)).filter(|&(a, _)| ![4, 9, 14, 18].contains(&a)).collect();
    assert_eq!(got, expect);
}

#[test]
fn delete_spilled_index_key_in_place_frees_and_reuses_chain() {
    // A deleted SPILLED index key's overflow chain must be freed by the fast route (which
    // touches only the leaf cell) and reused on reinsert. Repeated in-place
    // delete+reinsert of the same spilled key must not grow the file, and must reach the
    // same state as the fallback.
    let ps = 512u32;

    let build = |in_txn: bool| -> (MemPager, PageId, Vec<PageId>) {
        let mut pager = MemPager::new(ps);
        init_database(&mut pager).unwrap();
        let root = create_index_btree(&mut pager).unwrap();
        // Two small anchor keys + one big spilled key on a single-leaf root.
        index_insert(&mut pager, root, &key(1, 1)).unwrap();
        index_insert(&mut pager, root, &key(9, 9)).unwrap();
        index_insert(&mut pager, root, &big_key(5, 5)).unwrap();
        if in_txn {
            pager.begin().unwrap();
        }
        let mut counts = Vec::new();
        for _ in 0..6 {
            assert!(index_delete(&mut pager, root, &big_key(5, 5)).unwrap());
            index_insert(&mut pager, root, &big_key(5, 5)).unwrap();
            counts.push(pager.page_count().unwrap());
        }
        if in_txn {
            pager.commit().unwrap();
        }
        (pager, root, counts)
    };

    let (fast, froot, fast_counts) = build(true);
    let (slow, sroot, _) = build(false);

    for w in fast_counts.windows(2) {
        assert_eq!(w[0], w[1], "repeated spilled index delete+reinsert must reuse freed pages, not grow the file");
    }
    assert_eq!(scan_all(&fast, froot), vec![(1, 1), (5, 5), (9, 9)], "keys read back in order");
    assert_eq!(scan_all(&slow, sroot), scan_all(&fast, froot), "fast and fallback agree");
    assert_eq!(
        fast.page_count().unwrap(),
        slow.page_count().unwrap(),
        "fast and fallback reach the same page count after a spilled delete+reinsert",
    );
}
