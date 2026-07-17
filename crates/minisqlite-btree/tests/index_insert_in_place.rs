//! Tests for the FAST in-place index insert route (`index_insert` under an open
//! transaction, reaching `Pager::page_mut` + the fileformat in-place cell editor).
//! Like the table tests in `insert_in_place.rs`, every OTHER index test calls
//! `index_insert` in auto-commit, where `page_mut` fails closed and the path falls
//! back to the whole-page `PageBuilder` rebuild — so those cover only the FALLBACK
//! route. These wrap inserts in `begin()`/`commit()` so the in-place splice runs, and
//! prove:
//!
//! 1. **Logical equivalence** — the same key set inserted in-transaction (fast) and in
//!    auto-commit (fallback) yields identical ordered scans, across many splits.
//! 2. **The fast route really ran** — for a non-sequential insert order the in-place
//!    splice leaves cells in INSERTION order within the content area, whereas the
//!    rebuild packs them in KEY order, so the raw page bytes differ even though the
//!    logical (pointer-array) order is identical.
//! 3. **Uniqueness / overflow / durability** — a duplicate key is still an idempotent
//!    no-op inside a transaction, spilled keys round-trip, and committed in-place
//!    inserts survive a fresh cursor.

use minisqlite_btree::{create_index_btree, index_insert, init_database, IndexCursor};
use minisqlite_fileformat::{decode_record, encode_record, PageView};
use minisqlite_pager::{MemPager, PageId, Pager};
use minisqlite_types::Value;

/// An index key record `[Integer(first), Integer(rowid)]` (mirrors `index.rs`).
fn key(first: i64, rowid: i64) -> Vec<u8> {
    encode_record(&[Value::Integer(first), Value::Integer(rowid)])
}

/// A large index key whose tail spills onto overflow pages: a long blob column plus
/// the rowid, so inserting it exercises the overflow chain on the in-place route.
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

/// Recursively re-parse every page as an index b-tree (fails closed on a malformed
/// page) and return the in-order leaf keys, asserting key order and the separator
/// invariant. This is the spec-validity check for the in-place-edited pages.
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

fn build_in_txn(page_size: u32, keys: &[Vec<u8>]) -> (MemPager, PageId) {
    let mut pager = MemPager::new(page_size);
    init_database(&mut pager).unwrap();
    let root = create_index_btree(&mut pager).unwrap();
    pager.begin().unwrap();
    for k in keys {
        index_insert(&mut pager, root, k).unwrap();
    }
    pager.commit().unwrap();
    (pager, root)
}

fn build_autocommit(page_size: u32, keys: &[Vec<u8>]) -> (MemPager, PageId) {
    let mut pager = MemPager::new(page_size);
    init_database(&mut pager).unwrap();
    let root = create_index_btree(&mut pager).unwrap();
    for k in keys {
        index_insert(&mut pager, root, k).unwrap();
    }
    (pager, root)
}

fn assert_routes_agree(page_size: u32, keys: &[Vec<u8>]) {
    let (fast, froot) = build_in_txn(page_size, keys);
    let (slow, sroot) = build_autocommit(page_size, keys);
    let usable = page_size as usize;
    assert_eq!(scan_all(&fast, froot), scan_all(&slow, sroot), "fast and fallback index scans disagree");
    assert_eq!(valid_keys(&fast, froot, usable), valid_keys(&slow, sroot, usable));
}

fn keys_for(pairs: &[(i64, i64)]) -> Vec<Vec<u8>> {
    pairs.iter().map(|&(a, b)| key(a, b)).collect()
}

#[test]
fn sequential_index_bulk_in_place_matches_fallback() {
    for &ps in &[512u32, 1024, 4096] {
        let pairs: Vec<(i64, i64)> = (1..=2000).map(|i| (i, i)).collect();
        assert_routes_agree(ps, &keys_for(&pairs));
    }
}

#[test]
fn shuffled_index_bulk_in_place_matches_fallback() {
    for &ps in &[512u32, 4096] {
        let pairs = shuffled((1..=2000).map(|i| (i, i)).collect());
        assert_routes_agree(ps, &keys_for(&pairs));
    }
}

#[test]
fn duplicate_index_key_in_txn_is_noop() {
    // Index keys are unique: re-inserting the identical key in a transaction is an
    // idempotent no-op (it must not corrupt the in-place-edited page or double the key).
    let ps = 4096u32;
    let pairs: Vec<(i64, i64)> = (1..=50).map(|i| (i, i)).collect();
    let mut keys = keys_for(&pairs);
    keys.extend(keys_for(&pairs)); // insert every key a second time
    let (pager, root) = build_in_txn(ps, &keys);
    let scan = scan_all(&pager, root);
    assert_eq!(scan.len(), 50, "duplicate keys did not add entries");
    assert_eq!(scan, pairs, "keys read back in order, exactly once each");
}

#[test]
fn in_place_index_layout_differs_from_rebuild() {
    // PROOF the fast route ran: insert keys in DESCENDING order into a single leaf. The
    // in-place splice places each new (smaller) key at the lowest content offset, so the
    // content area ends up in INSERTION order; the PageBuilder rebuild packs cells in KEY
    // order. Same logical (pointer-array) order, different physical bytes.
    let ps = 4096u32;
    let pairs: Vec<(i64, i64)> = (1..=8).rev().map(|i| (i, i)).collect(); // 8,7,...,1
    let keys = keys_for(&pairs);

    let (fast, froot) = build_in_txn(ps, &keys);
    let (slow, sroot) = build_autocommit(ps, &keys);

    // Single leaf (few small keys), so the root page holds them all.
    assert_eq!(scan_all(&fast, froot), scan_all(&slow, sroot), "logical order identical");
    let fast_page = fast.read_page(froot).unwrap().to_vec();
    let slow_page = slow.read_page(sroot).unwrap().to_vec();
    assert_ne!(
        fast_page, slow_page,
        "in-place and rebuild must lay the content area out differently for a non-sequential insert"
    );
    // The fast page is still a spec-valid index leaf that scans in key order.
    assert_eq!(scan_all(&fast, froot), (1..=8).map(|i| (i, i)).collect::<Vec<_>>());
}

#[test]
fn overflow_index_key_in_place_roundtrips() {
    // Large keys that spill onto overflow pages, inserted in a transaction (in-place
    // route), must round-trip byte-exact and dedup a repeat — the overflow chain is
    // allocated before the in-place splice, exactly as the fallback does.
    let ps = 512u32;
    let mut keys: Vec<Vec<u8>> = (1..=30).map(|i| big_key(i, i)).collect();
    keys.push(big_key(15, 15)); // duplicate spilled key ⇒ no-op
    let (pager, root) = build_in_txn(ps, &keys);
    let scan = scan_all(&pager, root);
    assert_eq!(scan.len(), 30, "duplicate spilled key added nothing");
    assert_eq!(scan, (1..=30).map(|i| (i, i)).collect::<Vec<_>>());
}

#[test]
fn in_place_index_inserts_survive_commit() {
    // Committed in-place index inserts are durable: a cursor opened after commit (no
    // active transaction) reads exactly them, in order.
    let ps = 1024u32;
    let pairs = shuffled((1..=600).map(|i| (i, i)).collect());
    let (pager, root) = build_in_txn(ps, &keys_for(&pairs));
    let mut expect: Vec<(i64, i64)> = pairs.clone();
    expect.sort_unstable();
    assert_eq!(scan_all(&pager, root), expect, "committed in-place index inserts read back exactly");
}
