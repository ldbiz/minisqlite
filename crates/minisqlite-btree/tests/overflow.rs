//! End-to-end tests for overflow payloads and reverse cursor navigation.
//!
//! Overflow: rows whose payload exceeds one page must spill onto a chain of
//! overflow pages and read back byte-for-byte through `TableCursor`, at multiple
//! page sizes and after the tree splits into interior pages. Reverse navigation:
//! `last` + `prev` must walk the whole tree in descending rowid order — the exact
//! reverse of a `first` + `next` forward scan — including the empty and single-row
//! edge cases.

use minisqlite_btree::{create_table_btree, init_database, table_insert, TableCursor};
use minisqlite_fileformat::PageView;
use minisqlite_pager::{MemPager, PageId, Pager};

/// A deterministic pseudo-random payload of `len` bytes, distinct per `seed` (an
/// LCG stream) so a round-trip checks every byte and cross-row contamination is
/// caught (row A's bytes showing up for row B).
fn big_payload(len: usize, seed: u64) -> Vec<u8> {
    let mut v = Vec::with_capacity(len);
    let mut state = seed.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for _ in 0..len {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        v.push((state >> 40) as u8);
    }
    v
}

/// A deterministic shuffle of `1..=n` (a fixed LCG permutation) so tests exercise
/// out-of-order insertion — and thus real splitting — without a PRNG dependency.
fn shuffled(n: i64) -> Vec<i64> {
    let mut ids: Vec<i64> = (1..=n).collect();
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
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

/// A fresh in-memory database with one empty table b-tree; returns `(pager, root)`.
fn empty_table(page_size: u32) -> (MemPager, PageId) {
    let mut pager = MemPager::new(page_size);
    init_database(&mut pager).unwrap();
    let root = create_table_btree(&mut pager).unwrap();
    (pager, root)
}

/// Forward scan (`first` + `next`) returning `(rowid, payload)` in ascending order.
fn scan_forward(pager: &dyn Pager, root: PageId) -> Vec<(i64, Vec<u8>)> {
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

/// Read a single row's payload by rowid through `TableCursor::seek_exact`, or `None`
/// if the rowid is absent. Used to check a row reads back byte-for-byte after a
/// REPLACE without walking the whole table.
fn read_row(pager: &dyn Pager, root: PageId, rowid: i64) -> Option<Vec<u8>> {
    let mut cur = TableCursor::open(pager, root).unwrap();
    if cur.seek_exact(rowid).unwrap() {
        Some(cur.payload().unwrap().into_owned())
    } else {
        None
    }
}

/// Reverse scan (`last` + `prev`) returning `(rowid, payload)` in descending order.
fn scan_reverse(pager: &dyn Pager, root: PageId) -> Vec<(i64, Vec<u8>)> {
    let mut cur = TableCursor::open(pager, root).unwrap();
    let mut out = Vec::new();
    if !cur.last().unwrap() {
        return out;
    }
    loop {
        out.push((cur.rowid(), cur.payload().unwrap().into_owned()));
        if !cur.prev().unwrap() {
            break;
        }
    }
    out
}

/// Insert one big row and read it back both directions, byte-for-byte.
fn roundtrip_single(page_size: u32, len: usize) {
    let (mut pager, root) = empty_table(page_size);
    let data = big_payload(len, 1234);
    table_insert(&mut pager, root, 1, &data).unwrap();

    let mut cur = TableCursor::open(&pager, root).unwrap();
    assert!(cur.first().unwrap());
    assert_eq!(cur.rowid(), 1);
    assert_eq!(cur.payload().unwrap().into_owned(), data, "forward read of a {len}-byte row @ {page_size}");
    assert!(!cur.next().unwrap());

    // The same single row through the reverse entry point.
    assert!(cur.last().unwrap());
    assert_eq!(cur.rowid(), 1);
    assert_eq!(cur.payload().unwrap().into_owned(), data, "reverse read of a {len}-byte row @ {page_size}");
    assert!(!cur.prev().unwrap());
}

#[test]
fn big_row_roundtrip_multiple_page_sizes() {
    // 5000 bytes exceeds one page at both sizes; 40_000 spans many overflow pages.
    roundtrip_single(512, 5000);
    roundtrip_single(4096, 5000);
    roundtrip_single(512, 40_000);
    roundtrip_single(4096, 40_000);
    roundtrip_single(65536, 200_000);
}

#[test]
fn interleaved_big_and_small_rows_scan_exactly() {
    // Even rowids get spilling payloads, odd rowids stay inline; a full scan must
    // return every row byte-for-byte, exercising overflow cells packed among small
    // cells across several split leaves.
    let (mut pager, root) = empty_table(512);
    let mut expected: Vec<(i64, Vec<u8>)> = Vec::new();
    for id in 1..=20i64 {
        let data = if id % 2 == 0 { big_payload(2000, id as u64) } else { big_payload(30, id as u64) };
        table_insert(&mut pager, root, id, &data).unwrap();
        expected.push((id, data));
    }
    assert!(pager.page_count().unwrap() > 3, "big cells among small ones must have split the leaf");
    assert_eq!(scan_forward(&pager, root), expected, "full scan must return every row byte-for-byte");
}

#[test]
fn spilled_row_reports_overflow_on_page() {
    // A single spilled row keeps the table root a leaf; parse it directly and
    // confirm the on-page cell records the overflow head and the full payload length.
    let page_size = 512u32;
    let (mut pager, root) = empty_table(page_size);
    let data = big_payload(5000, 77);
    table_insert(&mut pager, root, 1, &data).unwrap();

    let usable = page_size as usize;
    let page = pager.read_page(root).unwrap();
    let view = PageView::new(page, root, usable).unwrap();
    assert!(view.page_type().is_leaf(), "one row leaves the root a leaf");
    assert_eq!(view.cell_count(), 1);
    let cell = view.table_leaf_cell(0).unwrap();
    assert_eq!(cell.rowid, 1);
    assert!(cell.overflow_page.is_some(), "a spilled row must carry an overflow pointer");
    assert_eq!(cell.payload_len, data.len() as u64, "payload_len is the FULL length, not the inline part");
    assert!((cell.local_payload.len() as u64) < cell.payload_len, "only a prefix stays inline");
}

#[test]
fn last_and_prev_walk_reverse_of_forward() {
    // Enough shuffled rows on small pages to force a multi-level tree (interior of
    // interiors), so `prev` climbs through more than one interior level.
    let (mut pager, root) = empty_table(512);
    for &id in &shuffled(600) {
        table_insert(&mut pager, root, id, &big_payload(40, id as u64)).unwrap();
    }

    let forward = scan_forward(&pager, root);
    let mut forward_rev = forward.clone();
    forward_rev.reverse();
    assert_eq!(scan_reverse(&pager, root), forward_rev, "reverse scan == reverse of forward scan");

    // `last` lands on the maximum rowid.
    let mut cur = TableCursor::open(&pager, root).unwrap();
    assert!(cur.last().unwrap());
    assert_eq!(cur.rowid(), 600);
}

#[test]
fn prev_from_a_seek_walks_descending() {
    // `prev` must work from any positioned state, not only from `last` — e.g. an
    // `ORDER BY rowid DESC` scan that starts at a range's upper bound via `seek_exact`
    // and steps down across leaf and interior boundaries.
    let (mut pager, root) = empty_table(512);
    for id in 1..=200i64 {
        table_insert(&mut pager, root, id, &big_payload(30, id as u64)).unwrap();
    }
    let mut cur = TableCursor::open(&pager, root).unwrap();
    assert!(cur.seek_exact(150).unwrap());
    let mut got = Vec::new();
    loop {
        got.push(cur.rowid());
        if !cur.prev().unwrap() {
            break;
        }
    }
    let expected: Vec<i64> = (1..=150).rev().collect();
    assert_eq!(got, expected, "prev from rowid 150 descends 150..=1 across pages");
}

#[test]
fn last_prev_on_empty_tree() {
    let (pager, root) = empty_table(4096);
    let mut cur = TableCursor::open(&pager, root).unwrap();
    assert!(!cur.last().unwrap(), "last on an empty tree is false");
    assert!(!cur.prev().unwrap(), "prev while unpositioned stays false");
}

#[test]
fn last_prev_on_single_row_tree() {
    let (mut pager, root) = empty_table(4096);
    table_insert(&mut pager, root, 42, &big_payload(50, 42)).unwrap();

    let mut cur = TableCursor::open(&pager, root).unwrap();
    assert!(cur.last().unwrap());
    assert_eq!(cur.rowid(), 42);
    assert!(!cur.prev().unwrap(), "one element, then the start of the tree");
    // `last` is repeatable after walking off the front.
    assert!(cur.last().unwrap());
    assert_eq!(cur.rowid(), 42);
}

#[test]
fn overflow_roundtrips_after_splits() {
    // Many large rows on a small page: forces many leaves and interior splits while
    // every row also spills onto overflow pages. Both scan directions must be exact.
    let page_size = 512u32;
    let n = 120i64;
    let (mut pager, root) = empty_table(page_size);
    for &id in &shuffled(n) {
        table_insert(&mut pager, root, id, &big_payload(1500, id as u64)).unwrap();
    }
    assert!(pager.page_count().unwrap() > 20, "120 spilling rows must have grown a multi-page tree");

    let scanned = scan_forward(&pager, root);
    assert_eq!(scanned.len(), n as usize);
    for (id, data) in &scanned {
        assert_eq!(*data, big_payload(1500, *id as u64), "row {id} must round-trip after splits");
    }

    let reverse_ids: Vec<i64> = scan_reverse(&pager, root).iter().map(|(id, _)| *id).collect();
    let mut expected_rev: Vec<i64> = (1..=n).collect();
    expected_rev.reverse();
    assert_eq!(reverse_ids, expected_rev, "reverse scan visits every rowid once, descending");
}

#[test]
fn replace_overflowed_row_reclaims_old_chain_for_reuse() {
    // The core leak fix: REPLACING an overflowed row must return its old chain to the
    // freelist so a later big insert reuses those pages instead of growing the file.
    let page_size = 512u32;
    let (mut pager, root) = empty_table(page_size);
    // page 1 = sqlite_schema, page 2 = this table's root leaf.
    const BASE: PageId = 2;

    // A big row @ rowid 1 that spans many overflow pages.
    let big1 = big_payload(40_000, 1);
    table_insert(&mut pager, root, 1, &big1).unwrap();
    assert_eq!(read_row(&pager, root, 1), Some(big1), "the big row reads back after insert");
    let c1 = pager.page_count().unwrap();
    let chain_len = c1 - BASE;
    assert!(
        chain_len >= 10,
        "a 40_000-byte row on a 512 page must span many overflow pages (got {chain_len})"
    );

    // REPLACE rowid 1 with a SMALL inline row. This allocates no overflow pages, so
    // any growth here would be pure leak; instead the old chain is freed (the count
    // does not shrink, but the pages become reusable).
    let small = big_payload(20, 111);
    table_insert(&mut pager, root, 1, &small).unwrap();
    assert_eq!(read_row(&pager, root, 1), Some(small.clone()), "rowid 1 now reads the small row");
    assert_eq!(
        pager.page_count().unwrap(),
        c1,
        "an inline REPLACE allocates nothing (its old chain went to the freelist)"
    );

    // Insert a NEW big row @ rowid 2 of the same size. Its overflow chain must reuse
    // rowid 1's freed pages rather than grow the file by a second full chain.
    let big2 = big_payload(40_000, 2);
    table_insert(&mut pager, root, 2, &big2).unwrap();
    let after = pager.page_count().unwrap();
    assert!(
        after < c1 + chain_len,
        "rowid 1's freed chain must be reused: page_count {after} should stay near {c1}, \
         not {} (a leak would add another full chain of {chain_len})",
        c1 + chain_len
    );
    assert!(after <= c1 + 2, "growth is at most a leaf split, not a whole chain (got {after}, C1={c1})");

    // Both surviving rows still read back byte-for-byte.
    assert_eq!(read_row(&pager, root, 1), Some(small), "rowid 1 unchanged by the rowid-2 insert");
    assert_eq!(read_row(&pager, root, 2), Some(big2), "rowid 2 reads back its full big payload");
}

#[test]
fn repeated_big_replace_reads_newest_and_stays_bounded() {
    // REPLACE a big row with another big row, repeatedly. Each replace must free the
    // previous chain, so the file stays bounded (growth is one transient chain — the
    // new chain is allocated before the old is freed — never N chains) and the row
    // always reads back the NEWEST payload, never a stale chain.
    let page_size = 512u32;
    let (mut pager, root) = empty_table(page_size);
    const BASE: PageId = 2;

    table_insert(&mut pager, root, 1, &big_payload(30_000, 1000)).unwrap();
    let c1 = pager.page_count().unwrap();
    let chain_len = c1 - BASE;
    assert!(chain_len >= 8, "the payload must span several overflow pages (got {chain_len})");

    for seed in 1..=8u64 {
        let data = big_payload(30_000, seed);
        table_insert(&mut pager, root, 1, &data).unwrap();
        assert_eq!(
            read_row(&pager, root, 1),
            Some(data),
            "REPLACE #{seed} reads back the new payload, not the old chain"
        );
    }

    let after = pager.page_count().unwrap();
    // A leak would make the count scale with the 8 replaces (~c1 + 8*chain_len).
    assert!(
        after <= c1 + chain_len + 2,
        "repeated REPLACE must not leak: page_count {after} should stay near {} (one transient \
         extra chain), far below the {} a per-replace leak would reach",
        c1 + chain_len,
        c1 + 8 * chain_len
    );
}

#[test]
fn replace_reclaim_survives_a_split_tree() {
    // The reclaim must also fire on the split path (`collect_leaf_cells`), not only the
    // in-place fast path. Fill a multi-leaf tree of big rows, then REPLACE half of them
    // with small rows: every old chain is freed, later big inserts reuse the pages, and
    // every surviving row still reads back correctly.
    let page_size = 512u32;
    const LEN: usize = 12_000; // ~23 overflow pages per row on a 512-byte page
    let (mut pager, root) = empty_table(page_size);

    // Measure one chain's page cost from the first row (its inline prefix sits on the
    // root leaf, so the whole delta is the overflow chain).
    table_insert(&mut pager, root, 1, &big_payload(LEN, 1)).unwrap();
    let chain_len = pager.page_count().unwrap() - 2;
    assert!(chain_len >= 8, "each row must span several overflow pages (got {chain_len})");

    // 24 big rows total force interior splits while every row also spills.
    for id in 2..=24i64 {
        table_insert(&mut pager, root, id, &big_payload(LEN, id as u64)).unwrap();
    }
    let peak = pager.page_count().unwrap();

    // REPLACE the 12 even rowids with small inline rows (frees 12 overflow chains via
    // the split-path capture; the inline rows allocate nothing themselves).
    for id in (2..=24i64).step_by(2) {
        table_insert(&mut pager, root, id, &big_payload(15, 9000 + id as u64)).unwrap();
    }
    assert_eq!(pager.page_count().unwrap(), peak, "inline replaces allocate nothing");

    // Insert 12 new big rows: their chains must reuse the 12 freed ones, so total
    // growth is only b-tree structure (a few leaf/interior splits) — far below the
    // 12 full chains a leak would add.
    for id in 25..=36i64 {
        table_insert(&mut pager, root, id, &big_payload(LEN, id as u64)).unwrap();
    }
    let after = pager.page_count().unwrap();
    let leaked = peak + 12 * chain_len; // page_count if the 12 old chains had leaked
    assert!(
        after < peak + 2 * chain_len,
        "new big rows must reuse the freed chains: growth {} should be a few structural pages, \
         far below the ~{} a per-replace leak would reach (after={after}, peak={peak}, chain={chain_len})",
        after - peak,
        leaked
    );

    // Every row reads back its current payload: even rowids <= 24 the small
    // replacement, everything else (odd <= 24, and the new 25..=36) a big payload.
    for id in 1..=36i64 {
        let expected = if id <= 24 && id % 2 == 0 {
            big_payload(15, 9000 + id as u64)
        } else {
            big_payload(LEN, id as u64)
        };
        assert_eq!(read_row(&pager, root, id), Some(expected), "row {id} reads back its current value");
    }
}

#[test]
fn replace_that_overflows_the_leaf_frees_the_old_chain_via_split_path() {
    // Covers the SPLIT-path REPLACE reclaim: `collect_leaf_cells` capturing the dropped
    // same-rowid cell's overflow head (insert.rs). The other reclaim tests never reach
    // it — they replace a spilled row with a SMALLER one, so the leaf shrinks and the
    // in-place fast path (`try_build_leaf` -> Fits) captures the head. Here the
    // replacement is a big fully-inline payload on an already-populated leaf, so the
    // rebuilt leaf does not fit: `try_build_leaf` bails to `Overflow` and the split path
    // must free the old chain. Proof of the free is reuse — a later big insert lands on
    // the freed pages instead of growing the file by a whole new chain.
    let page_size = 512u32;
    let usable = page_size as usize;
    let (mut pager, root) = empty_table(page_size);

    // Three tiny inline rows so the target does not sit alone on the leaf (a lone cell
    // is rebuilt as its own page and never overflows).
    for id in 1..=3i64 {
        table_insert(&mut pager, root, id, &big_payload(40, id as u64)).unwrap();
    }

    // Target rowid 4: a big spilled row. Its chain cost is the page-count delta (the
    // leaf does not split — a few tiny cells plus one spilled cell fit one page).
    let before_target = pager.page_count().unwrap();
    table_insert(&mut pager, root, 4, &big_payload(20_000, 4)).unwrap();
    let chain_len = pager.page_count().unwrap() - before_target;
    assert!(chain_len >= 8, "the target must span several overflow pages (got {chain_len})");
    {
        let page = pager.read_page(root).unwrap();
        let view = PageView::new(page, root, usable).unwrap();
        assert!(view.page_type().is_leaf(), "setup: tiny rows + one spilled row stay on a single leaf");
    }
    let pre = pager.page_count().unwrap();

    // REPLACE rowid 4 with a big fully-inline payload: 440 <= usable-35 (=477 here) so
    // it does NOT spill, but its cell plus the three inline rows far exceeds one page,
    // so the rebuilt leaf overflows and balance-deepens — the split path runs.
    let inline_big = big_payload(440, 999);
    table_insert(&mut pager, root, 4, &inline_big).unwrap();
    assert_eq!(read_row(&pager, root, 4), Some(inline_big.clone()), "rowid 4 now reads the inline payload");
    {
        let page = pager.read_page(root).unwrap();
        let view = PageView::new(page, root, usable).unwrap();
        assert!(
            view.page_type().is_interior(),
            "the REPLACE must have overflowed the leaf and split it (proving the split path ran)"
        );
    }

    // Insert a NEW big spilled row: its chain must reuse rowid 4's freed pages. If the
    // split-path reclaim did not fire, rowid 4's old chain is orphaned AND this chain
    // grows the file, pushing page_count to at least `pre + chain_len`.
    table_insert(&mut pager, root, 5, &big_payload(20_000, 5)).unwrap();
    let after = pager.page_count().unwrap();
    assert!(
        after < pre + chain_len,
        "split-path REPLACE must free the old chain for reuse: page_count {after} should stay \
         well below {} (a leak orphans the old chain and adds a whole new one of {chain_len})",
        pre + chain_len
    );

    // Every row still reads back its current value byte-for-byte.
    for id in 1..=3i64 {
        assert_eq!(read_row(&pager, root, id), Some(big_payload(40, id as u64)), "row {id} intact");
    }
    assert_eq!(read_row(&pager, root, 4), Some(inline_big), "rowid 4 = inline replacement");
    assert_eq!(read_row(&pager, root, 5), Some(big_payload(20_000, 5)), "rowid 5 = new big payload");
}

#[test]
fn replace_spilled_row_among_others_reads_new_bytes_both_directions() {
    // This change makes the overflow write path run on REPLACE too (previously any
    // overflowing payload failed closed). Insert several spilled rows, REPLACE one of
    // them with a DIFFERENT spilled payload of a different length, and confirm both scan
    // directions return the NEW bytes — pinning the intersection of overflow-write with
    // REPLACE's drop-old-same-rowid splice (and any resulting re-split).
    let (mut pager, root) = empty_table(512);
    for id in 1..=10i64 {
        table_insert(&mut pager, root, id, &big_payload(1500, id as u64)).unwrap();
    }
    let replacement = big_payload(3000, 999);
    table_insert(&mut pager, root, 5, &replacement).unwrap(); // REPLACE rowid 5

    let want: Vec<(i64, Vec<u8>)> = (1..=10i64)
        .map(|id| (id, if id == 5 { replacement.clone() } else { big_payload(1500, id as u64) }))
        .collect();
    assert_eq!(scan_forward(&pager, root), want, "forward scan returns the replaced spilled bytes, one row per rowid");
    let mut want_rev = want.clone();
    want_rev.reverse();
    assert_eq!(scan_reverse(&pager, root), want_rev, "reverse scan returns the replaced spilled bytes");
}

#[test]
fn reverse_scan_reads_spilled_payloads_byte_exact_across_splits() {
    // `overflow_roundtrips_after_splits` checks only rowids on the reverse pass; this
    // pins the spilled *bytes* on a descending scan through a multi-page tree, so a
    // `prev`-path overflow-reassembly bug can't hide behind a correct rowid sequence.
    let (mut pager, root) = empty_table(512);
    let n = 60i64;
    for &id in &shuffled(n) {
        table_insert(&mut pager, root, id, &big_payload(1200, id as u64)).unwrap();
    }
    assert!(pager.page_count().unwrap() > 20, "60 spilling rows must have grown a multi-page tree");
    let mut expected: Vec<(i64, Vec<u8>)> =
        (1..=n).map(|id| (id, big_payload(1200, id as u64))).collect();
    expected.reverse();
    assert_eq!(scan_reverse(&pager, root), expected, "descending scan returns every spilled payload byte-for-byte");
}

#[test]
fn seek_reads_a_spilled_payload_byte_exact() {
    // Positioning via seek (not first/last) must read spilled payloads correctly too;
    // `payload()` is positioning-agnostic, but pin it so a future change can't regress
    // the seek entry points.
    let (mut pager, root) = empty_table(512);
    for id in 1..=40i64 {
        table_insert(&mut pager, root, id, &big_payload(1500, id as u64)).unwrap();
    }
    let mut cur = TableCursor::open(&pager, root).unwrap();
    assert!(cur.seek_exact(23).unwrap());
    assert_eq!(cur.rowid(), 23);
    assert_eq!(cur.payload().unwrap().into_owned(), big_payload(1500, 23), "seek_exact reads the spilled row's bytes");
    assert!(cur.seek_ge(37).unwrap());
    assert_eq!(cur.rowid(), 37);
    assert_eq!(cur.payload().unwrap().into_owned(), big_payload(1500, 37), "seek_ge reads the spilled row's bytes");
}
