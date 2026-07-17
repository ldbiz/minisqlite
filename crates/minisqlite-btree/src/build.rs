//! Building and *planning the split of* table b-tree pages from in-memory cell lists.
//!
//! Every page the b-tree writes is produced here through `PageBuilder`, so pages are
//! always defragmented (no freeblocks, no fragmented bytes) — which real sqlite reads
//! back, and which lets the insert path reason about free space as one contiguous
//! region.
//!
//! Two page shapes:
//! - **leaf**: a list of `(rowid, encoded_cell)` in ascending rowid order.
//! - **interior**: parallel `children[0..=m]` and `keys[0..m]`, where `keys[j]` is the
//!   separator between `children[j]` and `children[j+1]` and equals the largest rowid
//!   in `children[j]`'s subtree. `children[m]` is the right-most pointer.
//!
//! This module is a pure functional core: it never touches the pager. The two
//! `plan_*_groups` functions decide *how* an overfull cell list partitions across
//! pages (returning cell-index ranges); `insert.rs` allocates the page ids, calls
//! `build_*` per group, and writes them. Keeping planning pure makes the split
//! arithmetic — the part that must preserve the separator invariant — testable in
//! isolation, and keeps page allocation in the thin pager shell.
//!
//! **N-way, not 2-way.** A single insert can force a leaf (or interior) to split into
//! *more than two* pages: a cell that is large relative to its neighbours may not fit
//! on a page beside either its left or its right neighbours, so it needs a page of its
//! own. A 2-way "fill left, rest goes right" split is wrong — its right page can
//! overflow — so the planners pack greedily into as many pages as it takes.

use minisqlite_fileformat::{encode_table_interior_cell, PageBuilder, PageType};
use minisqlite_pager::PageId;
use minisqlite_types::{Error, Result};

/// An owned leaf cell awaiting placement: its rowid (for choosing separators) and its
/// fully encoded cell bytes.
pub(crate) type LeafCell = (i64, Vec<u8>);

/// A page id used only to *probe* capacity while planning a split. Every page a split
/// produces is a fresh child or a reused non-root page — never page 1, whose 100-byte
/// database header reduces its capacity (page 1 is always the root, and a root that
/// overflows is grown by balance-deeper, not by `split_*`). So the probe id only needs
/// to differ from 1 to get the normal (full-capacity) page layout, and the planned
/// partition is valid for any non-page-1 target id.
const PROBE_ID: PageId = 2;

/// Build a leaf table page from `cells` (in order). `Ok(None)` if they do not all fit
/// on one page — the signal to split.
pub(crate) fn build_leaf(
    page_id: PageId,
    cells: &[LeafCell],
    page_size: usize,
    usable: usize,
) -> Result<Option<Vec<u8>>> {
    let mut b = PageBuilder::new(page_size, usable, page_id, PageType::LeafTable);
    for (_rowid, cell) in cells {
        if !b.add_cell(cell) {
            return Ok(None);
        }
    }
    Ok(Some(b.finish()))
}

/// Build an interior table page from parallel `children` / `keys` lists (see module
/// docs: `children.len() == keys.len() + 1`). `Ok(None)` if it overflows one page.
pub(crate) fn build_interior(
    page_id: PageId,
    children: &[u32],
    keys: &[i64],
    page_size: usize,
    usable: usize,
) -> Result<Option<Vec<u8>>> {
    // A hard check, not a debug_assert: `children.len() == keys.len() + 1` is the
    // interior invariant, and a violation here silently reuses `children.last()` as
    // both a cell's left child and the right pointer — a malformed page with no error.
    // A release build compiles out debug_assert, and new callers (index
    // b-trees, delete/rebalance) will reuse this, so fail closed rather than corrupt.
    if children.len() != keys.len() + 1 {
        return Err(Error::format("interior page needs exactly one more child than keys"));
    }
    let right_ptr = *children.last().ok_or_else(|| Error::format("interior page needs a child"))?;
    let mut b = PageBuilder::new(page_size, usable, page_id, PageType::InteriorTable);
    b.set_right_most_pointer(right_ptr);
    let mut tmp = Vec::new();
    for j in 0..keys.len() {
        tmp.clear();
        encode_table_interior_cell(children[j], keys[j], &mut tmp);
        if !b.add_cell(&tmp) {
            return Ok(None);
        }
    }
    Ok(Some(b.finish()))
}

/// Partition an overfull leaf's ordered `cells` into the contiguous groups that each
/// fit one page. Returns `(start, end)` cell-index ranges covering `0..cells.len()`
/// with no gaps; every group holds `>= 1` cell and fits a page. Greedy fill-left: pack
/// each page to capacity, spill to the next — which keeps left pages dense for
/// ascending inserts (the new, largest row starts a fresh trailing page) and, for a
/// cell too large to share with a neighbour, gives it a page of its own.
///
/// Errors only if a single cell cannot fit an empty page — an overflow-sized row that
/// should have been rejected before insert (this layer stores payloads inline; see
/// `insert::table_insert`). It never returns an empty plan.
pub(crate) fn plan_leaf_groups(
    cells: &[LeafCell],
    page_size: usize,
    usable: usize,
) -> Result<Vec<(usize, usize)>> {
    if cells.is_empty() {
        return Err(Error::format("cannot plan a leaf split with no cells"));
    }
    let mut groups = Vec::new();
    let mut start = 0usize;
    let mut b = PageBuilder::new(page_size, usable, PROBE_ID, PageType::LeafTable);
    let mut i = 0usize;
    while i < cells.len() {
        if b.add_cell(&cells[i].1) {
            i += 1;
            continue;
        }
        // cells[i] did not fit the current page. Close the group (if it has anything)
        // and retry cells[i] on a fresh page.
        if i == start {
            return Err(Error::format(
                "a leaf cell does not fit an empty page (an overflow row was not rejected)",
            ));
        }
        groups.push((start, i));
        start = i;
        b = PageBuilder::new(page_size, usable, PROBE_ID, PageType::LeafTable);
    }
    groups.push((start, cells.len()));
    Ok(groups)
}

/// Partition an overfull interior page's `children` / `keys` into the contiguous
/// groups that each fit one page. Returns `(start, end)` cell-index ranges where group
/// `(s, e)` becomes `build_interior(children[s..=e], keys[s..e])`: its cells are
/// `s..e-1`, its right pointer is `children[e]`, and — for every group but the last —
/// `keys[e]` is the separator *promoted* to the parent (it leaves this page entirely).
/// That promoted key equals the group's own subtree max, preserving the separator
/// invariant for any choice of boundary. Consecutive groups skip the promoted cell:
/// `next.start == prev.end + 1`.
///
/// **Every group holds `K = e - s >= 2` keys whenever this is a real split (more than
/// one group).** fileformat2 §1.6 requires a non-page-1 interior b-tree page to have
/// `K >= 2` (only a page-1 interior may, rarely, have `K == 1`), and every group here
/// becomes a *non-root* interior page (the children built by `split_interior_in_place`
/// and `balance_deeper_interior`), so all must satisfy `K >= 2`.
///
/// Greedy fill-left packs each group but the last to page capacity — always many tiny
/// interior cells (a table interior cell is a 4-byte child pointer plus a rowid varint,
/// ~5-9 bytes, so an interior fits `>= 4` even at the smallest supported page size), so
/// only the *last* group can fall short: on ascending inserts the remainder lands with
/// exactly 0 or 1 keys. The tail fix-up rebalances the final two groups so both keep
/// `>= 2` keys — it pulls the split point back to give the last group exactly two cells;
/// the already-full previous group shrinks by at most two cells and still fits. Because
/// interior cells are tiny this always has room; a genuine shortfall is only reachable
/// on a pathological page size that fits `< 4` interior cells (which no supported page
/// size allows), so that case fails closed with `Error::format` rather than emit a
/// `K < 2` page.
pub(crate) fn plan_interior_groups(
    children: &[u32],
    keys: &[i64],
    page_size: usize,
    usable: usize,
) -> Result<Vec<(usize, usize)>> {
    if children.len() != keys.len() + 1 {
        return Err(Error::format("interior split needs exactly one more child than keys"));
    }
    let m = keys.len();
    if m == 0 {
        return Err(Error::format("cannot split an interior page with no cells"));
    }
    // Greedy fill-left: pack each group to capacity, spill to the next. The last group
    // is the remainder and may momentarily hold 0 keys (`(m, m)` — a promoted last cell
    // stranded the final child) or 1 key; the tail fix-up below rebalances it to `>= 2`.
    let mut groups: Vec<(usize, usize)> = Vec::new();
    let mut start = 0usize;
    let mut tmp = Vec::new();
    loop {
        // Probe how far cells `[start..)` fit on one interior page. The right-most
        // pointer lives in the fixed 12-byte page header, so its value does not affect
        // the free space cells consume — a placeholder is fine while probing.
        let mut fit_end = start;
        let mut b = PageBuilder::new(page_size, usable, PROBE_ID, PageType::InteriorTable);
        b.set_right_most_pointer(0);
        while fit_end < m {
            tmp.clear();
            encode_table_interior_cell(children[fit_end], keys[fit_end], &mut tmp);
            if b.add_cell(&tmp) {
                fit_end += 1;
            } else {
                break;
            }
        }
        if fit_end == m {
            // Everything from `start` fits one page: the final group covers
            // children[start..=m]. `start == m` (a zero-key remainder) lands here too —
            // this check precedes the empty-page error below, so it is not an error.
            groups.push((start, m));
            break;
        }
        if fit_end == start {
            // A single interior cell (<= 4 + 9 bytes) does not fit an empty page — no
            // supported page size allows this; fail closed rather than loop forever.
            return Err(Error::format("an interior cell does not fit an empty page"));
        }
        // A full group ending at `fit_end` (promoting `keys[fit_end]`); continue after it.
        groups.push((start, fit_end));
        start = fit_end + 1;
    }

    // Tail fix-up: guarantee every group has `K >= 2` on a real split. Only the last
    // group can be short (all earlier groups were filled to capacity, `>= 4` cells).
    if groups.len() >= 2 {
        let li = groups.len() - 1;
        let (ls, le) = groups[li];
        debug_assert_eq!(le, m, "the last interior group must cover through m");
        if le - ls < 2 {
            let (ps, _pe) = groups[li - 1];
            // The last two groups span keys[ps..m] (`m - ps` keys) over children[ps..=m].
            // Re-splitting that span into two groups that each keep `>= 2` keys — with one
            // key promoted between them — needs `>= 5` keys. The previous group was filled
            // to capacity (`>= 4` keys), so `m - ps >= 5` always holds here; a shortfall is
            // only reachable on an unsupported page size that fits `< 4` interior cells, so
            // fail closed rather than emit a `K < 2` page.
            if m - ps < 5 {
                return Err(Error::format(
                    "interior split cannot give every group >= 2 keys (page too small for a valid interior)",
                ));
            }
            // `split = m - 3` makes the final group `(m - 2, m)` hold exactly two cells
            // (`children[m-2..=m]` / `keys[m-2..m]`); the previous group `(ps, m - 3)`
            // keeps `m - 3 - ps >= 2` cells and, being smaller than before, still fits.
            let split = m - 3;
            groups[li - 1] = (ps, split);
            groups[li] = (split + 1, m);
        }
    }

    Ok(groups)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encode a leaf cell the way the insert path does, so plan tests use real sizes.
    fn leaf_cell(rowid: i64, payload_len: usize) -> LeafCell {
        use minisqlite_fileformat::encode_table_leaf_cell;
        let payload = vec![0xABu8; payload_len];
        let mut bytes = Vec::new();
        encode_table_leaf_cell(payload.len() as u64, rowid, &payload, None, &mut bytes);
        (rowid, bytes)
    }

    #[test]
    fn leaf_plan_single_page_when_all_fit() {
        let cells: Vec<LeafCell> = (1..=3).map(|r| leaf_cell(r, 10)).collect();
        let groups = plan_leaf_groups(&cells, 4096, 4096).unwrap();
        assert_eq!(groups, vec![(0, 3)]);
    }

    #[test]
    fn leaf_plan_three_way_for_a_big_middle_cell() {
        // 512-byte page: five 40-byte rows, then a 300-byte inline row, then more
        // 40-byte rows. A 2-way split cannot place the big row without overflowing one
        // side; the planner must use >= 3 pages, each non-empty and covering 0..n.
        let mut cells: Vec<LeafCell> = Vec::new();
        for r in 1..=5 {
            cells.push(leaf_cell(r, 40));
        }
        cells.push(leaf_cell(6, 300));
        for r in 7..=12 {
            cells.push(leaf_cell(r, 40));
        }
        let groups = plan_leaf_groups(&cells, 512, 512).unwrap();
        assert!(groups.len() >= 3, "expected an N-way (>=3) split, got {groups:?}");
        // Contiguous cover, every group non-empty.
        assert_eq!(groups.first().unwrap().0, 0);
        assert_eq!(groups.last().unwrap().1, cells.len());
        for w in groups.windows(2) {
            assert_eq!(w[0].1, w[1].0, "groups must be contiguous");
        }
        for &(s, e) in &groups {
            assert!(e > s, "every group holds at least one cell");
            // And each group actually fits a page.
            assert!(build_leaf(2, &cells[s..e], 512, 512).unwrap().is_some());
        }
    }

    #[test]
    fn interior_plan_minimal_overflow_splits_two_ways_both_k_at_least_two() {
        // The minimal real split: fill an interior to exactly one-past capacity (m = f + 1,
        // f = the page's interior fanout). This is the shape most prone to a short tail — a
        // pure greedy strands the final child on a zero-cell page, and giving the tail "one
        // fewer cell" only lifts it to a single key. The tail fix-up must instead split two
        // ways with BOTH groups holding >= 2 keys. Ascending keys 1..=m are exactly the case
        // where the short tail arises.
        let usable = 512usize;
        // Find f + 1: the smallest key count that overflows one interior page (splits into
        // more than one group), so this pins the true minimal-overflow shape, not a generic
        // large split.
        let mut m = 2usize;
        loop {
            let ch: Vec<u32> = (1..=(m as u32 + 1)).collect();
            let ks: Vec<i64> = (1..=m as i64).collect();
            if plan_interior_groups(&ch, &ks, usable, usable).unwrap().len() > 1 {
                break;
            }
            m += 1;
        }
        let children: Vec<u32> = (1..=(m as u32 + 1)).collect();
        let keys: Vec<i64> = (1..=m as i64).collect();
        let groups = plan_interior_groups(&children, &keys, usable, usable).unwrap();

        assert_eq!(groups.len(), 2, "one-past-capacity (m={m}) must split into exactly two groups");
        // Contiguity with a one-cell gap (the promoted key) between groups, full cover.
        assert_eq!(groups.first().unwrap().0, 0, "cover must start at 0");
        assert_eq!(groups.last().unwrap().1, keys.len(), "cover must end at m");
        for w in groups.windows(2) {
            assert_eq!(w[1].0, w[0].1 + 1, "one promoted key sits between groups");
        }
        for (gi, &(s, e)) in groups.iter().enumerate() {
            // The fix-up's guarantee: every group is K >= 2, never a zero- or one-cell tail.
            assert!(e - s >= 2, "group {gi} {s}..{e}: K={} must be >= 2", e - s);
            assert!(
                build_interior(2, &children[s..=e], &keys[s..e], usable, usable).unwrap().is_some(),
                "group {gi} must fit one page"
            );
        }
    }

    #[test]
    fn interior_plan_gives_every_group_at_least_two_keys() {
        // fileformat2 §1.6: a non-page-1 interior b-tree page must have K >= 2 keys. Every
        // group `plan_interior_groups` returns becomes a NON-root interior page, so each must
        // hold >= 2 keys (>= 3 children) — not merely >= 1. A readable spec anchor at a single
        // representative overfull size: it documents the invariant but is NOT the regression
        // witness — n=200's tail residue happens to land >= 2 even under the old buggy tail, so
        // the actual guard is `interior_plan_every_group_two_keys_across_all_tail_residues`
        // (which sweeps the residues that DO produce the short tail). Assert `e - s >= 2` for
        // every group of an overfull interior, and re-confirm the (s, e) contract the callers
        // in insert.rs depend on: contiguous cover of 0..m with one promoted key between
        // consecutive groups, and every group fits one page.
        let usable = 512usize;
        let n = 200usize; // far more than a 512-byte interior page holds
        let children: Vec<u32> = (1..=(n as u32 + 1)).collect();
        let keys: Vec<i64> = (1..=n as i64).collect();
        let groups = plan_interior_groups(&children, &keys, usable, usable).unwrap();

        assert!(groups.len() >= 2, "an overfull interior must split into >= 2 groups");
        assert_eq!(groups.first().unwrap().0, 0, "cover must start at 0");
        assert_eq!(groups.last().unwrap().1, keys.len(), "cover must end at m");
        for w in groups.windows(2) {
            assert_eq!(w[1].0, w[0].1 + 1, "exactly one promoted key sits between groups");
        }
        for (gi, &(s, e)) in groups.iter().enumerate() {
            assert!(
                e - s >= 2,
                "group {gi} {s}..{e} has K={} keys; a non-root interior needs K >= 2",
                e - s
            );
            assert!(
                build_interior(2, &children[s..=e], &keys[s..e], usable, usable).unwrap().is_some(),
                "group {gi} must fit one page"
            );
        }
    }

    #[test]
    fn interior_plan_every_group_two_keys_across_all_tail_residues() {
        // The K=1 (and K=0) tail is a function of `m mod (fanout + 1)`: as `m` sweeps a
        // range wider than one page's fanout, it necessarily hits the residues that leave the
        // greedy remainder with 0 or 1 keys — exactly where the tail fix-up must fire. Sweep
        // `m` across such a range and assert the K >= 2 invariant (plus the cover/contiguity
        // contract) holds for every one, so a fix that only handles a single hand-picked `m`
        // is caught.
        let usable = 512usize;
        for n in 4usize..=140 {
            let children: Vec<u32> = (1..=(n as u32 + 1)).collect();
            let keys: Vec<i64> = (1..=n as i64).collect();
            let groups = plan_interior_groups(&children, &keys, usable, usable).unwrap();
            // Only assert the >=2 invariant on real splits; a lone group means it all fit.
            if groups.len() >= 2 {
                assert_eq!(groups.first().unwrap().0, 0, "n={n}: cover must start at 0");
                assert_eq!(groups.last().unwrap().1, keys.len(), "n={n}: cover must end at m");
                for w in groups.windows(2) {
                    assert_eq!(w[1].0, w[0].1 + 1, "n={n}: one promoted key between groups");
                }
                for (gi, &(s, e)) in groups.iter().enumerate() {
                    assert!(e - s >= 2, "n={n} group {gi} {s}..{e}: K={} < 2", e - s);
                    assert!(
                        build_interior(2, &children[s..=e], &keys[s..e], usable, usable)
                            .unwrap()
                            .is_some(),
                        "n={n} group {gi} must fit one page"
                    );
                }
            }
        }
    }
}
