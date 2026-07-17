//! Redistribution PLANNING for the classic-B index rebalance (the DELETE side's
//! underflow fix). Pure functions over in-memory cell lists — no pager — so the
//! packing arithmetic that must preserve the classic-B invariants is testable in
//! isolation, and the page I/O (gather, reuse ids, free) stays in `index_delete`.
//!
//! ## The redistribution problem
//! When a delete leaves a non-root node underfull, `index_delete` GATHERS that node
//! with an adjacent sibling (and the divider between them, pulled DOWN — it is a real
//! key), yielding one in-order sequence of leaf cells, or one interior's
//! `children`/`dividers` lists. These functions REDISTRIBUTE that sequence into the
//! FEWEST pages that are each valid:
//! - `g == 1` (everything fits one page) is a **merge**: the two nodes become one.
//! - `g >= 2` is a **rebalance**: the boundary item(s) are PROMOTED back to the parent
//!   as its new divider(s), classic-B style (an item moved UP, present once).
//!
//! ## Group convention (shared by both planners)
//! A returned group `(s, e)` puts `items[s..e]` on its page (`K = e - s` cells); for
//! every group but the last, `item[e]` is PROMOTED to the parent (it lands on no
//! page). The last group is `(s, m)` with nothing promoted after it. Consecutive
//! groups therefore satisfy `next.s == prev.e + 1` (the one-item promote gap), the
//! first starts at `s = 0`, and the last ends at `e = m` — a full cover of the items
//! with exactly `g - 1` promoted between `g` groups. For a leaf, `items` are the
//! gathered cells; for an interior, `items` are the dividers and group `(s, e)` owns
//! `children[s..=e]` with `dividers[s..e]` (so `children[e]` is its right pointer and
//! `dividers[e]` the promoted one). This is the same shape `plan_interior_groups`
//! (the table b-tree's splitter) returns, so `index_delete` splices results into the
//! parent the same way `index::propagate_split` does.
//!
//! ## Why fewest pages, and why every non-last group is full
//! Greedy fill-left packs each group to capacity, so every group but the last is
//! FULL. A full index page holds many cells: at usable size `U`, one cell's on-page
//! size is bounded by the spill threshold `X ≈ U/4` (a larger key spills its tail to
//! overflow pages, keeping only its `<= X` inline prefix), so `>= 4` items fit before
//! a page overflows (for `U` up to ~8 KiB — the tests use 512). Only the LAST group
//! can fall short of the per-group minimum (`>= 1` cell for a leaf, `>= 2` dividers /
//! `>= 3` children for a non-root interior, per fileformat2 §1.6); the tail fix-up
//! re-splits the final two groups so both meet it, relying on the previous (full)
//! group having spare items to lend.
//!
//! ## Infeasibility (fold in another sibling)
//! `plan_index_interior_groups` returns `Ok(None)` when the gathered dividers are too
//! few to form even the tail as two `K >= 2` pages (`< 5` dividers span the last two
//! groups). The caller then folds in one more adjacent sibling — supplying more
//! dividers — and re-plans, matching real sqlite's up-to-3-way balance. This only
//! arises at large usable sizes (>= 16 KiB) where a page holds just 3 near-maxLocal
//! dividers; there `index_insert` (via `split_index_interior`) fails closed on the same
//! shape, so it never *builds* such an interior — the delete fold is reached only when
//! DELETING from a large-page file real sqlite wrote. (Greedy fill can also report
//! `None` when a fully balanced N-way split would just fit; that is safe — the caller
//! folds or fails closed, never emitting a malformed page.) Leaves are always feasible
//! (a single cell always fits, `>= 1` per group is trivial), so `plan_index_leaf_groups`
//! returns `Ok(Some(_))` unless the input is corrupt.

use minisqlite_fileformat::{
    encode_index_interior_cell, encode_index_leaf_cell, PageBuilder, PageType,
};
use minisqlite_pager::PageId;
use minisqlite_types::{Error, Result};

use crate::index_build::IndexCell;

/// A page id used only to PROBE capacity while planning. Every page a redistribution
/// writes is a reused non-root child id (index roots and their children are never
/// page 1, whose 100-byte database header would shrink capacity), so the probe only
/// needs to differ from 1 to get the normal full-capacity layout.
const PROBE_ID: PageId = 2;

/// The on-disk fields of an [`IndexCell`], borrowed. Mirrors the (private) accessors
/// in `index_build` so the planner can size and encode a cell without widening that
/// module's surface — the encoding IS `encode_index_*_cell`, identical to
/// `IndexCell::encode_*`, so there is no second source of truth for the byte layout.
fn cell_fields(cell: &IndexCell) -> (u64, &[u8], Option<u32>) {
    match cell {
        IndexCell::Inline(p) => (p.len() as u64, p.as_slice(), None),
        IndexCell::Spilled { payload_len, inline, first_overflow } => {
            (*payload_len, inline.as_slice(), Some(*first_overflow))
        }
    }
}

fn encode_leaf_cell(cell: &IndexCell, out: &mut Vec<u8>) {
    let (payload_len, local, overflow) = cell_fields(cell);
    encode_index_leaf_cell(payload_len, local, overflow, out);
}

fn encode_interior_cell(left_child: u32, cell: &IndexCell, out: &mut Vec<u8>) {
    let (payload_len, local, overflow) = cell_fields(cell);
    encode_index_interior_cell(left_child, payload_len, local, overflow, out);
}

/// Plan the redistribution of a gathered, in-order LEAF cell sequence into the fewest
/// leaf pages, each fitting one page and holding `>= 1` cell, with one cell PROMOTED
/// between consecutive groups (classic-B: the promoted cell becomes a parent divider).
/// See the module docs for the `(s, e)` group convention. `Ok(Some(_))` always for a
/// non-empty `cells` at a supported page size; `Err` only on a corrupt cell that
/// cannot fit an empty page.
pub(crate) fn plan_index_leaf_groups(
    cells: &[IndexCell],
    page_size: usize,
    usable: usize,
) -> Result<Option<Vec<(usize, usize)>>> {
    let m = cells.len();
    if m == 0 {
        return Err(Error::format("index leaf redistribution: no cells to place"));
    }
    let mut groups: Vec<(usize, usize)> = Vec::new();
    let mut start = 0usize;
    let mut tmp = Vec::new();
    loop {
        let mut b = PageBuilder::new(page_size, usable, PROBE_ID, PageType::LeafIndex);
        let mut fit_end = start;
        while fit_end < m {
            tmp.clear();
            encode_leaf_cell(&cells[fit_end], &mut tmp);
            if b.add_cell(&tmp) {
                fit_end += 1;
            } else {
                break;
            }
        }
        if fit_end == m {
            groups.push((start, m));
            break;
        }
        if fit_end == start {
            // A single leaf cell exceeds an empty page. Every cell keeps its inline
            // part <= X (a larger key spills), so this is unreachable except on a
            // corrupt cell — fail closed rather than loop forever.
            return Err(Error::format("index leaf redistribution: a cell exceeds an empty page"));
        }
        // cells[start..fit_end] fit; cells[fit_end] did not, so PROMOTE it (it lands
        // on no page) and continue after it.
        groups.push((start, fit_end));
        start = fit_end + 1;
    }

    // Tail fix-up: guarantee the last group holds >= 1 cell. Fill-left leaves every
    // earlier group full, so only the last can be empty (a promotion consumed the
    // final cell, leaving `(m, m)`).
    if groups.len() >= 2 {
        let li = groups.len() - 1;
        let (ls, le) = groups[li];
        debug_assert_eq!(le, m, "the last leaf group must cover through m");
        if le <= ls {
            let (ps, _pe) = groups[li - 1];
            // The last two groups span cells[ps..m] with one promote between. Two
            // non-empty leaves + a promoted cell need >= 3 cells; a full previous
            // group has >= 4, so `m - ps >= 3` holds. Give the last group exactly one.
            if m - ps < 3 {
                return Err(Error::format(
                    "index leaf redistribution: too few cells to give the last group a cell",
                ));
            }
            groups[li - 1] = (ps, m - 2); // cells[ps..m-2], promote cells[m-2]
            groups[li] = (m - 1, m); // cells[m-1..m] = one cell
        }
    }
    Ok(Some(groups))
}

/// Plan the redistribution of a gathered interior's `children`/`dividers` (with the
/// pulled-down parent divider(s) already interleaved) into the fewest interior pages,
/// each fitting one page and — on a real (multi-group) split — holding `K >= 2`
/// dividers (`>= 3` children) per fileformat2 §1.6, with one divider PROMOTED between
/// consecutive groups. See the module docs for the `(s, e)` convention (group `(s, e)`
/// owns `children[s..=e]` + `dividers[s..e]`, promoting `dividers[e]`). Returns
/// `Ok(None)` when the dividers are too few to make the tail two `K >= 2` pages — the
/// signal to fold in another sibling and re-plan.
pub(crate) fn plan_index_interior_groups(
    children: &[u32],
    dividers: &[IndexCell],
    page_size: usize,
    usable: usize,
) -> Result<Option<Vec<(usize, usize)>>> {
    let m = dividers.len();
    if children.len() != m + 1 {
        return Err(Error::format(
            "index interior redistribution needs exactly one more child than dividers",
        ));
    }
    if m == 0 {
        // A single-child interior carries no dividers to redistribute; the gather
        // never yields this (it always includes a sibling with >= 2 dividers).
        return Err(Error::format("index interior redistribution: no dividers to place"));
    }

    // Greedy fill-left, promoting the first divider that does not fit (it goes to the
    // parent, taking no page space) and starting the next group after it.
    let mut groups: Vec<(usize, usize)> = Vec::new();
    let mut start = 0usize;
    let mut tmp = Vec::new();
    loop {
        let mut b = PageBuilder::new(page_size, usable, PROBE_ID, PageType::InteriorIndex);
        b.set_right_most_pointer(0); // placeholder: the right pointer is header, not a cell
        let mut fit_end = start;
        while fit_end < m {
            tmp.clear();
            encode_interior_cell(children[fit_end], &dividers[fit_end], &mut tmp);
            if b.add_cell(&tmp) {
                fit_end += 1;
            } else {
                break;
            }
        }
        if fit_end == m {
            groups.push((start, m));
            break;
        }
        if fit_end == start {
            return Err(Error::format(
                "index interior redistribution: a divider cell exceeds an empty page",
            ));
        }
        groups.push((start, fit_end)); // dividers[start..fit_end], promote dividers[fit_end]
        start = fit_end + 1;
    }

    // Tail fix-up: guarantee the last group has K >= 2. Fill-left leaves every earlier
    // group full (K >= 4 at supported page sizes), so only the last can be short
    // (0 or 1 dividers). Re-split the last two groups so both keep K >= 2.
    if groups.len() >= 2 {
        let li = groups.len() - 1;
        let (ls, le) = groups[li];
        debug_assert_eq!(le, m, "the last interior group must cover through m");
        if le - ls < 2 {
            let (ps, _pe) = groups[li - 1];
            // Two K>=2 groups + one promoted divider need >= 5 dividers spanning the
            // last two groups. A full previous group has >= 4, so this holds at
            // supported page sizes; when it does not (very large page + near-maxLocal
            // spilled dividers), signal infeasible so the caller folds in a sibling.
            if m - ps < 5 {
                return Ok(None);
            }
            groups[li - 1] = (ps, m - 3); // dividers[ps..m-3], promote dividers[m-3]
            groups[li] = (m - 2, m); // dividers[m-2..m] = two dividers
        }
    }

    // Verify every group fits AND (on a real split) is a valid non-root interior
    // (K >= 2). Fill-left + tail fix-up guarantees this at supported page sizes, but a
    // failure here (rather than a silent malformed page) means "fold in a sibling":
    // more dividers give the packer room to form valid pages.
    let real_split = groups.len() >= 2;
    for &(s, e) in &groups {
        if real_split && e - s < 2 {
            return Ok(None);
        }
        if !group_interior_fits(children, dividers, s, e, page_size, usable)? {
            return Ok(None);
        }
    }
    Ok(Some(groups))
}

/// Whether interior group `(s, e)` — `children[s..=e]` with `dividers[s..e]` — fits
/// one page. A pure fit check via `PageBuilder` (through the shared encoder), matching
/// exactly what `build_index_interior` would accept.
fn group_interior_fits(
    children: &[u32],
    dividers: &[IndexCell],
    s: usize,
    e: usize,
    page_size: usize,
    usable: usize,
) -> Result<bool> {
    let mut b = PageBuilder::new(page_size, usable, PROBE_ID, PageType::InteriorIndex);
    b.set_right_most_pointer(children[e]);
    let mut tmp = Vec::new();
    for d in s..e {
        tmp.clear();
        encode_interior_cell(children[d], &dividers[d], &mut tmp);
        if !b.add_cell(&tmp) {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A distinct inline leaf/divider cell of `len` bytes whose first 4 bytes encode
    /// `j`, so a lost / duplicated / reordered item is detectable by conservation.
    fn cell(j: usize, len: usize) -> IndexCell {
        let mut b = vec![0u8; len.max(4)];
        b[..4].copy_from_slice(&(j as u32).to_be_bytes());
        IndexCell::Inline(b)
    }

    /// Assert the shared group-cover contract: first starts at 0, last ends at m,
    /// consecutive groups leave exactly one promoted item between them.
    fn assert_cover(groups: &[(usize, usize)], m: usize) {
        assert_eq!(groups.first().unwrap().0, 0, "cover starts at 0");
        assert_eq!(groups.last().unwrap().1, m, "cover ends at m");
        for w in groups.windows(2) {
            assert_eq!(w[1].0, w[0].1 + 1, "exactly one promoted item between groups");
        }
    }

    #[test]
    fn leaf_all_fit_is_one_group_merge() {
        let cells: Vec<IndexCell> = (0..3).map(|j| cell(j, 10)).collect();
        let groups = plan_index_leaf_groups(&cells, 4096, 4096).unwrap().unwrap();
        assert_eq!(groups, vec![(0, 3)], "everything fits -> a single (merge) group");
    }

    #[test]
    fn leaf_overflow_splits_with_promoted_cells_and_full_cover() {
        // Many near-maxLocal leaf cells on a 512B page force a multi-group split.
        let x = (512 - 12) * 64 / 255 - 23; // index inline threshold at U=512
        let cells: Vec<IndexCell> = (0..20).map(|j| cell(j, x)).collect();
        let groups = plan_index_leaf_groups(&cells, 512, 512).unwrap().unwrap();
        assert!(groups.len() >= 2, "near-maxLocal cells must split, got {groups:?}");
        assert_cover(&groups, cells.len());
        for &(s, e) in &groups {
            assert!(e > s, "every leaf group holds >= 1 cell ({s}..{e})");
            assert!(
                crate::index_build::build_index_leaf(2, &cells[s..e], 512, 512).unwrap().is_some(),
                "group {s}..{e} must fit one page"
            );
        }
    }

    #[test]
    fn leaf_tail_fixup_never_leaves_an_empty_last_group() {
        // Sweep cell counts across a range wider than one page's fanout so the greedy
        // remainder necessarily lands on the empty-tail residue for some counts; every
        // one must still yield a full cover with a non-empty last group.
        let x = (512 - 12) * 64 / 255 - 23;
        for n in 3..=40usize {
            let cells: Vec<IndexCell> = (0..n).map(|j| cell(j, x)).collect();
            let groups = plan_index_leaf_groups(&cells, 512, 512).unwrap().unwrap();
            assert_cover(&groups, n);
            for &(s, e) in &groups {
                assert!(e > s, "n={n}: leaf group {s}..{e} must be non-empty");
            }
        }
    }

    #[test]
    fn interior_all_fit_is_one_group_merge() {
        let children: Vec<u32> = (0..6).collect();
        let dividers: Vec<IndexCell> = (0..5).map(|j| cell(j, 8)).collect();
        let groups = plan_index_interior_groups(&children, &dividers, 4096, 4096).unwrap().unwrap();
        assert_eq!(groups, vec![(0, 5)], "small dividers all fit -> a single (merge) group");
    }

    #[test]
    fn interior_overflow_splits_into_k_ge_2_groups_full_cover() {
        // Near-maxLocal dividers on a 512B page force a multi-group interior split.
        let x = (512 - 12) * 64 / 255 - 23;
        let m = 20usize;
        let children: Vec<u32> = (0..=m as u32).collect();
        let dividers: Vec<IndexCell> = (0..m).map(|j| cell(j, x)).collect();
        let groups = plan_index_interior_groups(&children, &dividers, 512, 512).unwrap().unwrap();
        assert!(groups.len() >= 2, "near-maxLocal dividers must split, got {groups:?}");
        assert_cover(&groups, m);
        for &(s, e) in &groups {
            assert!(e - s >= 2, "group {s}..{e} K={} must be >= 2 (§1.6)", e - s);
            assert!(
                group_interior_fits(&children, &dividers, s, e, 512, 512).unwrap(),
                "group {s}..{e} must fit one page"
            );
        }
    }

    #[test]
    fn interior_tail_fixup_gives_last_group_k_ge_2_across_residues() {
        // Sweep m across more than one page's fanout so the greedy remainder hits the
        // K=0/K=1 tail residues where the fix-up must fire; every real split must keep
        // K >= 2 on every group (the §1.6 invariant this balancer maintains).
        let x = (512 - 12) * 64 / 255 - 23;
        for m in 5..=60usize {
            let children: Vec<u32> = (0..=m as u32).collect();
            let dividers: Vec<IndexCell> = (0..m).map(|j| cell(j, x)).collect();
            let groups = plan_index_interior_groups(&children, &dividers, 512, 512).unwrap().unwrap();
            if groups.len() >= 2 {
                assert_cover(&groups, m);
                for &(s, e) in &groups {
                    assert!(e - s >= 2, "m={m} group {s}..{e}: K={} < 2", e - s);
                }
            }
        }
    }

    #[test]
    fn interior_child_divider_mismatch_is_error() {
        let dividers: Vec<IndexCell> = (0..5).map(|j| cell(j, 8)).collect();
        let bad_children: Vec<u32> = (0..5).collect(); // should be 6
        assert!(plan_index_interior_groups(&bad_children, &dividers, 512, 512).is_err());
    }

    #[test]
    fn interior_infeasible_two_node_gather_signals_fold_at_large_page() {
        // At a large usable size the fixed per-cell overhead shrinks against the ~U/4
        // inline bound, so only 3 near-maxLocal dividers fit an interior page. A gathered
        // window of exactly 4 dividers (5 children) then GENUINELY cannot split into two
        // K>=2 pages: one page holds at most 3, and two K>=2 pages with a promoted
        // divider need >= 2+1+2 = 5 dividers. So the planner returns `Ok(None)` — the
        // signal for `rebalance_child` to fold in a third sibling (real sqlite's
        // up-to-3-way balance) — and folding up to >= 5 dividers makes it feasible.
        //
        // NB: this large-page corner is DEFENSIVE. `split_index_interior` fails closed
        // for the same reason at these sizes (a non-root interior with >3 maxLocal
        // dividers cannot be built), so `index_insert` never produces such interiors and
        // the delete fold is unreachable through the public API — the 512B tests are the
        // real coverage. This pins the planner's None-then-feasible contract directly.
        let u = 65536usize;
        let x = (u - 12) * 64 / 255 - 23; // index inline threshold at usable u
        let children = |m: usize| (0..=m as u32).collect::<Vec<_>>();
        let dividers = |m: usize| (0..m).map(|j| cell(j, x)).collect::<Vec<_>>();

        // 4 dividers / 5 children: genuinely infeasible as two K>=2 pages -> fold signal.
        assert!(
            plan_index_interior_groups(&children(4), &dividers(4), u, u).unwrap().is_none(),
            "a 4-divider gather at usable={u} must signal infeasible (fold a sibling)"
        );

        // 5 and 6 dividers (a folded 3-node gather): feasible again, every group K>=2.
        for m in [5usize, 6] {
            let groups = plan_index_interior_groups(&children(m), &dividers(m), u, u).unwrap().unwrap();
            assert!(groups.len() >= 2, "m={m} wide dividers must split, got {groups:?}");
            for &(s, e) in &groups {
                assert!(e - s >= 2, "m={m} folded group {s}..{e} must be K>=2");
            }
        }
    }
}
