//! Building and splitting index b-tree pages from in-memory cell lists — the index
//! analogue of the table `build` module.
//!
//! Every index page the b-tree writes is produced here through `PageBuilder`, so
//! pages are always defragmented (no freeblocks, no fragmented bytes), which real
//! sqlite reads back and which lets the insert path reason about free space as one
//! contiguous region.
//!
//! Two page shapes:
//! - **leaf**: a list of [`IndexCell`] in ascending key order.
//! - **interior**: parallel `children[0..=m]` and `dividers[0..m]`, where
//!   `dividers[j]` is a REAL index entry (a genuine key, present exactly ONCE in the
//!   whole tree) that sits strictly between `children[j]`'s subtree (all keys <
//!   `dividers[j]`) and `children[j+1]`'s subtree (all keys > `dividers[j]`).
//!   `children[m]` is the right-most pointer. This is the CLASSIC B-tree shape real
//!   sqlite uses: an in-order traversal interleaves each interior divider between its
//!   two child subtrees, so a full scan visits interior and leaf entries alike. (This
//!   is *unlike* a B+-tree, where a divider would be a duplicated copy of a leaf key;
//!   the table b-tree is different again — its interior cells are pure rowid
//!   separators with no key payload.)
//!
//! Splitting keeps the left half on the caller-chosen `left_id` and puts the right
//! half on `right_id`, returning the divider to promote to the parent. The promoted
//! divider is MOVED up (removed from both halves), never copied, so no key is ever
//! duplicated between a leaf and an interior page.

use minisqlite_fileformat::{
    encode_index_interior_cell, encode_index_leaf_cell, PageBuilder, PageType,
};
use minisqlite_pager::PageId;
use minisqlite_types::{Error, Result};

/// An owned index cell awaiting placement on a page. Modeled as an enum so the
/// invalid combination "no overflow page, yet the declared total length disagrees
/// with the inline bytes" is unrepresentable: `encode_index_*_cell` writes
/// `varint(payload_len)` then the inline bytes verbatim, so such a mismatch would
/// serialize a cell whose declared length disagrees with its bytes AND claims no
/// overflow — a silently corrupt cell `PageView` would mis-parse into its neighbour.
/// Both variants preserve the on-disk fields faithfully, so a cell read back to be
/// rebuilt or promoted keeps its exact bytes and overflow linkage.
#[derive(Debug, Clone)]
pub(crate) enum IndexCell {
    /// The whole key record stored inline; the bytes double as the comparison key.
    Inline(Vec<u8>),
    /// A spilled key: `payload_len` total bytes, of which `inline` stay on the page,
    /// and the overflow chain begins at `first_overflow`.
    Spilled { payload_len: u64, inline: Vec<u8>, first_overflow: u32 },
}

impl IndexCell {
    /// A fresh key given the split `write_overflow_chain` returned: `inline_len`
    /// leading bytes stay on the page and `overflow_page` heads the chain for the rest
    /// (`None` when the whole key fit inline, in which case `inline_len` is its full
    /// length).
    pub(crate) fn new_key(key_record: &[u8], inline_len: usize, overflow_page: Option<u32>) -> IndexCell {
        match overflow_page {
            None => IndexCell::Inline(key_record.to_vec()),
            Some(first_overflow) => IndexCell::Spilled {
                payload_len: key_record.len() as u64,
                inline: key_record[..inline_len].to_vec(),
                first_overflow,
            },
        }
    }

    /// Reconstruct an owned cell from a parsed on-page cell (leaf or interior),
    /// preserving its exact bytes and overflow linkage so a rebuilt / re-split /
    /// promoted cell is never truncated.
    pub(crate) fn from_parsed(payload_len: u64, local_payload: &[u8], overflow_page: Option<u32>) -> IndexCell {
        match overflow_page {
            None => IndexCell::Inline(local_payload.to_vec()),
            Some(first_overflow) => {
                IndexCell::Spilled { payload_len, inline: local_payload.to_vec(), first_overflow }
            }
        }
    }

    /// The total key length P (inline bytes plus any spilled tail).
    fn payload_len(&self) -> u64 {
        match self {
            IndexCell::Inline(p) => p.len() as u64,
            IndexCell::Spilled { payload_len, .. } => *payload_len,
        }
    }

    /// The inline bytes stored on the page.
    fn local(&self) -> &[u8] {
        match self {
            IndexCell::Inline(p) => p,
            IndexCell::Spilled { inline, .. } => inline,
        }
    }

    /// The overflow chain head, if the key spilled.
    fn overflow_page(&self) -> Option<u32> {
        match self {
            IndexCell::Inline(_) => None,
            IndexCell::Spilled { first_overflow, .. } => Some(*first_overflow),
        }
    }

    fn encode_leaf(&self, out: &mut Vec<u8>) {
        encode_index_leaf_cell(self.payload_len(), self.local(), self.overflow_page(), out);
    }

    fn encode_interior(&self, left_child: u32, out: &mut Vec<u8>) {
        encode_index_interior_cell(left_child, self.payload_len(), self.local(), self.overflow_page(), out);
    }
}

/// Build a leaf index page from `cells` (in order). `Ok(None)` if they do not all
/// fit on one page — the signal to split.
pub(crate) fn build_index_leaf(
    page_id: PageId,
    cells: &[IndexCell],
    page_size: usize,
    usable: usize,
) -> Result<Option<Vec<u8>>> {
    let mut b = PageBuilder::new(page_size, usable, page_id, PageType::LeafIndex);
    let mut tmp = Vec::new();
    for c in cells {
        tmp.clear();
        c.encode_leaf(&mut tmp);
        if !b.add_cell(&tmp) {
            return Ok(None);
        }
    }
    Ok(Some(b.finish()))
}

/// Build an interior index page from parallel `children` / `dividers` lists (see
/// module docs: `children.len() == dividers.len() + 1`). `Ok(None)` on overflow.
pub(crate) fn build_index_interior(
    page_id: PageId,
    children: &[u32],
    dividers: &[IndexCell],
    page_size: usize,
    usable: usize,
) -> Result<Option<Vec<u8>>> {
    if children.len() != dividers.len() + 1 {
        return Err(Error::format(
            "interior index page needs exactly one more child than dividers",
        ));
    }
    let right_ptr = *children.last().ok_or_else(|| Error::format("interior index page needs a child"))?;
    let mut b = PageBuilder::new(page_size, usable, page_id, PageType::InteriorIndex);
    b.set_right_most_pointer(right_ptr);
    let mut tmp = Vec::new();
    for j in 0..dividers.len() {
        tmp.clear();
        dividers[j].encode_interior(children[j], &mut tmp);
        if !b.add_cell(&tmp) {
            return Ok(None);
        }
    }
    Ok(Some(b.finish()))
}

/// The outcome of splitting an index page: the rebuilt left page bytes (for
/// `left_id`), the new right page bytes (for `right_id`), and the divider to promote
/// to the parent.
pub(crate) struct IndexSplit {
    pub left: Vec<u8>,
    pub right: Vec<u8>,
    pub sep: IndexCell,
}

/// Split a leaf's ordered cell list into a left leaf, a promoted divider, and a
/// right leaf — a CLASSIC B-tree leaf split (the shape real sqlite writes). The
/// divider is a DISTINCT entry MOVED UP out of the leaves (`cells[k]`), removed from
/// both halves, so it appears exactly ONCE in the whole tree — not copied and kept
/// in the left leaf as a B+-tree would.
///
/// Partition: left = `cells[0..k]`, promoted divider = `cells[k]`, right =
/// `cells[k+1..]`. Greedy fill finds how many cells fit the left page (`g`); the
/// divider index is `k = min(g, len-2)` so the left keeps as much as fits WHILE the
/// right still keeps at least one cell (if greedy would place all-but-the-last on the
/// left, `k` is pulled back to `len-2`). Left is then a strict subset of what already
/// fit, so it fits; the right is `< |divider| + X <= 2X ≈ U/2 < cap`, so it fits too.
///
/// Invariant held: left non-empty (`k >= 1`), exactly one divider, right non-empty
/// (`k <= len-2`), and the divider key is strictly greater than every left key and
/// strictly less than every right key (index keys are unique, so ordering is strict).
///
/// Bounds: an index leaf cell's ON-PAGE size is bounded by the spill threshold
/// X = ((U-12)*64/255)-23 ≈ U/4 — a larger key SPILLS its tail onto overflow pages,
/// keeping only its `<= X` inline prefix on the page (it is not rejected). A split
/// happens only after splicing ONE cell into a page whose cells already fit — so a
/// real split always has >= 3 cells (a 1- or 2-cell set is <= ~U/2 and fits one page,
/// never splitting). Because every cell is `<= X`, a single cell always fits an empty
/// page, so `g == 0` is unreachable except on a corrupt/miscomputed cell — fail closed
/// rather than emit a bad page. The `build_index_leaf` guards fail closed too if the
/// fit invariant is ever violated.
///
/// A promoted cell that spilled carries its `first_overflow` head UP with it (the
/// `.clone()` preserves the [`IndexCell::Spilled`] variant), so its overflow chain is
/// now referenced exactly once — by the interior divider, with no copy left in a leaf.
pub(crate) fn split_index_leaf(
    cells: &[IndexCell],
    left_id: PageId,
    right_id: PageId,
    page_size: usize,
    usable: usize,
) -> Result<IndexSplit> {
    // Greedy: count how many leading cells fit one page (`g`). This only measures the
    // fit; the halves are rebuilt below once the divider index `k` is chosen.
    let mut probe = PageBuilder::new(page_size, usable, left_id, PageType::LeafIndex);
    let mut tmp = Vec::new();
    let mut g = 0usize;
    while g < cells.len() {
        tmp.clear();
        cells[g].encode_leaf(&mut tmp);
        if probe.add_cell(&tmp) {
            g += 1;
        } else {
            break;
        }
    }
    if g == 0 {
        // A single cell did not fit an empty page. Every cell keeps its inline part
        // <= X (a larger key spills), so this is unreachable except on a corrupt or
        // miscomputed cell — fail closed rather than emit a bad page.
        return Err(Error::format("index leaf split: a single cell exceeds an empty page"));
    }
    if cells.len() < 3 {
        // A classic-B leaf split needs >= 3 cells (left + divider + right, each
        // non-empty). A 1- or 2-cell set fits one page and must never reach a split;
        // if it did, the caller's overflow invariant is broken — fail closed.
        return Err(Error::format("index leaf split needs at least three cells"));
    }
    // Divider index: keep as much on the left as fits, but never so much that the
    // right leaf would be empty. `g >= 1` and `len >= 3` make `1 <= k <= len-2`.
    let k = g.min(cells.len() - 2);
    let sep = cells[k].clone();
    let left = build_index_leaf(left_id, &cells[0..k], page_size, usable)?
        .ok_or_else(|| Error::format("index leaf split left page overflowed unexpectedly"))?;
    let right = build_index_leaf(right_id, &cells[k + 1..], page_size, usable)?
        .ok_or_else(|| Error::format("index leaf split right page overflowed unexpectedly"))?;
    Ok(IndexSplit { left, right, sep })
}

/// Split an interior page's `children` / `dividers` across two well-formed non-root
/// interiors, promoting the divider at the chosen split index `s`. Left keeps
/// `children[0..=s]` with `dividers[0..s]` (K = s dividers, right pointer
/// `children[s]`); the promoted divider is `dividers[s]`; right keeps
/// `children[s+1..=m]` with `dividers[s+1..m]` (K = m-1-s dividers, right pointer
/// `children[m]`). The promoted divider `dividers[s]` is MOVED up — removed from the
/// left dividers and NOT placed in the right — so, as a classic B-tree entry, it
/// appears exactly once (now in the parent), with its whole subtree still partitioned
/// around it: every key under `children[0..=s]` is `< dividers[s]` and every key under
/// `children[s+1..=m]` is `> dividers[s]`. Its overflow chain (if it spilled) moves up
/// with it via the `.clone()`, referenced once.
///
/// BOTH halves must be well-formed non-root interiors with K >= 2 dividers (>= 3
/// children), per fileformat2 §1.6 ("In all other cases, K is 2 or more"). A split
/// half is never the root (the root grows via balance-deeper, keeping its page id), so
/// the sole K<2 exception §1.6 allows — page 1 as an interior — never applies here.
/// That constrains `s` to `[2, m-3]` (left K = s >= 2, right K = m-1-s >= 2), which
/// needs `m >= 5`. For usable page sizes up to 8 KiB that always holds: each divider's
/// inline part is <= ~U/4 (a larger key spills its tail to overflow), so >= 4 dividers
/// fit and the page overflows only at the 5th cell. At usable >= 16 KiB, though, the
/// fixed per-cell overhead shrinks against ~U/4 so only 3 near-maxLocal (spilled)
/// dividers fit, and a non-root interior can overflow at `m = 4` — then `[2, m-3]` is
/// empty and NO 2-way split yields two K>=2 halves (matching real sqlite there needs a
/// 3-way sibling rebalance, which this 2-way split does not do). That case fails closed
/// (below), never emitting a malformed page.
///
/// Within `[2, m-3]`, `s` is chosen to BALANCE on-page bytes between the halves so
/// variable-width keys split evenly and neither half is left near-empty. The balanced
/// pick is then verified to fit both pages; only extreme key-size variance could make
/// it overflow, in which case we fall back to the largest fitting `s` (packing the
/// left fullest while keeping the right K>=2), and fail closed if no `s` in `[2, m-3]`
/// keeps both halves within one page each — never emitting a malformed or overflowing
/// page. The balanced pick and its single fit-check are O(m); the fallback re-checks
/// candidates and is O(m^2), but is reached only under that extreme variance.
pub(crate) fn split_index_interior(
    children: &[u32],
    dividers: &[IndexCell],
    left_id: PageId,
    right_id: PageId,
    page_size: usize,
    usable: usize,
) -> Result<IndexSplit> {
    let m = dividers.len();
    if children.len() != m + 1 {
        return Err(Error::format(
            "interior index split needs exactly one more child than dividers",
        ));
    }
    // Two K>=2 halves need m >= 5 (left>=2 + promoted + right>=2), so [2, m-3] is
    // non-empty. That holds at every usable size up to 8 KiB; at >= 16 KiB a non-root
    // interior with near-maxLocal spilled dividers can overflow at m=4 (see the doc
    // comment), where matching real sqlite would take a 3-way sibling rebalance this
    // 2-way split does not implement. Either way, fail closed rather than emit a K<2
    // (malformed) non-root interior — a silent on-disk corruption.
    if m < 5 {
        return Err(Error::format(
            "interior index split needs >= 5 dividers to make two K>=2 halves",
        ));
    }

    // On-page cost of each divider as an interior cell: the 2-byte cell pointer plus
    // the encoded body (4-byte left child + varint payload_len + inline bytes). The
    // 12-byte interior header (including the right-most pointer) is identical on both
    // halves, so balancing the sum of cell costs balances on-page fill. A divider's
    // cost is independent of which half holds it (children[j] is always dividers[j]'s
    // left child, on either page), so one pass computes it for every candidate split.
    let mut costs = Vec::with_capacity(m);
    let mut tmp = Vec::new();
    let mut total: usize = 0;
    for j in 0..m {
        tmp.clear();
        dividers[j].encode_interior(children[j], &mut tmp);
        let cost = tmp.len() + 2;
        total += cost;
        costs.push(cost);
    }

    // Choose the promoted index s in [2, m-3] whose split most evenly balances on-page
    // bytes. left(s) = sum costs[0..s]; the promoted divider costs[s] lands on NEITHER
    // half; right(s) = total - left(s) - costs[s].
    let lo = 2usize;
    let hi = m - 3; // m >= 5 => hi >= 2 == lo, so the range is non-empty.
    let mut prefix = 0usize; // = sum costs[0..s] as s advances
    let mut best_s = lo;
    let mut best_imbalance = usize::MAX;
    for s in 0..m {
        if (lo..=hi).contains(&s) {
            let left = prefix;
            let right = total - prefix - costs[s];
            let imbalance = left.abs_diff(right);
            if imbalance < best_imbalance {
                best_imbalance = imbalance;
                best_s = s;
            }
        }
        prefix += costs[s];
    }

    // Verify the balanced split fits both pages (it does for a minimally-overflowed
    // interior). On extreme key-size variance, fall back to the largest s in [2, m-3]
    // whose left AND right (both K>=2) fit; if none does, fail closed.
    if let Some(split) =
        try_build_interior_split(children, dividers, best_s, left_id, right_id, page_size, usable)?
    {
        return Ok(split);
    }
    for s in (lo..=hi).rev() {
        if s == best_s {
            continue; // already tried above
        }
        if let Some(split) =
            try_build_interior_split(children, dividers, s, left_id, right_id, page_size, usable)?
        {
            return Ok(split);
        }
    }
    Err(Error::format(
        "interior index split: no split point keeps both halves K>=2 within one page each",
    ))
}

/// Build both halves of an interior split at promoted index `s`: left keeps
/// `dividers[0..s]`, right keeps `dividers[s+1..]`, and `dividers[s]` is promoted.
/// Returns `Ok(None)` if EITHER half overflows one page (so the caller can try a
/// different `s`); `Ok(Some(_))` only when both fit. The caller guarantees
/// `2 <= s <= m-3`, so both halves already have K >= 2 dividers by construction.
fn try_build_interior_split(
    children: &[u32],
    dividers: &[IndexCell],
    s: usize,
    left_id: PageId,
    right_id: PageId,
    page_size: usize,
    usable: usize,
) -> Result<Option<IndexSplit>> {
    let m = dividers.len();
    let left = build_index_interior(left_id, &children[0..=s], &dividers[0..s], page_size, usable)?;
    let right =
        build_index_interior(right_id, &children[s + 1..=m], &dividers[s + 1..m], page_size, usable)?;
    Ok(match (left, right) {
        (Some(left), Some(right)) => Some(IndexSplit { left, right, sep: dividers[s].clone() }),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    //! Direct, order-independent tests of `split_index_interior` at the function
    //! level: whatever minimally-overflowing interior the insert path hands it, both
    //! returned halves must be well-formed non-root interiors (K >= 2, fitting one
    //! page each) with every child and divider conserved exactly once and the
    //! separator moved up. This pins the §1.6 fix on the pure function itself,
    //! independent of any build order / key count / shuffle seed that the end-to-end
    //! `tests/index.rs` guard depends on.
    use super::*;
    use minisqlite_fileformat::PageView;

    /// A distinct `len`-byte inline divider key whose first 4 bytes encode `j` (so a
    /// lost / duplicated / reordered divider is detectable by the conservation checks).
    fn divkey(j: usize, len: usize) -> IndexCell {
        let mut b = vec![0u8; len.max(4)];
        b[..4].copy_from_slice(&(j as u32).to_be_bytes());
        IndexCell::Inline(b)
    }

    /// Parse a built interior index page back into its `(children, inline divider
    /// bytes)`, asserting it really is an interior index page.
    fn read_interior(page: &[u8], page_id: PageId, usable: usize) -> (Vec<u32>, Vec<Vec<u8>>) {
        let view = PageView::new(page, page_id, usable).unwrap();
        assert!(
            view.page_type().is_index() && view.page_type().is_interior(),
            "a split half must be an interior index page",
        );
        let count = view.cell_count() as usize;
        let mut children = Vec::with_capacity(count + 1);
        let mut dividers = Vec::with_capacity(count);
        for i in 0..count {
            let c = view.index_interior_cell(i).unwrap();
            children.push(c.left_child);
            dividers.push(c.local_payload.to_vec());
        }
        children.push(view.right_most_pointer().unwrap());
        (children, dividers)
    }

    /// Grow `(children, dividers)` of `sizes(j)`-byte inline keys until the interior
    /// overflows one page, returning the MINIMAL overflowing lists — exactly the shape
    /// the insert path (which adds one divider at a time) hands `split_index_interior`.
    fn minimal_overflow(
        page_size: usize,
        usable: usize,
        sizes: impl Fn(usize) -> usize,
    ) -> (Vec<u32>, Vec<IndexCell>) {
        let mut children: Vec<u32> = vec![1000];
        let mut dividers: Vec<IndexCell> = Vec::new();
        for j in 0..200_000usize {
            dividers.push(divkey(j, sizes(j)));
            children.push(2000 + j as u32);
            if build_index_interior(2, &children, &dividers, page_size, usable).unwrap().is_none() {
                return (children, dividers);
            }
        }
        panic!("interior never overflowed within the bound");
    }

    /// Split `(children, dividers)` and assert the §1.6 well-formedness the fix
    /// guarantees: both halves are K>=2 interiors that fit their page, every child and
    /// divider is conserved once and in order, and the separator is moved up (present
    /// in neither half).
    fn assert_split_well_formed(children: &[u32], dividers: &[IndexCell], page_size: usize, usable: usize) {
        const LEFT: PageId = 7;
        const RIGHT: PageId = 8;
        let m = dividers.len();
        let split = split_index_interior(children, dividers, LEFT, RIGHT, page_size, usable).unwrap();
        let (lchild, ldiv) = read_interior(&split.left, LEFT, usable);
        let (rchild, rdiv) = read_interior(&split.right, RIGHT, usable);

        assert!(ldiv.len() >= 2, "left half K={} must be >= 2 (§1.6)", ldiv.len());
        assert!(rdiv.len() >= 2, "right half K={} must be >= 2 (§1.6)", rdiv.len());

        // Divider conservation: left ++ [sep] ++ right == original, in order, once each.
        let orig: Vec<Vec<u8>> = dividers.iter().map(|d| d.local().to_vec()).collect();
        let mut combined = ldiv.clone();
        combined.push(split.sep.local().to_vec());
        combined.extend(rdiv.iter().cloned());
        assert_eq!(combined, orig, "dividers must be conserved (none lost/duplicated/reordered)");
        assert_eq!(combined.len(), m, "exactly m dividers across both halves plus the promoted sep");

        // Child conservation: left children ++ right children == the original children.
        let mut cchild = lchild.clone();
        cchild.extend(rchild.iter().copied());
        assert_eq!(cchild, children, "children must be conserved in order (none lost/duplicated)");

        // The separator moved UP, not copied into either half.
        let sep = split.sep.local().to_vec();
        assert!(!ldiv.contains(&sep) && !rdiv.contains(&sep), "promoted separator must not remain in a half");
    }

    #[test]
    fn split_interior_uniform_small_keys_gives_two_k_ge_2_halves() {
        for &ps in &[512usize, 4096] {
            let (children, dividers) = minimal_overflow(ps, ps, |_| 12);
            assert!(dividers.len() >= 5, "uniform small keys overflow at m>=5 for usable<=4096");
            assert_split_well_formed(&children, &dividers, ps, ps);
        }
    }

    #[test]
    fn split_interior_tight_m5_near_maxlocal_keys() {
        // Near-maxLocal inline dividers (index X = (512-12)*64/255-23 = 102 at usable
        // 512) pack only 4 per interior, so it overflows at exactly m=5 — the tightest
        // real split, where the only legal split index is s=2 (both halves K=2).
        let x = (512 - 12) * 64 / 255 - 23; // 102
        let (children, dividers) = minimal_overflow(512, 512, |_| x);
        assert_eq!(dividers.len(), 5, "near-maxLocal keys overflow at exactly m=5 on a 512B page");
        assert_split_well_formed(&children, &dividers, 512, 512);
    }

    #[test]
    fn split_interior_variable_width_keys_stay_k_ge_2() {
        // Alternate tiny and near-maxLocal inline dividers so the byte-balanced split
        // index differs from a divider-count midpoint; both halves must still be K>=2
        // and fit. Exercises the byte-balance path on variable-width keys (the case it
        // was written for), which the fixed-width end-to-end tests do not cover.
        for &ps in &[512usize, 4096] {
            let x = (ps - 12) * 64 / 255 - 23;
            let (children, dividers) = minimal_overflow(ps, ps, |j| if j % 2 == 0 { 8 } else { x });
            assert_split_well_formed(&children, &dividers, ps, ps);
        }
    }

    #[test]
    fn split_interior_below_5_dividers_fails_closed() {
        // Two K>=2 halves need m>=5 (left>=2 + promoted + right>=2); m<5 cannot split
        // into two well-formed non-root interiors. This is reachable on a large page
        // (>= 16 KiB) with near-maxLocal spilled dividers, where an interior can
        // overflow at m=4 — so the function must FAIL CLOSED, never emit a K<2 page.
        for m in 0..=4usize {
            let children: Vec<u32> = (0..=m as u32).map(|i| 100 + i).collect();
            let dividers: Vec<IndexCell> = (0..m).map(|j| divkey(j, 8)).collect();
            assert!(
                split_index_interior(&children, &dividers, 7, 8, 512, 512).is_err(),
                "m={m} (<5) must fail closed, not emit a K<2 interior",
            );
        }
    }

    #[test]
    fn split_interior_rejects_child_divider_count_mismatch() {
        // children must be dividers + 1; a mismatch is a corrupt call — fail closed.
        let dividers: Vec<IndexCell> = (0..6).map(|j| divkey(j, 8)).collect();
        let bad_children: Vec<u32> = (0..6).collect(); // should be 7
        assert!(split_index_interior(&bad_children, &dividers, 7, 8, 512, 512).is_err());
    }
}
