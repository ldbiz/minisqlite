//! Pure navigation over one parsed b-tree page. These read nothing from the pager;
//! they operate on a borrowed [`PageView`] the caller already opened, so both the
//! writer (descent) and the reader (cursor) share one correct implementation of the
//! separator rule.
//!
//! The table-b-tree separator invariant (fileformat2 §1.6): for an interior cell
//! `(left_child, key)`, every rowid in `left_child`'s subtree is `<= key`, and the
//! right-most pointer holds every rowid greater than all separators. So `key` is
//! the largest rowid on the left, and a search for `R` takes the first cell whose
//! `key >= R` (else the right-most pointer).

use minisqlite_fileformat::PageView;
use minisqlite_types::{Error, Result};

/// Choose the child to descend into for `rowid` on an interior table page.
/// Returns `(child_page, pointer_index)` where `pointer_index` is the position of
/// the pointer taken (`0..=cell_count`), which the descent records so a later
/// sibling step (the cursor's `next`) can resume. Binary search: O(log fanout).
pub(crate) fn choose_child(view: &PageView, rowid: i64) -> Result<(u32, u32)> {
    let count = view.cell_count() as usize;
    // First index whose separator key is >= rowid (keys are ascending).
    let (mut lo, mut hi) = (0usize, count);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let key = view.table_interior_cell(mid)?.rowid;
        if rowid <= key {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    if lo < count {
        Ok((view.table_interior_cell(lo)?.left_child, lo as u32))
    } else {
        let rp = right_pointer(view)?;
        Ok((rp, count as u32))
    }
}

/// The child page at `pointer_index` on an interior page: cell `i`'s left child for
/// `i < cell_count`, else (`== cell_count`) the right-most pointer.
pub(crate) fn pointer_at(view: &PageView, pointer_index: u32) -> Result<u32> {
    let count = view.cell_count() as u32;
    if pointer_index < count {
        Ok(view.table_interior_cell(pointer_index as usize)?.left_child)
    } else if pointer_index == count {
        right_pointer(view)
    } else {
        Err(Error::format("pointer index past the end of an interior page"))
    }
}

/// The right-most child pointer, or a format error if the page is not interior.
fn right_pointer(view: &PageView) -> Result<u32> {
    view.right_most_pointer()
        .ok_or_else(|| Error::format("interior page is missing its right-most pointer"))
}

/// Binary search a leaf table page for `rowid`. `Ok(i)` = present at cell `i`;
/// `Err(pos)` = absent, and `pos` is the first cell index whose rowid is `> rowid`
/// (the insertion point). Leaf cells are ascending by rowid.
pub(crate) fn leaf_search(view: &PageView, rowid: i64) -> Result<core::result::Result<usize, usize>> {
    let count = view.cell_count() as usize;
    let (mut lo, mut hi) = (0usize, count);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let key = view.table_leaf_cell(mid)?.rowid;
        match rowid.cmp(&key) {
            std::cmp::Ordering::Equal => return Ok(Ok(mid)),
            std::cmp::Ordering::Less => hi = mid,
            std::cmp::Ordering::Greater => lo = mid + 1,
        }
    }
    Ok(Err(lo))
}

#[cfg(test)]
mod tests {
    use super::*;
    use minisqlite_fileformat::{encode_table_interior_cell, PageBuilder, PageType};

    /// Build a real interior table page from parallel `children` (length `m + 1`, the
    /// last is the right-most pointer) and ascending `keys` (length `m`), then return
    /// its bytes so a `PageView` parses it exactly as the on-disk codec would — no
    /// hand-rolled layout, the same builder the insert path uses.
    fn interior_page(children: &[u32], keys: &[i64]) -> Vec<u8> {
        assert_eq!(children.len(), keys.len() + 1, "children must be keys + 1");
        let mut b = PageBuilder::new(512, 512, 2, PageType::InteriorTable);
        b.set_right_most_pointer(*children.last().unwrap());
        let mut tmp = Vec::new();
        for j in 0..keys.len() {
            tmp.clear();
            encode_table_interior_cell(children[j], keys[j], &mut tmp);
            assert!(b.add_cell(&tmp), "interior cell {j} must fit the test page");
        }
        b.finish()
    }

    /// A rowid EQUAL to a separator must descend into that separator's LEFT child.
    ///
    /// This is the exact boundary a `<`/`<=` mutation flips. The separator invariant
    /// (module docs) is that `keys[j]` is the *largest* rowid in `children[j]`'s
    /// subtree, so a row whose rowid equals `keys[j]` lives in `children[j]` — descent
    /// must take pointer index `j`, not `j + 1`. Changing `rowid <= key` to
    /// `rowid < key` routes it one child too far right, and the row becomes unreachable.
    #[test]
    fn key_equal_to_separator_descends_left() {
        let children = [100u32, 200, 300, 400];
        let keys = [10i64, 20, 30];
        let bytes = interior_page(&children, &keys);
        let view = PageView::new(&bytes, 2, 512).unwrap();

        assert_eq!(choose_child(&view, 10).unwrap(), (100, 0));
        assert_eq!(choose_child(&view, 20).unwrap(), (200, 1));
        assert_eq!(choose_child(&view, 30).unwrap(), (300, 2));
    }

    /// A rowid one past a separator descends into the NEXT child (the right-most
    /// pointer for a key past the last separator).
    #[test]
    fn key_one_past_separator_descends_right() {
        let children = [100u32, 200, 300, 400];
        let keys = [10i64, 20, 30];
        let bytes = interior_page(&children, &keys);
        let view = PageView::new(&bytes, 2, 512).unwrap();

        assert_eq!(choose_child(&view, 11).unwrap(), (200, 1));
        assert_eq!(choose_child(&view, 21).unwrap(), (300, 2));
        assert_eq!(choose_child(&view, 31).unwrap(), (400, 3)); // right-most pointer
    }

    /// Below every separator lands on the left-most child; above every separator lands
    /// on the right-most pointer (`pointer_index == cell_count`). Extremes included so
    /// the binary-search bounds can't be off at the ends.
    #[test]
    fn below_min_and_above_max() {
        let children = [100u32, 200, 300, 400];
        let keys = [10i64, 20, 30];
        let bytes = interior_page(&children, &keys);
        let view = PageView::new(&bytes, 2, 512).unwrap();

        assert_eq!(choose_child(&view, 9).unwrap(), (100, 0));
        assert_eq!(choose_child(&view, i64::MIN).unwrap(), (100, 0));
        assert_eq!(choose_child(&view, 1000).unwrap(), (400, 3));
        assert_eq!(choose_child(&view, i64::MAX).unwrap(), (400, 3));
    }

    /// Exhaust every rowid in a window spanning the separators and check `choose_child`
    /// against a dead-simple reference (the first pointer whose separator is `>= rowid`,
    /// else the right pointer). A proof by exhaustion over the enumerable domain: the
    /// binary search must equal the linear rule at EVERY value, so no off-by-one at any
    /// separator — the `<` mutant diverges from this reference at each separator value.
    #[test]
    fn choose_child_matches_reference_rule_exhaustively() {
        let children = [100u32, 200, 300, 400];
        let keys = [10i64, 20, 30];
        let bytes = interior_page(&children, &keys);
        let view = PageView::new(&bytes, 2, 512).unwrap();

        for rowid in -5..=40i64 {
            // Spec (module docs): the first cell whose key `>= rowid`, else the
            // right-most pointer at index `cell_count`.
            let idx = keys.iter().position(|&k| rowid <= k).unwrap_or(keys.len());
            assert_eq!(
                choose_child(&view, rowid).unwrap(),
                (children[idx], idx as u32),
                "choose_child({rowid}) must match the separator rule"
            );
        }
    }
}
