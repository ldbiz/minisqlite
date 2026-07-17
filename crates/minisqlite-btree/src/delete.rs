//! `table_delete`: remove a rowid-keyed row from a table b-tree, keeping the tree
//! structurally valid, balanced, and readable by real sqlite.
//!
//! The leaf's single dropped cell has a FAST and a FALLBACK route, ONE delete path.
//! The fast route edits the leaf bytes IN PLACE (`Pager::page_mut` +
//! `fileformat::delete_cell`) — an O(cell) splice that leaves a freeblock (the page is
//! not defragmented). That needs a mutable page borrow, which requires an active
//! transaction; when there is none (auto-commit) `page_mut` fails closed and the path
//! falls back to rebuilding the whole leaf from scratch through `PageBuilder` (which
//! *defragments* — no freeblocks — exactly the shape real sqlite reads back). Both
//! routes go through the copy-on-write buffer / `put_page`, so a delete touching page 1
//! preserves its 100-byte database header, and both leave the identical cell set,
//! cell-pointer array, and total free space — they differ only in physical layout,
//! which no reader or upward fix-up keys off (all of it reads cells through the
//! cell-pointer array and decides on cell_count, never the raw contiguous-gap layout).
//! The structural machinery below — interior rebuilds, separator retargets, splits,
//! rotates, merges, root collapse — always builds whole pages through `PageBuilder`;
//! only the surviving leaf's single-cell drop is done in place.
//!
//! Shape of the algorithm:
//! 1. Descend from the root to the target leaf, recording the interior path as
//!    `(page_id, pointer_index)` so a page that empties can be unlinked from its
//!    parent and a separator that goes stale can be retargeted.
//! 2. Search the leaf. Absent -> `Ok(false)`, tree untouched.
//! 3. Drop the target cell from the leaf (in place on the fast route, else a whole-page
//!    rebuild — either way any survivor's overflow pointer is preserved). An empty
//!    *root* leaf is a valid empty table.
//! 4. Maintain two invariants upward from the leaf, in one bottom-up pass
//!    ([`propagate`]):
//!    - **Separator == left-subtree max.** Deleting the largest rowid of a
//!      non-right-most subtree makes the bordering separator stale; retarget it to
//!      the subtree's new maximum. (Exactly one separator on the path can be stale,
//!      so this is O(height), not a re-derivation of every key.)
//!    - **No interior points at an empty child.** When a leaf (or, recursively, an
//!      interior) empties, drop its pointer and bordering separator from the parent.
//!      A parent that empties in turn propagates the same removal up; the root that
//!      empties becomes the empty-table leaf. A root left with a single child
//!      collapses height into the root page id (the id never changes — the inverse
//!      of insert's balance-deeper).
//!    - **No under-min non-root interior (fileformat2 §1.6: `K >= 2`).** Dropping a
//!      child pointer can leave a non-root interior with fewer than two keys (one or
//!      two children), which real sqlite never writes. Before continuing upward we
//!      **rebalance** it with an adjacent sibling ([`rebalance`]): if a sibling can
//!      spare a child (stays `K >= 2` after lending) we *rotate* one child through the
//!      parent separator (parent fan-out unchanged, so the fix stops here); otherwise
//!      we *merge* the under-min interior and an adjacent sibling — with the bordering
//!      parent separator pulled down between them — into a single page, free the other,
//!      and drop that child from the parent, which may cascade the same handling up. A
//!      merge chain that reaches a two-child root collapses it into the root id. The
//!      root itself is exempt from `K >= 2`: a two-child (`K == 1`) root is valid and
//!      kept; only a single-child root collapses.
//!
//! Balance is preserved because we only ever remove a *whole empty subtree* (same
//! height as its siblings), retarget a key, or move a *whole child subtree* (its
//! pointer) between two same-height interiors during a rotate/merge — never a live
//! cell between leaves. A separator that a rotate or merge makes stale is retargeted
//! to the max rowid of the subtree it now borders; that max is *carried up* from the
//! deletion point (the changed right-most subtree's new max), never re-derived by a
//! fresh descent, so a delete stays O(tree height).
//!
//! Freeing reclaimed space is part of delete. Every b-tree page this operation
//! orphans — an emptied non-root leaf or interior, and each page a root collapse
//! pulls up or bypasses — is collected while the tree is fixed up and returned to the
//! freelist with `Pager::free_page` at the very end of [`table_delete`]; a deleted
//! row's overflow chain is released with
//! [`crate::overflow_io::free_overflow_chain`]. The frees run last, after every page
//! write for the delete is done and no page-view borrow is live, because `free_page`
//! takes `&mut` and may repurpose a freed page as a freelist trunk — so no later read
//! or write may touch a freed page. The root page id is never freed: an empty root
//! leaf is the valid empty table. A later `allocate_page` reuses the freed pages
//! before growing the file, so a delete-heavy workload stays bounded, as real sqlite
//! does.

use minisqlite_fileformat::{delete_cell, encode_table_leaf_cell, PageBuilder, PageType, PageView};
use minisqlite_pager::{PageId, Pager};
use minisqlite_types::{Error, Result};

use crate::build::build_interior;
use crate::insert::{balance_deeper_interior, propagate_split, split_interior_in_place};
use crate::nav::{choose_child, leaf_search};
use crate::tree::{put_page, usable_of};

/// Delete the row with `rowid` from the table b-tree rooted at `root`. Returns
/// `Ok(true)` if a row was removed, `Ok(false)` if `rowid` was absent (the tree is
/// left unchanged). Runs within whatever transaction the caller has open (it does
/// not begin/commit).
pub fn table_delete(pager: &mut dyn Pager, root: PageId, rowid: i64) -> Result<bool> {
    let page_size = pager.page_size() as usize;
    let usable = usable_of(pager);

    // (1) Descend to the target leaf, recording the interior path.
    let mut path: Vec<(PageId, u32)> = Vec::new();
    let mut current = root;
    let leaf_id = loop {
        let data = pager.read_page(current)?;
        let view = PageView::new(data, current, usable)?;
        if !view.page_type().is_table() {
            return Err(Error::format("table_delete reached a non-table b-tree page"));
        }
        if view.page_type().is_leaf() {
            break current;
        }
        let (child, pointer_index) = choose_child(&view, rowid)?;
        path.push((current, pointer_index));
        current = child;
    };

    // (2) Search the leaf. Absent -> the tree is unchanged. When present, copy out the
    // target cell's overflow-chain head (a `u32`) while the view borrow is still live,
    // so its pages can be freed after the writes below — a spilled row leaks its whole
    // chain otherwise. `leaf_search` returns an in-range index, so the cell read is safe.
    let (idx, leaf_count, target_overflow) = {
        let data = pager.read_page(leaf_id)?;
        let view = PageView::new(data, leaf_id, usable)?;
        match leaf_search(&view, rowid)? {
            Ok(i) => (i, view.cell_count() as usize, view.table_leaf_cell(i)?.overflow_page),
            Err(_) => return Ok(false),
        }
    };

    // B-tree pages this delete unlinks and must reclaim. Collected as the tree is fixed
    // up and freed together in (6), once every page write is done and no page-view
    // borrow is live: `free_page` needs `&mut` and may repurpose a freed page as a
    // freelist trunk, so nothing may read or write these pages afterward. The root id is
    // never added (an empty root leaf is the valid empty table).
    let mut orphans: Vec<PageId> = Vec::new();

    // (3) The leaf keeps at least one row, or it is the root (an empty root leaf is
    // a valid empty table): drop its target cell (in place on the fast route, else a
    // whole-page rebuild). Otherwise it empties and is not the root, so (5) unlinks it
    // and reclaims the now-dead leaf page.
    if leaf_count > 1 || path.is_empty() {
        let new_max = drop_leaf_cell(pager, leaf_id, idx, leaf_count, page_size, usable)?;
        // (4) If we removed the leaf's maximum rowid, a separator up the path may now
        // be stale; retarget it. (The root leaf has no separators above it. Retargeting
        // moves no pages, so it orphans nothing.)
        if let Some(new_max) = new_max {
            if !path.is_empty() {
                propagate(pager, &path, root, Prop::MaxChanged(new_max), page_size, usable, &mut orphans)?;
            }
        }
    } else {
        // (5) The leaf emptied and is not the root: unlink it (collapsing upward,
        // rebalancing any interior that drops below K>=2) and reclaim the dead leaf
        // page itself. The leaf is the child at `path.last().1` of its parent.
        orphans.push(leaf_id);
        let leaf_idx = path.last().expect("a non-root leaf has a parent on the path").1 as usize;
        propagate(
            pager,
            &path,
            root,
            Prop::Removed { idx: leaf_idx, rightmost_max: None },
            page_size,
            usable,
            &mut orphans,
        )?;
    }

    // (6) Reclaim the deleted row's overflow chain and every orphaned b-tree page, now
    // that all writes for this delete are complete and no view borrow is live. Order is
    // irrelevant: each page is already fully unlinked from the surviving tree.
    if let Some(first) = target_overflow {
        crate::overflow_io::free_overflow_chain(pager, first, usable)?;
    }
    // Debug-only backstop for the orphan set. `free_page` gives no safety net here: it
    // rejects only page 0/1/out-of-range and does NOT detect a double-free or a freed
    // *table* root (>= 2), so either would corrupt silently and only surface far away when
    // `allocate_page` later hands the same physical page out twice. The set is disjoint and
    // root-free by construction: the sites that push here — an emptied non-root leaf/interior,
    // a merged-away non-root interior sibling (`rebalance`), and a root collapse's pulled-up
    // pages (`collapse_root`) — each orphan a distinct, fully-unlinked non-root page. Fail
    // loud at the cause if that reasoning is ever wrong rather than leaking downstream.
    debug_assert!(
        orphans.iter().all(|&id| id != root && id != 1),
        "delete must never free the root (={root}) or page 1"
    );
    debug_assert!(
        {
            let mut seen = orphans.clone();
            seen.sort_unstable();
            seen.dedup();
            seen.len() == orphans.len()
        },
        "delete orphan set must contain no duplicate page ids (double-free guard)"
    );
    for id in orphans {
        pager.free_page(id)?;
    }
    Ok(true)
}

/// What one level of [`propagate`] must tell the level above it.
#[derive(Clone, Copy)]
enum Prop {
    /// The child subtree we came from still exists, but its maximum rowid changed to
    /// this value; retarget the separator bordering it (unless it hangs off the
    /// right-most pointer, in which case this level's own maximum changed and the fix
    /// continues one level up). Targets the descent pointer `path[level].1`.
    MaxChanged(i64),
    /// The child pointer at index `idx` of the level being processed must be unlinked
    /// (its subtree emptied, or it merged away). `rightmost_max`, when `Some(m)`, says
    /// that — independently of this removal — the node's *right-most* child now has
    /// subtree max `m` (a lower merge kept a right-most child but changed its max), so
    /// this node's own max is `m` and must be carried up. `idx` is explicit rather than
    /// `path`-derived because a merge removes a sibling of the descended child, at a
    /// different index than the descent pointer.
    Removed { idx: usize, rightmost_max: Option<i64> },
}

/// Result of [`rebalance`]: whether the fix-up is complete at this level (a rotate
/// leaves the parent fan-out unchanged) or must continue upward (a merge dropped a
/// child from the parent).
enum Next {
    /// The tree is fully repaired; stop the bottom-up pass.
    Stop,
    /// Continue the bottom-up pass at the parent level with this state.
    Propagate(Prop),
}

/// Bottom-up fix-up along the recorded interior `path`, carrying a [`Prop`] from the
/// child level to its parent. Handles separator retargeting (`MaxChanged`),
/// empty-subtree unlinking with height collapse, and rebalance-on-underflow (rotate /
/// merge) that keeps every non-root interior at `K >= 2` — all in a single pass.
/// `path` must be non-empty. Any page this unlinks (an emptied non-root interior, a
/// page a merge makes dead, or the pages a root collapse pulls up) is pushed to
/// `orphans` for the caller to free once all writes are done; a `MaxChanged` pass moves
/// no pages and pushes nothing.
fn propagate(
    pager: &mut dyn Pager,
    path: &[(PageId, u32)],
    root: PageId,
    initial: Prop,
    page_size: usize,
    usable: usize,
    orphans: &mut Vec<PageId>,
) -> Result<()> {
    let mut state = initial;
    for level in (0..path.len()).rev() {
        let pid = path[level].0;
        match state {
            Prop::MaxChanged(new_max) => {
                // The changed subtree is the one we descended into, at `path[level].1`.
                let child_ptr = path[level].1 as usize;
                let cell_count = {
                    let data = pager.read_page(pid)?;
                    PageView::new(data, pid, usable)?.cell_count() as usize
                };
                if child_ptr < cell_count {
                    // A real separator borders the changed subtree; retarget it. A
                    // non-right-most child's change does not move `pid`'s own maximum,
                    // so no separator above `pid` is affected — stop here.
                    let (children, mut keys) = read_interior_lists(pager, pid, usable)?;
                    keys[child_ptr] = new_max;
                    if let Some(bytes) = build_interior(pid, &children, &keys, page_size, usable)? {
                        return put_page(pager, pid, bytes);
                    }
                    // The retargeted separator encodes WIDER than the one it replaced
                    // (classically the deleted max was a small positive whose in-order
                    // predecessor is a NEGATIVE rowid: a 1-byte varint becomes a 9-byte
                    // one), so a near-full `pid` no longer fits one page. That is not an
                    // error — real sqlite SPLITS/rebalances and the DELETE succeeds. Grow
                    // the tree exactly as an insert split does, reusing insert's one split
                    // implementation so the delete path can never drift from its tree
                    // shape. A retarget changes no child count, so the split's halves are
                    // still `K >= 2` by `plan_interior_groups`, and `pid`'s own maximum is
                    // unchanged (this is a non-right-most separator), so nothing above
                    // `pid` needs a further fix — the split propagation completes it.
                    if level == 0 {
                        // `pid` is the root (`path[0].0 == root`); grow depth in place so
                        // the root page id stays stable.
                        return balance_deeper_interior(pager, root, &children, &keys, page_size, usable);
                    }
                    // Splitting `pid` (at `level`) promotes new siblings into `pid`'s
                    // parent, whose descent entry is `path[level - 1]`; propagate into the
                    // ancestors above `pid` via `&path[..level]` (do NOT pass the full
                    // path — that would re-descend into `pid`).
                    let (nc, ns) = split_interior_in_place(pager, pid, &children, &keys, page_size, usable)?;
                    return propagate_split(pager, &path[..level], root, ns, nc, page_size, usable);
                }
                // The changed subtree hangs off the right-most pointer (no bordering
                // separator), so `pid`'s own maximum changed; keep climbing with the
                // same new maximum.
            }
            Prop::Removed { idx, rightmost_max } => {
                let (mut children, mut keys) = read_interior_lists(pager, pid, usable)?;
                if idx >= children.len() {
                    return Err(Error::format("delete: removal pointer index out of range"));
                }
                let old_len = children.len();
                let was_rightmost = idx + 1 == old_len;
                children.remove(idx);
                // Drop the separator bordering the removed child: keys[idx] for a left
                // child, the last key for the right-most pointer, and nothing when the
                // page had a single child (no separators at all).
                let dropped_key = if idx < keys.len() {
                    Some(keys.remove(idx))
                } else if !keys.is_empty() {
                    keys.pop()
                } else {
                    None
                };
                debug_assert!(
                    children.is_empty() || children.len() == keys.len() + 1,
                    "interior child/key counts must stay paired after a removal: {} children, {} keys",
                    children.len(),
                    keys.len(),
                );

                if children.is_empty() {
                    // `pid` is now empty (only reachable defensively — the K>=2 invariant
                    // means a non-root interior has >= 3 children before a removal).
                    if level == 0 {
                        // The root emptied: it becomes the empty-table leaf, id intact.
                        let bytes =
                            PageBuilder::new(page_size, usable, pid, PageType::LeafTable).finish();
                        return put_page(pager, pid, bytes);
                    }
                    // `pid` is a dead non-root interior: it will be unlinked from its
                    // parent as the removal propagates up, so reclaim it now.
                    orphans.push(pid);
                    state = Prop::Removed { idx: path[level - 1].1 as usize, rightmost_max: None };
                    continue;
                }

                // `pid`'s corrected subtree max, if it changed. `rightmost_max` (a max
                // carried up by a lower merge that kept a right-most child) and
                // `was_rightmost` (we removed `pid`'s own right-most child, so its new max
                // is the dropped separator) are mutually exclusive.
                debug_assert!(
                    !(rightmost_max.is_some() && was_rightmost),
                    "a carried right-most max and a right-most removal cannot co-occur",
                );
                let node_max_changed: Option<i64> = if let Some(m) = rightmost_max {
                    Some(m)
                } else if was_rightmost {
                    Some(
                        dropped_key
                            .ok_or_else(|| Error::format("delete: right-most removal without a separator"))?,
                    )
                } else {
                    None
                };

                if level == 0 {
                    // The root is exempt from K>=2. A single child collapses height into
                    // the root id; two-or-more children stay a valid root (K>=1), and its
                    // own max change is irrelevant (no parent separator to retarget).
                    if children.len() == 1 {
                        return collapse_root(pager, root, children[0], page_size, usable, orphans);
                    }
                    let bytes = build_interior(pid, &children, &keys, page_size, usable)?
                        .ok_or_else(|| Error::format("delete: interior overflow after removing a child"))?;
                    return put_page(pager, pid, bytes);
                }

                if children.len() >= 3 {
                    // `pid` (non-root) survives at K>=2. Rebuild it, then propagate its
                    // own max change (if any) — a right-most removal or a carried max.
                    let bytes = build_interior(pid, &children, &keys, page_size, usable)?
                        .ok_or_else(|| Error::format("delete: interior overflow after removing a child"))?;
                    put_page(pager, pid, bytes)?;
                    match node_max_changed {
                        Some(m) => {
                            state = Prop::MaxChanged(m);
                            continue;
                        }
                        None => return Ok(()),
                    }
                }

                // `pid` (non-root) dropped below K>=2 (one or two children): rebalance it
                // with an adjacent sibling before continuing upward.
                match rebalance(
                    pager, path, level, root, pid, children, keys, node_max_changed, page_size, usable,
                    orphans,
                )? {
                    Next::Stop => return Ok(()),
                    Next::Propagate(next) => {
                        state = next;
                        continue;
                    }
                }
            }
        }
    }
    // Climbed past the root: a `MaxChanged` here means the tree's global-maximum
    // subtree changed, which no separator references, so there is nothing to do.
    Ok(())
}

/// Rebalance an under-min non-root interior `node` (its post-removal `nc` / `nk` child
/// and key lists, holding one or two children) with an adjacent sibling, reachable
/// through the parent at `path[level - 1]`. `node_max_changed` is `node`'s corrected
/// subtree max when the removal that under-filled it also changed its max (else `None`,
/// meaning the parent's separator for `node` is still correct).
///
/// Strategy (classic B-tree underflow, adapted to the table b-tree's "separator = left
/// subtree max" routing keys):
/// - **Rotate** when an adjacent sibling has a child to spare (stays `K >= 2` after
///   lending, i.e. `>= 4` children) and `node` has two children (a rotate then lifts it
///   to `K == 2`): move the bordering parent separator and one sibling child into
///   `node`, and the sibling's now-bordering separator up to the parent. The parent's
///   fan-out is unchanged, so the pass stops (unless `node` is the parent's right-most
///   child and its max changed — then the parent's max changed too, carried up).
/// - **Merge** otherwise: concatenate `node`, the pulled-down parent separator, and an
///   adjacent sibling into one page (an under-min interior plus a `K == 2` sibling is
///   five children — always one page, interior cells being tiny), write it into one of
///   the two ids, free the other, and remove the dropped child from the parent (a
///   `Removed` carried up, which may cascade the same handling / collapse the root).
///   Prefer merging with the right sibling: the merged node then keeps the right
///   sibling's (unchanged) max, so no parent separator goes stale. Merging left is used
///   only when `node` is the parent's right-most child; then `node`'s max becomes the
///   parent's max and is carried up via `rightmost_max`.
#[allow(clippy::too_many_arguments)]
fn rebalance(
    pager: &mut dyn Pager,
    path: &[(PageId, u32)],
    level: usize,
    root: PageId,
    node_id: PageId,
    mut nc: Vec<u32>,
    mut nk: Vec<i64>,
    node_max_changed: Option<i64>,
    page_size: usize,
    usable: usize,
    orphans: &mut Vec<PageId>,
) -> Result<Next> {
    let (parent_id, p_idx_u32) = path[level - 1];
    let p_idx = p_idx_u32 as usize;
    let (pchildren, mut pkeys) = read_interior_lists(pager, parent_id, usable)?;
    debug_assert_eq!(
        pchildren.get(p_idx).copied(),
        Some(node_id),
        "the parent's pointer at p_idx must reference the underflowing node",
    );
    let has_left = p_idx > 0;
    let has_right = p_idx + 1 < pchildren.len();
    debug_assert!(has_left || has_right, "an under-min non-root interior always has a sibling");

    // Try to rotate first (it keeps tree height and stops propagation). A rotate lifts
    // `node` to exactly K==2, so it only helps when `node` currently has two children.
    if nc.len() == 2 {
        if has_left {
            let left_id = pchildren[p_idx - 1];
            let (mut lc, mut lk) = read_interior_lists(pager, left_id, usable)?;
            if lc.len() >= 4 {
                // ROTATE from the left sibling: its right-most child moves to the front
                // of `node`, bordered by the old parent separator (= that child's max);
                // the sibling's now-dropped last key becomes the parent separator.
                let sep = pkeys[p_idx - 1];
                let lm = lc.pop().expect("left sibling has >= 4 children");
                let lm_sep = lk.pop().expect("left sibling has >= 3 keys");
                nc.insert(0, lm);
                nk.insert(0, sep);
                pkeys[p_idx - 1] = lm_sep;
                // `node`'s right-most child is unchanged by a left rotate, but the removal
                // may have changed its max; reflect that in the parent separator bordering
                // `node`, or carry it up when `node` is the parent's right-most child.
                let mut carried: Option<i64> = None;
                if let Some(m) = node_max_changed {
                    if p_idx < pkeys.len() {
                        pkeys[p_idx] = m;
                    } else {
                        carried = Some(m);
                    }
                }
                write_interior(pager, left_id, &lc, &lk, page_size, usable)?;
                write_interior(pager, node_id, &nc, &nk, page_size, usable)?;
                // A left rotate retargets one or two parent separators; on a near-full
                // parent a retarget that widens (a small positive separator replaced by a
                // negative, i.e. 9-byte, one) can overflow it. That is the same class as a
                // `MaxChanged` retarget overflow: real sqlite grows the tree, so write-or-
                // split the parent rather than fail. `carried` is `Some(m)` only when `node`
                // is the parent's right-most child and its max changed, so the parent's own
                // max became `m` and must be carried up.
                return write_parent_or_split(
                    pager, path, level, root, parent_id, &pchildren, &pkeys, carried, page_size,
                    usable, orphans,
                );
            }
        }
        if has_right {
            let right_id = pchildren[p_idx + 1];
            let (mut rc, mut rk) = read_interior_lists(pager, right_id, usable)?;
            if rc.len() >= 4 {
                // ROTATE from the right sibling: its left-most child moves to the back of
                // `node`, bordered by `node`'s old max; the moved child's key becomes the
                // new parent separator (= `node`'s new max). `node` has a right sibling,
                // so pkeys[p_idx] exists and holds `node`'s current max unless the removal
                // changed it.
                let node_max = node_max_changed.unwrap_or(pkeys[p_idx]);
                let r0 = rc.remove(0);
                let r0_sep = rk.remove(0);
                nc.push(r0);
                nk.push(node_max);
                pkeys[p_idx] = r0_sep;
                write_interior(pager, node_id, &nc, &nk, page_size, usable)?;
                write_interior(pager, right_id, &rc, &rk, page_size, usable)?;
                // A right rotate installs `node`'s new max (`r0_sep`) as the bordering
                // parent separator; on a near-full parent that can widen it past one page.
                // `node` has a right sibling, so it is NOT the parent's right-most child and
                // its max change is captured inside the parent (no carry). Write-or-split.
                return write_parent_or_split(
                    pager, path, level, root, parent_id, &pchildren, &pkeys, None, page_size,
                    usable, orphans,
                );
            }
        }
    }

    // No sibling can spare a child: MERGE. Prefer the right sibling (the merged node
    // keeps the right sibling's max, so no parent separator goes stale).
    if has_right {
        let right_id = pchildren[p_idx + 1];
        let (rc, rk) = read_interior_lists(pager, right_id, usable)?;
        // The separator between `node`'s part and the right sibling's part in the merged
        // page is `node`'s current max (corrected if the removal changed it), not the
        // possibly-stale parent separator.
        let node_max = node_max_changed.unwrap_or(pkeys[p_idx]);
        let mut mc = nc;
        mc.extend(rc);
        let mut mk = nk;
        mk.push(node_max);
        mk.extend(rk);
        write_interior(pager, right_id, &mc, &mk, page_size, usable)?;
        orphans.push(node_id);
        // The parent loses `node` (at p_idx); the stale pkeys[p_idx] it drops is the one
        // we folded (corrected) into the merged page, so no separator is left stale.
        Ok(Next::Propagate(Prop::Removed { idx: p_idx, rightmost_max: None }))
    } else {
        // `node` is the parent's right-most child: merge with the left sibling, keeping
        // `node`'s id so it stays the right-most child. The pulled-down separator is the
        // left|node parent separator (= the left sibling's max, unaffected by `node`'s
        // removal).
        let left_id = pchildren[p_idx - 1];
        let (lc, lk) = read_interior_lists(pager, left_id, usable)?;
        let sep = pkeys[p_idx - 1];
        let mut mc = lc;
        mc.extend(nc);
        let mut mk = lk;
        mk.push(sep);
        mk.extend(nk);
        write_interior(pager, node_id, &mc, &mk, page_size, usable)?;
        orphans.push(left_id);
        // The parent loses the left sibling (at p_idx - 1); the merged node (kept at
        // `node`'s id) stays the parent's right-most child. If `node`'s max changed, the
        // parent's max changed too — carry it up as `rightmost_max`.
        Ok(Next::Propagate(Prop::Removed { idx: p_idx - 1, rightmost_max: node_max_changed }))
    }
}

/// Write the PARENT of a just-rotated node, splitting it if the rotate widened a
/// separator past one page. A rotate leaves the parent's fan-out UNCHANGED — it only
/// changes a separator's value, hence its varint width — so an overflow here is the same
/// class as the `MaxChanged` retarget overflow in [`propagate`]: a separator that
/// re-encodes wider (a small positive replaced by a negative 9-byte one, or a value that
/// crosses a varint band) can push a near-full parent over. Real sqlite grows the tree and
/// the DELETE succeeds; it does not fail closed. So this reuses insert's split machinery,
/// exactly as the primary fix does.
///
/// `carry` is `Some(m)` ONLY for a left rotate whose `node` is the parent's right-most
/// child and whose subtree max changed to `m` (so the parent's own max became `m` and must
/// be carried up); every other rotate captured its max change inside the parent's own
/// separators and passes `None`.
///
/// - **Fits:** write the parent and return `Stop`, or `Propagate(MaxChanged(m))` to carry
///   the right-most max up (unchanged from the pre-split behavior).
/// - **Overflows:** grow the tree. When `carry` is `Some(m)`, carry `m` up FIRST. In that
///   case `node`'s subtree lies entirely above the (non-negative) left-sibling separator
///   that widened, so `m > 0`, and `m <=` the parent's old max, i.e. `m` is
///   narrower-or-equal: its ancestor retarget can NEVER itself overflow, so the carry only
///   retargets one ancestor separator (or climbs the right spine) and never splits. That
///   leaves the parent split's `propagate_split` to shift the now-corrected ancestor
///   separator into place bordering `node`'s new right-most page. Then split the parent
///   (`balance_deeper_interior` if it is the root) and `Stop` — the propagation has fixed
///   everything above the parent.
#[allow(clippy::too_many_arguments)]
fn write_parent_or_split(
    pager: &mut dyn Pager,
    path: &[(PageId, u32)],
    level: usize,
    root: PageId,
    parent_id: PageId,
    pchildren: &[u32],
    pkeys: &[i64],
    carry: Option<i64>,
    page_size: usize,
    usable: usize,
    orphans: &mut Vec<PageId>,
) -> Result<Next> {
    if let Some(bytes) = build_interior(parent_id, pchildren, pkeys, page_size, usable)? {
        put_page(pager, parent_id, bytes)?;
        return Ok(match carry {
            Some(m) => Next::Propagate(Prop::MaxChanged(m)),
            None => Next::Stop,
        });
    }
    // The parent overflowed from a widened separator. It sits at `path[level - 1]` (the
    // node being rebalanced is at `level`, so `level >= 1` and this does not underflow).
    let parent_level = level - 1;
    if let Some(m) = carry {
        // Load-bearing invariant, checked (the doc comment proves it in prose): reaching the
        // overflow arm with `carry = Some` means the ONLY parent-key change was the left
        // separator bordering the right-most `node` re-encoding wider, which can only happen
        // when it crossed from non-negative to negative. `node` sits entirely to the right of
        // that separator, so its max `m` is strictly positive — hence `m <=` the parent's old
        // (positive, same-or-wider varint) max, i.e. the ancestor retarget below is
        // narrower-or-equal and can NEVER itself overflow/split. If this ever fires, the
        // carry-then-split ordering has been invalidated and the retarget could double-split
        // overlapping ancestors and silently corrupt the tree, so fail loud in debug.
        debug_assert!(
            m > 0,
            "carry+overflow retarget must be narrower-or-equal (m must be > 0), got m={m}"
        );
        // Carry `node`'s max up before splitting so the split's `propagate_split` leaves a
        // correct separator bordering `node`'s new page (see the doc comment: this retarget
        // is narrower-or-equal and never itself splits). When the parent is the root there
        // is no ancestor to carry into — its max is the tree's global max, unreferenced.
        if parent_level > 0 {
            propagate(pager, &path[..parent_level], root, Prop::MaxChanged(m), page_size, usable, orphans)?;
        }
    }
    if parent_level == 0 {
        // The parent is the root; grow depth in place (its page id stays stable).
        balance_deeper_interior(pager, root, pchildren, pkeys, page_size, usable)?;
    } else {
        // Splitting the parent promotes new siblings into the grandparent, whose descent
        // entry is `path[parent_level - 1]`; propagate through `&path[..parent_level]`.
        let (nc, ns) = split_interior_in_place(pager, parent_id, pchildren, pkeys, page_size, usable)?;
        propagate_split(pager, &path[..parent_level], root, ns, nc, page_size, usable)?;
    }
    Ok(Next::Stop)
}

/// Build interior page `id` from `children` / `keys` and write it (header-preserving),
/// failing closed if it overflows. This is used only for the DELETE rebalance writes that
/// provably fit one page: the rotate CHILD moves (a rotate lifts `node` from one/two
/// children to exactly `K == 2` and shrinks the lending sibling — both shrink or barely
/// grow) and the MERGE writes (an under-min node plus a `K <= 2` sibling is at most five
/// children = four tiny interior cells). The one rebalance write that CAN widen a
/// separator and overflow — installing a rotate's retargeted separator into the PARENT —
/// does NOT come here; it goes through [`write_parent_or_split`], which splits like insert
/// rather than failing. So a `None` from [`build_interior`] here is a real corruption
/// (a write that was supposed to fit did not), not the widening-overflow case.
fn write_interior(
    pager: &mut dyn Pager,
    id: PageId,
    children: &[u32],
    keys: &[i64],
    page_size: usize,
    usable: usize,
) -> Result<()> {
    let bytes = build_interior(id, children, keys, page_size, usable)?
        .ok_or_else(|| Error::format("delete: rebalanced interior overflowed one page"))?;
    put_page(pager, id, bytes)
}

/// Collapse a root that has a single child by pulling the child's content up into
/// the root page id (so the id never changes), reducing tree height — the inverse of
/// insert's balance-deeper. Loops through any single-child pass-throughs beneath the
/// root. Every page whose content is pulled up, and every pass-through bypassed on the
/// way down, is no longer reachable once the root is rewritten, so its id is pushed to
/// `orphans` for the caller to free. A page kept as the root's single child (when the
/// reduced-capacity page-1 root cannot absorb it) stays reachable and is NOT freed.
fn collapse_root(
    pager: &mut dyn Pager,
    root: PageId,
    single_child: PageId,
    page_size: usize,
    usable: usize,
    orphans: &mut Vec<PageId>,
) -> Result<()> {
    let mut child = single_child;
    loop {
        let is_leaf = {
            let data = pager.read_page(child)?;
            let view = PageView::new(data, child, usable)?;
            if !view.page_type().is_table() {
                return Err(Error::format("delete: non-table page beneath the root"));
            }
            view.page_type().is_leaf()
        };

        if is_leaf {
            // Pull the child leaf's cells into the root leaf page. The reduced-capacity
            // root (page 1) may not hold a full child leaf; then keep the root as a
            // single-child interior pointing at `child` (a valid pass-through) rather
            // than corrupt the tree.
            if let Some(bytes) = try_copy_leaf(pager, root, child, page_size, usable)? {
                // `child`'s cells now live on the root, so the child page is dead.
                orphans.push(child);
                return put_page(pager, root, bytes);
            }
            // `child` stays reachable as the root's single child, so it is not freed.
            let bytes = build_interior(root, &[child], &[], page_size, usable)?
                .ok_or_else(|| Error::format("delete: root single-child interior failed to build"))?;
            return put_page(pager, root, bytes);
        }

        // Interior child: pull its (children, keys) up into the root.
        let (children, keys) = read_interior_lists(pager, child, usable)?;
        if children.len() >= 2 {
            match build_interior(root, &children, &keys, page_size, usable)? {
                Some(bytes) => {
                    // `child`'s children/keys now live on the root, so the child is dead.
                    orphans.push(child);
                    return put_page(pager, root, bytes);
                }
                None => {
                    // Does not fit the reduced-capacity root (page 1): keep a
                    // single-child interior pointing at `child` (still reachable, not
                    // freed).
                    let bytes = build_interior(root, &[child], &[], page_size, usable)?.ok_or_else(
                        || Error::format("delete: root single-child interior failed to build"),
                    )?;
                    return put_page(pager, root, bytes);
                }
            }
        }
        if children.len() == 1 {
            // The child is itself a single-child pass-through: rewriting the root to
            // point further down bypasses it, so reclaim it and descend.
            orphans.push(child);
            child = children[0];
            continue;
        }
        return Err(Error::format("delete: root's single child had no children"));
    }
}

/// Drop the cell at `idx` from leaf `leaf_id`, returning the leaf's new maximum rowid
/// when `idx` was its last (largest) cell (so the caller can retarget a stale separator
/// up the path), else `None`. `leaf_count` is the leaf's cell count BEFORE the drop.
///
/// FAST route: edit the page bytes in place ([`Pager::page_mut`] +
/// [`delete_cell`](minisqlite_fileformat::delete_cell)) — an O(cell) splice that leaves
/// a freeblock (the page is not defragmented). A returned mutable borrow cannot
/// auto-commit, so `page_mut` requires an active transaction and fails closed otherwise;
/// then this FALLS BACK to [`rebuild_leaf_dropping`] + [`put_page`], the original O(page)
/// whole-leaf rebuild (which defragments). Both routes leave the identical surviving cell
/// set, cell-pointer array, cell_count, and total free space, and compute the identical
/// `new_max` (a rowid is layout-independent); they differ only in physical layout
/// (freeblock vs defragmented), which no reader or upward fix-up keys off.
///
/// A removal only shrinks the page, so it always fits — there is no split on delete, and
/// the fast edit cannot fail for lack of room (a `delete_cell` error would be page
/// corruption, propagated loud via `?` rather than leaving a half-edited page).
fn drop_leaf_cell(
    pager: &mut dyn Pager,
    leaf_id: PageId,
    idx: usize,
    leaf_count: usize,
    page_size: usize,
    usable: usize,
) -> Result<Option<i64>> {
    // The mutable borrow lives ONLY inside the `Ok` arm; on `Err` nothing is borrowed,
    // so the fallback below can re-borrow the pager (mirrors insert's `insert_leaf_cell`).
    match pager.page_mut(leaf_id) {
        Ok(page) => {
            delete_cell(page, leaf_id, usable, idx)?;
            // `delete_cell` must shrink the leaf by EXACTLY one cell: the `new_max` read
            // below (and the fallback's separator retarget) assume the survivors are cells
            // `0..leaf_count-1`. Assert that post-condition at its cause (debug-only, so no
            // release cost) — a regression here would otherwise install a stale separator
            // silently up the path.
            debug_assert_eq!(
                PageView::new(page, leaf_id, usable)
                    .expect("edited leaf must re-parse")
                    .cell_count() as usize,
                leaf_count - 1,
                "delete_cell dropped != 1 cell from the table leaf"
            );
            // Read the new maximum off the just-edited page (an immutable reborrow of
            // `page`) only when the dropped cell was the leaf's last (largest) one. This
            // is exactly the fallback's condition and value.
            let new_max = if leaf_count >= 2 && idx + 1 == leaf_count {
                let view = PageView::new(page, leaf_id, usable)?;
                let last = view.cell_count() as usize - 1;
                Some(view.table_leaf_cell(last)?.rowid)
            } else {
                None
            };
            return Ok(new_max);
        }
        Err(_) => { /* no active transaction: fall back to the whole-page rebuild. */ }
    }
    let (bytes, new_max) = rebuild_leaf_dropping(pager, leaf_id, idx, page_size, usable)?;
    put_page(pager, leaf_id, bytes)?;
    Ok(new_max)
}

/// Rebuild leaf `leaf_id` without the cell at `skip_idx`, re-encoding every survivor
/// (preserving its overflow pointer). Returns the page bytes and, when `skip_idx`
/// was the leaf's maximum (last) cell, the new maximum rowid so the caller can
/// retarget a stale separator. Removing a cell only shrinks the page, so the
/// survivors always fit the page they came from. This is the FALLBACK route of
/// [`drop_leaf_cell`], used when in-place mutation is unavailable (auto-commit).
fn rebuild_leaf_dropping(
    pager: &dyn Pager,
    leaf_id: PageId,
    skip_idx: usize,
    page_size: usize,
    usable: usize,
) -> Result<(Vec<u8>, Option<i64>)> {
    let data = pager.read_page(leaf_id)?;
    let view = PageView::new(data, leaf_id, usable)?;
    let count = view.cell_count() as usize;
    let mut b = PageBuilder::new(page_size, usable, leaf_id, PageType::LeafTable);
    let mut tmp = Vec::new();
    for i in 0..count {
        if i == skip_idx {
            continue;
        }
        let cell = view.table_leaf_cell(i)?;
        tmp.clear();
        encode_table_leaf_cell(cell.payload_len, cell.rowid, cell.local_payload, cell.overflow_page, &mut tmp);
        if !b.add_cell(&tmp) {
            return Err(Error::format("delete: leaf overflowed after removing a cell"));
        }
    }
    // If we removed the last (largest) cell, the new maximum is the previous cell.
    let new_max = if count >= 2 && skip_idx + 1 == count {
        Some(view.table_leaf_cell(count - 2)?.rowid)
    } else {
        None
    };
    Ok((b.finish(), new_max))
}

/// Try to rebuild `dest` as a leaf holding `src_leaf`'s cells. `Ok(None)` if they do
/// not fit `dest` — only possible when `dest` is the reduced-capacity page 1, whose
/// 100-byte database header leaves less room than a full-size source page.
fn try_copy_leaf(
    pager: &dyn Pager,
    dest: PageId,
    src_leaf: PageId,
    page_size: usize,
    usable: usize,
) -> Result<Option<Vec<u8>>> {
    let data = pager.read_page(src_leaf)?;
    let view = PageView::new(data, src_leaf, usable)?;
    let count = view.cell_count() as usize;
    let mut b = PageBuilder::new(page_size, usable, dest, PageType::LeafTable);
    let mut tmp = Vec::new();
    for i in 0..count {
        let cell = view.table_leaf_cell(i)?;
        tmp.clear();
        encode_table_leaf_cell(cell.payload_len, cell.rowid, cell.local_payload, cell.overflow_page, &mut tmp);
        if !b.add_cell(&tmp) {
            return Ok(None);
        }
    }
    Ok(Some(b.finish()))
}

/// Read an interior table page into owned `(children, keys)` lists, where
/// `children[j]` is the child at pointer index `j` (the last being the right-most
/// pointer) and `keys[j]` the separator bordering `children[j]`. Mirrors the shape
/// [`build_interior`] consumes.
fn read_interior_lists(pager: &dyn Pager, pid: PageId, usable: usize) -> Result<(Vec<u32>, Vec<i64>)> {
    let data = pager.read_page(pid)?;
    let view = PageView::new(data, pid, usable)?;
    if !view.page_type().is_interior() || !view.page_type().is_table() {
        return Err(Error::format("delete: expected an interior table page"));
    }
    let count = view.cell_count() as usize;
    let mut children = Vec::with_capacity(count + 1);
    let mut keys = Vec::with_capacity(count);
    for i in 0..count {
        let cell = view.table_interior_cell(i)?;
        children.push(cell.left_child);
        keys.push(cell.rowid);
    }
    children.push(
        view.right_most_pointer()
            .ok_or_else(|| Error::format("delete: interior page missing its right-most pointer"))?,
    );
    Ok((children, keys))
}
