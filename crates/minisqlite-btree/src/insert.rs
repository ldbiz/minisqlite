//! `table_insert`: insert (or REPLACE) a rowid-keyed row into a table b-tree, with
//! page splitting and root balance-deeper.
//!
//! Shape of the algorithm:
//! 1. Descend from `root` to the target leaf, recording the interior path as
//!    `(page_id, pointer_index)` so a split can be propagated to each parent.
//! 2. Decide whether the new cell fits the leaf (see `plan_leaf_edit`). If it fits,
//!    splice it in — the FAST path edits the leaf bytes IN PLACE (`page_mut` +
//!    `insert_cell`), an O(cell) amortized splice, so a bulk load does not re-encode a
//!    whole page per row. REPLACE is an in-place delete-then-insert.
//! 3. Otherwise split the leaf into N pages and propagate the promoted separators up
//!    the path, splitting interior pages as needed.
//! 4. When the page that must split is the root, **balance-deeper** instead of
//!    promoting: move the root's content down to new child pages and turn the root
//!    into an interior page — so the root page id never changes (schema/`root_page`
//!    references depend on it). Page 1 stays the root with its 100-byte header.
//!
//! A single insert can force an N-way (not just 2-way) split: a cell that is large
//! relative to its neighbours may need a page of its own, so a leaf can split into
//! three or more pages at once and promote several separators together. The split
//! *arithmetic* (which cells land where) is planned purely in `build::plan_*_groups`;
//! this module allocates the ids, builds each page, and splices the promoted
//! separators into the parent.
//!
//! Fast route and fallback route, ONE insert path. The in-place splice needs a
//! mutable page borrow (`Pager::page_mut`), which requires an active transaction (a
//! returned `&mut` cannot auto-commit). When there is none — e.g. a bare
//! `table_insert` in auto-commit — `page_mut` fails closed and the path falls back to
//! the O(page) whole-page rebuild (`try_build_leaf` + `put_page`), preserving the old
//! behavior exactly. The split/balance-deeper machinery is shared by both routes and
//! still builds pages through `PageBuilder`/`put_page`, so page 1's database header is
//! preserved and split pages stay defragmented.

use minisqlite_fileformat::{
    delete_cell, encode_table_leaf_cell, insert_cell, leaf_cell_len, leaf_free_space, CellKind,
    PageBuilder, PageType, PageView, TableLeafCell,
};
use minisqlite_pager::{PageId, Pager};
use minisqlite_types::{Error, Result};

use crate::build::{build_interior, build_leaf, plan_interior_groups, plan_leaf_groups, LeafCell};
use crate::nav::choose_child;
use crate::tree::{put_page, usable_of};

/// Insert `payload` (an encoded record, treated as opaque bytes) under `rowid` into
/// the table b-tree rooted at `root`, keeping cells in ascending rowid order. If
/// `rowid` already exists its payload is replaced. Runs within whatever transaction
/// the caller has open (it does not begin/commit).
pub fn table_insert(pager: &mut dyn Pager, root: PageId, rowid: i64, payload: &[u8]) -> Result<()> {
    let page_size = pager.page_size() as usize;
    let usable = usable_of(pager);

    // A payload larger than the page's inline threshold spills onto a chain of
    // overflow pages; the on-page cell then keeps only the inline prefix plus a
    // pointer to the chain head. `write_overflow_chain` allocates and writes the
    // chain (nothing when the payload fits inline) and returns how many bytes stay
    // inline. The resulting cell is bounded (`local <= usable-35 < usable`), so the
    // split/balance-deeper planning below is unchanged.
    //
    // A REPLACE over a row that already spilled would orphan that old chain; the leaf
    // pass below captures the replaced cell's overflow head so it is freed back to the
    // freelist before the new cell is written (see `replaced_overflow`). The new chain
    // is allocated here first, so a replace grows by at most one transient chain and
    // then reuses the freed pages — never a leak.
    let (local_len, overflow_page) =
        crate::overflow_io::write_overflow_chain(pager, payload, CellKind::TableLeaf, usable)?;
    let mut new_cell = Vec::new();
    encode_table_leaf_cell(
        payload.len() as u64,
        rowid,
        &payload[..local_len],
        overflow_page,
        &mut new_cell,
    );

    // (1) Descend to the target leaf, recording the interior path.
    let mut path: Vec<(PageId, u32)> = Vec::new();
    let mut current = root;
    let leaf_id = loop {
        let data = pager.read_page(current)?;
        let view = PageView::new(data, current, usable)?;
        if !view.page_type().is_table() {
            return Err(Error::format("table_insert reached a non-table b-tree page"));
        }
        if view.page_type().is_leaf() {
            break current;
        }
        let (child, pointer_index) = choose_child(&view, rowid)?;
        path.push((current, pointer_index));
        current = child;
    };

    // (2) Under a READ borrow of the leaf, decide whether the new cell fits in place
    // and (for a REPLACE) capture the replaced row's overflow head. The fit test is
    // byte-for-byte equivalent to whether a `PageBuilder` rebuild of the leaf's cells
    // (with this one spliced in / replaced) fits one page (see `plan_leaf_edit`), so the
    // split path below triggers at exactly the same points as the whole-page rebuild.
    let (plan, replaced_overflow) = {
        let data = pager.read_page(leaf_id)?;
        plan_leaf_edit(data, leaf_id, rowid, &new_cell, usable)?
    };

    // The page borrow is dropped, so the mutable `free_page` calls are now legal.
    // Reclaim the replaced row's old overflow chain before writing the new leaf, so
    // its pages return to the freelist instead of leaking (fileformat2 §1.5).
    if let Some(old_first) = replaced_overflow {
        crate::overflow_io::free_overflow_chain(pager, old_first, usable)?;
    }

    let cells = match plan {
        // (Fast route) The cell fits: splice it into the leaf in place — an O(cell)
        // edit of the page bytes — falling back to a whole-page `PageBuilder` rebuild
        // only when in-place mutation is unavailable (see `insert_leaf_cell`).
        LeafPlan::InPlace { pos, is_replace } => {
            return insert_leaf_cell(
                pager, leaf_id, rowid, &new_cell, pos, is_replace, page_size, usable,
            );
        }
        LeafPlan::Split(cells) => cells,
    };

    // (3)/(4) The leaf overflowed. If it is the root, balance-deeper; else split it
    // (into N pages) and propagate the promoted separators up.
    if path.is_empty() {
        return balance_deeper_leaf(pager, root, &cells, page_size, usable);
    }
    let (new_children, seps) = split_leaf_in_place(pager, leaf_id, &cells, page_size, usable)?;
    propagate_split(pager, &path, root, seps, new_children, page_size, usable)
}

/// How to place the new cell on the leaf: splice it in place, or split the leaf.
enum LeafPlan {
    /// The cell fits: insert it at key-order `pos`. `is_replace` marks a same-rowid
    /// REPLACE, whose existing cell sits at that same `pos` and is deleted first (so
    /// `Some(other_index)` is not a representable state — the delete and insert share
    /// one position).
    InPlace { pos: usize, is_replace: bool },
    /// The cell does not fit one page: the ordered cell list to split into N pages.
    Split(Vec<LeafCell>),
}

/// Binary-search a table leaf `view` for `rowid`. Returns the key-order position where
/// a cell with `rowid` belongs (the lower bound over the ascending rowids) and, if a
/// cell with exactly `rowid` is already there, that cell (a REPLACE). O(log N), so a
/// sequential bulk load — every rowid larger than all present — lands at `pos == count`
/// with a couple of comparisons instead of an O(N) scan.
fn locate_rowid<'a>(
    view: &PageView<'a>,
    rowid: i64,
) -> Result<(usize, Option<TableLeafCell<'a>>)> {
    let n = view.cell_count() as usize;
    let (mut lo, mut hi) = (0usize, n);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if view.table_leaf_cell(mid)?.rowid < rowid {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    let existing = if lo < n {
        let c = view.table_leaf_cell(lo)?;
        if c.rowid == rowid { Some(c) } else { None }
    } else {
        None
    };
    Ok((lo, existing))
}

/// Decide, under a READ borrow of the leaf page `data`, how to place `new_cell` (for
/// `rowid`): an in-place splice at its key-order position, or (when it will not fit one
/// page) the full ordered cell list to split. Also returns the overflow-chain head of
/// any same-rowid cell being REPLACEd, for the caller to free.
///
/// The fit test is derived from the leaf's total reclaimable free space and is exactly
/// equivalent to whether a `PageBuilder` could repack all the cells plus this one:
///   - a plain insert adds one cell (`new_cell.len()` bytes) and one 2-byte pointer, so
///     it fits iff `total_free >= new_cell.len() + 2`;
///   - a REPLACE swaps a same-rowid cell (`old_len` on-page bytes, same pointer), so it
///     fits iff `total_free + old_len >= new_cell.len()`.
/// Both match `try_build_leaf`'s `PageBuilder` capacity exactly (re-encoding a cell
/// preserves its length), so the in-place route and the fallback route split at the same
/// boundary. `insert_cell` reclaims freeblocks/fragments by defragmenting when the
/// contiguous gap is short, so a "fits" verdict here always succeeds in place.
fn plan_leaf_edit(
    data: &[u8],
    leaf_id: PageId,
    rowid: i64,
    new_cell: &[u8],
    usable: usize,
) -> Result<(LeafPlan, Option<u32>)> {
    let view = PageView::new(data, leaf_id, usable)?;
    let (pos, existing) = locate_rowid(&view, rowid)?;
    let replaced_overflow = existing.and_then(|c| c.overflow_page);
    let free = leaf_free_space(data, leaf_id, usable)?;
    let fits = match existing {
        None => free.total >= new_cell.len() + 2,
        Some(_) => {
            let off = view.cell_pointer(pos);
            let old_len = leaf_cell_len(data, leaf_id, usable, off)?;
            free.total + old_len >= new_cell.len()
        }
    };
    if fits {
        Ok((LeafPlan::InPlace { pos, is_replace: existing.is_some() }, replaced_overflow))
    } else {
        Ok((LeafPlan::Split(collect_leaf_cells(&view, rowid, new_cell)?), replaced_overflow))
    }
}

/// Place `new_cell` on `leaf_id` at key-order `pos`, first deleting the same-rowid cell
/// (at that same `pos`) when `is_replace`. The FAST route edits the page bytes in place through
/// [`Pager::page_mut`] — an O(cell) splice, no whole-page re-encode. Because a returned
/// `&mut` borrow cannot auto-commit, `page_mut` requires an active transaction and fails
/// closed otherwise; then this FALLS BACK to rebuilding the whole leaf with
/// [`try_build_leaf`] and writing it via [`put_page`] (which auto-commits), the original
/// O(page) behavior. The caller has already verified the cell fits (see `plan_leaf_edit`),
/// so a fallback rebuild that overflows is corruption and fails loud rather than silently
/// splitting a page the fit check accepted.
#[allow(clippy::too_many_arguments)]
fn insert_leaf_cell(
    pager: &mut dyn Pager,
    leaf_id: PageId,
    rowid: i64,
    new_cell: &[u8],
    pos: usize,
    is_replace: bool,
    page_size: usize,
    usable: usize,
) -> Result<()> {
    // The mutable borrow lives ONLY inside the `Ok` arm; on `Err` nothing is borrowed,
    // so the fallback below can re-borrow the pager.
    match pager.page_mut(leaf_id) {
        Ok(page) => {
            if is_replace {
                delete_cell(page, leaf_id, usable, pos)?;
            }
            insert_cell(page, leaf_id, usable, pos, new_cell)?;
            return Ok(());
        }
        Err(_) => { /* no active transaction: fall back to the whole-page rebuild. */ }
    }
    let bytes = {
        let data = pager.read_page(leaf_id)?;
        let view = PageView::new(data, leaf_id, usable)?;
        match try_build_leaf(&view, leaf_id, rowid, new_cell, page_size, usable)? {
            LeafBuild::Fits(bytes) => bytes,
            LeafBuild::Overflow => {
                return Err(Error::format(
                    "leaf rebuild overflowed after the in-place fit check accepted it",
                ));
            }
        }
    };
    put_page(pager, leaf_id, bytes)
}

/// Whether a whole-page leaf rebuild fit or overflowed (the split signal).
enum LeafBuild {
    Fits(Vec<u8>),
    Overflow,
}

/// Rebuild `view` (a leaf) with `new_cell` (for `rowid`) spliced into rowid order,
/// dropping any existing cell with the same rowid (REPLACE). Streams straight into a
/// `PageBuilder`, re-encoding existing cells from their borrowed bytes. This is the
/// FALLBACK route used only when `page_mut` is unavailable (auto-commit): the caller has
/// already freed any replaced overflow chain, so this does not re-derive one.
///
/// Returns `LeafBuild::Fits(bytes)` when the result fits one page, or
/// `LeafBuild::Overflow` when it does not (the signal to split).
fn try_build_leaf(
    view: &PageView,
    leaf_id: PageId,
    rowid: i64,
    new_cell: &[u8],
    page_size: usize,
    usable: usize,
) -> Result<LeafBuild> {
    let mut b = PageBuilder::new(page_size, usable, leaf_id, PageType::LeafTable);
    let count = view.cell_count() as usize;
    let mut inserted = false;
    let mut tmp = Vec::new();
    for i in 0..count {
        let cell = view.table_leaf_cell(i)?;
        if !inserted && rowid <= cell.rowid {
            if !b.add_cell(new_cell) {
                return Ok(LeafBuild::Overflow);
            }
            inserted = true;
            if rowid == cell.rowid {
                continue; // REPLACE: drop the old same-rowid cell.
            }
        }
        tmp.clear();
        encode_table_leaf_cell(cell.payload_len, cell.rowid, cell.local_payload, cell.overflow_page, &mut tmp);
        if !b.add_cell(&tmp) {
            return Ok(LeafBuild::Overflow);
        }
    }
    if !inserted && !b.add_cell(new_cell) {
        return Ok(LeafBuild::Overflow);
    }
    Ok(LeafBuild::Fits(b.finish()))
}

/// Collect a leaf's cells as an owned, ordered list with `new_cell` spliced in (or
/// replacing the same rowid). `new_cell` is the already-encoded cell — including any
/// overflow pointer — so a spilled new row is placed byte-identically to the in-place
/// route. Used only on the split path, so the per-cell copies are amortized across a
/// whole page's worth of inserts.
fn collect_leaf_cells(view: &PageView, rowid: i64, new_cell: &[u8]) -> Result<Vec<LeafCell>> {
    let count = view.cell_count() as usize;
    let mut out: Vec<LeafCell> = Vec::with_capacity(count + 1);
    let mut inserted = false;
    for i in 0..count {
        let cell = view.table_leaf_cell(i)?;
        if !inserted && rowid <= cell.rowid {
            out.push((rowid, new_cell.to_vec()));
            inserted = true;
            if rowid == cell.rowid {
                continue; // REPLACE: drop the old same-rowid cell.
            }
        }
        let mut cb = Vec::new();
        encode_table_leaf_cell(cell.payload_len, cell.rowid, cell.local_payload, cell.overflow_page, &mut cb);
        out.push((cell.rowid, cb));
    }
    if !inserted {
        out.push((rowid, new_cell.to_vec()));
    }
    Ok(out)
}

/// Split an overfull leaf's ordered `cells` into `>= 2` pages: the first reuses
/// `first_id` (the leaf being split, kept in place), the rest are freshly allocated.
/// Returns `(new_children, seps)` — the ids of the pages *after* the first and, in
/// parallel, the separator (max rowid) that precedes each — ready to splice into the
/// parent. Both vectors have length `groups - 1`.
fn split_leaf_in_place(
    pager: &mut dyn Pager,
    first_id: PageId,
    cells: &[LeafCell],
    page_size: usize,
    usable: usize,
) -> Result<(Vec<PageId>, Vec<i64>)> {
    let groups = plan_leaf_groups(cells, page_size, usable)?;
    let mut new_children = Vec::with_capacity(groups.len().saturating_sub(1));
    let mut seps = Vec::with_capacity(groups.len().saturating_sub(1));
    for (gi, &(s, e)) in groups.iter().enumerate() {
        let id = if gi == 0 { first_id } else { pager.allocate_page()? };
        let bytes = build_leaf(id, &cells[s..e], page_size, usable)?
            .ok_or_else(|| Error::format("leaf split page overflowed after planning"))?;
        put_page(pager, id, bytes)?;
        if gi + 1 < groups.len() {
            // The separator that precedes the NEXT page = this page's max rowid.
            seps.push(cells[e - 1].0);
        }
        if gi > 0 {
            new_children.push(id);
        }
    }
    Ok((new_children, seps))
}

/// Split an overfull interior page's `children` / `keys` into `>= 2` pages: the first
/// reuses `first_id`, the rest are freshly allocated. Returns `(new_children, seps)` —
/// the ids of the pages after the first and the promoted separators between them,
/// ready to splice into the parent. Both vectors have length `groups - 1`.
///
/// `pub(crate)` so DELETE (`delete.rs`) can reuse the one split implementation when a
/// separator retarget widens an interior past one page — the delete path must never
/// drift from insert's tree shape. Mirrors `index.rs` exposing its split machinery to
/// `index_delete`.
pub(crate) fn split_interior_in_place(
    pager: &mut dyn Pager,
    first_id: PageId,
    children: &[u32],
    keys: &[i64],
    page_size: usize,
    usable: usize,
) -> Result<(Vec<PageId>, Vec<i64>)> {
    let groups = plan_interior_groups(children, keys, page_size, usable)?;
    let mut new_children = Vec::with_capacity(groups.len().saturating_sub(1));
    let mut seps = Vec::with_capacity(groups.len().saturating_sub(1));
    for (gi, &(s, e)) in groups.iter().enumerate() {
        let id = if gi == 0 { first_id } else { pager.allocate_page()? };
        let bytes = build_interior(id, &children[s..=e], &keys[s..e], page_size, usable)?
            .ok_or_else(|| Error::format("interior split page overflowed after planning"))?;
        put_page(pager, id, bytes)?;
        if gi + 1 < groups.len() {
            // `keys[e]` is the cell promoted out of this group (its subtree max).
            seps.push(keys[e]);
        }
        if gi > 0 {
            new_children.push(id);
        }
    }
    Ok((new_children, seps))
}

/// Propagate a set of promoted `(seps, new_children)` up the recorded interior `path`,
/// from the deepest parent toward the root. `new_children[t]` is a new sibling of the
/// page the path descended into, and `seps[t]` is the separator that precedes it (both
/// in ascending order, `seps.len() == new_children.len()`). At each interior we splice
/// them in after the descended pointer; if the page still fits we write it and stop;
/// if it overflows we split it (promoting again) and continue; if the overflowing page
/// is the root we balance-deeper and stop.
///
/// `pub(crate)` so DELETE (`delete.rs`) can reuse this exact upward-split propagation
/// after a separator retarget (or a rotate) overflows an interior — one split
/// implementation shared with insert, as `index.rs` shares it with `index_delete`.
pub(crate) fn propagate_split(
    pager: &mut dyn Pager,
    path: &[(PageId, u32)],
    root: PageId,
    mut seps: Vec<i64>,
    mut new_children: Vec<PageId>,
    page_size: usize,
    usable: usize,
) -> Result<()> {
    for level in (0..path.len()).rev() {
        let (pid, pointer_index) = path[level];
        let (mut children, mut keys) = read_interior_lists(pager, pid, usable)?;
        // The child at `pointer_index` split into itself plus `new_children`. Insert
        // each new child pointer after it, and each separator before it, keeping the
        // interior's `(child, key)` alignment (see build::plan_interior_groups).
        let j = pointer_index as usize;
        for (t, &child) in new_children.iter().enumerate() {
            children.insert(j + 1 + t, child);
        }
        for (t, &s) in seps.iter().enumerate() {
            keys.insert(j + t, s);
        }

        if let Some(bytes) = build_interior(pid, &children, &keys, page_size, usable)? {
            return put_page(pager, pid, bytes);
        }
        // pid overflowed.
        if level == 0 {
            return balance_deeper_interior(pager, root, &children, &keys, page_size, usable);
        }
        let (nc, ns) = split_interior_in_place(pager, pid, &children, &keys, page_size, usable)?;
        new_children = nc;
        seps = ns;
    }
    // Unreachable: a non-empty path always resolves at level 0 (fit or balance-deeper).
    Err(Error::format("split propagation ran past the root"))
}

/// Read an interior table page into owned `(children, keys)` lists (see build docs).
fn read_interior_lists(pager: &dyn Pager, pid: PageId, usable: usize) -> Result<(Vec<u32>, Vec<i64>)> {
    let data = pager.read_page(pid)?;
    let view = PageView::new(data, pid, usable)?;
    if !view.page_type().is_interior() || !view.page_type().is_table() {
        return Err(Error::format("expected an interior table page during split propagation"));
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
            .ok_or_else(|| Error::format("interior page missing its right-most pointer"))?,
    );
    Ok((children, keys))
}

/// Grow the tree one level while keeping `root`'s page id: move the root's
/// (overflowing) leaf content down to one or more fresh child pages and make the root
/// an interior page over them. If the content fits a single (full-capacity) child the
/// root gets one right pointer and zero cells — the shape real sqlite's balance-deeper
/// writes, and the only way page 1's reduced capacity is reconciled with a full child.
/// Page 1 stays the root with its database header intact.
fn balance_deeper_leaf(
    pager: &mut dyn Pager,
    root: PageId,
    cells: &[LeafCell],
    page_size: usize,
    usable: usize,
) -> Result<()> {
    let groups = plan_leaf_groups(cells, page_size, usable)?;
    let mut children = Vec::with_capacity(groups.len());
    let mut keys = Vec::with_capacity(groups.len().saturating_sub(1));
    for (gi, &(s, e)) in groups.iter().enumerate() {
        let id = pager.allocate_page()?;
        let bytes = build_leaf(id, &cells[s..e], page_size, usable)?
            .ok_or_else(|| Error::format("leaf child overflowed after planning during balance-deeper"))?;
        put_page(pager, id, bytes)?;
        children.push(id);
        if gi + 1 < groups.len() {
            keys.push(cells[e - 1].0);
        }
    }
    write_root_interior(pager, root, &children, &keys, page_size, usable)
}

/// Balance-deeper for an interior root: move the root's (overflowing) interior content
/// down to one or more fresh child interior pages and make the root point to them.
/// Mirrors `balance_deeper_leaf`.
///
/// `pub(crate)` so DELETE (`delete.rs`) can grow the tree when a separator retarget (or
/// a rotate) overflows the *root* interior — the root page id stays stable, exactly as
/// insert requires. Mirrors `index.rs` exposing the same to `index_delete`.
pub(crate) fn balance_deeper_interior(
    pager: &mut dyn Pager,
    root: PageId,
    children: &[u32],
    keys: &[i64],
    page_size: usize,
    usable: usize,
) -> Result<()> {
    let groups = plan_interior_groups(children, keys, page_size, usable)?;
    let mut root_children = Vec::with_capacity(groups.len());
    let mut root_keys = Vec::with_capacity(groups.len().saturating_sub(1));
    for (gi, &(s, e)) in groups.iter().enumerate() {
        let id = pager.allocate_page()?;
        let bytes = build_interior(id, &children[s..=e], &keys[s..e], page_size, usable)?
            .ok_or_else(|| Error::format("interior child overflowed after planning during balance-deeper"))?;
        put_page(pager, id, bytes)?;
        root_children.push(id);
        if gi + 1 < groups.len() {
            root_keys.push(keys[e]);
        }
    }
    write_root_interior(pager, root, &root_children, &root_keys, page_size, usable)
}

/// Write `root` as an interior page over the given `children` / `keys`. Used only by
/// balance-deeper, where `children` are freshly built child pages holding all of the
/// root's former content, so the root itself holds only pointers.
fn write_root_interior(
    pager: &mut dyn Pager,
    root: PageId,
    children: &[u32],
    keys: &[i64],
    page_size: usize,
    usable: usize,
) -> Result<()> {
    let root_bytes = build_interior(root, children, keys, page_size, usable)?.ok_or_else(|| {
        Error::format("root interior overflowed after balance-deeper (too many separators)")
    })?;
    put_page(pager, root, root_bytes)
}
