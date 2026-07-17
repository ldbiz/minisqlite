//! Auto_vacuum / incremental_vacuum RECLAMATION: shrinking an auto_vacuum database by
//! removing free pages from the file (fileformat2 §1.8, pragma.html "auto_vacuum" /
//! "incremental_vacuum"). This is the write-side companion to [`crate::ptrmap_build`],
//! which keeps the ptrmap VALID; this module makes the file SMALLER.
//!
//! ## What "reclaim" does
//! A free page can only be removed from the file if it is at the very END (the file is
//! a flat array of pages; only the tail can be truncated). So reclamation works
//! tail-inward: repeatedly look at the last page and
//!   - a **free** page → drop it (truncate one page, one page reclaimed);
//!   - a **pointer-map** page that now covers nothing (its data pages were all
//!     truncated) → drop it (structural, not counted against the reclaim budget);
//!   - a **live** page → RELOCATE it into a lower free slot (rewriting the single
//!     pointer that names it), which turns the tail into a free page the next step
//!     drops — this is how an *interior* free page is reclaimed.
//! When the whole freelist is gone the file has zero free pages (FULL's contract);
//! [`reclaim`] stops early after `max` pages for `PRAGMA incremental_vacuum(N)`.
//!
//! ## Correctness stance: never corrupt, decline instead
//! Relocation rewrites live b-tree structure, so a bug there could corrupt the file.
//! Two guards make that impossible to persist:
//!   1. [`reclaim`] runs inside its own transaction ([`Cow::incremental_vacuum`]); any
//!      error rolls the whole thing back, leaving the pre-reclaim, fully-valid database.
//!   2. Before truncating, [`verify_forest`] re-walks the forest and refuses (errors) if
//!      any live pointer still names a page past the new end or a page went unaccounted
//!      — so a missed pointer rewrite declines rather than truncating a referenced page.
//! The ptrmap itself is re-derived from scratch by [`crate::ptrmap_build::finalize`] at
//! commit, so reclamation only has to leave the FOREST (pointers + freelist + page
//! count) consistent; it never hand-writes ptrmap entries.

use std::collections::{BTreeSet, HashMap};

use minisqlite_fileformat::freelist::parse_trunk;
use minisqlite_fileformat::ptrmap::{
    decode_ptrmap_entry, ptrmap_offset_of, ptrmap_page_of, PTRMAP_BTREE, PTRMAP_ENTRY_SIZE,
    PTRMAP_OVERFLOW1, PTRMAP_OVERFLOW2, PTRMAP_ROOTPAGE,
};
use minisqlite_fileformat::{
    header_offset_for, read_serial_value, read_varint, serial_type_payload_len, DatabaseHeader,
    PageType, PageView, HEADER_SIZE,
};
use minisqlite_types::{Error, Result, Value};

use crate::cow::Cow;
use crate::store::PageStore;
use crate::PageId;

/// The column index of `rootpage` in a `sqlite_schema` row (`type, name, tbl_name,
/// rootpage, sql`) — the pointer a moved b-tree ROOT is named by.
const SCHEMA_ROOTPAGE_COL: usize = 3;

/// The result of a reclamation pass: how many free pages were removed from the file,
/// and whether a table/index ROOT page moved (which changes a `sqlite_schema.rootpage`,
/// so the engine must reload its cached catalog before it queries by the old root).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VacuumOutcome {
    /// Free pages removed from the file this pass.
    pub reclaimed: u32,
    /// A root page moved, so `sqlite_schema` rootpages changed on disk and any cached
    /// catalog is stale.
    pub root_moved: bool,
}

/// Reclaim up to `max` free pages (all of them when `max` is `None`, i.e. FULL commit or
/// `incremental_vacuum` with no/`<1` argument). Runs inside the caller's active write
/// transaction; the caller commits (which triggers [`crate::ptrmap_build::finalize`] to
/// rebuild the ptrmap for the smaller file) or rolls back. A no-op returning
/// `reclaimed == 0` for a non-auto_vacuum database or one with no free pages, so the
/// default path never pays for this.
pub(crate) fn reclaim<S: PageStore>(cow: &mut Cow<S>, max: Option<u32>) -> Result<VacuumOutcome> {
    let count = cow.page_count()?;
    if count < 1 {
        return Ok(VacuumOutcome::default());
    }
    let header = match read_header(cow)? {
        Some(h) if h.uses_ptrmap() => h,
        _ => return Ok(VacuumOutcome::default()),
    };
    let usable = header.usable_size();
    let budget = max.unwrap_or(u32::MAX);
    if budget == 0 {
        return Ok(VacuumOutcome::default());
    }

    // The live set of free pages (trunks + leaves), held in memory and mutated as pages
    // are reclaimed/relocated; the on-disk freelist is rebuilt from what remains at the
    // end, so the loop never has to splice a specific page out of a trunk chain.
    let mut free: BTreeSet<PageId> = load_freelist(cow, &header, count)?;
    if free.is_empty() {
        return Ok(VacuumOutcome::default());
    }

    let mut target = count;
    let mut reclaimed = 0u32;
    let mut root_moved = false;
    // Maps a page relocated during THIS pass to its new home, so a later step that needs
    // page P's parent (from the ptrmap) resolves the parent's CURRENT location even after
    // the parent itself moved.
    let mut moved: HashMap<PageId, PageId> = HashMap::new();

    while reclaimed < budget && target > 1 {
        let last = target;
        if is_ptrmap_page(usable, last) {
            // A trailing ptrmap page covers only pages > `last`, which are already gone;
            // drop it (structural overhead, not a reclaimed free page).
            target -= 1;
            continue;
        }
        if free.remove(&last) {
            // The last page is free: just drop it.
            target -= 1;
            reclaimed += 1;
            continue;
        }
        // The last page is LIVE. To reclaim an interior free page we must move this live
        // page down into one. Pick the lowest free slot (< last, and it stays <= the
        // final size, so it is never itself relocated later).
        let Some(&dest) = free.iter().next() else {
            // No free slot below to move into: nothing left to reclaim.
            break;
        };
        free.remove(&dest);
        if relocate_page(cow, last, dest, usable, &mut moved)? {
            root_moved = true;
        }
        target -= 1;
        reclaimed += 1;
    }

    // Strip any trailing pointer-map page the reclamation exposed (it now covers no live
    // data). Safe: a ptrmap page only ever covers pages that come AFTER it.
    while target > 1 && is_ptrmap_page(usable, target) {
        target -= 1;
    }

    if target == count {
        // Nothing was removed (e.g. every free page sits under a live tail page and no
        // relocation happened). Leave the database exactly as it was.
        return Ok(VacuumOutcome { reclaimed, root_moved });
    }

    // Rebuild the on-disk freelist from the pages still free (all < `target`), then check
    // the forest before shrinking. Order matters: `free_in_txn` validates ids against the
    // current (pre-truncation) page count, so rebuild first.
    rebuild_freelist(cow, &free)?;
    verify_forest(cow, target, usable, &free)?;
    cow.truncate_to(target)?;

    Ok(VacuumOutcome { reclaimed, root_moved })
}

/// Relocate live page `from` into free slot `dest` by rewriting the ONE pointer that
/// names `from` (identified by `from`'s ptrmap back-pointer) and copying its bytes.
/// Returns `true` if `from` was a table/index ROOT (so `sqlite_schema` changed).
///
/// The ptrmap is NOT updated here — [`crate::ptrmap_build::finalize`] re-derives the
/// whole ptrmap from the relocated forest at commit. Only the forest pointer, the page
/// content, and (for a root) the schema row are rewritten. Fails closed on any
/// inconsistency so a relocation bug declines rather than corrupts.
fn relocate_page<S: PageStore>(
    cow: &mut Cow<S>,
    from: PageId,
    dest: PageId,
    usable: usize,
    moved: &mut HashMap<PageId, PageId>,
) -> Result<bool> {
    let (kind, orig_parent) = read_ptrmap(cow, from, usable)?;
    // The ptrmap records the ORIGINAL parent; if it too moved this pass, follow the
    // relocation map to its current home (that is where the pointer to `from` now lives).
    let parent = resolve_moved(moved, orig_parent);

    let mut root_moved = false;
    match kind {
        PTRMAP_BTREE => rewrite_child_pointer(cow, parent, from, dest, usable)?,
        PTRMAP_OVERFLOW1 => rewrite_overflow_head(cow, parent, from, dest, usable)?,
        PTRMAP_OVERFLOW2 => rewrite_overflow_next(cow, parent, from, dest)?,
        PTRMAP_ROOTPAGE => {
            rewrite_schema_rootpage(cow, from, dest, usable)?;
            root_moved = true;
        }
        other => {
            return Err(Error::Format(format!(
                "reclaim: page {from} has unexpected ptrmap type {other}; declining to relocate"
            )));
        }
    }

    // Copy the page's bytes into its new home. `from`'s own child/overflow pointers move
    // with it unchanged (its children did not move), so the subtree stays intact.
    let bytes = cow.read_page(from)?.to_vec();
    cow.stage_write(dest, &bytes)?;
    moved.insert(from, dest);
    Ok(root_moved)
}

/// Follow the relocation chain to a page's current location. Bounded by the map size so
/// a corrupt cycle cannot loop.
fn resolve_moved(moved: &HashMap<PageId, PageId>, mut page: PageId) -> PageId {
    let mut steps = 0usize;
    while let Some(&next) = moved.get(&page) {
        page = next;
        steps += 1;
        if steps > moved.len() {
            break;
        }
    }
    page
}

/// Rewrite the child pointer in interior b-tree page `parent` that equals `from`,
/// setting it to `dest`. Scans the interior cells and the right-most pointer.
///
/// Shared with [`crate::roots_first`], which rewrites the same forest pointers when it
/// moves a non-root page out of a slot a b-tree root must occupy (§1.8 roots-first).
pub(crate) fn rewrite_child_pointer<S: PageStore>(
    cow: &mut Cow<S>,
    parent: PageId,
    from: PageId,
    dest: PageId,
    usable: usize,
) -> Result<()> {
    let page = cow.read_page(parent)?.to_vec();
    let view = PageView::new(&page, parent, usable)?;
    let n = view.cell_count() as usize;
    match view.page_type() {
        PageType::InteriorTable => {
            for i in 0..n {
                if view.table_interior_cell(i)?.left_child == from {
                    return overwrite_be32(cow, parent, view.cell_pointer(i), dest);
                }
            }
        }
        PageType::InteriorIndex => {
            for i in 0..n {
                if view.index_interior_cell(i)?.left_child == from {
                    return overwrite_be32(cow, parent, view.cell_pointer(i), dest);
                }
            }
        }
        other => {
            return Err(Error::Format(format!(
                "reclaim: b-tree child {from}'s parent {parent} is a {other:?} page, not interior"
            )));
        }
    }
    if view.right_most_pointer() == Some(from) {
        // The right-most pointer sits at header offset + 8 on an interior page.
        return overwrite_be32(cow, parent, header_offset_for(parent) + 8, dest);
    }
    Err(Error::Format(format!(
        "reclaim: interior page {parent} has no child pointer equal to {from}; declining"
    )))
}

/// Rewrite the overflow-chain HEAD pointer in b-tree page `parent` (the cell whose
/// spilled payload begins at `from`), setting it to `dest`. The 4-byte overflow pointer
/// sits immediately after a cell's inline payload.
///
/// Shared with [`crate::roots_first`] (see [`rewrite_child_pointer`]).
pub(crate) fn rewrite_overflow_head<S: PageStore>(
    cow: &mut Cow<S>,
    parent: PageId,
    from: PageId,
    dest: PageId,
    usable: usize,
) -> Result<()> {
    let page = cow.read_page(parent)?.to_vec();
    let view = PageView::new(&page, parent, usable)?;
    let n = view.cell_count() as usize;
    for i in 0..n {
        let cp = view.cell_pointer(i);
        let (local_len, overflow, content_off) = match view.page_type() {
            PageType::LeafTable => {
                let c = view.table_leaf_cell(i)?;
                let (_pl, a) = varint_at(&page, cp)?;
                let (_rid, b) = varint_at(&page, cp + a)?;
                (c.local_payload.len(), c.overflow_page, cp + a + b)
            }
            PageType::LeafIndex => {
                let c = view.index_leaf_cell(i)?;
                let (_pl, a) = varint_at(&page, cp)?;
                (c.local_payload.len(), c.overflow_page, cp + a)
            }
            PageType::InteriorIndex => {
                let c = view.index_interior_cell(i)?;
                let (_pl, a) = varint_at(&page, cp + 4)?;
                (c.local_payload.len(), c.overflow_page, cp + 4 + a)
            }
            other => {
                return Err(Error::Format(format!(
                    "reclaim: overflow head {from}'s parent {parent} is a {other:?} page"
                )));
            }
        };
        if overflow == Some(from) {
            return overwrite_be32(cow, parent, content_off + local_len, dest);
        }
    }
    Err(Error::Format(format!(
        "reclaim: page {parent} has no cell whose overflow head is {from}; declining"
    )))
}

/// Rewrite the "next overflow page" pointer (the first 4 bytes) of overflow page
/// `parent`, from `from` to `dest`.
///
/// Shared with [`crate::roots_first`] (see [`rewrite_child_pointer`]).
pub(crate) fn rewrite_overflow_next<S: PageStore>(
    cow: &mut Cow<S>,
    parent: PageId,
    from: PageId,
    dest: PageId,
) -> Result<()> {
    let page = cow.read_page(parent)?;
    let next = u32::from_be_bytes([page[0], page[1], page[2], page[3]]);
    if next != from {
        return Err(Error::Format(format!(
            "reclaim: overflow page {parent} next is {next}, expected {from}; declining"
        )));
    }
    overwrite_be32(cow, parent, 0, dest)
}

/// Rewrite the `rootpage` column of the `sqlite_schema` row whose value is `from`,
/// setting it to `dest`. The new value is encoded in the SAME serial width the old one
/// used — always possible because `dest < from`, so it fits — an in-place same-size
/// overwrite that never resizes the cell or rebalances the schema b-tree.
///
/// Shared with [`crate::roots_first`], which moves a b-tree root DOWN to a low slot
/// (`dest < from` holds there too) to satisfy §1.8 roots-first.
pub(crate) fn rewrite_schema_rootpage<S: PageStore>(
    cow: &mut Cow<S>,
    from: PageId,
    dest: PageId,
    usable: usize,
) -> Result<()> {
    let mut stack = vec![1u32];
    let mut visited = BTreeSet::new();
    while let Some(p) = stack.pop() {
        if !visited.insert(p) {
            continue;
        }
        let page = cow.read_page(p)?.to_vec();
        let view = PageView::new(&page, p, usable)?;
        match view.page_type() {
            PageType::InteriorTable => {
                for i in 0..view.cell_count() as usize {
                    stack.push(view.table_interior_cell(i)?.left_child);
                }
                if let Some(rm) = view.right_most_pointer() {
                    stack.push(rm);
                }
            }
            PageType::LeafTable => {
                for i in 0..view.cell_count() as usize {
                    let cp = view.cell_pointer(i);
                    let cell = view.table_leaf_cell(i)?;
                    // Read `rootpage` (col 3) from the inline `local_payload` only. It
                    // almost always stays inline (it precedes the large `sql` column, so a
                    // record spills AFTER it), but that is NOT guaranteed: a pathological
                    // schema row whose type/name/tbl_name are large enough to push col 3
                    // past the inline boundary makes `rootpage_field` return Err ("past
                    // inline payload"), which propagates out and DECLINES this reclaim —
                    // fail-closed, never corrupt. `ptrmap_build` assembles overflow so it
                    // derives that root correctly regardless, so the file stays valid and
                    // cross-readable; only the (pathological-case) compaction is skipped.
                    if let Some((rel, width)) = rootpage_field(cell.local_payload, from)? {
                        let (_pl, a) = varint_at(&page, cp)?;
                        let (_rid, b) = varint_at(&page, cp + a)?;
                        let record_off = cp + a + b;
                        return write_int_in_place(cow, p, record_off + rel, width, dest);
                    }
                }
            }
            other => {
                return Err(Error::Format(format!(
                    "reclaim: sqlite_schema page {p} is a {other:?} page, expected a table b-tree"
                )));
            }
        }
    }
    Err(Error::Format(format!(
        "reclaim: no sqlite_schema row has rootpage {from}; declining to relocate a root"
    )))
}

/// If this schema record's `rootpage` column (index 3) equals `from`, return
/// `(offset_within_record, width)` of that column's value so the caller can overwrite it
/// in place. `Ok(None)` when this row's rootpage is not `from` (or is not an integer).
fn rootpage_field(record: &[u8], from: PageId) -> Result<Option<(usize, usize)>> {
    let (header_len, hn) =
        read_varint(record).ok_or_else(|| Error::Format("reclaim: truncated record header".into()))?;
    let header_len = header_len as usize;
    let header_end = header_len.min(record.len());
    let mut type_off = hn;
    let mut body_off = header_len;
    let mut col = 0usize;
    while type_off < header_end {
        let (serial, n) = read_varint(&record[type_off..header_end])
            .ok_or_else(|| Error::Format("reclaim: truncated record serial".into()))?;
        type_off += n;
        let width = serial_type_payload_len(serial);
        if col == SCHEMA_ROOTPAGE_COL {
            let end = body_off
                .checked_add(width)
                .ok_or_else(|| Error::Format("reclaim: record body offset overflow".into()))?;
            let body = record
                .get(body_off..end)
                .ok_or_else(|| Error::Format("reclaim: rootpage column past inline payload".into()))?;
            return match read_serial_value(serial, body) {
                Value::Integer(v) if v == from as i64 => Ok(Some((body_off, width))),
                _ => Ok(None),
            };
        }
        body_off = body_off
            .checked_add(width)
            .ok_or_else(|| Error::Format("reclaim: record body offset overflow".into()))?;
        col += 1;
    }
    Ok(None)
}

/// Overwrite the `width`-byte big-endian integer at `off` in page `page` with `dest`.
/// `dest < from` guarantees it fits in the same width the old (larger) root used.
fn write_int_in_place<S: PageStore>(
    cow: &mut Cow<S>,
    page: PageId,
    off: usize,
    width: usize,
    dest: PageId,
) -> Result<()> {
    let buf = cow.page_mut(page)?;
    let end = off
        .checked_add(width)
        .filter(|&e| e <= buf.len())
        .ok_or_else(|| Error::Format(format!("reclaim: rootpage field past page {page}")))?;
    let value = dest as u64;
    for (i, slot) in buf[off..end].iter_mut().enumerate() {
        *slot = ((value >> (8 * (width - 1 - i))) & 0xff) as u8;
    }
    Ok(())
}

/// Overwrite 4 big-endian bytes at `off` in page `id` with `value`.
fn overwrite_be32<S: PageStore>(cow: &mut Cow<S>, id: PageId, off: usize, value: u32) -> Result<()> {
    let buf = cow.page_mut(id)?;
    if off + 4 > buf.len() {
        return Err(Error::Format(format!("reclaim: pointer offset {off} out of range on page {id}")));
    }
    buf[off..off + 4].copy_from_slice(&value.to_be_bytes());
    Ok(())
}

/// Read a varint at `off` in `page`, returning `(value, byte_len)`.
fn varint_at(page: &[u8], off: usize) -> Result<(u64, usize)> {
    let rest = page.get(off..).ok_or_else(|| Error::Format("reclaim: cell offset past page".into()))?;
    read_varint(rest).ok_or_else(|| Error::Format("reclaim: truncated varint in cell".into()))
}

/// Read page `p`'s ptrmap entry as `(type, parent)`.
fn read_ptrmap<S: PageStore>(cow: &Cow<S>, p: PageId, usable: usize) -> Result<(u8, PageId)> {
    let b = ptrmap_page_of(usable, p);
    let off = ptrmap_offset_of(usable, p);
    let ptrmap = cow.read_page(b)?;
    let slice = ptrmap
        .get(off..off + PTRMAP_ENTRY_SIZE)
        .ok_or_else(|| Error::Format(format!("reclaim: ptrmap entry for page {p} past page {b}")))?;
    Ok(decode_ptrmap_entry(slice))
}

/// Whether page `p` sits at a ptrmap position.
fn is_ptrmap_page(usable: usize, p: PageId) -> bool {
    minisqlite_fileformat::is_ptrmap_page(usable, p)
}

/// Load the freelist (trunks + leaves) into a set, bounded by `count`.
fn load_freelist<S: PageStore>(
    cow: &Cow<S>,
    header: &DatabaseHeader,
    count: PageId,
) -> Result<BTreeSet<PageId>> {
    let usable = header.usable_size();
    let mut set = BTreeSet::new();
    let mut trunk = header.first_freelist_trunk;
    let mut steps: u64 = 0;
    let limit = count as u64 + 1;
    while trunk != 0 {
        steps += 1;
        if steps > limit {
            return Err(Error::Format("reclaim: freelist trunk chain longer than the database".into()));
        }
        set.insert(trunk);
        let (next, leaves) = parse_trunk(cow.read_page(trunk)?, usable)?;
        for leaf in leaves {
            set.insert(leaf);
        }
        trunk = next;
    }
    Ok(set)
}

/// Clear the on-disk freelist header, then re-add each remaining free page so the trunk
/// structure references only surviving pages (all `< target`).
///
/// Shared with [`crate::roots_first`], which rebuilds the freelist after moving a b-tree
/// root into a free low slot (the set of free pages changes: the consumed slot leaves,
/// the vacated root slot joins).
pub(crate) fn rebuild_freelist<S: PageStore>(
    cow: &mut Cow<S>,
    remaining: &BTreeSet<PageId>,
) -> Result<()> {
    clear_freelist_header(cow)?;
    for &p in remaining {
        crate::alloc::free_in_txn(cow, p)?;
    }
    Ok(())
}

/// Zero the freelist head (offset 32) and count (offset 36) on page 1.
fn clear_freelist_header<S: PageStore>(cow: &mut Cow<S>) -> Result<()> {
    let mut page1 = cow.read_page(1)?.to_vec();
    let head: &mut [u8; HEADER_SIZE] =
        (&mut page1[..HEADER_SIZE]).try_into().expect("page 1 is at least HEADER_SIZE bytes");
    let mut header = DatabaseHeader::read(head)?;
    header.first_freelist_trunk = 0;
    header.freelist_count = 0;
    header.write(head);
    cow.stage_write(1, &page1)
}

/// Verify-or-abort: walk the whole forest (page 1 + every `sqlite_schema` root) and refuse
/// to truncate if any live page lies past `target`, or any page in `1..=target` is neither
/// reachable, free, nor a ptrmap position. This catches a relocation that missed a pointer
/// (leaving it aimed at a page about to be truncated) so reclamation DECLINES rather than
/// truncating a still-referenced page.
fn verify_forest<S: PageStore>(
    cow: &Cow<S>,
    target: PageId,
    usable: usize,
    free: &BTreeSet<PageId>,
) -> Result<()> {
    let mut reachable: BTreeSet<PageId> = BTreeSet::new();
    reachable.insert(1);
    // page 1 is the schema root; walk it and every table/index root it names.
    walk_reachable(cow, 1, usable, &mut reachable)?;
    for root in collect_roots(cow, usable)? {
        walk_reachable(cow, root, usable, &mut reachable)?;
    }
    for &p in &reachable {
        if p > target {
            return Err(Error::Format(format!(
                "reclaim: live page {p} would survive past the truncation point {target}; declining"
            )));
        }
        if free.contains(&p) {
            return Err(Error::Format(format!(
                "reclaim: page {p} is both reachable and on the freelist; declining"
            )));
        }
    }
    for p in 2..=target {
        if is_ptrmap_page(usable, p) || reachable.contains(&p) || free.contains(&p) {
            continue;
        }
        return Err(Error::Format(format!(
            "reclaim: page {p} is unaccounted for after relocation (not reachable, free, or ptrmap); declining"
        )));
    }
    Ok(())
}

/// Collect every table/index root (`sqlite_schema.rootpage >= 2`) by walking the schema
/// table b-tree. Bounded by a visited set.
fn collect_roots<S: PageStore>(cow: &Cow<S>, usable: usize) -> Result<Vec<PageId>> {
    let mut roots = Vec::new();
    let mut visited = BTreeSet::new();
    let mut stack = vec![1u32];
    while let Some(p) = stack.pop() {
        if !visited.insert(p) {
            continue;
        }
        let page = cow.read_page(p)?.to_vec();
        let view = PageView::new(&page, p, usable)?;
        match view.page_type() {
            PageType::InteriorTable => {
                for i in 0..view.cell_count() as usize {
                    stack.push(view.table_interior_cell(i)?.left_child);
                }
                if let Some(rm) = view.right_most_pointer() {
                    stack.push(rm);
                }
            }
            PageType::LeafTable => {
                for i in 0..view.cell_count() as usize {
                    let cell = view.table_leaf_cell(i)?;
                    if let Some(root) = decode_rootpage(cell.local_payload)? {
                        roots.push(root);
                    }
                }
            }
            other => {
                return Err(Error::Format(format!(
                    "reclaim: sqlite_schema page {p} is a {other:?} page, expected a table b-tree"
                )));
            }
        }
    }
    Ok(roots)
}

/// Decode the `rootpage` column (index 3) of a schema record, returning the root page
/// (>= 2) or `None` for a view/trigger (rootpage 0) or a non-integer.
fn decode_rootpage(record: &[u8]) -> Result<Option<PageId>> {
    let (header_len, hn) =
        read_varint(record).ok_or_else(|| Error::Format("reclaim: truncated record header".into()))?;
    let header_len = header_len as usize;
    let header_end = header_len.min(record.len());
    let mut type_off = hn;
    let mut body_off = header_len;
    let mut col = 0usize;
    while type_off < header_end {
        let (serial, n) = read_varint(&record[type_off..header_end])
            .ok_or_else(|| Error::Format("reclaim: truncated record serial".into()))?;
        type_off += n;
        let width = serial_type_payload_len(serial);
        if col == SCHEMA_ROOTPAGE_COL {
            let end = body_off.checked_add(width);
            let body = end.and_then(|e| record.get(body_off..e));
            return Ok(match body.map(|b| read_serial_value(serial, b)) {
                Some(Value::Integer(v)) if v >= 2 && v <= PageId::MAX as i64 => Some(v as PageId),
                _ => None,
            });
        }
        body_off = body_off
            .checked_add(width)
            .ok_or_else(|| Error::Format("reclaim: record body offset overflow".into()))?;
        col += 1;
    }
    Ok(None)
}

/// Walk the b-tree rooted at `root`, inserting every page it points at into `reachable`.
/// Bounded by `reachable` (each page processed once), so a corrupt cycle cannot loop.
fn walk_reachable<S: PageStore>(
    cow: &Cow<S>,
    root: PageId,
    usable: usize,
    reachable: &mut BTreeSet<PageId>,
) -> Result<()> {
    if root == 0 {
        return Ok(());
    }
    let mut stack = vec![root];
    while let Some(p) = stack.pop() {
        if !reachable.insert(p) {
            continue;
        }
        let page = cow.read_page(p)?.to_vec();
        let view = PageView::new(&page, p, usable)?;
        let n = view.cell_count() as usize;
        match view.page_type() {
            PageType::InteriorTable => {
                for i in 0..n {
                    stack.push(view.table_interior_cell(i)?.left_child);
                }
                if let Some(rm) = view.right_most_pointer() {
                    stack.push(rm);
                }
            }
            PageType::LeafTable => {
                for i in 0..n {
                    if let Some(ov) = view.table_leaf_cell(i)?.overflow_page {
                        walk_overflow(cow, ov, usable, reachable)?;
                    }
                }
            }
            PageType::InteriorIndex => {
                for i in 0..n {
                    let cell = view.index_interior_cell(i)?;
                    stack.push(cell.left_child);
                    if let Some(ov) = cell.overflow_page {
                        walk_overflow(cow, ov, usable, reachable)?;
                    }
                }
                if let Some(rm) = view.right_most_pointer() {
                    stack.push(rm);
                }
            }
            PageType::LeafIndex => {
                for i in 0..n {
                    if let Some(ov) = view.index_leaf_cell(i)?.overflow_page {
                        walk_overflow(cow, ov, usable, reachable)?;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Walk an overflow chain from `first`, inserting each page into `reachable`. Bounded by
/// `reachable` so a corrupt self-referential chain cannot loop.
fn walk_overflow<S: PageStore>(
    cow: &Cow<S>,
    first: PageId,
    usable: usize,
    reachable: &mut BTreeSet<PageId>,
) -> Result<()> {
    let mut cur = first;
    while cur != 0 {
        if !reachable.insert(cur) {
            break;
        }
        cur = minisqlite_fileformat::parse_overflow_page(cow.read_page(cur)?, usable)?.next;
    }
    Ok(())
}

/// Parse the page-1 header through the overlay. `None` when page 1 is not a header yet.
fn read_header<S: PageStore>(cow: &Cow<S>) -> Result<Option<DatabaseHeader>> {
    let page1 = cow.read_page(1)?;
    let head: &[u8; HEADER_SIZE] = match page1.get(..HEADER_SIZE).and_then(|s| s.try_into().ok()) {
        Some(h) => h,
        None => return Ok(None),
    };
    Ok(DatabaseHeader::read(head).ok())
}

#[cfg(test)]
mod tests {
    //! Relocation-mechanics and verify-or-abort unit tests. These drive the RISKIEST
    //! reclamation internals directly with hand-built page images, so a relocation bug
    //! surfaces here (in the pager crate, which builds independently of the SQL layer)
    //! and not only through the end-to-end facade suite.
    //!
    //! Fixture strategy: build every page into an OPEN transaction's overlay
    //! ([`staged`]) and run `relocate_page` / `reclaim` / `verify_forest` in that same
    //! transaction, then roll back. Nothing is ever committed, so `ptrmap_build::finalize`
    //! (the commit-time ptrmap rebuild) never runs and the hand-built ptrmap / freelist /
    //! schema pages are read back EXACTLY as written — the tests pin these functions'
    //! behavior, not the commit path's.

    use super::*;
    use crate::store::MemStore;
    use minisqlite_fileformat::freelist::write_trunk;
    use minisqlite_fileformat::{
        CellKind, PageBuilder, PTRMAP_FREEPAGE, decode_record, encode_index_interior_cell,
        encode_index_leaf_cell, encode_record, encode_table_interior_cell, encode_table_leaf_cell,
        payload_split, put_ptrmap_entry,
    };

    /// Page size == usable size (reserved 0) for every fixture. 512 is the SQLite minimum
    /// (§1.3.4), so a modest payload spills onto overflow pages and — because a ptrmap page
    /// covers `J = 512/5 = 102` data pages — only page 2 is a pointer-map page in these
    /// tiny files, keeping the §1.8 offset math trivial to reason about by hand.
    const PS: usize = 512;

    fn store() -> MemStore {
        MemStore::new(PS as u32)
    }

    /// Build a database from explicit page images (`pages[0]` is page 1, `pages[1]` page 2,
    /// …) into an OPEN transaction and return the `Cow` with that transaction still active.
    /// Every image must be exactly `PS` bytes. The caller runs the function under test in
    /// this transaction and then rolls back; nothing commits, so no ptrmap finalize runs.
    fn staged(pages: Vec<Vec<u8>>) -> Cow<MemStore> {
        let mut cow = Cow::new(store());
        cow.begin().expect("begin");
        for (i, bytes) in pages.iter().enumerate() {
            assert_eq!(bytes.len(), PS, "page {} image must be exactly {PS} bytes", i + 1);
            let id = cow.grow_one().expect("grow_one");
            cow.stage_write(id, bytes).expect("stage_write");
        }
        cow
    }

    fn zeros() -> Vec<u8> {
        vec![0u8; PS]
    }

    /// Assert a decoded record is exactly one `Integer(expected)`. `Value` has no
    /// `PartialEq` (only `Debug`), so a direct `assert_eq!` on decoded rows will not
    /// compile — decode and match the single column instead.
    fn assert_one_int(record: &[u8], expected: i64) {
        let vals = decode_record(record);
        assert_eq!(vals.len(), 1, "expected exactly one column, got {vals:?}");
        match &vals[0] {
            Value::Integer(v) => assert_eq!(*v, expected, "integer column value"),
            other => panic!("expected Integer({expected}), got {other:?}"),
        }
    }

    fn set_be32(page: &mut [u8], off: usize, v: u32) {
        page[off..off + 4].copy_from_slice(&v.to_be_bytes());
    }

    fn be32_at(page: &[u8], off: usize) -> u32 {
        u32::from_be_bytes([page[off], page[off + 1], page[off + 2], page[off + 3]])
    }

    /// A ptrmap page (page 2) carrying the given `(data_page, type, parent)` entries at the
    /// §1.8 offsets for usable size `PS`.
    fn ptrmap_page(entries: &[(PageId, u8, PageId)]) -> Vec<u8> {
        let mut page = zeros();
        for &(p, kind, parent) in entries {
            put_ptrmap_entry(&mut page, ptrmap_offset_of(PS, p), kind, parent);
        }
        page
    }

    /// Write a valid auto_vacuum page-1 [`DatabaseHeader`] into the first 100 bytes of an
    /// already-built page-1 image (which carries the schema b-tree from byte 100 on).
    fn write_header(
        page1: &mut [u8],
        largest_root: u32,
        incremental: u32,
        first_trunk: u32,
        freelist_count: u32,
    ) {
        let header = DatabaseHeader {
            page_size: PS as u32,
            reserved_space: 0,
            largest_root_btree: largest_root,
            incremental_vacuum: incremental,
            first_freelist_trunk: first_trunk,
            freelist_count,
            ..DatabaseHeader::default()
        };
        let head: &mut [u8; HEADER_SIZE] =
            (&mut page1[..HEADER_SIZE]).try_into().expect("page 1 is at least HEADER_SIZE bytes");
        header.write(head);
    }

    /// A one-column table-leaf page holding rows `(rowid, Integer(value))` in ascending
    /// rowid order, built for `page_number`.
    fn int_leaf(page_number: u32, rows: &[(i64, i64)]) -> Vec<u8> {
        let mut b = PageBuilder::new(PS, PS, page_number, PageType::LeafTable);
        for &(rowid, value) in rows {
            let rec = encode_record(&[Value::Integer(value)]);
            let mut cell = Vec::new();
            encode_table_leaf_cell(rec.len() as u64, rowid, &rec, None, &mut cell);
            assert!(b.add_cell(&cell), "row {rowid} must fit on page {page_number}");
        }
        b.finish()
    }

    /// A page-1 `sqlite_schema` leaf naming a single table `t` with the given `rootpage`
    /// (record columns: `type, name, tbl_name, rootpage, sql`). No database header yet —
    /// the caller adds one with [`write_header`] when the test needs `uses_ptrmap`.
    fn schema_leaf_naming_root(rootpage: i64) -> Vec<u8> {
        let rec = encode_record(&[
            Value::Text("table".into()),
            Value::Text("t".into()),
            Value::Text("t".into()),
            Value::Integer(rootpage),
            Value::Text("CREATE TABLE t(a)".into()),
        ]);
        let mut cell = Vec::new();
        encode_table_leaf_cell(rec.len() as u64, 1, &rec, None, &mut cell);
        let mut b = PageBuilder::new(PS, PS, 1, PageType::LeafTable);
        assert!(b.add_cell(&cell), "schema row must fit on page 1");
        b.finish()
    }

    // ---- relocate_page: OVERFLOW2 (a later overflow page; rewrite the next pointer) ----

    #[test]
    fn relocate_rewrites_overflow_next_pointer() {
        // Chain: overflow page 4 -> page 5 (chain tail). Move live page 5 into free slot 6.
        // The ptrmap types page 5 as OVERFLOW2 with parent 4, so relocation rewrites page
        // 4's leading 4-byte "next overflow page" pointer from 5 to 6 and copies 5 -> 6.
        let (from, dest, parent) = (5u32, 6u32, 4u32);
        let mut p4 = zeros();
        set_be32(&mut p4, 0, from); // page 4's next-overflow pointer names page 5
        p4[4..].iter_mut().for_each(|b| *b = 0xC4); // recognizable body, left untouched
        let mut p5 = vec![0xABu8; PS]; // the page being moved
        set_be32(&mut p5, 0, 0); // page 5 is the chain tail

        let mut cow = staged(vec![
            zeros(),
            ptrmap_page(&[(from, PTRMAP_OVERFLOW2, parent)]),
            zeros(),
            p4,
            p5.clone(),
            zeros(),
        ]);
        let mut moved = HashMap::new();
        let root_moved = relocate_page(&mut cow, from, dest, PS, &mut moved).unwrap();

        assert!(!root_moved, "an overflow page is not a root");
        assert_eq!(moved.get(&from), Some(&dest), "the move is recorded");
        assert_eq!(be32_at(cow.read_page(parent).unwrap(), 0), dest, "parent next -> dest");
        assert_eq!(cow.read_page(dest).unwrap(), &p5[..], "dest is a byte copy of the moved page");
        // The rest of the parent page is untouched (the 0xC4 body after the pointer).
        assert_eq!(be32_at(cow.read_page(parent).unwrap(), 4), 0xC4C4_C4C4, "body untouched");
        cow.rollback().unwrap();
    }

    // ---- relocate_page: OVERFLOW1 (an overflow-chain head; rewrite the cell pointer) ----

    #[test]
    fn relocate_rewrites_overflow_head_in_leaf_cell() {
        // Page 3 is a table-leaf page whose single cell spills its payload onto page 5.
        // The ptrmap types page 5 as OVERFLOW1 with parent 3, so relocation rewrites the
        // 4-byte overflow pointer that follows the cell's inline payload from 5 to 6.
        let (from, dest, parent) = (5u32, 6u32, 3u32);
        let total = 2000u64;
        let split = payload_split(PS, total, CellKind::TableLeaf);
        assert!(split.has_overflow, "2000 bytes must spill on a 512-byte page");
        let local = vec![0x5Au8; split.local];
        let mut cell = Vec::new();
        encode_table_leaf_cell(total, 42, &local, Some(from), &mut cell);
        let mut b = PageBuilder::new(PS, PS, parent, PageType::LeafTable);
        assert!(b.add_cell(&cell), "the spilling cell must fit on the parent page");
        let p3 = b.finish();
        let p5 = vec![0xABu8; PS];

        let mut cow = staged(vec![
            zeros(),
            ptrmap_page(&[(from, PTRMAP_OVERFLOW1, parent)]),
            p3,
            zeros(),
            p5.clone(),
            zeros(),
        ]);
        let mut moved = HashMap::new();
        assert!(!relocate_page(&mut cow, from, dest, PS, &mut moved).unwrap());

        let parent_bytes = cow.read_page(parent).unwrap().to_vec();
        let view = PageView::new(&parent_bytes, parent, PS).unwrap();
        assert_eq!(
            view.table_leaf_cell(0).unwrap().overflow_page,
            Some(dest),
            "the cell's overflow head now points at dest"
        );
        assert_eq!(cow.read_page(dest).unwrap(), &p5[..], "dest is a byte copy of the moved page");
        cow.rollback().unwrap();
    }

    // The overflow-head parent above is a table-leaf page; the two tests below cover the
    // OTHER two page types `rewrite_overflow_head` handles, whose offset math to LOCATE the
    // overflow-head pointer genuinely DIFFERS (reclaim.rs, the `rewrite_overflow_head`
    // per-page-type arms):
    //   * index-leaf     `[payload-len varint][key][overflow(4B)]`            -> cp + a
    //   * index-interior `[left-child(4B)][payload-len varint][key][overflow]`-> cp + 4 + a
    // vs a table-leaf's `[payload-len varint][rowid varint][payload][overflow]` -> cp + a + b.
    // A wrong offset (a missing/extra varint, or the missing `+4` left-child on the interior
    // arm) silently overwrites the wrong 4 bytes and corrupts the relocated overflow pointer,
    // so each arm needs its own value-exact pin. Only the table-leaf arm was covered; the
    // facade suite reaches neither index arm (its index keys never spill to overflow).

    #[test]
    fn relocate_rewrites_overflow_head_in_index_leaf_cell() {
        // Page 3 is an INDEX-leaf page whose single cell's key spills onto page 5. An
        // index-leaf cell has NO rowid varint (unlike a table-leaf cell), so the overflow
        // head sits at `cp + a` (payload-len varint only) + the inline key length. The
        // ptrmap types page 5 as OVERFLOW1 with parent 3, so relocation rewrites that
        // 4-byte pointer from 5 to 6.
        let (from, dest, parent) = (5u32, 6u32, 3u32);
        let total = 2000u64;
        let split = payload_split(PS, total, CellKind::Index);
        assert!(split.has_overflow, "2000 bytes must spill on a 512-byte index page");
        let local = vec![0x5Au8; split.local];
        let mut cell = Vec::new();
        encode_index_leaf_cell(total, &local, Some(from), &mut cell);
        let mut b = PageBuilder::new(PS, PS, parent, PageType::LeafIndex);
        assert!(b.add_cell(&cell), "the spilling index-leaf cell must fit on the parent page");
        let p3 = b.finish();
        let p5 = vec![0xABu8; PS];

        let mut cow = staged(vec![
            zeros(),
            ptrmap_page(&[(from, PTRMAP_OVERFLOW1, parent)]),
            p3,
            zeros(),
            p5.clone(),
            zeros(),
        ]);
        let mut moved = HashMap::new();
        assert!(!relocate_page(&mut cow, from, dest, PS, &mut moved).unwrap());

        let parent_bytes = cow.read_page(parent).unwrap().to_vec();
        let view = PageView::new(&parent_bytes, parent, PS).unwrap();
        let c = view.index_leaf_cell(0).unwrap();
        assert_eq!(c.overflow_page, Some(dest), "the index-leaf overflow head now points at dest");
        assert_eq!(c.local_payload, &local[..], "the inline key payload is intact");
        assert_eq!(cow.read_page(dest).unwrap(), &p5[..], "dest is a byte copy of the moved page");
        cow.rollback().unwrap();
    }

    #[test]
    fn relocate_rewrites_overflow_head_in_index_interior_cell() {
        // Page 3 is an INDEX-interior page whose single cell's key spills onto page 5. An
        // index-interior cell prefixes the payload with a 4-byte left-child pointer, so the
        // overflow head sits at `cp + 4 + a` (left child + payload-len varint) + the inline
        // key length — the `+4` that neither leaf arm has. The right-most pointer (8) and the
        // left child (7) must stay put. The ptrmap types page 5 as OVERFLOW1 with parent 3.
        let (from, dest, parent) = (5u32, 6u32, 3u32);
        let total = 2000u64;
        let split = payload_split(PS, total, CellKind::Index);
        assert!(split.has_overflow, "2000 bytes must spill on a 512-byte index page");
        let local = vec![0x5Au8; split.local];
        let mut cell = Vec::new();
        encode_index_interior_cell(7, total, &local, Some(from), &mut cell);
        let mut b = PageBuilder::new(PS, PS, parent, PageType::InteriorIndex);
        b.set_right_most_pointer(8);
        assert!(b.add_cell(&cell), "the spilling index-interior cell must fit on the parent page");
        let p3 = b.finish();
        let p5 = vec![0xABu8; PS];

        let mut cow = staged(vec![
            zeros(),
            ptrmap_page(&[(from, PTRMAP_OVERFLOW1, parent)]),
            p3,
            zeros(),
            p5.clone(),
            zeros(),
        ]);
        let mut moved = HashMap::new();
        assert!(!relocate_page(&mut cow, from, dest, PS, &mut moved).unwrap());

        let parent_bytes = cow.read_page(parent).unwrap().to_vec();
        let view = PageView::new(&parent_bytes, parent, PS).unwrap();
        let c = view.index_interior_cell(0).unwrap();
        assert_eq!(c.overflow_page, Some(dest), "the index-interior overflow head now points at dest");
        assert_eq!(c.left_child, 7, "the cell's left child is untouched");
        assert_eq!(c.local_payload, &local[..], "the inline key payload is intact");
        assert_eq!(view.right_most_pointer(), Some(8), "the right-most pointer is untouched");
        assert_eq!(cow.read_page(dest).unwrap(), &p5[..], "dest is a byte copy of the moved page");
        cow.rollback().unwrap();
    }

    // ---- relocate_page: BTREE via the interior right-most pointer ----

    #[test]
    fn relocate_rewrites_interior_right_most_child() {
        // Page 3 is an interior table page whose RIGHT-MOST child is page 5. The ptrmap
        // types page 5 as a non-root b-tree with parent 3, so relocation rewrites the
        // right-most pointer (interior header offset + 8) from 5 to 6.
        let (from, dest, parent) = (5u32, 6u32, 3u32);
        let mut b = PageBuilder::new(PS, PS, parent, PageType::InteriorTable);
        b.set_right_most_pointer(from);
        let p3 = b.finish();
        let p5 = vec![0xABu8; PS];

        let mut cow = staged(vec![
            zeros(),
            ptrmap_page(&[(from, PTRMAP_BTREE, parent)]),
            p3,
            zeros(),
            p5.clone(),
            zeros(),
        ]);
        let mut moved = HashMap::new();
        assert!(!relocate_page(&mut cow, from, dest, PS, &mut moved).unwrap());

        let parent_bytes = cow.read_page(parent).unwrap().to_vec();
        let view = PageView::new(&parent_bytes, parent, PS).unwrap();
        assert_eq!(view.right_most_pointer(), Some(dest), "right-most child now points at dest");
        assert_eq!(cow.read_page(dest).unwrap(), &p5[..], "dest is a byte copy of the moved page");
        cow.rollback().unwrap();
    }

    // ---- relocate_page: BTREE via a cell's left-child pointer ----

    #[test]
    fn relocate_rewrites_interior_cell_left_child() {
        // Page 3 is an interior table page whose CELL left-child is page 5 (its right-most
        // is a different page, 7, which must stay put). Relocation rewrites the 4-byte
        // left-child pointer at the cell's start from 5 to 6 and leaves right-most alone.
        let (from, dest, parent) = (5u32, 6u32, 3u32);
        let mut b = PageBuilder::new(PS, PS, parent, PageType::InteriorTable);
        b.set_right_most_pointer(7);
        let mut cell = Vec::new();
        encode_table_interior_cell(from, 100, &mut cell);
        assert!(b.add_cell(&cell), "the interior cell must fit");
        let p3 = b.finish();
        let p5 = vec![0xABu8; PS];

        let mut cow = staged(vec![
            zeros(),
            ptrmap_page(&[(from, PTRMAP_BTREE, parent)]),
            p3,
            zeros(),
            p5.clone(),
            zeros(),
        ]);
        let mut moved = HashMap::new();
        assert!(!relocate_page(&mut cow, from, dest, PS, &mut moved).unwrap());

        let parent_bytes = cow.read_page(parent).unwrap().to_vec();
        let view = PageView::new(&parent_bytes, parent, PS).unwrap();
        assert_eq!(view.table_interior_cell(0).unwrap().left_child, dest, "left child -> dest");
        assert_eq!(view.right_most_pointer(), Some(7), "the other child pointer is untouched");
        assert_eq!(cow.read_page(dest).unwrap(), &p5[..]);
        cow.rollback().unwrap();
    }

    // ---- relocate_page: BTREE via an INDEX-interior parent ----
    //
    // Mirror the two InteriorTable tests above but for an index-interior parent (page type
    // 0x02), so `rewrite_child_pointer`'s separate `PageType::InteriorIndex` arm runs (it
    // parses cells via `index_interior_cell`). The child pointer it rewrites — a cell's left
    // child, or the right-most pointer — sits at the SAME offset in both interior layouts, so
    // these pin the index arm's EXISTENCE and its value-exact rewrite (a deleted/merged arm
    // falls to the `other =>` error and `unwrap` panics), NOT a distinct parse offset; the
    // genuinely differing per-page-type offset math is covered by the index overflow-head
    // tests above. Both are planner-independent, unlike the facade-level index test (a
    // table-scan plan could satisfy it without ever reading a rewritten index pointer).

    #[test]
    fn relocate_rewrites_index_interior_right_most_child() {
        // Page 3 is an interior INDEX page whose RIGHT-MOST child is page 5, plus one
        // non-matching key cell (left-child 7) that must stay put — so this also proves
        // the index-cell parse in the loop does not false-match before falling through
        // to the right-most pointer. The ptrmap types page 5 as a non-root b-tree with
        // parent 3, so relocation rewrites the right-most pointer (interior header
        // offset + 8) from 5 to 6.
        let (from, dest, parent) = (5u32, 6u32, 3u32);
        let key = encode_record(&[Value::Integer(42)]);
        let mut b = PageBuilder::new(PS, PS, parent, PageType::InteriorIndex);
        b.set_right_most_pointer(from);
        let mut cell = Vec::new();
        encode_index_interior_cell(7, key.len() as u64, &key, None, &mut cell);
        assert!(b.add_cell(&cell), "the index-interior cell must fit");
        let p3 = b.finish();
        let p5 = vec![0xABu8; PS];

        let mut cow = staged(vec![
            zeros(),
            ptrmap_page(&[(from, PTRMAP_BTREE, parent)]),
            p3,
            zeros(),
            p5.clone(),
            zeros(),
        ]);
        let mut moved = HashMap::new();
        assert!(!relocate_page(&mut cow, from, dest, PS, &mut moved).unwrap());

        let parent_bytes = cow.read_page(parent).unwrap().to_vec();
        let view = PageView::new(&parent_bytes, parent, PS).unwrap();
        assert_eq!(view.right_most_pointer(), Some(dest), "right-most child now points at dest");
        let c = view.index_interior_cell(0).unwrap();
        assert_eq!(c.left_child, 7, "the non-matching cell's left child is untouched");
        assert_eq!(c.local_payload, key.as_slice(), "the index key payload is intact");
        assert_eq!(cow.read_page(dest).unwrap(), &p5[..], "dest is a byte copy of the moved page");
        cow.rollback().unwrap();
    }

    #[test]
    fn relocate_rewrites_index_interior_cell_left_child() {
        // Page 3 is an interior INDEX page whose CELL left-child is page 5 (its right-most
        // is a different page, 7, which must stay put). Relocation rewrites the 4-byte
        // left-child pointer at the cell's start from 5 to 6, leaves right-most alone, and
        // leaves the key payload after the pointer intact (a wrong write offset would corrupt it).
        let (from, dest, parent) = (5u32, 6u32, 3u32);
        let key = encode_record(&[Value::Integer(42)]);
        let mut b = PageBuilder::new(PS, PS, parent, PageType::InteriorIndex);
        b.set_right_most_pointer(7);
        let mut cell = Vec::new();
        encode_index_interior_cell(from, key.len() as u64, &key, None, &mut cell);
        assert!(b.add_cell(&cell), "the index-interior cell must fit");
        let p3 = b.finish();
        let p5 = vec![0xABu8; PS];

        let mut cow = staged(vec![
            zeros(),
            ptrmap_page(&[(from, PTRMAP_BTREE, parent)]),
            p3,
            zeros(),
            p5.clone(),
            zeros(),
        ]);
        let mut moved = HashMap::new();
        assert!(!relocate_page(&mut cow, from, dest, PS, &mut moved).unwrap());

        let parent_bytes = cow.read_page(parent).unwrap().to_vec();
        let view = PageView::new(&parent_bytes, parent, PS).unwrap();
        let c = view.index_interior_cell(0).unwrap();
        assert_eq!(c.left_child, dest, "left child -> dest");
        assert_eq!(c.local_payload, key.as_slice(), "the index key payload after the pointer is intact");
        assert_eq!(view.right_most_pointer(), Some(7), "the other child pointer is untouched");
        assert_eq!(cow.read_page(dest).unwrap(), &p5[..]);
        cow.rollback().unwrap();
    }

    // ---- relocate_page: ROOTPAGE (rewrite sqlite_schema.rootpage in place) ----

    #[test]
    fn relocate_root_rewrites_schema_rootpage() {
        // Page 5 is a table root named by the schema on page 1 (rootpage = 5). Relocating
        // it DOWN to page 3 (dest < from, so the new value fits the same serial width) must
        // rewrite the schema row's rootpage column in place and report a root move.
        let (from, dest) = (5u32, 3u32);
        let p1 = schema_leaf_naming_root(from as i64);
        let p5 = vec![0xABu8; PS];

        let mut cow = staged(vec![
            p1,
            ptrmap_page(&[(from, PTRMAP_ROOTPAGE, 0)]),
            zeros(), // page 3 = dest
            zeros(), // page 4
            p5.clone(), // page 5 = from
        ]);
        let mut moved = HashMap::new();
        let root_moved = relocate_page(&mut cow, from, dest, PS, &mut moved).unwrap();

        assert!(root_moved, "relocating a table root reports root_moved = true");
        assert_eq!(
            collect_roots(&cow, PS).unwrap(),
            vec![dest],
            "the schema now names the relocated root {dest}"
        );
        assert_eq!(cow.read_page(dest).unwrap(), &p5[..], "dest is a byte copy of the moved root");
        cow.rollback().unwrap();
    }

    // ---- relocate_page: fail-closed declines (never a silent mis-relocation) ----

    #[test]
    fn relocate_declines_on_unexpected_ptrmap_type() {
        // A ptrmap type of 9 is not a real §1.8 entry kind; relocation must decline rather
        // than guess which pointer to rewrite.
        let (from, dest) = (5u32, 6u32);
        let mut cow = staged(vec![
            zeros(),
            ptrmap_page(&[(from, 9, 0)]),
            zeros(),
            zeros(),
            vec![0xABu8; PS],
            zeros(),
        ]);
        let mut moved = HashMap::new();
        let err = relocate_page(&mut cow, from, dest, PS, &mut moved).unwrap_err();
        assert!(
            matches!(&err, Error::Format(m) if m.contains("unexpected ptrmap type")),
            "unexpected error: {err:?}"
        );
        assert!(moved.is_empty(), "a declined relocation records no move");
        cow.rollback().unwrap();
    }

    #[test]
    fn relocate_declines_when_parent_pointer_is_missing() {
        // The ptrmap types page 5 as a b-tree child of page 3, but page 3 names neither 5
        // (its cell child is 7, its right-most is 8): a corrupt back-pointer must decline.
        let (from, dest, parent) = (5u32, 6u32, 3u32);
        let mut b = PageBuilder::new(PS, PS, parent, PageType::InteriorTable);
        b.set_right_most_pointer(8);
        let mut cell = Vec::new();
        encode_table_interior_cell(7, 100, &mut cell);
        assert!(b.add_cell(&cell));
        let p3 = b.finish();

        let mut cow = staged(vec![
            zeros(),
            ptrmap_page(&[(from, PTRMAP_BTREE, parent)]),
            p3,
            zeros(),
            vec![0xABu8; PS],
            zeros(),
        ]);
        let mut moved = HashMap::new();
        let err = relocate_page(&mut cow, from, dest, PS, &mut moved).unwrap_err();
        assert!(
            matches!(&err, Error::Format(m) if m.contains("no child pointer equal to")),
            "unexpected error: {err:?}"
        );
        cow.rollback().unwrap();
    }

    // ---- resolve_moved: follow the relocation chain, bounded against cycles ----

    #[test]
    fn resolve_moved_follows_chain_then_stops() {
        let mut moved = HashMap::new();
        moved.insert(5u32, 6u32);
        moved.insert(6u32, 7u32);
        assert_eq!(resolve_moved(&moved, 5), 7, "5 -> 6 -> 7 (terminal)");
        assert_eq!(resolve_moved(&moved, 9), 9, "an unmapped page resolves to itself");
    }

    #[test]
    fn resolve_moved_is_bounded_under_a_cycle() {
        // A corrupt cycle must terminate via the step bound rather than loop forever.
        let mut moved = HashMap::new();
        moved.insert(1u32, 2u32);
        moved.insert(2u32, 1u32);
        let _ = resolve_moved(&moved, 1); // returns (value unspecified); the point is it halts
    }

    // ---- verify_forest: accept a consistent forest, decline a broken one ----

    #[test]
    fn verify_forest_accepts_a_consistent_forest() {
        // Page 1 = empty schema (no tables), page 2 = ptrmap, page 3 = free. Every page in
        // 2..=3 is accounted for (ptrmap / free), so the forest verifies.
        let cow = staged(vec![
            PageBuilder::new(PS, PS, 1, PageType::LeafTable).finish(),
            zeros(),
            zeros(),
        ]);
        let free: BTreeSet<PageId> = [3].into_iter().collect();
        assert!(verify_forest(&cow, 3, PS, &free).is_ok());
    }

    #[test]
    fn verify_forest_declines_an_unaccounted_page() {
        // Page 3 is neither reachable, free, nor a ptrmap position: an unaccounted page
        // must make verify decline rather than truncate over it.
        let cow = staged(vec![
            PageBuilder::new(PS, PS, 1, PageType::LeafTable).finish(),
            zeros(),
            zeros(),
        ]);
        let free = BTreeSet::new();
        let err = verify_forest(&cow, 3, PS, &free).unwrap_err();
        assert!(
            matches!(&err, Error::Format(m) if m.contains("unaccounted for")),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn verify_forest_declines_a_live_page_past_the_target() {
        // Page 1's schema names table root page 3, an interior page whose right-most child
        // is page 4. Truncating to 3 would strand page 4 (a live, reachable page) past the
        // new end — the signature of a relocation that missed a pointer. Verify must decline.
        let p3 = {
            let mut b = PageBuilder::new(PS, PS, 3, PageType::InteriorTable);
            b.set_right_most_pointer(4);
            b.finish()
        };
        let cow = staged(vec![
            schema_leaf_naming_root(3),
            ptrmap_page(&[(3, PTRMAP_ROOTPAGE, 0), (4, PTRMAP_BTREE, 3)]),
            p3,
            PageBuilder::new(PS, PS, 4, PageType::LeafTable).finish(),
        ]);
        let free = BTreeSet::new();
        let err = verify_forest(&cow, 3, PS, &free).unwrap_err();
        assert!(
            matches!(&err, Error::Format(m) if m.contains("past the truncation point")),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn verify_forest_declines_a_reachable_page_on_the_freelist() {
        // Page 3 is reachable (the table root named by the schema) but also listed free:
        // a page cannot be both, so verify declines.
        let cow = staged(vec![
            schema_leaf_naming_root(3),
            ptrmap_page(&[(3, PTRMAP_ROOTPAGE, 0)]),
            PageBuilder::new(PS, PS, 3, PageType::LeafTable).finish(),
        ]);
        let free: BTreeSet<PageId> = [3].into_iter().collect();
        let err = verify_forest(&cow, 3, PS, &free).unwrap_err();
        assert!(
            matches!(&err, Error::Format(m) if m.contains("both reachable and on the freelist")),
            "unexpected error: {err:?}"
        );
    }

    // ---- reclaim: the whole tail-inward loop (relocate a live tail + drop a free page) --

    #[test]
    fn reclaim_relocates_live_tail_into_interior_hole_then_truncates() {
        // File [1: header+schema(root=3), 2: ptrmap, 3: interior root(right-most=6),
        //       4: freelist trunk(leaf 5), 5: free leaf, 6: leaf child with two rows].
        // FULL reclaim must: relocate live page 6 DOWN into free slot 4 (rewriting root 3's
        // right-most 6 -> 4), drop free page 5, clear the freelist, verify, and truncate
        // 6 -> 4. Two free pages are reclaimed; the rows ride along on the relocated page.
        let mut p1 = schema_leaf_naming_root(3);
        write_header(&mut p1, /*largest_root*/ 3, /*incremental*/ 0, /*trunk*/ 4, /*flc*/ 2);
        let p3 = {
            let mut b = PageBuilder::new(PS, PS, 3, PageType::InteriorTable);
            b.set_right_most_pointer(6);
            b.finish()
        };
        let mut p4 = zeros();
        write_trunk(&mut p4, 0, &[5], PS).unwrap(); // trunk: no next, one leaf (page 5)
        let p6 = int_leaf(6, &[(1, 100), (2, 200)]);

        let mut cow = staged(vec![
            p1,
            ptrmap_page(&[
                (3, PTRMAP_ROOTPAGE, 0),
                (4, PTRMAP_FREEPAGE, 0),
                (5, PTRMAP_FREEPAGE, 0),
                (6, PTRMAP_BTREE, 3),
            ]),
            p3,
            p4,
            zeros(), // page 5: free leaf
            p6,
        ]);

        let outcome = reclaim(&mut cow, None).unwrap();

        assert_eq!(outcome.reclaimed, 2, "two free pages (interior 4 + tail 5) reclaimed");
        assert!(!outcome.root_moved, "a non-root child moved, so no schema rootpage changed");
        assert_eq!(cow.page_count().unwrap(), 4, "6 pages -> 4 after relocate + truncate");

        let h = read_header(&cow).unwrap().expect("valid header");
        assert_eq!(h.freelist_count, 0, "the freelist is empty after FULL reclaim");
        assert_eq!(h.first_freelist_trunk, 0, "no trunk remains");

        // The forest is consistent for the new size, and the child moved 6 -> 4.
        verify_forest(&cow, 4, PS, &BTreeSet::new()).expect("forest verifies at the new size");
        let mut reachable = BTreeSet::new();
        reachable.insert(1u32);
        walk_reachable(&cow, 1, PS, &mut reachable).unwrap();
        for root in collect_roots(&cow, PS).unwrap() {
            walk_reachable(&cow, root, PS, &mut reachable).unwrap();
        }
        assert!(reachable.contains(&4), "the child now lives on relocated page 4");
        assert!(!reachable.contains(&6), "nothing references the old tail page 6");

        // The rows survive verbatim on the relocated page.
        let p4now = cow.read_page(4).unwrap().to_vec();
        let view = PageView::new(&p4now, 4, PS).unwrap();
        assert_eq!(view.cell_count(), 2, "both rows rode along with the relocation");
        let r0 = view.table_leaf_cell(0).unwrap();
        let r1 = view.table_leaf_cell(1).unwrap();
        assert_eq!(r0.rowid, 1);
        assert_one_int(r0.local_payload, 100);
        assert_eq!(r1.rowid, 2);
        assert_one_int(r1.local_payload, 200);
        cow.rollback().unwrap();
    }

    #[test]
    fn reclaim_relocates_two_live_pages_across_a_moved_parent() {
        // The `resolve_moved` cross-move (reclaim.rs:170) driven END-TO-END, not in
        // isolation: a single FULL pass relocates TWO live tail pages where the SECOND
        // relocation's ptrmap parent is the FIRST-relocated page. This is reachable by
        // ordinary DML (DELETE frees low pages, then a vacuum relocates tail-inward), and
        // whenever a parent page sits at a HIGHER page number than its child, the parent
        // moves first — so the child's pointer rewrite MUST follow `resolve_moved` to the
        // parent's new home. The existing end-to-end reclaim test relocates one page with an
        // empty `moved` map; only `resolve_moved_*` (isolated) touched a non-empty map.
        //
        // File [1: schema(root=6), 2: ptrmap, 3: freelist trunk(leaf 4), 4: free leaf,
        //       5: C leaf child of P, 6: P interior root (right-most = 5)].
        // reclaim(None) walks the tail down:
        //   * page 6 (P, root) is LIVE -> relocate into lowest free slot 3; rewrite schema
        //     rootpage 6 -> 3 and record moved[6] = 3.
        //   * page 5 (C, child) is LIVE -> relocate into free slot 4; its ptrmap parent is 6,
        //     which ALREADY moved, so `resolve_moved(6) == 3` and the child pointer must be
        //     rewritten in page 3 (P's NEW home): right-most 5 -> 4.
        // Then the freelist empties and the file truncates 6 -> 4.
        let mut p1 = schema_leaf_naming_root(6);
        write_header(&mut p1, /*largest_root*/ 6, /*incremental*/ 0, /*trunk*/ 3, /*flc*/ 2);
        let mut p3 = zeros();
        write_trunk(&mut p3, 0, &[4], PS).unwrap(); // trunk: no next, one leaf (page 4)
        let p5 = int_leaf(5, &[(1, 700)]); // C: the child page, one row
        let p6 = {
            let mut b = PageBuilder::new(PS, PS, 6, PageType::InteriorTable);
            b.set_right_most_pointer(5); // P (root) parents C at page 5
            b.finish()
        };

        let mut cow = staged(vec![
            p1,
            ptrmap_page(&[
                (3, PTRMAP_FREEPAGE, 0),
                (4, PTRMAP_FREEPAGE, 0),
                (5, PTRMAP_BTREE, 6), // C's parent is the root P at page 6
                (6, PTRMAP_ROOTPAGE, 0),
            ]),
            p3,
            zeros(), // page 4: free leaf
            p5,
            p6,
        ]);

        let outcome = reclaim(&mut cow, None).unwrap();

        assert_eq!(outcome.reclaimed, 2, "both free pages (trunk 3 + leaf 4) are reclaimed");
        assert!(outcome.root_moved, "the interior root P moved, so a schema rootpage changed");
        assert_eq!(cow.page_count().unwrap(), 4, "6 pages -> 4 after two relocations + truncate");

        let h = read_header(&cow).unwrap().expect("valid header");
        assert_eq!(h.freelist_count, 0, "FULL reclaim empties the freelist");
        assert_eq!(h.first_freelist_trunk, 0, "no trunk remains");

        // The root moved 6 -> 3 (schema rewrite).
        assert_eq!(collect_roots(&cow, PS).unwrap(), vec![3], "root P relocated 6 -> 3");

        // THE CROSS-MOVE ASSERTION: P now lives at page 3, and the child pointer that named
        // C was rewritten IN P'S NEW HOME (page 3) to C's new home (page 4). This is only
        // correct if `resolve_moved(6)` returned 3; a broken lookup would rewrite page 6
        // (about to be truncated) and leave page 3's right-most at 5 -> `verify_forest`
        // inside `reclaim` would have rejected the pass (so `.unwrap()` above would panic).
        let p3now = cow.read_page(3).unwrap().to_vec();
        let v3 = PageView::new(&p3now, 3, PS).unwrap();
        assert_eq!(v3.page_type(), PageType::InteriorTable, "P (interior root) relocated to page 3");
        assert_eq!(
            v3.right_most_pointer(),
            Some(4),
            "C's back-pointer was rewritten 5 -> 4 in P's NEW home via resolve_moved"
        );

        // The child's row rode along verbatim onto its new home page 4.
        let p4now = cow.read_page(4).unwrap().to_vec();
        let v4 = PageView::new(&p4now, 4, PS).unwrap();
        assert_eq!(v4.cell_count(), 1, "C's single row survived the relocation");
        let row = v4.table_leaf_cell(0).unwrap();
        assert_eq!(row.rowid, 1);
        assert_one_int(row.local_payload, 700);

        // The forest verifies at the new size and reaches the relocated child, not the old tail.
        verify_forest(&cow, 4, PS, &BTreeSet::new()).expect("forest verifies at the new size");
        let mut reachable = BTreeSet::new();
        reachable.insert(1u32);
        walk_reachable(&cow, 1, PS, &mut reachable).unwrap();
        for root in collect_roots(&cow, PS).unwrap() {
            walk_reachable(&cow, root, PS, &mut reachable).unwrap();
        }
        assert!(reachable.contains(&4), "the child now lives on relocated page 4");
        assert!(!reachable.contains(&5), "nothing references the old child page 5");
        assert!(!reachable.contains(&6), "nothing references the old root page 6");
        cow.rollback().unwrap();
    }

    #[test]
    fn reclaim_stops_after_the_budget() {
        // File [1: header+schema(root=3), 2: ptrmap, 3: live leaf root, 4: freelist trunk
        //       (leaves 5, 6), 5: free, 6: free]. Free pages 4, 5, 6 sit at the TAIL, so a
        // budget of 2 drops exactly the two free tail pages (6 then 5) with no relocation,
        // leaving page 4 free — proving `incremental_vacuum(N)` honors its cap exactly.
        let mut p1 = schema_leaf_naming_root(3);
        write_header(&mut p1, /*largest_root*/ 3, /*incremental*/ 1, /*trunk*/ 4, /*flc*/ 3);
        let p3 = int_leaf(3, &[(1, 111)]);
        let mut p4 = zeros();
        write_trunk(&mut p4, 0, &[5, 6], PS).unwrap();

        let mut cow = staged(vec![
            p1,
            ptrmap_page(&[
                (3, PTRMAP_ROOTPAGE, 0),
                (4, PTRMAP_FREEPAGE, 0),
                (5, PTRMAP_FREEPAGE, 0),
                (6, PTRMAP_FREEPAGE, 0),
            ]),
            p3,
            p4,
            zeros(),
            zeros(),
        ]);

        let outcome = reclaim(&mut cow, Some(2)).unwrap();
        assert_eq!(outcome.reclaimed, 2, "the budget of 2 reclaims exactly two free tail pages");
        assert!(!outcome.root_moved, "dropping free tail pages moves no root");
        assert_eq!(cow.page_count().unwrap(), 4, "6 pages -> 4 (two free tail pages dropped)");
        let h = read_header(&cow).unwrap().expect("valid header");
        assert_eq!(h.freelist_count, 1, "one free page (the old trunk page 4) remains");
        // The live row is untouched by a budgeted reclaim.
        let p3now = cow.read_page(3).unwrap().to_vec();
        let view = PageView::new(&p3now, 3, PS).unwrap();
        assert_one_int(view.table_leaf_cell(0).unwrap().local_payload, 111);
        cow.rollback().unwrap();
    }

    #[test]
    fn reclaim_is_a_noop_without_ptrmap() {
        // A non-auto_vacuum database (offset 52 == 0) is never compacted, even with a
        // populated freelist: reclaim returns the default outcome and changes nothing.
        let mut p1 = schema_leaf_naming_root(3);
        write_header(&mut p1, /*largest_root*/ 0, /*incremental*/ 0, /*trunk*/ 4, /*flc*/ 1);
        let mut p4 = zeros();
        write_trunk(&mut p4, 0, &[], PS).unwrap();

        let mut cow = staged(vec![
            p1,
            zeros(),
            PageBuilder::new(PS, PS, 3, PageType::LeafTable).finish(),
            p4,
        ]);
        let outcome = reclaim(&mut cow, None).unwrap();
        assert_eq!(outcome, VacuumOutcome::default(), "no ptrmap -> no reclamation");
        assert_eq!(cow.page_count().unwrap(), 4, "page count is unchanged");
        cow.rollback().unwrap();
    }
}
