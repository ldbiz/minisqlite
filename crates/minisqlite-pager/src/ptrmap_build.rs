//! The auto_vacuum / incremental_vacuum WRITE side: rebuilding the pointer-map
//! (ptrmap) pages and the page-1 "largest root b-tree page" field (offset 52) so an
//! auto_vacuum database this engine writes is a file real `sqlite3` reads back
//! (fileformat2 §1.8 + §1.3.12). The READ side (position math + entry codec) lives in
//! [`minisqlite_fileformat::ptrmap`]; this module is its writer.
//!
//! ## Why derive-at-commit, not incremental
//! A ptrmap entry is a pure function of the b-tree forest: every page's role (root /
//! interior-or-leaf b-tree / first-or-later overflow / freelist) and its parent are
//! determined by who points at it. Rather than thread parent/child bookkeeping through
//! every allocation, split, and free — a deep, error-prone hook into the hot b-tree
//! paths — this module DERIVES the whole ptrmap at [`finalize`], called once per commit
//! from [`Cow::commit`](crate::cow::Cow). Derivation-by-traversal cannot drift from the
//! actual structure the way incremental counters can, which is the correctness argument
//! for paying an O(live pages) walk per auto_vacuum commit. (The cost lands ONLY on
//! auto_vacuum databases: [`finalize`] is a single page-1 header read that returns
//! immediately when offset 52 is zero, so the default non-vacuum path is byte-identical
//! and pays nothing. Incremental maintenance is the eventual optimization, deferred.)
//!
//! ## What `finalize` guarantees
//! After it runs on an auto_vacuum database (offset 52 non-zero):
//! - every b-tree root page precedes every non-root / overflow / freelist page (§1.8
//!   roots-first), relocating a root to the front if a `CREATE INDEX`/`CREATE TABLE` left
//!   it at the tail (see [`crate::roots_first`]), and
//! - every ptrmap page at the position [`is_ptrmap_page`](minisqlite_fileformat::ptrmap)
//!   computes holds the correct 5-byte entry for each data page it covers (§1.8), and
//! - offset 52 records the largest root b-tree page (§1.3.12).
//! No b-tree pointer points at a ptrmap page, because [`crate::alloc`] skips ptrmap
//! positions when growing an auto_vacuum database — so the forest walk here never
//! encounters one, and a pointer-driven reader (schema load, table/index cursors) never
//! reads one either.
//!
//! ## Every allocated page is classified (no unrepresentable pages)
//! §1.8 can only describe a page as root / b-tree / overflow / freelist — there is no
//! "allocated but unreferenced" entry. [`finalize`] therefore guarantees every
//! allocated page falls into one of those buckets before it writes the ptrmap: the
//! forest walk classifies the reachable pages, and [`sweep_leaked_pages`] puts any
//! remaining allocated page (a leak) onto the freelist. The one leak source today is
//! `DROP TABLE`, which unlinks a table's schema row without freeing its b-tree pages;
//! the sweep turns those into legitimate FREEPAGE entries (and reclaimable space)
//! rather than leaving a page the ptrmap cannot represent. The forest walk fails closed
//! on a corrupt page (see [`walk_tree`]) so the sweep can never mistake a live page for
//! a leak. DDL/DML otherwise run as one wrapped write transaction, so finalize sees the
//! consistent end-of-statement state, never a page allocated-but-not-yet-linked
//! mid-statement.
//!
//! ## Companion: free-page reclamation (compaction)
//! This module makes an auto_vacuum database STRUCTURALLY VALID (correct ptrmap pages +
//! header). The free-page RECLAMATION half — FULL's truncate-to-zero-free-pages at every
//! commit and `PRAGMA incremental_vacuum(N)` — lives in [`crate::reclaim`], which relocates
//! the tail's live pages into freed slots (rewriting the one pointer each ptrmap
//! back-pointer names) and truncates the file. Reclamation leaves the FOREST consistent and
//! then calls [`finalize`] to re-derive the ptrmap for the smaller file, so the two stay
//! cleanly separated: this module owns "the ptrmap is correct for the current forest",
//! [`crate::reclaim`] owns "the file is as small as the mode requires".
//!
//! ## Known cost: derive-at-commit is O(live pages) per commit (deferred optimization)
//! Because [`finalize`] re-derives the WHOLE ptrmap each commit, an auto_vacuum database
//! pays an O(live pages) forest walk on every write (the non-vacuum path is untouched —
//! see above). For a large auto_vacuum database, or a long run of per-statement autocommit
//! writes, that is super-linear in aggregate. The correctness-first reason is documented
//! above (derivation cannot drift); the eventual fix is incremental maintenance — write
//! just the affected 5-byte entries as the allocator hands out / frees / relocates a page
//! (sqlite's `ptrmapPut`), which requires threading parent bookkeeping through the b-tree
//! hot paths and is left as a follow-up so it does not destabilize the validated derivation.

use std::collections::{BTreeMap, HashSet};

use minisqlite_fileformat::overflow::PageSource;
use minisqlite_fileformat::ptrmap::{
    ptrmap_entries_per_page, ptrmap_stride, put_ptrmap_entry, PTRMAP_BTREE, PTRMAP_ENTRY_SIZE,
    PTRMAP_FREEPAGE, PTRMAP_OVERFLOW1, PTRMAP_OVERFLOW2, PTRMAP_ROOTPAGE,
};
use minisqlite_fileformat::{
    assemble_payload, parse_overflow_page, parse_trunk, read_serial_value, DatabaseHeader,
    PageType, PageView, RecordCursor, TableLeafCell, HEADER_SIZE,
};
use minisqlite_types::{Error, Result, Value};

use crate::cow::Cow;
use crate::store::PageStore;
use crate::PageId;

/// The column index of `rootpage` in a `sqlite_schema` row (`type, name, tbl_name,
/// rootpage, sql`). It is an integer that precedes the potentially-large `sql` text,
/// so it is always in a cell's inline payload; still, [`schema_rootpage`] assembles
/// the overflow chain when one is present so a pathologically long name/type can never
/// hide the root.
const SCHEMA_ROOTPAGE_COL: usize = 3;

/// A ptrmap back-pointer: the entry `type` byte (§1.8) and the `parent` page number
/// (0 for a root or freelist page).
type Entry = (u8, u32);

/// Rebuild the pointer-map pages and page-1 offset 52 for the transaction about to
/// commit, first restoring §1.8 roots-first order (see [`crate::roots_first`]) if a
/// `CREATE INDEX`/`CREATE TABLE` left a root above existing non-root pages. A NO-OP (one
/// page-1 header read) unless the database is in auto_vacuum / incremental_vacuum mode
/// (offset 52 non-zero), so the default path is byte-identical.
///
/// Called from [`Cow::commit`](crate::cow::Cow) while the write transaction is still
/// active, so the ptrmap pages and the patched page 1 are staged into the same overlay
/// and committed atomically with the caller's changes.
///
/// Returns `true` iff it RELOCATED a b-tree root to satisfy roots-first (which rewrote a
/// `sqlite_schema.rootpage`). [`Cow::commit`] records that so the engine can reload a
/// cached catalog whose root pages just moved — the same reload the reclaim path performs
/// via [`crate::reclaim::VacuumOutcome::root_moved`]. `false` for the overwhelmingly
/// common commit (non-vacuum, or an auto_vacuum commit whose roots were already first).
pub(crate) fn finalize<S: PageStore>(cow: &mut Cow<S>) -> Result<bool> {
    let count = cow.page_count()?;
    if count < 1 {
        return Ok(false);
    }
    let header = match read_page1_header(cow)? {
        Some(h) if h.uses_ptrmap() => h,
        // Unformatted page 1, or a plain (non-vacuum) database: nothing to maintain.
        _ => return Ok(false),
    };
    let usable = header.usable_size();

    // Gate: a ptrmap is DERIVED from the schema b-tree forest, so there is nothing to
    // derive unless page 1 is a parseable b-tree page. A store with a valid 100-byte
    // header whose b-tree body is not yet formatted (a raw header written directly, or a
    // mid-format store) has no forest — skip rather than error the caller's commit. Every
    // engine-created auto_vacuum database reaches commit with page 1 already a formatted
    // schema b-tree, so this gate only ever fires outside the real engine path.
    {
        let page1 = cow.read_page(1)?.to_vec();
        if PageView::new(&page1, 1, usable).is_err() {
            return Ok(false);
        }
    }

    // Derive the ptrmap back-pointers from the forest, first restoring §1.8 ROOTS-FIRST if
    // a CREATE INDEX / second CREATE TABLE left a root above existing non-root pages (see
    // [`crate::roots_first`]). `enforce` establishes the invariant in a single pass, so the
    // loop runs at most twice: derive → relocate → re-derive (which confirms roots-first and
    // relocates nothing). A relocation that left the forest inconsistent makes the re-derive
    // fail CLOSED (`walk_tree` / `write_ptrmap_pages` error), rolling the whole commit back
    // to the pre-relocation, fully-valid state — never a persisted corrupt file.
    let mut guard = 0u32;
    let mut moved_roots = false;
    let (entries, largest_root) = loop {
        let (entries, roots, largest_root) = derive_entries(cow, count, usable)?;
        if !crate::roots_first::enforce(cow, &roots, &entries, count, usable)? {
            break (entries, largest_root);
        }
        // A relocation rewrote at least one `sqlite_schema.rootpage`; report it so the
        // caller reloads its cached catalog for the moved roots.
        moved_roots = true;
        guard += 1;
        if guard > 2 {
            return Err(Error::Format(
                "auto_vacuum: roots-first (§1.8) did not converge after relocation".into(),
            ));
        }
    };

    // Write the ptrmap pages that cover data pages up to `count`, then record offset 52 =
    // the largest root b-tree page (§1.3.12), preserving the incremental-vacuum flag
    // (offset 64) the PRAGMA set.
    write_ptrmap_pages(cow, &entries, count, usable)?;
    set_largest_root(cow, largest_root)?;
    Ok(moved_roots)
}

/// Walk the forest and classify every page in `2..=count` into its §1.8 back-pointer:
/// each table/index root (from `sqlite_schema`) as [`PTRMAP_ROOTPAGE`], their reachable
/// children/overflow via [`walk_tree`], any `DROP`-leaked page swept onto the freelist,
/// and every freelist page as [`PTRMAP_FREEPAGE`]. Returns `(entries, roots, largest
/// root)`. Called once per [`finalize`] loop iteration on the CURRENT (possibly just
/// relocated) forest, so the entries always match the bytes about to be written.
fn derive_entries<S: PageStore>(
    cow: &mut Cow<S>,
    count: PageId,
    usable: usize,
) -> Result<(BTreeMap<PageId, Entry>, Vec<PageId>, PageId)> {
    // 1. Derive a back-pointer entry for every live page by walking the forest.
    let mut entries: BTreeMap<PageId, Entry> = BTreeMap::new();
    let mut visited: HashSet<PageId> = HashSet::new();

    // The schema tree's root is page 1; it contributes its structural child pages and
    // any overflow, and its rows name every table/index root. Page 1 itself gets no
    // ptrmap entry (ptrmap coverage starts at page 3).
    let roots = collect_table_roots(cow, usable)?;
    walk_tree(cow, 1, usable, &mut entries, &mut visited)?;
    let mut largest_root: PageId = 1;
    for r in &roots {
        largest_root = largest_root.max(*r);
        entries.insert(*r, (PTRMAP_ROOTPAGE, 0));
        walk_tree(cow, *r, usable, &mut entries, &mut visited)?;
    }

    // 2. Reclaim any LEAKED page — allocated, but neither reachable from the forest nor
    //    on the freelist — by putting it on the freelist. `DROP TABLE` currently frees
    //    no pages (a documented catalog leak), so without this a dropped root would be
    //    an allocated page with no possible §1.8 classification: `write_ptrmap_pages`
    //    would have to invent an invalid type-0 entry (or, in debug, panic). Sweeping it
    //    onto the freelist makes it a legitimate FREEPAGE, keeps the file structurally
    //    valid for cross-read, AND makes the space reclaimable by compaction. A no-op
    //    (one O(1) balance check) on the overwhelmingly common leak-free commit. Re-read
    //    page 1 first, because a prior roots-first relocation may have moved the header
    //    freelist head/count.
    let header = read_page1_header(cow)?.ok_or_else(|| {
        Error::Format("auto_vacuum: page 1 header became unreadable during finalize".into())
    })?;
    sweep_leaked_pages(cow, &header, count, usable, &entries)?;

    // 3. Freelist pages (trunks + leaves, including anything just swept) are FREEPAGE
    //    entries with no parent. Re-read page 1 because the sweep may have changed the
    //    freelist head/count.
    let header = read_page1_header(cow)?.ok_or_else(|| {
        Error::Format("auto_vacuum: page 1 header became unreadable during finalize".into())
    })?;
    record_freelist(cow, &header, count, &mut entries)?;

    Ok((entries, roots, largest_root))
}

/// Reclaim leaked pages onto the freelist so every allocated page is classifiable as a
/// §1.8 ptrmap entry. A "leak" is a page in `2..=count` that is not a ptrmap position,
/// not reachable from the forest (`live`), and not already on the freelist. Today the
/// only source is `DROP TABLE`, which removes a table's schema row without freeing its
/// b-tree pages; in a healthy forest there are no leaks and this returns after a single
/// O(1) arithmetic check, so the common commit pays nothing.
///
/// Correctness rests on [`walk_tree`] failing closed on a corrupt page: if a live
/// subtree were silently dropped from `live`, those pages would look leaked and be
/// wrongly freed. With the fail-closed walk, a page absent from `live` and the freelist
/// is provably unreferenced, so freeing it is safe.
fn sweep_leaked_pages<S: PageStore>(
    cow: &mut Cow<S>,
    header: &DatabaseHeader,
    count: PageId,
    usable: usize,
    live: &BTreeMap<PageId, Entry>,
) -> Result<()> {
    // O(1) balance check: page 1 + ptrmap pages + live pages + freelist pages must
    // account for every page. When they do there is no leak, so the O(count) scan and
    // its per-page freelist lookups are skipped entirely — this is the hot path.
    let accounted =
        1u64 + ptrmap_page_count(count, usable) + live.len() as u64 + header.freelist_count as u64;
    if accounted == count as u64 {
        return Ok(());
    }

    // A discrepancy means at least one page is neither live nor free. Snapshot the
    // current freelist so an already-free page is never double-freed, then free each
    // genuinely-leaked page (in ascending order, deterministically).
    let free_set = freelist_set(cow, header, count)?;
    let mut leaked = Vec::new();
    for p in 2..=count {
        if minisqlite_fileformat::is_ptrmap_page(usable, p) {
            continue;
        }
        if live.contains_key(&p) || free_set.contains(&p) {
            continue;
        }
        leaked.push(p);
    }
    for p in leaked {
        crate::alloc::free_in_txn(cow, p)?;
    }
    Ok(())
}

/// The number of pointer-map pages in a file of `count` pages: the §1.8 positions
/// `2, 2+stride, 2+2*stride, …` that are `<= count`. Used only for the O(1) leak
/// balance check.
fn ptrmap_page_count(count: PageId, usable: usize) -> u64 {
    if count < 2 {
        return 0;
    }
    let stride = ptrmap_stride(usable);
    (count as u64 - 2) / stride + 1
}

/// The set of pages currently on the freelist (trunk pages and their leaves), bounded
/// by `count` so a corrupt trunk cycle cannot loop.
fn freelist_set<S: PageStore>(
    cow: &Cow<S>,
    header: &DatabaseHeader,
    count: PageId,
) -> Result<HashSet<PageId>> {
    let mut set = HashSet::new();
    for_each_freelist_page(cow, header, count, |p| {
        set.insert(p);
    })?;
    Ok(set)
}

/// Parse the page-1 header through the transaction overlay. `None` when page 1 is not a
/// valid SQLite header yet (an unformatted / mid-format store), which callers treat as
/// "not auto_vacuum, nothing to do".
fn read_page1_header<S: PageStore>(cow: &Cow<S>) -> Result<Option<DatabaseHeader>> {
    let page1 = cow.read_page(1)?;
    let head: &[u8; HEADER_SIZE] = match page1.get(..HEADER_SIZE).and_then(|s| s.try_into().ok()) {
        Some(h) => h,
        None => return Ok(None),
    };
    Ok(DatabaseHeader::read(head).ok())
}

/// Walk one b-tree (rooted at `root`) and record a ptrmap entry for every page it
/// points at: each child of an interior page is a non-root b-tree page (`PTRMAP_BTREE`,
/// parent = the interior page), and each cell whose payload spilled starts an overflow
/// chain (`PTRMAP_OVERFLOW1` for the first page, `PTRMAP_OVERFLOW2` for the rest). The
/// root's OWN entry is set by the caller (a real root is `PTRMAP_ROOTPAGE`; the schema
/// root page 1 has none).
///
/// Termination is bounded by `visited`: each page is processed at most once, so a
/// corrupt cyclic pointer cannot loop. A page a live pointer reached that does NOT
/// parse as a b-tree page aborts the commit (fail closed) rather than being skipped:
/// silently dropping its subtree would make those live pages look unreferenced, and
/// the leak sweep would then free pages that are actually in use.
fn walk_tree<S: PageStore>(
    cow: &Cow<S>,
    root: PageId,
    usable: usize,
    entries: &mut BTreeMap<PageId, Entry>,
    visited: &mut HashSet<PageId>,
) -> Result<()> {
    if root == 0 {
        return Ok(());
    }
    let mut stack = vec![root];
    while let Some(p) = stack.pop() {
        if !visited.insert(p) {
            continue;
        }
        // Copy the page out so no read borrow of `cow` is held across `walk_overflow`'s
        // own reads. One page at a time keeps peak extra memory to a single page.
        let page = cow.read_page(p)?.to_vec();
        // Fail CLOSED on a page that a live pointer reached but that does not parse as a
        // b-tree page: that is a corrupt forest, and swallowing it here would silently
        // drop the whole subtree from `entries` — which the leak sweep would then treat
        // as garbage and free (freeing LIVE pages). Propagating matches
        // `collect_table_roots`, so a broken forest aborts the commit instead of
        // persisting an invalid ptrmap. (A page never reached by any pointer is simply
        // never walked; this is only about pages a pointer *claims* are b-tree pages.)
        let view = PageView::new(&page, p, usable)?;
        let n = view.cell_count() as usize;
        match view.page_type() {
            PageType::InteriorTable => {
                for i in 0..n {
                    let child = view.table_interior_cell(i)?.left_child;
                    record_child(entries, &mut stack, child, p);
                }
                if let Some(rm) = view.right_most_pointer() {
                    record_child(entries, &mut stack, rm, p);
                }
            }
            PageType::LeafTable => {
                for i in 0..n {
                    let cell = view.table_leaf_cell(i)?;
                    if let Some(ov) = cell.overflow_page {
                        walk_overflow(cow, ov, p, usable, entries, visited)?;
                    }
                }
            }
            PageType::InteriorIndex => {
                for i in 0..n {
                    let cell = view.index_interior_cell(i)?;
                    record_child(entries, &mut stack, cell.left_child, p);
                    if let Some(ov) = cell.overflow_page {
                        walk_overflow(cow, ov, p, usable, entries, visited)?;
                    }
                }
                if let Some(rm) = view.right_most_pointer() {
                    record_child(entries, &mut stack, rm, p);
                }
            }
            PageType::LeafIndex => {
                for i in 0..n {
                    let cell = view.index_leaf_cell(i)?;
                    if let Some(ov) = cell.overflow_page {
                        walk_overflow(cow, ov, p, usable, entries, visited)?;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Record a non-root b-tree child (`PTRMAP_BTREE`, parent = `parent`) and queue it for
/// traversal. A zero pointer (an empty slot) is ignored.
fn record_child(
    entries: &mut BTreeMap<PageId, Entry>,
    stack: &mut Vec<PageId>,
    child: PageId,
    parent: PageId,
) {
    if child != 0 {
        entries.insert(child, (PTRMAP_BTREE, parent));
        stack.push(child);
    }
}

/// Walk an overflow chain starting at `first`, recording the first page as
/// `PTRMAP_OVERFLOW1` (parent = the b-tree page holding the cell) and every later page
/// as `PTRMAP_OVERFLOW2` (parent = the previous overflow page), per §1.8. Bounded by
/// `visited` so a corrupt self-referential chain cannot loop.
fn walk_overflow<S: PageStore>(
    cow: &Cow<S>,
    first: PageId,
    btree_parent: PageId,
    usable: usize,
    entries: &mut BTreeMap<PageId, Entry>,
    visited: &mut HashSet<PageId>,
) -> Result<()> {
    let mut cur = first;
    let mut parent = btree_parent;
    let mut first_link = true;
    while cur != 0 {
        if !visited.insert(cur) {
            break;
        }
        let kind = if first_link { PTRMAP_OVERFLOW1 } else { PTRMAP_OVERFLOW2 };
        entries.insert(cur, (kind, parent));
        // Read only the 4-byte next pointer, then drop the borrow.
        let next = parse_overflow_page(cow.read_page(cur)?, usable)?.next;
        parent = cur;
        cur = next;
        first_link = false;
    }
    Ok(())
}

/// Read every `rootpage` from `sqlite_schema` (root page 1), returning the table/index
/// b-tree roots (rootpage >= 2). Rows with rootpage 0 (views, triggers) contribute no
/// root. Walks the schema TABLE b-tree directly (the pager cannot depend on the b-tree
/// crate's cursor); bounded by `visited`.
fn collect_table_roots<S: PageStore>(cow: &Cow<S>, usable: usize) -> Result<Vec<PageId>> {
    let mut roots = Vec::new();
    let mut visited: HashSet<PageId> = HashSet::new();
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
                    if let Some(root) = schema_rootpage(cow, &cell, usable)? {
                        roots.push(root);
                    }
                }
            }
            // sqlite_schema is always a TABLE b-tree; an index page here is corruption.
            other => {
                return Err(Error::Format(format!(
                    "sqlite_schema page {p} is a {other:?} page, expected a table b-tree"
                )))
            }
        }
    }
    Ok(roots)
}

/// Decode the `rootpage` column of one `sqlite_schema` leaf cell, assembling the
/// overflow chain if the payload spilled. Returns the root page (>= 2) or `None` for a
/// rootpage of 0/NULL (a view or trigger) or an out-of-range value.
fn schema_rootpage<S: PageStore>(
    cow: &Cow<S>,
    cell: &TableLeafCell<'_>,
    usable: usize,
) -> Result<Option<PageId>> {
    let payload = match cell.overflow_page {
        None => cell.local_payload.to_vec(),
        Some(first) => assemble_payload(
            cell.local_payload,
            cell.payload_len as usize,
            first,
            usable,
            &CowSource(cow),
        )?,
    };
    let Some((serial, body)) = RecordCursor::new(&payload).nth(SCHEMA_ROOTPAGE_COL) else {
        return Ok(None);
    };
    match read_serial_value(serial, body) {
        Value::Integer(n) if n >= 2 && n <= PageId::MAX as i64 => Ok(Some(n as PageId)),
        _ => Ok(None),
    }
}

/// A [`PageSource`] over the transaction (overlay-then-committed), so overflow-chain
/// assembly can borrow pages for a rare over-long schema row.
struct CowSource<'a, S: PageStore>(&'a Cow<S>);

impl<S: PageStore> PageSource for CowSource<'_, S> {
    fn page(&self, page_no: u32) -> Result<&[u8]> {
        self.0.read_page(page_no)
    }
}

/// Record every freelist page (trunk pages and their leaves) as a `PTRMAP_FREEPAGE`
/// entry with no parent (§1.8).
fn record_freelist<S: PageStore>(
    cow: &Cow<S>,
    header: &DatabaseHeader,
    count: PageId,
    entries: &mut BTreeMap<PageId, Entry>,
) -> Result<()> {
    for_each_freelist_page(cow, header, count, |p| {
        entries.insert(p, (PTRMAP_FREEPAGE, 0));
    })
}

/// Visit every freelist page (each trunk, then each of its leaves) exactly once, in
/// chain order. Bounded by `count` so a corrupt trunk `next` cycle cannot loop. The
/// single freelist traversal shared by [`record_freelist`] and [`freelist_set`].
fn for_each_freelist_page<S: PageStore>(
    cow: &Cow<S>,
    header: &DatabaseHeader,
    count: PageId,
    mut visit: impl FnMut(PageId),
) -> Result<()> {
    let usable = header.usable_size();
    let mut trunk = header.first_freelist_trunk;
    let mut steps: u64 = 0;
    let limit = count as u64 + 1;
    while trunk != 0 {
        steps += 1;
        if steps > limit {
            return Err(Error::Format("freelist trunk chain longer than the database".into()));
        }
        visit(trunk);
        let (next, leaves) = parse_trunk(cow.read_page(trunk)?, usable)?;
        for leaf in leaves {
            visit(leaf);
        }
        trunk = next;
    }
    Ok(())
}

/// Build and stage each ptrmap page that covers data pages up to `count`. Ptrmap page
/// `B` (page 2, then every stride) holds the entries for data pages `B+1 ..= B+J`; a
/// page is staged only when its bytes actually change, so a commit that leaves the
/// forest unchanged rewrites no ptrmap page.
fn write_ptrmap_pages<S: PageStore>(
    cow: &mut Cow<S>,
    entries: &BTreeMap<PageId, Entry>,
    count: PageId,
    usable: usize,
) -> Result<()> {
    let page_size = cow.page_size() as usize;
    let per_page = ptrmap_entries_per_page(usable) as u32;
    let stride = ptrmap_stride(usable);
    let mut b: PageId = 2;
    while b as u64 <= count as u64 {
        let mut page = vec![0u8; page_size];
        for k in 0..per_page {
            let d = b + 1 + k;
            if d > count {
                break;
            }
            // `d` is never a ptrmap page: the stride window `B+1..=B+J` stops one short
            // of the next ptrmap page `B+J+1`. Every covered page must have a derived
            // entry by now (the forest walk classified the live pages, and
            // `sweep_leaked_pages` put every other allocated page on the freelist). A
            // missing entry means a page is neither reachable nor free — fail CLOSED
            // rather than writing a type-0 byte, which is not one of the five §1.8 entry
            // types (1..=5) and would make the file structurally invalid for cross-read.
            let (kind, parent) = match entries.get(&d) {
                Some(&e) => e,
                None => {
                    return Err(Error::Format(format!(
                        "auto_vacuum: page {d} has no ptrmap back-pointer (unreachable and not \
                         on the freelist); refusing to write an invalid §1.8 type-0 entry"
                    )));
                }
            };
            put_ptrmap_entry(&mut page, PTRMAP_ENTRY_SIZE * k as usize, kind, parent);
        }
        stage_if_changed(cow, b, &page)?;
        match (b as u64).checked_add(stride) {
            Some(nb) if nb <= u32::MAX as u64 => b = nb as PageId,
            _ => break,
        }
    }
    Ok(())
}

/// Record offset 52 (largest root b-tree page, §1.3.12) on page 1, preserving every
/// other header field (notably offset 64, the incremental-vacuum flag the PRAGMA set).
/// A no-op when the value is already correct, so a commit that does not change the root
/// set does not rewrite page 1's header here.
fn set_largest_root<S: PageStore>(cow: &mut Cow<S>, largest: PageId) -> Result<()> {
    let mut page1 = cow.read_page(1)?.to_vec();
    let head: &mut [u8; HEADER_SIZE] = (&mut page1[..HEADER_SIZE])
        .try_into()
        .expect("page 1 is at least HEADER_SIZE bytes");
    let mut header = DatabaseHeader::read(head)?;
    if header.largest_root_btree == largest {
        return Ok(());
    }
    header.largest_root_btree = largest;
    header.write(head);
    cow.stage_write(1, &page1)
}

/// Stage `bytes` for page `id` only when it differs from the page the transaction would
/// otherwise commit (overlay-then-committed). Keeps an unchanged ptrmap page out of the
/// write set, so a no-structural-change commit journals and writes nothing extra.
fn stage_if_changed<S: PageStore>(cow: &mut Cow<S>, id: PageId, bytes: &[u8]) -> Result<()> {
    let differs = cow.read_page(id).map(|cur| cur != bytes).unwrap_or(true);
    if differs {
        cow.stage_write(id, bytes)?;
    }
    Ok(())
}
