//! Decode-free entry counting over a b-tree, the storage half of the `count(*)` scan
//! fast path. It walks the tree following only child pointers and reading each page's
//! 2-byte cell count (fileformat2 §1.6, header offset 3) — it never decodes a cell
//! payload, reassembles an overflow chain, or materializes a row. A cell that spilled
//! to overflow pages is still ONE entry, and freelist pages are never in the tree, so
//! they are never visited.
//!
//! The count returned is EXACTLY the number of rows the matching read cursor's
//! `first`/`next` drain would yield — that equivalence is the correctness contract the
//! executor's `count_rows` override relies on. Achieving it needs the B+tree/B-tree
//! distinction of fileformat2 §1.6:
//!
//! * **Table b-tree** (rowid tables): a B+tree — "all data in the leaves", interior
//!   cells are pure rowid separators. So the row count is the sum of LEAF cell counts;
//!   interior cells are NOT counted. This matches [`TableCursor`](crate::TableCursor),
//!   which yields only leaf cells.
//! * **Index b-tree** (WITHOUT ROWID tables, and secondary indexes): a CLASSIC B-tree —
//!   every interior divider cell is itself a real entry present exactly once. So the
//!   entry count is the sum of ALL cells, leaf AND interior. This matches
//!   [`IndexCursor`](crate::IndexCursor), which interleaves interior dividers with leaf
//!   keys. Counting only leaf cells would UNDERCOUNT a multi-level index tree by the
//!   number of interior dividers.
//!
//! Traversal is bounded (a `count(*)` on a hostile/corrupt file must not hang or exhaust
//! memory): a DFS worklist popped LIFO holds only the pending-sibling spine, so peak
//! memory is O(fanout × height), and a `visited > page_count` guard turns a cyclic or
//! corrupt tree into a loud error instead of an unbounded walk. It reuses the existing
//! page-access/navigation helpers ([`PageView`], `nav::pointer_at`,
//! `index_nav::index_pointer_at`, `tree::usable_of`) rather than re-parsing pages.

use minisqlite_fileformat::PageView;
use minisqlite_pager::{PageId, Pager};
use minisqlite_types::{Error, Result};

use crate::index_nav::index_pointer_at;
use crate::nav::pointer_at;
use crate::tree::usable_of;

/// Which b-tree family a count walks, selecting both the valid page type and whether an
/// interior page's cells are themselves entries (see the module docs).
#[derive(Clone, Copy)]
enum TreeKind {
    /// Rowid table b-tree (B+tree): entries live only in leaves.
    Table,
    /// Index b-tree (classic B-tree): every cell — leaf or interior — is an entry.
    Index,
}

/// Number of rows in the rowid **table** b-tree rooted at `root`, counted without
/// decoding any payload — the sum of every leaf page's cell count (interior cells are
/// rowid separators, not rows; fileformat2 §1.6). Equals a full [`TableCursor`] scan's
/// row count. An empty table (a root leaf with zero cells) is `0`.
pub fn table_entry_count(pager: &dyn Pager, root: PageId) -> Result<usize> {
    count_entries(pager, root, TreeKind::Table)
}

/// Number of entries in the **index** b-tree rooted at `root`, counted without decoding
/// any key — the sum of EVERY page's cell count, leaf and interior, because an index
/// b-tree is a classic B-tree whose interior dividers are real entries (fileformat2
/// §1.6). Equals a full [`IndexCursor`] scan's entry count, so it is the row count of a
/// WITHOUT ROWID table (whose rows are its PRIMARY KEY index b-tree). An empty tree (a
/// root leaf with zero cells) is `0`.
pub fn index_entry_count(pager: &dyn Pager, root: PageId) -> Result<usize> {
    count_entries(pager, root, TreeKind::Index)
}

/// Shared decode-free traversal: descend from `root`, adding each page's contribution to
/// the total and following every child pointer, bounded against a cyclic/corrupt tree.
fn count_entries(pager: &dyn Pager, root: PageId, kind: TreeKind) -> Result<usize> {
    let usable = usable_of(pager);
    // Upper bound on legitimately reachable pages: a well-formed b-tree's pages are a
    // subset of the file's, each visited exactly once, so a walk that visits more pages
    // than exist has followed a cycle (or a corrupt pointer) — fail closed rather than
    // loop. This also caps total work at O(page_count).
    //
    // Snapshot invariant: `page_count` here and every `read_page` below MUST resolve
    // against the SAME pager snapshot (in WAL mode both key off the connection's pinned
    // snapshot). That is what makes this bound sound — `page_count` is then never smaller
    // than the tree's page set at that snapshot, so a concurrent grow/VACUUM on another
    // connection cannot make a valid tree spuriously trip the guard. A future refactor that
    // let `page_count` read a different (live/shared) size than `read_page` resolves against
    // would break this; keep them on one snapshot.
    let page_limit = u64::from(pager.page_count()?);
    let mut total: u64 = 0;
    let mut visited: u64 = 0;
    // DFS worklist of pages still to examine. Popped LIFO, so at any moment it holds only
    // the not-yet-descended siblings along the current root-to-leaf spine: peak size is
    // O(fanout × height), not O(pages).
    let mut stack: Vec<PageId> = Vec::new();
    stack.push(root);
    while let Some(pid) = stack.pop() {
        visited += 1;
        if visited > page_limit {
            return Err(Error::format(
                "b-tree entry count exceeded the database page count (cyclic or corrupt tree)",
            ));
        }
        let view = PageView::new(pager.read_page(pid)?, pid, usable)?;
        let page_type = view.page_type();
        // Fail closed if a page is the wrong b-tree family — a table count must never
        // wander into index pages, or vice versa (a corrupt root or child pointer).
        let family_ok = match kind {
            TreeKind::Table => page_type.is_table(),
            TreeKind::Index => page_type.is_index(),
        };
        if !family_ok {
            return Err(Error::format(
                "b-tree entry count reached a page of the wrong b-tree family",
            ));
        }
        let cells = view.cell_count();
        if page_type.is_leaf() {
            // Leaf cells are entries for both families.
            total += u64::from(cells);
        } else {
            // Interior page. For an index b-tree each divider cell is itself an entry
            // (classic B-tree); for a table b-tree interior cells are pure separators.
            if matches!(kind, TreeKind::Index) {
                total += u64::from(cells);
            }
            // Follow every child: the K left-child pointers (cells `0..K`) plus the
            // right-most pointer (index `K`).
            for i in 0..=u32::from(cells) {
                let child = match kind {
                    TreeKind::Table => pointer_at(&view, i)?,
                    TreeKind::Index => index_pointer_at(&view, i)?,
                };
                stack.push(child);
            }
        }
    }
    // `total` is bounded by page_count × cells-per-page, so it fits `usize` on any 64-bit
    // target; convert fail-closed (matching the rest of this fn) rather than silently
    // truncating on a hypothetical 32-bit target.
    usize::try_from(total)
        .map_err(|_| Error::format("b-tree entry count exceeds the platform's usize range"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        create_index_btree, create_table_btree, index_delete, index_insert, init_database,
        table_delete, table_insert, IndexCursor, TableCursor,
    };
    use minisqlite_fileformat::{encode_record, PageView};
    use minisqlite_pager::MemPager;
    use minisqlite_types::Value;

    /// A deterministic ~40-byte payload for a rowid, wide enough that a modest row count
    /// splits a small page and forces interior levels.
    fn payload_for(rowid: i64) -> Vec<u8> {
        let mut v = format!("row-{rowid}-").into_bytes();
        let fill = (rowid as u8).wrapping_mul(31).wrapping_add(7);
        while v.len() < 40 {
            v.push(fill);
        }
        v
    }

    /// Build a rowid table b-tree with rowids `1..=n` and return `(pager, root)`.
    fn build_table(page_size: u32, n: i64) -> (MemPager, PageId) {
        let mut pager = MemPager::new(page_size);
        init_database(&mut pager).unwrap();
        let root = create_table_btree(&mut pager).unwrap();
        for id in 1..=n {
            table_insert(&mut pager, root, id, &payload_for(id)).unwrap();
        }
        (pager, root)
    }

    /// An index key record `(key, rowid)`, both integers — the classic secondary-index
    /// / WITHOUT-ROWID key shape (trailing rowid, unique overall).
    fn index_key(k: i64, rowid: i64) -> Vec<u8> {
        encode_record(&[Value::Integer(k), Value::Integer(rowid)])
    }

    /// Build an index b-tree with keys `(i, i)` for `i in 1..=n` and return `(pager, root)`.
    fn build_index(page_size: u32, n: i64) -> (MemPager, PageId) {
        let mut pager = MemPager::new(page_size);
        init_database(&mut pager).unwrap();
        let root = create_index_btree(&mut pager).unwrap();
        for i in 1..=n {
            index_insert(&mut pager, root, &index_key(i, i)).unwrap();
        }
        (pager, root)
    }

    /// Rows a full forward `TableCursor` scan yields — the drain the fast count must equal.
    fn table_drain_count(pager: &dyn Pager, root: PageId) -> usize {
        let mut cur = TableCursor::open(pager, root).unwrap();
        let mut n = 0usize;
        if !cur.first().unwrap() {
            return 0;
        }
        loop {
            n += 1;
            if !cur.next().unwrap() {
                break;
            }
        }
        n
    }

    /// Entries a full forward `IndexCursor` scan yields (interleaving interior dividers) —
    /// the drain the fast index count must equal.
    fn index_drain_count(pager: &dyn Pager, root: PageId) -> usize {
        let mut cur = IndexCursor::open(pager, root).unwrap();
        let mut n = 0usize;
        if !cur.first().unwrap() {
            return 0;
        }
        loop {
            n += 1;
            if !cur.next().unwrap() {
                break;
            }
        }
        n
    }

    /// Whether the root page is an interior page (i.e. the tree has more than one level),
    /// so a test can assert it actually exercised the interior-cell path.
    fn root_is_interior(pager: &dyn Pager, root: PageId) -> bool {
        let usable = usable_of(pager);
        PageView::new(pager.read_page(root).unwrap(), root, usable)
            .unwrap()
            .page_type()
            .is_interior()
    }

    // --- Table b-tree: count == drain == n, across sizes and after deletes. ----------

    #[test]
    fn table_count_matches_drain_and_n_across_sizes() {
        // 0 and 1 are the empty / single-leaf boundaries; the larger sizes at a 512-byte
        // page force multiple leaves and then interior levels.
        for n in [0i64, 1, 2, 5, 50, 500] {
            let (pager, root) = build_table(512, n);
            let count = table_entry_count(&pager, root).unwrap();
            assert_eq!(count as i64, n, "table_entry_count must equal the inserted rows (n={n})");
            assert_eq!(
                count,
                table_drain_count(&pager, root),
                "table_entry_count must equal the TableCursor drain (n={n})"
            );
        }
    }

    #[test]
    fn table_count_empty_root_leaf_is_zero() {
        let (pager, root) = build_table(4096, 0);
        assert!(!root_is_interior(&pager, root), "an empty table is a single leaf root");
        assert_eq!(table_entry_count(&pager, root).unwrap(), 0);
    }

    #[test]
    fn table_count_exercises_interior_levels() {
        // 500 rows of ~40 bytes on 512-byte pages splits well past a single leaf, so the
        // root is an interior page — the interior-follow path is exercised, not just leaves.
        let (pager, root) = build_table(512, 500);
        assert!(root_is_interior(&pager, root), "500 rows at 512B must build interior levels");
        assert_eq!(table_entry_count(&pager, root).unwrap(), 500);
        assert_eq!(table_entry_count(&pager, root).unwrap(), table_drain_count(&pager, root));
    }

    #[test]
    fn table_count_after_deletes_matches_survivors() {
        // Deletes leave page gaps and free pages (which are never in the tree, so never
        // revisited). Delete every 3rd rowid of 1..=300 (rowids 1,4,…,298 = 100 rows), so
        // 200 survive: the fast count must equal both that exact figure and the drain.
        let (mut pager, root) = build_table(512, 300);
        let mut deleted = 0i64;
        for id in (1..=300).step_by(3) {
            assert!(table_delete(&mut pager, root, id).unwrap(), "rowid {id} present to delete");
            deleted += 1;
        }
        let survivors = 300 - deleted;
        assert_eq!(survivors, 200, "every-3rd deletion of 1..=300 leaves 200 rows");
        assert_eq!(
            table_entry_count(&pager, root).unwrap() as i64,
            survivors,
            "fast count must equal the exact survivor count after deletes"
        );
        assert_eq!(
            table_entry_count(&pager, root).unwrap(),
            table_drain_count(&pager, root),
            "fast count must equal the TableCursor drain after deletes"
        );
    }

    // --- Index b-tree (WITHOUT ROWID): count == drain == n, incl. interior dividers. -

    #[test]
    fn index_count_matches_drain_and_n_across_sizes() {
        for n in [0i64, 1, 2, 5, 50, 500] {
            let (pager, root) = build_index(512, n);
            let count = index_entry_count(&pager, root).unwrap();
            assert_eq!(count as i64, n, "index_entry_count must equal inserted keys (n={n})");
            assert_eq!(
                count,
                index_drain_count(&pager, root),
                "index_entry_count must equal the IndexCursor drain (n={n})"
            );
        }
    }

    #[test]
    fn index_count_counts_interior_dividers_not_just_leaves() {
        // The load-bearing case: a multi-level index tree. Its interior divider cells are
        // real entries, so counting leaves alone would UNDERCOUNT. The drain equality is
        // the proof (IndexCursor interleaves interior dividers), and we independently
        // assert the tree really has an interior root so leaf-only would visibly fail.
        let (pager, root) = build_index(512, 500);
        assert!(root_is_interior(&pager, root), "500 keys at 512B must build interior levels");
        let count = index_entry_count(&pager, root).unwrap();
        assert_eq!(count, 500, "every key counted once, including interior dividers");
        assert_eq!(count, index_drain_count(&pager, root), "must equal the IndexCursor drain");

        // Prove interior dividers are actually present (so the leaf-only bug is reachable
        // in this fixture): the summed LEAF cell count is strictly less than 500.
        let leaf_only = sum_leaf_cells_index(&pager, root);
        assert!(
            leaf_only < 500,
            "fixture must have interior dividers so leaf-only ({leaf_only}) undercounts 500"
        );
    }

    #[test]
    fn index_count_after_deletes_matches_survivors() {
        let (mut pager, root) = build_index(512, 400);
        for i in (1..=400).step_by(2) {
            assert!(index_delete(&mut pager, root, &index_key(i, i)).unwrap(), "key {i} present");
        }
        assert_eq!(
            index_entry_count(&pager, root).unwrap(),
            index_drain_count(&pager, root),
            "after deletes the fast index count must equal the drain"
        );
        assert_eq!(index_entry_count(&pager, root).unwrap(), 200, "even keys removed => 200 left");
    }

    // --- Fail-closed: the wrong family surfaces a format error, never a wrong count. --

    #[test]
    fn table_count_on_an_index_root_errors() {
        let (pager, root) = build_index(512, 50);
        assert!(
            table_entry_count(&pager, root).is_err(),
            "counting an index b-tree as a table must fail closed"
        );
    }

    #[test]
    fn index_count_on_a_table_root_errors() {
        let (pager, root) = build_table(512, 50);
        assert!(
            index_entry_count(&pager, root).is_err(),
            "counting a table b-tree as an index must fail closed"
        );
    }

    // --- Overflow: a cell that spills onto overflow pages is still ONE entry; the count
    // reads only the on-page cell count and must NEVER follow the chain. -----------------

    #[test]
    fn table_count_with_overflow_spilled_rows_counts_each_once() {
        // A 4 KB payload dwarfs a 512-byte page, so every row's cell spills onto overflow
        // pages. The decode-free count reads only each leaf's cell count (never the chain),
        // so it must still equal n and the TableCursor drain (which DOES reassemble chains).
        let mut pager = MemPager::new(512);
        init_database(&mut pager).unwrap();
        let root = create_table_btree(&mut pager).unwrap();
        let big = vec![0xABu8; 4000];
        let n = 12i64;
        for id in 1..=n {
            table_insert(&mut pager, root, id, &big).unwrap();
        }
        assert_eq!(
            table_entry_count(&pager, root).unwrap() as i64,
            n,
            "a spilled row is ONE entry, counted once"
        );
        assert_eq!(
            table_entry_count(&pager, root).unwrap(),
            table_drain_count(&pager, root),
            "overflow count must equal the TableCursor drain"
        );
    }

    #[test]
    fn index_count_with_overflow_spilled_keys_counts_each_once() {
        // A 4 KB blob in the key record forces each index cell to spill. The classic-B-tree
        // count (all cells, leaf + interior dividers) must still equal n and the IndexCursor
        // drain without walking any overflow chain.
        let mut pager = MemPager::new(512);
        init_database(&mut pager).unwrap();
        let root = create_index_btree(&mut pager).unwrap();
        let n = 12i64;
        for i in 1..=n {
            let key = encode_record(&[Value::Integer(i), Value::Blob(vec![0xCDu8; 4000])]);
            index_insert(&mut pager, root, &key).unwrap();
        }
        assert_eq!(
            index_entry_count(&pager, root).unwrap() as i64,
            n,
            "a spilled key is ONE entry, counted once"
        );
        assert_eq!(
            index_entry_count(&pager, root).unwrap(),
            index_drain_count(&pager, root),
            "overflow index count must equal the IndexCursor drain"
        );
    }

    /// Sum ONLY the leaf-page cell counts of an index tree — the (wrong) leaf-only count,
    /// used to prove a fixture actually has interior dividers.
    fn sum_leaf_cells_index(pager: &dyn Pager, root: PageId) -> usize {
        let usable = usable_of(pager);
        let mut stack = vec![root];
        let mut total = 0usize;
        while let Some(pid) = stack.pop() {
            let view = PageView::new(pager.read_page(pid).unwrap(), pid, usable).unwrap();
            if view.page_type().is_leaf() {
                total += view.cell_count() as usize;
            } else {
                let k = view.cell_count();
                for i in 0..=u32::from(k) {
                    stack.push(index_pointer_at(&view, i).unwrap());
                }
            }
        }
        total
    }
}
