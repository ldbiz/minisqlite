//! On-disk format — WRITE direction (Part B).
//!
//! Each test drives OUR writers (`codec::PageBuilder`/`encode_record`/the cell
//! encoders, and `minisqlite_btree::{init_database, create_table_btree,
//! table_insert}`) and checks the bytes they produce against the SAME hand-built,
//! spec-derived fixtures the read tests use (`common`). Two regimes, per the
//! byte-exactness rule:
//!
//!   * BYTE-EXACT — F1 (empty db), F2 (a few rows on one leaf), F3 (an overflow
//!     chain): the layout is fully determined by the logical contents, so our writer
//!     must reproduce the fixture bytes exactly.
//!   * SPEC-VALID + ROUND-TRIP — F4 (freelist) and F5 (b-tree split): the spec leaves
//!     WHICH pages land on the freelist and WHERE a tree splits to the implementation,
//!     so instead of forcing byte identity we assert the writer's output is itself
//!     spec-valid (page types, header fields, separator invariant, trunk format) and
//!     that our reader reads it back into the same logical rows.
//!
//! No `sqlite3` is invoked; every test always runs and always asserts.

mod common;

use common::*;

use minisqlite_pager::codec::{self, PageType, PageView, HEADER_SIZE};
use minisqlite_pager::{DiskPager, MemPager, PageId, Pager};
use minisqlite_types::Value;

use minisqlite_btree::{create_table_btree, init_database, table_insert};

// ---- F1: empty database (byte-exact) ----------------------------------------------

#[test]
fn f1_codec_composition_reproduces_fixture_bytes() {
    // The empty database is fully determined by the spec: our header writer composed
    // with an empty-leaf `PageBuilder` must be byte-for-byte the hand-built fixture.
    for &ps in &FIXTURE_PAGE_SIZES {
        assert_eq!(
            our_empty_db_page1(ps),
            f1_empty_db(ps),
            "empty-db page 1 (page size {ps}) must match the spec fixture byte-for-byte"
        );
    }
}

#[test]
fn f1_init_database_reproduces_fixture_bytes() {
    // The b-tree entry point `init_database` (over an in-memory pager, which does no
    // commit-time header maintenance) must also reproduce the fixture exactly.
    for &ps in &FIXTURE_PAGE_SIZES {
        let mut mp = MemPager::new(ps);
        init_database(&mut mp).unwrap();
        assert_eq!(
            mp.read_page(1).unwrap(),
            f1_empty_db(ps).as_slice(),
            "init_database page 1 (page size {ps}) must match the spec fixture byte-for-byte"
        );
    }
}

#[test]
fn f1_empty_db_survives_disk_commit_and_reopen() {
    // The durable path: format an empty db on disk, commit, reopen. The committed
    // page-1 header carries a bumped change counter (§1.3.6) so the FULL page differs
    // from the zero-counter fixture, but the b-tree region (offset 100+) is unchanged
    // and must still match, and the database must reopen as a valid empty schema.
    for &ps in &FIXTURE_PAGE_SIZES {
        let dir = TempDir::new("f1w");
        let path = dir.db("empty.db");
        drop(fresh_disk_db(&path, ps)); // formats + commits an empty db, then closes

        let pager = DiskPager::open(&path).unwrap();
        assert_eq!(pager.page_size(), ps);
        assert_eq!(pager.page_count().unwrap(), 1, "committed empty db is one page");
        assert_eq!(
            &pager.read_page(1).unwrap()[HEADER_SIZE..],
            &f1_empty_db(ps)[HEADER_SIZE..],
            "committed empty leaf b-tree region matches the fixture (page {ps})"
        );
        let hdr = header_of(&read_file(&path));
        assert_eq!(hdr.page_size, ps);
        assert!(hdr.in_header_size_valid() && hdr.database_size_pages == 1);
        assert!(scan_table(&pager, 1).is_empty(), "reopened empty schema has no rows");
    }
}

// ---- F2: several rows on one leaf (byte-exact) ------------------------------------

#[test]
fn f2_writer_reproduces_leaf_bytes() {
    for &ps in &FIXTURE_PAGE_SIZES {
        let dir = TempDir::new("f2w");
        let path = dir.db("rows.db");

        let mut p = fresh_disk_db(&path, ps);
        p.begin().unwrap();
        let root = create_table_btree(&mut p).unwrap();
        assert_eq!(root, 2, "the first user table roots at page 2");
        for (rowid, cols) in f2_rows() {
            // The payload is produced by OUR record writer; the byte-exact leaf
            // comparison below therefore also validates `encode_record`.
            table_insert(&mut p, root, rowid, &codec::encode_record(&cols)).unwrap();
        }
        p.commit().unwrap();
        drop(p);

        let raw = read_file(&path);
        assert_eq!(raw.len(), 2 * ps as usize, "schema page + one leaf");
        assert_eq!(
            page_slice(&raw, 2, ps),
            f2_leaf_page(ps).as_slice(),
            "writer's leaf page must match the spec fixture byte-for-byte (page {ps})"
        );

        // Round-trip: our reader reads our writer's page back into the same rows.
        let pager = DiskPager::open(&path).unwrap();
        let got = scan_table(&pager, 2);
        let want = f2_rows();
        assert_eq!(got.len(), want.len());
        for ((grow, gcols), (wrow, wcols)) in got.iter().zip(&want) {
            assert_eq!(grow, wrow);
            assert_row_eq(gcols, wcols, &format!("F2 round-trip row {wrow} (page {ps})"));
        }
    }
}

// ---- F3: a row spilling onto overflow pages (byte-exact, page size 512) -----------

#[test]
fn f3_writer_reproduces_overflow_chain_bytes() {
    // Independent pins first: our record writer reproduces the fixture record, and the
    // spill split matches the hand-computed spec values.
    assert_eq!(
        codec::encode_record(&[Value::Blob(f3_blob())]),
        f3_record(),
        "encode_record of the large BLOB must match the spec fixture record"
    );
    assert_eq!(f3_split(), (39, 1461), "U=512 P=1500 split: inline M=39, overflow 1461 (§1.6)");

    let dir = TempDir::new("f3w");
    let path = dir.db("overflow.db");

    let mut p = fresh_disk_db(&path, F3_PAGE_SIZE);
    p.begin().unwrap();
    let root = create_table_btree(&mut p).unwrap();
    assert_eq!(root, 2);
    table_insert(&mut p, root, 1, &codec::encode_record(&[Value::Blob(f3_blob())])).unwrap();
    p.commit().unwrap();
    drop(p);

    let raw = read_file(&path);
    assert_eq!(raw.len(), 5 * F3_PAGE_SIZE as usize, "schema + leaf + 3 overflow pages");
    // The spill leaf (page 2): inline prefix + first-overflow pointer.
    assert_eq!(
        page_slice(&raw, 2, F3_PAGE_SIZE),
        f3_leaf_page().as_slice(),
        "writer's spill leaf must match the spec fixture byte-for-byte"
    );
    // The overflow chain (pages 3,4,5): [next-page][content], last zero-padded.
    for (i, want_page) in f3_overflow_pages().iter().enumerate() {
        let page_no = 3 + i as u32;
        assert_eq!(
            page_slice(&raw, page_no, F3_PAGE_SIZE),
            want_page.as_slice(),
            "writer's overflow page {page_no} must match the spec fixture byte-for-byte"
        );
    }

    // Round-trip: reassemble the row through our reader.
    let pager = DiskPager::open(&path).unwrap();
    let rows = scan_table(&pager, 2);
    assert_eq!(rows.len(), 1);
    assert_row_eq(&rows[0].1, &[Value::Blob(f3_blob())], "F3 round-trip BLOB");
}

#[test]
fn f3_overflow_roundtrips_at_multiple_page_sizes() {
    // Overflow is page-size-parameterized (the split formula uses U); exercise the
    // write+read chain at two page sizes with payloads sized to overflow each, to show
    // the byte-exact 512 case is not a special case.
    for &ps in &[512u32, 4096] {
        let dir = TempDir::new("f3rt");
        let path = dir.db("ov.db");
        // 2*page_size is comfortably above the inline threshold X = U-35 for any U.
        let blob: Vec<u8> = (0..(ps as usize * 2)).map(|i| ((i * 7 + 3) % 251) as u8).collect();

        let mut p = fresh_disk_db(&path, ps);
        p.begin().unwrap();
        let root = create_table_btree(&mut p).unwrap();
        table_insert(&mut p, root, 1, &codec::encode_record(&[Value::Blob(blob.clone())])).unwrap();
        p.commit().unwrap();
        assert!(p.page_count().unwrap() > 2, "the payload must have spilled (page {ps})");
        drop(p);

        let pager = DiskPager::open(&path).unwrap();
        let rows = scan_table(&pager, root);
        assert_eq!(rows.len(), 1);
        assert_row_eq(&rows[0].1, &[Value::Blob(blob)], &format!("overflow round-trip (page {ps})"));
    }
}

// ---- F4: a non-empty freelist (spec-valid + round-trip) ---------------------------

#[test]
fn f4_writer_produces_spec_valid_freelist() {
    // The spec pins the trunk-page BYTE LAYOUT (§1.5) but not which pages land on the
    // freelist or in what order — that is our allocation policy. So we assert the
    // writer's freelist is spec-valid (header fields set, every trunk parses, the
    // freelist pages are exactly those freed) and round-trips (a later allocation
    // reuses a freed page), NOT byte-identity against an arbitrary fixture.
    for &ps in &[512u32, 4096] {
        let dir = TempDir::new("f4w");
        let path = dir.db("free.db");
        let mut p = fresh_disk_db(&path, ps);

        // Grow to pages 2..=6, then free three of them.
        p.begin().unwrap();
        let allocated: Vec<PageId> = (0..5).map(|_| p.allocate_page().unwrap()).collect();
        p.commit().unwrap();
        assert_eq!(allocated, vec![2, 3, 4, 5, 6], "sequential growth with no freelist yet");

        let freed = [3u32, 5, 6];
        p.begin().unwrap();
        for &f in &freed {
            p.free_page(f).unwrap();
        }
        p.commit().unwrap();
        drop(p);

        let raw = read_file(&path);
        let hdr = header_of(&raw);
        let usable = hdr.usable_size();
        assert_eq!(hdr.freelist_count, freed.len() as u32, "freelist_count @36 == pages freed");
        assert_ne!(hdr.first_freelist_trunk, 0, "freelist head @32 is set");

        // Walk the trunk chain: every trunk must parse to a spec-valid page, and the
        // union of trunks + their leaves must be exactly the freed pages.
        let pager = DiskPager::open(&path).unwrap();
        let mut freelist_pages = Vec::new();
        let mut trunk = hdr.first_freelist_trunk;
        while trunk != 0 {
            assert!((2..=6).contains(&trunk), "trunk page id in range");
            freelist_pages.push(trunk);
            let (next, leaves) = codec::parse_trunk(pager.read_page(trunk).unwrap(), usable).unwrap();
            assert!(leaves.len() <= codec::trunk_leaf_capacity(usable), "trunk within capacity");
            freelist_pages.extend(leaves);
            trunk = next;
        }
        freelist_pages.sort_unstable();
        assert_eq!(freelist_pages, freed.to_vec(), "freelist holds exactly the freed pages");
        assert_eq!(freelist_pages.len() as u32, hdr.freelist_count, "count == pages walked");

        // Round-trip: the next allocation reuses a freed page before the file grows.
        let mut pager = pager;
        pager.begin().unwrap();
        let reused = pager.allocate_page().unwrap();
        pager.commit().unwrap();
        assert!(freed.contains(&reused), "allocation reuses a freed page (got {reused})");
    }
}

// ---- F5: interior + leaf b-tree pages (spec-valid + round-trip) --------------------

/// Recursively validate a table b-tree: leaf rowids ascend, and every interior page
/// (type 0x05) holds the separator invariant (fileformat2 §1.6) — each divider key
/// equals the max rowid of its left subtree, and the next subtree's min rowid is
/// strictly greater.
fn assert_tree_spec_valid(pager: &dyn Pager, page: PageId, usable: usize) {
    let view = PageView::new(pager.read_page(page).unwrap(), page, usable).unwrap();
    match view.page_type() {
        PageType::LeafTable => {
            let mut prev: Option<i64> = None;
            for i in 0..view.cell_count() as usize {
                let r = view.table_leaf_cell(i).unwrap().rowid;
                if let Some(p) = prev {
                    assert!(p < r, "leaf {page}: rowids strictly ascending ({p} then {r})");
                }
                prev = Some(r);
            }
        }
        PageType::InteriorTable => {
            let n = view.cell_count() as usize;
            assert!(n >= 1, "interior page {page} has at least one divider cell");
            let right = view.right_most_pointer().expect("interior has a right-most pointer @8");
            // children[0..n] are the divider left-children; children[n] is right-most.
            let mut children: Vec<PageId> =
                (0..n).map(|i| view.table_interior_cell(i).unwrap().left_child).collect();
            children.push(right);
            for i in 0..n {
                let key = view.table_interior_cell(i).unwrap().rowid;
                assert_eq!(
                    key,
                    max_rowid(pager, children[i], usable),
                    "interior {page} divider {i}: separator == max rowid of left subtree"
                );
                assert!(
                    min_rowid(pager, children[i + 1], usable) > key,
                    "interior {page} divider {i}: next subtree min rowid > separator"
                );
            }
            for &c in &children {
                assert_tree_spec_valid(pager, c, usable);
            }
        }
        other => panic!("page {page} is a {other:?}, not a table b-tree page"),
    }
}

#[test]
fn f5_writer_builds_spec_valid_interior_tree() {
    // Enough rows on a small page to force balance-deeper into an interior root, at two
    // page sizes. WHERE the tree splits is our policy (not byte-pinned); we assert the
    // result is a spec-valid interior tree and round-trips to the same rows.
    for &(ps, n_rows) in &[(512u32, 400i64), (4096, 2000)] {
        let dir = TempDir::new("f5w");
        let path = dir.db("tree.db");

        let mut p = fresh_disk_db(&path, ps);
        p.begin().unwrap();
        let root = create_table_btree(&mut p).unwrap();
        assert_eq!(root, 2);
        let mut want = Vec::with_capacity(n_rows as usize);
        for rowid in 1..=n_rows {
            // Value distinct from the rowid, so a rowid/value confusion is caught.
            let val = Value::Integer(rowid * 3 + 1);
            table_insert(&mut p, root, rowid, &codec::encode_record(std::slice::from_ref(&val)))
                .unwrap();
            want.push((rowid, val));
        }
        p.commit().unwrap();
        drop(p);

        let pager = DiskPager::open(&path).unwrap();
        let usable = header_of(&read_file(&path)).usable_size();

        // The root must actually be an interior page (else the test proves nothing).
        let root_view = PageView::new(pager.read_page(root).unwrap(), root, usable).unwrap();
        assert_eq!(
            root_view.page_type(),
            PageType::InteriorTable,
            "{n_rows} rows on a {ps}-byte page must force an interior root"
        );
        assert!(pager.page_count().unwrap() > 3, "an interior root implies several leaves");

        // Structural: separator invariant holds throughout the tree.
        assert_tree_spec_valid(&pager, root, usable);

        // Round-trip: every row reads back in rowid order with the right value.
        let got = scan_table(&pager, root);
        assert_eq!(got.len(), want.len(), "row count (page {ps})");
        for ((grow, gcols), (wrow, wval)) in got.iter().zip(&want) {
            assert_eq!(grow, wrow, "rowid order across the split (page {ps})");
            assert_row_eq(gcols, std::slice::from_ref(wval), &format!("row {wrow} (page {ps})"));
        }
    }
}
