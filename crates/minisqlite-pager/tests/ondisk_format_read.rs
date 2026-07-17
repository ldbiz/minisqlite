//! On-disk format — READ direction (Part A).
//!
//! Each test hand-builds a `.db` byte sequence STRICTLY from the documented format
//! (`spec/sqlite-doc/fileformat2.html`) in `common`, writes it to a temp file, and
//! asserts OUR reader (`DiskPager` + `codec` + `minisqlite_btree::TableCursor`)
//! decodes it into the exact expected logical structure. The fixtures never go
//! through our own writers, so a reader bug is caught against an independent spec
//! authority. No `sqlite3` is invoked; every test always runs and always asserts.

mod common;

use common::*;

use minisqlite_pager::codec::{self, PageType, PageView};
use minisqlite_pager::{DiskPager, Pager};
use minisqlite_types::Value;

/// Open a `.db` file built from raw fixture bytes.
fn open_bytes(dir: &TempDir, name: &str, bytes: &[u8]) -> DiskPager {
    let path = dir.db(name);
    std::fs::write(&path, bytes).expect("write fixture");
    DiskPager::open(&path).expect("open fixture db")
}

/// Independent pin of the fixtures' varint encoder against literal spec examples
/// (fileformat2 §1.6), so a bug in the fixture builder itself cannot pass silently.
#[test]
fn spec_varint_matches_known_encodings() {
    assert_eq!(spec_varint(0), vec![0x00]);
    assert_eq!(spec_varint(1), vec![0x01]);
    assert_eq!(spec_varint(127), vec![0x7f]); // largest single-byte varint
    assert_eq!(spec_varint(128), vec![0x81, 0x00]); // smallest two-byte varint
    assert_eq!(spec_varint(1500), vec![0x8B, 0x5C]); // F3 payload length P
    assert_eq!(spec_varint(3006), vec![0x97, 0x3E]); // F3 BLOB serial type (12 + 2*1497)
    // A 9-byte varint carries the full low byte verbatim.
    assert_eq!(spec_varint(u64::MAX), vec![0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]);
}

// ---- F1: empty database -----------------------------------------------------------

#[test]
fn f1_empty_database_reads_as_empty_schema() {
    for &ps in &FIXTURE_PAGE_SIZES {
        let dir = TempDir::new("f1r");
        let pager = open_bytes(&dir, "empty.db", &f1_empty_db(ps));

        assert_eq!(pager.page_size(), ps, "page size decoded from header @16");
        assert_eq!(pager.page_count().unwrap(), 1, "a 1-page empty database");

        let raw = read_file(&dir.db("empty.db"));

        // Independently pin the load-bearing header bytes at their LITERAL §1.3
        // offsets, so the empty-db fixture's authority is the spec text itself — not
        // our `DatabaseHeader` decoder, nor its `default()` (which happens to share
        // these same "free" constants).
        assert_eq!(&raw[0..16], b"SQLite format 3\0", "magic string @0 (§1.3.1)");
        let want_ps_field: [u8; 2] =
            if ps == 65_536 { [0x00, 0x01] } else { (ps as u16).to_be_bytes() };
        assert_eq!(&raw[16..18], &want_ps_field, "page-size @16 (65536 stored as 1, §1.3.2)");
        assert_eq!(raw[18], 1, "write version @18 (rollback journal, §1.3.3)");
        assert_eq!(raw[19], 1, "read version @19");
        assert_eq!(raw[21], 64, "max embedded payload fraction @21 (§1.3.5)");
        assert_eq!(raw[22], 32, "min embedded payload fraction @22");
        assert_eq!(raw[23], 32, "leaf payload fraction @23");
        assert_eq!(&raw[44..48], &[0, 0, 0, 4], "schema format 4 @44 (§1.3.10)");
        assert_eq!(&raw[56..60], &[0, 0, 0, 1], "text encoding UTF-8 @56 (§1.3.11)");
        assert_eq!(raw[100], 0x0d, "leaf table b-tree page type @100 (§1.6)");

        // The 100-byte header decodes to the documented empty-db field values.
        let hdr = header_of(&raw);
        assert_eq!(hdr.page_size, ps);
        assert_eq!(hdr.schema_format, 4, "schema format 4 @44");
        assert_eq!(hdr.text_encoding, codec::TEXT_ENCODING_UTF8, "UTF-8 @56");
        assert_eq!(hdr.database_size_pages, 1, "in-header size @28");
        assert!(hdr.in_header_size_valid(), "change counter == version-valid-for");
        assert_eq!(hdr.first_freelist_trunk, 0);
        assert_eq!(hdr.freelist_count, 0);

        // The b-tree at offset 100 is an empty leaf table page (the empty schema).
        let view = PageView::new(pager.read_page(1).unwrap(), 1, hdr.usable_size()).unwrap();
        assert_eq!(view.page_type(), PageType::LeafTable);
        assert_eq!(view.cell_count(), 0);

        // A cursor over the empty schema finds no rows.
        assert!(scan_table(&pager, 1).is_empty(), "empty schema has no rows (page {ps})");
    }
}

// ---- F2: several rows on one leaf --------------------------------------------------

#[test]
fn f2_rows_read_back_in_key_order() {
    for &ps in &FIXTURE_PAGE_SIZES {
        let dir = TempDir::new("f2r");
        let pager = open_bytes(&dir, "rows.db", &f2_file(ps));
        assert_eq!(pager.page_size(), ps);
        assert_eq!(pager.page_count().unwrap(), 2);

        let usable = header_of(&read_file(&dir.db("rows.db"))).usable_size();
        // Page 2 is a leaf table page holding all three rows (no split).
        let view = PageView::new(pager.read_page(2).unwrap(), 2, usable).unwrap();
        assert_eq!(view.page_type(), PageType::LeafTable);
        assert_eq!(view.cell_count() as usize, f2_rows().len());
        for (i, (rowid, _)) in f2_rows().iter().enumerate() {
            let cell = view.table_leaf_cell(i).unwrap();
            assert_eq!(cell.rowid, *rowid, "cell {i} rowid in key order");
            assert!(cell.overflow_page.is_none(), "F2 rows are inline");
        }

        // The full scan yields the exact logical rows (mixed serial types).
        let got = scan_table(&pager, 2);
        let want = f2_rows();
        assert_eq!(got.len(), want.len());
        for ((grow, gcols), (wrow, wcols)) in got.iter().zip(&want) {
            assert_eq!(grow, wrow, "rowid (page {ps})");
            assert_row_eq(gcols, wcols, &format!("row {wrow} (page {ps})"));
        }
    }
}

// ---- F3: a row spilling onto overflow pages ---------------------------------------

#[test]
fn f3_overflow_row_reassembles_from_chain() {
    let dir = TempDir::new("f3r");
    let pager = open_bytes(&dir, "overflow.db", &f3_file());
    assert_eq!(pager.page_size(), F3_PAGE_SIZE);
    assert_eq!(pager.page_count().unwrap(), 5, "schema + leaf + 3 overflow pages");

    // Pin the spec-formula split the fixture relies on (independent of production).
    let (local, overflow) = f3_split();
    assert_eq!(local, 39, "U=512 P=1500: K>X branch inlines exactly M=39 (§1.6)");
    assert_eq!(overflow, 1461, "the spilled remainder P - local");

    let usable = F3_PAGE_SIZE as usize; // reserved 0
    // The leaf cell keeps the inline prefix plus the first-overflow-page pointer.
    let view = PageView::new(pager.read_page(2).unwrap(), 2, usable).unwrap();
    let cell = view.table_leaf_cell(0).unwrap();
    assert_eq!(cell.rowid, 1);
    assert_eq!(cell.payload_len, F3_PAYLOAD_LEN as u64, "declared total payload P");
    assert_eq!(cell.local_payload.len(), local, "inline bytes = M");
    assert_eq!(cell.overflow_page, Some(3), "chain head is page 3");

    // The overflow pages form the documented [next-page][content] chain 3 -> 4 -> 5 -> 0.
    for (page_no, want_next) in [(3u32, 4u32), (4, 5), (5, 0)] {
        let ov = codec::parse_overflow_page(pager.read_page(page_no).unwrap(), usable).unwrap();
        assert_eq!(ov.next, want_next, "overflow page {page_no} next pointer");
    }

    // The cursor reassembles the inline prefix + chain into the original record.
    let rows = scan_table(&pager, 2);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].0, 1);
    assert_row_eq(&rows[0].1, &[Value::Blob(f3_blob())], "F3 reassembled BLOB row");
}

// ---- F4: a non-empty freelist -----------------------------------------------------

#[test]
fn f4_single_trunk_freelist_reads() {
    for &ps in &FIXTURE_PAGE_SIZES {
        let dir = TempDir::new("f4r");
        let pager = open_bytes(&dir, "freelist.db", &f4_single_trunk_file(ps));
        assert_eq!(pager.page_size(), ps);
        assert_eq!(pager.page_count().unwrap(), 4);

        let raw = read_file(&dir.db("freelist.db"));
        let hdr = header_of(&raw);
        assert_eq!(hdr.first_freelist_trunk, 2, "freelist head @32 -> page 2");
        assert_eq!(hdr.freelist_count, 3, "total freelist pages @36 (1 trunk + 2 leaves)");

        // The trunk page parses to the documented (next, leaves) structure (§1.5).
        let usable = hdr.usable_size();
        let (next, leaves) = codec::parse_trunk(pager.read_page(2).unwrap(), usable).unwrap();
        assert_eq!(next, 0, "single trunk is the last (next = 0), page {ps}");
        assert_eq!(leaves, vec![3, 4], "trunk lists leaf pages 3 and 4, page {ps}");
    }
}

#[test]
fn f4_chained_trunk_freelist_walks_the_chain() {
    for &ps in &FIXTURE_PAGE_SIZES {
        let dir = TempDir::new("f4rc");
        let pager = open_bytes(&dir, "freelist2.db", &f4_chained_trunk_file(ps));
        assert_eq!(pager.page_size(), ps);
        assert_eq!(pager.page_count().unwrap(), 5);

        let hdr = header_of(&read_file(&dir.db("freelist2.db")));
        assert_eq!(hdr.first_freelist_trunk, 2);
        assert_eq!(hdr.freelist_count, 4, "2 trunks + 2 leaves");
        let usable = hdr.usable_size();

        // Follow the trunk chain page 2 -> page 5 -> end, collecting every freelist
        // page, and confirm the count matches the header's freelist_count.
        let mut freelist_pages = Vec::new();
        let mut trunk = hdr.first_freelist_trunk;
        while trunk != 0 {
            freelist_pages.push(trunk); // the trunk page itself counts toward freelist_count
            let (next, leaves) =
                codec::parse_trunk(pager.read_page(trunk).unwrap(), usable).unwrap();
            freelist_pages.extend(leaves);
            trunk = next;
        }
        freelist_pages.sort_unstable();
        assert_eq!(freelist_pages, vec![2, 3, 4, 5], "trunks 2,5 and leaves 3,4, page {ps}");
        assert_eq!(freelist_pages.len() as u32, hdr.freelist_count);
    }
}

// ---- F5: interior + leaf b-tree pages ---------------------------------------------

#[test]
fn f5_interior_tree_reads_and_scans() {
    for &ps in &FIXTURE_PAGE_SIZES {
        let dir = TempDir::new("f5r");
        let pager = open_bytes(&dir, "interior.db", &f5_file(ps));
        assert_eq!(pager.page_size(), ps);
        assert_eq!(pager.page_count().unwrap(), 4);

        let usable = header_of(&read_file(&dir.db("interior.db"))).usable_size();
        // Page 2 is an interior table page: right-most pointer 4, one divider (child 3, key 3).
        let root = PageView::new(pager.read_page(2).unwrap(), 2, usable).unwrap();
        assert_eq!(root.page_type(), PageType::InteriorTable, "root is interior (0x05), page {ps}");
        assert_eq!(root.right_most_pointer(), Some(4), "right-most child @8");
        assert_eq!(root.cell_count(), 1);
        let divider = root.table_interior_cell(0).unwrap();
        assert_eq!(divider.left_child, 3, "divider left child = page 3");
        assert_eq!(divider.rowid, F5_SEPARATOR_ROWID, "separator = max rowid of the left leaf");

        // The two children are leaf table pages with the expected rowid ranges.
        for (child, want_rowids) in [(3u32, [1i64, 2, 3].as_slice()), (4, [4, 5].as_slice())] {
            let leaf = PageView::new(pager.read_page(child).unwrap(), child, usable).unwrap();
            assert_eq!(leaf.page_type(), PageType::LeafTable);
            assert_eq!(leaf.cell_count() as usize, want_rowids.len());
            for (i, &r) in want_rowids.iter().enumerate() {
                assert_eq!(leaf.table_leaf_cell(i).unwrap().rowid, r, "leaf {child} cell {i}");
            }
        }

        // A full scan descends interior -> leaves and yields every row in rowid order.
        let got = scan_table(&pager, 2);
        let want = f5_rows();
        assert_eq!(got.len(), want.len());
        for ((grow, gcols), (wrow, wval)) in got.iter().zip(&want) {
            assert_eq!(grow, wrow, "rowid order across the interior split (page {ps})");
            assert_row_eq(gcols, std::slice::from_ref(wval), &format!("row {wrow} (page {ps})"));
        }
    }
}

// ---- Fail-closed: malformed input the spec says our reader must reject ------------

#[test]
fn page_zero_and_out_of_range_reads_are_rejected() {
    // Page numbers are 1-based (fileformat2 §1.2); page 0 does not exist and an id
    // past the page count is out of range. Both must fail closed, not read garbage.
    let dir = TempDir::new("oob");
    let pager = open_bytes(&dir, "rows.db", &f2_file(4096));
    let count = pager.page_count().unwrap();
    assert_eq!(count, 2);
    assert!(pager.read_page(0).is_err(), "page 0 is invalid");
    assert!(pager.read_page(count + 1).is_err(), "page past the count is out of range");
}

#[test]
fn bad_magic_is_rejected_on_open() {
    // A file whose first 16 bytes are not "SQLite format 3\0" is not a database
    // (§1.3.1). Opening it must fail closed rather than treating junk as a header.
    let dir = TempDir::new("magic");
    let mut bytes = f1_empty_db(4096);
    bytes[3] = b'X'; // corrupt the magic string
    let path = dir.db("bad.db");
    std::fs::write(&path, &bytes).unwrap();
    assert!(DiskPager::open(&path).is_err(), "bad magic must fail closed");
}

#[test]
fn non_page_multiple_length_is_rejected_on_open() {
    // A valid header but a file length that is not a whole number of pages is
    // corrupt (§1.3.7 fallback assumes file_len / page_size is exact). Fail closed.
    let dir = TempDir::new("len");
    let mut bytes = f1_empty_db(512); // 512 bytes = exactly one 512-byte page
    bytes.extend_from_slice(&[0u8; 10]); // now 522 bytes, not a multiple of 512
    let path = dir.db("ragged.db");
    std::fs::write(&path, &bytes).unwrap();
    assert!(DiskPager::open(&path).is_err(), "ragged file length must fail closed");
}

#[test]
fn invalid_page_type_byte_is_rejected() {
    // §1.6: the b-tree page-type byte must be one of 0x02/0x05/0x0a/0x0d. A page 2
    // whose type byte is bogus is not a b-tree page — PageView must reject it.
    let mut file = spec_empty_schema_page1(512, 2, 0, 0);
    let mut page2 = vec![0u8; 512];
    page2[0] = 0xEE; // not a valid page type
    file.extend_from_slice(&page2);
    let dir = TempDir::new("ptype");
    let pager = open_bytes(&dir, "badpage.db", &file);
    assert!(
        PageView::new(pager.read_page(2).unwrap(), 2, 512).is_err(),
        "an invalid page-type byte must be rejected by the reader"
    );
}
