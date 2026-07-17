//! On-disk format — READ direction for two variants real `sqlite3` writes that a
//! plain reader gets wrong: a UTF-16 text encoding (§1.3.13) and an auto_vacuum file
//! with pointer-map pages (§1.3.12, §1.8).
//!
//! Like the sibling `ondisk_format_read` tests, every fixture is a hand-built BYTE
//! sequence derived straight from `spec/sqlite-doc/fileformat2.html` — never through
//! our own writers, and never via `sqlite3`. The fixtures
//! reuse the spec-literal builders in `common` (`spec_varint`, `spec_pack_page`,
//! `spec_db_header`, `spec_table_leaf_cell`) and add three variant-specific pieces:
//! a UTF-16 text encoder (the std `encode_utf16`, the inverse of the decode under
//! test — pinned to literal bytes below so a fixture bug can't pass silently), a
//! generic record encoder (§2.1), and a ptrmap page builder (§1.8).
//!
//! The read is driven through the SAME functions the engine's read path uses:
//! `minisqlite_pager::text_encoding_of` / `read_database_header` (page-1 header),
//! `minisqlite_btree::TableCursor` (b-tree traversal), and `codec::decode_record_enc`
//! (the encoding-aware record decoder). So these are genuine read-path conformance
//! checks, not a self-referential loop.

mod common;

use common::*;

use minisqlite_pager::codec::{self, PageView, TextEncoding};
use minisqlite_pager::{DiskPager, PageId, Pager};
use minisqlite_types::Value;

/// Open a `.db` file built from raw fixture bytes.
fn open_bytes(dir: &TempDir, name: &str, bytes: &[u8]) -> DiskPager {
    let path = dir.db(name);
    std::fs::write(&path, bytes).expect("write fixture");
    DiskPager::open(&path).expect("open fixture db")
}

// ===================================================================================
// Variant-specific spec-literal builders
// ===================================================================================

/// Encode `s` in text encoding `enc` for a stored TEXT body (§1.3.13). UTF-8 is the
/// string's own bytes; UTF-16 lays each code unit down in the encoding's byte order.
/// `str::encode_utf16` is the std code-unit iterator — the INVERSE of the UTF-16
/// decode under test, so using it here keeps the fixture independent of the reader
/// (`utf16_bytes_match_spec_layout` pins its output to literal bytes).
fn encode_text(s: &str, enc: TextEncoding) -> Vec<u8> {
    match enc {
        TextEncoding::Utf8 => s.as_bytes().to_vec(),
        TextEncoding::Utf16le => s.encode_utf16().flat_map(|u| u.to_le_bytes()).collect(),
        TextEncoding::Utf16be => s.encode_utf16().flat_map(|u| u.to_be_bytes()).collect(),
    }
}

/// The serial type for a TEXT value of `byte_len` bytes: odd `N = 13 + 2*byte_len`
/// (§2.1). A UTF-16 body is always even-length, but this holds for any byte length.
fn text_serial(byte_len: usize) -> u64 {
    13 + 2 * byte_len as u64
}

/// Encode a record (§2.1) from `(serial, body)` fields: a header of the total header
/// length varint followed by each serial-type varint, then the packed bodies. The
/// header length counts itself, so it is solved to a fixed point (a large header could
/// need a 2-byte length varint). Independent of our `encode_record`.
fn spec_record(fields: &[(u64, Vec<u8>)]) -> Vec<u8> {
    let mut serials = Vec::new();
    for (serial, _) in fields {
        serials.extend_from_slice(&spec_varint(*serial));
    }
    // Solve `header_len == varint_len(header_len) + serials.len()` (§2.1): the length
    // varint counts itself. Start assuming a 1-byte length and widen if it grows.
    let mut len_bytes = 1usize;
    let header = loop {
        let header_len = len_bytes + serials.len();
        let v = spec_varint(header_len as u64);
        if v.len() == len_bytes {
            break v;
        }
        len_bytes = v.len();
    };
    let mut rec = header;
    rec.extend_from_slice(&serials);
    for (_, body) in fields {
        rec.extend_from_slice(body);
    }
    rec
}

/// The `sqlite_schema` record for `CREATE TABLE t(c TEXT)` with root page `root`,
/// every TEXT column encoded in `enc` (§1.3.13). Columns are the schema table's
/// `(type, name, tbl_name, rootpage, sql)`. `rootpage` is a small positive integer
/// (serial 1, i8).
fn create_table_record(enc: TextEncoding, root: u32) -> Vec<u8> {
    let t = |s: &str| encode_text(s, enc);
    let type_b = t("table");
    let name_b = t("t");
    let tbl_b = t("t");
    let sql_b = t("CREATE TABLE t(c TEXT)");
    assert!((1..=127).contains(&root), "fixture root page fits an i8 body");
    spec_record(&[
        (text_serial(type_b.len()), type_b),
        (text_serial(name_b.len()), name_b),
        (text_serial(tbl_b.len()), tbl_b),
        (1, vec![root as u8]), // rootpage: serial 1 (i8)
        (text_serial(sql_b.len()), sql_b),
    ])
}

/// Page 1: the 100-byte header overlaid on a `sqlite_schema` leaf b-tree page holding
/// the single `CREATE TABLE t` row (rowid 1). `header` carries the variant-specific
/// fields (text encoding at 56, or the auto_vacuum flag at 52).
fn schema_page1(page_size: u32, header: [u8; 100], enc: TextEncoding, root: u32) -> Vec<u8> {
    let cell = spec_table_leaf_cell(1, &create_table_record(enc, root));
    let mut page = spec_pack_page(page_size, 100, 0x0d, None, &[cell]);
    page[..100].copy_from_slice(&header);
    page
}

/// A table `t` leaf b-tree page: one table-leaf cell per `(rowid, text)` row, each a
/// single-column record `[TEXT c]` encoded in `enc`.
fn table_leaf(page_size: u32, enc: TextEncoding, rows: &[(i64, &str)]) -> Vec<u8> {
    let cells: Vec<Vec<u8>> = rows
        .iter()
        .map(|(rowid, s)| {
            let body = encode_text(s, enc);
            spec_table_leaf_cell(*rowid, &spec_record(&[(text_serial(body.len()), body)]))
        })
        .collect();
    spec_pack_page(page_size, 0, 0x0d, None, &cells)
}

/// `spec_db_header` with the text-encoding field (offset 56, §1.3.13) overridden.
fn header_with_encoding(page_size: u32, db_size_pages: u32, encoding_code: u32) -> [u8; 100] {
    let mut h = spec_db_header(page_size, db_size_pages, 0, 0);
    h[56..60].copy_from_slice(&encoding_code.to_be_bytes());
    h
}

/// `spec_db_header` with the "largest root b-tree page" field (offset 52, §1.3.12) set
/// non-zero, marking the file as auto_vacuum — which requires ptrmap pages (§1.8).
fn header_autovacuum(page_size: u32, db_size_pages: u32, largest_root: u32) -> [u8; 100] {
    let mut h = spec_db_header(page_size, db_size_pages, 0, 0);
    h[52..56].copy_from_slice(&largest_root.to_be_bytes());
    h
}

/// A ptrmap page (§1.8): an array of 5-byte entries `[1-byte type][4-byte big-endian
/// page number]`. The first ptrmap page is page 2 and carries back-links for pages
/// 3, 4, … in order, so the entry for page `p` sits at byte offset `(p - 3) * 5`.
/// `entries` are `(page, type, back_pointer)`. Unlisted slots are left zero.
fn ptrmap_page(page_size: u32, entries: &[(u32, u8, u32)]) -> Vec<u8> {
    let mut page = vec![0u8; page_size as usize];
    for &(page_no, kind, back) in entries {
        let off = (page_no as usize - 3) * 5;
        page[off] = kind;
        page[off + 1..off + 5].copy_from_slice(&back.to_be_bytes());
    }
    page
}

// ===================================================================================
// Read helper: encoding-aware table scan (the read path's decode seam)
// ===================================================================================

/// Full forward scan of the table b-tree rooted at `root`, decoding each row in text
/// encoding `enc` — `TableCursor` + `decode_record_enc`, exactly the read path the
/// executor uses. Never visits a page the b-tree pointers do not reach (so a ptrmap
/// page is naturally skipped).
fn scan_table_enc(pager: &dyn Pager, root: PageId, enc: TextEncoding) -> Vec<(i64, Vec<Value>)> {
    use minisqlite_btree::TableCursor;
    let mut cur = TableCursor::open(pager, root).expect("open table cursor");
    let mut out = Vec::new();
    let mut positioned = cur.first().expect("cursor first");
    while positioned {
        let payload = cur.payload().expect("cursor payload");
        out.push((cur.rowid(), codec::decode_record_enc(&payload, enc)));
        positioned = cur.next().expect("cursor next");
    }
    out
}

// ===================================================================================
// Fixture-encoder self-pin (independent authority for the UTF-16 layout)
// ===================================================================================

#[test]
fn utf16_bytes_match_spec_layout() {
    // ASCII 'h','i' are U+0068,U+0069: one code unit each. LE stores the low byte
    // first, BE the high byte first (§1.3.13 / Unicode). Pins the fixture encoder so a
    // transcription bug cannot masquerade as a passing decode.
    assert_eq!(encode_text("hi", TextEncoding::Utf16le), vec![0x68, 0x00, 0x69, 0x00]);
    assert_eq!(encode_text("hi", TextEncoding::Utf16be), vec![0x00, 0x68, 0x00, 0x69]);
    // U+1F600 is a SUPPLEMENTARY code point: the surrogate pair D83D DE00. LE lays each
    // surrogate's low byte first. This is the exact case a naive "one u16 = one char"
    // decoder gets wrong, so pinning it here makes the round-trip test meaningful.
    assert_eq!(encode_text("\u{1F600}", TextEncoding::Utf16le), vec![0x3D, 0xD8, 0x00, 0xDE]);
    assert_eq!(encode_text("\u{1F600}", TextEncoding::Utf16be), vec![0xD8, 0x3D, 0xDE, 0x00]);
    // UTF-8 is the string's own bytes ("é" is two UTF-8 bytes 0xC3 0xA9).
    assert_eq!(encode_text("café", TextEncoding::Utf8), "café".as_bytes());
}

// ===================================================================================
// PART 1 — UTF-16 text read
// ===================================================================================

/// The rows both UTF-16 tests read back: an ASCII string, a Latin-1 string (é = a
/// single BMP code unit), and a string with a supplementary-plane emoji (a surrogate
/// pair) sandwiched between ASCII — so BMP, non-Latin, and surrogate-pair decoding are
/// all exercised end to end.
fn utf16_rows() -> [(i64, &'static str); 3] {
    [(1, "hi"), (2, "café"), (3, "a\u{1F600}z")]
}

fn utf16_reads_back_utf8(encoding_code: u32, enc: TextEncoding) {
    let ps = 512u32;
    let rows = utf16_rows();
    let dir = TempDir::new("u16r");
    let mut file = schema_page1(ps, header_with_encoding(ps, 2, encoding_code), enc, 2);
    file.extend_from_slice(&table_leaf(ps, enc, &rows));
    let pager = open_bytes(&dir, "u16.db", &file);

    assert_eq!(pager.page_size(), ps);
    assert_eq!(pager.page_count().unwrap(), 2);

    // The header's offset-56 code decodes to the declared UTF-16 encoding (§1.3.13).
    let raw = read_file(&dir.db("u16.db"));
    assert_eq!(&raw[56..60], &encoding_code.to_be_bytes(), "text encoding @56 (§1.3.13)");
    assert_eq!(header_of(&raw).text_encoding, encoding_code);
    assert_eq!(minisqlite_pager::text_encoding_of(&pager), enc, "pager reports the encoding");

    // The schema row (page 1) decodes its UTF-16 TEXT columns to the expected UTF-8.
    let schema = scan_table_enc(&pager, 1, enc);
    assert_eq!(schema.len(), 1, "one schema row");
    let cols = &schema[0].1;
    assert!(value_eq(&cols[0], &Value::Text("table".into())), "type: {:?}", cols[0]);
    assert!(value_eq(&cols[1], &Value::Text("t".into())), "name: {:?}", cols[1]);
    assert!(value_eq(&cols[3], &Value::Integer(2)), "rootpage: {:?}", cols[3]);
    assert!(
        value_eq(&cols[4], &Value::Text("CREATE TABLE t(c TEXT)".into())),
        "sql: {:?}",
        cols[4]
    );

    // The table rows (page 2) decode their UTF-16 TEXT value to the expected UTF-8
    // string — including the surrogate-pair emoji, proving pairs recombine.
    let got = scan_table_enc(&pager, 2, enc);
    assert_eq!(got.len(), rows.len());
    for ((rowid, want), (grid, gcols)) in rows.iter().zip(&got) {
        assert_eq!(*grid, *rowid, "rowid in key order");
        assert!(
            value_eq(&gcols[0], &Value::Text((*want).into())),
            "row {rowid}: got {:?}, want {want:?}",
            gcols[0]
        );
    }

    // Decoding the SAME bytes as UTF-8 would NOT yield the strings (a UTF-16 "hi" has
    // an embedded NUL), so the encoding threading is what makes the read correct — not
    // an accident of ASCII.
    let as_utf8 = scan_table_enc(&pager, 2, TextEncoding::Utf8);
    assert!(
        !value_eq(&as_utf8[0].1[0], &Value::Text("hi".into())),
        "UTF-16 bytes must not read as the UTF-8 string"
    );
}

#[test]
fn utf16le_file_reads_back_utf8_strings() {
    utf16_reads_back_utf8(codec::TEXT_ENCODING_UTF16LE, TextEncoding::Utf16le);
}

#[test]
fn utf16be_file_reads_back_utf8_strings() {
    utf16_reads_back_utf8(codec::TEXT_ENCODING_UTF16BE, TextEncoding::Utf16be);
}

// ===================================================================================
// PART 2 — auto_vacuum file read
// ===================================================================================

/// The rows the auto_vacuum tests read back (plain ASCII — encoding is orthogonal to
/// the vacuum layout, tested separately above).
fn av_rows() -> [(i64, &'static str); 3] {
    [(1, "alpha"), (2, "beta"), (3, "gamma")]
}

/// An auto_vacuum database: page 1 header has a non-zero offset-52 (largest root), so
/// the file carries a ptrmap page at page 2 (§1.8); the table `t` root is page 3 (NOT
/// 2). Three pages total: [schema][ptrmap][table leaf].
fn autovacuum_file(page_size: u32) -> Vec<u8> {
    let root = 3u32;
    // Largest root b-tree page = 3 (the only user table's root); marks auto_vacuum.
    let header = header_autovacuum(page_size, 3, root);
    let mut file = schema_page1(page_size, header, TextEncoding::Utf8, root);
    // Page 2: ptrmap. Its only live entry is for page 3, a b-tree ROOT page (type 1,
    // §1.8), whose back-pointer is 0. Byte 0 of the page is therefore 0x01.
    file.extend_from_slice(&ptrmap_page(page_size, &[(3, 1, 0)]));
    // Page 3: the table `t` leaf.
    file.extend_from_slice(&table_leaf(page_size, TextEncoding::Utf8, &av_rows()));
    file
}

/// The SAME logical table with NO auto_vacuum: no ptrmap page, table root = page 2.
fn plain_file(page_size: u32) -> Vec<u8> {
    let header = spec_db_header(page_size, 2, 0, 0); // offset 52 = 0 → non-vacuum
    let mut file = schema_page1(page_size, header, TextEncoding::Utf8, 2);
    file.extend_from_slice(&table_leaf(page_size, TextEncoding::Utf8, &av_rows()));
    file
}

#[test]
fn autovacuum_header_flags_ptrmap_pages() {
    let ps = 512u32;
    let dir = TempDir::new("avh");
    let pager = open_bytes(&dir, "av.db", &autovacuum_file(ps));

    // Offset 52 non-zero ⇒ auto/incremental-vacuum ⇒ ptrmap pages present (§1.3.12).
    let raw = read_file(&dir.db("av.db"));
    assert_ne!(&raw[52..56], &[0u8; 4], "largest-root @52 non-zero (§1.3.12)");
    let hdr = minisqlite_pager::read_database_header(&pager).expect("valid header");
    assert!(hdr.uses_ptrmap(), "non-zero offset 52 marks a ptrmap database");

    // The §1.8 ptrmap positions: page 2 is the first (and, in a 3-page db, only)
    // ptrmap page; pages 1 and 3 are not.
    assert!(hdr.is_ptrmap_page(2), "page 2 is the first ptrmap page (§1.8)");
    assert!(!hdr.is_ptrmap_page(1), "page 1 is the header/schema root, never ptrmap");
    assert!(!hdr.is_ptrmap_page(3), "page 3 is the first data page after the ptrmap");
}

#[test]
fn autovacuum_ptrmap_page_is_not_a_btree_page() {
    let ps = 512u32;
    let dir = TempDir::new("avp");
    let pager = open_bytes(&dir, "av.db", &autovacuum_file(ps));
    let usable = minisqlite_pager::read_database_header(&pager).unwrap().usable_size();

    // In THIS fixture page 2's first byte is a ptrmap entry's page-type 1 (a b-tree ROOT,
    // §1.8), which is not a valid b-tree page-type flag (2/5/10/13), so parsing page 2 as
    // a b-tree page fails closed. This parse-failure is FIXTURE-SPECIFIC, not a general
    // property: a ptrmap page whose first tracked page were a freelist page (entry type 2)
    // or a non-root b-tree page (type 5) would have byte 0 = 0x02/0x05, which ARE valid
    // interior/leaf flags, so `PageView::new` could actually succeed on such a ptrmap page.
    // The REAL guarantee that a read never mis-parses a ptrmap page is that b-tree
    // traversal is pointer-driven and never reaches page 2 at all — proven by
    // `autovacuum_rows_identical_to_non_autovacuum`. This case only documents the
    // fail-closed behavior for the common root-page (type 1) ptrmap entry.
    assert_eq!(pager.read_page(2).unwrap()[0], 0x01, "page 2 byte 0 is ptrmap type 1");
    assert!(
        PageView::new(pager.read_page(2).unwrap(), 2, usable).is_err(),
        "this fixture's ptrmap page (first entry type 1) does not parse as a b-tree page"
    );
}

#[test]
fn autovacuum_file_reads_schema_and_rows() {
    let ps = 512u32;
    let dir = TempDir::new("avr");
    let pager = open_bytes(&dir, "av.db", &autovacuum_file(ps));
    assert_eq!(pager.page_count().unwrap(), 3, "schema + ptrmap + table leaf");

    // The schema row names table `t` with root page 3 — the auto_vacuum layout puts the
    // ptrmap at page 2, so the b-tree root is page 3, and the reader follows THAT.
    let schema = scan_table_enc(&pager, 1, TextEncoding::Utf8);
    assert_eq!(schema.len(), 1);
    assert!(value_eq(&schema[0].1[1], &Value::Text("t".into())), "table name");
    assert!(value_eq(&schema[0].1[3], &Value::Integer(3)), "rootpage 3, past the ptrmap");

    // The table b-tree at page 3 reads back the exact rows: the traversal follows
    // pointers from the root and never touches the ptrmap page 2.
    let got = scan_table_enc(&pager, 3, TextEncoding::Utf8);
    let want = av_rows();
    assert_eq!(got.len(), want.len());
    for ((rowid, s), (grid, gcols)) in want.iter().zip(&got) {
        assert_eq!(*grid, *rowid);
        assert!(value_eq(&gcols[0], &Value::Text((*s).into())), "row {rowid}: {:?}", gcols[0]);
    }
}

#[test]
fn autovacuum_rows_identical_to_non_autovacuum() {
    // The spec bar (§1.8): an auto_vacuum file must read the SAME schema and rows as
    // the equivalent plain file — the ptrmap pages are invisible to a b-tree read.
    let ps = 512u32;
    let dir = TempDir::new("avc");
    let av = open_bytes(&dir, "av.db", &autovacuum_file(ps));
    let plain = open_bytes(&dir, "plain.db", &plain_file(ps));

    // Schema rows equal (ignoring the differing rootpage: 3 with ptrmap, 2 without).
    let av_schema = scan_table_enc(&av, 1, TextEncoding::Utf8);
    let plain_schema = scan_table_enc(&plain, 1, TextEncoding::Utf8);
    assert_eq!(av_schema.len(), plain_schema.len());
    assert!(value_eq(&av_schema[0].1[1], &plain_schema[0].1[1]), "same table name");
    assert!(value_eq(&av_schema[0].1[4], &plain_schema[0].1[4]), "same CREATE TABLE sql");

    // Table rows are byte-for-byte the same logical values (roots 3 vs 2).
    let av_rows_read = scan_table_enc(&av, 3, TextEncoding::Utf8);
    let plain_rows_read = scan_table_enc(&plain, 2, TextEncoding::Utf8);
    assert_eq!(av_rows_read.len(), plain_rows_read.len());
    for ((ar, ac), (pr, pc)) in av_rows_read.iter().zip(&plain_rows_read) {
        assert_eq!(ar, pr, "same rowid");
        assert_row_eq(ac, pc, "auto_vacuum vs plain row");
    }
}
