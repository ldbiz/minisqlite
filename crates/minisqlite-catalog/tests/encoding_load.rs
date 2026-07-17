//! Loading a UTF-16 database's schema through the real `SchemaCatalog::load` seam.
//!
//! `sqlite_schema` stores every object's type, name, and CREATE SQL as TEXT (§1.3.13),
//! so in a UTF-16 database those bytes are UTF-16 and `load` must transcode them to the
//! engine's UTF-8 before parsing the SQL — otherwise the CREATE TABLE fails to parse and
//! the database won't even open. This test pins that end of the read path in the catalog
//! crate itself (the pager-crate fixtures can't reach it — dependency direction): a
//! regression that dropped the `enc` threading in `load` (reverting `decode_record_enc`
//! to the UTF-8 `decode_record`) is caught here, not silently shipped.
//!
//! The page-1 bytes are hand-built from `spec/sqlite-doc/fileformat2.html` (§1.3.13 text
//! encoding, §2.1 records, §1.6 cells) via the crate's own public page/record writers —
//! never `sqlite3`. UTF-16 bodies use std `str::encode_utf16`,
//! the inverse of the decoder under test.

use minisqlite_catalog::{Catalog, SchemaCatalog};
use minisqlite_fileformat::{
    encode_table_leaf_cell, write_varint, DatabaseHeader, PageBuilder, PageType, TextEncoding,
    HEADER_SIZE, TEXT_ENCODING_UTF16BE, TEXT_ENCODING_UTF16LE,
};
use minisqlite_pager::{MemPager, Pager};

const PAGE_SIZE: u32 = 4096;

/// A stored TEXT column as `(serial_type, body)` for `s` in encoding `enc`: odd serial
/// `N = 13 + 2*len` (§2.1); the UTF-16 body lays each code unit down in the encoding's
/// byte order (std `encode_utf16` is the inverse of the decode under test).
fn text_field(s: &str, enc: TextEncoding) -> (u64, Vec<u8>) {
    let body: Vec<u8> = match enc {
        TextEncoding::Utf8 => s.as_bytes().to_vec(),
        TextEncoding::Utf16le => s.encode_utf16().flat_map(u16::to_le_bytes).collect(),
        TextEncoding::Utf16be => s.encode_utf16().flat_map(u16::to_be_bytes).collect(),
    };
    (13 + 2 * body.len() as u64, body)
}

/// Encode a record (§2.1): a self-counting header-length varint, each field's serial-type
/// varint, then the packed bodies (header length counts itself → fixed-point solve).
fn record(fields: &[(u64, Vec<u8>)]) -> Vec<u8> {
    let mut serials = Vec::new();
    for (serial, _) in fields {
        write_varint(*serial, &mut serials);
    }
    let mut len_bytes = 1usize;
    let header = loop {
        let mut h = Vec::new();
        let n = write_varint((len_bytes + serials.len()) as u64, &mut h);
        if n == len_bytes {
            break h;
        }
        len_bytes = n;
    };
    let mut rec = header;
    rec.extend_from_slice(&serials);
    for (_, body) in fields {
        rec.extend_from_slice(body);
    }
    rec
}

/// A table-leaf cell (§1.6) wrapping an inline `payload` for `rowid`.
fn leaf_cell(rowid: i64, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    encode_table_leaf_cell(payload.len() as u64, rowid, payload, None, &mut out);
    out
}

/// A `MemPager` whose page 1 is the 100-byte header (text encoding `code` at offset 56,
/// §1.3.13) over a `sqlite_schema` leaf holding a single `CREATE TABLE t(a TEXT, b TEXT)`
/// row (rooted at page 2), with every schema TEXT column encoded in `enc`.
fn utf16_schema_pager(enc: TextEncoding, code: u32) -> MemPager {
    let sql = "CREATE TABLE t(a TEXT, b TEXT)";
    let rec = record(&[
        text_field("table", enc),
        text_field("t", enc),
        text_field("t", enc),
        (1, vec![2u8]), // rootpage 2 (serial 1, i8)
        text_field(sql, enc),
    ]);
    let mut pb = PageBuilder::new(PAGE_SIZE as usize, PAGE_SIZE as usize, 1, PageType::LeafTable);
    assert!(pb.add_cell(&leaf_cell(1, &rec)), "schema cell fits page 1");
    let mut page1 = pb.finish();
    let header =
        DatabaseHeader { page_size: PAGE_SIZE, text_encoding: code, ..DatabaseHeader::default() };
    page1[..HEADER_SIZE].copy_from_slice(&header.to_bytes());

    let mut pager = MemPager::new(PAGE_SIZE);
    pager.begin().unwrap();
    assert_eq!(pager.allocate_page().unwrap(), 1, "page 1 first");
    pager.write_page(1, &page1).unwrap();
    pager.commit().unwrap();
    pager
}

fn utf16_schema_loads_as_utf8(code: u32, enc: TextEncoding) {
    let pager = utf16_schema_pager(enc, code);
    let mut cat = SchemaCatalog::new();
    cat.load(&pager).expect("load a schema stored in UTF-16");

    let t = cat.table("t").unwrap().expect("table t recovered from the UTF-16 schema");
    assert_eq!(t.name, "t", "table name decoded UTF-16 → UTF-8");
    assert_eq!(t.root_page, 2, "rootpage from the schema record");
    let cols: Vec<&str> = t.columns.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(cols, ["a", "b"], "columns recovered from the decoded UTF-16 CREATE SQL");
}

#[test]
fn utf16le_schema_loads_as_utf8() {
    utf16_schema_loads_as_utf8(TEXT_ENCODING_UTF16LE, TextEncoding::Utf16le);
}

#[test]
fn utf16be_schema_loads_as_utf8() {
    utf16_schema_loads_as_utf8(TEXT_ENCODING_UTF16BE, TextEncoding::Utf16be);
}
