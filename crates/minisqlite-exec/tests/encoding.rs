//! End-to-end read of a UTF-16 database through the engine's REAL read seam — the
//! `SchemaCatalog::load` schema decode and the `seq_scan` operator driven by
//! `StreamingExecutor` — proving the text-encoding threading reaches the production
//! cursors, not just the record decoder in isolation.
//!
//! The pager-crate fixtures (`ondisk_format_encoding_vacuum`) prove the record DECODER
//! transcodes UTF-16 (§1.3.13), but they RECONSTRUCT the read path with a local
//! `TableCursor` + `decode_record_enc`. A regression that left one production cursor on
//! the UTF-8 decode entrypoint (dropping the `enc` it should read from the header) would
//! keep every such test passing while `SELECT` from a UTF-16 file returned garbage. This
//! test closes that gap: it runs the actual `seq_scan` and `SchemaCatalog::load`, so the
//! threading is what makes it pass.
//!
//! The fixture bytes are hand-built from `spec/sqlite-doc/fileformat2.html` (§1.3.13
//! text encoding, §2.1 records, §1.6 cells) via the crate's OWN public page/record
//! writers — never `sqlite3`. UTF-16 bodies use the std
//! `str::encode_utf16`, the inverse of the decoder under test.

use minisqlite_catalog::{Catalog, SchemaCatalog};
use minisqlite_exec::{Executor, Runtime, StreamingExecutor};
use minisqlite_fileformat::{
    encode_table_leaf_cell, write_varint, DatabaseHeader, PageBuilder, PageType, TextEncoding,
    HEADER_SIZE, TEXT_ENCODING_UTF16BE, TEXT_ENCODING_UTF16LE,
};
use minisqlite_pager::{MemPager, Pager};
use minisqlite_plan::{Plan, PlanNode, SeqScan};
use minisqlite_types::Value;
use minisqlite_exec::PagerSet;
use minisqlite_types::DbIndex;

const PAGE_SIZE: u32 = 4096;

// ----- spec-literal fixture builders (independent of the codec under test) ----------

/// A stored TEXT column as `(serial_type, body)` for `s` in encoding `enc` (§1.3.13,
/// §2.1): odd serial `N = 13 + 2*len`; the UTF-16 body lays each code unit down in the
/// encoding's byte order. `str::encode_utf16` is the inverse of the UTF-16 decode under
/// test, so a fixture bug can't masquerade as a passing decode.
fn text_field(s: &str, enc: TextEncoding) -> (u64, Vec<u8>) {
    let body: Vec<u8> = match enc {
        TextEncoding::Utf8 => s.as_bytes().to_vec(),
        TextEncoding::Utf16le => s.encode_utf16().flat_map(u16::to_le_bytes).collect(),
        TextEncoding::Utf16be => s.encode_utf16().flat_map(u16::to_be_bytes).collect(),
    };
    (13 + 2 * body.len() as u64, body)
}

/// Encode a record (§2.1): a self-counting header-length varint, then each field's
/// serial-type varint, then the packed bodies. The header length counts itself, so it
/// is solved to a fixed point (a large header could need a 2-byte length varint).
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

/// A table-leaf cell (§1.6) wrapping an inline `payload` for `rowid`, via the crate's
/// own cell encoder (no overflow — the fixtures are tiny).
fn leaf_cell(rowid: i64, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    encode_table_leaf_cell(payload.len() as u64, rowid, payload, None, &mut out);
    out
}

/// The `sqlite_schema` record for `CREATE TABLE t(c TEXT)` rooted at page `root`, every
/// TEXT column encoded in `enc`: columns `(type, name, tbl_name, rootpage, sql)`.
fn schema_record(enc: TextEncoding, root: u8) -> Vec<u8> {
    record(&[
        text_field("table", enc),
        text_field("t", enc),
        text_field("t", enc),
        (1, vec![root]), // rootpage: serial 1 (i8)
        text_field("CREATE TABLE t(c TEXT)", enc),
    ])
}

/// A hand-built UTF-16 database in a `MemPager`: page 1 is the 100-byte header (text
/// encoding `code` at offset 56, §1.3.13) over the `sqlite_schema` leaf holding the one
/// `CREATE TABLE t` row; page 2 is the table `t` leaf, one single-column `[TEXT c]`
/// record per row (rowids in key order). The table root is page 2.
fn utf16_db(enc: TextEncoding, code: u32, rows: &[(i64, &str)]) -> MemPager {
    // Page 1: schema leaf (b-tree header at offset 100 for page 1), DB header overlaid.
    let mut pb = PageBuilder::new(PAGE_SIZE as usize, PAGE_SIZE as usize, 1, PageType::LeafTable);
    assert!(pb.add_cell(&leaf_cell(1, &schema_record(enc, 2))), "schema cell fits page 1");
    let mut page1 = pb.finish();
    let header =
        DatabaseHeader { page_size: PAGE_SIZE, text_encoding: code, ..DatabaseHeader::default() };
    page1[..HEADER_SIZE].copy_from_slice(&header.to_bytes());

    // Page 2: table `t` leaf, one `[TEXT c]` record per row in ascending rowid order.
    let mut pb2 = PageBuilder::new(PAGE_SIZE as usize, PAGE_SIZE as usize, 2, PageType::LeafTable);
    for (rowid, s) in rows {
        let payload = record(&[text_field(s, enc)]);
        assert!(pb2.add_cell(&leaf_cell(*rowid, &payload)), "row {rowid} cell fits page 2");
    }
    let page2 = pb2.finish();

    let mut pager = MemPager::new(PAGE_SIZE);
    pager.begin().unwrap();
    assert_eq!(pager.allocate_page().unwrap(), 1, "page 1 first");
    assert_eq!(pager.allocate_page().unwrap(), 2, "page 2 second");
    pager.write_page(1, &page1).unwrap();
    pager.write_page(2, &page2).unwrap();
    pager.commit().unwrap();
    pager
}

// ----- plan + executor harness (mirrors tests/exec.rs) ------------------------------

fn seqscan(table: &str, column_count: usize) -> PlanNode {
    PlanNode::SeqScan(SeqScan { db: DbIndex::MAIN, table: table.to_string(), column_count })
}

fn plan(root: PlanNode, columns: &[&str]) -> Plan {
    Plan {
        root,
        result_columns: columns.iter().map(|s| s.to_string()).collect(),
        ctes: Vec::new(),
        subqueries: Vec::new(),
        mutates: false,
        generated: Vec::new(),
    }
}

fn run(plan: &Plan, cat: &SchemaCatalog, pager: &mut MemPager) -> Vec<Vec<Value>> {
    let mut rt = Runtime::new();
    let mut exec = StreamingExecutor;
    let mut cur = exec.execute(plan, cat, PagerSet::One { db: DbIndex::MAIN, pager: &mut *pager }).unwrap();
    let mut out = Vec::new();
    while let Some(row) = cur.next_row(&mut rt).unwrap() {
        out.push(row);
    }
    out
}

fn text_of(v: &Value) -> &str {
    match v {
        Value::Text(s) => s,
        other => panic!("expected Text, got {other:?}"),
    }
}

// ----- the end-to-end assertions ----------------------------------------------------

/// Rows exercising ASCII, a BMP non-ASCII code unit (é), and a supplementary-plane
/// surrogate pair (😀) sandwiched in ASCII — so every UTF-16 decode branch runs through
/// the production `seq_scan`, not just the isolated decoder.
fn utf16_rows() -> [(i64, &'static str); 3] {
    [(1, "hi"), (2, "café"), (3, "a\u{1F600}z")]
}

fn utf16_reads_through_executor(code: u32, enc: TextEncoding) {
    let rows = utf16_rows();
    let mut pager = utf16_db(enc, code, &rows);

    // CATALOG threading (schemacatalog.rs `load` → `decode_record_enc`): the table name
    // and CREATE SQL are stored UTF-16 and must reload as UTF-8. A revert to the UTF-8
    // decode would corrupt the SQL and the CREATE TABLE would fail to parse (or name
    // the column wrong), so recovering column `c` proves the schema decoded correctly.
    let mut cat = SchemaCatalog::new();
    cat.load(&pager).expect("load schema from a UTF-16 page-1 header + record");
    let t = cat.table("t").unwrap().expect("table t present after loading UTF-16 schema");
    assert_eq!(t.name, "t", "table name decoded UTF-16 → UTF-8");
    assert_eq!(t.root_page, 2, "rootpage from the schema record");
    assert_eq!(t.columns.len(), 1, "one column parsed from the decoded CREATE SQL");
    assert_eq!(t.columns[0].name, "c", "column name recovered from the decoded CREATE SQL");

    // EXEC threading (scan.rs `seq_scan` → `decode_table_row_enc`): a full scan through
    // the real executor decodes each row's UTF-16 TEXT to UTF-8. A bare SeqScan emits
    // `[c, rowid]`; column 0 is the value.
    let got = run(&plan(seqscan("t", 1), &["c"]), &cat, &mut pager);
    assert_eq!(got.len(), rows.len(), "one row per stored cell");
    for ((rowid, want), row) in rows.iter().zip(&got) {
        assert_eq!(
            text_of(&row[0]),
            *want,
            "row {rowid}: UTF-16 stored TEXT transcoded to UTF-8 through seq_scan"
        );
    }
}

#[test]
fn utf16le_database_reads_through_executor_as_utf8() {
    utf16_reads_through_executor(TEXT_ENCODING_UTF16LE, TextEncoding::Utf16le);
}

#[test]
fn utf16be_database_reads_through_executor_as_utf8() {
    utf16_reads_through_executor(TEXT_ENCODING_UTF16BE, TextEncoding::Utf16be);
}
