//! Shared fixtures and helpers for the on-disk-format tests (`ondisk_format_read`,
//! `ondisk_format_write`).
//!
//! These tests validate the SQLite on-disk file format in BOTH directions —
//! reading `.db` bytes our reader must decode, and writing `.db` bytes our writer
//! must produce — WITHOUT invoking real `sqlite3`. The reference
//! is the documented format in `spec/sqlite-doc/fileformat2.html`.
//!
//! The fixtures here are hand-constructed BYTE SEQUENCES derived straight from that
//! spec, with every non-obvious byte justified in a comment citing its field/offset.
//! Deliberately, they do NOT go through our own header/page/record writers
//! (`DatabaseHeader::to_bytes`, `PageBuilder`, `encode_record`, the cell encoders) —
//! those are exactly what the write-direction tests check against these fixtures. A
//! fixture is thus an INDEPENDENT spec artifact, and each is cross-checked twice:
//!   * READ  — our reader decodes the fixture bytes into the expected logical values;
//!   * WRITE — our writer reproduces the fixture bytes from the logical values.
//! A transcription error in a fixture fails at least one direction, so the pair is a
//! genuine spec conformance check, not a self-referential loop.
//!
//! Pure spill ARITHMETIC (`codec::payload_split`) is used where the spec defines a
//! formula rather than a byte layout — it is separately and exhaustively pinned in
//! `minisqlite-fileformat`, and the write tests additionally pin its results to
//! hand-computed literals. (The overflow-page capacity is a trivial `page_size - 4`
//! and is computed inline in `f3_overflow_pages`, not via a codec helper.)

#![allow(dead_code)] // Each test binary uses a subset of these shared helpers.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use minisqlite_pager::codec::{
    payload_split, CellKind, DatabaseHeader, PageBuilder, PageType, DEFAULT_PAGE_SIZE, HEADER_SIZE,
};
use minisqlite_pager::{DiskPager, PageId, Pager};
use minisqlite_types::Value;

/// The three page sizes the byte-exact fixtures sweep (the smallest, the default,
/// and the largest the format allows). Every legal page size is a power of two in
/// `[512, 65536]`; these three exercise the header page-size field (including the
/// 65536 -> stored `1` sentinel) and the empty-page cell-content sentinel.
pub const FIXTURE_PAGE_SIZES: [u32; 3] = [512, 4096, 65536];

// ===================================================================================
// Temp files
// ===================================================================================

/// A unique temp directory for a test's `.db` (plus any `-journal`/`-wal`) files,
/// removed on drop so nothing leaks into the shared temp dir. Uniqueness is pid + a
/// process-wide counter + nanos, so parallel tests never collide.
pub struct TempDir {
    path: PathBuf,
}

impl TempDir {
    pub fn new(tag: &str) -> TempDir {
        static N: AtomicU32 = AtomicU32::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
        let mut path = std::env::temp_dir();
        path.push(format!("msodf-{tag}-{pid}-{n}-{nanos}"));
        std::fs::create_dir_all(&path).expect("create per-test temp dir");
        TempDir { path }
    }

    pub fn db(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

// ===================================================================================
// Spec-literal byte builders (independent of our codec's writers)
// ===================================================================================

/// Encode a SQLite varint (fileformat2 §1.6): a big-endian static Huffman encoding,
/// 1..=9 bytes. Every byte but the last of a multi-byte varint has its high bit set
/// and contributes its low 7 bits, most-significant group first; the 9th byte (used
/// only when the value needs >56 bits) contributes all 8 bits. This is an
/// independent re-implementation from the spec text — the write tests compare our
/// crate's `write_varint` output against fixtures built with THIS encoder, and
/// `spec_varint_matches_known_encodings` pins it against literal spec examples.
pub fn spec_varint(v: u64) -> Vec<u8> {
    // Values with any of bits 56..=63 set do not fit eight 7-bit groups and take the
    // 9-byte form: eight continuation bytes then a full 8-bit final byte.
    if v & 0xff00_0000_0000_0000 != 0 {
        let mut out = vec![0u8; 9];
        out[8] = v as u8;
        let mut rest = v >> 8;
        for slot in out[..8].iter_mut().rev() {
            *slot = (rest as u8 & 0x7f) | 0x80;
            rest >>= 7;
        }
        return out;
    }
    // Emit 7-bit groups, most-significant first; every group but the last carries the
    // 0x80 continuation flag.
    let mut groups = vec![(v & 0x7f) as u8];
    let mut rest = v >> 7;
    while rest != 0 {
        groups.push((rest & 0x7f) as u8);
        rest >>= 7;
    }
    groups.reverse();
    let last = groups.len() - 1;
    for g in &mut groups[..last] {
        *g |= 0x80;
    }
    groups
}

/// A hand-built 100-byte database header (fileformat2 §1.3), laid out at LITERAL
/// documented offsets. Fixed fields carry the spec-mandated constants; the caller
/// supplies the page size, the in-header page count, and the freelist head/count.
///
/// `file_change_counter` (offset 24) and `version-valid-for` (offset 92) are both 0,
/// so they are equal and the in-header database size at offset 28 is VALID (§1.3.7) —
/// meaning our reader trusts `db_size_pages` for the page count.
pub fn spec_db_header(
    page_size: u32,
    db_size_pages: u32,
    first_freelist_trunk: u32,
    freelist_count: u32,
) -> [u8; 100] {
    let mut h = [0u8; 100];
    // offset 0, 16 bytes: the magic string "SQLite format 3\0" (§1.3.1).
    h[0..16].copy_from_slice(b"SQLite format 3\0");
    // offset 16, u16 big-endian: page size; 65536 is stored as the value 1 (§1.3.2).
    let ps_field: u16 = if page_size == 65_536 { 1 } else { page_size as u16 };
    h[16..18].copy_from_slice(&ps_field.to_be_bytes());
    h[18] = 1; // offset 18: file format write version = 1 (rollback journal, §1.3.3).
    h[19] = 1; // offset 19: file format read version  = 1 (rollback journal).
    h[20] = 0; // offset 20: bytes of reserved space per page = 0 (§1.3.4).
    h[21] = 64; // offset 21: maximum embedded payload fraction, must be 64 (§1.3.5).
    h[22] = 32; // offset 22: minimum embedded payload fraction, must be 32.
    h[23] = 32; // offset 23: leaf payload fraction, must be 32.
    // offset 24, u32: file change counter = 0.
    h[28..32].copy_from_slice(&db_size_pages.to_be_bytes()); // offset 28: in-header size (§1.3.7).
    h[32..36].copy_from_slice(&first_freelist_trunk.to_be_bytes()); // offset 32: first trunk (§1.3.8).
    h[36..40].copy_from_slice(&freelist_count.to_be_bytes()); // offset 36: total freelist pages.
    // offset 40, u32: schema cookie = 0.
    h[44..48].copy_from_slice(&4u32.to_be_bytes()); // offset 44: schema format number = 4 (§1.3.10).
    // offset 48: default page cache size = 0.
    // offset 52: largest root b-tree page = 0 (not auto/incremental-vacuum).
    h[56..60].copy_from_slice(&1u32.to_be_bytes()); // offset 56: text encoding = 1 (UTF-8).
    // offsets 60/64/68: user version / incremental-vacuum flag / application id = 0.
    // offsets 72..92: reserved for expansion, must be zero (§1.3.17).
    // offset 92, u32: version-valid-for = 0 (== change counter -> in-header size valid).
    // offset 96, u32: SQLITE_VERSION_NUMBER = 0.
    h
}

/// Assemble one b-tree page (fileformat2 §1.6), packing `cells` toward the END of the
/// usable region and writing the cell-pointer array in the given (key) order — the
/// exact layout the spec prescribes and our `PageBuilder` produces (cells packed at
/// the end, pointers in key order). `header_at` is 100 for page 1 (whose first 100
/// bytes hold the database header) and 0 for every other page. Reserved space is 0,
/// so the usable size equals `page_size`.
///
/// Cell physical placement within the content area is left arbitrary by the spec
/// ("the actual location of keys within the page is arbitrary"); this packs them from
/// the end in the order given, which is what our writer does, so the write-direction
/// byte-exact check is deterministic. Any spec reader (ours or real SQLite) locates
/// cells via the pointer array, not their physical order.
pub fn spec_pack_page(
    page_size: u32,
    header_at: usize,
    page_type_byte: u8,
    right_most: Option<u32>,
    cells: &[Vec<u8>],
) -> Vec<u8> {
    let ps = page_size as usize;
    let mut buf = vec![0u8; ps];
    let header_len = if right_most.is_some() { 12 } else { 8 };

    // Cell content grows DOWN from the end of the usable region (= page size here).
    let mut content_top = ps;
    let mut pointers: Vec<u16> = Vec::with_capacity(cells.len());
    for cell in cells {
        content_top -= cell.len();
        buf[content_top..content_top + cell.len()].copy_from_slice(cell);
        pointers.push(content_top as u16);
    }

    // B-tree page header (§1.6).
    buf[header_at] = page_type_byte; // offset 0: page type flag.
    // offset 1, u16: first freeblock = 0 (a freshly built page has none).
    buf[header_at + 3..header_at + 5].copy_from_slice(&(cells.len() as u16).to_be_bytes()); // offset 3: cell count.
    // offset 5, u16: cell content area start; a value of 0 means 65536 (empty 64 KiB page).
    let content_field: u16 = if content_top >= 65_536 { 0 } else { content_top as u16 };
    buf[header_at + 5..header_at + 7].copy_from_slice(&content_field.to_be_bytes());
    buf[header_at + 7] = 0; // offset 7: fragmented free bytes = 0.
    if let Some(rm) = right_most {
        // offset 8, u32: right-most child pointer (interior pages only, §1.6).
        buf[header_at + 8..header_at + 12].copy_from_slice(&rm.to_be_bytes());
    }

    // Cell pointer array (§1.6): K 2-byte big-endian offsets, in key order.
    let cpa = header_at + header_len;
    for (i, &p) in pointers.iter().enumerate() {
        buf[cpa + i * 2..cpa + i * 2 + 2].copy_from_slice(&p.to_be_bytes());
    }
    buf
}

/// A table-leaf cell (fileformat2 §1.6, header 0x0d): `varint(P)`, `varint(rowid)`,
/// then the `record` payload. `record.len()` is the total payload P (no overflow —
/// this helper is for rows that fit inline; the spilled F3 cell is built explicitly).
pub fn spec_table_leaf_cell(rowid: i64, record: &[u8]) -> Vec<u8> {
    let mut c = spec_varint(record.len() as u64);
    c.extend_from_slice(&spec_varint(rowid as u64));
    c.extend_from_slice(record);
    c
}

/// A table-interior cell (fileformat2 §1.6, header 0x05): a 4-byte big-endian left
/// child page number, then `varint(rowid)` (the separator key).
pub fn spec_table_interior_cell(left_child: u32, rowid: i64) -> Vec<u8> {
    let mut c = left_child.to_be_bytes().to_vec();
    c.extend_from_slice(&spec_varint(rowid as u64));
    c
}

/// A freelist trunk page (fileformat2 §1.5): an array of 4-byte big-endian integers —
/// `int[0]` = next trunk page (0 if last), `int[1]` = L (leaf count), then L leaf
/// page numbers. The rest of the usable area is unused (left zero).
pub fn spec_freelist_trunk(page_size: u32, next: u32, leaves: &[u32]) -> Vec<u8> {
    let mut buf = vec![0u8; page_size as usize];
    buf[0..4].copy_from_slice(&next.to_be_bytes());
    buf[4..8].copy_from_slice(&(leaves.len() as u32).to_be_bytes());
    for (i, &leaf) in leaves.iter().enumerate() {
        buf[8 + i * 4..8 + i * 4 + 4].copy_from_slice(&leaf.to_be_bytes());
    }
    buf
}

/// Page 1 of a valid database whose sole b-tree is an EMPTY `sqlite_schema` leaf: the
/// 100-byte header over an empty leaf table b-tree page. Used as page 1 of the
/// multi-page fixtures (F2/F3/F4/F5) whose data lives on later pages.
pub fn spec_empty_schema_page1(
    page_size: u32,
    db_size_pages: u32,
    first_freelist_trunk: u32,
    freelist_count: u32,
) -> Vec<u8> {
    // An empty leaf table page (type 0x0d, 0 cells) with the b-tree header at offset
    // 100, then the database header overlaid on the first 100 bytes.
    let mut page = spec_pack_page(page_size, 100, 0x0d, None, &[]);
    let hdr = spec_db_header(page_size, db_size_pages, first_freelist_trunk, freelist_count);
    page[0..100].copy_from_slice(&hdr);
    page
}

// ===================================================================================
// Value comparison (Value has no PartialEq; Real compares by bit pattern)
// ===================================================================================

pub fn value_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Null, Value::Null) => true,
        (Value::Integer(x), Value::Integer(y)) => x == y,
        (Value::Real(x), Value::Real(y)) => x.to_bits() == y.to_bits(),
        (Value::Text(x), Value::Text(y)) => x == y,
        (Value::Blob(x), Value::Blob(y)) => x == y,
        _ => false,
    }
}

pub fn assert_row_eq(got: &[Value], want: &[Value], ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: column count {got:?} vs {want:?}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(value_eq(g, w), "{ctx}: column {i}: {g:?} != {w:?}");
    }
}

// ===================================================================================
// Disk helpers
// ===================================================================================

/// Read a whole `.db` file into memory (fixtures are small — the largest is a
/// 65536-byte page or two).
pub fn read_file(path: &Path) -> Vec<u8> {
    std::fs::read(path).expect("read db file")
}

/// The `page_size`-byte slice for 1-based page `id` in a raw database file.
pub fn page_slice(raw: &[u8], id: u32, page_size: u32) -> &[u8] {
    let ps = page_size as usize;
    let off = (id as usize - 1) * ps;
    &raw[off..off + ps]
}

/// Decode page 1's database header from raw file bytes (validates the magic and page
/// size via our reader).
pub fn header_of(raw: &[u8]) -> DatabaseHeader {
    let mut buf = [0u8; HEADER_SIZE];
    buf.copy_from_slice(&raw[..HEADER_SIZE]);
    DatabaseHeader::read(&buf).expect("valid database header")
}

/// Page 1 of an empty database as OUR writers produce it (the codec composition the
/// F1 write-direction test checks byte-for-byte against the hand-built fixture):
/// `DatabaseHeader::to_bytes()` overlaid on an empty `PageBuilder` leaf page.
pub fn our_empty_db_page1(page_size: u32) -> Vec<u8> {
    let usable = page_size as usize; // reserved 0
    let mut page = PageBuilder::new(page_size as usize, usable, 1, PageType::LeafTable).finish();
    let hdr = DatabaseHeader { page_size, ..DatabaseHeader::default() };
    page[0..HEADER_SIZE].copy_from_slice(&hdr.to_bytes());
    page
}

/// Open a fresh on-disk database at `page_size`, formatted with an empty
/// `sqlite_schema` on page 1, ready for `create_table_btree` / `table_insert`.
///
/// `DiskPager` fixes a brand-new file to [`DEFAULT_PAGE_SIZE`], so for any other size
/// we first seed a one-page file carrying only a page-size-declaring header, then open
/// and (re)write page 1 as a proper empty schema leaf inside a committed transaction —
/// so every byte the file keeps is one our pager committed. (This is TEST SETUP for the
/// later table pages; the write-direction assertions only inspect pages >= 2.)
pub fn fresh_disk_db(path: &Path, page_size: u32) -> DiskPager {
    if page_size != DEFAULT_PAGE_SIZE {
        let mut seed = vec![0u8; page_size as usize];
        seed[..HEADER_SIZE].copy_from_slice(&spec_db_header(page_size, 1, 0, 0));
        std::fs::write(path, &seed).expect("seed page-size header");
    }
    let mut p = DiskPager::open(path).expect("open fresh db");
    assert_eq!(p.page_size(), page_size, "pager adopts the intended page size");
    p.begin().unwrap();
    if p.page_count().unwrap() == 0 {
        assert_eq!(p.allocate_page().unwrap(), 1, "fresh db allocates page 1 first");
    }
    p.write_page(1, &our_empty_db_page1(page_size)).unwrap();
    p.commit().unwrap();
    p
}

/// Full forward scan of the table b-tree rooted at `root`, decoding each row's
/// record into `Value`s. Drives `TableCursor::{first,next,rowid,payload}` and the
/// record decoder — the read side of a table.
pub fn scan_table(pager: &dyn Pager, root: PageId) -> Vec<(i64, Vec<Value>)> {
    use minisqlite_btree::TableCursor;
    let mut cur = TableCursor::open(pager, root).expect("open table cursor");
    let mut out = Vec::new();
    let mut positioned = cur.first().expect("cursor first");
    while positioned {
        let payload = cur.payload().expect("cursor payload");
        out.push((cur.rowid(), minisqlite_pager::codec::decode_record(&payload)));
        positioned = cur.next().expect("cursor next");
    }
    out
}

/// The largest rowid reachable in the table subtree rooted at `page`: descend the
/// right spine to the last leaf and take its last cell. Used to check the interior
/// separator invariant (§1.6) on writer-built trees.
pub fn max_rowid(pager: &dyn Pager, page: PageId, usable: usize) -> i64 {
    use minisqlite_pager::codec::PageView;
    let view = PageView::new(pager.read_page(page).unwrap(), page, usable).unwrap();
    if view.page_type().is_leaf() {
        let last = view.cell_count() as usize - 1;
        view.table_leaf_cell(last).unwrap().rowid
    } else {
        let child = view.right_most_pointer().unwrap();
        max_rowid(pager, child, usable)
    }
}

/// The smallest rowid reachable in the table subtree rooted at `page`: descend the
/// left spine to the first leaf and take its first cell.
pub fn min_rowid(pager: &dyn Pager, page: PageId, usable: usize) -> i64 {
    use minisqlite_pager::codec::PageView;
    let view = PageView::new(pager.read_page(page).unwrap(), page, usable).unwrap();
    if view.page_type().is_leaf() {
        view.table_leaf_cell(0).unwrap().rowid
    } else {
        // Left-most child is the left_child of cell 0.
        let child = view.table_interior_cell(0).unwrap().left_child;
        min_rowid(pager, child, usable)
    }
}

// ===================================================================================
// Fixtures
// ===================================================================================

// ---- F1: empty database -----------------------------------------------------------

/// F1 — an empty database: a single page holding the 100-byte header followed by an
/// empty leaf table b-tree page (the empty `sqlite_schema`). Fully hand-built.
pub fn f1_empty_db(page_size: u32) -> Vec<u8> {
    spec_empty_schema_page1(page_size, 1, 0, 0)
}

// ---- F2: several rows on one leaf --------------------------------------------------

/// The logical rows of F2, as `(rowid, columns)`. The first column is an INTEGER
/// PRIMARY KEY (rowid alias), which the record stores as NULL (§2.3) — its value is
/// the rowid. The columns exercise the full spread of serial types: NULL(0), the
/// integer constants 0(8) and 1(9), i8(1), i16(2), i24(3), i32(4), i48(5), i64(6),
/// TEXT (odd >=13), BLOB (even >=12, incl. empty), and REAL(7) — so the on-disk
/// byte-exact check covers every integer WIDTH, not just the fileformat crate's own
/// unit tests.
pub fn f2_rows() -> Vec<(i64, Vec<Value>)> {
    vec![
        (
            1,
            vec![
                Value::Null,             // rowid alias -> NULL
                Value::Integer(0),       // serial 8
                Value::Integer(1),       // serial 9
                Value::Text("a".into()), // serial 15
                Value::Blob(vec![0xFF]), // serial 14
            ],
        ),
        (
            2,
            vec![
                Value::Null,              // rowid alias -> NULL
                Value::Integer(200),      // serial 2 (i16): 200 > i8 max
                Value::Text("xy".into()), // serial 17
                Value::Blob(vec![]),      // serial 12 (empty blob)
                Value::Integer(-1),       // serial 1 (i8)
            ],
        ),
        (
            3,
            vec![
                Value::Null,           // rowid alias -> NULL
                Value::Integer(-1000), // serial 2 (i16)
                Value::Real(1.5),      // serial 7 (f64)
                Value::Text("".into()), // serial 13 (empty text)
                Value::Null,           // serial 0
            ],
        ),
        (
            // The wider integer widths (serial 3/4/5/6), each chosen just past the
            // previous width's max so the space-optimal serial type is unambiguous.
            4,
            vec![
                Value::Null,                       // rowid alias -> NULL
                Value::Integer(0x12_3456),         // serial 3 (i24): > i16 max
                Value::Integer(0x1234_5678),       // serial 4 (i32): > i24 max
                Value::Integer(0x1234_5678_9ABC),  // serial 5 (i48): > i32 max
                Value::Integer(0x1234_5678_9ABC_DEF0), // serial 6 (i64): > i48 max
            ],
        ),
    ]
}

/// F2 page-2 leaf, hand-built from the spec. The three records (§2.1) are written as
/// explicit byte literals so every byte is checkable against fileformat2.html.
pub fn f2_leaf_page(page_size: u32) -> Vec<u8> {
    // Row rowid=1: [NULL, INTEGER 0, INTEGER 1, TEXT "a", BLOB {FF}]
    //   record header: len=6 (§2.1, header counts itself), serials 0,8,9,15,14
    //   record body:   TEXT "a" = 0x61, BLOB byte 0xFF
    let rec1: &[u8] = &[0x06, 0x00, 0x08, 0x09, 0x0F, 0x0E, 0x61, 0xFF];
    // Row rowid=2: [NULL, INTEGER 200, TEXT "xy", BLOB {}, INTEGER -1]
    //   header: len=6, serials 0, 2 (i16), 17 (text 2B), 12 (blob 0B), 1 (i8)
    //   body:   200 -> big-endian i16 0x00C8; "xy" = 0x78 0x79; empty blob (0B); -1 -> i8 0xFF
    let rec2: &[u8] = &[0x06, 0x00, 0x02, 0x11, 0x0C, 0x01, 0x00, 0xC8, 0x78, 0x79, 0xFF];
    // Row rowid=3: [NULL, INTEGER -1000, REAL 1.5, TEXT "", NULL]
    //   header: len=6, serials 0, 2 (i16), 7 (f64), 13 (text 0B), 0 (NULL)
    //   body:   -1000 -> i16 0xFC18; 1.5 -> f64 big-endian 0x3FF8000000000000; empty text; NULL (0B)
    let rec3: &[u8] = &[
        0x06, 0x00, 0x02, 0x07, 0x0D, 0x00, // header
        0xFC, 0x18, // -1000
        0x3F, 0xF8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 1.5
    ];
    // Row rowid=4: [NULL, INTEGER i24, INTEGER i32, INTEGER i48, INTEGER i64]
    //   header: len=6, serials 0, 3 (i24), 4 (i32), 5 (i48), 6 (i64)
    //   body (all big-endian, minimal-width two's complement, §2.1):
    //     0x123456           -> i24 12 34 56
    //     0x12345678         -> i32 12 34 56 78
    //     0x123456789ABC     -> i48 12 34 56 78 9A BC
    //     0x123456789ABCDEF0 -> i64 12 34 56 78 9A BC DE F0
    let rec4: &[u8] = &[
        0x06, 0x00, 0x03, 0x04, 0x05, 0x06, // header
        0x12, 0x34, 0x56, // i24
        0x12, 0x34, 0x56, 0x78, // i32
        0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, // i48
        0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, // i64
    ];

    let cells = vec![
        spec_table_leaf_cell(1, rec1),
        spec_table_leaf_cell(2, rec2),
        spec_table_leaf_cell(3, rec3),
        spec_table_leaf_cell(4, rec4),
    ];
    spec_pack_page(page_size, 0, 0x0d, None, &cells)
}

/// F2 full file: page 1 = empty schema (in-header size 2), page 2 = the table leaf.
pub fn f2_file(page_size: u32) -> Vec<u8> {
    let mut file = spec_empty_schema_page1(page_size, 2, 0, 0);
    file.extend_from_slice(&f2_leaf_page(page_size));
    file
}

// ---- F3: a row that spills to overflow pages (page size 512) -----------------------

/// F3 is built at a 512-byte page size, where a modest payload overflows.
pub const F3_PAGE_SIZE: u32 = 512;
/// Total payload P of the F3 row (a record wrapping one large BLOB).
pub const F3_PAYLOAD_LEN: usize = 1500;
/// The single BLOB column's byte length: P minus the 3-byte record header
/// (`[len=3][serial=3006][..]`), where serial 3006 = 12 + 2*1497 (§2.1).
pub const F3_BLOB_LEN: usize = 1497;

/// Deterministic, position-dependent BLOB content, so reassembly ORDER is checked (a
/// constant fill would pass even if chunks were swapped).
pub fn f3_blob() -> Vec<u8> {
    (0..F3_BLOB_LEN).map(|i| ((i * 31 + 7) % 251) as u8).collect()
}

/// The F3 record (§2.1): header `[len=3][serial 3006]` then the BLOB body. Built from
/// the spec varint encoder so it is independent of our record writer.
pub fn f3_record() -> Vec<u8> {
    // serial for a BLOB of F3_BLOB_LEN bytes: 12 + 2*len (§2.1, even N>=12).
    let serial = 12 + 2 * F3_BLOB_LEN as u64; // = 3006
    // header length varint counts itself + the one serial varint = 1 + 2 = 3.
    let mut rec = spec_varint(3);
    rec.extend_from_slice(&spec_varint(serial));
    rec.extend_from_slice(&f3_blob());
    assert_eq!(rec.len(), F3_PAYLOAD_LEN); // plain assert: holds under --release too
    rec
}

/// The inline/overflow split of the F3 payload, from the spec formula (§1.6). Callers
/// pin the returned values to the hand-computed literals (local 39, overflow 1461).
pub fn f3_split() -> (usize, usize) {
    let s = payload_split(F3_PAGE_SIZE as usize, F3_PAYLOAD_LEN as u64, CellKind::TableLeaf);
    (s.local, s.overflow)
}

/// F3 page-2 leaf: one table-leaf cell for a spilled row (§1.6, header 0x0d):
/// `varint(P)`, `varint(rowid)`, the inline payload prefix, then the 4-byte first
/// overflow page number. The chain starts at page 3 (page 1 = schema, page 2 = this
/// leaf), matching the allocation order the writer uses.
pub fn f3_leaf_page() -> Vec<u8> {
    let record = f3_record();
    let (local, _overflow) = f3_split();
    let mut cell = spec_varint(F3_PAYLOAD_LEN as u64); // varint(P = 1500)
    cell.extend_from_slice(&spec_varint(1)); // varint(rowid = 1)
    cell.extend_from_slice(&record[..local]); // inline payload prefix
    cell.extend_from_slice(&3u32.to_be_bytes()); // first overflow page = 3 (§1.6 last cell field)
    spec_pack_page(F3_PAGE_SIZE, 0, 0x0d, None, &[cell])
}

/// The three overflow pages of F3 (page numbers 3, 4, 5). Each overflow page
/// (fileformat2 §1.7) is `[4-byte big-endian next page (0 on the last)][content]`;
/// the usable content capacity is `page_size - 4` = 508 bytes. The last page's
/// content is shorter than the capacity, so its tail is zero-padded.
pub fn f3_overflow_pages() -> [Vec<u8>; 3] {
    let record = f3_record();
    let (local, overflow) = f3_split();
    let cap = F3_PAGE_SIZE as usize - 4; // 508 (§1.7)
    let tail = &record[local..]; // the spilled bytes (P - local)
    assert_eq!(tail.len(), overflow);

    let page = |next: u32, chunk: &[u8]| -> Vec<u8> {
        let mut p = vec![0u8; F3_PAGE_SIZE as usize];
        p[0..4].copy_from_slice(&next.to_be_bytes());
        p[4..4 + chunk.len()].copy_from_slice(chunk);
        p
    };
    // 1461 bytes -> 508 + 508 + 445 across pages 3 -> 4 -> 5 -> 0.
    let c0 = &tail[0..cap];
    let c1 = &tail[cap..2 * cap];
    let c2 = &tail[2 * cap..];
    [page(4, c0), page(5, c1), page(0, c2)]
}

/// F3 full file: page 1 = empty schema (in-header size 5), page 2 = spill leaf,
/// pages 3-5 = the overflow chain.
pub fn f3_file() -> Vec<u8> {
    let mut file = spec_empty_schema_page1(F3_PAGE_SIZE, 5, 0, 0);
    file.extend_from_slice(&f3_leaf_page());
    for p in f3_overflow_pages() {
        file.extend_from_slice(&p);
    }
    file
}

// ---- F4: a database with a non-empty freelist -------------------------------------

/// F4 single-trunk freelist file: a 4-page database whose header names a trunk on
/// page 2 listing leaf pages 3 and 4 (freelist_count = trunk + 2 leaves = 3). The
/// freelist LAYOUT (page numbers, order) is our policy, not a spec byte contract, so
/// the read test checks the trunk's structure field-by-field rather than byte-identity;
/// swept over page sizes so the trunk decoder is exercised at 512/4096/65536.
pub fn f4_single_trunk_file(page_size: u32) -> Vec<u8> {
    let ps = page_size;
    // Header: first freelist trunk = page 2, total freelist pages = 3, in-header size 4.
    let mut file = spec_empty_schema_page1(ps, 4, 2, 3);
    // Page 2: trunk, next = 0 (last trunk), leaves = [3, 4].
    file.extend_from_slice(&spec_freelist_trunk(ps, 0, &[3, 4]));
    // Pages 3 and 4: freelist leaf pages (§1.5 says they carry no information).
    file.extend_from_slice(&vec![0u8; ps as usize]); // page 3
    file.extend_from_slice(&vec![0u8; ps as usize]); // page 4
    file
}

/// F4 chained-trunk freelist file: a 5-page database whose freelist is TWO trunk
/// pages — page 2 (leaves 3,4) chained to page 5 (no leaves). Total freelist pages =
/// 2 trunks + 2 leaves = 4. Swept over page sizes like the single-trunk fixture.
pub fn f4_chained_trunk_file(page_size: u32) -> Vec<u8> {
    let ps = page_size;
    let mut file = spec_empty_schema_page1(ps, 5, 2, 4);
    file.extend_from_slice(&spec_freelist_trunk(ps, 5, &[3, 4])); // page 2: next = 5
    file.extend_from_slice(&vec![0u8; ps as usize]); // page 3: leaf
    file.extend_from_slice(&vec![0u8; ps as usize]); // page 4: leaf
    file.extend_from_slice(&spec_freelist_trunk(ps, 0, &[])); // page 5: last trunk, no leaves
    file
}

// ---- F5: interior + leaf b-tree pages ---------------------------------------------

/// The F5 logical rows: rowids 1..=5, each a single INTEGER column valued `rowid*11`
/// (distinct from the rowid, so a rowid/value confusion is caught). All fit i8.
pub fn f5_rows() -> Vec<(i64, Value)> {
    (1..=5).map(|r| (r, Value::Integer(r * 11))).collect()
}

/// A one-column record `[INTEGER v]` for small `v` (i8): header `[len=2][serial 1]`
/// then the 1-byte body (§2.1).
fn f5_record(v: i64) -> Vec<u8> {
    assert!((-128..=127).contains(&v) && v != 0 && v != 1, "F5 values are plain i8 (serial 1)");
    vec![0x02, 0x01, v as u8]
}

/// The F5 rowid at which the (single) interior divider splits the tree: the max rowid
/// of the left leaf (page 3), per the separator invariant (§1.6).
pub const F5_SEPARATOR_ROWID: i64 = 3;

/// F5 full file: a 4-page database whose table (root = page 2) is an INTERIOR table
/// page over two leaves. The interior LAYOUT (where the split falls) is our policy,
/// so this hand-built read fixture pins a valid interior shape rather than the writer's
/// exact split; swept over page sizes so the interior/right-most-pointer decode path is
/// exercised at 512/4096/65536 (a real writer would need thousands of rows to force an
/// interior root at 65536, which the READ direction covers directly here instead).
///   page 1: empty schema (in-header size 4)
///   page 2: interior table page — right-most pointer = 4, one divider cell (left
///           child 3, key = 3) so leaf 3 holds rowids <= 3 and leaf 4 holds rowids > 3
///   page 3: leaf with rowids 1,2,3
///   page 4: leaf with rowids 4,5
pub fn f5_file(page_size: u32) -> Vec<u8> {
    let ps = page_size;
    let mut file = spec_empty_schema_page1(ps, 4, 0, 0);

    // Page 2: interior table page (type 0x05). Right-most pointer -> page 4; one
    // divider cell (left child = page 3, separator rowid = 3 = max rowid in page 3).
    let divider = spec_table_interior_cell(3, F5_SEPARATOR_ROWID);
    file.extend_from_slice(&spec_pack_page(ps, 0, 0x05, Some(4), &[divider]));

    // Page 3: leaf with rowids 1,2,3.
    let leaf3 = vec![
        spec_table_leaf_cell(1, &f5_record(11)),
        spec_table_leaf_cell(2, &f5_record(22)),
        spec_table_leaf_cell(3, &f5_record(33)),
    ];
    file.extend_from_slice(&spec_pack_page(ps, 0, 0x0d, None, &leaf3));

    // Page 4: leaf with rowids 4,5.
    let leaf4 =
        vec![spec_table_leaf_cell(4, &f5_record(44)), spec_table_leaf_cell(5, &f5_record(55))];
    file.extend_from_slice(&spec_pack_page(ps, 0, 0x0d, None, &leaf4));

    file
}
