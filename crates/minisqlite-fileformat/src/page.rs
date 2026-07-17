//! B-tree page header, cells, and a page builder (fileformat2 §1.6). A b-tree page
//! is laid out as: an optional 100-byte database header (page 1 only), the 8- or
//! 12-byte page header, the cell pointer array, unallocated space, then the cell
//! content area growing down from the end of the usable region.
//!
//! Reads BORROW from the page slice: a parsed cell's payload is a sub-slice of the
//! page, never a copy, so scanning a page allocates nothing per row. The 4 cell
//! shapes (table/index × leaf/interior) are parsed and built by dedicated
//! functions so a caller works with typed cells, not raw offsets.

use crate::bytes::{be16, be32, try_be32, write_be16, write_be32};
use crate::overflow::{payload_split, CellKind};
use crate::varint::{read_varint, write_varint};
use minisqlite_types::{Error, Result};

/// The b-tree page type byte at offset 0 of the page header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageType {
    /// 0x02 — interior index b-tree page.
    InteriorIndex,
    /// 0x05 — interior table b-tree page.
    InteriorTable,
    /// 0x0a — leaf index b-tree page.
    LeafIndex,
    /// 0x0d — leaf table b-tree page.
    LeafTable,
}

impl PageType {
    /// Decode the page-type byte; any other value is not a b-tree page.
    pub fn from_byte(b: u8) -> Option<PageType> {
        match b {
            0x02 => Some(PageType::InteriorIndex),
            0x05 => Some(PageType::InteriorTable),
            0x0a => Some(PageType::LeafIndex),
            0x0d => Some(PageType::LeafTable),
            _ => None,
        }
    }

    pub fn to_byte(self) -> u8 {
        match self {
            PageType::InteriorIndex => 0x02,
            PageType::InteriorTable => 0x05,
            PageType::LeafIndex => 0x0a,
            PageType::LeafTable => 0x0d,
        }
    }

    pub fn is_interior(self) -> bool {
        matches!(self, PageType::InteriorIndex | PageType::InteriorTable)
    }

    pub fn is_leaf(self) -> bool {
        !self.is_interior()
    }

    pub fn is_table(self) -> bool {
        matches!(self, PageType::InteriorTable | PageType::LeafTable)
    }

    pub fn is_index(self) -> bool {
        !self.is_table()
    }

    /// Header size: 8 bytes for a leaf page, 12 for an interior page (the extra 4
    /// hold the right-most child pointer).
    pub fn header_len(self) -> usize {
        if self.is_interior() { 12 } else { 8 }
    }
}

/// The b-tree page header at offset 0 within the header region (`is_leaf` pages
/// omit `right_most_pointer`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageHeader {
    pub page_type: PageType,
    /// Offset of the first freeblock, or 0 if none.
    pub first_freeblock: u16,
    pub cell_count: u16,
    /// Start of the cell content area; the stored value 0 means 65536.
    pub cell_content_start: u32,
    pub fragmented_free_bytes: u8,
    /// Right-most child pointer (interior pages only).
    pub right_most_pointer: Option<u32>,
}

/// A borrowing reader over one b-tree page. Holds the page bytes plus where the
/// header starts (100 on page 1, else 0) and the usable page size `U` (needed for
/// the payload-spill split).
#[derive(Debug, Clone, Copy)]
pub struct PageView<'a> {
    data: &'a [u8],
    header_offset: usize,
    page_type: PageType,
    usable_size: usize,
}

/// A parsed table-leaf cell (0x0d): rowid plus content, with the inline portion of
/// the payload borrowed from the page and the overflow chain head if it spilled.
#[derive(Debug, Clone, Copy)]
pub struct TableLeafCell<'a> {
    /// Total payload length P including any overflow.
    pub payload_len: u64,
    pub rowid: i64,
    /// Inline payload bytes stored on this page (a prefix of the full payload).
    pub local_payload: &'a [u8],
    /// First overflow page, if the payload spilled.
    pub overflow_page: Option<u32>,
}

/// A parsed table-interior cell (0x05): a child pointer and the separator rowid.
#[derive(Debug, Clone, Copy)]
pub struct TableInteriorCell {
    pub left_child: u32,
    pub rowid: i64,
}

/// A parsed index-leaf cell (0x0a): a key payload (the index record), inline
/// portion borrowed, with the overflow head if it spilled.
#[derive(Debug, Clone, Copy)]
pub struct IndexLeafCell<'a> {
    pub payload_len: u64,
    pub local_payload: &'a [u8],
    pub overflow_page: Option<u32>,
}

/// A parsed index-interior cell (0x02): a child pointer plus a key payload.
#[derive(Debug, Clone, Copy)]
pub struct IndexInteriorCell<'a> {
    pub left_child: u32,
    pub payload_len: u64,
    pub local_payload: &'a [u8],
    pub overflow_page: Option<u32>,
}

/// The header offset for a page: page 1 carries the 100-byte database header
/// first, so its b-tree header begins at byte 100; all other pages start at 0.
pub fn header_offset_for(page_number: u32) -> usize {
    if page_number == 1 { 100 } else { 0 }
}

impl<'a> PageView<'a> {
    /// Interpret `data` as page `page_number` with usable page size `usable_size`.
    /// Validates that the header and the full cell pointer array lie within the
    /// slice, so header-field and cell-pointer reads cannot go out of bounds.
    pub fn new(data: &'a [u8], page_number: u32, usable_size: usize) -> Result<PageView<'a>> {
        let header_offset = header_offset_for(page_number);
        let type_pos = header_offset;
        let type_byte = *data
            .get(type_pos)
            .ok_or_else(|| Error::Format("page too small for b-tree header".into()))?;
        let page_type = PageType::from_byte(type_byte)
            .ok_or_else(|| Error::Format(format!("invalid b-tree page type {type_byte:#x}")))?;
        let header_len = page_type.header_len();
        if header_offset + header_len > data.len() {
            return Err(Error::Format("page too small for b-tree header".into()));
        }
        let cell_count = be16(data, header_offset + 3) as usize;
        let ptr_array_end = header_offset + header_len + cell_count * 2;
        if ptr_array_end > data.len() {
            return Err(Error::Format("cell pointer array exceeds page".into()));
        }
        Ok(PageView { data, header_offset, page_type, usable_size })
    }

    pub fn page_type(&self) -> PageType {
        self.page_type
    }

    fn h(&self, rel: usize) -> usize {
        self.header_offset + rel
    }

    pub fn first_freeblock(&self) -> u16 {
        be16(self.data, self.h(1))
    }

    pub fn cell_count(&self) -> u16 {
        be16(self.data, self.h(3))
    }

    /// Start of the cell content area; the stored value 0 is reported as 65536
    /// (the format's sentinel for a full 64 KiB page).
    pub fn cell_content_start(&self) -> u32 {
        let v = be16(self.data, self.h(5));
        if v == 0 { 65536 } else { v as u32 }
    }

    pub fn fragmented_free_bytes(&self) -> u8 {
        self.data[self.h(7)]
    }

    /// The right-most child pointer, present only on interior pages.
    pub fn right_most_pointer(&self) -> Option<u32> {
        if self.page_type.is_interior() {
            Some(be32(self.data, self.h(8)))
        } else {
            None
        }
    }

    /// The full page header as a struct.
    pub fn header(&self) -> PageHeader {
        PageHeader {
            page_type: self.page_type,
            first_freeblock: self.first_freeblock(),
            cell_count: self.cell_count(),
            cell_content_start: self.cell_content_start(),
            fragmented_free_bytes: self.fragmented_free_bytes(),
            right_most_pointer: self.right_most_pointer(),
        }
    }

    fn cell_ptr_array_start(&self) -> usize {
        self.header_offset + self.page_type.header_len()
    }

    /// Byte offset of cell `index` within the page (from the cell pointer array).
    /// `index` must be `< cell_count()`.
    pub fn cell_pointer(&self, index: usize) -> usize {
        be16(self.data, self.cell_ptr_array_start() + index * 2) as usize
    }

    /// Iterate the cell content offsets in stored (key) order.
    pub fn cell_pointers(&self) -> impl Iterator<Item = usize> + '_ {
        let n = self.cell_count() as usize;
        (0..n).map(move |i| self.cell_pointer(i))
    }

    fn slice_from(&self, off: usize) -> Result<&'a [u8]> {
        self.data.get(off..).ok_or_else(|| Error::Format("cell offset past end of page".into()))
    }

    /// Split the payload at `content_off` into its inline slice and overflow head.
    fn split_payload(
        &self,
        content_off: usize,
        payload_len: u64,
        kind: CellKind,
    ) -> Result<(&'a [u8], Option<u32>)> {
        let split = payload_split(self.usable_size, payload_len, kind);
        let end = content_off + split.local;
        let local = self
            .data
            .get(content_off..end)
            .ok_or_else(|| Error::Format("cell payload past end of page".into()))?;
        let overflow_page = if split.has_overflow {
            Some(
                try_be32(self.data, end)
                    .ok_or_else(|| Error::Format("overflow pointer past end of page".into()))?,
            )
        } else {
            None
        };
        Ok((local, overflow_page))
    }

    /// Parse cell `index` as a table-leaf cell.
    pub fn table_leaf_cell(&self, index: usize) -> Result<TableLeafCell<'a>> {
        let off = self.cell_pointer(index);
        let (payload_len, n1) =
            read_varint(self.slice_from(off)?).ok_or_else(|| Error::Format("bad payload varint".into()))?;
        let (rowid, n2) = read_varint(self.slice_from(off + n1)?)
            .ok_or_else(|| Error::Format("bad rowid varint".into()))?;
        let content_off = off + n1 + n2;
        let (local_payload, overflow_page) =
            self.split_payload(content_off, payload_len, CellKind::TableLeaf)?;
        Ok(TableLeafCell { payload_len, rowid: rowid as i64, local_payload, overflow_page })
    }

    /// Parse cell `index` as a table-interior cell.
    pub fn table_interior_cell(&self, index: usize) -> Result<TableInteriorCell> {
        let off = self.cell_pointer(index);
        let left_child =
            try_be32(self.data, off).ok_or_else(|| Error::Format("bad left child pointer".into()))?;
        let (rowid, _) = read_varint(self.slice_from(off + 4)?)
            .ok_or_else(|| Error::Format("bad rowid varint".into()))?;
        Ok(TableInteriorCell { left_child, rowid: rowid as i64 })
    }

    /// Parse cell `index` as an index-leaf cell.
    pub fn index_leaf_cell(&self, index: usize) -> Result<IndexLeafCell<'a>> {
        let off = self.cell_pointer(index);
        let (payload_len, n1) =
            read_varint(self.slice_from(off)?).ok_or_else(|| Error::Format("bad payload varint".into()))?;
        let content_off = off + n1;
        let (local_payload, overflow_page) =
            self.split_payload(content_off, payload_len, CellKind::Index)?;
        Ok(IndexLeafCell { payload_len, local_payload, overflow_page })
    }

    /// Parse cell `index` as an index-interior cell.
    pub fn index_interior_cell(&self, index: usize) -> Result<IndexInteriorCell<'a>> {
        let off = self.cell_pointer(index);
        let left_child =
            try_be32(self.data, off).ok_or_else(|| Error::Format("bad left child pointer".into()))?;
        let (payload_len, n1) = read_varint(self.slice_from(off + 4)?)
            .ok_or_else(|| Error::Format("bad payload varint".into()))?;
        let content_off = off + 4 + n1;
        let (local_payload, overflow_page) =
            self.split_payload(content_off, payload_len, CellKind::Index)?;
        Ok(IndexInteriorCell { left_child, payload_len, local_payload, overflow_page })
    }
}

// ---- Cell serialization -------------------------------------------------------
//
// These emit the raw bytes of a cell (the same layout PageView parses). The
// caller computes the payload split via `overflow`, writes any overflow pages,
// then hands the inline slice and the first overflow page number here.

/// Append a cell's inline payload followed by the 4-byte first-overflow-page
/// pointer when the payload spilled. The three payload-bearing cell encoders share
/// this exact tail, so its on-disk shape is defined here once.
fn write_local_and_overflow(local_payload: &[u8], overflow_page: Option<u32>, out: &mut Vec<u8>) {
    out.extend_from_slice(local_payload);
    if let Some(p) = overflow_page {
        out.extend_from_slice(&p.to_be_bytes());
    }
}

/// Encode a table-leaf cell: `varint(P), varint(rowid), inline payload, [overflow
/// page]`. `payload_total_len` is P (including overflow); `local_payload` is the
/// inline prefix.
pub fn encode_table_leaf_cell(
    payload_total_len: u64,
    rowid: i64,
    local_payload: &[u8],
    overflow_page: Option<u32>,
    out: &mut Vec<u8>,
) {
    write_varint(payload_total_len, out);
    write_varint(rowid as u64, out);
    write_local_and_overflow(local_payload, overflow_page, out);
}

/// Encode a table-interior cell: `u32 left child, varint(rowid)`.
pub fn encode_table_interior_cell(left_child: u32, rowid: i64, out: &mut Vec<u8>) {
    out.extend_from_slice(&left_child.to_be_bytes());
    write_varint(rowid as u64, out);
}

/// Encode an index-leaf cell: `varint(P), inline payload, [overflow page]`.
pub fn encode_index_leaf_cell(
    payload_total_len: u64,
    local_payload: &[u8],
    overflow_page: Option<u32>,
    out: &mut Vec<u8>,
) {
    write_varint(payload_total_len, out);
    write_local_and_overflow(local_payload, overflow_page, out);
}

/// Encode an index-interior cell: `u32 left child, varint(P), inline payload,
/// [overflow page]`.
pub fn encode_index_interior_cell(
    left_child: u32,
    payload_total_len: u64,
    local_payload: &[u8],
    overflow_page: Option<u32>,
    out: &mut Vec<u8>,
) {
    out.extend_from_slice(&left_child.to_be_bytes());
    write_varint(payload_total_len, out);
    write_local_and_overflow(local_payload, overflow_page, out);
}

// ---- Page builder -------------------------------------------------------------

/// Builds one b-tree page: writes the header, the cell pointer array, and cell
/// content growing down from the end of the usable region. Cells added in key
/// order produce a cell pointer array in key order (the physical content order is
/// reversed, which is valid). The builder produces defragmented pages (no
/// freeblocks, no fragmented bytes), which real SQLite reads back.
#[derive(Debug)]
pub struct PageBuilder {
    buf: Vec<u8>,
    header_offset: usize,
    page_type: PageType,
    cell_count: usize,
    /// Current top of the cell content area (grows downward as cells are added).
    /// Initialized to the usable size (page size minus reserved bytes), the start
    /// of the cell content area for an empty page.
    content_top: usize,
    cell_pointers: Vec<u16>,
    right_most_pointer: u32,
}

impl PageBuilder {
    /// Start an empty page of `page_type` for `page_number`. `page_size` is the
    /// full page size; `usable_size` excludes the reserved region. Page 1 leaves
    /// its first 100 bytes untouched for the caller's database header.
    pub fn new(page_size: usize, usable_size: usize, page_number: u32, page_type: PageType) -> PageBuilder {
        let header_offset = header_offset_for(page_number);
        PageBuilder {
            buf: vec![0u8; page_size],
            header_offset,
            page_type,
            cell_count: 0,
            content_top: usable_size,
            cell_pointers: Vec::new(),
            right_most_pointer: 0,
        }
    }

    /// Set the right-most child pointer (interior pages only).
    pub fn set_right_most_pointer(&mut self, page: u32) {
        self.right_most_pointer = page;
    }

    fn cell_ptr_array_start(&self) -> usize {
        self.header_offset + self.page_type.header_len()
    }

    /// Bytes currently free for one more cell plus its 2-byte pointer entry.
    pub fn free_space(&self) -> usize {
        let ptr_end = self.cell_ptr_array_start() + self.cell_count * 2;
        self.content_top.saturating_sub(ptr_end)
    }

    /// Append a pre-encoded cell in key order. Returns `false` (adding nothing) if
    /// it does not fit, so the caller can split the page. Accounts for both the
    /// cell content and the new 2-byte cell pointer.
    pub fn add_cell(&mut self, cell: &[u8]) -> bool {
        let len = cell.len();
        if len > self.content_top {
            return false;
        }
        let new_top = self.content_top - len;
        let ptr_end = self.cell_ptr_array_start() + (self.cell_count + 1) * 2;
        if ptr_end > new_top {
            return false;
        }
        self.buf[new_top..new_top + len].copy_from_slice(cell);
        self.content_top = new_top;
        // A cell always leaves at least a few bytes below 65536, so the offset
        // fits a u16 (only an *empty* page's content start needs the 0 sentinel).
        self.cell_pointers.push(new_top as u16);
        self.cell_count += 1;
        true
    }

    pub fn cell_count(&self) -> usize {
        self.cell_count
    }

    /// Finalize the page: write the header fields and cell pointer array, and
    /// return the page bytes. On page 1 the first 100 bytes are left as the caller
    /// set them (the database header).
    pub fn finish(mut self) -> Vec<u8> {
        let ho = self.header_offset;
        self.buf[ho] = self.page_type.to_byte();
        write_be16(&mut self.buf, ho + 1, 0); // no freeblocks
        write_be16(&mut self.buf, ho + 3, self.cell_count as u16);
        let content_field = if self.content_top >= 65536 { 0 } else { self.content_top as u16 };
        write_be16(&mut self.buf, ho + 5, content_field);
        self.buf[ho + 7] = 0; // no fragmented bytes
        if self.page_type.is_interior() {
            write_be32(&mut self.buf, ho + 8, self.right_most_pointer);
        }
        let cpa = self.cell_ptr_array_start();
        for (i, &ptr) in self.cell_pointers.iter().enumerate() {
            write_be16(&mut self.buf, cpa + i * 2, ptr);
        }
        self.buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serial::{decode_record, encode_record};
    use minisqlite_types::Value;

    const PAGE: usize = 4096;
    const USABLE: usize = 4096;

    #[test]
    fn page_type_byte_roundtrip() {
        for t in [
            PageType::InteriorIndex,
            PageType::InteriorTable,
            PageType::LeafIndex,
            PageType::LeafTable,
        ] {
            assert_eq!(PageType::from_byte(t.to_byte()), Some(t));
        }
        // Pin the spec constants (§1.6), not just from∘to symmetry: a consistent
        // swap of two type bytes in both to_byte/from_byte would keep the round-trip
        // green while mislabeling pages real sqlite wrote.
        assert_eq!(PageType::InteriorIndex.to_byte(), 0x02);
        assert_eq!(PageType::InteriorTable.to_byte(), 0x05);
        assert_eq!(PageType::LeafIndex.to_byte(), 0x0a);
        assert_eq!(PageType::LeafTable.to_byte(), 0x0d);
        assert_eq!(PageType::from_byte(0x02), Some(PageType::InteriorIndex));
        assert_eq!(PageType::from_byte(0x05), Some(PageType::InteriorTable));
        assert_eq!(PageType::from_byte(0x0a), Some(PageType::LeafIndex));
        assert_eq!(PageType::from_byte(0x0d), Some(PageType::LeafTable));
        assert_eq!(PageType::from_byte(0), None);
        assert_eq!(PageType::from_byte(1), None);
        assert_eq!(PageType::InteriorTable.header_len(), 12);
        assert_eq!(PageType::LeafTable.header_len(), 8);
        assert!(PageType::LeafTable.is_table() && PageType::LeafTable.is_leaf());
        assert!(PageType::InteriorIndex.is_index() && PageType::InteriorIndex.is_interior());
    }

    #[test]
    fn build_and_read_table_leaf_page() {
        let rows = [
            (1i64, vec![Value::Integer(10), Value::Text("alpha".into())]),
            (2, vec![Value::Integer(20), Value::Text("beta".into())]),
            (5, vec![Value::Null, Value::Blob(vec![1, 2, 3])]),
        ];
        let mut b = PageBuilder::new(PAGE, USABLE, 3, PageType::LeafTable);
        for (rowid, row) in &rows {
            let payload = encode_record(row);
            let mut cell = Vec::new();
            // Short rows: no overflow, local == whole payload.
            encode_table_leaf_cell(payload.len() as u64, *rowid, &payload, None, &mut cell);
            assert!(b.add_cell(&cell));
        }
        let page = b.finish();

        // The raw type byte on disk is the spec literal (this is page 3, header at 0).
        assert_eq!(page[0], 0x0d, "leaf table page type byte");
        let view = PageView::new(&page, 3, USABLE).unwrap();
        assert_eq!(view.page_type(), PageType::LeafTable);
        assert_eq!(view.cell_count(), 3);
        assert_eq!(view.right_most_pointer(), None);
        for (i, (rowid, row)) in rows.iter().enumerate() {
            let cell = view.table_leaf_cell(i).unwrap();
            assert_eq!(cell.rowid, *rowid);
            assert!(cell.overflow_page.is_none());
            assert_eq!(cell.payload_len as usize, cell.local_payload.len());
            let decoded = decode_record(cell.local_payload);
            assert_eq!(decoded.len(), row.len());
        }
    }

    #[test]
    fn build_and_read_table_interior_page() {
        let entries = [(2u32, 100i64), (3, 200), (4, 300)];
        let mut b = PageBuilder::new(PAGE, USABLE, 9, PageType::InteriorTable);
        b.set_right_most_pointer(5);
        for (child, rowid) in &entries {
            let mut cell = Vec::new();
            encode_table_interior_cell(*child, *rowid, &mut cell);
            assert!(b.add_cell(&cell));
        }
        let page = b.finish();

        assert_eq!(page[0], 0x05, "interior table page type byte");
        let view = PageView::new(&page, 9, USABLE).unwrap();
        assert_eq!(view.page_type(), PageType::InteriorTable);
        assert_eq!(view.right_most_pointer(), Some(5));
        assert_eq!(view.cell_count(), 3);
        for (i, (child, rowid)) in entries.iter().enumerate() {
            let c = view.table_interior_cell(i).unwrap();
            assert_eq!(c.left_child, *child);
            assert_eq!(c.rowid, *rowid);
        }
    }

    #[test]
    fn build_and_read_index_pages() {
        // Index leaf.
        let keys = [
            encode_record(&[Value::Integer(1), Value::Integer(100)]),
            encode_record(&[Value::Integer(2), Value::Integer(200)]),
        ];
        let mut b = PageBuilder::new(PAGE, USABLE, 6, PageType::LeafIndex);
        for k in &keys {
            let mut cell = Vec::new();
            encode_index_leaf_cell(k.len() as u64, k, None, &mut cell);
            assert!(b.add_cell(&cell));
        }
        let page = b.finish();
        assert_eq!(page[0], 0x0a, "leaf index page type byte");
        let view = PageView::new(&page, 6, USABLE).unwrap();
        assert_eq!(view.page_type(), PageType::LeafIndex);
        for (i, k) in keys.iter().enumerate() {
            let c = view.index_leaf_cell(i).unwrap();
            assert!(c.overflow_page.is_none());
            assert_eq!(c.local_payload, k.as_slice());
        }

        // Index interior.
        let mut b2 = PageBuilder::new(PAGE, USABLE, 7, PageType::InteriorIndex);
        b2.set_right_most_pointer(42);
        let key = encode_record(&[Value::Text("k".into()), Value::Integer(9)]);
        let mut cell = Vec::new();
        encode_index_interior_cell(11, key.len() as u64, &key, None, &mut cell);
        assert!(b2.add_cell(&cell));
        let page2 = b2.finish();
        assert_eq!(page2[0], 0x02, "interior index page type byte");
        let view2 = PageView::new(&page2, 7, USABLE).unwrap();
        assert_eq!(view2.right_most_pointer(), Some(42));
        let c = view2.index_interior_cell(0).unwrap();
        assert_eq!(c.left_child, 11);
        assert_eq!(c.local_payload, key.as_slice());
    }

    #[test]
    fn page_one_header_lives_at_offset_100() {
        // Build page 1 as a leaf table page (the schema page shape) and confirm the
        // b-tree header is at byte 100, with the first 100 bytes free for the db
        // header. Cell offsets remain relative to the page start.
        let mut b = PageBuilder::new(PAGE, USABLE, 1, PageType::LeafTable);
        let payload = encode_record(&[Value::Text("schema".into())]);
        let mut cell = Vec::new();
        encode_table_leaf_cell(payload.len() as u64, 1, &payload, None, &mut cell);
        assert!(b.add_cell(&cell));
        let mut page = b.finish();
        // Overlay a database header in the first 100 bytes.
        let hdr = crate::header::DatabaseHeader::default().to_bytes();
        page[0..100].copy_from_slice(&hdr);

        assert_eq!(page[100], 0x0d, "b-tree header (leaf table type byte) at file offset 100");
        let view = PageView::new(&page, 1, USABLE).unwrap();
        assert_eq!(view.page_type(), PageType::LeafTable);
        assert_eq!(view.cell_count(), 1);
        let c = view.table_leaf_cell(0).unwrap();
        assert_eq!(c.rowid, 1);
        // The borrowed payload is a sub-slice of the page bytes (zero-copy read).
        let page_range = page.as_ptr_range();
        let payload_range = c.local_payload.as_ptr_range();
        assert!(payload_range.start >= page_range.start && payload_range.end <= page_range.end);
        let decoded = decode_record(c.local_payload);
        assert_eq!(decoded.len(), 1);
        assert!(matches!(&decoded[0], Value::Text(s) if s == "schema"));
    }

    #[test]
    fn empty_page_content_start_equals_usable() {
        let b = PageBuilder::new(PAGE, USABLE, 4, PageType::LeafTable);
        let page = b.finish();
        let view = PageView::new(&page, 4, USABLE).unwrap();
        assert_eq!(view.cell_count(), 0);
        assert_eq!(view.cell_content_start(), USABLE as u32);
    }

    #[test]
    fn empty_page_65536_uses_zero_sentinel() {
        let b = PageBuilder::new(65536, 65536, 4, PageType::LeafTable);
        let page = b.finish();
        // The raw stored value must be 0 (65536 does not fit in u16).
        assert_eq!(be16(&page, 5), 0);
        let view = PageView::new(&page, 4, 65536).unwrap();
        assert_eq!(view.cell_content_start(), 65536);
    }

    #[test]
    fn add_cell_reports_full_page() {
        let mut b = PageBuilder::new(PAGE, USABLE, 2, PageType::LeafTable);
        // Fill with big cells until one is refused; the refusal must add nothing.
        let big = vec![0xABu8; 1000];
        let mut added = 0;
        loop {
            let mut cell = Vec::new();
            encode_table_leaf_cell(big.len() as u64, added as i64 + 1, &big, None, &mut cell);
            if b.add_cell(&cell) {
                added += 1;
            } else {
                break;
            }
        }
        assert!(added >= 1 && added < 5, "≈4 one-KiB cells fit on a 4 KiB page, got {added}");
        assert_eq!(b.cell_count(), added);
        let page = b.finish();
        let view = PageView::new(&page, 2, USABLE).unwrap();
        assert_eq!(view.cell_count() as usize, added);
    }

    #[test]
    fn invalid_page_type_rejected() {
        let mut page = vec![0u8; PAGE];
        page[0] = 0xFF;
        assert!(PageView::new(&page, 3, USABLE).is_err());
    }

    #[test]
    fn corrupt_cell_count_rejected() {
        let mut page = vec![0u8; PAGE];
        page[0] = PageType::LeafTable.to_byte();
        // Claim more cells than the page could ever hold.
        write_be16(&mut page, 3, 60000);
        assert!(PageView::new(&page, 3, USABLE).is_err());
    }

    #[test]
    fn table_leaf_cell_with_overflow_reports_head() {
        // A payload that spills: build the cell with an inline prefix + overflow
        // pointer and confirm the reader recovers the split and the head page.
        let usable = 512usize;
        let payload = vec![0x5Au8; 2000];
        let split = payload_split(usable, payload.len() as u64, CellKind::TableLeaf);
        assert!(split.has_overflow);
        let local = &payload[..split.local];
        let mut cell = Vec::new();
        encode_table_leaf_cell(payload.len() as u64, 77, local, Some(123), &mut cell);
        let mut b = PageBuilder::new(usable, usable, 3, PageType::LeafTable);
        assert!(b.add_cell(&cell));
        let page = b.finish();
        let view = PageView::new(&page, 3, usable).unwrap();
        let c = view.table_leaf_cell(0).unwrap();
        assert_eq!(c.rowid, 77);
        assert_eq!(c.payload_len, 2000);
        assert_eq!(c.local_payload.len(), split.local);
        assert_eq!(c.overflow_page, Some(123));
    }

    #[test]
    fn rowid_sign_and_magnitude_roundtrip() {
        // rowid is a signed i64 stored as its u64 bit pattern via a varint
        // (`rowid as u64` on write, `as i64` on read). Guard that cast for
        // negatives and the extremes, on both table cell kinds.
        for &rowid in &[0i64, -1, 1, i64::MIN, i64::MAX, -1_000_000_000_000, 4_294_967_296] {
            let payload = encode_record(&[Value::Integer(42)]);
            let mut leaf = Vec::new();
            encode_table_leaf_cell(payload.len() as u64, rowid, &payload, None, &mut leaf);
            let mut lb = PageBuilder::new(PAGE, USABLE, 3, PageType::LeafTable);
            assert!(lb.add_cell(&leaf));
            let lp = lb.finish();
            let lv = PageView::new(&lp, 3, USABLE).unwrap();
            assert_eq!(lv.table_leaf_cell(0).unwrap().rowid, rowid, "leaf rowid {rowid}");

            let mut inner = Vec::new();
            encode_table_interior_cell(2, rowid, &mut inner);
            let mut ib = PageBuilder::new(PAGE, USABLE, 9, PageType::InteriorTable);
            ib.set_right_most_pointer(5);
            assert!(ib.add_cell(&inner));
            let ip = ib.finish();
            let iv = PageView::new(&ip, 9, USABLE).unwrap();
            assert_eq!(iv.table_interior_cell(0).unwrap().rowid, rowid, "interior rowid {rowid}");
        }
    }
}
