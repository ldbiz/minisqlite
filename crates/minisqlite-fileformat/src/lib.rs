//! The official SQLite on-disk file format codec — read AND write
//! (see `spec/sqlite-doc/fileformat2.html`). A pure bytes<->data layer: it does
//! no file I/O, caching, or transactions (those live in `minisqlite-pager`, built
//! on this). The bar is bidirectional and byte-exact: read files real sqlite3
//! wrote, and write files real sqlite3 reads back.
//!
//! This root is a thin re-export hub; the real code lives in the submodules:
//!
//! - [`varint`] — the 1–9 byte big-endian Huffman varint.
//! - [`serial`] — serial-type codes and the record (row) codec.
//! - [`record_cursor`] — zero-copy lazy iteration over a record's columns.
//! - [`header`] — the 100-byte database header.
//! - [`text_encoding`] — the database TEXT encoding (§1.3.13) and UTF-16→UTF-8 decode.
//! - [`page`] — b-tree page header, cells, and a page builder.
//! - [`page_edit`] — in-place, single-cell editing of a leaf page (the O(cell) insert path).
//! - [`overflow`] — the payload-spill threshold math and overflow-page chains.
//! - [`freelist`] — freelist trunk-page byte layout (§1.5).
//! - [`ptrmap`] — pointer-map page positions for auto_vacuum databases (§1.8).
//!
//! Decoders BORROW from the input page slice wherever possible (cell payloads are
//! sub-slices, never copies), so the pager and executor can read a row without
//! allocating a whole page.

mod bytes;

pub mod freelist;
pub mod header;
pub mod overflow;
pub mod page;
pub mod page_edit;
pub mod ptrmap;
pub mod record_cursor;
pub mod serial;
pub mod text_encoding;
pub mod varint;

pub use varint::{read_varint, varint_len, write_varint};

pub use serial::{
    decode_record, decode_record_enc, decode_record_into, decode_record_into_enc, encode_record,
    encode_record_enc, encode_record_into, encode_record_into_enc, read_serial_value,
    read_serial_value_enc, serial_type_of, serial_type_of_enc, serial_type_payload_len,
    write_serial_value, write_serial_value_enc,
};

pub use text_encoding::TextEncoding;

pub use ptrmap::{
    decode_ptrmap_entry, encode_ptrmap_entry, get_ptrmap_entry, is_ptrmap_page,
    ptrmap_entries_per_page, ptrmap_offset_of, ptrmap_page_of, ptrmap_stride, put_ptrmap_entry,
    PTRMAP_BTREE, PTRMAP_ENTRY_SIZE, PTRMAP_FREEPAGE, PTRMAP_OVERFLOW1, PTRMAP_OVERFLOW2,
    PTRMAP_ROOTPAGE,
};

pub use record_cursor::RecordCursor;

pub use header::{
    DatabaseHeader, DEFAULT_PAGE_SIZE, HEADER_SIZE, LEAF_PAYLOAD_FRACTION,
    MAX_EMBEDDED_PAYLOAD_FRACTION, MIN_EMBEDDED_PAYLOAD_FRACTION, MAGIC, TEXT_ENCODING_UTF16BE,
    TEXT_ENCODING_UTF16LE, TEXT_ENCODING_UTF8,
};

pub use page::{
    encode_index_interior_cell, encode_index_leaf_cell, encode_table_interior_cell,
    encode_table_leaf_cell, header_offset_for, IndexInteriorCell, IndexLeafCell, PageBuilder,
    PageHeader, PageType, PageView, TableInteriorCell, TableLeafCell,
};

pub use page_edit::{
    defragment, delete_cell, insert_cell, leaf_cell_len, leaf_free_space, FreeSpace,
};

pub use overflow::{
    assemble_payload, overflow_chunks, overflow_content_mut, overflow_page_capacity,
    parse_overflow_page, payload_split, set_overflow_next, split_payload_for_write,
    write_overflow_page, CellKind, OverflowPage, PageSource, PayloadSplit,
};

pub use freelist::{
    parse_trunk, trunk_leaf_capacity, trunk_leaf_count, trunk_leaves, trunk_next, write_trunk,
    TRUNK_HEADER_LEN,
};
