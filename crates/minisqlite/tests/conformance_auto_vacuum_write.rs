//! Conformance battery: the **WRITE side of auto_vacuum / incremental_vacuum**
//! (fileformat2 §1.3.12 + §1.8). The anti-"painted-over-header" gate.
//!
//! The two header-byte cases in `conformance_fileformat.rs` only prove offsets 52/64
//! are non-zero — they cannot tell a real ptrmap structure from a writer that scribbled
//! two header integers and produced a file real sqlite would reject. This file closes
//! that gap: it WRITES real auto_vacuum databases through `minisqlite::Connection` (a
//! multi-page table b-tree, an index, an overflow row, and freed pages), reopens them to
//! confirm the ROWS read back, then reads the RAW file bytes and INDEPENDENTLY walks the
//! b-trees / overflow chains / freelist to reconstruct the `(type, parent)` every
//! non-page-1, non-ptrmap page MUST have — and checks the on-disk ptrmap entry against
//! it, page by page. It is impossible to pass by writing header bytes alone: every one of
//! the five §1.8 entry types is reconstructed from the structure and asserted present.
//!
//! Companion to `conformance_auto_vacuum.rs`: that sibling spot-checks a FULL-mode file
//! (root/overflow entries at computed positions, a round-trip). This file is the
//! EXHAUSTIVE form — it reconstructs the WHOLE ptrmap and diffs every page's `(type,
//! parent)`, and it additionally covers what the spot-check sibling deliberately does not:
//! INCREMENTAL mode (offset 64 = 1), the freelist (type-2) entries that DELETE produces,
//! `off 52` equal to the independently-computed largest root, and a file large enough to
//! span MULTIPLE ptrmap pages (small 512-byte pages, page count past the second ptrmap
//! page at 105). Kept a separate file (not merged into the sibling) so the two are
//! independent commit cells — the repo's file-granularity rule for parallel work.
//!
//! Everything here is transcribed from the spec (`spec/sqlite-doc/fileformat2.html`
//! §1.3, §1.5, §1.6, §1.7, §1.8), NEVER imported from the engine crates — the tests link
//! only the `minisqlite` facade. The independent decoders
//! (varint, payload split, b-tree cell shapes, ptrmap addressing) are deliberately a
//! second implementation: if they and the engine's codec ever disagree, that disagreement
//! is a real bug this test is meant to surface, not something to paper over.

mod conformance;
use conformance::*;

use minisqlite::{Connection, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Temp-file harness (a unique on-disk path per test, cleaned up even on panic).
// Local copy — the identical guard lives in the sibling on-disk conformance files;
// hoisting it touches the shared harness seam, so it stays local (see the note in
// `conformance_fileformat.rs`).
// ---------------------------------------------------------------------------

struct TempDb {
    path: PathBuf,
}

impl TempDb {
    fn new() -> TempDb {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
        let mut path = std::env::temp_dir();
        path.push(format!("minisqlite_av_{pid}_{n}_{nanos}.db"));
        let db = TempDb { path };
        db.remove_all();
        db
    }

    fn open(&self) -> Connection {
        let path: &Path = &self.path;
        Connection::open(path).unwrap_or_else(|e| panic!("Connection::open failed for {path:?}: {e:?}"))
    }

    fn sidecar(&self, suffix: &str) -> PathBuf {
        let mut s = self.path.as_os_str().to_os_string();
        s.push(suffix);
        PathBuf::from(s)
    }

    fn remove_all(&self) {
        let _ = std::fs::remove_file(&self.path);
        for suffix in ["-journal", "-wal", "-shm"] {
            let _ = std::fs::remove_file(self.sidecar(suffix));
        }
    }
}

impl Drop for TempDb {
    fn drop(&mut self) {
        self.remove_all();
    }
}

fn read_db(tmp: &TempDb) -> Vec<u8> {
    std::fs::read(&tmp.path).unwrap_or_else(|e| panic!("failed to read db file {:?}: {e}", tmp.path))
}

// ---------------------------------------------------------------------------
// Big-endian primitives + the record varint (§1.3 header + §2.1 varint), decoded
// independently of the engine.
// ---------------------------------------------------------------------------

fn be16(b: &[u8], off: usize) -> u16 {
    u16::from_be_bytes([b[off], b[off + 1]])
}

fn be32(b: &[u8], off: usize) -> u32 {
    u32::from_be_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

/// A big-endian base-128 varint (§2.1): up to 8 payload bytes with the high bit as the
/// continuation flag, and a 9th byte contributing all 8 bits. Returns `(value, len)`.
fn read_varint(b: &[u8]) -> (u64, usize) {
    let mut result: u64 = 0;
    for i in 0..8 {
        let byte = b[i];
        result = (result << 7) | u64::from(byte & 0x7f);
        if byte & 0x80 == 0 {
            return (result, i + 1);
        }
    }
    result = (result << 8) | u64::from(b[8]);
    (result, 9)
}

// ---------------------------------------------------------------------------
// Pointer-map addressing (§1.8) — a second implementation of the position math, so a
// bug in the engine's `ptrmap.rs` cannot hide behind a shared helper.
// ---------------------------------------------------------------------------

/// J = U/5: the number of 5-byte entries a ptrmap page holds.
fn ptrmap_entries_per_page(usable: usize) -> u32 {
    (usable / 5) as u32
}

/// The spacing between ptrmap pages: J + 1.
fn ptrmap_stride(usable: usize) -> u32 {
    ptrmap_entries_per_page(usable) + 1
}

/// Whether `page` sits at a §1.8 ptrmap position: page 2, then every J+1 pages.
fn is_ptrmap_page(usable: usize, page: u32) -> bool {
    page >= 2 && (page - 2) % ptrmap_stride(usable) == 0
}

/// The ptrmap page covering `page`: the greatest ptrmap position strictly below it.
fn ptrmap_page_for(usable: usize, page: u32) -> u32 {
    let stride = ptrmap_stride(usable);
    2 + ((page - 2) / stride) * stride
}

/// The byte offset of `page`'s entry within its covering ptrmap page: (p - cover - 1)*5.
fn ptrmap_entry_offset(usable: usize, page: u32) -> usize {
    let cover = ptrmap_page_for(usable, page);
    (page - cover - 1) as usize * 5
}

// The five §1.8 entry types, transcribed independently.
const KIND_ROOT: u8 = 1;
const KIND_FREELIST: u8 = 2;
const KIND_OVERFLOW_HEAD: u8 = 3;
const KIND_OVERFLOW_NEXT: u8 = 4;
const KIND_BTREE: u8 = 5;

// B-tree page-type bytes (§1.6).
const PT_INTERIOR_INDEX: u8 = 0x02;
const PT_INTERIOR_TABLE: u8 = 0x05;
const PT_LEAF_INDEX: u8 = 0x0a;
const PT_LEAF_TABLE: u8 = 0x0d;

/// A read-only view over the raw on-disk database bytes, with the decoded page geometry.
struct RawDb {
    raw: Vec<u8>,
    page_size: usize,
    usable: usize,
    page_count: u32,
}

impl RawDb {
    fn parse(raw: Vec<u8>) -> RawDb {
        assert!(raw.len() >= 100, "db file shorter than the 100-byte header");
        let page_size = match be16(&raw, 16) {
            1 => 65_536,
            n => n as usize,
        };
        let reserved = raw[20] as usize;
        let usable = page_size - reserved;
        assert!(usable >= 480, "usable page size {usable} below the §1.3.4 floor");
        assert_eq!(raw.len() % page_size, 0, "file length is not a whole number of pages");
        let page_count = (raw.len() / page_size) as u32;
        RawDb { raw, page_size, usable, page_count }
    }

    /// The bytes of page `n` (1-based).
    fn page(&self, n: u32) -> &[u8] {
        let start = (n as usize - 1) * self.page_size;
        &self.raw[start..start + self.page_size]
    }

    /// Where a page's b-tree header begins: byte 100 on page 1 (after the db header), 0
    /// elsewhere (§1.6).
    fn header_offset(&self, n: u32) -> usize {
        if n == 1 { 100 } else { 0 }
    }

    fn largest_root(&self) -> u32 {
        be32(&self.raw, 52)
    }

    fn incremental_flag(&self) -> u32 {
        be32(&self.raw, 64)
    }

    fn first_freelist_trunk(&self) -> u32 {
        be32(&self.raw, 32)
    }

    /// The on-disk `(type, parent)` recorded for `page` in its covering ptrmap page.
    fn ptrmap_entry(&self, page: u32) -> (u8, u32) {
        let cover = ptrmap_page_for(self.usable, page);
        let off = ptrmap_entry_offset(self.usable, page);
        let p = self.page(cover);
        (p[off], be32(p, off + 1))
    }

    fn is_valid_data_page(&self, page: u32) -> bool {
        page >= 2 && page <= self.page_count && !is_ptrmap_page(self.usable, page)
    }
}

// ---------------------------------------------------------------------------
// Payload split (§1.6): how many bytes of a P-byte payload stay on the b-tree page, and
// whether any spill to overflow. A second copy of the format's exact (quirky) arithmetic.
// ---------------------------------------------------------------------------

/// Returns `(local_len, has_overflow)` for a payload of `p` bytes. `table_leaf` selects
/// the table-leaf X formula; index cells (leaf and interior) use the index formula.
fn payload_local(usable: usize, p: usize, table_leaf: bool) -> (usize, bool) {
    let u = usable;
    let x = if table_leaf { u - 35 } else { (u - 12) * 64 / 255 - 23 };
    if p <= x {
        return (p, false);
    }
    let m = (u - 12) * 32 / 255 - 23;
    let k = m + (p - m) % (u - 4);
    let local = if k <= x { k } else { m };
    (local, true)
}

// ---------------------------------------------------------------------------
// The independent classifier: walk every b-tree, overflow chain, and the freelist from
// the raw bytes and build the `page -> (type, parent)` map the ptrmap MUST match.
// ---------------------------------------------------------------------------

/// Reconstruct the expected ptrmap contents by walking the real structure, mirroring the
/// §1.8 rule with a wholly separate decoder from the engine's.
fn expected_entries(db: &RawDb) -> BTreeMap<u32, (u8, u32)> {
    let mut map: BTreeMap<u32, (u8, u32)> = BTreeMap::new();

    // Page 1 (sqlite_schema) is a root but carries NO entry; walk it for any child /
    // overflow pages it owns, then every table/index root it names.
    walk_btree(db, 1, &mut map);
    for root in schema_roots(db) {
        assert!(
            db.is_valid_data_page(root),
            "schema rootpage {root} is out of range or on a ptrmap position (page_count {})",
            db.page_count
        );
        // A root is never a child of another b-tree, so it must not already be classified.
        assert!(!map.contains_key(&root), "rootpage {root} was already seen as a non-root page");
        map.insert(root, (KIND_ROOT, 0));
        walk_btree(db, root, &mut map);
    }

    // Freelist trunks + leaves are type 2 (§1.5).
    collect_freelist(db, &mut map);

    map
}

/// Read `sqlite_schema` (the table b-tree rooted at page 1) and return every non-zero
/// `rootpage` (column 3 of `(type, name, tbl_name, rootpage, sql)`). Views/triggers store
/// rootpage 0 and are skipped.
fn schema_roots(db: &RawDb) -> Vec<u32> {
    let mut roots = Vec::new();
    for payload in table_leaf_payloads(db, 1) {
        if let Some(rp) = record_int_column(&payload, 3) {
            if rp > 0 {
                roots.push(rp as u32);
            }
        }
    }
    roots
}

/// Every table-leaf cell's FULL payload (overflow assembled) under the table b-tree rooted
/// at `root`. Used only for the tiny `sqlite_schema` table, but overflow is assembled so a
/// long schema row would still decode correctly.
fn table_leaf_payloads(db: &RawDb, root: u32) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut stack = vec![root];
    // Visited-guarded like `walk_btree`: a corrupt interior cycle must terminate rather
    // than spin the verifier (every traversal must be bounded by page_count).
    let mut visited = std::collections::BTreeSet::new();
    while let Some(n) = stack.pop() {
        if !visited.insert(n) {
            continue;
        }
        assert!(
            visited.len() as u32 <= db.page_count,
            "schema b-tree walk visited more pages than exist (a cycle)"
        );
        let page = db.page(n);
        let ho = db.header_offset(n);
        let pt = page[ho];
        let cells = be16(page, ho + 3) as usize;
        match pt {
            PT_LEAF_TABLE => {
                let ptr_base = ho + 8;
                for i in 0..cells {
                    let off = be16(page, ptr_base + i * 2) as usize;
                    let (p_len, n1) = read_varint(&page[off..]);
                    let (_rowid, n2) = read_varint(&page[off + n1..]);
                    let content = off + n1 + n2;
                    out.push(assemble_payload(db, page, content, p_len as usize, true));
                }
            }
            PT_INTERIOR_TABLE => {
                let ptr_base = ho + 12;
                for i in 0..cells {
                    let off = be16(page, ptr_base + i * 2) as usize;
                    stack.push(be32(page, off));
                }
                stack.push(be32(page, ho + 8)); // right-most child
            }
            other => panic!("page {n} in the schema b-tree has non-table type {other:#x}"),
        }
    }
    out
}

/// Assemble a possibly-overflowing payload into a single `Vec`: the inline prefix on the
/// b-tree page followed by the content of each overflow page (§1.7).
fn assemble_payload(db: &RawDb, page: &[u8], content_off: usize, p_len: usize, table_leaf: bool) -> Vec<u8> {
    let (local, has_overflow) = payload_local(db.usable, p_len, table_leaf);
    let mut buf = page[content_off..content_off + local].to_vec();
    if has_overflow {
        let mut next = be32(page, content_off + local);
        // Bounded like the other walks: stop once the payload is complete (past that,
        // `take` would be 0 — never make no progress), and cap the hop count by page_count
        // so a cyclic/self-pointing chain fails loudly instead of hanging.
        let mut guard = 0u32;
        while next != 0 && buf.len() < p_len {
            guard += 1;
            assert!(guard <= db.page_count, "overflow chain longer than the file (a cycle)");
            let op = db.page(next);
            let take = (p_len - buf.len()).min(db.usable - 4);
            buf.extend_from_slice(&op[4..4 + take]);
            next = be32(op, 0);
        }
    }
    buf.truncate(p_len);
    buf
}

/// Decode the signed integer stored in column `col` of a record, or `None` if the column
/// is absent or not an integer serial type (§2.1).
fn record_int_column(payload: &[u8], col: usize) -> Option<i64> {
    let (header_len, mut type_off) = read_varint(payload);
    let header_end = (header_len as usize).min(payload.len());
    let mut body_off = header_len as usize;
    let mut idx = 0;
    while type_off < header_end {
        let (serial, n) = read_varint(&payload[type_off..header_end]);
        type_off += n;
        let len = serial_payload_len(serial);
        if idx == col {
            return decode_int_serial(serial, payload.get(body_off..body_off + len)?);
        }
        body_off += len;
        idx += 1;
    }
    None
}

/// Body length of a serial type (§2.1 table).
fn serial_payload_len(serial: u64) -> usize {
    match serial {
        0 | 8 | 9 => 0,
        1 => 1,
        2 => 2,
        3 => 3,
        4 => 4,
        5 => 6,
        6 | 7 => 8,
        s if s >= 12 && s % 2 == 0 => ((s - 12) / 2) as usize,
        s if s >= 13 => ((s - 13) / 2) as usize,
        _ => 0, // 10/11 are reserved; treat as empty
    }
}

/// Decode an integer-class serial type (1..=6 and the 0/1 constants 8/9) as a signed i64.
fn decode_int_serial(serial: u64, body: &[u8]) -> Option<i64> {
    match serial {
        8 => Some(0),
        9 => Some(1),
        1..=6 => {
            let mut v: i64 = if body.first().copied().unwrap_or(0) & 0x80 != 0 { -1 } else { 0 };
            for &b in body {
                v = (v << 8) | i64::from(b);
            }
            Some(v)
        }
        _ => None,
    }
}

/// Walk the b-tree rooted at `root`, recording a type-5 entry for every page BELOW it and
/// type-3/4 entries for each cell's overflow chain. `root`'s own entry (type 1, or none
/// for page 1) is the caller's responsibility. Iterative + visited-guarded, so a corrupt
/// cycle terminates.
fn walk_btree(db: &RawDb, root: u32, map: &mut BTreeMap<u32, (u8, u32)>) {
    let mut stack = vec![root];
    while let Some(n) = stack.pop() {
        let page = db.page(n);
        let ho = db.header_offset(n);
        let pt = page[ho];
        let cells = be16(page, ho + 3) as usize;
        let (children, overflow_heads) = match pt {
            PT_INTERIOR_TABLE => {
                let mut ch = Vec::new();
                let base = ho + 12;
                for i in 0..cells {
                    let off = be16(page, base + i * 2) as usize;
                    ch.push(be32(page, off));
                }
                ch.push(be32(page, ho + 8));
                (ch, Vec::new())
            }
            PT_INTERIOR_INDEX => {
                let (mut ch, mut ov) = (Vec::new(), Vec::new());
                let base = ho + 12;
                for i in 0..cells {
                    let off = be16(page, base + i * 2) as usize;
                    ch.push(be32(page, off));
                    let (p_len, n1) = read_varint(&page[off + 4..]);
                    let content = off + 4 + n1;
                    if let Some(h) = overflow_head(db, page, content, p_len as usize, false) {
                        ov.push(h);
                    }
                }
                ch.push(be32(page, ho + 8));
                (ch, ov)
            }
            PT_LEAF_TABLE => {
                let mut ov = Vec::new();
                let base = ho + 8;
                for i in 0..cells {
                    let off = be16(page, base + i * 2) as usize;
                    let (p_len, n1) = read_varint(&page[off..]);
                    let (_rowid, n2) = read_varint(&page[off + n1..]);
                    let content = off + n1 + n2;
                    if let Some(h) = overflow_head(db, page, content, p_len as usize, true) {
                        ov.push(h);
                    }
                }
                (Vec::new(), ov)
            }
            PT_LEAF_INDEX => {
                let mut ov = Vec::new();
                let base = ho + 8;
                for i in 0..cells {
                    let off = be16(page, base + i * 2) as usize;
                    let (p_len, n1) = read_varint(&page[off..]);
                    let content = off + n1;
                    if let Some(h) = overflow_head(db, page, content, p_len as usize, false) {
                        ov.push(h);
                    }
                }
                (Vec::new(), ov)
            }
            other => panic!("page {n} has an invalid b-tree type byte {other:#x}"),
        };
        for head in overflow_heads {
            record_overflow_chain(db, n, head, map);
        }
        for child in children {
            assert!(
                db.is_valid_data_page(child),
                "b-tree page {n} points at invalid child {child} (page_count {})",
                db.page_count
            );
            if !map.contains_key(&child) {
                map.insert(child, (KIND_BTREE, n));
                stack.push(child);
            }
        }
    }
}

/// The first overflow page of a cell, if its payload spilled.
fn overflow_head(db: &RawDb, page: &[u8], content_off: usize, p_len: usize, table_leaf: bool) -> Option<u32> {
    let (local, has_overflow) = payload_local(db.usable, p_len, table_leaf);
    if has_overflow {
        Some(be32(page, content_off + local))
    } else {
        None
    }
}

/// Record a cell's overflow chain: head is type 3 (parent = the b-tree page), each later
/// page type 4 (parent = the previous page). Bounded by the visited guard.
fn record_overflow_chain(db: &RawDb, parent_btree: u32, head: u32, map: &mut BTreeMap<u32, (u8, u32)>) {
    assert!(
        db.is_valid_data_page(head),
        "overflow head {head} (parent b-tree {parent_btree}) is out of range or a ptrmap position",
    );
    if map.contains_key(&head) {
        return;
    }
    map.insert(head, (KIND_OVERFLOW_HEAD, parent_btree));
    let mut prev = head;
    loop {
        let next = be32(db.page(prev), 0);
        if next == 0 {
            break;
        }
        assert!(
            db.is_valid_data_page(next),
            "overflow page {next} (after {prev}) is out of range or a ptrmap position",
        );
        if map.contains_key(&next) {
            break;
        }
        map.insert(next, (KIND_OVERFLOW_NEXT, prev));
        prev = next;
    }
}

/// Record every freelist trunk and leaf as type 2 (§1.5), walking the trunk chain from the
/// header's first-trunk pointer. Bounded by a page-count guard against a corrupt cycle.
fn collect_freelist(db: &RawDb, map: &mut BTreeMap<u32, (u8, u32)>) {
    let mut trunk = db.first_freelist_trunk();
    let mut guard = 0u32;
    while trunk != 0 {
        assert!(db.is_valid_data_page(trunk), "freelist trunk {trunk} is out of range or a ptrmap position");
        guard += 1;
        assert!(guard <= db.page_count, "freelist trunk chain longer than the file (a cycle)");
        map.insert(trunk, (KIND_FREELIST, 0));
        let page = db.page(trunk);
        let next = be32(page, 0);
        let leaf_count = be32(page, 4) as usize;
        for i in 0..leaf_count {
            let leaf = be32(page, 8 + i * 4);
            assert!(db.is_valid_data_page(leaf), "freelist leaf {leaf} is out of range or a ptrmap position");
            map.insert(leaf, (KIND_FREELIST, 0));
        }
        trunk = next;
    }
}

// ---------------------------------------------------------------------------
// The verifier: reconcile the reconstructed map against the on-disk ptrmap pages, byte by
// byte, plus header + full coverage checks. Returns the set of types seen so a caller can
// assert every type was exercised.
// ---------------------------------------------------------------------------

/// Verify the raw auto_vacuum file `db` against the structure it encodes, for the given
/// `incremental` mode. Returns the distinct entry types observed.
fn verify_ptrmap(db: &RawDb, incremental: bool) -> std::collections::BTreeSet<u8> {
    // Header (§1.3.12): offset 52 non-zero marks a vacuum file; offset 64 is non-zero iff
    // INCREMENTAL. The §1.3.12 invariant "off52==0 ⇒ off64==0" is implied by off52!=0 here.
    assert_ne!(db.largest_root(), 0, "off 52 (largest root b-tree) must be non-zero in a vacuum file");
    if incremental {
        assert_ne!(db.incremental_flag(), 0, "off 64 must be non-zero for INCREMENTAL");
    } else {
        assert_eq!(db.incremental_flag(), 0, "off 64 must be zero for FULL auto_vacuum");
    }

    // The largest root page number in the header must equal the true maximum root
    // (page 1 is always a root, so it is at least 1).
    let mut expected = expected_entries(db);
    let max_root = expected
        .iter()
        .filter(|(_, (k, _))| *k == KIND_ROOT)
        .map(|(p, _)| *p)
        .max()
        .unwrap_or(1)
        .max(1);
    assert_eq!(
        db.largest_root(),
        max_root,
        "off 52 must equal the largest root b-tree page number",
    );

    // Every reconstructed entry is for a real data page, and no parent back-pointer lands
    // on a ptrmap page (b-tree/overflow links never point at a ptrmap page). A parent MAY
    // be page 1 (the schema root can be an interior page whose children back-point to it),
    // so page 1 is not excluded here.
    for (&p, &(_kind, parent)) in &expected {
        assert!(db.is_valid_data_page(p), "reconstructed entry for invalid page {p}");
        assert!(!is_ptrmap_page(db.usable, parent), "entry {p} has ptrmap-page parent {parent}");
    }

    // §1.8 ROOTS-FIRST (fileformat2.html:1094-1096): "all b-tree root pages must come
    // before any non-root b-tree page, cell payload overflow page, or freelist page."
    // Every root created after data (the ubiquitous CREATE INDEX on a populated table, a
    // second CREATE TABLE after inserts) must have been relocated to the front — otherwise
    // real sqlite CORRUPTS the file the moment it auto-vacuums or `PRAGMA integrity_check`s
    // it (the auto-vacuum logic refuses to move a root). `max_root` above is the largest
    // b-tree root (page 1, always the smallest, needs no check); assert every non-root /
    // overflow / freelist page sits strictly ABOVE it. Reconstructed independently, so a
    // relocation that missed a pointer would already have failed the coverage diff below;
    // this catches the placement itself.
    for (&p, &(kind, _)) in &expected {
        if matches!(kind, KIND_FREELIST | KIND_OVERFLOW_HEAD | KIND_OVERFLOW_NEXT | KIND_BTREE) {
            assert!(
                p > max_root,
                "§1.8 roots-first violated: non-root page {p} (type {kind}) precedes the \
                 largest root b-tree page {max_root}",
            );
        }
    }

    // Every non-page-1, non-ptrmap page in the file MUST have exactly the reconstructed
    // entry on disk; and a ptrmap page must itself carry NO entry.
    let mut seen = std::collections::BTreeSet::new();
    for p in 2..=db.page_count {
        if is_ptrmap_page(db.usable, p) {
            assert!(
                !expected.contains_key(&p),
                "ptrmap page {p} must not itself have a ptrmap entry",
            );
            continue;
        }
        let want = expected.remove(&p).unwrap_or_else(|| {
            panic!("page {p} is unaccounted for: no b-tree/overflow/freelist role found for it")
        });
        let got = db.ptrmap_entry(p);
        assert_eq!(
            got, want,
            "ptrmap entry for page {p}: on-disk (type {}, parent {}) != reconstructed (type {}, parent {})",
            got.0, got.1, want.0, want.1,
        );
        seen.insert(want.0);
    }
    assert!(
        expected.is_empty(),
        "reconstructed entries for pages outside 2..=page_count: {expected:?}",
    );

    // Note: every §1.8 ptrmap POSITION being present in the file needs no separate check —
    // the coverage loop above reads each data page's covering ptrmap page via
    // `db.ptrmap_entry` (→ `db.page(cover)`), which panics if a covering position lay past
    // the file end, so a skipped/short ptrmap page fails loudly right there.
    seen
}

// ---------------------------------------------------------------------------
// The builds. Small 512-byte pages so a b-tree splits (type 5) and rows cross the second
// ptrmap page (105) without a huge write; a 4 KiB text forces an overflow chain
// (type 3/4); a scratch table whose rows are all DELETEd stocks the freelist (type 2);
// the index adds a root (type 1) and non-root index pages (type 5).
//
// Freelist coverage uses DELETE, not DROP, on purpose: `DELETE` reclaims the emptied
// b-tree's data pages to the freelist (see `minisqlite-btree`'s `delete.rs`), whereas
// `DROP TABLE` does NOT yet reclaim the dropped object's pages (a pre-existing, documented
// gap in `minisqlite_catalog`'s `drop_object`). Building the freelist via DELETE keeps
// this a strong, well-formed check — every page is reachable or free, so the reconstructed
// map must match the on-disk ptrmap exactly, with no orphaned pages to explain away.
// ---------------------------------------------------------------------------

/// Build a rich auto_vacuum database of the given mode (1 = FULL, 2 = INCREMENTAL) at
/// `tmp`, exercising all five §1.8 entry types, then close it so the header flushes.
fn build_rich_db(tmp: &TempDb, mode: i64) {
    let mut db = tmp.open();
    // page_size and auto_vacuum are both fixed at creation, so set them before any object.
    let _ = db.query("PRAGMA journal_mode=DELETE");
    let _ = db.query("PRAGMA page_size = 512");
    let _ = db.query(&format!("PRAGMA auto_vacuum = {mode}"));

    exec(&mut db, "CREATE TABLE t(a TEXT)");
    // Bulk-load in one explicit transaction (one commit ⇒ one ptrmap rebuild): enough
    // rows to force a multi-level b-tree and to grow past the 2nd ptrmap page (105).
    exec(&mut db, "BEGIN");
    for i in 0..300 {
        exec(&mut db, &format!("INSERT INTO t(a) VALUES ('row-{i:04}-{}')", "x".repeat(40)));
    }
    exec(&mut db, "COMMIT");

    // An index adds a root (type 1) and non-root index pages (type 5).
    exec(&mut db, "CREATE INDEX it ON t(a)");

    // A single very large row forces an overflow chain (type 3 head + type 4 tail).
    exec(&mut db, &format!("INSERT INTO t(a) VALUES ('{}')", "Z".repeat(4000)));

    // A scratch table, populated across several pages then fully emptied with DELETE,
    // reclaims its data pages to the freelist (type 2 trunk + leaves) that this commit's
    // rebuild records. DELETE (not DROP) so the pages are genuinely freed, not leaked.
    exec(&mut db, "CREATE TABLE scratch(x TEXT)");
    exec(&mut db, "BEGIN");
    for i in 0..60 {
        exec(&mut db, &format!("INSERT INTO scratch(x) VALUES ('s-{i:04}-{}')", "y".repeat(40)));
    }
    exec(&mut db, "COMMIT");
    exec(&mut db, "DELETE FROM scratch");
    // The connection drops here → close + flush.
}

/// Reopen `tmp` and confirm the ROWS survived: the 301 rows in `t` (300 loaded + the big
/// one), the big row's exact length, and that the index answers a lookup. This proves the
/// ptrmap writes did not corrupt the actual data (they must be independent overlay pages).
fn assert_rows_survive(tmp: &TempDb) {
    let mut db = tmp.open();
    let count = query(&mut db, "SELECT count(*) FROM t");
    assert!(matches!(count.rows[0][0], Value::Integer(301)), "expected 301 rows, got {:?}", count.rows[0][0]);

    let big = query(&mut db, "SELECT length(a) FROM t WHERE length(a) > 1000");
    assert_eq!(big.rows.len(), 1, "exactly one oversized row");
    assert!(matches!(big.rows[0][0], Value::Integer(4000)), "big row length 4000, got {:?}", big.rows[0][0]);

    // An indexed lookup returns the specific row (the index b-tree must be intact). The
    // literal is built exactly as the row was inserted, so it matches byte-for-byte.
    let needle = format!("row-0100-{}", "x".repeat(40));
    let hit = query(&mut db, &format!("SELECT a FROM t WHERE a = '{needle}'"));
    assert_eq!(hit.rows.len(), 1, "the index lookup finds row 0100");

    // The scratch table still exists but is empty (its data pages were reclaimed to the
    // freelist by DELETE); the emptied b-tree still answers a scan, returning no rows.
    let scratch = query(&mut db, "SELECT count(*) FROM scratch");
    assert!(matches!(scratch.rows[0][0], Value::Integer(0)), "scratch emptied, got {:?}", scratch.rows[0][0]);
}

#[test]
fn full_auto_vacuum_file_is_structurally_valid() {
    let tmp = TempDb::new();
    build_rich_db(&tmp, 1);
    assert_rows_survive(&tmp);

    let db = RawDb::parse(read_db(&tmp));
    assert_eq!(db.page_size, 512, "page_size = 512 was requested before any table");

    // FULL auto_vacuum truncates the freelist off the file at EVERY commit (pragma.html
    // "auto_vacuum"), so the final `DELETE FROM scratch` in `build_rich_db` leaves ZERO
    // free pages and a SMALLER file. A compacted FULL db therefore has no freelist head
    // and no type-2 (FREELIST) entries — that coverage (and the >105 multi-ptrmap-page
    // span) lives in the INCREMENTAL sibling below, which keeps its free pages. Every
    // remaining page is reachable, so `verify_ptrmap` still reconstructs the whole ptrmap.
    assert_eq!(
        db.first_freelist_trunk(),
        0,
        "FULL auto_vacuum must compact to zero free pages at commit (freelist head == 0)",
    );

    // Non-vacuity: the criterion requires >=2 ROOTPAGE entries. The fixture builds three
    // non-page-1 roots (table `t`, index `it`, and table `scratch` whose root survives the
    // DELETE); `verify_ptrmap` byte-checks each root's entry, so this precondition guards
    // against a future fixture shrink silently dropping below multi-root coverage.
    let roots = schema_roots(&db);
    assert!(roots.len() >= 2, "non-vacuity: fixture must contain >=2 root b-trees; got {} ({roots:?})", roots.len());

    let seen = verify_ptrmap(&db, /*incremental=*/ false);
    assert!(
        !seen.contains(&KIND_FREELIST),
        "a compacted FULL db must have no freelist (type-2) entries; found some (seen: {seen:?})",
    );
    for kind in [KIND_ROOT, KIND_OVERFLOW_HEAD, KIND_OVERFLOW_NEXT, KIND_BTREE] {
        assert!(seen.contains(&kind), "entry type {kind} was never exercised (seen: {seen:?})");
    }
}

#[test]
fn incremental_auto_vacuum_file_is_structurally_valid() {
    let tmp = TempDb::new();
    build_rich_db(&tmp, 2);
    assert_rows_survive(&tmp);

    let db = RawDb::parse(read_db(&tmp));
    assert_eq!(db.page_size, 512);
    assert!(db.page_count > 105, "the build must cross the 2nd ptrmap page (105); got {} pages", db.page_count);

    // Non-vacuity: >=2 ROOTPAGE entries (table + index [+ scratch]); see the FULL-mode test.
    let roots = schema_roots(&db);
    assert!(roots.len() >= 2, "non-vacuity: fixture must contain >=2 root b-trees; got {} ({roots:?})", roots.len());

    let seen = verify_ptrmap(&db, /*incremental=*/ true);
    for kind in [KIND_ROOT, KIND_FREELIST, KIND_OVERFLOW_HEAD, KIND_OVERFLOW_NEXT, KIND_BTREE] {
        assert!(seen.contains(&kind), "entry type {kind} was never exercised (seen: {seen:?})");
    }
}

/// A small FULL auto_vacuum database (one tiny table, one row) still produces a correct
/// single-ptrmap-page file. Guards the minimal shape the two header-byte conformance
/// cases hit, but with the full structural check — the exact `[page1][ptrmap 2][root 3]`
/// layout the read-path fixture documents, page 3 = (type 1 root, parent 0).
#[test]
fn minimal_auto_vacuum_file_has_root_entry_on_page_two() {
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        let _ = db.query("PRAGMA journal_mode=DELETE");
        let _ = db.query("PRAGMA auto_vacuum = 1");
        exec(&mut db, "CREATE TABLE t(a)");
        exec(&mut db, "INSERT INTO t(a) VALUES (1)");
    }
    let db = RawDb::parse(read_db(&tmp));
    assert!(is_ptrmap_page(db.usable, 2), "page 2 is the first ptrmap page");
    assert!(!is_ptrmap_page(db.usable, 3), "page 3 is the first data page");
    // Page 3 is the table root: type 1, parent 0, and its entry sits at offset 0 of page 2.
    assert_eq!(ptrmap_page_for(db.usable, 3), 2);
    assert_eq!(ptrmap_entry_offset(db.usable, 3), 0);
    let seen = verify_ptrmap(&db, false);
    assert!(seen.contains(&KIND_ROOT), "the table root must be recorded as type 1");
}

/// An exact reproducer: populate a table in its own transaction,
/// THEN `CREATE INDEX`. The index root is allocated at the TAIL (above `t`'s leaf pages),
/// so without the §1.8 roots-first relocation this file has a non-root page (a leaf of `t`)
/// preceding a root (`it`) — a file real sqlite corrupts on the next auto-vacuum /
/// integrity_check. `verify_ptrmap` now asserts roots-first, and the explicit witness
/// checks the two roots ended up at the two lowest data slots (pages 3 and 4). Both modes.
#[test]
fn roots_first_holds_after_create_index_on_populated_table() {
    for mode in [1i64, 2] {
        let tmp = TempDb::new();
        {
            let mut db = tmp.open();
            let _ = db.query("PRAGMA journal_mode=DELETE");
            let _ = db.query("PRAGMA page_size = 512");
            let _ = db.query(&format!("PRAGMA auto_vacuum = {mode}"));
            exec(&mut db, "CREATE TABLE t(a)");
            exec(&mut db, "BEGIN");
            for i in 0..100 {
                exec(&mut db, &format!("INSERT INTO t(a) VALUES ('row-{i:04}-{}')", "x".repeat(50)));
            }
            exec(&mut db, "COMMIT");
            // t's root is page 3, its leaves are pages 4.. ; this index root is allocated at
            // the tail and MUST be relocated to the front by the commit (§1.8).
            exec(&mut db, "CREATE INDEX it ON t(a)");
        }
        let db = RawDb::parse(read_db(&tmp));
        assert_eq!(db.page_size, 512, "mode {mode}");

        // Two roots (table `t` + index `it`); the full structural + roots-first checks.
        let roots = schema_roots(&db);
        assert_eq!(roots.len(), 2, "mode {mode}: table t and index it, got {roots:?}");
        verify_ptrmap(&db, mode == 2);

        // Explicit witness of the invariant: the two roots occupy the two
        // lowest data slots (pages 3 and 4 at 512-byte pages, page 2 being the ptrmap page),
        // strictly below every leaf/overflow page.
        let mut sorted = roots.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![3, 4], "mode {mode}: roots at the lowest data slots; got {sorted:?}");

        // The data and the index survive the relocation (root moves must rewrite the schema
        // rootpage AND the moved leaf's parent pointer, or a lookup would miss).
        let mut conn = tmp.open();
        let count = query(&mut conn, "SELECT count(*) FROM t");
        assert!(
            matches!(count.rows[0][0], Value::Integer(100)),
            "mode {mode}: 100 rows survive, got {:?}",
            count.rows[0][0]
        );
        let needle = format!("row-0042-{}", "x".repeat(50));
        let hit = query(&mut conn, &format!("SELECT count(*) FROM t WHERE a = '{needle}'"));
        assert!(
            matches!(hit.rows[0][0], Value::Integer(1)),
            "mode {mode}: index lookup finds the relocated-root index's row, got {:?}",
            hit.rows[0][0]
        );
    }
}

/// Roots-first when the lowest open slot is a FREE page: DELETE rows so low data pages land
/// on the freelist, THEN `CREATE INDEX`. INCREMENTAL keeps free pages (FULL would compact
/// them away), so `roots_first` must handle a free target slot (the root takes it; the
/// vacated root slot becomes free and the freelist is rebuilt). `verify_ptrmap` enforces
/// roots-first plus full ptrmap correctness, including the surviving type-2 free entries.
#[test]
fn roots_first_holds_when_low_slots_are_free_before_create_index() {
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        let _ = db.query("PRAGMA journal_mode=DELETE");
        let _ = db.query("PRAGMA page_size = 512");
        let _ = db.query("PRAGMA auto_vacuum = 2"); // INCREMENTAL: free pages persist
        exec(&mut db, "CREATE TABLE t(a)");
        exec(&mut db, "BEGIN");
        for i in 0..200 {
            exec(&mut db, &format!("INSERT INTO t(a) VALUES ('row-{i:04}-{}')", "x".repeat(40)));
        }
        exec(&mut db, "COMMIT");
        // Empty most of the table so many of `t`'s data pages are freed onto the freelist,
        // leaving free pages interleaved among the low slots a relocated root wants.
        exec(&mut db, "DELETE FROM t WHERE rowid > 15");
        exec(&mut db, "CREATE INDEX it ON t(a)");
    }
    let db = RawDb::parse(read_db(&tmp));
    // Full ptrmap correctness + roots-first + type-2 (freelist) coverage.
    let seen = verify_ptrmap(&db, /*incremental=*/ true);
    assert!(seen.contains(&KIND_ROOT), "roots present");

    let mut conn = tmp.open();
    let count = query(&mut conn, "SELECT count(*) FROM t");
    assert!(
        matches!(count.rows[0][0], Value::Integer(15)),
        "15 rows remain after the delete, got {:?}",
        count.rows[0][0]
    );
    // The index resolves against the surviving rows (root relocation kept the index intact).
    let needle = format!("row-0007-{}", "x".repeat(40));
    let hit = query(&mut conn, &format!("SELECT count(*) FROM t WHERE a = '{needle}'"));
    assert!(matches!(hit.rows[0][0], Value::Integer(1)), "index finds a surviving row, got {:?}", hit.rows[0][0]);
}

/// The FREE-TARGET branch of `roots_first::enforce`, exercised DETERMINISTICALLY (the sister
/// test above can hit the SWAP branch instead, since its low slots may stay live). Create
/// three tables FIRST so their roots are exactly pages 3, 4, 5, populate the OUTER two, then
/// `DROP TABLE mid`. The dropped root (page 4) is swept onto the freelist during that commit,
/// so when `enforce` re-packs the two survivors to the lowest slots [3, 4], the target slot 4
/// holds a FREE page — forcing the branch that MOVES a root into a free slot (no live blocker
/// to swap) and rebuilds the freelist. The witness that this branch (not SWAP) ran: `hi`'s old
/// root slot (page 5) must itself become a FREE page afterwards (INCREMENTAL keeps it), and
/// the root set must be exactly [3, 4].
#[test]
fn roots_first_free_target_branch_when_dropped_root_frees_a_low_slot() {
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        let _ = db.query("PRAGMA journal_mode=DELETE");
        let _ = db.query("PRAGMA page_size = 512");
        let _ = db.query("PRAGMA auto_vacuum = 2"); // INCREMENTAL: the freed slot persists as the target
        // Roots become pages 3 (lo), 4 (mid), 5 (hi) in creation order, before any data.
        exec(&mut db, "CREATE TABLE lo(a)");
        exec(&mut db, "CREATE TABLE mid(a)");
        exec(&mut db, "CREATE TABLE hi(a)");
        // Populate only the outer two so `hi`'s root at page 5 is genuinely "too high" once
        // `mid` is gone, and the file has real b-tree structure to reconstruct.
        exec(&mut db, "BEGIN");
        for i in 0..60 {
            exec(&mut db, &format!("INSERT INTO lo(a) VALUES ('lo-{i:04}-{}')", "x".repeat(50)));
            exec(&mut db, &format!("INSERT INTO hi(a) VALUES ('hi-{i:04}-{}')", "y".repeat(50)));
        }
        exec(&mut db, "COMMIT");
        // Dropping the MIDDLE table frees its root (page 4); the commit's roots-first pass must
        // then move `hi`'s root (page 5) DOWN into that free slot 4 (the free-target branch).
        exec(&mut db, "DROP TABLE mid");
    }
    let db = RawDb::parse(read_db(&tmp));
    assert_eq!(db.page_size, 512);

    // Two surviving roots, re-packed to the two lowest data slots.
    let mut roots = schema_roots(&db);
    roots.sort_unstable();
    assert_eq!(roots, vec![3, 4], "survivors re-packed to the lowest slots; got {roots:?}");

    // Full structural correctness + roots-first, and the type-2 entries the freed slots carry.
    let seen = verify_ptrmap(&db, /*incremental=*/ true);
    assert!(seen.contains(&KIND_FREELIST), "the vacated slot must be recorded as a free page");

    // Witness that the FREE-TARGET branch (not SWAP) ran: `hi`'s old root slot (page 5) is now
    // a free page. A SWAP would instead have parked a live blocker there.
    assert_eq!(
        db.ptrmap_entry(5).0,
        KIND_FREELIST,
        "hi's vacated root slot (page 5) must be freed by the free-target relocation"
    );

    // Both surviving tables round-trip through the relocation.
    let mut conn = tmp.open();
    assert!(matches!(query(&mut conn, "SELECT count(*) FROM lo").rows[0][0], Value::Integer(60)));
    assert!(matches!(query(&mut conn, "SELECT count(*) FROM hi").rows[0][0], Value::Integer(60)));
    let hit = query(&mut conn, &format!("SELECT count(*) FROM hi WHERE a = 'hi-0042-{}'", "y".repeat(50)));
    assert!(matches!(hit.rows[0][0], Value::Integer(1)), "relocated-root table still reads its rows");
}

/// TWO roots misplaced in a SINGLE commit, so `enforce` relocates both in one pass — the
/// zip-pairing of misplaced roots to target slots and the `moved`/`current` bookkeeping over
/// more than one swap. Populate `t` (root 3, leaves 4, 5, …), then create TWO indexes inside
/// ONE transaction: both index roots are allocated at the tail, and the single COMMIT's
/// roots-first pass must move BOTH down to the lowest free data slots [4, 5] at once. If the
/// multi-move bookkeeping were wrong (e.g. a blocker repointed to a stale slot), either the
/// verifier's independent walk or an index lookup below would fail.
#[test]
fn roots_first_relocates_two_roots_created_in_one_commit() {
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        let _ = db.query("PRAGMA journal_mode=DELETE");
        let _ = db.query("PRAGMA page_size = 512");
        let _ = db.query("PRAGMA auto_vacuum = 1"); // FULL
        exec(&mut db, "CREATE TABLE t(a, b)");
        exec(&mut db, "BEGIN");
        for i in 0..100 {
            exec(&mut db, &format!("INSERT INTO t(a, b) VALUES ('a-{i:04}-{}', 'b-{i:04}')", "x".repeat(40)));
        }
        exec(&mut db, "COMMIT");
        // Both index roots are created at the tail in ONE transaction, so a single commit
        // leaves two roots above `t`'s leaves — relocated together by one enforce pass.
        exec(&mut db, "BEGIN");
        exec(&mut db, "CREATE INDEX it_a ON t(a)");
        exec(&mut db, "CREATE INDEX it_b ON t(b)");
        exec(&mut db, "COMMIT");
    }
    let db = RawDb::parse(read_db(&tmp));
    assert_eq!(db.page_size, 512);

    // Three roots (t + two indexes), re-packed to the three lowest data slots.
    let mut roots = schema_roots(&db);
    roots.sort_unstable();
    assert_eq!(roots, vec![3, 4, 5], "all three roots at the lowest data slots; got {roots:?}");
    verify_ptrmap(&db, /*incremental=*/ false);

    // Both indexes resolve after the simultaneous relocation (each moved root's schema
    // rootpage AND its swapped blocker's parent pointer had to be rewritten correctly).
    let mut conn = tmp.open();
    assert!(matches!(query(&mut conn, "SELECT count(*) FROM t").rows[0][0], Value::Integer(100)));
    let hit_a = query(&mut conn, &format!("SELECT count(*) FROM t WHERE a = 'a-0031-{}'", "x".repeat(40)));
    assert!(matches!(hit_a.rows[0][0], Value::Integer(1)), "index it_a finds its row after the dual relocation");
    let hit_b = query(&mut conn, "SELECT count(*) FROM t WHERE b = 'b-0077'");
    assert!(matches!(hit_b.rows[0][0], Value::Integer(1)), "index it_b finds its row after the dual relocation");
}

/// THREE roots created by a SINGLE statement, relocated together in one commit. A
/// `CREATE TABLE t2(k INTEGER, b UNIQUE, c UNIQUE)` allocates the table root PLUS the two
/// `sqlite_autoindex_*` roots for the UNIQUE columns atomically — so one commit's
/// roots-first pass must relocate all THREE misplaced roots at once (the multi-move
/// zip-pairing and `moved`/`current` bookkeeping over several swaps in a single pass). This
/// is a DIFFERENT single-statement trigger than the sibling two-`CREATE INDEX`-in-a-
/// transaction test above. The base table `t` is
/// populated AND carries a 4 KiB overflow row first, so the three tail roots displace real
/// non-root and overflow pages when relocated to the front, and the file keeps a type-3/4
/// overflow chain the verifier re-checks after the move.
///
/// Value-exact: the commit succeeds; the file is structurally valid and §1.8 roots-first
/// (independent verifier); the four roots land at the four lowest data slots [3,4,5,6];
/// `PRAGMA page_count`/`freelist_count` are sane; and after a REOPEN every base row and
/// BOTH auto-index lookup paths (`=`, `BETWEEN`, a known-absent key) return exactly the
/// right values, cross-checked against the table/rowid path.
#[test]
fn roots_first_relocates_three_roots_from_one_create_table_with_unique_columns() {
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        let _ = db.query("PRAGMA journal_mode=DELETE");
        let _ = db.query("PRAGMA page_size = 512");
        let _ = db.query("PRAGMA auto_vacuum = 2"); // INCREMENTAL: free/relocated pages persist
        exec(&mut db, "CREATE TABLE t(a TEXT)");
        // Populate t across many low pages and force an overflow chain (a 4 KiB row), so the
        // three roots created below are allocated ABOVE real non-root / overflow pages.
        exec(&mut db, "BEGIN");
        for i in 0..120 {
            exec(&mut db, &format!("INSERT INTO t(a) VALUES ('row-{i:04}-{}')", "x".repeat(40)));
        }
        exec(&mut db, &format!("INSERT INTO t(a) VALUES ('{}')", "Z".repeat(4000)));
        exec(&mut db, "COMMIT");

        // ONE statement, THREE new roots (t2 + sqlite_autoindex_t2_1 + _2), all allocated at
        // the tail: the single commit's roots-first pass relocates all three to the front.
        exec(&mut db, "CREATE TABLE t2(k INTEGER, b UNIQUE, c UNIQUE)");

        // Populate t2 so both UNIQUE auto-indexes carry real keys to look up afterward.
        exec(&mut db, "BEGIN");
        for i in 0..40 {
            exec(&mut db, &format!("INSERT INTO t2(k, b, c) VALUES ({i}, 'b-{i:04}', 'c-{i:04}')"));
        }
        exec(&mut db, "COMMIT");
    }

    // ---- Structural checks on the raw file (an independent second decoder). ----
    let db = RawDb::parse(read_db(&tmp));
    assert_eq!(db.page_size, 512);

    // Four roots (t, t2, and t2's two auto-indexes) re-packed to the four lowest data slots
    // [3,4,5,6] (page 2 is the ptrmap page). t was the only pre-existing root (page 3), so
    // this proves the single CREATE TABLE statement created three roots and all three were
    // relocated down from the tail in one pass.
    let mut roots = schema_roots(&db);
    roots.sort_unstable();
    assert_eq!(
        roots,
        vec![3, 4, 5, 6],
        "t + t2 + two UNIQUE auto-indexes must occupy the four lowest data slots; got {roots:?}",
    );

    // Full structural correctness: every ptrmap entry matches the independently reconstructed
    // (type, parent), off 52 == the largest root, no pointer targets a ptrmap page, and §1.8
    // roots-first holds (every non-root / overflow / freelist page sits above the largest
    // root). This is what a missed pointer or stale-slot repoint in the multi-move would trip.
    let seen = verify_ptrmap(&db, /*incremental=*/ true);
    assert!(seen.contains(&KIND_ROOT), "roots recorded (type 1)");
    assert!(
        seen.contains(&KIND_OVERFLOW_HEAD) && seen.contains(&KIND_OVERFLOW_NEXT),
        "the 4 KiB row's overflow chain (type 3 head + type 4 tail) survived the relocation (seen: {seen:?})",
    );

    // ---- Reopen and assert value-exact data + index correctness (round-trip). ----
    let mut conn = tmp.open();

    // page/freelist counts are sane: page_count matches the on-disk file geometry, and a
    // pure-swap multi-root relocation (no DELETEs) leaves zero free pages.
    assert_scalar(&mut conn, "PRAGMA page_count", int(db.page_count as i64));
    assert_scalar(&mut conn, "PRAGMA freelist_count", int(0));

    // Base table intact: 121 rows (120 + the big one), and the big row's exact length.
    assert_scalar(&mut conn, "SELECT count(*) FROM t", int(121));
    assert_scalar(&mut conn, "SELECT length(a) FROM t WHERE length(a) > 1000", int(4000));

    // t2 intact.
    assert_scalar(&mut conn, "SELECT count(*) FROM t2", int(40));

    // Both UNIQUE auto-indexes answer correctly AFTER the relocation, each cross-checked
    // against the rowid/table path, for equality, a range, and a known-absent key. A wrong
    // schema-rootpage or blocker repoint in the multi-move would corrupt one of these.
    assert_scalar(&mut conn, "SELECT k FROM t2 WHERE b = 'b-0017'", int(17));
    assert_scalar(&mut conn, "SELECT b FROM t2 WHERE k = 17", text("b-0017"));
    assert_scalar(&mut conn, "SELECT k FROM t2 WHERE c = 'c-0033'", int(33));
    assert_scalar(&mut conn, "SELECT c FROM t2 WHERE k = 33", text("c-0033"));
    // Range scan over the `b` auto-index: rows 10..=19 inclusive.
    assert_scalar(
        &mut conn,
        "SELECT count(*) FROM t2 WHERE b BETWEEN 'b-0010' AND 'b-0019'",
        int(10),
    );
    // Known-absent keys are absent on both index paths.
    assert_scalar(&mut conn, "SELECT count(*) FROM t2 WHERE b = 'b-9999'", int(0));
    assert_scalar(&mut conn, "SELECT count(*) FROM t2 WHERE c = 'c-9999'", int(0));

    // Full cross-check: every (k, b, c) triple read via a plain scan equals the expected
    // set, so no relocation-induced row corruption slipped past the point lookups above.
    let expected: Vec<Vec<Value>> = (0..40)
        .map(|i| vec![int(i), text(&format!("b-{i:04}")), text(&format!("c-{i:04}"))])
        .collect();
    assert_rows_unordered(&mut conn, "SELECT k, b, c FROM t2", &expected);
}
