//! Conformance battery: the **byte-level on-disk page-1 database header**.
//!
//! Every other on-disk test in this repo does a SELF round-trip (write via
//! minisqlite, reopen via minisqlite, SELECT) — which passes even if the engine
//! writes a non-standard header that real `sqlite3` could never read back. This
//! file closes that gap: it opens a FILE-backed `minisqlite::Connection`, performs
//! a committed write, closes the connection (flushing the header to the main db
//! file), then reads the RAW bytes with `std::fs::read` and checks the first 100
//! bytes against the documented format.
//!
//! Every expected value is transcribed from the SQLite documentation, NEVER from
//! whatever the current engine happens to emit:
//!   - `spec/sqlite-doc/fileformat2.html` §1.3 ("The Database Header", the offset
//!     table), §1.3.1 (the magic header string), §1.3.2 (page size), §1.3.3 (file
//!     format versions), §1.3.4 (reserved bytes / the 480-byte usable-size floor),
//!     §1.3.5 (payload fractions), §1.3.6 (file change counter), §1.3.7 (in-header
//!     database size validity), §1.3.8 (freelist), §1.3.9 (schema cookie), §1.3.10
//!     (schema format), §1.3.11 (suggested/default cache size), §1.3.12 (vacuum
//!     settings), §1.3.13 (text encoding), §1.3.14 (user_version), §1.3.15
//!     (application_id), §1.3.16 / #validfor (version-valid-for), §1.3.17
//!     (reserved-for-expansion region).
//!   - `spec/sqlite-doc/pragma.html`: `user_version` / `application_id` write the
//!     header fields at offsets 60 / 68; `encoding` defaults to UTF-8; and the
//!     Category-D "header-writing" set-paths — `page_size` (16), `encoding` (56),
//!     `auto_vacuum` (52/64), `schema_version` (40), `default_cache_size` (48) —
//!     each mandate the specific header byte they must write.
//!
//! A failing case here is the intended signal of a WRITE-side format divergence —
//! a header real `sqlite3` would not accept. Many fields are spec-MANDATED
//! CONSTANTS, so a failing assertion on those is unambiguously a write-side
//! format bug.
//!
//! Each case is its own `#[test]` (one header behavior) so one discrepancy fails
//! exactly that case rather than masking the rest — and, for the operation-driven
//! fields, so a WAL-not-checkpointed-on-close divergence is isolated to its case.

mod conformance;
use conformance::*;

// A file-backed connection is opened DIRECTLY (the harness `mem()` is in-memory
// only); `exec` / `query` from the harness take `&mut Connection` and work on it.
use minisqlite::Connection;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Temp-file harness: a unique on-disk path per test, cleaned up even on panic.
// `TempDb` is duplicated verbatim across FOUR on-disk test files — this one plus
// `conformance_ondisk_roundtrip.rs`, `conformance_ondisk_overflow.rs`, and
// `conformance_concurrency.rs` (only the temp-name prefix differs). That is past
// the point where it should be hoisted into the shared `conformance` harness, but
// consolidation edits that high-contention shared seam plus the three siblings, so
// it belongs to the harness owner, not this single tests-only file — kept local.
// ---------------------------------------------------------------------------

/// An RAII guard over a unique temporary database path. Every test gets its own file
/// (isolated + idempotent), and `Drop` deletes the database AND its `-journal` /
/// `-wal` / `-shm` sidecars — so cleanup happens even when an assertion panics and
/// nothing is left behind in the temp dir. No `.db` file is ever committed.
struct TempDb {
    path: PathBuf,
}

impl TempDb {
    /// Build a fresh, unique path under the system temp dir and remove any stale file
    /// (and sidecars) that a previous crashed run might have left at the same name.
    /// Uniqueness combines the process id, a per-process atomic counter, and a
    /// nanosecond timestamp so parallel test threads never collide.
    fn new() -> TempDb {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
        let mut path = std::env::temp_dir();
        path.push(format!("minisqlite_fmt_{pid}_{n}_{nanos}.db"));
        let db = TempDb { path };
        db.remove_all();
        db
    }

    /// Open (or reopen) a file-backed connection to this path, panicking with context
    /// on failure so a test body can use it unwrapped.
    fn open(&self) -> Connection {
        let path: &Path = &self.path;
        Connection::open(path)
            .unwrap_or_else(|e| panic!("Connection::open failed for {path:?}: {e:?}"))
    }

    /// The `<path><suffix>` sibling (e.g. `-journal`), by appending to the FULL path
    /// including its extension — SQLite's own naming convention (`foo.db-journal`).
    fn sidecar(&self, suffix: &str) -> PathBuf {
        let mut s = self.path.as_os_str().to_os_string();
        s.push(suffix);
        PathBuf::from(s)
    }

    /// Delete the database file and every sidecar SQLite may have created. Missing
    /// files are ignored — the expected, non-broken state, not an error to surface.
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

// ---------------------------------------------------------------------------
// Byte helpers. Every multibyte header field is BIG-ENDIAN (§1.3): reading these
// little-endian is the classic bug, so both helpers are big-endian, used everywhere.
// ---------------------------------------------------------------------------

/// The database header is exactly 100 bytes (§1.3). Transcribed, not imported from
/// the engine crate (tests link only the `minisqlite` facade).
const HEADER_SIZE: usize = 100;

/// The 16 magic bytes every SQLite file begins with, incl. the trailing NUL (§1.3.1).
const MAGIC: &[u8; 16] = b"SQLite format 3\x00";

/// Big-endian u16 at `off`.
fn be16(b: &[u8], off: usize) -> u16 {
    u16::from_be_bytes([b[off], b[off + 1]])
}

/// Big-endian u32 at `off`.
fn be32(b: &[u8], off: usize) -> u32 {
    u32::from_be_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

/// The effective page size in bytes: the offset-16 field, with the sentinel `1`
/// decoded to 65536 (§1.3.2).
fn effective_page_size(raw: &[u8]) -> u32 {
    let field = be16(raw, 16);
    if field == 1 { 65_536 } else { field as u32 }
}

/// Best-effort request for rollback-journal (non-WAL) mode, run at the very start of
/// each Category-C test (before the first write) so a just-committed page-1 value is
/// read from the MAIN db file rather than a lagging `-wal` sidecar. The engine
/// currently defaults to rollback AND applies a `journal_mode` switch only at the
/// *next* open, so this is inert today — kept as documented, harmless best-effort: it
/// cannot rescue a WAL-lag failure if the default ever changes mid-connection (that
/// would need the switch before the first write and after each reopen), and it can
/// never mask a real assertion because the result is ignored on purpose (setup must
/// not hide the check). Categories A and B hold regardless of journal mode.
fn request_rollback_journal(db: &mut Connection) {
    let _ = db.query("PRAGMA journal_mode=DELETE");
}

/// Read the whole on-disk database file, asserting a full 100-byte page-1 header is
/// present. A committed write must flush one to the MAIN db file; a 0-byte or short
/// file fails here with a legible reason rather than panicking on an out-of-range
/// slice later. The full file is returned so length-based checks (A6, B7) can use
/// `raw.len()`.
fn read_db(tmp: &TempDb) -> Vec<u8> {
    let raw = std::fs::read(&tmp.path)
        .unwrap_or_else(|e| panic!("failed to read db file {:?}: {e}", tmp.path));
    assert!(
        raw.len() >= HEADER_SIZE,
        "db file {:?} is only {} byte(s); a committed write must flush a full \
         {HEADER_SIZE}-byte page-1 header to the main db file",
        tmp.path,
        raw.len(),
    );
    raw
}

/// Create a fresh file-backed db, perform one committed write (a `CREATE TABLE`, which
/// forces page 1 and its header to be written and flushed), cleanly close it, and
/// return the raw on-disk bytes. The minimal committed state that the header checks
/// need. Category A/B cases share this because they assert format properties that
/// hold for any conformant single-table database.
fn fresh_db_bytes() -> Vec<u8> {
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
    } // the connection is dropped here (closed + flushed)
    read_db(&tmp)
}

// ===========================================================================
// CATEGORY A — MANDATED CONSTANTS. These hold for ANY valid SQLite file
// independent of engine choices; a failure is unambiguously a write-side bug.
// ===========================================================================

#[test]
fn magic_header_string_is_exact_16_bytes() {
    // §1.3.1: every valid SQLite database begins with the 16 bytes
    // 53 51 4c 69 74 65 20 66 6f 72 6d 61 74 20 33 00 = "SQLite format 3\0"
    // (including the trailing NUL). The whole slice is checked, not a prefix.
    let raw = fresh_db_bytes();
    assert_eq!(
        &raw[0..16],
        MAGIC.as_slice(),
        "off 0 magic header: got {:02x?}, expected {:02x?} (\"SQLite format 3\\0\")",
        &raw[0..16],
        MAGIC.as_slice(),
    );
}

#[test]
fn max_embedded_payload_fraction_is_64() {
    // §1.3.5: the maximum embedded payload fraction (offset 21) MUST be 64.
    let raw = fresh_db_bytes();
    assert_eq!(raw[21], 64, "off 21 max embedded payload fraction: got {}, must be 64", raw[21]);
}

#[test]
fn min_embedded_payload_fraction_is_32() {
    // §1.3.5: the minimum embedded payload fraction (offset 22) MUST be 32.
    let raw = fresh_db_bytes();
    assert_eq!(raw[22], 32, "off 22 min embedded payload fraction: got {}, must be 32", raw[22]);
}

#[test]
fn leaf_payload_fraction_is_32() {
    // §1.3.5: the leaf payload fraction (offset 23) MUST be 32.
    let raw = fresh_db_bytes();
    assert_eq!(raw[23], 32, "off 23 leaf payload fraction: got {}, must be 32", raw[23]);
}

#[test]
fn reserved_expansion_region_is_all_zero() {
    // §1.3.17: bytes 72..92 (20 bytes) are reserved for expansion and MUST be zero.
    let raw = fresh_db_bytes();
    let region = &raw[72..92];
    assert!(
        region.iter().all(|&x| x == 0),
        "off 72..92 reserved-for-expansion region must be all zero, got {region:02x?}"
    );
}

#[test]
fn file_length_is_a_whole_number_of_pages() {
    // §1.3.2: the page size lives at offset 16 (big-endian u16; the sentinel 1 means
    // 65536). A valid SQLite file is an exact multiple of the page size and holds at
    // least one page — a non-multiple length OR a bogus page-size field is a
    // cross-read blocker for real sqlite3.
    let raw = fresh_db_bytes();
    let eff = effective_page_size(&raw);
    // Guard the divisor first: a bogus page-size field of 0 makes `effective_page_size`
    // return 0, and the modulo below would then panic with a cryptic divide-by-zero
    // instead of the legible message this test intends (page-size *validity* itself is
    // pinned separately by `page_size_is_valid_power_of_two_or_65536`).
    assert!(
        eff != 0,
        "off 16 page-size field is 0 — not a valid SQLite page size, so the file cannot be a whole number of pages"
    );
    assert!(
        raw.len() as u64 % eff as u64 == 0,
        "off 16 page-size {eff}: file length {} is not a whole number of pages",
        raw.len(),
    );
    assert!(
        raw.len() >= 512,
        "file length {} is smaller than the 512-byte minimum page",
        raw.len(),
    );
}

#[test]
fn usable_page_size_is_at_least_480() {
    // §1.3.4: the "usable size" is the page size (offset 16) minus the reserved
    // bytes-per-page (offset 20), and SQLite forbids it dropping below 480. So a
    // conformant writer must keep page_size - reserved >= 480.
    let raw = fresh_db_bytes();
    let eff = effective_page_size(&raw);
    let reserved = raw[20] as u32;
    let usable = eff.saturating_sub(reserved);
    assert!(
        usable >= 480,
        "usable size (page {eff} - reserved {reserved} = {usable}) is below the 480-byte minimum (§1.3.4)"
    );
}

// ===========================================================================
// CATEGORY B — STRUCTURAL / VALID-RANGE. The engine picks the exact value, but
// the spec constrains the domain; assert membership/relationships, not a value.
// ===========================================================================

#[test]
fn page_size_is_valid_power_of_two_or_65536() {
    // §1.3.2: the offset-16 u16 is either the sentinel 1 (meaning 65536) or a power
    // of two in [512, 32768]. Any other value is not a readable SQLite page size.
    let raw = fresh_db_bytes();
    let ps = be16(&raw, 16);
    let ok = ps == 1 || ((512..=32768).contains(&ps) && ps.is_power_of_two());
    assert!(
        ok,
        "off 16 page-size field {ps:#06x} ({ps}) is neither the sentinel 1 nor a power of two in [512, 32768]"
    );
}

#[test]
fn file_format_versions_are_in_valid_range() {
    // §1.3.3: the write version (offset 18) and read version (offset 19) are each 1
    // (rollback journal) or 2 (WAL) in current SQLite. No other value is defined.
    let raw = fresh_db_bytes();
    let (w, r) = (raw[18], raw[19]);
    assert!(w == 1 || w == 2, "off 18 write-version: got {w}, expected 1 (rollback) or 2 (WAL)");
    assert!(r == 1 || r == 2, "off 19 read-version: got {r}, expected 1 (rollback) or 2 (WAL)");
}

#[test]
fn file_format_write_and_read_versions_match() {
    // §1.3.3: in current SQLite both bytes are 1 together (rollback) or 2 together
    // (WAL) — a conformant writer never emits a mismatched pair for a file it created.
    let raw = fresh_db_bytes();
    assert_eq!(
        raw[18], raw[19],
        "off 18/19 write-version {} != read-version {} — a file real sqlite3 writes has them equal",
        raw[18], raw[19],
    );
}

#[test]
fn schema_format_number_is_1_through_4() {
    // §1.3.10: the schema format number (offset 44) is one of 1, 2, 3, 4. A database
    // that has a schema (we create a table) has a defined, in-range format number.
    let raw = fresh_db_bytes();
    let fmt = be32(&raw, 44);
    assert!((1..=4).contains(&fmt), "off 44 schema-format: got {fmt}, expected one of 1..=4");
}

#[test]
fn text_encoding_is_1_through_3() {
    // §1.3.13: the text encoding (offset 56) is 1 (UTF-8), 2 (UTF-16le), or 3
    // (UTF-16be). "No other values are allowed."
    let raw = fresh_db_bytes();
    let enc = be32(&raw, 56);
    assert!((1..=3).contains(&enc), "off 56 text-encoding: got {enc}, expected 1, 2, or 3");
}

#[test]
fn non_vacuum_db_has_zero_vacuum_fields() {
    // §1.3.12: with no auto_vacuum / incremental_vacuum, the largest-root-b-tree page
    // (offset 52) is 0, and then the incremental-vacuum flag (offset 64) must also be
    // 0 ("If the integer at offset 52 is zero then the integer at offset 64 must also
    // be zero"). We create the db without any vacuum pragma, so both must be zero.
    let raw = fresh_db_bytes();
    let largest_root = be32(&raw, 52);
    let incr_vacuum = be32(&raw, 64);
    assert_eq!(
        largest_root, 0,
        "off 52 largest-root-btree: got {largest_root}, expected 0 for a non-vacuum db"
    );
    assert_eq!(
        incr_vacuum, 0,
        "off 64 incremental-vacuum flag: got {incr_vacuum}, expected 0 when offset 52 is 0"
    );
}

#[test]
fn fresh_db_has_empty_freelist() {
    // §1.3.8: the first-freelist-trunk page (offset 32) is 0 and the freelist page
    // count (offset 36) is 0 when the freelist is empty. A freshly created db with a
    // single CREATE TABLE and no drops/deletes has an empty freelist.
    let raw = fresh_db_bytes();
    let trunk = be32(&raw, 32);
    let count = be32(&raw, 36);
    assert_eq!(trunk, 0, "off 32 first-freelist-trunk: got {trunk}, expected 0 for an empty freelist");
    assert_eq!(count, 0, "off 36 freelist-page-count: got {count}, expected 0 for an empty freelist");
}

#[test]
fn in_header_size_matches_file_length_when_valid() {
    // §1.3.7: the in-header database size in pages (offset 28) is trustworthy only
    // when the change counter (offset 24) equals the version-valid-for number
    // (offset 92). When valid, it must equal the real page count:
    // be32(28) * page_size == file length, and be >= 1.
    let raw = fresh_db_bytes();
    let change_counter = be32(&raw, 24);
    let valid_for = be32(&raw, 92);
    if change_counter == valid_for {
        let pages = be32(&raw, 28);
        let eff = effective_page_size(&raw);
        assert!(pages >= 1, "off 28 in-header page-count is {pages}, expected >= 1 when valid");
        assert_eq!(
            pages as u64 * eff as u64,
            raw.len() as u64,
            "off 28 in-header size ({pages} pages * {eff} bytes) != file length {}",
            raw.len(),
        );
    }
    // When be32(92) != be32(24) the in-header size is documented as untrustworthy, so
    // we deliberately assert nothing about offset 28 in that branch. This is the
    // spec's own validity rule (§1.3.7).
}

// ===========================================================================
// CATEGORY C — OPERATION-DETERMINED. The header must reflect operations run
// through the facade. Each is its own test so a WAL-not-checkpointed divergence
// is isolated. See `request_rollback_journal` for the WAL caveat.
// ===========================================================================

#[test]
fn user_version_defaults_to_zero() {
    // §1.3.14 + pragma.html "user_version": a freshly created db that never set the
    // user version reads back 0 at offset 60.
    let raw = fresh_db_bytes();
    assert_eq!(be32(&raw, 60), 0, "off 60 user_version default: expected 0 on a fresh db");
}

#[test]
fn user_version_persists_small_value() {
    // §1.3.14 + pragma.html: `PRAGMA user_version = N` writes N to offset 60 (a 4-byte
    // big-endian integer). After a clean close the main-file header reads N back.
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        request_rollback_journal(&mut db);
        exec(&mut db, "CREATE TABLE t(a)");
        exec(&mut db, "PRAGMA user_version = 12345");
    }
    let raw = read_db(&tmp);
    assert_eq!(
        be32(&raw, 60),
        12345,
        "off 60 user_version: expected 12345 after `PRAGMA user_version = 12345`, got {}",
        be32(&raw, 60),
    );
}

#[test]
fn user_version_persists_large_value() {
    // §1.3.14: offset 60 is a full 32-bit field, so a large value round-trips through
    // the header bytes. i32::MAX (2147483647) exercises the high-bit region without
    // sign ambiguity in the big-endian byte read.
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        request_rollback_journal(&mut db);
        exec(&mut db, "CREATE TABLE t(a)");
        exec(&mut db, "PRAGMA user_version = 2147483647");
    }
    let raw = read_db(&tmp);
    assert_eq!(
        be32(&raw, 60),
        2147483647,
        "off 60 user_version: expected 2147483647 (i32::MAX), got {}",
        be32(&raw, 60),
    );
}

#[test]
fn application_id_defaults_to_zero() {
    // §1.3.15 + pragma.html "application_id": a fresh db that never set the
    // Application ID reads back 0 at offset 68.
    let raw = fresh_db_bytes();
    assert_eq!(be32(&raw, 68), 0, "off 68 application_id default: expected 0 on a fresh db");
}

#[test]
fn application_id_persists_value() {
    // §1.3.15 + pragma.html: `PRAGMA application_id = N` writes N to offset 68 (a
    // 4-byte big-endian integer). 0x01020304 = 16909060 makes all four bytes distinct,
    // so a big-endian vs little-endian byte-order slip in the writer is caught here.
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        request_rollback_journal(&mut db);
        exec(&mut db, "CREATE TABLE t(a)");
        exec(&mut db, "PRAGMA application_id = 16909060"); // 0x01020304
    }
    let raw = read_db(&tmp);
    assert_eq!(
        be32(&raw, 68),
        0x0102_0304,
        "off 68 application_id: expected 0x01020304 (16909060) after `PRAGMA application_id`, got {:#010x}",
        be32(&raw, 68),
    );
}

#[test]
fn text_encoding_defaults_to_utf8() {
    // §1.3.13 + pragma.html "encoding": a database created with no encoding pragma
    // defaults to UTF-8, which is the code 1 at offset 56. A firm equality, not a range.
    let raw = fresh_db_bytes();
    assert_eq!(be32(&raw, 56), 1, "off 56 text-encoding: expected 1 (UTF-8) by default, got {}", be32(&raw, 56));
}

#[test]
fn schema_cookie_changes_on_ddl() {
    // §1.3.9: the schema cookie (offset 40) is incremented whenever the schema
    // changes. After the first CREATE TABLE it is non-zero; a second CREATE TABLE (a
    // further schema change) must make it strictly larger.
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        request_rollback_journal(&mut db);
        exec(&mut db, "CREATE TABLE t1(a)");
    }
    let cookie1 = be32(&read_db(&tmp), 40);
    {
        let mut db = tmp.open();
        exec(&mut db, "CREATE TABLE t2(b)");
    }
    let cookie2 = be32(&read_db(&tmp), 40);
    assert!(cookie1 >= 1, "off 40 schema-cookie after first DDL: got {cookie1}, expected >= 1");
    assert!(
        cookie2 > cookie1,
        "off 40 schema-cookie must grow on a schema change: {cookie1} -> {cookie2}"
    );
}

#[test]
fn change_counter_increments_on_write() {
    // §1.3.6: the file change counter (offset 24) is incremented whenever the db is
    // unlocked after being modified. Two separate committed writes (a CREATE TABLE,
    // then an INSERT after reopen) must strictly increase it. Only the monotonic
    // increase is pinned — the absolute value is implementation-specific.
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        request_rollback_journal(&mut db);
        exec(&mut db, "CREATE TABLE t(a INTEGER)");
    }
    let cc1 = be32(&read_db(&tmp), 24);
    {
        let mut db = tmp.open();
        exec(&mut db, "INSERT INTO t VALUES (1)");
    }
    let cc2 = be32(&read_db(&tmp), 24);
    assert!(cc2 > cc1, "off 24 change-counter must increase after a write: {cc1} -> {cc2}");
}

#[test]
fn version_valid_for_matches_change_counter_after_clean_close() {
    // §1.3.16 / #validfor: after a clean write + close, the version-valid-for number
    // (offset 92) equals the change counter (offset 24) — which is exactly what makes
    // the in-header database size (offset 28) trustworthy (§1.3.7).
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        request_rollback_journal(&mut db);
        exec(&mut db, "CREATE TABLE t(a INTEGER)");
        exec(&mut db, "INSERT INTO t VALUES (1), (2), (3)");
    }
    let raw = read_db(&tmp);
    assert_eq!(
        be32(&raw, 92),
        be32(&raw, 24),
        "off 92 version-valid-for {} != off 24 change-counter {} after a clean close",
        be32(&raw, 92),
        be32(&raw, 24),
    );
}

// ===========================================================================
// CATEGORY D — HEADER-WRITING PRAGMA SET PATHS (loud-signal)
//
// `PRAGMA page_size`, `encoding`, `auto_vacuum`, `schema_version`, and
// `default_cache_size` each mandate a specific page-1 header BYTE (like
// `user_version`/`application_id` in Category C). Each case runs the pragma with
// ERROR TOLERANCE — `let _ = db.query(..)`, the
// same shape `request_rollback_journal` uses — then asserts the header field. So
// whether the pragma parse-errors OR silently no-ops today, the field keeps its
// default and the assertion fails as a legible byte-mismatch (the intended loud
// signal), and when the engine implements the pragma the byte becomes correct and
// the test passes with NO edit here. The header-byte assertion is the
// invariant; the pragma's own return is deliberately ignored (setup must never
// hide the check).
//
// TIMING (pragma.html): `page_size`, `encoding`, and `auto_vacuum` fix a property
// chosen when the database is CREATED, so they are issued BEFORE the first
// `CREATE TABLE`; `schema_version` (offset 40) and `default_cache_size` (offset 48)
// write their field directly, so they are issued after the table exists. Every
// expected value is transcribed from `spec/sqlite-doc/fileformat2.html` §1.3 and
// `pragma.html`, never engine output.
// ===========================================================================

#[test]
fn page_size_pragma_sets_header_page_size() {
    // §1.3.2 (offset 16): the page size is a 2-byte big-endian value set for the life
    // of the database when it is CREATED (pragma.html "page_size": the new size is
    // used "when the database is first created"), so the pragma precedes the first
    // table. A conformant write both records 8192 at offset 16 AND lays the file out
    // in 8192-byte pages, so the file length is a whole number of 8192.
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        request_rollback_journal(&mut db);
        // Tolerant: a parse-error or a no-op today leaves offset 16 at its default, so
        // the byte assertion below is the loud signal (never the ignored return).
        let _ = db.query("PRAGMA page_size = 8192");
        exec(&mut db, "CREATE TABLE t(a)");
    }
    let raw = read_db(&tmp);
    assert_eq!(
        be16(&raw, 16),
        8192,
        "off 16 page-size: expected 8192 after `PRAGMA page_size = 8192` before CREATE, got {}",
        be16(&raw, 16),
    );
    assert_eq!(
        raw.len() as u32 % 8192,
        0,
        "file length {} must be a whole number of 8192-byte pages — the write must USE \
         the new page size, not merely record it",
        raw.len(),
    );
}

#[test]
fn page_size_pragma_65536_uses_sentinel_one() {
    // §1.3.2 (offset 16): 65536 does not fit in the two-byte field, so it is stored as
    // the sentinel value 1 (byte pattern 0x00 0x01), "thought of as a magic number to
    // represent the 65536 page size". A conformant writer therefore records 1 at
    // offset 16 for a 65536-byte page, and `effective_page_size` decodes it back to
    // 65536. Set before CREATE (the size is fixed at db creation).
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        request_rollback_journal(&mut db);
        let _ = db.query("PRAGMA page_size = 65536");
        exec(&mut db, "CREATE TABLE t(a)");
    }
    let raw = read_db(&tmp);
    assert_eq!(
        be16(&raw, 16),
        1,
        "off 16 page-size: 65536 must be encoded as the sentinel 1, got {}",
        be16(&raw, 16),
    );
    assert_eq!(
        effective_page_size(&raw),
        65_536,
        "off 16 sentinel 1 must decode to a 65536-byte page, got {}",
        effective_page_size(&raw),
    );
}

#[test]
fn encoding_pragma_utf16le_sets_header() {
    // §1.3.13 (offset 56): the text encoding is 1 (UTF-8), 2 (UTF-16le), 3 (UTF-16be).
    // `PRAGMA encoding = 'UTF-16le'` sets the encoding the main database is CREATED
    // with — "it is not possible to change the text encoding of a database after it has
    // been created" (pragma.html "encoding") — so it precedes the first table. A
    // conformant write records 2 at offset 56.
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        request_rollback_journal(&mut db);
        let _ = db.query("PRAGMA encoding = 'UTF-16le'");
        exec(&mut db, "CREATE TABLE t(a)");
    }
    let raw = read_db(&tmp);
    assert_eq!(
        be32(&raw, 56),
        2,
        "off 56 text-encoding: expected 2 (UTF-16le) after `PRAGMA encoding='UTF-16le'`, got {}",
        be32(&raw, 56),
    );
}

#[test]
fn encoding_pragma_utf16be_sets_header() {
    // §1.3.13 (offset 56): 3 means UTF-16be. Set before CREATE — the encoding is fixed
    // at db creation (pragma.html "encoding"). A conformant write records 3 at offset
    // 56.
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        request_rollback_journal(&mut db);
        let _ = db.query("PRAGMA encoding = 'UTF-16be'");
        exec(&mut db, "CREATE TABLE t(a)");
    }
    let raw = read_db(&tmp);
    assert_eq!(
        be32(&raw, 56),
        3,
        "off 56 text-encoding: expected 3 (UTF-16be) after `PRAGMA encoding='UTF-16be'`, got {}",
        be32(&raw, 56),
    );
}

/// Extract a `Text` value or panic — the row-back checks below want the string.
fn as_text(v: &minisqlite::Value) -> String {
    match v {
        minisqlite::Value::Text(s) => s.clone(),
        other => panic!("expected a Text value, got {other:?}"),
    }
}

/// Extract an `Integer` value or panic.
fn as_int(v: &minisqlite::Value) -> i64 {
    match v {
        minisqlite::Value::Integer(i) => *i,
        other => panic!("expected an Integer value, got {other:?}"),
    }
}

/// End-to-end UTF-16le round trip. `PRAGMA encoding='UTF-16le'` before the first table
/// fixes the whole database as UTF-16le (§1.3.13), so the schema row, the data rows, and
/// a secondary index's key records are ALL written in UTF-16le. Reopening reads them back
/// through the same header, and the raw file carries little-endian UTF-16 (ASCII 'x' as
/// the two bytes 0x78 0x00). This is the write half of the on-disk format: the engine's own
/// reader agrees with its writer, and the bytes are the layout real sqlite reads.
#[test]
fn utf16le_database_round_trips_data_and_index() {
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        request_rollback_journal(&mut db);
        let _ = db.query("PRAGMA encoding = 'UTF-16le'");
        exec(&mut db, "CREATE TABLE t(a TEXT, b INTEGER)");
        exec(&mut db, "CREATE INDEX t_a ON t(a)");
        // 'á' (U+00E1) is a non-ASCII BMP char, so it exercises a two-byte UTF-16 unit
        // whose high byte is non-zero — not just ASCII-in-16-bits.
        exec(&mut db, "INSERT INTO t VALUES ('xyz', 1), ('\u{00e1}bc', 2), ('mmm', 3)");
    }
    // Reopen: the header says UTF-16le, so every read transcodes UTF-16le back to UTF-8.
    {
        let mut db = tmp.open();
        let qr = query(&mut db, "SELECT a, b FROM t ORDER BY b");
        let got: Vec<(String, i64)> =
            qr.rows.iter().map(|r| (as_text(&r[0]), as_int(&r[1]))).collect();
        assert_eq!(
            got,
            vec![
                ("xyz".to_string(), 1),
                ("\u{00e1}bc".to_string(), 2),
                ("mmm".to_string(), 3),
            ],
            "UTF-16le rows must read back verbatim after a reopen",
        );
        // An equality lookup on the indexed TEXT column: the seek key is encoded in the
        // same UTF-16le the stored key uses, so it lands and returns the right row.
        let qr = query(&mut db, "SELECT b FROM t WHERE a = '\u{00e1}bc'");
        assert_eq!(qr.rows.len(), 1, "the index lookup finds exactly the one matching row");
        assert_eq!(as_int(&qr.rows[0][0]), 2, "index lookup returns the matching row's b");
    }
    let raw = read_db(&tmp);
    assert_eq!(be32(&raw, 56), 2, "off 56 must be 2 (UTF-16le) for the round trip");
    // 'xyz' is stored little-endian: 'x'=0x78, then 0x00, etc.
    let needle = [0x78u8, 0x00, 0x79, 0x00, 0x7a, 0x00];
    assert!(
        raw.windows(needle.len()).any(|w| w == needle),
        "the file must contain 'xyz' as little-endian UTF-16 bytes",
    );
}

/// The UTF-16be twin of the round trip above: encoding 3 at offset 56, and ASCII 'x' is
/// stored big-endian (0x00 0x78). Same schema/data/index writes, same reopen read-back.
#[test]
fn utf16be_database_round_trips_data_and_index() {
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        request_rollback_journal(&mut db);
        let _ = db.query("PRAGMA encoding = 'UTF-16be'");
        exec(&mut db, "CREATE TABLE t(a TEXT, b INTEGER)");
        exec(&mut db, "CREATE INDEX t_a ON t(a)");
        exec(&mut db, "INSERT INTO t VALUES ('xyz', 1), ('\u{00e1}bc', 2), ('mmm', 3)");
    }
    {
        let mut db = tmp.open();
        let qr = query(&mut db, "SELECT a, b FROM t ORDER BY b");
        let got: Vec<(String, i64)> =
            qr.rows.iter().map(|r| (as_text(&r[0]), as_int(&r[1]))).collect();
        assert_eq!(
            got,
            vec![
                ("xyz".to_string(), 1),
                ("\u{00e1}bc".to_string(), 2),
                ("mmm".to_string(), 3),
            ],
            "UTF-16be rows must read back verbatim after a reopen",
        );
        let qr = query(&mut db, "SELECT b FROM t WHERE a = '\u{00e1}bc'");
        assert_eq!(qr.rows.len(), 1, "the index lookup finds exactly the one matching row");
        assert_eq!(as_int(&qr.rows[0][0]), 2, "index lookup returns the matching row's b");
    }
    let raw = read_db(&tmp);
    assert_eq!(be32(&raw, 56), 3, "off 56 must be 3 (UTF-16be) for the round trip");
    // 'xyz' is stored big-endian: 0x00 then 'x'=0x78, etc.
    let needle = [0x00u8, 0x78, 0x00, 0x79, 0x00, 0x7a];
    assert!(
        raw.windows(needle.len()).any(|w| w == needle),
        "the file must contain 'xyz' as big-endian UTF-16 bytes",
    );
}

// ---------------------------------------------------------------------------
// UTF-16 DML/DDL read-back correctness. The round-trip tests above only exercise
// INSERT + full-scan + one index seek; these pin the DML/DDL paths that DECODE a
// stored row and then compare or re-encode it (AUTOINCREMENT high-water, CREATE
// INDEX backfill, INSERT OR REPLACE victim, UPSERT DO UPDATE, FK cascade). Each
// read-back site must decode in the DB's text encoding — a bare UTF-8 decode reads
// garbage on a UTF-16 database, so each test would fail if that site regressed to
// `decode_table_row`/`decode_table_row_skipping_virtual` (deleted for this reason).
// ---------------------------------------------------------------------------

/// AUTOINCREMENT high-water survives on a UTF-16 database. `sqlite_sequence.name` is TEXT,
/// stored UTF-16; `read_sequence` must decode it in that encoding. A bare UTF-8 decode reads
/// 't\0…' and never matches the table name, so the high-water is lost and the freed rowid is
/// reused — real sqlite never reuses an AUTOINCREMENT rowid (`autoinc.html`). The DELETE is
/// load-bearing: it drops `max(rowid)` to 0, so a lost high-water is observable as reuse.
#[test]
fn utf16_autoincrement_high_water_is_not_lost() {
    let tmp = TempDb::new();
    let mut db = tmp.open();
    let _ = db.query("PRAGMA encoding = 'UTF-16le'");
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY AUTOINCREMENT, v)");
    exec(&mut db, "INSERT INTO t(v) VALUES ('a')"); // id = 1; write_sequence stores ('t', 1)
    exec(&mut db, "DELETE FROM t"); // table empty -> max(rowid) = 0
    exec(&mut db, "INSERT INTO t(v) VALUES ('b')"); // must NOT reuse rowid 1
    let qr = query(&mut db, "SELECT id FROM t");
    assert_eq!(qr.rows.len(), 1, "one row after delete + reinsert");
    assert_eq!(
        as_int(&qr.rows[0][0]),
        2,
        "AUTOINCREMENT never reuses a rowid; the high-water must persist on a UTF-16 DB",
    );
}

/// A secondary index built by BACKFILL over an already-populated UTF-16 table has keys in the
/// DB encoding. `index_build` decodes each existing row to compute its key; a bare UTF-8
/// decode yields corrupt UTF-16 keys, so an index-driven lookup misses. `INDEXED BY i` forces
/// the index access path so a full-table-scan fallback (which reads the data rows correctly)
/// cannot mask a corrupt key.
#[test]
fn utf16_create_index_backfill_keys_are_correct() {
    let tmp = TempDb::new();
    let mut db = tmp.open();
    let _ = db.query("PRAGMA encoding = 'UTF-16le'");
    exec(&mut db, "CREATE TABLE t(a TEXT)");
    // Rows exist BEFORE the index, so CREATE INDEX must backfill from stored records.
    exec(&mut db, "INSERT INTO t VALUES ('x'), ('\u{00e1}y')");
    exec(&mut db, "CREATE INDEX i ON t(a)");
    let qr = query(&mut db, "SELECT a FROM t INDEXED BY i WHERE a = 'x'");
    assert_eq!(qr.rows.len(), 1, "backfilled index finds the ASCII row");
    assert_eq!(as_text(&qr.rows[0][0]), "x");
    let qr = query(&mut db, "SELECT a FROM t INDEXED BY i WHERE a = '\u{00e1}y'");
    assert_eq!(qr.rows.len(), 1, "backfilled index finds the non-ASCII row");
    assert_eq!(as_text(&qr.rows[0][0]), "\u{00e1}y");
}

/// INSERT OR REPLACE removes the victim row's secondary-index entries on a UTF-16 DB. The
/// victim is decoded to recompute its OLD index keys to delete; a bare UTF-8 decode computes
/// the wrong key, deletes nothing, and leaves the real entry orphaned. Proven through a UNIQUE
/// index: after replacing (1,'á') the value 'á' is free, so re-inserting it must be allowed —
/// a lingering orphan entry would wrongly raise a UNIQUE violation.
#[test]
fn utf16_insert_or_replace_deletes_victim_index_entry() {
    let tmp = TempDb::new();
    let mut db = tmp.open();
    let _ = db.query("PRAGMA encoding = 'UTF-16le'");
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, a TEXT)");
    exec(&mut db, "CREATE UNIQUE INDEX i ON t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1, '\u{00e1}')");
    exec(&mut db, "INSERT OR REPLACE INTO t VALUES (1, 'z')"); // victim (1,'á') removed
    try_exec(&mut db, "INSERT INTO t VALUES (2, '\u{00e1}')").expect(
        "the replaced victim's 'á' index entry must be gone, so re-inserting 'á' is allowed",
    );
    let qr = query(&mut db, "SELECT count(*) FROM t");
    assert_eq!(as_int(&qr.rows[0][0]), 2, "both the replaced row and the re-inserted 'á' row exist");
    let qr = query(&mut db, "SELECT id FROM t WHERE a = '\u{00e1}'");
    assert_eq!(qr.rows.len(), 1, "exactly the new row holds 'á'");
    assert_eq!(as_int(&qr.rows[0][0]), 2);
}

/// UPSERT DO UPDATE reads the conflicting row's OLD values in the DB encoding. `SET a = a||…`
/// binds against the existing row, decoded here; a bare UTF-8 decode of the UTF-16 bytes for
/// 'x' ([78 00]) reads "x\0", so the concatenation would produce "x\0z". The real old value
/// is 'x', so the result must be 'xz'.
#[test]
fn utf16_upsert_do_update_reads_existing_row_in_db_encoding() {
    let tmp = TempDb::new();
    let mut db = tmp.open();
    let _ = db.query("PRAGMA encoding = 'UTF-16le'");
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, a TEXT)");
    exec(&mut db, "INSERT INTO t VALUES (1, 'x')");
    exec(&mut db, "INSERT INTO t VALUES (1, 'ignored') ON CONFLICT(id) DO UPDATE SET a = a || 'z'");
    let qr = query(&mut db, "SELECT a FROM t WHERE id = 1");
    assert_eq!(
        as_text(&qr.rows[0][0]),
        "xz",
        "DO UPDATE must concatenate onto the REAL old value 'x', not a UTF-8 misread of it",
    );
}

/// An ON DELETE SET NULL cascade preserves a child's non-FK TEXT column on a UTF-16 DB. The
/// cascade decodes each matching child (`decode_scanned_row`), sets its FK column, and
/// re-encodes the row; a bare UTF-8 decode + UTF-16 re-encode corrupts every OTHER TEXT
/// column. Reopened to prove the rewritten row persisted as valid UTF-16.
#[test]
fn utf16_fk_set_null_cascade_preserves_text_columns() {
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        let _ = db.query("PRAGMA encoding = 'UTF-16le'");
        exec(&mut db, "PRAGMA foreign_keys = ON");
        exec(&mut db, "CREATE TABLE parent(id INTEGER PRIMARY KEY)");
        exec(
            &mut db,
            "CREATE TABLE child(id INTEGER PRIMARY KEY, \
             pid INTEGER REFERENCES parent(id) ON DELETE SET NULL, data TEXT)",
        );
        exec(&mut db, "INSERT INTO parent VALUES (1)");
        exec(&mut db, "INSERT INTO child VALUES (10, 1, 'keepme')");
        exec(&mut db, "DELETE FROM parent WHERE id = 1"); // SET NULL cascade rewrites child 10
    }
    let mut db = tmp.open();
    let qr = query(&mut db, "SELECT pid, data FROM child WHERE id = 10");
    assert_eq!(qr.rows.len(), 1, "the child row survives a SET NULL cascade");
    assert!(matches!(qr.rows[0][0], minisqlite::Value::Null), "the FK column is SET NULL");
    assert_eq!(
        as_text(&qr.rows[0][1]),
        "keepme",
        "the cascade must not corrupt the child's non-FK TEXT column",
    );
}

/// A supplementary-plane (astral) character round-trips on a UTF-16 database. `serial.rs`
/// sizes a UTF-16 TEXT payload as `2 * s.encode_utf16().count()`; a `chars().count()` mistake
/// would size U+1F600 (😀) at 2 bytes instead of its 4-byte surrogate pair, truncating it.
/// The round-trip read-back is the load-bearing check (a bad length corrupts the record); the
/// raw-byte and `length()` assertions pin the exact encoding and character count.
#[test]
fn utf16_supplementary_plane_char_round_trips() {
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        let _ = db.query("PRAGMA encoding = 'UTF-16le'");
        exec(&mut db, "CREATE TABLE t(a TEXT)");
        exec(&mut db, "INSERT INTO t VALUES ('\u{1F600}')"); // 😀 — 2 UTF-16 units = 4 bytes
    }
    let mut db = tmp.open();
    let qr = query(&mut db, "SELECT a, length(a) FROM t");
    assert_eq!(as_text(&qr.rows[0][0]), "\u{1F600}", "the astral char must round-trip verbatim");
    assert_eq!(as_int(&qr.rows[0][1]), 1, "length() counts one character (SQLite char count)");
    // U+1F600 -> surrogate pair D83D DE00 -> little-endian bytes 3D D8 00 DE.
    let raw = read_db(&tmp);
    let needle = [0x3Du8, 0xD8, 0x00, 0xDE];
    assert!(
        raw.windows(needle.len()).any(|w| w == needle),
        "the file must contain the astral char as its UTF-16le surrogate pair",
    );
}

#[test]
fn auto_vacuum_full_sets_header_fields() {
    // §1.3.12 (offsets 52 and 64): in auto-/incremental-vacuum mode the file carries
    // ptrmap pages, so the largest-root-b-tree page at offset 52 is non-zero; the
    // incremental-vacuum flag at offset 64 is "true for incremental_vacuum and false
    // for auto_vacuum". `PRAGMA auto_vacuum = 1` selects FULL, which must be set before
    // any table exists ("auto-vacuuming must be turned on before any tables are
    // created", pragma.html "auto_vacuum"). So offset 52 is non-zero and offset 64 is 0.
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        request_rollback_journal(&mut db);
        let _ = db.query("PRAGMA auto_vacuum = 1");
        exec(&mut db, "CREATE TABLE t(a)");
    }
    let raw = read_db(&tmp);
    assert!(
        be32(&raw, 52) != 0,
        "off 52 largest-root-btree: expected non-zero in FULL auto_vacuum mode, got 0"
    );
    assert_eq!(
        be32(&raw, 64),
        0,
        "off 64 incremental-vacuum flag: expected 0 (false) for FULL auto_vacuum, got {}",
        be32(&raw, 64),
    );
}

#[test]
fn auto_vacuum_incremental_sets_header_flag() {
    // §1.3.12 (offsets 52 and 64): INCREMENTAL vacuum carries ptrmap pages (offset 52
    // non-zero) AND sets the incremental-vacuum flag at offset 64 to true (non-zero).
    // `PRAGMA auto_vacuum = 2` selects INCREMENTAL and must be set before any table is
    // created (pragma.html "auto_vacuum").
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        request_rollback_journal(&mut db);
        let _ = db.query("PRAGMA auto_vacuum = 2");
        exec(&mut db, "CREATE TABLE t(a)");
    }
    let raw = read_db(&tmp);
    assert!(
        be32(&raw, 52) != 0,
        "off 52 largest-root-btree: expected non-zero in INCREMENTAL vacuum mode, got 0"
    );
    assert!(
        be32(&raw, 64) != 0,
        "off 64 incremental-vacuum flag: expected non-zero (true) for INCREMENTAL auto_vacuum, got 0"
    );
}

#[test]
fn schema_version_pragma_sets_cookie() {
    // §1.3.9 (offset 40) + pragma.html "schema_version": `PRAGMA schema_version = N`
    // writes N directly to the 4-byte big-endian schema cookie at offset 40, so unlike
    // page_size/encoding/auto_vacuum (fixed at db creation) it can be issued AFTER the
    // table exists. schema_version is a documented "dangerous" pragma — misusing it can
    // desync a prepared statement and corrupt the database — but its header-byte effect
    // is well-defined, and that byte is all this test asserts.
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        request_rollback_journal(&mut db);
        exec(&mut db, "CREATE TABLE t(a)");
        let _ = db.query("PRAGMA schema_version = 424242");
    }
    let raw = read_db(&tmp);
    assert_eq!(
        be32(&raw, 40),
        424242,
        "off 40 schema-cookie: expected 424242 after `PRAGMA schema_version = 424242`, got {}",
        be32(&raw, 40),
    );
}

#[test]
fn page_size_pragma_512_uses_minimum() {
    // §1.3.2 (offset 16): 512 is the smallest legal page size (the field must be a
    // power of two in [512, 32768], or the sentinel 1). Setting it before CREATE pins
    // the lower boundary of the domain — distinct from the 8192 and 65536-sentinel
    // cases above — and confirms the minimum is written literally, not rejected nor
    // treated as a sentinel. A conformant write records 512 at offset 16.
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        request_rollback_journal(&mut db);
        let _ = db.query("PRAGMA page_size = 512");
        exec(&mut db, "CREATE TABLE t(a)");
    }
    let raw = read_db(&tmp);
    assert_eq!(
        be16(&raw, 16),
        512,
        "off 16 page-size: expected 512 (the minimum) after `PRAGMA page_size = 512`, got {}",
        be16(&raw, 16),
    );
}

#[test]
fn default_cache_size_pragma_sets_header() {
    // §1.3.11 (offset 48) + pragma.html "default_cache_size": `PRAGMA
    // default_cache_size = N` stores the suggested cache size (a page count) in the
    // 4-byte big-endian field at offset 48. Like schema_version it writes the field
    // directly, so it can be issued AFTER the table exists. The pragma is DEPRECATED
    // (kept only for backwards compatibility), so it matters little on the write path
    // — but its header-byte effect is well-defined and it is the same class
    // as the others (`_ => Ok(None)` no-op today). §1.3.11 stores the value as a
    // signed integer whose absolute value is the size, so a positive 500 is written
    // verbatim; this rounds out the gate to ALL settable page-1 header fields.
    let tmp = TempDb::new();
    {
        let mut db = tmp.open();
        request_rollback_journal(&mut db);
        exec(&mut db, "CREATE TABLE t(a)");
        let _ = db.query("PRAGMA default_cache_size = 500");
    }
    let raw = read_db(&tmp);
    assert_eq!(
        be32(&raw, 48),
        500,
        "off 48 default_cache_size: expected 500 after `PRAGMA default_cache_size = 500`, got {}",
        be32(&raw, 48),
    );
}
