//! Conformance battery: the whole-database HEADER PRAGMAs are **NAMESPACE-AWARE**
//! (`main` / `temp` / an `ATTACH`-ed database).
//!
//! These pragmas read/write a field of ONE database file's page-1 header —
//! `user_version`, `application_id`, `schema_version` (the schema cookie),
//! `default_cache_size`, `page_size`, `encoding`, `page_count`, `freelist_count`,
//! `auto_vacuum` — so each must target the database its schema qualifier names, exactly
//! like the object-introspection pragmas in `conformance_pragma_namespace.rs`. Every
//! expected value is TRANSCRIBED FROM THE SQLITE DOCS in `spec/sqlite-doc/`, never from
//! what the engine currently returns; assertions are never weakened to pass.
//!
//! Spec sources (all under `spec/sqlite-doc/`):
//!   * `pragma.html` — "A pragma may have an optional schema-name before the pragma name.
//!     The schema-name is the name of an ATTACH-ed database or 'main' or 'temp' … If the
//!     optional schema name is omitted, 'main' is assumed." So an UNQUALIFIED header
//!     pragma targets `main` UNCONDITIONALLY — NOT the temp→main→attached search order
//!     the OBJECT pragmas use (those resolve an object NAME; a header pragma has no
//!     object argument). This is the crucial distinction pinned below.
//!   * `fileformat2.html` §1.3 — the page-1 header fields, all PER-DATABASE-FILE:
//!     page_size off 16, freelist count off 36 (§1.3.5), schema cookie off 40 (§1.3.9),
//!     default cache size off 48 (§1.3.11), auto-vacuum largest-root-btree off 52
//!     (§1.3.12), text encoding off 56 (§1.3.13), user_version off 60 (§1.3.14),
//!     application_id off 68 (§1.3.15). A freshly-created empty database has each of
//!     these integer fields = 0, page_size the default (4096), UTF-8 encoding, and
//!     exactly one page (page 1, whose first 100 bytes ARE the header).
//!   * `lang_attach.html` — `ATTACH ':memory:' AS aux` adds an in-memory database.
//!
//! DISCRIMINATOR style: distinct per-database values (a settable field set to a
//! different value in main vs aux; a different page size; a grown page count) prove the
//! qualifier — not a hardcoded `main` — selects the answering file. A regression that
//! silently reads/writes `main` (the OLD behavior this file guards against) returns the
//! WRONG database's value and fails loudly, rather than accidentally matching.

mod conformance;

use conformance::*;

// ===========================================================================
// Settable integer header fields (user_version off 60, application_id off 68,
// schema_version/cookie off 40): a DISTINCT value per database proves routing.
// ===========================================================================

#[test]
fn settable_header_fields_honor_the_qualifier() {
    // pragma.html: `PRAGMA schema.user_version = N` sets that schema's page-1 field, and
    // the unqualified form defaults to main. Set DISTINCT values in main vs aux for each
    // settable field; a read of the wrong file would return the wrong number.
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");

    // Three independent header fields, distinct main/aux values (no DDL in between, so
    // nothing else touches the cookie).
    exec(&mut db, "PRAGMA main.user_version = 11");
    exec(&mut db, "PRAGMA aux.user_version = 22");
    exec(&mut db, "PRAGMA main.application_id = 33");
    exec(&mut db, "PRAGMA aux.application_id = 44");
    exec(&mut db, "PRAGMA main.schema_version = 55");
    exec(&mut db, "PRAGMA aux.schema_version = 66");

    // Each qualifier reports its OWN database's field...
    assert_scalar(&mut db, "PRAGMA main.user_version", int(11));
    assert_scalar(&mut db, "PRAGMA aux.user_version", int(22));
    assert_scalar(&mut db, "PRAGMA main.application_id", int(33));
    assert_scalar(&mut db, "PRAGMA aux.application_id", int(44));
    assert_scalar(&mut db, "PRAGMA main.schema_version", int(55));
    assert_scalar(&mut db, "PRAGMA aux.schema_version", int(66));

    // ...and the UNQUALIFIED form is main (pragma.html "main is assumed"), NOT aux and
    // NOT a temp→main→attached search. This is the discriminator: a resolver that fell
    // through to aux would answer 22/44/66 here.
    assert_scalar(&mut db, "PRAGMA user_version", int(11));
    assert_scalar(&mut db, "PRAGMA application_id", int(33));
    assert_scalar(&mut db, "PRAGMA schema_version", int(55));
}

#[test]
fn user_version_is_a_signed_32bit_field_per_database() {
    // fileformat2 §1.3.14: user_version is a 4-byte integer, reported by the pragma as a
    // 32-bit SIGNED value. A negative value round-trips independently per database, so
    // the sign handling is not accidentally shared/overwritten across namespaces.
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "PRAGMA main.user_version = -1");
    exec(&mut db, "PRAGMA aux.user_version = -5");
    assert_scalar(&mut db, "PRAGMA main.user_version", int(-1));
    assert_scalar(&mut db, "PRAGMA aux.user_version", int(-5));
    assert_scalar(&mut db, "PRAGMA user_version", int(-1)); // unqualified == main
}

#[test]
fn default_cache_size_honors_the_qualifier() {
    // pragma.html "default_cache_size" (fileformat2 §1.3.11, off 48): a per-file header
    // integer, read/written per database like user_version. Distinct main/aux values.
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "PRAGMA main.default_cache_size = 111");
    exec(&mut db, "PRAGMA aux.default_cache_size = 222");
    assert_scalar(&mut db, "PRAGMA main.default_cache_size", int(111));
    assert_scalar(&mut db, "PRAGMA aux.default_cache_size", int(222));
    assert_scalar(&mut db, "PRAGMA default_cache_size", int(111)); // unqualified == main
}

// ===========================================================================
// page_size: a SET-when-fresh rebuild of the RESOLVED db's pager (off 16). A distinct
// per-db page size is the discriminator; `main` must NOT be resized by `aux.page_size`.
// ===========================================================================

#[test]
fn page_size_set_and_get_honor_the_qualifier() {
    // pragma.html "page_size" (fileformat2 §1.3.2, off 16): the page size is fixed at
    // creation, settable only while the database is still empty. `ATTACH ':memory:'`
    // gives a fresh aux, so `PRAGMA aux.page_size = 8192` resizes AUX only. main keeps
    // the default 4096; aux reports 8192; the unqualified GET is main (4096). A
    // regression that resized/read `main` would report 8192 for `main.page_size` (or
    // 4096 for `aux.page_size`) and fail here.
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "PRAGMA aux.page_size = 8192");

    assert_scalar(&mut db, "PRAGMA aux.page_size", int(8192));
    assert_scalar(&mut db, "PRAGMA main.page_size", int(4096));
    assert_scalar(&mut db, "PRAGMA page_size", int(4096)); // unqualified == main
}

// ===========================================================================
// page_count: read live from the RESOLVED db's pager. Grow aux so its count differs.
// ===========================================================================

#[test]
fn page_count_honors_the_qualifier() {
    // pragma.html #pragma_page_count: the total pages in the named database. A fresh
    // empty database is exactly one page (fileformat2 §1.3: page 1 holds the header).
    // Creating one rowid table allocates its root page (page 2), so aux grows to 2 pages
    // while main (no objects) stays at 1. Each qualifier reports its OWN count;
    // unqualified is main. A lookup that read main's pager for `aux.page_count` would
    // return 1 and fail loudly.
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.t(x)"); // aux: page 1 + table root page 2 = 2 pages

    assert_scalar(&mut db, "PRAGMA aux.page_count", int(2));
    assert_scalar(&mut db, "PRAGMA main.page_count", int(1));
    assert_scalar(&mut db, "PRAGMA page_count", int(1)); // unqualified == main
}

// ===========================================================================
// freelist_count: per-file header field (off 36). A fresh database's freelist is empty
// (count 0). This shares the exact `read_page1_header_of(db)` path the settable GETs
// exercise, so proving those route correctly proves this does too; here we pin the
// routing + the fresh-empty value + no-panic across every namespace.
// ===========================================================================

#[test]
fn freelist_count_honors_the_qualifier() {
    // fileformat2 §1.3.5: the freelist page count (off 36) is 0 on a freshly-created
    // database. Both main and a fresh aux report 0; unqualified is main. (freelist_count
    // has no SET form, so distinct values would require freeing pages — an
    // engine-specific count this file will not transcribe; the shared header-read path
    // is already discriminated by the settable fields above.)
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    assert_scalar(&mut db, "PRAGMA aux.freelist_count", int(0));
    assert_scalar(&mut db, "PRAGMA main.freelist_count", int(0));
    assert_scalar(&mut db, "PRAGMA freelist_count", int(0)); // unqualified == main
}

// ===========================================================================
// encoding: per-file header field (off 56), settable only while the db is empty.
// ===========================================================================

#[test]
fn encoding_set_and_get_honor_the_qualifier() {
    // fileformat2 §1.3.13 / pragma.html "encoding": each database file records its own
    // text encoding, chosen at creation. A fresh aux may still choose it, so
    // `PRAGMA aux.encoding = 'UTF-16le'` records UTF-16le in AUX only; main stays UTF-8;
    // unqualified reports main (UTF-8). The labels are exactly SQLite's spellings.
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "PRAGMA aux.encoding = 'UTF-16le'");

    assert_scalar(&mut db, "PRAGMA aux.encoding", text("UTF-16le"));
    assert_scalar(&mut db, "PRAGMA main.encoding", text("UTF-8"));
    assert_scalar(&mut db, "PRAGMA encoding", text("UTF-8")); // unqualified == main
}

// ===========================================================================
// Writing a QUALIFIED temp field on a fresh connection materializes temp and PERSISTS,
// exactly as sqlite treats its (auto-created) temp database — and stays isolated from
// main. This pins the not-live-temp WRITE path (no panic, correct persistence).
// ===========================================================================

#[test]
fn write_to_not_live_temp_persists_and_is_isolated_from_main() {
    // On a fresh connection `temp` is reserved-but-not-materialized. `PRAGMA
    // temp.user_version = 99` must materialize temp and store 99 in ITS header (sqlite's
    // temp database always exists and holds its own user_version), NOT silently write
    // main. Reading temp back gives 99; main and the unqualified form remain 0.
    let mut db = mem();
    exec(&mut db, "PRAGMA temp.user_version = 99");

    assert_scalar(&mut db, "PRAGMA temp.user_version", int(99));
    assert_scalar(&mut db, "PRAGMA main.user_version", int(0));
    assert_scalar(&mut db, "PRAGMA user_version", int(0)); // unqualified == main, untouched
}

// ===========================================================================
// Not-live temp READS: a fresh connection with NO temp store must report the
// fresh-empty default for every header pragma, WITHOUT panicking on the absent slot.
// ===========================================================================

#[test]
fn not_live_temp_header_reads_are_fresh_defaults_not_panic() {
    // `temp` resolves to DbIndex(1) unconditionally (pragma.html: `temp` is a valid
    // schema name), but its store is created lazily. A header GET on the not-yet-live
    // temp store must return the FRESH-EMPTY default (fileformat2 §1.3: every header
    // integer field 0, default 4096 page size, UTF-8, one page) rather than index-panic.
    let mut db = mem();

    // Integer header fields default to 0 (fileformat2 §1.3.5/§1.3.9/§1.3.11/§1.3.12/
    // §1.3.14/§1.3.15).
    assert_scalar(&mut db, "PRAGMA temp.user_version", int(0));
    assert_scalar(&mut db, "PRAGMA temp.application_id", int(0));
    assert_scalar(&mut db, "PRAGMA temp.schema_version", int(0));
    assert_scalar(&mut db, "PRAGMA temp.default_cache_size", int(0));
    assert_scalar(&mut db, "PRAGMA temp.freelist_count", int(0));
    assert_scalar(&mut db, "PRAGMA temp.auto_vacuum", int(0));
    // Page size defaults to 4096; a fresh/empty database is exactly one page.
    assert_scalar(&mut db, "PRAGMA temp.page_size", int(4096));
    assert_scalar(&mut db, "PRAGMA temp.page_count", int(1));
    // Text encoding defaults to UTF-8 (fileformat2 §1.3.13: code 0/1 both mean UTF-8).
    assert_scalar(&mut db, "PRAGMA temp.encoding", text("UTF-8"));

    // `temporary` is the accepted long spelling and resolves identically.
    assert_scalar(&mut db, "PRAGMA temporary.user_version", int(0));
    assert_scalar(&mut db, "PRAGMA temporary.page_count", int(1));
}

// ===========================================================================
// Coupling guard: the not-live fresh-empty default the engine reports for a
// reserved-but-unmaterialized namespace MUST equal what a store ACTUALLY materializes to.
// The not-live read path answers from a zeroed default, while a store's real header is
// formatted when it is created; if those two ever diverge, `PRAGMA temp.X` would disagree
// with itself across temp's first write. A fresh `ATTACH ':memory:'` is materialized-empty,
// so pin its (real, created) header to the SAME fileformat2 §1.3 fresh-empty literals the
// not-live-temp test above uses — comparing each side to the SPEC (never to each other, so
// a shared bug can't hide the drift).
// ===========================================================================

#[test]
fn freshly_materialized_store_matches_the_not_live_fresh_defaults() {
    // `aux` is a freshly-created, empty, LIVE database (its page 1 is formatted at ATTACH
    // time); the not-live `temp` above answers from the zeroed default. Both must report
    // the fileformat2 §1.3 fresh-empty values transcribed here. If a store's real fresh
    // format ever diverged from the not-live default (e.g. a different default page size,
    // or a pre-populated field), the two tests would then disagree and this reddens.
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    assert_scalar(&mut db, "PRAGMA aux.user_version", int(0));
    assert_scalar(&mut db, "PRAGMA aux.application_id", int(0));
    assert_scalar(&mut db, "PRAGMA aux.schema_version", int(0));
    assert_scalar(&mut db, "PRAGMA aux.default_cache_size", int(0));
    assert_scalar(&mut db, "PRAGMA aux.freelist_count", int(0));
    assert_scalar(&mut db, "PRAGMA aux.auto_vacuum", int(0));
    assert_scalar(&mut db, "PRAGMA aux.page_size", int(4096));
    assert_scalar(&mut db, "PRAGMA aux.page_count", int(1));
    assert_scalar(&mut db, "PRAGMA aux.encoding", text("UTF-8"));
}

// ===========================================================================
// Unknown qualifier: an unattached schema name resolves to no database, so a header GET
// yields the empty result (columns, zero rows) and a SET is a silent no-op — never an
// error, never a panic, and never a fallback that reads/writes main.
// ===========================================================================

#[test]
fn unknown_qualifier_header_reads_are_empty_with_columns() {
    // Mirrors the introspection pragmas' "no such database" convention: an
    // unknown/unattached qualifier yields the pragma's column with ZERO rows.
    let mut db = mem();
    assert_columns(&mut db, "PRAGMA nope.user_version", &["user_version"]);
    assert_rows(&mut db, "PRAGMA nope.user_version", &[]);
    assert_columns(&mut db, "PRAGMA nope.page_count", &["page_count"]);
    assert_rows(&mut db, "PRAGMA nope.page_count", &[]);
    assert_columns(&mut db, "PRAGMA nope.encoding", &["encoding"]);
    assert_rows(&mut db, "PRAGMA nope.encoding", &[]);
}

#[test]
fn unknown_qualifier_header_set_is_noop_and_does_not_touch_main() {
    // A SET through an unknown qualifier must not error and must not fall back to main:
    // `PRAGMA nope.user_version = 7` changes nothing, and main stays 0. (The whole
    // point of namespace-awareness: a bogus qualifier never corrupts main.)
    let mut db = mem();
    // Must not error (execute succeeds and produces no rows).
    exec(&mut db, "PRAGMA nope.user_version = 7");
    exec(&mut db, "PRAGMA nope.page_size = 512");
    exec(&mut db, "PRAGMA nope.encoding = 'UTF-16be'");
    // main is untouched by any of the bogus-qualifier writes.
    assert_scalar(&mut db, "PRAGMA main.user_version", int(0));
    assert_scalar(&mut db, "PRAGMA main.page_size", int(4096));
    assert_scalar(&mut db, "PRAGMA main.encoding", text("UTF-8"));
}

// ===========================================================================
// Regression guard: a plain main-only connection (no temp, no attach) is byte-for-byte
// unchanged from before the namespace change — the hot path must not regress.
// ===========================================================================

#[test]
fn main_only_header_pragmas_unchanged() {
    // With no temp objects and no attachments, an unqualified header pragma resolves to
    // main with no observable difference. Every fresh value is the fileformat2 §1.3
    // default; a set/get round-trip works; and an explicit `main.` qualifier reaches the
    // same field.
    let mut db = mem();

    // Fresh defaults (fileformat2 §1.3).
    assert_scalar(&mut db, "PRAGMA user_version", int(0));
    assert_scalar(&mut db, "PRAGMA application_id", int(0));
    assert_scalar(&mut db, "PRAGMA schema_version", int(0));
    assert_scalar(&mut db, "PRAGMA default_cache_size", int(0));
    assert_scalar(&mut db, "PRAGMA freelist_count", int(0));
    assert_scalar(&mut db, "PRAGMA auto_vacuum", int(0));
    assert_scalar(&mut db, "PRAGMA page_size", int(4096));
    assert_scalar(&mut db, "PRAGMA page_count", int(1));
    assert_scalar(&mut db, "PRAGMA encoding", text("UTF-8"));

    // A set/get round-trip on the unqualified (== main) form, and the explicit main.
    // qualifier reads the same field back.
    exec(&mut db, "PRAGMA user_version = 42");
    assert_scalar(&mut db, "PRAGMA user_version", int(42));
    assert_scalar(&mut db, "PRAGMA main.user_version", int(42));

    // A GET column name is exactly the pragma name (SQLite names the result column so).
    assert_columns(&mut db, "PRAGMA user_version", &["user_version"]);
    assert_columns(&mut db, "PRAGMA page_count", &["page_count"]);
    assert_columns(&mut db, "PRAGMA encoding", &["encoding"]);
}
