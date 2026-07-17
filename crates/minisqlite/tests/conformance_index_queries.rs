//! Conformance battery: **indexes are a pure PERFORMANCE structure**.
//!
//! The whole point of this file: an index is a redundant, query-accelerating
//! copy of table data (`spec/sqlite-doc/lang_createindex.html`). Therefore
//! adding, using, or dropping an index must NEVER change a query's result set,
//! and the index must stay consistent as rows are INSERTed / UPDATEd / DELETEd.
//! So for every query below the EXPECTED result is derived from the DATA and the
//! documented WHERE / ORDER BY / collation semantics — never from what the engine
//! returns, and never dependent on whether an index exists. A case that fails is
//! the intended signal that the engine diverges from the spec; an assertion is
//! NEVER weakened to match engine output.
//!
//! Spec sources (all under `spec/sqlite-doc/`):
//!   * `lang_createindex.html` — CREATE INDEX / UNIQUE INDEX / DROP INDEX. An
//!     index is a redundant copy of data used only to accelerate queries. §1.1
//!     "Unique Indexes": a duplicate entry is an error, but "all NULL values are
//!     considered different from all other NULL values and are thus unique" (a
//!     UNIQUE index admits many NULLs). §1.2 indexes on expressions, §1.3
//!     descending indexes, §1.5 the COLLATE clause (default BINARY), and the
//!     intro's optional WHERE clause -> a "partial index".
//!   * `lang_select.html` §2.3 "WHERE clause filtering" — "Only rows for which
//!     the WHERE clause expression evaluates to true are included"; a row whose
//!     predicate is false OR NULL is excluded. §4 "The ORDER BY clause" — with no
//!     ASC/DESC, rows sort "in ascending (smaller values first) order by
//!     default", and "SQLite considers NULL values to be smaller than any other
//!     values for sorting purposes" (NULLs first in ASC, last in DESC).
//!   * `datatype3.html` §4 sort order — NULL < INTEGER/REAL < TEXT < BLOB; an
//!     INTEGER and a REAL compare numerically; two TEXT values compare by their
//!     collating sequence. §7 "Collating Sequences" — BINARY compares "using
//!     memcmp()" (byte order of the UTF-8 bytes); NOCASE folds the 26 ASCII
//!     upper-case letters before comparing.
//!
//! Structure: the read-query sections use the helpers below to assert a query
//! returns the spec-derived rows BOTH with no index and with an index created
//! BEFORE the rows — directly pinning "identical whether or not an index is
//! present". The backfill section builds the index AFTER the rows: an index is a
//! copy of ALL existing rows, so `CREATE INDEX` backfills the pre-existing rows
//! into the new index and a lookup finds them. DML-maintenance, DROP INDEX, DESC,
//! and the advanced index features each get their own cases.
//!
//! `Value` has no `PartialEq`/`Ord`, so every comparison goes through the shared
//! harness assertions — never `==` on a `Value`.

mod conformance;
use conformance::*;

// `Value` names the helper signatures below; `Connection` is the fixture return
// type. The harness imports both privately, so they are not in scope via
// `conformance::*`.
use minisqlite::{Connection, Value};

// ===========================================================================
// Shared small fixtures. Kept tiny (a handful of rows) — this is a correctness
// suite, not a benchmark.
// ===========================================================================

// Integer key column with a duplicate key (2 appears twice), a spread for
// ranges (2,5,8) and one NULL key. Used by equality / range / order / backfill.
const INT_SCHEMA: &str = "CREATE TABLE t(a INTEGER, b TEXT)";
const INT_INDEX: &str = "CREATE INDEX i ON t(a)";
const INT_DATA: &str = "INSERT INTO t VALUES (2,'two'),(5,'five'),(2,'twob'),(8,'eight'),(NULL,'nil')";

// Two NULL-keyed rows plus two non-NULL, for the IS NULL / IS NOT NULL cases.
const NULL_DATA: &str = "INSERT INTO t VALUES (1,'one'),(2,'two'),(NULL,'nilA'),(NULL,'nilB')";

// Distinct-keyed rows for the DML-maintenance / DROP / incremental sections.
const DML_DATA: &str = "INSERT INTO t VALUES (1,'x'),(2,'y'),(3,'z')";

// Two-column index fixture (leftmost-prefix + composite equality + a duplicate
// (a,b)=(1,10) pair).
const MC_SCHEMA: &str = "CREATE TABLE t(a INTEGER, b INTEGER, c TEXT)";
const MC_INDEX: &str = "CREATE INDEX i ON t(a,b)";
const MC_DATA: &str = "INSERT INTO t VALUES (1,10,'p'),(1,20,'q'),(2,10,'r'),(2,20,'s'),(1,10,'t')";
// Composite fixture with a NULL in the TRAILING index column ('n' row): for the
// leading-equality + trailing-range NULL-exclusion case.
const MC_NULL_DATA: &str = "INSERT INTO t VALUES (1,10,'p'),(1,20,'q'),(1,NULL,'n'),(2,10,'r')";

// Covering fixture: index (a,b) covers a `SELECT b ... WHERE a` query.
const COV_SCHEMA: &str = "CREATE TABLE t(a INTEGER, b INTEGER)";
const COV_INDEX: &str = "CREATE INDEX i ON t(a,b)";
const COV_DATA: &str = "INSERT INTO t VALUES (1,10),(2,20),(3,30),(2,25)";

// TEXT key: BINARY collation orders by UTF-8 byte value (a < b < m < p).
const TXT_SCHEMA: &str = "CREATE TABLE t(s TEXT, n INTEGER)";
const TXT_INDEX: &str = "CREATE INDEX i ON t(s)";
const TXT_DATA: &str = "INSERT INTO t VALUES ('apple',1),('mango',2),('pear',3),('banana',4)";
// TEXT with a NULL key (n=5). A separate fixture so the NULL-free ORDER BY /
// equality expectations above are not perturbed; used for the upper-bound-range
// and NULL-through-a-TEXT-index generalizations.
const TXT_NULL_DATA: &str =
    "INSERT INTO t VALUES ('apple',1),('mango',2),('pear',3),('banana',4),(NULL,5)";

// REAL key.
const REAL_SCHEMA: &str = "CREATE TABLE t(x REAL, n INTEGER)";
const REAL_INDEX: &str = "CREATE INDEX i ON t(x)";
const REAL_DATA: &str = "INSERT INTO t VALUES (1.5,1),(2.5,2),(3.5,3)";
// REAL with a NULL key (n=4) for the upper-bound-range generalization.
const REAL_NULL_DATA: &str = "INSERT INTO t VALUES (1.5,1),(2.5,2),(3.5,3),(NULL,4)";

// Mixed INTEGER + REAL in a no-affinity column: the stored classes stay distinct
// (1 and 3 are INTEGER, 2.0 is REAL) but compare numerically (datatype3.html §4).
const MIX_SCHEMA: &str = "CREATE TABLE t(v, n TEXT)";
const MIX_INDEX: &str = "CREATE INDEX i ON t(v)";
const MIX_DATA: &str = "INSERT INTO t VALUES (1,'i1'),(2.0,'r'),(3,'i2')";
// Mixed column with a NULL key ('nul') for the upper-bound-range generalization.
const MIX_NULL_DATA: &str = "INSERT INTO t VALUES (1,'i1'),(2.0,'r'),(3,'i2'),(NULL,'nul')";

// ===========================================================================
// Helpers. Each builds fresh in-memory databases and asserts the spec-derived
// expected rows; none of these four returns a Connection (the per-section
// fixtures further down, `desc_fixture` / `dml_fixture`, do).
// ===========================================================================

/// The pure-performance invariant for a read query with NO fully-ordering ORDER
/// BY: the result is the same multiset with no index and with an index created
/// BEFORE the rows (the landed path). Both builds must equal `expected`.
fn same_unordered(schema: &str, index: &str, data: &str, q: &str, expected: &[Vec<Value>]) {
    // (a) No index at all — the full-scan baseline.
    let mut plain = mem();
    exec(&mut plain, schema);
    exec(&mut plain, data);
    assert_rows_unordered(&mut plain, q, expected);

    // (b) Index created BEFORE the rows, so it is maintained as they insert.
    let mut indexed = mem();
    exec(&mut indexed, schema);
    exec(&mut indexed, index);
    exec(&mut indexed, data);
    assert_rows_unordered(&mut indexed, q, expected);
}

/// Same invariant, but for a query whose ORDER BY makes the row sequence total
/// and deterministic, so an ORDERED comparison is valid.
fn same_ordered(schema: &str, index: &str, data: &str, q: &str, expected: &[Vec<Value>]) {
    let mut plain = mem();
    exec(&mut plain, schema);
    exec(&mut plain, data);
    assert_rows(&mut plain, q, expected);

    let mut indexed = mem();
    exec(&mut indexed, schema);
    exec(&mut indexed, index);
    exec(&mut indexed, data);
    assert_rows(&mut indexed, q, expected);
}

/// The BACKFILL path: the index is created AFTER the rows already exist. The
/// spec says the index is a copy of ALL existing rows, so a lookup must still
/// find them — `CREATE INDEX` backfills the pre-existing rows into the new index
/// (`minisqlite_exec::build_index`), so these lookups return the same rows a
/// query with no index would. The `expected` is spec-derived.
fn backfill_unordered(schema: &str, index: &str, data: &str, q: &str, expected: &[Vec<Value>]) {
    let mut db = mem();
    exec(&mut db, schema);
    exec(&mut db, data);
    exec(&mut db, index); // built on already-present rows
    assert_rows_unordered(&mut db, q, expected);
}

/// An ADVANCED index feature (partial / expression / COLLATE) that the engine
/// may not implement yet. The CREATE goes through `try_exec` and its result is
/// ignored here (a dedicated `*_can_be_created` test pins that it should
/// succeed), because the query's result must be CORRECT whether or not the index
/// exists. The index is attempted BEFORE the rows so a supported feature is
/// exercised on the maintained path, isolating it from the separate backfill bug.
fn advanced_index_query_unordered(
    schema: &str,
    index: &str,
    data: &str,
    q: &str,
    expected: &[Vec<Value>],
) {
    let mut db = mem();
    exec(&mut db, schema);
    let _ = try_exec(&mut db, index); // may be unimplemented; correctness must hold regardless
    exec(&mut db, data);
    assert_rows_unordered(&mut db, q, expected);
}

// ===========================================================================
// 1. EQUALITY via an index created BEFORE the inserts (lang_select §2.3).
//    Data keys: 2, 2, 5, 8, NULL.
// ===========================================================================

#[test]
fn equality_unique_key_same_with_and_without_index() {
    // WHERE a = 5 selects exactly the one row with that key; identical with and
    // without the index (an index cannot change which rows match).
    same_unordered(
        INT_SCHEMA,
        INT_INDEX,
        INT_DATA,
        "SELECT b FROM t WHERE a = 5",
        &[vec![text("five")]],
    );
}

#[test]
fn equality_duplicate_key_returns_all_matches() {
    // The key 2 appears twice; equality returns BOTH rows (an index is a copy of
    // every matching row, not a set of distinct keys).
    same_unordered(
        INT_SCHEMA,
        INT_INDEX,
        INT_DATA,
        "SELECT b FROM t WHERE a = 2",
        &[vec![text("two")], vec![text("twob")]],
    );
}

#[test]
fn equality_miss_returns_zero_rows() {
    // No row has key 3, so the predicate is true for none: zero rows, index or
    // not (a seek that finds nothing is still zero rows, never an error).
    same_unordered(
        INT_SCHEMA,
        INT_INDEX,
        INT_DATA,
        "SELECT b FROM t WHERE a = 3",
        &[],
    );
}

// ===========================================================================
// 2. EQUALITY / RANGE via an index created AFTER the inserts — BACKFILL.
//    CREATE INDEX copies the pre-existing rows into the new index
//    (lang_createindex.html: an index is a redundant copy of ALL existing rows),
//    so a lookup finds every prior row — identical to the no-index scan.
// ===========================================================================

#[test]
fn backfill_equality_hit_is_correct() {
    // key 5 exists before the index is built; the lookup must still find it.
    backfill_unordered(
        INT_SCHEMA,
        INT_INDEX,
        INT_DATA,
        "SELECT b FROM t WHERE a = 5",
        &[vec![text("five")]],
    );
}

#[test]
fn backfill_equality_duplicate_key_is_correct() {
    // Both pre-existing key-2 rows must appear once the index is built.
    backfill_unordered(
        INT_SCHEMA,
        INT_INDEX,
        INT_DATA,
        "SELECT b FROM t WHERE a = 2",
        &[vec![text("two")], vec![text("twob")]],
    );
}

#[test]
fn backfill_equality_miss_is_zero_rows() {
    // A genuine miss is zero rows either way; a backfilled index adds only the
    // rows that exist, so a key no row has still returns nothing (never a
    // spurious hit).
    backfill_unordered(
        INT_SCHEMA,
        INT_INDEX,
        INT_DATA,
        "SELECT b FROM t WHERE a = 7",
        &[],
    );
}

#[test]
fn backfill_range_is_correct() {
    // A range over the backfilled index must return every pre-existing row in
    // range (keys 5 and 8 for a >= 5); NULL never satisfies a comparison.
    backfill_unordered(
        INT_SCHEMA,
        INT_INDEX,
        INT_DATA,
        "SELECT b FROM t WHERE a >= 5",
        &[vec![text("five")], vec![text("eight")]],
    );
}

// ===========================================================================
// 3. RANGE via index (lang_select §2.3; datatype3 §4 numeric comparison).
//    Data keys: 2, 2, 5, 8, NULL. NULL satisfies no comparison, so 'nil' never
//    appears in any range result.
// ===========================================================================

#[test]
fn range_gt_same_with_and_without_index() {
    // a > 2  -> keys 5, 8.
    same_unordered(
        INT_SCHEMA,
        INT_INDEX,
        INT_DATA,
        "SELECT b FROM t WHERE a > 2",
        &[vec![text("five")], vec![text("eight")]],
    );
}

#[test]
fn range_ge_same_with_and_without_index() {
    // a >= 2 -> keys 2, 2, 5, 8 (the boundary is included).
    same_unordered(
        INT_SCHEMA,
        INT_INDEX,
        INT_DATA,
        "SELECT b FROM t WHERE a >= 2",
        &[
            vec![text("two")],
            vec![text("twob")],
            vec![text("five")],
            vec![text("eight")],
        ],
    );
}

#[test]
fn range_lt_same_with_and_without_index() {
    // a < 5 -> keys 2, 2 (8 excluded; NULL excluded).
    same_unordered(
        INT_SCHEMA,
        INT_INDEX,
        INT_DATA,
        "SELECT b FROM t WHERE a < 5",
        &[vec![text("two")], vec![text("twob")]],
    );
}

#[test]
fn range_le_same_with_and_without_index() {
    // a <= 5 -> keys 2, 2, 5 (the boundary is included).
    same_unordered(
        INT_SCHEMA,
        INT_INDEX,
        INT_DATA,
        "SELECT b FROM t WHERE a <= 5",
        &[vec![text("two")], vec![text("twob")], vec![text("five")]],
    );
}

#[test]
fn range_between_same_with_and_without_index() {
    // BETWEEN is inclusive on both ends: a BETWEEN 2 AND 5 -> keys 2, 2, 5.
    same_unordered(
        INT_SCHEMA,
        INT_INDEX,
        INT_DATA,
        "SELECT b FROM t WHERE a BETWEEN 2 AND 5",
        &[vec![text("two")], vec![text("twob")], vec![text("five")]],
    );
}

#[test]
fn range_two_sided_same_with_and_without_index() {
    // Two one-sided bounds ANDed together are the same closed interval as
    // BETWEEN: a >= 2 AND a <= 5 -> keys 2, 2, 5.
    same_unordered(
        INT_SCHEMA,
        INT_INDEX,
        INT_DATA,
        "SELECT b FROM t WHERE a >= 2 AND a <= 5",
        &[vec![text("two")], vec![text("twob")], vec![text("five")]],
    );
}

#[test]
fn range_ordered_by_key_ascending() {
    // A range with ORDER BY a is totally ordered on the projected key (the two
    // key-2 rows project the identical value int(2), so their relative order is
    // immaterial to the row sequence). a >= 2 AND a <= 8 -> 2, 2, 5, 8 ascending.
    same_ordered(
        INT_SCHEMA,
        INT_INDEX,
        INT_DATA,
        "SELECT a FROM t WHERE a >= 2 AND a <= 8 ORDER BY a",
        &[vec![int(2)], vec![int(2)], vec![int(5)], vec![int(8)]],
    );
}

// ===========================================================================
// 4. ORDER BY served by an ASC index (lang_select §4; lang_createindex §1.4).
//    Ascending is the default; NULL sorts smaller than every value, so it comes
//    first. Projecting only `a`, tied key-2 rows are indistinguishable, so the
//    sequence [NULL, 2, 2, 5, 8] is deterministic for an ordered assert.
// ===========================================================================

#[test]
fn order_by_default_ascending_nulls_first() {
    same_ordered(
        INT_SCHEMA,
        INT_INDEX,
        INT_DATA,
        "SELECT a FROM t ORDER BY a",
        &[
            vec![null()],
            vec![int(2)],
            vec![int(2)],
            vec![int(5)],
            vec![int(8)],
        ],
    );
}

#[test]
fn order_by_asc_keyword_matches_default() {
    // Explicit ASC is identical to the default ascending order.
    same_ordered(
        INT_SCHEMA,
        INT_INDEX,
        INT_DATA,
        "SELECT a FROM t ORDER BY a ASC",
        &[
            vec![null()],
            vec![int(2)],
            vec![int(2)],
            vec![int(5)],
            vec![int(8)],
        ],
    );
}

// ===========================================================================
// 5. DESC index + ORDER BY DESC (lang_createindex §1.3; lang_select §4).
//    DESC sorts larger values first, and NULL (smallest) lands at the END of a
//    descending order. If `CREATE INDEX ... (a DESC)` is unimplemented the
//    dedicated `_can_be_created` test fails loudly; the two ORDER BY cases guard
//    the CREATE so they exercise the DESC-index path only when it is available.
// ===========================================================================

/// The single fixture for the DESC section: keys 2, 5, 2, 8, NULL in one column.
fn desc_fixture() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (2),(5),(2),(8),(NULL)");
    db
}

#[test]
fn desc_index_can_be_created() {
    // lang_createindex §1.3: a column may be followed by DESC. This must be
    // accepted; if the engine rejects it, this case fails loudly (not skipped).
    let mut db = desc_fixture();
    let r = try_exec(&mut db, "CREATE INDEX i ON t(a DESC)");
    assert!(
        r.is_ok(),
        "CREATE INDEX i ON t(a DESC) should be accepted (lang_createindex.html §1.3); got {r:?}"
    );
}

#[test]
fn desc_index_serves_order_by_desc() {
    // With a DESC index present, ORDER BY a DESC is larger-first with NULL last:
    // 8, 5, 2, 2, NULL. Guarded on the CREATE so this exercises the DESC path
    // only where it exists (the `_can_be_created` test owns the gap).
    let mut db = desc_fixture();
    if try_exec(&mut db, "CREATE INDEX i ON t(a DESC)").is_ok() {
        assert_rows(
            &mut db,
            "SELECT a FROM t ORDER BY a DESC",
            &[
                vec![int(8)],
                vec![int(5)],
                vec![int(2)],
                vec![int(2)],
                vec![null()],
            ],
        );
    }
}

#[test]
fn desc_index_order_by_asc_still_correct() {
    // A DESC index must not corrupt an ascending scan: ORDER BY a is still
    // NULL, 2, 2, 5, 8. Guarded on the DESC CREATE as above.
    let mut db = desc_fixture();
    if try_exec(&mut db, "CREATE INDEX i ON t(a DESC)").is_ok() {
        assert_rows(
            &mut db,
            "SELECT a FROM t ORDER BY a",
            &[
                vec![null()],
                vec![int(2)],
                vec![int(2)],
                vec![int(5)],
                vec![int(8)],
            ],
        );
    }
}

#[test]
fn desc_index_range_upper_bound_excludes_null() {
    // The DESC (reverse-scanned) mirror of the ascending `<`/`<=` NULL-exclusion:
    // an upper-bound-only range (a <= 5) must EXCLUDE the NULL-keyed row (NULL <=
    // 5 is NULL, not true — lang_select §2.3, datatype3 §3). A reverse index scan
    // starting at the high end must not emit NULL keys any more than a forward
    // one does. The no-index baseline is always checked; the DESC-index arm is
    // guarded on the CREATE (the `desc_index_can_be_created` companion owns that
    // gap), so an unimplemented DESC index is never a silent skip.
    // Shared by both arms so the data / query / expected can never drift apart.
    let data = "INSERT INTO t VALUES (2,'two'),(5,'five'),(8,'eight'),(NULL,'nil')";
    let q = "SELECT b FROM t WHERE a <= 5";
    let want = [vec![text("two")], vec![text("five")]];

    let mut plain = mem();
    exec(&mut plain, INT_SCHEMA);
    exec(&mut plain, data);
    assert_rows_unordered(&mut plain, q, &want);

    let mut indexed = mem();
    exec(&mut indexed, INT_SCHEMA);
    if try_exec(&mut indexed, "CREATE INDEX i ON t(a DESC)").is_ok() {
        exec(&mut indexed, data);
        assert_rows_unordered(&mut indexed, q, &want);
    }
}

// ===========================================================================
// 6. MULTI-COLUMN index (a, b) (lang_createindex; lang_select §2.3, §4).
//    Composite equality, leftmost-prefix equality, and ORDER BY a, b.
// ===========================================================================

#[test]
fn multicol_equality_on_both_columns() {
    // WHERE a=1 AND b=10 matches the two rows with that composite key -> c 'p','t'.
    same_unordered(
        MC_SCHEMA,
        MC_INDEX,
        MC_DATA,
        "SELECT c FROM t WHERE a = 1 AND b = 10",
        &[vec![text("p")], vec![text("t")]],
    );
}

#[test]
fn multicol_leftmost_prefix_equality() {
    // WHERE a=2 uses only the leftmost index column; both a=2 rows match -> 'r','s'.
    same_unordered(
        MC_SCHEMA,
        MC_INDEX,
        MC_DATA,
        "SELECT c FROM t WHERE a = 2",
        &[vec![text("r")], vec![text("s")]],
    );
}

#[test]
fn multicol_order_by_a_b() {
    // ORDER BY a, b projecting (a, b): the tied (1,10) rows project the identical
    // pair, so the sequence is deterministic: (1,10),(1,10),(1,20),(2,10),(2,20).
    same_ordered(
        MC_SCHEMA,
        MC_INDEX,
        MC_DATA,
        "SELECT a, b FROM t ORDER BY a, b",
        &[
            vec![int(1), int(10)],
            vec![int(1), int(10)],
            vec![int(1), int(20)],
            vec![int(2), int(10)],
            vec![int(2), int(20)],
        ],
    );
}

#[test]
fn multicol_trailing_range_excludes_null() {
    // The upper-bound NULL-exclusion (lang_select §2.3; datatype3 §3) on the
    // TRAILING column of a composite index: `WHERE a = 1 AND b <= 15` seeks the
    // a=1 group by the equality prefix, then range-scans b. The a=1 group's rows
    // are b=10('p'), b=20('q'), b=NULL('n'); only b=10 satisfies b <= 15, and the
    // b=NULL row is excluded (`NULL <= 15` is NULL, not true). (a=2 row 'r' is
    // filtered by the equality prefix.) Expected {'p'} — identical with and
    // without the index. The same forward index range scan that leaks NULL on a
    // leading column (`range_le_*`) must not leak it at the bottom of the a=1
    // group either, so this pins the trailing-column instance of the same shape.
    same_unordered(
        MC_SCHEMA,
        MC_INDEX,
        MC_NULL_DATA,
        "SELECT c FROM t WHERE a = 1 AND b <= 15",
        &[vec![text("p")]],
    );
}

// ===========================================================================
// 7. UNIQUE index result correctness (lang_createindex §1.1).
//    Each stored row is returned once; a duplicate is rejected; NULLs are all
//    distinct, so many are allowed and IS NULL returns them all.
// ===========================================================================

#[test]
fn unique_index_lookup_returns_row_once() {
    // A UNIQUE index still stores one entry per row; a lookup returns exactly the
    // matching row, and the table holds every distinct-keyed row.
    let mut db = mem();
    exec(&mut db, INT_SCHEMA);
    exec(&mut db, "CREATE UNIQUE INDEX i ON t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1,'x'),(2,'y'),(3,'z')");
    assert_rows_unordered(&mut db, "SELECT b FROM t WHERE a = 2", &[vec![text("y")]]);
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(3));
}

#[test]
fn unique_index_rejects_duplicate() {
    // §1.1: "Any attempt to insert a duplicate entry will result in an error."
    let mut db = mem();
    exec(&mut db, INT_SCHEMA);
    exec(&mut db, "CREATE UNIQUE INDEX i ON t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1,'x')");
    assert_exec_error(&mut db, "INSERT INTO t VALUES (1,'z')");
}

#[test]
fn unique_index_allows_multiple_nulls() {
    // §1.1: "all NULL values are considered different from all other NULL values
    // and are thus unique" — so two NULL keys are not a violation; both survive.
    let mut db = mem();
    exec(&mut db, INT_SCHEMA);
    exec(&mut db, "CREATE UNIQUE INDEX i ON t(a)");
    exec(&mut db, "INSERT INTO t VALUES (NULL,'a'),(NULL,'b')");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2));
}

#[test]
fn unique_index_where_is_null_returns_all_null_rows() {
    // The NULL-keyed rows are all retained and all matched by IS NULL.
    let mut db = mem();
    exec(&mut db, INT_SCHEMA);
    exec(&mut db, "CREATE UNIQUE INDEX i ON t(a)");
    exec(&mut db, "INSERT INTO t VALUES (NULL,'a'),(NULL,'b'),(1,'x')");
    assert_rows_unordered(
        &mut db,
        "SELECT b FROM t WHERE a IS NULL",
        &[vec![text("a")], vec![text("b")]],
    );
}

// ===========================================================================
// 8. NULL handling through an index (lang_select §2.3; datatype3 §3 NULL).
//    Equality never matches a NULL-keyed row; IS NULL / IS NOT NULL partition
//    the table. The first block uses an integer key (1, 2, NULL, NULL); the
//    second repeats the same invariants through a TEXT index, since the rule is
//    key-type-generic (a NULL key satisfies no `=` and is matched only by IS
//    NULL) and must not silently break for one storage class.
// ===========================================================================

#[test]
fn equality_never_matches_null_row() {
    // a = 1 matches only the key-1 row; the two NULL-keyed rows are never equal
    // to any value (a = 1 is NULL for them, not true).
    same_unordered(
        INT_SCHEMA,
        INT_INDEX,
        NULL_DATA,
        "SELECT b FROM t WHERE a = 1",
        &[vec![text("one")]],
    );
}

#[test]
fn where_is_null_returns_null_rows() {
    // IS NULL returns exactly the two NULL-keyed rows.
    same_unordered(
        INT_SCHEMA,
        INT_INDEX,
        NULL_DATA,
        "SELECT b FROM t WHERE a IS NULL",
        &[vec![text("nilA")], vec![text("nilB")]],
    );
}

#[test]
fn where_is_not_null_returns_the_rest() {
    // IS NOT NULL returns the complement — the two non-NULL rows.
    same_unordered(
        INT_SCHEMA,
        INT_INDEX,
        NULL_DATA,
        "SELECT b FROM t WHERE a IS NOT NULL",
        &[vec![text("one")], vec![text("two")]],
    );
}

#[test]
fn text_equality_never_matches_null_row() {
    // The same rule through a TEXT index: s = 'mango' matches only that key; the
    // NULL-keyed row (n=5) is never equal to any value.
    same_unordered(
        TXT_SCHEMA,
        TXT_INDEX,
        TXT_NULL_DATA,
        "SELECT n FROM t WHERE s = 'mango'",
        &[vec![int(2)]],
    );
}

#[test]
fn text_where_is_null_returns_null_row() {
    // IS NULL returns exactly the NULL-keyed row (n=5) through a TEXT index.
    same_unordered(
        TXT_SCHEMA,
        TXT_INDEX,
        TXT_NULL_DATA,
        "SELECT n FROM t WHERE s IS NULL",
        &[vec![int(5)]],
    );
}

#[test]
fn text_where_is_not_null_returns_the_rest() {
    // IS NOT NULL returns the four non-NULL rows through a TEXT index.
    same_unordered(
        TXT_SCHEMA,
        TXT_INDEX,
        TXT_NULL_DATA,
        "SELECT n FROM t WHERE s IS NOT NULL",
        &[vec![int(1)], vec![int(2)], vec![int(3)], vec![int(4)]],
    );
}

// ===========================================================================
// 9. TEXT and REAL keys (datatype3 §4 sort order, §7 BINARY collation).
//    TEXT compares by BINARY = memcmp of UTF-8 bytes; REAL numerically; a mixed
//    INTEGER/REAL column compares numerically across the two classes.
// ===========================================================================

#[test]
fn text_index_equality() {
    // Equality on a TEXT key returns the matching row.
    same_unordered(
        TXT_SCHEMA,
        TXT_INDEX,
        TXT_DATA,
        "SELECT n FROM t WHERE s = 'mango'",
        &[vec![int(2)]],
    );
}

#[test]
fn text_index_range_binary_order() {
    // BINARY (byte) order: 'apple'/'banana' start with 'a'/'b' (< 'm'), while
    // 'mango'/'pear' are >= 'm'. So s >= 'm' -> 'mango'(2), 'pear'(3).
    same_unordered(
        TXT_SCHEMA,
        TXT_INDEX,
        TXT_DATA,
        "SELECT n FROM t WHERE s >= 'm'",
        &[vec![int(2)], vec![int(3)]],
    );
}

#[test]
fn text_index_order_by_binary() {
    // ORDER BY s ascending in BINARY byte order: apple < banana < mango < pear.
    same_ordered(
        TXT_SCHEMA,
        TXT_INDEX,
        TXT_DATA,
        "SELECT s FROM t ORDER BY s",
        &[
            vec![text("apple")],
            vec![text("banana")],
            vec![text("mango")],
            vec![text("pear")],
        ],
    );
}

#[test]
fn real_index_equality() {
    // Equality on a REAL key.
    same_unordered(
        REAL_SCHEMA,
        REAL_INDEX,
        REAL_DATA,
        "SELECT n FROM t WHERE x = 2.5",
        &[vec![int(2)]],
    );
}

#[test]
fn real_index_range() {
    // x >= 2.0 -> 2.5(2), 3.5(3); 1.5 excluded.
    same_unordered(
        REAL_SCHEMA,
        REAL_INDEX,
        REAL_DATA,
        "SELECT n FROM t WHERE x >= 2.0",
        &[vec![int(2)], vec![int(3)]],
    );
}

#[test]
fn mixed_int_real_equality_is_numeric() {
    // datatype3 §4: an INTEGER and a REAL compare numerically, so v = 2 matches
    // the stored REAL 2.0 (row 'r'), even though 2 is an INTEGER literal.
    same_unordered(
        MIX_SCHEMA,
        MIX_INDEX,
        MIX_DATA,
        "SELECT n FROM t WHERE v = 2",
        &[vec![text("r")]],
    );
}

#[test]
fn mixed_int_real_range_is_numeric() {
    // v >= 2 compares numerically across classes: REAL 2.0 ('r') and INTEGER 3
    // ('i2') qualify; INTEGER 1 does not.
    same_unordered(
        MIX_SCHEMA,
        MIX_INDEX,
        MIX_DATA,
        "SELECT n FROM t WHERE v >= 2",
        &[vec![text("r")], vec![text("i2")]],
    );
}

// ===========================================================================
// 9b. UPPER-bound-only range EXCLUDES NULL for every key type (lang_select §2.3;
//     datatype3 §3). `NULL <= k` is NULL (not true), so a NULL-keyed row is
//     excluded from an `a <= k` / `s <= k` / `x <= k` result — regardless of
//     whether an index backs the query. This is the same shape as the integer
//     `range_lt`/`range_le` cases, generalized to TEXT, REAL, and a no-affinity
//     column: the index range scan is key-type-generic, so a NULL leak on the
//     upper-bound (no-lower-bound) direction must be pinned for each storage
//     class, not just INTEGER, or a future type-specialized index path could fix
//     one and leave the others silently wrong. Each `*_NULL` fixture adds one
//     NULL-keyed row; the expected set excludes it. (The `>=`/`>` lower-bound
//     tests above seek PAST NULL and so do not expose this — only `<`/`<=` do.)
// ===========================================================================

#[test]
fn range_le_text_excludes_null() {
    // s <= 'm' in BINARY order: 'apple'(1) and 'banana'(4) qualify ('a','b' < 'm');
    // 'mango'/'pear' exceed 'm'; the NULL-keyed row (5) is excluded (NULL <= 'm'
    // is NULL, not true).
    same_unordered(
        TXT_SCHEMA,
        TXT_INDEX,
        TXT_NULL_DATA,
        "SELECT n FROM t WHERE s <= 'm'",
        &[vec![int(1)], vec![int(4)]],
    );
}

#[test]
fn range_le_real_excludes_null() {
    // x <= 2.5 -> 1.5(1), 2.5(2); 3.5 excluded; the NULL-keyed row (4) excluded.
    same_unordered(
        REAL_SCHEMA,
        REAL_INDEX,
        REAL_NULL_DATA,
        "SELECT n FROM t WHERE x <= 2.5",
        &[vec![int(1)], vec![int(2)]],
    );
}

#[test]
fn range_le_mixed_excludes_null() {
    // v <= 2 compares numerically: INTEGER 1 ('i1') and REAL 2.0 ('r') qualify;
    // INTEGER 3 excluded; the NULL-keyed row ('nul') excluded.
    same_unordered(
        MIX_SCHEMA,
        MIX_INDEX,
        MIX_NULL_DATA,
        "SELECT n FROM t WHERE v <= 2",
        &[vec![text("i1")], vec![text("r")]],
    );
}

// ===========================================================================
// 10. Index maintenance across UPDATE (lang_update; lang_createindex).
//     Moving a key must update the index: the old key stops matching and the new
//     key starts matching, with the row's other columns intact. Updating a
//     NON-indexed column must not disturb key lookups.
// ===========================================================================

/// Build the distinct-keyed table (1,'x'),(2,'y'),(3,'z') with an index on `a`
/// created before the rows (the maintained path).
fn dml_fixture() -> Connection {
    let mut db = mem();
    exec(&mut db, INT_SCHEMA);
    exec(&mut db, INT_INDEX);
    exec(&mut db, DML_DATA);
    db
}

#[test]
fn update_moves_key_old_key_gone() {
    // After moving key 2 -> 99, a lookup on the OLD key returns nothing.
    let mut db = dml_fixture();
    exec(&mut db, "UPDATE t SET a = 99 WHERE a = 2");
    assert_rows_unordered(&mut db, "SELECT b FROM t WHERE a = 2", &[]);
}

#[test]
fn update_moves_key_new_key_present() {
    // The moved row is found under the NEW key with its other column intact.
    let mut db = dml_fixture();
    exec(&mut db, "UPDATE t SET a = 99 WHERE a = 2");
    assert_rows_unordered(&mut db, "SELECT b FROM t WHERE a = 99", &[vec![text("y")]]);
}

#[test]
fn update_non_indexed_column_leaves_lookup_unaffected() {
    // Updating b (not in the index) must not change which key finds the row: a=2
    // still finds it, now with the new b value.
    let mut db = dml_fixture();
    exec(&mut db, "UPDATE t SET b = 'yy' WHERE a = 2");
    assert_rows_unordered(&mut db, "SELECT b FROM t WHERE a = 2", &[vec![text("yy")]]);
}

#[test]
fn update_order_by_reflects_new_key() {
    // A full ascending scan reflects the moved key: 1, 3, 99.
    let mut db = dml_fixture();
    exec(&mut db, "UPDATE t SET a = 99 WHERE a = 2");
    assert_rows(
        &mut db,
        "SELECT a FROM t ORDER BY a",
        &[vec![int(1)], vec![int(3)], vec![int(99)]],
    );
}

// ===========================================================================
// 11. Index maintenance across DELETE (lang_delete; lang_createindex).
//     A deleted key stops matching; other keys are unaffected; a full scan omits
//     the deleted row.
// ===========================================================================

#[test]
fn delete_removes_key_from_lookup() {
    let mut db = dml_fixture();
    exec(&mut db, "DELETE FROM t WHERE a = 2");
    assert_rows_unordered(&mut db, "SELECT b FROM t WHERE a = 2", &[]);
}

#[test]
fn delete_leaves_other_keys() {
    let mut db = dml_fixture();
    exec(&mut db, "DELETE FROM t WHERE a = 2");
    assert_rows_unordered(&mut db, "SELECT b FROM t WHERE a = 1", &[vec![text("x")]]);
    assert_rows_unordered(&mut db, "SELECT b FROM t WHERE a = 3", &[vec![text("z")]]);
}

#[test]
fn delete_order_by_reflects_deletion() {
    let mut db = dml_fixture();
    exec(&mut db, "DELETE FROM t WHERE a = 2");
    assert_rows(
        &mut db,
        "SELECT a FROM t ORDER BY a",
        &[vec![int(1)], vec![int(3)]],
    );
}

// ===========================================================================
// 12. Incremental INSERT after the index exists (the normal maintained path).
//     Each new row becomes visible to a lookup immediately, and earlier rows
//     stay visible.
// ===========================================================================

#[test]
fn incremental_insert_each_lookup_correct() {
    let mut db = mem();
    exec(&mut db, INT_SCHEMA);
    exec(&mut db, INT_INDEX); // index exists before any rows
    exec(&mut db, "INSERT INTO t VALUES (1,'x')");
    assert_rows_unordered(&mut db, "SELECT b FROM t WHERE a = 1", &[vec![text("x")]]);
    exec(&mut db, "INSERT INTO t VALUES (2,'y')");
    assert_rows_unordered(&mut db, "SELECT b FROM t WHERE a = 2", &[vec![text("y")]]);
    // The earlier key is still found after a later insert.
    assert_rows_unordered(&mut db, "SELECT b FROM t WHERE a = 1", &[vec![text("x")]]);
    exec(&mut db, "INSERT INTO t VALUES (3,'z')");
    assert_rows_unordered(&mut db, "SELECT b FROM t WHERE a = 3", &[vec![text("z")]]);
}

// ===========================================================================
// 13. DROP INDEX then query (lang_dropindex.html). Dropping the index must not
//     change results — the query falls back to a full scan.
// ===========================================================================

#[test]
fn drop_index_equality_unchanged() {
    let mut db = dml_fixture();
    exec(&mut db, "DROP INDEX i");
    assert_rows_unordered(&mut db, "SELECT b FROM t WHERE a = 2", &[vec![text("y")]]);
}

#[test]
fn drop_index_order_by_unchanged() {
    let mut db = dml_fixture();
    exec(&mut db, "DROP INDEX i");
    assert_rows(
        &mut db,
        "SELECT a FROM t ORDER BY a",
        &[vec![int(1)], vec![int(2)], vec![int(3)]],
    );
}

// ===========================================================================
// 14. COVERING query (index (a, b) contains everything a `SELECT b WHERE a`
//     needs). Whether or not a covering optimization applies, the values must be
//     correct. Data: (1,10),(2,20),(3,30),(2,25).
// ===========================================================================

#[test]
fn covering_projection_where_equality() {
    // a = 2 -> b values 20 and 25.
    same_unordered(
        COV_SCHEMA,
        COV_INDEX,
        COV_DATA,
        "SELECT b FROM t WHERE a = 2",
        &[vec![int(20)], vec![int(25)]],
    );
}

#[test]
fn covering_projection_range_ordered_by_b() {
    // a >= 2 -> rows (2,20),(2,25),(3,30); ORDER BY b -> 20, 25, 30.
    same_ordered(
        COV_SCHEMA,
        COV_INDEX,
        COV_DATA,
        "SELECT b FROM t WHERE a >= 2 ORDER BY b",
        &[vec![int(20)], vec![int(25)], vec![int(30)]],
    );
}

// ===========================================================================
// 15. ADVANCED index features (lang_createindex §1.1 partial / §1.2 expression /
//     §1.5 collation). These MAY be unimplemented: each `_can_be_created` test
//     asserts the CREATE is accepted (a loud failure if not), and each query
//     test asserts the spec-correct RESULT, which must hold whether or not the
//     advanced index is used.
// ===========================================================================

// Partial index fixture: a > 0 for rows 5,2; a <= 0 for rows -1,0.
const PARTIAL_DATA: &str = "INSERT INTO t VALUES (5,'pos'),(2,'small'),(-1,'neg'),(0,'zero')";

#[test]
fn partial_index_can_be_created() {
    // Intro: "If the optional WHERE clause is included, then the index is a
    // partial index." The CREATE should be accepted.
    let mut db = mem();
    exec(&mut db, INT_SCHEMA);
    let r = try_exec(&mut db, "CREATE INDEX i ON t(a) WHERE a > 0");
    assert!(
        r.is_ok(),
        "partial index CREATE (WHERE a > 0) should be accepted (lang_createindex.html); got {r:?}"
    );
}

#[test]
fn partial_index_query_implying_predicate_correct() {
    // A query whose predicate implies a > 0 (a = 5) may use the partial index;
    // the result is the matching row regardless.
    advanced_index_query_unordered(
        INT_SCHEMA,
        "CREATE INDEX i ON t(a) WHERE a > 0",
        PARTIAL_DATA,
        "SELECT b FROM t WHERE a = 5",
        &[vec![text("pos")]],
    );
}

#[test]
fn partial_index_query_not_implying_predicate_correct() {
    // A query whose predicate does NOT imply a > 0 (a = -1 references a row the
    // partial index deliberately omits) must still return that row via a full
    // scan — using the partial index here would be a correctness bug.
    advanced_index_query_unordered(
        INT_SCHEMA,
        "CREATE INDEX i ON t(a) WHERE a > 0",
        PARTIAL_DATA,
        "SELECT b FROM t WHERE a = -1",
        &[vec![text("neg")]],
    );
}

#[test]
fn expression_index_can_be_created() {
    // §1.2: an index key may be an expression over the table's own columns.
    let mut db = mem();
    exec(&mut db, MC_SCHEMA);
    let r = try_exec(&mut db, "CREATE INDEX i ON t(a + b)");
    assert!(
        r.is_ok(),
        "expression index CREATE ON t(a + b) should be accepted (lang_createindex.html §1.2); got {r:?}"
    );
}

#[test]
fn expression_index_query_correct() {
    // Rows (a,b,c): (1,2,'x') a+b=3, (3,4,'y') a+b=7, (2,2,'z') a+b=4. The query
    // WHERE a + b = 3 selects 'x' whether or not the expression index is used.
    advanced_index_query_unordered(
        MC_SCHEMA,
        "CREATE INDEX i ON t(a + b)",
        "INSERT INTO t VALUES (1,2,'x'),(3,4,'y'),(2,2,'z')",
        "SELECT c FROM t WHERE a + b = 3",
        &[vec![text("x")]],
    );
}

#[test]
fn nocase_index_can_be_created() {
    // §1.5: a COLLATE clause may follow the indexed column.
    let mut db = mem();
    exec(&mut db, TXT_SCHEMA);
    let r = try_exec(&mut db, "CREATE INDEX i ON t(s COLLATE NOCASE)");
    assert!(
        r.is_ok(),
        "NOCASE index CREATE ON t(s COLLATE NOCASE) should be accepted (lang_createindex.html §1.5); got {r:?}"
    );
}

#[test]
fn nocase_index_case_insensitive_lookup_correct() {
    // datatype3 §7: under NOCASE the 26 ASCII upper-case letters fold, so
    // 'Apple' matches 'apple'. The comparison requests NOCASE explicitly (the
    // column's default collation is BINARY); the NOCASE index may back it.
    advanced_index_query_unordered(
        TXT_SCHEMA,
        "CREATE INDEX i ON t(s COLLATE NOCASE)",
        "INSERT INTO t VALUES ('Apple',1),('banana',2),('CHERRY',3)",
        "SELECT n FROM t WHERE s = 'apple' COLLATE NOCASE",
        &[vec![int(1)]],
    );
}
