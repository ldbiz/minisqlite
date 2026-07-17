//! Conformance battery: the COVERING-INDEX (index-only scan) optimization is a pure
//! PERFORMANCE change and MUST be byte-for-byte result-identical to the table-fetch path
//! (`spec/sqlite-doc/optoverview.html` §"Covering Indexes").
//!
//! The central check here is RESULT-EQUIVALENCE. For each query shape, the SAME query runs
//! against two databases built from identical data:
//!   * one WITHOUT the relevant index — the guaranteed-correct full-scan / table-fetch
//!     ORACLE (covering can never engage), and
//!   * one WITH the index — where the planner may mark the index scan `covering` and the
//!     executor reads values straight from the index entry, SKIPPING the by-rowid fetch.
//! The two result sets must be identical (same values; and, for a fully-ordered query, same
//! order). Any wrong covering determination — a mis-mapped column, a wrongly-covered NULL, a
//! decline that should have engaged returning different bytes — makes the WITH-index run
//! diverge from the oracle and fails loudly. The index is created BOTH before the rows (the
//! maintained path) and after them (the `CREATE INDEX` backfill path), since a covering read
//! must be identical over either.
//!
//! Plan-shape assertions (that `covering` flips true ONLY when provable) live in
//! `minisqlite-plan`'s `compile::covering` unit tests; this file pins the observable RESULT
//! through the real facade, which is what ultimately matters.
//!
//! `minisqlite::Value` has no `PartialEq`/`Ord`, so every comparison goes through the shared
//! harness assertions — never `==` on a `Value`.

mod conformance;
use conformance::*;

use minisqlite::{Connection, Value};

// ===========================================================================
// Fixtures — tiny (this is a correctness suite, not a benchmark). Each carries
// duplicates, a spread for ranges, and (where noted) NULLs, so the covering read
// is exercised on repeated keys and the NULL storage class.
// ===========================================================================

// Single-column index on `a`; `b` is NOT covered. Duplicate key (2 twice), a NULL key.
const A_SCHEMA: &str = "CREATE TABLE t(a INTEGER, b TEXT)";
const A_INDEX: &str = "CREATE INDEX i ON t(a)";
const A_DATA: &str =
    "INSERT INTO t VALUES (2,'two'),(5,'five'),(2,'twob'),(8,'eight'),(NULL,'nil')";

// Composite index on (a, b); `c` is NOT covered. Duplicate (a,b)=(1,10) pair.
const AB_SCHEMA: &str = "CREATE TABLE t(a INTEGER, b INTEGER, c TEXT)";
const AB_INDEX: &str = "CREATE INDEX i ON t(a,b)";
const AB_DATA: &str =
    "INSERT INTO t VALUES (1,10,'p'),(1,20,'q'),(2,10,'r'),(2,20,'s'),(1,10,'t')";
// Composite fixture with a NULL in the trailing index column.
const AB_NULL_DATA: &str =
    "INSERT INTO t VALUES (1,10,'p'),(1,20,'q'),(1,NULL,'n'),(2,10,'r')";

// A DESCENDING index — the covering read walks the entry the same way regardless of the
// index's b-tree direction, so the stored values are identical.
const DESC_INDEX: &str = "CREATE INDEX i ON t(a DESC)";

// TEXT key with a NULL, to cover the TEXT storage class and NULL through a text index.
const TXT_SCHEMA: &str = "CREATE TABLE t(s TEXT, n INTEGER)";
const TXT_INDEX: &str = "CREATE INDEX i ON t(s)";
const TXT_DATA: &str =
    "INSERT INTO t VALUES ('apple',1),('mango',2),('pear',3),('banana',4),(NULL,5)";

// An EXPRESSION index. It keys a computed value, not a named column, so it must NOT be
// treated as covering column `a` by name — but the result must still be correct.
const EXPR_SCHEMA: &str = "CREATE TABLE t(a INTEGER, b TEXT)";
const EXPR_INDEX: &str = "CREATE INDEX i ON t(abs(a))";
const EXPR_DATA: &str = "INSERT INTO t VALUES (2,'two'),(-2,'negtwo'),(5,'five'),(-5,'negfive')";

// A table whose INTEGER PRIMARY KEY column aliases the rowid, with a secondary index on a
// non-key column — the covering read must surface the rowid alias correctly.
const PK_SCHEMA: &str = "CREATE TABLE t(id INTEGER PRIMARY KEY, v INTEGER)";
const PK_INDEX: &str = "CREATE INDEX i ON t(v)";
const PK_DATA: &str = "INSERT INTO t VALUES (1,100),(2,200),(3,100),(4,300)";

// REAL storage class in the covered column (with a duplicate and a NULL) — the covering read
// must reproduce a REAL value bit-identically, distinct from an INTEGER of the same magnitude.
const REAL_SCHEMA: &str = "CREATE TABLE t(a REAL, b TEXT)";
const REAL_INDEX: &str = "CREATE INDEX i ON t(a)";
const REAL_DATA: &str =
    "INSERT INTO t VALUES (1.5,'a'),(2.5,'b'),(2.5,'c'),(3.5,'d'),(NULL,'n')";

// BLOB storage class in the covered column (with a NULL). BLOB sorts above every other class,
// and the covering read must return the exact bytes.
const BLOB_SCHEMA: &str = "CREATE TABLE t(a BLOB, b TEXT)";
const BLOB_INDEX: &str = "CREATE INDEX i ON t(a)";
const BLOB_DATA: &str =
    "INSERT INTO t VALUES (x'01','one'),(x'02','two'),(x'0203','twoish'),(NULL,'nil')";

// ===========================================================================
// Equivalence harness. The no-index database is the ORACLE; the with-index
// database (built two ways: index-before-rows and index-after-rows backfill)
// must return the identical result.
// ===========================================================================

fn oracle_rows(schema: &str, data: &str, q: &str) -> Vec<Vec<Value>> {
    let mut db = mem();
    exec(&mut db, schema);
    exec(&mut db, data);
    query(&mut db, q).rows
}

fn db_index_before(schema: &str, index: &str, data: &str) -> Connection {
    let mut db = mem();
    exec(&mut db, schema);
    exec(&mut db, index);
    exec(&mut db, data);
    db
}

fn db_index_after(schema: &str, index: &str, data: &str) -> Connection {
    let mut db = mem();
    exec(&mut db, schema);
    exec(&mut db, data);
    exec(&mut db, index); // backfill the pre-existing rows into the new index
    db
}

/// UNORDERED (multiset) equivalence: the with-index result equals the no-index oracle,
/// order-insensitive — for queries with no fully-deterministic ORDER BY (the index path and
/// the table-scan path may legitimately emit rows in different orders).
fn same_unordered(schema: &str, index: &str, data: &str, q: &str) {
    let expected = oracle_rows(schema, data, q);
    let mut before = db_index_before(schema, index, data);
    assert_rows_unordered(&mut before, q, &expected);
    let mut after = db_index_after(schema, index, data);
    assert_rows_unordered(&mut after, q, &expected);
}

/// ORDERED equivalence: the with-index result equals the oracle row-for-row — for a query
/// whose order is deterministic (a single-row aggregate, or a full ORDER BY).
fn same_ordered(schema: &str, index: &str, data: &str, q: &str) {
    let expected = oracle_rows(schema, data, q);
    let mut before = db_index_before(schema, index, data);
    assert_rows(&mut before, q, &expected);
    let mut after = db_index_after(schema, index, data);
    assert_rows(&mut after, q, &expected);
}

// ===========================================================================
// count(*) over an index range — the headline covering case: zero columns read
// above the scan, whole predicate served by the seek, so the per-row table fetch
// is skipped and only entries are counted.
// ===========================================================================

#[test]
fn count_over_index_range_matches_oracle() {
    same_ordered(A_SCHEMA, A_INDEX, A_DATA, "SELECT count(*) FROM t WHERE a BETWEEN 2 AND 8");
}

#[test]
fn count_over_index_equality_matches_oracle() {
    same_ordered(A_SCHEMA, A_INDEX, A_DATA, "SELECT count(*) FROM t WHERE a = 2");
}

#[test]
fn count_over_composite_leading_equality_matches_oracle() {
    same_ordered(AB_SCHEMA, AB_INDEX, AB_DATA, "SELECT count(*) FROM t WHERE a = 1");
}

// ===========================================================================
// Projecting only indexed columns — the covering read supplies them from the
// index entry (equality, range, and the trailing rowid, which every entry carries).
// ===========================================================================

#[test]
fn project_indexed_column_range_matches_oracle() {
    same_unordered(A_SCHEMA, A_INDEX, A_DATA, "SELECT a FROM t WHERE a >= 2");
}

#[test]
fn project_indexed_column_equality_matches_oracle() {
    same_unordered(A_SCHEMA, A_INDEX, A_DATA, "SELECT a FROM t WHERE a = 2");
}

#[test]
fn project_indexed_column_and_rowid_matches_oracle() {
    same_unordered(A_SCHEMA, A_INDEX, A_DATA, "SELECT a, rowid FROM t WHERE a >= 2");
}

#[test]
fn project_indexed_column_ordered_matches_oracle() {
    // ORDER BY on the indexed column — the index walk serves the order AND covers `a`.
    same_ordered(A_SCHEMA, A_INDEX, A_DATA, "SELECT a FROM t WHERE a >= 0 ORDER BY a");
}

#[test]
fn composite_covered_subset_matches_oracle() {
    // Index (a,b) covers a query needing only b under an a-equality (with a duplicate b).
    same_unordered(AB_SCHEMA, AB_INDEX, AB_DATA, "SELECT b FROM t WHERE a = 1");
}

#[test]
fn composite_both_covered_columns_match_oracle() {
    same_unordered(AB_SCHEMA, AB_INDEX, AB_DATA, "SELECT a, b FROM t WHERE a = 1");
}

#[test]
fn composite_covered_ordered_matches_oracle() {
    same_ordered(AB_SCHEMA, AB_INDEX, AB_DATA, "SELECT a, b FROM t WHERE a >= 1 ORDER BY a, b");
}

// ===========================================================================
// Covering must NOT engage — a needed column is absent from the index, or a
// residual predicate reads one. The result must still equal the oracle.
// ===========================================================================

#[test]
fn project_non_indexed_column_matches_oracle() {
    // `b` is not in the a-index: covering declines, the table fetch supplies b. Still equal.
    same_unordered(A_SCHEMA, A_INDEX, A_DATA, "SELECT a, b FROM t WHERE a = 2");
}

#[test]
fn composite_needing_uncovered_column_matches_oracle() {
    // Index (a,b) does not carry `c`.
    same_unordered(AB_SCHEMA, AB_INDEX, AB_DATA, "SELECT c FROM t WHERE a = 1");
}

#[test]
fn residual_where_on_uncovered_column_matches_oracle() {
    // `c = 'p'` cannot be served by the (a,b)-index; it stays a residual Filter reading the
    // uncovered column `c`, so covering must decline — result still matches the oracle.
    same_unordered(AB_SCHEMA, AB_INDEX, AB_DATA, "SELECT a FROM t WHERE a >= 1 AND c = 'p'");
}

#[test]
fn select_star_matches_oracle() {
    // `*` needs every column; the index covers only a subset — declines but stays correct.
    same_unordered(A_SCHEMA, A_INDEX, A_DATA, "SELECT * FROM t WHERE a = 2");
    same_unordered(AB_SCHEMA, AB_INDEX, AB_DATA, "SELECT * FROM t WHERE a = 1");
}

// ===========================================================================
// GROUP BY / ORDER BY on the indexed column — the grouping reads only covered
// columns above the (covering) scan.
// ===========================================================================

#[test]
fn group_by_indexed_column_matches_oracle() {
    same_ordered(
        A_SCHEMA,
        A_INDEX,
        A_DATA,
        "SELECT a, count(*) FROM t WHERE a >= 0 GROUP BY a ORDER BY a",
    );
}

#[test]
fn group_by_composite_covered_matches_oracle() {
    same_ordered(
        AB_SCHEMA,
        AB_INDEX,
        AB_DATA,
        "SELECT a, count(*) FROM t GROUP BY a ORDER BY a",
    );
}

// ===========================================================================
// NULLs in the indexed column — the covering read must reproduce a NULL key
// value byte-identically (NULL is a real storage class in the index entry).
// ===========================================================================

#[test]
fn null_keys_project_indexed_column_matches_oracle() {
    // `a >= 2` excludes the NULL row in both paths; `a IS NULL` selects it.
    same_unordered(A_SCHEMA, A_INDEX, A_DATA, "SELECT a FROM t WHERE a >= 2");
    same_unordered(A_SCHEMA, A_INDEX, A_DATA, "SELECT a FROM t WHERE a IS NULL");
    same_ordered(A_SCHEMA, A_INDEX, A_DATA, "SELECT count(*) FROM t WHERE a IS NULL");
}

#[test]
fn composite_trailing_null_matches_oracle() {
    same_unordered(AB_SCHEMA, AB_INDEX, AB_NULL_DATA, "SELECT a, b FROM t WHERE a = 1");
    same_unordered(AB_SCHEMA, AB_INDEX, AB_NULL_DATA, "SELECT b FROM t WHERE a = 1 ORDER BY b");
}

#[test]
fn text_index_project_and_range_matches_oracle() {
    same_unordered(TXT_SCHEMA, TXT_INDEX, TXT_DATA, "SELECT s FROM t WHERE s >= 'banana'");
    same_ordered(TXT_SCHEMA, TXT_INDEX, TXT_DATA, "SELECT s FROM t WHERE s >= 'a' ORDER BY s");
    same_ordered(TXT_SCHEMA, TXT_INDEX, TXT_DATA, "SELECT count(*) FROM t WHERE s >= 'banana'");
}

// ===========================================================================
// DESC index — the entry stores the same values; only the walk direction differs.
// ===========================================================================

#[test]
fn desc_index_matches_oracle() {
    same_ordered(A_SCHEMA, DESC_INDEX, A_DATA, "SELECT count(*) FROM t WHERE a BETWEEN 2 AND 8");
    same_unordered(A_SCHEMA, DESC_INDEX, A_DATA, "SELECT a FROM t WHERE a >= 2");
    same_ordered(A_SCHEMA, DESC_INDEX, A_DATA, "SELECT a FROM t WHERE a >= 0 ORDER BY a DESC");
}

// ===========================================================================
// Expression index — keys a computed value, so it covers no named column. The
// result must match the oracle whether or not the planner uses the index.
// ===========================================================================

#[test]
fn expression_index_named_column_matches_oracle() {
    same_unordered(EXPR_SCHEMA, EXPR_INDEX, EXPR_DATA, "SELECT a FROM t WHERE abs(a) = 2");
    same_unordered(EXPR_SCHEMA, EXPR_INDEX, EXPR_DATA, "SELECT a, b FROM t WHERE abs(a) = 5");
}

#[test]
fn expression_index_count_matches_oracle() {
    same_ordered(EXPR_SCHEMA, EXPR_INDEX, EXPR_DATA, "SELECT count(*) FROM t WHERE abs(a) = 2");
}

// ===========================================================================
// INTEGER PRIMARY KEY alias — a secondary index on `v`; the covering read must
// surface the rowid/alias correctly (the alias maps to the rowid register).
// ===========================================================================

#[test]
fn intpk_secondary_index_matches_oracle() {
    same_unordered(PK_SCHEMA, PK_INDEX, PK_DATA, "SELECT v FROM t WHERE v = 100");
    same_unordered(PK_SCHEMA, PK_INDEX, PK_DATA, "SELECT id, v FROM t WHERE v = 100");
    same_ordered(PK_SCHEMA, PK_INDEX, PK_DATA, "SELECT count(*) FROM t WHERE v >= 100");
    same_unordered(PK_SCHEMA, PK_INDEX, PK_DATA, "SELECT id FROM t WHERE v = 100 ORDER BY id");
}

// ===========================================================================
// DML consistency — after UPDATE/DELETE the covering read reflects the change
// (the index is maintained, so index-only reads see the new values), identical
// to the oracle over the same mutations.
// ===========================================================================

#[test]
fn covering_reflects_updates_and_deletes() {
    let q = "SELECT a FROM t WHERE a >= 0 ORDER BY a";
    let mutations = "UPDATE t SET a = 9 WHERE a = 2; DELETE FROM t WHERE a = 8;";

    // Oracle: no index, apply mutations, read.
    let mut oracle = mem();
    exec(&mut oracle, A_SCHEMA);
    exec(&mut oracle, A_DATA);
    exec(&mut oracle, mutations);
    let expected = query(&mut oracle, q).rows;

    // With index: same mutations maintain the index; the covering read must agree.
    let mut db = db_index_before(A_SCHEMA, A_INDEX, A_DATA);
    exec(&mut db, mutations);
    assert_rows(&mut db, q, &expected);
}

// ===========================================================================
// REAL and BLOB storage classes in the covered column — the covering read must
// reproduce the exact value/class the table row decodes to (INTEGER/TEXT/NULL are
// exercised above; these add the remaining two classes).
// ===========================================================================

#[test]
fn real_covered_column_matches_oracle() {
    same_unordered(REAL_SCHEMA, REAL_INDEX, REAL_DATA, "SELECT a FROM t WHERE a >= 2.0");
    same_ordered(REAL_SCHEMA, REAL_INDEX, REAL_DATA, "SELECT a FROM t WHERE a >= 0 ORDER BY a");
    same_ordered(REAL_SCHEMA, REAL_INDEX, REAL_DATA, "SELECT count(*) FROM t WHERE a BETWEEN 2.0 AND 3.0");
}

#[test]
fn blob_covered_column_matches_oracle() {
    same_unordered(BLOB_SCHEMA, BLOB_INDEX, BLOB_DATA, "SELECT a FROM t WHERE a >= x'02'");
    same_ordered(BLOB_SCHEMA, BLOB_INDEX, BLOB_DATA, "SELECT a FROM t WHERE a >= x'00' ORDER BY a");
    same_ordered(BLOB_SCHEMA, BLOB_INDEX, BLOB_DATA, "SELECT count(*) FROM t WHERE a >= x'02'");
}

// ===========================================================================
// ALTER TABLE ADD COLUMN with a DEFAULT: pre-existing rows have a SHORT record
// (missing the new column), and both the table decode and the index backfill must
// surface the DEFAULT. Indexing the defaulted column and covering-reading it must
// reproduce that default byte-identically to the table-fetch oracle.
// ===========================================================================

#[test]
fn add_column_default_short_rows_matches_oracle() {
    // Rows inserted here predate `d`, so their stored record is 2 wide; reading `d` yields the
    // declared DEFAULT 7 (lang_altertable.html §4: ADD COLUMN makes no change to table content).
    let setup = "CREATE TABLE t(a INTEGER, b TEXT); \
                 INSERT INTO t VALUES (1,'x'),(2,'y'),(3,'z'); \
                 ALTER TABLE t ADD COLUMN d INTEGER DEFAULT 7;";
    let q_d = "SELECT d FROM t WHERE d >= 7 ORDER BY d";
    let q_ad = "SELECT a, d FROM t WHERE d = 7 ORDER BY a";
    let q_count = "SELECT count(*) FROM t WHERE d >= 7";

    // Oracle: NO index on d — the by-rowid fetch decodes the short row and applies the default.
    let mut oracle = mem();
    exec(&mut oracle, setup);
    let e_d = query(&mut oracle, q_d).rows;
    let e_ad = query(&mut oracle, q_ad).rows;
    let e_count = query(&mut oracle, q_count).rows;

    // With an index on the defaulted column: CREATE INDEX backfills the short rows to the
    // default, and the covering read must reproduce it identically.
    let mut db = mem();
    exec(&mut db, setup);
    exec(&mut db, "CREATE INDEX id ON t(d)");
    assert_rows(&mut db, q_d, &e_d);
    assert_rows(&mut db, q_ad, &e_ad);
    assert_rows(&mut db, q_count, &e_count);
}

// ===========================================================================
// Covering inside a CTE body and a non-correlated subquery. `mark_covering`
// deliberately extends into Materialized/Recursive CTE bodies and NON-correlated
// subplans (where the leaf row sits at base 0), so the register mapping is exact
// there; a correlated subplan is left alone. These pin the result through those
// paths against the no-index oracle.
// ===========================================================================

#[test]
fn cte_body_covering_matches_oracle() {
    same_ordered(
        A_SCHEMA,
        A_INDEX,
        A_DATA,
        "WITH c AS (SELECT a FROM t WHERE a >= 2) SELECT a FROM c ORDER BY a",
    );
    same_ordered(
        A_SCHEMA,
        A_INDEX,
        A_DATA,
        "WITH c AS (SELECT a FROM t WHERE a >= 2) SELECT count(*) FROM c",
    );
}

#[test]
fn non_correlated_subquery_covering_matches_oracle() {
    // The inner `SELECT a FROM t WHERE a >= 5` is non-correlated, so `mark_covering` may cover
    // its leaf; the outer WHERE carries an IN-subquery, so the OUTER chain declines covering.
    // Either way the result must equal the no-index oracle.
    same_ordered(
        A_SCHEMA,
        A_INDEX,
        A_DATA,
        "SELECT a FROM t WHERE a IN (SELECT a FROM t WHERE a >= 5) ORDER BY a",
    );
    same_ordered(
        A_SCHEMA,
        A_INDEX,
        A_DATA,
        "SELECT count(*) FROM t WHERE a IN (SELECT a FROM t WHERE a >= 2)",
    );
}
