//! Conformance battery: the **UPSERT conflict target that names a PARTIAL unique index** —
//! `INSERT ... ON CONFLICT(<cols-or-exprs>) WHERE <pred> DO UPDATE / DO NOTHING`, end to end at
//! the pinned `minisqlite` facade.
//!
//! This is the intersection of two documented features exercised separately elsewhere: UPSERT
//! (`conformance_upsert.rs`) and PARTIAL indexes (`conformance_partial_index_*.rs`). The conflict
//! target of an UPSERT may carry its own `WHERE`, which names a PARTIAL unique index; the target's
//! columns/expressions AND its `WHERE` must BOTH match the index. Real `sqlite3` SUCCEEDS on these;
//! the engine previously rejected any `ON CONFLICT(...) WHERE ...` outright.
//!
//! Every expected value is TRANSCRIBED FROM THE SPEC, never from what the engine returns.
//! Spec sources (all under `spec/sqlite-doc/`):
//!
//!   * `lang_upsert.html` §2: "The UPSERT processing happens only for uniqueness constraints. A
//!     'uniqueness constraint' is an explicit UNIQUE or PRIMARY KEY constraint ..., or a unique
//!     index." A PARTIAL unique index IS a unique index, so `ON CONFLICT(cols) WHERE <pred>` names
//!     it. §2: the conflict target "must ... match a single uniqueness constraint"; a target
//!     matching NO constraint is an error at prepare time.
//!   * `lang_upsert.html` §2: a bare column in DO UPDATE is the ORIGINAL (pre-INSERT) row value;
//!     `excluded.col` is the value the omitted INSERT would have written.
//!   * `partialindex.html` §2 / §2.1: "Only rows of the table for which the WHERE clause evaluates
//!     to true are included in the index." "A partial index definition may include the UNIQUE
//!     keyword. If it does, then SQLite requires every entry *in the index* to be unique." So a
//!     partial UNIQUE index constrains ONLY the rows whose predicate is true — two rows that share
//!     the indexed value do NOT conflict unless BOTH are in the index. THIS is what makes an
//!     "outside the predicate" duplicate coexist through the UPSERT path.
//!   * `lang_createindex.html` §1.2 ("Indexes On Expressions"): an index key may be a general
//!     expression over the table's own columns, e.g. `lower(a)`; combined with a `WHERE`, that is
//!     an expression-based PARTIAL unique index.
//!
//! Each assertion is the spec-correct post-upsert STATE and is never weakened or `#[ignore]`d to
//! pass; each behavior is its own `#[test]` so one discrepancy fails exactly that case.

mod conformance;
use conformance::*;

use minisqlite::{Connection, Error};

/// A rowid table `t(a, b, note)` with a COLUMN-based PARTIAL unique index on `a` restricted to
/// rows where `b > 0`, pre-seeded with one IN-predicate row (`a = 1`, `b = 5`, in the index).
///
/// ```sql
/// CREATE TABLE t(a INTEGER, b INTEGER, note TEXT);
/// CREATE UNIQUE INDEX i ON t(a) WHERE b > 0;
/// INSERT INTO t VALUES (1, 5, 'orig');
/// ```
fn seeded_partial_col_db() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b INTEGER, note TEXT)");
    exec(&mut db, "CREATE UNIQUE INDEX i ON t(a) WHERE b > 0");
    exec(&mut db, "INSERT INTO t VALUES (1, 5, 'orig')"); // in the index (b > 0), key a = 1
    db
}

// ---------------------------------------------------------------------------
// Column-based partial index: the target fires (or not) exactly as membership dictates.
// ---------------------------------------------------------------------------

/// §2 + partialindex §2.1: an IN-predicate candidate (`b = 8 > 0`) whose `a` duplicates the seeded
/// in-index `a = 1` collides on the partial unique index, so `ON CONFLICT(a) WHERE b > 0` fires the
/// DO UPDATE on the EXISTING row. `excluded.note` is the would-be-inserted value; the bare `a`/`b`
/// stay the original (the update rewrites the pre-existing row, not the candidate).
#[test]
fn partial_col_inside_predicate_conflict_do_update_fires() {
    let mut db = seeded_partial_col_db();
    exec(
        &mut db,
        "INSERT INTO t VALUES (1, 8, 'new') ON CONFLICT(a) WHERE b > 0 DO UPDATE SET note = excluded.note",
    );
    // Existing (1, 5, 'orig') updated in place: note -> 'new'; a and b unchanged; still one row.
    assert_rows(&mut db, "SELECT a, b, note FROM t", &[vec![int(1), int(5), text("new")]]);
}

/// §2: an IN-predicate candidate whose `a` matches NO in-index entry does not conflict, so the DO
/// UPDATE does not fire and the row inserts normally. `(2, 5)` is in the index (distinct key), so
/// both rows survive.
#[test]
fn partial_col_inside_predicate_no_match_inserts() {
    let mut db = seeded_partial_col_db();
    exec(
        &mut db,
        "INSERT INTO t VALUES (2, 5, 'new') ON CONFLICT(a) WHERE b > 0 DO UPDATE SET note = excluded.note",
    );
    assert_rows_unordered(
        &mut db,
        "SELECT a, b, note FROM t",
        &[vec![int(1), int(5), text("orig")], vec![int(2), int(5), text("new")]],
    );
}

/// THE headline case (partialindex §2.1): an OUTSIDE-predicate candidate (`b = -1`, NOT in the
/// index) whose `a` duplicates an existing OUTSIDE-predicate row's `a` does NOT conflict — neither
/// row is in the partial index, so there is no in-index entry to collide with. The plain INSERT
/// proceeds and BOTH rows coexist. This is the exact behavior the partial-maintenance fix enables;
/// before it, index maintenance dropped the predicate and this over-enforced.
#[test]
fn partial_col_outside_predicate_duplicate_coexists() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b INTEGER, note TEXT)");
    exec(&mut db, "CREATE UNIQUE INDEX i ON t(a) WHERE b > 0");
    exec(&mut db, "INSERT INTO t VALUES (1, -1, 'first')"); // b <= 0 => NOT in the index
    exec(
        &mut db,
        "INSERT INTO t VALUES (1, -1, 'second') ON CONFLICT(a) WHERE b > 0 DO UPDATE SET note = excluded.note",
    );
    // No in-index collision => the candidate INSERTS; two rows share a = 1.
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2));
    assert_rows(
        &mut db,
        "SELECT note FROM t ORDER BY note",
        &[vec![text("first")], vec![text("second")]],
    );
}

/// §2: `DO NOTHING` on an in-predicate collision skips the offending row — no insert, no update, no
/// error. The seeded row is untouched.
#[test]
fn partial_col_inside_predicate_conflict_do_nothing_skips() {
    let mut db = seeded_partial_col_db();
    exec(&mut db, "INSERT INTO t VALUES (1, 9, 'new') ON CONFLICT(a) WHERE b > 0 DO NOTHING");
    assert_rows(&mut db, "SELECT a, b, note FROM t", &[vec![int(1), int(5), text("orig")]]);
}

/// §2 + partialindex §2.1: `DO NOTHING` does NOT over-apply — an outside-predicate candidate is not
/// in the index, so there is no conflict for DO NOTHING to swallow; the row inserts.
#[test]
fn partial_col_outside_predicate_do_nothing_still_inserts() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b INTEGER, note TEXT)");
    exec(&mut db, "CREATE UNIQUE INDEX i ON t(a) WHERE b > 0");
    exec(&mut db, "INSERT INTO t VALUES (1, -1, 'first')");
    exec(&mut db, "INSERT INTO t VALUES (1, -1, 'second') ON CONFLICT(a) WHERE b > 0 DO NOTHING");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2));
}

/// partialindex §2 ("Only rows ... for which the WHERE clause evaluates to true are included"): a
/// candidate whose PREDICATE COLUMN IS NULL is EXCLUDED from the index — `NULL > 0` is NULL, not
/// true, exactly like a false predicate. `(1, NULL)` shares the seeded in-index row's `a = 1`, but
/// the candidate is never admitted to the `WHERE b > 0` index, so there is no in-index entry to
/// collide with: the DO UPDATE does NOT fire, the row INSERTs, the seeded row is untouched, and the
/// two `a = 1` rows coexist. (The `b <= 0` case is covered above; this pins the distinct NULL arm.)
#[test]
fn partial_col_null_predicate_candidate_inserts() {
    let mut db = seeded_partial_col_db();
    exec(
        &mut db,
        "INSERT INTO t VALUES (1, NULL, 'new') ON CONFLICT(a) WHERE b > 0 DO UPDATE SET note = excluded.note",
    );
    // Candidate excluded from the index (b IS NULL) => no collision => plain insert; two a = 1 rows.
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2));
    assert_rows_unordered(
        &mut db,
        "SELECT a, b, note FROM t",
        &[vec![int(1), int(5), text("orig")], vec![int(1), null(), text("new")]],
    );
}

// ---------------------------------------------------------------------------
// Expression-based partial index: `lower(a)` restricted to `b > 0`.
// ---------------------------------------------------------------------------

/// §1.2 + partialindex §2.1: an EXPRESSION-based partial unique index `lower(a) WHERE b > 0` is a
/// valid conflict target. An in-predicate candidate `('FOO', 5)` collides with the seeded
/// `('Foo', 5)` on key `lower = 'foo'`, so the DO UPDATE fires on the existing row.
#[test]
fn partial_expr_inside_predicate_conflict_do_update_fires() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a TEXT, b INTEGER, n INTEGER)");
    exec(&mut db, "CREATE UNIQUE INDEX i ON t(lower(a)) WHERE b > 0");
    exec(&mut db, "INSERT INTO t VALUES ('Foo', 5, 1)"); // in index, key lower('Foo') = 'foo'
    exec(
        &mut db,
        "INSERT INTO t VALUES ('FOO', 5, 9) ON CONFLICT(lower(a)) WHERE b > 0 DO UPDATE SET n = excluded.n",
    );
    // Existing row's a ('Foo') unchanged; n set to the candidate's 9; still one row.
    assert_rows(&mut db, "SELECT a, b, n FROM t", &[vec![text("Foo"), int(5), int(9)]]);
}

/// partialindex §2.1: an OUTSIDE-predicate expression-key duplicate coexists. Both `('Foo', -1)`
/// and `('FOO', -1)` have `b <= 0`, so neither is in the `lower(a) WHERE b > 0` index; the second
/// inserts rather than colliding.
#[test]
fn partial_expr_outside_predicate_duplicate_coexists() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a TEXT, b INTEGER, n INTEGER)");
    exec(&mut db, "CREATE UNIQUE INDEX i ON t(lower(a)) WHERE b > 0");
    exec(&mut db, "INSERT INTO t VALUES ('Foo', -1, 1)");
    exec(
        &mut db,
        "INSERT INTO t VALUES ('FOO', -1, 2) ON CONFLICT(lower(a)) WHERE b > 0 DO UPDATE SET n = excluded.n",
    );
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2));
    assert_rows(&mut db, "SELECT n FROM t ORDER BY n", &[vec![int(1)], vec![int(2)]]);
}

// ---------------------------------------------------------------------------
// A target whose columns match but whose WHERE does NOT is an error (§2).
// ---------------------------------------------------------------------------

/// §2: the conflict target must MATCH the partial index. The match on the WHERE clause is
/// STRUCTURAL, not semantic implication: the index is `WHERE b > 0` but the target is `WHERE
/// b >= 0`, a DIFFERENT predicate, so it names no uniqueness constraint and — exactly like real
/// sqlite at prepare — the statement is rejected.
#[test]
fn partial_target_mismatched_where_is_rejected() {
    let mut db = seeded_partial_col_db();
    let e = assert_exec_error(
        &mut db,
        "INSERT INTO t VALUES (1, 5, 'x') ON CONFLICT(a) WHERE b >= 0 DO NOTHING",
    );
    match e {
        Error::Sql(m) => assert!(
            m.contains("does not match any PRIMARY KEY or UNIQUE constraint"),
            "expected the no-matching-constraint error; got {m:?}"
        ),
        other => panic!("expected a Sql error, got {other:?}"),
    }
}

/// §2 + partialindex: a PLAIN `ON CONFLICT(a)` (NO WHERE) does NOT name a partial index — real
/// sqlite requires the matching WHERE to select one. When the ONLY constraint on `a` is a partial
/// index, a bare column target matches nothing and is rejected (it must not silently bind to the
/// partial index and then over-enforce).
#[test]
fn plain_target_does_not_name_a_partial_index() {
    let mut db = seeded_partial_col_db();
    let e =
        assert_exec_error(&mut db, "INSERT INTO t VALUES (1, 9, 'x') ON CONFLICT(a) DO NOTHING");
    match e {
        Error::Sql(m) => assert!(
            m.contains("does not match any PRIMARY KEY or UNIQUE constraint"),
            "expected the no-matching-constraint error; got {m:?}"
        ),
        other => panic!("expected a Sql error, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Clause routing: a partial-index target and a PRIMARY KEY target in one chain.
// ---------------------------------------------------------------------------

/// §2: chained clauses each carry their own target, and the FIRST whose constraint the candidate
/// violates runs. The table has BOTH an INTEGER PRIMARY KEY (`id`) and a PARTIAL unique index on
/// `a WHERE b > 0`; the two clauses target each in turn. A candidate that collides only on the
/// rowid takes `ON CONFLICT(id)`; one that collides only on the partial index takes
/// `ON CONFLICT(a) WHERE b > 0` — proving the executor routes by the RESOLVED target (rowid vs the
/// partial index's root), not by position.
#[test]
fn chained_pk_and_partial_targets_route_correctly() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, a INTEGER, b INTEGER, tag TEXT)");
    exec(&mut db, "CREATE UNIQUE INDEX i ON t(a) WHERE b > 0");
    exec(&mut db, "INSERT INTO t VALUES (1, 10, 5, 'orig')"); // id=1; a=10,b=5>0 in the partial index

    let chain = "ON CONFLICT(id) DO UPDATE SET tag = 'by_id' \
                 ON CONFLICT(a) WHERE b > 0 DO UPDATE SET tag = 'by_partial'";

    // id=1 collides (rowid); a=20 has no in-index match => only the rowid conflict => first clause.
    exec(&mut db, &format!("INSERT INTO t VALUES (1, 20, 5, 'x') {chain}"));
    assert_rows(
        &mut db,
        "SELECT id, a, b, tag FROM t",
        &[vec![int(1), int(10), int(5), text("by_id")]],
    );

    // id=2 is free, but a=10,b=5>0 collides on the partial index with the (id=1) row => second
    // clause rewrites the EXISTING id=1 row; the id=2 candidate is not inserted.
    exec(&mut db, &format!("INSERT INTO t VALUES (2, 10, 5, 'y') {chain}"));
    assert_rows(
        &mut db,
        "SELECT id, a, b, tag FROM t",
        &[vec![int(1), int(10), int(5), text("by_partial")]],
    );
}

/// A per-row routing counterpart: with the same chain, an OUTSIDE-predicate candidate whose `a`
/// duplicates the seeded row but with `b <= 0` is NOT in the partial index and (given a free rowid)
/// hits NO constraint — so neither clause fires and it inserts as a new row. The bare column target
/// clause must not swallow a row the partial index does not actually contain.
#[test]
fn chained_partial_target_ignores_outside_predicate_row() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, a INTEGER, b INTEGER, tag TEXT)");
    exec(&mut db, "CREATE UNIQUE INDEX i ON t(a) WHERE b > 0");
    exec(&mut db, "INSERT INTO t VALUES (1, 10, 5, 'orig')");

    let chain = "ON CONFLICT(id) DO UPDATE SET tag = 'by_id' \
                 ON CONFLICT(a) WHERE b > 0 DO UPDATE SET tag = 'by_partial'";

    // id=2 free; a=10 but b=-1 => the candidate is NOT in the partial index => no conflict at all.
    exec(&mut db, &format!("INSERT INTO t VALUES (2, 10, -1, 'kept') {chain}"));
    assert_rows_unordered(
        &mut db,
        "SELECT id, a, b, tag FROM t",
        &[
            vec![int(1), int(10), int(5), text("orig")],
            vec![int(2), int(10), int(-1), text("kept")],
        ],
    );
}

// ---------------------------------------------------------------------------
// WITHOUT ROWID: a column-based partial secondary UNIQUE index is a valid conflict target
// (the executor's `decide_upsert_wr` path — a WR row is identified by its PRIMARY KEY, and a
// partial secondary index gates membership on the same predicate). Expression indexes on a WR
// table are unsupported (rejected at CREATE), so this covers the column-based WR case.
// ---------------------------------------------------------------------------

/// partialindex §2.1 on a WITHOUT ROWID table: an in-predicate candidate whose secondary-index
/// key `a = 1` collides with an existing in-index row fires the DO UPDATE on that existing row
/// (found by its PRIMARY KEY), leaving the candidate's PK uninserted.
#[test]
fn wr_partial_col_inside_predicate_conflict_do_update_fires() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k TEXT PRIMARY KEY, a INTEGER, b INTEGER, note TEXT) WITHOUT ROWID");
    exec(&mut db, "CREATE UNIQUE INDEX i ON t(a) WHERE b > 0");
    exec(&mut db, "INSERT INTO t VALUES ('k1', 1, 5, 'orig')"); // a=1,b=5>0 in the partial index
    exec(
        &mut db,
        "INSERT INTO t VALUES ('k2', 1, 8, 'new') ON CONFLICT(a) WHERE b > 0 DO UPDATE SET note = excluded.note",
    );
    // The existing 'k1' row (owner of the in-index a=1) is updated; 'k2' never lands.
    assert_rows(&mut db, "SELECT k, a, b, note FROM t", &[vec![text("k1"), int(1), int(5), text("new")]]);
}

/// partialindex §2.1 on a WITHOUT ROWID table: two OUTSIDE-predicate rows sharing `a` coexist —
/// neither is in the `WHERE b > 0` secondary index, so the second (a distinct PRIMARY KEY)
/// inserts rather than colliding.
#[test]
fn wr_partial_col_outside_predicate_duplicate_coexists() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k TEXT PRIMARY KEY, a INTEGER, b INTEGER, note TEXT) WITHOUT ROWID");
    exec(&mut db, "CREATE UNIQUE INDEX i ON t(a) WHERE b > 0");
    exec(&mut db, "INSERT INTO t VALUES ('k1', 1, -1, 'first')"); // b <= 0 => NOT in the index
    exec(
        &mut db,
        "INSERT INTO t VALUES ('k2', 1, -1, 'second') ON CONFLICT(a) WHERE b > 0 DO UPDATE SET note = excluded.note",
    );
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2));
    assert_rows(
        &mut db,
        "SELECT k, note FROM t ORDER BY k",
        &[vec![text("k1"), text("first")], vec![text("k2"), text("second")]],
    );
}
