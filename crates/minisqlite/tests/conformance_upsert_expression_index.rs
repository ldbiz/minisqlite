//! Conformance battery: the **UPSERT conflict target that names an indexed EXPRESSION** —
//! `INSERT ... ON CONFLICT(<expr>) DO UPDATE / DO NOTHING`, end to end at the pinned
//! `minisqlite` facade.
//!
//! This is the intersection of two documented features already exercised separately:
//! UPSERT (`conformance_upsert.rs`) and indexes on expressions
//! (`conformance_expression_index.rs`). Nothing here re-tests either in isolation; every
//! case pins that an `ON CONFLICT` target which names an INDEXED EXPRESSION resolves to that
//! unique index and drives the upsert decision.
//!
//! Every expected value is TRANSCRIBED FROM THE SPEC, never from what the engine returns.
//! Spec sources:
//!
//!   * `spec/sqlite-doc/lang_upsert.html` §2: "The UPSERT processing happens only for
//!     uniqueness constraints. A 'uniqueness constraint' is an explicit UNIQUE or PRIMARY KEY
//!     constraint ..., or a unique index." An index on an expression IS a unique index, so
//!     `ON CONFLICT(<expr>)` names it. §2: "the conflict target ... must ... match a single
//!     uniqueness constraint"; a target matching NO constraint is an error.
//!   * `spec/sqlite-doc/lang_upsert.html` §2: a bare column in DO UPDATE is the ORIGINAL
//!     (pre-INSERT) row value; `excluded.col` is the value the omitted INSERT would have
//!     written.
//!   * `spec/sqlite-doc/lang_createindex.html` §1.2 ("Indexes On Expressions"): an index key
//!     may be a general expression over the table's own columns; a UNIQUE such index rejects
//!     a duplicate KEY VALUE. `t(a+b)` keys the sum, so `(1,2)` and `(2,1)` share key `3`.
//!
//! Each assertion is the spec-correct post-upsert STATE and is never weakened to pass; each
//! behavior is its own `#[test]` so one discrepancy fails exactly that case.

mod conformance;
use conformance::*;

use minisqlite::{Connection, Error};

/// A rowid table `t(a, b, note)` with a UNIQUE index on the expression `a + b`, pre-seeded
/// with one row whose key is `3`. The lever every conflict test below pulls: a second row
/// whose `a + b` is also `3` (e.g. `(2, 1, ...)`) collides on that expression index.
fn seeded_expr_index_db() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b INTEGER, note TEXT)");
    exec(&mut db, "CREATE UNIQUE INDEX u ON t(a + b)");
    exec(&mut db, "INSERT INTO t VALUES (1, 2, 'orig')"); // key a+b = 3
    db
}

// ---------------------------------------------------------------------------
// The expression target fires the upsert.
// ---------------------------------------------------------------------------

/// §2 + §1.2: `ON CONFLICT(a + b)` names the unique expression index `u`. The incoming
/// `(2, 1, 'new')` has key `a+b = 3`, colliding with the seeded `(1, 2, 'orig')`, so the
/// INSERT behaves as a DO UPDATE of the EXISTING row: its `note` becomes `'touched'` while
/// its bare `a`/`b` stay the original `1`/`2` (the update rewrites the pre-existing row, not
/// the candidate).
#[test]
fn expr_index_target_do_update_rewrites_existing_row() {
    let mut db = seeded_expr_index_db();
    exec(
        &mut db,
        "INSERT INTO t VALUES (2, 1, 'new') ON CONFLICT(a + b) DO UPDATE SET note = 'touched'",
    );
    assert_rows(&mut db, "SELECT a, b, note FROM t", &[vec![int(1), int(2), text("touched")]]);
}

/// §2: `ON CONFLICT(a + b) DO NOTHING` on the same collision skips the offending row
/// entirely — no insert, no update, no error. The table is unchanged (still one row).
#[test]
fn expr_index_target_do_nothing_skips_the_row() {
    let mut db = seeded_expr_index_db();
    exec(&mut db, "INSERT INTO t VALUES (2, 1, 'new') ON CONFLICT(a + b) DO NOTHING");
    assert_rows(&mut db, "SELECT a, b, note FROM t", &[vec![int(1), int(2), text("orig")]]);
}

/// §2: when the candidate does NOT collide on the expression key, the DO UPDATE does not
/// fire and the row inserts normally. `(4, 5, 'new')` has key `9` (distinct from `3`), so
/// both rows survive.
#[test]
fn expr_index_target_no_conflict_inserts_normally() {
    let mut db = seeded_expr_index_db();
    exec(
        &mut db,
        "INSERT INTO t VALUES (4, 5, 'new') ON CONFLICT(a + b) DO UPDATE SET note = 'touched'",
    );
    assert_rows_unordered(
        &mut db,
        "SELECT a, b, note FROM t",
        &[vec![int(1), int(2), text("orig")], vec![int(4), int(5), text("new")]],
    );
}

// ---------------------------------------------------------------------------
// DO UPDATE value semantics through an expression target (bare vs excluded).
// ---------------------------------------------------------------------------

/// §2: `excluded.note` is the value the omitted INSERT carried. On the collision, the
/// existing row's `note` is set to the candidate's `'new'`.
#[test]
fn expr_index_target_do_update_reads_excluded() {
    let mut db = seeded_expr_index_db();
    exec(
        &mut db,
        "INSERT INTO t VALUES (2, 1, 'new') ON CONFLICT(a + b) DO UPDATE SET note = excluded.note",
    );
    assert_rows(&mut db, "SELECT a, b, note FROM t", &[vec![int(1), int(2), text("new")]]);
}

/// §2: a BARE column in DO UPDATE is the ORIGINAL row value. `SET a = a + 100` reads the
/// existing `a` (1) → 101; because the key `a + b` becomes `101 + 2 = 103`, distinct from
/// any other row, the in-place update is allowed. (This also confirms the DO UPDATE path
/// re-maintains the expression index for the rewritten row.)
#[test]
fn expr_index_target_do_update_bare_is_existing_value() {
    let mut db = seeded_expr_index_db();
    exec(
        &mut db,
        "INSERT INTO t VALUES (2, 1, 'new') ON CONFLICT(a + b) DO UPDATE SET a = a + 100",
    );
    assert_rows(&mut db, "SELECT a, b, note FROM t", &[vec![int(101), int(2), text("orig")]]);
}

// ---------------------------------------------------------------------------
// Other expression shapes: a function key, and a mixed name+expression key.
// ---------------------------------------------------------------------------

/// §1.2 + §2: a function-expression index `lower(s)` is a valid conflict target.
/// `('FOO', ...)` collides with the seeded `('Foo', ...)` on key `'foo'`, so the DO UPDATE
/// fires on the existing row.
#[test]
fn function_expression_index_target() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(s TEXT, n INTEGER)");
    exec(&mut db, "CREATE UNIQUE INDEX u ON t(lower(s))");
    exec(&mut db, "INSERT INTO t VALUES ('Foo', 1)"); // key 'foo'
    exec(
        &mut db,
        "INSERT INTO t VALUES ('FOO', 9) ON CONFLICT(lower(s)) DO UPDATE SET n = excluded.n",
    );
    // Existing row's s ('Foo') is unchanged; n is set to the candidate's 9.
    assert_rows(&mut db, "SELECT s, n FROM t", &[vec![text("Foo"), int(9)]]);
}

/// §2: a mixed target — one plain column and one expression — resolves to the matching
/// UNIQUE index `(a, b + c)`. The candidate collides on that composite key, firing the
/// DO UPDATE. (The `note` column is outside the key so it is free to change.)
#[test]
fn mixed_name_and_expression_index_target() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b INTEGER, c INTEGER, note TEXT)");
    exec(&mut db, "CREATE UNIQUE INDEX u ON t(a, b + c)");
    exec(&mut db, "INSERT INTO t VALUES (1, 2, 3, 'orig')"); // key (1, b+c=5)
    exec(
        &mut db,
        "INSERT INTO t VALUES (1, 4, 1, 'new') ON CONFLICT(a, b + c) DO UPDATE SET note = 'touched'",
    );
    // Candidate key (1, 4+1=5) == existing (1, 5) → collides; existing row updated in place.
    assert_rows(
        &mut db,
        "SELECT a, b, c, note FROM t",
        &[vec![int(1), int(2), int(3), text("touched")]],
    );
}

// ---------------------------------------------------------------------------
// Clause routing: an expression target and a column target in one chain.
// ---------------------------------------------------------------------------

/// §2 + §4: chained clauses each carry their own target, and the FIRST whose constraint the
/// candidate violates runs. Here the table has BOTH an INTEGER PRIMARY KEY (`id`) and a
/// unique expression index (`a + b`); the two clauses target each in turn. A candidate that
/// collides only on the rowid takes the `ON CONFLICT(id)` clause; one that collides only on
/// the expression key takes the `ON CONFLICT(a + b)` clause — proving the executor routes by
/// the RESOLVED target kind, not by position alone.
#[test]
fn chained_rowid_and_expression_targets_route_correctly() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, a INTEGER, b INTEGER, tag TEXT)");
    exec(&mut db, "CREATE UNIQUE INDEX u ON t(a + b)");
    exec(&mut db, "INSERT INTO t VALUES (1, 10, 20, 'orig')"); // id=1, key a+b=30

    let chain = "ON CONFLICT(id) DO UPDATE SET tag = 'by_id' \
                 ON CONFLICT(a + b) DO UPDATE SET tag = 'by_expr'";

    // id=1 collides (rowid), a+b=10 does NOT → first clause (by_id) fires.
    exec(&mut db, &format!("INSERT INTO t VALUES (1, 5, 5, 'x') {chain}"));
    assert_rows(&mut db, "SELECT id, a, b, tag FROM t", &[vec![int(1), int(10), int(20), text("by_id")]]);

    // id=2 is free, but a+b=30 collides with the (id=1) row → second clause (by_expr) fires,
    // rewriting the EXISTING id=1 row. The (id=2) candidate is not inserted.
    exec(&mut db, &format!("INSERT INTO t VALUES (2, 25, 5, 'y') {chain}"));
    assert_rows(
        &mut db,
        "SELECT id, a, b, tag FROM t",
        &[vec![int(1), int(10), int(20), text("by_expr")]],
    );
}

// ---------------------------------------------------------------------------
// A target that matches no uniqueness constraint is an error (§2).
// ---------------------------------------------------------------------------

/// §2: the conflict target must match a uniqueness constraint. `a + 1` is not the key of any
/// index (only `a + b` is), so — exactly like real sqlite at prepare — the statement is
/// rejected rather than silently accepted. The match is EXACT: a differently-written
/// expression fails closed.
#[test]
fn expression_target_matching_no_index_is_rejected() {
    let mut db = seeded_expr_index_db();
    let e = assert_exec_error(
        &mut db,
        "INSERT INTO t VALUES (2, 1, 'new') ON CONFLICT(a + 1) DO NOTHING",
    );
    match e {
        Error::Sql(m) => assert!(
            m.contains("does not match any PRIMARY KEY or UNIQUE constraint"),
            "expected the no-matching-constraint error; got {m:?}"
        ),
        other => panic!("expected a Sql error, got {other:?}"),
    }
}

/// §2: a NON-unique expression index is not a uniqueness constraint, so it cannot be a
/// conflict target. `CREATE INDEX` (no UNIQUE) on `a + b` leaves the table with no unique
/// expression constraint, so `ON CONFLICT(a + b)` is rejected.
#[test]
fn non_unique_expression_index_is_not_a_conflict_target() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b INTEGER, note TEXT)");
    exec(&mut db, "CREATE INDEX u ON t(a + b)"); // NOT unique
    exec(&mut db, "INSERT INTO t VALUES (1, 2, 'orig')");
    let e = assert_exec_error(
        &mut db,
        "INSERT INTO t VALUES (2, 1, 'new') ON CONFLICT(a + b) DO NOTHING",
    );
    match e {
        Error::Sql(m) => assert!(
            m.contains("does not match any PRIMARY KEY or UNIQUE constraint"),
            "expected the no-matching-constraint error; got {m:?}"
        ),
        other => panic!("expected a Sql error, got {other:?}"),
    }
}
