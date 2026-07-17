//! Conformance battery for UNIQUE/non-UNIQUE PARTIAL index MAINTENANCE across every
//! write verb — the behavior the predicate-gating fix must deliver, asserted against
//! the documented `sqlite3` semantics (never against what this engine happens to do).
//!
//! ## The binding contract (transcribed from the spec)
//! Source: `spec/sqlite-doc/partialindex.html`.
//!   * §2: "Only rows of the table for which the WHERE clause evaluates to true are
//!     included in the index. If the WHERE clause expression evaluates to NULL or to
//!     false for some rows of the table, then those rows are omitted from the index."
//!   * §2.1 (Unique Partial Indexes): a partial UNIQUE index "requires every entry *in
//!     the index* to be unique" — uniqueness is enforced ONLY across the rows the
//!     predicate admits.
//!
//! A correct engine must therefore, for EVERY maintenance path (INSERT, UPDATE, DELETE,
//! CREATE INDEX backfill), key/probe/delete an index entry for a row IFF the row's WHERE
//! predicate is TRUE — evaluating the OLD row for the delete side of an UPDATE and the
//! NEW row for its insert side. This file pins the consequences that are cleanly
//! observable through the `minisqlite` facade:
//!   * backfill of a populated table indexes only matching rows (a pre-existing OUT
//!     duplicate does not fail the create; an IN duplicate does);
//!   * an UPDATE that moves a row OUT frees its value; one that moves a row IN (or
//!     re-keys within the predicate) still enforces uniqueness;
//!   * a DELETE of an OUT row leaves the IN entry intact, and a DELETE of the IN row
//!     frees the value;
//!   * a NULL predicate omits the row (distinct from FALSE);
//!   * a NON-unique partial index never hides rows from a read that cannot use it.
//!
//! It complements `conformance_partial_index_gap.rs` (the over-enforcement pins on
//! INSERT/UPDATE/backfill): this file adds the value-freeing / under-enforcement-guard
//! directions and the DELETE + NULL + read-correctness cases. The assertions encode
//! the SPEC's behavior — they are not weakened, `#[ignore]`d, or special-cased; a
//! failure means the engine diverges from `sqlite3`.

mod conformance;

use conformance::*;
// `Connection`/`Error` are named in the fixture signature and the constraint-class
// assertions; the harness imports them privately, so this file names the pinned facade
// surface directly, exactly like the sibling conformance files.
use minisqlite::{Connection, Error};

/// A fresh in-memory database with `t(a, b)` and a UNIQUE PARTIAL index on `a` over the
/// rows where `b > 0` — the same shape the gap battery uses:
///
/// ```sql
/// CREATE TABLE t(a, b);
/// CREATE UNIQUE INDEX u ON t(a) WHERE b > 0;
/// ```
///
/// The index holds an entry ONLY for a row with `b > 0`, and UNIQUE is enforced only
/// across those in-index entries.
fn unique_partial() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "CREATE UNIQUE INDEX u ON t(a) WHERE b > 0");
    db
}

/// Assert `sql` fails as a CONSTRAINT violation (the class real sqlite raises for a
/// UNIQUE conflict), not an unrelated Sql/Io/Format error — so a test cannot pass for
/// the wrong reason without brittle-matching the (unasserted) message text.
fn assert_constraint(db: &mut Connection, sql: &str) {
    let e = assert_exec_error(db, sql);
    assert!(matches!(e, Error::Constraint(_)), "expected a CONSTRAINT violation for {sql:?}, got {e:?}");
}

// ---- CREATE INDEX backfill --------------------------------------------------------

/// Backfilling a UNIQUE partial index over a populated table whose duplicate `a` values
/// are ALL OUTSIDE the predicate CREATES successfully (the index is built over zero
/// rows), and the resulting index is LIVE: a later in-predicate insert of that `a`
/// succeeds once, then a second in-predicate insert of the same `a` conflicts.
#[test]
fn backfill_skips_out_of_predicate_duplicates_then_index_is_live() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    // Two rows share a = 1 but both have b <= 0, so neither is in `WHERE b > 0`.
    exec(&mut db, "INSERT INTO t VALUES (1, -1)");
    exec(&mut db, "INSERT INTO t VALUES (1, -2)");
    // sqlite backfills 0 entries, so the create succeeds despite the duplicate a = 1.
    exec(&mut db, "CREATE UNIQUE INDEX u ON t(a) WHERE b > 0");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2));
    // The index is live: the FIRST in-predicate a = 1 has no existing entry → accepted.
    exec(&mut db, "INSERT INTO t VALUES (1, 5)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(3));
    // A SECOND in-predicate a = 1 collides with the entry just written → UNIQUE.
    assert_constraint(&mut db, "INSERT INTO t VALUES (1, 6)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(3));
}

/// Backfilling a UNIQUE partial index over a populated table with a duplicate `a` value
/// INSIDE the predicate FAILS the create (two in-index entries would collide), and the
/// pre-existing rows are left untouched.
#[test]
fn backfill_rejects_in_predicate_duplicate() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    // Both rows are in `WHERE b > 0` and share a = 1 → an in-index duplicate.
    exec(&mut db, "INSERT INTO t VALUES (1, 5)");
    exec(&mut db, "INSERT INTO t VALUES (1, 6)");
    assert_constraint(&mut db, "CREATE UNIQUE INDEX u ON t(a) WHERE b > 0");
    // The failed CREATE INDEX left the table rows in place.
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2));
}

// ---- UPDATE (both sides gate on their own frame) ----------------------------------

/// An UPDATE moving a row OUT of the predicate deletes its index entry, freeing the
/// value: a later in-predicate insert of that same `a` then succeeds. This exercises the
/// UPDATE delete-side gate (the OLD row was in the index, so its entry is removed) and
/// insert-side gate (the NEW row is out, so no entry is re-added).
#[test]
fn update_out_of_predicate_frees_the_value() {
    let mut db = unique_partial();
    exec(&mut db, "INSERT INTO t VALUES (1, 5)"); // in-index a = 1
    // Move it out (b: 5 → -1): the a = 1 entry must be removed.
    exec(&mut db, "UPDATE t SET b = -1 WHERE a = 1");
    // The value is free again: a fresh in-index a = 1 is accepted (no lingering entry).
    exec(&mut db, "INSERT INTO t VALUES (1, 5)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2));
    // Both rows share a = 1; only one (the new b = 5 row) is in the index.
    assert_rows(&mut db, "SELECT a, b FROM t ORDER BY b", &[vec![int(1), int(-1)], vec![int(1), int(5)]]);
}

/// An UPDATE moving a row INTO the predicate adds its entry and MUST then conflict with
/// an existing in-index entry for the same `a`. Guards against a fix that gates the
/// delete side but forgets to add/probe the new entry (silent UNDER-enforcement). The
/// aborted UPDATE leaves the row unchanged.
#[test]
fn update_into_predicate_conflicts_with_existing_entry() {
    let mut db = unique_partial();
    exec(&mut db, "INSERT INTO t VALUES (1, 5)"); // in-index a = 1
    // Out-of-index a = 1 (b <= 0): accepted, because it is not in the index and so does
    // not collide with the in-index a = 1 above.
    exec(&mut db, "INSERT INTO t VALUES (1, -1)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2));
    // Move the OUT row IN (b: -1 → 9): it would become a second in-index a = 1 → UNIQUE.
    assert_constraint(&mut db, "UPDATE t SET b = 9 WHERE b < 0");
    // The aborted UPDATE left the row at b = -1 (statement rolled back on the violation).
    assert_rows(&mut db, "SELECT a, b FROM t ORDER BY b", &[vec![int(1), int(-1)], vec![int(1), int(5)]]);
}

/// An UPDATE that RE-KEYS a row within the predicate (both OLD and NEW in-index) into an
/// existing in-index value still conflicts — the ordinary UNIQUE enforcement must remain
/// for the in-index subset after the fix.
#[test]
fn update_within_predicate_still_enforces() {
    let mut db = unique_partial();
    exec(&mut db, "INSERT INTO t VALUES (1, 5)"); // in-index a = 1
    exec(&mut db, "INSERT INTO t VALUES (2, 6)"); // in-index a = 2
    // Re-key the a = 2 row to a = 1 (stays in-index): collides with the a = 1 entry.
    assert_constraint(&mut db, "UPDATE t SET a = 1 WHERE a = 2");
    assert_rows(&mut db, "SELECT a, b FROM t ORDER BY a", &[vec![int(1), int(5)], vec![int(2), int(6)]]);
}

// ---- DELETE -----------------------------------------------------------------------

/// Deleting an OUT-of-predicate row is a no-op on the index: the IN-index entry for the
/// same `a` survives, so a subsequent in-predicate insert of that `a` still conflicts.
/// (If DELETE ignored the predicate and mis-computed the out row's key, it could not
/// remove the in row's entry — its rowid suffix differs — but this still pins that the
/// in-index entry is intact and enforcing after the delete.)
#[test]
fn delete_out_of_predicate_row_keeps_in_entry() {
    let mut db = unique_partial();
    exec(&mut db, "INSERT INTO t VALUES (1, 5)"); // in-index a = 1
    exec(&mut db, "INSERT INTO t VALUES (1, -1)"); // out-of-index
    exec(&mut db, "DELETE FROM t WHERE b < 0"); // remove the out row
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(1));
    // The in-index a = 1 entry is still present and enforcing.
    assert_constraint(&mut db, "INSERT INTO t VALUES (1, 7)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(1));
}

/// Deleting an IN-predicate row removes its index entry, freeing the value: a later
/// in-predicate insert of that same `a` then succeeds.
#[test]
fn delete_in_predicate_row_frees_the_value() {
    let mut db = unique_partial();
    exec(&mut db, "INSERT INTO t VALUES (1, 5)"); // in-index a = 1
    exec(&mut db, "DELETE FROM t WHERE a = 1"); // removes the in row + its entry
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
    exec(&mut db, "INSERT INTO t VALUES (1, 7)"); // a = 1 is free again → accepted
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(1));
}

// ---- NULL predicate (distinct from FALSE) -----------------------------------------

/// A row whose predicate evaluates to NULL is OMITTED from the index, exactly like a
/// FALSE one (`partialindex.html` §2). With `WHERE b > 0` and `b` NULL, `b > 0` is NULL,
/// so two rows sharing `a` with NULL `b` are both omitted and both accepted under a
/// UNIQUE partial index.
#[test]
fn null_predicate_omits_row_from_unique_index() {
    let mut db = unique_partial();
    exec(&mut db, "INSERT INTO t VALUES (1, NULL)");
    exec(&mut db, "INSERT INTO t VALUES (1, NULL)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2));
    // A row that IS in the predicate still enforces against other in-index rows, but the
    // NULL-predicate rows above never entered the index, so this a = 1 is the first
    // in-index entry and is accepted.
    exec(&mut db, "INSERT INTO t VALUES (1, 5)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(3));
    // ...and a SECOND in-index a = 1 now conflicts.
    assert_constraint(&mut db, "INSERT INTO t VALUES (1, 6)");
}

// ---- Reads never lose rows to a partial index -------------------------------------

/// A NON-unique partial index must never hide rows from a query that cannot use it: a
/// read that filters on the indexed column must still return the OUT-of-predicate rows
/// (the planner declines a partial index for scans, so the answer is the full table).
/// If a read wrongly used the partial index, the `b <= 0` row would vanish.
#[test]
fn nonunique_partial_index_does_not_hide_rows_from_reads() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "CREATE INDEX idx ON t(a) WHERE b > 0");
    exec(&mut db, "INSERT INTO t VALUES (1, 5)"); // in-index
    exec(&mut db, "INSERT INTO t VALUES (1, -1)"); // out-of-index
    exec(&mut db, "INSERT INTO t VALUES (2, 3)"); // in-index, different a
    // Both a = 1 rows are returned, including the out-of-predicate one.
    assert_rows(&mut db, "SELECT b FROM t WHERE a = 1 ORDER BY b", &[vec![int(-1)], vec![int(5)]]);
    assert_scalar(&mut db, "SELECT count(*) FROM t WHERE a = 1", int(2));
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(3));
    // A non-unique partial index permits duplicate in-index values (no uniqueness):
    // inserting another a = 2, b > 0 is fine.
    exec(&mut db, "INSERT INTO t VALUES (2, 9)");
    assert_scalar(&mut db, "SELECT count(*) FROM t WHERE a = 2", int(2));
}
