//! Conformance battery pinning the partial-index over-enforcement gap.
//!
//! ## What a partial index is (the binding contract)
//! Source: `spec/sqlite-doc/partialindex.html`.
//!   * §1 (intro): "A partial index is an index over a subset of the rows of a
//!     table." A `CREATE INDEX ... WHERE <pred>` is a partial index (§2: "Any index
//!     that includes the WHERE clause at the end is considered to be a partial
//!     index").
//!   * §2: "Only rows of the table for which the WHERE clause evaluates to true are
//!     included in the index. If the WHERE clause expression evaluates to NULL or to
//!     false for some rows of the table, then those rows are omitted from the index."
//!   * §2.1 (Unique Partial Indexes): "A partial index definition may include the
//!     UNIQUE keyword. If it does, then SQLite requires every entry *in the index* to
//!     be unique. This provides a mechanism for enforcing uniqueness across some
//!     subset of the rows in a table." So a partial UNIQUE index constrains ONLY the
//!     rows whose predicate is true — two rows that share the indexed value do NOT
//!     conflict unless BOTH are in the index.
//!
//! ## This is a DELIBERATE expected-fail pinning a KNOWN gap
//! The catalog stores only `IndexDef.partial: bool` — NOT the WHERE predicate
//! (`crates/minisqlite-catalog/src/builder.rs`: `partial: stmt.where_clause.is_some()`).
//! With no stored predicate, index maintenance
//! (`crates/minisqlite-exec/src/ops/dml_index.rs`) covers EVERY row regardless of the
//! WHERE clause. A partial UNIQUE index therefore OVER-ENFORCES: it rejects a
//! duplicate that real `sqlite3` ACCEPTS, because neither of the "conflicting" rows is
//! actually in the (partial) index. Reads are safe (the planner declines a partial
//! index for scans); the over-enforcement is SILENT on the write path.
//!
//! EVERY write verb feeds that one predicate-less maintenance path, so the gap is a
//! single class with several instances. This battery pins the ones cleanly observable
//! through the facade — INSERT, UPDATE, and the CREATE INDEX backfill, each reaching the
//! shared UNIQUE probe by a DIFFERENT operator — so a future fix that gates one verb but
//! forgets another is still caught, and all pass together when the fix lands.
//! (DELETE over-population and the FK-cascade re-index share the same root cause but are
//! not cleanly facade-observable, so they are not pinned here.)
//!
//! The ROOT FIX: the CATALOG must store the predicate
//! (`IndexDef.partial_predicate: Option<Expr>`, populated in `index_def_from_ast` on
//! both create and schema-reload), the PLAN must bind it per DML node, and EXEC
//! (`dml_index.rs::index_key_values`) must gate on it (skip rows whose predicate is
//! false). When that lands, the assertions below pass WITHOUT any change here.
//!
//! ## Note to future readers
//! These assertions encode the DOCUMENTED behavior transcribed from the spec above,
//! never from what this engine currently returns. An honest failing assertion IS the
//! pin: it fails today because the engine is wrong, and it passes by itself the moment
//! the catalog+plan+exec predicate fix lands; it is not weakened, `#[ignore]`d, or
//! special-cased to pass. The one CONTROL test
//! (`partial_unique_still_enforces_inside_predicate`) passes today and must STAY
//! correct after the fix — it guards against a fix that swings into UNDER-enforcement.

mod conformance;

use conformance::*;
// `Connection`/`Error` are named in the helper signature / the control's error-class
// assertion below. The harness imports them privately (so `conformance::*` does not
// re-export them); this file depends only on the pinned `minisqlite` facade surface,
// exactly like the harness and its siblings.
use minisqlite::{Connection, Error};

/// A fresh in-memory database with a table `t(a, b)` and a UNIQUE PARTIAL index on `a`
/// restricted to the rows where `b > 0`:
///
/// ```sql
/// CREATE TABLE t(a, b);
/// CREATE UNIQUE INDEX u ON t(a) WHERE b > 0;
/// ```
///
/// Per `partialindex.html` §2/§2.1 the index holds an entry ONLY for a row with
/// `b > 0`, and UNIQUE is enforced only ACROSS those in-index entries. Two rows that
/// share `a` therefore conflict IFF both have `b > 0`.
fn partial_unique() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "CREATE UNIQUE INDEX u ON t(a) WHERE b > 0");
    db
}

/// Assert a write statement that real `sqlite3` ACCEPTS actually succeeds.
///
/// This is the pin. Real sqlite accepts the statement because a UNIQUE *partial* index
/// constrains only the rows whose WHERE predicate is true (`partialindex.html` §2.1:
/// SQLite "requires every entry *in the index* to be unique"), so at each call site
/// below no two entries that are actually IN the index collide. The engine wrongly
/// REJECTS it today: index maintenance covers EVERY row, ignoring the predicate the
/// catalog does not yet store — so this panics with the gap explained. When the
/// predicate fix lands, `try_exec` returns `Ok` and this is a silent no-op. Shared by
/// the INSERT, UPDATE, and CREATE INDEX backfill witnesses (all reach the one path).
fn accepted(db: &mut Connection, sql: &str) {
    if let Err(e) = try_exec(db, sql) {
        panic!(
            "PARTIAL-INDEX OVER-ENFORCEMENT GAP (expected-fail until the catalog stores \
             the partial-index predicate): real sqlite ACCEPTS this statement because a \
             UNIQUE partial index \
             only constrains rows whose WHERE predicate is true (partialindex.html §2.1: \
             SQLite \"requires every entry IN THE INDEX to be unique\"), so no two \
             entries that are IN the index collide. The engine wrongly REJECTED it: \
             index maintenance covers EVERY row, ignoring the predicate the catalog does \
             not yet store.\n  sql: {sql}\n  error: {e:?}"
        );
    }
}

/// THE headline witness — INSERT (FAILS today). Both rows have `b = -1`, so NEITHER is
/// in the `WHERE b > 0` index; there is no in-index entry for `a = 1` at all, so the
/// second insert cannot violate uniqueness. Real sqlite accepts both rows
/// (`count(*) = 2`).
///
/// The engine indexes both rows (dropping the predicate), so the second insert sees a
/// phantom `a = 1` index entry from the first row and rejects the row real sqlite keeps
/// — the OVER-ENFORCEMENT this file pins.
#[test]
fn partial_unique_does_not_enforce_outside_predicate() {
    let mut db = partial_unique();
    accepted(&mut db, "INSERT INTO t VALUES (1, -1)");
    accepted(&mut db, "INSERT INTO t VALUES (1, -1)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2));
}

/// CONTROL — INSERT (PASSES today, and must STAY correct after the fix). Both rows have
/// `b = 5 > 0`, so BOTH are in the `WHERE b > 0` index; the second insert is a genuine
/// UNIQUE violation on the shared `a = 2` entry, which real sqlite rejects. This holds
/// whether index maintenance gates on the predicate or not, so it is the guard that a
/// future predicate fix does not swing the other way into UNDER-enforcement.
#[test]
fn partial_unique_still_enforces_inside_predicate() {
    let mut db = partial_unique();
    exec(&mut db, "INSERT INTO t VALUES (2, 5)");
    // A second in-index row with the same `a` is a real UNIQUE violation. `query`
    // executes the INSERT (the facade routes DML through the same path as `execute`), so
    // a successful insert would return an empty result and FAIL this assertion; only a
    // real violation returns an error. Assert the error CLASS (a constraint violation,
    // not an unrelated Sql/Io/Format error) so the control cannot pass for the wrong
    // reason — without brittle-matching the (unasserted) message wording.
    let e = assert_query_error(&mut db, "INSERT INTO t VALUES (2, 5)");
    assert!(
        matches!(e, Error::Constraint(_)),
        "the in-predicate duplicate must fail as a CONSTRAINT violation, got {e:?}"
    );
}

/// A mixed in/out INSERT case (FAILS today). The first row `(3, -1)` has `b <= 0`, so
/// it is OMITTED from the index; the second row `(3, 5)` has `b > 0`, so it is the ONLY
/// `a = 3` entry in the index. There is no in-index collision, so real sqlite accepts
/// both rows (`count(*) = 2`).
///
/// The engine indexes the first row too (predicate dropped), so the second insert
/// collides with that phantom `a = 3` entry and is wrongly rejected — the same
/// over-enforcement, exposed even when the surviving row IS inside the predicate.
#[test]
fn partial_unique_mixed_in_and_out() {
    let mut db = partial_unique();
    accepted(&mut db, "INSERT INTO t VALUES (3, -1)");
    accepted(&mut db, "INSERT INTO t VALUES (3, 5)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2));
}

/// UPDATE over-enforcement (FAILS today). The over-enforcement affects INSERT and
/// UPDATE alike, and UPDATE maintains the index through a SEPARATE operator
/// (`crates/minisqlite-exec/src/ops/update.rs`), so a fix that gates INSERT but forgets
/// UPDATE would silently leave this half wrong — this pins it. Both rows have `b <= 0`,
/// so NEITHER is in the `WHERE b > 0` index; re-keying one to `a = 1` creates no
/// in-index collision, so real sqlite ACCEPTS the UPDATE (`count(*) = 2`).
///
/// The engine indexed both rows (predicate dropped), so re-keying `(2, -1)` to `a = 1`
/// collides with the phantom `a = 1` entry and is wrongly rejected. (The two setup
/// inserts have distinct `a`, so they do not collide even under the buggy maintenance.)
#[test]
fn partial_unique_update_does_not_enforce_outside_predicate() {
    let mut db = partial_unique();
    exec(&mut db, "INSERT INTO t VALUES (1, -1)");
    exec(&mut db, "INSERT INTO t VALUES (2, -1)");
    accepted(&mut db, "UPDATE t SET a = 1 WHERE a = 2");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2));
}

/// CREATE INDEX backfill over-enforcement (FAILS today). The backfill builds the index
/// through YET ANOTHER path (`crates/minisqlite-exec/src/index_build.rs`) and probes the
/// same predicate-less UNIQUE check. Both existing rows have `b <= 0`, so the partial
/// index `WHERE b > 0` is built over ZERO rows and real sqlite CREATES it successfully;
/// the engine over-populates the backfill and falsely raises UNIQUE.
///
/// This is the cleanest witness: a single statement, and its setup (two inserts made
/// BEFORE any index exists) cannot itself trip the gap — so the failure is unambiguously
/// the backfill.
#[test]
fn partial_unique_backfill_ignores_predicate() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "INSERT INTO t VALUES (1, -1)");
    exec(&mut db, "INSERT INTO t VALUES (1, -1)");
    accepted(&mut db, "CREATE UNIQUE INDEX u ON t(a) WHERE b > 0");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2));
}
