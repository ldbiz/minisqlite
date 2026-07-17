//! Conformance battery: **WITHOUT ROWID feature holes** — shapes real `sqlite3`
//! accepts that the engine used to refuse with a loud "not yet supported":
//!   * a TABLE-LEVEL single-column INTEGER PRIMARY KEY (`PRIMARY KEY(a)` with
//!     `a INTEGER`) on a WITHOUT ROWID table;
//!   * `INSERT ... ON CONFLICT` (UPSERT) targeting a WITHOUT ROWID table's PRIMARY KEY;
//!   * `CREATE TRIGGER` bodies firing on a WITHOUT ROWID table.
//!
//! Every expected value here is TRANSCRIBED FROM THE SQLITE DOCS, never from what the
//! engine currently returns; a spec-correct case is never weakened or `#[ignore]`d to
//! make it pass. Each case is its own `#[test]` so an unimplemented hole fails exactly
//! that case.
//!
//! Spec sources (all under `spec/sqlite-doc/`):
//!   * `withoutrowid.html`: a WITHOUT ROWID table stores each row in a b-tree KEYED BY
//!     its PRIMARY KEY (so a plain scan visits rows in PK order); "the PRIMARY KEY ...
//!     is not omitted"; "NOT NULL is enforced on every column of the PRIMARY KEY".
//!     "WITHOUT ROWID is ... available on ... ordinary tables" and "may contain a
//!     PRIMARY KEY that ... is an INTEGER". A WITHOUT ROWID table does NOT make an
//!     INTEGER PRIMARY KEY an alias for a rowid (there is no rowid).
//!   * `lang_createtable.html` §5: an INTEGER PRIMARY KEY becomes a rowid alias only on
//!     a rowid table; "the rowid ... is omitted in WITHOUT ROWID tables".
//!   * `lang_UPSERT.html`: "An UPSERT is an ordinary INSERT ... that is followed by the
//!     special ON CONFLICT clause"; on a conflict of the named uniqueness constraint it
//!     runs `DO NOTHING` (skip) or `DO UPDATE SET ...` (update the existing row); the
//!     `excluded.` alias names the row that would have been inserted. UPSERT applies to
//!     WITHOUT ROWID tables (its PRIMARY KEY is a uniqueness constraint).
//!   * `lang_createtrigger.html`: a `FOR EACH ROW` trigger fires once per affected row;
//!     `NEW`/`OLD` reference the row values. Triggers apply to WITHOUT ROWID tables.

mod conformance;
use conformance::*;

use minisqlite::Error;

// -------------------------------------------------------------------------------------
// Hole (c): table-level single-column INTEGER PRIMARY KEY on a WITHOUT ROWID table.
// -------------------------------------------------------------------------------------

#[test]
fn table_level_integer_pk_round_trips_in_key_order() {
    // `PRIMARY KEY(a)` with `a INTEGER` (table-level) on a WITHOUT ROWID table is a
    // normal integer PRIMARY KEY (NOT a rowid alias — WR tables have no rowid). The row
    // is stored keyed by `a`, so a plain scan visits rows in ascending `a` order
    // regardless of INSERT order, and every column reads back its stored value.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT, PRIMARY KEY(a)) WITHOUT ROWID");
    exec(&mut db, "INSERT INTO t VALUES (2, 'two'), (1, 'one'), (3, 'three')");
    assert_rows(
        &mut db,
        "SELECT a, b FROM t",
        &[
            vec![int(1), text("one")],
            vec![int(2), text("two")],
            vec![int(3), text("three")],
        ],
    );
    // The non-first column selected alone still reads its own value (a width/register
    // slip would surface here).
    assert_rows(&mut db, "SELECT b FROM t", &[vec![text("one")], vec![text("two")], vec![text("three")]]);
}

#[test]
fn table_level_integer_pk_lookup_and_uniqueness() {
    // The PRIMARY KEY is the WR b-tree key: a WHERE on it selects the matching row, and
    // inserting a duplicate key is a PRIMARY KEY constraint violation (not a silent
    // second row). withoutrowid.html: uniqueness is enforced on the PK.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT, PRIMARY KEY(a)) WITHOUT ROWID");
    exec(&mut db, "INSERT INTO t VALUES (10, 'ten'), (20, 'twenty')");
    assert_scalar(&mut db, "SELECT b FROM t WHERE a = 10", text("ten"));
    assert_scalar(&mut db, "SELECT b FROM t WHERE a = 20", text("twenty"));

    let dup = try_exec(&mut db, "INSERT INTO t VALUES (10, 'again')");
    match dup {
        Err(Error::Constraint(m)) => {
            assert!(m.starts_with("UNIQUE constraint failed") || m.starts_with("PRIMARY KEY"), "got {m:?}");
        }
        other => panic!("expected a PRIMARY KEY constraint violation, got {other:?}"),
    }
    // The failed insert added no row: still exactly two rows, unchanged.
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2));
    assert_scalar(&mut db, "SELECT b FROM t WHERE a = 10", text("ten"));
}

#[test]
fn table_level_integer_pk_is_not_a_rowid_alias() {
    // lang_createtable.html §5: a WITHOUT ROWID table omits the rowid, so an INTEGER
    // PRIMARY KEY there is NOT a rowid alias — referencing `rowid` is "no such column".
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT, PRIMARY KEY(a)) WITHOUT ROWID");
    exec(&mut db, "INSERT INTO t VALUES (5, 'five')");
    assert!(
        try_exec(&mut db, "SELECT rowid FROM t").is_err(),
        "a WITHOUT ROWID table has no rowid column"
    );
}

#[test]
fn table_level_composite_integer_pk_orders_by_declared_key() {
    // A table-level composite PK `(b, a)` with integer members orders the WR b-tree by
    // (b, a) in that declared order: rows sort by b first, then a.
    let mut db = mem();
    exec(
        &mut db,
        "CREATE TABLE t(a INTEGER, b INTEGER, c TEXT, PRIMARY KEY(b, a)) WITHOUT ROWID",
    );
    exec(&mut db, "INSERT INTO t VALUES (2, 1, 'x'), (1, 1, 'y'), (1, 2, 'z')");
    // Key order (b, a): (1,1) < (1,2) < (2,1) → c = y, x, z.
    assert_rows(
        &mut db,
        "SELECT c FROM t",
        &[vec![text("y")], vec![text("x")], vec![text("z")]],
    );
}
