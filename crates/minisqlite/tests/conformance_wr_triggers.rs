//! Conformance battery: **CREATE TRIGGER bodies firing on a WITHOUT ROWID table**
//! (hole (b) of the WITHOUT ROWID feature set). Real `sqlite3` fires `BEFORE`/`AFTER`
//! `INSERT`/`UPDATE`/`DELETE` `FOR EACH ROW` triggers on a WITHOUT ROWID table exactly
//! as on a rowid table; the engine used to refuse every one with a loud
//! "triggers on WITHOUT ROWID table ... are not yet supported".
//!
//! The load-bearing subtlety this battery pins: a WITHOUT ROWID row has **no rowid
//! register**, so its `OLD`/`NEW` trigger frame is width `N` (the N stored columns),
//! NOT the rowid table's width `N+1`. The plan side binds `OLD.col_k → Column(k)` and
//! `NEW.col_k → Column(N + k)` for a WR target; if the executor built the frame at the
//! wrong width, an `OLD.*`/`NEW.*` read would shift into the wrong half and surface as a
//! garbage/NULL value here. The multi-column `OLD`+`NEW` cases below are the width probes.
//!
//! Every expected value is TRANSCRIBED FROM THE SQLITE DOCS, never from what the engine
//! currently returns; a spec-correct case is never weakened or `#[ignore]`d to make it
//! pass. Each case is its own `#[test]`.
//!
//! Spec sources (all under `spec/sqlite-doc/`):
//!   * `lang_createtrigger.html`: a `FOR EACH ROW` trigger fires once per affected row;
//!     `NEW` references the row being inserted/updated, `OLD` the row being updated/
//!     deleted. A `BEFORE` trigger runs "before the associated ... operation" (so before
//!     the row is written); an `AFTER` trigger runs after. A `WHEN` clause gates whether
//!     the body runs for a given row. `UPDATE OF <columns>` fires only when one of the
//!     named columns is in the UPDATE's SET list. `RAISE(IGNORE)` "abandons the current
//!     ... row" without an error and continues with the next.
//!   * `withoutrowid.html`: a WITHOUT ROWID table has no rowid; it stores rows in a
//!     PRIMARY KEY index b-tree (so a plain scan visits rows in PRIMARY KEY order).
//!   * `pragma.html` (`recursive_triggers`): OFF by default, so a trigger's own writes
//!     do not recursively re-fire the trigger.

mod conformance;
use conformance::*;

use minisqlite::Error;

// -------------------------------------------------------------------------------------
// AFTER INSERT — NEW.* binds at width N; fires once per inserted row, post-write.
// -------------------------------------------------------------------------------------

#[test]
fn after_insert_for_each_row_logs_new_values() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k TEXT PRIMARY KEY, v TEXT) WITHOUT ROWID");
    exec(&mut db, "CREATE TABLE log(k TEXT, v TEXT)");
    exec(
        &mut db,
        "CREATE TRIGGER t_ai AFTER INSERT ON t BEGIN \
           INSERT INTO log(k, v) VALUES (NEW.k, NEW.v); END",
    );
    exec(&mut db, "INSERT INTO t VALUES ('b', 'two'), ('a', 'one')");
    // The base rows landed and read back in PRIMARY KEY order.
    assert_rows(
        &mut db,
        "SELECT k, v FROM t",
        &[vec![text("a"), text("one")], vec![text("b"), text("two")]],
    );
    // The trigger fired once per inserted row with the correct NEW values (a width-N
    // frame slip would read NEW.k / NEW.v from the wrong register and mis-log here).
    assert_rows_unordered(
        &mut db,
        "SELECT k, v FROM log",
        &[vec![text("a"), text("one")], vec![text("b"), text("two")]],
    );
}

#[test]
fn after_insert_does_not_fire_for_a_skipped_or_ignore_row() {
    // A conflicting `INSERT OR IGNORE` row is not written, so its AFTER INSERT trigger
    // must not fire (the trigger logs each NEW; an unwritten row leaves no log entry).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k TEXT PRIMARY KEY, v TEXT) WITHOUT ROWID");
    exec(&mut db, "CREATE TABLE log(k TEXT)");
    exec(&mut db, "CREATE TRIGGER t_ai AFTER INSERT ON t BEGIN INSERT INTO log VALUES (NEW.k); END");
    exec(&mut db, "INSERT INTO t VALUES ('a', 'one')");
    // Second insert conflicts on the PRIMARY KEY and is ignored → no write, no fire.
    exec(&mut db, "INSERT OR IGNORE INTO t VALUES ('a', 'dup'), ('b', 'two')");
    assert_rows_unordered(&mut db, "SELECT k FROM log", &[vec![text("a")], vec![text("b")]]);
    assert_rows(
        &mut db,
        "SELECT k, v FROM t",
        &[vec![text("a"), text("one")], vec![text("b"), text("two")]],
    );
}

// -------------------------------------------------------------------------------------
// BEFORE INSERT — sees NEW before the write; a WHEN + RAISE(IGNORE) drops the row.
// -------------------------------------------------------------------------------------

#[test]
fn before_insert_when_raise_ignore_skips_the_row() {
    // A BEFORE INSERT trigger with `WHEN NEW.v < 0` that RAISEs IGNORE abandons exactly
    // the rows it matches (lang_createtrigger.html: RAISE(IGNORE) abandons the current
    // row, no error). The non-matching rows are inserted normally.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k TEXT PRIMARY KEY, v INTEGER) WITHOUT ROWID");
    exec(
        &mut db,
        "CREATE TRIGGER t_bi BEFORE INSERT ON t WHEN NEW.v < 0 BEGIN \
           SELECT RAISE(IGNORE); END",
    );
    exec(&mut db, "INSERT INTO t VALUES ('a', 1), ('b', -5), ('c', 3), ('d', -1)");
    // Only the non-negative rows survive; the RAISE(IGNORE) rows never reached storage.
    assert_rows(
        &mut db,
        "SELECT k, v FROM t",
        &[vec![text("a"), int(1)], vec![text("c"), int(3)]],
    );
}

// -------------------------------------------------------------------------------------
// AFTER UPDATE — OLD and NEW both bind at width N (the strongest register-slip probe).
// -------------------------------------------------------------------------------------

#[test]
fn after_update_logs_old_and_new_across_all_columns() {
    // A 3-column WR table with a single-column PK. An AFTER UPDATE trigger logs every
    // OLD.* and NEW.* column. For a WR target OLD.col_k is Column(k) and NEW.col_k is
    // Column(N + k); a wrong frame width would cross the two halves and mis-log.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a TEXT PRIMARY KEY, b TEXT, c TEXT) WITHOUT ROWID");
    exec(&mut db, "CREATE TABLE log(oa TEXT, ob TEXT, oc TEXT, na TEXT, nb TEXT, nc TEXT)");
    exec(
        &mut db,
        "CREATE TRIGGER t_au AFTER UPDATE ON t BEGIN \
           INSERT INTO log VALUES (OLD.a, OLD.b, OLD.c, NEW.a, NEW.b, NEW.c); END",
    );
    exec(&mut db, "INSERT INTO t VALUES ('x', 'b1', 'c1')");
    exec(&mut db, "UPDATE t SET b = 'b2', c = 'c2' WHERE a = 'x'");
    // The updated base row.
    assert_rows(&mut db, "SELECT a, b, c FROM t", &[vec![text("x"), text("b2"), text("c2")]]);
    // OLD is the pre-update row, NEW the post-update row — each column in its own slot.
    assert_rows(
        &mut db,
        "SELECT oa, ob, oc, na, nb, nc FROM log",
        &[vec![text("x"), text("b1"), text("c1"), text("x"), text("b2"), text("c2")]],
    );
}

#[test]
fn after_update_that_changes_the_primary_key_sees_old_and_new_pk() {
    // Changing the PRIMARY KEY of a WR row rewrites its b-tree entry (old key deleted,
    // new key inserted). The AFTER UPDATE trigger still sees OLD.k (the old key) and
    // NEW.k (the new key) — the frame carries both PK values, not one.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k TEXT PRIMARY KEY, v TEXT) WITHOUT ROWID");
    exec(&mut db, "CREATE TABLE log(ok TEXT, nk TEXT)");
    exec(
        &mut db,
        "CREATE TRIGGER t_au AFTER UPDATE ON t BEGIN INSERT INTO log VALUES (OLD.k, NEW.k); END",
    );
    exec(&mut db, "INSERT INTO t VALUES ('old', 'v')");
    exec(&mut db, "UPDATE t SET k = 'new' WHERE k = 'old'");
    assert_rows(&mut db, "SELECT k, v FROM t", &[vec![text("new"), text("v")]]);
    assert_rows(&mut db, "SELECT ok, nk FROM log", &[vec![text("old"), text("new")]]);
}

#[test]
fn update_of_named_column_filters_the_trigger() {
    // `UPDATE OF b` fires only when column b is in the SET list (lang_createtrigger.html).
    // An UPDATE that sets only a different column must not fire it.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a TEXT PRIMARY KEY, b TEXT, c TEXT) WITHOUT ROWID");
    exec(&mut db, "CREATE TABLE log(tag TEXT)");
    exec(
        &mut db,
        "CREATE TRIGGER t_aub AFTER UPDATE OF b ON t BEGIN INSERT INTO log VALUES ('b-changed'); END",
    );
    exec(&mut db, "INSERT INTO t VALUES ('x', 'b1', 'c1')");
    // Sets only c: the OF b trigger does NOT fire.
    exec(&mut db, "UPDATE t SET c = 'c2' WHERE a = 'x'");
    assert_scalar(&mut db, "SELECT count(*) FROM log", int(0));
    // Sets b: the trigger fires once.
    exec(&mut db, "UPDATE t SET b = 'b2' WHERE a = 'x'");
    assert_rows(&mut db, "SELECT tag FROM log", &[vec![text("b-changed")]]);
}

// -------------------------------------------------------------------------------------
// DELETE — OLD binds at width N; BEFORE can protect a row.
// -------------------------------------------------------------------------------------

#[test]
fn after_delete_for_each_row_logs_old_values() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k TEXT PRIMARY KEY, v TEXT) WITHOUT ROWID");
    exec(&mut db, "CREATE TABLE log(k TEXT, v TEXT)");
    exec(
        &mut db,
        "CREATE TRIGGER t_ad AFTER DELETE ON t BEGIN INSERT INTO log VALUES (OLD.k, OLD.v); END",
    );
    exec(&mut db, "INSERT INTO t VALUES ('a', 'one'), ('b', 'two'), ('c', 'three')");
    exec(&mut db, "DELETE FROM t WHERE k IN ('a', 'c')");
    // The surviving base row.
    assert_rows(&mut db, "SELECT k, v FROM t", &[vec![text("b"), text("two")]]);
    // AFTER DELETE fired once per removed row with that row's OLD values.
    assert_rows_unordered(
        &mut db,
        "SELECT k, v FROM log",
        &[vec![text("a"), text("one")], vec![text("c"), text("three")]],
    );
}

#[test]
fn before_delete_when_raise_ignore_protects_the_row() {
    // A BEFORE DELETE trigger that RAISEs IGNORE for protected rows keeps them: the row
    // is never removed and (being a BEFORE fire) no removal or AFTER work happens for it.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k TEXT PRIMARY KEY, keep INTEGER) WITHOUT ROWID");
    exec(
        &mut db,
        "CREATE TRIGGER t_bd BEFORE DELETE ON t WHEN OLD.keep = 1 BEGIN SELECT RAISE(IGNORE); END",
    );
    exec(&mut db, "INSERT INTO t VALUES ('a', 0), ('b', 1), ('c', 0)");
    exec(&mut db, "DELETE FROM t");
    // Only the protected row 'b' survives; 'a' and 'c' were deleted.
    assert_rows(&mut db, "SELECT k, keep FROM t", &[vec![text("b"), int(1)]]);
}

// -------------------------------------------------------------------------------------
// Composite PRIMARY KEY — the width-N frame must place every column correctly.
// -------------------------------------------------------------------------------------

#[test]
fn composite_pk_after_insert_new_columns_have_no_register_slip() {
    // A WR table with a composite PK `(a, b)` and a trailing payload `c`. The AFTER
    // INSERT trigger logs all three NEW columns; with a width-N frame each lands in its
    // own register regardless of how many columns the PK spans.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT, c TEXT, PRIMARY KEY(a, b)) WITHOUT ROWID");
    exec(&mut db, "CREATE TABLE log(a INTEGER, b TEXT, c TEXT)");
    exec(
        &mut db,
        "CREATE TRIGGER t_ai AFTER INSERT ON t BEGIN INSERT INTO log VALUES (NEW.a, NEW.b, NEW.c); END",
    );
    exec(&mut db, "INSERT INTO t VALUES (1, 'x', 'first'), (1, 'y', 'second')");
    assert_rows_unordered(
        &mut db,
        "SELECT a, b, c FROM log",
        &[vec![int(1), text("x"), text("first")], vec![int(1), text("y"), text("second")]],
    );
}

// -------------------------------------------------------------------------------------
// recursive_triggers OFF (default): a trigger's own write does not re-fire it.
// -------------------------------------------------------------------------------------

#[test]
fn after_insert_does_not_recurse_by_default() {
    // With `PRAGMA recursive_triggers` OFF (the default), a trigger that inserts into its
    // OWN table does not recursively re-fire (pragma.html). One user INSERT therefore
    // yields the user row plus exactly ONE trigger-inserted row — not an unbounded chain.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k TEXT PRIMARY KEY, v TEXT) WITHOUT ROWID");
    exec(
        &mut db,
        "CREATE TRIGGER t_ai AFTER INSERT ON t WHEN NEW.k = 'seed' BEGIN \
           INSERT INTO t VALUES ('derived', NEW.v); END",
    );
    exec(&mut db, "INSERT INTO t VALUES ('seed', 'v')");
    // 'seed' (user) + 'derived' (trigger, which did not itself re-fire) = 2 rows.
    assert_rows(
        &mut db,
        "SELECT k, v FROM t",
        &[vec![text("derived"), text("v")], vec![text("seed"), text("v")]],
    );
}

// -------------------------------------------------------------------------------------
// A WR table that carries triggers still enforces its PRIMARY KEY (no silent bypass).
// -------------------------------------------------------------------------------------

#[test]
fn triggers_do_not_disable_primary_key_uniqueness() {
    // Enabling triggers on a WR table must not weaken its constraints: a duplicate PK is
    // still a constraint violation, and the AFTER INSERT trigger does not fire for the
    // rejected row (it aborts before any write).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k TEXT PRIMARY KEY, v TEXT) WITHOUT ROWID");
    exec(&mut db, "CREATE TABLE log(k TEXT)");
    exec(&mut db, "CREATE TRIGGER t_ai AFTER INSERT ON t BEGIN INSERT INTO log VALUES (NEW.k); END");
    exec(&mut db, "INSERT INTO t VALUES ('a', 'one')");
    match try_exec(&mut db, "INSERT INTO t VALUES ('a', 'dup')") {
        Err(Error::Constraint(_)) => {}
        other => panic!("expected a PRIMARY KEY constraint violation, got {other:?}"),
    }
    // The rejected insert wrote no base row and fired no trigger.
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(1));
    assert_rows(&mut db, "SELECT k FROM log", &[vec![text("a")]]);
}
