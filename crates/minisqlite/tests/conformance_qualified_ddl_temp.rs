//! Conformance battery: **schema-qualified (`temp.`) write-side DDL issued while the
//! `temp` store is NOT yet materialized**, exercised through the pinned
//! `minisqlite::Connection` facade.
//!
//! A fresh connection has only `main` live; the `temp` schema (index 1) is created
//! LAZILY on the first real temp object. A `temp.`-qualified `CREATE INDEX` / `DROP` /
//! `ALTER TABLE` names the (not-yet-materialized) temp namespace, and must behave as if
//! `temp` existed but were EMPTY — never crash, and never materialize `temp` as a side
//! effect. These cases previously panicked (a dead-namespace out-of-bounds index into the
//! per-namespace pager/catalog vectors); every `#[test]` here drives them through
//! `Connection`, so a panic fails the test — the assertions therefore prove BOTH "no
//! panic" and the exact SQL result.
//!
//! "Behave as an empty temp store" is a UNIVERSAL contract, not just the "no such <kind>"
//! cases: the concrete `SchemaCatalog` checks the `sqlite_`-reserved rule BEFORE existence
//! and before `IF EXISTS`, so a `temp.`-qualified DDL on a reserved name (e.g.
//! `temp.sqlite_master`) must be the reserved-name error even on a non-live temp — and the
//! `IF EXISTS` form must ERROR, not silently succeed. The non-live path therefore does not
//! hand-roll messages; it runs the real mutation against a THROWAWAY empty temp store, so
//! every message (reserved, `IF EXISTS` no-op, "no such …") comes verbatim from the one
//! source of truth. The `live_temp_*` guards re-assert the same literals through the REAL
//! materialized catalog, cross-checking that the two paths agree.
//!
//! Every expectation is TRANSCRIBED FROM THE SQLITE DOCS in `spec/sqlite-doc/`, never from
//! what the engine currently returns; an assertion is never weakened to pass.
//!
//! Spec sources (all under `spec/sqlite-doc/`):
//!   * `lang_createindex.html`: an index inherits the schema of the table it indexes; a
//!     `schema.`-qualified index name must match its table's schema. A `temp.`-qualified
//!     index whose (unqualified) table is not a table of the temp schema is therefore
//!     "no such table: <table>".
//!   * `lang_droptable.html` / `lang_dropindex.html` / `lang_dropview.html` /
//!     `lang_droptrigger.html`: `DROP <kind> schema.name` on an object absent from that
//!     schema is an error ("no such <kind>: <name>"); the `IF EXISTS` clause turns that
//!     error into a silent no-op.
//!   * `lang_altertable.html`: `ALTER TABLE schema.name` on a table absent from that
//!     schema is "no such table: <name>" (ALTER has no `IF EXISTS`).
//!   * `pragma.html` #pragma_database_list: `temp` (seq 1) is listed ONLY once a temp
//!     object materializes its schema — a failed/no-op `temp.`-qualified DDL must NOT make
//!     it appear.
//!   * `schematab.html` §2: the temp schema's table is reachable as `temp.sqlite_master`,
//!     where a real temp object (e.g. an index on a temp table) is listed.
//!
//! The object NAME in each message is the bare object name (`x`, `t`), not the
//! `temp.`-qualified spelling — matching the concrete per-namespace `SchemaCatalog`, the
//! one source of truth for these strings. Each behavior is its own small `#[test]`.

mod conformance;
use conformance::*;

use minisqlite::{Connection, Error};

/// `db.execute(sql)` must be `Err(Error::Sql(expected))` — the EXACT message, and the
/// `Sql` variant (a pure syntax/name error, not `Constraint`/`Format`/`Io`). Pinning the
/// exact string (not a substring) holds the contract precisely; matching the variant stops
/// a regression that errors for the wrong reason from passing. If the engine PANICS
/// instead of returning `Err`, the `#[test]` fails here — which is exactly the crash this
/// suite guards against.
#[track_caller]
fn assert_sql_err(db: &mut Connection, sql: &str, expected: &str) {
    match db.execute(sql) {
        Ok(()) => panic!("expected Err(Sql({expected:?})) but `{sql}` succeeded"),
        Err(Error::Sql(m)) => {
            assert_eq!(m, expected, "wrong error message for `{sql}`");
        }
        Err(other) => panic!("expected Err(Sql({expected:?})) for `{sql}`, got {other:?}"),
    }
}

/// The `temp` schema is NOT materialized: `PRAGMA database_list` lists ONLY `main`. Used to
/// prove that a failed/no-op `temp.`-qualified DDL did not lazily create the temp store.
#[track_caller]
fn assert_temp_absent(db: &mut Connection) {
    assert_rows(db, "PRAGMA database_list", &[vec![int(0), text("main"), text("")]]);
}

// ---------------------------------------------------------------------------
// The crash cases: `temp.`-qualified write DDL before temp exists. Each must be a
// clean SQL error (no panic) and must leave temp unmaterialized.
// ---------------------------------------------------------------------------

#[test]
fn create_index_temp_qualified_when_temp_absent_is_no_such_table() {
    // The index's schema is `temp`, which owns no table `t` (temp is not materialized),
    // so it is "no such table: t" — even though a `main.t` exists (an index must live in
    // its table's schema; the temp schema has no such table). Was a panic.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    assert_sql_err(&mut db, "CREATE INDEX temp.ix ON t(a)", "no such table: t");
    assert_temp_absent(&mut db);
}

#[test]
fn create_index_temp_qualified_when_table_also_absent_is_no_such_table() {
    // With no `t` anywhere either, the answer is the same "no such table: t" (the temp
    // schema still has no table `t`), never a panic.
    let mut db = mem();
    assert_sql_err(&mut db, "CREATE INDEX temp.ix ON t(a)", "no such table: t");
    assert_temp_absent(&mut db);
}

#[test]
fn drop_temp_qualified_when_temp_absent_is_no_such_object_per_kind() {
    // DROP (no IF EXISTS) of a `temp.`-qualified object that cannot exist (temp is not
    // materialized) is the concrete catalog's per-kind "no such <kind>: <name>", never a
    // panic. The name is the bare object name, not the `temp.`-qualified spelling.
    let mut db = mem();
    assert_sql_err(&mut db, "DROP TABLE temp.x", "no such table: x");
    assert_sql_err(&mut db, "DROP INDEX temp.ix", "no such index: ix");
    assert_sql_err(&mut db, "DROP VIEW temp.v", "no such view: v");
    assert_sql_err(&mut db, "DROP TRIGGER temp.tr", "no such trigger: tr");
    assert_temp_absent(&mut db);
}

#[test]
fn drop_if_exists_temp_qualified_when_temp_absent_is_noop() {
    // `DROP <kind> IF EXISTS temp.<name>` on a not-yet-materialized temp is a silent Ok
    // no-op (as against a live-but-empty temp schema) and must NOT materialize temp.
    let mut db = mem();
    exec(&mut db, "DROP TABLE IF EXISTS temp.x");
    exec(&mut db, "DROP INDEX IF EXISTS temp.ix");
    exec(&mut db, "DROP VIEW IF EXISTS temp.v");
    exec(&mut db, "DROP TRIGGER IF EXISTS temp.tr");
    // Nothing was created: temp is still absent.
    assert_temp_absent(&mut db);
}

#[test]
fn alter_table_temp_qualified_when_temp_absent_is_no_such_table() {
    // `ALTER TABLE temp.<name>` on a not-yet-materialized temp is "no such table: <name>"
    // (ALTER has no IF EXISTS), never a panic. The router is action-agnostic, so every
    // action takes the same path.
    let mut db = mem();
    assert_sql_err(&mut db, "ALTER TABLE temp.x RENAME TO y", "no such table: x");
    assert_sql_err(&mut db, "ALTER TABLE temp.x ADD COLUMN c INTEGER", "no such table: x");
    assert_sql_err(&mut db, "ALTER TABLE temp.x RENAME COLUMN a TO b", "no such table: x");
    assert_temp_absent(&mut db);
}

// ---------------------------------------------------------------------------
// Reserved-name (`sqlite_`) cases. The concrete `SchemaCatalog` checks the
// `sqlite_`-reserved rule BEFORE existence AND before `IF EXISTS`, so a
// `temp.`-qualified DDL naming a reserved object is a RESERVED-name error even
// while temp is not materialized — never the plain "no such <kind>", and (for
// the `IF EXISTS` form) never a silent success. The non-live path runs the real
// mutation against a throwaway EMPTY temp store, so it produces these messages
// verbatim; each must also leave temp unmaterialized. Spec: `sqlite_`-prefixed
// names are reserved for internal use and may not be created/dropped/altered/
// indexed (`lang_createtable.html` §3, `lang_createindex.html`; the schema table
// `sqlite_master`, `schematab.html` §2).
// ---------------------------------------------------------------------------

#[test]
fn drop_table_temp_qualified_reserved_name_may_not_be_dropped() {
    // The `sqlite_` reserved rule fires before existence, so `DROP TABLE
    // temp.sqlite_master` is "table sqlite_master may not be dropped", NOT
    // "no such table: sqlite_master".
    let mut db = mem();
    assert_sql_err(
        &mut db,
        "DROP TABLE temp.sqlite_master",
        "table sqlite_master may not be dropped",
    );
    assert_temp_absent(&mut db);
}

#[test]
fn drop_table_if_exists_temp_qualified_reserved_name_still_errors() {
    // The reserved check precedes the existence test, so it fires EVEN UNDER IF EXISTS:
    // `DROP TABLE IF EXISTS temp.sqlite_master` is an ERROR, never the forbidden silent
    // "false success" an ordinary absent object would no-op to.
    let mut db = mem();
    assert_sql_err(
        &mut db,
        "DROP TABLE IF EXISTS temp.sqlite_master",
        "table sqlite_master may not be dropped",
    );
    assert_temp_absent(&mut db);
}

#[test]
fn drop_index_temp_qualified_reserved_name_cannot_be_dropped() {
    // A `sqlite_`-prefixed index (e.g. an auto-index backing a UNIQUE/PRIMARY KEY) cannot
    // be dropped directly; the reserved check fires before existence and before IF EXISTS,
    // for both the plain and the `IF EXISTS` form.
    let mut db = mem();
    assert_sql_err(
        &mut db,
        "DROP INDEX temp.sqlite_autoindex_tt_1",
        "index associated with UNIQUE or PRIMARY KEY constraint cannot be dropped",
    );
    assert_sql_err(
        &mut db,
        "DROP INDEX IF EXISTS temp.sqlite_autoindex_tt_1",
        "index associated with UNIQUE or PRIMARY KEY constraint cannot be dropped",
    );
    assert_temp_absent(&mut db);
}

#[test]
fn create_index_temp_qualified_on_reserved_table_may_not_be_indexed() {
    // Indexing a `sqlite_`-prefixed TABLE is rejected as "table sqlite_master may not be
    // indexed" — the reserved-table rule, not "no such table".
    let mut db = mem();
    assert_sql_err(
        &mut db,
        "CREATE INDEX temp.ix ON sqlite_master(a)",
        "table sqlite_master may not be indexed",
    );
    assert_temp_absent(&mut db);
}

#[test]
fn create_index_temp_qualified_reserved_index_name_is_reserved() {
    // A `sqlite_`-prefixed INDEX NAME is reserved; the check fires before the target table
    // is resolved, so it errors even with no table `t` anywhere.
    let mut db = mem();
    assert_sql_err(
        &mut db,
        "CREATE INDEX temp.sqlite_ix ON t(a)",
        "object name reserved for internal use: sqlite_ix",
    );
    assert_temp_absent(&mut db);
}

#[test]
fn alter_table_temp_qualified_reserved_name_may_not_be_altered() {
    // `ALTER TABLE temp.sqlite_master …` is rejected as "table sqlite_master may not be
    // altered" (the reserved rule), not "no such table", before temp is materialized —
    // for both ADD COLUMN and RENAME TO.
    let mut db = mem();
    assert_sql_err(
        &mut db,
        "ALTER TABLE temp.sqlite_master ADD COLUMN c INTEGER",
        "table sqlite_master may not be altered",
    );
    assert_sql_err(
        &mut db,
        "ALTER TABLE temp.sqlite_master RENAME TO x",
        "table sqlite_master may not be altered",
    );
    assert_temp_absent(&mut db);
}

#[test]
fn reserved_name_ddl_is_identical_whether_temp_is_live_or_not() {
    // The core contract: a `temp.`-qualified statement returns the SAME result
    // whether or not temp is materialized. Before the fix, `DROP TABLE IF EXISTS
    // temp.sqlite_master` was a silent Ok on a NON-LIVE temp but an error on a LIVE-empty
    // temp — divergence by temp state alone. This asserts the two runs agree with EACH
    // OTHER (a true differential — no shared literal, so it catches EITHER path drifting on
    // its own), and separately pins the exact message (so it also catches BOTH paths
    // drifting together to the same wrong text).
    const STMT: &str = "DROP TABLE IF EXISTS temp.sqlite_master";
    const MSG: &str = "table sqlite_master may not be dropped";

    // Non-live temp (fresh connection): the throwaway-store emulation.
    let mut before = mem();
    let non_live = match before.execute(STMT) {
        Err(Error::Sql(m)) => m,
        other => panic!("non-live temp: expected Err(Sql(..)), got {other:?}"),
    };
    assert_temp_absent(&mut before); // the erroring statement did not materialize temp

    // Live-but-empty temp (materialized by an unrelated temp table): the real catalog path.
    let mut after = mem();
    exec(&mut after, "CREATE TEMP TABLE t(a)");
    let live = match after.execute(STMT) {
        Err(Error::Sql(m)) => m,
        other => panic!("live-empty temp: expected Err(Sql(..)), got {other:?}"),
    };

    // Differential: temp state alone must not change the result...
    assert_eq!(non_live, live, "non-live temp diverges from a live-but-empty temp store");
    // ...and it must be the exact reserved-name error (not some other text both agreed on).
    assert_eq!(non_live, MSG, "reserved-name DROP must be the catalog's reserved message");
}

#[test]
fn reserved_name_ddl_in_transaction_errors_exactly_and_keeps_txn_usable() {
    // The non-live emulation is transaction-agnostic: it runs against a throwaway store and
    // never touches the connection's open transaction. Inside a `BEGIN`, a reserved-name
    // `temp.`-qualified DROP still errors with the EXACT catalog message (the companion peer
    // test in `conformance_temp_tables.rs` only substring-pins the in-transaction slice for a
    // non-reserved name), and — because the scratch store is discarded without disturbing
    // `self` — the transaction stays open and usable: a following write + COMMIT succeeds.
    let mut db = mem();
    exec(&mut db, "BEGIN");
    assert_sql_err(
        &mut db,
        "DROP TABLE IF EXISTS temp.sqlite_master",
        "table sqlite_master may not be dropped",
    );
    exec(&mut db, "CREATE TABLE m(a)");
    exec(&mut db, "INSERT INTO m VALUES (1)");
    exec(&mut db, "COMMIT");
    assert_scalar(&mut db, "SELECT a FROM m", int(1));
    assert_temp_absent(&mut db);
}

// ---------------------------------------------------------------------------
// Regression guards: once temp IS materialized, the same qualified DDL works
// exactly as it did — the fix must not degrade the live-temp path.
// ---------------------------------------------------------------------------

#[test]
fn live_temp_create_index_qualified_succeeds_in_temp() {
    // A valid `temp.`-qualified CREATE INDEX over a temp table succeeds and the index is a
    // temp object: present in `temp.sqlite_master`, absent from `main`'s schema table.
    let mut db = mem();
    exec(&mut db, "CREATE TEMP TABLE tt(a)"); // materializes temp
    exec(&mut db, "INSERT INTO tt VALUES (1), (2)");
    exec(&mut db, "CREATE INDEX temp.ix ON tt(a)");

    assert_rows_unordered(
        &mut db,
        "SELECT name FROM temp.sqlite_master WHERE type = 'index' AND name = 'ix'",
        &[vec![text("ix")]],
    );
    assert_rows_unordered(&mut db, "SELECT name FROM sqlite_master WHERE type = 'index'", &[]);
    // The index is usable.
    assert_rows(&mut db, "SELECT a FROM tt WHERE a = 2", &[vec![int(2)]]);
}

#[test]
fn live_temp_drop_missing_errors_and_if_exists_noops() {
    // Against a materialized (but for these names empty) temp schema, a plain DROP of a
    // missing object errors, while IF EXISTS is a no-op — the same behavior the non-live
    // path emulates.
    let mut db = mem();
    exec(&mut db, "CREATE TEMP TABLE tt(a)"); // materializes temp

    assert_sql_err(&mut db, "DROP TABLE temp.missing", "no such table: missing");
    assert_sql_err(&mut db, "DROP INDEX temp.missing", "no such index: missing");
    exec(&mut db, "DROP TABLE IF EXISTS temp.missing");
    exec(&mut db, "DROP INDEX IF EXISTS temp.missing");
    // The real temp table is untouched by the missing-object drops.
    assert_rows_unordered(
        &mut db,
        "SELECT name FROM temp.sqlite_master WHERE type = 'table'",
        &[vec![text("tt")]],
    );
}

#[test]
fn live_temp_drop_existing_succeeds() {
    // A `temp.`-qualified DROP of an object that DOES exist in temp removes it.
    let mut db = mem();
    exec(&mut db, "CREATE TEMP TABLE tt(a)");
    exec(&mut db, "DROP TABLE temp.tt");
    assert_rows_unordered(
        &mut db,
        "SELECT name FROM temp.sqlite_master WHERE type = 'table'",
        &[],
    );
}

#[test]
fn live_temp_reserved_name_ddl_matches_the_non_live_emulation() {
    // With temp materialized (empty of these reserved names), the SAME reserved-name DDL
    // yields the SAME messages the non-live throwaway-store emulation produces — a
    // cross-check of the emulation against the REAL per-namespace `SchemaCatalog` (an
    // INDEPENDENT source of truth), so a drift in EITHER path (a stale hardcoded string, a
    // changed catalog message) is caught. These literals are the same ones the non-live
    // `*_reserved_*` tests above assert; both paths must agree.
    let mut db = mem();
    exec(&mut db, "CREATE TEMP TABLE tt(a)"); // materialize temp

    assert_sql_err(
        &mut db,
        "DROP TABLE temp.sqlite_master",
        "table sqlite_master may not be dropped",
    );
    assert_sql_err(
        &mut db,
        "DROP TABLE IF EXISTS temp.sqlite_master",
        "table sqlite_master may not be dropped",
    );
    assert_sql_err(
        &mut db,
        "DROP INDEX temp.sqlite_autoindex_tt_1",
        "index associated with UNIQUE or PRIMARY KEY constraint cannot be dropped",
    );
    assert_sql_err(
        &mut db,
        "CREATE INDEX temp.ix ON sqlite_master(a)",
        "table sqlite_master may not be indexed",
    );
    assert_sql_err(
        &mut db,
        "CREATE INDEX temp.sqlite_ix ON tt(a)",
        "object name reserved for internal use: sqlite_ix",
    );
    assert_sql_err(
        &mut db,
        "ALTER TABLE temp.sqlite_master ADD COLUMN c INTEGER",
        "table sqlite_master may not be altered",
    );

    // Every rejected reserved-name DDL left the real temp table untouched.
    assert_rows_unordered(
        &mut db,
        "SELECT name FROM temp.sqlite_master WHERE type = 'table'",
        &[vec![text("tt")]],
    );
}

// ---------------------------------------------------------------------------
// Sanity: main-qualified and attached-qualified DDL are unaffected.
// ---------------------------------------------------------------------------

#[test]
fn main_qualified_ddl_still_works_and_never_touches_temp() {
    // `main.`-qualified DDL routes to the always-live main store; none of it materializes
    // temp.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE main.m(a)");
    exec(&mut db, "INSERT INTO main.m VALUES (1)");
    exec(&mut db, "CREATE INDEX main.mi ON m(a)");
    exec(&mut db, "ALTER TABLE main.m RENAME TO m2");
    assert_rows_unordered(
        &mut db,
        "SELECT name FROM main.sqlite_master WHERE type = 'index'",
        &[vec![text("mi")]],
    );
    exec(&mut db, "DROP TABLE main.m2");
    assert_temp_absent(&mut db);
}

#[test]
fn attached_qualified_ddl_still_works() {
    // After ATTACH, an `aux.`-qualified DDL routes to the live attached store (index 2).
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.a(x)");
    exec(&mut db, "INSERT INTO aux.a VALUES (1)");
    exec(&mut db, "CREATE INDEX aux.ai ON a(x)");
    exec(&mut db, "ALTER TABLE aux.a RENAME TO a2");
    exec(&mut db, "DROP TABLE aux.a2");
    assert_rows_unordered(
        &mut db,
        "SELECT name FROM aux.sqlite_master WHERE type = 'table'",
        &[],
    );
}
