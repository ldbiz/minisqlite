//! Conformance battery: **schema introspection via the schema table**
//! (`sqlite_master` / `sqlite_schema`) and the `PRAGMA table_info` probe.
//!
//! Every expected result here is TRANSCRIBED FROM THE SQLITE DOCS in
//! `spec/sqlite-doc/`, never from what the engine currently returns — a failing
//! case is the intended signal that the engine diverges from the spec. Assertions
//! are NEVER weakened to make the suite pass; a real divergence is left as a genuine
//! failing assertion. Only a case that HANGS/aborts may be `#[ignore]`-d (none here).
//!
//! Spec sources (all under `spec/sqlite-doc/`):
//!   * `schematab.html` §1 "Introduction" — every database has one schema table
//!     `sqlite_schema` shaped exactly `(type text, name text, tbl_name text,
//!     rootpage integer, sql text)`, with one row per table/index/view/trigger
//!     and **no entry for the schema table itself**.
//!   * `schematab.html` §2 "Alternative Names" — the schema table is also
//!     reachable as `sqlite_master` (works anywhere), plus the TEMP-only aliases.
//!   * `schematab.html` §3 "Interpretation Of The Schema Table":
//!       - `type` is one of `'table'`, `'index'`, `'view'`, `'trigger'`.
//!       - `name` holds the object's name.
//!       - `tbl_name`: for a table/view it "is a copy of the name column"; for an
//!         index it "is the name of the table that is indexed"; for a trigger it
//!         is the table/view that fires it.
//!       - `rootpage` "stores the page number of the root b-tree page for tables
//!         and indexes" (0 or NULL for views/triggers/virtual tables).
//!       - `sql` stores the CREATE statement text (normalized), "usually a copy
//!         of the original statement", and "modified by subsequent ALTER TABLE
//!         statements"; NULL for the internal auto-indexes.
//!   * `schematab.html` §4 — the schema table's content changes as DDL runs
//!     (CREATE adds a row, DROP removes it).
//!   * `fileformat2.html` §2.6 "Storage Of The SQL Database Schema" — the schema
//!     b-tree is page 1 (so user tables/indexes root at a page > 1), the same
//!     five-column shape, no self-row, and §2.6.1 the alternative names.
//!   * `pragma.html` #pragma_table_info — "returns one row for each normal
//!     column"; the result columns are `cid`, `name`, `type`, `notnull`,
//!     `dflt_value`, `pk`. (PROBE.)
//!
//! The CREATE statement text used here is already in the schema table's
//! *normalized* form (a single `CREATE TABLE`/`CREATE INDEX` in upper case, one
//! space after the first two keywords, no leading space, no schema qualifier, no
//! TEMP), so the documented normalization is a no-op and the stored `sql` must
//! equal the input byte-for-byte (`schematab.html` §3 normalization rules).
//!
//! Each case is its own small `#[test]` so an unsupported feature fails exactly
//! that case and never masks the rest.

mod conformance;

use conformance::*;
// `Value` (no `PartialEq`) and `Connection` are needed for the local `rootpage`
// helper, which pattern-matches the storage class directly; every *value*
// comparison still goes through the harness.
use minisqlite::{Connection, Value};

// ---------------------------------------------------------------------------
// Local helper: rootpage is a positive b-tree page number, class Integer.
// The exact page is NOT fixed by the spec, so we assert only class + `> 1`
// (page 1 is the schema table). Kept local (not in the shared harness) because
// it is specific to this file's schema-introspection checks.
// ---------------------------------------------------------------------------

fn assert_rootpage_is_btree_page(db: &mut Connection, object_name: &str) {
    // `object_name` is only ever a literal identifier from the tests below
    // ("t", "ix"), so interpolating it into the SQL is safe here. Do not feed
    // this helper an untrusted name: one containing a `'` would break the
    // quoting and mis-query rather than fail loudly.
    let sql = format!("SELECT rootpage FROM sqlite_master WHERE name = '{object_name}'");
    let qr = try_query(db, &sql)
        .unwrap_or_else(|e| panic!("query failed\n  sql: {sql}\n  error: {e:?}"));
    assert_eq!(
        qr.rows.len(),
        1,
        "expected exactly one sqlite_master row for {object_name:?}\n  sql: {sql}\n  rows: {:?}",
        qr.rows
    );
    match &qr.rows[0][0] {
        // schematab.html §3 / fileformat2.html §2.6: tables/indexes carry the page
        // number of their root b-tree page; page 1 is the schema table, so a user
        // object roots at page > 1.
        Value::Integer(page) => assert!(
            *page > 1,
            "rootpage for {object_name:?} must be a b-tree page number > 1 (page 1 is the schema \
             table); got {page}"
        ),
        other => panic!(
            "rootpage for {object_name:?} must be Integer-class per schematab.html §3; got {other:?}"
        ),
    }
}

// ===========================================================================
// Column shape of the schema table (schematab.html §1; fileformat2.html §2.6).
// ===========================================================================

#[test]
fn schema_table_select_star_has_five_columns_in_order() {
    // schematab.html §1: the schema table is shaped
    // `(type, name, tbl_name, rootpage, sql)` in that order. `SELECT *` expands
    // to those columns in declaration order.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
    assert_columns(
        &mut db,
        "SELECT * FROM sqlite_master",
        &["type", "name", "tbl_name", "rootpage", "sql"],
    );
}

#[test]
fn sqlite_schema_alias_select_star_has_same_columns() {
    // §2 / fileformat2 §2.6.1: `sqlite_schema` is the modern name for the very
    // same table, so its column shape is identical.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
    assert_columns(
        &mut db,
        "SELECT * FROM sqlite_schema",
        &["type", "name", "tbl_name", "rootpage", "sql"],
    );
}

// ===========================================================================
// A created TABLE's row: type / name / tbl_name / verbatim sql (schematab §3).
// ===========================================================================

#[test]
fn created_table_row_type_name_tblname_sql_verbatim() {
    // §3: a CREATE TABLE adds one row with type='table', name and tbl_name equal
    // to the table name, and `sql` the (normalized) CREATE text — here identical
    // to the input, which is already normalized.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
    assert_rows(
        &mut db,
        "SELECT type, name, tbl_name, sql FROM sqlite_master WHERE type = 'table' AND name = 't'",
        &[vec![
            text("table"),
            text("t"),
            text("t"),
            text("CREATE TABLE t(a INTEGER, b TEXT)"),
        ]],
    );
}

#[test]
fn created_table_type_column_is_table() {
    // §3: `type` for an ordinary table is the text string 'table'.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
    assert_scalar(
        &mut db,
        "SELECT type FROM sqlite_master WHERE name = 't'",
        text("table"),
    );
}

#[test]
fn created_table_tbl_name_is_copy_of_name() {
    // §3: "For a table or view, the tbl_name column is a copy of the name column."
    let mut db = mem();
    exec(&mut db, "CREATE TABLE things(a INTEGER, b TEXT)");
    assert_rows(
        &mut db,
        "SELECT name, tbl_name FROM sqlite_master WHERE type = 'table' AND name = 'things'",
        &[vec![text("things"), text("things")]],
    );
}

// ===========================================================================
// `sqlite_schema` and `sqlite_master` are the same table (schematab §2;
// fileformat2 §2.6.1).
// ===========================================================================

#[test]
fn sqlite_schema_returns_same_table_row_as_sqlite_master() {
    // §2: the query against `sqlite_schema` returns exactly the row seen through
    // the legacy `sqlite_master` name.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
    assert_rows(
        &mut db,
        "SELECT type, name, tbl_name, sql FROM sqlite_schema WHERE type = 'table' AND name = 't'",
        &[vec![
            text("table"),
            text("t"),
            text("t"),
            text("CREATE TABLE t(a INTEGER, b TEXT)"),
        ]],
    );
}

#[test]
fn sqlite_master_and_sqlite_schema_agree_on_table_count() {
    // §2: the two names address one table, so a count through either is the same.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t1(a)");
    exec(&mut db, "CREATE TABLE t2(a)");
    assert_scalar(
        &mut db,
        "SELECT count(*) FROM sqlite_master WHERE type = 'table'",
        int(2),
    );
    assert_scalar(
        &mut db,
        "SELECT count(*) FROM sqlite_schema WHERE type = 'table'",
        int(2),
    );
}

// ===========================================================================
// A created INDEX's row (schematab §3): type='index', tbl_name = indexed table,
// verbatim sql.
// ===========================================================================

#[test]
fn created_index_row_type_name_tblname() {
    // §3: an index's row has type='index', name=index name, and tbl_name = "the
    // name of the table that is indexed".
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
    exec(&mut db, "CREATE INDEX ix ON t(a)");
    assert_rows(
        &mut db,
        "SELECT type, name, tbl_name FROM sqlite_master WHERE type = 'index' AND name = 'ix'",
        &[vec![text("index"), text("ix"), text("t")]],
    );
}

#[test]
fn created_index_sql_is_verbatim() {
    // §3: the index row's `sql` is the (normalized) CREATE INDEX text — identical
    // to the already-normalized input here.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
    exec(&mut db, "CREATE INDEX ix ON t(a)");
    assert_scalar(
        &mut db,
        "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = 'ix'",
        text("CREATE INDEX ix ON t(a)"),
    );
}

// ===========================================================================
// Internal auto-indexes for UNIQUE / PRIMARY KEY (schematab.html §3;
// fileformat2.html §2.6.2). A UNIQUE or PRIMARY KEY constraint on an ordinary
// table creates an internal index named `sqlite_autoindex_<table>_<N>` (N from
// 1) whose `sql` is NULL; an INTEGER PRIMARY KEY (the rowid alias) creates none.
// ===========================================================================

#[test]
fn unique_constraint_creates_autoindex_row_with_null_sql() {
    // §3: "UNIQUE and PRIMARY KEY constraints on tables cause SQLite to create
    // internal indexes with names of the form 'sqlite_autoindex_TABLE_N' where
    // TABLE is replaced by the name of the table ... and N is an integer
    // beginning with 1", and (§3, the `sql` column) "The sqlite_schema.sql is
    // NULL for the internal indexes that are automatically created by UNIQUE or
    // PRIMARY KEY constraints."
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a UNIQUE)");
    assert_rows(
        &mut db,
        "SELECT type, name, tbl_name, sql FROM sqlite_master WHERE type = 'index' ORDER BY name",
        &[vec![text("index"), text("sqlite_autoindex_t_1"), text("t"), null()]],
    );
}

#[test]
fn integer_primary_key_creates_no_autoindex_row() {
    // §3: "The 'sqlite_autoindex_TABLE_N' name is never allocated for an INTEGER
    // PRIMARY KEY" — it aliases the rowid, so no separate index row is created.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, b)");
    assert_scalar(
        &mut db,
        "SELECT count(*) FROM sqlite_master WHERE type = 'index'",
        int(0),
    );
}

// ===========================================================================
// Multiple objects list deterministically under ORDER BY (schematab §1).
// ===========================================================================

#[test]
fn multiple_tables_listed_in_name_order() {
    // §1: every table has a row; ORDER BY name gives a deterministic listing.
    // Created out of alphabetical order to make the ordering meaningful.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t2(a)");
    exec(&mut db, "CREATE TABLE t1(a)");
    exec(&mut db, "CREATE TABLE t3(a)");
    assert_rows(
        &mut db,
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name",
        &[vec![text("t1")], vec![text("t2")], vec![text("t3")]],
    );
}

// ===========================================================================
// DROP removes the schema row (schematab §4; lang_droptable / lang_dropindex).
// ===========================================================================

#[test]
fn drop_table_removes_its_schema_row() {
    // §4: dropping the object removes its row from the schema table.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
    exec(&mut db, "DROP TABLE t");
    assert_scalar(
        &mut db,
        "SELECT count(*) FROM sqlite_master WHERE name = 't'",
        int(0),
    );
}

#[test]
fn drop_index_removes_its_schema_row_but_keeps_the_table() {
    // §4: an index's row disappears once the index is dropped, while the row for
    // the table it indexed is left intact.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
    exec(&mut db, "CREATE INDEX ix ON t(a)");
    exec(&mut db, "DROP INDEX ix");
    assert_scalar(
        &mut db,
        "SELECT count(*) FROM sqlite_master WHERE name = 'ix'",
        int(0),
    );
    assert_scalar(
        &mut db,
        "SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name = 't'",
        int(1),
    );
}

// ===========================================================================
// rootpage is a positive b-tree page number (schematab §3; fileformat2 §2.6).
// ===========================================================================

#[test]
fn table_rootpage_is_positive_btree_page() {
    // §3: a table's rootpage is the page number of its root b-tree page. It is
    // Integer-class and > 1 (page 1 is the schema table). The exact page number
    // is NOT fixed by the spec, so it is never hardcoded.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
    assert_rootpage_is_btree_page(&mut db, "t");
}

#[test]
fn index_rootpage_is_positive_btree_page() {
    // §3: an index has its own root b-tree page, likewise Integer-class and > 1.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
    exec(&mut db, "CREATE INDEX ix ON t(a)");
    assert_rootpage_is_btree_page(&mut db, "ix");
}

// ===========================================================================
// The schema table does not list itself (schematab §1; fileformat2 §2.6).
// ===========================================================================

#[test]
fn fresh_schema_table_is_empty() {
    // §1: a fresh database has no user objects and there is no self-row, so the
    // schema table is empty.
    let mut db = mem();
    assert_scalar(&mut db, "SELECT count(*) FROM sqlite_master", int(0));
}

#[test]
fn schema_table_has_no_row_for_itself() {
    // §1 / fileformat2 §2.6: "there is no entry for the sqlite_schema table
    // itself" — under either name — even after user objects exist.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
    assert_scalar(
        &mut db,
        "SELECT count(*) FROM sqlite_master \
         WHERE name = 'sqlite_master' OR name = 'sqlite_schema'",
        int(0),
    );
}

// ===========================================================================
// Total object count excludes the self-row (schematab §1).
// ===========================================================================

#[test]
fn total_object_count_excludes_self() {
    // §1: one row per object and no self-row, so a table plus an index give
    // exactly two rows (neither table has a UNIQUE/PRIMARY KEY, so there are no
    // internal auto-indexes to count).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
    exec(&mut db, "CREATE INDEX ix ON t(a)");
    assert_scalar(&mut db, "SELECT count(*) FROM sqlite_master", int(2));
}

// ===========================================================================
// ALTER TABLE RENAME updates the schema row (schematab §3: sql is "modified by
// subsequent ALTER TABLE statements"; lang_altertable.html).
// ===========================================================================

#[test]
fn alter_rename_new_name_present() {
    // §3: after RENAME the object appears under the new name.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE a(x)");
    exec(&mut db, "ALTER TABLE a RENAME TO b");
    assert_rows(
        &mut db,
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'b'",
        &[vec![text("b")]],
    );
}

#[test]
fn alter_rename_old_name_absent() {
    // §3: the old name no longer has a row after the rename.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE a(x)");
    exec(&mut db, "ALTER TABLE a RENAME TO b");
    assert_scalar(
        &mut db,
        "SELECT count(*) FROM sqlite_master WHERE name = 'a'",
        int(0),
    );
}

#[test]
fn alter_rename_updates_sql_to_new_name() {
    // §3: the stored `sql` is "modified by subsequent ALTER TABLE statements", so
    // it must recreate the object under its NEW name. The spec does not pin the
    // exact quoting/normalization of the rewritten text, so this asserts the
    // spec-true, quoting-agnostic property: the sql names the new table and no
    // longer names the old one. Distinctive names (`alpha`/`beta`) keep the
    // substring test robust (neither appears incidentally in CREATE/TABLE/etc.).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE alpha(x)");
    exec(&mut db, "ALTER TABLE alpha RENAME TO beta");
    let qr = try_query(
        &mut db,
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'beta'",
    )
    .unwrap_or_else(|e| panic!("query failed while reading renamed sql: {e:?}"));
    assert_eq!(
        qr.rows.len(),
        1,
        "expected exactly one schema row for the renamed table `beta`; got {:?}",
        qr.rows
    );
    match &qr.rows[0][0] {
        Value::Text(s) => {
            assert!(
                s.contains("beta"),
                "renamed table's sql must name the new table `beta`; got {s:?}"
            );
            assert!(
                !s.contains("alpha"),
                "renamed table's sql must no longer name the old table `alpha`; got {s:?}"
            );
        }
        other => panic!("the `sql` column of a table row must be Text; got {other:?}"),
    }
}

// ===========================================================================
// Filtering / counting works like any ordinary table (schematab §1).
// ===========================================================================

#[test]
fn count_tables_equals_number_created() {
    // §1: one row per table, so a filtered count equals the number created.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t1(a)");
    exec(&mut db, "CREATE TABLE t2(a)");
    exec(&mut db, "CREATE TABLE t3(a)");
    assert_scalar(
        &mut db,
        "SELECT count(*) FROM sqlite_master WHERE type = 'table'",
        int(3),
    );
}

#[test]
fn count_indexes_equals_number_created() {
    // §1: likewise for indexes (explicit `CREATE INDEX`es only; no constraints
    // here means no internal auto-indexes).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b, c)");
    exec(&mut db, "CREATE INDEX ix_a ON t(a)");
    exec(&mut db, "CREATE INDEX ix_b ON t(b)");
    assert_scalar(
        &mut db,
        "SELECT count(*) FROM sqlite_master WHERE type = 'index'",
        int(2),
    );
}

// ===========================================================================
// PROBE — views and triggers also get schema rows (schematab §3). These pin
// documented behavior on surface the engine may not implement yet; a failure is
// left as a genuine failing assertion, never a reason to weaken it.
// ===========================================================================

#[test]
fn probe_view_appears_in_schema_with_type_view() {
    // §3: a view has type='view' and, like a table, tbl_name = name.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
    exec(&mut db, "CREATE VIEW v AS SELECT a FROM t");
    assert_rows(
        &mut db,
        "SELECT type, name, tbl_name FROM sqlite_master WHERE type = 'view' AND name = 'v'",
        &[vec![text("view"), text("v"), text("v")]],
    );
}

#[test]
fn probe_trigger_appears_in_schema_with_type_trigger() {
    // §3: a trigger has type='trigger' and tbl_name = the table that fires it.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
    exec(
        &mut db,
        "CREATE TRIGGER trg AFTER INSERT ON t BEGIN SELECT 1; END",
    );
    assert_rows(
        &mut db,
        "SELECT type, name, tbl_name FROM sqlite_master WHERE type = 'trigger' AND name = 'trg'",
        &[vec![text("trigger"), text("trg"), text("t")]],
    );
}

// ===========================================================================
// PROBE — PRAGMA table_info (pragma.html #pragma_table_info). Documented to
// return one row per column with columns cid,name,type,notnull,dflt_value,pk.
// The engine honors only PRAGMA user_version today, so these are expected to
// fail (empty result) until table_info lands — left failing, not weakened.
// ===========================================================================

#[test]
fn probe_pragma_table_info_columns() {
    // #pragma_table_info: the result set columns are cid, name, type, notnull,
    // dflt_value, pk (in that order).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
    assert_columns(
        &mut db,
        "PRAGMA table_info(t)",
        &["cid", "name", "type", "notnull", "dflt_value", "pk"],
    );
}

#[test]
fn probe_pragma_table_info_one_row_per_column() {
    // #pragma_table_info: "returns one row for each normal column" — two here.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
    let qr = try_query(&mut db, "PRAGMA table_info(t)")
        .unwrap_or_else(|e| panic!("PRAGMA table_info(t) errored: {e:?}"));
    assert_eq!(
        qr.rows.len(),
        2,
        "PRAGMA table_info(t) must return one row per column (2 for `t(a, b)`); got {:?}",
        qr.rows
    );
}

#[test]
fn probe_pragma_table_info_row_values() {
    // #pragma_table_info: each row is (cid, name, type, notnull, dflt_value, pk).
    // For `t(a INTEGER, b TEXT)` with no NOT NULL / DEFAULT / PRIMARY KEY: `cid`
    // is the 0-based rank, `type` the declared type, `notnull`=0 (column may be
    // NULL), `dflt_value`=NULL (no default), `pk`=0 (not part of a primary key).
    // Rows come in `cid` (declaration) order — PRAGMA output can't take an
    // ORDER BY, but that order is the documented, deterministic one. This pins
    // the row *content*, so a future impl returning the right count but wrong
    // cells still fails.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
    assert_rows(
        &mut db,
        "PRAGMA table_info(t)",
        &[
            vec![int(0), text("a"), text("INTEGER"), int(0), null(), int(0)],
            vec![int(1), text("b"), text("TEXT"), int(0), null(), int(0)],
        ],
    );
}
