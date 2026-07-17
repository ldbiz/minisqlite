//! Conformance battery: the **schema-introspection PRAGMAs** that read the catalog
//! — `table_info`, `index_list`, `index_info`, `index_xinfo`, `foreign_key_list`.
//!
//! Every expected result is TRANSCRIBED FROM THE SQLITE DOCS in `spec/sqlite-doc/`
//! (`pragma.html`), never from what the engine currently returns — a failing case is
//! the intended signal that the engine diverges from the spec. Assertions are NEVER
//! weakened to make the suite pass.
//!
//! Spec sources (all under `spec/sqlite-doc/pragma.html`):
//!   * #pragma_table_info — one row per normal column; columns `cid`, `name`, `type`
//!     (declared type "if given, else ''"), `notnull`, `dflt_value` (the default
//!     value, NULL when none), `pk` (0, or the 1-based index within the primary key).
//!   * #pragma_index_list — one row per index; columns `seq`, `name`, `unique`
//!     (1/0), `origin` ('c' for CREATE INDEX, 'u' for a UNIQUE constraint, 'pk' for a
//!     PRIMARY KEY constraint), `partial` (1/0).
//!   * #pragma_index_info — one row per key column; columns are the rank within the
//!     index (0 = left-most), the rank within the table (-1 = rowid, -2 = expression),
//!     and the column name (NULL for rowid/expression). The result columns are named
//!     `seqno`, `cid`, `name`.
//!   * #pragma_index_xinfo — like index_info for EVERY column (key columns then the
//!     auxiliary columns that locate the table row), adding `desc` (1 if DESC),
//!     `coll` (the collation), and `key` (1 for a key column, 0 for an auxiliary one).
//!     On a rowid table the sole auxiliary column is the rowid (cid = -1, name = NULL,
//!     key = 0).
//!   * #pragma_foreign_key_list — one row per foreign key created by a REFERENCES
//!     clause.
//!
//! Ordering note: for the multi-index / multi-constraint spec order the docs do not
//! pin a stable `seq` direction, so every `index_list` case here uses a table with a
//! SINGLE index — its `seq` is unambiguously 0 regardless of ordering convention, so
//! the assertion pins the payload without over-pinning an order the spec leaves open.

mod conformance;

use conformance::*;

// ===========================================================================
// PRAGMA table_info — cid / name / type / notnull / dflt_value / pk.
// ===========================================================================

#[test]
fn table_info_integer_primary_key_reports_pk_1() {
    // #pragma_table_info: `pk` is the 1-based index of the column within the primary
    // key. An INTEGER PRIMARY KEY is a single-column key, so its column reports pk=1;
    // it is not implicitly NOT NULL in table_info (notnull=0).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT)");
    assert_rows(
        &mut db,
        "PRAGMA table_info(t)",
        &[
            vec![int(0), text("a"), text("INTEGER"), int(0), null(), int(1)],
            vec![int(1), text("b"), text("TEXT"), int(0), null(), int(0)],
        ],
    );
}

#[test]
fn table_info_column_level_text_pk_reports_pk_1() {
    // #pragma_table_info: a non-integer column-level PRIMARY KEY is likewise a
    // single-column key (pk=1). `b` has no declared type, so its `type` is ''.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a TEXT PRIMARY KEY, b)");
    assert_rows(
        &mut db,
        "PRAGMA table_info(t)",
        &[
            vec![int(0), text("a"), text("TEXT"), int(0), null(), int(1)],
            vec![int(1), text("b"), text(""), int(0), null(), int(0)],
        ],
    );
}

#[test]
fn table_info_notnull_and_default_are_reported() {
    // #pragma_table_info: `notnull` is 1 for a NOT NULL column; `dflt_value` is the
    // default value's SQL text (a bare number renders as itself), NULL when absent.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER NOT NULL DEFAULT 5, b TEXT)");
    assert_rows(
        &mut db,
        "PRAGMA table_info(t)",
        &[
            vec![int(0), text("a"), text("INTEGER"), int(1), text("5"), int(0)],
            vec![int(1), text("b"), text("TEXT"), int(0), null(), int(0)],
        ],
    );
}

#[test]
fn table_info_text_default_keeps_sql_quoting() {
    // #pragma_table_info: a string default is reported with its SQL single-quotes,
    // as stored in the schema (e.g. DEFAULT 'x' -> the text `'x'`).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a TEXT DEFAULT 'x')");
    assert_rows(
        &mut db,
        "PRAGMA table_info(t)",
        &[vec![int(0), text("a"), text("TEXT"), int(0), text("'x'"), int(0)]],
    );
}

#[test]
fn table_info_columns_are_named_per_spec() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    assert_columns(
        &mut db,
        "PRAGMA table_info(t)",
        &["cid", "name", "type", "notnull", "dflt_value", "pk"],
    );
}

#[test]
fn table_info_missing_table_is_empty_with_columns() {
    // #pragma_table_info of an unknown table returns the fixed columns and no rows.
    let mut db = mem();
    assert_columns(
        &mut db,
        "PRAGMA table_info(nope)",
        &["cid", "name", "type", "notnull", "dflt_value", "pk"],
    );
    let qr = try_query(&mut db, "PRAGMA table_info(nope)")
        .unwrap_or_else(|e| panic!("table_info of a missing table must not error: {e:?}"));
    assert_eq!(qr.rows.len(), 0, "table_info of a missing table returns no rows");
}

// ===========================================================================
// PRAGMA index_list — seq / name / unique / origin / partial.
// ===========================================================================

#[test]
fn index_list_columns_are_named_per_spec() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    assert_columns(
        &mut db,
        "PRAGMA index_list(t)",
        &["seq", "name", "unique", "origin", "partial"],
    );
}

#[test]
fn index_list_explicit_index_has_origin_c() {
    // #pragma_index_list: a CREATE INDEX index has origin 'c'; a non-UNIQUE index has
    // unique=0 and (no WHERE clause) partial=0. Single index => seq is 0.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "CREATE INDEX ix ON t(a)");
    assert_rows(
        &mut db,
        "PRAGMA index_list(t)",
        &[vec![int(0), text("ix"), int(0), text("c"), int(0)]],
    );
}

#[test]
fn index_list_unique_explicit_index_reports_unique_1() {
    // #pragma_index_list: a CREATE UNIQUE INDEX still has origin 'c' but unique=1.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "CREATE UNIQUE INDEX uix ON t(a)");
    assert_rows(
        &mut db,
        "PRAGMA index_list(t)",
        &[vec![int(0), text("uix"), int(1), text("c"), int(0)]],
    );
}

#[test]
fn index_list_unique_constraint_has_origin_u() {
    // #pragma_index_list: a UNIQUE constraint auto-creates an index named
    // `sqlite_autoindex_<table>_1` with unique=1 and origin 'u'.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a UNIQUE, b)");
    assert_rows(
        &mut db,
        "PRAGMA index_list(t)",
        &[vec![int(0), text("sqlite_autoindex_t_1"), int(1), text("u"), int(0)]],
    );
}

#[test]
fn index_list_primary_key_constraint_has_origin_pk() {
    // #pragma_index_list: a non-integer PRIMARY KEY auto-creates an index with
    // origin 'pk'. (An INTEGER PRIMARY KEY is the rowid alias and creates no index.)
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a TEXT PRIMARY KEY, b)");
    assert_rows(
        &mut db,
        "PRAGMA index_list(t)",
        &[vec![int(0), text("sqlite_autoindex_t_1"), int(1), text("pk"), int(0)]],
    );
}

#[test]
fn index_list_no_indexes_is_empty() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    let qr = try_query(&mut db, "PRAGMA index_list(t)")
        .unwrap_or_else(|e| panic!("index_list must not error: {e:?}"));
    assert_eq!(qr.rows.len(), 0, "a table with no indexes lists none");
}

// ===========================================================================
// PRAGMA index_info / index_xinfo — the columns of an index.
// ===========================================================================

#[test]
fn index_info_columns_are_named_and_cid_is_table_ordinal() {
    // #pragma_index_info: one row per key column — seqno (rank in index), cid (rank
    // within the table), name. For `t(a,b,c)` indexed on `(b,c)`: b is table column 1,
    // c is table column 2.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b, c)");
    exec(&mut db, "CREATE INDEX ix ON t(b, c)");
    assert_columns(&mut db, "PRAGMA index_info(ix)", &["seqno", "cid", "name"]);
    assert_rows(
        &mut db,
        "PRAGMA index_info(ix)",
        &[
            vec![int(0), int(1), text("b")],
            vec![int(1), int(2), text("c")],
        ],
    );
}

#[test]
fn index_xinfo_lists_key_columns_then_rowid_auxiliary() {
    // #pragma_index_xinfo: key columns (key=1) then the rowid auxiliary (cid=-1,
    // name=NULL, key=0) on a rowid table. `desc`/`coll` are read per key column from
    // the index metadata; for this plain (no COLLATE / no DESC) index they compute to
    // desc=0 and coll='BINARY', and the rowid auxiliary is always desc=0/coll='BINARY'.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "CREATE INDEX ix ON t(a)");
    assert_columns(
        &mut db,
        "PRAGMA index_xinfo(ix)",
        &["seqno", "cid", "name", "desc", "coll", "key"],
    );
    assert_rows(
        &mut db,
        "PRAGMA index_xinfo(ix)",
        &[
            vec![int(0), int(0), text("a"), int(0), text("BINARY"), int(1)],
            vec![int(1), int(-1), null(), int(0), text("BINARY"), int(0)],
        ],
    );
}

#[test]
fn index_xinfo_desc_key_column_reports_desc_1() {
    // #pragma_index_xinfo: a DESC key column reports desc=1 (previously hardcoded 0, a
    // byte-for-byte divergence from real sqlite). `desc` is the 4th output column.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "CREATE INDEX ix ON t(a DESC)");
    assert_rows(
        &mut db,
        "PRAGMA index_xinfo(ix)",
        &[
            vec![int(0), int(0), text("a"), int(1), text("BINARY"), int(1)],
            vec![int(1), int(-1), null(), int(0), text("BINARY"), int(0)],
        ],
    );
}

#[test]
fn index_xinfo_explicit_collate_reports_collation_name() {
    // #pragma_index_xinfo: an explicit per-key `COLLATE NOCASE` reports coll='NOCASE'
    // (previously hardcoded 'BINARY'). `coll` is the 5th output column; desc stays 0.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "CREATE INDEX ix ON t(a COLLATE NOCASE)");
    assert_rows(
        &mut db,
        "PRAGMA index_xinfo(ix)",
        &[
            vec![int(0), int(0), text("a"), int(0), text("NOCASE"), int(1)],
            vec![int(1), int(-1), null(), int(0), text("BINARY"), int(0)],
        ],
    );
}

#[test]
fn index_xinfo_inherits_column_declared_collation() {
    // #pragma_index_xinfo: with no explicit COLLATE on the index key, coll inherits the
    // target table column's declared COLLATE (datatype3.html §7.1). Column `a` is
    // declared COLLATE NOCASE, so an index on the bare `a` reports coll='NOCASE'.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a TEXT COLLATE NOCASE, b)");
    exec(&mut db, "CREATE INDEX ix ON t(a)");
    assert_rows(
        &mut db,
        "PRAGMA index_xinfo(ix)",
        &[
            vec![int(0), int(0), text("a"), int(0), text("NOCASE"), int(1)],
            vec![int(1), int(-1), null(), int(0), text("BINARY"), int(0)],
        ],
    );
}

#[test]
fn index_xinfo_explicit_collate_preserves_written_case() {
    // #pragma_index_xinfo: `coll` echoes the COLLATE token VERBATIM — real sqlite reports
    // the collation name in the exact case it was written, NOT a canonicalized upper-case
    // form. A lower-case `COLLATE nocase` reports coll='nocase', and a lower-case `COLLATE
    // rtrim` reports coll='rtrim'. (An upper-case token round-trips upper-case too — see
    // index_xinfo_explicit_collate_reports_collation_name — so both cases prove the name is
    // passed through unchanged rather than normalized.)
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "CREATE INDEX ix_lower ON t(a COLLATE nocase)");
    assert_rows(
        &mut db,
        "PRAGMA index_xinfo(ix_lower)",
        &[
            vec![int(0), int(0), text("a"), int(0), text("nocase"), int(1)],
            vec![int(1), int(-1), null(), int(0), text("BINARY"), int(0)],
        ],
    );
    exec(&mut db, "CREATE INDEX ix_rtrim ON t(b COLLATE rtrim)");
    assert_rows(
        &mut db,
        "PRAGMA index_xinfo(ix_rtrim)",
        &[
            vec![int(0), int(1), text("b"), int(0), text("rtrim"), int(1)],
            vec![int(1), int(-1), null(), int(0), text("BINARY"), int(0)],
        ],
    );
}

#[test]
fn index_xinfo_inherits_column_collation_verbatim_case() {
    // #pragma_index_xinfo: an inherited collation is reported VERBATIM as well. Column `a`
    // is declared `COLLATE nOcAsE` (mixed case); an index on the bare `a` inherits it and
    // reports coll='nOcAsE' exactly, not a normalized 'NOCASE'. This pins that the whole
    // inheritance chain preserves case, not just the explicit-override path.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a TEXT COLLATE nOcAsE, b)");
    exec(&mut db, "CREATE INDEX ix ON t(a)");
    assert_rows(
        &mut db,
        "PRAGMA index_xinfo(ix)",
        &[
            vec![int(0), int(0), text("a"), int(0), text("nOcAsE"), int(1)],
            vec![int(1), int(-1), null(), int(0), text("BINARY"), int(0)],
        ],
    );
}

#[test]
fn index_xinfo_plain_key_column_stays_asc_binary() {
    // #pragma_index_xinfo regression guard: the desc/coll fix must NOT over-apply. A
    // plain key column (no COLLATE, no DESC) still reports desc=0, coll='BINARY'. The
    // index is on the SECOND column, pinning that cid tracks the right table ordinal (1)
    // while desc/coll stay the baseline.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "CREATE INDEX ix ON t(b)");
    assert_rows(
        &mut db,
        "PRAGMA index_xinfo(ix)",
        &[
            vec![int(0), int(1), text("b"), int(0), text("BINARY"), int(1)],
            vec![int(1), int(-1), null(), int(0), text("BINARY"), int(0)],
        ],
    );
}

#[test]
fn index_xinfo_multi_column_mixes_desc_and_collation_per_column() {
    // #pragma_index_xinfo: each key column reports its OWN desc/coll. For `(a, b DESC,
    // c COLLATE NOCASE)`: `a` plain (desc=0/BINARY), `b` DESC (desc=1/BINARY), `c`
    // COLLATE NOCASE (desc=0/NOCASE). The trailing rowid auxiliary is desc=0/BINARY, key=0.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b, c)");
    exec(&mut db, "CREATE INDEX ix ON t(a, b DESC, c COLLATE NOCASE)");
    assert_rows(
        &mut db,
        "PRAGMA index_xinfo(ix)",
        &[
            vec![int(0), int(0), text("a"), int(0), text("BINARY"), int(1)],
            vec![int(1), int(1), text("b"), int(1), text("BINARY"), int(1)],
            vec![int(2), int(2), text("c"), int(0), text("NOCASE"), int(1)],
            vec![int(3), int(-1), null(), int(0), text("BINARY"), int(0)],
        ],
    );
}

#[test]
fn index_xinfo_single_column_combines_collate_and_desc() {
    // #pragma_index_xinfo: one key column may carry BOTH an explicit COLLATE and DESC:
    // `a COLLATE NOCASE DESC` reports desc=1 AND coll='NOCASE' together.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "CREATE INDEX ix ON t(a COLLATE NOCASE DESC)");
    assert_rows(
        &mut db,
        "PRAGMA index_xinfo(ix)",
        &[
            vec![int(0), int(0), text("a"), int(1), text("NOCASE"), int(1)],
            vec![int(1), int(-1), null(), int(0), text("BINARY"), int(0)],
        ],
    );
}

#[test]
fn index_xinfo_auto_index_inherits_column_collation() {
    // #pragma_index_xinfo over a UNIQUE-constraint auto-index (`sqlite_autoindex_t_1`):
    // the constraint carries no explicit COLLATE, so the key column inherits the column's
    // declared COLLATE NOCASE. Exercises the auto-index path, not just CREATE INDEX.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a TEXT COLLATE NOCASE UNIQUE, b)");
    assert_rows(
        &mut db,
        "PRAGMA index_xinfo(sqlite_autoindex_t_1)",
        &[
            vec![int(0), int(0), text("a"), int(0), text("NOCASE"), int(1)],
            vec![int(1), int(-1), null(), int(0), text("BINARY"), int(0)],
        ],
    );
}

// ---------------------------------------------------------------------------
// PRAGMA index_xinfo — a WITHOUT ROWID table's auxiliary columns are its PRIMARY
// KEY columns (there is no rowid), listed after the key columns with key=0.
//
// The secondary index reachable on a WITHOUT ROWID table here is a UNIQUE-constraint
// auto-index: the catalog registers it at CREATE TABLE time and index_xinfo is a pure
// catalog read, so it introspects without any INSERT. (A `CREATE INDEX` on a WR base
// table is a separate, deliberate engine gap; the aux-column derivation itself is also
// unit-tested over a CREATE INDEX shape in minisqlite-catalog/src/introspect.rs.)
// ---------------------------------------------------------------------------

#[test]
fn index_xinfo_without_rowid_appends_auxiliary_pk_columns() {
    // #pragma_index_xinfo: a WITHOUT ROWID table has no rowid, so a secondary index's
    // auxiliary columns are the table's PRIMARY KEY columns needed to locate the row
    // (withoutrowid.html; the b-tree record is de-duplicated PK columns then data). For
    // `w(a, b, c, PRIMARY KEY(a, b), UNIQUE(c)) WITHOUT ROWID`, the UNIQUE(c) auto-index
    // `sqlite_autoindex_w_2` (the PK reserves ordinal 1 but owns no index) lists the key
    // column c (key=1), then the PK columns a, b appended as auxiliary columns (key=0), in
    // PRIMARY KEY order.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE w(a, b, c, PRIMARY KEY(a, b), UNIQUE(c)) WITHOUT ROWID");
    assert_rows(
        &mut db,
        "PRAGMA index_xinfo(sqlite_autoindex_w_2)",
        &[
            vec![int(0), int(2), text("c"), int(0), text("BINARY"), int(1)],
            vec![int(1), int(0), text("a"), int(0), text("BINARY"), int(0)],
            vec![int(2), int(1), text("b"), int(0), text("BINARY"), int(0)],
        ],
    );
}

#[test]
fn index_xinfo_without_rowid_deduplicates_pk_column_already_in_key() {
    // #pragma_index_xinfo: a PK column already present as an index key column is NOT repeated
    // as an auxiliary column (withoutrowid.html, "de-duplicated PRIMARY KEY columns"). The
    // UNIQUE(c, a) auto-index keeps `a` as a key column (key=1); only the remaining PK column
    // `b` is appended as an auxiliary column (key=0).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE w(a, b, c, PRIMARY KEY(a, b), UNIQUE(c, a)) WITHOUT ROWID");
    assert_rows(
        &mut db,
        "PRAGMA index_xinfo(sqlite_autoindex_w_2)",
        &[
            vec![int(0), int(2), text("c"), int(0), text("BINARY"), int(1)],
            vec![int(1), int(0), text("a"), int(0), text("BINARY"), int(1)],
            vec![int(2), int(1), text("b"), int(0), text("BINARY"), int(0)],
        ],
    );
}

#[test]
fn index_xinfo_without_rowid_aux_columns_carry_pk_desc_and_collation() {
    // #pragma_index_xinfo: each appended auxiliary PK column reports the PRIMARY KEY's OWN
    // per-column desc/coll, not a flat default. `PRIMARY KEY(a DESC, b COLLATE NOCASE)` yields
    // aux `a` desc=1/BINARY and aux `b` desc=0/NOCASE, after the plain UNIQUE key column c.
    let mut db = mem();
    exec(
        &mut db,
        "CREATE TABLE w(a, b, c, PRIMARY KEY(a DESC, b COLLATE NOCASE), UNIQUE(c)) WITHOUT ROWID",
    );
    assert_rows(
        &mut db,
        "PRAGMA index_xinfo(sqlite_autoindex_w_2)",
        &[
            vec![int(0), int(2), text("c"), int(0), text("BINARY"), int(1)],
            vec![int(1), int(0), text("a"), int(1), text("BINARY"), int(0)],
            vec![int(2), int(1), text("b"), int(0), text("NOCASE"), int(0)],
        ],
    );
}

#[test]
fn index_xinfo_without_rowid_integer_pk_aux_column() {
    // #pragma_index_xinfo: a WITHOUT ROWID table keyed by a single INTEGER PRIMARY KEY (which
    // reserves NO auto-index — the table b-tree IS the key) still appends that PK column as an
    // auxiliary column. `w(k INTEGER PRIMARY KEY, v UNIQUE) WITHOUT ROWID`: the UNIQUE(v)
    // auto-index `sqlite_autoindex_w_1` lists key column v, then aux PK k (desc=0,
    // coll='BINARY', key=0) — the no-spec fallback path (the integer PK owns no spec).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE w(k INTEGER PRIMARY KEY, v UNIQUE) WITHOUT ROWID");
    assert_rows(
        &mut db,
        "PRAGMA index_xinfo(sqlite_autoindex_w_1)",
        &[
            vec![int(0), int(1), text("v"), int(0), text("BINARY"), int(1)],
            vec![int(1), int(0), text("k"), int(0), text("BINARY"), int(0)],
        ],
    );
}

// ===========================================================================
// PRAGMA foreign_key_list — documented column shape; no FK metadata yet.
// ===========================================================================

#[test]
fn foreign_key_list_has_documented_columns_and_no_rows_without_fks() {
    // #pragma_foreign_key_list: a table with no REFERENCES clause has no foreign
    // keys, so the pragma returns the fixed columns and zero rows.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    assert_columns(
        &mut db,
        "PRAGMA foreign_key_list(t)",
        &["id", "seq", "table", "from", "to", "on_update", "on_delete", "match"],
    );
    let qr = try_query(&mut db, "PRAGMA foreign_key_list(t)")
        .unwrap_or_else(|e| panic!("foreign_key_list must not error: {e:?}"));
    assert_eq!(qr.rows.len(), 0, "a table with no foreign keys lists none");
}

// ===========================================================================
// Case-folding: the object name argument is matched case-insensitively.
// ===========================================================================

#[test]
fn introspection_pragmas_fold_the_object_name() {
    // SQLite identifiers are case-insensitive over ASCII, so `table_info(T)` finds a
    // table created as `t`, and `index_info(IX)` finds index `ix`.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "CREATE INDEX ix ON t(a)");
    let ti = try_query(&mut db, "PRAGMA table_info(T)")
        .unwrap_or_else(|e| panic!("table_info(T) must fold the name: {e:?}"));
    assert_eq!(ti.rows.len(), 2, "table_info(T) resolves table t");
    let ii = try_query(&mut db, "PRAGMA index_info(IX)")
        .unwrap_or_else(|e| panic!("index_info(IX) must fold the name: {e:?}"));
    assert_eq!(ii.rows.len(), 1, "index_info(IX) resolves index ix");
}
