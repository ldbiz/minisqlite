//! Conformance battery: the schema-introspection PRAGMAs are **NAMESPACE-AWARE**
//! (`main` / `temp` / an `ATTACH`-ed database).
//!
//! The six pragmas — `table_info`, `table_xinfo`, `index_list`, `index_info`,
//! `index_xinfo`, `foreign_key_list` — must resolve WHICH database their object
//! argument lives in from the pragma's schema qualifier and the argument name, then
//! report that database's object. Every expected result is TRANSCRIBED FROM THE
//! SQLITE DOCS in `spec/sqlite-doc/`, never from what the engine currently returns;
//! assertions are never weakened to pass.
//!
//! Spec sources (all under `spec/sqlite-doc/`):
//!   * `pragma.html` — every one of these pragmas has the form
//!     `PRAGMA schema.table_info(table-name)` / `PRAGMA schema.index_info(index-name)`;
//!     the optional schema-name is an ATTACH-ed db name or `main`/`temp`, and "If the
//!     schema-name is omitted, main is assumed" governs the pragma's SCHEMA default.
//!     The per-row column semantics are pinned by `conformance_pragma_introspection.rs`;
//!     this file pins only WHICH namespace answers.
//!   * `lang_naming.html` — an UNQUALIFIED object reference resolves `temp`, then
//!     `main`, then attached databases in attach order, taking the FIRST match. So an
//!     unqualified pragma argument follows that search order (the same resolution the
//!     engine uses for the same name in `SELECT ... FROM t`), NOT an unconditional
//!     "main"; a temp table shadows a same-named main table.
//!   * `lang_attach.html` — `ATTACH ':memory:' AS aux` adds an in-memory database.
//!
//! DISCRIMINATOR style: same-named objects are given DIFFERENT shapes per namespace
//! (distinct columns / column counts / index names), and a qualifier pointed at the
//! WRONG namespace is asserted to return the empty-but-columns result. That way a
//! regression that silently reads `main` (the old hardcoded behavior) fails loudly
//! rather than accidentally matching.

mod conformance;

use conformance::*;

// ===========================================================================
// PRAGMA schema.table_info — the qualifier selects the named namespace.
// ===========================================================================

#[test]
fn qualified_table_info_selects_the_named_namespace() {
    // pragma.html: `PRAGMA schema.table_info(t)` reports the table `t` in `schema`.
    // Same name `t` in all three namespaces with DISTINCT shapes proves the qualifier
    // (not a hardcoded `main`) picks the answering store.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)"); // main.t
    exec(&mut db, "CREATE TEMP TABLE t(b TEXT, c REAL)"); // temp.t shadows main
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.t(d BLOB, e, f NUMERIC)"); // aux.t

    assert_rows(
        &mut db,
        "PRAGMA main.table_info(t)",
        &[vec![int(0), text("a"), text("INTEGER"), int(0), null(), int(0)]],
    );
    assert_rows(
        &mut db,
        "PRAGMA temp.table_info(t)",
        &[
            vec![int(0), text("b"), text("TEXT"), int(0), null(), int(0)],
            vec![int(1), text("c"), text("REAL"), int(0), null(), int(0)],
        ],
    );
    assert_rows(
        &mut db,
        "PRAGMA aux.table_info(t)",
        &[
            vec![int(0), text("d"), text("BLOB"), int(0), null(), int(0)],
            // `e` has no declared type, so `type` is '' (pragma.html: "else ''").
            vec![int(1), text("e"), text(""), int(0), null(), int(0)],
            vec![int(2), text("f"), text("NUMERIC"), int(0), null(), int(0)],
        ],
    );
}

#[test]
fn qualified_temporary_spelling_selects_temp() {
    // pragma.html / lang_naming.html: `temporary` is an accepted spelling of the temp
    // schema qualifier, equivalent to `temp`.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "CREATE TEMP TABLE t(b TEXT, c REAL)");
    assert_rows(
        &mut db,
        "PRAGMA temporary.table_info(t)",
        &[
            vec![int(0), text("b"), text("TEXT"), int(0), null(), int(0)],
            vec![int(1), text("c"), text("REAL"), int(0), null(), int(0)],
        ],
    );
}

#[test]
fn unqualified_table_info_resolves_temp_over_main_matching_select_star() {
    // lang_naming.html: an unqualified name resolves temp -> main, first match. So an
    // unqualified `table_info(t)` reports TEMP's columns (temp shadows main), IDENTICAL
    // to the columns `SELECT * FROM t` produces — the pragma must not diverge from SQL
    // name resolution. The qualified `main.table_info(t)` still reaches main.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)"); // main.t(a)
    exec(&mut db, "CREATE TEMP TABLE t(b TEXT, c REAL)"); // temp.t(b, c)

    // The pragma resolves to temp...
    assert_rows(
        &mut db,
        "PRAGMA table_info(t)",
        &[
            vec![int(0), text("b"), text("TEXT"), int(0), null(), int(0)],
            vec![int(1), text("c"), text("REAL"), int(0), null(), int(0)],
        ],
    );
    // ...exactly as `SELECT * FROM t` resolves to temp (consistency with SQL).
    assert_columns(&mut db, "SELECT * FROM t", &["b", "c"]);
    // The explicit `main` qualifier still reaches main's single column.
    assert_rows(
        &mut db,
        "PRAGMA main.table_info(t)",
        &[vec![int(0), text("a"), text("INTEGER"), int(0), null(), int(0)]],
    );
}

#[test]
fn unqualified_table_info_finds_attached_only_table() {
    // lang_naming.html: the search order falls through to attached databases, so an
    // unqualified `table_info(u)` for a table that lives ONLY in `aux` finds it. Before
    // namespace-awareness this returned 0 rows (it only ever looked in main).
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.u(x INTEGER, y TEXT)");

    assert_rows(
        &mut db,
        "PRAGMA table_info(u)",
        &[
            vec![int(0), text("x"), text("INTEGER"), int(0), null(), int(0)],
            vec![int(1), text("y"), text("TEXT"), int(0), null(), int(0)],
        ],
    );
    // A qualifier RESTRICTS: `main` has no `u`, so the qualified form is empty (with
    // columns) — the discriminator against a lookup that silently falls back to main.
    assert_columns(
        &mut db,
        "PRAGMA main.table_info(u)",
        &["cid", "name", "type", "notnull", "dflt_value", "pk"],
    );
    assert_rows(&mut db, "PRAGMA main.table_info(u)", &[]);
}

// ===========================================================================
// PRAGMA schema.table_xinfo — full column list incl. generated, in a non-main
// namespace (pragma.html #pragma_table_xinfo: hidden 0 normal / 2 VIRTUAL / 3 STORED).
// ===========================================================================

#[test]
fn qualified_table_xinfo_reports_generated_columns_in_temp() {
    // #pragma_table_xinfo lists EVERY column with its true cid and a trailing `hidden`
    // flag; a temp table exercises the flag set (0 normal, 2 VIRTUAL, 3 STORED) in a
    // NON-main namespace. Same generated-column shape pinned by
    // `conformance_generated_columns.rs::table_xinfo_reports_hidden_flags`, here routed
    // via the `temp.` qualifier.
    let mut db = mem();
    exec(
        &mut db,
        "CREATE TEMP TABLE g(a INTEGER, b INT AS (a + 1) VIRTUAL, c INT AS (a * 2) STORED, d INT)",
    );
    assert_columns(
        &mut db,
        "PRAGMA temp.table_xinfo(g)",
        &["cid", "name", "type", "notnull", "dflt_value", "pk", "hidden"],
    );
    assert_rows(
        &mut db,
        "PRAGMA temp.table_xinfo(g)",
        &[
            vec![int(0), text("a"), text("INTEGER"), int(0), null(), int(0), int(0)],
            vec![int(1), text("b"), text("INT"), int(0), null(), int(0), int(2)],
            vec![int(2), text("c"), text("INT"), int(0), null(), int(0), int(3)],
            vec![int(3), text("d"), text("INT"), int(0), null(), int(0), int(0)],
        ],
    );
    // Unqualified resolves to temp too (g lives only there): same rows.
    assert_rows(
        &mut db,
        "PRAGMA table_xinfo(g)",
        &[
            vec![int(0), text("a"), text("INTEGER"), int(0), null(), int(0), int(0)],
            vec![int(1), text("b"), text("INT"), int(0), null(), int(0), int(2)],
            vec![int(2), text("c"), text("INT"), int(0), null(), int(0), int(3)],
            vec![int(3), text("d"), text("INT"), int(0), null(), int(0), int(0)],
        ],
    );
}

// ===========================================================================
// PRAGMA schema.index_list — the index list of the table in the named namespace.
// (Row shape per conformance_pragma_introspection.rs: seq, name, unique, origin, partial.)
// ===========================================================================

#[test]
fn index_list_qualified_and_unqualified_temp() {
    // A CREATE INDEX on a temp table lives in temp (origin 'c', unique 0, partial 0).
    // `PRAGMA temp.index_list(tt)` and the unqualified form (tt is temp-only) both
    // report it.
    let mut db = mem();
    exec(&mut db, "CREATE TEMP TABLE tt(a, b)");
    exec(&mut db, "CREATE INDEX itemp ON tt(a)"); // resolves to temp (tt is temp)

    assert_rows(
        &mut db,
        "PRAGMA temp.index_list(tt)",
        &[vec![int(0), text("itemp"), int(0), text("c"), int(0)]],
    );
    assert_rows(
        &mut db,
        "PRAGMA index_list(tt)",
        &[vec![int(0), text("itemp"), int(0), text("c"), int(0)]],
    );
}

#[test]
fn index_list_qualified_and_unqualified_attached() {
    // Same, in an ATTACH-ed database. `at` and its index live only in `aux`.
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.at(a, b)");
    exec(&mut db, "CREATE INDEX iaux ON at(a)"); // resolves to aux (at is aux-only)

    assert_rows(
        &mut db,
        "PRAGMA aux.index_list(at)",
        &[vec![int(0), text("iaux"), int(0), text("c"), int(0)]],
    );
    assert_rows(
        &mut db,
        "PRAGMA index_list(at)",
        &[vec![int(0), text("iaux"), int(0), text("c"), int(0)]],
    );
    // The `main` qualifier restricts to main, which has no `at` -> empty.
    assert_rows(&mut db, "PRAGMA main.index_list(at)", &[]);
}

#[test]
fn index_list_unqualified_resolves_temp_over_main() {
    // `index_list` shares the table-argument resolver (`owner_db`) with `table_info`, so
    // its shadowing must match: with a same-named table `t` in BOTH main and temp, each
    // carrying a DIFFERENTLY-named index, an unqualified `index_list(t)` reports TEMP's
    // index (temp shadows main), while each qualifier reaches its own store. Distinct
    // index names prove which store answered — a regression that read main would return
    // `im` instead of `it` and fail loudly.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)"); // main.t
    exec(&mut db, "CREATE INDEX im ON t(a)"); // resolves to main (only main.t exists): main.im
    exec(&mut db, "CREATE TEMP TABLE t(m, n)"); // temp.t shadows main.t
    exec(&mut db, "CREATE INDEX it ON t(m)"); // resolves to temp (temp.t shadows): temp.it

    // Unqualified -> temp (shadows main): reports temp's index `it`.
    assert_rows(
        &mut db,
        "PRAGMA index_list(t)",
        &[vec![int(0), text("it"), int(0), text("c"), int(0)]],
    );
    // Qualifiers reach each store independently.
    assert_rows(
        &mut db,
        "PRAGMA temp.index_list(t)",
        &[vec![int(0), text("it"), int(0), text("c"), int(0)]],
    );
    assert_rows(
        &mut db,
        "PRAGMA main.index_list(t)",
        &[vec![int(0), text("im"), int(0), text("c"), int(0)]],
    );
}

// ===========================================================================
// PRAGMA schema.foreign_key_list — the FKs of the table in the named namespace.
// (Row shape per conformance_foreign_keys.rs: id, seq, table, from, to, on_update,
// on_delete, match.)
// ===========================================================================

#[test]
fn foreign_key_list_qualified_and_unqualified_attached() {
    // pragma.html #pragma_foreign_key_list: one row per FK column. A two-table FK lives
    // in `aux`; `PRAGMA aux.foreign_key_list(child)` (and the unqualified form, child is
    // aux-only) reports it. `y REFERENCES parent(id)` -> to='id', both actions the
    // "NO ACTION" default, match 'NONE'.
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.parent(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE aux.child(x INTEGER, y INTEGER REFERENCES parent(id))");

    let expected = vec![vec![
        int(0),
        int(0),
        text("parent"),
        text("y"),
        text("id"),
        text("NO ACTION"),
        text("NO ACTION"),
        text("NONE"),
    ]];
    assert_rows(&mut db, "PRAGMA aux.foreign_key_list(child)", &expected);
    assert_rows(&mut db, "PRAGMA foreign_key_list(child)", &expected);
    // `main` has no `child` -> the qualifier restricts to an empty result.
    assert_rows(&mut db, "PRAGMA main.foreign_key_list(child)", &[]);
}

// ===========================================================================
// PRAGMA schema.index_info / index_xinfo — an INDEX-argument pragma. There is no
// owner_db for an index, so an unqualified index name resolves by walking the live
// namespaces in search order (temp, main, attached) for the first that defines it.
// (Row shapes per conformance_pragma_introspection.rs.)
// ===========================================================================

#[test]
fn index_info_qualified_and_unqualified_temp() {
    // #pragma_index_info: seqno (rank in index), cid (rank in the table), name. Index
    // `it` on temp `tt(a, b)` over `b`: b is table column 1 -> (0, 1, 'b').
    let mut db = mem();
    exec(&mut db, "CREATE TEMP TABLE tt(a, b)");
    exec(&mut db, "CREATE INDEX it ON tt(b)");

    assert_rows(&mut db, "PRAGMA temp.index_info(it)", &[vec![int(0), int(1), text("b")]]);
    // Unqualified: `it` lives only in temp, found by the search order.
    assert_rows(&mut db, "PRAGMA index_info(it)", &[vec![int(0), int(1), text("b")]]);
    // A qualifier pointed at main (no such index) restricts to an empty result.
    assert_rows(&mut db, "PRAGMA main.index_info(it)", &[]);
}

#[test]
fn index_info_qualified_and_unqualified_attached() {
    // Same for an ATTACH-ed database. Index `ia` on aux `at(x, y)` over `x`: x is table
    // column 0 -> (0, 0, 'x').
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.at(x, y)");
    exec(&mut db, "CREATE INDEX ia ON at(x)");

    assert_rows(&mut db, "PRAGMA aux.index_info(ia)", &[vec![int(0), int(0), text("x")]]);
    assert_rows(&mut db, "PRAGMA index_info(ia)", &[vec![int(0), int(0), text("x")]]);
}

#[test]
fn index_xinfo_qualified_temp_includes_rowid_auxiliary() {
    // #pragma_index_xinfo on a rowid table: the key column(s) (key=1) then the rowid
    // auxiliary (cid=-1, name=NULL, key=0). Plain key column -> desc=0, coll='BINARY'.
    // Index `it` on temp `tt(a, b)` over `b` (table column 1).
    let mut db = mem();
    exec(&mut db, "CREATE TEMP TABLE tt(a, b)");
    exec(&mut db, "CREATE INDEX it ON tt(b)");

    let expected = vec![
        vec![int(0), int(1), text("b"), int(0), text("BINARY"), int(1)],
        vec![int(1), int(-1), null(), int(0), text("BINARY"), int(0)],
    ];
    assert_rows(&mut db, "PRAGMA temp.index_xinfo(it)", &expected);
    assert_rows(&mut db, "PRAGMA index_xinfo(it)", &expected);
}

#[test]
fn index_xinfo_qualified_attached_includes_rowid_auxiliary() {
    // Same, in an ATTACH-ed database: index `ia` on aux `at(x, y)` over `x`.
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.at(x, y)");
    exec(&mut db, "CREATE INDEX ia ON at(x)");

    let expected = vec![
        vec![int(0), int(0), text("x"), int(0), text("BINARY"), int(1)],
        vec![int(1), int(-1), null(), int(0), text("BINARY"), int(0)],
    ];
    assert_rows(&mut db, "PRAGMA aux.index_xinfo(ia)", &expected);
    assert_rows(&mut db, "PRAGMA index_xinfo(ia)", &expected);
}

#[test]
fn unqualified_index_resolves_temp_before_attached() {
    // lang_naming.html search order applied to an index name: with the SAME index name
    // `dup` defined in BOTH temp and aux (on differently-shaped tables), an unqualified
    // `index_info(dup)` resolves to TEMP (searched before attached), while each qualifier
    // reaches its own store. Distinct cid/name prove which store answered.
    let mut db = mem();
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TEMP TABLE tt(a, b)"); // temp
    exec(&mut db, "CREATE INDEX dup ON tt(b)"); // resolves to temp (tt is temp): temp.dup over 'b'
    exec(&mut db, "CREATE TABLE aux.at(x, y)"); // aux
    exec(&mut db, "CREATE INDEX dup ON at(x)"); // resolves to aux (at is aux-only): aux.dup over 'x'

    // Unqualified -> temp (searched first): b at cid 1.
    assert_rows(&mut db, "PRAGMA index_info(dup)", &[vec![int(0), int(1), text("b")]]);
    // Qualifiers reach each store independently.
    assert_rows(&mut db, "PRAGMA temp.index_info(dup)", &[vec![int(0), int(1), text("b")]]);
    assert_rows(&mut db, "PRAGMA aux.index_info(dup)", &[vec![int(0), int(0), text("x")]]);
}

#[test]
fn unqualified_index_resolves_temp_over_main() {
    // Pins the OTHER edge of the index search order: temp BEFORE main. The same index
    // name `dup` is defined in BOTH main and temp (on differently-shaped tables), so an
    // unqualified `index_info(dup)` must resolve to TEMP (searched first), not main. This
    // is the index-path analogue of `unqualified_table_info_resolves_temp_over_main...`;
    // a resolver that walked `main -> temp -> ...` would answer with main's `b`/cid 1 and
    // fail loudly here.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE mt(a, b)"); // main.mt
    exec(&mut db, "CREATE INDEX dup ON mt(b)"); // resolves to main (only main.mt exists): main.dup over 'b' (cid 1)
    exec(&mut db, "CREATE TEMP TABLE tt(p, q)"); // temp.tt
    exec(&mut db, "CREATE INDEX dup ON tt(p)"); // resolves to temp (tt is temp): temp.dup over 'p' (cid 0)

    // Unqualified -> temp (searched before main): p at cid 0.
    assert_rows(&mut db, "PRAGMA index_info(dup)", &[vec![int(0), int(0), text("p")]]);
    // Qualifiers reach each store independently: main's `dup` is over `b` at cid 1.
    assert_rows(&mut db, "PRAGMA temp.index_info(dup)", &[vec![int(0), int(0), text("p")]]);
    assert_rows(&mut db, "PRAGMA main.index_info(dup)", &[vec![int(0), int(1), text("b")]]);
}

// ===========================================================================
// Fail-closed cases: unknown qualifier and not-live temp -> empty result, NO panic.
// ===========================================================================

#[test]
fn unknown_qualifier_returns_empty_with_columns() {
    // An unknown/unattached qualifier resolves to no namespace; the engine reports it as
    // the empty-but-columns result (its "no such object" convention), not an error/panic.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    assert_columns(
        &mut db,
        "PRAGMA nope.table_info(t)",
        &["cid", "name", "type", "notnull", "dflt_value", "pk"],
    );
    assert_rows(&mut db, "PRAGMA nope.table_info(t)", &[]);
    // Same for an index-argument pragma.
    assert_rows(&mut db, "PRAGMA nope.index_info(t)", &[]);
}

#[test]
fn not_live_temp_pragma_is_empty_not_panic() {
    // `PRAGMA temp.table_info(t)` on a fresh connection with NO temp store must NOT panic:
    // `temp` resolves to DbIndex(1) unconditionally, and the checked per-namespace lookup
    // returns None for the absent store -> an empty (but columns-present) result, i.e. an
    // empty temp schema.
    let mut db = mem();
    assert_columns(
        &mut db,
        "PRAGMA temp.table_info(t)",
        &["cid", "name", "type", "notnull", "dflt_value", "pk"],
    );
    assert_rows(&mut db, "PRAGMA temp.table_info(t)", &[]);
    // The other five, likewise no-panic on a not-live temp store.
    assert_rows(&mut db, "PRAGMA temp.table_xinfo(t)", &[]);
    assert_rows(&mut db, "PRAGMA temp.index_list(t)", &[]);
    assert_rows(&mut db, "PRAGMA temp.index_info(i)", &[]);
    assert_rows(&mut db, "PRAGMA temp.index_xinfo(i)", &[]);
    assert_rows(&mut db, "PRAGMA temp.foreign_key_list(t)", &[]);
}

// ===========================================================================
// Regression guard: a plain main-only connection is byte-for-byte unchanged.
// ===========================================================================

#[test]
fn main_only_unqualified_pragmas_unchanged() {
    // The hot path: with no temp objects and no attachments, unqualified resolution must
    // reach `main` with no observable difference from before the namespace change.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT)");
    exec(&mut db, "CREATE INDEX ix ON t(b)");

    assert_rows(
        &mut db,
        "PRAGMA table_info(t)",
        &[
            vec![int(0), text("a"), text("INTEGER"), int(0), null(), int(1)],
            vec![int(1), text("b"), text("TEXT"), int(0), null(), int(0)],
        ],
    );
    assert_rows(
        &mut db,
        "PRAGMA index_list(t)",
        &[vec![int(0), text("ix"), int(0), text("c"), int(0)]],
    );
    assert_rows(&mut db, "PRAGMA index_info(ix)", &[vec![int(0), int(1), text("b")]]);
    assert_rows(
        &mut db,
        "PRAGMA index_xinfo(ix)",
        &[
            vec![int(0), int(1), text("b"), int(0), text("BINARY"), int(1)],
            vec![int(1), int(-1), null(), int(0), text("BINARY"), int(0)],
        ],
    );
    // An explicit `main.` qualifier reaches the same main object.
    assert_rows(
        &mut db,
        "PRAGMA main.table_info(t)",
        &[
            vec![int(0), text("a"), text("INTEGER"), int(0), null(), int(1)],
            vec![int(1), text("b"), text("TEXT"), int(0), null(), int(0)],
        ],
    );
}
