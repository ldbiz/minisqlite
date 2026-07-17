//! Conformance battery: the **`pragma_*` schema-introspection TABLE-VALUED FUNCTIONS**
//! (`spec/sqlite-doc/pragma.html` §2 "PRAGMA functions").
//!
//! pragma.html §2: "PRAGMAs that return results and that have no side-effects can be
//! accessed from ordinary SELECT statements as table-valued functions. For each
//! participating PRAGMA, the corresponding table-valued function has the same name as the
//! PRAGMA with a 7-character 'pragma_' prefix. The PRAGMA argument and schema, if any, are
//! passed as arguments to the table-valued function, with the schema as an optional, last
//! argument." So `SELECT * FROM pragma_table_info('t')` must return the SAME rows as the
//! `PRAGMA table_info(t)` STATEMENT, and the two-argument form
//! `pragma_table_info('t','main')` selects the schema.
//!
//! Two independent checks per pragma, both required:
//!   1. **Grounding** — the STATEMENT form is asserted against rows transcribed from
//!      pragma.html §2 (never from what the engine happens to return). A failing case is
//!      the intended "engine diverges from the spec" signal; assertions are never
//!      weakened to pass.
//!   2. **Equivalence** — the TVF form is asserted to return EXACTLY those same rows, and
//!      [`assert_same_rows`] additionally diffs the TVF result against the live statement
//!      result cell-by-cell. That is the core promise of pragma.html §2, and it also pins
//!      richer outputs this file does not hand-enumerate.
//!
//! pragma.html §2 also lists the advantages the TVF form adds over the statement — "the
//! query can return just a subset of the PRAGMA columns, can include a WHERE clause, can
//! use aggregate functions, and the table-valued function can be just one of several data
//! sources in a join" — each of which is exercised below (column/row subset, `WHERE`,
//! `count(*)`, and the documented `pragma_index_list` ⋈ `pragma_index_info` LATERAL join).

mod conformance;

use conformance::*;

use minisqlite::Connection;

/// Assert a `pragma_*` table-valued function returns EXACTLY the rows of its corresponding
/// `PRAGMA` statement — the equivalence pragma.html §2 promises. Both are run through the
/// facade against the SAME database and compared cell-by-cell (order-sensitive: the TVF's
/// `SELECT *` yields the pragma's fixed columns in declaration order, the same order the
/// statement returns). `Value` has no `PartialEq`, so cells are compared with `value_eq`.
fn assert_same_rows(db: &mut Connection, stmt_sql: &str, tvf_sql: &str) {
    let stmt = query(db, stmt_sql);
    let tvf = query(db, tvf_sql);
    let same = stmt.rows.len() == tvf.rows.len()
        && stmt.rows.iter().zip(tvf.rows.iter()).all(|(rs, rt)| {
            rs.len() == rt.len() && rs.iter().zip(rt.iter()).all(|(a, b)| value_eq(a, b))
        });
    assert!(
        same,
        "pragma TVF must return the same rows as its statement\n  statement: {stmt_sql}\n  \
         tvf:       {tvf_sql}\n  statement rows: {:?}\n  tvf rows:       {:?}",
        stmt.rows, tvf.rows
    );
}

// ===========================================================================
// pragma_table_info — cid / name / type / notnull / dflt_value / pk.
// ===========================================================================

#[test]
fn pragma_table_info_matches_statement_and_spec() {
    // pragma.html #pragma_table_info: one row per NORMAL column — cid, name, type
    // (declared type verbatim, else ''), notnull (0/1), dflt_value (the default's SQL text
    // incl. quotes, NULL when none), pk (0, or the 1-based position in the primary key).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE ti(a INTEGER PRIMARY KEY, b TEXT NOT NULL DEFAULT 'x', c)");
    let expected = &[
        vec![int(0), text("a"), text("INTEGER"), int(0), null(), int(1)],
        vec![int(1), text("b"), text("TEXT"), int(1), text("'x'"), int(0)],
        vec![int(2), text("c"), text(""), int(0), null(), int(0)],
    ];
    // Grounding: the statement form matches the spec.
    assert_rows(&mut db, "PRAGMA table_info(ti)", expected);
    // Equivalence: the TVF form matches the spec, and the live statement result.
    assert_columns(
        &mut db,
        "SELECT * FROM pragma_table_info('ti')",
        &["cid", "name", "type", "notnull", "dflt_value", "pk"],
    );
    assert_rows(&mut db, "SELECT * FROM pragma_table_info('ti')", expected);
    assert_same_rows(&mut db, "PRAGMA table_info(ti)", "SELECT * FROM pragma_table_info('ti')");
}

#[test]
fn pragma_table_info_case_insensitive_function_name() {
    // SQLite function names are case-insensitive, so the TVF resolves regardless of case.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    assert_same_rows(&mut db, "PRAGMA table_info(t)", "SELECT * FROM PRAGMA_Table_Info('t')");
}

// ===========================================================================
// pragma_table_xinfo — adds the trailing `hidden` flag (0 normal, 2 VIRTUAL, 3 STORED).
// ===========================================================================

#[test]
fn pragma_table_xinfo_matches_statement_and_spec() {
    // pragma.html #pragma_table_xinfo: every column incl. generated, with a trailing
    // `hidden` (0 normal, 2 a VIRTUAL generated column, 3 a STORED one). `cid` is the TRUE
    // 0-based table index (no renumbering).
    let mut db = mem();
    exec(
        &mut db,
        "CREATE TABLE tx(a INTEGER, b INT AS (a + 1) VIRTUAL, c INT AS (a * 2) STORED, d INT)",
    );
    let expected = &[
        vec![int(0), text("a"), text("INTEGER"), int(0), null(), int(0), int(0)],
        vec![int(1), text("b"), text("INT"), int(0), null(), int(0), int(2)],
        vec![int(2), text("c"), text("INT"), int(0), null(), int(0), int(3)],
        vec![int(3), text("d"), text("INT"), int(0), null(), int(0), int(0)],
    ];
    assert_rows(&mut db, "PRAGMA table_xinfo(tx)", expected);
    assert_columns(
        &mut db,
        "SELECT * FROM pragma_table_xinfo('tx')",
        &["cid", "name", "type", "notnull", "dflt_value", "pk", "hidden"],
    );
    assert_rows(&mut db, "SELECT * FROM pragma_table_xinfo('tx')", expected);
    assert_same_rows(&mut db, "PRAGMA table_xinfo(tx)", "SELECT * FROM pragma_table_xinfo('tx')");
}

// ===========================================================================
// pragma_index_list — seq / name / unique / origin / partial.
// ===========================================================================

#[test]
fn pragma_index_list_matches_statement_and_spec() {
    // pragma.html #pragma_index_list: one row per index — seq, name, unique (0/1), origin
    // ('c' for CREATE INDEX), partial (0/1). A single index has an unambiguous seq of 0.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE il(a, b)");
    exec(&mut db, "CREATE INDEX ix ON il(a)");
    let expected = &[vec![int(0), text("ix"), int(0), text("c"), int(0)]];
    assert_rows(&mut db, "PRAGMA index_list(il)", expected);
    assert_columns(
        &mut db,
        "SELECT * FROM pragma_index_list('il')",
        &["seq", "name", "unique", "origin", "partial"],
    );
    assert_rows(&mut db, "SELECT * FROM pragma_index_list('il')", expected);
    assert_same_rows(&mut db, "PRAGMA index_list(il)", "SELECT * FROM pragma_index_list('il')");
}

// ===========================================================================
// pragma_index_info — seqno / cid / name (one row per key column).
// ===========================================================================

#[test]
fn pragma_index_info_matches_statement_and_spec() {
    // pragma.html #pragma_index_info: one row per key column — seqno (rank in the index),
    // cid (the column's ordinal in the table), name. Index on (b, c) of t(a, b, c): b is
    // table column 1, c is table column 2.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE ii(a, b, c)");
    exec(&mut db, "CREATE INDEX ixi ON ii(b, c)");
    let expected = &[vec![int(0), int(1), text("b")], vec![int(1), int(2), text("c")]];
    assert_rows(&mut db, "PRAGMA index_info(ixi)", expected);
    assert_columns(&mut db, "SELECT * FROM pragma_index_info('ixi')", &["seqno", "cid", "name"]);
    assert_rows(&mut db, "SELECT * FROM pragma_index_info('ixi')", expected);
    assert_same_rows(&mut db, "PRAGMA index_info(ixi)", "SELECT * FROM pragma_index_info('ixi')");
}

// ===========================================================================
// pragma_index_xinfo — seqno / cid / name / desc / coll / key.
// ===========================================================================

#[test]
fn pragma_index_xinfo_matches_statement_and_spec() {
    // pragma.html #pragma_index_xinfo: like index_info for EVERY index-record column — key
    // columns (key=1) then the rowid auxiliary (cid=-1, name=NULL, key=0) on a rowid table —
    // adding desc (1 if DESC) and coll (the collating sequence, 'BINARY' by default).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE ixx(a, b)");
    exec(&mut db, "CREATE INDEX ixx_ix ON ixx(a)");
    let expected = &[
        vec![int(0), int(0), text("a"), int(0), text("BINARY"), int(1)],
        vec![int(1), int(-1), null(), int(0), text("BINARY"), int(0)],
    ];
    assert_rows(&mut db, "PRAGMA index_xinfo(ixx_ix)", expected);
    assert_columns(
        &mut db,
        "SELECT * FROM pragma_index_xinfo('ixx_ix')",
        &["seqno", "cid", "name", "desc", "coll", "key"],
    );
    assert_rows(&mut db, "SELECT * FROM pragma_index_xinfo('ixx_ix')", expected);
    assert_same_rows(
        &mut db,
        "PRAGMA index_xinfo(ixx_ix)",
        "SELECT * FROM pragma_index_xinfo('ixx_ix')",
    );
}

#[test]
fn pragma_index_xinfo_without_rowid_aux_pk_columns_match_statement() {
    // pragma.html #pragma_index_xinfo: a WITHOUT ROWID table has no rowid, so a secondary
    // index's auxiliary columns are the table's PRIMARY KEY columns not already in the key
    // (withoutrowid.html), listed AFTER the key columns with key=0. Both the statement and
    // the TVF form must include those aux columns and return the SAME rows — the core
    // equivalence pragma.html §2 promises, holding for the WITHOUT ROWID auxiliary case too.
    // The reachable WR secondary index is a UNIQUE-constraint auto-index (introspection is a
    // catalog read; it is registered at CREATE TABLE time). `w(a, b, c, PRIMARY KEY(a, b),
    // UNIQUE(c)) WITHOUT ROWID`: the UNIQUE(c) index `sqlite_autoindex_w_2` -> key c, aux a, b.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE w(a, b, c, PRIMARY KEY(a, b), UNIQUE(c)) WITHOUT ROWID");
    let expected = &[
        vec![int(0), int(2), text("c"), int(0), text("BINARY"), int(1)],
        vec![int(1), int(0), text("a"), int(0), text("BINARY"), int(0)],
        vec![int(2), int(1), text("b"), int(0), text("BINARY"), int(0)],
    ];
    assert_rows(&mut db, "PRAGMA index_xinfo(sqlite_autoindex_w_2)", expected);
    assert_rows(&mut db, "SELECT * FROM pragma_index_xinfo('sqlite_autoindex_w_2')", expected);
    assert_same_rows(
        &mut db,
        "PRAGMA index_xinfo(sqlite_autoindex_w_2)",
        "SELECT * FROM pragma_index_xinfo('sqlite_autoindex_w_2')",
    );
}

// ===========================================================================
// pragma_foreign_key_list — id / seq / table / from / to / on_update / on_delete / match.
// ===========================================================================

#[test]
fn pragma_foreign_key_list_matches_statement_and_spec() {
    // pragma.html #pragma_foreign_key_list: one row per column of each foreign key —
    // id, seq, table (parent), from (child col), to (parent col), on_update, on_delete
    // (the action names; the omitted default is 'NO ACTION'), match (always 'NONE').
    let mut db = mem();
    exec(&mut db, "CREATE TABLE parent(pid INTEGER PRIMARY KEY, code TEXT)");
    exec(&mut db, "CREATE TABLE child(x INT REFERENCES parent(pid) ON DELETE CASCADE, y)");
    let expected = &[vec![
        int(0),
        int(0),
        text("parent"),
        text("x"),
        text("pid"),
        text("NO ACTION"),
        text("CASCADE"),
        text("NONE"),
    ]];
    assert_rows(&mut db, "PRAGMA foreign_key_list(child)", expected);
    assert_columns(
        &mut db,
        "SELECT * FROM pragma_foreign_key_list('child')",
        &["id", "seq", "table", "from", "to", "on_update", "on_delete", "match"],
    );
    assert_rows(&mut db, "SELECT * FROM pragma_foreign_key_list('child')", expected);
    assert_same_rows(
        &mut db,
        "PRAGMA foreign_key_list(child)",
        "SELECT * FROM pragma_foreign_key_list('child')",
    );
}

// ===========================================================================
// Schema-qualified second argument selects the schema (pragma.html §2).
// ===========================================================================

#[test]
fn pragma_tvf_second_argument_selects_schema() {
    // pragma.html §2: "the schema [is] an optional, last argument". The two-argument form
    // `pragma_table_info('ti','main')` targets the main schema, exactly as the qualified
    // statement `PRAGMA main.table_info(ti)` does.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE ti(a INTEGER PRIMARY KEY, b TEXT)");
    assert_same_rows(
        &mut db,
        "PRAGMA main.table_info(ti)",
        "SELECT * FROM pragma_table_info('ti', 'main')",
    );
    // A non-existent schema qualifier selects nothing (empty result, no error).
    let qr = try_query(&mut db, "SELECT * FROM pragma_table_info('ti', 'nope')")
        .unwrap_or_else(|e| panic!("an unknown schema qualifier must not error: {e:?}"));
    assert_eq!(qr.rows.len(), 0, "an unknown schema qualifier yields no rows");
}

// ===========================================================================
// The advantages the TVF form adds over the statement (pragma.html §2): column/row
// subset, WHERE, aggregates, and use as one source in a join.
// ===========================================================================

#[test]
fn pragma_tvf_supports_column_and_row_subset_and_where() {
    // pragma.html §2: the TVF "can return just a subset of the PRAGMA columns, can include
    // a WHERE clause". Select one column, filtered — the primary-key column of `ti`.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE ti(a INTEGER PRIMARY KEY, b TEXT, c)");
    assert_rows(
        &mut db,
        "SELECT name FROM pragma_table_info('ti') WHERE pk = 1",
        &[vec![text("a")]],
    );
}

#[test]
fn pragma_tvf_supports_aggregates() {
    // pragma.html §2: the TVF "can use aggregate functions". count(*) over table_info is
    // the column count (3 normal columns here).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE ti(a INTEGER PRIMARY KEY, b TEXT, c)");
    assert_scalar(&mut db, "SELECT count(*) FROM pragma_table_info('ti')", int(3));
}

#[test]
fn pragma_tvf_lateral_join_index_list_to_index_info() {
    // pragma.html §2 documents exactly this join: to list every indexed column of a schema,
    // join pragma_index_list against pragma_index_info(il.name). The inner TVF's argument
    // references the OUTER row (implicitly LATERAL), so it re-resolves per index.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b, c)");
    exec(&mut db, "CREATE INDEX ix1 ON t(a)");
    exec(&mut db, "CREATE INDEX ix2 ON t(c, b)");
    // For each index, its key columns in order — ix1: [a]; ix2: [c, b].
    assert_rows_unordered(
        &mut db,
        "SELECT il.name, ii.seqno, ii.name \
           FROM pragma_index_list('t') AS il \
           JOIN pragma_index_info(il.name) AS ii \
          ORDER BY il.name, ii.seqno",
        &[
            vec![text("ix1"), int(0), text("a")],
            vec![text("ix2"), int(0), text("c")],
            vec![text("ix2"), int(1), text("b")],
        ],
    );
}

// ===========================================================================
// Empty-result and error cases.
// ===========================================================================

#[test]
fn pragma_tvf_absent_or_null_object_is_empty() {
    // A missing object yields the fixed columns and zero rows (the statement form's
    // convention); a NULL name argument likewise yields nothing.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    let missing = try_query(&mut db, "SELECT * FROM pragma_table_info('nope')")
        .unwrap_or_else(|e| panic!("table_info of a missing table must not error: {e:?}"));
    assert_eq!(missing.rows.len(), 0, "a missing table yields no rows");
    let null_arg = try_query(&mut db, "SELECT * FROM pragma_table_info(NULL)")
        .unwrap_or_else(|e| panic!("a NULL object argument must not error: {e:?}"));
    assert_eq!(null_arg.rows.len(), 0, "a NULL object argument yields no rows");
}

#[test]
fn unknown_table_valued_function_is_a_loud_error() {
    // A name that is neither a JSON TVF nor a pragma_* TVF stays a loud error, and a
    // pragma_*-looking name that does not correspond to a real introspection PRAGMA is not
    // silently accepted either.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    assert_query_error(&mut db, "SELECT * FROM not_a_function('t')");
    assert_query_error(&mut db, "SELECT * FROM pragma_no_such_thing('t')");
}

#[test]
fn pragma_tvf_too_many_arguments_is_an_error() {
    // The object name plus the optional trailing schema are the ONLY arguments (pragma.html
    // §2). A third argument is a loud error, matching SQLite's "wrong number of arguments".
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    assert_query_error(&mut db, "SELECT * FROM pragma_table_info('t', 'main', 'extra')");
}

#[test]
fn pragma_tvf_zero_arguments_is_empty_not_an_error() {
    // With no object argument there is nothing to describe, so the TVF yields the fixed
    // columns and zero rows rather than erroring — the same empty shape as an absent object.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    let qr = try_query(&mut db, "SELECT * FROM pragma_table_info()")
        .unwrap_or_else(|e| panic!("a zero-argument pragma TVF must not error: {e:?}"));
    assert_eq!(qr.rows.len(), 0, "no object argument yields no rows");
    assert_columns(
        &mut db,
        "SELECT * FROM pragma_table_info()",
        &["cid", "name", "type", "notnull", "dflt_value", "pk"],
    );
}

#[test]
fn pragma_tvf_coerces_non_text_object_argument_to_text() {
    // pragma.html §2 passes the argument to the table-valued function; SQLite's TVF reads it
    // via sqlite3_value_text, so a NON-text argument is coerced to its text form —
    // pragma_table_info(5) looks up the table named "5". This is the one place the TVF and
    // the PRAGMA STATEMENT legitimately differ (the statement treats a bare numeric literal
    // as "no name" → empty); the TVF's text coercion matches real sqlite. Pinned here so the
    // coercion — and the difference — is visible rather than silent.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE \"5\"(a, b)");
    assert_rows(
        &mut db,
        "SELECT name FROM pragma_table_info(5)",
        &[vec![text("a")], vec![text("b")]],
    );
}

#[test]
fn pragma_index_info_second_argument_selects_schema() {
    // The trailing schema argument drives the INDEX-name resolution path
    // (resolve_index_target / owner_db_of_index), not only the table path: an explicit
    // 'main' matches the qualified statement `PRAGMA main.index_info(ix)`, and an unknown
    // schema qualifier selects nothing.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "CREATE INDEX ix ON t(b)");
    assert_same_rows(
        &mut db,
        "PRAGMA main.index_info(ix)",
        "SELECT * FROM pragma_index_info('ix', 'main')",
    );
    let qr = try_query(&mut db, "SELECT * FROM pragma_index_info('ix', 'nope')")
        .unwrap_or_else(|e| panic!("an unknown schema qualifier must not error: {e:?}"));
    assert_eq!(qr.rows.len(), 0, "an unknown schema qualifier yields no index rows");
}

#[test]
fn pragma_composite_primary_key_tvf_matches_statement() {
    // A TABLE-level composite PRIMARY KEY. Two things are pinned: the correct `pk` ordinals,
    // and that the TVF returns the SAME rows as the statement.
    //
    // pragma.html #pragma_table_info: `pk` is each column's 1-based position WITHIN the
    // primary key (0 for a non-member). For `PRIMARY KEY(a, b)`, `a` is pk=1, `b` is pk=2,
    // and `c` (not in the key) is pk=0. This was once a shared gap — both forms reported
    // pk=0 because the catalog carried no table-level PK-column order — now CLOSED:
    // `TableDef::primary_key` records the ordered PK-column indices and `table_info_pk` reads
    // them. Because the statement path and the TVF path run the ONE shared `pragma_rows`
    // builder, both report those same 1-based ordinals — which is exactly the equivalence
    // this pins (statement matches spec, TVF matches spec, and the two agree cell-for-cell).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b, c, PRIMARY KEY(a, b))");
    let expected = &[
        vec![int(0), text("a"), text(""), int(0), null(), int(1)],
        vec![int(1), text("b"), text(""), int(0), null(), int(2)],
        vec![int(2), text("c"), text(""), int(0), null(), int(0)],
    ];
    assert_rows(&mut db, "PRAGMA table_info(t)", expected);
    assert_rows(&mut db, "SELECT * FROM pragma_table_info('t')", expected);
    assert_same_rows(&mut db, "PRAGMA table_info(t)", "SELECT * FROM pragma_table_info('t')");
}

// ===========================================================================
// Regression guard: wiring the pragma_* TVFs must not disturb the JSON TVFs.
// ===========================================================================

#[test]
fn json_each_still_works_alongside_pragma_tvfs() {
    // json_each / json_tree are the other FROM table-valued functions; the pragma wiring
    // shares their FROM-clause path, so a smoke check keeps them working.
    let mut db = mem();
    assert_rows(
        &mut db,
        "SELECT value FROM json_each('[10,20,30]') ORDER BY value",
        &[vec![int(10)], vec![int(20)], vec![int(30)]],
    );
}
