//! Conformance: a VIEW / FROM-subquery / CTE output column that is a bare column
//! reference INHERITS that column's AFFINITY and COLLATION, while a column mapping to any
//! other expression has NO affinity + BINARY collation.
//!
//! Source of truth: `spec/sqlite-doc/datatype3.html`
//!   §3.2 "Affinity Of Expressions"
//!   §3.3 "Column Affinity For Views And Subqueries" (the `v1(x,y,z)` example)
//!   §3.3.1 "Column Affinity For Compound Views"
//!   §7.1 collation of a comparison operand (a bare column carries its column's collation)
//!
//! §3.3: "The columns of a VIEW or FROM-clause subquery are really the expressions in the
//! result set of the SELECT ... the affinity for columns of a VIEW or subquery are
//! determined by the expression affinity rules above." So a derived column that maps
//! directly to a base column has that column's affinity; a derived column that maps to an
//! expression (arithmetic, literal, function, concat) has NO affinity.
//!
//! The observable consequence is a comparison RESULT SET: applying INTEGER affinity to a
//! text literal (`a = '1'` → `a = 1`) makes the row match, and inheriting NOCASE makes
//! `a = 'ABC'` match a stored `'abc'`. A derived column that is an EXPRESSION keeps NONE
//! affinity, so `x = '1'` compares an integer against text and does NOT match — the
//! discriminating negative proving the inheritance is expression-directed, not blanket.
//!
//! Every expected value is transcribed from the spec / stated sqlite behavior,
//! never from what this engine returns; a case that reveals an engine bug is left as a
//! genuine failing assertion rather than weakened to pass.

mod conformance;

use conformance::*;
// `Connection` is not re-exported by the harness (it re-exports only its `pub fn`
// helpers), so name it from the facade for the shared table-setup helper.
use minisqlite::Connection;

/// A fresh db with `t(a INTEGER)` holding a single row `(1)`. The INTEGER affinity on the
/// derived column is what turns `a = '1'` into `a = 1` (a match).
fn t_int() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    db
}

// ---- INTEGER affinity inherited through a subquery / view / CTE --------------

/// `SELECT * FROM (SELECT a FROM t) WHERE a = '1'`: the subquery column `a` maps directly
/// to `t.a`, so it inherits INTEGER affinity (§3.3). The comparison then applies NUMERIC
/// affinity to the text literal `'1'` (§4.2 rule (i)), converting it to the integer 1,
/// which equals the stored 1 → the row is returned.
#[test]
fn subquery_bare_column_inherits_integer_affinity() {
    let mut db = t_int();
    assert_rows(
        &mut db,
        "SELECT * FROM (SELECT a FROM t) WHERE a = '1'",
        &[vec![int(1)]],
    );
}

/// Same inheritance through a VIEW: `v.a` maps to `t.a`, so `WHERE a = '1'` matches.
#[test]
fn view_bare_column_inherits_integer_affinity() {
    let mut db = t_int();
    exec(&mut db, "CREATE VIEW v AS SELECT a FROM t");
    assert_rows(&mut db, "SELECT * FROM v WHERE a = '1'", &[vec![int(1)]]);
}

/// Same inheritance through a CTE: `c.a` maps to `t.a`, so `WHERE a = '1'` matches.
#[test]
fn cte_bare_column_inherits_integer_affinity() {
    let mut db = t_int();
    assert_rows(
        &mut db,
        "WITH c AS (SELECT a FROM t) SELECT * FROM c WHERE a = '1'",
        &[vec![int(1)]],
    );
}

/// `SELECT *` in the subquery is a bare column reference per column, so the star-expanded
/// derived column `a` inherits INTEGER affinity just as an explicit `SELECT a` does.
#[test]
fn star_subquery_preserves_integer_affinity() {
    let mut db = t_int();
    assert_rows(
        &mut db,
        "SELECT * FROM (SELECT * FROM t) WHERE a = '1'",
        &[vec![int(1)]],
    );
}

/// The affinity flows through nested subqueries: `a` is a bare column reference at each
/// layer, so the outermost `a` still carries `t.a`'s INTEGER affinity (§3.3 re-applied per
/// layer, since a derived column's affinity is itself read back by the next layer up).
#[test]
fn nested_subquery_inherits_integer_affinity_through_layers() {
    let mut db = t_int();
    assert_rows(
        &mut db,
        "SELECT * FROM (SELECT a FROM (SELECT a FROM t)) WHERE a = '1'",
        &[vec![int(1)]],
    );
}

// ---- The discriminating NEGATIVE: an expression column stays NONE ------------

/// `SELECT * FROM (SELECT a + 0 AS x FROM t) WHERE x = '1'`: `a + 0` is an arithmetic
/// expression, which has NO affinity (§3.2 "otherwise, an expression has no affinity"), so
/// the derived column `x` is NONE-affinity. `x = '1'` then compares the integer 1 against
/// the text `'1'` with NO conversion (§4.2 rule (iii)) → INTEGER < TEXT (§4.1) → not equal
/// → NO rows. If `x` wrongly inherited INTEGER affinity this would return the row, so this
/// pins the inheritance to bare columns only.
#[test]
fn subquery_expression_column_keeps_no_affinity() {
    let mut db = t_int();
    assert_rows(
        &mut db,
        "SELECT * FROM (SELECT a + 0 AS x FROM t) WHERE x = '1'",
        &[],
    );
}

/// The same negative through a VIEW, matching the §3.3 example (`v1.y` maps to `a+c`, so it
/// "has no affinity, since [it] map[s] into expression a+c ... and expressions always have
/// no affinity").
#[test]
fn view_expression_column_keeps_no_affinity() {
    let mut db = t_int();
    exec(&mut db, "CREATE VIEW vx AS SELECT a + 0 AS x FROM t");
    assert_rows(&mut db, "SELECT * FROM vx WHERE x = '1'", &[]);
}

// ---- NOCASE collation inherited through a subquery ---------------------------

/// `SELECT * FROM (SELECT a FROM t2) WHERE a = 'ABC'` where `t2.a` is `TEXT COLLATE
/// NOCASE`: the derived column `a` inherits the NOCASE collating sequence (a bare column
/// reference carries its column's collation, §7.1). The comparison of the stored `'abc'`
/// with `'ABC'` under NOCASE is equal → the row is returned. Under the default BINARY
/// (the pre-fix behavior) `'abc' != 'ABC'` and no row would return.
#[test]
fn subquery_bare_column_inherits_nocase_collation() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t2(a TEXT COLLATE NOCASE)");
    exec(&mut db, "INSERT INTO t2 VALUES ('abc')");
    assert_rows(
        &mut db,
        "SELECT * FROM (SELECT a FROM t2) WHERE a = 'ABC'",
        &[vec![text("abc")]],
    );
}

/// The NOCASE inheritance also holds through a VIEW.
#[test]
fn view_bare_column_inherits_nocase_collation() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t2(a TEXT COLLATE NOCASE)");
    exec(&mut db, "INSERT INTO t2 VALUES ('abc')");
    exec(&mut db, "CREATE VIEW v2 AS SELECT a FROM t2");
    assert_rows(&mut db, "SELECT * FROM v2 WHERE a = 'ABC'", &[vec![text("abc")]]);
}

/// An expression column does NOT inherit the source column's collation: `a || ''` is a
/// concat expression, so its collation is BINARY (§7.1: a column with an operator applied
/// is no longer a column). `'abc' = 'ABC'` under BINARY is not equal → no row. This is the
/// collation analogue of the affinity negative above.
#[test]
fn subquery_expression_column_uses_binary_collation() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t2(a TEXT COLLATE NOCASE)");
    exec(&mut db, "INSERT INTO t2 VALUES ('abc')");
    assert_rows(
        &mut db,
        "SELECT * FROM (SELECT a || '' AS z FROM t2) WHERE z = 'ABC'",
        &[],
    );
}

// ---- Compound derived body: affinity from a constituent arm (§3.3.1) ---------

/// §3.3.1: "When the SELECT that implements a VIEW or FROM-clause subquery is a compound
/// SELECT then the affinity of each column ... will be the affinity of the corresponding
/// result column for one of the individual SELECT statements". The engine uses the
/// LEFTMOST arm, whose `a` maps to `t.a` (INTEGER), so `a = '1'` applies INTEGER affinity
/// and both `UNION ALL` rows match.
#[test]
fn compound_subquery_inherits_a_constituent_arm_affinity() {
    let mut db = t_int();
    assert_rows_unordered(
        &mut db,
        "SELECT * FROM (SELECT a FROM t UNION ALL SELECT a FROM t) WHERE a = '1'",
        &[vec![int(1)], vec![int(1)]],
    );
}

// ---- CAST in the projection takes the cast type's affinity -------------------

/// A `CAST(_ AS T)` projected column has T's affinity (§3.2). Here `b` is TEXT but
/// `CAST(b AS INTEGER)` gives the derived column `y` INTEGER affinity, so `y = '1'` applies
/// NUMERIC affinity to `'1'` and matches the stored integer 1 (the value `CAST('1' AS
/// INTEGER)` yields inside the subquery).
#[test]
fn subquery_cast_column_takes_cast_type_affinity() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE tc(b TEXT)");
    exec(&mut db, "INSERT INTO tc VALUES ('1')");
    assert_rows(
        &mut db,
        "SELECT * FROM (SELECT CAST(b AS INTEGER) AS y FROM tc) WHERE y = '1'",
        &[vec![int(1)]],
    );
}

// ---- Qualified `table.*` inherits affinity too ------------------------------

/// A qualified `t.*` in the subquery expands to the same bare column references as `*`, so
/// the derived column `a` still inherits `t.a`'s INTEGER affinity (`WHERE a = '1'` matches).
/// This exercises the `expand_star_cols(Some(t))` branch, distinct from the unqualified
/// `expand_star_cols(None)` path the other star test covers.
#[test]
fn qualified_table_star_subquery_preserves_integer_affinity() {
    let mut db = t_int();
    assert_rows(
        &mut db,
        "SELECT * FROM (SELECT t.* FROM t) WHERE a = '1'",
        &[vec![int(1)]],
    );
}

// ---- INTEGER PRIMARY KEY (rowid alias) inherits INTEGER through `*` ----------

/// An `INTEGER PRIMARY KEY` column is a rowid alias, so a bare reference to it (here via
/// `SELECT *`) reads as the integer rowid and carries INTEGER affinity (§3.1). Through the
/// subquery the derived `id` therefore inherits INTEGER, so `id = '1'` applies NUMERIC
/// affinity and matches the stored rowid 1. This pins the rowid-alias branch of the shared
/// base-column metadata used by star expansion, which a plain-`INTEGER` column would not
/// distinguish.
#[test]
fn subquery_star_inherits_integer_primary_key_alias_affinity() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE tk(id INTEGER PRIMARY KEY, v TEXT)");
    exec(&mut db, "INSERT INTO tk VALUES (1, 'x')");
    assert_rows(
        &mut db,
        "SELECT * FROM (SELECT * FROM tk) WHERE id = '1'",
        &[vec![int(1), text("x")]],
    );
}

// ---- NATURAL/USING coalesced `*` column uses the LEFT copy's metadata --------

/// A USING/NATURAL shared column appears once under `*` as the LEFT copy, so through a
/// subquery it inherits the LEFT source column's collation. Here `l.k` is `TEXT COLLATE
/// NOCASE` and `r.k` is plain TEXT (BINARY); the coalesced `k` surviving the inner
/// `SELECT *` carries `l.k`'s NOCASE, so the outer `k = 'ABC'` matches the stored `'abc'`.
/// If the RIGHT copy's BINARY collation had been used instead, this would return 0 rows —
/// so it pins the LEFT-copy-wins rule in the star metadata path.
#[test]
fn coalesced_using_star_inherits_left_copy_collation() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE l(k TEXT COLLATE NOCASE)");
    exec(&mut db, "CREATE TABLE r(k TEXT)");
    exec(&mut db, "INSERT INTO l VALUES ('abc')");
    exec(&mut db, "INSERT INTO r VALUES ('abc')");
    assert_rows(
        &mut db,
        "SELECT * FROM (SELECT * FROM l JOIN r USING(k)) WHERE k = 'ABC'",
        &[vec![text("abc")]],
    );
}
