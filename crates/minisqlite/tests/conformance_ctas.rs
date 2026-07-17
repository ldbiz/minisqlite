//! Conformance battery: `CREATE TABLE ... AS SELECT` (CTAS).
//!
//! Every expected value is TRANSCRIBED FROM THE SPEC,
//! `spec/sqlite-doc/lang_createtable.html` §2.1 "CREATE TABLE ... AS SELECT
//! Statements", never from what the engine returns. The binding rules pinned below
//! (verbatim from §2.1) are:
//!
//!   * "The table has the same number of columns as the SELECT statement returns.
//!     The name of each column is the same as the name of the corresponding column in
//!     the result set of the SELECT statement." — so the new table's column COUNT and
//!     NAMES come from the SELECT's result columns; an `AS` alias names the new column,
//!     and a bare column reference keeps that column's name.
//!   * "Tables created using CREATE TABLE AS are initially populated with the rows of
//!     data returned by the SELECT statement." — the new table holds exactly the
//!     SELECT's result rows (a data copy), so projection / WHERE / GROUP BY / JOIN /
//!     UNION / subquery sources each copy in precisely their result set.
//!   * "Rows are assigned contiguously ascending rowid values, starting with 1, in the
//!     order that they are returned by the SELECT statement." — the copy's rowids are
//!     1..N following the SELECT's (here `ORDER BY`-determined) result order.
//!   * "A table created using CREATE TABLE AS has no PRIMARY KEY and no constraints of
//!     any kind. The default value of each column is NULL." — the source's PRIMARY
//!     KEY, UNIQUE constraints and INTEGER-PRIMARY-KEY rowid alias are NOT carried
//!     over; the copy is a PLAIN table, so a value the source would reject as a
//!     duplicate inserts into the copy without error.
//!   * "The declared type of each column is determined by the expression affinity of
//!     the corresponding expression in the result set" per the §2.1 mapping table:
//!     TEXT affinity -> "TEXT", NUMERIC -> "NUM", INTEGER -> "INT", REAL -> "REAL",
//!     BLOB/NONE -> "" (the empty string). Expression affinity itself is
//!     `spec/sqlite-doc/datatype3.html` §3.2: a simple reference to a column of a real
//!     table carries THAT column's affinity; an expression built with an operator, a
//!     function, or a bare literal has NO affinity (so its declared type is "").
//!
//! Note: CTAS is *parsed* (`minisqlite_sql` builds `CreateTableBody::AsSelect`
//! from a full `parse_select`) but the catalog REJECTS it in `table_def_from_ast`
//! (`crates/minisqlite-catalog/src/builder.rs`) with the error string
//! `"CREATE TABLE ... AS SELECT is not yet supported in the catalog"`. The reject
//! happens at catalog-build time, so almost every CTAS `exec()` below fails today.
//! Each test asserts the spec-correct resulting STATE of the created table, so it
//! fails now and will pass with no test change once the engine implements CTAS; the
//! spec-correct assertion is left intact rather than weakened to pass.
//!
//! ONE exception is documented at its own test (`ctas_if_not_exists_existing_table_is_noop`):
//! `CREATE TABLE IF NOT EXISTS <existing> AS SELECT ...` is a no-op that the catalog
//! short-circuits BEFORE it reaches the CTAS build, so it already behaves correctly
//! today. That test pins that the no-op is PRESERVED once CTAS lands (a naive
//! implementation that runs the SELECT before the existence check would wrongly
//! clobber or error on the existing table).

mod conformance;
use conformance::*;

// ---------------------------------------------------------------------------
// §2.1 — the new table is populated with the SELECT's rows (data copy), and its
// columns are the SELECT's result columns.
// ---------------------------------------------------------------------------

/// §2.1: `CREATE TABLE t2 AS SELECT * FROM t1` copies every row of t1 into t2. The
/// new table holds exactly the SELECT's result set.
#[test]
fn ctas_select_star_copies_all_rows() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t1(a INT, b TEXT)");
    exec(&mut db, "INSERT INTO t1 VALUES (1,'p'),(2,'q'),(3,'r')");
    exec(&mut db, "CREATE TABLE t2 AS SELECT * FROM t1");
    assert_rows_unordered(
        &mut db,
        "SELECT a, b FROM t2",
        &[
            vec![int(1), text("p")],
            vec![int(2), text("q")],
            vec![int(3), text("r")],
        ],
    );
}

/// §2.1: a projection + WHERE CTAS copies ONLY the selected columns and ONLY the
/// matching rows. Asserting `SELECT *` pins both: the row width proves column `c` was
/// not carried (t2 has just a,b), and the rows prove only a>2 was copied.
#[test]
fn ctas_projection_and_where_copies_only_selected_columns_and_rows() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t1(a INT, b TEXT, c INT)");
    exec(&mut db, "INSERT INTO t1 VALUES (1,'p',10),(2,'q',20),(3,'r',30),(4,'s',40)");
    exec(&mut db, "CREATE TABLE t2 AS SELECT a, b FROM t1 WHERE a > 2");
    assert_rows(
        &mut db,
        "SELECT * FROM t2 ORDER BY a",
        &[vec![int(3), text("r")], vec![int(4), text("s")]],
    );
}

// ---------------------------------------------------------------------------
// §2.1 — the column NAME is the SELECT result column name (AS aliases name the
// new columns; a computed expression's name comes from its alias).
// ---------------------------------------------------------------------------

/// §2.1: "The name of each column is the same as the name of the corresponding column
/// in the result set of the SELECT statement." An `AS` alias on a computed column is
/// that result column's name, so the new table's columns are named `x` and `y`.
#[test]
fn ctas_aliased_expressions_name_the_new_columns() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t1(a INT, b TEXT)");
    exec(&mut db, "INSERT INTO t1 VALUES (1,'hi'),(2,'yo')");
    exec(&mut db, "CREATE TABLE t2 AS SELECT a+1 AS x, upper(b) AS y FROM t1");
    assert_columns(&mut db, "SELECT * FROM t2", &["x", "y"]);
}

/// §2.1: the aliased computed columns are populated with the SELECT's computed VALUES
/// — `a+1` and `upper(b)` — not the source columns. (1->2,'hi'->'HI'; 2->3,'yo'->'YO'.)
#[test]
fn ctas_aliased_expressions_hold_the_computed_values() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t1(a INT, b TEXT)");
    exec(&mut db, "INSERT INTO t1 VALUES (1,'hi'),(2,'yo')");
    exec(&mut db, "CREATE TABLE t2 AS SELECT a+1 AS x, upper(b) AS y FROM t1");
    assert_rows_unordered(
        &mut db,
        "SELECT x, y FROM t2",
        &[vec![int(2), text("HI")], vec![int(3), text("YO")]],
    );
}

// ---------------------------------------------------------------------------
// §2.1 — an empty SELECT result still CREATES the table (with its columns) and
// populates it with zero rows.
// ---------------------------------------------------------------------------

/// §2.1: a CTAS whose SELECT returns no rows still creates the table and populates it
/// with the (empty) result — so `count(*)` is 0, not an error.
#[test]
fn ctas_empty_result_creates_table_with_zero_rows() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t1(a INT)");
    exec(&mut db, "INSERT INTO t1 VALUES (1),(2)");
    exec(&mut db, "CREATE TABLE t2 AS SELECT a FROM t1 WHERE 0");
    assert_scalar(&mut db, "SELECT count(*) FROM t2", int(0));
}

/// §2.1: an empty-result CTAS still creates the COLUMN, so selecting it succeeds and
/// returns zero rows — it must NOT be a "no such column" error. (An empty expected set
/// asserts the query resolves the column AND yields no rows.)
#[test]
fn ctas_empty_result_still_creates_the_column() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t1(a INT)");
    exec(&mut db, "INSERT INTO t1 VALUES (1),(2)");
    exec(&mut db, "CREATE TABLE t2 AS SELECT a FROM t1 WHERE 0");
    assert_rows(&mut db, "SELECT a FROM t2", &[]);
}

// ---------------------------------------------------------------------------
// §2.1 — the SELECT may be any query: aggregate/GROUP BY, JOIN, or a compound
// (UNION). The copy is precisely that query's result set.
// ---------------------------------------------------------------------------

/// §2.1: CTAS from a GROUP BY aggregate copies in one row per group, with the
/// aggregate's aliased column. ('a' -> 2 rows, 'b' -> 1 row.)
#[test]
fn ctas_from_group_by_aggregate() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t1(g TEXT, v INT)");
    exec(&mut db, "INSERT INTO t1 VALUES ('a',1),('a',2),('b',3)");
    exec(&mut db, "CREATE TABLE t2 AS SELECT g, count(*) AS n FROM t1 GROUP BY g");
    assert_rows_unordered(
        &mut db,
        "SELECT g, n FROM t2",
        &[vec![text("a"), int(2)], vec![text("b"), int(1)]],
    );
}

/// §2.1: CTAS from a JOIN copies in the join's result rows, with the projected
/// (aliased) columns. Each l row matches its r row on id.
#[test]
fn ctas_from_inner_join() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE l(id INT, name TEXT)");
    exec(&mut db, "CREATE TABLE r(id INT, val INT)");
    exec(&mut db, "INSERT INTO l VALUES (1,'x'),(2,'y')");
    exec(&mut db, "INSERT INTO r VALUES (1,10),(2,20)");
    exec(
        &mut db,
        "CREATE TABLE t2 AS SELECT l.name AS name, r.val AS val FROM l JOIN r ON l.id = r.id",
    );
    assert_rows_unordered(
        &mut db,
        "SELECT name, val FROM t2",
        &[vec![text("x"), int(10)], vec![text("y"), int(20)]],
    );
}

/// §2.1: CTAS from a compound SELECT (UNION) copies in the compound's result. UNION
/// removes duplicates, so {1,2} ∪ {2,3} = {1,2,3}. The compound's column name comes
/// from the first SELECT (`x`).
#[test]
fn ctas_from_union_compound() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE a(x INT)");
    exec(&mut db, "CREATE TABLE b(x INT)");
    exec(&mut db, "INSERT INTO a VALUES (1),(2)");
    exec(&mut db, "INSERT INTO b VALUES (2),(3)");
    exec(&mut db, "CREATE TABLE t2 AS SELECT x FROM a UNION SELECT x FROM b");
    assert_rows_unordered(
        &mut db,
        "SELECT x FROM t2",
        &[vec![int(1)], vec![int(2)], vec![int(3)]],
    );
}

// ---------------------------------------------------------------------------
// §2.1 — rowid assignment: "Rows are assigned contiguously ascending rowid
// values, starting with 1, in the order that they are returned by the SELECT
// statement."
// ---------------------------------------------------------------------------

/// §2.1: the copy's rows get contiguous rowids starting at 1 in the SELECT's result
/// ORDER. `SELECT a FROM t1 ORDER BY a DESC` yields 30,20,10, so t2 gets rowid 1->30,
/// 2->20, 3->10. This pins BOTH the "starts at 1, contiguous" rule and the "in the
/// order returned by the SELECT" rule — a CTAS that copied the right value SET in the
/// wrong order (or mis-seeded rowids) would fail here, where the deliberately
/// order-blind copy tests above would not.
#[test]
fn ctas_assigns_rowids_in_select_order_from_one() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t1(a INT)");
    exec(&mut db, "INSERT INTO t1 VALUES (10),(30),(20)");
    exec(&mut db, "CREATE TABLE t2 AS SELECT a FROM t1 ORDER BY a DESC");
    assert_rows(
        &mut db,
        "SELECT rowid, a FROM t2 ORDER BY rowid",
        &[
            vec![int(1), int(30)],
            vec![int(2), int(20)],
            vec![int(3), int(10)],
        ],
    );
}

// ---------------------------------------------------------------------------
// §2.1 — "no PRIMARY KEY and no constraints of any kind": the copy is a plain
// table, so a value the SOURCE would reject as a duplicate inserts fine.
// ---------------------------------------------------------------------------

/// §2.1: "no PRIMARY KEY". The source's INTEGER PRIMARY KEY (and its rowid alias) is
/// NOT carried to the copy, so re-inserting a row whose `id` duplicates an existing
/// value SUCCEEDS — leaving two rows with id=1 (the source would have rejected it).
#[test]
fn ctas_does_not_copy_primary_key() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE src(id INTEGER PRIMARY KEY, v TEXT)");
    exec(&mut db, "INSERT INTO src VALUES (1,'a'),(2,'b')");
    exec(&mut db, "CREATE TABLE t2 AS SELECT * FROM src");
    // No PK on the copy: this duplicate id must NOT raise a uniqueness error.
    exec(&mut db, "INSERT INTO t2 VALUES (1,'dup')");
    assert_scalar(&mut db, "SELECT count(*) FROM t2 WHERE id = 1", int(2));
}

/// §2.1: "no constraints of any kind". A source UNIQUE constraint is not carried, so a
/// duplicate value in that column inserts into the copy without error (3 rows total).
#[test]
fn ctas_does_not_copy_unique_constraint() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE src(x INTEGER UNIQUE, y TEXT)");
    exec(&mut db, "INSERT INTO src VALUES (1,'a'),(2,'b')");
    exec(&mut db, "CREATE TABLE t2 AS SELECT * FROM src");
    // No UNIQUE on the copy: this duplicate x must NOT raise a uniqueness error.
    exec(&mut db, "INSERT INTO t2 VALUES (1,'dup')");
    assert_scalar(&mut db, "SELECT count(*) FROM t2", int(3));
}

// ---------------------------------------------------------------------------
// CTAS with IF NOT EXISTS (lang_createtable.html §1 — the CREATE TABLE prefix
// applies to the AS SELECT form too).
// ---------------------------------------------------------------------------

/// §1/§2.1: `CREATE TABLE IF NOT EXISTS <existing> AS SELECT ...` is a no-op when the
/// table already exists — no error, and the existing table is UNCHANGED (its row stays
/// 1; the SELECT's 99 is never written).
///
/// NOTE: this case ALREADY passes today. The catalog short-circuits the
/// `IF NOT EXISTS` duplicate to `Ok(())` BEFORE it reaches the CTAS build in
/// `table_def_from_ast`, so the (unsupported) SELECT is never evaluated. The test is
/// kept because it pins that the no-op is PRESERVED once CTAS lands: an implementation
/// that ran the SELECT before the existence check would wrongly clobber or error on
/// the existing table.
#[test]
fn ctas_if_not_exists_existing_table_is_noop() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t2(a INT)");
    exec(&mut db, "INSERT INTO t2 VALUES (1)");
    exec(&mut db, "CREATE TABLE IF NOT EXISTS t2 AS SELECT 99 AS a");
    assert_rows(&mut db, "SELECT a FROM t2", &[vec![int(1)]]);
}

/// §1/§2.1: `CREATE TABLE IF NOT EXISTS <new> AS SELECT ...` on a name that does NOT
/// exist creates and populates the table via CTAS (IF NOT EXISTS does not suppress a
/// legitimate creation).
#[test]
fn ctas_if_not_exists_new_name_creates_table() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE IF NOT EXISTS t2 AS SELECT 42 AS a");
    assert_rows(&mut db, "SELECT a FROM t2", &[vec![int(42)]]);
}

// ---------------------------------------------------------------------------
// §2.1 — a literal-only SELECT (no FROM) is a valid CTAS source.
// ---------------------------------------------------------------------------

/// §2.1: a literal-only CTAS creates a one-row table holding those literal values.
#[test]
fn ctas_literal_only_copies_values() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t2 AS SELECT 1 AS a, 'x' AS b");
    assert_rows(&mut db, "SELECT a, b FROM t2", &[vec![int(1), text("x")]]);
}

/// §2.1: the literal-only SELECT's `AS` aliases name the new columns (`a`, `b`).
#[test]
fn ctas_literal_only_names_columns() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t2 AS SELECT 1 AS a, 'x' AS b");
    assert_columns(&mut db, "SELECT * FROM t2", &["a", "b"]);
}

// ---------------------------------------------------------------------------
// §2.1 — a scalar subquery in the result set is a result column like any other:
// its alias names the new column and its value is copied per row.
// ---------------------------------------------------------------------------

/// §2.1: a projected scalar subquery `(SELECT count(*) FROM t1) AS total` is a result
/// column — the new table gets a `total` column holding that value (3) for every row.
#[test]
fn ctas_scalar_subquery_projection_named_and_valued() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t1(a INT)");
    exec(&mut db, "INSERT INTO t1 VALUES (1),(2),(3)");
    exec(&mut db, "CREATE TABLE t2 AS SELECT a, (SELECT count(*) FROM t1) AS total FROM t1");
    assert_rows_unordered(
        &mut db,
        "SELECT a, total FROM t2",
        &[
            vec![int(1), int(3)],
            vec![int(2), int(3)],
            vec![int(3), int(3)],
        ],
    );
}

// ---------------------------------------------------------------------------
// §2.1 — "The declared type of each column is determined by the expression
// affinity of the corresponding expression" (the §2.1 affinity->declared-type
// table), with expression affinity per datatype3.html §3.2. Pinned via
// `PRAGMA table_info`, whose `type` column is the declared type verbatim.
// ---------------------------------------------------------------------------

/// §2.1 affinity table + datatype3 §3.2 (a bare column reference to a real table
/// carries that column's affinity): a `SELECT *` copy's declared types are the
/// affinity-derived names, NOT the source's declared-type spellings —
/// INTEGER->"INT", TEXT->"TEXT", REAL->"REAL", BLOB->"", NUMERIC->"NUM", none->"".
/// The full `table_info` row also pins §2.1's "no constraints of any kind" (notnull=0,
/// pk=0) and "default value ... is NULL" (dflt_value=NULL) for the copy.
#[test]
fn ctas_column_reference_affinity_sets_declared_types() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE src(i INTEGER, t TEXT, r REAL, b BLOB, n NUMERIC, x)");
    exec(&mut db, "CREATE TABLE t2 AS SELECT * FROM src");
    assert_rows(
        &mut db,
        "PRAGMA table_info(t2)",
        &[
            vec![int(0), text("i"), text("INT"), int(0), null(), int(0)],
            vec![int(1), text("t"), text("TEXT"), int(0), null(), int(0)],
            vec![int(2), text("r"), text("REAL"), int(0), null(), int(0)],
            vec![int(3), text("b"), text(""), int(0), null(), int(0)],
            vec![int(4), text("n"), text("NUM"), int(0), null(), int(0)],
            vec![int(5), text("x"), text(""), int(0), null(), int(0)],
        ],
    );
}

/// datatype3 §3.2: an expression built with an operator (`a+1`) or a function
/// (`upper(b)`) has NO affinity, so per the §2.1 table its CTAS column's declared type
/// is "" (the BLOB/NONE row of the mapping) — regardless of the source column types.
#[test]
fn ctas_non_column_expression_has_no_affinity() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t1(a INTEGER, b TEXT)");
    exec(&mut db, "INSERT INTO t1 VALUES (1,'hi')");
    exec(&mut db, "CREATE TABLE t2 AS SELECT a+1 AS x, upper(b) AS y FROM t1");
    assert_rows(
        &mut db,
        "PRAGMA table_info(t2)",
        &[
            vec![int(0), text("x"), text(""), int(0), null(), int(0)],
            vec![int(1), text("y"), text(""), int(0), null(), int(0)],
        ],
    );
}

/// datatype3 §3.2: "An expression of the form CAST(expr AS type) has an affinity that
/// is the same as a column with a declared type of type." So a CTAS column built from
/// `CAST(a AS TEXT)` takes TEXT affinity -> declared type "TEXT", and `CAST(a AS
/// INTEGER)` takes INTEGER affinity -> "INT" (per the §2.1 mapping) — independent of
/// the source column's own (here untyped) affinity. This is a THIRD, distinct
/// expression-affinity rule beyond the column-reference and no-affinity cases above.
#[test]
fn ctas_cast_expression_sets_affinity_declared_type() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t1(a)");
    exec(&mut db, "INSERT INTO t1 VALUES (5)");
    exec(&mut db, "CREATE TABLE t2 AS SELECT CAST(a AS TEXT) AS s, CAST(a AS INTEGER) AS i FROM t1");
    assert_rows(
        &mut db,
        "PRAGMA table_info(t2)",
        &[
            vec![int(0), text("s"), text("TEXT"), int(0), null(), int(0)],
            vec![int(1), text("i"), text("INT"), int(0), null(), int(0)],
        ],
    );
}
