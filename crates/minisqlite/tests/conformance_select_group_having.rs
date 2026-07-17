//! Conformance battery: GROUP BY and HAVING semantics.
//!
//! Every expectation here is transcribed from the SQLite documentation, NOT
//! from what the engine returns. The binding sources are:
//!   * `spec/sqlite-doc/lang_select.html` §2.4 "Generation of the set of result
//!     rows" — how GROUP BY forms groups (NULLs compare equal, so all NULLs land
//!     in ONE group), how HAVING discards groups, and the rule that an aggregate
//!     query WITHOUT a GROUP BY always yields exactly one row even over zero
//!     input rows.
//!   * `spec/sqlite-doc/lang_select.html` §2.5 "Bare columns in an aggregate
//!     query" — the special min()/max() association for bare columns.
//!   * `spec/sqlite-doc/lang_select.html` (ORDER BY): "SQLite considers NULL
//!     values to be smaller than any other values for sorting purposes", so a
//!     NULL group sorts first under `ORDER BY ... ASC`.
//!   * `spec/sqlite-doc/lang_aggfunc.html` — count(*) counts rows; count(X) and
//!     sum(X)/avg(X)/min(X)/max(X) ignore NULL; sum() over no non-NULL rows is
//!     NULL and is an integer when all inputs are integers; avg() is always
//!     floating point; count(DISTINCT X) counts distinct non-NULL values.
//!
//! If the engine disagrees with a documented value, the assertion stays spec-correct
//! (it FAILS) rather than being weakened to match the engine.
//!
//! For determinism, every grouped query carries an ORDER BY on a RESULT column
//! (by name or by the `ORDER BY 1` ordinal) so the row order is fixed and
//! `assert_rows` (ordered) can be used.

mod conformance;

use conformance::*;
// The harness (`conformance/mod.rs`) imports `Connection`/`Value` privately (not
// `pub use`), so they are not in scope through `conformance::*`; the fixture
// helpers below need them, so import them directly.
use minisqlite::{Connection, Value};

/// The shared fixture used across most cases:
///   dept | amt
///   -----+-----
///    a   | 10
///    a   | 20
///    b   | 5
///    b   | 5
///    c   | 100
///    a   | NULL
/// Six rows; group 'a' contains a NULL amt, which the aggregates must ignore
/// (except count(*), which counts the row).
fn sales_db() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE sales(dept TEXT, amt INTEGER)");
    exec(
        &mut db,
        "INSERT INTO sales VALUES ('a',10),('a',20),('b',5),('b',5),('c',100),('a',NULL)",
    );
    db
}

/// An empty single-column table, used by the "aggregate over zero input rows"
/// cases.
fn et_db() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE et(x INTEGER)");
    db
}

// ---- count / sum / count(X) per group ---------------------------------------

/// count(*) returns the number of ROWS in each group (lang_aggfunc: "count(*)
/// ... returns the total number of rows in the group"), so group 'a' is 3 even
/// though one of its amt values is NULL.
#[test]
fn group_by_count_star() {
    let mut db = sales_db();
    assert_rows(
        &mut db,
        "SELECT dept, count(*) FROM sales GROUP BY dept ORDER BY dept",
        &[
            vec![text("a"), int(3)],
            vec![text("b"), int(2)],
            vec![text("c"), int(1)],
        ],
    );
}

/// sum(X) sums the non-NULL values in each group (lang_aggfunc: "return the sum
/// of all non-NULL values in the group"); the NULL amt in group 'a' is ignored,
/// so sum is 30, not NULL. All inputs are integers, so the result class is
/// INTEGER.
#[test]
fn group_by_sum_ignores_null() {
    let mut db = sales_db();
    assert_rows(
        &mut db,
        "SELECT dept, sum(amt) FROM sales GROUP BY dept ORDER BY dept",
        &[
            vec![text("a"), int(30)],
            vec![text("b"), int(10)],
            vec![text("c"), int(100)],
        ],
    );
}

/// count(X) counts only the rows where X is not NULL (lang_aggfunc: "count(X)
/// ... returns a count of the number of times that X is not NULL in a group"),
/// so group 'a' is 2 (10 and 20; the NULL is not counted).
#[test]
fn group_by_count_column_ignores_null() {
    let mut db = sales_db();
    assert_rows(
        &mut db,
        "SELECT dept, count(amt) FROM sales GROUP BY dept ORDER BY dept",
        &[
            vec![text("a"), int(2)],
            vec![text("b"), int(2)],
            vec![text("c"), int(1)],
        ],
    );
}

/// Several aggregates in a single grouped query, each evaluated across the rows
/// of its group independently.
#[test]
fn group_by_multiple_aggregates() {
    let mut db = sales_db();
    assert_rows(
        &mut db,
        "SELECT dept, count(*), sum(amt), count(amt) FROM sales GROUP BY dept ORDER BY dept",
        &[
            vec![text("a"), int(3), int(30), int(2)],
            vec![text("b"), int(2), int(10), int(2)],
            vec![text("c"), int(1), int(100), int(1)],
        ],
    );
}

// ---- HAVING -----------------------------------------------------------------

/// HAVING is evaluated once per group as a boolean; groups for which it is false
/// are discarded (§2.4). Group sums are a=30, b=10, c=100; `sum(amt) > 15` keeps
/// a and c.
#[test]
fn having_on_aggregate_sum() {
    let mut db = sales_db();
    assert_rows(
        &mut db,
        "SELECT dept, sum(amt) FROM sales GROUP BY dept HAVING sum(amt) > 15 ORDER BY dept",
        &[vec![text("a"), int(30)], vec![text("c"), int(100)]],
    );
}

/// A HAVING on a different aggregate than the one selected. Group row counts are
/// a=3, b=2, c=1; `count(*) = 1` keeps only c, whose sum is 100.
#[test]
fn having_count_selects_single_group() {
    let mut db = sales_db();
    assert_rows(
        &mut db,
        "SELECT dept, sum(amt) FROM sales GROUP BY dept HAVING count(*) = 1 ORDER BY dept",
        &[vec![text("c"), int(100)]],
    );
}

/// A HAVING clause that is a non-aggregate expression is evaluated with respect
/// to the group's rows (here the grouping column `dept`, constant within a
/// group), discarding group 'b'.
#[test]
fn having_on_grouping_column() {
    let mut db = sales_db();
    assert_rows(
        &mut db,
        "SELECT dept, sum(amt) FROM sales GROUP BY dept HAVING dept <> 'b' ORDER BY dept",
        &[vec![text("a"), int(30)], vec![text("c"), int(100)]],
    );
}

/// §2.4: "The HAVING expression may refer to values, even aggregate functions,
/// that are not in the result." Here sum(amt) drives the filter but is not
/// selected; only dept is returned.
#[test]
fn having_references_aggregate_not_in_result() {
    let mut db = sales_db();
    assert_rows(
        &mut db,
        "SELECT dept FROM sales GROUP BY dept HAVING sum(amt) > 15 ORDER BY dept",
        &[vec![text("a")], vec![text("c")]],
    );
}

/// A HAVING that no group satisfies produces zero result rows (§2.4: the number
/// of result rows equals the number of surviving groups).
#[test]
fn having_that_matches_no_group_yields_no_rows() {
    let mut db = sales_db();
    assert_rows(
        &mut db,
        "SELECT dept, count(*) FROM sales GROUP BY dept HAVING count(*) > 100 ORDER BY dept",
        &[],
    );
}

// ---- GROUP BY on an expression / a column not in the result -----------------

/// §2.4: GROUP BY may group on an expression. Grouping by `amt % 2` over the
/// non-NULL amts {10,20,5,5,100} yields the even group {10,20,100} (count 3) and
/// the odd group {5,5} (count 2). `ORDER BY 1` sorts by the first result column
/// (the parity), so 0 precedes 1.
#[test]
fn group_by_expression_parity() {
    let mut db = sales_db();
    assert_rows(
        &mut db,
        "SELECT amt%2, count(*) FROM sales WHERE amt IS NOT NULL GROUP BY amt%2 ORDER BY 1",
        &[vec![int(0), int(3)], vec![int(1), int(2)]],
    );
}

/// §2.4: "The expressions in the GROUP BY clause do not have to be expressions
/// that appear in the result." Here we group by dept but select only count(*),
/// so dept itself is not a result column. The three groups a/b/c yield counts
/// 3/2/1; ordering is done by `ORDER BY 1` (the sole result column, the count)
/// so the assertion stays ordered — we deliberately do not `ORDER BY dept`,
/// since dept is not projected. (SQLite does permit `ORDER BY` on a
/// non-projected column, but this engine currently rejects that; that ORDER BY
/// gap is out of this file's scope (covered by the `conformance_select_orderby.rs`
/// suite).)
#[test]
fn group_by_column_not_in_result() {
    let mut db = sales_db();
    assert_rows(
        &mut db,
        "SELECT count(*) FROM sales GROUP BY dept ORDER BY 1",
        &[vec![int(1)], vec![int(2)], vec![int(3)]],
    );
}

// ---- NULL grouping ----------------------------------------------------------

/// §2.4: "For the purposes of grouping rows, NULL values are considered equal."
/// So the two NULL keys collapse into ONE group. And per the ORDER BY rule
/// ("SQLite considers NULL values to be smaller than any other values for
/// sorting purposes"), that NULL group sorts first.
#[test]
fn group_by_null_forms_one_group_and_sorts_first() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE g(k)");
    exec(&mut db, "INSERT INTO g VALUES (1),(NULL),(1),(NULL),(2)");
    assert_rows(
        &mut db,
        "SELECT k, count(*) FROM g GROUP BY k ORDER BY k",
        &[
            vec![null(), int(2)],
            vec![int(1), int(2)],
            vec![int(2), int(1)],
        ],
    );
}

/// Grouping by multiple columns: rows are grouped by the tuple (dept, region),
/// and NULL region values compare equal to each other (forming one ('a', NULL)
/// group) while sorting before non-NULL regions.
#[test]
fn group_by_multiple_columns_with_null() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE sr(dept TEXT, region TEXT, amt INTEGER)");
    exec(
        &mut db,
        "INSERT INTO sr VALUES \
         ('a','north',10),('a','north',20),('a',NULL,5),('a',NULL,7),('b','south',100)",
    );
    assert_rows(
        &mut db,
        "SELECT dept, region, count(*), sum(amt) FROM sr \
         GROUP BY dept, region ORDER BY dept, region",
        &[
            vec![text("a"), null(), int(2), int(12)],
            vec![text("a"), text("north"), int(2), int(30)],
            vec![text("b"), text("south"), int(1), int(100)],
        ],
    );
}

// ---- DISTINCT within an aggregate -------------------------------------------

/// count(DISTINCT X) counts distinct non-NULL values per group (lang_aggfunc:
/// "count(distinct X) will return the number of distinct values of column X
/// instead of the total number of non-null values"). Group 'a' has amts
/// {10,20,NULL} => 2 distinct non-NULL; group 'b' has {5,5} => 1.
#[test]
fn count_distinct_per_group() {
    let mut db = sales_db();
    assert_rows(
        &mut db,
        "SELECT dept, count(DISTINCT amt) FROM sales GROUP BY dept ORDER BY dept",
        &[
            vec![text("a"), int(2)],
            vec![text("b"), int(1)],
            vec![text("c"), int(1)],
        ],
    );
}

// ---- min / max / avg per group ----------------------------------------------

/// min(X)/max(X) return the min/max non-NULL value in each group (lang_aggfunc:
/// they "return NULL if and only if there are no non-NULL values in the group").
#[test]
fn min_max_per_group_ignore_null() {
    let mut db = sales_db();
    assert_rows(
        &mut db,
        "SELECT dept, min(amt), max(amt) FROM sales GROUP BY dept ORDER BY dept",
        &[
            vec![text("a"), int(10), int(20)],
            vec![text("b"), int(5), int(5)],
            vec![text("c"), int(100), int(100)],
        ],
    );
}

/// avg(X) averages the non-NULL values and is ALWAYS floating point
/// (lang_aggfunc: "The result of avg() is always a floating point value
/// whenever there is at least one non-NULL input even if all inputs are
/// integers"). Group 'a' averages {10,20} = 15.0 (the NULL is ignored).
#[test]
fn avg_per_group_is_real_and_ignores_null() {
    let mut db = sales_db();
    assert_rows(
        &mut db,
        "SELECT dept, avg(amt) FROM sales GROUP BY dept ORDER BY dept",
        &[
            vec![text("a"), real(15.0)],
            vec![text("b"), real(5.0)],
            vec![text("c"), real(100.0)],
        ],
    );
}

/// A group whose aggregated column is entirely NULL: sum() over no non-NULL rows
/// is NULL (lang_aggfunc: "If there are no non-NULL input rows then sum()
/// returns NULL"), count(X) is 0, but count(*) still counts the rows.
#[test]
fn sum_of_all_null_group_is_null() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE sn(g TEXT, v INTEGER)");
    exec(&mut db, "INSERT INTO sn VALUES ('a',NULL),('a',NULL),('b',5)");
    assert_rows(
        &mut db,
        "SELECT g, sum(v), count(v), count(*) FROM sn GROUP BY g ORDER BY g",
        &[
            vec![text("a"), null(), int(0), int(2)],
            vec![text("b"), int(5), int(1), int(1)],
        ],
    );
}

// ---- aggregate query WITHOUT GROUP BY: exactly one row ----------------------

/// §2.4: "An aggregate query without a GROUP BY clause always returns exactly
/// one row of data." Over the whole (non-empty) table: 6 rows, sum 140.
#[test]
fn aggregate_without_group_by_returns_one_row() {
    let mut db = sales_db();
    assert_rows(
        &mut db,
        "SELECT count(*), sum(amt) FROM sales",
        &[vec![int(6), int(140)]],
    );
}

/// §2.4: the single-row guarantee holds "even if there are zero rows of input
/// data." Over an empty table, count(*) is 0 while sum() and avg() are NULL
/// (no non-NULL inputs) — but there is still exactly one result row.
#[test]
fn aggregate_without_group_by_on_empty_table_returns_one_row() {
    let mut db = et_db();
    assert_rows(
        &mut db,
        "SELECT count(*), sum(x), avg(x) FROM et",
        &[vec![int(0), null(), null()]],
    );
}

/// The same single-row rule via the scalar helper: count(*) over an empty table
/// is exactly one row holding 0.
#[test]
fn count_over_empty_table_is_one_row_zero() {
    let mut db = et_db();
    assert_scalar(&mut db, "SELECT count(*) FROM et", int(0));
}

/// In contrast to the no-GROUP-BY case, an aggregate query WITH a GROUP BY over
/// an empty table has zero groups and therefore returns zero rows (§2.4: result
/// rows == number of groups).
#[test]
fn group_by_over_empty_table_returns_no_rows() {
    let mut db = et_db();
    assert_rows(&mut db, "SELECT x, count(*) FROM et GROUP BY x ORDER BY x", &[]);
}

// ---- bare columns: documented min()/max() association (§2.5) -----------------
//
// §2.5: "If there is exactly one min() or max() aggregate in the query, then all
// bare columns in the result set take values from an input row which also
// contains the minimum or maximum." This is deterministic ONLY when the extreme
// value is unique within the group (limitation 1: if the extreme occurs on two+
// rows, the bare value may come from any of them). The `mm` fixture below makes
// every per-group min and max unique, so the associated `tag` is well-defined.
// (The subtler `HAVING amt=max(amt)` bare-column probe is intentionally NOT
// asserted because its bare value is arbitrary.)

fn mm_db() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE mm(g TEXT, v INTEGER, tag TEXT)");
    exec(
        &mut db,
        "INSERT INTO mm VALUES ('a',1,'lo'),('a',3,'hi'),('b',5,'x'),('b',2,'y')",
    );
    db
}

/// With exactly one max() aggregate, the bare `tag` column comes from each
/// group's maximum-`v` row: group 'a' max v=3 -> 'hi'; group 'b' max v=5 -> 'x'.
#[test]
fn bare_column_takes_value_from_max_row() {
    let mut db = mm_db();
    assert_rows(
        &mut db,
        "SELECT g, tag, max(v) FROM mm GROUP BY g ORDER BY g",
        &[
            vec![text("a"), text("hi"), int(3)],
            vec![text("b"), text("x"), int(5)],
        ],
    );
}

/// Symmetric case for min(): the bare `tag` comes from each group's minimum-`v`
/// row: group 'a' min v=1 -> 'lo'; group 'b' min v=2 -> 'y'.
#[test]
fn bare_column_takes_value_from_min_row() {
    let mut db = mm_db();
    assert_rows(
        &mut db,
        "SELECT g, tag, min(v) FROM mm GROUP BY g ORDER BY g",
        &[
            vec![text("a"), text("lo"), int(1)],
            vec![text("b"), text("y"), int(2)],
        ],
    );
}

// ---- §2.5 GENERAL bare columns: valid in ANY aggregate query ------------------
//
// §2.5 opens: "bare columns … are a documented feature" and "the use of bare columns
// in an aggregate query is a feature, not a bug"; the min()/max() association above is
// the ONE case with a defined source row. In EVERY OTHER aggregate query (zero min/max,
// or two-or-more) a bare column is STILL legal — it just "takes a value from an
// arbitrary row" that is "undefined and might change from one release to the next".
// So these pin what the spec DEFINES: the query SUCCEEDS (it is NOT an error — the old
// engine wrongly rejected it) and the group key + aggregate results are exact; the
// arbitrary `tag` is asserted only to be SOME row of its group, never a pinned value
// (which would test the engine's arbitrary choice, not the spec).

/// Asserts a bare-column row: group key `g` is exact, the arbitrary `tag` must be one
/// of the group's actual values (never NULL, never another group's value).
fn assert_bare_group(row: &[Value], g: &str, tag_choices: &[&str]) {
    assert!(value_eq(&row[0], &text(g)), "group key: expected {g:?}, got {:?}", row[0]);
    assert!(
        tag_choices.iter().any(|t| value_eq(&row[1], &text(t))),
        "bare tag {:?} is not one of this group's rows {tag_choices:?}",
        row[1]
    );
}

/// Zero min/max aggregates (a lone `sum()`): §2.5 still applies — the query SUCCEEDS
/// and `tag` is an arbitrary row of each group. Was wrongly the ungrouped-column error.
#[test]
fn bare_column_general_case_with_only_a_non_minmax_aggregate() {
    let mut db = mm_db();
    let qr = query(&mut db, "SELECT g, tag, sum(v) FROM mm GROUP BY g ORDER BY g");
    assert_eq!(qr.rows.len(), 2, "one row per group");
    assert_bare_group(&qr.rows[0], "a", &["lo", "hi"]);
    assert!(value_eq(&qr.rows[0][2], &int(4)), "sum(v) for 'a' = 1+3");
    assert_bare_group(&qr.rows[1], "b", &["x", "y"]);
    assert!(value_eq(&qr.rows[1][2], &int(7)), "sum(v) for 'b' = 5+2");
}

/// Two min/max aggregates: the single-min/max ASSOCIATION does not apply (the source
/// row is ambiguous between them), but the query is STILL a legal aggregate query —
/// `tag` is simply arbitrary. Was wrongly rejected as an ungrouped column.
#[test]
fn bare_column_general_case_with_two_minmax_aggregates() {
    let mut db = mm_db();
    let qr = query(&mut db, "SELECT g, tag, max(v), min(v) FROM mm GROUP BY g ORDER BY g");
    assert_eq!(qr.rows.len(), 2, "one row per group");
    assert_bare_group(&qr.rows[0], "a", &["lo", "hi"]);
    assert!(value_eq(&qr.rows[0][2], &int(3)), "max(v) for 'a'");
    assert!(value_eq(&qr.rows[0][3], &int(1)), "min(v) for 'a'");
    assert_bare_group(&qr.rows[1], "b", &["x", "y"]);
    assert!(value_eq(&qr.rows[1][2], &int(5)), "max(v) for 'b'");
    assert!(value_eq(&qr.rows[1][3], &int(2)), "min(v) for 'b'");
}

/// Exactly one min/max ALONGSIDE another aggregate still triggers §2.5 (limitation 2
/// bars only *two-or-more* min/max, not other aggregates). `count(*)` accompanies the
/// sole `max(v)`, and the bare `tag` MUST come from each group's max-`v` row — the case
/// a naive "exactly one aggregate total" rule wrongly turns into a hard error.
#[test]
fn bare_column_captured_with_one_minmax_and_another_aggregate() {
    let mut db = mm_db();
    assert_rows(
        &mut db,
        "SELECT g, tag, max(v), count(*) FROM mm GROUP BY g ORDER BY g",
        &[
            vec![text("a"), text("hi"), int(3), int(2)],
            vec![text("b"), text("x"), int(5), int(2)],
        ],
    );
}

/// The sole min/max is the SECOND aggregate (`sum(v), max(v)`): the marker's aggregate
/// index and the captured-column offset (past ALL results) must track the min/max's
/// real position, not assume index 0. `tag` still comes from the max-`v` row.
#[test]
fn bare_column_captured_when_minmax_is_not_the_first_aggregate() {
    let mut db = mm_db();
    assert_rows(
        &mut db,
        "SELECT g, tag, sum(v), max(v) FROM mm GROUP BY g ORDER BY g",
        &[
            vec![text("a"), text("hi"), int(4), int(3)],
            vec![text("b"), text("x"), int(7), int(5)],
        ],
    );
}

/// The sole min/max lives only in HAVING (`HAVING max(v) > 2`), the sole reachability
/// path with no other coverage: the bare `tag` in the SELECT list must still associate
/// with each group's max-`v` row. Both groups pass (max 3 and 5 are > 2). Deleting the
/// HAVING scan in the detector would make this wrongly reject `tag`.
#[test]
fn bare_column_captured_when_minmax_is_only_in_having() {
    let mut db = mm_db();
    assert_rows(
        &mut db,
        "SELECT g, tag FROM mm GROUP BY g HAVING max(v) > 2 ORDER BY g",
        &[
            vec![text("a"), text("hi")],
            vec![text("b"), text("x")],
        ],
    );
}

/// The min/max sits inside a NESTED expression (`max(v) + 0`), not a top-level call —
/// which is exactly why the detector recurses through Binary/Case/etc. rather than
/// scanning only top-level result columns. `tag` still comes from the max-`v` row.
#[test]
fn bare_column_captured_with_minmax_in_a_nested_expression() {
    let mut db = mm_db();
    assert_rows(
        &mut db,
        "SELECT g, tag, max(v) + 0 FROM mm GROUP BY g ORDER BY g",
        &[
            vec![text("a"), text("hi"), int(3)],
            vec![text("b"), text("x"), int(5)],
        ],
    );
}

// ---- §2.5 general bare columns that are FUNCTIONALLY DEPENDENT on the key -----
//
// When the bare column is UNIQUELY determined by the group key (every row of a group
// shares one value — e.g. grouping by a table's primary key, or by a column another
// column functionally depends on), §2.5's "arbitrary row" is deterministic: ANY row
// of the group yields the same value, so the result is fully defined regardless of
// which row SQLite (or this engine) happens to pick. These are the high-value general
// bare-column cases — a `GROUP BY <unique key>, <dependent bare column>` pattern that
// real applications rely on and that the old engine WRONGLY rejected outside the
// single-min/max case. `fd_db` groups by the primary key `id`, on which `name` (and
// the joined child aggregate) functionally depend.

fn fd_db() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE a(id INTEGER PRIMARY KEY, name TEXT)");
    exec(&mut db, "CREATE TABLE b(aid INTEGER, val INTEGER)");
    exec(&mut db, "INSERT INTO a VALUES (1,'x'),(2,'y'),(3,'z')");
    exec(&mut db, "INSERT INTO b VALUES (1,10),(1,20),(2,5)");
    db
}

/// The canonical real-world shape: group by the parent's primary key and select a bare
/// non-key parent column beside an aggregate over the child. `name` is functionally
/// dependent on `id`, so it is well-defined. Old engine: rejected outside §2.5 min/max.
#[test]
fn bare_column_functionally_dependent_with_count_over_left_join() {
    let mut db = fd_db();
    assert_rows(
        &mut db,
        "SELECT a.name, count(b.val) FROM a LEFT JOIN b ON a.id=b.aid GROUP BY a.id ORDER BY a.name",
        &[
            vec![text("x"), int(2)],
            vec![text("y"), int(1)],
            vec![text("z"), int(0)],
        ],
    );
}

/// Two aggregates beside a functionally-dependent bare column: the captured column's
/// offset is `num_keys + 2` (past BOTH aggregate results), so this pins the general
/// offset arithmetic just as the min/max `not_the_first_aggregate` case does.
#[test]
fn bare_column_functionally_dependent_with_two_aggregates() {
    let mut db = fd_db();
    assert_rows(
        &mut db,
        "SELECT a.name, count(b.val), sum(b.val) \
         FROM a LEFT JOIN b ON a.id=b.aid GROUP BY a.id ORDER BY a.name",
        &[
            vec![text("x"), int(2), int(30)],
            vec![text("y"), int(1), int(5)],
            vec![text("z"), int(0), null()],
        ],
    );
}

/// A GROUP BY with NO aggregate at all and a bare non-key column: a legal aggregate
/// query (the GROUP BY makes it one) whose bare `name` is functionally dependent on the
/// key `id`. Equivalent to `SELECT DISTINCT id, name` here, and defined for that reason.
#[test]
fn bare_column_functionally_dependent_with_no_aggregate() {
    let mut db = fd_db();
    assert_rows(
        &mut db,
        "SELECT id, name FROM a GROUP BY id ORDER BY id",
        &[
            vec![int(1), text("x")],
            vec![int(2), text("y")],
            vec![int(3), text("z")],
        ],
    );
}

/// An implicit aggregate (no GROUP BY) over an EMPTY table still yields exactly one row
/// (§2.4); its bare column has no input row to draw from, so it is NULL while `count(*)`
/// is 0. Pins the empty-group branch of the general capture (captured = None → NULL).
#[test]
fn bare_column_over_empty_implicit_group_is_null() {
    let mut db = fd_db();
    exec(&mut db, "CREATE TABLE empty(x INTEGER, y TEXT)");
    assert_rows(&mut db, "SELECT y, count(*) FROM empty", &[vec![null(), int(0)]]);
}

// ---- result column naming with aliases --------------------------------------

/// Aliases give the grouped result columns their well-defined names (the bare
/// aggregate column name is otherwise unspecified, so only the aliased form is
/// asserted).
#[test]
fn group_by_result_uses_column_aliases() {
    let mut db = sales_db();
    assert_columns(
        &mut db,
        "SELECT dept AS d, count(*) AS n FROM sales GROUP BY dept ORDER BY dept",
        &["d", "n"],
    );
}
