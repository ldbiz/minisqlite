//! Conformance battery: GROUP BY integer-ordinal and output-alias resolution.
//!
//! `spec/sqlite-doc/lang_select.html` (GROUP BY / HAVING processing): "each of the
//! expressions specified as part of the GROUP BY clause is evaluated for each row of
//! the dataset **according to the processing rules stated below for ORDER BY
//! expressions**." Those ORDER BY rules are: (1) a constant positive integer K is an
//! alias for the K-th result column (1-based); (2) an identifier matching an
//! output-column alias is an alias for that column; (3) otherwise it is an arbitrary
//! expression. So `GROUP BY 1` groups by the FIRST output column's source expression
//! (NOT the constant integer 1), and `GROUP BY <alias>` groups by that aliased column.
//!
//! Regression: the planner previously bound each GROUP BY key literally, so
//! `SELECT a, count(*) FROM t GROUP BY 1` grouped by the constant 1 — every row in ONE
//! group — instead of by `a`, and `GROUP BY <alias>` errored `no such column`. These
//! cases pin the ORDER-BY-style resolution, the precedence (an input column of the same
//! name wins over an output alias), the "only INTEGER is an ordinal" rule (a
//! TRUE/FALSE/REAL/TEXT/NULL/subquery constant stays a single-group constant), and the
//! out-of-range error wording.
//!
//! Every expectation is derived from the SQLite documentation, never from what the
//! engine returns; a disagreement stays a FAILING assertion rather than weakened to pass.

mod conformance;

use conformance::*;
// The harness imports these privately (no `pub use`), so bring them in directly for the
// fixtures, the two-query equivalence helper, and the error-message matches below.
use minisqlite::{Connection, Error, QueryResult, Value};

/// The shared fixture:
///   a | b
///   --+----
///   1 | 10
///   1 | 20
///   2 | 20
///   2 | 20
///   3 | 30
/// Three distinct `a` values (counts 2, 2, 1); `a%2` splits into {a=2}(count 2, parity 0)
/// and {a=1,a=1,a=3}(count 3, parity 1); `a+b` has four distinct sums (22 twice).
fn t_db() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1,10),(1,20),(2,20),(2,20),(3,30)");
    db
}

/// Assert two query results have identical rows (same shape, every cell `value_eq`).
/// Used for the ordinal-vs-explicit-expression equivalences: `GROUP BY 1` must produce
/// exactly what `GROUP BY <first output source>` produces.
fn assert_same_rows(a: &QueryResult, b: &QueryResult) {
    assert_eq!(a.rows.len(), b.rows.len(), "row count differs:\n  a: {:?}\n  b: {:?}", a.rows, b.rows);
    for (i, (ra, rb)) in a.rows.iter().zip(b.rows.iter()).enumerate() {
        assert_eq!(ra.len(), rb.len(), "row {i} width differs:\n  a: {ra:?}\n  b: {rb:?}");
        for (j, (x, y)) in ra.iter().zip(rb.iter()).enumerate() {
            assert!(value_eq(x, y), "row {i} col {j} differs: {x:?} vs {y:?}");
        }
    }
}

// ---- integer ordinals -------------------------------------------------------

/// `GROUP BY 1` groups by the FIRST output column's source (`a`), NOT the constant 1.
/// Three distinct `a` values => three groups (the old bug collapsed this to ONE row).
#[test]
fn ordinal_one_groups_by_first_output_column() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT a, count(*) FROM t GROUP BY 1 ORDER BY 1",
        &[
            vec![int(1), int(2)],
            vec![int(2), int(2)],
            vec![int(3), int(1)],
        ],
    );
}

/// `GROUP BY 1` follows the SOURCE expression of output column 1 — here `a%2` — so it
/// groups by parity: parity 0 is the two `a=2` rows (count 2), parity 1 the two `a=1`
/// plus the `a=3` row (count 3).
#[test]
fn ordinal_one_groups_by_first_column_expression() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT a%2 AS p, count(*) FROM t GROUP BY 1 ORDER BY 1",
        &[vec![int(0), int(2)], vec![int(1), int(3)]],
    );
}

/// Two ordinals `GROUP BY 1, 2` group by the tuple (`a`, `b`): (1,10),(1,20),(2,20)×2,
/// (3,30) => four groups, the (2,20) group having count 2.
#[test]
fn ordinals_one_two_group_by_the_tuple() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT a, b, count(*) FROM t GROUP BY 1, 2 ORDER BY 1, 2",
        &[
            vec![int(1), int(10), int(1)],
            vec![int(1), int(20), int(1)],
            vec![int(2), int(20), int(2)],
            vec![int(3), int(30), int(1)],
        ],
    );
}

/// Mixed forms in one GROUP BY: an ordinal and a bare column (`GROUP BY 1, b`) resolve
/// independently — column 1 (`a`) and the input column `b` — same tuple grouping as
/// `GROUP BY 1, 2` / `GROUP BY a, b`.
#[test]
fn mixed_ordinal_and_column_group_by() {
    let mut db = t_db();
    let via_mixed = query(&mut db, "SELECT a, b, count(*) FROM t GROUP BY 1, b ORDER BY 1, 2");
    let via_columns = query(&mut db, "SELECT a, b, count(*) FROM t GROUP BY a, b ORDER BY a, b");
    assert_same_rows(&via_mixed, &via_columns);
}

// ---- output-column aliases --------------------------------------------------

/// `GROUP BY <alias>` where the alias names an output column groups by that column's
/// source (`a`). Old bug: errored `no such column: z`.
#[test]
fn alias_groups_by_the_aliased_column() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT a AS z, count(*) FROM t GROUP BY z ORDER BY z",
        &[
            vec![int(1), int(2)],
            vec![int(2), int(2)],
            vec![int(3), int(1)],
        ],
    );
}

/// An alias for an EXPRESSION output column: `GROUP BY s` groups by `a+b`. Four distinct
/// sums; 22 occurs twice ((2,20) twice).
#[test]
fn alias_groups_by_the_expression_it_names() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT a+b AS s, count(*) FROM t GROUP BY s ORDER BY s",
        &[
            vec![int(11), int(1)],
            vec![int(21), int(1)],
            vec![int(22), int(2)],
            vec![int(33), int(1)],
        ],
    );
}

/// PRECEDENCE: a name that IS an input column groups by the INPUT column, even when a
/// same-named output alias exists. `SELECT b AS a ... GROUP BY a` groups by the table
/// column `a` (three groups), NOT by the alias `a` (which projects `b`, whose two
/// distinct values would give a different grouping). This pins that the input column
/// wins — the documented GROUP BY precedence (unlike ORDER BY, where the alias wins).
#[test]
fn input_column_wins_over_a_same_named_output_alias() {
    let mut db = t_db();
    // `b AS a` makes an OUTPUT column named "a", but the GROUP BY name "a" also matches the
    // INPUT column `a`; the input column must win. Grouping by input `a` gives 3 groups with
    // row counts {2,2,1}. (Had the alias won, it would group by `b`: b=10(1), b=20(3),
    // b=30(1) => counts {1,3,1}, distinguishable below.) `ORDER BY 2` sorts by the count
    // column, so the count multiset is deterministic regardless of the §2.5 bare `b` value.
    let qr = query(&mut db, "SELECT b AS a, count(*) FROM t GROUP BY a ORDER BY 2");
    assert_eq!(qr.rows.len(), 3, "grouped by table column a => 3 groups, not by alias");
    let counts: Vec<Value> = qr.rows.iter().map(|r| r[1].clone()).collect();
    // Ascending by count: [1, 2, 2] for input-column `a`; the alias-`b` reading would be
    // [1, 1, 3], which fails at counts[1].
    assert!(value_eq(&counts[0], &int(1)), "smallest a-group count is 1 (a=3)");
    assert!(value_eq(&counts[1], &int(2)), "input-column grouping => second count is 2, not 1");
    assert!(value_eq(&counts[2], &int(2)));
}

// ---- equivalence: ordinal/alias == the underlying expression ----------------

/// `... GROUP BY 1 ORDER BY 1` returns identically to `... GROUP BY a ORDER BY a` — the
/// ordinal is a pure alias for the first output column.
#[test]
fn ordinal_equivalent_to_the_named_column() {
    let mut db = t_db();
    let via_ordinal = query(&mut db, "SELECT a, count(*) FROM t GROUP BY 1 ORDER BY 1");
    let via_column = query(&mut db, "SELECT a, count(*) FROM t GROUP BY a ORDER BY a");
    assert_same_rows(&via_ordinal, &via_column);
}

/// `GROUP BY 1` over an expression column equals `GROUP BY <that expression>`.
#[test]
fn ordinal_over_expression_equivalent_to_the_expression() {
    let mut db = t_db();
    let via_ordinal = query(&mut db, "SELECT a%2 AS p, count(*) FROM t GROUP BY 1 ORDER BY 1");
    let via_expr = query(&mut db, "SELECT a%2 AS p, count(*) FROM t GROUP BY a%2 ORDER BY a%2");
    assert_same_rows(&via_ordinal, &via_expr);
}

// ---- `SELECT *` ordinals ----------------------------------------------------

/// `SELECT *, count(*) ... GROUP BY 1` groups by the FIRST star-expanded column (`a`) —
/// the star column's synthesized bare-column source binds to that column. The star also
/// projects `b`, a §2.5 bare column of each a-group (unasserted, arbitrary within a
/// group; here every a-group's b happens to make the count the load-bearing value).
#[test]
fn ordinal_one_over_star_groups_by_first_expanded_column() {
    let mut db = t_db();
    let qr = query(&mut db, "SELECT *, count(*) FROM t GROUP BY 1 ORDER BY 1");
    assert_eq!(qr.rows.len(), 3, "grouped by the first star column a => 3 groups");
    // Column 0 is `a` (1,2,3); the last column is count(*) (2,2,1). Column 1 (`b`) is an
    // arbitrary row of each group, so it is not asserted.
    assert!(value_eq(&qr.rows[0][0], &int(1)));
    assert!(value_eq(&qr.rows[0][2], &int(2)));
    assert!(value_eq(&qr.rows[1][0], &int(2)));
    assert!(value_eq(&qr.rows[1][2], &int(2)));
    assert!(value_eq(&qr.rows[2][0], &int(3)));
    assert!(value_eq(&qr.rows[2][2], &int(1)));
}

// ---- only INTEGER is an ordinal: constants stay ONE group -------------------

/// Each non-integer constant GROUP BY term is an ordinary constant expression, so every
/// row lands in ONE group and `count(*)` is the whole-table count (5). A REAL `2.0` is
/// NOT the ordinal 2 (which would be out of range for a one-column result); TRUE/FALSE
/// are the boolean literals, not ordinals; NULL and a scalar subquery likewise.
#[test]
fn non_integer_constants_form_a_single_group() {
    let cases = [
        "SELECT count(*) FROM t GROUP BY 'x'",
        "SELECT count(*) FROM t GROUP BY 2.0",
        "SELECT count(*) FROM t GROUP BY TRUE",
        "SELECT count(*) FROM t GROUP BY FALSE",
        "SELECT count(*) FROM t GROUP BY NULL",
        "SELECT count(*) FROM t GROUP BY (SELECT 1)",
    ];
    for sql in cases {
        let mut db = t_db();
        // Exactly one group over all five rows.
        assert_rows(&mut db, sql, &[vec![int(5)]]);
    }
}

/// A negative literal `-1` parses as unary-minus over `Integer(1)`, NOT an `Integer`
/// node, so it is a constant expression (one group), never ordinal 1.
#[test]
fn negative_literal_is_not_an_ordinal() {
    let mut db = t_db();
    assert_rows(&mut db, "SELECT count(*) FROM t GROUP BY -1", &[vec![int(5)]]);
}

/// The SUBTLE only-INTEGER case: `1.0` is a REAL literal, never the ordinal 1. Over a
/// multi-column result `GROUP BY 1.0` is a constant term => ONE group over all rows; had a
/// REAL been mistreated as an ordinal it would resolve to column 1 (`a`) and return THREE
/// rows. (`2.0` above only proves REAL is excluded because it would be out of range; `1.0`
/// is the in-range value that would silently mis-group.)
#[test]
fn real_literal_one_point_zero_is_not_an_ordinal() {
    let mut db = t_db();
    let qr = query(&mut db, "SELECT a, count(*) FROM t GROUP BY 1.0");
    assert_eq!(qr.rows.len(), 1, "a REAL GROUP BY term is a constant => a single group");
    assert!(value_eq(&qr.rows[0][1], &int(5)), "the single group counts all five rows");
}

// ---- out-of-range ordinals: loud error --------------------------------------

/// `GROUP BY 0` (below 1) is a loud error naming the term's 1-based position and the
/// valid range (`should be between 1 and <num output columns>`).
#[test]
fn ordinal_zero_is_out_of_range() {
    let mut db = t_db();
    let e = assert_query_error(&mut db, "SELECT a FROM t GROUP BY 0");
    match e {
        Error::Sql(ref msg)
            if msg.contains("1st GROUP BY term out of range - should be between 1 and 1") => {}
        other => panic!("expected a GROUP BY out-of-range error, got: {other:?}"),
    }
}

/// `GROUP BY 2` against a single-output-column result (`SELECT a`) is above the range.
#[test]
fn ordinal_above_range_is_out_of_range() {
    let mut db = t_db();
    let e = assert_query_error(&mut db, "SELECT a FROM t GROUP BY 2");
    match e {
        Error::Sql(ref msg)
            if msg.contains("1st GROUP BY term out of range - should be between 1 and 1") => {}
        other => panic!("expected a GROUP BY out-of-range error, got: {other:?}"),
    }
}

/// The offending term's POSITION is reported: the SECOND GROUP BY term out of range over
/// a two-column result yields "2nd GROUP BY term out of range - should be between 1 and 2".
#[test]
fn out_of_range_reports_the_offending_term_position() {
    let mut db = t_db();
    let e = assert_query_error(&mut db, "SELECT a, b FROM t GROUP BY 1, 5");
    match e {
        Error::Sql(ref msg)
            if msg.contains("2nd GROUP BY term out of range - should be between 1 and 2") => {}
        other => panic!("expected a positioned GROUP BY out-of-range error, got: {other:?}"),
    }
}

/// The range check counts STAR-EXPANDED output columns, not select-list items: `*` expands
/// to (a, b), so `SELECT *, count(*)` has THREE output columns and `GROUP BY 4` is out of
/// range against width 3 (the range would be 1..2 if the literal two select items were
/// counted instead of the expanded columns).
#[test]
fn star_ordinal_out_of_range_uses_the_expanded_width() {
    let mut db = t_db();
    let e = assert_query_error(&mut db, "SELECT *, count(*) FROM t GROUP BY 4");
    match e {
        Error::Sql(ref msg)
            if msg.contains("1st GROUP BY term out of range - should be between 1 and 3") => {}
        other => panic!("expected an out-of-range error over the expanded width 3, got: {other:?}"),
    }
}

/// An ordinal that resolves to an AGGREGATE output column is a loud error, not a silent
/// group. `SELECT count(*) FROM t GROUP BY 1` resolves to output column 1 (`count(*)`);
/// grouping by an aggregate is illegal (SQLite: "aggregate functions are not allowed in the
/// GROUP BY clause"), and this engine raises the equivalent "misuse of aggregate function"
/// while binding the group key. Crucially it must ERROR — the old constant-`1` bug instead
/// returned a single `count` with no error, so this pins the fix from the other side.
#[test]
fn ordinal_landing_on_an_aggregate_output_column_errors() {
    let mut db = t_db();
    let e = assert_query_error(&mut db, "SELECT count(*) FROM t GROUP BY 1");
    match e {
        Error::Sql(ref msg) if msg.to_ascii_lowercase().contains("aggregate") => {}
        other => panic!("expected an aggregate-misuse error, got: {other:?}"),
    }
}

// ---- collation carried by an ordinal ----------------------------------------

/// An ordinal that resolves to a `COLLATE NOCASE` output column carries that column's
/// collation, so grouping is case-insensitive: 'A' and 'a' collapse into one group. The
/// ordinal form must agree row-for-row with the explicit `GROUP BY s COLLATE NOCASE`.
#[test]
fn ordinal_to_collate_nocase_output_column_groups_case_insensitively() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE tc(s TEXT)");
    exec(&mut db, "INSERT INTO tc VALUES ('A'),('a'),('B')");
    let via_ordinal =
        query(&mut db, "SELECT s COLLATE NOCASE, count(*) FROM tc GROUP BY 1 ORDER BY 2 DESC, 1");
    let via_expr = query(
        &mut db,
        "SELECT s COLLATE NOCASE, count(*) FROM tc GROUP BY s COLLATE NOCASE ORDER BY 2 DESC, 1",
    );
    assert_eq!(via_ordinal.rows.len(), 2, "NOCASE ordinal groups 'A' and 'a' together => 2 groups");
    assert!(value_eq(&via_ordinal.rows[0][1], &int(2)), "merged {{A,a}} group has count 2");
    assert!(value_eq(&via_ordinal.rows[1][1], &int(1)), "singleton {{B}} group has count 1");
    assert_same_rows(&via_ordinal, &via_expr);
}
