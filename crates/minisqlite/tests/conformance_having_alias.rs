//! Conformance battery: HAVING referencing a SELECT-list output column ALIAS.
//!
//! `spec/sqlite-doc/lang_select.html` (GROUP BY / HAVING name resolution): an
//! unqualified identifier in the HAVING clause that does NOT name a column of the
//! queried tables is resolved, as a fallback, against the SELECT result-column
//! alias list — the same fallback GROUP BY uses ("evaluated ... according to the
//! processing rules stated below for ORDER BY expressions"). When it matches an
//! output alias, SQLite substitutes that column's SOURCE expression and binds it
//! against the post-aggregate row. So
//!   `SELECT a, count(*) c FROM t GROUP BY a HAVING c > 1`
//! resolves `c` to `count(*)` and keeps groups whose count exceeds 1.
//!
//! Regression: the planner previously bound HAVING names literally, so a bare
//! alias reference (`HAVING c > 1`) errored `no such column: c` instead of
//! resolving to the aliased aggregate — an error-vs-rows divergence from real
//! SQLite. These cases pin the alias resolution at many positions (comparison,
//! AND/NOT, arithmetic, a scalar function, BETWEEN, IN-list, CASE, CAST, COLLATE,
//! LIKE, ISNULL/NOTNULL), case-insensitive alias matching, the interaction with
//! an ordinal GROUP BY, the alias-for-a-group-key case, the non-regression of
//! alias-free HAVING clauses (the rewrite is a no-op for them), and the
//! precedence rule (an INPUT column of the same name WINS over an output alias —
//! identical to GROUP BY, unlike ORDER BY).
//!
//! They also guard a critical ordering requirement: the resolved HAVING is threaded
//! into the aggregate PRE-SCANS, not just the binder, so an alias that expands to
//! an aggregate is counted consistently. That only affects output when a §2.5
//! bare-column capture is present, so a dedicated test combines the two.
//!
//! Every expectation is derived from the SQLite documentation and hand computation
//! over the fixture, never from what the engine returns; a disagreement stays a
//! FAILING assertion rather than weakened to pass.

mod conformance;

use conformance::*;
// The harness imports `Connection` privately (no `pub use`), so bring it in
// directly for the fixture's return type.
use minisqlite::Connection;

/// The fixture:
///   a | b
///   --+----
///   1 | x
///   1 | y
///   2 | z
///   3 | x
///   3 | q
///   3 | r
/// Group counts by `a`: a=1 -> 2, a=2 -> 1, a=3 -> 3. Group sums of `a`
/// (sum(a) = a * rows-in-group): a=1 -> 2, a=2 -> 2, a=3 -> 9.
fn t_db() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
    exec(&mut db, "INSERT INTO t VALUES (1,'x'),(1,'y'),(2,'z'),(3,'x'),(3,'q'),(3,'r')");
    db
}

/// The §2.5 "bare column" fixture (same shape as `mm_db()` in
/// `conformance_select_group_having.rs`; replicated here because each
/// `tests/*.rs` file is its own binary and cannot share a sibling's fixture).
///   g | v | tag
///   --+---+----
///   a | 1 | lo
///   a | 3 | hi
///   b | 5 | x
///   b | 2 | y
/// Per group of `g`: 'a' has max(v)=3 at row tag='hi', count 2; 'b' has
/// max(v)=5 at row tag='x', count 2.
fn mm_db() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE mm(g TEXT, v INTEGER, tag TEXT)");
    exec(&mut db, "INSERT INTO mm VALUES ('a',1,'lo'),('a',3,'hi'),('b',5,'x'),('b',2,'y')");
    db
}

// ---- the core alias-in-HAVING cases ----------------------------------------

/// The headline case: a bare alias `c` (= `count(*)`) in HAVING resolves to that
/// aggregate. Keeps a=1 (count 2) and a=3 (count 3); drops a=2 (count 1).
#[test]
fn alias_in_simple_comparison() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT a, count(*) c FROM t GROUP BY a HAVING c > 1 ORDER BY a",
        &[vec![int(1), int(2)], vec![int(3), int(3)]],
    );
}

/// The alias resolves independently at EACH position of a compound predicate:
/// `c > 1 AND c < 3` becomes `count(*) > 1 AND count(*) < 3` (two distinct,
/// non-deduped aggregate calls), keeping only count == 2 => a=1.
#[test]
fn alias_in_compound_and_predicate() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT a, count(*) c FROM t GROUP BY a HAVING c > 1 AND c < 3 ORDER BY a",
        &[vec![int(1), int(2)]],
    );
}

/// The alias resolves inside arithmetic: `c + 1 > 2` <=> `count(*) > 1`, so a=1
/// and a=3 survive.
#[test]
fn alias_in_arithmetic() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT a, count(*) c FROM t GROUP BY a HAVING c + 1 > 2 ORDER BY a",
        &[vec![int(1), int(2)], vec![int(3), int(3)]],
    );
}

/// An ordinal GROUP BY and a HAVING alias together: `GROUP BY 1` groups by the
/// first output column (`a`) and `HAVING c` resolves to `count(*)`. Both the
/// ordinal and the alias resolve against the same output-column list.
#[test]
fn ordinal_group_by_with_having_alias() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT a, count(*) c FROM t GROUP BY 1 HAVING c > 1 ORDER BY 1",
        &[vec![int(1), int(2)], vec![int(3), int(3)]],
    );
}

/// An alias for a GROUP-BY key: `g` (= `a`) is the group key, and `HAVING g >= 2`
/// resolves to that key and filters on it. Keeps a=2 (count 1) and a=3 (count 3).
#[test]
fn alias_for_a_group_key() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT a AS g, count(*) FROM t GROUP BY g HAVING g >= 2 ORDER BY g",
        &[vec![int(2), int(1)], vec![int(3), int(3)]],
    );
}

/// The alias resolves inside a SCALAR function argument: `abs(c)` <=>
/// `abs(count(*))`. This is exactly the position that requires descending into a
/// function's arguments during the rewrite.
#[test]
fn alias_inside_scalar_function() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT a, count(*) c FROM t GROUP BY a HAVING abs(c) > 1 ORDER BY a",
        &[vec![int(1), int(2)], vec![int(3), int(3)]],
    );
}

// ---- alias at more positions ("at ANY position in the HAVING expression") ----

/// BETWEEN: `c BETWEEN 2 AND 3` resolves the alias in the operand and both bounds'
/// neighbor — keeps counts in [2,3] => a=1 and a=3.
#[test]
fn alias_in_between() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT a, count(*) c FROM t GROUP BY a HAVING c BETWEEN 2 AND 3 ORDER BY a",
        &[vec![int(1), int(2)], vec![int(3), int(3)]],
    );
}

/// A bare IN value-list (NOT a subquery) has its scalar operands rewritten:
/// `c IN (2, 5)` keeps count == 2 => a=1.
#[test]
fn alias_in_in_value_list() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT a, count(*) c FROM t GROUP BY a HAVING c IN (2, 5) ORDER BY a",
        &[vec![int(1), int(2)]],
    );
}

/// Inside a CASE expression: `CASE WHEN c > 1 THEN 1 ELSE 0 END = 1` keeps c > 1.
#[test]
fn alias_in_case_expression() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT a, count(*) c FROM t GROUP BY a HAVING CASE WHEN c > 1 THEN 1 ELSE 0 END = 1 ORDER BY a",
        &[vec![int(1), int(2)], vec![int(3), int(3)]],
    );
}

/// Under a NOT / parenthesized subexpression: `NOT (c = 1)` keeps count != 1.
#[test]
fn alias_under_not_and_parens() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT a, count(*) c FROM t GROUP BY a HAVING NOT (c = 1) ORDER BY a",
        &[vec![int(1), int(2)], vec![int(3), int(3)]],
    );
}

/// Two DISTINCT aggregate aliases in one HAVING: `c` (= `count(*)`) and `s`
/// (= `sum(a)`) both appear in the SELECT list AND are referenced in HAVING. The
/// HAVING substitutions add TWO further (non-deduped) aggregate calls after the
/// two projected ones; the BINDER must give each its own post-aggregate register
/// so the predicate reads the right value. `c > 1 AND s >= 6` keeps only a=3
/// (count 3, sum 9).
///
/// NOTE: this pins the *binder* side (`bind_grouped_output`) of register layout,
/// NOT the aggregate *pre-scan* threading (`single_minmax_special` /
/// `count_aggregates`). With no bare column and no correlated subquery here,
/// `bare_base`/`subplan_outer_width` are unused, so this stays passing even if the
/// pre-scans saw raw `having`. The §2.5 test below (`bare_column_and_count_alias_
/// in_having_keeps_prescan_offsets`) is what guards that pre-scan hazard.
#[test]
fn two_distinct_aggregate_aliases_in_having_resolve_independently() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT a, count(*) c, sum(a) s FROM t GROUP BY a HAVING c > 1 AND s >= 6 ORDER BY a",
        &[vec![int(3), int(3), int(9)]],
    );
}

// ---- the CRITICAL-ORDERING hazard: pre-scan threading of the resolved HAVING --
//
// The fix threads the RESOLVED HAVING (not the raw `having`) into the two
// aggregate pre-scans `single_minmax_special` and `count_aggregates`, so a HAVING
// alias that expands to an aggregate is counted by the pre-scan exactly as the
// binder later collects it. Getting this wrong corrupts output registers — but it
// is only observable when a HAVING alias resolves to an aggregate AND the query
// has a §2.5 single-min/max bare-column capture (the only shape where the
// pre-scan's `bare_base` reaches the output). The tests above never combine those.

/// §2.5 bare-column capture + a HAVING alias that resolves to a NON-min/max
/// aggregate. `SELECT g, tag, max(v) AS mx, count(*) AS c ... HAVING c > 1`:
///   * a single `max(v)` keeps the §2.5 association ON, so the bare `tag` is
///     captured from each group's max-`v` row (g='a' -> tag='hi' at v=3; g='b' ->
///     tag='x' at v=5);
///   * `c` resolves to `count(*)`, adding a THIRD aggregate. With the resolved
///     HAVING threaded into both pre-scans, `count_aggregates` sees 3 aggregates
///     and `bare_base = num_keys(1) + 3 = 4`, so the captured `tag` lands at reg 4
///     — clear of max@1, count@2, and the HAVING count@3.
///
/// This is the exact hazard the critical ordering requirement warns about:
/// revert either pre-scan (aggregate.rs `single_minmax_special` / `count_
/// aggregates`) to the raw `having` and the count undercounts by one, so the
/// captured `tag` collides with the HAVING-count register — the value goes to
/// Integer(2) (a debug build panics on the `build_minmax_bare` register assert
/// first). Either way this assertion fails while the alias-only tests still
/// pass. (A min/max alias in HAVING is deliberately avoided: it would add a
/// second, non-deduped min/max and disable §2.5, diverging from real SQLite.)
#[test]
fn bare_column_and_count_alias_in_having_keeps_prescan_offsets() {
    let mut db = mm_db();
    assert_rows(
        &mut db,
        "SELECT g, tag, max(v) AS mx, count(*) AS c FROM mm GROUP BY g HAVING c > 1 ORDER BY g",
        &[
            vec![text("a"), text("hi"), int(3), int(2)],
            vec![text("b"), text("x"), int(5), int(2)],
        ],
    );
}

// ---- alias resolution at more expression positions -------------------------
//
// The rewrite descends every `Expr` arm; these pin the alias fallback INSIDE the
// less-common nodes (CAST/COLLATE/LIKE/ISNULL/NOTNULL). A dropped inner-rewrite in
// any of those arms would re-surface `no such column: c` (a panic via `query()`),
// so each returns rows only if the alias was resolved in that position.

/// Case-INSENSITIVE alias match (`eq_ignore_ascii_case`): the SELECT alias is
/// lower-case `c`, HAVING spells it `C`. It still resolves to `count(*)`.
#[test]
fn alias_match_is_case_insensitive() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT a, count(*) c FROM t GROUP BY a HAVING C > 1 ORDER BY a",
        &[vec![int(1), int(2)], vec![int(3), int(3)]],
    );
}

/// Alias inside a `CAST`: `CAST(c AS REAL) > 1.5` -> `count(*) > 1.5` keeps the
/// groups whose count is >= 2 (a=1, a=3).
#[test]
fn alias_inside_cast() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT a, count(*) c FROM t GROUP BY a HAVING CAST(c AS REAL) > 1.5 ORDER BY a",
        &[vec![int(1), int(2)], vec![int(3), int(3)]],
    );
}

/// Alias inside a `COLLATE`: the alias `mb` (= `max(b)`) is compared under
/// `NOCASE`, so group a=3's `max(b)`='x' matches the literal 'X'. (max(b) per
/// group: a=1 -> 'y', a=2 -> 'z', a=3 -> 'x'.) Only a=3 survives.
#[test]
fn alias_inside_collate() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT a, max(b) mb, count(*) FROM t GROUP BY a HAVING mb COLLATE NOCASE = 'X' ORDER BY a",
        &[vec![int(3), text("x"), int(3)]],
    );
}

/// Alias inside a `LIKE` (left side): `mb LIKE 'x'` -> `max(b) LIKE 'x'` keeps
/// only group a=3 whose `max(b)`='x'.
#[test]
fn alias_inside_like() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT a, max(b) mb, count(*) FROM t GROUP BY a HAVING mb LIKE 'x' ORDER BY a",
        &[vec![int(3), text("x"), int(3)]],
    );
}

/// Alias inside a postfix `NOTNULL`: `c NOTNULL` (always true — a count is never
/// NULL) AND `c > 1`. The alias must resolve inside the NOTNULL arm; the net
/// filter keeps a=1 and a=3.
#[test]
fn alias_inside_notnull() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT a, count(*) c FROM t GROUP BY a HAVING c NOTNULL AND c > 1 ORDER BY a",
        &[vec![int(1), int(2)], vec![int(3), int(3)]],
    );
}

/// Alias inside a postfix `ISNULL`: `c ISNULL` -> `count(*) ISNULL`, always false
/// (a count is never NULL), so NO group survives. The empty result (rather than a
/// `no such column: c` panic) proves the alias resolved inside the ISNULL arm.
#[test]
fn alias_inside_isnull() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT a, count(*) c FROM t GROUP BY a HAVING c ISNULL ORDER BY a",
        &[],
    );
}

// ---- precedence: an input column of the same name WINS ----------------------

/// PRECEDENCE (identical to GROUP BY, unlike ORDER BY): when a HAVING name is BOTH
/// a table column and a result-set alias, the INPUT column wins. Here `count(*)`
/// is aliased `a`, but the table column `a` (the group key) also matches. `HAVING
/// a > 1` must therefore filter on the GROUP KEY a (a in {2,3}), NOT on the alias
/// `count(*)`. `ORDER BY 1` sorts by the sole output column (the count), giving
/// counts [1, 3]. Had the alias won, `HAVING count(*) > 1` would keep a in {1,3}
/// with counts [2, 3] — distinguishable at position 0.
#[test]
fn input_column_wins_over_a_same_named_output_alias() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT count(*) AS a FROM t GROUP BY a HAVING a > 1 ORDER BY 1",
        &[vec![int(1)], vec![int(3)]],
    );
}

// ---- non-regression: alias-free HAVING is byte-for-byte unaffected ----------

/// A direct aggregate in HAVING (no alias) is unchanged: `count(*) > 1` keeps a=1
/// and a=3. The rewrite has no rewritable node here, so it is a no-op.
#[test]
fn direct_aggregate_having_unchanged() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT a, count(*) FROM t GROUP BY a HAVING count(*) > 1 ORDER BY a",
        &[vec![int(1), int(2)], vec![int(3), int(3)]],
    );
}

/// A HAVING on a non-projected aggregate written out in full (no alias):
/// `sum(a) >= 6` keeps only a=3 (sum 9). Also a no-op for the rewrite.
#[test]
fn direct_sum_having_unchanged() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT a, count(*) FROM t GROUP BY a HAVING sum(a) >= 6 ORDER BY a",
        &[vec![int(3), int(3)]],
    );
}

/// A HAVING that references the group key by its REAL column name (`a`, not an
/// alias) is unchanged: `a <> 2` keeps a=1 and a=3.
#[test]
fn group_key_by_real_name_unchanged() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT a, count(*) FROM t GROUP BY a HAVING a <> 2 ORDER BY a",
        &[vec![int(1), int(2)], vec![int(3), int(3)]],
    );
}

/// A plain grouped query with NO HAVING is unaffected (all three groups returned).
#[test]
fn no_having_unchanged() {
    let mut db = t_db();
    assert_rows(
        &mut db,
        "SELECT a, count(*) FROM t GROUP BY a ORDER BY a",
        &[vec![int(1), int(2)], vec![int(2), int(1)], vec![int(3), int(3)]],
    );
}
