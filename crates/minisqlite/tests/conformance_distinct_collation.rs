//! Conformance tests: `DISTINCT`, aggregate-`DISTINCT`, `min`/`max`, and
//! compound-`SELECT` duplicate elimination / extremum selection honour per-column /
//! explicit collation — not BINARY always.
//!
//! Every expected value is derived from the SQLite manual, never from what this
//! engine returns:
//!   * `spec/sqlite-doc/datatype3.html` §7 / §7.1 — the built-in collating
//!     functions (BINARY/NOCASE/RTRIM) and the ordered rules for which collating
//!     function a comparison uses (explicit postfix COLLATE, else a column's
//!     declared collation, else BINARY).
//!   * `spec/sqlite-doc/lang_select.html` §2.6 "DISTINCT processing" — DISTINCT
//!     removes rows that compare equal under `IS DISTINCT FROM`, i.e. it USES the
//!     §7.1 collation rules (an explicit COLLATE included). Aggregate `DISTINCT`
//!     dedups its argument tuple the same way (its argument's §7.1 collation).
//!   * `spec/sqlite-doc/lang_aggfunc.html` — `min(X)`/`max(X)` return the value
//!     that would sort first/last in an `ORDER BY X`, and `ORDER BY` uses the §7.1
//!     collation of `X`. So on a NOCASE/RTRIM column (or with an explicit COLLATE
//!     on the argument) the extremum is chosen under that collation, not bytewise.
//!     The single-min/max bare-column capture (lang_select.html §2.5) picks the
//!     row of that SAME extremum, so it must agree.
//!   * `spec/sqlite-doc/lang_corefunc.html` max_scalar / min_scalar — the
//!     multi-argument `min`/`max` "searches its arguments from left to right for an
//!     argument that defines a collating function and uses that collating function for
//!     all string comparisons", else BINARY. So `max('B' COLLATE NOCASE, 'a')` compares
//!     under NOCASE ('B'>'a' folded) and returns 'B', not the BINARY 'a'.
//!   * `spec/sqlite-doc/lang_select.html` compound-SELECT "duplicate rows"
//!     paragraph — UNION/INTERSECT/EXCEPT compare "as if the columns ... were the
//!     operands of the equals (=) operator, EXCEPT that greater precedence is NOT
//!     assigned to a collation sequence specified with the postfix COLLATE operator ...
//!     No affinity transformations are applied." So each arm's output DEFINES a
//!     collation (an explicit postfix COLLATE, else the column's declared collation);
//!     the arms fold left→right (first that defines one wins). A postfix COLLATE is
//!     honored for its OWN arm but earns NO precedence over a column collation in an
//!     EARLIER arm ("no greater precedence") — it is not ignored. No affinity applies.
//!
//! These pin the fix for a SILENT correctness bug: the operators used to dedup /
//! compare under BINARY unconditionally, disagreeing with `GROUP BY` (the one dedup
//! operator that was already collation-correct). The expected value is the spec's.

mod conformance;

use conformance::*;
use minisqlite::Connection;

/// The canonical NOCASE fixture: a `TEXT COLLATE NOCASE` column with rows
/// `'B','a','C','A'` (insertion = rowid = scan order). Under NOCASE the 'a'/'A'
/// pair folds to one value, so the three distinct groups are {B, a, C} and the
/// first-seen representative of the folded group is `'a'` (rowid 2).
fn t_nocase() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(b TEXT COLLATE NOCASE)");
    exec(&mut db, "INSERT INTO t VALUES ('B'),('a'),('C'),('A')");
    db
}

/// The BINARY control fixture: same rows, but the column has the default BINARY
/// collation, so all four values are distinct.
fn u_binary() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE u(b TEXT)");
    exec(&mut db, "INSERT INTO u VALUES ('B'),('a'),('C'),('A')");
    db
}

/// Number of rows a query returns. Used for the compound cases so the assertion
/// is on cardinality (what these cases pin) and independent of WHICH folded
/// representative a UNION emits.
fn nrows(db: &mut Connection, sql: &str) -> usize {
    query(db, sql).rows.len()
}

// -----------------------------------------------------------------------------
// FIX 1 + FIX 2 — DISTINCT / aggregate-DISTINCT honour the column's collation.
// -----------------------------------------------------------------------------

/// Case 1: on a NOCASE column, aggregate `count(DISTINCT b)` folds case, so
/// the four rows 'B','a','C','A' collapse to the three distinct values {b,a,c} =>
/// 3 (§2.6 + §7.1 rule 2). The BINARY-always bug returned 4.
#[test]
fn count_distinct_on_nocase_column_folds_case() {
    let mut db = t_nocase();
    assert_scalar(&mut db, "SELECT count(DISTINCT b) FROM t", int(3));
}

/// Case 1 (row-level DISTINCT): `SELECT DISTINCT b` on the NOCASE column
/// likewise yields three rows; counting the derived table gives 3.
#[test]
fn select_distinct_on_nocase_column_folds_case() {
    let mut db = t_nocase();
    assert_scalar(&mut db, "SELECT count(*) FROM (SELECT DISTINCT b FROM t)", int(3));
}

/// Case 2 (BINARY control): the SAME rows in a default-collation column stay
/// four distinct values under `count(DISTINCT)` — the fix must not touch BINARY
/// dedup. Paired with case 1 this pins that the folding above IS the column
/// collation at work, not a blanket change.
#[test]
fn count_distinct_on_binary_column_keeps_all() {
    let mut db = u_binary();
    assert_scalar(&mut db, "SELECT count(DISTINCT b) FROM u", int(4));
}

/// Case 3: an explicit `COLLATE NOCASE` on the aggregate argument is now
/// honoured (§7.1 rule 1: explicit COLLATE wins), so `count(DISTINCT b COLLATE
/// NOCASE)` over the BINARY column folds to 3. The bug ignored the explicit
/// COLLATE and returned 4.
#[test]
fn count_distinct_explicit_collate_on_binary_column_folds() {
    let mut db = u_binary();
    assert_scalar(&mut db, "SELECT count(DISTINCT b COLLATE NOCASE) FROM u", int(3));
}

/// Case 4: `SELECT DISTINCT b` and `SELECT b GROUP BY b` MUST return the same
/// multiset — SQLite defines DISTINCT as grouping by the same §7.1 collation. Both
/// keep the first-seen row of each folded group (scan order B,a,C,A), so both are
/// exactly {B, a, C}. Asserting both against the same literal proves they agree.
#[test]
fn distinct_and_group_by_agree_under_nocase() {
    let mut db = t_nocase();
    let expected = &[vec![text("B")], vec![text("a")], vec![text("C")]];
    assert_rows_unordered(&mut db, "SELECT DISTINCT b FROM t", expected);
    assert_rows_unordered(&mut db, "SELECT b FROM t GROUP BY b", expected);
}

/// Case 5 (RTRIM): a `COLLATE RTRIM` column treats 'a' and 'a ' as equal
/// (trailing spaces ignored, §7), so `count(DISTINCT b)` is 1 — exercising the
/// aggregate-DISTINCT path under a non-NOCASE collation.
#[test]
fn count_distinct_on_rtrim_column_ignores_trailing_space() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE tr(b TEXT COLLATE RTRIM)");
    exec(&mut db, "INSERT INTO tr VALUES ('a'),('a ')");
    assert_scalar(&mut db, "SELECT count(DISTINCT b) FROM tr", int(1));
}

/// Case 5 (RTRIM, row-level DISTINCT): the same fold applies to
/// `SELECT DISTINCT b`, so the derived table has one row.
#[test]
fn select_distinct_on_rtrim_column_ignores_trailing_space() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE tr(b TEXT COLLATE RTRIM)");
    exec(&mut db, "INSERT INTO tr VALUES ('a'),('a ')");
    assert_scalar(&mut db, "SELECT count(*) FROM (SELECT DISTINCT b FROM tr)", int(1));
}

/// A typeless column holding `1` (INTEGER), `1.0` (REAL), `2` (INTEGER). The
/// column has NONE affinity, so `1.0` is stored as REAL — the cross-storage-class
/// pair `1`/`1.0` that numeric value-folding must collapse.
///
/// (A literal `(VALUES (1),(1.0),(2)) AS s(v)` form needs a
/// column-aliased VALUES derived table, which this engine's parser does not yet
/// accept — a separate parser gap — so the SAME invariant is pinned
/// through a base table here.)
fn nums_1_1r_2() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE nums(v)");
    exec(&mut db, "INSERT INTO nums VALUES (1),(1.0),(2)");
    db
}

/// Case 8: numeric DISTINCT is unchanged by the collation fix — collation
/// only folds TEXT. `1` and `1.0` compare numerically equal (`cell_key` value-
/// folding), so `DISTINCT v` over {1, 1.0, 2} is {1, 2} => 2. A regression that
/// keyed numbers under a text collation would wrongly return 3.
#[test]
fn distinct_numeric_value_folding_unchanged() {
    let mut db = nums_1_1r_2();
    assert_scalar(&mut db, "SELECT count(*) FROM (SELECT DISTINCT v FROM nums)", int(2));
}

/// Numeric aggregate-DISTINCT likewise folds `1`/`1.0` and keeps `2` distinct: the
/// aggregate path's `cell_key` value-folding is collation-independent.
#[test]
fn count_distinct_numeric_value_folding_unchanged() {
    let mut db = nums_1_1r_2();
    assert_scalar(&mut db, "SELECT count(DISTINCT v) FROM nums", int(2));
}

// -----------------------------------------------------------------------------
// FIX 3 — compound SELECT (UNION/INTERSECT/EXCEPT) uses the COLUMN collation.
// -----------------------------------------------------------------------------

/// The compound fixture: a NOCASE left column and a BINARY right column, each with
/// one case-variant value.
fn l_nocase_r_binary() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE l(x TEXT COLLATE NOCASE)");
    exec(&mut db, "INSERT INTO l VALUES ('A')");
    exec(&mut db, "CREATE TABLE r(y TEXT)");
    exec(&mut db, "INSERT INTO r VALUES ('a')");
    db
}

/// Case 6 (UNION): the compound dedup uses the LEFT column's collation
/// (left-arm precedence). `x` is NOCASE, so 'A' and 'a' fold and the UNION yields
/// ONE row. The BINARY-always bug returned two.
#[test]
fn compound_union_uses_left_column_nocase() {
    let mut db = l_nocase_r_binary();
    assert_eq!(nrows(&mut db, "SELECT x FROM l UNION SELECT y FROM r"), 1);
}

/// Case 6 (UNION ALL): `UNION ALL` never dedups, so both rows survive
/// regardless of collation — a control that the fold above is the UNION dedup, not
/// something applied to the rows themselves.
#[test]
fn compound_union_all_never_dedups() {
    let mut db = l_nocase_r_binary();
    assert_eq!(nrows(&mut db, "SELECT x FROM l UNION ALL SELECT y FROM r"), 2);
}

/// Case 6 (INTERSECT): under the left column's NOCASE, 'A' (left) and 'a'
/// (right) are the same value, so the intersection is that one row.
#[test]
fn compound_intersect_uses_left_column_nocase() {
    let mut db = l_nocase_r_binary();
    assert_eq!(nrows(&mut db, "SELECT x FROM l INTERSECT SELECT y FROM r"), 1);
}

/// Case 6 (EXCEPT): 'A' minus 'a' under NOCASE removes the only left row, so
/// the result is empty. Under a wrong BINARY comparison 'A' would survive (1 row).
#[test]
fn compound_except_uses_left_column_nocase() {
    let mut db = l_nocase_r_binary();
    assert_eq!(nrows(&mut db, "SELECT x FROM l EXCEPT SELECT y FROM r"), 0);
}

/// Case 7 (corrected): a compound assigns a postfix COLLATE no GREATER precedence than
/// a column collation across arms, but it does NOT ignore the COLLATE. In `SELECT 'A'
/// COLLATE NOCASE UNION SELECT 'a'` the LEFT arm DEFINES NOCASE (its postfix COLLATE) and
/// the right defines nothing, so the fold uses NOCASE and 'A'/'a' collapse => ONE row.
///
/// (A naive reading asserted TWO rows by misreading "greater precedence is
/// not assigned to a postfix COLLATE" as "ignore the postfix COLLATE entirely"; real sqlite3
/// still honors it per arm and folds to one — see the module doc. The INTERSECT/EXCEPT
/// variants exercise the same per-arm collation through the other two set operators.)
#[test]
fn compound_union_honors_left_arm_postfix_collate() {
    let mut db = mem();
    assert_eq!(nrows(&mut db, "SELECT 'A' COLLATE NOCASE UNION SELECT 'a'"), 1);
    // INTERSECT: 'A' and 'a' are one value under NOCASE, so the intersection is that row.
    assert_eq!(nrows(&mut db, "SELECT 'A' COLLATE NOCASE INTERSECT SELECT 'a'"), 1);
    // EXCEPT: 'A' minus 'a' under NOCASE removes the only left row => empty.
    assert_eq!(nrows(&mut db, "SELECT 'A' COLLATE NOCASE EXCEPT SELECT 'a'"), 0);
}

/// A postfix COLLATE in the RIGHT arm is honored too, when the LEFT arm defines no
/// collation: `SELECT 'A' UNION SELECT 'a' COLLATE NOCASE` folds left→right — the left
/// literal defines nothing, the right defines NOCASE — so it dedups to ONE row.
#[test]
fn compound_union_honors_right_arm_postfix_collate_when_left_undefined() {
    let mut db = mem();
    assert_eq!(nrows(&mut db, "SELECT 'A' UNION SELECT 'a' COLLATE NOCASE"), 1);
}

/// A postfix COLLATE on a (BINARY) COLUMN reference also DEFINES that arm's collation:
/// `SELECT x COLLATE NOCASE FROM lb UNION SELECT y FROM rb` (both columns BINARY) folds
/// under the left arm's NOCASE => ONE row (the postfix wins WITHIN its own arm over the
/// column's BINARY).
#[test]
fn compound_union_postfix_collate_on_column_defines_arm() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE lb(x TEXT)");
    exec(&mut db, "INSERT INTO lb VALUES ('A')");
    exec(&mut db, "CREATE TABLE rb(y TEXT)");
    exec(&mut db, "INSERT INTO rb VALUES ('a')");
    assert_eq!(nrows(&mut db, "SELECT x COLLATE NOCASE FROM lb UNION SELECT y FROM rb"), 1);
}

/// The "no GREATER precedence" clause, isolated: the LEFT arm is a BINARY COLUMN and the
/// RIGHT arm carries a postfix COLLATE NOCASE. Under the `=` rule the explicit COLLATE would
/// win; in a compound it does NOT outrank the EARLIER column collation, so the fold stays
/// BINARY and 'A'/'a' remain distinct => TWO rows. This is the case that distinguishes the
/// correct rule ("first arm that defines a collation") from "honor any postfix COLLATE
/// anywhere" — and the control that is already correct.
#[test]
fn compound_union_postfix_collate_gets_no_precedence_over_earlier_column() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE lb(x TEXT)");
    exec(&mut db, "INSERT INTO lb VALUES ('A')");
    assert_eq!(nrows(&mut db, "SELECT x FROM lb UNION SELECT 'a' COLLATE NOCASE"), 2);
}

/// Case 7 control: two literal arms with NO collation at all compare BINARY, so
/// distinct case variants stay two rows — pins that the folds above are the COLLATE at
/// work, not some other divergence.
#[test]
fn compound_union_literals_are_binary() {
    let mut db = mem();
    assert_eq!(nrows(&mut db, "SELECT 'A' UNION SELECT 'a'"), 2);
    // Byte-identical literals still dedup to one row (UNION does dedup).
    assert_eq!(nrows(&mut db, "SELECT 'a' UNION SELECT 'a'"), 1);
}

/// FIX 3 depth check: a THREE-arm compound folds the column collation across all
/// arms (SQLite's `multiSelectCollSeq` scans arms left→right). With a NOCASE first
/// arm, 'A','a','A' all fold => one row. This exercises the nested-SetOp
/// propagation, not just the two-arm case.
#[test]
fn compound_three_way_union_propagates_left_column_collation() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE l(x TEXT COLLATE NOCASE)");
    exec(&mut db, "INSERT INTO l VALUES ('A')");
    exec(&mut db, "CREATE TABLE r(y TEXT)");
    exec(&mut db, "INSERT INTO r VALUES ('a')");
    exec(&mut db, "CREATE TABLE s(z TEXT)");
    exec(&mut db, "INSERT INTO s VALUES ('A')");
    assert_eq!(
        nrows(&mut db, "SELECT x FROM l UNION SELECT y FROM r UNION SELECT z FROM s"),
        1,
    );
}

/// FIX 3 right-arm precedence: when the LEFT arm's output is NOT a column (a
/// literal, so no column collation) but the RIGHT arm's is a NOCASE column, the
/// comparison inherits the right column's collation (§7.1: left operand yields,
/// right column's collation applies). `'a'` and `'A'` (from a NOCASE column) fold
/// => one row.
#[test]
fn compound_union_falls_through_to_right_column_collation() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE rr(y TEXT COLLATE NOCASE)");
    exec(&mut db, "INSERT INTO rr VALUES ('A')");
    assert_eq!(nrows(&mut db, "SELECT 'a' UNION SELECT y FROM rr"), 1);
}

// -----------------------------------------------------------------------------
// min(X) / max(X) select the extremum under the argument's collation.
//
// lang_aggfunc.html: min/max is the value that sorts first/last in `ORDER BY X`,
// and ORDER BY uses X's §7.1 collation. The witness: a NOCASE
// column with 'B','a' has min='a', max='B' (since NOCASE 'a'<'b'), whereas the
// BINARY-always bug returned min='B', max='a' ('B'=0x42 < 'a'=0x61).
// -----------------------------------------------------------------------------

/// The min/max witness fixture: a `TEXT COLLATE NOCASE` column with 'B' then 'a'.
fn mm_nocase() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE mm(b TEXT COLLATE NOCASE)");
    exec(&mut db, "INSERT INTO mm VALUES ('B'),('a')");
    db
}

/// The exact reproducer: on a NOCASE column, `min(b)`='a' and `max(b)`='B'.
/// The winning value is returned VERBATIM (original case), the extremum chosen under
/// NOCASE. The BINARY-always bug returned 'B'/'a'.
#[test]
fn min_max_on_nocase_column_folds_case() {
    let mut db = mm_nocase();
    assert_scalar(&mut db, "SELECT min(b) FROM mm", text("a"));
    assert_scalar(&mut db, "SELECT max(b) FROM mm", text("B"));
}

/// BINARY control on the SAME rows: `min`='B', `max`='a' (bytewise). Paired with the
/// case above this pins that the fold is the column's NOCASE at work, not a blanket
/// change — and matches the exact pre-fix (buggy) answer for the NOCASE column, so a
/// revert makes the NOCASE test fail while this one still passes.
#[test]
fn min_max_on_binary_column_is_bytewise() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE mb(b TEXT)");
    exec(&mut db, "INSERT INTO mb VALUES ('B'),('a')");
    assert_scalar(&mut db, "SELECT min(b) FROM mb", text("B"));
    assert_scalar(&mut db, "SELECT max(b) FROM mb", text("a"));
}

/// An explicit `COLLATE NOCASE` on the min/max argument is honoured (§7.1 rule 1:
/// explicit COLLATE wins) even over a BINARY column: `min(b COLLATE NOCASE)`='a',
/// `max(b COLLATE NOCASE)`='B'. Ties the arg-collation resolution (best_effort, honors
/// explicit COLLATE) into the accumulator seam.
#[test]
fn min_max_explicit_collate_argument_folds() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE mb(b TEXT)");
    exec(&mut db, "INSERT INTO mb VALUES ('B'),('a')");
    assert_scalar(&mut db, "SELECT min(b COLLATE NOCASE) FROM mb", text("a"));
    assert_scalar(&mut db, "SELECT max(b COLLATE NOCASE) FROM mb", text("B"));
}

/// RTRIM: 'a' and 'a  ' are equal (trailing spaces ignored, §7), so on scan order
/// 'a','a  ' both min and max keep the FIRST-seen 'a' (a tie preserves first-seen,
/// returned verbatim). The discriminator vs BINARY: under BINARY 'a'<'a  ' (prefix
/// sorts first) so `max` would be 'a  '; RTRIM makes it 'a'.
#[test]
fn min_max_on_rtrim_column_ignores_trailing_space() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE mr(b TEXT COLLATE RTRIM)");
    exec(&mut db, "INSERT INTO mr VALUES ('a'),('a  ')");
    assert_scalar(&mut db, "SELECT min(b) FROM mr", text("a"));
    assert_scalar(&mut db, "SELECT max(b) FROM mr", text("a"));
}

/// The single-min/max bare-column capture (lang_select.html §2.5): `SELECT tag, max(b)`
/// with no GROUP BY captures `tag` from the row that produced the extremum. Under NOCASE
/// `max(b)`='B' (row 'rowB') and `min(b)`='a' (row 'rowa'); the captured `tag` must be
/// that SAME row. This pins fix #2 (the capture keys the extremum by the argument's
/// collation, in lockstep with the reported min/max). The BINARY bug would capture the
/// opposite rows (max→'rowa', min→'rowB').
#[test]
fn min_max_bare_column_capture_uses_arg_collation() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE mc(b TEXT COLLATE NOCASE, tag TEXT)");
    exec(&mut db, "INSERT INTO mc VALUES ('B','rowB'),('a','rowa')");
    assert_rows(&mut db, "SELECT tag, max(b) FROM mc", &[vec![text("rowB"), text("B")]]);
    assert_rows(&mut db, "SELECT tag, min(b) FROM mc", &[vec![text("rowa"), text("a")]]);
}

/// Grouped min/max also fold per group under the column collation. Two NOCASE groups
/// ({x,X} and {y}) each pick their extremum under NOCASE; the folded group's min/max is
/// its single value returned verbatim. Confirms the per-group accumulator carries the
/// collation, not just the whole-table case.
#[test]
fn grouped_min_max_folds_under_nocase() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE mg(g INTEGER, b TEXT COLLATE NOCASE)");
    exec(&mut db, "INSERT INTO mg VALUES (1,'B'),(1,'a'),(1,'C')");
    // Group 1 under NOCASE: min='a', max='C'.
    assert_rows(&mut db, "SELECT min(b), max(b) FROM mg GROUP BY g", &[vec![text("a"), text("C")]]);
}

/// Window min/max (`max(b) OVER ()`) also compare under the argument's collation — the
/// same accumulator seam serves the window path. Over the NOCASE column {'B','a'} every
/// row sees `max`='B'; `SELECT DISTINCT` collapses to that one value.
#[test]
fn window_min_max_folds_under_nocase() {
    let mut db = mm_nocase();
    assert_scalar(&mut db, "SELECT DISTINCT max(b) OVER () FROM mm", text("B"));
    assert_scalar(&mut db, "SELECT DISTINCT min(b) OVER () FROM mm", text("a"));
}

// -----------------------------------------------------------------------------
// Window PARTITION BY groups under the key's collation (datatype3.html §7.1),
// like GROUP BY. `PARTITION BY b` where b is NOCASE folds 'a'/'A' into one
// partition; the BINARY-always bug split them.
// -----------------------------------------------------------------------------

/// On a NOCASE column {'a','A','b'}, `PARTITION BY b` forms two partitions — {'a','A'}
/// (count 2) and {'b'} (count 1) — so `count(*) OVER (PARTITION BY b)` is {2,2,1}. The
/// BINARY-always bug gave three singleton partitions => {1,1,1}.
#[test]
fn window_partition_by_folds_under_nocase() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE wp(b TEXT COLLATE NOCASE)");
    exec(&mut db, "INSERT INTO wp VALUES ('a'),('A'),('b')");
    assert_rows_unordered(
        &mut db,
        "SELECT count(*) OVER (PARTITION BY b) FROM wp",
        &[vec![int(2)], vec![int(2)], vec![int(1)]],
    );
}

/// BINARY control on the same shape: a default-collation column keeps 'a','A','b' as
/// three distinct partitions => {1,1,1}. Pins that the fold above is the column's NOCASE
/// at work, not a blanket change.
#[test]
fn window_partition_by_binary_column_keeps_partitions() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE wpb(b TEXT)");
    exec(&mut db, "INSERT INTO wpb VALUES ('a'),('A'),('b')");
    assert_rows_unordered(
        &mut db,
        "SELECT count(*) OVER (PARTITION BY b) FROM wpb",
        &[vec![int(1)], vec![int(1)], vec![int(1)]],
    );
}

/// An explicit `PARTITION BY b COLLATE NOCASE` over a BINARY column also folds (§7.1
/// rule 1: explicit COLLATE wins), giving {2,2,1}.
#[test]
fn window_partition_by_explicit_collate_folds() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE wpc(b TEXT)");
    exec(&mut db, "INSERT INTO wpc VALUES ('a'),('A'),('b')");
    assert_rows_unordered(
        &mut db,
        "SELECT count(*) OVER (PARTITION BY b COLLATE NOCASE) FROM wpc",
        &[vec![int(2)], vec![int(2)], vec![int(1)]],
    );
}

// -----------------------------------------------------------------------------
// Scalar (multi-argument) min(X,Y,...) / max(X,Y,...) compare strings under the
// collation of the FIRST argument that DEFINES one (lang_corefunc.html max_scalar /
// min_scalar), else BINARY. This is the SCALAR function (>= 2 args) — distinct from the
// single-argument aggregate min/max above, and it must agree with it on a shared column.
// -----------------------------------------------------------------------------

/// The witness: `max('B' COLLATE NOCASE, 'a')` folds under NOCASE, where 'B' > 'a'
/// (case-folded), so the maximum is 'B' and the minimum 'a' (returned verbatim). The
/// BINARY-always bug compared bytewise ('B'=0x42 < 'a'=0x61) and returned max='a', min='B'.
#[test]
fn scalar_min_max_first_arg_explicit_collate_folds() {
    let mut db = mem();
    assert_scalar(&mut db, "SELECT max('B' COLLATE NOCASE, 'a')", text("B"));
    assert_scalar(&mut db, "SELECT min('B' COLLATE NOCASE, 'a')", text("a"));
}

/// BINARY control on the same literals: with NO argument defining a collation the scalar
/// min/max compare bytewise, so `max('B','a')`='a' and `min('B','a')`='B'. Paired with the
/// case above this pins that the fold is the NOCASE at work, not a blanket change — and a
/// revert reddens the case above while this one still passes.
#[test]
fn scalar_min_max_no_collation_is_bytewise() {
    let mut db = mem();
    assert_scalar(&mut db, "SELECT max('B', 'a')", text("a"));
    assert_scalar(&mut db, "SELECT min('B', 'a')", text("B"));
}

/// The collation comes from the FIRST argument that defines one, scanning left→right (the
/// positional rule): in `max('a', 'B' COLLATE NOCASE)` the first arg defines nothing and the
/// second defines NOCASE, so NOCASE applies and the maximum is 'B' (min 'a'). Under BINARY it
/// would be max='a'. This distinguishes the positional rule from `=`'s left-operand
/// precedence (which would still see the explicit COLLATE regardless of position).
#[test]
fn scalar_min_max_uses_first_arg_that_defines_collation() {
    let mut db = mem();
    assert_scalar(&mut db, "SELECT max('a', 'B' COLLATE NOCASE)", text("B"));
    assert_scalar(&mut db, "SELECT min('a', 'B' COLLATE NOCASE)", text("a"));
}

/// A COLUMN argument defines its declared collation for the scalar min/max: with `b` a
/// NOCASE column holding 'a', `max(b, 'B')` folds under NOCASE (the first arg `b` defines
/// NOCASE), so the maximum is 'B' and the minimum 'a'. This is the internal-consistency case
/// — the scalar `max(b,'B')` now agrees with the aggregate `max(b)`
/// over the same NOCASE column instead of diverging.
#[test]
fn scalar_min_max_column_arg_defines_collation() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(b TEXT COLLATE NOCASE)");
    exec(&mut db, "INSERT INTO t VALUES ('a')");
    assert_scalar(&mut db, "SELECT max(b, 'B') FROM t", text("B"));
    assert_scalar(&mut db, "SELECT min(b, 'B') FROM t", text("a"));
    // Internal consistency: the aggregate over {'a','B'} on a NOCASE column agrees.
    exec(&mut db, "INSERT INTO t VALUES ('B')");
    assert_scalar(&mut db, "SELECT max(b) FROM t", text("B"));
}

/// The scalar analogue of the compound "no GREATER precedence" control
/// (`compound_union_postfix_collate_gets_no_precedence_over_earlier_column`): a default-BINARY
/// COLUMN as the FIRST argument DEFINES a collation (BINARY) and so WINS the left→right
/// positional scan — a later explicit `COLLATE NOCASE` on a subsequent argument earns no
/// precedence. With `x` a BINARY column holding 'a', `max(x, 'B' COLLATE NOCASE)` compares
/// bytewise ('a'=0x61 > 'B'=0x42) => max='a', min='B'. Were the later COLLATE (wrongly)
/// allowed to win, NOCASE would fold 'B'>'a' and give max='B'. This pins the scalar side of
/// the boundary: `defined_collation` returns Some(BINARY) for a plain column (not None),
/// matching real sqlite3's sqlite3ExprCollSeq — a column with no explicit COLLATE resolves to
/// the default BINARY, which is non-NULL and stops the scan (unlike a literal, which defines
/// nothing and falls through — see `scalar_min_max_uses_first_arg_that_defines_collation`).
#[test]
fn scalar_min_max_binary_column_first_arg_blocks_later_collate() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE cb(x TEXT)"); // BINARY column
    exec(&mut db, "INSERT INTO cb VALUES ('a')");
    assert_scalar(&mut db, "SELECT max(x, 'B' COLLATE NOCASE) FROM cb", text("a"));
    assert_scalar(&mut db, "SELECT min(x, 'B' COLLATE NOCASE) FROM cb", text("B"));
}

/// The collation fix folds only TEXT: numeric comparison and the NULL-if-any-NULL rule are
/// unchanged. `max(1,2,3)`=3 numerically; a NULL argument makes the whole call NULL even
/// when another argument carries an explicit COLLATE.
#[test]
fn scalar_min_max_numeric_and_null_unchanged() {
    let mut db = mem();
    assert_scalar(&mut db, "SELECT max(1, 2, 3)", int(3));
    assert_scalar(&mut db, "SELECT min(3, 1, 2)", int(1));
    assert_scalar(&mut db, "SELECT max('a' COLLATE NOCASE, NULL)", null());
}

// -----------------------------------------------------------------------------
// nullif(X,Y) uses the SAME positional collation rule as min/max
// (lang_corefunc.html nullif): the first argument that defines a collating function.
// -----------------------------------------------------------------------------

/// `nullif` compares under the first argument that defines a collation. With `x` a BINARY
/// column holding 'a', `nullif(x, 'A' COLLATE NOCASE)` uses the LEFT column's BINARY — a
/// postfix COLLATE on the RIGHT earns no precedence over the earlier-defined collation — so
/// 'a' != 'A' and nullif returns 'a'. Under `=`'s left-operand rule (rule 1 > rule 2) the
/// right's explicit NOCASE would win and wrongly fold the pair to NULL.
#[test]
fn nullif_uses_positional_collation_not_equals_rule() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE tb(x TEXT)"); // BINARY column
    exec(&mut db, "INSERT INTO tb VALUES ('a')");
    assert_scalar(&mut db, "SELECT nullif(x, 'A' COLLATE NOCASE) FROM tb", text("a"));
    // Control: when the FIRST argument defines NOCASE (a NOCASE column), it applies, so
    // 'a' and 'A' fold equal and nullif returns NULL.
    exec(&mut db, "CREATE TABLE tn(x TEXT COLLATE NOCASE)");
    exec(&mut db, "INSERT INTO tn VALUES ('a')");
    assert_scalar(&mut db, "SELECT nullif(x, 'A') FROM tn", null());
}
