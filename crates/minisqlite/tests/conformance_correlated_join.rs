//! Conformance battery for a CORRELATED subquery / trigger action whose FROM clause is a
//! JOIN (two or more sources), end to end through the `minisqlite` facade. It also covers
//! the FROM-leaf shapes the join relaxation newly
//! exposes under a correlated/trigger outer: RIGHT/FULL null-fill (§9), a derived
//! table / VIEW / CTE reference as the leading (or single) source (§10), an
//! ON-clause-only correlation with a memo hit (§11), a left-spine lateral TVF (§12), the two
//! deliberate loud residual gaps (§8 parenthesized sub-join operand, §13 right-operand TVF),
//! and a recursive-CTE reference as a correlated source (§14).
//!
//! A correlated subquery is re-evaluated with the enclosing (outer) row prepended, and a
//! trigger action runs with the `OLD ++ NEW` frame prepended. Every FROM leaf must
//! contribute that outer/frame prefix EXACTLY ONCE (`outer ++ local`); a JOIN carries it on
//! its LEFT spine only, laying its own sources after it — `outer ++ left ++ right` — so an
//! outer `Column(i < outer_width)` reads the prefix and a join column reads at
//! `>= outer_width`. This file proves that the answer for such a FROM equals the SQL-correct
//! answer for every join flavor and leaf kind.
//!
//! Every expected value is computed BY HAND from the SQLite documented semantics, never
//! from what the engine happens to return:
//!
//!   * `spec/sqlite-doc/lang_select.html` — the ON / USING / NATURAL join rules and the
//!     LEFT JOIN NULL-extension of the unmatched right side; a comma (`,`) FROM is a CROSS
//!     join (no `ON`, every pair kept).
//!   * `spec/sqlite-doc/lang_expr.html` §11 "Subquery Expressions" — a scalar subquery is
//!     the first row of its SELECT (after its own ORDER BY/LIMIT), NULL if it returns no
//!     rows; §10 EXISTS is 1/0; §8 IN membership with three-valued NULL logic.
//!   * `spec/sqlite-doc/lang_createtrigger.html` — a trigger BODY statement sees OLD/NEW;
//!     an `AFTER UPDATE` action's `INSERT ... SELECT` observes `OLD.col` / `NEW.col`.
//!
//! A case that reveals an engine bug is left as a genuine failing assertion rather than
//! weakened to pass. `Value` has no `PartialEq`, so every check goes through the shared
//! harness (`assert_rows` / `assert_scalar` / `value_eq`); never compare a `Value` with `==`.

mod conformance;

use conformance::*;
use minisqlite::Connection;

// ---------------------------------------------------------------------------
// Canonical fixture. `o` is the OUTER table a correlated subquery correlates
// on (`o.cat`); `a` and `b` are the two joined inner sources. `a.k = b.k` is
// the equijoin, and `b` has TWO rows for k=1 (so an inner join multiplies) and
// a lone k=9 with NO `a` partner (so a LEFT JOIN null-extends nothing on that
// side, and an inner join drops a's unmatched k=3).
//
//   o(id, cat):   (1,100) (2,200) (3,300)
//   a(cat, k):    (100,1) (100,2) (200,3)
//   b(k, val):    (1,'x') (1,'x2') (2,'y') (9,'z')
//
// Inner-join a⋈b ON a.k=b.k (before any WHERE) is exactly:
//   (a.cat=100,a.k=1,b.val='x'), (100,1,'x2'), (100,2,'y')
// — a(200,3) has no b match, so it drops (inner) or null-extends (left).
// ---------------------------------------------------------------------------

fn join_fixture() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE o(id INTEGER, cat INTEGER)");
    exec(&mut db, "INSERT INTO o VALUES (1,100),(2,200),(3,300)");
    exec(&mut db, "CREATE TABLE a(cat INTEGER, k INTEGER)");
    exec(&mut db, "INSERT INTO a VALUES (100,1),(100,2),(200,3)");
    exec(&mut db, "CREATE TABLE b(k INTEGER, val TEXT)");
    exec(&mut db, "INSERT INTO b VALUES (1,'x'),(1,'x2'),(2,'y'),(9,'z')");
    db
}

// ===========================================================================
// (1) Correlated SCALAR subquery whose FROM is an INNER JOIN.
// ===========================================================================

#[test]
fn correlated_scalar_count_over_inner_join() {
    // `(SELECT count(*) FROM a JOIN b ON a.k=b.k WHERE a.cat=o.cat)` per outer row. The
    // inner join has three rows (all a.cat=100): two for a.k=1 (b has k=1 twice) and one
    // for a.k=2. WHERE a.cat=o.cat then counts:
    //   o.cat=100 -> all 3;  o.cat=200 -> 0 (a(200,3) never joined b);  o.cat=300 -> 0.
    // The correlated `o.cat` is read from the prepended outer row; the join's own columns
    // (a.cat, a.k, b.k) are read after it. A doubled/mis-sized prefix would corrupt these.
    let mut db = join_fixture();
    assert_rows(
        &mut db,
        "SELECT o.id, (SELECT count(*) FROM a JOIN b ON a.k = b.k WHERE a.cat = o.cat) \
         FROM o ORDER BY o.id",
        &[vec![int(1), int(3)], vec![int(2), int(0)], vec![int(3), int(0)]],
    );
}

#[test]
fn correlated_scalar_value_with_order_by_and_limit_over_inner_join() {
    // A scalar subquery whose FROM is a join and which ORDER BYs + LIMITs its own rows:
    // `(SELECT b.val ... WHERE a.cat=o.cat ORDER BY b.val LIMIT 1)`. For o.cat=100 the
    // joined b.val values are {'x','x2','y'}, so ORDER BY b.val LIMIT 1 is 'x'; for
    // o.cat=200 / 300 the join is empty so the scalar subquery is NULL. Proves ORDER BY +
    // LIMIT compose correctly ON TOP of a correlated joined FROM.
    let mut db = join_fixture();
    assert_rows(
        &mut db,
        "SELECT o.id, (SELECT b.val FROM a JOIN b ON a.k = b.k WHERE a.cat = o.cat \
         ORDER BY b.val LIMIT 1) FROM o ORDER BY o.id",
        &[vec![int(1), text("x")], vec![int(2), null()], vec![int(3), null()]],
    );
}

// ===========================================================================
// (2) Correlated EXISTS / IN subquery whose FROM is a join.
// ===========================================================================

#[test]
fn correlated_exists_over_inner_join() {
    // EXISTS shape. A row is kept iff the correlated joined subquery would return a row:
    // only o.cat=100 has matching a⋈b rows, so only id=1 survives.
    let mut db = join_fixture();
    assert_rows(
        &mut db,
        "SELECT o.id FROM o \
         WHERE EXISTS (SELECT 1 FROM a JOIN b ON a.k = b.k WHERE a.cat = o.cat) \
         ORDER BY o.id",
        &[vec![int(1)]],
    );
}

#[test]
fn correlated_in_over_inner_join_probes_per_row() {
    // IN shape over a join, with BOTH the subject and the WHERE correlated. Rows 1 and 2
    // share cat=100 (candidate a.k set {1,2}) but differ in `probe`: 2 is in the set (keep
    // 1), 9 is not (drop 2). Row 3 has cat=200 whose joined set is empty, so probe=3 fails.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE oi(id INTEGER, cat INTEGER, probe INTEGER)");
    exec(&mut db, "INSERT INTO oi VALUES (1,100,2),(2,100,9),(3,200,3)");
    exec(&mut db, "CREATE TABLE a(cat INTEGER, k INTEGER)");
    exec(&mut db, "INSERT INTO a VALUES (100,1),(100,2),(200,3)");
    exec(&mut db, "CREATE TABLE b(k INTEGER, val TEXT)");
    exec(&mut db, "INSERT INTO b VALUES (1,'x'),(1,'x2'),(2,'y'),(9,'z')");
    assert_rows(
        &mut db,
        "SELECT oi.id FROM oi \
         WHERE oi.probe IN (SELECT a.k FROM a JOIN b ON a.k = b.k WHERE a.cat = oi.cat) \
         ORDER BY oi.id",
        &[vec![int(1)]],
    );
}

// ===========================================================================
// (3) Correlated subquery whose FROM is a LEFT JOIN — NULL-extension of the
// unmatched right side must be correct UNDER the outer prefix.
// ===========================================================================

#[test]
fn correlated_left_join_null_extends_unmatched_right() {
    // LEFT JOIN keeps every `a` row, NULL-extending `b` when a.k has no partner. a(200,3)
    // has no b, so it survives as (200,3,NULL,NULL). Two assertions pin the extension:
    //   count(*):      o.cat=200 -> 1 (the null-extended a(200,3) row is present),
    //   count(b.val):  o.cat=200 -> 0 (its b.val is NULL, so a non-NULL count omits it).
    // The count(*)=1 vs count(b.val)=0 gap IS the null-extended row; a broken extension
    // under the prefix would drop the row (count(*)=0) or leak a non-NULL b.val.
    let mut db = join_fixture();
    assert_rows(
        &mut db,
        "SELECT o.id, (SELECT count(*) FROM a LEFT JOIN b ON a.k = b.k WHERE a.cat = o.cat) \
         FROM o ORDER BY o.id",
        &[vec![int(1), int(3)], vec![int(2), int(1)], vec![int(3), int(0)]],
    );
    assert_rows(
        &mut db,
        "SELECT o.id, (SELECT count(b.val) FROM a LEFT JOIN b ON a.k = b.k WHERE a.cat = o.cat) \
         FROM o ORDER BY o.id",
        &[vec![int(1), int(3)], vec![int(2), int(0)], vec![int(3), int(0)]],
    );
}

// ===========================================================================
// (4) A trigger (AFTER UPDATE) whose action's FROM is a join referencing
// OLD / NEW — assert the resulting table state.
// ===========================================================================

#[test]
fn after_update_trigger_action_joins_two_tables_referencing_old_new() {
    // The action `INSERT INTO log SELECT NEW.id, OLD.val, NEW.val, q.tag FROM p JOIN q ON
    // p.pk=q.pk WHERE p.cat = NEW.cat` runs with the OLD++NEW frame prepended (base 2W).
    // Updating the lone row (cat=5, val 100 -> 200): the join over p.cat=5 is {(A,ta),
    // (B,tb)} (p(9,C) is filtered out), so two log rows land, each carrying OLD.val=100,
    // NEW.val=200, NEW.id=1 and its own q.tag. This exercises a JOIN + WHERE that mixes the
    // frame columns (NEW.cat) with the join's own columns (p.cat, p.pk, q.pk).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, cat INTEGER, val INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1, 5, 100)");
    exec(&mut db, "CREATE TABLE p(cat INTEGER, pk TEXT)");
    exec(&mut db, "INSERT INTO p VALUES (5,'A'),(5,'B'),(9,'C')");
    exec(&mut db, "CREATE TABLE q(pk TEXT, tag TEXT)");
    exec(&mut db, "INSERT INTO q VALUES ('A','ta'),('B','tb'),('C','tc')");
    exec(&mut db, "CREATE TABLE log(id INTEGER, oldv INTEGER, newv INTEGER, tag TEXT)");
    exec(
        &mut db,
        "CREATE TRIGGER trg AFTER UPDATE ON t BEGIN \
           INSERT INTO log(id, oldv, newv, tag) \
             SELECT NEW.id, OLD.val, NEW.val, q.tag \
             FROM p JOIN q ON p.pk = q.pk \
             WHERE p.cat = NEW.cat; \
         END",
    );
    exec(&mut db, "UPDATE t SET val = 200 WHERE id = 1");
    // The base row updated.
    assert_rows(&mut db, "SELECT id, cat, val FROM t", &[vec![int(1), int(5), int(200)]]);
    // The trigger's joined action inserted the two matching rows, each with OLD/NEW values.
    assert_rows(
        &mut db,
        "SELECT id, oldv, newv, tag FROM log ORDER BY tag",
        &[
            vec![int(1), int(100), int(200), text("ta")],
            vec![int(1), int(100), int(200), text("tb")],
        ],
    );
}

// ===========================================================================
// (5) Aggregate + GROUP BY inside a correlated joined-FROM subquery.
// ===========================================================================

#[test]
fn correlated_group_by_aggregate_over_inner_join() {
    // The scalar subquery groups the joined rows by a.k and returns the LARGEST group count
    // (ORDER BY the output count DESC, LIMIT 1). For o.cat=100 the groups are {k=1:2,
    // k=2:1}, so the unique top count is 2 (no tie -> deterministic); for o.cat=200 / 300
    // the join is empty (no groups) so the scalar subquery is NULL. Proves GROUP BY + an
    // aggregate + ORDER BY (on the output column) + LIMIT all compose on a correlated
    // joined FROM.
    let mut db = join_fixture();
    assert_rows(
        &mut db,
        "SELECT o.id, (SELECT count(*) AS c FROM a JOIN b ON a.k = b.k WHERE a.cat = o.cat \
         GROUP BY a.k ORDER BY c DESC LIMIT 1) FROM o ORDER BY o.id",
        &[vec![int(1), int(2)], vec![int(2), null()], vec![int(3), null()]],
    );
}

#[test]
fn correlated_group_by_aggregate_multiset_consumed_by_in() {
    // The GROUP BY subquery returns a MULTI-ROW set of per-group counts, consumed by a
    // correlated IN (no ORDER BY, so this is robust to the aggregate ORDER BY surface). For
    // o.cat=100 the group counts are {2, 1}, so `2 IN (...)` holds and id=1 is kept; for
    // o.cat=200 / 300 the join is empty (no groups) so the set is empty and IN is false.
    let mut db = join_fixture();
    assert_rows(
        &mut db,
        "SELECT o.id FROM o \
         WHERE 2 IN (SELECT count(*) FROM a JOIN b ON a.k = b.k WHERE a.cat = o.cat GROUP BY a.k) \
         ORDER BY o.id",
        &[vec![int(1)]],
    );
}

#[test]
fn correlated_distinct_aggregate_over_inner_join() {
    // `count(DISTINCT b.k)` over the correlated join. For o.cat=100 the joined b.k values
    // are {1, 1, 2} (a.k=1 matches b(1,'x') AND b(1,'x2')), so the DISTINCT count is 2,
    // versus a plain count of 3 — DISTINCT must fold the duplicate under the outer prefix.
    let mut db = join_fixture();
    assert_rows(
        &mut db,
        "SELECT o.id, (SELECT count(DISTINCT b.k) FROM a JOIN b ON a.k = b.k WHERE a.cat = o.cat) \
         FROM o ORDER BY o.id",
        &[vec![int(1), int(2)], vec![int(2), int(0)], vec![int(3), int(0)]],
    );
}

#[test]
fn correlated_select_distinct_operator_over_inner_join() {
    // The `SELECT DISTINCT` row-dedup operator inside the correlated join: the distinct
    // b.k set for o.cat=100 is {1, 2}, so `1 IN (SELECT DISTINCT b.k ...)` holds (keep id=1)
    // while cat=200/300 have empty joins. Exercises the Distinct operator (not just an
    // aggregate) composing on top of a correlated joined FROM.
    let mut db = join_fixture();
    assert_rows(
        &mut db,
        "SELECT o.id FROM o \
         WHERE 1 IN (SELECT DISTINCT b.k FROM a JOIN b ON a.k = b.k WHERE a.cat = o.cat) \
         ORDER BY o.id",
        &[vec![int(1)]],
    );
}

// ===========================================================================
// (6) REGRESSION GUARD: an ordinary TOP-LEVEL 2-table join (no outer prefix,
// base_offset == 0) must return identical rows — the change is a strict no-op
// when there is no correlated/trigger outer row.
// ===========================================================================

#[test]
fn top_level_inner_join_is_unchanged_by_the_outer_prefix_change() {
    // No subquery, no trigger: a plain `a JOIN b ON a.k=b.k` at the top level. This checks
    // the RESULT ROWS are unchanged — `outer` is empty here, so the prefix path is a no-op
    // (the right child is built with the same empty outer either way). The register LAYOUT of
    // the base-0 path is pinned separately by the plan-unit tests in select.rs / trigger.rs;
    // this is the end-to-end guard on the core path the change must not touch.
    let mut db = join_fixture();
    assert_rows(
        &mut db,
        "SELECT a.cat, a.k, b.val FROM a JOIN b ON a.k = b.k ORDER BY a.cat, a.k, b.val",
        &[
            vec![int(100), int(1), text("x")],
            vec![int(100), int(1), text("x2")],
            vec![int(100), int(2), text("y")],
        ],
    );
}

// ===========================================================================
// (7) Additional join FLAVORS under a correlated outer: CROSS (comma), USING,
// and NATURAL — each must behave as at the top level.
// ===========================================================================

#[test]
fn correlated_cross_comma_join() {
    // A comma FROM is a CROSS join (no ON, every pair kept). `(SELECT count(*) FROM a, b
    // WHERE a.cat=o.cat)` is |a rows with cat| * |all b| = |a_cat| * 4:
    //   o.cat=100 -> 2*4=8;  o.cat=200 -> 1*4=4;  o.cat=300 -> 0.
    let mut db = join_fixture();
    assert_rows(
        &mut db,
        "SELECT o.id, (SELECT count(*) FROM a, b WHERE a.cat = o.cat) FROM o ORDER BY o.id",
        &[vec![int(1), int(8)], vec![int(2), int(4)], vec![int(3), int(0)]],
    );
}

#[test]
fn correlated_using_join() {
    // `a JOIN b USING (k)` equijoins on the shared column `k` (and coalesces it). Same
    // matched rows as the ON form, so the correlated counts match test (1):
    //   o.cat=100 -> 3;  o.cat=200 -> 0;  o.cat=300 -> 0.
    // Pins that the USING-coalesced column resolves against the correct ABSOLUTE registers
    // under a nonzero base_offset.
    let mut db = join_fixture();
    assert_rows(
        &mut db,
        "SELECT o.id, (SELECT count(*) FROM a JOIN b USING (k) WHERE a.cat = o.cat) \
         FROM o ORDER BY o.id",
        &[vec![int(1), int(3)], vec![int(2), int(0)], vec![int(3), int(0)]],
    );
}

#[test]
fn correlated_natural_join() {
    // `a NATURAL JOIN b` joins on all common columns; `a(cat,k)` and `b(k,val)` share only
    // `k`, so it equals `USING (k)` — same correlated counts as tests (1) and USING.
    let mut db = join_fixture();
    assert_rows(
        &mut db,
        "SELECT o.id, (SELECT count(*) FROM a NATURAL JOIN b WHERE a.cat = o.cat) \
         FROM o ORDER BY o.id",
        &[vec![int(1), int(3)], vec![int(2), int(0)], vec![int(3), int(0)]],
    );
}

// ===========================================================================
// (8) The one deliberate residual GAP, proven end to end: a PARENTHESIZED
// sub-join used as an OPERAND of a larger join, under a correlated outer, is a
// PRECISE loud error (a sub-join is built in a local 0-based space that cannot
// compose with the outer prefix). A FLAT correlated join is supported; this
// narrow shape fails loud rather than mis-bind. See from.rs `build_subjoin`.
// ===========================================================================

#[test]
fn correlated_parenthesized_subjoin_operand_is_a_loud_error() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE o(id INTEGER, cat INTEGER)");
    exec(&mut db, "INSERT INTO o VALUES (1,100)");
    exec(&mut db, "CREATE TABLE a(cat INTEGER, k INTEGER)");
    exec(&mut db, "INSERT INTO a VALUES (100,1)");
    exec(&mut db, "CREATE TABLE b(k INTEGER, z INTEGER)");
    exec(&mut db, "INSERT INTO b VALUES (1,7)");
    exec(&mut db, "CREATE TABLE c(k INTEGER, w TEXT)");
    exec(&mut db, "INSERT INTO c VALUES (1,'m')");
    // `(b JOIN c ON b.k=c.k)` is a parenthesized sub-join used as the right operand of
    // `a JOIN (...)`, inside a subquery correlated on `o.cat` — the residual gap.
    let err = assert_query_error(
        &mut db,
        "SELECT o.id, (SELECT count(*) FROM a JOIN (b JOIN c ON b.k = c.k) ON a.k = b.k \
         WHERE a.cat = o.cat) FROM o",
    );
    assert!(
        format!("{err:?}").contains("parenthesized join"),
        "expected a precise 'parenthesized join ... not yet supported' error, got {err:?}"
    );
}

// ===========================================================================
// (9) RIGHT / FULL OUTER JOIN under a correlated / trigger outer — the
// unmatched-RIGHT null-fill path (`fill_left` / `next_unmatched_right` in
// exec `ops/join.rs`), which RE-PREPENDS the outer prefix onto a synthesized
// `outer ++ NULLs(left_width) ++ right_local` row. This path is reached ONLY
// for RIGHT/FULL joins with a NON-EMPTY outer, so a top-level RIGHT/FULL test
// (outer == &[]) can never exercise it: these are the tests that would go red
// if the outer-threading in `fill_left` were dropped.
// ===========================================================================

#[test]
fn correlated_right_join_null_extended_row_survives_correlated_where() {
    // RIGHT JOIN preserves EVERY `b` row; `b.k=5` has no `a` partner so it flows through
    // `fill_left`/`next_unmatched_right` as (a=NULL, b.k=5). Correlating the WHERE on the
    // RIGHT column `b.k` (not `a.*`, which is NULL here) keeps that null-extended row, and
    // `o.lim` is read from the OUTER prefix. RIGHT JOIN a⋈b = {(a.k=2,b.k=2),(a=NULL,b.k=5)}:
    //   o.lim=0   -> both b.k>=0   -> count 2;  o.lim=100 -> neither b.k>=100 -> count 0.
    // If `fill_left` dropped the prefix, `b.k`/`o.lim` would misread (wrong count or an
    // out-of-range Column panic) — this is the guard for the introduced edge.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE o(id INTEGER, lim INTEGER)");
    exec(&mut db, "INSERT INTO o VALUES (1,0),(2,100)");
    exec(&mut db, "CREATE TABLE a(k INTEGER, av TEXT)");
    exec(&mut db, "INSERT INTO a VALUES (2,'a2')");
    exec(&mut db, "CREATE TABLE b(k INTEGER, bv TEXT)");
    exec(&mut db, "INSERT INTO b VALUES (2,'b2'),(5,'b5')");
    assert_rows(
        &mut db,
        "SELECT o.id, (SELECT count(*) FROM a RIGHT JOIN b ON a.k = b.k WHERE b.k >= o.lim) \
         FROM o ORDER BY o.id",
        &[vec![int(1), int(2)], vec![int(2), int(0)]],
    );
}

#[test]
fn correlated_full_join_null_fills_both_sides_under_outer() {
    // FULL JOIN exercises BOTH null-fill paths under a non-empty outer at once: an
    // unmatched-LEFT row (a.k=3, b=NULL) via `fill_right`, and an unmatched-RIGHT row
    // (a=NULL, b.k=5) via `fill_left`. `coalesce(a.k,b.k)` -> {2, 3, 5}; correlating on
    // `o.lim`:  o.lim=0 keeps all 3, o.lim=100 keeps none.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE o(id INTEGER, lim INTEGER)");
    exec(&mut db, "INSERT INTO o VALUES (1,0),(2,100)");
    exec(&mut db, "CREATE TABLE a(k INTEGER, av TEXT)");
    exec(&mut db, "INSERT INTO a VALUES (2,'a2'),(3,'a3')");
    exec(&mut db, "CREATE TABLE b(k INTEGER, bv TEXT)");
    exec(&mut db, "INSERT INTO b VALUES (2,'b2'),(5,'b5')");
    assert_rows(
        &mut db,
        "SELECT o.id, (SELECT count(*) FROM a FULL JOIN b ON a.k = b.k \
         WHERE coalesce(a.k, b.k) >= o.lim) FROM o ORDER BY o.id",
        &[vec![int(1), int(3)], vec![int(2), int(0)]],
    );
    // The null sides are placed correctly UNDER the outer prefix: count(a.k) omits the
    // unmatched-RIGHT row (a.k NULL there); count(b.bv) omits the unmatched-LEFT row (b.bv
    // NULL there). Both are 2 for o.lim=0. A prefix/width slip would miscount these.
    assert_rows(
        &mut db,
        "SELECT o.id, (SELECT count(a.k) FROM a FULL JOIN b ON a.k = b.k \
         WHERE coalesce(a.k, b.k) >= o.lim) FROM o ORDER BY o.id",
        &[vec![int(1), int(2)], vec![int(2), int(0)]],
    );
    assert_rows(
        &mut db,
        "SELECT o.id, (SELECT count(b.bv) FROM a FULL JOIN b ON a.k = b.k \
         WHERE coalesce(a.k, b.k) >= o.lim) FROM o ORDER BY o.id",
        &[vec![int(1), int(2)], vec![int(2), int(0)]],
    );
}

#[test]
fn after_update_trigger_action_right_join_null_fills_under_frame() {
    // Trigger-action variant of the RIGHT-join null-fill: the action's FROM is a RIGHT JOIN
    // and the WHERE correlates on `NEW.lim` (from the OLD++NEW frame). NEW.lim=3 keeps only
    // the unmatched-RIGHT row (a=NULL, b.k=5) — routed through `fill_left` with the FRAME as
    // the outer. The inserted log row proves a=NULL (av NULL) with b.bv='b5' survived and
    // that NEW.lim was read from the frame prefix, not a mis-shifted register.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, lim INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1, 0)");
    exec(&mut db, "CREATE TABLE a(k INTEGER, av TEXT)");
    exec(&mut db, "INSERT INTO a VALUES (2,'a2')");
    exec(&mut db, "CREATE TABLE b(k INTEGER, bv TEXT)");
    exec(&mut db, "INSERT INTO b VALUES (2,'b2'),(5,'b5')");
    exec(&mut db, "CREATE TABLE log(bk INTEGER, av TEXT, bv TEXT)");
    exec(
        &mut db,
        "CREATE TRIGGER trg AFTER UPDATE ON t BEGIN \
           INSERT INTO log(bk, av, bv) \
             SELECT b.k, a.av, b.bv FROM a RIGHT JOIN b ON a.k = b.k \
             WHERE b.k >= NEW.lim; \
         END",
    );
    exec(&mut db, "UPDATE t SET lim = 3 WHERE id = 1");
    assert_rows(
        &mut db,
        "SELECT bk, av, bv FROM log ORDER BY bk",
        &[vec![int(5), null(), text("b5")]],
    );
}

// ===========================================================================
// (10) A DERIVED TABLE / VIEW / CTE reference as the LEFTMOST leaf (or single
// source) of a correlated subquery. These lower to `CteScan`, a FROM leaf that
// (like every other leaf) must contribute the outer prefix EXACTLY ONCE. The
// body is drained STANDALONE (empty outer) and the prefix is prepended at the
// CteScan output boundary; without that, the base-0 body mis-binds (silent
// wrong rows) or throws "column register index out of range".
// ===========================================================================

#[test]
fn correlated_derived_table_as_left_operand_of_comma_join() {
    // `(SELECT cat FROM a) d, b` — a derived table as the LEFT/first source of a comma
    // (CROSS) join, correlated on `o.cat`. d = {100,100,200}; d,b is 3*2=6 pairs; WHERE
    // d.cat=o.cat=100 keeps the 2 d-rows with cat=100 times |b|=2 -> 4.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE o(cat INTEGER)");
    exec(&mut db, "INSERT INTO o VALUES (100)");
    exec(&mut db, "CREATE TABLE a(cat INTEGER)");
    exec(&mut db, "INSERT INTO a VALUES (100),(100),(200)");
    exec(&mut db, "CREATE TABLE b(x INTEGER)");
    exec(&mut db, "INSERT INTO b VALUES (7),(8)");
    assert_scalar(
        &mut db,
        "SELECT (SELECT count(*) FROM (SELECT cat FROM a) d, b WHERE d.cat = o.cat) FROM o",
        int(4),
    );
}

#[test]
fn correlated_derived_table_as_left_operand_of_equijoin() {
    // Hash-join variant: `(SELECT cat,k FROM a) d JOIN b ON d.k=b.k`. The hash left key is
    // `d.k` evaluated against the standalone left row — which must be the CteScan's own
    // width, so a dropped prefix would put `d.k` out of range. d={(100,1)}; join b(k=1) ->
    // 1 row; WHERE d.cat=100=o.cat -> count 1.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE o(cat INTEGER)");
    exec(&mut db, "INSERT INTO o VALUES (100)");
    exec(&mut db, "CREATE TABLE a(cat INTEGER, k INTEGER)");
    exec(&mut db, "INSERT INTO a VALUES (100,1)");
    exec(&mut db, "CREATE TABLE b(k INTEGER)");
    exec(&mut db, "INSERT INTO b VALUES (1)");
    assert_scalar(
        &mut db,
        "SELECT (SELECT count(*) FROM (SELECT cat,k FROM a) d JOIN b ON d.k = b.k \
         WHERE d.cat = o.cat) FROM o",
        int(1),
    );
}

#[test]
fn correlated_derived_table_left_operand_correlated_on_derived_column() {
    // The correlation reads a DERIVED column (`d.k = t.id`), so the outer prefix and the
    // derived leaf's own columns must both resolve. d={1,2}; d JOIN u ON d.k=u.k =
    // {(1,1),(2,2)}; WHERE d.k=t.id -> exactly one row per outer id.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1),(2)");
    exec(&mut db, "CREATE TABLE x(k INTEGER)");
    exec(&mut db, "INSERT INTO x VALUES (1),(2)");
    exec(&mut db, "CREATE TABLE u(k INTEGER)");
    exec(&mut db, "INSERT INTO u VALUES (1),(2)");
    assert_rows(
        &mut db,
        "SELECT t.id, (SELECT count(*) FROM (SELECT k FROM x) d JOIN u ON d.k = u.k \
         WHERE d.k = t.id) FROM t ORDER BY t.id",
        &[vec![int(1), int(1)], vec![int(2), int(1)]],
    );
}

#[test]
fn correlated_single_source_derived_table_subquery() {
    // SINGLE-source correlated subquery over a derived table (no join): the CteScan is the
    // ONLY source, so it alone must carry the outer prefix. d={1,2}; WHERE d.k=t.id -> 1
    // per id. (This shape was broken before the leaf-contract fix even without a join.)
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1),(2)");
    exec(&mut db, "CREATE TABLE x(k INTEGER)");
    exec(&mut db, "INSERT INTO x VALUES (1),(2)");
    assert_rows(
        &mut db,
        "SELECT t.id, (SELECT count(*) FROM (SELECT k FROM x) d WHERE d.k = t.id) \
         FROM t ORDER BY t.id",
        &[vec![int(1), int(1)], vec![int(2), int(1)]],
    );
}

#[test]
fn correlated_cte_reference_as_left_operand_of_join() {
    // A CTE reference (also lowers to CteScan) as the LEFT operand of a correlated join.
    // Same shape/answer as the derived-table equijoin case.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE o(cat INTEGER)");
    exec(&mut db, "INSERT INTO o VALUES (100)");
    exec(&mut db, "CREATE TABLE a(cat INTEGER, k INTEGER)");
    exec(&mut db, "INSERT INTO a VALUES (100,1)");
    exec(&mut db, "CREATE TABLE b(k INTEGER)");
    exec(&mut db, "INSERT INTO b VALUES (1)");
    assert_scalar(
        &mut db,
        "WITH d AS (SELECT cat, k FROM a) \
         SELECT (SELECT count(*) FROM d JOIN b ON d.k = b.k WHERE d.cat = o.cat) FROM o",
        int(1),
    );
}

#[test]
fn correlated_view_as_left_operand_of_join() {
    // A VIEW reference (also lowers to CteScan) as the LEFT operand of a correlated join.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE o(cat INTEGER)");
    exec(&mut db, "INSERT INTO o VALUES (100)");
    exec(&mut db, "CREATE TABLE a(cat INTEGER, k INTEGER)");
    exec(&mut db, "INSERT INTO a VALUES (100,1)");
    exec(&mut db, "CREATE TABLE b(k INTEGER)");
    exec(&mut db, "INSERT INTO b VALUES (1)");
    exec(&mut db, "CREATE VIEW dv AS SELECT cat, k FROM a");
    assert_scalar(
        &mut db,
        "SELECT (SELECT count(*) FROM dv JOIN b ON dv.k = b.k WHERE dv.cat = o.cat) FROM o",
        int(1),
    );
}

#[test]
fn correlated_derived_table_as_right_operand_still_works() {
    // The ASYMMETRY that confirms the root cause: the SAME derived table as the RIGHT
    // operand always worked (the right child is built with an empty outer, so its base-0
    // body binds correctly and the left spine carries the prefix). Pinned so the left/right
    // symmetry of the fix is explicit.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE o(cat INTEGER)");
    exec(&mut db, "INSERT INTO o VALUES (100)");
    exec(&mut db, "CREATE TABLE a(cat INTEGER, k INTEGER)");
    exec(&mut db, "INSERT INTO a VALUES (100,1)");
    exec(&mut db, "CREATE TABLE b(k INTEGER)");
    exec(&mut db, "INSERT INTO b VALUES (1)");
    assert_scalar(
        &mut db,
        "SELECT (SELECT count(*) FROM b JOIN (SELECT cat,k FROM a) d ON d.k = b.k \
         WHERE d.cat = o.cat) FROM o",
        int(1),
    );
}

// ===========================================================================
// (11) Correlation placed ONLY in the join ON constraint (not WHERE), with two
// outer rows SHARING the correlated key so the correlated-subquery memo cache
// takes a HIT. This locks the invariant that `correlated_cols` captures outer
// refs appearing in the ON clause: if a refactor bound ON off a collector-less
// scope, the memo key would be incomplete and serve a stale result to the
// second same-key row — a silent wrong answer this test would catch.
// ===========================================================================

#[test]
fn correlated_reference_only_in_on_clause_with_shared_key() {
    // `... a2 JOIN b2 ON a2.k=b2.k AND b2.val=o2.z` — o2.z appears ONLY in the ON. ids 1&2
    // share z=7 (a memo HIT); id 3 has z=9. a2⋈b2 on k gives {k=1:val 7 twice, k=2:val 9}:
    //   z=7 -> the two val=7 rows -> 2;  z=9 -> the one val=9 row -> 1.
    // Expected (1,2),(2,2),(3,1). A memo key that dropped the ON correlation would reuse
    // id=1's result for id=3 and return 2 there.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE o2(id INTEGER, z INTEGER)");
    exec(&mut db, "INSERT INTO o2 VALUES (1,7),(2,7),(3,9)");
    exec(&mut db, "CREATE TABLE a2(k INTEGER)");
    exec(&mut db, "INSERT INTO a2 VALUES (1),(2)");
    exec(&mut db, "CREATE TABLE b2(k INTEGER, val INTEGER)");
    exec(&mut db, "INSERT INTO b2 VALUES (1,7),(2,9),(1,7)");
    assert_rows(
        &mut db,
        "SELECT id, (SELECT count(*) FROM a2 JOIN b2 ON a2.k = b2.k AND b2.val = o2.z) \
         FROM o2 ORDER BY id",
        &[vec![int(1), int(2)], vec![int(2), int(2)], vec![int(3), int(1)]],
    );
}

// ===========================================================================
// (12) A lateral table-valued function (`json_each`) on the LEFT spine of a
// correlated join. TableFunctionScan DOES call `with_outer`, so the left-spine
// TVF works (only a RIGHT-operand TVF is the deliberate loud gap); pinned here
// so the asymmetry is explicit and can't silently regress.
// ===========================================================================

#[test]
fn correlated_left_spine_table_function_join() {
    // `json_each(o.data) je JOIN u ON u.k = je.value` — the TVF's arg reads the OUTER row
    // (`o.data`), and the join's right is a base table. The two outer rows are given
    // DIFFERENT expected counts so a "TVF arg stuck on the first outer row" bug is caught,
    // not just a total break: o.id=1 (data '[1,3]') -> elements {1,3}, u={1,2}, only 1
    // matches -> count 1; o.id=2 (data '[1,2]') -> elements {1,2}, both match u -> count 2.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE o(id INTEGER, data TEXT)");
    exec(&mut db, "INSERT INTO o VALUES (1,'[1,3]'),(2,'[1,2]')");
    exec(&mut db, "CREATE TABLE u(k INTEGER)");
    exec(&mut db, "INSERT INTO u VALUES (1),(2)");
    assert_rows(
        &mut db,
        "SELECT o.id, (SELECT count(*) FROM json_each(o.data) je JOIN u ON u.k = je.value) \
         FROM o ORDER BY o.id",
        &[vec![int(1), int(1)], vec![int(2), int(2)]],
    );
}

// ===========================================================================
// (13) The second deliberate residual GAP, proven end to end: a table-valued
// function used as the RIGHT operand of a join under a correlated outer is a
// PRECISE loud error (a TVF is implicitly LATERAL and needs a per-left
// IndexNestedLoop seek, which is gated to base 0). The LEFT-spine form works
// (§12); only the right-operand form is gapped. See from.rs
// `choose_right_and_strategy`.
// ===========================================================================

#[test]
fn correlated_right_operand_table_function_is_a_loud_error() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE o(id INTEGER, data TEXT)");
    exec(&mut db, "INSERT INTO o VALUES (1,'[1,2]')");
    exec(&mut db, "CREATE TABLE u(k INTEGER)");
    exec(&mut db, "INSERT INTO u VALUES (1),(2)");
    // `u JOIN json_each(o.data)` puts the lateral TVF as the RIGHT operand inside a
    // subquery correlated on `o.data` — the residual gap; must fail loud, not mis-bind.
    let err = assert_query_error(
        &mut db,
        "SELECT o.id, (SELECT count(*) FROM u JOIN json_each(o.data) je ON u.k = je.value) \
         FROM o",
    );
    assert!(
        format!("{err:?}").contains("table-valued function")
            && format!("{err:?}").contains("not yet supported"),
        "expected a precise 'table-valued function ... not yet supported' error, got {err:?}"
    );
}

// ===========================================================================
// (14) A RECURSIVE CTE reference as the (single) source of a correlated
// subquery. A recursive CTE lowers to a `CteScan` over a `CtePlan::Recursive`,
// which runs the fixpoint generator; its output flows through the SAME
// `CteScanCursor::next_row` prefix boundary as a materialized CTE, so the
// correlated prefix is prepended once while the recursive frontier stays
// unprefixed (read at base 0 by `RecursiveScan`). Correct by construction —
// pinned here so it can't silently regress.
// ===========================================================================

#[test]
fn correlated_subquery_over_a_recursive_cte_reference() {
    // `cnt` is the recursive series {1,2,3,4,5}; the correlated subquery counts how many of
    // its rows are <= the outer `t.id`. t.id=1 -> {1} -> 1; t.id=3 -> {1,2,3} -> 3.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1),(3)");
    assert_rows(
        &mut db,
        "WITH RECURSIVE cnt(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM cnt WHERE n < 5) \
         SELECT t.id, (SELECT count(*) FROM cnt WHERE cnt.n <= t.id) FROM t ORDER BY t.id",
        &[vec![int(1), int(1)], vec![int(3), int(3)]],
    );
}
