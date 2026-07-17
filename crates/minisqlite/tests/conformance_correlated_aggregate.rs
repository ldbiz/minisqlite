//! Conformance battery for CORRELATED subqueries in the post-aggregate context of an
//! aggregate query — a scalar / EXISTS / IN subquery in the SELECT list or HAVING of a
//! `GROUP BY` (or aggregate) query that references the outer query's GROUP BY column.
//!
//! Every expected value here is DERIVED BY HAND from the SQLite documentation, never from
//! what this engine happens to return. The binding sources are:
//!
//!   * `spec/sqlite-doc/lang_select.html` §2 "Generation of the set of result rows":
//!     an aggregate query with GROUP BY produces one result row per group; the SELECT-list
//!     and HAVING expressions are evaluated once per group over that group's rows. The
//!     "HAVING clause" (`lang_select.html`) filters the groups: a group is kept only when its
//!     HAVING expression is TRUE (a FALSE or NULL HAVING drops the group, exactly like a WHERE
//!     predicate). (§2.5 is reserved throughout this codebase for the "bare columns in an
//!     aggregate query" special case — kept distinct here to avoid overloading the citation.)
//!   * `spec/sqlite-doc/lang_expr.html` "Subquery Expressions": "The value of a subquery
//!     expression is the first row of the result from the enclosed SELECT statement" (its
//!     first column, in scalar context) and "is NULL if the enclosed SELECT statement
//!     returns no rows."
//!   * `spec/sqlite-doc/lang_expr.html` "The EXISTS operator": "always evaluates to one of
//!     the integer values 0 and 1" — 1 if the subquery would return one or more rows, else 0.
//!   * `spec/sqlite-doc/lang_expr.html` "The IN operator": `x IN (subquery)` is TRUE when x
//!     equals some value the subquery returns, FALSE when the (non-NULL) subquery returns
//!     no matching row and no NULLs, NULL under the documented NULL cases.
//!   * `spec/sqlite-doc/lang_expr.html` "Correlated Subqueries": a subquery that references
//!     an outer column "is reevaluated each time its result is required" — so each GROUP
//!     row re-runs the subquery with THAT group's key value. This is the property the
//!     per-group-distinct test pins.
//!
//! In SQLite booleans ARE integers: TRUE = 1, FALSE = 0, UNKNOWN = NULL. `Value` has no
//! `PartialEq`, so every assertion goes through the shared harness; never compare with `==`.
//!
//! A case that reveals an engine bug is left as a genuine failing assertion rather than
//! weakened to pass.

mod conformance;

use conformance::*;
use minisqlite::Connection;

// ---------------------------------------------------------------------------
// Canonical fixture:
//
//   emp(id, dept, sal):
//     (1,'a',100), (2,'a',200),           -- group 'a': 2 rows, sum(sal)=300
//     (3,'b',300), (4,'b',50), (5,'b',150) -- group 'b': 3 rows, sum(sal)=500
//   budget(dept, amt): ('a',1000), ('b',2000), ('c',500)
//
// A fresh in-memory database per test keeps them independent and deterministic.
// ---------------------------------------------------------------------------

fn canonical() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE emp(id INTEGER, dept TEXT, sal INTEGER)");
    exec(
        &mut db,
        "INSERT INTO emp VALUES (1,'a',100),(2,'a',200),(3,'b',300),(4,'b',50),(5,'b',150)",
    );
    exec(&mut db, "CREATE TABLE budget(dept TEXT, amt INTEGER)");
    exec(&mut db, "INSERT INTO budget VALUES ('a',1000),('b',2000),('c',500)");
    db
}

// ===========================================================================
// T1 — correlated SCALAR subquery in the SELECT list, correlating on the
//      GROUP BY key e.dept.
// ===========================================================================

#[test]
fn t1_correlated_scalar_in_select_list_on_group_key() {
    let mut db = canonical();
    // Hand-derivation. GROUP BY dept -> two groups.
    //   group 'a': count(*) = 2; scalar subquery (SELECT amt FROM budget WHERE dept='a')
    //     -> first row's amt = 1000.
    //   group 'b': count(*) = 3; (SELECT amt FROM budget WHERE dept='b') -> 2000.
    // Each group re-evaluates the subquery with ITS OWN dept, so 'a' gets 1000 and 'b'
    // gets 2000 — the whole point of the outer-register mapping being per-group.
    assert_rows(
        &mut db,
        "SELECT dept, count(*), (SELECT amt FROM budget b WHERE b.dept = e.dept) AS bud \
         FROM emp e GROUP BY dept ORDER BY dept",
        &[
            vec![text("a"), int(2), int(1000)],
            vec![text("b"), int(3), int(2000)],
        ],
    );
}

#[test]
fn t1_output_column_names() {
    let mut db = canonical();
    // The correlated scalar column takes its `AS bud` alias; the bare aggregate takes its
    // source text. (Pins that adding the subquery does not disturb output naming.)
    assert_columns(
        &mut db,
        "SELECT dept, count(*) AS n, (SELECT amt FROM budget b WHERE b.dept = e.dept) AS bud \
         FROM emp e GROUP BY dept ORDER BY dept",
        &["dept", "n", "bud"],
    );
}

// ===========================================================================
// Scalar subquery returns NO rows for some group -> NULL for that group.
// ===========================================================================

#[test]
fn scalar_subquery_with_no_matching_row_is_null_for_that_group() {
    let mut db = canonical();
    // The subquery adds `AND b.amt > 1500`, so:
    //   group 'a': (SELECT amt FROM budget WHERE dept='a' AND amt>1500) -> budget('a',1000);
    //     1000 > 1500 is false -> NO rows -> scalar value NULL.
    //   group 'b': budget('b',2000); 2000 > 1500 is true -> amt = 2000.
    // Confirms a scalar subquery with an empty result is NULL per-group (not an error, and
    // not the other group's value).
    assert_rows(
        &mut db,
        "SELECT dept, count(*), \
                (SELECT amt FROM budget b WHERE b.dept = e.dept AND b.amt > 1500) AS bigbud \
         FROM emp e GROUP BY dept ORDER BY dept",
        &[
            vec![text("a"), int(2), null()],
            vec![text("b"), int(3), int(2000)],
        ],
    );
}

// ===========================================================================
// T2 — correlated subquery in HAVING that discriminates groups (and exercises
//      a group whose subquery returns no rows -> HAVING NULL -> group dropped).
// ===========================================================================

#[test]
fn t2_correlated_subquery_in_having_discriminates_groups() {
    // A dedicated fixture so the surviving set is non-trivial and hand-verifiable.
    //   emp as canonical: sum(sal) is 'a'=300, 'b'=500.
    //   budget(dept, amt): ('a',250), ('b',600).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE emp(id INTEGER, dept TEXT, sal INTEGER)");
    exec(
        &mut db,
        "INSERT INTO emp VALUES (1,'a',100),(2,'a',200),(3,'b',300),(4,'b',50),(5,'b',150)",
    );
    exec(&mut db, "CREATE TABLE budget(dept TEXT, amt INTEGER)");
    exec(&mut db, "INSERT INTO budget VALUES ('a',250),('b',600)");

    // HAVING sum(sal) > (SELECT amt FROM budget WHERE dept = e.dept AND amt < 400).
    // Hand-derivation:
    //   group 'a': subquery = amt WHERE dept='a' AND amt<400 -> budget('a',250): 250<400
    //     true -> 250. HAVING: sum(sal)=300 > 250 -> TRUE -> KEEP. Row: a|300.
    //   group 'b': subquery = amt WHERE dept='b' AND amt<400 -> budget('b',600): 600<400
    //     false -> NO rows -> scalar NULL. HAVING: 500 > NULL -> NULL -> NOT true -> DROP.
    // Surviving set: {a|300}. This both discriminates and exercises HAVING-NULL drop.
    assert_rows(
        &mut db,
        "SELECT dept, sum(sal) FROM emp e GROUP BY dept \
         HAVING sum(sal) > (SELECT amt FROM budget b WHERE b.dept = e.dept AND b.amt < 400) \
         ORDER BY dept",
        &[vec![text("a"), int(300)]],
    );
}

// ===========================================================================
// T3 — correlated EXISTS in the SELECT list of a GROUP BY query.
// ===========================================================================

#[test]
fn t3_correlated_exists_in_select_list() {
    let mut db = canonical();
    // EXISTS(SELECT 1 FROM budget WHERE dept=e.dept AND amt>1500):
    //   group 'a': budget('a',1000); 1000>1500 false -> no rows -> EXISTS = 0.
    //   group 'b': budget('b',2000); 2000>1500 true  -> a row  -> EXISTS = 1.
    assert_rows(
        &mut db,
        "SELECT dept, count(*), \
                EXISTS(SELECT 1 FROM budget b WHERE b.dept = e.dept AND b.amt > 1500) AS big \
         FROM emp e GROUP BY dept ORDER BY dept",
        &[
            vec![text("a"), int(2), int(0)],
            vec![text("b"), int(3), int(1)],
        ],
    );
}

// ===========================================================================
// Per-group distinctness — each group gets ITS OWN correlated value, never a
// single shared one. This is the test that would fail if the outer-register
// mapping were wrong (e.g. reading a fixed slot for every group).
// ===========================================================================

#[test]
fn each_group_gets_its_own_correlated_value() {
    // Three groups, three DISTINCT budget amounts. If the correlated register were mis-mapped
    // to a single shared slot, every group would show the same amount; three distinct results
    // prove each group's dept key drives its own subquery evaluation.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE emp(id INTEGER, dept TEXT, sal INTEGER)");
    exec(
        &mut db,
        "INSERT INTO emp VALUES (1,'a',100),(2,'a',200),(3,'b',300),(4,'b',50),\
         (5,'b',150),(6,'c',75)",
    );
    exec(&mut db, "CREATE TABLE budget(dept TEXT, amt INTEGER)");
    exec(&mut db, "INSERT INTO budget VALUES ('a',10),('b',20),('c',30)");
    // group 'a' -> 10, 'b' -> 20, 'c' -> 30 (each distinct).
    assert_rows(
        &mut db,
        "SELECT dept, (SELECT amt FROM budget b WHERE b.dept = e.dept) AS bud \
         FROM emp e GROUP BY dept ORDER BY dept",
        &[
            vec![text("a"), int(10)],
            vec![text("b"), int(20)],
            vec![text("c"), int(30)],
        ],
    );
}

// ===========================================================================
// Correlated IN subquery in HAVING (the third correlated-subquery flavour).
// ===========================================================================

#[test]
fn correlated_in_subquery_in_having() {
    // budget amounts chosen so the group's count(*) matches only for 'a'.
    //   group 'a': count(*)=2; (SELECT amt FROM budget WHERE dept='a') = {2}; 2 IN {2} TRUE.
    //   group 'b': count(*)=3; (SELECT amt FROM budget WHERE dept='b') = {99}; 3 IN {99} FALSE.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE emp(id INTEGER, dept TEXT, sal INTEGER)");
    exec(
        &mut db,
        "INSERT INTO emp VALUES (1,'a',100),(2,'a',200),(3,'b',300),(4,'b',50),(5,'b',150)",
    );
    exec(&mut db, "CREATE TABLE budget(dept TEXT, amt INTEGER)");
    exec(&mut db, "INSERT INTO budget VALUES ('a',2),('b',99)");
    assert_rows(
        &mut db,
        "SELECT dept, count(*) FROM emp e GROUP BY dept \
         HAVING count(*) IN (SELECT b.amt FROM budget b WHERE b.dept = e.dept) \
         ORDER BY dept",
        &[vec![text("a"), int(2)]],
    );
}

// ===========================================================================
// GROUP BY on an EXPRESSION key: a group over a computed key still supplies a
// per-group value to the correlated subquery (the subquery correlates on the
// same grouped COLUMN, sal, which here IS the sole group key).
// ===========================================================================

#[test]
fn correlated_scalar_on_a_single_column_group_key_that_is_not_dept() {
    // Group by sal (each distinct sal is its own group here). The subquery correlates on the
    // group key e.sal; each group looks up its own matching lim row. This varies WHICH column
    // is the group key (sal, not dept) to guard against a hard-coded key register.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE emp(id INTEGER, dept TEXT, sal INTEGER)");
    exec(&mut db, "INSERT INTO emp VALUES (1,'a',100),(2,'b',100),(3,'c',200)");
    exec(&mut db, "CREATE TABLE lim(sal INTEGER, tag TEXT)");
    exec(&mut db, "INSERT INTO lim VALUES (100,'lo'),(200,'hi')");
    // group sal=100 (2 rows) -> tag 'lo'; group sal=200 (1 row) -> tag 'hi'.
    assert_rows(
        &mut db,
        "SELECT sal, count(*), (SELECT tag FROM lim l WHERE l.sal = e.sal) AS t \
         FROM emp e GROUP BY sal ORDER BY sal",
        &[
            vec![int(100), int(2), text("lo")],
            vec![int(200), int(1), text("hi")],
        ],
    );
}

// ===========================================================================
// Nested: a scalar subquery in the projection that CONTAINS a further subquery
// correlated to the outer GROUP BY key (through the middle, no-FROM subquery).
// This pins that the outer_width composition stays correct across nesting under
// a grouping parent.
// ===========================================================================

#[test]
fn nested_subquery_correlated_to_the_group_key_through_a_no_from_middle() {
    let mut db = canonical();
    // M = (SELECT (inner)); inner = (SELECT amt FROM budget WHERE dept = e.dept).
    //   group 'a': inner = 1000 -> M = 1000.
    //   group 'b': inner = 2000 -> M = 2000.
    assert_rows(
        &mut db,
        "SELECT dept, (SELECT (SELECT amt FROM budget b WHERE b.dept = e.dept)) \
         FROM emp e GROUP BY dept ORDER BY dept",
        &[vec![text("a"), int(1000)], vec![text("b"), int(2000)]],
    );
}

// ===========================================================================
// A correlated post-aggregate subquery whose OWN body contains a WINDOW function
// (the window is INSIDE the subquery, not the enclosing aggregate query). This is
// ACCEPTED and correct: the outer-query window reject (compile/aggregate.rs) keys
// on the OUTER query's window collector, which is empty here, so it does not fire;
// the subquery's own window binds over its own produced row `[post-agg outer ++ own
// FROM]`, whose width the window-input-width fix computes correctly. Pins that this
// interaction cannot silently regress.
// ===========================================================================

#[test]
fn correlated_subquery_with_its_own_inner_window_function_is_correct() {
    let mut db = canonical();
    // Subquery: `SELECT sum(b.amt) OVER () FROM budget b WHERE b.dept = e.dept`, correlated on
    // the GROUP BY key e.dept. Per group, budget WHERE dept=<key> is a single row, so
    // `sum(b.amt) OVER ()` over that one-row frame is that row's amt, and the scalar subquery
    // takes its first row.
    //   group 'a': count=2; budget dept='a' -> amt 1000; sum OVER () = 1000.
    //   group 'b': count=3; budget dept='b' -> amt 2000; sum OVER () = 2000.
    assert_rows(
        &mut db,
        "SELECT dept, count(*), \
         (SELECT sum(b.amt) OVER () FROM budget b WHERE b.dept = e.dept) \
         FROM emp e GROUP BY dept ORDER BY dept",
        &[
            vec![text("a"), int(2), int(1000)],
            vec![text("b"), int(3), int(2000)],
        ],
    );
}

// ===========================================================================
// Residual LOUD reject: a correlated reference to a NON-GROUP-BY column. A bare
// non-group column has no stable post-aggregate register (§2.5 arbitrary-row
// semantics are unimplemented for a correlated REFERENCE), so this stays a LOUD
// error, not a wrong row. (The grandparent cases below are the other residuals.)
// ===========================================================================

#[test]
fn correlated_ref_to_a_non_group_key_column_errors() {
    let mut db = canonical();
    // The subquery correlates on e.sal, which is NOT the GROUP BY key (dept is). A bare
    // non-group column has no stable post-aggregate register (§2.5 arbitrary-row semantics
    // are unimplemented for a correlated reference), so this is a LOUD error, not a wrong row.
    assert_query_error(
        &mut db,
        "SELECT dept, (SELECT amt FROM budget b WHERE b.amt = e.sal) FROM emp e GROUP BY dept",
    );
}

// ===========================================================================
// The row-WIDENING combinations — a §2.5 capture, a post-aggregate window call,
// and an ORDER BY-introduced aggregate — are now SUPPORTED alongside a correlated
// projection subquery: the planner re-binds the subquery with the true (wider)
// outer row width so its own sources sit past the captured / window / extra-aggregate
// columns. (Deeper coverage lives in `conformance_correlated_subquery_aggregate.rs`.)
// ===========================================================================

#[test]
fn correlated_subquery_with_a_bare_column_capture_is_supported() {
    let mut db = canonical();
    // `e.id` is a §2.5 bare column; with the single `max(sal)` aggregate it takes the value
    // from each group's maximum-`sal` row (lang_select.html §2.5): group 'a' max sal 200 → id
    // 2; group 'b' max sal 300 → id 3. The correlated subquery is re-bound to sit after the
    // captured column. `(SELECT amt ...)`: a→1000, b→2000.
    assert_rows(
        &mut db,
        "SELECT dept, id, max(sal), (SELECT amt FROM budget b WHERE b.dept = e.dept) \
         FROM emp e GROUP BY dept ORDER BY dept",
        &[
            vec![text("a"), int(2), int(200), int(1000)],
            vec![text("b"), int(3), int(300), int(2000)],
        ],
    );
}

#[test]
fn correlated_subquery_with_a_window_function_is_supported() {
    let mut db = canonical();
    // `count(*) OVER ()` is a post-aggregate window call: over the two group rows its default
    // frame counts both, so it is 2 on every row (lang_window). The correlated subquery is
    // re-bound to sit after the window column. `(SELECT amt ...)`: a→1000, b→2000.
    assert_rows(
        &mut db,
        "SELECT dept, count(*) OVER (), (SELECT amt FROM budget b WHERE b.dept = e.dept) \
         FROM emp e GROUP BY dept ORDER BY dept",
        &[
            vec![text("a"), int(2), int(1000)],
            vec![text("b"), int(2), int(2000)],
        ],
    );
}

#[test]
fn correlated_subquery_with_an_order_by_aggregate_is_supported() {
    let mut db = canonical();
    // The ORDER BY introduces an aggregate (sum(sal), a→300, b→500) not present in the
    // projection, widening the real post-aggregate row; the correlated subquery is re-bound
    // against that true width. Rows come back in sum(sal) order (a<b). `(SELECT amt ...)`:
    // a→1000, b→2000.
    assert_rows(
        &mut db,
        "SELECT dept, (SELECT amt FROM budget b WHERE b.dept = e.dept) \
         FROM emp e GROUP BY dept ORDER BY sum(sal)",
        &[vec![text("a"), int(1000)], vec![text("b"), int(2000)]],
    );
}

// ===========================================================================
// Residual LOUD reject — the GRANDPARENT case: a correlated subquery in an
// aggregate's post-aggregate context that resolves THROUGH the aggregate to a
// column of a query ENCLOSING the aggregate. The executor prepends ONLY the
// aggregate operator's output row `[group_keys.., agg_results..]`; the enclosing
// query's row is not present, so that value has NO register in the runtime
// post-aggregate row. Returning any register would read a group-key / aggregate
// slot — a SILENT wrong answer (or an out-of-range read). It MUST be loud.
//
// Locality is decided by NAME, not register range: a grandparent's 0-based
// register can numerically overlap the aggregate's own FROM registers, so a range
// test would misclassify (and could silently remap) such a reference.
// ===========================================================================

#[test]
fn grandparent_ref_through_an_aggregate_in_having_errors_loudly() {
    // The witness. Hand-derivation of the CORRECT answer: `o.k` = 'zzz' is a constant
    // grandparent reference; `EXISTS(SELECT 1 FROM budget WHERE b.dept='zzz')` is FALSE (budget
    // holds only 'a','b'), so HAVING drops BOTH groups, the inner aggregate returns 0 rows, and
    // the scalar subquery is NULL -> the correct row is `zzz|` (NULL). This engine cannot thread
    // 'zzz' into the innermost subquery (its runtime outer is emp's POST-aggregate row, which
    // has no o.k slot), so the ONLY acceptable behaviors are that NULL or a LOUD error — never
    // `zzz|1` (reading a group's dept key), the silent wrong answer the old non-local branch
    // produced. We require and pin the loud error.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE emp(id INTEGER, dept TEXT)");
    exec(&mut db, "INSERT INTO emp VALUES (1,'a'),(2,'b')");
    exec(&mut db, "CREATE TABLE budget(dept TEXT)");
    exec(&mut db, "INSERT INTO budget VALUES ('a'),('b')");
    exec(&mut db, "CREATE TABLE o(k TEXT)");
    exec(&mut db, "INSERT INTO o VALUES ('zzz')");
    let err = assert_query_error(
        &mut db,
        "SELECT o.k, \
         (SELECT count(*) FROM emp e GROUP BY e.dept \
          HAVING EXISTS (SELECT 1 FROM budget b WHERE b.dept = o.k)) \
         FROM o",
    );
    // Bind to the DISTINCTIVE reject so a broad unrelated regression can't satisfy it vacuously.
    assert!(
        format!("{err:?}").contains("enclosing the aggregate"),
        "expected the post-aggregate enclosing-reference reject, got {err:?}"
    );
}

#[test]
fn grandparent_ref_through_an_aggregate_in_select_list_errors_loudly() {
    // A minimal variant: `(SELECT o.k)` reads o.k THROUGH the aggregate over emp.
    // The correct value is 'zzz' for the single group; the engine cannot supply it from the
    // post-aggregate row `[dept_key]`, so it must reject rather than return the group key
    // ('a'/'b'). Pins that the SELECT-list path rejects identically to HAVING.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE emp(id INTEGER, dept TEXT)");
    exec(&mut db, "INSERT INTO emp VALUES (1,'a'),(2,'b')");
    exec(&mut db, "CREATE TABLE o(k TEXT)");
    exec(&mut db, "INSERT INTO o VALUES ('zzz')");
    let err = assert_query_error(
        &mut db,
        "SELECT o.k, (SELECT (SELECT o.k) FROM emp e GROUP BY e.dept) FROM o",
    );
    assert!(
        format!("{err:?}").contains("enclosing the aggregate"),
        "expected the post-aggregate enclosing-reference reject, got {err:?}"
    );
}

#[test]
fn grandparent_ref_colliding_with_a_group_key_register_errors_loudly() {
    // The MAXIMALLY-ADVERSARIAL grandparent witness: the grandparent reference's register
    // NUMERICALLY EQUALS the aggregate's group-key register, so the DELETED register-range
    // check would not have merely rejected with a different message — it would have SILENTLY
    // returned a wrong VALUE. Here `dept` is emp's FIRST column (reg 0) so the group key is at
    // reg 0, and grandparent `o.k` also resolves to reg 0. Under the old register-range path
    // the HAVING's `o.k` bound WITHOUT error to the post-aggregate dept slot (the base-0
    // correlation trial hit `col_to_group.get(&0) == Some(0)` and silently remapped it, and the
    // rebind kept a reg-0 read), so `EXISTS(SELECT 1 FROM budget WHERE b.dept = <group dept>)`
    // is TRUE for BOTH groups (budget has 'a','b'), both groups survive, the inner count is 1,
    // and the query returned `zzz|1`. The CORRECT answer is `zzz|` (NULL): `o.k` = 'zzz' is a
    // constant, `EXISTS(budget WHERE dept='zzz')` is FALSE, so HAVING drops both groups and the
    // scalar subquery is NULL. This engine cannot supply 'zzz' from emp's post-aggregate row,
    // so it must REJECT — never return the silent `zzz|1`. The name-based check ignores
    // registers, so it rejects regardless of the collision. (`assert_query_error` panics if the
    // query SUCCEEDS, so this fails on the old silent `zzz|1`.)
    let mut db = mem();
    exec(&mut db, "CREATE TABLE o(k TEXT)");
    exec(&mut db, "INSERT INTO o VALUES ('zzz')");
    exec(&mut db, "CREATE TABLE emp(dept TEXT, sal INTEGER)");
    exec(&mut db, "INSERT INTO emp VALUES ('a',100),('b',200)");
    exec(&mut db, "CREATE TABLE budget(dept TEXT)");
    exec(&mut db, "INSERT INTO budget VALUES ('a'),('b')");
    let err = assert_query_error(
        &mut db,
        "SELECT o.k, \
         (SELECT count(*) FROM emp e GROUP BY e.dept \
          HAVING EXISTS (SELECT 1 FROM budget b WHERE b.dept = o.k)) \
         FROM o",
    );
    assert!(
        format!("{err:?}").contains("enclosing the aggregate"),
        "expected the post-aggregate enclosing-reference reject, got {err:?}"
    );
}

// ===========================================================================
// SAFETY PROPERTY: a NON-correlated subquery containing a window function, in the
// SELECT list of a multi-aggregate GROUP BY, must plan and run UNCHANGED. Its
// window result binds to the subquery's OWN row width, never the enclosing
// aggregate's (wider) post-aggregate width — a non-correlated subquery runs with
// an EMPTY outer at exec, so an inflated width reads out of range.
// ===========================================================================

#[test]
fn non_correlated_windowed_subquery_in_a_multi_aggregate_group_by() {
    // The witness. Hand-derivation (lang_select.html GROUP BY + lang_expr.html
    // scalar-subquery / lang_window.html count OVER ()): GROUP BY a -> groups {1},{2}; for each,
    // count(*)=1, sum(a)=a, max(a)=a. The scalar subquery `SELECT count(*) OVER () FROM s`
    // computes count(*) OVER () = 3 over all 3 rows of s and returns its first row's value = 3.
    //   a=1 -> 1|1|1|1|3 ;  a=2 -> 2|1|2|2|3.
    // subplan_outer_width = num_keys(1) + num_aggregates(3) = 4 > s's own FROM width 2, so the
    // pre-fix inflated window-input width bound `count(*) OVER ()` past the real row -> the exec
    // "column register index out of range". This must return the rows, not err.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1),(2)");
    exec(&mut db, "CREATE TABLE s(x)");
    exec(&mut db, "INSERT INTO s VALUES (10),(20),(30)");
    assert_rows(
        &mut db,
        "SELECT a, count(*), sum(a), max(a), (SELECT count(*) OVER () FROM s) \
         FROM t GROUP BY a ORDER BY a",
        &[
            vec![int(1), int(1), int(1), int(1), int(3)],
            vec![int(2), int(1), int(2), int(2), int(3)],
        ],
    );
}

#[test]
fn non_correlated_windowed_subquery_matches_a_nonaggregate_parent() {
    // Control: the SAME subquery under a NON-grouping parent already worked
    // and must keep its answer — proving the fix targets the window-input width itself, not the
    // aggregate path. Each of t's 2 rows sees the scalar subquery = 3.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "INSERT INTO t VALUES (1),(2)");
    exec(&mut db, "CREATE TABLE s(x)");
    exec(&mut db, "INSERT INTO s VALUES (10),(20),(30)");
    assert_rows(
        &mut db,
        "SELECT a, (SELECT count(*) OVER () FROM s) FROM t ORDER BY a",
        &[vec![int(1), int(3)], vec![int(2), int(3)]],
    );
}
