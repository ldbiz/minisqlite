//! Conformance: JOINs (CROSS / INNER / comma / LEFT / RIGHT / FULL, plus the ON,
//! USING, and NATURAL constraints).
//!
//! Every expected value below is transcribed from the SQLite documentation, NOT
//! from what this engine returns. The binding source is
//! `spec/sqlite-doc/lang_select.html` §2 ("Determination of input data (FROM
//! clause processing)"), §2.2 ("Special handling of CROSS JOIN"), and §2.3
//! ("WHERE clause filtering"). The load-bearing rules:
//!
//!   * "All joins in SQLite are based on the cartesian product of the left and
//!     right-hand datasets. The columns of the cartesian product dataset are, in
//!     order, all the columns of the left-hand dataset followed by all the
//!     columns of the right-hand dataset." So an N-row × M-row join scans N×M
//!     candidate rows and the row width is the sum of the two widths.
//!   * "If the join-operator is 'CROSS JOIN', 'INNER JOIN', 'JOIN' or a comma
//!     (',') and there is no ON or USING clause, then the result of the join is
//!     simply the cartesian product." §2.2: INNER JOIN, JOIN, "," and CROSS JOIN
//!     are result-equivalent (they differ only in optimizer table-ordering).
//!   * ON: "Only rows for which the expression evaluates to true are included."
//!     A NULL ON result (e.g. a NULL join key) is therefore NOT a match.
//!   * USING (X): X "must exist in the datasets to both the left and right"; the
//!     kept rows are those where "lhs.X = rhs.X" is true, and "the column from
//!     the right-hand dataset is omitted from the joined dataset" — a single
//!     shared column, the ONLY difference from the equivalent ON constraint.
//!   * NATURAL: "an implicit USING clause is added ... [containing] each of the
//!     column names that appear in both the left and right-hand input datasets."
//!   * LEFT [OUTER] JOIN: after ON/USING filtering, "an extra row is added to the
//!     output for each row in the original left-hand input dataset that does not
//!     match any row in the right-hand dataset", NULL-filled on the right.
//!     RIGHT / FULL are the documented mirror / union.
//!   * §2.3: for an outer join "the extra NULL rows ... are added after ON clause
//!     processing but before WHERE clause processing." A right-referencing
//!     constraint in the ON clause keeps the all-NULL rows; the same constraint
//!     in WHERE excludes them, making the LEFT JOIN behave like an inner join.
//!
//! `Value` has no `PartialEq`, so every assertion goes through the shared harness
//! (`assert_rows` / `assert_rows_unordered` / `assert_scalar` / `assert_columns`);
//! never compare a `Value` with `==`. ORDER BY sort keys are chosen to be
//! non-NULL so ordered assertions do not entangle NULL-ordering (a separate
//! concern); outer joins that place NULLs in a sort position use the multiset
//! (`assert_rows_unordered`) form instead.

mod conformance;
use conformance::*;

// `Connection` is used only in the private `setup()` helper's signature below; the
// harness imports it privately, so it is not in scope via `conformance::*`.
use minisqlite::Connection;

/// The shared fixture used by every test: `emp` (with one NULL `dept_id`), `dept`
/// (with one department that matches no employee), and `deptu` — a schema-clone
/// of `dept` whose `dept_id`/`dname` column names overlap `emp`'s so USING and
/// NATURAL have a well-defined common column. A fresh database per test keeps
/// them independent.
fn setup() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE emp(id INTEGER, name TEXT, dept_id INTEGER)");
    exec(&mut db, "INSERT INTO emp VALUES (1,'ann',10),(2,'bob',20),(3,'cy',NULL)");
    exec(&mut db, "CREATE TABLE dept(dept_id INTEGER, dname TEXT)");
    exec(&mut db, "INSERT INTO dept VALUES (10,'eng'),(20,'sales'),(30,'ops')");
    exec(&mut db, "CREATE TABLE deptu(dept_id INTEGER, dname TEXT)");
    exec(&mut db, "INSERT INTO deptu VALUES (10,'eng'),(20,'sales'),(30,'ops')");
    db
}

// ---------------------------------------------------------------------------
// CROSS JOIN / comma join — the pure cartesian product (§2, §2.2).
//
// emp has 3 rows and dept has 3 rows, so the product has 3 × 3 = 9 rows,
// regardless of any NULL in a join column (no ON/USING filter is applied).
// ---------------------------------------------------------------------------

#[test]
fn cross_join_count_is_cartesian_product() {
    let mut db = setup();
    assert_scalar(&mut db, "SELECT count(*) FROM emp CROSS JOIN dept", int(9));
}

#[test]
fn comma_join_count_is_cartesian_product() {
    // "," is result-equivalent to CROSS/INNER JOIN; with no WHERE it is the full
    // cartesian product (§2.2).
    let mut db = setup();
    assert_scalar(&mut db, "SELECT count(*) FROM emp, dept", int(9));
}

#[test]
fn cross_join_small_explicit_rows() {
    // A tiny 2 × 2 product spelled out in full: every left row pairs with every
    // right row. ORDER BY over non-NULL keys pins the order for `assert_rows`.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE a(x INTEGER)");
    exec(&mut db, "INSERT INTO a VALUES (1),(2)");
    exec(&mut db, "CREATE TABLE b(y TEXT)");
    exec(&mut db, "INSERT INTO b VALUES ('p'),('q')");
    assert_rows(
        &mut db,
        "SELECT x, y FROM a CROSS JOIN b ORDER BY x, y",
        &[
            vec![int(1), text("p")],
            vec![int(1), text("q")],
            vec![int(2), text("p")],
            vec![int(2), text("q")],
        ],
    );
}

// ---------------------------------------------------------------------------
// INNER JOIN ... ON, and the equivalent comma join + WHERE (§2, §2.2).
//
// The ON predicate `emp.dept_id = dept.dept_id` is true only for ann (10→eng)
// and bob (20→sales). cy has a NULL dept_id, so `NULL = dept.dept_id` is NULL,
// never true — cy is not in the result. dept 30 ('ops') matches no employee.
// ---------------------------------------------------------------------------

#[test]
fn inner_join_on_matches_pairs() {
    let mut db = setup();
    assert_rows(
        &mut db,
        "SELECT name, dname FROM emp JOIN dept ON emp.dept_id=dept.dept_id ORDER BY name",
        &[
            vec![text("ann"), text("eng")],
            vec![text("bob"), text("sales")],
        ],
    );
}

#[test]
fn comma_join_where_equals_inner_join() {
    // A comma join filtered by the same equality in WHERE is identical to the
    // INNER JOIN ... ON above (§2.2: no difference between a constraint in WHERE
    // and one in ON for an inner/cross join).
    let mut db = setup();
    assert_rows(
        &mut db,
        "SELECT name, dname FROM emp, dept WHERE emp.dept_id=dept.dept_id ORDER BY name",
        &[
            vec![text("ann"), text("eng")],
            vec![text("bob"), text("sales")],
        ],
    );
}

#[test]
fn inner_join_with_where_filter() {
    // ON selects the matched pairs; the WHERE then keeps only the 'eng' row.
    let mut db = setup();
    assert_rows(
        &mut db,
        "SELECT name, dname FROM emp JOIN dept ON emp.dept_id=dept.dept_id \
         WHERE dname='eng' ORDER BY name",
        &[vec![text("ann"), text("eng")]],
    );
}

// ---------------------------------------------------------------------------
// LEFT [OUTER] JOIN (§2).
//
// Every left (emp) row survives: ann and bob match, and cy — whose NULL dept_id
// matches nothing — is kept with NULLs in the right-hand (dname) column.
// ---------------------------------------------------------------------------

#[test]
fn left_join_keeps_unmatched_left_row() {
    let mut db = setup();
    assert_rows(
        &mut db,
        "SELECT name, dname FROM emp LEFT JOIN dept ON emp.dept_id=dept.dept_id ORDER BY name",
        &[
            vec![text("ann"), text("eng")],
            vec![text("bob"), text("sales")],
            vec![text("cy"), null()],
        ],
    );
}

#[test]
fn left_outer_join_spelling_is_equivalent() {
    // "LEFT OUTER JOIN" is documented as a synonym for "LEFT JOIN".
    let mut db = setup();
    assert_rows(
        &mut db,
        "SELECT name, dname FROM emp LEFT OUTER JOIN dept ON emp.dept_id=dept.dept_id \
         ORDER BY name",
        &[
            vec![text("ann"), text("eng")],
            vec![text("bob"), text("sales")],
            vec![text("cy"), null()],
        ],
    );
}

// ---------------------------------------------------------------------------
// USING (§2). `emp JOIN deptu USING (dept_id)` keeps rows where the shared
// dept_id compares equal and yields a SINGLE dept_id column (the right copy is
// omitted). The projected values match the inner join; the column check pins the
// coalesced-column shape.
// ---------------------------------------------------------------------------

#[test]
fn using_join_matches_pairs() {
    let mut db = setup();
    assert_rows(
        &mut db,
        "SELECT name, dname FROM emp JOIN deptu USING (dept_id) ORDER BY name",
        &[
            vec![text("ann"), text("eng")],
            vec![text("bob"), text("sales")],
        ],
    );
}

#[test]
fn using_join_yields_single_shared_column() {
    // Joined columns = all left columns, then right columns with the USING column
    // omitted: emp(id, name, dept_id) then deptu's dname (its dept_id removed) —
    // lang_select.html §2, "the column from the right-hand dataset is omitted from the
    // joined dataset". The single shared dept_id sits at the LEFT table's position.
    let mut db = setup();
    assert_columns(
        &mut db,
        "SELECT * FROM emp JOIN deptu USING (dept_id)",
        &["id", "name", "dept_id", "dname"],
    );
}

// ---------------------------------------------------------------------------
// NATURAL JOIN (§2). The only common column between emp and deptu is dept_id,
// so NATURAL adds an implicit `USING (dept_id)` — same matched pairs, same single
// shared column as the explicit USING above.
// ---------------------------------------------------------------------------

#[test]
fn natural_join_matches_on_common_column() {
    let mut db = setup();
    assert_rows(
        &mut db,
        "SELECT name, dname FROM emp NATURAL JOIN deptu ORDER BY name",
        &[
            vec![text("ann"), text("eng")],
            vec![text("bob"), text("sales")],
        ],
    );
}

#[test]
fn natural_join_yields_single_shared_column() {
    // NATURAL uses USING semantics, so the shared dept_id appears once: [id, name,
    // dept_id, dname], the right copy omitted (lang_select.html §2), exactly like the
    // explicit `using_join_yields_single_shared_column` above.
    let mut db = setup();
    assert_columns(
        &mut db,
        "SELECT * FROM emp NATURAL JOIN deptu",
        &["id", "name", "dept_id", "dname"],
    );
}

// ---------------------------------------------------------------------------
// Three-table inner join (§2: "(A join B) join C", left to right).
//
// emp⋈dept gives {ann→eng(10), bob→sales(20)}; joining proj on dept_id fans ann
// out to its two dept-10 projects and bob to his one dept-20 project.
// ---------------------------------------------------------------------------

#[test]
fn three_table_inner_join() {
    let mut db = setup();
    exec(&mut db, "CREATE TABLE proj(pid INTEGER, dept_id INTEGER, pname TEXT)");
    exec(
        &mut db,
        "INSERT INTO proj VALUES (100,10,'apollo'),(101,20,'zephyr'),(102,10,'borealis')",
    );
    assert_rows(
        &mut db,
        "SELECT name, dname, pname \
         FROM emp JOIN dept ON emp.dept_id=dept.dept_id \
                  JOIN proj ON dept.dept_id=proj.dept_id \
         ORDER BY name, pname",
        &[
            vec![text("ann"), text("eng"), text("apollo")],
            vec![text("ann"), text("eng"), text("borealis")],
            vec![text("bob"), text("sales"), text("zephyr")],
        ],
    );
}

// ---------------------------------------------------------------------------
// Outer join: ON vs WHERE placement (§2.3). The NULL-filled rows for unmatched
// left rows are added AFTER ON processing but BEFORE WHERE processing, so a
// right-referencing constraint behaves very differently in the two positions.
// ---------------------------------------------------------------------------

#[test]
fn left_join_where_referencing_right_is_effectively_inner() {
    // The LEFT JOIN produces the cy/NULL row; `WHERE dept.dept_id IS NOT NULL`
    // then drops it (NULL IS NOT NULL is false), so the result collapses to the
    // inner join's rows — the documented "effectively inner" case.
    let mut db = setup();
    assert_rows(
        &mut db,
        "SELECT name, dname FROM emp LEFT JOIN dept ON emp.dept_id=dept.dept_id \
         WHERE dept.dept_id IS NOT NULL ORDER BY name",
        &[
            vec![text("ann"), text("eng")],
            vec![text("bob"), text("sales")],
        ],
    );
}

#[test]
fn left_join_constraint_in_on_keeps_left_rows() {
    // The extra predicate `dept.dname='eng'` lives in the ON clause, so it only
    // decides what counts as a match: ann matches, bob's dept-20 row fails the
    // predicate (so bob is unmatched → NULL), and cy is unmatched → NULL. All
    // three left rows survive.
    let mut db = setup();
    assert_rows(
        &mut db,
        "SELECT name, dname FROM emp LEFT JOIN dept \
         ON emp.dept_id=dept.dept_id AND dept.dname='eng' ORDER BY name",
        &[
            vec![text("ann"), text("eng")],
            vec![text("bob"), null()],
            vec![text("cy"), null()],
        ],
    );
}

#[test]
fn left_join_constraint_in_where_drops_rows() {
    // The SAME predicate in WHERE runs after the NULL-fill: it rejects bob
    // ('sales' ≠ 'eng') and cy (NULL ≠ 'eng'), leaving only ann. Contrast with
    // `left_join_constraint_in_on_keeps_left_rows` (§2.3).
    let mut db = setup();
    assert_rows(
        &mut db,
        "SELECT name, dname FROM emp LEFT JOIN dept ON emp.dept_id=dept.dept_id \
         WHERE dept.dname='eng' ORDER BY name",
        &[vec![text("ann"), text("eng")]],
    );
}

// ---------------------------------------------------------------------------
// RIGHT / FULL JOIN (§2). Documented mirror / union of LEFT JOIN. These are
// newer SQLite features; if the engine does not support them the spec-correct
// assertion FAILS rather than being weakened. Multiset form is used
// because the unmatched rows carry NULL in a sort key.
// ---------------------------------------------------------------------------

#[test]
fn right_join_adds_unmatched_right_row() {
    // Matched pairs plus dept 30 ('ops'), which no employee matches, NULL-filled
    // on the left. cy (an unmatched LEFT row) is NOT added by a RIGHT JOIN.
    let mut db = setup();
    assert_rows_unordered(
        &mut db,
        "SELECT name, dname FROM emp RIGHT JOIN dept ON emp.dept_id=dept.dept_id",
        &[
            vec![text("ann"), text("eng")],
            vec![text("bob"), text("sales")],
            vec![null(), text("ops")],
        ],
    );
}

#[test]
fn full_join_adds_unmatched_from_both_sides() {
    // Union of LEFT and RIGHT: matched pairs, plus cy (unmatched left, NULL
    // dname) and dept 30 'ops' (unmatched right, NULL name).
    let mut db = setup();
    assert_rows_unordered(
        &mut db,
        "SELECT name, dname FROM emp FULL JOIN dept ON emp.dept_id=dept.dept_id",
        &[
            vec![text("ann"), text("eng")],
            vec![text("bob"), text("sales")],
            vec![text("cy"), null()],
            vec![null(), text("ops")],
        ],
    );
}

// ---------------------------------------------------------------------------
// NULL join keys never match (§2, ON processing, lines ~11145-11148: "Only rows
// for which the expression evaluates to true are included").
//
// The main INNER-join fixture excludes cy only because no dept row has a NULL
// dept_id, so it cannot by itself distinguish "a NULL key never matches" from
// "there was simply no right row". Here BOTH sides carry a NULL key, so the only
// surviving pair is the non-NULL match: `NULL = NULL` is NULL (not true), so the
// two NULL-keyed rows do not join each other.
// ---------------------------------------------------------------------------

#[test]
fn inner_join_null_key_never_matches() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE l(k INTEGER, lv TEXT)");
    exec(&mut db, "INSERT INTO l VALUES (1,'a'),(NULL,'lnull')");
    exec(&mut db, "CREATE TABLE r(k INTEGER, rv TEXT)");
    exec(&mut db, "INSERT INTO r VALUES (1,'x'),(NULL,'rnull')");
    assert_rows(
        &mut db,
        "SELECT lv, rv FROM l JOIN r ON l.k=r.k ORDER BY lv",
        &[vec![text("a"), text("x")]],
    );
}

#[test]
fn left_join_null_key_keeps_left_with_nulls() {
    // The left NULL-keyed row matches nothing (NULL = anything is never true), so
    // a LEFT JOIN keeps it with NULLs on the right (§2, LEFT JOIN NULL-fill).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE l(k INTEGER, lv TEXT)");
    exec(&mut db, "INSERT INTO l VALUES (1,'a'),(NULL,'lnull')");
    exec(&mut db, "CREATE TABLE r(k INTEGER, rv TEXT)");
    exec(&mut db, "INSERT INTO r VALUES (1,'x'),(NULL,'rnull')");
    assert_rows(
        &mut db,
        "SELECT lv, rv FROM l LEFT JOIN r ON l.k=r.k ORDER BY lv",
        &[
            vec![text("a"), text("x")],
            vec![text("lnull"), null()],
        ],
    );
}

// ---------------------------------------------------------------------------
// NATURAL with NO common columns has no effect => cartesian product (§2, lines
// ~11171-11173: "If the left and right-hand input datasets feature no common
// column names, then the NATURAL keyword has no effect on the results of the
// join").
// ---------------------------------------------------------------------------

#[test]
fn natural_join_no_common_columns_is_cartesian_product() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE p(pk INTEGER)");
    exec(&mut db, "INSERT INTO p VALUES (1),(2)");
    exec(&mut db, "CREATE TABLE q(qk INTEGER)");
    exec(&mut db, "INSERT INTO q VALUES (10),(20)");
    assert_rows(
        &mut db,
        "SELECT pk, qk FROM p NATURAL JOIN q ORDER BY pk, qk",
        &[
            vec![int(1), int(10)],
            vec![int(1), int(20)],
            vec![int(2), int(10)],
            vec![int(2), int(20)],
        ],
    );
}

// ---------------------------------------------------------------------------
// USING/NATURAL — the coalesced join column is a single, unambiguously
// referenceable column carrying the join key (§2, lines ~11163-11166). These
// pin the *value* of the shared column (the `*_yields_single_shared_column`
// tests above pin only its shape), and extend the shape rule to the outer-join
// path and to a join over MULTIPLE common columns.
// ---------------------------------------------------------------------------

#[test]
fn using_join_coalesced_column_is_referenceable() {
    // An unqualified `dept_id` resolves to the single coalesced column and carries the
    // join key value (§2: the right copy is omitted, leaving one unambiguous name).
    let mut db = setup();
    assert_rows(
        &mut db,
        "SELECT name, dept_id FROM emp JOIN deptu USING (dept_id) ORDER BY name",
        &[
            vec![text("ann"), int(10)],
            vec![text("bob"), int(20)],
        ],
    );
}

#[test]
fn left_join_using_coalesced_column_on_outer_path() {
    // On a LEFT JOIN the coalesced column still resolves to one column; the unmatched
    // left row (cy) carries its own NULL dept_id — COALESCE(NULL, NULL) = NULL, since
    // the right side is also NULL-extended on that outer-left row.
    let mut db = setup();
    assert_rows(
        &mut db,
        "SELECT name, dept_id FROM emp LEFT JOIN deptu USING (dept_id) ORDER BY name",
        &[
            vec![text("ann"), int(10)],
            vec![text("bob"), int(20)],
            vec![text("cy"), null()],
        ],
    );
}

// ---------------------------------------------------------------------------
// USING/NATURAL coalescing on the RIGHT-outer path (§2). A RIGHT JOIN preserves the
// right dataset and NULL-fills the LEFT columns on a right row that matches no left
// row. The coalesced shared column is `COALESCE(left, right)`, so on such an
// unmatched-right row it is the PRESENT right value, not the NULL-extended left copy.
// This is exactly where a left-register-only resolution would be wrong, and both the
// bare-name path (`resolve_column_expr`) AND the `SELECT *` path (`expand_star_exprs`)
// must agree on that value (and with real sqlite). `deptu(30,'ops')` matches no
// employee; the LEFT emp row `cy` (NULL dept_id) is dropped by a RIGHT join. ORDER BY
// dname is non-NULL on every row (the right side is preserved), so the order is fixed.
// ---------------------------------------------------------------------------

#[test]
fn right_join_using_bare_name_coalesces_on_unmatched_right_row() {
    // Bare `dept_id` is COALESCE(emp.dept_id, deptu.dept_id); on the unmatched-right
    // 'ops' row emp.dept_id is NULL, so it yields the right value 30.
    let mut db = setup();
    assert_rows(
        &mut db,
        "SELECT dept_id, dname FROM emp RIGHT JOIN deptu USING (dept_id) ORDER BY dname",
        &[
            vec![int(10), text("eng")],
            vec![int(30), text("ops")],
            vec![int(20), text("sales")],
        ],
    );
}

#[test]
fn right_join_using_star_coalesces_on_unmatched_right_row() {
    // `SELECT *` must carry the SAME coalesced dept_id as the bare name above: the
    // 'ops' row's dept_id is the present right value 30, NOT the NULL-extended left
    // copy. A plain Column(left) here is the bug — it reads NULL and diverges from both
    // the bare name and real sqlite. Coalesced output shape is [id, name, dept_id, dname].
    let mut db = setup();
    assert_rows(
        &mut db,
        "SELECT * FROM emp RIGHT JOIN deptu USING (dept_id) ORDER BY dname",
        &[
            vec![int(1), text("ann"), int(10), text("eng")],
            vec![null(), null(), int(30), text("ops")],
            vec![int(2), text("bob"), int(20), text("sales")],
        ],
    );
}

#[test]
fn natural_right_join_star_coalesces_on_unmatched_right_row() {
    // NATURAL adds an implicit USING (dept_id): same RIGHT-outer coalescing as the
    // explicit-USING case, exercising the NATURAL shared-set path. The unmatched-right
    // 'ops' row's `*` dept_id is the present right value 30.
    let mut db = setup();
    assert_rows(
        &mut db,
        "SELECT * FROM emp NATURAL RIGHT JOIN deptu ORDER BY dname",
        &[
            vec![int(1), text("ann"), int(10), text("eng")],
            vec![null(), null(), int(30), text("ops")],
            vec![int(2), text("bob"), int(20), text("sales")],
        ],
    );
}

#[test]
fn left_join_using_star_coalesces_on_unmatched_left_row() {
    // The LEFT-outer mirror of the RIGHT-join witnesses above, and the end-to-end
    // complement of the LEFT-join bare-name test: a LEFT JOIN preserves the LEFT dataset
    // and NULL-fills the RIGHT columns on a left row that matches no right row. The
    // coalesced shared column is COALESCE(left, right); on such an unmatched-LEFT row it
    // is the PRESENT left value (right NULL-extended). The shared fixture has no
    // unmatched-left row with a NON-NULL key (emp's 10/20 both match; cy is NULL), so a
    // local fixture supplies one (k=99, absent from the right). This pins that `*` (and a
    // bare name) coalesces genuinely: a hypothetical right-only resolution would read
    // NULL here, so it distinguishes COALESCE(left,right) from Column(right) end-to-end,
    // just as the RIGHT-join cases distinguish it from Column(left).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE l(k INTEGER, a TEXT)");
    exec(&mut db, "INSERT INTO l VALUES (10,'p'),(99,'q')");
    exec(&mut db, "CREATE TABLE r(k INTEGER, b TEXT)");
    exec(&mut db, "INSERT INTO r VALUES (10,'x')");
    // Bare name: the unmatched-left 'q' row's k is the present left value 99, not NULL.
    assert_rows(
        &mut db,
        "SELECT k, a FROM l LEFT JOIN r USING (k) ORDER BY a",
        &[vec![int(10), text("p")], vec![int(99), text("q")]],
    );
    // `*` agrees, at the coalesced [k, a, b] shape (the right copy of k dropped).
    assert_rows(
        &mut db,
        "SELECT * FROM l LEFT JOIN r USING (k) ORDER BY a",
        &[
            vec![int(10), text("p"), text("x")],
            vec![int(99), text("q"), null()],
        ],
    );
}

#[test]
fn natural_join_multiple_common_columns_matches() {
    // Two common columns (a, b): the implicit USING contains BOTH, so a row joins
    // only when a AND b are equal (§2, lines ~11168-11171).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE m1(a INTEGER, b INTEGER, v1 TEXT)");
    exec(&mut db, "INSERT INTO m1 VALUES (1,1,'p'),(1,2,'q'),(2,2,'r')");
    exec(&mut db, "CREATE TABLE m2(a INTEGER, b INTEGER, v2 TEXT)");
    exec(&mut db, "INSERT INTO m2 VALUES (1,1,'x'),(2,2,'y'),(1,3,'z')");
    assert_rows(
        &mut db,
        "SELECT v1, v2 FROM m1 NATURAL JOIN m2 ORDER BY v1",
        &[
            vec![text("p"), text("x")],
            vec![text("r"), text("y")],
        ],
    );
}

#[test]
fn using_join_multiple_columns_matches() {
    // Explicit USING (a, b) is the same multi-column equi-join as the NATURAL
    // case above.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE m1(a INTEGER, b INTEGER, v1 TEXT)");
    exec(&mut db, "INSERT INTO m1 VALUES (1,1,'p'),(1,2,'q'),(2,2,'r')");
    exec(&mut db, "CREATE TABLE m2(a INTEGER, b INTEGER, v2 TEXT)");
    exec(&mut db, "INSERT INTO m2 VALUES (1,1,'x'),(2,2,'y'),(1,3,'z')");
    assert_rows(
        &mut db,
        "SELECT v1, v2 FROM m1 JOIN m2 USING (a, b) ORDER BY v1",
        &[
            vec![text("p"), text("x")],
            vec![text("r"), text("y")],
        ],
    );
}

#[test]
fn natural_join_multiple_common_columns_single_shape() {
    // Both common columns are coalesced to a single copy each: joined columns are
    // a, b (from the left), v1, then v2 (m2's a and b omitted). Exercises the
    // omission over N>1 shared columns, not just one.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE m1(a INTEGER, b INTEGER, v1 TEXT)");
    exec(&mut db, "INSERT INTO m1 VALUES (1,1,'p')");
    exec(&mut db, "CREATE TABLE m2(a INTEGER, b INTEGER, v2 TEXT)");
    exec(&mut db, "INSERT INTO m2 VALUES (1,1,'x')");
    assert_columns(
        &mut db,
        "SELECT * FROM m1 NATURAL JOIN m2",
        &["a", "b", "v1", "v2"],
    );
}

// ---------------------------------------------------------------------------
// USING/NATURAL coalescing across a THREE-table chain (§2). A join chain
// `t1 JOIN t2 USING(k) JOIN t3 USING(k)` is left-associative: the second join's
// left input is the ALREADY-coalesced `(t1 JOIN t2)` dataset, so its shared `k`
// coalesces the first join's `k` with t3's — i.e. the whole-chain value is
// COALESCE(t1.k, t2.k, t3.k), present wherever ANY of the three has it. The plan
// records ONE coalesced pair per step ({k: t1.k, t2.k} then {k: t1.k, t3.k}), so a
// resolution that used only the FIRST pair would fold to COALESCE(t1.k, t2.k) and
// DROP t3.k — reading NULL on a row present only in t3 (t1, t2 NULL-extended by the
// RIGHT join). These pin the N-ary fold over the whole chain, for both the bare name
// and `*`. The RIGHT JOIN onto t3 makes the third operand load-bearing: t3's k=30
// row matches nothing in (t1 JOIN t2), so t1.k and t2.k are NULL there and only
// COALESCE-ing t3.k in yields the documented present value.
// ---------------------------------------------------------------------------

/// Three tables sharing a `k` column: t1/t2 match only at k=10/20, and t3 carries a
/// k=30 row that matches neither — the unmatched-right row a RIGHT join preserves.
fn setup_using_chain() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t1(k INTEGER, a TEXT)");
    exec(&mut db, "INSERT INTO t1 VALUES (10,'a10'),(20,'a20')");
    exec(&mut db, "CREATE TABLE t2(k INTEGER, b TEXT)");
    exec(&mut db, "INSERT INTO t2 VALUES (10,'b10'),(20,'b20')");
    exec(&mut db, "CREATE TABLE t3(k INTEGER, c TEXT)");
    exec(&mut db, "INSERT INTO t3 VALUES (10,'c10'),(30,'c30')");
    db
}

#[test]
fn using_chain_coalesces_bare_name_across_three_tables() {
    // Bare `k` over `t1 JOIN t2 USING(k) RIGHT JOIN t3 USING(k)` is the whole-chain
    // COALESCE(t1.k, t2.k, t3.k). On the RIGHT-preserved t3 row k=30, t1.k and t2.k are
    // NULL-extended, so k is the present t3 value 30 — NOT the NULL a two-operand
    // COALESCE(t1.k, t2.k) would read. (k=20 is dropped: it matches no t3 row and a
    // RIGHT join does not preserve the left.)
    let mut db = setup_using_chain();
    assert_rows(
        &mut db,
        "SELECT k, c FROM t1 JOIN t2 USING (k) RIGHT JOIN t3 USING (k) ORDER BY c",
        &[vec![int(10), text("c10")], vec![int(30), text("c30")]],
    );
}

#[test]
fn using_chain_coalesces_star_across_three_tables() {
    // `SELECT *` must carry the SAME whole-chain coalesced `k` as the bare name above:
    // one `k` column (t2's and t3's copies dropped), and on the k=30 row its value is the
    // present t3 value 30, with a and b NULL-extended. Coalesced shape is [k, a, b, c].
    let mut db = setup_using_chain();
    assert_rows(
        &mut db,
        "SELECT * FROM t1 JOIN t2 USING (k) RIGHT JOIN t3 USING (k) ORDER BY c",
        &[
            vec![int(10), text("a10"), text("b10"), text("c10")],
            vec![int(30), null(), null(), text("c30")],
        ],
    );
}

// ---------------------------------------------------------------------------
// The coalesced USING/NATURAL column is a first-class column: it can be a GROUP BY
// key, appear inside a compound projection expression over that key, and be filtered
// in HAVING — resolving to the single shared column throughout (§2, the shared column
// is unambiguous). These guard that the coalescing integrates with the grouping/
// aggregate path, not just with a bare projection. The INNER join keeps ann→10 and
// bob→20, giving two single-row groups (dept_id 10 and 20).
// ---------------------------------------------------------------------------

#[test]
fn coalesced_column_is_a_valid_group_by_key() {
    // GROUP BY over the coalesced dept_id groups by the single shared column; ORDER BY
    // dept_id names the output column (present in the SELECT list), so this is a clean
    // grouped read over the coalesced key.
    let mut db = setup();
    assert_rows(
        &mut db,
        "SELECT dept_id, count(*) FROM emp JOIN deptu USING (dept_id) \
         GROUP BY dept_id ORDER BY dept_id",
        &[vec![int(10), int(1)], vec![int(20), int(1)]],
    );
}

#[test]
fn coalesced_group_key_in_a_compound_projection() {
    // The coalesced dept_id resolves inside a compound projection (`dept_id + 100`) by
    // matching the GROUP BY key, so each group projects its key arithmetic. ORDER BY 1
    // sorts by the projected first column (avoiding an unrelated ORDER-BY-on-aggregate
    // path). Groups 10/20 → 110/120.
    let mut db = setup();
    assert_rows(
        &mut db,
        "SELECT dept_id + 100, count(*) FROM emp JOIN deptu USING (dept_id) \
         GROUP BY dept_id ORDER BY 1",
        &[vec![int(110), int(1)], vec![int(120), int(1)]],
    );
}

#[test]
fn coalesced_group_key_in_having() {
    // HAVING filters on the coalesced dept_id (resolved as the group key), keeping only
    // the dept_id=20 group.
    let mut db = setup();
    assert_rows(
        &mut db,
        "SELECT dept_id, count(*) FROM emp JOIN deptu USING (dept_id) \
         GROUP BY dept_id HAVING dept_id >= 20 ORDER BY dept_id",
        &[vec![int(20), int(1)]],
    );
}

// ---------------------------------------------------------------------------
// `SELECT *` over a coalesced GROUP BY key (§2 + lang_select.html grouping). A bare
// `k` in the projection already resolves to its group key (the binder matches a
// projection sub-expression against a whole GROUP BY key), but `*` expands to the
// coalesced column by REGISTER, and a coalesced group key binds to COALESCE(...)
// rather than a single register — so an unqualified `*` whose ONLY columns are the
// coalesced group key(s) must still project the group value, exactly as the bare
// reference does. Real sqlite returns the grouped rows here; the coalesced column is
// a first-class group key whether reached by name or by `*`. These pin that a `*`
// coalesced group key reads the group value (the star analogue of the bare-name /
// compound-projection / HAVING cases above), for both USING and NATURAL, single and
// multiple shared keys.
// ---------------------------------------------------------------------------

#[test]
fn coalesced_group_key_star_projects_group_value() {
    // Both tables have ONLY the shared `k`, so `SELECT *` is exactly the coalesced key.
    // `ak` has k = 1 (×2), 2; `bk` has k = 1, 2 (×2); the inner USING join keeps every
    // equal-k pair, and GROUP BY k collapses to the distinct present keys {1, 2}. `*`
    // projects the single coalesced k = the group value, NOT the "must appear in GROUP
    // BY" error a register-only star lookup would raise (the coalesced key never enters
    // the register→group map, so `*` must match it the way a bare `k` does).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE ak(k INTEGER)");
    exec(&mut db, "INSERT INTO ak VALUES (1),(1),(2)");
    exec(&mut db, "CREATE TABLE bk(k INTEGER)");
    exec(&mut db, "INSERT INTO bk VALUES (1),(2),(2)");
    assert_rows(
        &mut db,
        "SELECT * FROM ak JOIN bk USING (k) GROUP BY k ORDER BY k",
        &[vec![int(1)], vec![int(2)]],
    );
}

#[test]
fn coalesced_group_key_natural_star_projects_multiple_group_values() {
    // NATURAL over two shared columns (a, b) coalesces BOTH; with only those two columns
    // present, `SELECT *` is [a, b] and GROUP BY a, b makes each a coalesced group key.
    // Each star column must match its own group key (a→group a, b→group b), so the
    // grouped `*` reads [group a, group b] — the distinct present pairs {(1,1),(2,2)}.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE m1(a INTEGER, b INTEGER)");
    exec(&mut db, "INSERT INTO m1 VALUES (1,1),(1,1),(2,2)");
    exec(&mut db, "CREATE TABLE m2(a INTEGER, b INTEGER)");
    exec(&mut db, "INSERT INTO m2 VALUES (1,1),(2,2),(2,2)");
    assert_rows(
        &mut db,
        "SELECT * FROM m1 NATURAL JOIN m2 GROUP BY a, b ORDER BY a, b",
        &[vec![int(1), int(1)], vec![int(2), int(2)]],
    );
}

#[test]
fn qualified_coalesced_key_stays_left_copy_on_outer_group() {
    // The non-regression guard for the star fix above: a QUALIFIED `a.k` (NOT a `*`)
    // must stay the LEFT table's copy, NOT the coalesced group value. On the RIGHT-outer
    // group k=2 (b's unmatched row, `a` NULL-extended) real sqlite reports a.k = NULL —
    // the left copy — even though the coalesced group value is 2. The star fix is
    // deliberately star-path-local (a qualified reference binds through a different path),
    // so this must be unchanged by it. ORDER BY 2 sorts by the non-NULL max(bv) column.
    let mut db = setup_minmax_coalesce();
    assert_rows(
        &mut db,
        "SELECT a.k, max(bv) FROM a RIGHT JOIN b USING (k) GROUP BY k ORDER BY 2",
        &[vec![int(1), text("b1")], vec![null(), text("b2")]],
    );
}

// ---------------------------------------------------------------------------
// USING/NATURAL coalescing through the single-min/max §2.5 bare-column capture.
// lang_select.html §2.5: when a query has exactly one min()/max() and other bare
// (non-aggregated, non-grouped) columns, those columns take their value from the
// EXTREMUM row. A coalesced shared column is one such bare column, so its captured
// value must be the whole COALESCE over its component copies — NOT the left copy
// alone, which is NULL-extended when the extremum lands on an unmatched outer row.
// This is the exact shape the plain projection already coalesces; a single max()
// must not revert it to the left copy. Fixture: `a` has only k=1, `b` has k=1 and
// k=2, so `a RIGHT JOIN b` preserves b's k=2 row with `a` NULL-extended, and
// max(bv)='b2' lands on THAT row — making the third/right operand load-bearing.
// ---------------------------------------------------------------------------

/// `a` (one row, k=1) and `b` (k=1 and the extra k=2) so a RIGHT/NATURAL join over
/// the shared `k` preserves b's unmatched k=2 row (a NULL-extended there), and
/// max(bv) selects that row as the extremum.
fn setup_minmax_coalesce() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE a(k INTEGER, av TEXT)");
    exec(&mut db, "INSERT INTO a VALUES (1,'a1')");
    exec(&mut db, "CREATE TABLE b(k INTEGER, bv TEXT)");
    exec(&mut db, "INSERT INTO b VALUES (1,'b1'),(2,'b2')");
    db
}

#[test]
fn single_max_captures_coalesced_bare_name_on_outer_row() {
    // §2.5 bare-column capture of the coalesced `k`: max(bv)='b2' picks b's unmatched
    // k=2 row, where a.k is NULL-extended. `k` must be COALESCE(a.k, b.k) = 2, not the
    // captured left copy a.k = NULL. (RIGHT JOIN drops a's unmatched k=1... no: a's k=1
    // DOES match b's k=1; the load-bearing row is b's k=2.)
    let mut db = setup_minmax_coalesce();
    assert_rows(
        &mut db,
        "SELECT k, max(bv) FROM a RIGHT JOIN b USING (k)",
        &[vec![int(2), text("b2")]],
    );
}

#[test]
fn single_max_captures_coalesced_star_on_outer_row() {
    // The `*` capture path must fold the coalesced `k` too: `SELECT *, max(bv)` over the
    // same join yields [k, av, bv, max(bv)]. On the extremum k=2 row, k is the coalesced
    // value 2 (not the NULL left copy), av is the NULL-extended a.av, and bv is b's 'b2'.
    let mut db = setup_minmax_coalesce();
    assert_rows(
        &mut db,
        "SELECT *, max(bv) FROM a RIGHT JOIN b USING (k)",
        &[vec![int(2), null(), text("b2"), text("b2")]],
    );
}

#[test]
fn single_max_captures_natural_coalesced_bare_name_on_outer_row() {
    // NATURAL adds an implicit USING(k) (k is the only common column), so the §2.5
    // capture folds the coalesced key identically to the explicit-USING case — pinning
    // the fold on the NATURAL shared-set path, not just USING.
    let mut db = setup_minmax_coalesce();
    assert_rows(
        &mut db,
        "SELECT k, max(bv) FROM a NATURAL RIGHT JOIN b",
        &[vec![int(2), text("b2")]],
    );
}

// ---------------------------------------------------------------------------
// NATURAL may not carry an ON or USING constraint (§2, line ~11173: "A USING or
// ON clause may not be added to a join that specifies the NATURAL keyword"). The
// parser must reject the combination.
// ---------------------------------------------------------------------------

#[test]
fn natural_join_with_on_is_error() {
    // Pin the REASON, not merely that it errored: `emp NATURAL JOIN deptu` is valid on its
    // own (the sibling tests run it successfully), so the ONLY failure source is the illegal
    // NATURAL+ON combination. Asserting the message keeps an unrelated error from keeping
    // this case falsely passing.
    let mut db = setup();
    let e = assert_query_error(
        &mut db,
        "SELECT * FROM emp NATURAL JOIN deptu ON emp.dept_id=deptu.dept_id",
    );
    assert!(
        e.to_string().to_lowercase().contains("natural"),
        "expected the NATURAL+ON rejection, got: {e:?}",
    );
}

#[test]
fn natural_join_with_using_is_error() {
    // As above: only the illegal NATURAL+USING combination can make this valid join fail,
    // so pin that the error is about NATURAL rather than any error at all.
    let mut db = setup();
    let e = assert_query_error(&mut db, "SELECT * FROM emp NATURAL JOIN deptu USING (dept_id)");
    assert!(
        e.to_string().to_lowercase().contains("natural"),
        "expected the NATURAL+USING rejection, got: {e:?}",
    );
}

// ---------------------------------------------------------------------------
// A rowid equijoin whose key is a TEXT column (`ON r.rowid = t.b`, t.b TEXT).
// The comparison applies NUMERIC affinity to the text operand (§ affinity rules
// for a rowid = column comparison), and the planner runs it as a per-left rowid
// seek (an index-nested-loop; see minisqlite-plan `rowid_seek_key`). The seek
// value is coerced with sqlite's OP_SeekRowid rule (`must_be_int`: NUMERIC
// affinity, then a lossless integer or no row) — the SAME affinity the ON
// applies — and the join re-checks the full ON after each seek, so it keeps
// exactly the rows `r.rowid = t.b` keeps: a numeric string ('10') and an
// integral-real string ('30.0') reach their integer rowid, while a non-matching
// rowid ('99'), a non-numeric string ('abc'), and NULL name no row. This pins
// the affinity-carrying key producing the right rows through the real
// planner + executor, not merely the fast plan shape.
// ---------------------------------------------------------------------------

#[test]
fn rowid_equijoin_on_a_text_key_returns_the_coerced_rows() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(b TEXT)");
    exec(&mut db, "INSERT INTO t(b) VALUES ('10'),('20'),('30.0'),('99'),('abc'),(NULL)");
    exec(&mut db, "CREATE TABLE r(v TEXT)");
    exec(&mut db, "INSERT INTO r(rowid, v) VALUES (10,'ten'),(20,'twenty'),(30,'thirty')");
    assert_rows_unordered(
        &mut db,
        "SELECT t.b, r.v FROM t JOIN r ON r.rowid = t.b",
        &[
            vec![text("10"), text("ten")],
            vec![text("20"), text("twenty")],
            vec![text("30.0"), text("thirty")],
        ],
    );
}

#[test]
fn left_rowid_equijoin_on_a_text_key_null_fills_a_missed_seek() {
    // The LEFT-join composition of the same text-keyed rowid seek: a left row whose text
    // key coerces and hits ('10' -> r.rowid 10) pairs normally, while a left row whose key
    // NAMES NO ROW under OP_SeekRowid — a non-numeric string ('abc') or NULL — must still
    // be emitted once, right half NULL-filled (§2.3 LEFT JOIN: unmatched left rows are
    // preserved). This pins the coerced-key seek MISS interacting with the IndexNestedLoop
    // null-fill, the piece the INNER text test and the integer-keyed LEFT test don't cross.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(b TEXT)");
    exec(&mut db, "INSERT INTO t(b) VALUES ('10'),('abc'),(NULL)");
    exec(&mut db, "CREATE TABLE r(v TEXT)");
    exec(&mut db, "INSERT INTO r(rowid, v) VALUES (10,'ten'),(20,'twenty')");
    assert_rows_unordered(
        &mut db,
        "SELECT t.b, r.v FROM t LEFT JOIN r ON r.rowid = t.b",
        &[
            vec![text("10"), text("ten")],
            vec![text("abc"), null()],
            vec![null(), null()],
        ],
    );
}
