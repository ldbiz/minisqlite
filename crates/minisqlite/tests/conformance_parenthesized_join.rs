//! Conformance battery: PARENTHESIZED JOINs in the FROM clause — a
//! `table-or-subquery` that is itself a bracketed `join-clause`, e.g.
//! `FROM (a JOIN b ON ...) JOIN c ON ...` and
//! `FROM a LEFT JOIN (b JOIN c ON ...) ON ...`.
//!
//! Every expected value is TRANSCRIBED FROM THE SPEC,
//! `spec/sqlite-doc/lang_select.html` §2.1 ("Determination of input data (FROM
//! clause processing)"), never from what the engine returns. Binding rules used:
//!
//!   * GRAMMAR: the `table-or-subquery` railroad diagram lists a parenthesized
//!     `join-clause` — `( join-clause )` — as one production. So a parenthesized
//!     join is a SUB-JOIN evaluated as a unit and then used, whole, as one operand
//!     (left or right) of the surrounding join. The constituent tables keep their
//!     names outside the parentheses (unlike a `( select-stmt )` derived table,
//!     which needs an alias and hides its inner names).
//!   * §2.1: "All joins in SQLite are based on the cartesian product of the left
//!     and right-hand datasets. The columns of the cartesian product dataset are,
//!     in order, all the columns of the left-hand dataset followed by all the
//!     columns of the right-hand dataset."
//!   * §2.1 (ON): "Only rows for which the expression evaluates to true are
//!     included." A NULL ON result (e.g. a NULL join key) is therefore NOT a match.
//!   * §2.1 (LEFT [OUTER] JOIN): after ON/USING filtering, "an extra row is added
//!     to the output for each row in the original left-hand input dataset that does
//!     not match any row in the right-hand dataset", NULL-filled on the right.
//!   * §2.1 (USING): "the column from the right-hand dataset is omitted from the
//!     joined dataset"; the kept rows are those where `lhs.X = rhs.X` is true.
//!   * §2.1 (NATURAL): "an implicit USING clause is added ... [containing] each of
//!     the column names that appear in both the left and right-hand input datasets."
//!   * §2.1 (LEFT-ASSOCIATIVITY): "When more than two tables are joined together as
//!     part of a FROM clause, the join operations are processed in order from left
//!     to right. In other words, the FROM clause (A join-op-1 B join-op-2 C) is
//!     computed as ((A join-op-1 B) join-op-2 C)." So for INNER joins parentheses
//!     are result-preserving — `(a JOIN b) JOIN c`, `a JOIN (b JOIN c)`, and the
//!     flat `a JOIN b JOIN c` all yield the SAME row multiset (the cartesian
//!     product is associative and the conjunctive ON filters commute). For OUTER
//!     joins, parentheses CHANGE the meaning: `a LEFT JOIN (b JOIN c ON ...)`
//!     first computes `(b JOIN c)` then LEFT-joins `a` to that composite, so its
//!     NULL-extension differs from `(a LEFT JOIN b) ... JOIN c`.
//!   * §2.3: an outer join's extra NULL rows "are added after ON clause processing
//!     but before WHERE clause processing" (a WHERE over a paren join filters the
//!     already-NULL-extended rows).
//!
//! Not yet implemented: a parenthesized join in FROM is *parsed*
//! (`minisqlite_sql`'s `parse_table_or_subquery` yields `TableOrSubquery::Join`)
//! but the planner REJECTS it in `crates/minisqlite-plan/src/compile/from.rs`
//! (`resolve_table`, the `TableOrSubquery::Join(_)` arm) with
//! `Error::sql("a parenthesized join in FROM is not yet supported")`. The reject
//! happens at PLAN time, so EVERY query below `panic!`s today through the harness
//! `query`/`exec` helpers — that panic is the intended loud, honest failure. Each
//! test asserts the spec-correct RESULT SET, so it fails now and will pass with
//! no test change once the engine implements parenthesized joins. The spec-correct
//! assertion is never weakened to make it pass, none are `#[ignore]`d, and none
//! assert the "not yet supported" text.

mod conformance;
use conformance::*;

// `Connection` (fixture-helper signatures) and `Value` (the shared expected-row
// helper's return type) are re-exported by the facade but are only imported
// PRIVATELY inside the harness module, so they are not in scope via `conformance::*`.
use minisqlite::{Connection, Value};

// ---------------------------------------------------------------------------
// Fixtures. A fresh in-memory database per test keeps them independent.
// ---------------------------------------------------------------------------

/// A three-table CHAIN: `a.ak = b.bk` links a→b, and `b.bk2 = c.ck` links b→c.
/// `a3`/`b3` have no `c` (b3.bk2=400 matches no ck), and a3 matches only b3, so
/// the inner 3-way join is `(a1,b1,c1),(a2,b2,c2),(a2,b2b,c3)` — with fan-out
/// (a2→two rows) and a chain end (b3) that only an OUTER join keeps.
fn chain() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE a(ak INTEGER, av TEXT)");
    exec(&mut db, "INSERT INTO a VALUES (1,'a1'),(2,'a2'),(3,'a3')");
    exec(&mut db, "CREATE TABLE b(bk INTEGER, bk2 INTEGER, bv TEXT)");
    exec(&mut db, "INSERT INTO b VALUES (1,100,'b1'),(2,200,'b2'),(2,300,'b2b'),(3,400,'b3')");
    exec(&mut db, "CREATE TABLE c(ck INTEGER, cv TEXT)");
    exec(&mut db, "INSERT INTO c VALUES (100,'c1'),(200,'c2'),(300,'c3')");
    db
}

/// A single-key fixture tuned to expose OUTER-join grouping: `a3` matches no `b`,
/// `b2` matches no `c`, and `c` holds only `k=1`. This lets `(b JOIN c)` drop `b2`
/// while `(b LEFT JOIN c)` keeps it, and lets an unmatched `a3` distinguish an
/// outer LEFT from an inner outer join.
fn keyed3() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE a(ak INTEGER, av TEXT)");
    exec(&mut db, "INSERT INTO a VALUES (1,'a1'),(2,'a2'),(3,'a3')");
    exec(&mut db, "CREATE TABLE b(bk INTEGER, bv TEXT)");
    exec(&mut db, "INSERT INTO b VALUES (1,'b1'),(2,'b2')");
    exec(&mut db, "CREATE TABLE c(ck INTEGER, cv TEXT)");
    exec(&mut db, "INSERT INTO c VALUES (1,'c1')");
    db
}

/// Three tables that all share the column name `k` (and nothing else), so USING(k)
/// and NATURAL have a well-defined single common column. `c` lacks `k=3`, so an
/// outer join over the shared key has an unmatched row to NULL-extend.
fn sharedk() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE a(k INTEGER, av TEXT)");
    exec(&mut db, "INSERT INTO a VALUES (1,'a1'),(2,'a2'),(3,'a3')");
    exec(&mut db, "CREATE TABLE b(k INTEGER, bv TEXT)");
    exec(&mut db, "INSERT INTO b VALUES (1,'b1'),(2,'b2'),(3,'b3')");
    exec(&mut db, "CREATE TABLE c(k INTEGER, cv TEXT)");
    exec(&mut db, "INSERT INTO c VALUES (1,'c1'),(2,'c2')");
    db
}

/// Four tables for joining TWO parenthesized sub-joins together: `a.ak=b.bk`,
/// `c.ck=d.dk`, and the composites linked by `a.ak=c.ck`. `a3`/`b3` have no `c`/`d`
/// partner, so an outer join between the two composites has a row to NULL-extend.
fn four() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE a(ak INTEGER, av TEXT)");
    exec(&mut db, "INSERT INTO a VALUES (1,'a1'),(2,'a2'),(3,'a3')");
    exec(&mut db, "CREATE TABLE b(bk INTEGER, bv TEXT)");
    exec(&mut db, "INSERT INTO b VALUES (1,'b1'),(2,'b2'),(3,'b3')");
    exec(&mut db, "CREATE TABLE c(ck INTEGER, cv TEXT)");
    exec(&mut db, "INSERT INTO c VALUES (1,'c1'),(2,'c2')");
    exec(&mut db, "CREATE TABLE d(dk INTEGER, dv TEXT)");
    exec(&mut db, "INSERT INTO d VALUES (1,'d1'),(2,'d2')");
    db
}

/// The hand-derived inner 3-way result of the `chain()` fixture, projecting
/// `a.av, b.bv, c.cv` ordered by `a.av, b.bv`. Shared by the four INNER
/// result-preserving tests so the "parentheses don't change an inner join"
/// oracle is written down exactly once.
fn chain_inner_three_way() -> Vec<Vec<Value>> {
    vec![
        vec![text("a1"), text("b1"), text("c1")],
        vec![text("a2"), text("b2"), text("c2")],
        vec![text("a2"), text("b2b"), text("c3")],
    ]
}

// ---------------------------------------------------------------------------
// INNER parenthesized joins are RESULT-PRESERVING (§2.1 left-associativity).
// `(a JOIN b) JOIN c`, `a JOIN (b JOIN c)`, and the flat `a JOIN b JOIN c` all
// compute the same row multiset; each test pins that hand-derived multiset.
// ---------------------------------------------------------------------------

/// §2.1 (left-associativity): a LEFT-hand parenthesized sub-join `(a JOIN b)` then
/// joined to `c` equals the flat left-deep 3-way join — parentheses on the natural
/// grouping are a no-op for INNER joins.
#[test]
fn inner_paren_on_left_matches_flat_three_way() {
    let mut db = chain();
    assert_rows(
        &mut db,
        "SELECT a.av, b.bv, c.cv \
         FROM (a JOIN b ON a.ak=b.bk) JOIN c ON b.bk2=c.ck \
         ORDER BY a.av, b.bv",
        &chain_inner_three_way(),
    );
}

/// §2.1 (associativity of the cartesian product): a RIGHT-hand parenthesized
/// sub-join `a JOIN (b JOIN c)` re-associates the same three tables and yields the
/// SAME row set as the flat/left grouping — `(b JOIN c)` is computed first, then
/// `a` joins the composite.
#[test]
fn inner_paren_on_right_matches_flat_three_way() {
    let mut db = chain();
    assert_rows(
        &mut db,
        "SELECT a.av, b.bv, c.cv \
         FROM a JOIN (b JOIN c ON b.bk2=c.ck) ON a.ak=b.bk \
         ORDER BY a.av, b.bv",
        &chain_inner_three_way(),
    );
}

/// §2.1: an OUTER pair of parentheses wrapping a left parenthesized join —
/// `((a JOIN b) JOIN c)` — is still just the inner 3-way join; nested/redundant
/// parentheses around an all-INNER join never change the result.
#[test]
fn inner_wrapped_left_nesting_matches_flat() {
    let mut db = chain();
    assert_rows(
        &mut db,
        "SELECT a.av, b.bv, c.cv \
         FROM ((a JOIN b ON a.ak=b.bk) JOIN c ON b.bk2=c.ck) \
         ORDER BY a.av, b.bv",
        &chain_inner_three_way(),
    );
}

/// §2.1: mixed nesting — an outer pair of parentheses wrapping a RIGHT
/// parenthesized join, `(a JOIN (b JOIN c) ON ...)` — likewise preserves the inner
/// 3-way result, exercising a distinct parse shape from the left-nested case.
#[test]
fn inner_wrapped_right_mixed_nesting_matches_flat() {
    let mut db = chain();
    assert_rows(
        &mut db,
        "SELECT a.av, b.bv, c.cv \
         FROM (a JOIN (b JOIN c ON b.bk2=c.ck) ON a.ak=b.bk) \
         ORDER BY a.av, b.bv",
        &chain_inner_three_way(),
    );
}

/// §2.1 / §2.3: a WHERE over a parenthesized inner join filters the join's rows and
/// the projection reads columns from all three tables. `b.bk2>=200` drops the
/// `b1` (bk2=100) row, leaving the two `a2` rows.
#[test]
fn paren_inner_join_where_filters_and_projects_all_three() {
    let mut db = chain();
    assert_rows(
        &mut db,
        "SELECT a.av, b.bv, c.cv \
         FROM (a JOIN b ON a.ak=b.bk) JOIN c ON b.bk2=c.ck \
         WHERE b.bk2 >= 200 ORDER BY b.bv",
        &[
            vec![text("a2"), text("b2"), text("c2")],
            vec![text("a2"), text("b2b"), text("c3")],
        ],
    );
}

/// §2.1 (LEFT JOIN NULL-fill): a parenthesized INNER sub-join `(a JOIN b)` then
/// LEFT-joined to `c` keeps every `(a,b)` row; the `b3` row (bk2=400 matches no
/// `c`) survives with NULLs in the `c` columns. `a3⋈b3` proves the composite's
/// rows flow into the outer LEFT join.
#[test]
fn paren_inner_then_left_join_null_extends_unmatched_c() {
    let mut db = chain();
    assert_rows(
        &mut db,
        "SELECT a.av, b.bv, c.cv \
         FROM (a JOIN b ON a.ak=b.bk) LEFT JOIN c ON b.bk2=c.ck \
         ORDER BY a.av, b.bv",
        &[
            vec![text("a1"), text("b1"), text("c1")],
            vec![text("a2"), text("b2"), text("c2")],
            vec![text("a2"), text("b2b"), text("c3")],
            vec![text("a3"), text("b3"), null()],
        ],
    );
}

/// §2.1 / GRAMMAR: a parenthesized join can be the ENTIRE FROM clause — the
/// `( join-clause )` production with nothing after it. This is the JOIN form of a
/// parenthesized `table-or-subquery`, distinct from a `( select-stmt )` derived
/// table (which is a different, already-supported feature); here the constituent
/// tables `a`/`b` stay individually named. `a3⋈b3` and the `a2` fan-out are kept.
#[test]
fn sole_parenthesized_join_as_entire_from() {
    let mut db = chain();
    assert_rows(
        &mut db,
        "SELECT a.av, b.bv FROM (a JOIN b ON a.ak=b.bk) ORDER BY a.av, b.bv",
        &[
            vec![text("a1"), text("b1")],
            vec![text("a2"), text("b2")],
            vec![text("a2"), text("b2b")],
            vec![text("a3"), text("b3")],
        ],
    );
}

// ---------------------------------------------------------------------------
// OUTER-join grouping CHANGES the result (§2.1). These are the semantically
// load-bearing cases: the same three tables and keys, re-grouped by parentheses,
// yield DISTINCT NULL-extended row sets. Each 6-column row is
// (a.ak, a.av, b.bk, b.bv, c.ck, c.cv). ORDER BY a.ak (always non-NULL: `a` is
// the preserved left side) fixes the order.
// ---------------------------------------------------------------------------

/// §2.1: `a LEFT JOIN (b JOIN c ON b.bk=c.ck) ON a.ak=b.bk` first computes the
/// INNER `(b JOIN c)` = `{(b1,c1)}` (b2 has no c), then LEFT-joins `a`. a1 matches;
/// a2 and a3 match NOTHING in the composite, so BOTH are NULL-extended across all
/// of b AND c. This is the grouping that makes the whole `(b JOIN c)` the outer
/// join's right operand.
#[test]
fn left_join_to_paren_inner_subjoin_null_extends_both() {
    let mut db = keyed3();
    assert_rows(
        &mut db,
        "SELECT a.ak, a.av, b.bk, b.bv, c.ck, c.cv \
         FROM a LEFT JOIN (b JOIN c ON b.bk=c.ck) ON a.ak=b.bk \
         ORDER BY a.ak",
        &[
            vec![int(1), text("a1"), int(1), text("b1"), int(1), text("c1")],
            vec![int(2), text("a2"), null(), null(), null(), null()],
            vec![int(3), text("a3"), null(), null(), null(), null()],
        ],
    );
}

/// §2.1: the OTHER grouping of the same tables — `(a LEFT JOIN b ON a.ak=b.bk) JOIN
/// c ON b.bk=c.ck` — computes `(a LEFT JOIN b)` first (a3 gets a NULL b), then an
/// INNER join to `c` on `b.bk=c.ck`. Only a1's row survives: a2's b2 has no c
/// (dropped by the inner join) and a3's NULL b.bk never equals c.ck. Contrast with
/// `left_join_to_paren_inner_subjoin_null_extends_both` (3 rows there, 1 here) —
/// this pair PROVES parentheses change an OUTER join's meaning.
#[test]
fn paren_left_ab_then_inner_c_drops_unmatched_left() {
    let mut db = keyed3();
    assert_rows(
        &mut db,
        "SELECT a.ak, a.av, b.bk, b.bv, c.ck, c.cv \
         FROM (a LEFT JOIN b ON a.ak=b.bk) JOIN c ON b.bk=c.ck \
         ORDER BY a.ak",
        &[vec![int(1), text("a1"), int(1), text("b1"), int(1), text("c1")]],
    );
}

/// §2.1: changing only the INNER sub-join's TYPE — `a LEFT JOIN (b LEFT JOIN c ON
/// b.bk=c.ck) ON a.ak=b.bk` — makes the composite keep `b2` (with a NULL c), so a2
/// now matches on `b.bk=2` and carries `b2` with NULL `c`. a3 still matches nothing
/// in the composite (no bk=3), so it stays fully NULL-extended. Contrast the a2 row
/// with `left_join_to_paren_inner_subjoin_null_extends_both` (b NULL there, b2
/// here): the sub-join's own ON/type is evaluated as a unit before the outer join.
#[test]
fn left_join_to_paren_left_subjoin_null_extends_only_c() {
    let mut db = keyed3();
    assert_rows(
        &mut db,
        "SELECT a.ak, a.av, b.bk, b.bv, c.ck, c.cv \
         FROM a LEFT JOIN (b LEFT JOIN c ON b.bk=c.ck) ON a.ak=b.bk \
         ORDER BY a.ak",
        &[
            vec![int(1), text("a1"), int(1), text("b1"), int(1), text("c1")],
            vec![int(2), text("a2"), int(2), text("b2"), null(), null()],
            vec![int(3), text("a3"), null(), null(), null(), null()],
        ],
    );
}

/// §2.1: an INNER outer join over the same parenthesized `(b LEFT JOIN c)` —
/// `a JOIN (b LEFT JOIN c ON b.bk=c.ck) ON a.ak=b.bk` — drops the unmatched a3
/// (the composite has no bk=3 row), unlike the LEFT version in
/// `left_join_to_paren_left_subjoin_null_extends_only_c` which keeps a3. Isolates
/// the OUTER operator's effect while the parenthesized right operand is held fixed.
#[test]
fn inner_join_to_paren_left_subjoin_keeps_matched_rows_only() {
    let mut db = keyed3();
    assert_rows(
        &mut db,
        "SELECT a.ak, a.av, b.bk, b.bv, c.ck, c.cv \
         FROM a JOIN (b LEFT JOIN c ON b.bk=c.ck) ON a.ak=b.bk \
         ORDER BY a.ak",
        &[
            vec![int(1), text("a1"), int(1), text("b1"), int(1), text("c1")],
            vec![int(2), text("a2"), int(2), text("b2"), null(), null()],
        ],
    );
}

// ---------------------------------------------------------------------------
// A NULL join key never matches (§2.1: for an ON clause "Only rows for which the
// expression evaluates to true are included" — a NULL comparison result is not
// true), exercised THROUGH a parenthesized sub-join.
// ---------------------------------------------------------------------------

/// §2.1 (ON keeps-true): inside `(l JOIN r ON l.k=r.k)` the NULL-keyed rows
/// `lnull`/`rnull` do NOT join — `NULL = anything`, and even `NULL = NULL`, is
/// NULL (not true) — so the composite holds only the two non-NULL matches, and the
/// outer `t JOIN (…)` sees just those. Pins that a parenthesized sub-join applies
/// ON's NULL-is-not-a-match rule as a unit before the surrounding join (the module
/// doc states this rule; this is the row that exercises it).
#[test]
fn paren_subjoin_null_join_key_never_matches() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE l(k INTEGER, lv TEXT)");
    exec(&mut db, "INSERT INTO l VALUES (1,'a'),(2,'b'),(NULL,'lnull')");
    exec(&mut db, "CREATE TABLE r(k INTEGER, rv TEXT)");
    exec(&mut db, "INSERT INTO r VALUES (1,'x'),(2,'y'),(NULL,'rnull')");
    exec(&mut db, "CREATE TABLE t(k INTEGER, tv TEXT)");
    exec(&mut db, "INSERT INTO t VALUES (1,'t1'),(2,'t2')");
    assert_rows(
        &mut db,
        "SELECT l.lv, r.rv, t.tv \
         FROM t JOIN (l JOIN r ON l.k=r.k) ON t.k=l.k \
         ORDER BY t.tv",
        &[
            vec![text("a"), text("x"), text("t1")],
            vec![text("b"), text("y"), text("t2")],
        ],
    );
}

// ---------------------------------------------------------------------------
// USING / NATURAL / CROSS inside the parentheses (§2.1 USING/NATURAL, §2.2 CROSS).
// These couple the parenthesized-join feature with shared-column coalescing and
// the cross product; all are grounded in lang_select.html. NOTE for the eventual
// implementer: because of that coupling, if one of these four lags red after
// parenthesized-join support lands while the pure INNER/LEFT cases above pass,
// suspect the USING/NATURAL coalescing (or CROSS) path, not the parenthesization.
// ---------------------------------------------------------------------------

/// §2.1 (USING): `(a JOIN b USING(k)) JOIN c USING(k)` — the inner USING coalesces
/// `k` into a single column, and the outer USING(k) then matches `c` against that
/// coalesced key. `k=3` is dropped (no `c`), leaving two fully-joined rows. The
/// bare `k` in the projection resolves to the single shared column.
#[test]
fn paren_using_join_threads_coalesced_key() {
    let mut db = sharedk();
    assert_rows(
        &mut db,
        "SELECT k, av, bv, cv \
         FROM (a JOIN b USING(k)) JOIN c USING(k) ORDER BY k",
        &[
            vec![int(1), text("a1"), text("b1"), text("c1")],
            vec![int(2), text("a2"), text("b2"), text("c2")],
        ],
    );
}

/// §2.1 (USING + LEFT JOIN): `(a JOIN b USING(k)) LEFT JOIN c USING(k)` keeps the
/// `k=3` composite row with a NULL `cv` (c has no k=3). The coalesced `k` on that
/// unmatched-right row is the present left value (3), since `COALESCE(left,right)`
/// prefers the non-NULL side — pinning USING coalescing on the outer path through a
/// parenthesized sub-join.
#[test]
fn paren_using_then_left_join_null_extends_missing_key() {
    let mut db = sharedk();
    assert_rows(
        &mut db,
        "SELECT k, av, bv, cv \
         FROM (a JOIN b USING(k)) LEFT JOIN c USING(k) ORDER BY k",
        &[
            vec![int(1), text("a1"), text("b1"), text("c1")],
            vec![int(2), text("a2"), text("b2"), text("c2")],
            vec![int(3), text("a3"), text("b3"), null()],
        ],
    );
}

/// §2.2 (CROSS JOIN) inside the parentheses: `a JOIN (b CROSS JOIN c) ON a.k=b.k`.
/// `(b CROSS JOIN c)` is the full 3×2 cartesian product (no ON), then `a` inner-joins
/// it on `a.k=b.k`. Each of a1/a2/a3 fans out to the two `c` rows sharing its `b`,
/// giving 6 rows. (b.k and c.k both exist in the composite; every reference is
/// qualified so nothing is ambiguous.)
#[test]
fn paren_cross_join_inside_then_outer_inner() {
    let mut db = sharedk();
    assert_rows(
        &mut db,
        "SELECT a.av, b.bv, c.cv \
         FROM a JOIN (b CROSS JOIN c) ON a.k=b.k \
         ORDER BY a.av, c.cv",
        &[
            vec![text("a1"), text("b1"), text("c1")],
            vec![text("a1"), text("b1"), text("c2")],
            vec![text("a2"), text("b2"), text("c1")],
            vec![text("a2"), text("b2"), text("c2")],
            vec![text("a3"), text("b3"), text("c1")],
            vec![text("a3"), text("b3"), text("c2")],
        ],
    );
}

/// §2.1 (NATURAL = implicit USING): `(a NATURAL JOIN b) NATURAL JOIN c` — `k` is the
/// only common name at each step, so both joins reduce to an implicit USING(k) and
/// coalesce to one `k`. `k=3` is dropped (no `c`), leaving two fully-joined rows,
/// exercising the NATURAL shared-set path through a parenthesized sub-join.
#[test]
fn paren_natural_join_inside_threads_common_column() {
    let mut db = sharedk();
    assert_rows(
        &mut db,
        "SELECT k, av, bv, cv \
         FROM (a NATURAL JOIN b) NATURAL JOIN c ORDER BY k",
        &[
            vec![int(1), text("a1"), text("b1"), text("c1")],
            vec![int(2), text("a2"), text("b2"), text("c2")],
        ],
    );
}

// ---------------------------------------------------------------------------
// Joining TWO parenthesized sub-joins (§2.1). A parenthesized sub-join can be
// BOTH operands at once; the outer operator's type decides how the two composites
// combine. Rows are (a.av, b.bv, c.cv, d.dv), ordered by a.av (non-NULL).
// ---------------------------------------------------------------------------

/// §2.1: `(a JOIN b ON a.ak=b.bk) JOIN (c JOIN d ON c.ck=d.dk) ON a.ak=c.ck` inner-
/// joins two parenthesized composites `AB` and `CD` on `a.ak=c.ck`. a3's composite
/// row has no `c` partner (CD has no ck=3), so the INNER outer join drops it,
/// leaving two rows.
#[test]
fn both_sides_parenthesized_inner_join() {
    let mut db = four();
    assert_rows(
        &mut db,
        "SELECT a.av, b.bv, c.cv, d.dv \
         FROM (a JOIN b ON a.ak=b.bk) JOIN (c JOIN d ON c.ck=d.dk) ON a.ak=c.ck \
         ORDER BY a.av",
        &[
            vec![text("a1"), text("b1"), text("c1"), text("d1")],
            vec![text("a2"), text("b2"), text("c2"), text("d2")],
        ],
    );
}

/// §2.1 (LEFT JOIN NULL-fill between two composites): the same two parenthesized
/// sub-joins under `... LEFT JOIN (c JOIN d ...) ON a.ak=c.ck` keep the `a3,b3`
/// composite row with NULLs across the entire `(c JOIN d)` operand — the outer
/// LEFT join NULL-extends the whole right composite when it matches nothing.
/// Contrast with `both_sides_parenthesized_inner_join` (a3 dropped there).
#[test]
fn both_sides_parenthesized_left_join_null_extends_right_composite() {
    let mut db = four();
    assert_rows(
        &mut db,
        "SELECT a.av, b.bv, c.cv, d.dv \
         FROM (a JOIN b ON a.ak=b.bk) LEFT JOIN (c JOIN d ON c.ck=d.dk) ON a.ak=c.ck \
         ORDER BY a.av",
        &[
            vec![text("a1"), text("b1"), text("c1"), text("d1")],
            vec![text("a2"), text("b2"), text("c2"), text("d2")],
            vec![text("a3"), text("b3"), null(), null()],
        ],
    );
}
