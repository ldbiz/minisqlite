//! Conformance battery: common table expressions (the `WITH` clause), both
//! ordinary and `WITH RECURSIVE`. Every expected value here is TRANSCRIBED FROM
//! THE SPEC in `spec/sqlite-doc/lang_with.html`, never from what the engine
//! returns:
//!
//! - Ordinary CTE = "works as if it were a view that exists for the duration of a
//!   single statement" (§2). A `WITH` clause may contain ordinary CTEs even with
//!   the `RECURSIVE` keyword; `RECURSIVE` "does not force common table expressions
//!   to be recursive" (§2).
//! - Recursive CTE shape (§3): the body is a compound select; non-recursive
//!   `initial-select`(s) come first, then recursive `recursive-select`(s), joined
//!   by `UNION` or `UNION ALL`. "more than one of each is allowed" (§3).
//! - The fixpoint algorithm (§3): run initial-select into a queue; while the queue
//!   is non-empty, extract one row, insert it into the recursive table, then run
//!   the recursive-select pretending that row is the recursive table's only row,
//!   appending results to the queue.
//!   - `UNION`: "only add rows to the queue if no identical row has been previously
//!     added" (whole-row dedup; "NULL values compare equal to one another").
//!   - `UNION ALL`: "all rows ... are always added to the queue even if they are
//!     repeats".
//!   - `LIMIT`: "determines the maximum number of rows that will ever be added to
//!     the recursive table ... Once the limit is reached, the recursion stops."
//! - The counting examples (§3.1) and the tree/graph examples (§3.2–§3.4) supply
//!   several concrete inputs whose outputs are computed directly from the algorithm
//!   above.
//! - Materialization hints (§4): `AS MATERIALIZED` / `AS NOT MATERIALIZED` are
//!   "non-binding hints to the query planner" that change the plan strategy but
//!   never the result set, so each hinted query must match its un-hinted form.
//! - Caveats (§5): SQLite "does not enforce" the rule that `RECURSIVE` must precede
//!   a recursive CTE, so a self-referential CTE is recursive even without the
//!   keyword.
//!
//! Assertions state DOCUMENTED behavior; a case that reveals an engine bug is left as a
//! genuine failing assertion rather than weakened to pass. Cases are split into many small
//! `#[test]` fns so one failing (or engine-rejected) case never masks the rest.
//!
//! TERMINATION: every recursive CTE here is bounded. Most carry a `WHERE` stop
//! predicate; the acyclic diamond-DAG walks drain on their own; and the `LIMIT 0`
//! / `OFFSET` cases keep a `WHERE` belt so they terminate (and fail loudly) even if
//! the engine ignores `LIMIT`/`OFFSET`. Only the Section C cases have NO independent
//! stop and lean on the engine honoring `UNION` dedup (a cycle) or `LIMIT`
//! (outer/body) to terminate — each is isolated in its own `#[test]` so that if a
//! future engine cannot bound one and loops, only that case is affected. `Value`
//! has no `PartialEq`; all comparisons go through the shared harness.

mod conformance;

use conformance::*;
use minisqlite::Connection;

// =============================================================================
// Section A — Ordinary (non-recursive) common table expressions (spec §2), and
// the materialization hints (§4), which are result-transparent.
// =============================================================================

#[test]
fn ordinary_cte_single_row() {
    // §2: an ordinary CTE is a view for the duration of the statement.
    let mut db = mem();
    assert_rows(&mut db, "WITH c AS (SELECT 1 AS n) SELECT n FROM c", &[vec![int(1)]]);
}

#[test]
fn ordinary_cte_output_column_name() {
    // The projected column keeps the name it was given inside the CTE body.
    let mut db = mem();
    assert_columns(&mut db, "WITH c AS (SELECT 1 AS n) SELECT n FROM c", &["n"]);
}

#[test]
fn ordinary_cte_values_with_column_name_list() {
    // cte-table-name may carry a "(col, col, ...)" list; VALUES supplies rows.
    let mut db = mem();
    assert_rows(
        &mut db,
        "WITH c(a,b) AS (VALUES (1,'x'),(2,'y')) SELECT a,b FROM c ORDER BY a",
        &[vec![int(1), text("x")], vec![int(2), text("y")]],
    );
}

#[test]
fn ordinary_cte_column_name_list_supplies_names() {
    // The "(a,b)" list names the CTE's output columns.
    let mut db = mem();
    assert_columns(&mut db, "WITH c(a,b) AS (VALUES (1,2)) SELECT a,b FROM c", &["a", "b"]);
}

#[test]
fn ordinary_cte_column_name_list_renames_body_columns() {
    // The column-name list renames the body's columns: body column `n` is exposed
    // as `x`.
    let mut db = mem();
    assert_rows(&mut db, "WITH c(x) AS (SELECT 1 AS n) SELECT x FROM c", &[vec![int(1)]]);
}

#[test]
fn ordinary_multiple_ctes_in_one_with_clause() {
    // A WITH clause may define several CTEs, each usable in the body.
    let mut db = mem();
    assert_rows(
        &mut db,
        "WITH c1 AS (SELECT 1 AS n), c2 AS (SELECT 2 AS n) \
         SELECT n FROM c1 UNION ALL SELECT n FROM c2 ORDER BY n",
        &[vec![int(1)], vec![int(2)]],
    );
}

#[test]
fn cte_name_shadows_a_base_table_of_the_same_name() {
    // §2 + the CTE name-resolution rule ("resolve CTE names in FROM ahead of base
    // tables"): when a CTE and a real table share a name, a FROM reference to that name
    // WITHIN the WITH statement resolves to the CTE, not the base table. Here the real
    // table `c` holds 9 while the CTE `c` yields 1, so `SELECT x FROM c` must read the
    // CTE's 1 — a regression that resolved the base table instead would return 9. This is
    // the only test that pins CTE-over-base-table PRECEDENCE (the others resolve a FROM
    // name that only a CTE defines, which cannot distinguish precedence from mere lookup).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE c(x)");
    exec(&mut db, "INSERT INTO c VALUES (9)");
    assert_rows(&mut db, "WITH c AS (SELECT 1 AS x) SELECT x FROM c", &[vec![int(1)]]);
}

#[test]
fn ordinary_cte_referenced_twice_in_self_join() {
    // The CTE `c` (a two-row set {1,2}) is referenced twice in a cross join, so the
    // result is the 2x2 Cartesian product ordered by (a.n, b.n).
    let mut db = mem();
    assert_rows(
        &mut db,
        "WITH c AS (SELECT 1 AS n UNION SELECT 2) \
         SELECT a.n, b.n FROM c a, c b ORDER BY a.n, b.n",
        &[
            vec![int(1), int(1)],
            vec![int(1), int(2)],
            vec![int(2), int(1)],
            vec![int(2), int(2)],
        ],
    );
}

#[test]
fn ordinary_cte_over_real_table() {
    // A CTE that filters a real base table behaves like a view over it.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x)");
    exec(&mut db, "INSERT INTO t(x) VALUES (10),(20),(30)");
    assert_rows(
        &mut db,
        "WITH big AS (SELECT x FROM t WHERE x>=20) SELECT x FROM big ORDER BY x",
        &[vec![int(20)], vec![int(30)]],
    );
}

#[test]
fn ordinary_cte_references_earlier_cte() {
    // §2: a CTE is a view visible to subsequent code, so a later CTE may reference
    // an earlier one. c1 = {3}; c2 = {n+1} = {4}.
    let mut db = mem();
    assert_rows(
        &mut db,
        "WITH c1 AS (SELECT 3 AS n), c2 AS (SELECT n+1 AS m FROM c1) SELECT m FROM c2",
        &[vec![int(4)]],
    );
}

#[test]
fn ordinary_cte_referenced_in_two_compound_arms() {
    // The single CTE `c` (value 5) is scanned in both arms of a compound select.
    let mut db = mem();
    assert_rows(
        &mut db,
        "WITH c AS (SELECT 5 AS n) SELECT n FROM c UNION ALL SELECT n*2 FROM c ORDER BY n",
        &[vec![int(5)], vec![int(10)]],
    );
}

#[test]
fn ordinary_cte_in_where_subquery() {
    // The CTE is used inside a subquery in the WHERE clause (a view usable anywhere
    // a table is). keep = {2,3}; rows of t whose x is in keep are 2 and 3.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x)");
    exec(&mut db, "INSERT INTO t(x) VALUES (1),(2),(3)");
    assert_rows(
        &mut db,
        "WITH keep AS (SELECT 2 AS v UNION SELECT 3) \
         SELECT x FROM t WHERE x IN (SELECT v FROM keep) ORDER BY x",
        &[vec![int(2)], vec![int(3)]],
    );
}

#[test]
fn ordinary_cte_at_start_of_scalar_subquery() {
    // §5: a WITH clause may appear at the beginning of a subquery. Here the subquery
    // is a scalar subquery in the SELECT list, so the test isolates WITH placement
    // without also requiring FROM-clause subquery support.
    let mut db = mem();
    assert_rows(
        &mut db,
        "SELECT (WITH c AS (SELECT 42 AS v) SELECT v FROM c)",
        &[vec![int(42)]],
    );
}

#[test]
fn ordinary_cte_at_start_of_from_subquery() {
    // §5: a WITH clause may appear at the beginning of a FROM-clause subquery. The
    // inner CTE `c` is scoped to that subquery and must resolve when the enclosing
    // query computes the subquery `d`'s schema (Phase 1), not only at compile (Phase 2).
    let mut db = mem();
    assert_rows(
        &mut db,
        "SELECT n FROM (WITH c(n) AS (SELECT 2) SELECT n FROM c) d",
        &[vec![int(2)]],
    );
}

#[test]
fn recursive_cte_at_start_of_from_subquery() {
    // A recursive CTE inside a FROM-clause subquery: the counting walk (§3.1) bounded
    // by a WHERE belt, wrapped as a derived table `d`.
    let mut db = mem();
    assert_rows(
        &mut db,
        "SELECT x FROM (WITH RECURSIVE cnt(x) AS \
         (SELECT 1 UNION ALL SELECT x+1 FROM cnt WHERE x < 3) SELECT x FROM cnt) d \
         ORDER BY x",
        &[vec![int(1)], vec![int(2)], vec![int(3)]],
    );
}

#[test]
fn recursive_keyword_does_not_force_recursion() {
    // §2: "The use of RECURSIVE does not force common table expressions to be
    // recursive." Here `c` is an ordinary (non-recursive) CTE under RECURSIVE.
    let mut db = mem();
    assert_rows(&mut db, "WITH RECURSIVE c AS (SELECT 7 AS n) SELECT n FROM c", &[vec![int(7)]]);
}

// ---- Materialization hints (§4) ---------------------------------------------
// §4: `AS MATERIALIZED` / `AS NOT MATERIALIZED` are "non-binding hints to the
// query planner about how the CTE should be implemented" — they change only the
// plan strategy, never the result set. So each hinted form must return exactly
// what the un-hinted form returns.

#[test]
fn ordinary_cte_materialized_hint() {
    // The MATERIALIZED hint forces an ephemeral table but is result-transparent.
    let mut db = mem();
    assert_rows(
        &mut db,
        "WITH c AS MATERIALIZED (SELECT 1 AS n) SELECT n FROM c",
        &[vec![int(1)]],
    );
}

#[test]
fn ordinary_cte_not_materialized_hint() {
    // NOT MATERIALIZED asks the planner to inline the CTE as a subquery; still
    // result-transparent, so the single row is unchanged.
    let mut db = mem();
    assert_rows(
        &mut db,
        "WITH c AS NOT MATERIALIZED (SELECT 1 AS n) SELECT n FROM c",
        &[vec![int(1)]],
    );
}

// =============================================================================
// Section B — Recursive common table expressions, WHERE-bounded (spec §3)
// =============================================================================

#[test]
fn recursive_counter_one_to_five() {
    // §3.1 counting pattern, bounded by `WHERE n<5`: seed 1, then n+1 up to 5.
    let mut db = mem();
    assert_rows(
        &mut db,
        "WITH RECURSIVE cnt(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM cnt WHERE n<5) \
         SELECT n FROM cnt ORDER BY n",
        &[vec![int(1)], vec![int(2)], vec![int(3)], vec![int(4)], vec![int(5)]],
    );
}

#[test]
fn recursive_sum_one_to_ten() {
    // The recursive table holds 1..10 (bounded by `WHERE n<10`); sum is 55. sum()
    // over all-integer inputs is an integer.
    let mut db = mem();
    assert_scalar(
        &mut db,
        "WITH RECURSIVE cnt(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM cnt WHERE n<10) \
         SELECT sum(n) FROM cnt",
        int(55),
    );
}

#[test]
fn recursive_fibonacci_pairs() {
    // Two-column recursion carrying (a,b): (0,1)->(1,1)->(1,2)->(2,3)->(3,5)->
    // (5,8)->(8,13)->(13,21). The step stops when b>=20, so (13,21) is the last row
    // added (its b=21 fails `WHERE b<20`). Projecting `a` in ascending order:
    let mut db = mem();
    assert_rows(
        &mut db,
        "WITH RECURSIVE fib(a,b) AS (SELECT 0,1 UNION ALL SELECT b,a+b FROM fib WHERE b<20) \
         SELECT a FROM fib ORDER BY a",
        &[
            vec![int(0)],
            vec![int(1)],
            vec![int(1)],
            vec![int(2)],
            vec![int(3)],
            vec![int(5)],
            vec![int(8)],
            vec![int(13)],
        ],
    );
}

#[test]
fn recursive_values_seed() {
    // §3.1 uses `VALUES(1)` as the initial-select. Bounded by `WHERE x<4`.
    let mut db = mem();
    assert_rows(
        &mut db,
        "WITH RECURSIVE cnt(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM cnt WHERE x<4) \
         SELECT x FROM cnt ORDER BY x",
        &[vec![int(1)], vec![int(2)], vec![int(3)], vec![int(4)]],
    );
}

#[test]
fn recursive_countdown() {
    // Decreasing recursion bounded by `WHERE n>1`: 5,4,3,2,1.
    let mut db = mem();
    assert_rows(
        &mut db,
        "WITH RECURSIVE c(n) AS (SELECT 5 UNION ALL SELECT n-1 FROM c WHERE n>1) \
         SELECT n FROM c ORDER BY n DESC",
        &[vec![int(5)], vec![int(4)], vec![int(3)], vec![int(2)], vec![int(1)]],
    );
}

#[test]
fn recursive_multiple_initial_selects() {
    // §3: "more than one [initial-select] ... is allowed." Two seeds {1,2}, step
    // n+2 bounded by `WHERE n<5`: 1->3->5(stop), 2->4->6(stop). Table = {1..6}.
    let mut db = mem();
    assert_rows(
        &mut db,
        "WITH RECURSIVE t(n) AS (\
            SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT n+2 FROM t WHERE n<5\
         ) SELECT n FROM t ORDER BY n",
        &[
            vec![int(1)],
            vec![int(2)],
            vec![int(3)],
            vec![int(4)],
            vec![int(5)],
            vec![int(6)],
        ],
    );
}

#[test]
fn recursive_without_recursive_keyword() {
    // §5: "The SQL:1999 spec requires that the RECURSIVE keyword follow WITH in
    // any WITH clause that includes a recursive common table expression. However,
    // for compatibility with SqlServer and Oracle, SQLite does not enforce this
    // rule." So a self-referential CTE is recursive even without `RECURSIVE`.
    // Bounded by `WHERE n<3`: 1,2,3.
    let mut db = mem();
    assert_rows(
        &mut db,
        "WITH cnt(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM cnt WHERE n<3) \
         SELECT n FROM cnt ORDER BY n",
        &[vec![int(1)], vec![int(2)], vec![int(3)]],
    );
}

// ---- LIMIT / OFFSET semantics on the recursive table (WHERE-belted) ----------
// §3 defines how `LIMIT`/`OFFSET` bound what is added to the recursive table.
// These cases pin that behavior but keep a `WHERE` stop predicate as a belt, so
// they terminate quickly even if the engine ignores `LIMIT`/`OFFSET` (they then
// return the WHERE-bounded rows and fail loudly, never loop). The unbounded
// `LIMIT`-only cases live in Section C.

#[test]
fn recursive_limit_zero_adds_no_rows() {
    // §3 (2643): "A limit of zero means that no rows are ever added to the recursive
    // table." With `LIMIT 0` the recursive table is empty, so the query returns no
    // rows. If the engine ignored `LIMIT` it would instead return the WHERE-bounded
    // 1..5 — a loud mismatch, not a hang.
    let mut db = mem();
    assert_rows(
        &mut db,
        "WITH RECURSIVE cnt(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM cnt WHERE n<5 LIMIT 0) \
         SELECT n FROM cnt",
        &[],
    );
}

#[test]
fn recursive_offset_skips_leading_rows() {
    // §3 (2646-2652): "The OFFSET clause ... prevents the first N rows from being
    // added to the recursive table. The first N rows are still processed by the
    // recursive-select ... Rows are not counted toward fulfilling the LIMIT until
    // all OFFSET rows have been skipped." The WHERE-bounded sequence is 1,2,3,4,5;
    // `OFFSET 2` skips 1,2 (still processed, so they generate successors) and
    // `LIMIT 3` then admits the next three, 3,4,5.
    let mut db = mem();
    assert_rows(
        &mut db,
        "WITH RECURSIVE cnt(n) AS \
            (SELECT 1 UNION ALL SELECT n+1 FROM cnt WHERE n<5 LIMIT 3 OFFSET 2) \
         SELECT n FROM cnt ORDER BY n",
        &[vec![int(3)], vec![int(4)], vec![int(5)]],
    );
}

// ---- UNION vs UNION ALL in recursion (whole-row dedup) -----------------------
// A diamond DAG e: 5->2, 5->4, 4->2. Node 2 is reachable by two paths (5->2 and
// 5->4->2), so the recursive walk produces it twice. UNION ALL keeps both copies;
// UNION admits each whole row only once. The graph is acyclic, so BOTH forms
// terminate without any WHERE/LIMIT — the contrast is purely the dedup.

fn insert_diamond_dag(db: &mut Connection) {
    exec(db, "CREATE TABLE e(a, b)");
    exec(db, "INSERT INTO e(a,b) VALUES (5,2),(5,4),(4,2)");
}

#[test]
fn recursive_union_all_keeps_duplicate_paths() {
    // Reachable-from-5 with UNION ALL: 5 (seed), then 2,4 (from 5), then 2 (from 4).
    // Node 2 appears twice.
    let mut db = mem();
    insert_diamond_dag(&mut db);
    assert_rows(
        &mut db,
        "WITH RECURSIVE t(n) AS (\
            SELECT 5 UNION ALL SELECT e.b FROM e JOIN t ON e.a = t.n\
         ) SELECT n FROM t ORDER BY n",
        &[vec![int(2)], vec![int(2)], vec![int(4)], vec![int(5)]],
    );
}

#[test]
fn recursive_union_dedups_duplicate_paths() {
    // Same walk with UNION: the second arrival at node 2 is discarded, so each of
    // {2,4,5} appears exactly once.
    let mut db = mem();
    insert_diamond_dag(&mut db);
    assert_rows(
        &mut db,
        "WITH RECURSIVE t(n) AS (\
            SELECT 5 UNION SELECT e.b FROM e JOIN t ON e.a = t.n\
         ) SELECT n FROM t ORDER BY n",
        &[vec![int(2)], vec![int(4)], vec![int(5)]],
    );
}

// ---- Recursion joined to a real table (hierarchy walk, spec §3.2/§3.4) -------

fn insert_org_tree(db: &mut Connection) {
    // The org data from spec §3.4 (chain-of-command tree).
    exec(db, "CREATE TABLE org(name TEXT, boss TEXT)");
    exec(db, "INSERT INTO org VALUES('Alice',NULL)");
    exec(db, "INSERT INTO org VALUES('Bob','Alice')");
    exec(db, "INSERT INTO org VALUES('Cindy','Alice')");
    exec(db, "INSERT INTO org VALUES('Dave','Bob')");
    exec(db, "INSERT INTO org VALUES('Emma','Bob')");
    exec(db, "INSERT INTO org VALUES('Fred','Cindy')");
    exec(db, "INSERT INTO org VALUES('Gail','Cindy')");
}

#[test]
fn recursive_hierarchy_names_and_levels() {
    // Walk the tree from Alice, carrying a depth level. The set of (name, level) is
    // fully determined by the tree; the WITHIN-level order is not (spec §3.4), so
    // this asserts the multiset.
    let mut db = mem();
    insert_org_tree(&mut db);
    // Built via `concat!` of adjacent literals because each fragment here starts
    // with a LEADING space, and a `\` line-continuation eats the newline plus the
    // next line's leading whitespace — which would glue `ALL` onto `SELECT`. (The
    // `\` style used elsewhere is safe: those breaks place the space BEFORE the `\`,
    // which is preserved.)
    assert_rows_unordered(
        &mut db,
        concat!(
            "WITH RECURSIVE under_alice(name, level) AS (",
            " VALUES('Alice', 0)",
            " UNION ALL",
            " SELECT org.name, under_alice.level+1",
            " FROM org JOIN under_alice ON org.boss = under_alice.name",
            ") SELECT name, level FROM under_alice",
        ),
        &[
            vec![text("Alice"), int(0)],
            vec![text("Bob"), int(1)],
            vec![text("Cindy"), int(1)],
            vec![text("Dave"), int(2)],
            vec![text("Emma"), int(2)],
            vec![text("Fred"), int(2)],
            vec![text("Gail"), int(2)],
        ],
    );
}

#[test]
fn recursive_hierarchy_subtree_count() {
    // Aggregation OVER a recursive CTE's result (allowed; only the recursive-select
    // itself may not aggregate). Bob's subtree is {Bob, Dave, Emma} -> 3.
    let mut db = mem();
    insert_org_tree(&mut db);
    assert_scalar(
        &mut db,
        concat!(
            "WITH RECURSIVE chain(name) AS (",
            " VALUES('Bob')",
            " UNION ALL",
            " SELECT org.name FROM org JOIN chain ON org.boss = chain.name",
            ") SELECT count(*) FROM chain",
        ),
        int(3),
    );
}

// =============================================================================
// Section C — Recursions bounded by dedup / LIMIT (isolated; see module note)
// =============================================================================
// Each case below relies on the engine to bound a recursion that has no WHERE
// stop predicate. They are isolated in their own `#[test]` fns: if the engine
// cannot bound one and loops, only that case is affected. Expected values are
// the spec's.

#[test]
fn recursive_union_terminates_on_cycle() {
    // Spec §3.3: "UNION is used instead of UNION ALL to prevent the recursion from
    // entering an infinite loop if the graph contains cycles." Directed cycle
    // 1->2->3->1; starting from 1, UNION dedup stops the walk after {1,2,3} because
    // the back-edge 3->1 reproduces the already-seen row 1.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE edge(a, b)");
    exec(&mut db, "INSERT INTO edge(a,b) VALUES (1,2),(2,3),(3,1)");
    assert_rows(
        &mut db,
        "WITH RECURSIVE nodes(x) AS (\
            SELECT 1 UNION SELECT edge.b FROM edge JOIN nodes ON edge.a = nodes.x\
         ) SELECT x FROM nodes ORDER BY x",
        &[vec![int(1)], vec![int(2)], vec![int(3)]],
    );
}

#[test]
fn recursive_outer_limit_bounds_infinite_recursion() {
    // Spec §3.1: the second counting example relies on LIMIT to stop an otherwise
    // unbounded recursion. Here `cnt` has no WHERE, and the outer `LIMIT 3` must
    // bound it to the first three rows (the optimizer streams each recursive row
    // straight to the result, so only three are ever produced).
    //
    // Asserted as a MULTISET: without an `ORDER BY`, §3 (2653-2658) leaves the
    // queue-extraction order undefined, so the spec-faithful claim is the set
    // {1,2,3}, not a fixed sequence. We deliberately do NOT add `ORDER BY n` to pin
    // an order, because sorting would force the full (infinite) result to be
    // collected first, defeating the very streaming that lets `LIMIT` bound the
    // recursion. (This linear recursion holds only one queue row at a time, so the
    // set is deterministically {1,2,3} regardless of extraction order.)
    let mut db = mem();
    assert_rows_unordered(
        &mut db,
        "WITH RECURSIVE cnt(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM cnt) \
         SELECT n FROM cnt LIMIT 3",
        &[vec![int(1)], vec![int(2)], vec![int(3)]],
    );
}

#[test]
fn recursive_body_limit_bounds_recursion() {
    // Spec §3 / §3.1: "The LIMIT clause ... determines the maximum number of rows
    // that will ever be added to the recursive table ... Once the limit is reached,
    // the recursion stops." A `LIMIT 4` inside the CTE body bounds the otherwise
    // unbounded counter to 1,2,3,4.
    let mut db = mem();
    assert_rows(
        &mut db,
        "WITH RECURSIVE cnt(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM cnt LIMIT 4) \
         SELECT n FROM cnt ORDER BY n",
        &[vec![int(1)], vec![int(2)], vec![int(3)], vec![int(4)]],
    );
}

// ---- Negative / NULL body LIMIT (the value-coercion edge) --------------------
// A recursive-CTE body LIMIT is a scalar LIMIT expression: its VALUE-coercion
// follows the general rule in `lang_select.html` §5 (16002-16004), even though
// its RECURSION effect is described in `lang_with.html` §3. The two ends of that
// rule diverge — a NEGATIVE limit is "no upper bound" (unbounded), but a NULL
// (or otherwise non-integral) limit is an ERROR — so they are pinned separately.
// The `LIMIT -1` case keeps a `WHERE` belt so it terminates loudly (returning the
// bounded set, or `[]` if the sign were mishandled) rather than looping.

#[test]
fn recursive_negative_limit_is_unbounded() {
    // §5 (16004): "If the LIMIT expression evaluates to a negative value, then there
    // is no upper bound on the number of rows returned." A body `LIMIT -1` imposes NO
    // cap on the recursive table, so the recursion runs to its natural WHERE-bounded
    // termination (1..5) exactly as if no LIMIT were present. This is the
    // recursion-specific "no limit" behavior: the fixpoint drains on its own stop
    // predicate. A bug that clamped -1 to 0 (or otherwise capped) would return a
    // truncated set — a loud mismatch against the WHERE-belted 1..5, never a hang.
    let mut db = mem();
    assert_rows(
        &mut db,
        "WITH RECURSIVE cnt(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM cnt WHERE n<5 LIMIT -1) \
         SELECT n FROM cnt ORDER BY n",
        &[vec![int(1)], vec![int(2)], vec![int(3)], vec![int(4)], vec![int(5)]],
    );
}

#[test]
fn recursive_null_limit_is_a_datatype_error() {
    // §5 (16002-16004): "If the [LIMIT] expression evaluates to a NULL value or any
    // other value that cannot be losslessly converted to an integer, an error is
    // returned." So a body `LIMIT NULL` is an ERROR, NOT "no limit" — NULL is paired
    // with negative only in loose summaries; the spec pairs NULL with the non-integral
    // REJECT set and reserves "unbounded" for negatives alone. The engine lowers the
    // body LIMIT to the shared `Limit` operator, so NULL is rejected by the same
    // `OP_MustBeInt` coercion as a top-level `LIMIT NULL` — this case proves that
    // rejection survives the recursive-CTE compile->wrapper path (the operator-level
    // `limit_null_errors` cannot, since it never goes through CTE compilation).
    let mut db = mem();
    let err = try_query(
        &mut db,
        "WITH RECURSIVE cnt(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM cnt WHERE n<5 LIMIT NULL) \
         SELECT n FROM cnt",
    )
    .expect_err("a NULL body LIMIT on a recursive CTE must error, not be treated as unbounded");
    // Pin the REASON, not merely that it errored: the surrounding query is valid (the
    // WHERE-belted counter, exercised by the sibling cases), so the ONLY failure
    // source is the `OP_MustBeInt` "datatype mismatch" on `LIMIT NULL`. Asserting the
    // message keeps a future UNRELATED error from making this case falsely pass.
    assert!(
        err.to_string().contains("datatype mismatch"),
        "expected a `datatype mismatch` error for a NULL body LIMIT, got: {err:?}",
    );
}
