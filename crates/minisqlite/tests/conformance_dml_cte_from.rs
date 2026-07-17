//! Conformance battery for two DOCUMENTED DML features the engine currently PARSES
//! BUT REJECTS at plan time:
//!
//!   A. `UPDATE ... FROM` — the join-driven UPDATE (`spec/sqlite-doc/lang_update.html`
//!      §2.2 "UPDATE FROM"). The target table is joined against the tables named in
//!      `FROM`; the `WHERE` clause matches target rows to FROM rows, and the `SET`
//!      values may read the matched FROM row's columns as well as the target's own
//!      (pre-update) columns.
//!   B. A leading `WITH` (CTE) clause on INSERT / UPDATE / DELETE
//!      (`spec/sqlite-doc/lang_with.html` §1: "All common table expressions ... are
//!      created by prepending a WITH clause in front of a SELECT, INSERT, DELETE, or
//!      UPDATE statement"; §3 for the RECURSIVE algorithm), plus
//!      `lang_insert.html` / `lang_update.html` / `lang_delete.html` for how the CTE
//!      feeds each DML's source/query.
//!
//! Every expected value is TRANSCRIBED FROM THE SPEC + the seeded DATA, never from what
//! the engine returns. Binding rules used below:
//!
//!   * §2.2: "you can join the target table against other tables in the database in
//!     order to help compute which rows need updating and what the new values should be
//!     on those rows." So the join drives BOTH row selection and the new values.
//!   * §2.2: "If the join between the target table and the FROM clause results in
//!     multiple output rows for the same target table row, then only one of those output
//!     rows is used ... The output row selected is arbitrary." Therefore EVERY join here
//!     is DETERMINISTIC — each target row matches AT MOST ONE FROM row — so the expected
//!     values are exact; the unspecified multi-match case is deliberately NOT pinned.
//!   * §2.2 (inner-join semantics): the FROM helps compute "which rows need updating", so
//!     a target row with NO matching FROM row is left UNCHANGED (not NULLed) — the FROM
//!     acts as an inner join for the purpose of which rows get written.
//!   * §2.2: "The target table is not included in the FROM clause, unless the intent is
//!     to do a self-join ... the table in the FROM clause must be aliased to a different
//!     name than the target table." (Exercised by the self-join case, which reads only a
//!     column the SET does not write, so the old/new-snapshot question never arises.)
//!   * lang_with.html §1: a CTE "act[s] like a temporary view that exists only for the
//!     duration of a single SQL statement", visible to the DML's source/query and its
//!     subqueries.
//!   * lang_with.html §3: the recursive-CTE queue algorithm — the initial-select seeds
//!     the recursive table and each extracted row runs the recursive-select until it
//!     yields no rows (e.g. `SELECT 1 UNION ALL SELECT n+1 FROM seq WHERE n<5` = 1..5).
//!
//! Note (parse-but-reject today): each statement PARSES — the grammar builds
//! `Update.from`, `Insert.with`, `Update.with`, and `Delete.with` (a leading `WITH` is
//! routed onto the DML by `parse_dml_or_select_stmt`; `parse_update` reads the `FROM`
//! clause with `self.parse_from()`) — but the PLANNER rejects it, so every target
//! `exec()` below fails at PLAN time. That failure is the intended loud, honest signal.
//! Reject sites, anchored on fn + guard expression + error string (NOT volatile lines):
//!
//!   * `compile_update_with_parent` (`minisqlite-plan/src/compile/update.rs`):
//!       `if stmt.from.is_some()` -> Error::sql("UPDATE ... FROM is not yet supported")
//!       `if stmt.with.is_some()` -> Error::sql("WITH on UPDATE is not yet supported")
//!   * `compile_delete_with_parent` (`minisqlite-plan/src/compile/delete.rs`):
//!       `if stmt.with.is_some()` -> Error::sql("WITH on DELETE is not yet supported")
//!   * `compile_insert_with_parent` (`minisqlite-plan/src/compile/insert.rs`):
//!       `if stmt.with.is_some()` -> Error::sql("WITH on INSERT is not yet supported")
//!
//! Each test asserts the SPEC-CORRECT post-statement STATE via a follow-up SELECT, so it
//! fails now (a "not yet supported" error at the target `exec`) and will pass with no
//! test change once the planner implements the feature; the spec-correct assertion is left
//! intact rather than weakened to pass.

mod conformance;
use conformance::*;

// ===========================================================================
// A. UPDATE ... FROM  (lang_update.html §2.2)
// ===========================================================================

/// §2.2: the basic join update. `UPDATE t SET v = o.v FROM o WHERE o.id = t.id` takes
/// each target row's new `v` from the FROM row it joins to; with a 1:1 key match the
/// result is exact.
#[test]
fn update_from_basic_join_takes_matching_value() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v)");
    exec(&mut db, "CREATE TABLE o(id, v)");
    exec(&mut db, "INSERT INTO t VALUES (1, 10), (2, 20)");
    exec(&mut db, "INSERT INTO o VALUES (1, 100), (2, 200)");
    exec(&mut db, "UPDATE t SET v = o.v FROM o WHERE o.id = t.id");
    assert_rows(
        &mut db,
        "SELECT id, v FROM t ORDER BY id",
        &[vec![int(1), int(100)], vec![int(2), int(200)]],
    );
}

/// §2.2: the SET clause may assign SEVERAL columns from one matched FROM row — each
/// target column takes its value from the same joined row.
#[test]
fn update_from_updates_multiple_columns() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, a, b)");
    exec(&mut db, "CREATE TABLE o(id, a, b)");
    exec(&mut db, "INSERT INTO t VALUES (1, 0, 0), (2, 0, 0)");
    exec(&mut db, "INSERT INTO o VALUES (1, 11, 12), (2, 21, 22)");
    exec(&mut db, "UPDATE t SET a = o.a, b = o.b FROM o WHERE o.id = t.id");
    assert_rows(
        &mut db,
        "SELECT id, a, b FROM t ORDER BY id",
        &[vec![int(1), int(11), int(12)], vec![int(2), int(21), int(22)]],
    );
}

/// §2.2: the WHERE clause both joins AND filters. `o.id = t.id AND t.id >= 2` matches
/// every row on the key but restricts writes to `t.id >= 2`, so row 1 (join matches but
/// filter fails) is left unchanged while rows 2 and 3 are updated.
#[test]
fn update_from_where_filters_which_rows_change() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v)");
    exec(&mut db, "CREATE TABLE o(id, v)");
    exec(&mut db, "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)");
    exec(&mut db, "INSERT INTO o VALUES (1, 100), (2, 200), (3, 300)");
    exec(&mut db, "UPDATE t SET v = o.v FROM o WHERE o.id = t.id AND t.id >= 2");
    assert_rows(
        &mut db,
        "SELECT id, v FROM t ORDER BY id",
        &[vec![int(1), int(10)], vec![int(2), int(200)], vec![int(3), int(300)]],
    );
}

/// §2.2: the FROM table may be aliased; the SET/WHERE reference it by the alias. Using a
/// DIFFERENT table under alias `src` (not a self-join), each target row takes `src.v`.
#[test]
fn update_from_table_alias() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v)");
    exec(&mut db, "CREATE TABLE o(id, v)");
    exec(&mut db, "INSERT INTO t VALUES (1, 10), (2, 20)");
    exec(&mut db, "INSERT INTO o VALUES (1, 100), (2, 200)");
    exec(&mut db, "UPDATE t SET v = src.v FROM o AS src WHERE src.id = t.id");
    assert_rows(
        &mut db,
        "SELECT id, v FROM t ORDER BY id",
        &[vec![int(1), int(100)], vec![int(2), int(200)]],
    );
}

/// §2.2: the FROM clause may itself be a JOIN of two tables. `FROM a, b WHERE a.id = t.id
/// AND b.k = a.k` chains t -> a -> b (both keys unique, so 1:1:1) and the target takes
/// `b.val` from the transitively-joined row.
#[test]
fn update_from_two_table_join() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v)");
    exec(&mut db, "CREATE TABLE a(id, k)");
    exec(&mut db, "CREATE TABLE b(k, val)");
    exec(&mut db, "INSERT INTO t VALUES (1, 0), (2, 0)");
    exec(&mut db, "INSERT INTO a VALUES (1, 'x'), (2, 'y')");
    exec(&mut db, "INSERT INTO b VALUES ('x', 100), ('y', 200)");
    exec(&mut db, "UPDATE t SET v = b.val FROM a, b WHERE a.id = t.id AND b.k = a.k");
    assert_rows(
        &mut db,
        "SELECT id, v FROM t ORDER BY id",
        &[vec![int(1), int(100)], vec![int(2), int(200)]],
    );
}

/// §2.2: a SET expression may COMBINE the target's own (pre-update) column with a FROM
/// column. `SET v = t.v + o.delta` reads the original `t.v` plus the joined `o.delta`,
/// so `10 + 5 = 15` and `20 + 7 = 27`.
#[test]
fn update_from_set_combines_target_and_from_columns() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v)");
    exec(&mut db, "CREATE TABLE o(id, delta)");
    exec(&mut db, "INSERT INTO t VALUES (1, 10), (2, 20)");
    exec(&mut db, "INSERT INTO o VALUES (1, 5), (2, 7)");
    exec(&mut db, "UPDATE t SET v = t.v + o.delta FROM o WHERE o.id = t.id");
    assert_rows(
        &mut db,
        "SELECT id, v FROM t ORDER BY id",
        &[vec![int(1), int(15)], vec![int(2), int(27)]],
    );
}

/// §2.2 (inner-join semantics): a target row with NO matching FROM row is left UNCHANGED
/// (not set to NULL). `o` has only `id = 2`, so rows 1 and 3 keep their original `v` and
/// only row 2 is updated.
#[test]
fn update_from_unmatched_target_row_is_unchanged() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v)");
    exec(&mut db, "CREATE TABLE o(id, v)");
    exec(&mut db, "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)");
    exec(&mut db, "INSERT INTO o VALUES (2, 200)");
    exec(&mut db, "UPDATE t SET v = o.v FROM o WHERE o.id = t.id");
    assert_rows(
        &mut db,
        "SELECT id, v FROM t ORDER BY id",
        &[vec![int(1), int(10)], vec![int(2), int(200)], vec![int(3), int(30)]],
    );
}

/// §2.2 (the documented worked example): the inventory/sales adjustment. The FROM
/// sub-select aggregates daily sales per item (one row per `itemId`, so the join is 1:1)
/// and the inventory quantity is reduced by that amount: item 1 loses 3+4=7 (100->93),
/// item 2 loses 5 (50->45).
#[test]
fn update_from_documented_inventory_example() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE inventory(itemId INTEGER PRIMARY KEY, quantity)");
    exec(&mut db, "CREATE TABLE sales(itemId, quantity)");
    exec(&mut db, "INSERT INTO inventory VALUES (1, 100), (2, 50)");
    exec(&mut db, "INSERT INTO sales VALUES (1, 3), (1, 4), (2, 5)");
    exec(
        &mut db,
        "UPDATE inventory SET quantity = quantity - daily.amt \
         FROM (SELECT sum(quantity) AS amt, itemId FROM sales GROUP BY itemId) AS daily \
         WHERE inventory.itemId = daily.itemId",
    );
    assert_rows(
        &mut db,
        "SELECT itemId, quantity FROM inventory ORDER BY itemId",
        &[vec![int(1), int(93)], vec![int(2), int(45)]],
    );
}

/// §2.2: a self-join must alias the target table under a different name (`t AS other`).
/// Here each row's `v` is set from the `ref` column of the row it points at
/// (`other.id = t.ref`). Only `v` is written and only the never-written `ref` is read
/// from `other`, so the result is independent of any old/new-snapshot detail: row1
/// (ref=3) -> other row 3's ref=2; row2 (ref=1) -> row 1's ref=3; row3 (ref=2) -> row
/// 2's ref=1.
#[test]
fn update_from_self_join_requires_alias() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v, ref)");
    exec(&mut db, "INSERT INTO t VALUES (1, 0, 3), (2, 0, 1), (3, 0, 2)");
    exec(&mut db, "UPDATE t SET v = other.ref FROM t AS other WHERE other.id = t.ref");
    assert_rows(
        &mut db,
        "SELECT id, v, ref FROM t ORDER BY id",
        &[vec![int(1), int(2), int(3)], vec![int(2), int(3), int(1)], vec![int(3), int(1), int(2)]],
    );
}

/// §2.2 (inner-join semantics, extreme case): if the FROM table is EMPTY the join yields
/// no rows, so no target row is updated — the whole table is left unchanged.
#[test]
fn update_from_empty_source_updates_nothing() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v)");
    exec(&mut db, "CREATE TABLE o(id, v)");
    exec(&mut db, "INSERT INTO t VALUES (1, 10), (2, 20)");
    exec(&mut db, "UPDATE t SET v = o.v FROM o WHERE o.id = t.id");
    assert_rows(
        &mut db,
        "SELECT id, v FROM t ORDER BY id",
        &[vec![int(1), int(10)], vec![int(2), int(20)]],
    );
}

// ===========================================================================
// B. WITH (CTE) on INSERT / UPDATE / DELETE  (lang_with.html §1, §3)
// ===========================================================================

/// §1 (WITH on INSERT): a CTE built from a `VALUES` list, with an explicit column name,
/// feeds an `INSERT ... SELECT`. `WITH c(x) AS (VALUES (1),(2),(3)) INSERT INTO t(x)
/// SELECT x FROM c` inserts rows 1, 2, 3.
#[test]
fn with_values_cte_feeds_insert_select() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x)");
    exec(&mut db, "WITH c(x) AS (VALUES (1),(2),(3)) INSERT INTO t(x) SELECT x FROM c");
    assert_rows(
        &mut db,
        "SELECT x FROM t ORDER BY x",
        &[vec![int(1)], vec![int(2)], vec![int(3)]],
    );
}

/// §1 (WITH on INSERT): a COMPUTED ordinary CTE derived from another table feeds the
/// insert. `c` doubles each `src.n`, and `INSERT INTO t(doubled) SELECT m FROM c` stores
/// 2, 4, 6.
#[test]
fn with_computed_cte_feeds_insert_select() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE src(n)");
    exec(&mut db, "CREATE TABLE t(doubled)");
    exec(&mut db, "INSERT INTO src VALUES (1), (2), (3)");
    exec(&mut db, "WITH c AS (SELECT n * 2 AS m FROM src) INSERT INTO t(doubled) SELECT m FROM c");
    assert_rows(
        &mut db,
        "SELECT doubled FROM t ORDER BY doubled",
        &[vec![int(2)], vec![int(4)], vec![int(6)]],
    );
}

/// §1 (WITH on UPDATE): the CTE selects which rows to touch. `c` holds the ids from `src`
/// with `flag = 1` ({1, 3}); `UPDATE t SET v = 99 WHERE id IN (SELECT id FROM c)` updates
/// exactly those rows and leaves row 2 unchanged.
#[test]
fn with_cte_filters_updated_rows() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE src(id, flag)");
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v)");
    exec(&mut db, "INSERT INTO src VALUES (1, 1), (2, 0), (3, 1)");
    exec(&mut db, "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)");
    exec(&mut db, "WITH c AS (SELECT id FROM src WHERE flag = 1) UPDATE t SET v = 99 WHERE id IN (SELECT id FROM c)");
    assert_rows(
        &mut db,
        "SELECT id, v FROM t ORDER BY id",
        &[vec![int(1), int(99)], vec![int(2), int(20)], vec![int(3), int(99)]],
    );
}

/// §1 (WITH on UPDATE): the CTE supplies a VALUE used by the SET. `c` aggregates a total
/// bonus (5+10+15 = 30) and `SET v = v + (SELECT total FROM c)` adds it to every row:
/// 100 -> 130, 200 -> 230.
#[test]
fn with_cte_supplies_update_value() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE bonus(amt)");
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v)");
    exec(&mut db, "INSERT INTO bonus VALUES (5), (10), (15)");
    exec(&mut db, "INSERT INTO t VALUES (1, 100), (2, 200)");
    exec(&mut db, "WITH c AS (SELECT sum(amt) AS total FROM bonus) UPDATE t SET v = v + (SELECT total FROM c)");
    assert_rows(
        &mut db,
        "SELECT id, v FROM t ORDER BY id",
        &[vec![int(1), int(130)], vec![int(2), int(230)]],
    );
}

/// §1 (WITH on DELETE): the CTE selects which rows to remove. `c` holds ids from `src`
/// with `flag = 1` ({1, 3}); `DELETE FROM t WHERE id IN (SELECT id FROM c)` removes those
/// rows, leaving only row 2.
#[test]
fn with_cte_filters_deleted_rows() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE src(id, flag)");
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v)");
    exec(&mut db, "INSERT INTO src VALUES (1, 1), (2, 0), (3, 1)");
    exec(&mut db, "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)");
    exec(&mut db, "WITH c AS (SELECT id FROM src WHERE flag = 1) DELETE FROM t WHERE id IN (SELECT id FROM c)");
    assert_rows(&mut db, "SELECT id, v FROM t ORDER BY id", &[vec![int(2), int(20)]]);
}

/// §1 (WITH on DELETE): the CTE supplies a scalar threshold from a SEPARATE table (so no
/// snapshot-of-the-delete-target subtlety). `cut` = 15, and `DELETE FROM t WHERE v <
/// (SELECT cutoff FROM cut)` removes only row 1 (v = 10).
#[test]
fn with_cte_scalar_threshold_delete() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE thresholds(kind, cutoff)");
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v)");
    exec(&mut db, "INSERT INTO thresholds VALUES ('low', 15)");
    exec(&mut db, "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)");
    exec(&mut db, "WITH cut AS (SELECT cutoff FROM thresholds WHERE kind = 'low') DELETE FROM t WHERE v < (SELECT cutoff FROM cut)");
    assert_rows(
        &mut db,
        "SELECT id, v FROM t ORDER BY id",
        &[vec![int(2), int(20)], vec![int(3), int(30)]],
    );
}

/// §3 (RECURSIVE WITH on INSERT): the recursive sequence `SELECT 1 UNION ALL SELECT n+1
/// FROM seq WHERE n < 5` yields 1..5 (queue algorithm: 1 -> 2 -> 3 -> 4 -> 5, then n=5
/// fails `n<5` and the recursion stops); `INSERT INTO t(n) SELECT n FROM seq` stores all
/// five rows.
#[test]
fn with_recursive_cte_feeds_insert() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(n)");
    exec(
        &mut db,
        "WITH RECURSIVE seq(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM seq WHERE n < 5) \
         INSERT INTO t(n) SELECT n FROM seq",
    );
    assert_rows(
        &mut db,
        "SELECT n FROM t ORDER BY n",
        &[vec![int(1)], vec![int(2)], vec![int(3)], vec![int(4)], vec![int(5)]],
    );
}

/// §3 (RECURSIVE WITH on DELETE): a recursive CTE generates the even ids {2, 4}
/// (`SELECT 2 UNION ALL SELECT n+2 FROM evens WHERE n+2 <= 4`) and
/// `DELETE FROM t WHERE id IN (SELECT n FROM evens)` removes rows 2 and 4, leaving
/// 1, 3, 5.
#[test]
fn with_recursive_cte_feeds_delete() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v)");
    exec(&mut db, "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30), (4, 40), (5, 50)");
    exec(
        &mut db,
        "WITH RECURSIVE evens(n) AS (SELECT 2 UNION ALL SELECT n + 2 FROM evens WHERE n + 2 <= 4) \
         DELETE FROM t WHERE id IN (SELECT n FROM evens)",
    );
    assert_rows(
        &mut db,
        "SELECT id, v FROM t ORDER BY id",
        &[vec![int(1), int(10)], vec![int(3), int(30)], vec![int(5), int(50)]],
    );
}
