//! Conformance battery: **a cross-namespace SUBQUERY inside a DML expression** — an
//! `UPDATE ... SET` assignment value, an UPSERT `DO UPDATE` `SET`/`WHERE`, AND a `RETURNING`
//! clause — exercised through the pinned `minisqlite::Connection` facade.
//!
//! The value on the right of an `UPDATE ... SET col = <expr>` is a general scalar expression
//! (`lang_update.html`: the SET grammar is `column-name = expr`); an UPSERT `DO UPDATE`
//! `SET`/`WHERE` is the same UPDATE-shaped grammar (`lang_upsert.html` §2: the INSERT "behaves
//! as an UPDATE"); and a `RETURNING` expression is likewise a general scalar expression
//! (`lang_returning.html`: the output columns are `expr`s like a SELECT's), so ALL may contain
//! a SELECT subquery — and that subquery may read ANY database the connection can name: the
//! `main` database, `temp`, or an `ATTACH`-ed database (`lang_attach.html` / `lang_naming.html`:
//! a `schema.object` qualifier reaches exactly that store; an unqualified name resolves
//! temp → main → attached in attach order). Real sqlite executes
//! `UPDATE t SET a = (SELECT max(x) FROM aux.u)` and
//! `INSERT INTO t VALUES (2) RETURNING (SELECT max(x) FROM aux.u)`; each is an ordinary read
//! of another namespace, no different from `INSERT INTO t VALUES ((SELECT … FROM aux.u))`,
//! which this engine already accepts.
//!
//! Every expectation here is derived from the SQLite docs in `spec/sqlite-doc/` and plain
//! SQL semantics, never from what the engine happens to return; a failing case is the
//! intended signal the engine diverges from the spec.
//!
//! Spec sources (`spec/sqlite-doc/`):
//!   * `lang_update.html`: the SET value is `expr` — a full scalar expression, so a
//!     scalar subquery is legal there (as it is anywhere `expr` appears).
//!   * `lang_returning.html`: a RETURNING output column is an `expr`; §3 explicitly allows
//!     subqueries in RETURNING ("If there are subqueries in the RETURNING clause …"), and
//!     §2.2 restricts only subqueries that reference the table being modified — a subquery
//!     reading a DIFFERENT table (an attached db / `temp`) is well-defined.
//!   * `lang_attach.html`: `ATTACH ':memory:' AS aux` adds an addressable in-memory
//!     database whose tables are reached as `aux.tbl`.
//!   * `lang_naming.html`: name resolution order (temp, main, attached) and the
//!     `schema.object` qualifier that reaches one specific store.
//!
//! ## DISCRIMINATORS vs GUARDS (verified by reverting the eval-view change at each site)
//! Before the fix, each DML write loop evaluated its cross-namespace-capable expressions
//! under a SINGLE-namespace read view pinned to the target's own namespace
//! (`Pagers::One { db }` in `minisqlite-exec/src/ops/{update,insert,delete}.rs`), which FAILS
//! CLOSED on any read that names a different namespace with `single-namespace context (db 0)
//! cannot reach namespace N`. Three expression classes were affected:
//!   * the `UPDATE ... SET` assignment eval (`update.rs`);
//!   * the UPSERT `DO UPDATE` `SET`-assignment AND `WHERE`-predicate eval (`insert.rs`,
//!     `do_upsert_update` — a DO UPDATE is an UPDATE, so both are subquery-legal); and
//!   * the `RETURNING` eval at ALL FIVE DML sites — rowid INSERT, WITHOUT ROWID INSERT, and
//!     UPSERT `DO UPDATE` (`insert.rs`), UPDATE (`update.rs`), and DELETE (`delete.rs`).
//! The fix evaluates all three classes under the whole-slice shared view (`self.pagers.source()`
//! / `pagers.source()`, routed through `ops::returning::eval_returning` for the RETURNING
//! class), the SAME view INSERT's `VALUES` always used.
//!   * DISCRIMINATORS — a SET or RETURNING subquery that reads a DIFFERENT namespace than the
//!     target. Each errored before the fix and now returns the correct value; each goes RED
//!     again the moment the eval view is reverted to `Pagers::One`.
//!   * GUARDS — a same-namespace subquery and a plain (no-subquery) statement. These always
//!     worked (the eval already reached the target's own store, and a plain assignment /
//!     RETURNING column needs no other store) and must keep working; they pin that the
//!     reorder did not disturb the hot path or the pre-update read semantics.

mod conformance;
use conformance::*;

// ===========================================================================
// DISCRIMINATORS — a SET subquery reading another namespace (RED before the fix
// with "cannot reach namespace N"; GREEN after).
// ===========================================================================

#[test]
fn update_set_noncorrelated_subquery_reads_attached_db() {
    // A non-correlated scalar subquery in the SET
    // value reads an ATTACH-ed database. `aux` is namespace 2 (main 0, temp 1, aux 2), so
    // the old single-namespace eval raised "cannot reach namespace 2".
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
    exec(&mut db, "INSERT INTO t VALUES (1, 'x')");
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.u(x INTEGER)");
    exec(&mut db, "INSERT INTO aux.u VALUES (100)");

    exec(&mut db, "UPDATE t SET a = (SELECT max(x) FROM aux.u) WHERE a = 1");

    // t.a took the subquery's value from aux.u; b (unassigned) is unchanged.
    assert_rows(&mut db, "SELECT a, b FROM t", &[vec![int(100), text("x")]]);
}

#[test]
fn update_set_correlated_subquery_reads_attached_db() {
    // A CORRELATED scalar subquery — the subquery's WHERE
    // references the outer UPDATE row (`t.k`) and reads a matching row from the attached
    // `aux.u`. Two rows, each taking ITS OWN correlated match, proving the outer row
    // (`old`, the pre-update row) is threaded into the cross-namespace subquery per row.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k INTEGER, v INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1, 0), (2, 0)");
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.u(k INTEGER, val INTEGER)");
    exec(&mut db, "INSERT INTO aux.u VALUES (1, 100), (2, 200)");

    exec(&mut db, "UPDATE t SET v = (SELECT val FROM aux.u WHERE aux.u.k = t.k)");

    // Each t row got the aux.u row whose k matched its own k.
    assert_rows_unordered(
        &mut db,
        "SELECT k, v FROM t",
        &[vec![int(1), int(100)], vec![int(2), int(200)]],
    );
    // The attached source is untouched by the UPDATE (it was read, not written).
    assert_rows_unordered(
        &mut db,
        "SELECT k, val FROM aux.u",
        &[vec![int(1), int(100)], vec![int(2), int(200)]],
    );
}

#[test]
fn update_set_subquery_reads_temp_namespace() {
    // The same mechanism against the `temp` namespace (db 1): a SET subquery reads a
    // `CREATE TEMP TABLE` source. The old eval raised "cannot reach namespace 1".
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    exec(&mut db, "CREATE TEMP TABLE u(x INTEGER)");
    exec(&mut db, "INSERT INTO temp.u VALUES (100)");

    exec(&mut db, "UPDATE t SET a = (SELECT max(x) FROM temp.u) WHERE a = 1");

    // t.a took the value from temp.u; the temp source is unchanged.
    assert_rows(&mut db, "SELECT a FROM t", &[vec![int(100)]]);
    assert_rows(&mut db, "SELECT x FROM temp.u", &[vec![int(100)]]);
}

#[test]
fn update_set_subquery_reads_temp_under_shadow() {
    // Cross-namespace SET subquery under TEMP SHADOWING: `temp.t` shadows `main.t`, but
    // the UPDATE targets `main.t` (qualified) while the subquery reads a DIFFERENT temp
    // table `temp.other`. The target resolves to main (db 0); the subquery reaches temp
    // (db 1). This pins that the target namespace and the subquery namespace are
    // independent — the write goes to main.t, the read comes from temp.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1)"); // main.t = {1} (created before any shadow)
    exec(&mut db, "CREATE TEMP TABLE t(a INTEGER)"); // temp.t shadows main.t
    exec(&mut db, "INSERT INTO temp.t VALUES (999)"); // distinct data to prove non-interference
    exec(&mut db, "CREATE TEMP TABLE other(x INTEGER)");
    exec(&mut db, "INSERT INTO temp.other VALUES (55)");

    exec(&mut db, "UPDATE main.t SET a = (SELECT max(x) FROM temp.other) WHERE a = 1");

    // main.t took temp.other's value; the temp.t shadow is untouched (the write resolved
    // to main, not the shadow), and temp.other (the read source) is unchanged.
    assert_rows(&mut db, "SELECT a FROM main.t", &[vec![int(55)]]);
    assert_rows(&mut db, "SELECT a FROM temp.t", &[vec![int(999)]]);
    assert_rows(&mut db, "SELECT x FROM temp.other", &[vec![int(55)]]);
}

#[test]
fn update_set_self_reference_plus_attached_subquery() {
    // The SET value combines a PRE-UPDATE self reference (`a`) with a cross-namespace
    // subquery (`(SELECT max(x) FROM aux.u)`). The reorder must keep evaluating the whole
    // expr against `old` (the pre-update row) — so `a` is the pre-update 10 — while the
    // subquery reaches `aux`. Expected: 10 + 5 = 15. This is a discriminator (the subquery
    // reads aux) that ALSO guards the "eval against old" invariant.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (10)");
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.u(x INTEGER)");
    exec(&mut db, "INSERT INTO aux.u VALUES (5)");

    exec(&mut db, "UPDATE t SET a = a + (SELECT max(x) FROM aux.u)");

    assert_rows(&mut db, "SELECT a FROM t", &[vec![int(15)]]);
}

// ===========================================================================
// GUARDS — must stay GREEN both before and after the fix.
// ===========================================================================

#[test]
fn update_set_same_db_subquery_control() {
    // A same-namespace SET subquery: both the target `t` and the subquery source `u` live
    // in `main`. This always worked (the single-namespace eval already reached the target's
    // own store) and must keep working — the reorder to the whole-slice view must not change
    // it. A same-db subquery reads the SAME pager the write targets (read-your-writes within
    // the statement is preserved because `Pagers::Set` resolves `set[db]`, the exact store
    // the old `Pagers::One` reborrowed).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    exec(&mut db, "CREATE TABLE u(x INTEGER)");
    exec(&mut db, "INSERT INTO u VALUES (42)");

    exec(&mut db, "UPDATE t SET a = (SELECT max(x) FROM u) WHERE a = 1");

    assert_rows(&mut db, "SELECT a FROM t", &[vec![int(42)]]);
    // The main-only subquery source is unchanged.
    assert_rows(&mut db, "SELECT x FROM u", &[vec![int(42)]]);
}

#[test]
fn update_plain_no_subquery_control() {
    // The plain UPDATE hot path: literal SET values, no subquery, all in `main`. This
    // exercises the common case that must be byte-for-byte unchanged by the reorder.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
    exec(&mut db, "INSERT INTO t VALUES (1, 'x'), (2, 'y')");

    exec(&mut db, "UPDATE t SET a = 5, b = 'z' WHERE a = 1");

    assert_rows_unordered(
        &mut db,
        "SELECT a, b FROM t",
        &[vec![int(5), text("z")], vec![int(2), text("y")]],
    );
}

#[test]
fn update_from_cross_db_source_table() {
    // BONUS coverage: `UPDATE t SET … FROM <cross-db table>` — the FROM clause joins the
    // target `main.t` against `aux.u` and the SET reads a column of the joined attached row.
    // The phase-1 join scan already runs under the whole-slice `source()` view (it must, to
    // reach the attached FROM table), and this fix aligns the SET-assignment eval to that
    // SAME view — so the whole cross-namespace UPDATE surface (FROM join + SET expr) reads
    // any namespace. This is a GUARD (the FROM scan path was already whole-slice), kept to
    // pin cross-db `UPDATE … FROM` end to end. (`lang_update.html`: the optional FROM clause;
    // `lang_attach.html`: the attached source is `aux.u`.)
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k INTEGER, v INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1, 0), (2, 0)");
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.u(k INTEGER, val INTEGER)");
    exec(&mut db, "INSERT INTO aux.u VALUES (1, 100), (2, 200)");

    exec(&mut db, "UPDATE t SET v = aux.u.val FROM aux.u WHERE aux.u.k = t.k");

    assert_rows_unordered(
        &mut db,
        "SELECT k, v FROM t",
        &[vec![int(1), int(100)], vec![int(2), int(200)]],
    );
}

#[test]
fn update_set_preupdate_read_semantics_control() {
    // Pre-update read semantics (a regression guard for the reorder): every SET expr is
    // evaluated against the row AS IT WAS before any assignment, so `SET a = a + 1, b = a`
    // uses the PRE-update `a` for BOTH — a becomes 2, and b becomes the OLD a (1), not the
    // just-assigned 2 (SQLite's documented behavior). The fix keeps `eval(expr, old, …)`,
    // so this must still hold.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1, 0)");

    exec(&mut db, "UPDATE t SET a = a + 1, b = a");

    // a = pre-update a + 1 = 2; b = pre-update a = 1 (NOT the new a=2).
    assert_rows(&mut db, "SELECT a, b FROM t", &[vec![int(2), int(1)]]);
}

// ===========================================================================
// UPSERT DO UPDATE SET / WHERE DISCRIMINATORS — a subquery inside an UPSERT
// `DO UPDATE`'s SET value or WHERE predicate that reads another namespace. A
// `DO UPDATE` behaves as an UPDATE (`lang_upsert.html` §2), so both its SET and
// WHERE are subquery-legal. Each errored before the fix with "cannot reach
// namespace N" (`do_upsert_update` evaluated SET/WHERE on the single-namespace
// `Pagers::One` view) and now executes; each goes RED again if `do_upsert_update`
// is reverted to the single-namespace view.
// ===========================================================================

#[test]
fn upsert_do_update_set_subquery_reads_attached_db() {
    // The DO UPDATE SET value is a scalar subquery over the ATTACHed `aux.u` (db 2). A
    // conflict on k=1 routes to DO UPDATE; the fix evaluates the SET under the whole-slice
    // source view, so the subquery reaches aux and v becomes 100. Before the fix this raised
    // "cannot reach namespace 2".
    let mut db = mem();
    exec(&mut db, "CREATE TABLE p(k INTEGER PRIMARY KEY, v INTEGER)");
    exec(&mut db, "INSERT INTO p VALUES (1, 0)");
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.u(x INTEGER)");
    exec(&mut db, "INSERT INTO aux.u VALUES (100)");

    exec(
        &mut db,
        "INSERT INTO p VALUES (1, 9) ON CONFLICT(k) DO UPDATE SET v = (SELECT max(x) FROM aux.u)",
    );

    // The existing row was updated in place with aux.u's max; no second row inserted.
    assert_rows(&mut db, "SELECT k, v FROM p", &[vec![int(1), int(100)]]);
    // The attached source is unchanged (read, not written).
    assert_rows(&mut db, "SELECT x FROM aux.u", &[vec![int(100)]]);
}

#[test]
fn upsert_do_update_where_true_cross_db_applies() {
    // The DO UPDATE WHERE predicate holds a scalar subquery over the ATTACHed `aux.u` (db 2).
    // `lang_upsert.html` §2.1: a NULL/false WHERE makes the DO UPDATE a no-op. Here the
    // predicate `(SELECT max(x) FROM aux.u) > 50` is TRUE (100 > 50), so the update applies —
    // the point is the cross-namespace read INSIDE the WHERE, which errored before the fix.
    // `SET v = 7` is deliberately NEITHER the original 0 NOR the candidate 9, so asserting
    // v == 7 proves the DO UPDATE genuinely ran (not "the candidate INSERT row won").
    let mut db = mem();
    exec(&mut db, "CREATE TABLE p(k INTEGER PRIMARY KEY, v INTEGER)");
    exec(&mut db, "INSERT INTO p VALUES (1, 0)");
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.u(x INTEGER)");
    exec(&mut db, "INSERT INTO aux.u VALUES (100)");

    exec(
        &mut db,
        "INSERT INTO p VALUES (1, 9) ON CONFLICT(k) DO UPDATE SET v = 7 \
         WHERE (SELECT max(x) FROM aux.u) > 50",
    );

    // WHERE true → the DO UPDATE applied (v = 7, not 0 and not the candidate 9); no second row.
    assert_rows(&mut db, "SELECT k, v FROM p", &[vec![int(1), int(7)]]);
}

#[test]
fn upsert_do_update_where_false_cross_db_is_noop() {
    // Companion to the WHERE discriminator: a cross-namespace WHERE that evaluates FALSE makes
    // the DO UPDATE a no-op (existing row unchanged, no error, nothing inserted —
    // `lang_upsert.html` §2.1). This pins that the cross-db WHERE predicate is genuinely
    // EVALUATED (not merely reached): reading aux.u's max (100) and testing `> 500` yields
    // false, so v stays at its original 0. Before the fix this errored on the aux read instead
    // of quietly no-op'ing.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE p(k INTEGER PRIMARY KEY, v INTEGER)");
    exec(&mut db, "INSERT INTO p VALUES (1, 0)");
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.u(x INTEGER)");
    exec(&mut db, "INSERT INTO aux.u VALUES (100)");

    exec(
        &mut db,
        "INSERT INTO p VALUES (1, 9) ON CONFLICT(k) DO UPDATE SET v = 9 \
         WHERE (SELECT max(x) FROM aux.u) > 500",
    );

    // WHERE false → DO UPDATE is a no-op: v stays 0 (NOT 9), and no new row was inserted.
    assert_rows(&mut db, "SELECT k, v FROM p", &[vec![int(1), int(0)]]);
}

// ===========================================================================
// UPSERT DO UPDATE SET GUARD — a same-namespace UPSERT SET subquery. Always
// worked and must keep working after the view-swap; pins that the cross-db
// UPSERT discriminators fail specifically on the CROSS-namespace reach.
// ===========================================================================

#[test]
fn upsert_do_update_set_same_db_subquery_control() {
    // Both the target `p` and the DO UPDATE SET subquery source `u` live in `main`. GREEN
    // before AND after the fix; when `do_upsert_update`'s view is reverted to single-namespace
    // this stays GREEN (the subquery reads the target's own namespace), proving the cross-db
    // UPSERT discriminators fail on the CROSS-namespace reach, not on UPSERT subqueries at all.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE p(k INTEGER PRIMARY KEY, v INTEGER)");
    exec(&mut db, "INSERT INTO p VALUES (1, 0)");
    exec(&mut db, "CREATE TABLE u(x INTEGER)");
    exec(&mut db, "INSERT INTO u VALUES (42)");

    exec(
        &mut db,
        "INSERT INTO p VALUES (1, 9) ON CONFLICT(k) DO UPDATE SET v = (SELECT max(x) FROM u)",
    );

    assert_rows(&mut db, "SELECT k, v FROM p", &[vec![int(1), int(42)]]);
}

// ===========================================================================
// RETURNING DISCRIMINATORS — a subquery inside a `RETURNING` clause that reads
// another namespace, at ALL FIVE DML RETURNING eval sites. Each errored before
// the fix with "cannot reach namespace N" (the RETURNING eval was on the
// single-namespace `Pagers::One` view) and now returns the correct value; each
// goes RED again the moment `ops::returning::eval_returning`'s view is reverted
// to a single-namespace one. RETURNING yields one result row per changed row, so
// each expectation is asserted directly on the DML statement's own result rows.
// Every RETURNING here also references a COLUMN of the changed row alongside the
// cross-namespace subquery, pinning that the eval binds the written/deleted row
// AND reaches the other namespace in the same output row.
// ===========================================================================

#[test]
fn insert_returning_subquery_reads_attached_db() {
    // Site 1/5 — rowid-table INSERT RETURNING (ops/insert.rs, rowid path). The
    // witness (A): a scalar subquery over the ATTACHed `aux.u` (namespace 2) in RETURNING.
    // The old single-namespace RETURNING eval raised "cannot reach namespace 2".
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.u(x INTEGER)");
    exec(&mut db, "INSERT INTO aux.u VALUES (100)");

    // The one inserted row yields one RETURNING row: its column `a` (=2) and aux.u's max.
    assert_rows(
        &mut db,
        "INSERT INTO t VALUES (2) RETURNING a, (SELECT max(x) FROM aux.u)",
        &[vec![int(2), int(100)]],
    );
    // The row was actually inserted (RETURNING is not a dry run).
    assert_rows_unordered(&mut db, "SELECT a FROM t", &[vec![int(1)], vec![int(2)]]);
}

#[test]
fn insert_without_rowid_returning_subquery_reads_attached_db() {
    // Site 2/5 — WITHOUT ROWID INSERT RETURNING (ops/insert.rs, WR path). The
    // witness (D). A WR RETURNING binds against `[c0..c_{N-1}]` (no rowid register), and the
    // cross-namespace subquery must reach `aux` (db 2) the same as a rowid table's.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE w(k INTEGER PRIMARY KEY, v INTEGER) WITHOUT ROWID");
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.u(x INTEGER)");
    exec(&mut db, "INSERT INTO aux.u VALUES (100)");

    assert_rows(
        &mut db,
        "INSERT INTO w VALUES (1, 0) RETURNING k, v, (SELECT max(x) FROM aux.u)",
        &[vec![int(1), int(0), int(100)]],
    );
    assert_rows(&mut db, "SELECT k, v FROM w", &[vec![int(1), int(0)]]);
}

#[test]
fn insert_upsert_returning_subquery_reads_attached_db() {
    // Site 3/5 — UPSERT `DO UPDATE` RETURNING (ops/insert.rs, do_upsert_update path). The
    // witness (E): a conflict routes to DO UPDATE, whose RETURNING subquery reads
    // `aux` (db 2). The fix hands the binding row back to the caller, which evaluates
    // RETURNING under the shared view once the DO UPDATE's `&mut` write borrow is released.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE p(k INTEGER PRIMARY KEY, v INTEGER)");
    exec(&mut db, "INSERT INTO p VALUES (1, 0)");
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.u(x INTEGER)");
    exec(&mut db, "INSERT INTO aux.u VALUES (100)");

    // Conflict on k=1 → DO UPDATE SET v=9; RETURNING reports the POST-update v and aux's max.
    assert_rows(
        &mut db,
        "INSERT INTO p VALUES (1, 9) ON CONFLICT(k) DO UPDATE SET v = 9 \
         RETURNING v, (SELECT max(x) FROM aux.u)",
        &[vec![int(9), int(100)]],
    );
    // The existing row was updated in place (no second row inserted).
    assert_rows(&mut db, "SELECT k, v FROM p", &[vec![int(1), int(9)]]);
}

#[test]
fn update_returning_subquery_reads_attached_db() {
    // Site 4/5 — UPDATE RETURNING (ops/update.rs). RETURNING reports the UPDATED `a`
    // (=5) alongside the cross-namespace read.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.u(x INTEGER)");
    exec(&mut db, "INSERT INTO aux.u VALUES (100)");

    assert_rows(
        &mut db,
        "UPDATE t SET a = 5 RETURNING a, (SELECT max(x) FROM aux.u)",
        &[vec![int(5), int(100)]],
    );
    assert_rows(&mut db, "SELECT a FROM t", &[vec![int(5)]]);
}

#[test]
fn delete_returning_subquery_reads_attached_db() {
    // Site 5/5 — DELETE RETURNING (ops/delete.rs). The witness (B). RETURNING
    // reports the DELETED row's `a` (=1) alongside the cross-namespace read; the row is gone.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.u(x INTEGER)");
    exec(&mut db, "INSERT INTO aux.u VALUES (100)");

    assert_rows(
        &mut db,
        "DELETE FROM t WHERE a = 1 RETURNING a, (SELECT max(x) FROM aux.u)",
        &[vec![int(1), int(100)]],
    );
    // The row was actually deleted.
    assert_rows(&mut db, "SELECT count(*) FROM t", &[vec![int(0)]]);
}

#[test]
fn update_returning_subquery_reads_temp_namespace() {
    // Broaden the RETURNING class to the `temp` namespace (db 1): an UPDATE RETURNING
    // subquery over a `CREATE TEMP TABLE`. The old eval raised "cannot reach namespace 1".
    // This pins that the RETURNING fix reaches temp, not just an attached db.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    exec(&mut db, "CREATE TEMP TABLE u(x INTEGER)");
    exec(&mut db, "INSERT INTO temp.u VALUES (100)");

    assert_rows(
        &mut db,
        "UPDATE t SET a = 5 RETURNING a, (SELECT max(x) FROM temp.u)",
        &[vec![int(5), int(100)]],
    );
    assert_rows(&mut db, "SELECT a FROM t", &[vec![int(5)]]);
    // The temp source is unchanged (it was read, not written).
    assert_rows(&mut db, "SELECT x FROM temp.u", &[vec![int(100)]]);
}

// ===========================================================================
// RETURNING GUARD — a same-namespace RETURNING subquery. It always worked (the
// single-namespace eval already reached the target's own store) and must keep
// working after the fix; it pins that the eval-view swap did not disturb a
// same-db RETURNING subquery (read-your-writes within the statement preserved,
// because `Pagers::Set` resolves `set[db]`, the exact store the old
// `Pagers::One` reborrowed).
// ===========================================================================

#[test]
fn insert_returning_same_db_subquery_control() {
    // Both the target `t` and the RETURNING subquery source `u` live in `main`. GREEN before
    // AND after the fix; when the helper's view is reverted to single-namespace this stays
    // GREEN (the subquery reads the target's own namespace), proving the RED discriminators
    // fail specifically on the CROSS-namespace reach, not on RETURNING subqueries in general.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    exec(&mut db, "CREATE TABLE u(x INTEGER)");
    exec(&mut db, "INSERT INTO u VALUES (42)");

    assert_rows(
        &mut db,
        "INSERT INTO t VALUES (2) RETURNING a, (SELECT max(x) FROM u)",
        &[vec![int(2), int(42)]],
    );
    assert_rows_unordered(&mut db, "SELECT a FROM t", &[vec![int(1)], vec![int(2)]]);
}

// ===========================================================================
// COMBINED / EDGE cases — interactions the single-clause tests cover only
// separately, plus the NULL (empty scalar subquery) path through the reordered
// cross-namespace eval.
// ===========================================================================

#[test]
fn update_set_and_returning_both_cross_db_in_one_statement() {
    // A single UPDATE carrying BOTH a cross-db SET subquery AND a cross-db RETURNING subquery.
    // Pins the interaction the per-clause tests cover only in isolation: the SET eval's shared
    // `source()` scope must fully close before the RETURNING eval opens its OWN `source()`
    // view (the write borrow sits between them), and RETURNING must observe the value SET just
    // wrote (post-update row). Both subqueries reach `aux` (db 2); both errored before the fix.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.u(x INTEGER)");
    exec(&mut db, "INSERT INTO aux.u VALUES (100)");

    // SET a = 100 (from aux); RETURNING reports the UPDATED a (=100) AND aux's max again.
    assert_rows(
        &mut db,
        "UPDATE t SET a = (SELECT max(x) FROM aux.u) RETURNING a, (SELECT max(x) FROM aux.u)",
        &[vec![int(100), int(100)]],
    );
    assert_rows(&mut db, "SELECT a FROM t", &[vec![int(100)]]);
}

#[test]
fn update_set_cross_db_subquery_empty_source_is_null() {
    // A cross-db SET subquery whose source table is EMPTY: `max()` over no rows is NULL (SQL
    // aggregate semantics), so the reordered cross-namespace SET eval must produce NULL, and
    // INTEGER affinity leaves NULL as NULL. Pins the NULL path through the reordered eval —
    // before the fix this errored on the aux read before ever producing a value.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1)");
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.u(x INTEGER)"); // deliberately empty

    exec(&mut db, "UPDATE t SET a = (SELECT max(x) FROM aux.u)");

    // max() over an empty table is NULL.
    assert_rows(&mut db, "SELECT a FROM t", &[vec![null()]]);
}

#[test]
fn insert_returning_cross_db_subquery_empty_source_is_null() {
    // RETURNING variant of the empty-source NULL path: a cross-db RETURNING subquery whose
    // source is empty yields NULL in the output row (not an error, not a missing column). Pins
    // the NULL path through `eval_returning` under the whole-slice view.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.u(x INTEGER)"); // deliberately empty

    assert_rows(
        &mut db,
        "INSERT INTO t VALUES (5) RETURNING a, (SELECT max(x) FROM aux.u)",
        &[vec![int(5), null()]],
    );
    assert_rows(&mut db, "SELECT a FROM t", &[vec![int(5)]]);
}
