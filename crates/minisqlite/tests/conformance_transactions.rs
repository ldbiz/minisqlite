//! Conformance battery: transaction control on a SINGLE connection —
//! BEGIN / COMMIT / ROLLBACK, the keyword spellings, the DEFERRED / IMMEDIATE /
//! EXCLUSIVE modes, the error cases, and SAVEPOINT / RELEASE / ROLLBACK TO.
//!
//! Every expected value here is TRANSCRIBED FROM THE SPEC in `spec/sqlite-doc/`,
//! never from what the engine returns. Binding sources:
//!
//!   * `lang_transaction.html` §1 (Transaction Control Syntax): the grammars
//!     `BEGIN [DEFERRED|IMMEDIATE|EXCLUSIVE] [TRANSACTION]`,
//!     `COMMIT|END [TRANSACTION]`, and
//!     `ROLLBACK [TRANSACTION] [TO [SAVEPOINT] name]`.
//!   * `lang_transaction.html` §2 (Transactions): "Any command that accesses the
//!     database ... will automatically start a transaction if one is not already
//!     in effect. Automatically started transactions are committed when the last
//!     SQL statement finishes." (autocommit default); "END TRANSACTION is an
//!     alias for COMMIT"; "Transactions created using BEGIN...COMMIT do not
//!     nest."; "An attempt to invoke the BEGIN command within a transaction will
//!     fail with an error".
//!   * `lang_transaction.html` §2.1 (read vs write): CREATE / DELETE / INSERT /
//!     UPDATE are write statements that start (or upgrade to) a write
//!     transaction — so a ROLLBACK reverts their effects.
//!   * `lang_transaction.html` §2.2 (DEFERRED, IMMEDIATE, EXCLUSIVE): "The
//!     default transaction behavior is DEFERRED."; all three are accepted mode
//!     keywords.
//!   * `lang_savepoint.html` §2 (Savepoints): "The ROLLBACK TO command reverts
//!     the state of the database back to what it was just after the corresponding
//!     SAVEPOINT ... does not cancel the transaction ... restarts the transaction
//!     again at the beginning. All intervening SAVEPOINTs are canceled"; RELEASE
//!     "is like a COMMIT for a SAVEPOINT"; "If a RELEASE command releases the
//!     outermost savepoint ... then RELEASE is the same as COMMIT"; "The COMMIT
//!     command may be used to release all savepoints and commit the transaction
//!     even if the transaction was originally started by a SAVEPOINT"; an unknown
//!     name in RELEASE returns an error; an inner RELEASE can still be undone by
//!     an outer ROLLBACK ("Content is not actually committed ... until the
//!     outermost transaction commits").
//!   * `lang_savepoint.html` §3 (Transaction Nesting Rules): "The ROLLBACK
//!     command without a TO clause rolls backs all transactions and leaves the
//!     transaction stack empty."; "If the savepoint-name in a ROLLBACK TO command
//!     does not match any SAVEPOINT on the stack, then the ROLLBACK command fails
//!     with an error and leaves the state of the database unchanged."
//!
//! Because a single in-memory `Connection` keeps transaction state across
//! `execute`/`query` calls, each test drives a multi-statement sequence on ONE
//! `db`. `Value` has no `PartialEq`, so every result check goes through the shared
//! harness (`assert_scalar` / `assert_rows_unordered` / `assert_*_error`).
//!
//! Expected values are transcribed from the SQLite documentation, not from what
//! this engine returns; a case that reveals an engine bug is left as a genuine
//! failing assertion rather than weakened to pass. Some transaction error messages
//! (e.g. sqlite3's "cannot commit - no transaction is active") are real sqlite's
//! exact text; those are cited in comments for context, but the assertions only
//! check WHETHER a statement errors. Only a case that HANGS or PANICS the engine
//! itself is `#[ignore]`d, in its own test — an ordinary mismatch must stay visible.

mod conformance;

use conformance::*;
use minisqlite::Connection;

// ---- fixtures ----------------------------------------------------------------

/// A fresh in-memory database with an empty `CREATE TABLE t(a)` already created
/// under autocommit (so the table exists independent of any later transaction).
fn new_t() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    db
}

/// `new_t()` plus two autocommitted rows (`a=1`, `a=2`). Gives a following
/// explicit transaction a committed baseline of two rows to revert to.
fn seeded_t() -> Connection {
    let mut db = new_t();
    exec(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "INSERT INTO t VALUES(2)");
    db
}

// =============================================================================
// COMMIT / ROLLBACK core (lang_transaction §2)
// =============================================================================

/// COMMIT persists every write done since BEGIN: two inserts inside one
/// transaction, committed, leave two rows.
#[test]
fn commit_persists_inserts() {
    let mut db = new_t();
    exec(&mut db, "BEGIN");
    exec(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "INSERT INTO t VALUES(2)");
    exec(&mut db, "COMMIT");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2));
}

/// ROLLBACK reverts an INSERT. The change is visible WITHIN the transaction on
/// the same connection (count 3), but ROLLBACK undoes it (back to 2).
#[test]
fn rollback_reverts_insert() {
    let mut db = seeded_t();
    exec(&mut db, "BEGIN");
    exec(&mut db, "INSERT INTO t VALUES(3)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(3)); // visible inside tx
    exec(&mut db, "ROLLBACK");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2)); // insert undone
}

/// ROLLBACK restores rows removed by a DELETE: the delete empties the table
/// inside the transaction (count 0), and rollback brings the two rows back.
#[test]
fn rollback_restores_deleted_rows() {
    let mut db = seeded_t();
    exec(&mut db, "BEGIN");
    exec(&mut db, "DELETE FROM t");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0)); // gone inside tx
    exec(&mut db, "ROLLBACK");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2)); // restored
}

/// ROLLBACK reverts an UPDATE: the new values are visible inside the transaction
/// and the original values return after rollback. Checked by the actual row
/// values (not an aggregate), so the assertion does not depend on `sum()` typing.
#[test]
fn rollback_reverts_update() {
    let mut db = seeded_t(); // rows a=1, a=2
    exec(&mut db, "BEGIN");
    exec(&mut db, "UPDATE t SET a = a + 100");
    assert_rows_unordered(&mut db, "SELECT a FROM t", &[vec![int(101)], vec![int(102)]]);
    exec(&mut db, "ROLLBACK");
    assert_rows_unordered(&mut db, "SELECT a FROM t", &[vec![int(1)], vec![int(2)]]);
}

// =============================================================================
// Autocommit default (lang_transaction §2 / §2.3)
// =============================================================================

/// A bare INSERT outside any BEGIN auto-starts and auto-commits its own
/// transaction, so a subsequent SELECT on the same connection sees the row.
#[test]
fn autocommit_bare_insert_visible() {
    let mut db = new_t();
    exec(&mut db, "INSERT INTO t VALUES(1)"); // no BEGIN → autocommit
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(1));
}

/// Proof that autocommit truly COMMITS (not merely "is visible"): a bare insert
/// is durable, so a later explicit BEGIN…ROLLBACK undoes only the row added
/// inside that transaction — the autocommitted row survives.
#[test]
fn autocommit_committed_row_survives_later_rollback() {
    let mut db = new_t();
    exec(&mut db, "INSERT INTO t VALUES(1)"); // autocommitted, durable
    exec(&mut db, "BEGIN");
    exec(&mut db, "INSERT INTO t VALUES(2)");
    exec(&mut db, "ROLLBACK"); // only the second insert is undone
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(1));
}

// =============================================================================
// Keyword spellings (lang_transaction §1 grammar; §2 END alias)
// =============================================================================

/// `BEGIN TRANSACTION` is accepted as the explicit-transaction spelling.
#[test]
fn begin_transaction_keyword() {
    let mut db = new_t();
    exec(&mut db, "BEGIN TRANSACTION");
    exec(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "COMMIT");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(1));
}

/// `COMMIT TRANSACTION` is accepted as a spelling of COMMIT.
#[test]
fn commit_transaction_keyword() {
    let mut db = new_t();
    exec(&mut db, "BEGIN");
    exec(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "COMMIT TRANSACTION");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(1));
}

/// `END` is a COMMIT synonym (lang_transaction §2). The count-1 result
/// distinguishes a real COMMIT from a mis-handling as ROLLBACK (which would be 0).
#[test]
fn end_is_commit_synonym() {
    let mut db = new_t();
    exec(&mut db, "BEGIN");
    exec(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "END");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(1));
}

/// `END TRANSACTION` is a COMMIT synonym: "END TRANSACTION is an alias for
/// COMMIT" (lang_transaction §2).
#[test]
fn end_transaction_is_commit_synonym() {
    let mut db = new_t();
    exec(&mut db, "BEGIN");
    exec(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "END TRANSACTION");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(1));
}

/// `ROLLBACK TRANSACTION` is accepted as a spelling of ROLLBACK.
#[test]
fn rollback_transaction_keyword() {
    let mut db = seeded_t();
    exec(&mut db, "BEGIN TRANSACTION");
    exec(&mut db, "INSERT INTO t VALUES(3)");
    exec(&mut db, "ROLLBACK TRANSACTION");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2));
}

// =============================================================================
// DEFERRED / IMMEDIATE / EXCLUSIVE modes (lang_transaction §2.2)
// =============================================================================
// All three mode keywords are accepted; on a single fresh connection none can
// fail with SQLITE_BUSY (there is no competing writer), so each completes a full
// insert+commit cycle.

/// `BEGIN DEFERRED` — the default mode, spelled explicitly.
#[test]
fn begin_deferred_accepted() {
    let mut db = new_t();
    exec(&mut db, "BEGIN DEFERRED");
    exec(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "COMMIT");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(1));
}

/// `BEGIN IMMEDIATE` — starts a write transaction immediately.
#[test]
fn begin_immediate_accepted() {
    let mut db = new_t();
    exec(&mut db, "BEGIN IMMEDIATE");
    exec(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "COMMIT");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(1));
}

/// `BEGIN EXCLUSIVE` — like IMMEDIATE, also accepted.
#[test]
fn begin_exclusive_accepted() {
    let mut db = new_t();
    exec(&mut db, "BEGIN EXCLUSIVE");
    exec(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "COMMIT");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(1));
}

/// A mode keyword may be combined with the optional TRANSACTION keyword:
/// `BEGIN DEFERRED TRANSACTION`.
#[test]
fn begin_deferred_transaction_keyword() {
    let mut db = new_t();
    exec(&mut db, "BEGIN DEFERRED TRANSACTION");
    exec(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "COMMIT");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(1));
}

// =============================================================================
// Error cases (lang_transaction §2; lang_savepoint §3)
// =============================================================================

/// COMMIT with no active transaction is an error. A fresh connection is in
/// autocommit mode with nothing to commit; real sqlite3 rejects this with
/// "cannot commit - no transaction is active". We assert only that it errors.
#[test]
fn commit_without_transaction_errors() {
    let mut db = new_t();
    assert_exec_error(&mut db, "COMMIT");
}

/// ROLLBACK with no active transaction is likewise an error (sqlite3: "cannot
/// rollback - no transaction is active").
#[test]
fn rollback_without_transaction_errors() {
    let mut db = new_t();
    assert_exec_error(&mut db, "ROLLBACK");
}

/// A nested BEGIN inside an open transaction fails: "An attempt to invoke the
/// BEGIN command within a transaction will fail with an error" (lang_transaction
/// §2; sqlite3: "cannot start a transaction within a transaction").
#[test]
fn nested_begin_errors() {
    let mut db = new_t();
    exec(&mut db, "BEGIN");
    assert_exec_error(&mut db, "BEGIN");
}

/// The failed nested BEGIN does not disturb the transaction already open, so the
/// original transaction still commits its work.
#[test]
fn nested_begin_leaves_original_transaction_active() {
    let mut db = new_t();
    exec(&mut db, "BEGIN");
    exec(&mut db, "INSERT INTO t VALUES(1)");
    assert_exec_error(&mut db, "BEGIN"); // rejected, first tx untouched
    exec(&mut db, "COMMIT");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(1));
}

// =============================================================================
// SAVEPOINT / RELEASE / ROLLBACK TO (lang_savepoint §2, §3)
// =============================================================================
// SAVEPOINT support may be partial in this engine; these cases assert the
// spec-documented behavior and are left to fail if unsupported rather than
// weakened to pass.

/// ROLLBACK TO reverts the database to "what it was just after the corresponding
/// SAVEPOINT" (lang_savepoint §2). The savepoint remains, so RELEASE afterwards
/// commits the (now unchanged) transaction.
#[test]
fn savepoint_rollback_to_reverts() {
    let mut db = seeded_t(); // committed baseline: 2 rows
    exec(&mut db, "SAVEPOINT sp1");
    exec(&mut db, "INSERT INTO t VALUES(3)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(3));
    exec(&mut db, "ROLLBACK TO sp1");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2)); // reverted to just-after sp1
    exec(&mut db, "RELEASE sp1");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2));
}

/// ROLLBACK TO "does not cancel the transaction ... [it] restarts the
/// transaction again at the beginning" (lang_savepoint §2). So an INSERT after a
/// ROLLBACK TO is still part of the savepoint transaction and is kept on RELEASE.
#[test]
fn savepoint_rollback_to_keeps_transaction_open() {
    let mut db = new_t();
    exec(&mut db, "SAVEPOINT sp1"); // outermost savepoint == BEGIN DEFERRED
    exec(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "ROLLBACK TO sp1"); // undoes the first insert; sp1 stays
    exec(&mut db, "INSERT INTO t VALUES(2)"); // still inside the transaction
    exec(&mut db, "RELEASE sp1"); // outermost release == COMMIT
    assert_rows_unordered(&mut db, "SELECT a FROM t", &[vec![int(2)]]); // only the 2nd insert
}

/// RELEASE of the outermost savepoint commits, so the inserted row persists
/// (lang_savepoint §2/§3: "If a RELEASE command releases the outermost savepoint
/// ... then RELEASE is the same as COMMIT").
#[test]
fn release_savepoint_keeps_changes() {
    let mut db = new_t();
    exec(&mut db, "SAVEPOINT sp1");
    exec(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "RELEASE sp1");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(1));
}

/// The release-stmt grammar allows the optional SAVEPOINT keyword:
/// `RELEASE SAVEPOINT name` (lang_savepoint §1).
#[test]
fn release_savepoint_keyword_accepted() {
    let mut db = new_t();
    exec(&mut db, "SAVEPOINT sp1");
    exec(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "RELEASE SAVEPOINT sp1");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(1));
}

/// The rollback-stmt grammar allows `ROLLBACK [TRANSACTION] TO [SAVEPOINT] name`;
/// here the full `ROLLBACK TO SAVEPOINT name` spelling.
#[test]
fn rollback_to_savepoint_keyword_accepted() {
    let mut db = seeded_t();
    exec(&mut db, "SAVEPOINT sp1");
    exec(&mut db, "INSERT INTO t VALUES(3)");
    exec(&mut db, "ROLLBACK TO SAVEPOINT sp1");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2));
    exec(&mut db, "RELEASE sp1");
}

/// Nested savepoints allow partial rollback (lang_savepoint §3): ROLLBACK TO the
/// inner savepoint undoes only work after it; a later ROLLBACK TO the outer
/// savepoint undoes the rest.
#[test]
fn nested_savepoints_partial_rollback() {
    let mut db = new_t();
    exec(&mut db, "SAVEPOINT sp1");
    exec(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "SAVEPOINT sp2");
    exec(&mut db, "INSERT INTO t VALUES(2)");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2));
    exec(&mut db, "ROLLBACK TO sp2"); // undoes only the second insert
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(1));
    exec(&mut db, "ROLLBACK TO sp1"); // undoes the first insert too
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
    exec(&mut db, "RELEASE sp1");
}

/// ROLLBACK TO an unknown savepoint name errors and leaves the database
/// unchanged (lang_savepoint §3).
#[test]
fn rollback_to_unknown_savepoint_errors() {
    let mut db = new_t();
    exec(&mut db, "SAVEPOINT sp1");
    exec(&mut db, "INSERT INTO t VALUES(1)");
    assert_exec_error(&mut db, "ROLLBACK TO nosuch");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(1)); // unchanged
    exec(&mut db, "RELEASE sp1");
}

/// RELEASE of an unknown savepoint name errors; the real savepoint is untouched
/// and can still be released (lang_savepoint §2).
#[test]
fn release_unknown_savepoint_errors() {
    let mut db = new_t();
    exec(&mut db, "SAVEPOINT sp1");
    assert_exec_error(&mut db, "RELEASE nosuch");
    exec(&mut db, "RELEASE sp1"); // sp1 was not affected by the failed RELEASE
}

/// COMMIT can commit a transaction that was started by SAVEPOINT rather than
/// BEGIN, releasing all savepoints (lang_savepoint §2).
#[test]
fn commit_releases_savepoint_started_transaction() {
    let mut db = new_t();
    exec(&mut db, "SAVEPOINT sp1");
    exec(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "COMMIT");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(1));
}

/// An inner savepoint may be RELEASEd, but an outer ROLLBACK still undoes that
/// work: "Content is not actually committed on the disk until the outermost
/// transaction commits" (lang_savepoint §2).
#[test]
fn inner_release_undone_by_outer_rollback() {
    let mut db = new_t();
    exec(&mut db, "BEGIN");
    exec(&mut db, "SAVEPOINT sp1");
    exec(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "RELEASE sp1"); // inner "commit"
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(1)); // visible within outer tx
    exec(&mut db, "ROLLBACK"); // outer rollback discards everything
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
}

/// A plain ROLLBACK (no TO clause) "rolls back all transactions and leaves the
/// transaction stack empty" (lang_savepoint §3), discarding every savepoint.
/// With the stack empty afterwards, a further COMMIT has nothing to commit and
/// errors.
#[test]
fn plain_rollback_discards_all_savepoints() {
    let mut db = new_t();
    exec(&mut db, "BEGIN");
    exec(&mut db, "SAVEPOINT sp1");
    exec(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "SAVEPOINT sp2");
    exec(&mut db, "INSERT INTO t VALUES(2)");
    exec(&mut db, "ROLLBACK");
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
    assert_exec_error(&mut db, "COMMIT"); // stack empty → nothing to commit
}

// =============================================================================
// DDL is transactional (lang_transaction §2.1: CREATE is a write statement)
// =============================================================================

/// A CREATE TABLE performed inside a transaction is reverted by ROLLBACK: after
/// the rollback the table does not exist, so querying it errors.
#[test]
fn rollback_reverts_create_table() {
    let mut db = mem();
    exec(&mut db, "BEGIN");
    exec(&mut db, "CREATE TABLE tmp(a)");
    exec(&mut db, "INSERT INTO tmp VALUES(1)");
    assert_scalar(&mut db, "SELECT count(*) FROM tmp", int(1)); // exists inside tx
    exec(&mut db, "ROLLBACK");
    assert_query_error(&mut db, "SELECT * FROM tmp"); // table is gone
}

// =============================================================================
// SAVEPOINT deep coverage (lang_savepoint §2, §3) — appended battery
// =============================================================================
// These deepen the savepoint cases above. Every expected value is derived from
// the documented transaction-stack model in `spec/sqlite-doc/lang_savepoint.html`
// (never from engine output). A spec-correct assertion is left to fail rather than
// weakened — a failure here is a candidate bug in the SAVEPOINT implementation. The
// stack model used
// throughout (from §3): SAVEPOINT pushes a named mark (starting a DEFERRED
// transaction when the stack was empty); ROLLBACK TO reverts to the state just
// after the matching mark, cancels marks above it, and KEEPS the matching mark
// with the transaction still open; RELEASE pops marks down to and including the
// most recent matching name, committing only if that empties the stack; a bare
// ROLLBACK / COMMIT empties the whole stack.

/// An OUTERMOST savepoint (no enclosing BEGIN) starts a transaction, and
/// releasing it commits durably: "When a SAVEPOINT is the outer-most savepoint
/// and it is not within a BEGIN...COMMIT then the behavior is the same as BEGIN
/// DEFERRED TRANSACTION" and "If a RELEASE command releases the outermost
/// savepoint ... then RELEASE is the same as COMMIT" (lang_savepoint §2). Proven
/// COMMITTED, not merely visible: a later independent BEGIN…ROLLBACK cannot undo
/// the released row, which is what a real COMMIT guarantees. (This in-memory
/// connection probe proves the write is outside any reversible transaction — a
/// proxy for durability; true cross-reopen on-disk durability is exercised by
/// the on-disk conformance suites, not here.)
#[test]
fn savepoint_outermost_release_commits_durably() {
    let mut db = new_t();
    exec(&mut db, "SAVEPOINT s"); // outermost == BEGIN DEFERRED
    exec(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "RELEASE s"); // outermost RELEASE == COMMIT
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(1));
    // Durability probe: an independent transaction's rollback leaves row 1 intact,
    // which only holds if RELEASE truly committed it.
    exec(&mut db, "BEGIN");
    exec(&mut db, "INSERT INTO t VALUES(2)");
    exec(&mut db, "ROLLBACK"); // undoes only row 2
    assert_rows_unordered(&mut db, "SELECT a FROM t", &[vec![int(1)]]);
}

/// After `ROLLBACK TO` an outermost savepoint the savepoint is NOT released and
/// the transaction stays OPEN: "unlike that plain ROLLBACK command ... the
/// ROLLBACK TO command does not cancel the transaction" (lang_savepoint §2). A
/// write made afterwards is therefore still uncommitted — proven because a bare
/// `ROLLBACK` (which needs an active transaction) succeeds and undoes it, after
/// which `COMMIT` errors (the stack is now empty / autocommit).
#[test]
fn savepoint_outermost_rollback_to_keeps_transaction_open_uncommitted() {
    let mut db = new_t();
    exec(&mut db, "SAVEPOINT s"); // outermost, transaction now open
    exec(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "ROLLBACK TO s"); // reverts to just-after-s; s remains; tx still open
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
    exec(&mut db, "INSERT INTO t VALUES(2)"); // still uncommitted (tx open)
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(1)); // row 2 really inserted (visible in the still-open tx), not silently dropped
    exec(&mut db, "ROLLBACK"); // bare ROLLBACK needs an open tx → undoes row 2
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
    assert_exec_error(&mut db, "COMMIT"); // stack empty → nothing to commit
}

/// RELEASE of an OUTER savepoint releases all INNER savepoints too: RELEASE
/// "causes all savepoints back to and including the most recent savepoint with a
/// matching name to be removed from the transaction stack" (lang_savepoint §2).
/// With an enclosing BEGIN the stack does not empty, so nothing commits yet, but
/// both `a` and the inner `b` are gone — a later `ROLLBACK TO` either one errors.
#[test]
fn release_outer_releases_inner_savepoints() {
    let mut db = new_t();
    exec(&mut db, "BEGIN");
    exec(&mut db, "SAVEPOINT a");
    exec(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "SAVEPOINT b");
    exec(&mut db, "INSERT INTO t VALUES(2)");
    exec(&mut db, "RELEASE a"); // removes b and a; work merged into the BEGIN tx
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2));
    assert_exec_error(&mut db, "ROLLBACK TO b"); // inner b was released
    assert_exec_error(&mut db, "ROLLBACK TO a"); // a was released
    exec(&mut db, "COMMIT"); // enclosing tx still open → commits both rows
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2));
}

/// `ROLLBACK TO` does NOT remove the savepoint, so it can be repeated: "The
/// SAVEPOINT with the matching name remains on the transaction stack" after a
/// ROLLBACK TO (lang_savepoint §3). Contrast RELEASE, which removes it. Here `s`
/// is rewound three times and then still releasable.
#[test]
fn rollback_to_repeatable_savepoint_remains() {
    let mut db = new_t();
    exec(&mut db, "SAVEPOINT s"); // mark: empty table
    exec(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "ROLLBACK TO s"); // → empty; s remains
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
    exec(&mut db, "INSERT INTO t VALUES(2)");
    exec(&mut db, "ROLLBACK TO s"); // → empty; s STILL remains
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
    exec(&mut db, "INSERT INTO t VALUES(3)");
    exec(&mut db, "ROLLBACK TO s"); // → empty; s STILL remains (third time)
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
    exec(&mut db, "RELEASE s"); // now removed → commit of the (empty) transaction
    assert_exec_error(&mut db, "COMMIT"); // stack empty afterwards
}

/// `ROLLBACK TO` cancels savepoints created AFTER the target but KEEPS the
/// target: "All intervening SAVEPOINTs are canceled" (lang_savepoint §2) while
/// "The SAVEPOINT with the matching name remains on the transaction stack"
/// (§3). After `ROLLBACK TO a`, `b` is gone (ROLLBACK TO/RELEASE b error) but `a`
/// is still releasable.
#[test]
fn rollback_to_cancels_inner_savepoints_keeps_target() {
    let mut db = new_t();
    exec(&mut db, "SAVEPOINT a"); // mark_a: empty table
    exec(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "SAVEPOINT b");
    exec(&mut db, "INSERT INTO t VALUES(2)");
    exec(&mut db, "ROLLBACK TO a"); // → empty; b canceled; a remains
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
    assert_exec_error(&mut db, "ROLLBACK TO b"); // b was canceled
    assert_exec_error(&mut db, "RELEASE b"); // b was canceled
    exec(&mut db, "RELEASE a"); // a survived → outermost release commits
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
}

/// Three-plus levels of nesting with selective rollback at different depths,
/// asserting exactly which rows survive at each step. Exercises §2 ("reverts ...
/// to what it was just after the corresponding SAVEPOINT"; "All intervening
/// SAVEPOINTs are canceled") and §3 ("the SAVEPOINT with the matching name
/// remains") across four marks s1..s4.
#[test]
fn deep_nested_savepoints_selective_rollback() {
    let mut db = new_t();
    exec(&mut db, "SAVEPOINT s1"); // mark_s1: {}
    exec(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "SAVEPOINT s2"); // mark_s2: {1}
    exec(&mut db, "INSERT INTO t VALUES(2)");
    exec(&mut db, "SAVEPOINT s3"); // mark_s3: {1,2}
    exec(&mut db, "INSERT INTO t VALUES(3)");
    exec(&mut db, "SAVEPOINT s4"); // mark_s4: {1,2,3}
    exec(&mut db, "INSERT INTO t VALUES(4)");
    assert_rows_unordered(
        &mut db,
        "SELECT a FROM t",
        &[vec![int(1)], vec![int(2)], vec![int(3)], vec![int(4)]],
    );
    exec(&mut db, "ROLLBACK TO s3"); // → {1,2}; s4 canceled
    assert_rows_unordered(&mut db, "SELECT a FROM t", &[vec![int(1)], vec![int(2)]]);
    exec(&mut db, "INSERT INTO t VALUES(5)"); // {1,2,5}
    exec(&mut db, "ROLLBACK TO s2"); // → {1}; s3 (and the 5) canceled
    assert_rows_unordered(&mut db, "SELECT a FROM t", &[vec![int(1)]]);
    exec(&mut db, "INSERT INTO t VALUES(6)"); // {1,6}
    exec(&mut db, "ROLLBACK TO s1"); // → {}; s2 canceled, row 1 undone
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
    // Only s1 remains; the deeper marks are gone.
    assert_exec_error(&mut db, "ROLLBACK TO s2");
    assert_exec_error(&mut db, "ROLLBACK TO s3");
    assert_exec_error(&mut db, "ROLLBACK TO s4");
    exec(&mut db, "INSERT INTO t VALUES(7)"); // {7}, still inside the s1 transaction
    exec(&mut db, "RELEASE s1"); // outermost release → commit {7}
    assert_rows_unordered(&mut db, "SELECT a FROM t", &[vec![int(7)]]);
}

/// A duplicate savepoint NAME (shadowing): `ROLLBACK TO` targets the MOST RECENT
/// savepoint with that name — "rolls back ... back to the most recent SAVEPOINT
/// with a matching name" (lang_savepoint §3), and names "need not be unique"
/// (§2). So rewinding to `s` reverts row 2 (added after the inner `s`) but keeps
/// row 1 (added before it).
#[test]
fn rollback_to_duplicate_name_targets_most_recent() {
    let mut db = new_t();
    exec(&mut db, "SAVEPOINT s"); // outer s, mark: {}
    exec(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "SAVEPOINT s"); // inner s (same name), mark: {1}
    exec(&mut db, "INSERT INTO t VALUES(2)");
    exec(&mut db, "ROLLBACK TO s"); // most-recent s → revert to {1}
    assert_rows_unordered(&mut db, "SELECT a FROM t", &[vec![int(1)]]);
    exec(&mut db, "COMMIT"); // release all savepoints, persist {1}
    assert_rows_unordered(&mut db, "SELECT a FROM t", &[vec![int(1)]]);
}

/// With a duplicate name, RELEASE removes only back to the most recent match and
/// leaves the PRIOR same-named savepoint intact: RELEASE "releases savepoints
/// backwards in time until it releases a savepoint with a matching
/// savepoint-name. Prior savepoints, even savepoints with matching
/// savepoint-names, are unchanged" (lang_savepoint §3). After releasing the
/// inner `s`, a `ROLLBACK TO s` now hits the OUTER `s` and rewinds to before
/// row 1.
#[test]
fn release_duplicate_name_leaves_prior_savepoint() {
    let mut db = new_t();
    exec(&mut db, "SAVEPOINT s"); // outer s, mark: {}
    exec(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "SAVEPOINT s"); // inner s, mark: {1}
    exec(&mut db, "INSERT INTO t VALUES(2)");
    exec(&mut db, "RELEASE s"); // removes inner s only; outer s remains; {1,2} merged
    assert_rows_unordered(&mut db, "SELECT a FROM t", &[vec![int(1)], vec![int(2)]]);
    exec(&mut db, "ROLLBACK TO s"); // now targets outer s → revert to {}
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
    exec(&mut db, "RELEASE s"); // outer s is outermost → commit (empty)
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
}

/// A bare `ROLLBACK` (no TO) cancels ALL savepoints and the whole transaction:
/// "The ROLLBACK command without a TO clause rolls backs all transactions and
/// leaves the transaction stack empty" (lang_savepoint §3). Afterwards every
/// savepoint name is invalid and there is nothing to COMMIT. (Uses a
/// savepoint-started transaction, i.e. no enclosing BEGIN.)
#[test]
fn plain_rollback_invalidates_all_savepoint_names() {
    let mut db = new_t();
    exec(&mut db, "SAVEPOINT a"); // outermost → starts the transaction
    exec(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "SAVEPOINT b");
    exec(&mut db, "INSERT INTO t VALUES(2)");
    exec(&mut db, "ROLLBACK"); // discards everything, empties the stack
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
    assert_exec_error(&mut db, "ROLLBACK TO a"); // name no longer on the stack
    assert_exec_error(&mut db, "ROLLBACK TO b");
    assert_exec_error(&mut db, "RELEASE a");
    assert_exec_error(&mut db, "RELEASE b");
    assert_exec_error(&mut db, "COMMIT"); // stack empty → nothing to commit
}

/// `COMMIT` releases ALL (including nested) savepoints and commits durably, even
/// when the transaction was started by SAVEPOINT: "The COMMIT command may be
/// used to release all savepoints and commit the transaction even if the
/// transaction was originally started by a SAVEPOINT command" (lang_savepoint
/// §2). After COMMIT the savepoint names are invalid and the rows are durable.
#[test]
fn commit_releases_all_nested_savepoints() {
    let mut db = new_t();
    exec(&mut db, "SAVEPOINT a"); // outermost → starts the transaction
    exec(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "SAVEPOINT b");
    exec(&mut db, "INSERT INTO t VALUES(2)");
    exec(&mut db, "COMMIT"); // releases a and b; commits {1,2}
    assert_rows_unordered(&mut db, "SELECT a FROM t", &[vec![int(1)], vec![int(2)]]);
    assert_exec_error(&mut db, "ROLLBACK TO a"); // names invalid after commit
    assert_exec_error(&mut db, "RELEASE b");
    // Durability probe: an independent transaction's rollback keeps {1,2}.
    exec(&mut db, "BEGIN");
    exec(&mut db, "INSERT INTO t VALUES(3)");
    exec(&mut db, "ROLLBACK");
    assert_rows_unordered(&mut db, "SELECT a FROM t", &[vec![int(1)], vec![int(2)]]);
}

/// `END` is a COMMIT alias (lang_transaction §2: "END TRANSACTION is an alias
/// for COMMIT"), so it likewise releases all savepoints and commits a
/// savepoint-started transaction (lang_savepoint §2). After END the savepoint
/// names are invalid.
#[test]
fn end_releases_savepoints() {
    let mut db = new_t();
    exec(&mut db, "SAVEPOINT a");
    exec(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "SAVEPOINT b");
    exec(&mut db, "INSERT INTO t VALUES(2)");
    exec(&mut db, "END"); // END == COMMIT → releases a and b, commits {1,2}
    assert_rows_unordered(&mut db, "SELECT a FROM t", &[vec![int(1)], vec![int(2)]]);
    assert_exec_error(&mut db, "RELEASE a"); // name invalid after commit
}

/// A mix of DML inside one savepoint all reverts together: `ROLLBACK TO` "reverts
/// the state of the database back to what it was just after the corresponding
/// SAVEPOINT" (lang_savepoint §2) — so an INSERT, an UPDATE, and a DELETE done
/// after the savepoint are ALL undone in one step.
#[test]
fn rollback_to_reverts_insert_update_delete_together() {
    let mut db = seeded_t(); // committed baseline {1,2}
    exec(&mut db, "SAVEPOINT s");
    exec(&mut db, "INSERT INTO t VALUES(3)"); // {1,2,3}
    exec(&mut db, "UPDATE t SET a = a + 10"); // {11,12,13}
    exec(&mut db, "DELETE FROM t WHERE a = 11"); // {12,13}
    assert_rows_unordered(&mut db, "SELECT a FROM t", &[vec![int(12)], vec![int(13)]]);
    exec(&mut db, "ROLLBACK TO s"); // all three DML reverted at once
    assert_rows_unordered(&mut db, "SELECT a FROM t", &[vec![int(1)], vec![int(2)]]);
    exec(&mut db, "RELEASE s"); // commit the (restored) state
    assert_rows_unordered(&mut db, "SELECT a FROM t", &[vec![int(1)], vec![int(2)]]);
}

/// `ROLLBACK TO` restores and then lets new work continue in the same
/// transaction: it "restarts the transaction again at the beginning" rather than
/// cancelling it (lang_savepoint §2). Inside an enclosing BEGIN: rewinding to `s`
/// drops row 1, a fresh row 2 is added, RELEASE merges it, and COMMIT persists
/// only row 2.
#[test]
fn rollback_to_then_new_work_within_begin() {
    let mut db = new_t();
    exec(&mut db, "BEGIN");
    exec(&mut db, "SAVEPOINT s"); // mark: {}
    exec(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "ROLLBACK TO s"); // drop row 1; s remains; tx open
    exec(&mut db, "INSERT INTO t VALUES(2)"); // new work after the rewind
    exec(&mut db, "RELEASE s"); // merge into the enclosing BEGIN tx
    assert_rows_unordered(&mut db, "SELECT a FROM t", &[vec![int(2)]]); // visible in tx
    exec(&mut db, "COMMIT"); // persist only row 2
    assert_rows_unordered(&mut db, "SELECT a FROM t", &[vec![int(2)]]);
}

/// Releasing the INNER savepoint keeps the OUTER one usable: RELEASE "merges a
/// named transaction into its parent" (lang_savepoint §2) and prior savepoints
/// "are unchanged" (§3). After `RELEASE b`, a `ROLLBACK TO a` still rewinds to
/// just after `a`, undoing the work of both savepoints.
#[test]
fn release_inner_then_rollback_to_outer() {
    let mut db = new_t();
    exec(&mut db, "BEGIN");
    exec(&mut db, "SAVEPOINT a"); // mark_a: {}
    exec(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "SAVEPOINT b");
    exec(&mut db, "INSERT INTO t VALUES(2)");
    exec(&mut db, "RELEASE b"); // b merged into a; a remains
    assert_rows_unordered(&mut db, "SELECT a FROM t", &[vec![int(1)], vec![int(2)]]);
    exec(&mut db, "ROLLBACK TO a"); // rewind to just-after-a → {}
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
    exec(&mut db, "COMMIT"); // commit the (emptied) enclosing transaction
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(0));
}
