//! Conformance battery: **FOREIGN KEY constraint enforcement** — the child-side
//! existence check (INSERT/UPDATE) and the parent-side `ON DELETE` / `ON UPDATE` actions,
//! all gated on the per-connection `PRAGMA foreign_keys` (SQLite's default is OFF).
//!
//! Every expectation is TRANSCRIBED FROM THE SPEC (`spec/sqlite-doc/foreignkeys.html`,
//! `spec/sqlite-doc/pragma.html`), never from what the engine currently returns — a
//! failing case is the intended signal that the engine diverges from the spec.
//!
//! Spec anchors:
//!   * foreignkeys.html §2 "Enabling Foreign Key Support": enforcement is "disabled by
//!     default ... so must be enabled separately for each database connection" via
//!     `PRAGMA foreign_keys = ON`; the GET form reports `0`/`1`.
//!   * foreignkeys.html §1: "Attempting to insert a row into the track table that does not
//!     correspond to any row in the artist table will fail", "as will attempting to delete
//!     a row from the artist table when there exist dependent rows". "if the foreign key
//!     column ... is NULL, then no corresponding entry in the ... table is required." (The
//!     spec's older CLI transcript prints the error lowercased; the modern engine emits it
//!     upper-cased — see the exact text noted below, which the assertions check.)
//!   * foreignkeys.html §1 terminology: "The parent key is the column or set of columns in
//!     the parent table that the foreign key constraint refers to. This is normally, but
//!     not always, the primary key of the parent table."
//!   * foreignkeys.html §4.1 (composite) / §6 (MATCH SIMPLE): "if any of the child key
//!     columns ... are NULL, then there is no requirement for a corresponding row in the
//!     parent table. ... All foreign key constraints in SQLite are handled as if MATCH
//!     SIMPLE were specified."
//!   * foreignkeys.html §4.3 "ON DELETE and ON UPDATE Actions": omitted action defaults to
//!     "NO ACTION"; RESTRICT prohibits deleting a parent key "when there exists one or
//!     more child keys mapped to it"; "CASCADE ... each row in the child table that was
//!     associated with the deleted parent row is also deleted"; "SET NULL ... the child
//!     key columns of all rows in the child table that mapped to the parent key are set to
//!     contain SQL NULL values."
//!   * pragma.html #pragma_foreign_key_list: "one row for each foreign key constraint
//!     created by a REFERENCES clause"; the column set is `id, seq, table, from, to,
//!     on_update, on_delete, match` (the `to` column is NULL when the FK references the
//!     parent's PRIMARY KEY without naming columns).
//!
//! The exact reject error text matches real sqlite: `FOREIGN KEY constraint failed`.
//!
//! DOCUMENTED FOLLOW-UPS — deliberately NOT covered here because the engine does not yet
//! implement them (a test would fail for a reason outside this battery's stable surface):
//!   * `WITHOUT ROWID` tables on an IMMEDIATE FK enforcement path fail CLOSED (a loud "not
//!     supported" error, never a silent pass or skip) — correct enforcement (a PK-index
//!     seek/scan) is a follow-up. Pinned here for a WR *parent* (the child-side check,
//!     `child_insert_referencing_without_rowid_parent_errors`) and a WR *child* on the
//!     immediate parent side (`parent_delete_with_without_rowid_child_errors`,
//!     `parent_update_with_without_rowid_child_errors`,
//!     `parent_delete_cascade_with_without_rowid_child_errors`).
//!   * DEFERRED constraint TIMING (`DEFERRABLE INITIALLY DEFERRED` / `PRAGMA
//!     defer_foreign_keys`, checked at COMMIT) is covered in its own battery,
//!     `conformance_deferred_fk.rs` — including the WR-child deferred recheck, which IS
//!     enforced (not fail-closed). This file exercises the immediate paths, plus two boundary
//!     cases: `deferred_parent_delete_with_without_rowid_child_is_deferred_not_rejected_early`
//!     (a deferred NO ACTION rejection correctly DEFERS to COMMIT instead of erroring early) and
//!     `deferred_parent_delete_cascade_with_without_rowid_child_still_errors` (a deferred FK with
//!     an explicit ON DELETE CASCADE *action* still errors IMMEDIATELY — only NO ACTION defers).
//!   * triggers firing on cascaded/SET-NULL child rewrites, and `changes()` counting of
//!     FK-action rows (SQLite's `sqlite3_changes()` excludes them).
//!   * `SET NULL` on a child that itself has generated columns (STORED recompute not wired).

mod conformance;

use conformance::*;

// ===========================================================================
// Gate / introspection — PRAGMA foreign_keys (default OFF) and foreign_key_list.
// ===========================================================================

#[test]
fn fk_off_by_default() {
    // foreignkeys.html §2: "the default setting for foreign key enforcement is OFF" —
    // `PRAGMA foreign_keys` on a fresh connection reports 0.
    let mut db = mem();
    assert_scalar(&mut db, "PRAGMA foreign_keys", int(0));
}

#[test]
fn fk_off_allows_orphan_child() {
    // foreignkeys.html §2: enforcement "must be enabled separately for each database
    // connection". With it OFF (the default), a child row may reference a non-existent
    // parent key — "while foreign key constraints are disabled, there is nothing to stop
    // the user from violating foreign key constraints" (§5). The FK is DECLARED but not
    // enforced.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE c(x INTEGER, y INTEGER REFERENCES p(id))");
    // No parent row with id=999 exists, yet this succeeds because the pragma is OFF.
    exec(&mut db, "INSERT INTO c(x, y) VALUES (1, 999)");
    assert_scalar(&mut db, "SELECT y FROM c", int(999));
}

#[test]
fn pragma_enable_reports_on() {
    // foreignkeys.html §2: after `PRAGMA foreign_keys = ON;`, `PRAGMA foreign_keys;`
    // reports 1.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    assert_scalar(&mut db, "PRAGMA foreign_keys", int(1));
}

#[test]
fn foreign_key_list_reports_fk() {
    // pragma.html #pragma_foreign_key_list: one row per FK column, columns
    // `id, seq, table, from, to, on_update, on_delete, match`. For a single-column
    // `y REFERENCES p(id)` with no explicit actions: id 0, seq 0, parent table `p`,
    // from `y`, to `id`, both actions the "NO ACTION" default (foreignkeys.html §4.3),
    // and match `NONE` (SQLite parses but ignores MATCH — §6).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE c(x INTEGER, y INTEGER REFERENCES p(id))");
    assert_columns(
        &mut db,
        "PRAGMA foreign_key_list(c)",
        &["id", "seq", "table", "from", "to", "on_update", "on_delete", "match"],
    );
    assert_rows(
        &mut db,
        "PRAGMA foreign_key_list(c)",
        &[vec![
            int(0),
            int(0),
            text("p"),
            text("y"),
            text("id"),
            text("NO ACTION"),
            text("NO ACTION"),
            text("NONE"),
        ]],
    );
}

#[test]
fn foreign_key_list_reports_to_null_for_bare_reference() {
    // pragma.html #pragma_foreign_key_list + foreignkeys.html §1/§3: a bare `REFERENCES p`
    // (no parent column list) targets the parent's PRIMARY KEY, so the pragma reports
    // `to` = NULL for that FK (there is no NAMED parent column). Everything else matches
    // the explicit-column form.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE c(x INTEGER, y INTEGER REFERENCES p)");
    assert_rows(
        &mut db,
        "PRAGMA foreign_key_list(c)",
        &[vec![
            int(0),
            int(0),
            text("p"),
            text("y"),
            null(),
            text("NO ACTION"),
            text("NO ACTION"),
            text("NONE"),
        ]],
    );
}

#[test]
fn foreign_key_list_reports_composite_seq_and_actions() {
    // pragma.html #pragma_foreign_key_list: a composite FK emits one row per child column,
    // `seq` its 0-based position, `from`/`to` the paired child/parent columns in order
    // (foreignkeys.html §4.1). `on_update`/`on_delete` carry the DECLARED action names
    // (foreignkeys.html §4.3 enumerates "CASCADE", "SET NULL", ...). The result column
    // order is `... to, on_update, on_delete, match`, so `ON DELETE CASCADE ON UPDATE SET
    // NULL` renders on_update='SET NULL' and on_delete='CASCADE'. (This is the REPORTING
    // surface; foreign_key_list echoes the declared action regardless of whether the
    // engine yet enforces it.)
    let mut db = mem();
    exec(&mut db, "CREATE TABLE p(a INTEGER, b INTEGER, PRIMARY KEY(a, b))");
    exec(
        &mut db,
        "CREATE TABLE c(a INTEGER, b INTEGER, \
         FOREIGN KEY(a, b) REFERENCES p(a, b) ON DELETE CASCADE ON UPDATE SET NULL)",
    );
    assert_rows(
        &mut db,
        "PRAGMA foreign_key_list(c)",
        &[
            vec![
                int(0),
                int(0),
                text("p"),
                text("a"),
                text("a"),
                text("SET NULL"),
                text("CASCADE"),
                text("NONE"),
            ],
            vec![
                int(0),
                int(1),
                text("p"),
                text("b"),
                text("b"),
                text("SET NULL"),
                text("CASCADE"),
                text("NONE"),
            ],
        ],
    );
}

// ===========================================================================
// Child INSERT — the referenced parent key must exist, unless a child key
// column is NULL (MATCH SIMPLE). Parent p(id INTEGER PRIMARY KEY),
// child c(x, y REFERENCES p(id)).
// ===========================================================================

#[test]
fn child_insert_existing_parent_ok() {
    // foreignkeys.html §1: a child row that maps to an existing parent row is accepted.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE c(x INTEGER, y INTEGER REFERENCES p(id))");
    exec(&mut db, "INSERT INTO p(id) VALUES (1)");
    exec(&mut db, "INSERT INTO c(x, y) VALUES (10, 1)");
    assert_rows(&mut db, "SELECT x, y FROM c", &[vec![int(10), int(1)]]);
}

#[test]
fn child_insert_missing_parent_rejected() {
    // foreignkeys.html §1: "Attempting to insert a row into the track table that does not
    // correspond to any row in the artist table will fail" -> `foreign key constraint
    // failed`.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE c(x INTEGER, y INTEGER REFERENCES p(id))");
    // No parent row with id=999.
    let e = assert_exec_error(&mut db, "INSERT INTO c(x, y) VALUES (10, 999)");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    // The rejected INSERT is atomic — the row never lands (foreignkeys.html §4.2: for an
    // immediate constraint, "the effects of the statement are reverted").
    assert_scalar(&mut db, "SELECT count(*) FROM c", int(0));
}

#[test]
fn child_insert_null_fk_skips_check() {
    // foreignkeys.html §1: "if the foreign key column for an entry ... is NULL, then no
    // corresponding entry in the ... table is required." A NULL child key is accepted even
    // though nothing in the parent matches.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE c(x INTEGER, y INTEGER REFERENCES p(id))");
    // Parent is empty; a NULL child key still inserts.
    exec(&mut db, "INSERT INTO c(x, y) VALUES (10, NULL)");
    assert_rows(&mut db, "SELECT x, y FROM c", &[vec![int(10), null()]]);
}

#[test]
fn child_insert_text_key_takes_integer_pk_parent_affinity() {
    // foreignkeys.html §3: the child key is compared "using the ... affinity ... of the
    // parent key column". The parent key is an INTEGER PRIMARY KEY, so the TEXT child value
    // takes NUMERIC/INTEGER affinity before the lookup: an integral value in text form
    // ('5.0', or the plain '5') losslessly converts to the integer 5 and MATCHES parent
    // row 5 — it is NOT a spurious FK violation. The value is still stored verbatim as
    // text in the TEXT child column (affinity governs comparison, not storage).
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE c(y TEXT REFERENCES p(id))");
    exec(&mut db, "INSERT INTO p(id) VALUES (5)");
    exec(&mut db, "INSERT INTO c(y) VALUES ('5.0')");
    exec(&mut db, "INSERT INTO c(y) VALUES ('5')");
    assert_rows_unordered(&mut db, "SELECT y FROM c", &[vec![text("5.0")], vec![text("5")]]);
}

#[test]
fn child_insert_fractional_text_key_has_no_integer_pk_parent() {
    // The flip side: a TEXT child value that does NOT losslessly convert to an integer
    // ('5.5' -> Real under NUMERIC affinity, 'abc' -> stays text) can equal no integer
    // rowid, so it names no parent row and the FK is violated (foreignkeys.html §1).
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE c(y TEXT REFERENCES p(id))");
    exec(&mut db, "INSERT INTO p(id) VALUES (5)");
    let e = assert_exec_error(&mut db, "INSERT INTO c(y) VALUES ('5.5')");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    let e = assert_exec_error(&mut db, "INSERT INTO c(y) VALUES ('abc')");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
}

// ===========================================================================
// WITHOUT ROWID parent — fail CLOSED. Correctly enforcing an FK whose PARENT is
// a WITHOUT ROWID table (its rows live in a PK-index b-tree, not a rowid table,
// so the child-side existence probe cannot scan it — foreignkeys.html §3,
// withoutrowid.html) is a documented follow-up. Until then the engine must NOT
// silently treat the constraint as satisfied: it fails closed with a loud error,
// so a VIOLATED such FK is surfaced rather than swallowed.
// ===========================================================================

#[test]
fn child_insert_referencing_without_rowid_parent_errors() {
    // With `PRAGMA foreign_keys = ON`, a child INSERT under an FK that references a
    // WITHOUT ROWID parent raises a loud "not supported" error rather than passing
    // silently. A silent success would report an unverified (and possibly violated)
    // FK as satisfied — the honest posture is fail-closed: refuse what we cannot yet
    // enforce. (Correct enforcement — a PK-index seek — is the follow-up.)
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    // The parent is WITHOUT ROWID; `exec` panics if the CREATE itself fails, so the
    // assertion below can only be satisfied by the FK check — no false pass on an
    // unrelated "unsupported" error.
    exec(&mut db, "CREATE TABLE p(k INTEGER PRIMARY KEY) WITHOUT ROWID");
    exec(&mut db, "CREATE TABLE c(x INTEGER REFERENCES p(k))");
    let e = assert_exec_error(&mut db, "INSERT INTO c(x) VALUES (1)");
    let msg = e.to_string();
    assert!(
        msg.contains("WITHOUT ROWID") && msg.contains("not supported"),
        "expected a loud 'WITHOUT ROWID ... not supported' FK error, got: {e}"
    );
    // Fail-closed is atomic: the rejected INSERT lands no row.
    assert_scalar(&mut db, "SELECT count(*) FROM c", int(0));
}

#[test]
fn child_insert_referencing_without_rowid_parent_ok_when_fk_off() {
    // Companion to the above: the error is specific to FK ENFORCEMENT, not a generic
    // "WITHOUT ROWID unsupported" failure. With the pragma OFF (SQLite's default) the
    // child-side check is skipped entirely (foreignkeys.html §2), so the very same
    // INSERT succeeds — proving the WR-parent + rowid-child setup is itself valid and
    // it is only the FK probe that fails closed.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE p(k INTEGER PRIMARY KEY) WITHOUT ROWID");
    exec(&mut db, "CREATE TABLE c(x INTEGER REFERENCES p(k))");
    exec(&mut db, "INSERT INTO c(x) VALUES (1)");
    assert_scalar(&mut db, "SELECT x FROM c", int(1));
}

// ===========================================================================
// WITHOUT ROWID child — fail CLOSED on the IMMEDIATE parent side. A WR child's
// rows live in a PK-index b-tree (not a rowid table), so the parent-side lookup
// that enforces ON DELETE / ON UPDATE cannot decode it. Correct enforcement
// (scanning — and, for CASCADE/SET NULL/SET DEFAULT, rewriting — the WR b-tree)
// is a follow-up; until then, rather than silently SKIP the action — which would
// let a parent DELETE/UPDATE orphan the WR child (fail OPEN, strictly worse than
// the WR-parent case above) — the engine fails closed with a loud "not supported"
// error. The DEFERRED path is the exception: it defers to the COMMIT-time recheck,
// which DOES scan a WR child (see `conformance_deferred_fk.rs`). The WR child's
// IMMEDIATE child-side FK is still enforced on INSERT. (withoutrowid.html,
// foreignkeys.html §4.2/§4.3.)
// ===========================================================================

#[test]
fn parent_delete_with_without_rowid_child_errors() {
    // A WR child holds a VALID reference (its child-side INSERT check passed), then the
    // parent is deleted. Real sqlite enforces the child's ON DELETE (default NO ACTION) and
    // REJECTS the delete. This engine cannot scan the WR child to find the dependent row, so
    // it fails CLOSED rather than silently drop the parent and orphan the child (fail OPEN).
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(k INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE c(id INTEGER PRIMARY KEY, x REFERENCES p(k)) WITHOUT ROWID");
    exec(&mut db, "INSERT INTO p VALUES (1)");
    // The child-side check passes here: the parent p is a rowid table, p.k=1 exists.
    exec(&mut db, "INSERT INTO c VALUES (10, 1)");
    let e = assert_exec_error(&mut db, "DELETE FROM p WHERE k = 1");
    let msg = e.to_string();
    assert!(
        msg.contains("WITHOUT ROWID") && msg.contains("not supported"),
        "expected a loud 'WITHOUT ROWID ... not supported' FK error, got: {e}"
    );
    // Fail-closed is atomic: the parent row is left in place (never silently orphaning c).
    assert_scalar(&mut db, "SELECT count(*) FROM p", int(1));
}

#[test]
fn parent_update_with_without_rowid_child_errors() {
    // The ON UPDATE analog: changing the referenced parent key must enforce the WR child's
    // ON UPDATE (default NO ACTION), which real sqlite REJECTS. Same fail-closed posture as
    // the DELETE case — the parent-side lookup routes through the same guard.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(k INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE c(id INTEGER PRIMARY KEY, x REFERENCES p(k)) WITHOUT ROWID");
    exec(&mut db, "INSERT INTO p VALUES (1)");
    exec(&mut db, "INSERT INTO c VALUES (10, 1)");
    let e = assert_exec_error(&mut db, "UPDATE p SET k = 2 WHERE k = 1");
    let msg = e.to_string();
    assert!(
        msg.contains("WITHOUT ROWID") && msg.contains("not supported"),
        "expected a loud 'WITHOUT ROWID ... not supported' FK error, got: {e}"
    );
    // Fail-closed is atomic: the parent key is unchanged.
    assert_rows(&mut db, "SELECT k FROM p", &[vec![int(1)]]);
}

#[test]
fn parent_delete_cascade_with_without_rowid_child_errors() {
    // Even a WRITE action (ON DELETE CASCADE) fails closed on a WR child: real sqlite would
    // cascade-delete the child, but this engine cannot rewrite the WR child's PK-index b-tree,
    // so it errors rather than silently leave the child un-cascaded (fail OPEN). This shows the
    // guard is uniform across actions, not only the default NO ACTION.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(k INTEGER PRIMARY KEY)");
    exec(
        &mut db,
        "CREATE TABLE c(id INTEGER PRIMARY KEY, x REFERENCES p(k) ON DELETE CASCADE) WITHOUT ROWID",
    );
    exec(&mut db, "INSERT INTO p VALUES (1)");
    exec(&mut db, "INSERT INTO c VALUES (10, 1)");
    let e = assert_exec_error(&mut db, "DELETE FROM p WHERE k = 1");
    let msg = e.to_string();
    assert!(
        msg.contains("WITHOUT ROWID") && msg.contains("not supported"),
        "expected a loud 'WITHOUT ROWID ... not supported' FK error, got: {e}"
    );
    assert_scalar(&mut db, "SELECT count(*) FROM p", int(1));
}

#[test]
fn deferred_parent_delete_with_without_rowid_child_is_deferred_not_rejected_early() {
    // Boundary of the immediate guard: when the FK is DEFERRABLE INITIALLY DEFERRED, the
    // parent-side NO ACTION rejection must DEFER — the immediate WR-child guard must NOT fire
    // early. The parent DELETE therefore SUCCEEDS mid-transaction (as real sqlite allows),
    // orphaning the WR child, and the COMMIT-time deferred recheck (which DOES scan a WR child)
    // catches it with the real `FOREIGN KEY constraint failed`, not a "not supported" error.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(k INTEGER PRIMARY KEY)");
    exec(
        &mut db,
        "CREATE TABLE c(id INTEGER PRIMARY KEY, \
         x REFERENCES p(k) DEFERRABLE INITIALLY DEFERRED) WITHOUT ROWID",
    );
    exec(&mut db, "INSERT INTO p VALUES (1)");
    exec(&mut db, "INSERT INTO c VALUES (10, 1)");
    exec(&mut db, "BEGIN");
    // Deferred NO ACTION: the guard defers, so this SUCCEEDS now (no early "not supported").
    exec(&mut db, "DELETE FROM p WHERE k = 1");
    // COMMIT rechecks the deferred FK (walking the WR child's b-tree) and finds the orphan.
    let e = assert_exec_error(&mut db, "COMMIT");
    assert!(
        e.to_string().contains("FOREIGN KEY constraint failed"),
        "expected the real deferred FK violation at COMMIT, got: {e}"
    );
    // The failed COMMIT leaves the transaction open; ROLLBACK restores the parent row.
    exec(&mut db, "ROLLBACK");
    assert_scalar(&mut db, "SELECT count(*) FROM p", int(1));
}

#[test]
fn deferred_parent_delete_cascade_with_without_rowid_child_still_errors() {
    // Pins the gate's NO-ACTION clause. Only a deferred *NO ACTION* rejection defers to COMMIT
    // (see the twin above); a DEFERRABLE INITIALLY DEFERRED FK carrying an explicit *action*
    // (ON DELETE CASCADE here) still fires IMMEDIATELY, so the WR-child guard must error at the
    // DELETE, not defer. Real sqlite applies referential ACTIONS at once regardless of deferral
    // (only the NO ACTION / RESTRICT *check* can defer); this engine cannot rewrite the WR
    // child's b-tree, so it fails closed immediately. Mutation: dropping the guard's
    // `matches!(NoAction)` clause (letting ANY deferred action defer) would let this DELETE
    // succeed mid-txn, turning this test red — so it pins the clause the boundary twin cannot.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(k INTEGER PRIMARY KEY)");
    exec(
        &mut db,
        "CREATE TABLE c(id INTEGER PRIMARY KEY, \
         x REFERENCES p(k) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED) WITHOUT ROWID",
    );
    exec(&mut db, "INSERT INTO p VALUES (1)");
    exec(&mut db, "INSERT INTO c VALUES (10, 1)");
    exec(&mut db, "BEGIN");
    // Deferrable, but ON DELETE CASCADE is an ACTION, not a NO ACTION check: it fires now.
    let e = assert_exec_error(&mut db, "DELETE FROM p WHERE k = 1");
    let msg = e.to_string();
    assert!(
        msg.contains("WITHOUT ROWID") && msg.contains("not supported"),
        "expected the immediate 'WITHOUT ROWID ... not supported' error (a deferred CASCADE \
         action still fires immediately), got: {e}"
    );
    // The statement aborted; nothing was deleted. Clean up the open transaction.
    exec(&mut db, "ROLLBACK");
    assert_scalar(&mut db, "SELECT count(*) FROM p", int(1));
}

// ===========================================================================
// Child UPDATE — the same existence rule applies to the post-update row.
// ===========================================================================

#[test]
fn child_update_to_missing_parent_rejected() {
    // foreignkeys.html §1: "Trying to modify the trackartist field ... does not work
    // either, since the new value ... Still does not correspond to any row in the artist
    // table." -> `foreign key constraint failed`.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE c(x INTEGER, y INTEGER REFERENCES p(id))");
    exec(&mut db, "INSERT INTO p(id) VALUES (1)");
    exec(&mut db, "INSERT INTO c(x, y) VALUES (10, 1)");
    let e = assert_exec_error(&mut db, "UPDATE c SET y = 999");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    // The rejected UPDATE leaves the row unchanged.
    assert_rows(&mut db, "SELECT x, y FROM c", &[vec![int(10), int(1)]]);
}

#[test]
fn child_update_to_null_ok() {
    // foreignkeys.html §1 (MATCH SIMPLE): updating the child key to NULL removes the
    // requirement for a matching parent row, so it is accepted.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE c(x INTEGER, y INTEGER REFERENCES p(id))");
    exec(&mut db, "INSERT INTO p(id) VALUES (1)");
    exec(&mut db, "INSERT INTO c(x, y) VALUES (10, 1)");
    exec(&mut db, "UPDATE c SET y = NULL");
    assert_rows(&mut db, "SELECT x, y FROM c", &[vec![int(10), null()]]);
}

#[test]
fn child_update_to_valid_parent_ok() {
    // foreignkeys.html §1: moving the child key to another EXISTING parent key is accepted.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE c(x INTEGER, y INTEGER REFERENCES p(id))");
    exec(&mut db, "INSERT INTO p(id) VALUES (1), (2)");
    exec(&mut db, "INSERT INTO c(x, y) VALUES (10, 1)");
    exec(&mut db, "UPDATE c SET y = 2");
    assert_rows(&mut db, "SELECT x, y FROM c", &[vec![int(10), int(2)]]);
}

// ===========================================================================
// Parent DELETE — the ON DELETE action decides what happens to dependents.
// ===========================================================================

#[test]
fn parent_delete_no_action_with_children_rejected() {
    // foreignkeys.html §4.3: the omitted ON DELETE action defaults to "NO ACTION". Under
    // it, deleting a parent row that still has dependent child rows fails (§1: "as will
    // attempting to delete a row from the artist table when there exist dependent rows").
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE c(x INTEGER, y INTEGER REFERENCES p(id))");
    exec(&mut db, "INSERT INTO p(id) VALUES (1)");
    exec(&mut db, "INSERT INTO c(x, y) VALUES (10, 1)");
    let e = assert_exec_error(&mut db, "DELETE FROM p WHERE id = 1");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    // The rejected DELETE leaves the parent row in place.
    assert_scalar(&mut db, "SELECT count(*) FROM p", int(1));
}

#[test]
fn parent_delete_no_action_no_children_ok() {
    // foreignkeys.html §1: "Once all the records that refer to a row in the artist table
    // have been deleted, it is possible to modify [or delete] the row." Delete the child
    // first, then the parent delete has no dependents and succeeds.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE c(x INTEGER, y INTEGER REFERENCES p(id))");
    exec(&mut db, "INSERT INTO p(id) VALUES (1)");
    exec(&mut db, "INSERT INTO c(x, y) VALUES (10, 1)");
    exec(&mut db, "DELETE FROM c WHERE y = 1");
    exec(&mut db, "DELETE FROM p WHERE id = 1");
    assert_scalar(&mut db, "SELECT count(*) FROM p", int(0));
}

#[test]
fn parent_delete_restrict_rejected() {
    // foreignkeys.html §4.3: "RESTRICT ... the application is prohibited from deleting ...
    // a parent key when there exists one or more child keys mapped to it." Rejects like
    // NO ACTION does for a single-statement DELETE.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(
        &mut db,
        "CREATE TABLE c(x INTEGER, y INTEGER REFERENCES p(id) ON DELETE RESTRICT)",
    );
    exec(&mut db, "INSERT INTO p(id) VALUES (1)");
    exec(&mut db, "INSERT INTO c(x, y) VALUES (10, 1)");
    let e = assert_exec_error(&mut db, "DELETE FROM p WHERE id = 1");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    assert_scalar(&mut db, "SELECT count(*) FROM p", int(1));
}

#[test]
fn parent_delete_cascade_removes_children() {
    // foreignkeys.html §4.3: "ON DELETE CASCADE ... each row in the child table that was
    // associated with the deleted parent row is also deleted." Deleting the parent removes
    // the dependent child row.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(
        &mut db,
        "CREATE TABLE c(x INTEGER, y INTEGER REFERENCES p(id) ON DELETE CASCADE)",
    );
    exec(&mut db, "INSERT INTO p(id) VALUES (1)");
    exec(&mut db, "INSERT INTO c(x, y) VALUES (10, 1)");
    exec(&mut db, "DELETE FROM p WHERE id = 1");
    // Both the parent and the cascaded child are gone.
    assert_scalar(&mut db, "SELECT count(*) FROM p", int(0));
    assert_rows(&mut db, "SELECT x, y FROM c", &[]);
}

#[test]
fn parent_delete_cascade_recursive() {
    // foreignkeys.html §4.3 + §6 ("foreign key actions are considered trigger programs" and
    // recurse): a CASCADE that deletes a child which is ITSELF a parent under CASCADE
    // cascades on to the grandchildren. p <- c(ON DELETE CASCADE) <- g(ON DELETE CASCADE):
    // deleting the p row removes the matching c row AND the g row that depended on it.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(
        &mut db,
        "CREATE TABLE c(id INTEGER PRIMARY KEY, pid INTEGER REFERENCES p(id) ON DELETE CASCADE)",
    );
    exec(
        &mut db,
        "CREATE TABLE g(id INTEGER PRIMARY KEY, cid INTEGER REFERENCES c(id) ON DELETE CASCADE)",
    );
    exec(&mut db, "INSERT INTO p(id) VALUES (1)");
    exec(&mut db, "INSERT INTO c(id, pid) VALUES (10, 1)");
    exec(&mut db, "INSERT INTO g(id, cid) VALUES (100, 10)");
    exec(&mut db, "DELETE FROM p WHERE id = 1");
    assert_scalar(&mut db, "SELECT count(*) FROM p", int(0));
    assert_scalar(&mut db, "SELECT count(*) FROM c", int(0));
    assert_scalar(&mut db, "SELECT count(*) FROM g", int(0));
}

#[test]
fn parent_delete_set_null_nulls_children() {
    // foreignkeys.html §4.3: "SET NULL ... the child key columns of all rows in the child
    // table that mapped to the parent key are set to contain SQL NULL values." The child
    // row SURVIVES; only its FK column becomes NULL.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(
        &mut db,
        "CREATE TABLE c(x INTEGER, y INTEGER REFERENCES p(id) ON DELETE SET NULL)",
    );
    exec(&mut db, "INSERT INTO p(id) VALUES (1)");
    exec(&mut db, "INSERT INTO c(x, y) VALUES (10, 1)");
    exec(&mut db, "DELETE FROM p WHERE id = 1");
    // The child row remains, with its FK column nulled.
    assert_rows(&mut db, "SELECT x, y FROM c", &[vec![int(10), null()]]);
}

// ===========================================================================
// Parent key resolution — `REFERENCES p` with no column list targets p's PK.
// ===========================================================================

#[test]
fn references_parent_pk_implicit() {
    // foreignkeys.html §1: the parent key is "normally ... the primary key of the parent
    // table"; §3 shorthand "Attaching a 'REFERENCES <parent-table>' clause ... creates a
    // foreign key constraint that maps the column to the primary key of <parent-table>."
    // So a bare `REFERENCES p` resolves against p's PRIMARY KEY: a missing key is rejected,
    // an existing one is accepted.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY, name TEXT)");
    exec(&mut db, "CREATE TABLE c(x INTEGER, y INTEGER REFERENCES p)");
    exec(&mut db, "INSERT INTO p(id, name) VALUES (1, 'a')");
    // Existing PK value accepted.
    exec(&mut db, "INSERT INTO c(x, y) VALUES (10, 1)");
    assert_rows(&mut db, "SELECT x, y FROM c", &[vec![int(10), int(1)]]);
    // Missing PK value rejected.
    let e = assert_exec_error(&mut db, "INSERT INTO c(x, y) VALUES (20, 999)");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    // The rejected INSERT is atomic — only the earlier valid row remains.
    assert_rows(&mut db, "SELECT x, y FROM c", &[vec![int(10), int(1)]]);
}

#[test]
fn references_table_level_pk_parent_implicit() {
    // A bare `REFERENCES p` targets p's PRIMARY KEY (foreignkeys.html §1/§3). When that PK is
    // declared as a single-column TABLE-level constraint whose type is NOT INTEGER, it is
    // neither a rowid alias nor a per-column PK flag, so the resolver must read the table's
    // declared PRIMARY KEY (`TableDef::primary_key`). A TEXT key is used deliberately: a
    // single-column table-level `PRIMARY KEY(id)` of INTEGER type IS a rowid alias
    // (lang_createtable.html §5) and would resolve via the alias fast-path instead, never
    // exercising this branch. A mapping child is accepted, a non-mapping one rejected — the FK
    // must be ENFORCED, never mis-reported as "which has no primary key".
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(code TEXT, name TEXT, PRIMARY KEY(code))");
    exec(&mut db, "CREATE TABLE c(x INTEGER, code TEXT REFERENCES p)");
    exec(&mut db, "INSERT INTO p(code, name) VALUES ('a1', 'alpha')");
    // Maps to an existing parent key -> accepted (no spurious "no primary key" error).
    exec(&mut db, "INSERT INTO c(x, code) VALUES (10, 'a1')");
    assert_rows(&mut db, "SELECT x, code FROM c", &[vec![int(10), text("a1")]]);
    // No parent row with code 'zzz' -> the FK is violated and the INSERT rejected.
    let e = assert_exec_error(&mut db, "INSERT INTO c(x, code) VALUES (11, 'zzz')");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    assert_scalar(&mut db, "SELECT count(*) FROM c", int(1));
}

#[test]
fn references_composite_pk_parent_implicit() {
    // A bare `REFERENCES p` to a parent whose PRIMARY KEY is COMPOSITE targets ALL its PK
    // columns in declaration order (foreignkeys.html §4.1) — the child key must be the same
    // arity. A composite PK is never a rowid alias and sets no per-column flag, so this is the
    // clearest exercise of the `TableDef::primary_key` resolver branch: a per-column-flag scan
    // sees an empty PK for `PRIMARY KEY(a, b)` and would wrongly reject every child.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(a INTEGER, b INTEGER, PRIMARY KEY(a, b))");
    exec(&mut db, "CREATE TABLE c(a INTEGER, b INTEGER, FOREIGN KEY(a, b) REFERENCES p)");
    exec(&mut db, "INSERT INTO p(a, b) VALUES (1, 2)");
    // (1, 2) maps to the parent tuple -> accepted.
    exec(&mut db, "INSERT INTO c(a, b) VALUES (1, 2)");
    assert_rows(&mut db, "SELECT a, b FROM c", &[vec![int(1), int(2)]]);
    // (1, 3) maps to no parent tuple -> rejected.
    let e = assert_exec_error(&mut db, "INSERT INTO c(a, b) VALUES (1, 3)");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    // MATCH SIMPLE: any NULL child key column skips the check (foreignkeys.html §4.1), so
    // (5, NULL) inserts even though no parent tuple begins with 5.
    exec(&mut db, "INSERT INTO c(a, b) VALUES (5, NULL)");
    assert_scalar(&mut db, "SELECT count(*) FROM c", int(2));
}

// ===========================================================================
// Composite foreign key — the whole tuple must match, but any NULL child key
// column skips the check (MATCH SIMPLE). p(a, b, PRIMARY KEY(a, b));
// c(a, b, FOREIGN KEY(a, b) REFERENCES p(a, b)).
// ===========================================================================

#[test]
fn composite_fk_all_match_ok() {
    // foreignkeys.html §4.1: "each entry in the song table is required to map to an entry
    // in the album table with the same combination" — a fully-matching composite tuple is
    // accepted.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(a INTEGER, b INTEGER, PRIMARY KEY(a, b))");
    exec(
        &mut db,
        "CREATE TABLE c(a INTEGER, b INTEGER, FOREIGN KEY(a, b) REFERENCES p(a, b))",
    );
    exec(&mut db, "INSERT INTO p(a, b) VALUES (1, 2)");
    exec(&mut db, "INSERT INTO c(a, b) VALUES (1, 2)");
    assert_rows(&mut db, "SELECT a, b FROM c", &[vec![int(1), int(2)]]);
}

#[test]
fn composite_fk_mismatch_rejected() {
    // foreignkeys.html §4.1: a composite child tuple with no matching parent tuple (here
    // (1,3) when only (1,2) exists) violates the constraint.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(a INTEGER, b INTEGER, PRIMARY KEY(a, b))");
    exec(
        &mut db,
        "CREATE TABLE c(a INTEGER, b INTEGER, FOREIGN KEY(a, b) REFERENCES p(a, b))",
    );
    exec(&mut db, "INSERT INTO p(a, b) VALUES (1, 2)");
    let e = assert_exec_error(&mut db, "INSERT INTO c(a, b) VALUES (1, 3)");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    // The rejected INSERT is atomic — no partial row lands.
    assert_scalar(&mut db, "SELECT count(*) FROM c", int(0));
}

#[test]
fn composite_fk_one_null_skips_check() {
    // foreignkeys.html §4.1 / §6 (MATCH SIMPLE): "if any of the child key columns ... are
    // NULL, then there is no requirement for a corresponding row in the parent table." So
    // (1, NULL) inserts even though no parent tuple begins with 1-then-NULL.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(a INTEGER, b INTEGER, PRIMARY KEY(a, b))");
    exec(
        &mut db,
        "CREATE TABLE c(a INTEGER, b INTEGER, FOREIGN KEY(a, b) REFERENCES p(a, b))",
    );
    exec(&mut db, "INSERT INTO p(a, b) VALUES (1, 2)");
    exec(&mut db, "INSERT INTO c(a, b) VALUES (1, NULL)");
    assert_rows(&mut db, "SELECT a, b FROM c", &[vec![int(1), null()]]);
}

// ===========================================================================
// Self-referential foreign key — the common NULL-root-then-children shape.
// ===========================================================================

#[test]
fn self_ref_null_root_then_children_ok() {
    // foreignkeys.html §1 (MATCH SIMPLE): a table may reference itself. The root row's
    // parent pointer is NULL (no parent required), and later rows point at an
    // already-inserted parent key, so every insert satisfies the constraint.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, parent INTEGER REFERENCES t(id))");
    // Root: NULL parent satisfies the FK outright.
    exec(&mut db, "INSERT INTO t(id, parent) VALUES (1, NULL)");
    // Children point at the existing root (id=1), then at each other.
    exec(&mut db, "INSERT INTO t(id, parent) VALUES (2, 1)");
    exec(&mut db, "INSERT INTO t(id, parent) VALUES (3, 2)");
    assert_rows(
        &mut db,
        "SELECT id, parent FROM t ORDER BY id",
        &[vec![int(1), null()], vec![int(2), int(1)], vec![int(3), int(2)]],
    );
    // A child pointing at a non-existent id is still rejected.
    let e = assert_exec_error(&mut db, "INSERT INTO t(id, parent) VALUES (4, 999)");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    // The rejected INSERT is atomic — the three valid rows are all that remain.
    assert_scalar(&mut db, "SELECT count(*) FROM t", int(3));
}

// ===========================================================================
// Parent UPDATE — the ON UPDATE action decides what happens to dependents when
// an UPDATE changes a REFERENCED parent key (foreignkeys.html §4.3). An UPDATE
// that leaves the referenced key unchanged fires nothing.
// ===========================================================================

#[test]
fn parent_update_no_action_with_children_rejected() {
    // foreignkeys.html §4.3: the omitted ON UPDATE action defaults to "NO ACTION". Under it,
    // changing a parent key that still has dependent child rows fails — the child would be
    // orphaned (§1: modifying a parent key referenced by a child is prohibited).
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE c(x INTEGER, y INTEGER REFERENCES p(id))");
    exec(&mut db, "INSERT INTO p(id) VALUES (1)");
    exec(&mut db, "INSERT INTO c(x, y) VALUES (10, 1)");
    let e = assert_exec_error(&mut db, "UPDATE p SET id = 2 WHERE id = 1");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    // The rejected UPDATE leaves the parent (and child) unchanged.
    assert_rows(&mut db, "SELECT id FROM p", &[vec![int(1)]]);
    assert_rows(&mut db, "SELECT x, y FROM c", &[vec![int(10), int(1)]]);
}

#[test]
fn parent_update_unreferenced_column_fires_nothing() {
    // foreignkeys.html §4.3: an ON UPDATE action fires only when the update modifies a
    // parent key column the child references. Updating a NON-referenced column of the parent
    // (here `name`) leaves the referenced key `id` intact, so even NO ACTION is satisfied.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY, name TEXT)");
    exec(&mut db, "CREATE TABLE c(x INTEGER, y INTEGER REFERENCES p(id))");
    exec(&mut db, "INSERT INTO p(id, name) VALUES (1, 'a')");
    exec(&mut db, "INSERT INTO c(x, y) VALUES (10, 1)");
    exec(&mut db, "UPDATE p SET name = 'b' WHERE id = 1");
    assert_rows(&mut db, "SELECT id, name FROM p", &[vec![int(1), text("b")]]);
    assert_rows(&mut db, "SELECT x, y FROM c", &[vec![int(10), int(1)]]);
}

#[test]
fn parent_update_no_children_ok() {
    // With no dependent rows, changing a parent key is unconstrained — NO ACTION has nothing
    // to reject (foreignkeys.html §1: once nothing refers to a parent key it may be modified).
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE c(x INTEGER, y INTEGER REFERENCES p(id))");
    exec(&mut db, "INSERT INTO p(id) VALUES (1)");
    exec(&mut db, "UPDATE p SET id = 2 WHERE id = 1");
    assert_rows(&mut db, "SELECT id FROM p", &[vec![int(2)]]);
}

#[test]
fn parent_update_restrict_rejected() {
    // foreignkeys.html §4.3: "RESTRICT ... the application is prohibited from [updating] ...
    // a parent key when there exists one or more child keys mapped to it." Rejects like NO
    // ACTION does for a single-statement UPDATE.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE c(x INTEGER, y INTEGER REFERENCES p(id) ON UPDATE RESTRICT)");
    exec(&mut db, "INSERT INTO p(id) VALUES (1)");
    exec(&mut db, "INSERT INTO c(x, y) VALUES (10, 1)");
    let e = assert_exec_error(&mut db, "UPDATE p SET id = 2 WHERE id = 1");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    assert_rows(&mut db, "SELECT id FROM p", &[vec![int(1)]]);
}

#[test]
fn parent_update_cascade_updates_children() {
    // foreignkeys.html §4.3: "ON UPDATE CASCADE ... each row in the child table that was
    // associated with the [updated] parent row is updated so that the child key columns hold
    // the new parent key values." Changing the parent key rewrites the dependent child key.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE c(x INTEGER, y INTEGER REFERENCES p(id) ON UPDATE CASCADE)");
    exec(&mut db, "INSERT INTO p(id) VALUES (1)");
    exec(&mut db, "INSERT INTO c(x, y) VALUES (10, 1)");
    exec(&mut db, "UPDATE p SET id = 2 WHERE id = 1");
    // The parent key moved and the child's FK followed it.
    assert_rows(&mut db, "SELECT id FROM p", &[vec![int(2)]]);
    assert_rows(&mut db, "SELECT x, y FROM c", &[vec![int(10), int(2)]]);
}

#[test]
fn parent_update_cascade_only_matching_children() {
    // ON UPDATE CASCADE rewrites ONLY the children mapped to the changed key; a child
    // referencing a DIFFERENT parent key is untouched (foreignkeys.html §4.3 — the action
    // applies to "each row ... that was associated with" the updated parent row).
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE c(x INTEGER, y INTEGER REFERENCES p(id) ON UPDATE CASCADE)");
    exec(&mut db, "INSERT INTO p(id) VALUES (1), (2)");
    exec(&mut db, "INSERT INTO c(x, y) VALUES (10, 1), (20, 2)");
    exec(&mut db, "UPDATE p SET id = 3 WHERE id = 1");
    // Only the child that referenced 1 moved to 3; the child on 2 is unchanged.
    assert_rows(
        &mut db,
        "SELECT x, y FROM c ORDER BY x",
        &[vec![int(10), int(3)], vec![int(20), int(2)]],
    );
}

#[test]
fn parent_update_cascade_recursive() {
    // foreignkeys.html §4.3 + §6 (FK actions "are considered trigger programs" and recurse):
    // an ON UPDATE CASCADE that rewrites a child which is ITSELF a referenced parent (its FK
    // column IS its own primary key) cascades on to the grandchildren.
    // p <- c(pid=PK, ON UPDATE CASCADE) <- g(cid, ON UPDATE CASCADE).
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(
        &mut db,
        "CREATE TABLE c(id INTEGER PRIMARY KEY REFERENCES p(id) ON UPDATE CASCADE)",
    );
    exec(
        &mut db,
        "CREATE TABLE g(gid INTEGER PRIMARY KEY, cid INTEGER REFERENCES c(id) ON UPDATE CASCADE)",
    );
    exec(&mut db, "INSERT INTO p(id) VALUES (1)");
    exec(&mut db, "INSERT INTO c(id) VALUES (1)");
    exec(&mut db, "INSERT INTO g(gid, cid) VALUES (100, 1)");
    exec(&mut db, "UPDATE p SET id = 5 WHERE id = 1");
    assert_rows(&mut db, "SELECT id FROM p", &[vec![int(5)]]);
    assert_rows(&mut db, "SELECT id FROM c", &[vec![int(5)]]);
    assert_rows(&mut db, "SELECT gid, cid FROM g", &[vec![int(100), int(5)]]);
}

#[test]
fn parent_update_set_null_nulls_children() {
    // foreignkeys.html §4.3: "ON UPDATE SET NULL ... the child key columns of all rows in the
    // child table that mapped to the [updated] parent key are set to contain SQL NULL." The
    // child survives detached; only its FK column becomes NULL.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE c(x INTEGER, y INTEGER REFERENCES p(id) ON UPDATE SET NULL)");
    exec(&mut db, "INSERT INTO p(id) VALUES (1)");
    exec(&mut db, "INSERT INTO c(x, y) VALUES (10, 1)");
    exec(&mut db, "UPDATE p SET id = 2 WHERE id = 1");
    // The child row remains, its FK column nulled.
    assert_rows(&mut db, "SELECT id FROM p", &[vec![int(2)]]);
    assert_rows(&mut db, "SELECT x, y FROM c", &[vec![int(10), null()]]);
}

#[test]
fn parent_update_cascade_composite_key() {
    // foreignkeys.html §4.1 + §4.3: a composite ON UPDATE CASCADE copies the WHOLE new
    // parent tuple into the matching child's key columns. Changing (1,2) -> (1,3) rewrites
    // the child (1,2) to (1,3).
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(a INTEGER, b INTEGER, PRIMARY KEY(a, b))");
    exec(
        &mut db,
        "CREATE TABLE c(a INTEGER, b INTEGER, \
         FOREIGN KEY(a, b) REFERENCES p(a, b) ON UPDATE CASCADE)",
    );
    exec(&mut db, "INSERT INTO p(a, b) VALUES (1, 2)");
    exec(&mut db, "INSERT INTO c(a, b) VALUES (1, 2)");
    exec(&mut db, "UPDATE p SET b = 3 WHERE a = 1 AND b = 2");
    assert_rows(&mut db, "SELECT a, b FROM p", &[vec![int(1), int(3)]]);
    assert_rows(&mut db, "SELECT a, b FROM c", &[vec![int(1), int(3)]]);
}

#[test]
fn parent_update_cascade_maintains_child_index() {
    // A CASCADE rewrite must MOVE the child's index entries, not just its record — a later
    // index-driven lookup on the child key must find the row at its NEW value and NOT at the
    // OLD one. (foreignkeys.html §4.3; index maintenance is required for a correct engine.)
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE c(x INTEGER, y INTEGER REFERENCES p(id) ON UPDATE CASCADE)");
    exec(&mut db, "CREATE INDEX ci ON c(y)");
    exec(&mut db, "INSERT INTO p(id) VALUES (1)");
    exec(&mut db, "INSERT INTO c(x, y) VALUES (10, 1)");
    exec(&mut db, "UPDATE p SET id = 2 WHERE id = 1");
    // The index now maps the NEW key; the OLD key finds nothing.
    assert_rows(&mut db, "SELECT x FROM c WHERE y = 2", &[vec![int(10)]]);
    assert_rows(&mut db, "SELECT x FROM c WHERE y = 1", &[]);
}

#[test]
fn parent_update_rowid_alias_cascade() {
    // The referenced parent key may be the INTEGER PRIMARY KEY (rowid). Moving the parent's
    // rowid must cascade to children referencing it (foreignkeys.html §3: a bare REFERENCES
    // targets the PK, here the rowid alias).
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE c(x INTEGER, y INTEGER REFERENCES p ON UPDATE CASCADE)");
    exec(&mut db, "INSERT INTO p(id) VALUES (1)");
    exec(&mut db, "INSERT INTO c(x, y) VALUES (10, 1)");
    exec(&mut db, "UPDATE p SET id = 7 WHERE id = 1");
    assert_rows(&mut db, "SELECT id FROM p", &[vec![int(7)]]);
    assert_rows(&mut db, "SELECT x, y FROM c", &[vec![int(10), int(7)]]);
}

// ===========================================================================
// SET DEFAULT — the child key columns are reset to their column DEFAULT, and
// the FK must still be satisfied afterwards (foreignkeys.html §4.3).
// ===========================================================================

#[test]
fn parent_delete_set_default_resets_children_to_valid_default() {
    // foreignkeys.html §4.3: "SET DEFAULT ... each of the child key columns is set to contain
    // the column's default value". The parent's DELETE resets the dependent child key to its
    // default; the default (0) references a surviving parent row, so the child stays valid.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(
        &mut db,
        "CREATE TABLE c(x INTEGER, y INTEGER DEFAULT 0 REFERENCES p(id) ON DELETE SET DEFAULT)",
    );
    exec(&mut db, "INSERT INTO p(id) VALUES (0), (1)");
    exec(&mut db, "INSERT INTO c(x, y) VALUES (10, 1)");
    exec(&mut db, "DELETE FROM p WHERE id = 1");
    // The child survives, its FK reset to the default 0 (which still exists in the parent).
    assert_rows(&mut db, "SELECT x, y FROM c", &[vec![int(10), int(0)]]);
    assert_rows(&mut db, "SELECT id FROM p", &[vec![int(0)]]);
}

#[test]
fn parent_delete_set_default_missing_default_parent_rejected() {
    // foreignkeys.html §4.3 (the spec's own DEFAULT-0 example): "if an 'ON DELETE SET
    // DEFAULT' action is configured, but there is no row in the parent table that corresponds
    // to the default values of the child key columns, deleting a parent key while dependent
    // child keys exist still causes a foreign key violation." Here parent 0 does NOT exist.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(
        &mut db,
        "CREATE TABLE c(x INTEGER, y INTEGER DEFAULT 0 REFERENCES p(id) ON DELETE SET DEFAULT)",
    );
    exec(&mut db, "INSERT INTO p(id) VALUES (1)");
    exec(&mut db, "INSERT INTO c(x, y) VALUES (10, 1)");
    let e = assert_exec_error(&mut db, "DELETE FROM p WHERE id = 1");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    // The rejected DELETE leaves parent and child unchanged.
    assert_rows(&mut db, "SELECT id FROM p", &[vec![int(1)]]);
    assert_rows(&mut db, "SELECT x, y FROM c", &[vec![int(10), int(1)]]);
}

#[test]
fn parent_delete_set_default_null_default_detaches_child() {
    // A child FK column with NO explicit DEFAULT has default NULL, so ON DELETE SET DEFAULT
    // sets it to NULL — which satisfies the FK by MATCH SIMPLE (foreignkeys.html §6), like
    // SET NULL. The child survives detached.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(
        &mut db,
        "CREATE TABLE c(x INTEGER, y INTEGER REFERENCES p(id) ON DELETE SET DEFAULT)",
    );
    exec(&mut db, "INSERT INTO p(id) VALUES (1)");
    exec(&mut db, "INSERT INTO c(x, y) VALUES (10, 1)");
    exec(&mut db, "DELETE FROM p WHERE id = 1");
    assert_rows(&mut db, "SELECT x, y FROM c", &[vec![int(10), null()]]);
}

#[test]
fn parent_update_set_default_resets_children_to_valid_default() {
    // foreignkeys.html §4.3: ON UPDATE SET DEFAULT resets the dependent child key to its
    // column default when the referenced parent key changes. The default (0) references a
    // surviving parent row, so the child stays valid.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(
        &mut db,
        "CREATE TABLE c(x INTEGER, y INTEGER DEFAULT 0 REFERENCES p(id) ON UPDATE SET DEFAULT)",
    );
    exec(&mut db, "INSERT INTO p(id) VALUES (0), (1)");
    exec(&mut db, "INSERT INTO c(x, y) VALUES (10, 1)");
    exec(&mut db, "UPDATE p SET id = 5 WHERE id = 1");
    // The parent key moved; the child was reset to the default 0 (which still exists).
    assert_rows_unordered(&mut db, "SELECT id FROM p", &[vec![int(0)], vec![int(5)]]);
    assert_rows(&mut db, "SELECT x, y FROM c", &[vec![int(10), int(0)]]);
}

#[test]
fn parent_update_set_default_missing_default_parent_rejected() {
    // The ON UPDATE analog of the spec's DEFAULT-0 violation: after the parent key moves,
    // the default 0 references no parent row, so the FK is violated (foreignkeys.html §4.3).
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE p(id INTEGER PRIMARY KEY)");
    exec(
        &mut db,
        "CREATE TABLE c(x INTEGER, y INTEGER DEFAULT 0 REFERENCES p(id) ON UPDATE SET DEFAULT)",
    );
    exec(&mut db, "INSERT INTO p(id) VALUES (1)");
    exec(&mut db, "INSERT INTO c(x, y) VALUES (10, 1)");
    let e = assert_exec_error(&mut db, "UPDATE p SET id = 5 WHERE id = 1");
    assert!(e.to_string().contains("FOREIGN KEY constraint failed"), "got: {e}");
    assert_rows(&mut db, "SELECT id FROM p", &[vec![int(1)]]);
    assert_rows(&mut db, "SELECT x, y FROM c", &[vec![int(10), int(1)]]);
}
