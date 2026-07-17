//! Conformance battery: **DML under namespace SHADOWING** — a temp/attached object with the
//! SAME name as a main object — exercised through the pinned `minisqlite::Connection` facade.
//!
//! These pin the correctness holes that appear when a DML statement resolves its target to one
//! namespace (main, via a `main.`/`temp.`/`aux.` qualifier or search order) but a consumer that
//! already KNOWS that namespace still reads the WRONG store's triggers / indexes / child tables /
//! `sqlite_sequence`. The fix threads the resolved `DbIndex` into each consumer so it uses the
//! `*_in(db)` catalog lookup instead of the namespace-blind bare form.
//!
//! Every expectation is TRANSCRIBED FROM THE SQLITE DOCS in `spec/sqlite-doc/` (and real
//! sqlite's documented behavior), never from what the engine currently returns — a failing case
//! is the intended signal the engine diverges.
//!
//! Spec sources (all under `spec/sqlite-doc/`):
//!   * `lang_naming.html` / name resolution: an unqualified object resolves temp, then main, then
//!     attached databases in attach order; a `schema.object` qualifier reaches exactly that store.
//!     So a temp/attached table SHADOWS a same-named main table.
//!   * `lang_createtrigger.html`: a trigger fires for DML on ITS table; a trigger belongs to the
//!     same schema as its table (a trigger cannot span databases). So `INSERT INTO main.t` fires
//!     main.t's triggers, never a same-named temp/attached t's.
//!   * `lang_createindex.html` / `datatype3.html`: a UNIQUE index rejects a duplicate key. An
//!     index lives in the same schema as its table, so a write to `main.t` is checked against and
//!     maintains main.t's indexes only.
//!   * `foreignkeys.html`: with `PRAGMA foreign_keys=ON`, a child row must reference an existing
//!     parent key; `ON DELETE/UPDATE CASCADE`/`SET NULL` act on the referencing children. A
//!     foreign key may NOT span databases, so enforcement + cascade stay within the write's own
//!     schema — a coincidentally-named table in another schema is irrelevant.
//!   * `autoinc.html`: AUTOINCREMENT tracks a high-water mark in that schema's own
//!     `sqlite_sequence`; a new rowid is one past `max(largest-ever-used, current table max)`.
//!   * `gencol.html`: a GENERATED column's value is computed by ITS table's generation
//!     expression — STORED values are materialized in the record, VIRTUAL values are computed
//!     on every read. The expression belongs to the table it is declared on, so a write/read
//!     of `main.t` uses main.t's own generation programs, never a same-named temp/attached t's.
//!
//! ## Two kinds of test here (verified by reverting each fix to its bare form)
//! * RED-GREEN DISCRIMINATORS — go red the moment the fix is reverted, so they isolate the bug:
//!   all the TRIGGER cases (plan-time `compile_triggers` picks the wrong / no trigger set), the
//!   three INDEX-maintenance cases (executor `build_index_plans` gets an empty index list from the
//!   temp shadow: INSERT and UPDATE skip the uniqueness probe so a duplicate wrongly succeeds, and
//!   DELETE removes the table row but not its index entry so the next indexed read rejects the
//!   dangling entry), the two SPACER-table cases (the FK CHILD-lookup and the AUTOINCREMENT
//!   high-water), which force the shadow onto a different root page so the bare cross-namespace read
//!   lands on the wrong (empty) b-tree, and the GENERATED-column
//!   VALUE-divergence cases (STORED insert/update, VIRTUAL read + index-scan). Generated
//!   programs are the CLEANEST discriminators of the family: they are pure in-memory
//!   computation applied to the row (there is no per-namespace pager or shared root-page number
//!   to accidentally realign them, the mechanism that masks the FK-parent / autoincrement
//!   cases), so a wrong-namespace program stores/computes a visibly wrong VALUE.
//!   The one GENERATED case that is NOT a discriminator is the mismatched-ARITY read
//!   (`generated_virtual_arity_mismatch_…`): `SELECT *` projects only main.t's own declared
//!   columns, so a wider shadow's extra program is never observable through the facade — it is
//!   kept as a regression guard alongside the FK-parent / autoincrement ones.
//! * CORRECTNESS / REGRESSION GUARDS — assert the correct end state but do NOT independently go
//!   red under the bare bug, because at the facade each namespace has its OWN pager: a
//!   wrong-namespace DEF resolved by the bare lookup is still opened on the WRITE's pager, and a
//!   same-position table in each store shares a root-page number, so the cross-namespace read
//!   lands on the correct store's bytes by coincidence and the other store is never observably
//!   mutated. This masks the FK PARENT-side cascade/set-null/update cases and the NO-SPACER
//!   AUTOINCREMENT case (its SPACER variant IS a discriminator — see above, exactly as the FK
//!   CHILD case has both a coincidence-masked guard and a spacer discriminator). Their fixes
//!   (`tables_in(db)`, `table_in(db, sqlite_sequence)`) are correct BY
//!   CONSTRUCTION (a cross-database FK / a shared `sqlite_sequence` is illegal — `foreignkeys.html`,
//!   `autoinc.html`), and these tests still guard against a total regression (a broken cascade, a
//!   panic under a shadow, the wrong final state). They are kept as regression guards, not weakened.
//!
//! Each behavior is its own small `#[test]` so one discrepancy fails exactly that case.

mod conformance;
use conformance::*;

use minisqlite::Connection;

/// `db.execute(sql)` must be `Err` and its message must contain `needle`. A bare `is_err()`
/// passes on ANY error — including the wrong one — so pinning a stable substring stops a
/// regression that fails for the wrong reason from passing.
#[track_caller]
fn assert_exec_error_contains(db: &mut Connection, sql: &str, needle: &str) {
    match db.execute(sql) {
        Ok(()) => panic!("expected an error containing {needle:?}, but `{sql}` succeeded"),
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains(needle),
                "error for `{sql}` was {msg:?}, expected to contain {needle:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Triggers: the DML target's OWN namespace's triggers fire (reproducers 1 & 2).
// ---------------------------------------------------------------------------

#[test]
fn main_after_insert_trigger_fires_on_qualified_insert_under_temp_shadow() {
    // Reproducer (1): a temp `t` shadows main's `t`, but `INSERT INTO main.t` must still fire
    // main.t's AFTER INSERT trigger. The bug read triggers via the search order (temp first), so
    // main's trigger silently never fired.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x)");
    exec(&mut db, "CREATE TABLE log(n)");
    exec(&mut db, "CREATE TRIGGER tr AFTER INSERT ON t BEGIN INSERT INTO log VALUES(NEW.x); END");
    exec(&mut db, "CREATE TEMP TABLE t(x)"); // temp.t shadows main.t; it has NO trigger

    exec(&mut db, "INSERT INTO main.t VALUES(42)");

    // main.t's trigger fired exactly once, seeing NEW.x = 42.
    assert_rows(&mut db, "SELECT n FROM log", &[vec![int(42)]]);
    // The row landed in main.t, and temp.t is still empty (the insert did not go there).
    assert_scalar(&mut db, "SELECT x FROM main.t", int(42));
    assert_scalar(&mut db, "SELECT count(*) FROM temp.t", int(0));

    // Prove temp.t genuinely has no trigger: an insert into it fires nothing (log unchanged).
    exec(&mut db, "INSERT INTO temp.t VALUES(99)");
    assert_rows(&mut db, "SELECT n FROM log", &[vec![int(42)]]);
    assert_scalar(&mut db, "SELECT count(*) FROM temp.t", int(1));
}

#[test]
fn attached_insert_does_not_fire_main_trigger() {
    // Reproducer (2): `INSERT INTO aux.t` must NOT fire main.t's trigger — the trigger belongs to
    // main, a different schema. The bug resolved the target's triggers by name via the search
    // order, found main's trigger, and (wrongly) fired it against the attached write.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x)");
    exec(&mut db, "CREATE TABLE log(n)");
    exec(&mut db, "CREATE TRIGGER tr AFTER INSERT ON t BEGIN INSERT INTO log VALUES(1); END");
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.t(x)"); // aux.t has NO trigger

    exec(&mut db, "INSERT INTO aux.t VALUES(5)");

    // main.t's trigger must not have fired: log is empty.
    assert_scalar(&mut db, "SELECT count(*) FROM log", int(0));
    // The row landed in aux.t, and main.t is untouched.
    assert_scalar(&mut db, "SELECT x FROM aux.t", int(5));
    assert_scalar(&mut db, "SELECT count(*) FROM main.t", int(0));
}

#[test]
fn main_after_delete_trigger_fires_under_temp_shadow() {
    // The DELETE analogue: `DELETE FROM main.t` under a temp `t` shadow fires main.t's AFTER
    // DELETE trigger (seeing OLD.x) and leaves the temp table untouched. Distinct data per
    // namespace (main 10, temp 20) proves which store answered.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x)");
    exec(&mut db, "CREATE TABLE log(n)");
    exec(&mut db, "CREATE TRIGGER trd AFTER DELETE ON t BEGIN INSERT INTO log VALUES(OLD.x); END");
    exec(&mut db, "INSERT INTO t VALUES(10)"); // main.t = {10} (no shadow yet -> main)
    exec(&mut db, "CREATE TEMP TABLE t(x)");
    exec(&mut db, "INSERT INTO temp.t VALUES(20)"); // temp.t = {20}

    exec(&mut db, "DELETE FROM main.t WHERE x = 10");

    // main.t's AFTER DELETE trigger fired with OLD.x = 10.
    assert_rows(&mut db, "SELECT n FROM log", &[vec![int(10)]]);
    // main.t's row is gone; temp.t is untouched.
    assert_scalar(&mut db, "SELECT count(*) FROM main.t", int(0));
    assert_scalar(&mut db, "SELECT x FROM temp.t", int(20));
}

#[test]
fn main_after_update_trigger_fires_under_temp_shadow() {
    // The UPDATE analogue: `UPDATE main.t` under a temp `t` shadow fires main.t's AFTER UPDATE
    // trigger (seeing NEW.x) and leaves the temp table untouched.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(x)");
    exec(&mut db, "CREATE TABLE log(n)");
    exec(&mut db, "CREATE TRIGGER tru AFTER UPDATE ON t BEGIN INSERT INTO log VALUES(NEW.x); END");
    exec(&mut db, "INSERT INTO t VALUES(10)"); // main.t = {10}
    exec(&mut db, "CREATE TEMP TABLE t(x)");
    exec(&mut db, "INSERT INTO temp.t VALUES(20)"); // temp.t = {20}

    exec(&mut db, "UPDATE main.t SET x = 11 WHERE x = 10");

    // main.t's AFTER UPDATE trigger fired with NEW.x = 11.
    assert_rows(&mut db, "SELECT n FROM log", &[vec![int(11)]]);
    // main.t updated; temp.t untouched.
    assert_scalar(&mut db, "SELECT x FROM main.t", int(11));
    assert_scalar(&mut db, "SELECT x FROM temp.t", int(20));
}

// ---------------------------------------------------------------------------
// Index maintenance: a write to main.t is checked against main.t's own indexes.
//
// The temp shadow deliberately has NO index. Bare `indexes_on("t")` resolves to the temp store
// (searched first), whose index list is EMPTY — so the buggy path performs NO uniqueness check
// and the duplicate write wrongly succeeds. The fix reads `indexes_on_in(main.db)` and enforces
// main.t's UNIQUE index. (This is deterministic: the buggy path is a clean no-op, not a
// cross-store read.)
// ---------------------------------------------------------------------------

#[test]
fn insert_unique_conflict_probes_target_namespace_index_under_temp_shadow() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "CREATE UNIQUE INDEX um ON t(a)"); // main.t's UNIQUE index
    exec(&mut db, "INSERT INTO t VALUES(1)"); // main.t = {1}
    exec(&mut db, "CREATE TEMP TABLE t(a)"); // temp.t shadows, NO index
    exec(&mut db, "INSERT INTO temp.t VALUES(1)"); // allowed: temp.t has no UNIQUE constraint

    // Re-inserting 1 into main.t must violate main.t's UNIQUE index.
    assert_exec_error_contains(&mut db, "INSERT INTO main.t VALUES(1)", "UNIQUE constraint failed");

    // main.t still holds exactly its one row; temp.t keeps its own independent row.
    assert_scalar(&mut db, "SELECT count(*) FROM main.t", int(1));
    assert_rows(&mut db, "SELECT a FROM temp.t", &[vec![int(1)]]);
}

#[test]
fn update_unique_conflict_probes_target_namespace_index_under_temp_shadow() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, b)");
    exec(&mut db, "CREATE UNIQUE INDEX um ON t(a)"); // main.t's UNIQUE index on a
    exec(&mut db, "INSERT INTO t VALUES(1, 'm1'), (2, 'm2')"); // main.t = {(1,m1),(2,m2)}
    exec(&mut db, "CREATE TEMP TABLE t(a, b)"); // temp.t shadows, NO index
    exec(&mut db, "INSERT INTO temp.t VALUES(1, 't1'), (2, 't2'), (3, 't3')");

    // Setting a=2 where a=1 would make two rows with a=2 — a UNIQUE violation on main.t.
    assert_exec_error_contains(
        &mut db,
        "UPDATE main.t SET a = 2 WHERE a = 1",
        "UNIQUE constraint failed",
    );

    // main.t is unchanged (the update was rejected); ordered by the NON-indexed column b so the
    // read reflects the table's real contents, not the index. temp.t is untouched.
    assert_rows(&mut db, "SELECT a FROM main.t ORDER BY b", &[vec![int(1)], vec![int(2)]]);
    assert_scalar(&mut db, "SELECT count(*) FROM temp.t", int(3));
}

#[test]
fn delete_index_maintenance_scoped_to_target_namespace_under_temp_shadow() {
    // The DELETE write path (delete.rs:132 -> `build_index_plans(node.db)` -> `indexes_on_in(db)`):
    // deleting a row from main.t must maintain main.t's OWN index — remove exactly the deleted
    // key's entry, keep the surviving keys' entries, and never touch a same-named temp shadow. This
    // is the DELETE analogue that rounds out the INSERT/UPDATE/DELETE index-maintenance
    // coverage; the shared `build_index_plans(db)` mechanism is the same one the INSERT/UPDATE cases
    // above discriminate. As with those, the temp shadow has NO index, so the bare `indexes_on("t")`
    // (temp searched first) returns an EMPTY list.
    //
    // RED-GREEN (verified by reverting `indexes_on_in(db)` -> `indexes_on`): under the bare bug the
    // DELETE removes the TABLE row for key 2 but skips index maintenance (empty temp index list), so
    // main.t's index keeps a DANGLING entry for the deleted rowid. The engine fails closed on the
    // very next indexed read (`Format("index entry points to missing table rowid 2")`), so the first
    // `ORDER BY a` assertion below goes red. The re-insert assertions additionally pin that the
    // DELETE removed exactly key 2's entry and left the rest of main.t's index intact and probeable.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)");
    exec(&mut db, "CREATE UNIQUE INDEX um ON t(a)"); // main.t's UNIQUE index (built before any shadow)
    exec(&mut db, "INSERT INTO t VALUES(1), (2), (3)"); // main.t = {1,2,3}, index {1,2,3}
    exec(&mut db, "CREATE TEMP TABLE t(a)"); // temp.t shadows, NO index
    exec(&mut db, "INSERT INTO temp.t VALUES(1), (2), (3)"); // temp.t independent, no UNIQUE

    exec(&mut db, "DELETE FROM main.t WHERE a = 2"); // must clear main.t's index entry for key 2

    // Full-scan truth: main.t now holds {1,3}.
    assert_rows(&mut db, "SELECT a FROM main.t ORDER BY a", &[vec![int(1)], vec![int(3)]]);
    // Indexed point lookup on the deleted key returns nothing (the index entry was removed, so no
    // stale entry surfaces a phantom row).
    assert_rows(&mut db, "SELECT a FROM main.t WHERE a = 2", &[]);
    // The freed key 2 can be re-inserted: its UNIQUE index entry is gone, so this passes the probe.
    exec(&mut db, "INSERT INTO main.t VALUES(2)");
    assert_rows(
        &mut db,
        "SELECT a FROM main.t ORDER BY a",
        &[vec![int(1)], vec![int(2)], vec![int(3)]],
    );
    // A SURVIVING key is still rejected as a duplicate — the DELETE removed only key 2's entry, it
    // did not wipe main.t's index. Under the bare bug the probe reads the temp shadow's empty index
    // and this duplicate INSERT wrongly succeeds (this is the discriminating assertion).
    assert_exec_error_contains(&mut db, "INSERT INTO main.t VALUES(1)", "UNIQUE constraint failed");

    // temp.t is entirely untouched by the main.t delete.
    assert_rows(
        &mut db,
        "SELECT a FROM temp.t ORDER BY a",
        &[vec![int(1)], vec![int(2)], vec![int(3)]],
    );
}

// ---------------------------------------------------------------------------
// Foreign keys: enforcement + cascade stay within the write's own schema.
// ---------------------------------------------------------------------------

#[test]
fn child_fk_parent_lookup_scoped_to_target_namespace_under_temp_shadow() {
    // The child side, common case (CORRECTNESS GUARD): `INSERT INTO main.child` looks its parent
    // up in main (`table_in(main)`) and sees main.parent's existing key, so the insert succeeds.
    // NB this simple shape does NOT go red under the bare bug: main.parent and the empty
    // temp.parent shadow share a root page (both are their store's first table), so the bare
    // cross-namespace read lands on main.parent's real bytes by coincidence. The discriminating
    // version is `child_fk_parent_lookup_isolated_by_spacer` below.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE parent(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE child(pid REFERENCES parent(id))");
    exec(&mut db, "INSERT INTO parent VALUES(1)"); // main.parent = {1}
    exec(&mut db, "CREATE TEMP TABLE parent(id INTEGER PRIMARY KEY)"); // temp.parent EMPTY, shadows

    // child.pid = 1 references main.parent(1), which exists -> the insert succeeds.
    exec(&mut db, "INSERT INTO main.child VALUES(1)");

    assert_rows(&mut db, "SELECT pid FROM main.child", &[vec![int(1)]]);
    // temp.parent is still empty (the lookup did not touch it).
    assert_scalar(&mut db, "SELECT count(*) FROM temp.parent", int(0));
}

#[test]
fn child_fk_parent_lookup_isolated_by_spacer() {
    // The child side, DISCRIMINATING version. Same bug as above (bare FK enforcement resolves the
    // parent by search order and finds the temp.parent shadow), but here a SPACER table forces the
    // shadow onto a DIFFERENT root page than the real parent, so the bare cross-namespace read no
    // longer coincidentally lands on the right data:
    //   * main store: `zzz_spacer` is main's first table (root page 2, left EMPTY), so `parent`
    //     is pushed to root page 3.
    //   * temp store: `parent` is temp's first table (root page 2).
    // Under the FIX, `table_in(main, "parent")` opens main.parent (root 3) = {1}, the key exists,
    // and the insert succeeds. Under the BARE bug, `table("parent")` resolves the temp shadow
    // (root 2); enforcement then opens root 2 on MAIN's pager — the empty `zzz_spacer` — finds no
    // key 1, and wrongly raises `FOREIGN KEY constraint failed`. Reading an EMPTY table makes the
    // "not found" verdict robust (no cross-schema record decode). This is exactly reproducer-style
    // proof that the parent lookup must be namespace-scoped.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE zzz_spacer(id INTEGER PRIMARY KEY)"); // main root 2, empty
    exec(&mut db, "CREATE TABLE parent(id INTEGER PRIMARY KEY)"); // main root 3
    exec(&mut db, "CREATE TABLE child(pid REFERENCES parent(id))"); // main root 4
    exec(&mut db, "INSERT INTO parent VALUES(1)"); // main.parent = {1}
    exec(&mut db, "CREATE TEMP TABLE parent(id INTEGER PRIMARY KEY)"); // temp root 2, EMPTY, shadows

    // References main.parent(1), which exists -> must succeed (bare wrongly rejects it).
    exec(&mut db, "INSERT INTO main.child VALUES(1)");

    assert_rows(&mut db, "SELECT pid FROM main.child", &[vec![int(1)]]);
    assert_scalar(&mut db, "SELECT count(*) FROM temp.parent", int(0));
}

#[test]
fn parent_delete_cascade_scoped_to_target_namespace_under_temp_shadow() {
    // The parent side (ON DELETE CASCADE): deleting main.parent cascades into main's child ONLY.
    // A coincidentally-named temp `child` (referencing a temp `parent`) must be untouched. The
    // bug enumerated children across ALL schemas (the `tables()` union) instead of just main's.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE parent(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE child(pid REFERENCES parent(id) ON DELETE CASCADE)");
    exec(&mut db, "INSERT INTO parent VALUES(1)");
    exec(&mut db, "INSERT INTO child VALUES(1)"); // main.child = {1}

    // A temp namespace with same-named parent+child, referencing each other, distinct data.
    exec(&mut db, "CREATE TEMP TABLE parent(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TEMP TABLE child(pid REFERENCES parent(id) ON DELETE CASCADE)");
    exec(&mut db, "INSERT INTO temp.parent VALUES(1), (2)");
    exec(&mut db, "INSERT INTO temp.child VALUES(2)"); // temp.child references temp.parent(2)

    exec(&mut db, "DELETE FROM main.parent WHERE id = 1");

    // main's parent + cascaded child are gone.
    assert_scalar(&mut db, "SELECT count(*) FROM main.parent", int(0));
    assert_scalar(&mut db, "SELECT count(*) FROM main.child", int(0));
    // temp's tables are completely untouched (temp.child still references temp.parent(2)).
    assert_rows(&mut db, "SELECT pid FROM temp.child", &[vec![int(2)]]);
    assert_rows_unordered(&mut db, "SELECT id FROM temp.parent", &[vec![int(1)], vec![int(2)]]);
}

#[test]
fn parent_delete_set_null_scoped_to_target_namespace_under_temp_shadow() {
    // The parent side (ON DELETE SET NULL): deleting main.parent nulls main.child's FK column
    // ONLY; a same-named temp child is untouched.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE parent(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE child(pid REFERENCES parent(id) ON DELETE SET NULL)");
    exec(&mut db, "INSERT INTO parent VALUES(1)");
    exec(&mut db, "INSERT INTO child VALUES(1)"); // main.child = {pid: 1}

    exec(&mut db, "CREATE TEMP TABLE parent(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TEMP TABLE child(pid REFERENCES parent(id) ON DELETE SET NULL)");
    exec(&mut db, "INSERT INTO temp.parent VALUES(1)");
    exec(&mut db, "INSERT INTO temp.child VALUES(1)"); // temp.child = {pid: 1}

    exec(&mut db, "DELETE FROM main.parent WHERE id = 1");

    // main.child's FK column was set to NULL (the row remains, pid is now NULL).
    assert_rows(&mut db, "SELECT pid FROM main.child", &[vec![null()]]);
    // temp.child is untouched: its pid is still 1.
    assert_rows(&mut db, "SELECT pid FROM temp.child", &[vec![int(1)]]);
}

#[test]
fn parent_update_cascade_scoped_to_target_namespace_under_temp_shadow() {
    // The parent side (ON UPDATE CASCADE): changing main.parent's key rewrites main.child's FK
    // column ONLY; a same-named temp child is untouched.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "CREATE TABLE parent(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE child(pid REFERENCES parent(id) ON UPDATE CASCADE)");
    exec(&mut db, "INSERT INTO parent VALUES(1)");
    exec(&mut db, "INSERT INTO child VALUES(1)");

    exec(&mut db, "CREATE TEMP TABLE parent(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TEMP TABLE child(pid REFERENCES parent(id) ON UPDATE CASCADE)");
    exec(&mut db, "INSERT INTO temp.parent VALUES(1)");
    exec(&mut db, "INSERT INTO temp.child VALUES(1)");

    exec(&mut db, "UPDATE main.parent SET id = 7 WHERE id = 1");

    // main.child's FK cascaded to the new key 7.
    assert_rows(&mut db, "SELECT pid FROM main.child", &[vec![int(7)]]);
    // temp.child is untouched: still 1.
    assert_rows(&mut db, "SELECT pid FROM temp.child", &[vec![int(1)]]);
}

#[test]
fn parent_delete_cascade_scoped_to_attached_namespace() {
    // The attached-namespace variant: deleting aux.parent cascades into aux's child only; a
    // same-named MAIN child is untouched. This proves the cascade uses the write's own schema
    // (`tables_in(aux)`), not the cross-schema union.
    let mut db = mem();
    exec(&mut db, "PRAGMA foreign_keys = ON");
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");

    // main: same-named parent/child, distinct data, must stay untouched.
    exec(&mut db, "CREATE TABLE parent(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE child(pid REFERENCES parent(id) ON DELETE CASCADE)");
    exec(&mut db, "INSERT INTO parent VALUES(1)");
    exec(&mut db, "INSERT INTO child VALUES(1)"); // main.child = {1}

    // aux: parent/child we will delete from.
    exec(&mut db, "CREATE TABLE aux.parent(id INTEGER PRIMARY KEY)");
    exec(&mut db, "CREATE TABLE aux.child(pid REFERENCES parent(id) ON DELETE CASCADE)");
    exec(&mut db, "INSERT INTO aux.parent VALUES(1)");
    exec(&mut db, "INSERT INTO aux.child VALUES(1)"); // aux.child = {1}

    exec(&mut db, "DELETE FROM aux.parent WHERE id = 1");

    // aux's parent + cascaded child are gone.
    assert_scalar(&mut db, "SELECT count(*) FROM aux.parent", int(0));
    assert_scalar(&mut db, "SELECT count(*) FROM aux.child", int(0));
    // main's same-named tables are untouched.
    assert_rows(&mut db, "SELECT id FROM main.parent", &[vec![int(1)]]);
    assert_rows(&mut db, "SELECT pid FROM main.child", &[vec![int(1)]]);
}

// ---------------------------------------------------------------------------
// AUTOINCREMENT: the high-water mark comes from the write's OWN schema's sqlite_sequence.
// ---------------------------------------------------------------------------

#[test]
fn autoincrement_high_water_scoped_to_target_namespace_under_temp_shadow() {
    // `sqlite_sequence` is per-schema (autoinc.html). main.t's high-water is 3 even after its
    // rows are deleted (AUTOINCREMENT never reuses), so the next main.t rowid is 4 — read from
    // MAIN's sqlite_sequence, not a same-named temp.t's. NB this simple shape is a CORRECTNESS
    // GUARD, not a discriminator: main and temp each get their sqlite_sequence at the same
    // root-page number, so the bare cross-namespace read lands on main's real seq bytes by
    // coincidence. The discriminating version is `autoincrement_high_water_discriminated_by_spacer`
    // below, which uses a spacer table to break that root-page coincidence.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY AUTOINCREMENT, v)");
    exec(&mut db, "INSERT INTO t(v) VALUES('a'), ('b'), ('c')"); // ids 1,2,3; high-water 3
    exec(&mut db, "DELETE FROM t"); // table now empty; high-water stays 3
    exec(&mut db, "CREATE TEMP TABLE t(id INTEGER PRIMARY KEY AUTOINCREMENT, v)"); // shadow
    exec(&mut db, "INSERT INTO temp.t(v) VALUES('x')"); // temp.t high-water 1

    exec(&mut db, "INSERT INTO main.t(v) VALUES('d')");

    // The new id is 4 (one past main's high-water 3), NOT 2 (one past temp's high-water 1).
    assert_rows(&mut db, "SELECT id, v FROM main.t", &[vec![int(4), text("d")]]);
    // temp.t kept its own independent sequence (its single row is id 1).
    assert_rows(&mut db, "SELECT id, v FROM temp.t", &[vec![int(1), text("x")]]);
}

#[test]
fn autoincrement_high_water_discriminated_by_spacer_under_temp_shadow() {
    // The DISCRIMINATING autoincrement case, using the same SPACER trick as the FK child test to
    // break the root-page coincidence that masks the guard above. `read_sequence`/`write_sequence`
    // read the write's OWN `sqlite_sequence` via `table_in(db, "sqlite_sequence")` (sequence.rs:64);
    // the bare `table("sqlite_sequence")` follows the temp->main->attached search order.
    //
    // The catalog creates `sqlite_sequence` EAGERLY at CREATE TABLE of an AUTOINCREMENT table
    // (sequence.rs module doc), so root pages fall out deterministically:
    //   * main store: `zzz_spacer` is created first (root page 2), then `t` (root 3), then main's
    //     `sqlite_sequence` (root 4).
    //   * temp store: `t` is created first (root 2), then temp's `sqlite_sequence` (root 3).
    // Under the FIX, `table_in(main, "sqlite_sequence")` opens main's seq (root 4) = {t: 3}, so the
    // next main.t id is 4. Under the BARE bug, `table("sqlite_sequence")` resolves TEMP's seq def
    // (root 3) but opens root 3 on MAIN's pager — which is main.t itself, now EMPTY after the
    // DELETE — finds no "t" row, returns None, and seeds from the (empty) table max 0, handing out
    // id 1. Reading an EMPTY b-tree makes the "not found" robust (no cross-schema record decode).
    let mut db = mem();
    // Spacer FIRST so main's sqlite_sequence lands on a different root page than temp's.
    exec(&mut db, "CREATE TABLE zzz_spacer(id INTEGER PRIMARY KEY)"); // main root 2 (plain PK, no seq)
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY AUTOINCREMENT, v)"); // main.t root 3; main seq root 4
    exec(&mut db, "INSERT INTO t(v) VALUES('a'), ('b'), ('c')"); // ids 1,2,3; main high-water 3
    exec(&mut db, "DELETE FROM t"); // main.t empty; high-water stays 3 (AUTOINCREMENT never reuses)
    exec(&mut db, "CREATE TEMP TABLE t(id INTEGER PRIMARY KEY AUTOINCREMENT, v)"); // temp.t root 2; temp seq root 3
    exec(&mut db, "INSERT INTO temp.t(v) VALUES('x')"); // temp high-water 1

    exec(&mut db, "INSERT INTO main.t(v) VALUES('d')");

    // Next main.t id is 4 (one past main's high-water 3), read from MAIN's sqlite_sequence.
    assert_rows(&mut db, "SELECT id, v FROM main.t", &[vec![int(4), text("d")]]);
    // temp.t kept its own independent sequence (its single row is id 1).
    assert_rows(&mut db, "SELECT id, v FROM temp.t", &[vec![int(1), text("x")]]);
}

// ---------------------------------------------------------------------------
// GENERATED columns: a write/read of main.t uses main.t's OWN generation programs.
//
// These are the sharpest discriminators of the whole family. The generation programs are
// bound once per statement and hung on `Plan::generated`; the read path (scan/index-scan)
// computes VIRTUAL columns and the write path (insert/update) computes every generated
// column. Before the fix the map was keyed by NAME only and populated by resolving the
// collected name with a BARE, search-order `catalog.table(name)` — so `INSERT/SELECT` on a
// `main.t` that a temp/attached `t` shadows bound the OTHER namespace's programs. Unlike the
// FK-parent / sqlite_sequence cases there is NO pager/root-page coincidence to mask it: the
// programs are pure computation over the in-memory row, so a wrong program yields a visibly
// wrong STORED value (write path) or computed VIRTUAL value (read path), or an out-of-range
// column when the two tables' arities differ. The fix carries the node's resolved `db` out of
// `collect_tables`, keys `TableGenerated` by `(db, name)`, resolves via `table_in(db, name)`,
// and every consumer looks up `generated_programs(node.db, name)`.
// ---------------------------------------------------------------------------

#[test]
fn generated_stored_insert_uses_target_namespace_program_under_temp_shadow() {
    // The primary witness (gencol.html): main.t's STORED `c` is `a + 1`; temp.t shadows it with
    // `c AS (a + 100)`. `INSERT INTO main.t(a) VALUES(5)` must materialize main.t's own c = 6.
    // Under the bug the name-keyed map bound temp.t's program, storing c = 105.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, c AS (a + 1) STORED)");
    exec(&mut db, "CREATE TEMP TABLE t(a, c AS (a + 100) STORED)"); // temp.t shadows main.t

    exec(&mut db, "INSERT INTO main.t(a) VALUES(5)");

    // main.t stored c = 5 + 1 = 6, computed by main.t's OWN program (not temp.t's a + 100 = 105).
    assert_rows(&mut db, "SELECT a, c FROM main.t", &[vec![int(5), int(6)]]);
    // temp.t is untouched and, when written itself, uses its own program (proves it really is
    // a + 100): inserting 5 there stores c = 105, so the two programs are genuinely different.
    exec(&mut db, "INSERT INTO temp.t(a) VALUES(5)");
    assert_rows(&mut db, "SELECT a, c FROM temp.t", &[vec![int(5), int(105)]]);
}

#[test]
fn generated_stored_insert_uses_target_namespace_program_attached() {
    // The attached variant. Search order is temp, then main, then attached, so a same-named
    // ATTACHED table does NOT shadow main by an unqualified name — but a QUALIFIED write does:
    // `INSERT INTO aux.t` targets aux.t, yet the bare `catalog.table("t")` resolves main FIRST,
    // so the bug bound MAIN's program to the aux write. main.t's c = a + 1, aux.t's c = a + 100.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, c AS (a + 1) STORED)"); // main.t
    exec(&mut db, "ATTACH DATABASE ':memory:' AS aux");
    exec(&mut db, "CREATE TABLE aux.t(a, c AS (a + 100) STORED)"); // aux.t, same name

    exec(&mut db, "INSERT INTO aux.t(a) VALUES(5)");

    // aux.t stored c = 5 + 100 = 105, its OWN program (not main.t's a + 1 = 6).
    assert_rows(&mut db, "SELECT a, c FROM aux.t", &[vec![int(5), int(105)]]);
    // main.t is untouched (empty).
    assert_scalar(&mut db, "SELECT count(*) FROM main.t", int(0));
}

#[test]
fn generated_stored_update_uses_target_namespace_program_under_temp_shadow() {
    // The UPDATE write path (update.rs): updating a base column RECOMPUTES the STORED generated
    // column from main.t's own program. main.t's c = a * 2; temp.t shadows with c = a * 100.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, c AS (a * 2) STORED)");
    exec(&mut db, "INSERT INTO t(a) VALUES(3)"); // main.t = {a:3, c:6} (no shadow yet -> main)
    exec(&mut db, "CREATE TEMP TABLE t(a, c AS (a * 100) STORED)"); // shadow
    exec(&mut db, "INSERT INTO temp.t(a) VALUES(3)"); // temp.t = {a:3, c:300}

    exec(&mut db, "UPDATE main.t SET a = 4 WHERE a = 3");

    // main.t recomputed c = 4 * 2 = 8 with its OWN program (not temp.t's 4 * 100 = 400).
    assert_rows(&mut db, "SELECT a, c FROM main.t", &[vec![int(4), int(8)]]);
    // temp.t is untouched (still a = 3, c = 300).
    assert_rows(&mut db, "SELECT a, c FROM temp.t", &[vec![int(3), int(300)]]);
}

#[test]
fn generated_virtual_read_uses_target_namespace_program_under_temp_shadow() {
    // The read path (scan.rs): a VIRTUAL column is computed on every read from main.t's own
    // program. main.t's v = a * 2; temp.t shadows with v = a * 100. Reading main.t must give
    // v = 10, not temp.t's 500. (VIRTUAL is never stored, so this isolates the on-read compute.)
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, v AS (a * 2) VIRTUAL)");
    exec(&mut db, "INSERT INTO t(a) VALUES(5)"); // main.t = {a:5} (v computed on read)
    exec(&mut db, "CREATE TEMP TABLE t(a, v AS (a * 100) VIRTUAL)"); // shadow

    // Reading main.t's VIRTUAL v computes 5 * 2 = 10 (main.t's program), not 5 * 100 = 500.
    assert_rows(&mut db, "SELECT a, v FROM main.t", &[vec![int(5), int(10)]]);
    // temp.t, read on its own, computes with its own program (proves they differ): empty here,
    // so insert a row and confirm 5 * 100 = 500.
    exec(&mut db, "INSERT INTO temp.t(a) VALUES(5)");
    assert_rows(&mut db, "SELECT a, v FROM temp.t", &[vec![int(5), int(500)]]);
}

#[test]
fn generated_virtual_indexscan_uses_target_namespace_program_under_temp_shadow() {
    // The index-scan read path (indexscan.rs): the same on-read VIRTUAL compute, but reached via
    // an index access path. main.t has an index on `a`, so a point lookup `WHERE a = 5` positions
    // on the index and fetches the row, then computes v from main.t's own program.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a, v AS (a * 2) VIRTUAL)");
    exec(&mut db, "CREATE INDEX mi ON t(a)"); // main.t's index -> drives an IndexScan path
    exec(&mut db, "INSERT INTO t(a) VALUES(5)"); // main.t = {a:5}
    exec(&mut db, "CREATE TEMP TABLE t(a, v AS (a * 100) VIRTUAL)"); // shadow, no index

    // The a = 5 lookup uses main.t's index; v is computed as 5 * 2 = 10 (main.t's program).
    assert_rows(&mut db, "SELECT a, v FROM main.t WHERE a = 5", &[vec![int(5), int(10)]]);
}

#[test]
fn generated_virtual_arity_mismatch_reads_only_target_columns_under_temp_shadow() {
    // The mismatched-ARITY witness (second reproducer), kept as a REGRESSION GUARD:
    // main.t has ONE column, temp.t shadows it with TWO (an extra VIRTUAL `v`). `SELECT * FROM
    // main.t` must yield exactly ONE column (5). NB this shape does NOT independently go red under
    // the bare bug (verified by reverting the fix): `SELECT *` expands to main.t's OWN declared
    // columns, so even when the buggy read binds temp.t's wider VIRTUAL program, the phantom
    // column is never projected and the result is still `[5]`. The value-divergence tests above
    // (STORED insert/update, VIRTUAL read/index-scan) are the true discriminators; this one guards
    // that a WIDER shadow does not crash the main read or leak an extra column, and pins the
    // documented correct shape. The fix resolves `table_in(main, "t")` = main.t (no generated
    // column), so the read takes the plain fast path with no program at all.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a)"); // main.t: ONE column, no generated column
    exec(&mut db, "INSERT INTO main.t VALUES(5)"); // main.t = {a:5}
    exec(&mut db, "CREATE TEMP TABLE t(a, v AS (a * 10) VIRTUAL)"); // shadow: TWO columns

    // `SELECT *` over main.t is exactly its one column; no phantom VIRTUAL column appears.
    assert_rows(&mut db, "SELECT * FROM main.t", &[vec![int(5)]]);
    assert_scalar(&mut db, "SELECT count(*) FROM main.t", int(1));
}
