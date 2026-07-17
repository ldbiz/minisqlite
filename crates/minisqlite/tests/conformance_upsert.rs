//! Conformance battery: the SQLite **UPSERT** clause on INSERT —
//! `INSERT ... ON CONFLICT (target) DO UPDATE SET ...` and
//! `INSERT ... ON CONFLICT DO NOTHING`.
//!
//! This is a DIFFERENT feature from the `INSERT OR IGNORE/REPLACE/ABORT/FAIL/ROLLBACK`
//! prefix forms (covered in `conformance_dml_insert.rs`); nothing here duplicates those.
//!
//! Every expected value is TRANSCRIBED FROM THE SPEC, `spec/sqlite-doc/lang_upsert.html`,
//! never from what the engine returns. Binding rules used below:
//!
//!   * §2: "UPSERT ... causes the INSERT to behave as an UPDATE or a no-op if the INSERT
//!     would violate a uniqueness constraint." The conflict target names the uniqueness
//!     constraint that triggers the upsert; if the insert would violate it, the insert is
//!     omitted and the DO NOTHING / DO UPDATE runs instead.
//!   * §2: "Column names in the expressions of a DO UPDATE refer to the original unchanged
//!     value of the column, before the attempted INSERT. To use the value that would have
//!     been inserted had the constraint not failed, add the special 'excluded.' table
//!     qualifier." So a BARE column = the pre-existing row's value; `excluded.col` = the
//!     value that the (omitted) INSERT would have written.
//!   * §2: "In the case of a multi-row insert, the upsert decision is made separately for
//!     each row of the insert."
//!   * §2: "The UPSERT processing happens only for uniqueness constraints. A 'uniqueness
//!     constraint' is an explicit UNIQUE or PRIMARY KEY constraint ... or a unique index."
//!     (This file therefore pins PRIMARY KEY, UNIQUE-column, and unique-index targets, and
//!     deliberately does NOT probe NOT NULL / CHECK interaction, which the
//!     engine does not enforce yet and which would muddy this file's signal.)
//!   * §2.1: the vocabulary (`DO UPDATE SET count=count+1`, also `vocabulary.count+1`),
//!     phonebook (`excluded.phonenumber`), and phonebook2 (`WHERE
//!     excluded.validDate>phonebook2.validDate`) worked examples.
//!   * §3: the DO UPDATE conflict-resolution algorithm is always ABORT ("DO UPDATE OR
//!     ABORT"); not exercised here (no in-update constraint violation is provoked).
//!   * §4 (History): a conflict target may be OMITTED on the last ON CONFLICT clause since
//!     SQLite 3.35.0 (for DO UPDATE as well as DO NOTHING).
//!
//! UPSERT is implemented: `compile_insert_with_parent`
//! (`crates/minisqlite-plan/src/compile/insert.rs`) lowers the `ON CONFLICT` clauses and
//! `bind_upsert_assignments` binds the DO UPDATE `SET` list (against the combined
//! `existing ++ excluded` row), and the executor applies the conflict decision per row.
//! Each test asserts the spec-correct post-upsert STATE (or the RETURNING result). A
//! spec-correct assertion is never weakened to make it pass, and none of these are
//! `#[ignore]`d.

mod conformance;
use conformance::*;

// ---------------------------------------------------------------------------
// §2.1 — DO UPDATE reading the EXISTING row value (the `vocabulary` example).
// ---------------------------------------------------------------------------

/// §2.1: `count=count+1` where the bare `count` is the ORIGINAL row value. Running the
/// same upsert twice inserts ('jovial', 1) then increments to ('jovial', 2).
#[test]
fn upsert_do_update_increments_existing_count() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE vocabulary(word TEXT PRIMARY KEY, count INT DEFAULT 1)");
    exec(&mut db, "INSERT INTO vocabulary(word) VALUES('jovial') ON CONFLICT(word) DO UPDATE SET count=count+1");
    exec(&mut db, "INSERT INTO vocabulary(word) VALUES('jovial') ON CONFLICT(word) DO UPDATE SET count=count+1");
    // 1st insert: no conflict -> plain insert, count takes its DEFAULT 1.
    // 2nd insert: 'jovial' conflicts -> count = (original 1) + 1 = 2.
    assert_rows(&mut db, "SELECT word, count FROM vocabulary", &[vec![text("jovial"), int(2)]]);
}

/// §2.1: "The 'count+1' expression could also be written as 'vocabulary.count'. PostgreSQL
/// requires the second form, but SQLite accepts either." The table-qualified bare reference
/// still names the ORIGINAL value, so the result is identical to the unqualified form.
#[test]
fn upsert_do_update_table_qualified_count() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE vocabulary(word TEXT PRIMARY KEY, count INT DEFAULT 1)");
    exec(&mut db, "INSERT INTO vocabulary(word) VALUES('jovial') ON CONFLICT(word) DO UPDATE SET count=vocabulary.count+1");
    exec(&mut db, "INSERT INTO vocabulary(word) VALUES('jovial') ON CONFLICT(word) DO UPDATE SET count=vocabulary.count+1");
    assert_rows(&mut db, "SELECT word, count FROM vocabulary", &[vec![text("jovial"), int(2)]]);
}

/// §2: when the INSERT does NOT violate the conflict target, the DO UPDATE does not fire —
/// the row is inserted normally. Running the vocabulary upsert ONCE on an empty table leaves
/// ('jovial', 1) (the DEFAULT), not an incremented count.
#[test]
fn upsert_no_conflict_is_plain_insert() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE vocabulary(word TEXT PRIMARY KEY, count INT DEFAULT 1)");
    exec(&mut db, "INSERT INTO vocabulary(word) VALUES('jovial') ON CONFLICT(word) DO UPDATE SET count=count+1");
    assert_rows(&mut db, "SELECT word, count FROM vocabulary", &[vec![text("jovial"), int(1)]]);
}

/// §2: a BARE column reference in DO UPDATE is the ORIGINAL row value, independent of the
/// value the conflicting INSERT carries. With original n=10 and an incoming (excluded) n=99,
/// `SET n=n+1` yields 11 (10+1), never 100 — isolating the bare=original rule with the
/// excluded value as a deliberate distractor (the vocabulary tests above cannot discriminate
/// this: there the omitted `count` defaults to 1, so bare and excluded coincide at 1).
#[test]
fn upsert_do_update_bare_is_original_not_excluded() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, n INT)");
    exec(&mut db, "INSERT INTO t VALUES(1, 10)");
    exec(&mut db, "INSERT INTO t VALUES(1, 99) ON CONFLICT(id) DO UPDATE SET n=n+1");
    assert_rows(&mut db, "SELECT id, n FROM t", &[vec![int(1), int(11)]]);
}

// ---------------------------------------------------------------------------
// §2 / §2.1 — DO UPDATE with the `excluded.` qualifier (the `phonebook` example).
// ---------------------------------------------------------------------------

/// §2.1: `excluded.phonenumber` is the value the omitted INSERT would have written, so on a
/// name conflict Alice's number is overwritten with the new one ('999').
#[test]
fn upsert_do_update_excluded_overwrites() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE phonebook(name TEXT PRIMARY KEY, phonenumber TEXT)");
    exec(&mut db, "INSERT INTO phonebook VALUES('Alice','111')");
    exec(&mut db, "INSERT INTO phonebook(name,phonenumber) VALUES('Alice','999') ON CONFLICT(name) DO UPDATE SET phonenumber=excluded.phonenumber");
    assert_rows(&mut db, "SELECT name, phonenumber FROM phonebook", &[vec![text("Alice"), text("999")]]);
}

/// §2.1: "the DO UPDATE clause acts only on the single row that experienced the constraint
/// error." Upserting Bob leaves Alice and Carol untouched and updates only Bob.
#[test]
fn upsert_do_update_affects_only_conflicting_row() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE phonebook(name TEXT PRIMARY KEY, phonenumber TEXT)");
    exec(&mut db, "INSERT INTO phonebook VALUES('Alice','111'),('Bob','222'),('Carol','333')");
    exec(&mut db, "INSERT INTO phonebook(name,phonenumber) VALUES('Bob','999') ON CONFLICT(name) DO UPDATE SET phonenumber=excluded.phonenumber");
    assert_rows(
        &mut db,
        "SELECT name, phonenumber FROM phonebook ORDER BY name",
        &[
            vec![text("Alice"), text("111")],
            vec![text("Bob"), text("999")],
            vec![text("Carol"), text("333")],
        ],
    );
}

/// §2: "causes the INSERT to behave as an UPDATE" — a real in-place UPDATE, so a data column
/// the SET does not mention keeps its ORIGINAL value. `w` is untouched by `SET v=excluded.v`,
/// so it stays 'keep' (not the excluded 'ignored', not NULL) — proving DO UPDATE modifies the
/// existing row rather than rebuilding a fresh row from the excluded values.
#[test]
fn upsert_do_update_preserves_unset_bystander_column() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, u TEXT UNIQUE, w TEXT, v TEXT)");
    exec(&mut db, "INSERT INTO t VALUES(1, 'x', 'keep', 'orig')");
    exec(&mut db, "INSERT INTO t VALUES(2, 'x', 'ignored', 'new') ON CONFLICT(u) DO UPDATE SET v=excluded.v");
    assert_rows(
        &mut db,
        "SELECT id, u, w, v FROM t ORDER BY id",
        &[vec![int(1), text("x"), text("keep"), text("new")]],
    );
}

/// §2 (bare = original) AND §2 (excluded = new value) combined in ONE expression: `SET count
/// = count + excluded.count`. Original 5 plus would-be-inserted 3 = 8. Derived from §2 (each
/// qualifier is defined separately; composing them in one expression is sound).
#[test]
fn upsert_do_update_mixes_original_and_excluded() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k TEXT PRIMARY KEY, count INT)");
    exec(&mut db, "INSERT INTO t VALUES('a', 5)");
    exec(&mut db, "INSERT INTO t VALUES('a', 3) ON CONFLICT(k) DO UPDATE SET count = count + excluded.count");
    assert_rows(&mut db, "SELECT k, count FROM t", &[vec![text("a"), int(8)]]);
}

// ---------------------------------------------------------------------------
// §2.1 — DO UPDATE ... WHERE gates the update into an optional no-op.
// ---------------------------------------------------------------------------

/// §2.1 (phonebook2): with `WHERE excluded.validDate>phonebook2.validDate`, a NEWER incoming
/// validDate satisfies the WHERE, so the update applies (phonenumber and validDate replaced).
#[test]
fn upsert_do_update_where_newer_date_applies() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE phonebook2(name TEXT PRIMARY KEY, phonenumber TEXT, validDate DATE)");
    exec(&mut db, "INSERT INTO phonebook2 VALUES('Alice','111','2018-01-01')");
    exec(&mut db, "INSERT INTO phonebook2(name,phonenumber,validDate) VALUES('Alice','999','2018-05-08') ON CONFLICT(name) DO UPDATE SET phonenumber=excluded.phonenumber, validDate=excluded.validDate WHERE excluded.validDate>phonebook2.validDate");
    assert_rows(
        &mut db,
        "SELECT name, phonenumber, validDate FROM phonebook2",
        &[vec![text("Alice"), text("999"), text("2018-05-08")]],
    );
}

/// §2.1 (phonebook2): "If the table already contains an entry with the same name and a
/// current validDate, then the WHERE clause causes the DO UPDATE to become a no-op." An
/// OLDER incoming validDate fails the WHERE, so the existing row is unchanged, no error, and
/// nothing new is inserted.
#[test]
fn upsert_do_update_where_older_date_is_noop() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE phonebook2(name TEXT PRIMARY KEY, phonenumber TEXT, validDate DATE)");
    exec(&mut db, "INSERT INTO phonebook2 VALUES('Alice','111','2018-05-08')");
    exec(&mut db, "INSERT INTO phonebook2(name,phonenumber,validDate) VALUES('Alice','999','2018-01-01') ON CONFLICT(name) DO UPDATE SET phonenumber=excluded.phonenumber, validDate=excluded.validDate WHERE excluded.validDate>phonebook2.validDate");
    assert_rows(
        &mut db,
        "SELECT name, phonenumber, validDate FROM phonebook2",
        &[vec![text("Alice"), text("111"), text("2018-05-08")]],
    );
}

/// §2.1: a trivially-TRUE `WHERE 1=1` gates the DO UPDATE ON, isolating the WHERE mechanism
/// from any date-comparison detail — the update applies.
#[test]
fn upsert_do_update_where_condition_true_applies() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)");
    exec(&mut db, "INSERT INTO t VALUES(1,'a')");
    exec(&mut db, "INSERT INTO t VALUES(1,'b') ON CONFLICT(id) DO UPDATE SET v=excluded.v WHERE 1=1");
    assert_rows(&mut db, "SELECT id, v FROM t", &[vec![int(1), text("b")]]);
}

/// §2.1: a trivially-FALSE `WHERE 1=0` makes the DO UPDATE a no-op — the existing row is
/// unchanged, no error, nothing inserted.
#[test]
fn upsert_do_update_where_condition_false_is_noop() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)");
    exec(&mut db, "INSERT INTO t VALUES(1,'a')");
    exec(&mut db, "INSERT INTO t VALUES(1,'b') ON CONFLICT(id) DO UPDATE SET v=excluded.v WHERE 1=0");
    assert_rows(&mut db, "SELECT id, v FROM t", &[vec![int(1), text("a")]]);
}

// ---------------------------------------------------------------------------
// §2 — DO NOTHING skips the offending row (no update, no error).
// ---------------------------------------------------------------------------

/// §2: on a conflict, `DO NOTHING` performs neither the insert nor an update — the existing
/// row is left unchanged and the conflicting VALUES row is skipped (row count unchanged).
#[test]
fn upsert_do_nothing_skips_conflict() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)");
    exec(&mut db, "INSERT INTO t VALUES(1, 'a')");
    exec(&mut db, "INSERT INTO t VALUES(1, 'b') ON CONFLICT(id) DO NOTHING");
    assert_rows(&mut db, "SELECT id, v FROM t ORDER BY id", &[vec![int(1), text("a")]]);
}

/// §2: with no conflict, `DO NOTHING` does not fire — the row is inserted normally.
#[test]
fn upsert_do_nothing_no_conflict_inserts() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)");
    exec(&mut db, "INSERT INTO t VALUES(1, 'a')");
    exec(&mut db, "INSERT INTO t VALUES(2, 'b') ON CONFLICT(id) DO NOTHING");
    assert_rows(
        &mut db,
        "SELECT id, v FROM t ORDER BY id",
        &[vec![int(1), text("a")], vec![int(2), text("b")]],
    );
}

/// §2: with no conflict, `DO UPDATE` likewise does not fire — the row is inserted normally
/// (Bob is a new name, so no name-conflict, so the DO UPDATE branch is skipped).
#[test]
fn upsert_do_update_no_conflict_inserts() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE phonebook(name TEXT PRIMARY KEY, phonenumber TEXT)");
    exec(&mut db, "INSERT INTO phonebook VALUES('Alice','111')");
    exec(&mut db, "INSERT INTO phonebook(name,phonenumber) VALUES('Bob','222') ON CONFLICT(name) DO UPDATE SET phonenumber=excluded.phonenumber");
    assert_rows(
        &mut db,
        "SELECT name, phonenumber FROM phonebook ORDER BY name",
        &[vec![text("Alice"), text("111")], vec![text("Bob"), text("222")]],
    );
}

// ---------------------------------------------------------------------------
// §2 — the uniqueness constraint that triggers the upsert may be a UNIQUE column,
// a unique index, or the PRIMARY KEY.
// ---------------------------------------------------------------------------

/// §2: the conflict target may be a UNIQUE column that is NOT the PRIMARY KEY. The new row's
/// `u='x'` collides with the existing row, so DO UPDATE fires on THAT row (id stays 1) and
/// sets v to the excluded value; no second row is inserted.
#[test]
fn upsert_conflict_target_unique_column() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, u TEXT UNIQUE, v)");
    exec(&mut db, "INSERT INTO t VALUES(1, 'x', 10)");
    exec(&mut db, "INSERT INTO t VALUES(2, 'x', 20) ON CONFLICT(u) DO UPDATE SET v=excluded.v");
    assert_rows(&mut db, "SELECT id, u, v FROM t ORDER BY id", &[vec![int(1), text("x"), int(20)]]);
}

/// §2: "a 'uniqueness constraint' is ... or a unique index." A conflict target matching a
/// `CREATE UNIQUE INDEX` column fires the upsert exactly like an inline UNIQUE constraint.
#[test]
fn upsert_conflict_target_unique_index() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, u TEXT, v)");
    exec(&mut db, "CREATE UNIQUE INDEX idx_u ON t(u)");
    exec(&mut db, "INSERT INTO t VALUES(1, 'x', 10)");
    exec(&mut db, "INSERT INTO t VALUES(2, 'x', 20) ON CONFLICT(u) DO UPDATE SET v=excluded.v");
    assert_rows(&mut db, "SELECT id, u, v FROM t ORDER BY id", &[vec![int(1), text("x"), int(20)]]);
}

/// §2: a PRIMARY KEY conflict target (here the INTEGER PRIMARY KEY / rowid alias) fires the
/// DO UPDATE, overwriting v with the excluded value.
#[test]
fn upsert_integer_pk_conflict_do_update() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)");
    exec(&mut db, "INSERT INTO t VALUES(1, 'a')");
    exec(&mut db, "INSERT INTO t VALUES(1, 'b') ON CONFLICT(id) DO UPDATE SET v=excluded.v");
    assert_rows(&mut db, "SELECT id, v FROM t", &[vec![int(1), text("b")]]);
}

// ---------------------------------------------------------------------------
// §2 — multi-row insert: the upsert decision is made separately per row.
// ---------------------------------------------------------------------------

/// §2: "In the case of a multi-row insert, the upsert decision is made separately for each
/// row." Of (1),(3),(2): 1 and 2 pre-exist (updated to the excluded v), 3 is new (inserted).
#[test]
fn upsert_multi_row_per_row_decision() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)");
    exec(&mut db, "INSERT INTO t VALUES(1, 'a'), (2, 'b')");
    exec(&mut db, "INSERT INTO t VALUES(1, 'x'), (3, 'y'), (2, 'z') ON CONFLICT(id) DO UPDATE SET v=excluded.v");
    assert_rows(
        &mut db,
        "SELECT id, v FROM t ORDER BY id",
        &[vec![int(1), text("x")], vec![int(2), text("z")], vec![int(3), text("y")]],
    );
}

/// §2 (derived): within ONE multi-row INSERT, a later row can conflict with an EARLIER row
/// that the same statement just inserted. Into an empty table `(4,'p'),(4,'q')`: (4,'p')
/// inserts, then (4,'q') would violate the id uniqueness constraint, so DO UPDATE fires and
/// sets v to the excluded 'q' -> (4,'q'). lang_upsert.html does not give this exact example;
/// it follows from §2 ("decision made separately for each row") plus the per-row conflict
/// rule applied against the table state after the earlier row was inserted.
#[test]
fn upsert_multi_row_same_key_second_updates_first() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)");
    exec(&mut db, "INSERT INTO t VALUES(4, 'p'), (4, 'q') ON CONFLICT(id) DO UPDATE SET v=excluded.v");
    assert_rows(&mut db, "SELECT id, v FROM t ORDER BY id", &[vec![int(4), text("q")]]);
}

/// §2: DO NOTHING is also decided per row in a batch — of (1),(2): 1 conflicts and is
/// skipped (existing row kept), 2 is new and inserted.
#[test]
fn upsert_do_nothing_batch_skips_only_conflict() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)");
    exec(&mut db, "INSERT INTO t VALUES(1, 'a')");
    exec(&mut db, "INSERT INTO t VALUES(1, 'skip'), (2, 'keep') ON CONFLICT(id) DO NOTHING");
    assert_rows(
        &mut db,
        "SELECT id, v FROM t ORDER BY id",
        &[vec![int(1), text("a")], vec![int(2), text("keep")]],
    );
}

// ---------------------------------------------------------------------------
// §4 (History, since 3.35.0) — the conflict target may be OMITTED on the last clause.
// ---------------------------------------------------------------------------

/// §4: since 3.35.0 `ON CONFLICT DO UPDATE` (no target) is allowed and fires for any
/// uniqueness violation not captured by a prior clause. Here the id conflict fires it.
#[test]
fn upsert_target_omitted_do_update() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)");
    exec(&mut db, "INSERT INTO t VALUES(1, 'a')");
    exec(&mut db, "INSERT INTO t VALUES(1, 'b') ON CONFLICT DO UPDATE SET v=excluded.v");
    assert_rows(&mut db, "SELECT id, v FROM t", &[vec![int(1), text("b")]]);
}

/// §2 / §4: the target-omitted `ON CONFLICT DO NOTHING` skips any uniqueness conflict with no
/// error. The duplicate `a=1` is skipped and the single row remains.
#[test]
fn upsert_target_omitted_do_nothing() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a UNIQUE)");
    exec(&mut db, "INSERT INTO t VALUES(1)");
    exec(&mut db, "INSERT INTO t VALUES(1) ON CONFLICT DO NOTHING");
    assert_rows(&mut db, "SELECT a FROM t", &[vec![int(1)]]);
}

// ---------------------------------------------------------------------------
// §2 — multiple chained ON CONFLICT clauses (since 3.35.0). "The ON CONFLICT clauses are
// checked in the order specified"; "Only a single ON CONFLICT clause, specifically the first
// ON CONFLICT clause with a matching conflict target, may run for each row." These expected
// values apply that stated rule (lang_upsert.html gives no worked example with output).
// ---------------------------------------------------------------------------

/// §2: a conflict on the FIRST clause's target fires that clause. The a=1 PRIMARY KEY conflict
/// matches `ON CONFLICT(a)` (first clause), so its DO UPDATE runs (v -> 'new') and the trailing
/// `ON CONFLICT(b) DO NOTHING` is bypassed. b keeps 'x' (only v is SET), so no new conflict.
#[test]
fn upsert_chained_first_clause_fires_on_its_target() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT UNIQUE, v TEXT)");
    exec(&mut db, "INSERT INTO t VALUES(1, 'x', 'orig')");
    exec(&mut db, "INSERT INTO t VALUES(1, 'y', 'new') ON CONFLICT(a) DO UPDATE SET v=excluded.v ON CONFLICT(b) DO NOTHING");
    assert_rows(&mut db, "SELECT a, b, v FROM t", &[vec![int(1), text("x"), text("new")]]);
}

/// §2: the matching clause is selected BY conflict target. Here a=2 is new (no PK conflict) but
/// b='x' conflicts, so the first clause `ON CONFLICT(a)` does NOT match and the second
/// `ON CONFLICT(b) DO NOTHING` fires — the row is skipped and the existing row is unchanged.
#[test]
fn upsert_chained_second_clause_fires_on_its_target() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT UNIQUE, v TEXT)");
    exec(&mut db, "INSERT INTO t VALUES(1, 'x', 'orig')");
    exec(&mut db, "INSERT INTO t VALUES(2, 'x', 'new') ON CONFLICT(a) DO UPDATE SET v=excluded.v ON CONFLICT(b) DO NOTHING");
    assert_rows(&mut db, "SELECT a, b, v FROM t", &[vec![int(1), text("x"), text("orig")]]);
}

// ---------------------------------------------------------------------------
// UPSERT + RETURNING (RETURNING is already implemented; this pins the combination).
// ---------------------------------------------------------------------------

/// §2 + RETURNING: `ON CONFLICT(id) DO UPDATE ... RETURNING id, v` returns the row as it
/// stands AFTER the update — the updated (1,'b'), not the pre-conflict (1,'a'). Asserted
/// directly on the RETURNING result of the upsert statement (a single upsert run).
#[test]
fn upsert_do_update_returning_updated_row() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)");
    exec(&mut db, "INSERT INTO t VALUES(1, 'a')");
    assert_rows(
        &mut db,
        "INSERT INTO t VALUES(1, 'b') ON CONFLICT(id) DO UPDATE SET v=excluded.v RETURNING id, v",
        &[vec![int(1), text("b")]],
    );
}

// ---------------------------------------------------------------------------
// DO UPDATE with a CORRELATED subquery — in a column-list source `SET (a,b) = (SELECT …)`,
// a scalar SET value `SET a = (SELECT …)`, a parenthesized row-value item, a single-name
// list, and the DO UPDATE `WHERE`. The DO UPDATE binds every expression against the combined
// `existing(W) ++ excluded(W)` row (`total_width() == 2W`), so a subquery may correlate to
// the EXISTING row (a bare / `table.`-qualified column) or the `excluded.` candidate (the
// values the omitted INSERT would have written, lang_upsert.html §2). This is the SAME
// correlation the plain `UPDATE` supports (rowvalue.html §2.3). Each test pins the real
// post-upsert STATE and is discriminating (a wrong scope width would read the excluded half
// / NULLs, a visibly different value).
// ---------------------------------------------------------------------------

/// rowvalue.html §2.3 + lang_upsert.html §2: a column-list subquery source CORRELATED to the
/// EXISTING conflict row (`src.sk = t.k`, `t.k` the pre-existing key 1). The conflict fires,
/// the subquery matches src row (100, 200), so `(a, b)` take those. A wrong `total_width()`
/// (`W` not `2W`) would rebase `src` onto the excluded half and read `0,0`/NULLs, so
/// asserting `100,200` discriminates the fix.
#[test]
fn upsert_do_update_correlated_column_list_subquery_source() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, a INTEGER, b INTEGER)");
    exec(&mut db, "CREATE TABLE src(sk INTEGER, x INTEGER, y INTEGER)");
    exec(&mut db, "INSERT INTO src VALUES (1, 100, 200)");
    exec(&mut db, "INSERT INTO t VALUES (1, 10, 20)");
    exec(
        &mut db,
        "INSERT INTO t VALUES (1, 0, 0) \
         ON CONFLICT(k) DO UPDATE SET (a, b) = (SELECT x, y FROM src WHERE src.sk = t.k)",
    );
    assert_rows(&mut db, "SELECT k, a, b FROM t", &[vec![int(1), int(100), int(200)]]);
}

/// A column-list subquery source CORRELATED to the `excluded.` candidate row. The candidate
/// carries `a = 7`, so `src.sk = excluded.a` matches src row (700, 800) — proving an
/// `excluded.`-qualified correlation binds to the candidate half `[W, 2W)`, not the existing
/// row (whose `a = 10` would match nothing → NULLs). This is the upsert-specific twist plain
/// `UPDATE` has no analog for (lang_upsert.html §2).
#[test]
fn upsert_do_update_correlated_column_list_subquery_source_to_excluded() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, a INTEGER, b INTEGER)");
    exec(&mut db, "CREATE TABLE src(sk INTEGER, x INTEGER, y INTEGER)");
    exec(&mut db, "INSERT INTO src VALUES (7, 700, 800)");
    exec(&mut db, "INSERT INTO t VALUES (1, 10, 20)");
    exec(
        &mut db,
        "INSERT INTO t VALUES (1, 7, 0) \
         ON CONFLICT(k) DO UPDATE SET (a, b) = (SELECT x, y FROM src WHERE src.sk = excluded.a)",
    );
    assert_rows(&mut db, "SELECT k, a, b FROM t", &[vec![int(1), int(700), int(800)]]);
}

/// A SCALAR SET value with a CORRELATED subquery — `SET a = (SELECT y FROM src WHERE
/// src.sk = t.k)` (the `insert.rs:689` reproducer). Conflict fires; `t.k = 1`
/// matches src `y = 999`, so `a = 999` and the unlisted `b` keeps 20. Previously this arm was
/// UNGUARDED and returned silently-wrong data; asserting `999` pins the root fix.
#[test]
fn upsert_do_update_correlated_scalar_subquery_value() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, a INTEGER, b INTEGER)");
    exec(&mut db, "CREATE TABLE src(sk INTEGER, y INTEGER)");
    exec(&mut db, "INSERT INTO src VALUES (1, 999)");
    exec(&mut db, "INSERT INTO t VALUES (1, 10, 20)");
    exec(
        &mut db,
        "INSERT INTO t VALUES (1, 0, 0) \
         ON CONFLICT(k) DO UPDATE SET a = (SELECT y FROM src WHERE src.sk = t.k)",
    );
    assert_rows(&mut db, "SELECT k, a, b FROM t", &[vec![int(1), int(999), int(20)]]);
}

/// A single-name list `SET (a) = (SELECT …)` (a width-1 row value, "just a scalar value" per
/// rowvalue.html §1) with a CORRELATED subquery (the `insert.rs:707` reproducer).
/// Same expected result as the scalar form — `a = 999`, `b` unchanged.
#[test]
fn upsert_do_update_correlated_single_name_list_subquery() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, a INTEGER, b INTEGER)");
    exec(&mut db, "CREATE TABLE src(sk INTEGER, y INTEGER)");
    exec(&mut db, "INSERT INTO src VALUES (1, 999)");
    exec(&mut db, "INSERT INTO t VALUES (1, 10, 20)");
    exec(
        &mut db,
        "INSERT INTO t VALUES (1, 0, 0) \
         ON CONFLICT(k) DO UPDATE SET (a) = (SELECT y FROM src WHERE src.sk = t.k)",
    );
    assert_rows(&mut db, "SELECT k, a, b FROM t", &[vec![int(1), int(999), int(20)]]);
}

/// A parenthesized row-value with a CORRELATED subquery ITEM mixed with a literal —
/// `SET (a, b) = ((SELECT y … WHERE src.sk = t.k), 5)` (the `insert.rs:695`
/// reproducer). `a` takes the correlated 999, `b` the literal 5.
#[test]
fn upsert_do_update_correlated_parenthesized_row_value_item() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, a INTEGER, b INTEGER)");
    exec(&mut db, "CREATE TABLE src(sk INTEGER, y INTEGER)");
    exec(&mut db, "INSERT INTO src VALUES (1, 999)");
    exec(&mut db, "INSERT INTO t VALUES (1, 10, 20)");
    exec(
        &mut db,
        "INSERT INTO t VALUES (1, 0, 0) \
         ON CONFLICT(k) DO UPDATE SET (a, b) = ((SELECT y FROM src WHERE src.sk = t.k), 5)",
    );
    assert_rows(&mut db, "SELECT k, a, b FROM t", &[vec![int(1), int(999), int(5)]]);
}

/// A CORRELATED subquery in the DO UPDATE `WHERE` (the `insert.rs:548`
/// reproducer). The predicate `(SELECT y FROM src WHERE src.sk = t.k) = 999` is TRUE
/// (`t.k = 1` → `y = 999`), so the `SET a = 111` runs. A wrong scope width would misbind the
/// subquery and flip the decision silently.
#[test]
fn upsert_do_update_correlated_where_true_applies_update() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, a INTEGER, b INTEGER)");
    exec(&mut db, "CREATE TABLE src(sk INTEGER, y INTEGER)");
    exec(&mut db, "INSERT INTO src VALUES (1, 999)");
    exec(&mut db, "INSERT INTO t VALUES (1, 5, 5)");
    exec(
        &mut db,
        "INSERT INTO t VALUES (1, 0, 0) \
         ON CONFLICT(k) DO UPDATE SET a = 111 WHERE (SELECT y FROM src WHERE src.sk = t.k) = 999",
    );
    assert_rows(&mut db, "SELECT k, a, b FROM t", &[vec![int(1), int(111), int(5)]]);
}

/// The mirror of the above with a FALSE correlated `WHERE`: `(SELECT y … WHERE src.sk = t.k)
/// = 0` is `999 = 0` → false, so the DO UPDATE is a no-op and the existing row is unchanged
/// (lang_upsert.html §2.1). Proves the WHERE subquery is genuinely evaluated (not accidentally
/// true), the discriminating twin of the true case.
#[test]
fn upsert_do_update_correlated_where_false_is_noop() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, a INTEGER, b INTEGER)");
    exec(&mut db, "CREATE TABLE src(sk INTEGER, y INTEGER)");
    exec(&mut db, "INSERT INTO src VALUES (1, 999)");
    exec(&mut db, "INSERT INTO t VALUES (1, 5, 5)");
    exec(
        &mut db,
        "INSERT INTO t VALUES (1, 0, 0) \
         ON CONFLICT(k) DO UPDATE SET a = 111 WHERE (SELECT y FROM src WHERE src.sk = t.k) = 0",
    );
    assert_rows(&mut db, "SELECT k, a, b FROM t", &[vec![int(1), int(5), int(5)]]);
}

/// DO UPDATE `SET (a,b) = (SELECT x,y FROM src)`: an UNCORRELATED subquery source — the
/// single source row's columns are assigned positionally to the name list on the conflicting
/// row (rowvalue.html §2.3, "The RHS can be any row value"). Only the conflicting row (k=1)
/// is updated; k=2 is untouched.
#[test]
fn upsert_do_update_subquery_source_uncorrelated() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, a INTEGER, b INTEGER)");
    exec(&mut db, "CREATE TABLE src(x INTEGER, y INTEGER)");
    exec(&mut db, "INSERT INTO src VALUES (77, 88)");
    exec(&mut db, "INSERT INTO t VALUES (1, 10, 20), (2, 30, 40)");
    exec(
        &mut db,
        "INSERT INTO t VALUES (1, 5, 6) ON CONFLICT(k) DO UPDATE SET (a, b) = (SELECT x, y FROM src)",
    );
    assert_rows(
        &mut db,
        "SELECT k, a, b FROM t ORDER BY k",
        &[vec![int(1), int(77), int(88)], vec![int(2), int(30), int(40)]],
    );
}

/// The UPSERT twin of `conformance_update_columnlist::subquery_source_no_rows_sets_columns_null`:
/// a CORRELATED DO UPDATE subquery source that matches ZERO rows sets every listed column to
/// NULL (scalar-subquery semantics — rowvalue.html §2 / lang_expr.html §5), applied positionally
/// to the whole row value. Here `src.sk = t.k` (t.k = existing key 1) finds no `src` row, so
/// both listed columns `(a, b)` become `(NULL, NULL)`. Discriminates a stray wrong value (e.g. the
/// excluded `0` half or the existing `10, 20`) from the spec-correct NULL.
#[test]
fn upsert_do_update_correlated_subquery_source_no_rows_sets_columns_null() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, a INTEGER, b INTEGER)");
    exec(&mut db, "CREATE TABLE src(sk INTEGER, x INTEGER, y INTEGER)");
    exec(&mut db, "INSERT INTO src VALUES (2, 100, 200)"); // no row with sk = 1
    exec(&mut db, "INSERT INTO t VALUES (1, 10, 20)");
    exec(
        &mut db,
        "INSERT INTO t VALUES (1, 0, 0) \
         ON CONFLICT(k) DO UPDATE SET (a, b) = (SELECT x, y FROM src WHERE src.sk = t.k)",
    );
    assert_rows(&mut db, "SELECT k, a, b FROM t", &[vec![int(1), null(), null()]]);
}
