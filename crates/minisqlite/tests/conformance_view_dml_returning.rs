//! Conformance: a `RETURNING` clause on a view DML redirected through an `INSTEAD OF`
//! trigger.
//!
//! Spec basis (`lang_returning.html`):
//!   * A `RETURNING` clause is available on top-level `INSERT` / `UPDATE` / `DELETE`. Its
//!     documented limitations (§3) list virtual tables, nested-in-trigger use, subquery use,
//!     ordering, top-level aggregates, and the modified-table-only rule — but NOT views with
//!     `INSTEAD OF` triggers, which real SQLite accepts.
//!   * §3.5: "The values emitted by the RETURNING clause are the values as seen by the
//!     top-level DELETE, INSERT, or UPDATE statement and do not reflect any subsequent value
//!     changes made by triggers." For a view redirect that means the NEW row the statement
//!     supplied / computed (INSERT, UPDATE) or the OLD row it deleted (DELETE) — NOT whatever
//!     the fired `INSTEAD OF` body actually stored.
//!
//! A view DML combined with `RETURNING` previously returned a loud "RETURNING on a view … is
//! not yet supported". Expected values are hand-derived from the spec. `RETURNING` output
//! order is arbitrary (§3.4), so result sets are compared as multisets (sorted by column 0).

mod conformance;
use conformance::{assert_rows_unordered, exec, int, mem, query, text, value_eq};
use minisqlite::{QueryResult, Value};

/// Seed `base(id, val)`, a pass-through view `v(id, val)`, and its three `INSTEAD OF`
/// triggers redirecting each event to `base`.
fn setup_passthrough(db: &mut minisqlite::Connection) {
    exec(db, "CREATE TABLE base(id INTEGER, val TEXT)");
    exec(db, "CREATE VIEW v(id, val) AS SELECT id, val FROM base");
    exec(
        db,
        "CREATE TRIGGER tr_ins INSTEAD OF INSERT ON v \
         BEGIN INSERT INTO base(id, val) VALUES(NEW.id, NEW.val); END",
    );
    exec(
        db,
        "CREATE TRIGGER tr_del INSTEAD OF DELETE ON v \
         BEGIN DELETE FROM base WHERE id = OLD.id; END",
    );
    exec(
        db,
        "CREATE TRIGGER tr_upd INSTEAD OF UPDATE ON v \
         BEGIN UPDATE base SET val = NEW.val WHERE id = NEW.id; END",
    );
}

/// A `RETURNING` result (captured, since re-running a DML would repeat its side effect) must
/// equal `expected` as a multiset. Both sides are sorted by column 0 (integer id) so row
/// order — arbitrary for RETURNING — does not matter; multiplicity does.
fn assert_returned(qr: &QueryResult, expected: &[Vec<Value>]) {
    let mut got = qr.rows.clone();
    got.sort_by(row_cmp);
    let mut want = expected.to_vec();
    want.sort_by(row_cmp);
    let same = got.len() == want.len()
        && got.iter().zip(&want).all(|(a, b)| {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| value_eq(x, y))
        });
    assert!(same, "RETURNING multiset mismatch\n  expected: {want:?}\n  actual:   {got:?}");
}

/// Order rows by their integer column 0 (all test data uses a distinct integer id there).
fn row_cmp(a: &Vec<Value>, b: &Vec<Value>) -> std::cmp::Ordering {
    match (&a[0], &b[0]) {
        (Value::Integer(x), Value::Integer(y)) => x.cmp(y),
        _ => std::cmp::Ordering::Equal,
    }
}

/// `INSERT INTO v … RETURNING id, val` returns the NEW (inserted) view rows, one per source
/// row, and the redirect still lands them in `base`.
#[test]
fn insert_returning_new_columns() {
    let mut db = mem();
    setup_passthrough(&mut db);
    let qr = query(&mut db, "INSERT INTO v(id, val) VALUES (1, 'a'), (2, 'b') RETURNING id, val");
    assert_returned(&qr, &[vec![int(1), text("a")], vec![int(2), text("b")]]);
    assert_rows_unordered(
        &mut db,
        "SELECT id, val FROM base",
        &[vec![int(1), text("a")], vec![int(2), text("b")]],
    );
}

/// `RETURNING id + 100 AS big, *` names the aliased expression and expands `*` to the view's
/// columns (the NEW row): one row (107, 7, 'g'), columns [big, id, val].
#[test]
fn insert_returning_star_and_expression() {
    let mut db = mem();
    setup_passthrough(&mut db);
    let qr = query(&mut db, "INSERT INTO v(id, val) VALUES (7, 'g') RETURNING id + 100 AS big, *");
    assert_eq!(qr.columns, vec!["big".to_string(), "id".to_string(), "val".to_string()]);
    assert_returned(&qr, &[vec![int(107), int(7), text("g")]]);
}

/// `DELETE FROM v … RETURNING id, val` returns the OLD (deleted) view rows.
#[test]
fn delete_returning_old_columns() {
    let mut db = mem();
    setup_passthrough(&mut db);
    exec(&mut db, "INSERT INTO base VALUES (1, 'a'), (2, 'b'), (3, 'c')");
    let qr = query(&mut db, "DELETE FROM v WHERE id >= 2 RETURNING id, val");
    assert_returned(&qr, &[vec![int(2), text("b")], vec![int(3), text("c")]]);
    assert_rows_unordered(&mut db, "SELECT id, val FROM base", &[vec![int(1), text("a")]]);
}

/// `UPDATE v … RETURNING id, val` returns the NEW (post-update) view rows — the updated
/// value, not the pre-update one.
#[test]
fn update_returning_new_columns() {
    let mut db = mem();
    setup_passthrough(&mut db);
    exec(&mut db, "INSERT INTO base VALUES (1, 'a'), (2, 'b')");
    let qr = query(&mut db, "UPDATE v SET val = 'Z' WHERE id = 1 RETURNING id, val");
    assert_returned(&qr, &[vec![int(1), text("Z")]]);
    assert_rows_unordered(
        &mut db,
        "SELECT id, val FROM base",
        &[vec![int(1), text("Z")], vec![int(2), text("b")]],
    );
}

/// §3.5: the RETURNING values are those the top-level statement saw, and do NOT reflect the
/// change the fired body made. The trigger stores `'STORED'`, but `RETURNING val` must show
/// the supplied NEW value `'orig'`; `base` ends up holding `'STORED'`.
#[test]
fn returning_reflects_the_statement_value_not_the_bodys_write() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE base(id INTEGER, val TEXT)");
    exec(&mut db, "CREATE VIEW v(id, val) AS SELECT id, val FROM base");
    exec(
        &mut db,
        "CREATE TRIGGER tr_ins INSTEAD OF INSERT ON v \
         BEGIN INSERT INTO base(id, val) VALUES(NEW.id, 'STORED'); END",
    );
    let qr = query(&mut db, "INSERT INTO v(id, val) VALUES (1, 'orig') RETURNING id, val");
    assert_returned(&qr, &[vec![int(1), text("orig")]]);
    assert_rows_unordered(&mut db, "SELECT id, val FROM base", &[vec![int(1), text("STORED")]]);
}
