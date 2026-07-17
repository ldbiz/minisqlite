#![allow(dead_code)]
//! Shared conformance test-support harness.
//!
//! Many sibling `tests/conformance_*.rs` files include this module via
//! `mod conformance;` and build the SQL conformance battery on the helpers here.
//! It is a load-bearing seam: the public function signatures below are pinned,
//! so renaming a helper or changing a signature breaks every dependent file. Add
//! helpers when a new pattern is needed; do not repurpose an existing one.
//! Because each including file uses only a subset of the helpers, the module is
//! `#![allow(dead_code)]`.
//!
//! It depends ONLY on the `minisqlite` facade crate — never on the internal
//! engine crates (`minisqlite-types`, `-exec`, …) — so the tests exercise the
//! exact public surface.
//!
//! Why the hand-written comparisons live here: `minisqlite::Value` derives only
//! `Debug` and `Clone` — it has NO `PartialEq` and NO `Ord`. Callers must
//! therefore never use `==` on a `Value`; they use [`value_eq`] for structural
//! equality and the assertions here for row/scalar checks. Concentrating that
//! logic in one place is the whole point of the harness: a single, correct
//! definition of "these values match" that every conformance file shares.

use minisqlite::{Connection, Error, QueryResult, Value};
use std::cmp::Ordering;
use std::fmt::Write as _;

// ---- Value constructors ------------------------------------------------------

/// The SQL NULL value.
pub fn null() -> Value {
    Value::Null
}

/// An INTEGER-class value.
pub fn int(x: i64) -> Value {
    Value::Integer(x)
}

/// A REAL-class value.
pub fn real(x: f64) -> Value {
    Value::Real(x)
}

/// A TEXT-class value.
pub fn text(s: &str) -> Value {
    Value::Text(s.to_string())
}

/// A BLOB-class value.
pub fn blob(bytes: &[u8]) -> Value {
    Value::Blob(bytes.to_vec())
}

// ---- Connections & raw run ---------------------------------------------------

/// A fresh in-memory database. Panics with context if the open fails, so a test
/// body can use it unwrapped.
pub fn mem() -> Connection {
    Connection::open_in_memory()
        .unwrap_or_else(|e| panic!("open_in_memory failed for a fresh database: {e:?}"))
}

/// Execute `sql` (DDL/DML/transaction), panicking with the sql and the error on
/// failure.
pub fn exec(db: &mut Connection, sql: &str) {
    if let Err(e) = db.execute(sql) {
        panic!("exec failed\n  sql: {sql}\n  error: {e:?}");
    }
}

/// Execute `sql`, returning the engine result unchanged (for tests that assert
/// on success/failure themselves).
pub fn try_exec(db: &mut Connection, sql: &str) -> Result<(), Error> {
    db.execute(sql)
}

/// Run `sql` as a query, panicking with the sql and the error on failure.
pub fn query(db: &mut Connection, sql: &str) -> QueryResult {
    match db.query(sql) {
        Ok(qr) => qr,
        Err(e) => panic!("query failed\n  sql: {sql}\n  error: {e:?}"),
    }
}

/// Run `sql` as a query, returning the engine result unchanged.
pub fn try_query(db: &mut Connection, sql: &str) -> Result<QueryResult, Error> {
    db.query(sql)
}

// ---- Structural comparison (Value has NO PartialEq) --------------------------

/// Exact structural equality of two `Value`s.
///
/// `Null == Null`; `Integer` by `==`; `Real` by `f64::total_cmp` being `Equal`
/// (so two identical NaN bit patterns are equal and `+0.0` is *not* equal to
/// `-0.0` — bit identity, not IEEE `==`); `Text` by `==`; `Blob` by byte
/// equality. Different storage classes are NEVER equal — in particular
/// `Integer(1)` does not equal `Real(1.0)`, so a diff stays faithful to the
/// class the engine actually returned.
pub fn value_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Null, Value::Null) => true,
        (Value::Integer(x), Value::Integer(y)) => x == y,
        (Value::Real(x), Value::Real(y)) => x.total_cmp(y) == Ordering::Equal,
        (Value::Text(x), Value::Text(y)) => x == y,
        (Value::Blob(x), Value::Blob(y)) => x == y,
        _ => false,
    }
}

/// Absolute-tolerance float comparison: `|a - b| <= eps`.
pub fn real_approx(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() <= eps
}

/// Storage-class rank for the harness's canonical total order. This is NOT
/// SQLite's ORDER BY class order; it only has to be consistent so both sides of
/// a multiset comparison can be sorted the same way. Integer and Real stay
/// distinct so the diff never silently equates them.
fn class_tag(v: &Value) -> u8 {
    match v {
        Value::Null => 0,
        Value::Integer(_) => 1,
        Value::Real(_) => 2,
        Value::Text(_) => 3,
        Value::Blob(_) => 4,
    }
}

/// A deterministic TOTAL order over `Value`, used only to canonicalize both
/// sides of a multiset comparison (`assert_rows_unordered`). Orders by class
/// tag, then within a class by the natural order of the payload (`total_cmp` for
/// reals, so NaN is ordered and the relation is total).
fn value_total_cmp(a: &Value, b: &Value) -> Ordering {
    let (ta, tb) = (class_tag(a), class_tag(b));
    if ta != tb {
        return ta.cmp(&tb);
    }
    match (a, b) {
        (Value::Null, Value::Null) => Ordering::Equal,
        (Value::Integer(x), Value::Integer(y)) => x.cmp(y),
        (Value::Real(x), Value::Real(y)) => x.total_cmp(y),
        (Value::Text(x), Value::Text(y)) => x.as_bytes().cmp(y.as_bytes()),
        (Value::Blob(x), Value::Blob(y)) => x.as_slice().cmp(y.as_slice()),
        // Equal class tags imply both values are the same variant.
        _ => unreachable!("equal class tag implies matching variants"),
    }
}

/// Lexicographic total order over rows; when one row is a prefix of the other,
/// the shorter one sorts first.
fn row_total_cmp(a: &[Value], b: &[Value]) -> Ordering {
    for (x, y) in a.iter().zip(b.iter()) {
        match value_total_cmp(x, y) {
            Ordering::Equal => continue,
            other => return other,
        }
    }
    a.len().cmp(&b.len())
}

/// Two row sets match when they have the same shape (row count and each row's
/// width) and every cell is `value_eq`. Order-sensitive; callers pre-sort for a
/// multiset comparison.
fn rows_eq(a: &[Vec<Value>], b: &[Vec<Value>]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b.iter()).all(|(ra, rb)| {
            ra.len() == rb.len() && ra.iter().zip(rb.iter()).all(|(x, y)| value_eq(x, y))
        })
}

fn render_rows(out: &mut String, label: &str, rows: &[Vec<Value>]) {
    let _ = writeln!(out, "  {label} ({} row(s)):", rows.len());
    for (i, r) in rows.iter().enumerate() {
        let _ = writeln!(out, "    [{i}] {r:?}");
    }
}

/// A readable multi-line diff for a failed row assertion: the SQL, then the
/// expected and actual row sets rendered via `Value`'s `Debug`.
fn rows_diff(kind: &str, sql: &str, expected: &[Vec<Value>], actual: &[Vec<Value>]) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "{kind} mismatch");
    let _ = writeln!(s, "  sql: {sql}");
    render_rows(&mut s, "expected", expected);
    render_rows(&mut s, "actual", actual);
    s
}

/// Shared 1x1 shape check for the scalar assertions; returns a borrow of the
/// single cell. `want` is a human description of the expectation, used only in
/// the panic message.
fn scalar_cell<'a>(qr: &'a QueryResult, sql: &str, want: &str) -> &'a Value {
    if qr.rows.len() != 1 || qr.rows[0].len() != 1 {
        let width = qr.rows.first().map(|r| r.len()).unwrap_or(0);
        panic!(
            "expected exactly 1 row x 1 column\n  sql: {sql}\n  expected: {want}\n  \
             actual shape: {} row(s) x {width} column(s)\n  actual rows: {:?}",
            qr.rows.len(),
            qr.rows
        );
    }
    &qr.rows[0][0]
}

// ---- Assertions --------------------------------------------------------------

/// ORDERED row comparison: row `i` / column `j` of the result must `value_eq`
/// expected `i` / `j`, and the row count and every row's width must match. On a
/// mismatch, panics with a multi-line diff of the SQL, expected, and actual.
pub fn assert_rows(db: &mut Connection, sql: &str, expected: &[Vec<Value>]) {
    let qr = query(db, sql);
    if !rows_eq(&qr.rows, expected) {
        panic!("{}", rows_diff("assert_rows (ordered)", sql, expected, &qr.rows));
    }
}

/// MULTISET row comparison: canonicalize both sides with the harness total order
/// and compare, so row order is irrelevant but multiplicity is not. On a
/// mismatch, panics with a diff of both sides shown in canonical sorted order.
pub fn assert_rows_unordered(db: &mut Connection, sql: &str, expected: &[Vec<Value>]) {
    let qr = query(db, sql);
    let mut actual = qr.rows;
    let mut want = expected.to_vec();
    actual.sort_by(|a, b| row_total_cmp(a, b));
    want.sort_by(|a, b| row_total_cmp(a, b));
    if !rows_eq(&actual, &want) {
        panic!(
            "{}",
            rows_diff(
                "assert_rows_unordered (multiset, shown in canonical sorted order)",
                sql,
                &want,
                &actual,
            )
        );
    }
}

/// The result must be exactly one row of one column, `value_eq` `expected`.
pub fn assert_scalar(db: &mut Connection, sql: &str, expected: Value) {
    let qr = query(db, sql);
    let got = scalar_cell(&qr, sql, &format!("{expected:?}"));
    if !value_eq(got, &expected) {
        panic!(
            "assert_scalar mismatch\n  sql: {sql}\n  expected: {expected:?}\n  actual:   {got:?}"
        );
    }
}

/// The result must be exactly one row of one column holding a `Real` within
/// `eps` of `expected`; a non-Real value (or a wrong shape) panics.
pub fn assert_scalar_approx(db: &mut Connection, sql: &str, expected: f64, eps: f64) {
    let qr = query(db, sql);
    let got = scalar_cell(&qr, sql, &format!("Real within {eps} of {expected}"));
    match got {
        Value::Real(x) if real_approx(*x, expected, eps) => {}
        Value::Real(x) => panic!(
            "assert_scalar_approx mismatch\n  sql: {sql}\n  \
             expected: {expected} +/- {eps}\n  actual:   {x}"
        ),
        other => panic!(
            "assert_scalar_approx expected a Real value\n  sql: {sql}\n  \
             expected: {expected} +/- {eps}\n  actual:   {other:?}"
        ),
    }
}

/// Result column names must equal `expected` exactly, in order.
pub fn assert_columns(db: &mut Connection, sql: &str, expected: &[&str]) {
    let qr = query(db, sql);
    let same = qr.columns.len() == expected.len()
        && qr.columns.iter().zip(expected.iter()).all(|(a, b)| a == b);
    if !same {
        panic!(
            "assert_columns mismatch\n  sql: {sql}\n  expected: {expected:?}\n  actual:   {:?}",
            qr.columns
        );
    }
}

/// `db.query(sql)` must be `Err`; returns the error (panics if it was `Ok`).
pub fn assert_query_error(db: &mut Connection, sql: &str) -> Error {
    match db.query(sql) {
        Err(e) => e,
        Ok(qr) => panic!(
            "assert_query_error: expected an error but the query succeeded\n  sql: {sql}\n  \
             columns: {:?}\n  rows: {:?}",
            qr.columns, qr.rows
        ),
    }
}

/// `db.execute(sql)` must be `Err`; returns the error (panics if it was `Ok`).
pub fn assert_exec_error(db: &mut Connection, sql: &str) -> Error {
    match db.execute(sql) {
        Err(e) => e,
        Ok(()) => {
            panic!("assert_exec_error: expected an error but execute succeeded\n  sql: {sql}")
        }
    }
}

// ---- No-FROM expression convenience ------------------------------------------

/// Evaluate `SELECT <expr>` in its OWN throwaway in-memory database; assert the
/// result is exactly 1x1 and return that value.
pub fn eval(expr: &str) -> Value {
    let mut db = mem();
    let sql = format!("SELECT {expr}");
    let qr = query(&mut db, &sql);
    if qr.rows.len() != 1 || qr.rows[0].len() != 1 {
        panic!(
            "eval expected exactly 1 row x 1 column\n  expr: {expr}\n  sql: {sql}\n  \
             actual rows: {:?}",
            qr.rows
        );
    }
    qr.rows
        .into_iter()
        .next()
        .and_then(|row| row.into_iter().next())
        .expect("the 1x1 shape checked above guarantees exactly one cell")
}

/// Evaluate `SELECT <expr>` and assert the single cell is `value_eq` `expected`.
pub fn eval_eq(expr: &str, expected: Value) {
    let got = eval(expr);
    if !value_eq(&got, &expected) {
        panic!(
            "eval_eq mismatch\n  expr: {expr}\n  expected: {expected:?}\n  actual:   {got:?}"
        );
    }
}
