//! Self-test for the shared conformance harness (`tests/conformance/mod.rs`),
//! and the first conformance file. It exercises every public helper against SQL
//! the engine already supports (no-FROM expressions, CREATE TABLE, INSERT,
//! SELECT + WHERE + ORDER BY, aggregates), so a regression in the harness — or a
//! break in that baseline SQL — surfaces here.
//!
//! If a case fails because the engine returns something unexpected, that is a
//! real signal: the case KEEPS asserting the spec-correct value rather than being
//! weakened to pass.

mod conformance;

use conformance::*;

#[test]
fn eval_covers_literals_ops_and_typeof() {
    eval_eq("1 + 2", int(3));
    eval_eq("2.5", real(2.5));
    eval_eq("'a' || 'b'", text("ab"));
    eval_eq("NULL", null());
    eval_eq("typeof(1)", text("integer"));

    // Blob literal round-trips to a harness `blob(..)` value (an engine-produced
    // blob, not one the harness built), closing the loop on the blob class.
    eval_eq("x'41ff'", blob(&[0x41, 0xff]));
    eval_eq("typeof(x'41ff')", text("blob"));

    // `eval` returns the value; compare it structurally (Value has no PartialEq).
    assert!(value_eq(&eval("3 * 4"), &int(12)));
}

#[test]
fn value_eq_is_exact_and_class_sensitive() {
    // Same-class equality across all five storage classes.
    assert!(value_eq(&null(), &null()));
    assert!(value_eq(&int(5), &int(5)));
    assert!(value_eq(&real(1.5), &real(1.5)));
    assert!(value_eq(&text("hi"), &text("hi")));
    assert!(value_eq(&blob(&[1, 2, 3]), &blob(&[1, 2, 3])));

    // Inequality within a class.
    assert!(!value_eq(&int(5), &int(6)));
    assert!(!value_eq(&real(1.5), &real(1.6)));
    assert!(!value_eq(&text("hi"), &text("ho")));
    assert!(!value_eq(&blob(&[1, 2]), &blob(&[1, 3])));

    // Different classes are never equal — notably Integer vs Real.
    assert!(!value_eq(&int(1), &real(1.0)));
    assert!(!value_eq(&null(), &int(0)));
    assert!(!value_eq(&text("1"), &int(1)));
    assert!(!value_eq(&blob(&[49]), &text("1")));

    // Real uses f64::total_cmp: identical NaN bits are equal, and +0.0 != -0.0.
    assert!(value_eq(&real(f64::NAN), &real(f64::NAN)));
    assert!(!value_eq(&real(0.0), &real(-0.0)));
}

#[test]
fn real_approx_within_epsilon() {
    assert!(real_approx(0.1 + 0.2, 0.3, 1e-9));
    assert!(real_approx(1.0, 1.0, 0.0));
    assert!(!real_approx(1.0, 1.1, 1e-9));
}

#[test]
fn table_rows_scalar_and_columns() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
    exec(&mut db, "INSERT INTO t VALUES (2, 'y'), (1, 'x')");

    // Direct use of the public `query` helper.
    let qr = query(&mut db, "SELECT a, b FROM t ORDER BY a");
    assert_eq!(qr.rows.len(), 2);

    // ORDERED: rows must appear in exactly this order.
    assert_rows(
        &mut db,
        "SELECT a, b FROM t ORDER BY a",
        &[vec![int(1), text("x")], vec![int(2), text("y")]],
    );

    // MULTISET: no ORDER BY. The engine returns rows in rowid (insertion) order
    // — (2,'y') then (1,'x') — so supplying the expected set in the OPPOSITE
    // order makes the canonicalizing sort load-bearing: a no-op sort would leave
    // the two sides in different orders and mismatch here.
    assert_rows_unordered(
        &mut db,
        "SELECT a, b FROM t",
        &[vec![int(1), text("x")], vec![int(2), text("y")]],
    );

    assert_scalar(&mut db, "SELECT count(*) FROM t", int(2));
    assert_columns(&mut db, "SELECT a AS one, b AS two FROM t", &["one", "two"]);
}

#[test]
fn scalar_approx_on_avg() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE nums(x INTEGER)");
    exec(&mut db, "INSERT INTO nums VALUES (1), (2)");
    // avg over {1, 2} is 1.5, which must come back as a REAL.
    assert_scalar_approx(&mut db, "SELECT avg(x) FROM nums", 1.5, 1e-9);
}

#[test]
fn error_and_try_variants() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER)");

    // try_exec / try_query success paths.
    assert!(try_exec(&mut db, "INSERT INTO t VALUES (1)").is_ok());
    assert!(try_query(&mut db, "SELECT a FROM t").is_ok());

    // try_exec / try_query failure paths.
    assert!(try_query(&mut db, "SELECT * FROM no_such_table").is_err());
    assert!(try_exec(&mut db, "CREATE TABLE").is_err());

    // assert_query_error / assert_exec_error return the Error they observed.
    let _e1 = assert_query_error(&mut db, "SELECT * FROM no_such_table");
    let _e2 = assert_exec_error(&mut db, "CREATE TABLE");
}

/// Assert that calling `f` unwinds (panics). Proves the harness's own assertions
/// actually FAIL on bad input — without this, a regression that turned an
/// `assert_*` into a no-op would leave every downstream conformance check
/// silently pass (a false all-clear). Each `f`'s panic message is captured by
/// the test harness and shown only if THIS test itself fails.
fn assert_panics(what: &str, f: impl FnOnce() + std::panic::UnwindSafe) {
    let outcome = std::panic::catch_unwind(f);
    assert!(
        outcome.is_err(),
        "expected `{what}` to panic on bad input, but it returned normally"
    );
}

#[test]
fn assertions_fail_loudly_on_mismatch() {
    // Each closure owns its own database so a caught unwind cannot leak state
    // into the next case.
    assert_panics("assert_rows wrong value", || {
        let mut db = mem();
        exec(&mut db, "CREATE TABLE t(a INTEGER)");
        exec(&mut db, "INSERT INTO t VALUES (1)");
        assert_rows(&mut db, "SELECT a FROM t", &[vec![int(2)]]);
    });
    assert_panics("assert_rows wrong row count", || {
        let mut db = mem();
        exec(&mut db, "CREATE TABLE t(a INTEGER)");
        exec(&mut db, "INSERT INTO t VALUES (1)");
        assert_rows(&mut db, "SELECT a FROM t", &[vec![int(1)], vec![int(2)]]);
    });
    assert_panics("assert_rows_unordered wrong multiset", || {
        let mut db = mem();
        exec(&mut db, "CREATE TABLE t(a INTEGER)");
        exec(&mut db, "INSERT INTO t VALUES (1), (2)");
        assert_rows_unordered(&mut db, "SELECT a FROM t", &[vec![int(1)], vec![int(3)]]);
    });
    assert_panics("assert_scalar wrong value", || {
        let mut db = mem();
        assert_scalar(&mut db, "SELECT 1", int(2));
    });
    // Integer(1) must NOT satisfy an expected Real(1.0): class-sensitivity.
    assert_panics("assert_scalar cross-class Integer vs Real", || {
        let mut db = mem();
        assert_scalar(&mut db, "SELECT 1", real(1.0));
    });
    assert_panics("assert_scalar_approx on a non-Real result", || {
        let mut db = mem();
        assert_scalar_approx(&mut db, "SELECT 1", 1.0, 1e-9);
    });
    assert_panics("assert_columns wrong name", || {
        let mut db = mem();
        assert_columns(&mut db, "SELECT 1 AS a", &["b"]);
    });
    assert_panics("assert_query_error on a valid query", || {
        let mut db = mem();
        let _ = assert_query_error(&mut db, "SELECT 1");
    });
    assert_panics("assert_exec_error on a valid statement", || {
        let mut db = mem();
        let _ = assert_exec_error(&mut db, "CREATE TABLE ok(a INTEGER)");
    });
    assert_panics("eval on a non-1x1 result", || {
        let _ = eval("1, 2");
    });
}
