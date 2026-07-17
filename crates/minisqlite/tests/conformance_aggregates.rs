//! Conformance battery: whole-table aggregate functions (NO GROUP BY).
//!
//! Every expectation here is transcribed from the SQLite documentation in
//! `spec/sqlite-doc/lang_aggfunc.html` §3 ("Descriptions of built-in aggregate
//! functions") and §1 (the DISTINCT rule) — NOT from what this engine returns.
//! If the engine disagrees, the spec-correct assertion stays and is left as a
//! genuine failing assertion (each cite below points at the section it must be
//! reconciled against) rather than weakened to pass.
//!
//! Facts under test (anchors are the spec's `<a name=...>` targets):
//!   * count(*)   — total number of rows in the group.                     (#count)
//!   * count(X)   — number of rows where X is not NULL.                     (#count)
//!   * DISTINCT   — duplicates are filtered before the aggregate sees them. (§1)
//!   * sum(X)     — INTEGER result iff every non-NULL input is an integer;
//!                  NULL over an empty / all-NULL group; an "integer overflow"
//!                  error when an all-integer sum overflows i64.            (#sumunc)
//!   * total(X)   — ALWAYS a REAL; 0.0 over an empty group; never overflows.(#sumunc)
//!   * avg(X)     — ALWAYS a REAL given >=1 non-NULL input; NULL otherwise;
//!                  computed as total()/count() over non-NULL inputs.       (#avg)
//!   * min/max(X) — ignore NULLs; NULL iff there are no non-NULL values. (#min_agg,#max_agg)
//!   * group_concat / string_agg — concatenate non-NULL X; default separator
//!                  ",", custom via the 2nd argument; string_agg is an alias.(#group_concat)
//!
//! ORDERING CAVEAT (spec anchor #aggorderby): "If no ORDER BY clause is
//! specified, the inputs to the aggregate occur in an arbitrary order." The
//! group_concat / string_agg tests below insert already-ordered rows and assert
//! the natural table-scan (rowid) order that real sqlite3 produces in practice.
//! A reordering engine is therefore a *spec-permitted* result, not a strict
//! violation — the deterministic form `group_concat(X ORDER BY Y)` pins it, and
//! is exercised in the "aggregate ORDER BY" section below.

mod conformance;

use conformance::*;

// `Connection` is used only in the private setup-helper signatures below, and
// `Error` only in the overflow test's variant/message check; the harness imports
// both privately, so neither is in scope via `conformance::*`.
use minisqlite::{Connection, Error};

// ---- Shared table setups (each test still gets its OWN fresh `mem()` db) ------

/// Setup A: an untyped column holding three integers and one NULL.
/// Derived facts: count(*)=4, count(x)=3, sum=6, total=6.0, avg=2.0, min=1, max=3.
fn table_n(db: &mut Connection) {
    exec(db, "CREATE TABLE n(x)");
    exec(db, "INSERT INTO n VALUES (1), (2), (3), (NULL)");
}

/// Setup B: an empty table (no rows at all).
fn table_empty(db: &mut Connection) {
    exec(db, "CREATE TABLE e(x)");
}

/// The `('a'),('b'),('c')` text table shared by the group_concat tests.
fn table_g(db: &mut Connection) {
    exec(db, "CREATE TABLE g(s)");
    exec(db, "INSERT INTO g VALUES ('a'), ('b'), ('c')");
}

/// Setup C: one integer and one real, so sum/avg exercise REAL results.
fn table_r(db: &mut Connection) {
    exec(db, "CREATE TABLE r(x)");
    exec(db, "INSERT INTO r VALUES (1), (2.5)");
}

/// The `(1),(1),(2),(3)` table shared by the DISTINCT-in-argument tests. The
/// three DISTINCT tests must reason over the SAME multiset, so it lives here
/// once rather than being copy-pasted (a divergent copy would silently break the
/// mutual sum/count/avg consistency).
fn table_dd(db: &mut Connection) {
    exec(db, "CREATE TABLE dd(x)");
    exec(db, "INSERT INTO dd VALUES (1), (1), (2), (3)");
}

// ---- Setup A: non-empty mixed integer/NULL column ----------------------------

#[test]
fn count_star_counts_every_row_including_null() {
    // #count: "count(*) ... returns the total number of rows in the group."
    let mut db = mem();
    table_n(&mut db);
    assert_scalar(&mut db, "SELECT count(*) FROM n", int(4));
}

#[test]
fn count_column_ignores_nulls() {
    // #count: "count(X) ... returns a count of the number of times that X is not
    // NULL". n has one NULL out of four rows, so count(x) = 3.
    let mut db = mem();
    table_n(&mut db);
    assert_scalar(&mut db, "SELECT count(x) FROM n", int(3));
}

#[test]
fn sum_of_all_integer_inputs_is_integer() {
    // #sumunc: "The result of sum() is an integer value if all non-NULL inputs
    // are integers." 1 + 2 + 3 = 6, with the NULL ignored.
    let mut db = mem();
    table_n(&mut db);
    assert_scalar(&mut db, "SELECT sum(x) FROM n", int(6));
}

#[test]
fn total_is_always_real() {
    // #sumunc: "The result of total() is always a floating point value." Same
    // sum as sum(x) above, but returned in the REAL storage class.
    let mut db = mem();
    table_n(&mut db);
    assert_scalar(&mut db, "SELECT total(x) FROM n", real(6.0));
}

#[test]
fn avg_is_real_and_ignores_nulls() {
    // #avg: avg is "always a floating point value whenever there is at least one
    // non-NULL input", computed as total()/count() over non-NULL inputs: 6/3 = 2.0.
    let mut db = mem();
    table_n(&mut db);
    assert_scalar(&mut db, "SELECT avg(x) FROM n", real(2.0));
}

#[test]
fn min_ignores_nulls() {
    // #min_agg: min ignores NULL and returns the smallest value in its own
    // storage class — here the integer 1.
    let mut db = mem();
    table_n(&mut db);
    assert_scalar(&mut db, "SELECT min(x) FROM n", int(1));
}

#[test]
fn max_ignores_nulls() {
    // #max_agg: max ignores NULL and returns the largest value — integer 3.
    let mut db = mem();
    table_n(&mut db);
    assert_scalar(&mut db, "SELECT max(x) FROM n", int(3));
}

// ---- DISTINCT with duplicates present (a second INSERT of (2)) ----------------

#[test]
fn count_distinct_filters_duplicates() {
    // §1: DISTINCT filters duplicate elements before the aggregate. With a
    // duplicate 2 present, count(x) still sees 4 non-NULL rows, while
    // count(DISTINCT x) collapses to the 3 distinct non-NULL values {1,2,3}.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE d(x)");
    exec(&mut db, "INSERT INTO d VALUES (1), (2), (3), (NULL), (2)");
    assert_scalar(&mut db, "SELECT count(x) FROM d", int(4));
    assert_scalar(&mut db, "SELECT count(DISTINCT x) FROM d", int(3));
}

// ---- Setup B: empty table (no rows) ------------------------------------------

#[test]
fn count_star_on_empty_is_zero() {
    // #count: no rows in the group => count(*) = 0.
    let mut db = mem();
    table_empty(&mut db);
    assert_scalar(&mut db, "SELECT count(*) FROM e", int(0));
}

#[test]
fn count_column_on_empty_is_zero() {
    // #count: no non-NULL X (indeed no rows) => count(x) = 0.
    let mut db = mem();
    table_empty(&mut db);
    assert_scalar(&mut db, "SELECT count(x) FROM e", int(0));
}

#[test]
fn sum_of_no_rows_is_null() {
    // #sumunc: "If there are no non-NULL input rows then sum() returns NULL".
    let mut db = mem();
    table_empty(&mut db);
    assert_scalar(&mut db, "SELECT sum(x) FROM e", null());
}

#[test]
fn total_of_no_rows_is_zero_real() {
    // #sumunc: "... but total() returns 0.0." (A REAL zero, not NULL.)
    let mut db = mem();
    table_empty(&mut db);
    assert_scalar(&mut db, "SELECT total(x) FROM e", real(0.0));
}

#[test]
fn avg_of_no_rows_is_null() {
    // #avg: "The result of avg() is NULL if there are no non-NULL inputs."
    let mut db = mem();
    table_empty(&mut db);
    assert_scalar(&mut db, "SELECT avg(x) FROM e", null());
}

#[test]
fn min_of_no_rows_is_null() {
    // #min_agg: min "returns NULL if and only if there are no non-NULL values".
    let mut db = mem();
    table_empty(&mut db);
    assert_scalar(&mut db, "SELECT min(x) FROM e", null());
}

#[test]
fn max_of_no_rows_is_null() {
    // #max_agg: max "returns NULL if and only if there are no non-NULL values".
    let mut db = mem();
    table_empty(&mut db);
    assert_scalar(&mut db, "SELECT max(x) FROM e", null());
}

// ---- All-NULL group: rows exist, but no non-NULL inputs ----------------------

#[test]
fn all_null_group_behaves_like_no_non_null_inputs() {
    // Distinct from the empty table: count(*) counts the rows, but every other
    // aggregate keys off "non-NULL inputs" (#sumunc/#avg/#min_agg/#max_agg) and so
    // behaves as if the group were empty.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE an(x)");
    exec(&mut db, "INSERT INTO an VALUES (NULL), (NULL)");
    assert_scalar(&mut db, "SELECT count(*) FROM an", int(2));
    assert_scalar(&mut db, "SELECT count(x) FROM an", int(0));
    assert_scalar(&mut db, "SELECT sum(x) FROM an", null());
    assert_scalar(&mut db, "SELECT total(x) FROM an", real(0.0));
    assert_scalar(&mut db, "SELECT avg(x) FROM an", null());
    assert_scalar(&mut db, "SELECT min(x) FROM an", null());
    assert_scalar(&mut db, "SELECT max(x) FROM an", null());
}

// ---- Setup C: real and mixed integer/real inputs -----------------------------

#[test]
fn sum_with_a_real_input_is_real() {
    // #sumunc: "If any input to sum() is neither an integer nor a NULL, then
    // sum() returns a floating point value." 1 + 2.5 = 3.5 (REAL).
    let mut db = mem();
    table_r(&mut db);
    assert_scalar(&mut db, "SELECT sum(x) FROM r", real(3.5));
}

#[test]
fn avg_of_mixed_numbers_is_real() {
    // #avg: (1 + 2.5) / 2 = 1.75, always REAL.
    let mut db = mem();
    table_r(&mut db);
    assert_scalar(&mut db, "SELECT avg(x) FROM r", real(1.75));
}

// ---- DISTINCT inside the aggregate argument ----------------------------------

#[test]
fn sum_distinct_sums_only_unique_values() {
    // §1: DISTINCT filters duplicates first, so sum(DISTINCT x) over {1,1,2,3}
    // sums the unique set {1,2,3} = 6 (all integers => INTEGER result, #sumunc).
    let mut db = mem();
    table_dd(&mut db);
    assert_scalar(&mut db, "SELECT sum(DISTINCT x) FROM dd", int(6));
}

#[test]
fn count_distinct_counts_only_unique_values() {
    // §1: count(DISTINCT x) over {1,1,2,3} counts the 3 distinct values.
    let mut db = mem();
    table_dd(&mut db);
    assert_scalar(&mut db, "SELECT count(DISTINCT x) FROM dd", int(3));
}

#[test]
fn avg_distinct_averages_only_unique_values() {
    // §1 + #avg: avg over the unique set {1,2,3} = 6/3 = 2.0 (REAL).
    let mut db = mem();
    table_dd(&mut db);
    assert_scalar(&mut db, "SELECT avg(DISTINCT x) FROM dd", real(2.0));
}

// ---- group_concat / string_agg (see ORDERING CAVEAT in the module docs) ------

#[test]
fn group_concat_default_separator_is_comma() {
    // #group_concat: "A comma (\",\") is used as the separator if Y is omitted."
    let mut db = mem();
    table_g(&mut db);
    assert_scalar(&mut db, "SELECT group_concat(s) FROM g", text("a,b,c"));
}

#[test]
fn group_concat_uses_custom_separator() {
    // #group_concat: "If parameter Y is present then it is used as the separator".
    let mut db = mem();
    table_g(&mut db);
    assert_scalar(&mut db, "SELECT group_concat(s, '-') FROM g", text("a-b-c"));
}

#[test]
fn string_agg_is_alias_for_group_concat() {
    // #group_concat: "The string_agg(X,Y) function is an alias for group_concat(X,Y)."
    let mut db = mem();
    table_g(&mut db);
    assert_scalar(&mut db, "SELECT string_agg(s, ';') FROM g", text("a;b;c"));
}

#[test]
fn group_concat_ignores_nulls() {
    // #group_concat: it concatenates "all non-NULL values of X", so a leading
    // NULL is skipped entirely (no stray separator): (NULL),('a'),('b') => "a,b".
    let mut db = mem();
    exec(&mut db, "CREATE TABLE gn(s)");
    exec(&mut db, "INSERT INTO gn VALUES (NULL), ('a'), ('b')");
    assert_scalar(&mut db, "SELECT group_concat(s) FROM gn", text("a,b"));
}

// ---- aggregate ORDER BY (lang_aggfunc.html #aggorderby) ----------------------
//
// #aggorderby: "aggregate functions … may take an ORDER BY clause following the
// last parameter." The clause "determines the order in which the values are
// processed", making the otherwise-arbitrary input order DETERMINISTIC. Unlike
// the caveat above, these fixtures insert rows in a SCRAMBLED order so a bug that
// ignored the ORDER BY (falling back to rowid order) would produce a different,
// FAILING string — the ORDER BY is doing real work.

/// Rows inserted OUT of value order so `ORDER BY v` must actively reorder them;
/// the natural rowid order is 3,1,2 (not the sorted 1,2,3).
fn table_scrambled(db: &mut Connection) {
    exec(db, "CREATE TABLE sc(v INTEGER)");
    exec(db, "INSERT INTO sc VALUES (3), (1), (2)");
}

#[test]
fn agg_order_by_ascending_sorts_the_inputs() {
    // ORDER BY v (ascending) over scrambled 3,1,2 yields the sorted "1,2,3",
    // NOT the rowid-order "3,1,2" a missing ORDER BY would give.
    let mut db = mem();
    table_scrambled(&mut db);
    assert_scalar(&mut db, "SELECT group_concat(v ORDER BY v) FROM sc", text("1,2,3"));
}

#[test]
fn agg_order_by_descending_sorts_the_inputs() {
    // ORDER BY v DESC yields "3,2,1".
    let mut db = mem();
    table_scrambled(&mut db);
    assert_scalar(&mut db, "SELECT group_concat(v ORDER BY v DESC) FROM sc", text("3,2,1"));
}

#[test]
fn agg_order_by_with_custom_separator() {
    // The ORDER BY follows the LAST argument, so the separator argument and the
    // ORDER BY coexist: `group_concat(v, '-' ORDER BY v)` => "1-2-3".
    let mut db = mem();
    table_scrambled(&mut db);
    assert_scalar(&mut db, "SELECT group_concat(v, '-' ORDER BY v) FROM sc", text("1-2-3"));
}

#[test]
fn agg_order_by_a_different_column_than_aggregated() {
    // The ordering key need not be the aggregated expression: order the `k`s by a
    // separate `v`. Rows (k,v)=('c',3),('a',1),('b',2); ORDER BY v => a,b,c input
    // order, so group_concat(k ORDER BY v) = "a,b,c".
    let mut db = mem();
    exec(&mut db, "CREATE TABLE kv(k TEXT, v INTEGER)");
    exec(&mut db, "INSERT INTO kv VALUES ('c',3),('a',1),('b',2)");
    assert_scalar(&mut db, "SELECT group_concat(k ORDER BY v) FROM kv", text("a,b,c"));
}

#[test]
fn agg_order_by_multiple_keys() {
    // A two-term ORDER BY: primary `g` ascending, secondary `v` DESCending.
    // Rows (g,v): (1,1),(1,3),(2,2); g asc then v desc within g=1 gives 3,1 then
    // g=2's 2 → the v-sequence 3,1,2 → "3,1,2".
    let mut db = mem();
    exec(&mut db, "CREATE TABLE gm(g INTEGER, v INTEGER)");
    exec(&mut db, "INSERT INTO gm VALUES (1,1),(1,3),(2,2)");
    assert_scalar(&mut db, "SELECT group_concat(v ORDER BY g, v DESC) FROM gm", text("3,1,2"));
}

#[test]
fn agg_order_by_distinct_combined() {
    // DISTINCT filters duplicates, ORDER BY sorts the survivors: scrambled
    // 2,3,1,2 → distinct {1,2,3} ordered → "1,2,3".
    let mut db = mem();
    exec(&mut db, "CREATE TABLE d(v INTEGER)");
    exec(&mut db, "INSERT INTO d VALUES (2),(3),(1),(2)");
    assert_scalar(&mut db, "SELECT group_concat(DISTINCT v ORDER BY v) FROM d", text("1,2,3"));
}

#[test]
fn agg_order_by_collate_nocase() {
    // The ordering key honors an explicit COLLATE: NOCASE orders 'B','a','C' as
    // a,B,C (case-insensitive), where BINARY would give B,C,a.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE cs(x TEXT)");
    exec(&mut db, "INSERT INTO cs VALUES ('B'),('a'),('C')");
    assert_scalar(
        &mut db,
        "SELECT group_concat(x ORDER BY x COLLATE NOCASE) FROM cs",
        text("a,B,C"),
    );
}

#[test]
fn agg_order_by_per_group_under_group_by() {
    // Each group orders independently: `GROUP BY k` with `ORDER BY v` per group.
    // k='a' has v={3,1} → "1,3"; k='b' has v={2} → "2". Outer ORDER BY k pins the
    // two output rows.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE pg(k TEXT, v INTEGER)");
    exec(&mut db, "INSERT INTO pg VALUES ('a',3),('b',2),('a',1)");
    assert_rows(
        &mut db,
        "SELECT k, group_concat(v ORDER BY v) FROM pg GROUP BY k ORDER BY k",
        &[vec![text("a"), text("1,3")], vec![text("b"), text("2")]],
    );
}

#[test]
fn agg_order_by_on_non_aggregate_function_is_error() {
    // #aggorderby applies to AGGREGATE functions; an ORDER BY inside a scalar
    // function call is a misuse and must be a loud error, never silently dropped.
    let mut db = mem();
    table_scrambled(&mut db);
    let e = assert_query_error(&mut db, "SELECT abs(v ORDER BY v) FROM sc");
    match e {
        Error::Sql(ref msg) if msg.to_lowercase().contains("order by") => {}
        other => panic!("expected an ORDER-BY-on-scalar error; got: {other:?}"),
    }
}

// ---- Integer overflow: sum() errors, total() does not ------------------------

/// Two i64::MAX rows: their true sum (2^64 - 2) overflows i64.
fn table_ov(db: &mut Connection) {
    exec(db, "CREATE TABLE ov(x)");
    exec(
        db,
        "INSERT INTO ov VALUES (9223372036854775807), (9223372036854775807)",
    );
}

#[test]
fn sum_of_overflowing_integers_raises_error() {
    // #sumunc: "Sum() will throw an 'integer overflow' exception if all inputs
    // are integers or NULL and an integer overflow occurs at any point during
    // the computation." Both inputs are integers, so this must error — and the
    // error must be the *overflow* one, not some incidental failure. Inspecting
    // the returned Error (the `Sql` variant whose message names the overflow)
    // pins this to the spec's distinguishing behavior rather than "some error".
    let mut db = mem();
    table_ov(&mut db);
    let e = assert_query_error(&mut db, "SELECT sum(x) FROM ov");
    match e {
        Error::Sql(ref msg) if msg.to_lowercase().contains("overflow") => {}
        other => panic!(
            "expected an integer-overflow error from sum(x); got a different error: {other:?}"
        ),
    }
}

#[test]
fn total_of_overflowing_integers_does_not_overflow() {
    // #sumunc: "Total() never throws an integer overflow." The same inputs that
    // overflow sum() yield a finite REAL from total(): 2 * 2^63 = 2^64, which is
    // exactly representable in f64 as 1.8446744073709552e19. The generous eps is
    // still negligible against a ~1.8e19 magnitude and only guards accumulation
    // order; the assertion's real job is "REAL class, no error".
    let mut db = mem();
    table_ov(&mut db);
    assert_scalar_approx(
        &mut db,
        "SELECT total(x) FROM ov",
        1.8446744073709552e19,
        1.0e6,
    );
}
