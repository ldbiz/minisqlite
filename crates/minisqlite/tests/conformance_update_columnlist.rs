//! Conformance battery: the **UPDATE column-name-list SET form**,
//! `UPDATE t SET (a, b, ...) = <row-value>` (SQLite ≥ 3.15.0).
//!
//! Every expected value is transcribed from the docs, never from what the engine
//! returns.
//!
//! Spec sources:
//!   * `spec/sqlite-doc/lang_update.html` §2: "an assignment in the SET clause can be a
//!     parenthesized list of column names on the left and a row value of the same size on
//!     the right." Also §2: "all scalar expressions are evaluated before any assignments
//!     are made" (so a row value reads the pre-UPDATE row), and "Columns that do not
//!     appear in the list of assignments are left unmodified."
//!   * `spec/sqlite-doc/rowvalue.html` §1: "A row value with a single column is just a
//!     scalar value" (so `(a) = (1)` is the plain assignment `a = 1`, and a 2-name list
//!     fed a scalar is a size mismatch). §2: a row value is either a parenthesized list of
//!     scalars OR "a subquery expression with two or more result columns." §2.3: "Row
//!     values can also be used in the SET clause of an UPDATE statement. The LHS must be a
//!     list of column names. The RHS can be any row value," with the worked example
//!     `UPDATE tab3 SET (a,b,c) = (SELECT x,y,z FROM tab4 WHERE tab4.w=tab3.d) WHERE ...`.
//!     §3.4: a real query mixes scalar assignments (`idxed=1, name=NULL`) with a
//!     column-list-subquery assignment in one SET. §3.5: "`UPDATE tab1 SET (a,b)=(b,a);`"
//!     and "`UPDATE tab1 SET a=b, b=a;`" "do exactly the same thing" — the row-value swap.
//!
//! Three categories, each classified from the spec + real SQLite (NOT from engine output):
//!
//!   (1) `// WORKS` — the parenthesized-row-value form (`(a,b) = (1,2)`); these assert the
//!       SPEC-CORRECT RESULT. A failure here is a real exec bug.
//!   (2) `// SPEC-CORRECT ERROR` — a row value must be "of the same size" as the name list,
//!       so a width mismatch in EITHER direction (too many / too few), on BOTH the
//!       parenthesized source and the subquery source, errors both here and in real SQLite.
//!       Asserted with `assert_exec_error` (the message is NOT pinned): a correct
//!       subquery-source implementation must reject a mismatched-width subquery rather than
//!       silently truncate or pad it.
//!   (3) `// SUBQUERY / ROW-VALUE SOURCE` — a subquery source (`(a,b) = (SELECT ...)`), valid
//!       in real SQLite (rowvalue.html §2, §2.3, §3.4) and implemented in
//!       `crates/minisqlite-plan/src/compile/update.rs`: `bind_assignments` compiles the
//!       subquery once and emits one `ScalarSubqueryColumn` per name, and the executor takes
//!       the subquery's first row and reads each column positionally. These (the
//!       `subquery_source_*` battery) assert the spec-correct post-UPDATE STATE via a
//!       follow-up SELECT.

mod conformance;
use conformance::*;

// =============================================================================
// (1) WORKS — parenthesized row-value source, already implemented.
//     lang_update.html §2 (column-name-list = row value); rowvalue.html §2.3.
// =============================================================================

// WORKS: a literal row value assigns each named column positionally — `(a,b) = (1,2)`
// sets a=1 and b=2 (row value "of the same size", lang_update.html §2).
#[test]
fn works_row_value_literal_assigns_both_columns() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, a INTEGER, b INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1, 10, 20)");
    exec(&mut db, "UPDATE t SET (a, b) = (1, 2)");
    assert_rows(&mut db, "SELECT k, a, b FROM t", &[vec![int(1), int(1), int(2)]]);
}

// WORKS: a three-wide literal row value — the same matched-arity path as the two-column
// case, exercising width 3 through a parenthesized source (a 3-column subquery source is
// covered by `subquery_source_three_column_correlated` in category (3)).
#[test]
fn works_three_column_literal_row_value_assigns_all() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, a INTEGER, b INTEGER, c INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1, 10, 20, 30)");
    exec(&mut db, "UPDATE t SET (a, b, c) = (1, 2, 3)");
    assert_rows(&mut db, "SELECT k, a, b, c FROM t", &[vec![int(1), int(1), int(2), int(3)]]);
}

// WORKS: the row-value swap. rowvalue.html §3.5 states `SET (a,b)=(b,a)` and
// `SET a=b, b=a` "do exactly the same thing"; lang_update.html §2 requires every RHS
// expression to be evaluated against the pre-UPDATE row before any assignment, so a and b
// swap (a←old b, b←old a) rather than both collapsing to one value.
#[test]
fn works_row_value_swaps_columns_using_old_values() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, a INTEGER, b INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1, 10, 20)");
    exec(&mut db, "UPDATE t SET (a, b) = (b, a)");
    // Pre-UPDATE (a,b)=(10,20) -> swapped (20,10); a left-to-right evaluate-and-assign
    // would instead give (20,20), so this discriminates the correct ordering.
    assert_rows(&mut db, "SELECT k, a, b FROM t", &[vec![int(1), int(20), int(10)]]);
}

// WORKS: a single-name list takes a scalar RHS. rowvalue.html §1: "A row value with a
// single column is just a scalar value", so the parser unwraps `(1)` to `1` and
// `SET (a) = (1)` is exactly the plain assignment `a = 1` (only a is changed).
#[test]
fn works_single_name_list_is_scalar_assignment() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, a INTEGER, b INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1, 10, 20)");
    exec(&mut db, "UPDATE t SET (a) = (7)");
    assert_rows(&mut db, "SELECT k, a, b FROM t", &[vec![int(1), int(7), int(20)]]);
}

// WORKS: a WHERE clause on the column-list UPDATE restricts it to matching rows
// (lang_update.html §2). Only k=2 changes; the other rows are untouched.
#[test]
fn works_column_list_with_where_updates_only_matching_rows() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, a INTEGER, b INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1, 10, 20), (2, 30, 40), (3, 50, 60)");
    exec(&mut db, "UPDATE t SET (a, b) = (100, 200) WHERE k = 2");
    assert_rows(
        &mut db,
        "SELECT k, a, b FROM t ORDER BY k",
        &[
            vec![int(1), int(10), int(20)],
            vec![int(2), int(100), int(200)],
            vec![int(3), int(50), int(60)],
        ],
    );
}

// WORKS: a column-list assignment may be mixed with an ordinary scalar assignment in the
// same SET list — rowvalue.html §3.4 shows exactly this (`idxed=1, name=NULL,
// (label,url,mtime) = (SELECT ...)`). Here the row-value part is a literal; the
// subquery-sourced mix is `subquery_source_mixed_with_scalar_assignment` in category
// (3). `(a,b) = (1,2), c = 3` sets a=1, b=2, c=3.
#[test]
fn works_mixed_column_list_and_scalar_assignment() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, a INTEGER, b INTEGER, c INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1, 10, 20, 30)");
    exec(&mut db, "UPDATE t SET (a, b) = (1, 2), c = 3");
    assert_rows(&mut db, "SELECT k, a, b, c FROM t", &[vec![int(1), int(1), int(2), int(3)]]);
}

// WORKS: the row value may hold arbitrary scalar expressions over the row being updated
// (lang_update.html §2). Evaluated against the pre-UPDATE row: a←old a+1, b←old b*2.
#[test]
fn works_row_value_of_expressions_uses_old_values() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, a INTEGER, b INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1, 10, 20)");
    exec(&mut db, "UPDATE t SET (a, b) = (a + 1, b * 2)");
    assert_rows(&mut db, "SELECT k, a, b FROM t", &[vec![int(1), int(11), int(40)]]);
}

// WORKS: "Columns that do not appear in the list of assignments are left unmodified"
// (lang_update.html §2). `(a,b) = (1,2)` leaves the bystander column `c` at its old value.
#[test]
fn works_column_list_leaves_unlisted_columns_unmodified() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, a INTEGER, b INTEGER, c TEXT)");
    exec(&mut db, "INSERT INTO t VALUES (1, 10, 20, 'keep')");
    exec(&mut db, "UPDATE t SET (a, b) = (1, 2)");
    assert_rows(
        &mut db,
        "SELECT k, a, b, c FROM t",
        &[vec![int(1), int(1), int(2), text("keep")]],
    );
}

// =============================================================================
// (2) SPEC-CORRECT ERROR — the row value must be "of the same size" as the name list
//     (lang_update.html §2). Real SQLite errors too, so these are permanent guards;
//     asserted with `assert_exec_error` and the message text is deliberately NOT pinned.
// =============================================================================

// SPEC-CORRECT ERROR: a parenthesized row value with TOO MANY values — `(a,b) = (1,2,3)`
// assigns a 3-wide row value to a 2-name list, a size mismatch that errors here and in real
// SQLite. Exercises the size-error arm in the too-many direction.
#[test]
fn error_row_value_too_many_values_is_size_error() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, a INTEGER, b INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1, 10, 20)");
    assert_exec_error(&mut db, "UPDATE t SET (a, b) = (1, 2, 3)");
}

// SPEC-CORRECT ERROR: a parenthesized row value with TOO FEW values — `(a,b,c) = (1,2)`
// assigns a 2-wide row value to a 3-name list, the mirror direction of the case above. This
// is the one that genuinely exercises the too-few branch of the size-error arm: unlike the
// scalar case below, `(1,2)` stays a parenthesized row value (2 columns), so it reaches the
// size-mismatch check rather than the scalar/gap path. Real SQLite also errors.
#[test]
fn error_row_value_too_few_values_is_size_error() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, a INTEGER, b INTEGER, c INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1, 10, 20, 30)");
    assert_exec_error(&mut db, "UPDATE t SET (a, b, c) = (1, 2)");
}

// SPEC-CORRECT ERROR: a bare scalar RHS to a multi-name list — `(a,b) = (1)`. rowvalue.html
// §1: a single-column row value "is just a scalar value", so the parser unwraps `(1)` to the
// scalar 1; assigning one scalar to a 2-name list is a size mismatch, an error in real SQLite.
// A scalar is neither a parenthesized row value, a single-name list, nor a subquery source,
// so `bind_assignments` routes it to the final size-mismatch arm ("2 columns assigned 1
// values"); only the fact that it errors is asserted, never the wording. PERMANENT guard: a
// bare scalar bound to a 2-name list must error rather than silently succeed.
#[test]
fn error_scalar_rhs_to_multi_name_list() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, a INTEGER, b INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1, 10, 20)");
    assert_exec_error(&mut db, "UPDATE t SET (a, b) = (1)");
}

// SPEC-CORRECT ERROR: a subquery source WIDER than the name list — `(a,b) = (SELECT x,y,z
// FROM src)` feeds a 3-column subquery to a 2-name list. rowvalue.html §2 / lang_update.html
// §2: the row value must be of the same size, so this is a size mismatch and an error in real
// SQLite. It errors via `bind_assignments`' explicit width check (`width != names.len()`);
// PERMANENT guard that the subquery source REJECTS a mismatched-width subquery — unlike the
// matched-width category-(3) tests, which assert success STATE — rather than silently
// truncating the extra column.
#[test]
fn error_subquery_source_wider_than_name_list() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, a INTEGER, b INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1, 10, 20)");
    exec(&mut db, "CREATE TABLE src(x INTEGER, y INTEGER, z INTEGER)");
    exec(&mut db, "INSERT INTO src VALUES (1, 2, 3)");
    assert_exec_error(&mut db, "UPDATE t SET (a, b) = (SELECT x, y, z FROM src)");
}

// SPEC-CORRECT ERROR: a subquery source NARROWER than the name list — `(a,b) = (SELECT x
// FROM src)` feeds a 1-column subquery to a 2-name list, the mirror of the case above and a
// size mismatch in real SQLite. It errors via the same explicit width check; PERMANENT guard
// that the subquery source rejects an under-wide subquery rather than silently padding the
// missing column (e.g. with NULL).
#[test]
fn error_subquery_source_narrower_than_name_list() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, a INTEGER, b INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1, 10, 20)");
    exec(&mut db, "CREATE TABLE src(x INTEGER, y INTEGER)");
    exec(&mut db, "INSERT INTO src VALUES (1, 2)");
    assert_exec_error(&mut db, "UPDATE t SET (a, b) = (SELECT x FROM src)");
}

// =============================================================================
// (3) SUBQUERY / ROW-VALUE SOURCE — `(a,b) = (SELECT ...)`, valid in real SQLite
//     (rowvalue.html §2, §2.3, §3.4) and implemented in
//     `crates/minisqlite-plan/src/compile/update.rs`: `bind_assignments` compiles the
//     subquery once and emits one `ScalarSubqueryColumn` per name; the executor runs the
//     subplan, takes its FIRST row, and reads each column positionally. Each test asserts
//     the spec-correct post-UPDATE STATE via a follow-up SELECT. Every subquery below
//     returns exactly one row per matched target row (except the no-rows edge case).
// =============================================================================

// Uncorrelated 2-column subquery — every matched target row takes the single source
// row's columns. rowvalue.html §2.3: "The RHS can be any row value" (a 2+-column subquery).
#[test]
fn subquery_source_two_column_uncorrelated() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, a INTEGER, b INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1, 10, 20)");
    exec(&mut db, "CREATE TABLE src(x INTEGER, y INTEGER)");
    exec(&mut db, "INSERT INTO src VALUES (100, 200)");
    exec(&mut db, "UPDATE t SET (a, b) = (SELECT x, y FROM src)");
    assert_rows(&mut db, "SELECT k, a, b FROM t", &[vec![int(1), int(100), int(200)]]);
}

// Uncorrelated source over MULTIPLE target rows — locks the cross-row cache-reuse path:
// the source is uncorrelated, so its first row is materialized ONCE (`CachedSubquery::FirstRow`)
// and every one of the two updated rows takes the SAME (100, 200), rather than re-running the
// subplan per row. Distinct from the single-row case above (which never exercises reuse across
// target rows).
#[test]
fn subquery_source_uncorrelated_reused_across_target_rows() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, a INTEGER, b INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1, 0, 0), (2, 0, 0)");
    exec(&mut db, "CREATE TABLE src(x INTEGER, y INTEGER)");
    exec(&mut db, "INSERT INTO src VALUES (100, 200)");
    exec(&mut db, "UPDATE t SET (a, b) = (SELECT x, y FROM src)");
    assert_rows(
        &mut db,
        "SELECT k, a, b FROM t ORDER BY k",
        &[vec![int(1), int(100), int(200)], vec![int(2), int(100), int(200)]],
    );
}

// Correlated 2-column subquery — each target row pulls its own matching source row
// via `src.k = t.k` (rowvalue.html §2.3 pattern).
#[test]
fn subquery_source_two_column_correlated() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, a INTEGER, b INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1, 0, 0), (2, 0, 0)");
    exec(&mut db, "CREATE TABLE src(k INTEGER, x INTEGER, y INTEGER)");
    exec(&mut db, "INSERT INTO src VALUES (1, 11, 12), (2, 21, 22)");
    exec(&mut db, "UPDATE t SET (a, b) = (SELECT x, y FROM src WHERE src.k = t.k)");
    assert_rows(
        &mut db,
        "SELECT k, a, b FROM t ORDER BY k",
        &[vec![int(1), int(11), int(12)], vec![int(2), int(21), int(22)]],
    );
}

// Mirrors the flagship rowvalue.html §2.3 example: a correlated subquery source with
// a WHERE on the UPDATE, so only k=2 changes and k=1 keeps its old values.
#[test]
fn subquery_source_correlated_with_where() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, a INTEGER, b INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1, 0, 0), (2, 0, 0)");
    exec(&mut db, "CREATE TABLE src(k INTEGER, x INTEGER, y INTEGER)");
    exec(&mut db, "INSERT INTO src VALUES (1, 11, 12), (2, 21, 22)");
    exec(&mut db, "UPDATE t SET (a, b) = (SELECT x, y FROM src WHERE src.k = t.k) WHERE t.k = 2");
    assert_rows(
        &mut db,
        "SELECT k, a, b FROM t ORDER BY k",
        &[vec![int(1), int(0), int(0)], vec![int(2), int(21), int(22)]],
    );
}

// Three-column correlated subquery — the row value's width matches the 3-name list.
#[test]
fn subquery_source_three_column_correlated() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, a INTEGER, b INTEGER, c INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1, 0, 0, 0), (2, 0, 0, 0)");
    exec(&mut db, "CREATE TABLE src(k INTEGER, x INTEGER, y INTEGER, z INTEGER)");
    exec(&mut db, "INSERT INTO src VALUES (1, 11, 12, 13), (2, 21, 22, 23)");
    exec(&mut db, "UPDATE t SET (a, b, c) = (SELECT x, y, z FROM src WHERE src.k = t.k)");
    assert_rows(
        &mut db,
        "SELECT k, a, b, c FROM t ORDER BY k",
        &[
            vec![int(1), int(11), int(12), int(13)],
            vec![int(2), int(21), int(22), int(23)],
        ],
    );
}

// Mirrors rowvalue.html §3.4: a scalar assignment (`c = 99`) mixed with a
// column-list-subquery assignment in one SET. Both parts must apply.
#[test]
fn subquery_source_mixed_with_scalar_assignment() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, a INTEGER, b INTEGER, c INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1, 0, 0, 0)");
    exec(&mut db, "CREATE TABLE src(x INTEGER, y INTEGER)");
    exec(&mut db, "INSERT INTO src VALUES (5, 6)");
    exec(&mut db, "UPDATE t SET c = 99, (a, b) = (SELECT x, y FROM src)");
    assert_rows(&mut db, "SELECT k, a, b, c FROM t", &[vec![int(1), int(5), int(6), int(99)]]);
}

// A subquery-sourced column list still leaves unlisted columns unmodified
// (lang_update.html §2) — `keep` retains its old value while a,b take the source columns.
#[test]
fn subquery_source_leaves_unlisted_columns_unmodified() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, a INTEGER, b INTEGER, keep TEXT)");
    exec(&mut db, "INSERT INTO t VALUES (1, 10, 20, 'orig')");
    exec(&mut db, "CREATE TABLE src(x INTEGER, y INTEGER)");
    exec(&mut db, "INSERT INTO src VALUES (5, 6)");
    exec(&mut db, "UPDATE t SET (a, b) = (SELECT x, y FROM src)");
    assert_rows(
        &mut db,
        "SELECT k, a, b, keep FROM t",
        &[vec![int(1), int(5), int(6), text("orig")]],
    );
}

// EDGE: a subquery source that returns NO rows sets every listed column to
// NULL — standard scalar-subquery semantics (rowvalue.html §2 / lang_expr.html §5: a
// scalar subquery with no rows is NULL), applied positionally to the whole row value.
// Here the correlated source matches no `src` row for k=1, so (a,b) become (NULL, NULL);
// the unlisted `c` is untouched.
#[test]
fn subquery_source_no_rows_sets_columns_null() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, a INTEGER, b INTEGER, c INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1, 10, 20, 30)");
    exec(&mut db, "CREATE TABLE src(k INTEGER, x INTEGER, y INTEGER)");
    exec(&mut db, "INSERT INTO src VALUES (2, 99, 98)"); // no row with k = 1
    exec(&mut db, "UPDATE t SET (a, b) = (SELECT x, y FROM src WHERE src.k = t.k)");
    assert_rows(
        &mut db,
        "SELECT k, a, b, c FROM t",
        &[vec![int(1), null(), null(), int(30)]],
    );
}

// EDGE: a subquery source that returns MORE THAN ONE row uses the FIRST row
// only (scalar-subquery semantics — the remaining rows are ignored), and both listed
// columns come from that SAME first row (not columns picked across different rows). The
// uncorrelated source is ordered so the first row is deterministic.
#[test]
fn subquery_source_multiple_rows_uses_first() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(k INTEGER PRIMARY KEY, a INTEGER, b INTEGER)");
    exec(&mut db, "INSERT INTO t VALUES (1, 0, 0)");
    exec(&mut db, "CREATE TABLE src(seq INTEGER PRIMARY KEY, x INTEGER, y INTEGER)");
    exec(&mut db, "INSERT INTO src VALUES (1, 11, 12), (2, 21, 22)");
    exec(&mut db, "UPDATE t SET (a, b) = (SELECT x, y FROM src ORDER BY seq)");
    assert_rows(&mut db, "SELECT k, a, b FROM t", &[vec![int(1), int(11), int(12)]]);
}
