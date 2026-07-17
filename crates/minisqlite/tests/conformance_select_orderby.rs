//! Conformance battery for the SQL `ORDER BY` clause.
//!
//! Every expected result here is transcribed from the SQLite manual in
//! `spec/sqlite-doc/`, NOT from what the engine currently returns:
//!
//! - `lang_select.html` §4 "The ORDER BY clause": rows are sorted by the
//!   left-most ORDER BY expression, ties broken by the next, and so on; each
//!   term may carry `ASC` (default, smaller first) or `DESC` (larger first). A
//!   NULL sorts before any other value, so NULLs lead an `ASC` ordering and
//!   trail a `DESC` one; `ASC NULLS LAST` / `DESC NULLS FIRST` (and the explicit
//!   `ASC NULLS FIRST` / `DESC NULLS LAST`) override that placement. A term that
//!   is a constant integer K aliases the K-th result column (1-based); a term
//!   that is an output-column alias sorts by that column; any other term is an
//!   evaluated expression.
//! - `datatype3.html` §6 "Sorting, Grouping and Compound SELECTs": an ORDER BY
//!   returns NULLs first, then INTEGER and REAL values interspersed in numeric
//!   order, then TEXT in collating-sequence order, then BLOB in memcmp() order.
//!   No storage-class conversion happens before the sort, so a returned value
//!   keeps the class it was stored as.
//! - `datatype3.html` §4.1 "Sort Order": NULL < everything; an INTEGER/REAL is
//!   less than any TEXT/BLOB; a TEXT is less than any BLOB; two BLOBs compare by
//!   memcmp().
//! - `datatype3.html` §7 "Collating Sequences": the default TEXT collation is
//!   BINARY (memcmp); `COLLATE NOCASE` folds the 26 ASCII uppercase letters to
//!   lowercase before comparing.
//!
//! Ties in the sort key are documented as "undefined" order, so every case here
//! either has a fully distinct sort key or supplies a deterministic tiebreaker —
//! order IS the property under test, so the expectation must be unambiguous.
//!
//! If the engine returns something different, the assertion stays spec-correct
//! and FAILS rather than being weakened. Only a hanging/panicking case would be
//! isolated and ignored.

mod conformance;

use conformance::*;
use minisqlite::Connection;

/// The canonical `t(a INTEGER, b TEXT)` fixture used across the basic ordering
/// cases: four rows with a duplicate `a` (two rows with `a = 1`) so that a
/// tiebreaker on `b` is required to make the order deterministic.
fn table_t() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a INTEGER, b TEXT)");
    exec(&mut db, "INSERT INTO t VALUES (3, 'x'), (1, 'z'), (2, 'y'), (1, 'a')");
    db
}

// ---- Basic ascending / descending, with deterministic tiebreaks -------------

#[test]
fn orderby_ascending_with_tiebreak() {
    // `ORDER BY a, b`: ascending `a`, ties broken by ascending `b`.
    let mut db = table_t();
    assert_rows(
        &mut db,
        "SELECT a, b FROM t ORDER BY a, b",
        &[
            vec![int(1), text("a")],
            vec![int(1), text("z")],
            vec![int(2), text("y")],
            vec![int(3), text("x")],
        ],
    );
}

#[test]
fn orderby_descending_with_tiebreak() {
    // `ORDER BY a DESC, b DESC`: larger values first on both keys.
    let mut db = table_t();
    assert_rows(
        &mut db,
        "SELECT a, b FROM t ORDER BY a DESC, b DESC",
        &[
            vec![int(3), text("x")],
            vec![int(2), text("y")],
            vec![int(1), text("z")],
            vec![int(1), text("a")],
        ],
    );
}

#[test]
fn orderby_mixed_directions() {
    // `ORDER BY a ASC, b DESC`: ascending `a`, ties broken by *descending* `b`,
    // so among the two `a = 1` rows 'z' precedes 'a'.
    let mut db = table_t();
    assert_rows(
        &mut db,
        "SELECT a, b FROM t ORDER BY a ASC, b DESC",
        &[
            vec![int(1), text("z")],
            vec![int(1), text("a")],
            vec![int(2), text("y")],
            vec![int(3), text("x")],
        ],
    );
}

#[test]
fn orderby_omitting_direction_equals_explicit_asc() {
    // "If neither ASC or DESC are specified, rows are sorted in ascending order
    // by default" (lang_select.html §4). Prove that equivalence directly: the
    // keyword-less ordering must return exactly the same rows as the explicit
    // `ASC` ordering. `table_t`'s `a` differs under ASC vs DESC, so a default
    // that resolved to DESC would make the two results diverge here. The
    // absolute ascending order itself is pinned by
    // `orderby_ascending_with_tiebreak`, so this case adds the equivalence, not
    // a duplicate of that literal.
    let mut db = table_t();
    let implicit = query(&mut db, "SELECT a, b FROM t ORDER BY a, b").rows;
    let explicit = query(&mut db, "SELECT a, b FROM t ORDER BY a ASC, b ASC").rows;
    let same = implicit.len() == explicit.len()
        && implicit.iter().zip(&explicit).all(|(r1, r2)| {
            r1.len() == r2.len() && r1.iter().zip(r2).all(|(x, y)| value_eq(x, y))
        });
    assert!(
        same,
        "omitting ASC must equal explicit ASC\n  implicit:     {implicit:?}\n  explicit ASC: {explicit:?}"
    );
}

// ---- Sorting by column position, alias, and expression ----------------------

#[test]
fn orderby_by_column_position() {
    // A constant integer term aliases the N-th result column (1-based), so
    // `ORDER BY 1, 2` is equivalent to `ORDER BY a, b` here.
    let mut db = table_t();
    assert_rows(
        &mut db,
        "SELECT a, b FROM t ORDER BY 1, 2",
        &[
            vec![int(1), text("a")],
            vec![int(1), text("z")],
            vec![int(2), text("y")],
            vec![int(3), text("x")],
        ],
    );
}

#[test]
fn orderby_by_output_alias() {
    // An ORDER BY term that matches an output column alias sorts by that column.
    let mut db = table_t();
    assert_rows(
        &mut db,
        "SELECT a AS k, b FROM t ORDER BY k, b",
        &[
            vec![int(1), text("a")],
            vec![int(1), text("z")],
            vec![int(2), text("y")],
            vec![int(3), text("x")],
        ],
    );
}

#[test]
fn orderby_by_column_absent_from_select_list() {
    // A simple SELECT's ORDER BY may reference any column, even one not in the
    // result set (lang_select.html §4). Sort by `a, b` but project only `b`;
    // the tiebreaker keeps the projected order deterministic.
    let mut db = table_t();
    assert_rows(
        &mut db,
        "SELECT b FROM t ORDER BY a, b",
        &[vec![text("a")], vec![text("z")], vec![text("y")], vec![text("x")]],
    );
}

#[test]
fn orderby_by_expression() {
    // A non-integer, non-alias term is an evaluated expression. `a * -1`
    // ascending negates the sign, so rows come back in descending `a`; the two
    // `a = 1` rows tie on the key and are broken by ascending `b`.
    let mut db = table_t();
    assert_rows(
        &mut db,
        "SELECT a, b FROM t ORDER BY a * -1, b",
        &[
            vec![int(3), text("x")],
            vec![int(2), text("y")],
            vec![int(1), text("a")],
            vec![int(1), text("z")],
        ],
    );
}

// ---- NULL placement in ORDER BY --------------------------------------------

/// `n(x)` holds two NULLs among the integers 1 and 2 to pin NULL placement.
fn table_n() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE n(x)");
    exec(&mut db, "INSERT INTO n VALUES (2), (NULL), (1), (NULL)");
    db
}

#[test]
fn orderby_nulls_default_ascending_first() {
    // NULLs are smaller than any value, so they lead an ascending order.
    let mut db = table_n();
    assert_rows(
        &mut db,
        "SELECT x FROM n ORDER BY x",
        &[vec![null()], vec![null()], vec![int(1)], vec![int(2)]],
    );
}

#[test]
fn orderby_nulls_default_descending_last() {
    // Descending reverses the order, so NULLs trail.
    let mut db = table_n();
    assert_rows(
        &mut db,
        "SELECT x FROM n ORDER BY x DESC",
        &[vec![int(2)], vec![int(1)], vec![null()], vec![null()]],
    );
}

#[test]
fn orderby_asc_nulls_last() {
    // Explicit override: ascending values but NULLs pushed to the end.
    let mut db = table_n();
    assert_rows(
        &mut db,
        "SELECT x FROM n ORDER BY x ASC NULLS LAST",
        &[vec![int(1)], vec![int(2)], vec![null()], vec![null()]],
    );
}

#[test]
fn orderby_asc_nulls_first_explicit() {
    // The explicit spelling of the ascending default.
    let mut db = table_n();
    assert_rows(
        &mut db,
        "SELECT x FROM n ORDER BY x ASC NULLS FIRST",
        &[vec![null()], vec![null()], vec![int(1)], vec![int(2)]],
    );
}

#[test]
fn orderby_desc_nulls_first() {
    // Explicit override: descending values but NULLs pulled to the front.
    let mut db = table_n();
    assert_rows(
        &mut db,
        "SELECT x FROM n ORDER BY x DESC NULLS FIRST",
        &[vec![null()], vec![null()], vec![int(2)], vec![int(1)]],
    );
}

#[test]
fn orderby_desc_nulls_last_explicit() {
    // The explicit spelling of the descending default.
    let mut db = table_n();
    assert_rows(
        &mut db,
        "SELECT x FROM n ORDER BY x DESC NULLS LAST",
        &[vec![int(2)], vec![int(1)], vec![null()], vec![null()]],
    );
}

// ---- Cross-storage-class ordering (datatype3.html §6) -----------------------

/// `m(v)` has no declared type (BLOB/no affinity), so each inserted literal
/// keeps its storage class: one value of every class.
fn table_m() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE m(v)");
    exec(&mut db, "INSERT INTO m VALUES (NULL), (1), (2.5), ('abc'), (x'00')");
    db
}

#[test]
fn orderby_cross_class_ascending() {
    // Class order for ascending ORDER BY: NULL, then numeric (INTEGER/REAL
    // interspersed), then TEXT, then BLOB. No class conversion occurs, so the
    // integer stays Integer and the real stays Real.
    let mut db = table_m();
    assert_rows(
        &mut db,
        "SELECT v FROM m ORDER BY v",
        &[
            vec![null()],
            vec![int(1)],
            vec![real(2.5)],
            vec![text("abc")],
            vec![blob(&[0x00])],
        ],
    );
}

#[test]
fn orderby_cross_class_descending() {
    // Descending reverses the whole class order, sending NULL to the end.
    let mut db = table_m();
    assert_rows(
        &mut db,
        "SELECT v FROM m ORDER BY v DESC",
        &[
            vec![blob(&[0x00])],
            vec![text("abc")],
            vec![real(2.5)],
            vec![int(1)],
            vec![null()],
        ],
    );
}

#[test]
fn orderby_integer_and_real_interspersed_numerically() {
    // §6: INTEGER and REAL values are interspersed in *numeric* order (not
    // grouped by class), and each keeps its stored class since no conversion
    // happens before the sort. Distinct numeric values avoid any tie ambiguity.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE mix(v)");
    exec(&mut db, "INSERT INTO mix VALUES (1), (1.5), (2), (2.5), (3)");
    assert_rows(
        &mut db,
        "SELECT v FROM mix ORDER BY v",
        &[
            vec![int(1)],
            vec![real(1.5)],
            vec![int(2)],
            vec![real(2.5)],
            vec![int(3)],
        ],
    );
}

// ---- TEXT collation in ORDER BY --------------------------------------------

/// `cf(s TEXT)` mixes letter case so that BINARY and NOCASE collation produce
/// *different* orders — the shared fixture behind the discriminating
/// BINARY-vs-NOCASE contrast pair below.
fn table_cf() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE cf(s TEXT)");
    exec(&mut db, "INSERT INTO cf VALUES ('Banana'), ('apple'), ('Cherry')");
    db
}

#[test]
fn orderby_text_default_binary_collation() {
    // A TEXT column with no COLLATE clause sorts with BINARY (memcmp), where all
    // ASCII uppercase letters (0x41..) precede all lowercase (0x61..). So
    // 'Banana' and 'Cherry' sort before 'apple'.
    let mut db = table_cf();
    assert_rows(
        &mut db,
        "SELECT s FROM cf ORDER BY s",
        &[vec![text("Banana")], vec![text("Cherry")], vec![text("apple")]],
    );
}

#[test]
fn orderby_collate_nocase_folds_case() {
    // `COLLATE NOCASE` folds ASCII case before comparing, so the order becomes
    // 'apple' < 'Banana' < 'Cherry' — different from the BINARY order above,
    // which is what makes this a discriminating check of NOCASE.
    let mut db = table_cf();
    assert_rows(
        &mut db,
        "SELECT s FROM cf ORDER BY s COLLATE NOCASE",
        &[vec![text("apple")], vec![text("Banana")], vec![text("Cherry")]],
    );
}

#[test]
fn orderby_collate_nocase_probe_all_lowercase() {
    // A probe on `t.b` ('x','z','y','a'). All values are
    // lowercase, so NOCASE matches BINARY here: a, x, y, z.
    let mut db = table_t();
    assert_rows(
        &mut db,
        "SELECT b FROM t ORDER BY b COLLATE NOCASE",
        &[vec![text("a")], vec![text("x")], vec![text("y")], vec![text("z")]],
    );
}

// ---- BLOB ordering (memcmp, §4.1 / §6) -------------------------------------

#[test]
fn orderby_blob_memcmp_order() {
    // BLOBs compare byte-by-byte with memcmp(): a shorter blob that is a prefix
    // of a longer one sorts first. For x'01', x'0100', x'00ff', x'02':
    //   x'00ff' (first byte 0x00) < x'01' < x'0100' (0x01 is a prefix of it)
    //   < x'02'.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE blb(v)");
    exec(&mut db, "INSERT INTO blb VALUES (x'01'), (x'0100'), (x'00ff'), (x'02')");
    assert_rows(
        &mut db,
        "SELECT v FROM blb ORDER BY v",
        &[
            vec![blob(&[0x00, 0xff])],
            vec![blob(&[0x01])],
            vec![blob(&[0x01, 0x00])],
            vec![blob(&[0x02])],
        ],
    );
}

// ---- ORDER BY in an AGGREGATE query (over the post-aggregate relation) --------
//
// lang_select.html §4: in an aggregate query the ORDER BY is evaluated over the
// post-GROUP BY relation, exactly like the result columns — so a term may be a
// group key, an aggregate function, or an expression over them, WHETHER OR NOT the
// SELECT list projects it. (SQLite: "the ORDER BY clause is applied to the result
// set" of the aggregation, whose columns are the group keys and aggregates.) These
// pin that a grouping key / aggregate the SELECT list omits still orders the result;
// a genuinely projected column is unaffected (covered elsewhere).

/// `agg(cat TEXT, v INTEGER)`: two groups ('p' with v=1,3; 'q' with v=2), so per
/// group count/sum/min/max are all distinct and every ordering below is deterministic.
fn table_agg() -> Connection {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE agg(cat TEXT, v INTEGER)");
    exec(&mut db, "INSERT INTO agg VALUES ('p', 1), ('q', 2), ('p', 3)");
    db
}

#[test]
fn orderby_group_key_not_in_result_set() {
    // `ORDER BY cat` where `cat` is the GROUP BY key but is NOT selected: the term
    // resolves to the group key and orders the (count-only) result. p→2, q→1, so
    // ascending cat gives 2 then 1.
    let mut db = table_agg();
    assert_rows(
        &mut db,
        "SELECT count(*) FROM agg GROUP BY cat ORDER BY cat",
        &[vec![int(2)], vec![int(1)]],
    );
}

#[test]
fn orderby_group_key_not_in_result_set_desc() {
    // Same, DESC: q (count 1) before p (count 2).
    let mut db = table_agg();
    assert_rows(
        &mut db,
        "SELECT count(*) FROM agg GROUP BY cat ORDER BY cat DESC",
        &[vec![int(1)], vec![int(2)]],
    );
}

#[test]
fn orderby_aggregate_not_in_result_set() {
    // `ORDER BY count(*)` where the count is NOT selected: it is computed as an
    // aggregate and orders the result. count(q)=1 < count(p)=2, so ascending gives
    // q then p.
    let mut db = table_agg();
    assert_rows(
        &mut db,
        "SELECT cat FROM agg GROUP BY cat ORDER BY count(*)",
        &[vec![text("q")], vec![text("p")]],
    );
}

#[test]
fn orderby_aggregate_not_in_result_set_desc_with_tiebreak() {
    // A two-key aggregate ORDER BY, neither key selected: primary `count(*) DESC`
    // (p=2 before q=1), tie-broken by `cat ASC`. Distinct counts here, so the DESC
    // count alone fixes the order; the tiebreaker guards the multi-key path.
    let mut db = table_agg();
    assert_rows(
        &mut db,
        "SELECT cat FROM agg GROUP BY cat ORDER BY count(*) DESC, cat ASC",
        &[vec![text("p")], vec![text("q")]],
    );
}

#[test]
fn orderby_expression_over_group_key_not_in_result_set() {
    // An EXPRESSION over the group key (`cat || '!'`), not just the bare key, is a
    // fresh post-aggregate expression bound in the grouping context: 'p!' < 'q!', so
    // ascending gives p's count (2) then q's (1).
    let mut db = table_agg();
    assert_rows(
        &mut db,
        "SELECT count(*) FROM agg GROUP BY cat ORDER BY cat || '!'",
        &[vec![int(2)], vec![int(1)]],
    );
}

#[test]
fn orderby_aggregate_only_in_having_and_order_by() {
    // The ordering aggregate (`sum(v)`) appears ONLY in ORDER BY (HAVING uses a
    // different one): sum(p)=4, sum(q)=2, so ascending sum gives q then p.
    let mut db = table_agg();
    assert_rows(
        &mut db,
        "SELECT cat FROM agg GROUP BY cat HAVING count(*) >= 1 ORDER BY sum(v)",
        &[vec![text("q")], vec![text("p")]],
    );
}

#[test]
fn orderby_qualified_group_key_not_in_result_set() {
    // A table-qualified group key (`agg.cat`) in ORDER BY resolves the same as the
    // bare key. p→2, q→1 ascending.
    let mut db = table_agg();
    assert_rows(
        &mut db,
        "SELECT count(*) FROM agg GROUP BY agg.cat ORDER BY agg.cat",
        &[vec![int(2)], vec![int(1)]],
    );
}
