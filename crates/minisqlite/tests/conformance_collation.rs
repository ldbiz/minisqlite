//! Conformance tests for SQLite's built-in collating sequences: BINARY, NOCASE,
//! and RTRIM.
//!
//! Every expected value here is transcribed from the SQLite documentation, never
//! from what this engine returns:
//!   * `spec/sqlite-doc/datatype3.html` §7 "Collating Sequences" — the three
//!     built-in collating functions and their definitions, and §7.1 the rules
//!     for which collating function a comparison / ORDER BY uses.
//!   * `spec/sqlite-doc/datatype3.html` §7.2 "Collation Sequence Examples" — the
//!     `t1` fixture and its result sets, copied verbatim from the manual.
//!   * `spec/sqlite-doc/lang_expr.html` (`collateop`) — the COLLATE operator is a
//!     unary postfix operator of the highest precedence, so `x = y COLLATE NOCASE`
//!     parses as `x = (y COLLATE NOCASE)` and the named collation governs the
//!     whole comparison (§7.1 rule 1).
//!
//! COLLATE support in this engine may be partial. A failing case is the signal,
//! not a defect in the test: it keeps asserting the spec-correct value and is
//! never weakened. Only a case that HANGS or PANICS is `#[ignore]`d (in its
//! own test), because an ordinary mismatch must stay visible.

mod conformance;

use conformance::*;
use minisqlite::{Connection, Value};

// -----------------------------------------------------------------------------
// §7 — direct COLLATE-operator behaviour on bare string comparisons.
//
// With no table column involved, the only collation in play is the one named by
// an explicit COLLATE operator, or BINARY by default (§7.1 rule 3). A SQL
// comparison yields the integer 1 for true and 0 for false, so these assert an
// `int(0)` / `int(1)` scalar.
// -----------------------------------------------------------------------------

/// §7.1 rule 3 / §7 BINARY: with no COLLATE operator and no column operand the
/// comparison uses BINARY, which compares with memcmp() and is case-sensitive,
/// so lower-case 'abc' and upper-case 'ABC' are NOT equal.
#[test]
fn binary_is_default_and_case_sensitive() {
    eval_eq("'abc' = 'ABC'", int(0));
}

/// §7 NOCASE folds the 26 upper-case ASCII letters to lower case before
/// comparing, so 'abc' equals 'ABC'. The COLLATE operator on the right operand
/// selects NOCASE for the whole comparison (§7.1 rule 1).
#[test]
fn nocase_operator_on_right_operand_equates_case_variants() {
    eval_eq("'abc' = 'ABC' COLLATE NOCASE", int(1));
}

/// §7.1 rule 1: an explicit COLLATE on the LEFT operand selects NOCASE just the
/// same — the operator applies to the whole comparison regardless of side.
#[test]
fn nocase_operator_on_left_operand_equates_case_variants() {
    eval_eq("'ABC' COLLATE NOCASE = 'abc'", int(1));
}

/// §7 NOCASE governs ordering comparisons too, not just equality: after folding,
/// 'abc' < 'abd' (from 'ABD'), so the `<` is true.
#[test]
fn nocase_orders_case_folded_strings() {
    eval_eq("'abc' < 'ABD' COLLATE NOCASE", int(1));
}

/// §7 RTRIM is "the same as binary, except that trailing space characters are
/// ignored", so 'abc' equals 'abc   ' under RTRIM.
#[test]
fn rtrim_ignores_trailing_spaces() {
    eval_eq("'abc' = 'abc   ' COLLATE RTRIM", int(1));
}

/// §7.1 rule 1 / §7 BINARY: an explicit COLLATE BINARY forces the default byte
/// comparison; two byte-identical strings are equal under it.
#[test]
fn binary_operator_compares_identical_strings_equal() {
    eval_eq("'abc' = 'abc' COLLATE BINARY", int(1));
}

// ---- Additional unambiguous §7 consequences ---------------------------------

/// §7 NOCASE: "only ASCII characters are case folded". The non-ASCII letters
/// 'é' (U+00E9) and 'É' (U+00C9) are therefore left untouched by NOCASE and, as
/// distinct byte sequences, compare unequal even under NOCASE.
#[test]
fn nocase_folds_only_ascii_letters() {
    eval_eq("'é' = 'É' COLLATE NOCASE", int(0));
}

/// §7 RTRIM ignores only TRAILING spaces; leading spaces remain significant, so
/// '  abc' and 'abc' differ under RTRIM.
#[test]
fn rtrim_does_not_ignore_leading_spaces() {
    eval_eq("'  abc' = 'abc' COLLATE RTRIM", int(0));
}

/// §7 BINARY: a trailing space is a significant byte under BINARY, so 'abc' and
/// 'abc ' are not equal — the direct contrast to RTRIM above.
#[test]
fn binary_treats_trailing_space_as_significant() {
    eval_eq("'abc' = 'abc ' COLLATE BINARY", int(0));
}

// ---- §7.1 rule 1: leftmost explicit COLLATE wins a multi-COLLATE comparison --

/// §7.1: "If two or more COLLATE operator subexpressions appear anywhere in a
/// comparison, the left most explicit collating function is used"
/// (datatype3.html:852-856). Here the LEFT operand's NOCASE is leftmost, so the
/// whole comparison folds and 'abc' equals 'ABC' despite the right operand's
/// explicit BINARY. A "rightmost wins" bug — or "any BINARY present wins" — would
/// give 0.
#[test]
fn rule1_leftmost_collate_wins_nocase() {
    eval_eq("'abc' COLLATE NOCASE = 'ABC' COLLATE BINARY", int(1));
}

/// §7.1 (datatype3.html:852-856), the mirror image: swap the operands' COLLATEs
/// and BINARY becomes leftmost and governs, so 'abc' and 'ABC' are unequal (0).
/// Paired with the case above, this pins "leftmost" specifically — not merely
/// "an explicit NOCASE appears somewhere".
#[test]
fn rule1_leftmost_collate_wins_binary() {
    eval_eq("'abc' COLLATE BINARY = 'ABC' COLLATE NOCASE", int(0));
}

// ---- §7: collation is inert for non-string (BLOB) comparisons ----------------

/// §7: "Collating functions only matter when comparing string values ... BLOBs
/// are always compared byte-by-byte using memcmp()" (datatype3.html:806-808). A
/// COLLATE NOCASE on a BLOB comparison is therefore inert: x'616263' ("abc") and
/// x'414243' ("ABC") differ byte-wise and stay unequal (0), while identical bytes
/// stay equal (1). If NOCASE were wrongly folding the blob bytes, the first would
/// be 1.
#[test]
fn blob_comparison_is_bytewise_ignoring_collate() {
    eval_eq("x'616263' = x'414243' COLLATE NOCASE", int(0));
    eval_eq("x'616263' = x'616263' COLLATE NOCASE", int(1));
}

// -----------------------------------------------------------------------------
// §7.2 — the worked "Collation Sequence Examples" fixture.
//
// Transcribed verbatim from datatype3.html §7.2. Column `a` has no COLLATE
// clause so it defaults to BINARY (§7.1); `b` is BINARY, `c` is RTRIM, `d` is
// NOCASE. Every test below builds a fresh copy of the fixture so the cases are
// independent.
// -----------------------------------------------------------------------------

/// The §7.2 `t1` fixture: the exact schema and four rows from the manual.
///
/// The manual writes the column list with `/* ... */` comments naming each
/// column's collating sequence; those comments are documentation, not schema, so
/// the DDL here uses the comment-free form (column `a` has no COLLATE clause and
/// so defaults to BINARY). Keeping comments out means these tests isolate
/// collation behaviour instead of also depending on SQL comment parsing.
///
/// WHITESPACE IS LOAD-BEARING: this is an RTRIM/BINARY fixture, so the trailing
/// spaces INSIDE the quoted literals are data, not formatting —
/// c = 'abc  '/'abc'/'abc '/'ABC', b row 4 = 'abc ', d = 'abc'/'ABC'/'Abc'/'abc'
/// (per datatype3.html:900-903). The only insignificant spaces are the ones
/// OUTSIDE the quotes (after a comma). Do not "align" or "tidy" the literals: a
/// stray space slipping inside a quote would silently change the data and make
/// every dependent assertion test the wrong thing with no signal.
fn t1() -> Connection {
    let mut db = mem();
    exec(
        &mut db,
        "CREATE TABLE t1(x INTEGER PRIMARY KEY, a, b COLLATE BINARY, c COLLATE RTRIM, d COLLATE NOCASE)",
    );
    exec(&mut db, "INSERT INTO t1 VALUES(1,'abc','abc', 'abc  ','abc')");
    exec(&mut db, "INSERT INTO t1 VALUES(2,'abc','abc', 'abc',  'ABC')");
    exec(&mut db, "INSERT INTO t1 VALUES(3,'abc','abc', 'abc ', 'Abc')");
    exec(&mut db, "INSERT INTO t1 VALUES(4,'abc','abc ','ABC',  'abc')");
    db
}

/// Build the single-column `x` result set the §7.2 examples expect.
fn xs(ids: &[i64]) -> Vec<Vec<Value>> {
    ids.iter().map(|&x| vec![int(x)]).collect()
}

/// §7.2: `a = b` — both operands are columns; the left column `a` is BINARY, so
/// the comparison uses BINARY. Rows where a and b are byte-equal: 1, 2, 3.
#[test]
fn ex_a_eq_b_uses_binary() {
    let mut db = t1();
    assert_rows(&mut db, "SELECT x FROM t1 WHERE a = b ORDER BY x", &xs(&[1, 2, 3]));
}

/// §7.2: `a = b COLLATE RTRIM` — the explicit COLLATE RTRIM (§7.1 rule 1) wins
/// over the columns' BINARY, so trailing spaces are ignored and row 4 (b='abc ')
/// now matches too: 1, 2, 3, 4.
#[test]
fn ex_a_eq_b_collate_rtrim_uses_rtrim() {
    let mut db = t1();
    assert_rows(
        &mut db,
        "SELECT x FROM t1 WHERE a = b COLLATE RTRIM ORDER BY x",
        &xs(&[1, 2, 3, 4]),
    );
}

/// §7.2: `d = a` — the left column `d` is NOCASE (§7.1 rule 2), so case is
/// folded and every row's d matches 'abc': 1, 2, 3, 4.
#[test]
fn ex_d_eq_a_uses_nocase() {
    let mut db = t1();
    assert_rows(&mut db, "SELECT x FROM t1 WHERE d = a ORDER BY x", &xs(&[1, 2, 3, 4]));
}

/// §7.2: `a = d` — the left column `a` is BINARY and takes precedence over d's
/// NOCASE (§7.1 rule 2, "precedence to the left operand"), so only the rows
/// whose d is exactly 'abc' match: 1, 4.
#[test]
fn ex_a_eq_d_left_binary_wins() {
    let mut db = t1();
    assert_rows(&mut db, "SELECT x FROM t1 WHERE a = d ORDER BY x", &xs(&[1, 4]));
}

/// §7.1 rule 2, the unary-`+` clause (datatype3.html:836-838): "a column name
/// preceded by one or more unary '+' operators and/or CAST operators is still
/// considered a column name." So `+d` keeps column d's NOCASE and `+d = a`
/// behaves exactly like `d = a`: the leading `+` must NOT downgrade the
/// comparison to BINARY. Expected [1,2,3,4]; a BINARY downgrade would wrongly
/// yield [1,4] (the `a = d` answer above). Not in the §7.2 example block, but a
/// direct consequence of rule 2 and a common spot for engines to slip. (`rule2_`
/// names a §7.1-rule-derived case, vs the `ex_` §7.2 verbatim examples.)
#[test]
fn rule2_unary_plus_preserves_column_collation() {
    let mut db = t1();
    assert_rows(&mut db, "SELECT x FROM t1 WHERE +d = a ORDER BY x", &xs(&[1, 2, 3, 4]));
}

/// §7.1 rule 2, the CAST half of the same clause (datatype3.html:836-838): a
/// column preceded by a CAST is "still considered a column name", so
/// `CAST(d AS TEXT)` keeps column d's NOCASE and `CAST(d AS TEXT) = a` matches
/// `d = a` → [1,2,3,4]. CAST and unary `+` are commonly distinct code paths, so a
/// passing `+d` does not prove `CAST(d)`; a CAST that drops the column's collation
/// downgrades to BINARY and wrongly yields [1,4].
#[test]
fn rule2_cast_preserves_column_collation() {
    let mut db = t1();
    assert_rows(
        &mut db,
        "SELECT x FROM t1 WHERE CAST(d AS TEXT) = a ORDER BY x",
        &xs(&[1, 2, 3, 4]),
    );
}

/// §7.2: `'abc' = c` — the left operand is a literal (not a column, no COLLATE),
/// so the column operand `c`'s RTRIM applies (§7.1 rule 2): 1, 2, 3.
#[test]
fn ex_literal_eq_c_uses_rtrim() {
    let mut db = t1();
    assert_rows(&mut db, "SELECT x FROM t1 WHERE 'abc' = c ORDER BY x", &xs(&[1, 2, 3]));
}

/// §7.2: `c = 'abc'` — the left column `c` is RTRIM (§7.1 rule 2): 1, 2, 3.
#[test]
fn ex_c_eq_literal_uses_rtrim() {
    let mut db = t1();
    assert_rows(&mut db, "SELECT x FROM t1 WHERE c = 'abc' ORDER BY x", &xs(&[1, 2, 3]));
}

/// §7.2: `GROUP BY d` — grouping uses column d's NOCASE, so 'abc', 'ABC', 'Abc'
/// collapse into ONE group of four rows: count 4.
#[test]
fn ex_group_by_d_uses_nocase() {
    let mut db = t1();
    assert_rows(&mut db, "SELECT count(*) FROM t1 GROUP BY d ORDER BY 1", &xs(&[4]));
}

/// §7.2: `GROUP BY (d || '')` — the operand is an expression, not a column, so
/// BINARY is used (§7.1); 'abc' (rows 1,4), 'ABC' (row 2) and 'Abc' (row 3) are
/// three distinct groups. Counts ordered ascending: 1, 1, 2.
#[test]
fn ex_group_by_d_concat_uses_binary() {
    let mut db = t1();
    assert_rows(
        &mut db,
        "SELECT count(*) FROM t1 GROUP BY (d || '') ORDER BY 1",
        &xs(&[1, 1, 2]),
    );
}

/// §7.2: `ORDER BY c, x` — sorting by column c uses its RTRIM collation. Under
/// RTRIM all four c values ('abc  ','abc','abc ','ABC') tie except 'ABC', which
/// sorts first (upper-case 'A' < 'a'); x breaks the remaining tie: 4, 1, 2, 3.
#[test]
fn ex_order_by_c_uses_rtrim() {
    let mut db = t1();
    assert_rows(&mut db, "SELECT x FROM t1 ORDER BY c, x", &xs(&[4, 1, 2, 3]));
}

/// §7.2: `ORDER BY (c || ''), x` — the sort key is an expression, so BINARY is
/// used (§7.1): 'ABC' (row 4) < 'abc' (row 2) < 'abc ' (row 3) < 'abc  ' (row 1),
/// i.e. 4, 2, 3, 1.
#[test]
fn ex_order_by_c_concat_uses_binary() {
    let mut db = t1();
    assert_rows(&mut db, "SELECT x FROM t1 ORDER BY (c||''), x", &xs(&[4, 2, 3, 1]));
}

/// §7.2: `ORDER BY c COLLATE NOCASE, x` — the explicit COLLATE NOCASE overrides
/// c's RTRIM. Folded, the c values compare as 'abc'(2) < 'abc '(3) < 'abc  '(1)
/// with 'ABC'(4) folding to 'abc' and tying row 2 first by x: 2, 4, 3, 1.
#[test]
fn ex_order_by_c_collate_nocase() {
    let mut db = t1();
    assert_rows(
        &mut db,
        "SELECT x FROM t1 ORDER BY c COLLATE NOCASE, x",
        &xs(&[2, 4, 3, 1]),
    );
}

// -----------------------------------------------------------------------------
// UNIQUE constraints / indexes honour the key column's collating sequence.
//
// A UNIQUE constraint "is implemented by creating a unique index"
// (lang_createtable.html §3.6), and an index column compares "using the
// collating sequence defined for that column in the CREATE TABLE statement …
// or if no collating sequence is otherwise defined, the built-in BINARY"
// (lang_createindex.html §1.5), overridable by a COLLATE clause on the index
// column. Two rows therefore collide on a UNIQUE key when their key columns
// compare EQUAL *under that collation* (datatype3.html §7): under NOCASE 'abc'
// and 'ABC' are the same key, so the second is a duplicate SQLite rejects with
// a `UNIQUE constraint failed` error — a BINARY-only equality check would miss
// it and wrongly admit the row. Each case pairs the collated reject with a
// BINARY control that admits the same pair, so the tests pin the collation
// (not merely "any error"/"any success").
// -----------------------------------------------------------------------------

/// §3.6 + §1.5 + §7: a column declared `COLLATE NOCASE UNIQUE` treats 'abc' and
/// 'ABC' as one key, so inserting both is a UNIQUE violation and only the first
/// row survives.
#[test]
fn nocase_unique_column_rejects_case_variant_duplicate() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE u(a TEXT COLLATE NOCASE UNIQUE)");
    exec(&mut db, "INSERT INTO u VALUES ('abc')");
    assert_exec_error(&mut db, "INSERT INTO u VALUES ('ABC')");
    assert_rows(&mut db, "SELECT count(*) FROM u", &[vec![int(1)]]);
}

/// §7.1 rule 3 control for the case above: a column with the default BINARY
/// collation compares case-sensitively, so 'abc' and 'ABC' are DISTINCT keys and
/// both inserts succeed. Paired with the NOCASE test, this pins that the reject
/// there is the collation at work, not a blanket "second insert fails".
#[test]
fn binary_unique_column_admits_case_variant() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE u(a TEXT UNIQUE)");
    exec(&mut db, "INSERT INTO u VALUES ('abc')");
    exec(&mut db, "INSERT INTO u VALUES ('ABC')");
    assert_rows(&mut db, "SELECT count(*) FROM u", &[vec![int(2)]]);
}

/// §7 RTRIM ignores trailing spaces, so a `COLLATE RTRIM UNIQUE` column treats
/// 'abc' and 'abc ' as one key — the second insert violates the constraint.
#[test]
fn rtrim_unique_column_rejects_trailing_space_variant() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE u(a TEXT COLLATE RTRIM UNIQUE)");
    exec(&mut db, "INSERT INTO u VALUES ('abc')");
    assert_exec_error(&mut db, "INSERT INTO u VALUES ('abc   ')");
    assert_rows(&mut db, "SELECT count(*) FROM u", &[vec![int(1)]]);
}

/// §1.5: a COLLATE clause on the INDEX column overrides the column's declared
/// collation. Here column `a` is BINARY but the UNIQUE index is built COLLATE
/// NOCASE, so the index — and thus the uniqueness test — folds case: 'abc' then
/// 'ABC' is a violation even though the column itself is case-sensitive.
#[test]
fn unique_index_collate_override_rejects_case_variant() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a TEXT)");
    exec(&mut db, "CREATE UNIQUE INDEX i ON t(a COLLATE NOCASE)");
    exec(&mut db, "INSERT INTO t VALUES ('abc')");
    assert_exec_error(&mut db, "INSERT INTO t VALUES ('ABC')");
    assert_rows(&mut db, "SELECT count(*) FROM t", &[vec![int(1)]]);
}

/// §3.6: building a UNIQUE index over EXISTING data that already holds a
/// collation-duplicate must fail (the backfill applies the same per-column
/// collation as the insert path). Rows 'abc' and 'ABC' are distinct under the
/// column's BINARY collation but collide once a NOCASE unique index is created,
/// so the CREATE UNIQUE INDEX itself is the `UNIQUE constraint failed` error.
#[test]
fn create_unique_index_nocase_over_existing_duplicate_fails() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a TEXT)");
    exec(&mut db, "INSERT INTO t VALUES ('abc'),('ABC')");
    assert_exec_error(&mut db, "CREATE UNIQUE INDEX i ON t(a COLLATE NOCASE)");
}

/// §7 + §3.6: a multi-column UNIQUE uses EACH column's own collation. With
/// `a COLLATE NOCASE` and `b` BINARY, ('abc','x') then ('ABC','x') collides (a
/// folds, b equal) and is rejected, but ('ABC','y') differs in the BINARY column
/// b and is admitted — so the surviving rows are exactly the two distinct keys.
#[test]
fn multicolumn_unique_uses_per_column_collation() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(a TEXT COLLATE NOCASE, b TEXT, UNIQUE(a, b))");
    exec(&mut db, "INSERT INTO t VALUES ('abc','x')");
    assert_exec_error(&mut db, "INSERT INTO t VALUES ('ABC','x')");
    exec(&mut db, "INSERT INTO t VALUES ('ABC','y')");
    assert_rows(&mut db, "SELECT count(*) FROM t", &[vec![int(2)]]);
}

/// §3.6: "NULL values are considered distinct from all other values, including
/// other NULLs" — this holds under NOCASE too, so a NOCASE UNIQUE column admits
/// any number of NULLs (the collation never enters, NULLs never collide).
#[test]
fn nocase_unique_permits_multiple_nulls() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE u(a TEXT COLLATE NOCASE UNIQUE)");
    exec(&mut db, "INSERT INTO u VALUES (NULL)");
    exec(&mut db, "INSERT INTO u VALUES (NULL)");
    assert_rows(&mut db, "SELECT count(*) FROM u", &[vec![int(2)]]);
}

/// The UPDATE self-probe under collation: changing the ONLY row's value to a
/// case variant of itself is not a conflict (the row's own index entry is
/// excluded by rowid), so `UPDATE … SET a='ABC'` on the sole 'abc' row succeeds
/// and leaves the value folded-in-place.
#[test]
fn nocase_unique_update_to_own_case_variant_is_allowed() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE u(a TEXT COLLATE NOCASE UNIQUE)");
    exec(&mut db, "INSERT INTO u VALUES ('abc')");
    exec(&mut db, "UPDATE u SET a = 'ABC'");
    assert_rows(&mut db, "SELECT a FROM u", &[vec![text("ABC")]]);
}

/// The UPDATE-into-conflict counterpart: with two distinct rows 'abc' and 'def',
/// updating the 'def' row to 'ABC' collides with the existing 'abc' under NOCASE
/// and is rejected, leaving both original rows intact.
#[test]
fn nocase_unique_update_into_existing_key_is_rejected() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE u(a TEXT COLLATE NOCASE UNIQUE)");
    exec(&mut db, "INSERT INTO u VALUES ('abc'),('def')");
    assert_exec_error(&mut db, "UPDATE u SET a = 'ABC' WHERE a = 'def'");
    assert_rows(&mut db, "SELECT a FROM u ORDER BY a", &[vec![text("abc")], vec![text("def")]]);
}
