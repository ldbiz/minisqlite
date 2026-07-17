//! Conformance battery — comparison operators and the IS family.
//!
//! Every expected value here is TRANSCRIBED FROM THE SPEC (`spec/sqlite-doc/`),
//! never from what the engine returns. Sources:
//!   - `lang_expr.html` §2 "Operators, and Parse-Affecting Attributes":
//!       * general rule (near `#binaryops`): "All operators generally evaluate to
//!         NULL when any operand is NULL" — so `=`, `<>`, `<`, `<=`, `>`, `>=`,
//!         `==`, `!=` all yield NULL when either side is NULL.
//!       * `#isisnot`: IS / IS NOT act like `=` / `!=` EXCEPT for NULL, where they
//!         compare NULL as an ordinary value and can NEVER evaluate to NULL.
//!       * `#isdf`: `IS NOT DISTINCT FROM` == `IS`; `IS DISTINCT FROM` == `IS NOT`.
//!       * operator table: `ISNULL`, `NOTNULL`, `NOT NULL` are postfix null tests.
//!   - `datatype3.html` §2.1 "Boolean Datatype": a boolean result is the INTEGER
//!     0 (false) or 1 (true) — every comparison below returns int(0)/int(1)/NULL.
//!   - `datatype3.html` §4.1 "Sort Order": storage-class order is
//!     NULL < INTEGER/REAL < TEXT < BLOB; two numerics compare numerically; two
//!     TEXT values compare by collating sequence (BINARY, i.e. unsigned byte
//!     order, for these literal operands per §7.1 `#colrules` rule 3: neither
//!     operand is a column and none carries COLLATE); two BLOBs by `memcmp()`
//!     (unsigned byte compare, shorter-prefix sorts first).
//!   - `datatype3.html` §4.2 "Type Conversions Prior To Comparison": affinity is
//!     applied per operand affinity; a LITERAL has NO affinity, so comparing two
//!     literals applies no conversion ("compared as is") and the §4.1 class order
//!     decides. Hence `1 = '1'` is 0 (INTEGER vs TEXT are different classes) and
//!     `1 < 'a'` is 1 (INTEGER sorts before TEXT).
//!
//! IMPORTANT distinction: §4.1's "NULL is less than any other value" governs
//! ORDER BY / sorting, NOT the comparison OPERATORS. `NULL < 1` is NULL (the
//! operand-NULL rule of lang_expr §2), it is not 1.
//!
//! If a case fails because the engine disagrees with the spec, the assertion
//! STAYS spec-correct (it fails) — it is never weakened to match the engine.

mod conformance;

use conformance::*;

// =============================================================================
// Equality:  `=`  and its synonym  `==`
// =============================================================================

#[test]
fn eq_numeric_integer() {
    // Same INTEGER class -> numeric comparison (datatype3 §4.1).
    eval_eq("1=1", int(1));
    eval_eq("1=2", int(0));
    eval_eq("2=2", int(1));
    eval_eq("0=0", int(1));
    eval_eq("42=42", int(1));
    eval_eq("42=43", int(0));
    eval_eq("-1=-1", int(1));
    eval_eq("-1=1", int(0));
}

#[test]
fn eq_double_equals_is_synonym_for_equals() {
    // lang_expr §2: `==` is an alternate spelling of `=`.
    eval_eq("1==1", int(1));
    eval_eq("1==2", int(0));
    eval_eq("2==2", int(1));
    eval_eq("'a'=='a'", int(1));
    eval_eq("'a'=='b'", int(0));
}

#[test]
fn eq_numeric_real_and_mixed() {
    // INTEGER vs REAL is a numeric (not cross-class) comparison: both are the
    // numeric storage classes (datatype3 §4.1).
    eval_eq("2.0=2", int(1));
    eval_eq("2=2.0", int(1));
    eval_eq("2.5=2.5", int(1));
    eval_eq("2.5=2", int(0));
    eval_eq("1.0==1", int(1));
    eval_eq("3.14=3.14", int(1));
    eval_eq("0.0=0", int(1));
}

#[test]
fn eq_text_binary_collation() {
    // Two TEXT values: BINARY (byte-order) collation for non-column expressions.
    eval_eq("'abc'='abc'", int(1));
    eval_eq("'abc'='abd'", int(0));
    eval_eq("'a'='a'", int(1));
    eval_eq("'a'='b'", int(0));
    eval_eq("''=''", int(1));
    eval_eq("'A'='a'", int(0)); // case-sensitive: 0x41 != 0x61
}

#[test]
fn eq_with_null_yields_null() {
    // lang_expr §2: any operand NULL -> the comparison is NULL.
    eval_eq("1=NULL", null());
    eval_eq("NULL=1", null());
    eval_eq("NULL=NULL", null());
    eval_eq("1==NULL", null());
    eval_eq("NULL==NULL", null());
    eval_eq("'a'=NULL", null());
    eval_eq("NULL='a'", null());
}

#[test]
fn eq_cross_class_is_never_equal() {
    // Literals have no affinity (datatype3 §4.2) so no conversion happens; a
    // value of one storage class never equals a value of another (§4.1).
    eval_eq("1='1'", int(0)); // INTEGER vs TEXT
    eval_eq("'1'=1", int(0));
    eval_eq("1=x'01'", int(0)); // INTEGER vs BLOB
    eval_eq("'a'=x'61'", int(0)); // TEXT vs BLOB (x'61' is byte 'a' but class differs)
    eval_eq("2.0='2.0'", int(0)); // REAL vs TEXT
}

// =============================================================================
// Inequality:  `<>`  and its synonym  `!=`
// =============================================================================

#[test]
fn ne_angle_numeric() {
    eval_eq("1<>2", int(1));
    eval_eq("1<>1", int(0));
    eval_eq("2<>2", int(0));
    eval_eq("2<>3", int(1));
    eval_eq("-1<>1", int(1));
}

#[test]
fn ne_bang_is_synonym() {
    // lang_expr §2: `!=` is an alternate spelling of `<>`.
    eval_eq("1!=1", int(0));
    eval_eq("1!=2", int(1));
    eval_eq("2!=2", int(0));
}

#[test]
fn ne_numeric_real() {
    eval_eq("2.0<>2", int(0)); // numerically equal -> not-equal is false
    eval_eq("2.5<>2", int(1));
    eval_eq("2.5!=2.5", int(0));
}

#[test]
fn ne_text() {
    eval_eq("'a'<>'b'", int(1));
    eval_eq("'a'<>'a'", int(0));
    eval_eq("'a'!='b'", int(1));
    eval_eq("'a'!='a'", int(0));
    eval_eq("'abc'<>'abd'", int(1));
}

#[test]
fn ne_with_null_yields_null() {
    eval_eq("1<>NULL", null());
    eval_eq("NULL<>1", null());
    eval_eq("NULL<>NULL", null());
    eval_eq("1!=NULL", null());
    eval_eq("NULL!=NULL", null());
}

#[test]
fn ne_cross_class_is_always_unequal() {
    // Different storage classes are never equal, so `<>` / `!=` are true.
    eval_eq("1<>'1'", int(1));
    eval_eq("1!='1'", int(1));
    eval_eq("'a'<>x'61'", int(1));
}

// =============================================================================
// Ordering:  `<`  `<=`  `>`  `>=`
// =============================================================================

#[test]
fn lt_numeric() {
    eval_eq("1<2", int(1));
    eval_eq("2<1", int(0));
    eval_eq("2<2", int(0));
    eval_eq("-1<0", int(1));
    eval_eq("0<-1", int(0));
}

#[test]
fn le_numeric() {
    eval_eq("2<=2", int(1));
    eval_eq("1<=2", int(1));
    eval_eq("2<=1", int(0));
    eval_eq("-1<=-1", int(1));
}

#[test]
fn gt_numeric() {
    eval_eq("3>2", int(1));
    eval_eq("2>3", int(0));
    eval_eq("3>3", int(0));
    eval_eq("0>-1", int(1));
}

#[test]
fn ge_numeric() {
    eval_eq("2>=3", int(0));
    eval_eq("3>=3", int(1));
    eval_eq("3>=2", int(1));
    eval_eq("-1>=0", int(0));
}

#[test]
fn ordering_is_numeric_not_lexicographic() {
    // Numeric operands compare by magnitude, NOT as strings ("2" < "10" lexically
    // would be false; numerically 2 < 10 is true). datatype3 §4.1.
    eval_eq("2<10", int(1));
    eval_eq("10>2", int(1));
    eval_eq("10<=2", int(0));
    eval_eq("100>=20", int(1));
    eval_eq("9<10", int(1));
}

#[test]
fn ordering_real() {
    eval_eq("2.0<2.5", int(1));
    eval_eq("2.5>2.0", int(1));
    eval_eq("2.5>=2.5", int(1));
    eval_eq("2.5<=2.5", int(1));
    eval_eq("1.5<2", int(1)); // REAL vs INTEGER, still numeric
    eval_eq("2<1.5", int(0));
    eval_eq("-1.0<0", int(1));
}

#[test]
fn ordering_text_binary() {
    // BINARY collation: byte-order comparison of TEXT.
    eval_eq("'a'<'b'", int(1));
    eval_eq("'b'<'a'", int(0));
    eval_eq("'a'<='a'", int(1));
    eval_eq("'a'<='b'", int(1));
    eval_eq("'b'>'a'", int(1));
    eval_eq("'a'>'b'", int(0));
    eval_eq("'a'>='b'", int(0));
    eval_eq("'b'>='a'", int(1));
    eval_eq("'abc'<'abd'", int(1));
}

#[test]
fn ordering_text_prefix_shorter_sorts_first() {
    // A prefix sorts before the longer string under BINARY: 'ab' < 'abc'.
    eval_eq("'ab'<'abc'", int(1));
    eval_eq("'abc'<'ab'", int(0));
    eval_eq("'abc'>'ab'", int(1));
}

#[test]
fn ordering_text_case_sensitive_binary() {
    // Uppercase ASCII (0x41..) sorts before lowercase (0x61..) under BINARY.
    eval_eq("'A'<'a'", int(1));
    eval_eq("'Z'<'a'", int(1));
    eval_eq("'a'<'A'", int(0));
}

#[test]
fn ordering_with_null_yields_null() {
    // The comparison OPERATORS yield NULL on a NULL operand (lang_expr §2). This
    // is distinct from ORDER BY, where §4.1 sorts NULL first.
    eval_eq("NULL<1", null());
    eval_eq("1<NULL", null());
    eval_eq("NULL<=1", null());
    eval_eq("1<=NULL", null());
    eval_eq("NULL>1", null());
    eval_eq("1>NULL", null());
    eval_eq("NULL>=1", null());
    eval_eq("1>=NULL", null());
    eval_eq("NULL<NULL", null());
    eval_eq("NULL<'a'", null());
    eval_eq("NULL>x'00'", null());
}

// =============================================================================
// IS / IS NOT   (lang_expr §2 #isisnot)
// =============================================================================

#[test]
fn is_operator_non_null_like_equals() {
    // With no NULL operand, IS behaves like `=`.
    eval_eq("1 IS 1", int(1));
    eval_eq("1 IS 2", int(0));
    eval_eq("2 IS 2", int(1));
    eval_eq("'a' IS 'a'", int(1));
    eval_eq("'a' IS 'b'", int(0));
    eval_eq("2.0 IS 2", int(1)); // like `=`: numeric comparison
}

#[test]
fn is_not_operator_non_null_like_not_equals() {
    // With no NULL operand, IS NOT behaves like `!=`.
    eval_eq("1 IS NOT 2", int(1));
    eval_eq("1 IS NOT 1", int(0));
    eval_eq("'a' IS NOT 'b'", int(1));
    eval_eq("'a' IS NOT 'a'", int(0));
}

#[test]
fn is_operator_with_null() {
    // Both NULL -> IS is 1; exactly one NULL -> IS is 0.
    eval_eq("NULL IS NULL", int(1));
    eval_eq("1 IS NULL", int(0));
    eval_eq("NULL IS 1", int(0));
    eval_eq("'a' IS NULL", int(0));
    eval_eq("NULL IS 'a'", int(0));
}

#[test]
fn is_not_operator_with_null() {
    // Both NULL -> IS NOT is 0; exactly one NULL -> IS NOT is 1.
    eval_eq("NULL IS NOT NULL", int(0));
    eval_eq("1 IS NOT NULL", int(1));
    eval_eq("NULL IS NOT 1", int(1));
    eval_eq("'a' IS NOT NULL", int(1));
}

#[test]
fn is_operator_cross_class() {
    // IS mirrors `=`: different storage classes are not equal.
    eval_eq("1 IS '1'", int(0));
    eval_eq("1 IS NOT '1'", int(1));
}

#[test]
fn is_and_is_not_never_evaluate_to_null() {
    // lang_expr §2 #isisnot: "It is not possible for an IS or IS NOT expression
    // to evaluate to NULL." Assert the result is a concrete value even when an
    // operand is NULL.
    assert!(!value_eq(&eval("1 IS NULL"), &null()));
    assert!(!value_eq(&eval("NULL IS NULL"), &null()));
    assert!(!value_eq(&eval("NULL IS 1"), &null()));
    assert!(!value_eq(&eval("1 IS NOT NULL"), &null()));
    assert!(!value_eq(&eval("NULL IS NOT NULL"), &null()));
}

// =============================================================================
// IS [NOT] DISTINCT FROM   (lang_expr §2 #isdf)
// =============================================================================

#[test]
fn is_not_distinct_from_equals_is() {
    // `IS NOT DISTINCT FROM` is an alternate spelling of `IS`.
    eval_eq("1 IS NOT DISTINCT FROM 1", int(1));
    eval_eq("1 IS NOT DISTINCT FROM 2", int(0));
    eval_eq("NULL IS NOT DISTINCT FROM NULL", int(1));
    eval_eq("1 IS NOT DISTINCT FROM NULL", int(0));
    eval_eq("NULL IS NOT DISTINCT FROM 1", int(0));
    eval_eq("'a' IS NOT DISTINCT FROM 'a'", int(1));
}

#[test]
fn is_distinct_from_equals_is_not() {
    // `IS DISTINCT FROM` means the same as `IS NOT`.
    eval_eq("1 IS DISTINCT FROM NULL", int(1));
    eval_eq("1 IS DISTINCT FROM 1", int(0));
    eval_eq("1 IS DISTINCT FROM 2", int(1));
    eval_eq("NULL IS DISTINCT FROM NULL", int(0));
    eval_eq("NULL IS DISTINCT FROM 1", int(1));
    eval_eq("'a' IS DISTINCT FROM 'b'", int(1));
}

// =============================================================================
// ISNULL / NOTNULL / NOT NULL   (postfix null tests, lang_expr §2 operator table)
// =============================================================================

#[test]
fn isnull_postfix() {
    eval_eq("NULL ISNULL", int(1));
    eval_eq("1 ISNULL", int(0));
    eval_eq("0 ISNULL", int(0));
    eval_eq("'a' ISNULL", int(0));
    eval_eq("x'00' ISNULL", int(0));
}

#[test]
fn notnull_postfix() {
    eval_eq("1 NOTNULL", int(1));
    eval_eq("NULL NOTNULL", int(0));
    eval_eq("'a' NOTNULL", int(1));
    eval_eq("0 NOTNULL", int(1));
}

#[test]
fn not_null_postfix() {
    // `x NOT NULL` is the two-word spelling of `x NOTNULL`.
    eval_eq("1 NOT NULL", int(1));
    eval_eq("NULL NOT NULL", int(0));
    eval_eq("'a' NOT NULL", int(1));
}

// =============================================================================
// Cross-class ordering   (datatype3 §4.1: NULL < INTEGER/REAL < TEXT < BLOB)
// =============================================================================

#[test]
fn cross_class_integer_vs_text() {
    // INTEGER sorts before TEXT regardless of the numeric/textual content.
    eval_eq("1 < 'a'", int(1));
    eval_eq("'a' > 1", int(1));
    eval_eq("'a' < 1", int(0));
    eval_eq("1 > 'a'", int(0));
    eval_eq("100 > 'a'", int(0)); // still INTEGER < TEXT
    eval_eq("100 < 'a'", int(1));
    eval_eq("1 <= 'a'", int(1));
    eval_eq("'a' >= 1", int(1));
    eval_eq("'a' <= 1", int(0));
}

#[test]
fn cross_class_real_vs_text() {
    // REAL is a numeric class, so it also sorts before TEXT.
    eval_eq("9e99 < 'a'", int(1));
    eval_eq("'a' > 9e99", int(1));
    eval_eq("2.5 < 'a'", int(1));
    eval_eq("'a' < 2.5", int(0));
}

#[test]
fn cross_class_text_vs_blob() {
    // TEXT sorts before BLOB, whatever the bytes are.
    eval_eq("'a' < x'00'", int(1));
    eval_eq("'z' < x'00'", int(1)); // content irrelevant: TEXT always < BLOB
    eval_eq("x'00' > 'z'", int(1));
    eval_eq("x'00' < 'z'", int(0));
    eval_eq("'a' <= x'00'", int(1));
    eval_eq("x'ff' >= 'z'", int(1));
}

#[test]
fn cross_class_numeric_vs_blob() {
    // INTEGER/REAL sort before BLOB.
    eval_eq("1 < x'00'", int(1));
    eval_eq("x'00' > 1", int(1));
    eval_eq("1.5 < x'00'", int(1));
    eval_eq("x'00' < 1", int(0));
}

#[test]
fn cross_class_null_operator_still_null() {
    // NULL as an operand of an ordering operator is still NULL, even against a
    // higher storage class (contrast with ORDER BY sorting).
    eval_eq("NULL < x'00'", null());
    eval_eq("NULL < 'a'", null());
    eval_eq("x'00' > NULL", null());
    eval_eq("'a' >= NULL", null());
}

// =============================================================================
// Same-class BLOB comparisons — memcmp()   (datatype3 §4.1: "When two BLOB
// values are compared, the result is determined using memcmp()")
// =============================================================================

#[test]
fn blob_vs_blob_equality() {
    // Two BLOBs are equal iff they hold identical bytes.
    eval_eq("x'01'=x'01'", int(1));
    eval_eq("x'01'=x'02'", int(0));
    eval_eq("x'0102'=x'0102'", int(1));
    eval_eq("x'0102'=x'0103'", int(0));
    eval_eq("x'01'<>x'02'", int(1));
    eval_eq("x'01'<>x'01'", int(0));
    eval_eq("x'01'!=x'02'", int(1));
    eval_eq("x'01'!=x'01'", int(0));
}

#[test]
fn blob_vs_blob_ordering_memcmp() {
    // memcmp() is an UNSIGNED byte compare, and when one blob is a prefix of the
    // other the shorter one sorts first.
    eval_eq("x'00'<x'01'", int(1));
    eval_eq("x'01'>x'00'", int(1));
    eval_eq("x'01'<x'00'", int(0));
    eval_eq("x'01'<=x'01'", int(1));
    eval_eq("x'01'>=x'01'", int(1));
    eval_eq("x'0100'>x'00'", int(1)); // first differing byte (0x01 > 0x00) decides
    eval_eq("x'00'<x'0000'", int(1)); // x'00' is a prefix of x'0000' -> shorter first
    eval_eq("x'0000'>x'00'", int(1));
    eval_eq("x'ff'>x'01'", int(1)); // unsigned: 0xff (255) > 0x01, NOT signed -1 < 1
    eval_eq("x'80'>x'7f'", int(1)); // unsigned: 0x80 (128) > 0x7f (127)
}

#[test]
fn blob_with_null_operand_yields_null() {
    // A NULL operand still makes the comparison NULL, BLOB class notwithstanding.
    eval_eq("x'00'=NULL", null());
    eval_eq("x'00'<>NULL", null());
    eval_eq("x'00'<NULL", null());
}
