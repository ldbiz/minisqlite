//! Conformance battery for the SQLite JSON **operators** `->` and `->>`
//! (`spec/sqlite-doc/json1.html` §4.10, "The -> and ->> operators").
//!
//! This is a DIFFERENT surface from the JSON *functions* covered in
//! `conformance_json.rs` (json_extract, json, json_array, ...). Both the functions and the
//! `->` / `->>` OPERATORS are implemented; this file is the operators' spec-correct RESULT
//! coverage. It had none before they landed — the operators were parse-rejected and the
//! only prior mention asserted that parser error (the old gap, not the documented result).
//!
//! Every expected value below is TRANSCRIBED FROM THE SPEC — the JSON literal plus the
//! documented rules — never from what this engine returns. The rules pinned (json1.html
//! §4.10 unless noted):
//!
//!   * `->` returns a text JSON REPRESENTATION of the selected subcomponent, or NULL if it
//!     does not exist. So a selected JSON *string* comes back QUOTED
//!     (`'{"a":"xyz"}' -> '$.a'` -> the 5-char text `"xyz"`), and a numeric/array/object
//!     subcomponent comes back as its JSON text.
//!   * `->>` returns an SQL REPRESENTATION (TEXT / INTEGER / REAL / NULL) of the same
//!     subcomponent, or NULL if it does not exist. So the same string comes back DEQUOTED
//!     (`->> '$.a'` -> the 3-char text `xyz`), a numeric subcomponent comes back as an SQL
//!     INTEGER or REAL, and a JSON boolean as an SQL INTEGER 1 (true) or 0 (false) — the
//!     value classes §4.8 json_extract enumerates for a single path. (The SQL representation
//!     of an array/object is its JSON text as a TEXT value — §4.8, "a text representation
//!     for JSON object and array values".)
//!   * Right-operand abbreviation (§4.10, "the ... operators also accept an alphanumeric
//!     text object label or integer array index"): an alphanumeric text label `X` is the
//!     JSON path `'$.X'`; a non-negative integer `N` is the JSON path `'$[N]'`; a negative
//!     integer `-K` is `'$[#-K]'` (SQLite 3.47.0+). A right operand beginning with `$` is a
//!     full JSON path.
//!   * A path/label/index that matches nothing -> NULL. A JSON `null` subcomponent ->
//!     JSON text `null` under `->`, SQL NULL under `->>`.
//!   * A SQL NULL left operand -> NULL (§3.1: "NULL values are interpreted as JSON nulls",
//!     so nothing is selected -> §4.10 absent-subcomponent -> NULL).
//!   * `->` and `->>` chain left-to-right (verbatim example `-> 'c' -> 2 ->> 'f'`).
//!
//! STATUS: the `->` / `->>` operators are IMPLEMENTED and this battery passes. The parser
//! folds `X -> P` / `X ->> P` into two-argument calls of the sentinel-named scalar operators
//! and the evaluator returns the spec-correct RESULT. Each test asserts that spec-transcribed
//! value (never the engine's output), so the assertions double as a regression gate — a
//! re-broken operator makes them fail. The spec-correct assertion is never weakened to
//! pass, and none of these are `#[ignore]`d.
//!
//! MALFORMED-JSON left operand — §3.1 says a JSON function fed text that is not well-formed
//! JSON "will usually throw an error" (and `->`/`->>` are not among the
//! json_valid()/json_quote()/json_error_position() exceptions), so the spec-correct result
//! is an ERROR. Now that the operators are live this is a genuine RUNTIME error (via
//! `parse_json_arg`); it is covered by the crate-level unit tests in `minisqlite-functions`'
//! `json/operators.rs` (`malformed_document_is_an_error`) rather than duplicated here.
//!
//! JSON SQL is quote-heavy, so SQL and expected JSON text use Rust RAW strings (`r#"..."#`)
//! where they contain `"`, matching the convention in `conformance_json.rs`.

mod conformance;
use conformance::*;

// ---------------------------------------------------------------------------
// The core distinction on a STRING subcomponent: `->` yields the JSON
// representation (QUOTED), `->>` yields the SQL representation (DEQUOTED).
// json1.html §4.10 verbatim examples:
//   '{"a":"xyz"}' -> '$.a'  -> '"xyz"'
//   '{"a":"xyz"}' ->> '$.a' -> 'xyz'
// ---------------------------------------------------------------------------

/// §4.10: `->` returns the JSON representation of a string subcomponent, i.e. WITH the
/// surrounding double-quotes (the 5-character text `"xyz"`). Verbatim doc example.
#[test]
fn arrow_string_returns_quoted_json_text() {
    eval_eq(r#"'{"a":"xyz"}' -> '$.a'"#, text(r#""xyz""#));
}

/// §4.10: `->>` returns the SQL representation of a string subcomponent, i.e. the dequoted
/// 3-character text `xyz`. Verbatim doc example; the `->` vs `->>` contrast on the same
/// input is the headline behavior of these operators.
#[test]
fn arrow2_string_returns_dequoted_sql_text() {
    eval_eq(r#"'{"a":"xyz"}' ->> '$.a'"#, text("xyz"));
}

// ---------------------------------------------------------------------------
// Right-operand ABBREVIATION: a bare alphanumeric label X is the path '$.X'.
// json1.html §4.10: "If the right operand is an alphanumeric text label X, then it
// is interpreted as the JSON path '$.X'."
// ---------------------------------------------------------------------------

/// §4.10 abbreviation: the bare label `'a'` means path `'$.a'`, so `-> 'a'` equals
/// `-> '$.a'` and returns the quoted JSON string `"xyz"`.
#[test]
fn arrow_bare_label_means_dollar_dot_label() {
    eval_eq(r#"'{"a":"xyz"}' -> 'a'"#, text(r#""xyz""#));
}

/// §4.10 abbreviation: the bare label `'a'` under `->>` returns the dequoted SQL text `xyz`,
/// identical to `->> '$.a'`.
#[test]
fn arrow2_bare_label_means_dollar_dot_label() {
    eval_eq(r#"'{"a":"xyz"}' ->> 'a'"#, text("xyz"));
}

/// §4.10 (verbatim): a full `$`-path selecting an array subcomponent returns that array's
/// JSON text under `->`.
#[test]
fn arrow_dollar_path_selects_array_json_text() {
    eval_eq(
        r#"'{"a":2,"c":[4,5,{"f":7}]}' -> '$.c'"#,
        text(r#"[4,5,{"f":7}]"#),
    );
}

/// §4.10 (verbatim): the bare label `'c'` selects the SAME array as `'$.c'` above — pinning
/// the documented equivalence between an alphanumeric label and the `$.label` path form.
#[test]
fn arrow_bare_label_equals_dollar_path_for_array() {
    eval_eq(
        r#"'{"a":2,"c":[4,5,{"f":7}]}' -> 'c'"#,
        text(r#"[4,5,{"f":7}]"#),
    );
}

// ---------------------------------------------------------------------------
// A NUMERIC subcomponent: `->` renders it as JSON text, `->>` returns an SQL INTEGER.
// (json1.html §4.10 + §4.8 json_extract: "INTEGER or REAL for a JSON numeric value".)
// ---------------------------------------------------------------------------

/// §4.10: `->` on the numeric field `a=2` returns the JSON representation of the number,
/// i.e. the text `2`.
#[test]
fn arrow_numeric_field_returns_json_text() {
    eval_eq(r#"'{"a":2}' -> 'a'"#, text("2"));
}

/// §4.10 / §4.8: `->>` on the numeric field `a=2` returns the SQL INTEGER 2 (not the text
/// `2`) — the storage class matters, so this asserts `int`, not `text`.
#[test]
fn arrow2_numeric_field_returns_sql_integer() {
    eval_eq(r#"'{"a":2}' ->> 'a'"#, int(2));
}

/// §4.10: `->` on a REAL numeric field `a=2.5` returns the JSON text `2.5`.
#[test]
fn arrow_real_field_returns_json_text() {
    eval_eq(r#"'{"a":2.5}' -> 'a'"#, text("2.5"));
}

/// §4.10 / §4.8: `->>` on a REAL numeric field returns an SQL REAL (json_extract yields
/// "INTEGER or REAL for a JSON numeric value"). Asserts the `real` storage class, distinct
/// from `int` and `text` — 2.5 is exactly representable so the comparison is exact.
#[test]
fn arrow2_real_field_returns_sql_real() {
    eval_eq(r#"'{"a":2.5}' ->> 'a'"#, real(2.5));
}

// ---------------------------------------------------------------------------
// A JSON BOOLEAN subcomponent. §4.8 json_extract: "an INTEGER zero for a JSON false value,
// an INTEGER one for a JSON true value" — so `->>` maps true/false to SQL 1/0, while `->`
// (JSON representation) renders the literal text `true`/`false`. This is the same
// JSON-representation-vs-SQL-representation split as the numeric and null cases, for a value
// class SQLite has no dedicated storage type for. (`->` rendering derived from the §4.10
// "text JSON representation" rule; the doc gives no verbatim boolean example.)
// ---------------------------------------------------------------------------

/// §4.10: `->` on the boolean field `a=true` returns the JSON text `true` (not SQL 1).
#[test]
fn arrow_json_true_returns_text_true() {
    eval_eq(r#"'{"a":true}' -> 'a'"#, text("true"));
}

/// §4.8: `->>` on the boolean field `a=true` returns the SQL INTEGER 1.
#[test]
fn arrow2_json_true_returns_sql_integer_one() {
    eval_eq(r#"'{"a":true}' ->> 'a'"#, int(1));
}

/// §4.8: `->>` on the boolean field `a=false` returns the SQL INTEGER 0.
#[test]
fn arrow2_json_false_returns_sql_integer_zero() {
    eval_eq(r#"'{"a":false}' ->> 'a'"#, int(0));
}

// ---------------------------------------------------------------------------
// An INTEGER right operand selects an array element: N -> path '$[N]'.
// json1.html §4.10 verbatim examples:
//   '[11,22,33,44]' -> 3   -> '44'
//   '[11,22,33,44]' ->> 3  -> 44
// ---------------------------------------------------------------------------

/// §4.10 (verbatim): integer right operand `3` selects array index 3; `->` returns that
/// element's JSON text `44`.
#[test]
fn arrow_integer_index_returns_json_text() {
    eval_eq(r#"'[11,22,33,44]' -> 3"#, text("44"));
}

/// §4.10 (verbatim): integer right operand `3` under `->>` returns the SQL INTEGER 44.
#[test]
fn arrow2_integer_index_returns_sql_integer() {
    eval_eq(r#"'[11,22,33,44]' ->> 3"#, int(44));
}

/// §4.10: integer right operand `0` selects the FIRST array element (path `'$[0]'`); on
/// `[3,2,1]` that is `3`, returned as an SQL INTEGER by `->>`. Pins the index-0 edge and
/// that `0` is a valid (non-negative) index, not a no-op.
#[test]
fn arrow2_integer_index_zero_selects_first_element() {
    eval_eq(r#"'[3,2,1]' ->> 0"#, int(3));
}

// ---------------------------------------------------------------------------
// A full `$`-path drilling through objects and arrays. json1.html §4.10 verbatim:
//   '{"a":2,"c":[4,5,{"f":7}]}' -> '$.c[2].f'   -> '7'
//   '{"a":2,"c":[4,5,{"f":7}]}' ->> '$.c[2].f'  -> 7
// ---------------------------------------------------------------------------

/// §4.10 (verbatim): a deep path `'$.c[2].f'` reaches the leaf `7`; `->` returns its JSON
/// text `7`.
#[test]
fn arrow_deep_path_returns_json_text() {
    eval_eq(r#"'{"a":2,"c":[4,5,{"f":7}]}' -> '$.c[2].f'"#, text("7"));
}

/// §4.10 (verbatim): the same deep path under `->>` returns the SQL INTEGER 7.
#[test]
fn arrow2_deep_path_returns_sql_integer() {
    eval_eq(r#"'{"a":2,"c":[4,5,{"f":7}]}' ->> '$.c[2].f'"#, int(7));
}

/// §4.10 (verbatim): the path `'$'` selects the whole document, and `->` returns its JSON
/// text. The input is already in SQLite's canonical minified form, so the rendering is
/// byte-for-byte the input.
#[test]
fn arrow_dollar_selects_whole_document() {
    eval_eq(
        r#"'{"a":2,"c":[4,5,{"f":7}]}' -> '$'"#,
        text(r#"{"a":2,"c":[4,5,{"f":7}]}"#),
    );
}

// ---------------------------------------------------------------------------
// A NON-scalar (array) subcomponent under `->` vs `->>`. `->` returns its JSON text;
// `->>` returns the SQL representation, which for an array/object is that same JSON text
// as a TEXT value (§4.8 json_extract: "a text representation for JSON object and array
// values"). Both therefore yield the text `[4,5]`.
// ---------------------------------------------------------------------------

/// §4.10: `->` on the array field `c` returns its JSON text `[4,5]`.
#[test]
fn arrow_array_subcomponent_returns_json_text() {
    eval_eq(r#"'{"c":[4,5]}' -> 'c'"#, text("[4,5]"));
}

/// §4.10 + §4.8: `->>` on the array field `c` returns the SQL representation of the array,
/// which is its JSON text `[4,5]` as a TEXT value. (Derived from the stated rules — the doc
/// gives no verbatim `->>`-on-array example.)
#[test]
fn arrow2_array_subcomponent_returns_sql_text() {
    eval_eq(r#"'{"c":[4,5]}' ->> 'c'"#, text("[4,5]"));
}

/// §4.10 (verbatim): `->` on a path selecting an OBJECT subcomponent (`'$.c[2]'`) returns
/// that object's JSON text `{"f":7}`.
#[test]
fn arrow_object_subcomponent_returns_json_text() {
    eval_eq(
        r#"'{"a":2,"c":[4,5,{"f":7}]}' -> '$.c[2]'"#,
        text(r#"{"f":7}"#),
    );
}

/// §4.10 + §4.8: `->>` on the same OBJECT subcomponent returns its SQL representation, which
/// for an object is its JSON text `{"f":7}` as a TEXT value. (Derived like the array case —
/// no verbatim `->>`-on-object example in the doc.)
#[test]
fn arrow2_object_subcomponent_returns_sql_text() {
    eval_eq(
        r#"'{"a":2,"c":[4,5,{"f":7}]}' ->> '$.c[2]'"#,
        text(r#"{"f":7}"#),
    );
}

// ---------------------------------------------------------------------------
// A subcomponent that does not exist -> NULL. json1.html §4.10: "or NULL if that
// subcomponent does not exist" (verbatim example `-> '$.x'` -> NULL).
// ---------------------------------------------------------------------------

/// §4.10 (verbatim): a path `'$.x'` matching no object member returns NULL.
#[test]
fn arrow_missing_key_returns_null() {
    eval_eq(r#"'{"a":2,"c":[4,5,{"f":7}]}' -> '$.x'"#, null());
}

/// §4.10: an integer index past the end of the array (index 5 of a 1-element array) selects
/// no subcomponent, so the result is NULL.
#[test]
fn arrow_index_out_of_range_returns_null() {
    eval_eq(r#"'[1]' -> 5"#, null());
}

// ---------------------------------------------------------------------------
// A JSON `null` subcomponent. json1.html §4.10 verbatim:
//   '{"a":null}' -> '$.a'   -> 'null'  (JSON text)
//   '{"a":null}' ->> '$.a'  -> NULL    (SQL NULL)
// This is the sharpest `->` vs `->>` divergence: same subcomponent, different value class.
// ---------------------------------------------------------------------------

/// §4.10 (verbatim): `->` on a JSON `null` subcomponent returns the JSON text `null` (a
/// 4-character TEXT value), NOT SQL NULL.
#[test]
fn arrow_json_null_returns_text_null() {
    eval_eq(r#"'{"a":null}' -> '$.a'"#, text("null"));
}

/// §4.10 (verbatim): `->>` on a JSON `null` subcomponent returns SQL NULL (not the text
/// `null`).
#[test]
fn arrow2_json_null_returns_sql_null() {
    eval_eq(r#"'{"a":null}' ->> '$.a'"#, null());
}

// ---------------------------------------------------------------------------
// A SQL NULL left operand. §3.1: "NULL values are interpreted as JSON nulls", so nothing
// can be selected out of it -> §4.10 absent-subcomponent -> NULL.
// ---------------------------------------------------------------------------

/// §3.1 + §4.10: a NULL left operand yields NULL (selecting `.a` out of JSON `null` finds
/// no subcomponent).
#[test]
fn arrow_null_left_operand_returns_null() {
    eval_eq(r#"NULL -> 'a'"#, null());
}

// ---------------------------------------------------------------------------
// Chaining: `->` / `->>` associate left-to-right, so successive extractions compose.
// json1.html §4.10 verbatim: '{"a":2,"c":[4,5,{"f":7}]}' -> 'c' -> 2 ->> 'f'  -> 7
// ---------------------------------------------------------------------------

/// §4.10 (verbatim): the mixed chain `-> 'c' -> 2 ->> 'f'` selects the object `{"f":7}`
/// (via `-> 'c'` then `-> 2`) and finally `->> 'f'` returns the SQL INTEGER 7, proving
/// left-to-right association and that intermediate `->` results remain JSON for the next hop.
#[test]
fn chain_mixed_arrow_and_arrow2_returns_sql_integer() {
    eval_eq(r#"'{"a":2,"c":[4,5,{"f":7}]}' -> 'c' -> 2 ->> 'f'"#, int(7));
}

/// §4.10: a pure `->` chain stays in JSON representation at every hop, so
/// `-> 'a' -> 'b'` on `{"a":{"b":5}}` returns the JSON text `5`.
#[test]
fn chain_arrow_only_returns_json_text() {
    eval_eq(r#"'{"a":{"b":5}}' -> 'a' -> 'b'"#, text("5"));
}

/// §4.10: the same chain terminated by `->>` returns the SQL representation of the leaf,
/// the INTEGER 5 — the only difference from the pure-`->` chain is the final operator.
#[test]
fn chain_arrow_then_arrow2_returns_sql_integer() {
    eval_eq(r#"'{"a":{"b":5}}' -> 'a' ->> 'b'"#, int(5));
}

// ---------------------------------------------------------------------------
// Negative array index inside a path (SQLite 3.47.0+). json1.html §4.10 verbatim:
//   '{"a":2,"c":[4,5],"f":7}' -> '$.c[#-1]'  -> '5'
// `#-1` indexes from the end, so it selects the last element (5); `->` renders its JSON text.
// ---------------------------------------------------------------------------

/// §4.10 (verbatim): the path `'$.c[#-1]'` selects the last element of `c` (`5`), and `->`
/// returns its JSON text `5`.
#[test]
fn arrow_negative_index_from_end_returns_json_text() {
    eval_eq(r#"'{"a":2,"c":[4,5],"f":7}' -> '$.c[#-1]'"#, text("5"));
}

// ---------------------------------------------------------------------------
// The operator applied to a table COLUMN, evaluated per row. Exercises the operator
// through a real projection (not just a literal-vs-literal expression) and shows the three
// `->>` result classes side by side: INTEGER, TEXT, and NULL (missing key).
// ---------------------------------------------------------------------------

/// §4.10: `j ->> 'k'` over rows of a TEXT column returns, per row, the SQL representation of
/// member `k`: an INTEGER for a numeric value, TEXT for a string value, and NULL where the
/// object has no member `k`.
#[test]
fn arrow2_over_table_column_per_row() {
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(id INTEGER PRIMARY KEY, j TEXT)");
    exec(
        &mut db,
        r#"INSERT INTO t(id, j) VALUES (1, '{"k":10}'), (2, '{"k":"v"}'), (3, '{"m":1}')"#,
    );
    assert_rows(
        &mut db,
        r#"SELECT id, j ->> 'k' FROM t ORDER BY id"#,
        &[
            vec![int(1), int(10)],
            vec![int(2), text("v")],
            vec![int(3), null()],
        ],
    );
}

// ---------------------------------------------------------------------------
// The JSON value-SUBTYPE of the operator result (json1.html §3.4 + §4.10). `->` returns
// the JSON *representation* of the selected node, so its non-NULL result carries the JSON
// subtype: a DIRECTLY wrapping JSON function embeds it as JSON instead of re-quoting it.
// `->>` returns the SQL representation and carries NO subtype, so a wrapping function
// quotes it. This is the operator analogue of the json_extract/json rule tested in
// conformance_json.rs, and the two forms MUST diverge inside a constructor even though
// their own text output for a container is identical (see arrow*_array_subcomponent_*).
// Values transcribed from the §3.4 subtype rule; real sqlite3 agrees (`[[4,5]]`, `[2]`,
// `[1]` for the `->` cases; `["[4,5]"]` for the `->>` guard).
// ---------------------------------------------------------------------------

/// §3.4 + §4.10: `-> '$.c'` selects the array `[4,5]` and, because it returns the JSON
/// representation, tags the result with the JSON subtype — so `json_array` EMBEDS it,
/// yielding `[[4,5]]` (not the re-quoted `["[4,5]"]`).
#[test]
fn arrow_container_result_embeds_in_constructor() {
    eval_eq(r#"json_array('{"c":[4,5]}' -> '$.c')"#, text("[[4,5]]"));
}

/// §3.4 + §4.10 (symmetric guard): `->> '$.c'` returns the SQL representation of the array
/// — its JSON text as an ordinary TEXT value with NO subtype — so `json_array` QUOTES it,
/// yielding `["[4,5]"]`. Paired with the `->` test above, this pins that ONLY `->` (not
/// `->>`) carries the subtype: dropping the subtype tag on `->` reddens the test above,
/// and mistakenly tagging `->>` reddens this one.
#[test]
fn arrow2_container_result_quotes_in_constructor() {
    eval_eq(r#"json_array('{"c":[4,5]}' ->> '$.c')"#, text(r#"["[4,5]"]"#));
}

/// §3.4 + §4.10: `->` tags EVERY selected node (it always renders JSON text), not just
/// containers — unlike single-path `json_extract`, which returns the SQL representation and
/// so tags only a container. So `-> 'a'` on a numeric field carries the subtype and
/// `json_array` embeds the number `2`, yielding `[2]` (not the re-quoted `["2"]`).
#[test]
fn arrow_scalar_result_embeds_in_constructor() {
    eval_eq(r#"json_array('{"a":2}' -> 'a')"#, text("[2]"));
}

/// §3.4 + §4.22: because `->` carries the JSON subtype, `json_quote` of a `->` result is a
/// no-op that returns it verbatim — `json_quote('{"a":[1]}' -> '$.a')` is `[1]`, not the
/// re-quoted `"[1]"`. This is the operator twin of `json_quote_of_json_result_is_a_noop`
/// in conformance_json.rs.
#[test]
fn json_quote_of_arrow_result_is_a_noop() {
    eval_eq(r#"json_quote('{"a":[1]}' -> '$.a')"#, text("[1]"));
}
