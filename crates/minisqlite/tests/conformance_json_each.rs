//! Conformance battery for the `json_each()` and `json_tree()` table-valued functions
//! (`spec/sqlite-doc/json1.html` §4.24), run through the pinned `minisqlite::Connection`
//! facade.
//!
//! Every expectation is TRANSCRIBED FROM THE SPEC (the schema and column semantics in
//! json1.html §4.24 and its worked examples), never from what this engine happens to
//! return. The visible schema both functions expose, in `SELECT *` order, is:
//!
//! ```text
//! key, value, type, atom, id, parent, fullkey, path
//! ```
//!
//! plus two HIDDEN input columns — `json` (the raw document argument) and `root` (the
//! start-path text) — excluded from `SELECT *` but selectable by name (json1.html §4.24).
//!
//! One DELIBERATE limitation is documented by tests here (see the marked case):
//!
//!   * **The `id` integer is engine-internal.** json1.html documents `id` as an unstable
//!     housekeeping number whose "only guarantee is that the id column will be different
//!     for every row". These tests therefore assert `id` UNIQUENESS and the `parent`↔`id`
//!     relationship, never a specific integer.
//!
//! JSON SQL is quote-heavy, so the SQL and expected JSON text use Rust RAW strings
//! (`r#"..."#`) throughout.

mod conformance;

use conformance::*;

// ============================================================================
// json_each(X): the immediate children of the top-level element (json1.html §4.24)
// ============================================================================

#[test]
fn json_each_over_array_keys_by_index() {
    // "The key column is the integer array index for elements of a JSON array." The
    // value/atom are the element's SQL value; type is 'integer'; fullkey is `$[i]`; the
    // container path is `$`.
    assert_rows(
        &mut mem(),
        r#"SELECT key, value, type, atom, fullkey, path FROM json_each('[10,20,30]')"#,
        &[
            vec![int(0), int(10), text("integer"), int(10), text("$[0]"), text("$")],
            vec![int(1), int(20), text("integer"), int(20), text("$[1]"), text("$")],
            vec![int(2), int(30), text("integer"), int(30), text("$[2]"), text("$")],
        ],
    );
}

#[test]
fn json_each_parent_is_always_null() {
    // "The parent column is always NULL for json_each()." (json1.html §4.24)
    assert_rows(
        &mut mem(),
        r#"SELECT parent FROM json_each('[10,20,30]')"#,
        &[vec![null()], vec![null()], vec![null()]],
    );
}

#[test]
fn json_each_over_object_keys_by_label_in_insertion_order() {
    // "the text label for elements of a JSON object." Object members keep their DOCUMENT
    // order, which is NOT sorted-by-label: the out-of-order keys c, b, a must come back in
    // that written order (an alphabetizing implementation would return a, b, c and fail).
    assert_rows(
        &mut mem(),
        r#"SELECT key, value, type, atom, fullkey, path FROM json_each('{"c":1,"b":"x","a":null}')"#,
        &[
            vec![text("c"), int(1), text("integer"), int(1), text("$.c"), text("$")],
            vec![text("b"), text("x"), text("text"), text("x"), text("$.b"), text("$")],
            vec![text("a"), null(), text("null"), null(), text("$.a"), text("$")],
        ],
    );
}

#[test]
fn json_each_over_primitive_yields_single_row() {
    // "or just the top-level element itself if the top-level element is a primitive
    // value." One row: key NULL, and the primitive-start rule makes path == fullkey.
    assert_rows(
        &mut mem(),
        r#"SELECT key, value, type, atom, fullkey, path FROM json_each('42')"#,
        &[vec![null(), int(42), text("integer"), int(42), text("$"), text("$")]],
    );
    assert_rows(
        &mut mem(),
        r#"SELECT key, value, type, atom, fullkey, path FROM json_each('"hi"')"#,
        &[vec![null(), text("hi"), text("text"), text("hi"), text("$"), text("$")]],
    );
}

#[test]
fn json_each_container_child_value_is_text_atom_null() {
    // "When type is 'array' or 'object', the value column will return the text JSON of
    // the array or object … The atom column is NULL for a JSON array or object."
    assert_rows(
        &mut mem(),
        r#"SELECT key, value, type, atom FROM json_each('[[1,2],{"x":3}]')"#,
        &[
            vec![int(0), text("[1,2]"), text("array"), null()],
            vec![int(1), text(r#"{"x":3}"#), text("object"), null()],
        ],
    );
}

#[test]
fn json_each_type_and_value_for_each_primitive_class() {
    // The 'type' vocabulary and the SQL value each primitive maps to (json1.html §4.24):
    // real→REAL, true→1, false→0, null→NULL, string→TEXT.
    assert_rows(
        &mut mem(),
        r#"SELECT type, value, atom FROM json_each('[1.5, true, false, null, "s"]')"#,
        &[
            vec![text("real"), real(1.5), real(1.5)],
            vec![text("true"), int(1), int(1)],
            vec![text("false"), int(0), int(0)],
            vec![text("null"), null(), null()],
            vec![text("text"), text("s"), text("s")],
        ],
    );
}

#[test]
fn json_each_number_type_distinguishes_integer_real_and_exponent() {
    // json1.html §4.24: `type` is 'integer' for a whole-number token and 'real' for one
    // with a fraction OR an exponent, and `value` keeps that storage class. A parser that
    // folds `1.0` to an integer, or treats an exponent token as text, fails here — and
    // `value_eq` keeps Integer(3) distinct from Real(3.0), so the classes really are pinned.
    assert_rows(
        &mut mem(),
        r#"SELECT type, value FROM json_each('[1.0, 3, 2.5, 1e2, 1.5e1]')"#,
        &[
            vec![text("real"), real(1.0)],
            vec![text("integer"), int(3)],
            vec![text("real"), real(2.5)],
            vec![text("real"), real(100.0)],
            vec![text("real"), real(15.0)],
        ],
    );
}

#[test]
fn json_each_large_and_negative_integers_keep_integer_class() {
    // Boundary integers keep type 'integer' and their exact i64 value — a value that fits
    // i64 is NOT coerced to real. i64::MAX, zero, and a negative all round-trip.
    assert_rows(
        &mut mem(),
        r#"SELECT type, value FROM json_each('[-42, 0, 9223372036854775807]')"#,
        &[
            vec![text("integer"), int(-42)],
            vec![text("integer"), int(0)],
            vec![text("integer"), int(9223372036854775807)],
        ],
    );
}

#[test]
fn json_each_empty_containers_have_no_children() {
    // json_each walks only immediate children, so an empty array/object yields no rows.
    assert_scalar(&mut mem(), r#"SELECT count(*) FROM json_each('[]')"#, int(0));
    assert_scalar(&mut mem(), r#"SELECT count(*) FROM json_each('{}')"#, int(0));
}

// ============================================================================
// json_each(X, P): walk the element identified by path P (json1.html §4.24)
// ============================================================================

#[test]
fn json_each_with_path_walks_the_selected_array() {
    // "treat the element identified by path P as the top-level element." The fullkey
    // still describes the true path from `$` (P is a prefix), and the container path is P.
    assert_rows(
        &mut mem(),
        r#"SELECT key, value, fullkey, path FROM json_each('{"a":[7,8,9]}', '$.a')"#,
        &[
            vec![int(0), int(7), text("$.a[0]"), text("$.a")],
            vec![int(1), int(8), text("$.a[1]"), text("$.a")],
            vec![int(2), int(9), text("$.a[2]"), text("$.a")],
        ],
    );
}

#[test]
fn json_each_with_path_to_primitive_is_single_row_path_equals_fullkey() {
    // A path selecting a primitive → one row; "the path to the current row in the case
    // where the iteration … only provides a single row of output" (path == fullkey).
    assert_rows(
        &mut mem(),
        r#"SELECT key, value, fullkey, path FROM json_each('{"a":5}', '$.a')"#,
        &[vec![null(), int(5), text("$.a"), text("$.a")]],
    );
}

#[test]
fn json_each_path_selecting_nothing_is_zero_rows() {
    // A path that matches nothing yields no rows (not an error).
    assert_scalar(&mut mem(), r#"SELECT count(*) FROM json_each('{"a":1}', '$.missing')"#, int(0));
    assert_scalar(&mut mem(), r#"SELECT count(*) FROM json_each('[1,2]', '$[9]')"#, int(0));
}

#[test]
fn json_each_null_document_or_null_path_is_zero_rows() {
    // A NULL first argument (a SQL NULL, not the JSON text "null") yields no rows; so
    // does a NULL path argument.
    assert_scalar(&mut mem(), r#"SELECT count(*) FROM json_each(NULL)"#, int(0));
    assert_scalar(&mut mem(), r#"SELECT count(*) FROM json_each('[1,2]', NULL)"#, int(0));
    // The JSON text "null" is a one-element primitive (distinct from SQL NULL).
    assert_rows(
        &mut mem(),
        r#"SELECT type, value FROM json_each('null')"#,
        &[vec![text("null"), null()]],
    );
}

// ============================================================================
// json_tree(X[,P]): recursive depth-first walk INCLUDING the start element
// ============================================================================

#[test]
fn json_tree_walks_root_then_descendants_depth_first() {
    // json_tree "recursively walk[s] through the JSON substructure starting with the
    // top-level element." The start element is the first row (key NULL, parent NULL).
    assert_rows(
        &mut mem(),
        r#"SELECT key, value, type, atom, fullkey, path FROM json_tree('{"a":1,"b":[2,3]}')"#,
        &[
            vec![null(), text(r#"{"a":1,"b":[2,3]}"#), text("object"), null(), text("$"), text("$")],
            vec![text("a"), int(1), text("integer"), int(1), text("$.a"), text("$")],
            vec![text("b"), text("[2,3]"), text("array"), null(), text("$.b"), text("$")],
            vec![int(0), int(2), text("integer"), int(2), text("$.b[0]"), text("$.b")],
            vec![int(1), int(3), text("integer"), int(3), text("$.b[1]"), text("$.b")],
        ],
    );
}

#[test]
fn json_tree_is_depth_first_not_breadth_first() {
    // "recursively walk" is DEPTH-first: with two sibling containers, json_tree fully
    // descends `a` (its elements) BEFORE visiting `b`. A breadth-first walk would list
    // `$.a`,`$.b` before any leaves, so pinning the full fullkey order distinguishes them.
    assert_rows(
        &mut mem(),
        r#"SELECT fullkey FROM json_tree('{"a":[1,2],"b":[3,4]}')"#,
        &[
            vec![text("$")],
            vec![text("$.a")],
            vec![text("$.a[0]")],
            vec![text("$.a[1]")],
            vec![text("$.b")],
            vec![text("$.b[0]")],
            vec![text("$.b[1]")],
        ],
    );
}

#[test]
fn json_tree_leaf_filter_by_atom_not_null() {
    // The spec's worked example: leaves are exactly the rows with a non-NULL atom
    // (json1.html §4.24.1 "SELECT big.rowid, fullkey, atom … WHERE atom IS NOT NULL").
    assert_rows(
        &mut mem(),
        r#"SELECT fullkey, atom FROM json_tree('{"a":1,"b":[2,3]}') WHERE atom IS NOT NULL"#,
        &[
            vec![text("$.a"), int(1)],
            vec![text("$.b[0]"), int(2)],
            vec![text("$.b[1]"), int(3)],
        ],
    );
}

#[test]
fn json_tree_leaf_filter_by_type_not_container() {
    // The equivalent spec example using the type column: suppress containers.
    assert_rows(
        &mut mem(),
        r#"SELECT fullkey, value FROM json_tree('{"a":1,"b":[2,3]}')
           WHERE type NOT IN ('object','array')"#,
        &[
            vec![text("$.a"), int(1)],
            vec![text("$.b[0]"), int(2)],
            vec![text("$.b[1]"), int(3)],
        ],
    );
}

#[test]
fn json_tree_deeply_nested_fullkeys() {
    // Nested objects/arrays: fullkey composes labels and indices; path is the container.
    assert_rows(
        &mut mem(),
        r#"SELECT fullkey, atom FROM json_tree('{"x":{"y":[{"z":9}]}}') WHERE atom IS NOT NULL"#,
        &[vec![text("$.x.y[0].z"), int(9)]],
    );
}

#[test]
fn json_tree_with_path_reparents_the_start_element() {
    // json_tree(X,'$.partlist') starts at $.partlist; a query keyed on `key` works
    // exactly as the spec's partlist/uuid example (json1.html §4.24.1).
    assert_rows(
        &mut mem(),
        r#"SELECT key, value FROM json_tree('{"id":1,"partlist":{"uuid":"6fa5"}}', '$.partlist')
           WHERE key='uuid'"#,
        &[vec![text("uuid"), text("6fa5")]],
    );
}

#[test]
fn json_tree_empty_array_is_one_row() {
    // Unlike json_each, json_tree includes the start element itself, so an empty array
    // yields exactly one row (the array), and an empty object likewise.
    assert_scalar(&mut mem(), r#"SELECT count(*) FROM json_tree('[]')"#, int(1));
    assert_scalar(&mut mem(), r#"SELECT count(*) FROM json_tree('{}')"#, int(1));
    assert_rows(
        &mut mem(),
        r#"SELECT key, type, value, atom FROM json_tree('[]')"#,
        &[vec![null(), text("array"), text("[]"), null()]],
    );
}

#[test]
fn json_tree_fullkey_roundtrips_through_json_extract_for_special_labels() {
    // "The fullkey column is a text path that uniquely identifies the current row
    // element within the original JSON string." (json1.html §4.24) The strongest,
    // spelling-agnostic invariant: each leaf's fullkey, fed back to json_extract, must
    // return that leaf's own atom — even when an object label needs quoting (space, dot).
    // The docs do not pin the exact quoting spelling, so we prove the ROUND-TRIP instead
    // of a literal string (which would only echo this implementation).
    let doc = r#"{"a b":{"c.d":7},"e":[8]}"#;
    let mut db = mem();
    // No leaf where json_extract(doc, fullkey) disagrees with the leaf's atom.
    assert_scalar(
        &mut db,
        &format!(
            r#"SELECT count(*) FROM json_tree('{doc}')
                 WHERE atom IS NOT NULL AND json_extract('{doc}', fullkey) IS NOT atom"#
        ),
        int(0),
    );
    // Guard against a vacuous pass: there really are two leaves to round-trip.
    assert_scalar(
        &mut db,
        &format!(r#"SELECT count(*) FROM json_tree('{doc}') WHERE atom IS NOT NULL"#),
        int(2),
    );
}

// ============================================================================
// id / parent semantics (json1.html §4.24): unique id, parent references container id
// ============================================================================

#[test]
fn json_tree_ids_are_unique() {
    // "The only guarantee is that the id column will be different for every row."
    let mut db = mem();
    let qr = query(&mut db, r#"SELECT count(*), count(DISTINCT id) FROM json_tree('{"a":[1,2],"b":{"c":3}}')"#);
    // Every row's id is distinct: count(*) == count(DISTINCT id).
    assert_eq!(qr.rows.len(), 1);
    assert!(value_eq(&qr.rows[0][0], &qr.rows[0][1]), "id must be unique per row: {:?}", qr.rows[0]);
}

#[test]
fn json_tree_parent_references_container_id() {
    // "the parent column is the id integer for the parent of the current element, or
    // NULL for the top-level JSON element." The parent of `$.a.b` equals the id of `$.a`.
    assert_scalar(
        &mut mem(),
        r#"SELECT (SELECT parent FROM json_tree('{"a":{"b":1}}') WHERE fullkey='$.a.b')
                = (SELECT id     FROM json_tree('{"a":{"b":1}}') WHERE fullkey='$.a')"#,
        int(1),
    );
    // The start element's parent is NULL.
    assert_scalar(
        &mut mem(),
        r#"SELECT parent FROM json_tree('{"a":1}') WHERE fullkey='$'"#,
        null(),
    );
}

// ============================================================================
// SELECT * : the eight visible columns, hidden json/root excluded (json1.html §4.24)
// ============================================================================

#[test]
fn star_exposes_exactly_the_eight_visible_columns() {
    // `SELECT *` returns the ordinary columns only, in schema order; the hidden `json`
    // and `root` columns are NOT part of the star expansion.
    assert_columns(
        &mut mem(),
        r#"SELECT * FROM json_each('[1]')"#,
        &["key", "value", "type", "atom", "id", "parent", "fullkey", "path"],
    );
    assert_columns(
        &mut mem(),
        r#"SELECT * FROM json_tree('{"a":1}')"#,
        &["key", "value", "type", "atom", "id", "parent", "fullkey", "path"],
    );
}

#[test]
fn hidden_json_column_echoes_the_document_argument() {
    // json1.html §4.24: the hidden `json` column is the original document argument, the
    // same on every row. Selectable by name (though excluded from `*`).
    assert_scalar(
        &mut mem(),
        r#"SELECT DISTINCT json FROM json_each('[1,2,3]')"#,
        text("[1,2,3]"),
    );
    assert_scalar(
        &mut mem(),
        r#"SELECT DISTINCT json FROM json_tree('{"a":1}')"#,
        text(r#"{"a":1}"#),
    );
    // The echo is the RAW argument, byte-for-byte — NOT a reparsed/reserialized (minified)
    // form. A non-canonical document (interior spaces) must come back verbatim, which pins
    // that the executor clones the original argument value rather than re-rendering the
    // parsed tree.
    assert_scalar(
        &mut mem(),
        r#"SELECT DISTINCT json FROM json_each('[1, 2,  3]')"#,
        text("[1, 2,  3]"),
    );
}

#[test]
fn hidden_root_column_is_the_start_path() {
    // json1.html §4.24: the hidden `root` column is the start-path argument; with no path
    // given it defaults to `$`, and with a path it is that path text.
    assert_scalar(
        &mut mem(),
        r#"SELECT DISTINCT root FROM json_each('[1,2,3]')"#,
        text("$"),
    );
    assert_scalar(
        &mut mem(),
        r#"SELECT DISTINCT root FROM json_tree('{"a":[1,2]}', '$.a')"#,
        text("$.a"),
    );
}

#[test]
fn hidden_columns_are_excluded_from_star_but_selectable_by_name() {
    // The hidden columns do not appear in `SELECT *` (pinned by
    // `star_exposes_exactly_the_eight_visible_columns`) yet resolve when named alongside
    // the visible columns.
    assert_rows(
        &mut mem(),
        r#"SELECT key, value, json, root FROM json_each('[7,8]')"#,
        &[
            vec![int(0), int(7), text("[7,8]"), text("$")],
            vec![int(1), int(8), text("[7,8]"), text("$")],
        ],
    );
}

#[test]
fn hidden_json_column_resolves_in_a_where_clause_not_only_projection() {
    // The hidden `json` column must materialize whenever ANY clause references it, not
    // only the SELECT list. If document materialization were tied to a PROJECTION
    // reference alone, `json` would read NULL inside WHERE and the filter would drop every
    // row (count 0). This pins that a WHERE reference is honored.
    assert_scalar(
        &mut mem(),
        r#"SELECT count(*) FROM json_each('[1,2,3]') WHERE json = '[1,2,3]'"#,
        int(3),
    );
    // A non-matching filter value proves the comparison is real (not a constant-true short
    // circuit that would also return 3 for the case above).
    assert_scalar(
        &mut mem(),
        r#"SELECT count(*) FROM json_each('[1,2,3]') WHERE json = 'nope'"#,
        int(0),
    );
}

#[test]
fn hidden_root_column_resolves_in_where_and_order_by() {
    // `root` (the start-path text) is likewise usable outside the projection.
    assert_scalar(
        &mut mem(),
        r#"SELECT count(*) FROM json_each('[1,2,3]', '$') WHERE root = '$'"#,
        int(3),
    );
    // Naming the hidden `json` column in ORDER BY must RESOLVE (not error). It is constant
    // per walk, so the tie-breaker `value` decides the order — the point is that the hidden
    // reference in ORDER BY is honored and the rows come back sorted by value.
    assert_rows(
        &mut mem(),
        r#"SELECT value FROM json_each('[30,10,20]') ORDER BY json, value"#,
        &[vec![int(10)], vec![int(20)], vec![int(30)]],
    );
}

#[test]
fn hidden_json_column_echoes_the_per_row_argument_when_correlated() {
    // In an implicit-LATERAL join the `json` column echoes each OUTER row's document, so
    // it re-evaluates per outer row just like the walk does (json1.html §4.24.1).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE big(doc)");
    exec(&mut db, r#"INSERT INTO big VALUES ('[1,2]'), ('[3]')"#);
    assert_rows(
        &mut db,
        r#"SELECT big.rowid, json_each.value, json_each.json
             FROM big, json_each(big.doc)
            ORDER BY big.rowid, json_each.value"#,
        &[
            vec![int(1), int(1), text("[1,2]")],
            vec![int(1), int(2), text("[1,2]")],
            vec![int(2), int(3), text("[3]")],
        ],
    );
}

// ============================================================================
// Errors: malformed JSON and malformed path (json1.html §4.24 / §3)
// ============================================================================

#[test]
fn malformed_json_argument_is_an_error() {
    // "json_each() requires well-formed JSON as its first argument." A malformed
    // document is a query error.
    assert_query_error(&mut mem(), r#"SELECT * FROM json_each('{')"#);
    assert_query_error(&mut mem(), r#"SELECT * FROM json_each('[1,2')"#);
    assert_query_error(&mut mem(), r#"SELECT * FROM json_tree('nope')"#);
}

#[test]
fn malformed_path_argument_is_an_error() {
    // A path that is not a well-formed JSON path (does not start with `$`) is an error.
    assert_query_error(&mut mem(), r#"SELECT * FROM json_each('[1]', 'notapath')"#);
}

#[test]
fn unknown_table_valued_function_is_an_error() {
    // Only json_each/json_tree are FROM-clause table-valued functions here; any other
    // name in that position is a loud error rather than a silent empty result.
    assert_query_error(&mut mem(), r#"SELECT * FROM not_a_tvf('[1]')"#);
}

#[test]
fn wrong_arity_is_an_error() {
    // The functions take one or two arguments; zero or three is an error.
    assert_query_error(&mut mem(), r#"SELECT * FROM json_each()"#);
    assert_query_error(&mut mem(), r#"SELECT * FROM json_tree('[1]', '$', 'extra')"#);
}

// ============================================================================
// Aliasing: a FROM alias renames the source qualifier (json1.html table-valued syntax)
// ============================================================================

#[test]
fn from_alias_renames_the_source() {
    // With `AS je`, the qualifier becomes `je` (not the function name).
    assert_rows(
        &mut mem(),
        r#"SELECT je.key, je.value FROM json_each('[7,8]') AS je"#,
        &[vec![int(0), int(7)], vec![int(1), int(8)]],
    );
}

#[test]
fn unaliased_source_qualifier_is_the_function_name() {
    // Without an alias the qualifier is the function's own name, so `json_each.value`
    // resolves — exactly as the spec examples write it.
    assert_rows(
        &mut mem(),
        r#"SELECT json_each.value FROM json_each('[7,8]') WHERE json_each.key = 1"#,
        &[vec![int(8)]],
    );
}

// ============================================================================
// Correlated / implicit-LATERAL joins (json1.html §4.24.1 worked examples)
// ============================================================================

#[test]
fn correlated_json_each_over_a_table_column() {
    // The spec's phone-number example: json_each(user.phone) is implicitly LATERAL — its
    // argument reads the current `user` row. "Which users have a phone number in the 704
    // area code?" (json1.html §4.24.1).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE user(name, phone)");
    exec(
        &mut db,
        r#"INSERT INTO user VALUES
            ('alice', '["704-111","980-222"]'),
            ('bob',   '["704-333"]'),
            ('carol', '["212-444"]')"#,
    );
    assert_rows_unordered(
        &mut db,
        r#"SELECT DISTINCT user.name FROM user, json_each(user.phone)
           WHERE json_each.value LIKE '704-%'"#,
        &[vec![text("alice")], vec![text("bob")]],
    );
}

#[test]
fn correlated_json_each_reevaluates_arg_per_outer_row() {
    // Each outer row drives its own walk: the total row count is the sum of each row's
    // array length (2 + 1 + 1 = 4), proving the argument is re-evaluated per outer row.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE user(name, phone)");
    exec(
        &mut db,
        r#"INSERT INTO user VALUES
            ('alice', '["704-111","980-222"]'),
            ('bob',   '["704-333"]'),
            ('carol', '["212-444"]')"#,
    );
    assert_scalar(
        &mut db,
        r#"SELECT count(*) FROM user, json_each(user.phone)"#,
        int(4),
    );
}

#[test]
fn correlated_json_tree_over_a_table_column() {
    // The spec's "big" example: json_tree(big.json) decomposes each row's JSON; the
    // rowid ties each element back to its source row (json1.html §4.24.1).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE big(json)");
    exec(
        &mut db,
        r#"INSERT INTO big VALUES ('{"a":1,"b":{"c":2}}'), ('[9,8]')"#,
    );
    assert_rows(
        &mut db,
        r#"SELECT big.rowid, fullkey, atom FROM big, json_tree(big.json)
           WHERE atom IS NOT NULL
           ORDER BY big.rowid, fullkey"#,
        &[
            vec![int(1), text("$.a"), int(1)],
            vec![int(1), text("$.b.c"), int(2)],
            vec![int(2), text("$[0]"), int(9)],
            vec![int(2), text("$[1]"), int(8)],
        ],
    );
}

#[test]
fn correlated_json_tree_with_constant_path() {
    // json_tree(big.json, '$.partlist') — a correlated document with a constant start
    // path — reproduces the spec's uuid-search shape (json1.html §4.24.1).
    let mut db = mem();
    exec(&mut db, "CREATE TABLE big(json)");
    exec(
        &mut db,
        r#"INSERT INTO big VALUES
            ('{"id":"A","partlist":{"uuid":"match"}}'),
            ('{"id":"B","partlist":{"uuid":"other"}}')"#,
    );
    assert_rows_unordered(
        &mut db,
        r#"SELECT json_extract(big.json,'$.id')
             FROM big, json_tree(big.json, '$.partlist')
            WHERE json_tree.key='uuid' AND json_tree.value='match'"#,
        &[vec![text("A")]],
    );
}

#[test]
fn correlated_json_each_in_a_scalar_subquery_from() {
    // A TVF may be the SOLE source of a correlated scalar subquery's FROM (not only a
    // join right operand): `(SELECT count(*) FROM json_each(t.data))` re-walks each outer
    // row's document, so the count matches that row's array length.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(data)");
    exec(&mut db, r#"INSERT INTO t VALUES ('[1,2,3]'), ('[]'), ('[9]')"#);
    assert_rows(
        &mut db,
        r#"SELECT (SELECT count(*) FROM json_each(t.data)) FROM t ORDER BY t.rowid"#,
        &[vec![int(3)], vec![int(0)], vec![int(1)]],
    );
}

#[test]
fn json_each_as_left_operand_of_a_join() {
    // A TVF may also be the LEFT operand (non-correlated here), joined to a base table.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE t(n)");
    exec(&mut db, "INSERT INTO t VALUES (1),(2)");
    // Cross join of a 2-element json_each with a 2-row table = 4 rows.
    assert_scalar(
        &mut db,
        r#"SELECT count(*) FROM json_each('[10,20]'), t"#,
        int(4),
    );
}

#[test]
fn correlated_json_each_in_left_join_keeps_unmatched_rows() {
    // A LEFT JOIN to a correlated json_each keeps a row whose JSON has no children (an
    // empty array), NULL-filling the TVF side — exercising the IndexNestedLoop Left path.
    let mut db = mem();
    exec(&mut db, "CREATE TABLE u(name, phone)");
    exec(
        &mut db,
        r#"INSERT INTO u VALUES ('alice', '["p1","p2"]'), ('bob', '[]')"#,
    );
    // alice → 2 rows; bob → no children, but LEFT JOIN keeps one NULL-filled row.
    assert_rows_unordered(
        &mut db,
        r#"SELECT u.name, je.value FROM u LEFT JOIN json_each(u.phone) AS je ON 1
           ORDER BY u.name"#,
        &[
            vec![text("alice"), text("p1")],
            vec![text("alice"), text("p2")],
            vec![text("bob"), null()],
        ],
    );
}
