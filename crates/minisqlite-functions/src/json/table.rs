//! Row generation for the `json_each()` and `json_tree()` table-valued functions
//! (`spec/sqlite-doc/json1.html` §4.24).
//!
//! Both functions walk a JSON value and return one row per element with the fixed
//! schema `key, value, type, atom, id, parent, fullkey, path`. The only difference is
//! reach: `json_each` visits the IMMEDIATE children of the selected element (or the
//! element itself if it is a primitive), while `json_tree` recursively visits the
//! selected element and every descendant, depth-first in document order.
//!
//! This module owns the pure walk (a `Vec<Value>` per row); the executor's table
//! operator (`minisqlite-exec` `ops::table_function`) evaluates the `X`/`P` arguments
//! and calls [`json_table_rows`], so the row-shape logic has one home and cannot drift.
//!
//! # Column semantics (json1.html §4.24)
//!
//! * `key` — the array index (INTEGER) or object label (TEXT) of the element within its
//!   parent; NULL for the walk-root element.
//! * `value` — the element's SQL value: a primitive as its SQL scalar, a container as
//!   its canonical JSON TEXT. (SQLite tags a container `value` with the JSON subtype so
//!   a directly-nested JSON function embeds rather than re-quotes it; that subtype is
//!   ephemeral and cannot ride a materialized row here, so a container `value` is plain
//!   text — the same subtype-loss the scalar JSON functions have across storage.)
//! * `type` — the JSON type name (`'null'`,`'true'`,`'false'`,`'integer'`,`'real'`,
//!   `'text'`,`'array'`,`'object'`).
//! * `atom` — the SQL value for a primitive; NULL for a container.
//! * `id` — a per-row document-order id. `json_tree` numbers pre-order so a `parent`
//!   references the container's id; `json_each` numbers its emitted rows in order.
//!   (json1.html documents the id as an internal, unstable number "different for every
//!   row"; this matches its ordering and keeps `parent`/`id` consistent, not SQLite's
//!   exact integers.)
//! * `parent` — for `json_tree`, the `id` of the element's container (NULL at the walk
//!   root); for `json_each`, always NULL (it is a single-level walk).
//! * `fullkey` — the full canonical path from `$` to the element (e.g. `$.a[2]`), with
//!   array subscripts resolved to concrete indices even when the start path `P` used
//!   `[#-N]`.
//! * `path` — the path to the element's CONTAINER (`fullkey` with its last component
//!   stripped), EXCEPT when the walk starts on a primitive and yields a single row, in
//!   which case `path` is that element's own `fullkey` (json1.html §4.24).
//!
//! # The two hidden columns (`json`, `root`)
//!
//! SQLite's schema also has hidden `json` / `root` columns (json1.html §4.24): `json` is
//! the raw first argument (the document) and `root` is the start-path text (default `$`).
//! They are excluded from `SELECT *` but selectable by name, and occupy the last two
//! register slots (`json`, then `root`), after the eight visible columns.
//!
//! `root` is a small path string and is returned here (in [`JsonTableRows`]) for the
//! executor to append to each row. `json` is the WHOLE document, so materializing it into
//! every row is `O(rows × |document|)` — quadratic for a flat array/object whose text
//! grows with its element count. SQLite computes a hidden column only when it is selected;
//! this generator therefore does NOT return `json` at all. The executor already holds the
//! evaluated first-argument `Value` and appends it (a clone) to each row ONLY when the
//! statement actually references the `json` column (see `minisqlite-exec`
//! `ops::table_function` and the planner's `emit_json` flag); otherwise it appends SQL
//! NULL in that slot. So the document is never copied on the hot path of the common
//! `SELECT value`/`SELECT *`/`count(*)` queries that never name `json`.

use std::borrow::Cow;
use std::fmt::Write as _;

use minisqlite_types::{Result, Value};

use super::parse::parse_json_arg;
use super::path::{parse_path, resolve_index_read, value_text, Step};
use super::value::{escape_string_into, Json};

/// Which JSON table-valued function to generate rows for. Shared with the planner
/// (`PlanNode::TableFunctionScan`) and the executor so the three agree on one vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonTableKind {
    /// `json_each` — the immediate children of the selected element (or the element
    /// itself if it is a primitive).
    Each,
    /// `json_tree` — the selected element and all its descendants, depth-first.
    Tree,
}

/// The number of VISIBLE columns both functions expose (`key…path`), i.e. the width of
/// each content row [`json_table_rows`] returns and the count `SELECT *` sees.
pub const JSON_TABLE_COLUMN_COUNT: usize = 8;

/// The number of HIDDEN input columns (`json`, `root`) appended after the visible ones
/// (json1.html §4.24): selectable by name but excluded from `SELECT *`. The executor row
/// width is [`JSON_TABLE_COLUMN_COUNT`] + this.
pub const JSON_TABLE_HIDDEN_COLUMN_COUNT: usize = 2;

/// The generated rows plus the constant hidden `root` value.
///
/// `rows` are the content rows (each [`JSON_TABLE_COLUMN_COUNT`] wide, `key…path`). `root`
/// is the small hidden start-path column (json1.html §4.24), identical for every row, so it
/// is returned once here and the executor appends it to each streamed row (when `rows` is
/// empty it is unused).
///
/// The other hidden column, `json`, is deliberately NOT returned: it is the whole document,
/// and copying it into every row is quadratic (see the module docs). The executor already
/// holds the evaluated first-argument `Value` and appends it only when the statement selects
/// `json`; otherwise it appends SQL NULL in that slot.
pub struct JsonTableRows {
    pub rows: Vec<Vec<Value>>,
    /// The hidden `root` column: the start-path text (`$` when no path was given).
    pub root: Value,
}

/// Generate the `json_each` / `json_tree` rows for JSON value `json` and optional start
/// path `path` (default `$`): the content rows (each the 8 columns in schema order) plus
/// the small constant hidden `root` value.
///
/// Returns zero rows when `json` is SQL NULL, when `path` is SQL NULL, or when `path`
/// selects nothing; a `"malformed JSON"` error when `json` is not well-formed JSON; and
/// a path error when `path` is not a well-formed JSON path. This is the single entry
/// point the executor's table operator calls. It does NOT copy the document (`json`
/// column): the executor materializes that only when the query references it.
pub fn json_table_rows(
    kind: JsonTableKind,
    json: &Value,
    path: Option<&Value>,
) -> Result<JsonTableRows> {
    // A SQL NULL document yields no rows (distinct from the JSON text "null", which is a
    // one-element primitive). Check before `parse_json_arg`, which maps SQL NULL to JSON
    // null rather than erroring.
    if matches!(json, Value::Null) {
        return Ok(JsonTableRows { rows: Vec::new(), root: Value::Null });
    }
    // A SQL NULL start path selects nothing (SQLite yields no rows rather than walking
    // from the default `$`).
    if let Some(Value::Null) = path {
        return Ok(JsonTableRows { rows: Vec::new(), root: Value::Null });
    }

    let root = parse_json_arg(json)?;

    let path_text: Cow<str> = match path {
        Some(v) => value_text(v),
        None => Cow::Borrowed("$"),
    };
    let steps = parse_path(&path_text)?;
    // The hidden `root` column is the start-path text (the 2nd argument, or `$`). Built
    // after `parse_path` borrowed `path_text`, so no extra clone.
    let root_col = Value::Text(path_text.into_owned());

    // Walk to the selected element, building its RESOLVED fullkey (concrete indices) and
    // its container's fullkey along the way. A path that selects nothing → zero rows.
    let Some(located) = locate(&root, &steps) else {
        return Ok(JsonTableRows { rows: Vec::new(), root: root_col });
    };

    let mut rows = Vec::new();
    match kind {
        JsonTableKind::Each => each_rows(located.node, &located.fullkey, &mut rows),
        JsonTableKind::Tree => {
            let mut next_id = 0i64;
            tree_rows(
                located.node,
                Value::Null,
                &located.fullkey,
                &located.container,
                None,
                true,
                &mut next_id,
                &mut rows,
            );
        }
    }
    Ok(JsonTableRows { rows, root: root_col })
}

/// The selected element plus the two paths the row builder needs: its own resolved
/// `fullkey` and the `fullkey` of its container (its `fullkey` with the last step
/// removed, tracked during the walk so a quoted label containing `.`/`[` never confuses
/// a string-strip).
struct Located<'a> {
    node: &'a Json,
    fullkey: String,
    container: String,
}

/// Follow `steps` from `root`, returning the selected node with its resolved `fullkey`
/// and container `fullkey`, or `None` if the path selects nothing (a missing key/index,
/// a type mismatch, or a `[#]` append subscript). Also builds the canonical path text.
///
/// INVARIANT (keep in lockstep with [`super::path::navigate`]): the per-step selection
/// rule here MUST match `navigate` exactly — object key = first matching member, array
/// index = [`resolve_index_read`] (the shared helper), any other pairing selects nothing.
/// `navigate` powers `json_extract`/`json_type`/etc. and this powers `json_each`/
/// `json_tree`; if the two diverge, the same path text would resolve differently between
/// them for the same document. A change to navigation semantics (duplicate-key handling,
/// subscript resolution, a JSON5 label nuance) must land in BOTH. This copy adds only the
/// side effect of recording the resolved fullkey/container path.
fn locate<'a>(root: &'a Json, steps: &[Step]) -> Option<Located<'a>> {
    let mut node = root;
    let mut fullkey = String::from("$");
    let mut container = String::from("$");
    for step in steps {
        // The container of the element reached AFTER this step is the current fullkey.
        container = fullkey.clone();
        match (node, step) {
            (Json::Object(members), Step::Key(k)) => {
                let (_, v) = members.iter().find(|(mk, _)| mk == k)?;
                append_label(&mut fullkey, k);
                node = v;
            }
            (Json::Array(items), Step::Index(idx)) => {
                let n = resolve_index_read(*idx, items.len())?;
                append_index(&mut fullkey, n);
                node = &items[n];
            }
            _ => return None,
        }
    }
    Some(Located { node, fullkey, container })
}

/// `json_each`: emit the immediate children of `selected`, or one row for `selected`
/// itself if it is a primitive. `selected_fullkey` is the walk-root's fullkey (which is
/// also the container path shared by every child).
fn each_rows(selected: &Json, selected_fullkey: &str, rows: &mut Vec<Vec<Value>>) {
    match selected {
        Json::Array(items) => {
            for (i, item) in items.iter().enumerate() {
                let mut fk = selected_fullkey.to_string();
                append_index(&mut fk, i);
                rows.push(make_row(Value::Integer(i as i64), item, i as i64, None, fk, selected_fullkey));
            }
        }
        Json::Object(members) => {
            for (i, (label, v)) in members.iter().enumerate() {
                let mut fk = selected_fullkey.to_string();
                append_label(&mut fk, label);
                rows.push(make_row(Value::Text(label.clone()), v, i as i64, None, fk, selected_fullkey));
            }
        }
        // A primitive selected element yields exactly one row for itself: key NULL, and
        // (the primitive-start exception) path == its own fullkey.
        primitive => {
            rows.push(make_row(Value::Null, primitive, 0, None, selected_fullkey.to_string(), selected_fullkey));
        }
    }
}

/// `json_tree`: emit `node` (pre-order) and recurse into its children depth-first.
/// `fullkey` is `node`'s own path; `container` is its container's path; `parent` is the
/// container's id (`None` at the walk root); `is_root` distinguishes the walk-root row
/// (whose `path` is its container, except when it is a primitive — then its own fullkey).
#[allow(clippy::too_many_arguments)]
fn tree_rows(
    node: &Json,
    key: Value,
    fullkey: &str,
    container: &str,
    parent: Option<i64>,
    is_root: bool,
    next_id: &mut i64,
    rows: &mut Vec<Vec<Value>>,
) {
    let my_id = *next_id;
    *next_id += 1;
    // Every non-root row's path is its container's fullkey. The walk root's path is its
    // container too, EXCEPT a primitive walk root (single-row output), whose path is its
    // own fullkey (json1.html §4.24).
    let path = if is_root && is_primitive(node) { fullkey } else { container };
    // `fullkey` is cloned here (unlike the `each_rows` sites, which move it): `tree_rows`
    // reuses the borrowed `fullkey` as the prefix for each child's fullkey below, so it
    // cannot give up ownership of it to the row.
    rows.push(make_row(key, node, my_id, parent, fullkey.to_string(), path));

    match node {
        Json::Array(items) => {
            for (i, item) in items.iter().enumerate() {
                let mut child_fk = fullkey.to_string();
                append_index(&mut child_fk, i);
                tree_rows(item, Value::Integer(i as i64), &child_fk, fullkey, Some(my_id), false, next_id, rows);
            }
        }
        Json::Object(members) => {
            for (label, v) in members {
                let mut child_fk = fullkey.to_string();
                append_label(&mut child_fk, label);
                tree_rows(v, Value::Text(label.clone()), &child_fk, fullkey, Some(my_id), false, next_id, rows);
            }
        }
        _ => {}
    }
}

/// Whether `node` is a primitive (not an array or object).
fn is_primitive(node: &Json) -> bool {
    !matches!(node, Json::Array(_) | Json::Object(_))
}

/// Assemble one output row `[key, value, type, atom, id, parent, fullkey, path]`.
///
/// `fullkey` is taken BY VALUE: every caller builds a fresh per-row `fullkey` string (the
/// `each_rows` children and the primitive/tree emits), so moving it straight into the
/// row's `Value::Text` avoids a second heap allocation + copy of a string that would
/// otherwise be borrowed here and dropped by the caller. `path` stays borrowed: it is the
/// container path SHARED across a walk's rows (each child of one container reuses it), so
/// it cannot be moved and is copied once per row.
fn make_row(
    key: Value,
    node: &Json,
    id: i64,
    parent: Option<i64>,
    fullkey: String,
    path: &str,
) -> Vec<Value> {
    let value = node.to_sql_scalar();
    // `atom` is the SQL value for a primitive, NULL for a container.
    let atom = if is_primitive(node) { value.clone() } else { Value::Null };
    vec![
        key,
        value,
        Value::Text(node.type_name().to_string()),
        atom,
        Value::Integer(id),
        parent.map(Value::Integer).unwrap_or(Value::Null),
        Value::Text(fullkey),
        Value::Text(path.to_string()),
    ]
}

/// Append `.label` (or `."label"` when the label is not a simple identifier) to a path
/// under construction, matching SQLite's `fullkey`/`path` rendering for object members.
fn append_label(path: &mut String, label: &str) {
    path.push('.');
    if is_simple_label(label) {
        path.push_str(label);
    } else {
        // A non-identifier label is double-quoted with JSON string escaping, so the
        // rendered path round-trips through the path parser's quoted-label form.
        escape_string_into(label, path);
    }
}

/// Append `[n]` (a resolved array subscript) to a path under construction.
fn append_index(path: &mut String, n: usize) {
    // Writing to a String is infallible; the `_ =` documents that we discard the Result.
    let _ = write!(path, "[{n}]");
}

/// Whether `label` renders bare (`.label`) in a path: a non-empty run of ASCII
/// letters/digits/underscore that does not start with a digit. Any other label
/// (empty, leading digit, spaces, punctuation, non-ASCII) is quoted. This matches the
/// simple object labels the json1.html examples use; exotic labels are quoted, which is
/// a safe over-approximation of SQLite's exact predicate.
fn is_simple_label(label: &str) -> bool {
    let mut chars = label.chars();
    match chars.next() {
        Some(c) if c == '_' || c.is_ascii_alphabetic() => {}
        _ => return false,
    }
    chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Column indices for readability in the assertions.
    const KEY: usize = 0;
    const VALUE: usize = 1;
    const TYPE: usize = 2;
    const ATOM: usize = 3;
    const ID: usize = 4;
    const PARENT: usize = 5;
    const FULLKEY: usize = 6;
    const PATH: usize = 7;

    /// Structural [`Value`] equality (`Value` derives neither `PartialEq` nor `Eq`, so
    /// the tests cannot use `==`; reals compare by bit pattern).
    fn veq(a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Null, Value::Null) => true,
            (Value::Integer(x), Value::Integer(y)) => x == y,
            (Value::Real(x), Value::Real(y)) => x.to_bits() == y.to_bits(),
            (Value::Text(x), Value::Text(y)) => x == y,
            (Value::Blob(x), Value::Blob(y)) => x == y,
            _ => false,
        }
    }

    /// Assert column `c` of row `r` equals `expected`.
    fn cell_eq(rows: &[Vec<Value>], r: usize, c: usize, expected: Value) {
        assert!(
            veq(&rows[r][c], &expected),
            "row {r} col {c}: got {:?}, want {expected:?}",
            rows[r][c]
        );
    }

    fn each(json: &str) -> Vec<Vec<Value>> {
        json_table_rows(JsonTableKind::Each, &Value::Text(json.to_string()), None).unwrap().rows
    }
    fn each_p(json: &str, p: &str) -> Vec<Vec<Value>> {
        json_table_rows(
            JsonTableKind::Each,
            &Value::Text(json.to_string()),
            Some(&Value::Text(p.to_string())),
        )
        .unwrap()
        .rows
    }
    fn tree(json: &str) -> Vec<Vec<Value>> {
        json_table_rows(JsonTableKind::Tree, &Value::Text(json.to_string()), None).unwrap().rows
    }
    fn tree_p(json: &str, p: &str) -> Vec<Vec<Value>> {
        json_table_rows(
            JsonTableKind::Tree,
            &Value::Text(json.to_string()),
            Some(&Value::Text(p.to_string())),
        )
        .unwrap()
        .rows
    }

    fn text(s: &str) -> Value {
        Value::Text(s.to_string())
    }
    fn int(i: i64) -> Value {
        Value::Integer(i)
    }

    #[test]
    fn each_over_array_yields_index_keyed_children() {
        let rows = each("[7,8,9]");
        assert_eq!(rows.len(), 3);
        // key is the array index; value/atom the element; parent always NULL.
        for (i, v) in [7, 8, 9].into_iter().enumerate() {
            cell_eq(&rows, i, KEY, int(i as i64));
            cell_eq(&rows, i, VALUE, int(v));
            cell_eq(&rows, i, ATOM, int(v));
            cell_eq(&rows, i, TYPE, text("integer"));
            cell_eq(&rows, i, PARENT, Value::Null);
            cell_eq(&rows, i, FULLKEY, text(&format!("$[{i}]")));
            cell_eq(&rows, i, PATH, text("$"));
        }
    }

    #[test]
    fn each_over_object_yields_label_keyed_children_in_order() {
        let rows = each(r#"{"a":1,"b":2}"#);
        assert_eq!(rows.len(), 2);
        cell_eq(&rows, 0, KEY, text("a"));
        cell_eq(&rows, 0, FULLKEY, text("$.a"));
        cell_eq(&rows, 0, PATH, text("$"));
        cell_eq(&rows, 1, KEY, text("b"));
        cell_eq(&rows, 1, VALUE, int(2));
    }

    #[test]
    fn each_over_primitive_yields_one_row_with_null_key() {
        // A primitive top-level element: one row, key NULL, path == fullkey == "$".
        let rows = each("42");
        assert_eq!(rows.len(), 1);
        cell_eq(&rows, 0, KEY, Value::Null);
        cell_eq(&rows, 0, VALUE, int(42));
        cell_eq(&rows, 0, ATOM, int(42));
        cell_eq(&rows, 0, FULLKEY, text("$"));
        cell_eq(&rows, 0, PATH, text("$"));
    }

    #[test]
    fn each_container_child_value_is_json_text_and_atom_null() {
        // A nested container child: value is its canonical JSON text, atom is NULL.
        let rows = each("[[1,2],3]");
        cell_eq(&rows, 0, TYPE, text("array"));
        cell_eq(&rows, 0, VALUE, text("[1,2]"));
        cell_eq(&rows, 0, ATOM, Value::Null);
        cell_eq(&rows, 1, VALUE, int(3));
    }

    #[test]
    fn each_with_path_walks_the_selected_element() {
        // json_each(X,'$.a') walks the array at $.a; fullkey carries the $.a prefix.
        let rows = each_p(r#"{"a":[10,20]}"#, "$.a");
        assert_eq!(rows.len(), 2);
        cell_eq(&rows, 0, KEY, int(0));
        cell_eq(&rows, 0, VALUE, int(10));
        cell_eq(&rows, 0, FULLKEY, text("$.a[0]"));
        cell_eq(&rows, 0, PATH, text("$.a"));
        cell_eq(&rows, 1, FULLKEY, text("$.a[1]"));
    }

    #[test]
    fn each_with_path_to_primitive_is_single_row_path_equals_fullkey() {
        // Selecting a primitive via a path: one row, path == fullkey == "$.a".
        let rows = each_p(r#"{"a":5}"#, "$.a");
        assert_eq!(rows.len(), 1);
        cell_eq(&rows, 0, KEY, Value::Null);
        cell_eq(&rows, 0, VALUE, int(5));
        cell_eq(&rows, 0, FULLKEY, text("$.a"));
        cell_eq(&rows, 0, PATH, text("$.a"));
    }

    #[test]
    fn each_from_end_index_resolves_in_fullkey() {
        // A `[#-1]` start path resolves to a concrete index in the fullkey.
        let rows = each_p("[[9,8,7]]", "$[#-1]");
        assert_eq!(rows.len(), 3);
        cell_eq(&rows, 0, FULLKEY, text("$[0][0]"));
        cell_eq(&rows, 0, PATH, text("$[0]"));
    }

    /// Extract the integer id of row `r`.
    fn id_of(rows: &[Vec<Value>], r: usize) -> i64 {
        match &rows[r][ID] {
            Value::Integer(i) => *i,
            other => panic!("id must be an integer, got {other:?}"),
        }
    }

    #[test]
    fn tree_root_first_then_depth_first_children() {
        // json_tree('{"a":1,"b":[2,3]}') — root object, then a, then b, then b's items.
        let rows = tree(r#"{"a":1,"b":[2,3]}"#);
        assert_eq!(rows.len(), 5);
        // Root object: key NULL, parent NULL, fullkey/path "$", value is the whole doc.
        cell_eq(&rows, 0, KEY, Value::Null);
        cell_eq(&rows, 0, TYPE, text("object"));
        cell_eq(&rows, 0, PARENT, Value::Null);
        cell_eq(&rows, 0, FULLKEY, text("$"));
        cell_eq(&rows, 0, PATH, text("$"));
        let root_id = id_of(&rows, 0);
        // Member a.
        cell_eq(&rows, 1, KEY, text("a"));
        cell_eq(&rows, 1, VALUE, int(1));
        cell_eq(&rows, 1, PARENT, int(root_id));
        cell_eq(&rows, 1, FULLKEY, text("$.a"));
        cell_eq(&rows, 1, PATH, text("$"));
        // Member b (an array): value is its JSON text, atom NULL.
        cell_eq(&rows, 2, KEY, text("b"));
        cell_eq(&rows, 2, TYPE, text("array"));
        cell_eq(&rows, 2, VALUE, text("[2,3]"));
        cell_eq(&rows, 2, ATOM, Value::Null);
        cell_eq(&rows, 2, PARENT, int(root_id));
        let b_id = id_of(&rows, 2);
        // Elements of b: parent is b's id, path is "$.b".
        cell_eq(&rows, 3, KEY, int(0));
        cell_eq(&rows, 3, VALUE, int(2));
        cell_eq(&rows, 3, PARENT, int(b_id));
        cell_eq(&rows, 3, FULLKEY, text("$.b[0]"));
        cell_eq(&rows, 3, PATH, text("$.b"));
        cell_eq(&rows, 4, FULLKEY, text("$.b[1]"));
    }

    #[test]
    fn tree_ids_are_unique_and_parents_reference_real_ids() {
        // Every id distinct; every non-NULL parent is some row's id (structural
        // consistency, which is the guarantee json1.html actually makes).
        let rows = tree(r#"{"x":[1,{"y":2}],"z":3}"#);
        let ids: Vec<i64> = (0..rows.len()).map(|r| id_of(&rows, r)).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), ids.len(), "ids must be unique");
        for r in &rows {
            if let Value::Integer(p) = r[PARENT] {
                assert!(ids.contains(&p), "parent {p} must reference a real id");
            }
        }
    }

    #[test]
    fn tree_with_path_reparents_and_prefixes_fullkey() {
        // json_tree(X,'$.b') — the selected array is the root row (key NULL, parent
        // NULL); its fullkey carries the $.b prefix and its path is the container "$".
        let rows = tree_p(r#"{"a":1,"b":[2,3]}"#, "$.b");
        assert_eq!(rows.len(), 3);
        cell_eq(&rows, 0, KEY, Value::Null);
        cell_eq(&rows, 0, TYPE, text("array"));
        cell_eq(&rows, 0, PARENT, Value::Null);
        cell_eq(&rows, 0, FULLKEY, text("$.b"));
        cell_eq(&rows, 0, PATH, text("$"));
        cell_eq(&rows, 1, FULLKEY, text("$.b[0]"));
        cell_eq(&rows, 1, PATH, text("$.b"));
    }

    #[test]
    fn null_document_yields_no_rows() {
        // A SQL NULL document → no rows (distinct from the JSON text "null").
        assert!(json_table_rows(JsonTableKind::Each, &Value::Null, None).unwrap().rows.is_empty());
        assert!(json_table_rows(JsonTableKind::Tree, &Value::Null, None).unwrap().rows.is_empty());
        // JSON text "null" is a one-element primitive.
        let rows = each("null");
        assert_eq!(rows.len(), 1);
        cell_eq(&rows, 0, TYPE, text("null"));
        cell_eq(&rows, 0, VALUE, Value::Null);
    }

    #[test]
    fn null_path_yields_no_rows() {
        let out = json_table_rows(
            JsonTableKind::Each,
            &Value::Text("[1,2]".into()),
            Some(&Value::Null),
        )
        .unwrap();
        assert!(out.rows.is_empty());
    }

    #[test]
    fn hidden_root_value_is_the_start_path() {
        // The hidden `root` column is the start-path text, defaulting to `$` (json1.html
        // §4.24). The hidden `json` column is materialized by the executor from its own
        // argument value (verified end-to-end in `conformance_json_each`), not here.
        let out = json_table_rows(JsonTableKind::Each, &Value::Text("[1,2,3]".into()), None).unwrap();
        assert!(veq(&out.root, &text("$")), "root defaults to $: {:?}", out.root);
        // With an explicit path, `root` is that path text.
        let out = json_table_rows(
            JsonTableKind::Tree,
            &Value::Text(r#"{"a":[1]}"#.into()),
            Some(&Value::Text("$.a".into())),
        )
        .unwrap();
        assert!(veq(&out.root, &text("$.a")), "root is the 2nd argument: {:?}", out.root);
    }

    #[test]
    fn each_over_object_preserves_insertion_order_not_sorted() {
        // Object members keep DOCUMENT order, which is NOT the same as sorted-by-label:
        // an out-of-order object must come back in its written order (an alphabetizing
        // implementation would return a, b, c and fail here).
        let rows = each(r#"{"c":1,"a":2,"b":3}"#);
        assert_eq!(rows.len(), 3);
        cell_eq(&rows, 0, KEY, text("c"));
        cell_eq(&rows, 0, VALUE, int(1));
        cell_eq(&rows, 1, KEY, text("a"));
        cell_eq(&rows, 1, VALUE, int(2));
        cell_eq(&rows, 2, KEY, text("b"));
        cell_eq(&rows, 2, VALUE, int(3));
    }

    #[test]
    fn tree_is_depth_first_not_breadth_first() {
        // Two sibling containers distinguish DFS from BFS: DFS fully descends `a` before
        // visiting `b`; BFS would list `$.a`,`$.b` before any leaves.
        let rows = tree(r#"{"a":[1,2],"b":[3,4]}"#);
        let fullkeys: Vec<&str> = rows
            .iter()
            .map(|r| match &r[FULLKEY] {
                Value::Text(s) => s.as_str(),
                other => panic!("fullkey must be text, got {other:?}"),
            })
            .collect();
        assert_eq!(
            fullkeys,
            ["$", "$.a", "$.a[0]", "$.a[1]", "$.b", "$.b[0]", "$.b[1]"],
            "json_tree must be depth-first in document order"
        );
    }

    #[test]
    fn path_selecting_nothing_yields_no_rows() {
        assert!(each_p(r#"{"a":1}"#, "$.missing").is_empty());
        assert!(tree_p("[1,2]", "$[9]").is_empty());
    }

    #[test]
    fn malformed_json_is_an_error() {
        let err = json_table_rows(JsonTableKind::Each, &Value::Text("{bad".into()), None);
        assert!(err.is_err(), "malformed JSON must be an error");
        let err = json_table_rows(JsonTableKind::Tree, &Value::Text("[1,2".into()), None);
        assert!(err.is_err());
    }

    #[test]
    fn bad_path_is_an_error() {
        let err = json_table_rows(
            JsonTableKind::Each,
            &Value::Text("[1]".into()),
            Some(&Value::Text("a".into())),
        );
        assert!(err.is_err(), "a path not starting with $ must be an error");
    }

    #[test]
    fn quoted_label_in_fullkey_when_not_a_simple_identifier() {
        // A label with a space is double-quoted in the fullkey.
        let rows = each(r#"{"a b":1}"#);
        cell_eq(&rows, 0, KEY, text("a b"));
        cell_eq(&rows, 0, FULLKEY, text(r#"$."a b""#));
    }

    #[test]
    fn true_false_types_and_values() {
        let rows = each("[true,false]");
        cell_eq(&rows, 0, TYPE, text("true"));
        cell_eq(&rows, 0, VALUE, int(1));
        cell_eq(&rows, 1, TYPE, text("false"));
        cell_eq(&rows, 1, VALUE, int(0));
    }
}
