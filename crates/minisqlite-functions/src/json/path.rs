//! JSONPath parsing, read navigation, and in-place edits (`spec/sqlite-doc/json1.html`
//! §3.3, §4.5, §4.8, §4.11, §4.18).
//!
//! A well-formed PATH is `$` followed by zero or more `.label` or `[index]` steps.
//! An array index is a non-negative integer `N`, the from-the-right form `#-N`
//! (`#-1` is the last element), or the bare `#` used only as an *append* target for
//! `json_insert`/`json_set`. Labels may be quoted (`."a b"`) to carry characters a
//! bare identifier cannot.
//!
//! Two navigation modes share the parsed step list:
//!
//! * **Read** ([`navigate`]) — follow existing structure and return the selected
//!   node, or `None` if the path selects nothing. Used by `json_extract`,
//!   `json_type`, and `json_array_length`.
//! * **Edit** ([`apply_edit`], [`remove_at`]) — walk to the parent of the final
//!   step and mutate. Create/overwrite is governed by [`SetMode`] so one walk backs
//!   `json_insert` (create only), `json_replace` (overwrite only), and `json_set`
//!   (both). A missing intermediate container makes the edit a no-op, matching
//!   SQLite (only the final step is created).

use std::borrow::Cow;

use minisqlite_types::{integer_to_text, real_to_text, Error, Result, Value};

use super::value::Json;

/// One step of a parsed path.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Step {
    /// `.label` — an object member by name.
    Key(String),
    /// `[index]` — an array element.
    Index(Index),
}

/// An array subscript within a path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Index {
    /// `[N]` — the N-th element from the start (0-based).
    At(usize),
    /// `[#-N]` — the N-th element from the end (`#-1` is the last).
    FromEnd(usize),
    /// `[#]` — the append position (one past the end); only meaningful as an edit
    /// target, and selects nothing on a read.
    Append,
}

/// How an edit creates vs. overwrites at the final path step (json1.html §4.11).
#[derive(Debug, Clone, Copy)]
pub(crate) enum SetMode {
    /// `json_insert`: create when absent, never overwrite.
    Insert,
    /// `json_replace`: overwrite when present, never create.
    Replace,
    /// `json_set`: create when absent and overwrite when present.
    Set,
}

impl SetMode {
    fn creates(self) -> bool {
        matches!(self, SetMode::Insert | SetMode::Set)
    }
    fn overwrites(self) -> bool {
        matches!(self, SetMode::Replace | SetMode::Set)
    }
}

/// The SQLite error for a malformed PATH argument (json1.html §3.3). `near` is the
/// unparsed remainder, matching SQLite's "JSON path error near '…'" shape.
fn path_error(near: &str) -> Error {
    Error::sql(format!("JSON path error near '{near}'"))
}

/// Coerce a value to text for use as a PATH, an object label, or a `json_pretty`
/// indent. NULL is handled by callers before this point; the `Null` arm yields the
/// empty string only to stay total (an empty path fails the leading-`$` check and
/// is reported as an error).
pub(crate) fn value_text(v: &Value) -> Cow<'_, str> {
    match v {
        Value::Text(s) => Cow::Borrowed(s.as_str()),
        Value::Integer(i) => Cow::Owned(integer_to_text(*i)),
        Value::Real(r) => Cow::Owned(real_to_text(*r)),
        Value::Blob(b) => String::from_utf8_lossy(b),
        Value::Null => Cow::Borrowed(""),
    }
}

/// Parse a path string into its steps. Errors (with SQLite's wording shape) when it
/// does not begin with exactly one `$`, or a step is malformed.
pub(crate) fn parse_path(path: &str) -> Result<Vec<Step>> {
    let b = path.as_bytes();
    if b.first() != Some(&b'$') {
        return Err(path_error(path));
    }
    let mut i = 1;
    let mut steps = Vec::new();
    while i < b.len() {
        match b[i] {
            b'.' => {
                i += 1;
                let (key, next) = parse_label(path, i)?;
                steps.push(Step::Key(key));
                i = next;
            }
            b'[' => {
                i += 1;
                let (index, next) = parse_index(path, i)?;
                steps.push(Step::Index(index));
                i = next;
            }
            _ => return Err(path_error(&path[i..])),
        }
    }
    Ok(steps)
}

/// Parse an object label at byte offset `i` (just past the `.`): either a
/// double-quoted string or a bare run up to the next `.`/`[`. Returns the decoded
/// label and the offset after it.
fn parse_label(path: &str, i: usize) -> Result<(String, usize)> {
    let b = path.as_bytes();
    if b.get(i) == Some(&b'"') {
        return parse_quoted_label(path, i);
    }
    let start = i;
    let mut j = i;
    while j < b.len() && b[j] != b'.' && b[j] != b'[' {
        j += 1;
    }
    if j == start {
        return Err(path_error(&path[start..]));
    }
    Ok((path[start..j].to_string(), j))
}

/// Parse a double-quoted object label (the opening quote is at `i`), returning the
/// decoded label and the offset just past the closing quote. Decoding delegates to
/// the parser's single JSON-string decoder ([`super::parse::decode_quoted`]) so a
/// quoted label resolves to the *same* string as the object key it targets —
/// surrogate pairs and every JSON5 escape included (json1.html §3.9). A malformed or
/// unterminated label is a path error.
fn parse_quoted_label(path: &str, i: usize) -> Result<(String, usize)> {
    super::parse::decode_quoted(path, i).ok_or_else(|| path_error(&path[i..]))
}

/// Parse an array index at byte offset `i` (just past the `[`): `N`, `#`, or `#-N`,
/// terminated by `]`. Returns the index and the offset after the `]`.
fn parse_index(path: &str, i: usize) -> Result<(Index, usize)> {
    let b = path.as_bytes();
    let (index, mut j) = if b.get(i) == Some(&b'#') {
        let mut j = i + 1;
        if b.get(j) == Some(&b'-') {
            j += 1;
            let (n, nj) = parse_uint(path, j)?;
            (Index::FromEnd(n), nj)
        } else {
            (Index::Append, j)
        }
    } else {
        let (n, nj) = parse_uint(path, i)?;
        (Index::At(n), nj)
    };
    if b.get(j) != Some(&b']') {
        return Err(path_error(&path[i..]));
    }
    j += 1;
    Ok((index, j))
}

/// Parse one or more decimal digits at byte offset `i` into a `usize` (saturating on
/// overflow), returning the value and the offset after the digits.
fn parse_uint(path: &str, i: usize) -> Result<(usize, usize)> {
    let b = path.as_bytes();
    let start = i;
    let mut j = i;
    let mut n: usize = 0;
    while j < b.len() && b[j].is_ascii_digit() {
        n = n.saturating_mul(10).saturating_add((b[j] - b'0') as usize);
        j += 1;
    }
    if j == start {
        return Err(path_error(&path[start..]));
    }
    Ok((n, j))
}

/// Resolve an array subscript against a known length for a *read*: `Some(i)` when it
/// selects an existing element, `None` otherwise. `[#]` and out-of-range subscripts
/// select nothing. Shared with the `json_each`/`json_tree` walk ([`super::table`]),
/// which resolves each `[#-N]` subscript of a start path to a concrete index for the
/// element's `fullkey`.
pub(crate) fn resolve_index_read(idx: Index, len: usize) -> Option<usize> {
    match idx {
        Index::At(n) => (n < len).then_some(n),
        Index::FromEnd(k) => len.checked_sub(k).filter(|&i| i < len),
        Index::Append => None,
    }
}

/// Follow `steps` from `root`, returning the selected node or `None` if the path
/// selects nothing (a missing key/index or a type mismatch such as a key applied to
/// an array). On a duplicate object label the first occurrence wins.
///
/// INVARIANT (keep in lockstep with [`super::table::locate`]): `json_each`/`json_tree`
/// re-implement this same per-step walk in `table::locate` so they can additionally
/// record the resolved fullkey. Any change to the selection semantics here (duplicate-key
/// handling, subscript resolution, a JSON5 label nuance) MUST be mirrored there, or the
/// same path text would resolve differently for `json_extract` vs `json_each`/`json_tree`.
pub(crate) fn navigate<'a>(root: &'a Json, steps: &[Step]) -> Option<&'a Json> {
    let mut cur = root;
    for step in steps {
        cur = match (cur, step) {
            (Json::Object(members), Step::Key(k)) => {
                members.iter().find(|(mk, _)| mk == k).map(|(_, v)| v)?
            }
            (Json::Array(items), Step::Index(idx)) => {
                let n = resolve_index_read(*idx, items.len())?;
                items.get(n)?
            }
            _ => return None,
        };
    }
    Some(cur)
}

/// Walk to the mutable parent container of the final step, following existing
/// structure. Returns `None` if any intermediate step is missing or hits a type
/// mismatch (in which case the caller leaves the value unchanged).
fn navigate_parent_mut<'a>(root: &'a mut Json, steps: &[Step]) -> Option<&'a mut Json> {
    let mut cur = root;
    for step in &steps[..steps.len() - 1] {
        cur = match (cur, step) {
            (Json::Object(members), Step::Key(k)) => {
                let mut found = None;
                for (mk, mv) in members.iter_mut() {
                    if *mk == *k {
                        found = Some(mv);
                        break;
                    }
                }
                found?
            }
            (Json::Array(items), Step::Index(idx)) => {
                let n = resolve_index_read(*idx, items.len())?;
                items.get_mut(n)?
            }
            _ => return None,
        };
    }
    Some(cur)
}

/// Apply one path/value edit under [`SetMode`]. An empty step list is the root path
/// `$`: overwrite modes replace the whole document, `Insert` is a no-op (the root
/// already exists). Otherwise walk to the parent and create/overwrite the final
/// step; a missing parent or a type mismatch leaves the value unchanged.
pub(crate) fn apply_edit(root: &mut Json, steps: &[Step], value: Json, mode: SetMode) {
    if steps.is_empty() {
        if mode.overwrites() {
            *root = value;
        }
        return;
    }
    let last = &steps[steps.len() - 1];
    let Some(parent) = navigate_parent_mut(root, steps) else {
        return;
    };
    match (parent, last) {
        (Json::Object(members), Step::Key(k)) => {
            let existing = members.iter_mut().find(|(mk, _)| *mk == *k);
            match existing {
                Some((_, v)) => {
                    if mode.overwrites() {
                        *v = value;
                    }
                }
                None => {
                    if mode.creates() {
                        members.push((k.clone(), value));
                    }
                }
            }
        }
        (Json::Array(items), Step::Index(idx)) => match idx {
            Index::At(n) => {
                if *n < items.len() {
                    if mode.overwrites() {
                        items[*n] = value;
                    }
                } else if mode.creates() {
                    // An index at or past the end appends one element.
                    items.push(value);
                }
            }
            Index::FromEnd(k) => {
                if let Some(i) = resolve_index_read(Index::FromEnd(*k), items.len()) {
                    if mode.overwrites() {
                        items[i] = value;
                    }
                }
                // Out of range from the end: no create target, so a no-op.
            }
            Index::Append => {
                if mode.creates() {
                    items.push(value);
                }
            }
        },
        // Type mismatch (e.g. a key on an array): silently ignored, as SQLite does.
        _ => {}
    }
}

/// Remove the element selected by `steps` in place. A missing element, a `[#]`/out-
/// of-range subscript, or a type mismatch is a silent no-op. Array removal shifts
/// later elements down (so sequential removes see the shifted positions). The root
/// path `$` is handled by the caller (it makes the whole result NULL).
pub(crate) fn remove_at(root: &mut Json, steps: &[Step]) {
    if steps.is_empty() {
        return; // caller handles `$` removal
    }
    let last = &steps[steps.len() - 1];
    let Some(parent) = navigate_parent_mut(root, steps) else {
        return;
    };
    match (parent, last) {
        (Json::Object(members), Step::Key(k)) => {
            if let Some(pos) = members.iter().position(|(mk, _)| *mk == *k) {
                members.remove(pos);
            }
        }
        (Json::Array(items), Step::Index(idx)) => {
            if let Some(i) = resolve_index_read(*idx, items.len()) {
                items.remove(i);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn steps(path: &str) -> Vec<Step> {
        parse_path(path).unwrap_or_else(|_| panic!("{path:?} should parse"))
    }

    #[test]
    fn parses_path_steps() {
        assert_eq!(steps("$"), vec![]);
        assert_eq!(steps("$.a"), vec![Step::Key("a".into())]);
        assert_eq!(
            steps("$.a[1]"),
            vec![Step::Key("a".into()), Step::Index(Index::At(1))]
        );
        assert_eq!(steps("$[#-1]"), vec![Step::Index(Index::FromEnd(1))]);
        assert_eq!(steps("$[#]"), vec![Step::Index(Index::Append)]);
        // Quoted label with a space.
        assert_eq!(steps(r#"$."a b""#), vec![Step::Key("a b".into())]);
        // Nested.
        assert_eq!(
            steps("$.c[2].f"),
            vec![Step::Key("c".into()), Step::Index(Index::At(2)), Step::Key("f".into())]
        );
    }

    #[test]
    fn rejects_bad_paths() {
        for bad in ["a", "", "$.", "$[", "$[x]", "$[1", "$[#-]"] {
            assert!(parse_path(bad).is_err(), "{bad:?} should be a path error");
        }
    }

    #[test]
    fn does_not_panic_on_hostile_paths() {
        // Any byte sequence must parse or error, never panic (no unchecked
        // indexing/slicing). Mixes truncated escapes, huge indices, and multibyte.
        for p in [
            "$", "$.", "$..", "$[", "$[]", "$[#", "$[#-", "$['a']", "$.a[", "$[-1]",
            "$[99999999999999999999999999]", "$\u{80}", "$.\u{1F600}", "$[#-999]",
            "\\", "$.\"", "$.\"\\", "$.\"\\u", "$.\"\\uZZZZ\"", "$.\"unterminated",
        ] {
            let _ = parse_path(p);
        }
    }

    #[test]
    fn navigate_reads_nodes() {
        let root = Json::Object(vec![
            ("a".into(), Json::Integer(2)),
            (
                "c".into(),
                Json::Array(vec![
                    Json::Integer(4),
                    Json::Integer(5),
                    Json::Object(vec![("f".into(), Json::Integer(7))]),
                ]),
            ),
        ]);
        assert_eq!(navigate(&root, &steps("$.a")), Some(&Json::Integer(2)));
        assert_eq!(navigate(&root, &steps("$.c[2].f")), Some(&Json::Integer(7)));
        // Last-relative index.
        assert_eq!(navigate(&root, &steps("$.c[#-1]")), Some(&Json::Object(vec![("f".into(), Json::Integer(7))])));
        // Missing / out of range / type mismatch -> None.
        assert_eq!(navigate(&root, &steps("$.x")), None);
        assert_eq!(navigate(&root, &steps("$.c[9]")), None);
        assert_eq!(navigate(&root, &steps("$.a.b")), None); // key on a number
        assert_eq!(navigate(&root, &steps("$[#]")), None); // append selects nothing on read
    }

    #[test]
    fn navigate_reads_first_of_duplicate_labels() {
        // On a read, the first occurrence of a duplicated key wins (json1.html §4.5).
        let root = Json::Object(vec![
            ("a".into(), Json::Integer(1)),
            ("a".into(), Json::Integer(2)),
        ]);
        assert_eq!(navigate(&root, &steps("$.a")), Some(&Json::Integer(1)));
    }

    #[test]
    fn quoted_label_decodes_like_the_object_key_it_targets() {
        // Regression: the path-label decoder must match the JSON string decoder so a
        // quoted label finds the key. A surrogate pair and a JSON5 escape are the
        // cases the old duplicate decoder got wrong (it produced garbage and the
        // navigation silently missed the key).
        let root = Json::Object(vec![
            ("\u{1F600}".into(), Json::Integer(1)), // key "😀"
            ("a\u{0b}b".into(), Json::Integer(2)),  // key with an embedded vertical tab
        ]);
        // Surrogate-pair escape resolves to the astral key.
        assert_eq!(navigate(&root, &steps(r#"$."\uD83D\uDE00""#)), Some(&Json::Integer(1)));
        // JSON5 `\v` escape resolves to the vertical-tab key.
        assert_eq!(navigate(&root, &steps(r#"$."a\vb""#)), Some(&Json::Integer(2)));
        // Plain `\uXXXX` (BMP) still works.
        let bmp = Json::Object(vec![("A".into(), Json::Integer(3))]);
        assert_eq!(navigate(&bmp, &steps(r#"$."\u0041""#)), Some(&Json::Integer(3)));
    }

    #[test]
    fn set_overwrites_and_creates() {
        let mut root = Json::Object(vec![("a".into(), Json::Integer(2)), ("c".into(), Json::Integer(4))]);
        // set overwrites an existing key.
        apply_edit(&mut root, &steps("$.a"), Json::Integer(99), SetMode::Set);
        assert_eq!(navigate(&root, &steps("$.a")), Some(&Json::Integer(99)));
        // set creates a missing key (appended in order).
        apply_edit(&mut root, &steps("$.e"), Json::Integer(9), SetMode::Set);
        assert_eq!(root.to_text(), r#"{"a":99,"c":4,"e":9}"#);
    }

    #[test]
    fn insert_only_creates_replace_only_overwrites() {
        let mut root = Json::Object(vec![("a".into(), Json::Integer(2))]);
        // insert does NOT overwrite an existing key.
        apply_edit(&mut root, &steps("$.a"), Json::Integer(99), SetMode::Insert);
        assert_eq!(navigate(&root, &steps("$.a")), Some(&Json::Integer(2)));
        // replace does NOT create a missing key.
        apply_edit(&mut root, &steps("$.z"), Json::Integer(1), SetMode::Replace);
        assert_eq!(navigate(&root, &steps("$.z")), None);
        // insert DOES create a missing key.
        apply_edit(&mut root, &steps("$.z"), Json::Integer(1), SetMode::Insert);
        assert_eq!(navigate(&root, &steps("$.z")), Some(&Json::Integer(1)));
    }

    #[test]
    fn array_append_and_index_edits() {
        let mut root = Json::Array(vec![Json::Integer(1), Json::Integer(2), Json::Integer(3)]);
        // `[#]` appends.
        apply_edit(&mut root, &steps("$[#]"), Json::Integer(99), SetMode::Insert);
        assert_eq!(root.to_text(), "[1,2,3,99]");
        // `[1]` overwrites under set.
        apply_edit(&mut root, &steps("$[1]"), Json::Integer(22), SetMode::Set);
        assert_eq!(root.to_text(), "[1,22,3,99]");
        // Missing intermediate -> no-op.
        apply_edit(&mut root, &steps("$.x.y"), Json::Integer(5), SetMode::Set);
        assert_eq!(root.to_text(), "[1,22,3,99]");
    }

    #[test]
    fn remove_shifts_array_indices() {
        let mut root = Json::Array((0..5).map(Json::Integer).collect());
        remove_at(&mut root, &steps("$[2]"));
        assert_eq!(root.to_text(), "[0,1,3,4]");
        // A second removal sees the shifted array.
        remove_at(&mut root, &steps("$[0]"));
        assert_eq!(root.to_text(), "[1,3,4]");
        // Removing a missing key is a no-op.
        let mut obj = Json::Object(vec![("x".into(), Json::Integer(1))]);
        remove_at(&mut obj, &steps("$.z"));
        assert_eq!(obj.to_text(), r#"{"x":1}"#);
    }

    #[test]
    fn root_overwrite_semantics() {
        let mut root = Json::Object(vec![("a".into(), Json::Integer(1))]);
        // set at `$` replaces the whole document.
        apply_edit(&mut root, &steps("$"), Json::Integer(5), SetMode::Set);
        assert_eq!(root, Json::Integer(5));
        // insert at `$` is a no-op (root already exists).
        apply_edit(&mut root, &steps("$"), Json::Integer(9), SetMode::Insert);
        assert_eq!(root, Json::Integer(5));
    }
}
