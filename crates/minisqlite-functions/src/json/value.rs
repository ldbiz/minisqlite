//! The in-memory JSON value model plus canonical / pretty rendering and the
//! bridge to and from SQLite [`Value`]s (`spec/sqlite-doc/json1.html` §3, §4).
//!
//! SQLite stores JSON as ordinary text; there is no distinct JSON storage class.
//! [`Json`] is the parsed tree the JSON functions operate on between reading text
//! (in `parse`) and writing canonical text back out here. Two invariants shape it:
//!
//! * **Object key order is preserved and duplicates are NOT removed.** SQLite keeps
//!   labels in insertion order and (for `json()` and friends) preserves duplicate
//!   labels (json1.html §4.1). So the object payload is an ordered `Vec`, never a
//!   map — a map would silently reorder and de-duplicate.
//! * **Integer vs real is preserved.** `json_type` must report 'integer' for `2`
//!   and 'real' for `2.5`, so the two are distinct variants rather than one number.
//!
//! String content is stored *decoded* (escape sequences resolved to their actual
//! characters). Re-escaping happens once here at render time, which keeps the model
//! simple and canonical output correct. This matches SQLite, which likewise
//! canonicalizes string escapes on a minify (e.g. `json('"\u0041"')` -> `'"A"'`, and
//! a control character re-escapes to `\uXXXX`) rather than preserving the input
//! spelling — so it is canonicalization, not a limitation.
//!
//! Numbers have two provenances, and they render differently — this is the SQLite
//! "verbatim vs canonicalized" number rule (json1.html §4.1: `json()` only strips
//! whitespace from canonical RFC-8259 input; only JSON5 spellings are converted):
//!
//! * [`Json::Number`] is a number parsed from JSON *text*. Its exact source token is
//!   preserved and emitted verbatim on minify, so `json('1e3')` → `'1e3'` and
//!   `json('[1.50]')` → `'[1.50]'` (a canonical token has no whitespace to strip and
//!   is already canonical). Only JSON5 number spellings (hex, leading `+`,
//!   leading/trailing `.`, `Infinity`) are canonicalized at parse time.
//! * [`Json::Integer`]/[`Json::Real`] are numbers *constructed from a SQL value*
//!   (`json_array(1.5)`, `json_quote(2.0)`, a numeric `json` argument). These have no
//!   source token, so they render by formatting the typed value via the engine's
//!   canonical number→text — which is the correct behavior for that provenance.

use minisqlite_types::{integer_to_text, real_to_text, Error, Result, Value};

/// SQLite's JSON value-subtype tag: ASCII `'J'` (json1.html §3.4). The JSON functions
/// stamp this on a result they produced (via
/// [`FnContext::set_result_subtype`](minisqlite_expr::FnContext::set_result_subtype))
/// so that when the result is passed *directly* as a `value` argument to another JSON
/// function, the inner function embeds it as literal JSON instead of re-quoting it as
/// a string. The subtype is ephemeral — it rides a value only within one expression
/// evaluation and never touches [`Value`] or stored rows — matching SQLite.
pub(crate) const JSON_SUBTYPE: u8 = 74;

/// The typed numeric value behind a [`NumLit`]. Kept as its own small enum (rather
/// than a [`Value`], which is not `PartialEq`) so [`Json`] can derive equality; an
/// integer token too large for i64 carries [`NumValue::Real`] while its
/// [`NumLit::is_integer`] stays true.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum NumValue {
    Int(i64),
    Real(f64),
}

/// A number parsed from JSON *text*, carrying its exact rendering so canonical
/// RFC-8259 tokens survive a minify verbatim (json1.html §4.1). `text` is the source
/// slice for a canonical number, or the canonicalized form for a JSON5 spelling;
/// `is_integer` drives `json_type` ('integer' vs 'real') independently of `value`
/// (a >i64 integer keeps type 'integer' but a `Real` `value`); `value` is the SQL
/// scalar single-path `json_extract` yields.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NumLit {
    pub(crate) text: String,
    pub(crate) is_integer: bool,
    pub(crate) value: NumValue,
}

/// A parsed JSON value: the three literals, numbers, strings, and the two
/// containers. Numbers appear as [`Json::Number`] when parsed from text (rendered
/// verbatim) and as [`Json::Integer`]/[`Json::Real`] when built from a SQL value
/// (rendered by formatting) — see the module docs for why the two exist.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Json {
    Null,
    Bool(bool),
    /// An integer built from a SQL value; renders via `integer_to_text`.
    Integer(i64),
    /// A real built from a SQL value; renders via `real_to_text`.
    Real(f64),
    /// A number parsed from JSON text; renders its preserved token verbatim.
    Number(NumLit),
    /// Decoded string content (escapes already resolved).
    Str(String),
    Array(Vec<Json>),
    /// Ordered, duplicate-preserving object members (label, value).
    Object(Vec<(String, Json)>),
}

impl Json {
    /// The `json_type()` name of this value: one of 'null', 'true', 'false',
    /// 'integer', 'real', 'text', 'array', 'object' (json1.html §4.20). Distinct
    /// from `Value::type_name` because JSON splits booleans out of the SQL storage
    /// classes and reports containers structurally.
    pub(crate) fn type_name(&self) -> &'static str {
        match self {
            Json::Null => "null",
            Json::Bool(true) => "true",
            Json::Bool(false) => "false",
            Json::Integer(_) => "integer",
            Json::Real(_) => "real",
            Json::Number(n) => {
                if n.is_integer {
                    "integer"
                } else {
                    "real"
                }
            }
            Json::Str(_) => "text",
            Json::Array(_) => "array",
            Json::Object(_) => "object",
        }
    }

    /// Render as canonical, minified RFC-8259 JSON text (no insignificant
    /// whitespace), the form every `json_*` text function returns.
    pub(crate) fn to_text(&self) -> String {
        let mut out = String::new();
        self.render_into(&mut out);
        out
    }

    /// Append the canonical rendering of `self` to `out`. Recursion depth is bounded
    /// by the parser's nesting limit (json1.html §3.5), so this cannot overflow the
    /// stack on any value the parser produced.
    pub(crate) fn render_into(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(true) => out.push_str("true"),
            Json::Bool(false) => out.push_str("false"),
            Json::Integer(i) => out.push_str(&integer_to_text(*i)),
            Json::Real(r) => out.push_str(&render_real(*r)),
            // A text-sourced number renders its preserved token verbatim.
            Json::Number(n) => out.push_str(&n.text),
            Json::Str(s) => escape_string_into(s, out),
            Json::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    item.render_into(out);
                }
                out.push(']');
            }
            Json::Object(members) => {
                out.push('{');
                for (i, (k, v)) in members.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    escape_string_into(k, out);
                    out.push(':');
                    v.render_into(out);
                }
                out.push('}');
            }
        }
    }

    /// Render as pretty-printed JSON with `indent` per nesting level (json1.html
    /// §4.17). Members/elements go one per line; a space follows every `:`; empty
    /// containers stay on one line as `{}` / `[]`.
    pub(crate) fn to_pretty(&self, indent: &str) -> String {
        let mut out = String::new();
        self.pretty_into(indent, 0, &mut out);
        out
    }

    fn pretty_into(&self, indent: &str, level: usize, out: &mut String) {
        match self {
            Json::Array(items) if !items.is_empty() => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push('\n');
                    push_indent(indent, level + 1, out);
                    item.pretty_into(indent, level + 1, out);
                }
                out.push('\n');
                push_indent(indent, level, out);
                out.push(']');
            }
            Json::Object(members) if !members.is_empty() => {
                out.push('{');
                for (i, (k, v)) in members.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push('\n');
                    push_indent(indent, level + 1, out);
                    escape_string_into(k, out);
                    out.push_str(": ");
                    v.pretty_into(indent, level + 1, out);
                }
                out.push('\n');
                push_indent(indent, level, out);
                out.push('}');
            }
            // Scalars and empty containers render exactly as canonical.
            other => other.render_into(out),
        }
    }

    /// The SQL value that single-path `json_extract`/`->>`/`json_each.value` produce
    /// for this node (json1.html §4.8): JSON null -> SQL NULL, booleans -> 0/1,
    /// numbers -> INTEGER/REAL, a JSON string -> its dequoted TEXT, and a container
    /// -> its canonical JSON TEXT.
    pub(crate) fn to_sql_scalar(&self) -> Value {
        match self {
            Json::Null => Value::Null,
            Json::Bool(true) => Value::Integer(1),
            Json::Bool(false) => Value::Integer(0),
            Json::Integer(i) => Value::Integer(*i),
            Json::Real(r) => Value::Real(*r),
            // A text-sourced number extracts as its typed SQL value (a >i64 integer
            // token carries a REAL value even though its json_type stays 'integer').
            Json::Number(n) => match n.value {
                NumValue::Int(i) => Value::Integer(i),
                NumValue::Real(r) => Value::Real(r),
            },
            Json::Str(s) => Value::Text(s.clone()),
            Json::Array(_) | Json::Object(_) => Value::Text(self.to_text()),
        }
    }
}

/// Encode a SQL [`Value`] as JSON for a `value` argument to `json_array`,
/// `json_object`, `json_insert`/`json_set`/`json_replace`, and the aggregates
/// (json1.html §3.4). A TEXT value becomes a *quoted JSON string* even if it looks
/// like JSON, an INTEGER/REAL becomes a JSON number, NULL becomes JSON null, and a
/// BLOB is an error ("JSON cannot hold BLOB values").
///
/// This is the NO-SUBTYPE fallback: it always quotes a TEXT value as a JSON string.
/// The subtype-aware embedding — where a TEXT value produced by another JSON function
/// (carrying the ephemeral JSON subtype, json1.html §3.4) is embedded as literal JSON
/// rather than re-quoted — lives in [`value_to_json_with_subtype`], which every
/// constructor's `value` operand goes through; this function is its fallback for an
/// ordinary (subtype-less) value. See the crate module docs for the subtype channel.
pub(crate) fn value_to_json(v: &Value) -> Result<Json> {
    match v {
        Value::Null => Ok(Json::Null),
        Value::Integer(i) => Ok(Json::Integer(*i)),
        Value::Real(r) => Ok(Json::Real(*r)),
        Value::Text(s) => Ok(Json::Str(s.clone())),
        Value::Blob(_) => Err(Error::sql("JSON cannot hold BLOB values")),
    }
}

/// Encode a `value` argument to a JSON node, honoring the ephemeral JSON value-subtype
/// (json1.html §3.4). When `subtype` is [`JSON_SUBTYPE`] and `v` is a TEXT value — i.e.
/// it came directly from another JSON function — the text is PARSED and embedded as its
/// actual JSON structure (so `json_array(json('[1,2]'))` is `[[1,2]]`, not `["[1,2]"]`).
/// Otherwise it falls back to [`value_to_json`], which quotes a TEXT value as a JSON
/// string even when it looks like JSON. NULL / numeric / BLOB handling is unchanged: a
/// subtype only ever rides a TEXT result, so those flow straight through the fallback.
pub(crate) fn value_to_json_with_subtype(v: &Value, subtype: u8) -> Result<Json> {
    if subtype == JSON_SUBTYPE && matches!(v, Value::Text(_)) {
        // A subtyped value is the output of a JSON function, hence well-formed JSON;
        // parse it back into structure to embed. (`parse_json_arg` errors on malformed
        // text, which a genuine subtyped value can never be — fail-closed if it were.)
        return super::parse::parse_json_arg(v);
    }
    value_to_json(v)
}

/// Render a SQL [`Value`] as a JSON value straight into `out`, with the same encoding
/// as [`value_to_json`] followed by [`Json::render_into`] (a TEXT value becomes a
/// quoted JSON string, a BLOB is an error), but without materializing an intermediate
/// [`Json`] — a TEXT value is escaped directly from its borrow, saving the `s.clone()`
/// a `Json::Str` would need. Used on the aggregate `step` hot path, which renders one
/// value per row and would otherwise allocate and drop a node every row.
pub(crate) fn render_value_into(v: &Value, out: &mut String) -> Result<()> {
    match v {
        Value::Null => out.push_str("null"),
        Value::Integer(i) => out.push_str(&integer_to_text(*i)),
        Value::Real(r) => out.push_str(&render_real(*r)),
        Value::Text(s) => escape_string_into(s, out),
        Value::Blob(_) => return Err(Error::sql("JSON cannot hold BLOB values")),
    }
    Ok(())
}

/// Render a `value` operand as a JSON value straight into `out`, honoring the ephemeral
/// JSON value-subtype (json1.html §3.4) — the subtype-aware counterpart of
/// [`render_value_into`], used on the aggregate `step` hot path (`json_group_array` /
/// `json_group_object`). When `subtype` is [`JSON_SUBTYPE`] and `v` is TEXT — i.e. the
/// value came directly from another JSON function — the text is PARSED and its actual
/// JSON structure is embedded (so `json_group_array(json('[1]'))` is `[[1]]`, not
/// `["[1]"]`). Every other case is the subtype-less fallback [`render_value_into`], which
/// quotes a TEXT value even when it looks like JSON — so the common no-subtype row keeps
/// the no-per-row-node fast path and only a genuinely-subtyped value pays for the parse.
/// A malformed subtyped value (impossible from a real JSON function) fails closed via
/// `parse_json_arg`, exactly like [`value_to_json_with_subtype`].
pub(crate) fn render_value_with_subtype_into(v: &Value, subtype: u8, out: &mut String) -> Result<()> {
    if subtype == JSON_SUBTYPE && matches!(v, Value::Text(_)) {
        super::parse::parse_json_arg(v)?.render_into(out);
        return Ok(());
    }
    render_value_into(v, out)
}

/// Render a REAL the way JSON output needs it. Finite values reuse the engine's
/// canonical float->text ([`real_to_text`]) so a real always reads back as a real.
/// Non-finite values cannot be written as RFC-8259 JSON, so — matching SQLite —
/// infinities render as the out-of-range literal `9.0e+999` and NaN as `null`
/// (a stored REAL is never NaN, so that arm is defensive only). Shared with `parse`,
/// which uses it to canonicalize JSON5 real spellings (`.5`, `Infinity`).
pub(crate) fn render_real(r: f64) -> String {
    if r.is_nan() {
        return "null".to_string();
    }
    if r.is_infinite() {
        return if r < 0.0 { "-9.0e+999".to_string() } else { "9.0e+999".to_string() };
    }
    real_to_text(r)
}

/// Append `level` copies of `indent` to `out` (pretty-print indentation).
fn push_indent(indent: &str, level: usize, out: &mut String) {
    for _ in 0..level {
        out.push_str(indent);
    }
}

/// Append `s` as a canonical, double-quoted JSON string literal, escaping exactly
/// what RFC-8259 requires: the quote and backslash, and the C0 control characters
/// (`< 0x20`) — the five with short escapes (`\b \f \n \r \t`) get those, the rest
/// get `\u00XX`. Everything else, including all non-ASCII UTF-8, is emitted
/// verbatim (SQLite does not `\u`-escape non-ASCII).
pub(crate) fn escape_string_into(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                // Remaining C0 controls have no short escape: use \u00XX.
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_names_cover_every_variant() {
        assert_eq!(Json::Null.type_name(), "null");
        assert_eq!(Json::Bool(true).type_name(), "true");
        assert_eq!(Json::Bool(false).type_name(), "false");
        assert_eq!(Json::Integer(1).type_name(), "integer");
        assert_eq!(Json::Real(1.5).type_name(), "real");
        assert_eq!(Json::Str("x".into()).type_name(), "text");
        assert_eq!(Json::Array(vec![]).type_name(), "array");
        assert_eq!(Json::Object(vec![]).type_name(), "object");
    }

    #[test]
    fn canonical_render_is_minified() {
        let obj = Json::Object(vec![
            ("this".into(), Json::Str("is".into())),
            ("a".into(), Json::Array(vec![Json::Str("test".into())])),
        ]);
        assert_eq!(obj.to_text(), r#"{"this":"is","a":["test"]}"#);
    }

    #[test]
    fn render_numbers_and_literals() {
        assert_eq!(Json::Null.to_text(), "null");
        assert_eq!(Json::Bool(true).to_text(), "true");
        assert_eq!(Json::Bool(false).to_text(), "false");
        assert_eq!(Json::Integer(-17).to_text(), "-17");
        assert_eq!(Json::Real(2.5).to_text(), "2.5");
        // An integral real keeps its decimal point so it stays a real.
        assert_eq!(Json::Real(3.0).to_text(), "3.0");
    }

    #[test]
    fn string_escaping_matches_canonical_json() {
        assert_eq!(Json::Str("plain".into()).to_text(), r#""plain""#);
        // Quote and backslash are escaped; the `{"six":7.7}` example from the doc.
        assert_eq!(Json::Str(r#"{"six":7.7}"#.into()).to_text(), r#""{\"six\":7.7}""#);
        // Control characters: short escapes and \u00XX.
        assert_eq!(Json::Str("a\tb\nc".into()).to_text(), r#""a\tb\nc""#);
        assert_eq!(Json::Str("\u{01}".into()).to_text(), r#""\u0001""#);
        // Non-ASCII stays as UTF-8, not \u-escaped.
        assert_eq!(Json::Str("é".into()).to_text(), "\"é\"");
    }

    #[test]
    fn infinity_and_nan_render_as_sqlite_does() {
        assert_eq!(Json::Real(f64::INFINITY).to_text(), "9.0e+999");
        assert_eq!(Json::Real(f64::NEG_INFINITY).to_text(), "-9.0e+999");
        assert_eq!(Json::Real(f64::NAN).to_text(), "null");
    }

    #[test]
    fn to_sql_scalar_maps_each_case() {
        assert!(matches!(Json::Null.to_sql_scalar(), Value::Null));
        assert!(matches!(Json::Bool(true).to_sql_scalar(), Value::Integer(1)));
        assert!(matches!(Json::Bool(false).to_sql_scalar(), Value::Integer(0)));
        assert!(matches!(Json::Integer(7).to_sql_scalar(), Value::Integer(7)));
        assert!(matches!(Json::Real(1.5).to_sql_scalar(), Value::Real(r) if r == 1.5));
        // A JSON string dequotes to plain TEXT.
        assert!(matches!(Json::Str("xyz".into()).to_sql_scalar(), Value::Text(s) if s == "xyz"));
        // A container returns its canonical JSON text.
        let arr = Json::Array(vec![Json::Integer(4), Json::Integer(5)]);
        assert!(matches!(arr.to_sql_scalar(), Value::Text(s) if s == "[4,5]"));
    }

    #[test]
    fn text_sourced_number_renders_verbatim_and_types_independently() {
        // A canonical exponent/trailing-zero token renders exactly as written.
        let exp = Json::Number(NumLit {
            text: "1e3".into(),
            is_integer: false,
            value: NumValue::Real(1000.0),
        });
        assert_eq!(exp.to_text(), "1e3");
        assert_eq!(exp.type_name(), "real");
        assert!(matches!(exp.to_sql_scalar(), Value::Real(r) if r == 1000.0));

        // A >i64 integer token keeps 'integer' type and its digits, but extracts as
        // a REAL — type and extracted value are independent.
        let big = Json::Number(NumLit {
            text: "9223372036854775808".into(),
            is_integer: true,
            value: NumValue::Real(9.223372036854776e18),
        });
        assert_eq!(big.to_text(), "9223372036854775808");
        assert_eq!(big.type_name(), "integer");
        assert!(matches!(big.to_sql_scalar(), Value::Real(_)));
    }

    #[test]
    fn value_to_json_encodes_value_arguments() {
        assert!(matches!(value_to_json(&Value::Null), Ok(Json::Null)));
        assert!(matches!(value_to_json(&Value::Integer(1)), Ok(Json::Integer(1))));
        assert!(matches!(value_to_json(&Value::Real(2.5)), Ok(Json::Real(r)) if r == 2.5));
        // A TEXT value becomes a quoted JSON string even if it looks like JSON.
        match value_to_json(&Value::Text("[1,2]".into())) {
            Ok(Json::Str(s)) => assert_eq!(s, "[1,2]"),
            other => panic!("expected Json::Str, got {other:?}"),
        }
        // A BLOB value argument is an error.
        assert!(matches!(value_to_json(&Value::Blob(vec![1])), Err(Error::Sql(_))));
    }

    #[test]
    fn render_value_with_subtype_embeds_only_subtyped_text() {
        // A JSON-subtyped TEXT value is embedded as its structure...
        let mut out = String::new();
        render_value_with_subtype_into(&Value::Text("[1,2]".into()), JSON_SUBTYPE, &mut out).unwrap();
        assert_eq!(out, "[1,2]");

        // ...but the SAME text with NO subtype is quoted as a JSON string (the guard that
        // stops a "parse anything JSON-looking" shortcut).
        let mut out = String::new();
        render_value_with_subtype_into(&Value::Text("[1,2]".into()), 0, &mut out).unwrap();
        assert_eq!(out, r#""[1,2]""#);

        // A non-TEXT value ignores the subtype entirely (it can only ride TEXT), matching
        // the plain render.
        let mut out = String::new();
        render_value_with_subtype_into(&Value::Integer(7), JSON_SUBTYPE, &mut out).unwrap();
        assert_eq!(out, "7");

        // A BLOB is still an error even when (nonsensically) tagged with the subtype.
        let mut out = String::new();
        assert!(render_value_with_subtype_into(&Value::Blob(vec![1]), JSON_SUBTYPE, &mut out).is_err());
    }

    #[test]
    fn pretty_print_matches_doc_layout() {
        // The json1.html §4.8 example structure, pretty-printed with 4 spaces.
        let v = Json::Object(vec![
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
        let expected = "{\n    \"a\": 2,\n    \"c\": [\n        4,\n        5,\n        {\n            \"f\": 7\n        }\n    ]\n}";
        assert_eq!(v.to_pretty("    "), expected);
    }

    #[test]
    fn pretty_print_empty_containers_stay_inline() {
        assert_eq!(Json::Object(vec![]).to_pretty("    "), "{}");
        assert_eq!(Json::Array(vec![]).to_pretty("    "), "[]");
        // A custom indent string is honored.
        let v = Json::Object(vec![("a".into(), Json::Integer(1))]);
        assert_eq!(v.to_pretty("  "), "{\n  \"a\": 1\n}");
    }
}
