//! The read/construct JSON scalar functions (`spec/sqlite-doc/json1.html` §4):
//! `json`, `json_array`, `json_object`, `json_extract`, `json_type`, `json_valid`,
//! `json_quote`, `json_array_length`, `json_pretty`, and `json_error_position`. The
//! mutating scalars (`json_insert`/`replace`/`set`/`remove`/`patch`) live in the
//! sibling `edit` module.
//!
//! NULL handling follows the doc's rule (json1.html §3.1): a NULL json argument
//! yields NULL for every function here except the three that "treat NULL specially"
//! — `json_valid(NULL)` -> NULL, `json_quote(NULL)` -> the text `'null'`, and
//! `json_error_position(NULL)` -> 0. A function that requires valid JSON raises
//! `"malformed JSON"` on bad input; `json_valid`, `json_quote`, and
//! `json_error_position` never do.

use std::borrow::Cow;
use std::sync::Arc;

use minisqlite_expr::{to_integer, FnContext, ScalarFunction};
use minisqlite_types::{Error, Result, Value};

use super::parse::{char_pos_of_byte, parse_json_arg, parse_text};
use super::path::{navigate, parse_path, value_text};
use super::value::{value_to_json_with_subtype, Json, JSON_SUBTYPE};
use crate::registry::{Arity, FunctionRegistry};

/// Register the read/construct JSON scalar functions.
pub(crate) fn register(reg: &mut FunctionRegistry) {
    reg.add_scalar("json", Arity::Exact(1), Arc::new(JsonFn));
    reg.add_scalar("json_array", Arity::Any, Arc::new(JsonArray));
    reg.add_scalar("json_object", Arity::Any, Arc::new(JsonObject));
    reg.add_scalar("json_extract", Arity::AtLeast(2), Arc::new(JsonExtract));
    reg.add_scalar("json_type", Arity::Range(1, 2), Arc::new(JsonType));
    reg.add_scalar("json_valid", Arity::Range(1, 2), Arc::new(JsonValid));
    reg.add_scalar("json_quote", Arity::Exact(1), Arc::new(JsonQuote));
    reg.add_scalar("json_array_length", Arity::Range(1, 2), Arc::new(JsonArrayLength));
    reg.add_scalar("json_pretty", Arity::Range(1, 2), Arc::new(JsonPretty));
    reg.add_scalar("json_error_position", Arity::Exact(1), Arc::new(JsonErrorPosition));
}

/// `json(X)` — validate `X` and return canonical minified JSON (json1.html §4.1).
/// JSON5 input is converted to canonical form; invalid input is an error.
#[derive(Debug)]
struct JsonFn;
impl ScalarFunction for JsonFn {
    fn call(&self, args: &[Value], ctx: &mut dyn FnContext) -> Result<Value> {
        debug_assert!(args.len() == 1, "json/1 arity precondition");
        if args[0].is_null() {
            return Ok(Value::Null);
        }
        let text = parse_json_arg(&args[0])?.to_text();
        // The result is canonical JSON: tag it so a wrapping JSON function embeds it.
        ctx.set_result_subtype(JSON_SUBTYPE);
        Ok(Value::Text(text))
    }
}

/// `json_array(v1, …)` — a JSON array of the value arguments (json1.html §4.3). A
/// TEXT value becomes a quoted JSON string; a BLOB argument is an error.
#[derive(Debug)]
struct JsonArray;
impl ScalarFunction for JsonArray {
    fn call(&self, args: &[Value], ctx: &mut dyn FnContext) -> Result<Value> {
        let mut items = Vec::with_capacity(args.len());
        for (i, a) in args.iter().enumerate() {
            // A value from another JSON function embeds; ordinary TEXT is quoted.
            items.push(value_to_json_with_subtype(a, ctx.arg_subtype(i))?);
        }
        let text = Json::Array(items).to_text();
        ctx.set_result_subtype(JSON_SUBTYPE);
        Ok(Value::Text(text))
    }
}

/// `json_object(label1, value1, …)` — a JSON object from label/value pairs
/// (json1.html §4.13). An odd argument count, or a label that is not TEXT, is an
/// error; a BLOB value is an error. Duplicate labels are preserved.
#[derive(Debug)]
struct JsonObject;
impl ScalarFunction for JsonObject {
    fn call(&self, args: &[Value], ctx: &mut dyn FnContext) -> Result<Value> {
        if args.len() % 2 != 0 {
            return Err(Error::sql("json_object() requires an even number of arguments"));
        }
        let mut members = Vec::with_capacity(args.len() / 2);
        for (pair_idx, pair) in args.chunks_exact(2).enumerate() {
            let label = match &pair[0] {
                Value::Text(s) => s.clone(),
                _ => return Err(Error::sql("json_object() labels must be TEXT")),
            };
            // The value is at absolute argument index `pair_idx*2 + 1`; a value from
            // another JSON function embeds, ordinary TEXT is quoted (json1.html §4.13).
            let value = value_to_json_with_subtype(&pair[1], ctx.arg_subtype(pair_idx * 2 + 1))?;
            members.push((label, value));
        }
        let text = Json::Object(members).to_text();
        ctx.set_result_subtype(JSON_SUBTYPE);
        Ok(Value::Text(text))
    }
}

/// `json_extract(X, P1, …)` — extract one or more values (json1.html §4.8). A single
/// path returns the SQL value of the selected node (a JSON scalar as its SQL type, a
/// container as JSON text), or NULL if it selects nothing. Multiple paths return a
/// JSON array of the selected nodes (missing -> `null`).
#[derive(Debug)]
struct JsonExtract;
impl ScalarFunction for JsonExtract {
    fn call(&self, args: &[Value], ctx: &mut dyn FnContext) -> Result<Value> {
        debug_assert!(args.len() >= 2, "json_extract/AtLeast(2) arity precondition");
        if args[0].is_null() {
            return Ok(Value::Null);
        }
        let root = parse_json_arg(&args[0])?;
        let paths = &args[1..];
        if paths.len() == 1 {
            if paths[0].is_null() {
                return Ok(Value::Null);
            }
            let steps = parse_path(&value_text(&paths[0]))?;
            Ok(match navigate(&root, &steps) {
                Some(node) => {
                    // A container extracts as JSON text and carries the JSON subtype so
                    // a wrapping JSON function embeds it; a scalar (including a dequoted
                    // JSON string) does not (json1.html §4.8, §3.4).
                    if matches!(node, Json::Array(_) | Json::Object(_)) {
                        ctx.set_result_subtype(JSON_SUBTYPE);
                    }
                    node.to_sql_scalar()
                }
                None => Value::Null,
            })
        } else {
            let mut arr = Vec::with_capacity(paths.len());
            for p in paths {
                if p.is_null() {
                    arr.push(Json::Null);
                    continue;
                }
                let steps = parse_path(&value_text(p))?;
                arr.push(navigate(&root, &steps).cloned().unwrap_or(Json::Null));
            }
            // The multi-path result is itself a JSON array (a container).
            let text = Json::Array(arr).to_text();
            ctx.set_result_subtype(JSON_SUBTYPE);
            Ok(Value::Text(text))
        }
    }
}

/// `json_type(X)` / `json_type(X, P)` — the type name of the whole value or the
/// element at path `P` (json1.html §4.20). A path that selects nothing yields NULL.
#[derive(Debug)]
struct JsonType;
impl ScalarFunction for JsonType {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        if args[0].is_null() {
            return Ok(Value::Null);
        }
        let root = parse_json_arg(&args[0])?;
        let node = match args.get(1) {
            None => &root,
            Some(p) => {
                if p.is_null() {
                    return Ok(Value::Null);
                }
                let steps = parse_path(&value_text(p))?;
                match navigate(&root, &steps) {
                    Some(n) => n,
                    None => return Ok(Value::Null),
                }
            }
        };
        Ok(Value::Text(node.type_name().to_string()))
    }
}

/// `json_valid(X)` / `json_valid(X, flags)` — whether `X` is well-formed per the
/// flag bitmask (json1.html §4.21). X or flags NULL -> NULL; flags default to 1
/// (canonical only); a flags value outside 1..=15 is an error. Bit 0x01 =
/// canonical RFC-8259, 0x02 = JSON5. The JSONB bits (0x04/0x08) are not supported
/// (JSONB is deferred), so a value is judged by its text interpretation.
#[derive(Debug)]
struct JsonValid;
impl ScalarFunction for JsonValid {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        if args[0].is_null() {
            return Ok(Value::Null);
        }
        let flags = match args.get(1) {
            None => 1,
            Some(f) => {
                if f.is_null() {
                    return Ok(Value::Null);
                }
                to_integer(f)
            }
        };
        if !(1..=15).contains(&flags) {
            return Err(Error::sql("json_valid() flags must be between 1 and 15"));
        }
        Ok(Value::Integer(if is_valid(&args[0], flags) { 1 } else { 0 }))
    }
}

/// Whether value `x` is well-formed JSON under `flags` (bit 0x01 canonical, 0x02
/// JSON5). A numeric SQL value is a canonical JSON number; a BLOB is read as text
/// (the legacy BLOB-input behavior). The JSONB bits contribute nothing here.
fn is_valid(x: &Value, flags: i64) -> bool {
    let want_canonical = flags & 0x01 != 0;
    let want_json5 = flags & 0x02 != 0;
    let text: Cow<str> = match x {
        Value::Text(s) => Cow::Borrowed(s.as_str()),
        Value::Blob(b) => String::from_utf8_lossy(b),
        // A numeric value is a valid canonical JSON number.
        Value::Integer(_) | Value::Real(_) => return want_canonical || want_json5,
        Value::Null => return false,
    };
    match parse_text(&text) {
        // canonical == parsed with no JSON5 extension; any parse is valid JSON5.
        Ok(p) => (want_canonical && !p.json5) || want_json5,
        Err(_) => false,
    }
}

/// `json_quote(X)` — the JSON representation of a SQL scalar (json1.html §4.22).
/// A number becomes a JSON number, a string a quoted JSON string, and NULL the text
/// `'null'`. When `X` already carries the JSON subtype (it came directly from another
/// JSON function), this is a no-op returning `X` verbatim (json1.html §4.22) — the
/// no-op branch below, via the ephemeral value-subtype channel documented in the crate
/// module docs.
#[derive(Debug)]
struct JsonQuote;
impl ScalarFunction for JsonQuote {
    fn call(&self, args: &[Value], ctx: &mut dyn FnContext) -> Result<Value> {
        debug_assert!(args.len() == 1, "json_quote/1 arity precondition");
        // No-op when X is already JSON from another JSON function: return it verbatim,
        // keeping the subtype so a further wrap still embeds it (json1.html §4.22).
        if ctx.arg_subtype(0) == JSON_SUBTYPE {
            ctx.set_result_subtype(JSON_SUBTYPE);
            return Ok(args[0].clone());
        }
        let node = match &args[0] {
            Value::Null => Json::Null,
            Value::Integer(i) => Json::Integer(*i),
            Value::Real(r) => Json::Real(*r),
            Value::Text(s) => Json::Str(s.clone()),
            Value::Blob(b) => Json::Str(String::from_utf8_lossy(b).into_owned()),
        };
        let text = node.to_text();
        // json_quote's result is a JSON representation, so it carries the subtype too.
        ctx.set_result_subtype(JSON_SUBTYPE);
        Ok(Value::Text(text))
    }
}

/// `json_array_length(X)` / `json_array_length(X, P)` — the element count of the
/// array at the top (or at path `P`), 0 if that element is not an array, NULL if
/// `P` selects nothing (json1.html §4.6).
#[derive(Debug)]
struct JsonArrayLength;
impl ScalarFunction for JsonArrayLength {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        if args[0].is_null() {
            return Ok(Value::Null);
        }
        let root = parse_json_arg(&args[0])?;
        let node = match args.get(1) {
            None => &root,
            Some(p) => {
                if p.is_null() {
                    return Ok(Value::Null);
                }
                let steps = parse_path(&value_text(p))?;
                match navigate(&root, &steps) {
                    Some(n) => n,
                    None => return Ok(Value::Null),
                }
            }
        };
        let len = match node {
            Json::Array(items) => items.len() as i64,
            _ => 0,
        };
        Ok(Value::Integer(len))
    }
}

/// `json_pretty(X)` / `json_pretty(X, indent)` — pretty-printed JSON (json1.html
/// §4.17). The indent defaults to four spaces when omitted or NULL.
#[derive(Debug)]
struct JsonPretty;
impl ScalarFunction for JsonPretty {
    fn call(&self, args: &[Value], ctx: &mut dyn FnContext) -> Result<Value> {
        if args[0].is_null() {
            return Ok(Value::Null);
        }
        let root = parse_json_arg(&args[0])?;
        let indent: Cow<str> = match args.get(1) {
            None | Some(Value::Null) => Cow::Borrowed("    "),
            Some(v) => value_text(v),
        };
        let text = root.to_pretty(&indent);
        // A pretty-printed JSON value still carries the JSON subtype (embedding it
        // re-parses and re-renders it minified).
        ctx.set_result_subtype(JSON_SUBTYPE);
        Ok(Value::Text(text))
    }
}

/// `json_error_position(X)` — 0 if `X` is well-formed JSON/JSON5, else the 1-based
/// character position of the first syntax error (json1.html §4.7). Never errors.
#[derive(Debug)]
struct JsonErrorPosition;
impl ScalarFunction for JsonErrorPosition {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        debug_assert!(args.len() == 1, "json_error_position/1 arity precondition");
        let pos = match &args[0] {
            // A NULL or numeric argument has no syntax error.
            Value::Null | Value::Integer(_) | Value::Real(_) => 0,
            Value::Text(s) => error_position(s),
            Value::Blob(b) => error_position(&String::from_utf8_lossy(b)),
        };
        Ok(Value::Integer(pos))
    }
}

/// The 1-based character position of the first parse error in `s`, or 0 if `s` is
/// well-formed JSON/JSON5.
fn error_position(s: &str) -> i64 {
    match parse_text(s) {
        Ok(_) => 0,
        Err(e) => char_pos_of_byte(s, e.byte_pos) as i64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::testutil::NullCtx;

    /// Call a scalar with the given args against a no-op context, expecting `Ok`.
    fn call(f: &dyn ScalarFunction, args: &[Value]) -> Value {
        f.call(args, &mut NullCtx).expect("json scalar should succeed")
    }

    /// Call a scalar, expecting an error (invalid JSON / path / arguments).
    fn call_err(f: &dyn ScalarFunction, args: &[Value]) -> Error {
        f.call(args, &mut NullCtx).expect_err("json scalar should error")
    }

    fn t(s: &str) -> Value {
        Value::Text(s.into())
    }
    fn text(v: &Value) -> &str {
        match v {
            Value::Text(s) => s,
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn json_minifies_and_validates() {
        assert_eq!(text(&call(&JsonFn, &[t(r#" { "this" : "is", "a": [ "test" ] } "#)])), r#"{"this":"is","a":["test"]}"#);
        // Whitespace round-trip from the doc.
        assert_eq!(text(&call(&JsonFn, &[t(r#"{ "a" : 1 }"#)])), r#"{"a":1}"#);
        // JSON5 is converted to canonical.
        assert_eq!(text(&call(&JsonFn, &[t("{a:1}")])), r#"{"a":1}"#);
        // A canonical number token survives the json() surface verbatim (the headline
        // witness, pinned here through the real ScalarFunction, not just parse_text).
        assert_eq!(text(&call(&JsonFn, &[t("1e3")])), "1e3");
        assert_eq!(text(&call(&JsonFn, &[t("[1.50]")])), "[1.50]");
        // A number argument is valid JSON.
        assert_eq!(text(&call(&JsonFn, &[Value::Integer(5)])), "5");
        // NULL -> NULL; malformed -> error.
        assert!(matches!(call(&JsonFn, &[Value::Null]), Value::Null));
        assert!(matches!(call_err(&JsonFn, &[t("{")]), Error::Sql(m) if m == "malformed JSON"));
    }

    #[test]
    fn json_preserves_duplicate_keys() {
        // json1.html §4.1: the current implementation preserves duplicate labels.
        assert_eq!(text(&call(&JsonFn, &[t(r#"{"a":1,"a":2}"#)])), r#"{"a":1,"a":2}"#);
    }

    #[test]
    fn json_array_builds_arrays() {
        assert_eq!(text(&call(&JsonArray, &[Value::Integer(1), t("2"), Value::Null])), r#"[1,"2",null]"#);
        // The doc's canonical example (json1.html §4.3).
        assert_eq!(text(&call(&JsonArray, &[Value::Integer(1), Value::Integer(2), t("3"), Value::Integer(4)])), r#"[1,2,"3",4]"#);
        // A TEXT value that looks like JSON is quoted, not inlined — including the
        // doc example that exercises `"` escaping inside a value string.
        assert_eq!(text(&call(&JsonArray, &[t("[1,2]")])), r#"["[1,2]"]"#);
        assert_eq!(
            text(&call(&JsonArray, &[Value::Integer(1), Value::Null, t("3"), t("[4,5]"), t(r#"{"six":7.7}"#)])),
            r#"[1,null,"3","[4,5]","{\"six\":7.7}"]"#
        );
        // Zero arguments -> empty array.
        assert_eq!(text(&call(&JsonArray, &[])), "[]");
        // A BLOB argument is an error.
        assert!(matches!(call_err(&JsonArray, &[Value::Blob(vec![1])]), Error::Sql(_)));
    }

    #[test]
    fn json_object_builds_objects() {
        assert_eq!(text(&call(&JsonObject, &[t("a"), Value::Integer(1), t("b"), t("x")])), r#"{"a":1,"b":"x"}"#);
        // The doc examples (json1.html §4.13): a literal-string value is quoted.
        assert_eq!(text(&call(&JsonObject, &[t("a"), Value::Integer(2), t("c"), Value::Integer(4)])), r#"{"a":2,"c":4}"#);
        assert_eq!(text(&call(&JsonObject, &[t("a"), Value::Integer(2), t("c"), t("{e:5}")])), r#"{"a":2,"c":"{e:5}"}"#);
        assert_eq!(text(&call(&JsonObject, &[])), "{}");
        // Odd argument count -> error.
        assert!(matches!(call_err(&JsonObject, &[t("a")]), Error::Sql(_)));
        // Non-TEXT label -> error.
        assert!(matches!(call_err(&JsonObject, &[Value::Integer(1), Value::Integer(2)]), Error::Sql(_)));
    }

    #[test]
    fn json_extract_single_and_multi() {
        let obj = t(r#"{"a":2,"c":[4,5,{"f":7}]}"#);
        // Single path: SQL scalar / JSON text of a container (json1.html §4.8).
        assert!(matches!(call(&JsonExtract, &[obj.clone(), t("$.c[2].f")]), Value::Integer(7)));
        assert_eq!(text(&call(&JsonExtract, &[obj.clone(), t("$.c")])), "[4,5,{\"f\":7}]");
        assert_eq!(text(&call(&JsonExtract, &[obj.clone(), t("$.c[2]")])), r#"{"f":7}"#);
        assert_eq!(text(&call(&JsonExtract, &[obj.clone(), t("$")])), r#"{"a":2,"c":[4,5,{"f":7}]}"#);
        // Missing single path -> NULL.
        assert!(matches!(call(&JsonExtract, &[obj.clone(), t("$.x")]), Value::Null));
        // From the doc: $.a[1]=20.
        assert!(matches!(call(&JsonExtract, &[t(r#"{"a":[10,20]}"#), t("$.a[1]")]), Value::Integer(20)));
        // Multiple paths -> JSON array (missing -> null).
        assert_eq!(text(&call(&JsonExtract, &[t(r#"{"a":2,"c":[4,5],"f":7}"#), t("$.c"), t("$.a")])), "[[4,5],2]");
        assert_eq!(text(&call(&JsonExtract, &[t(r#"{"a":2,"c":[4,5,{"f":7}]}"#), t("$.x"), t("$.a")])), "[null,2]");
        // A JSON string dequotes for a single path.
        assert_eq!(text(&call(&JsonExtract, &[t(r#"{"a":"xyz"}"#), t("$.a")])), "xyz");
        assert!(matches!(call(&JsonExtract, &[t(r#"{"a":null}"#), t("$.a")]), Value::Null));
        // Last-relative index from the doc.
        assert!(matches!(call(&JsonExtract, &[t(r#"{"a":2,"c":[4,5],"f":7}"#), t("$.c[#-1]")]), Value::Integer(5)));
    }

    #[test]
    fn json_type_reports_element_types() {
        let x = t(r#"{"a":[2,3.5,true,false,null,"x"]}"#);
        assert_eq!(text(&call(&JsonType, &[x.clone()])), "object");
        assert_eq!(text(&call(&JsonType, &[x.clone(), t("$.a")])), "array");
        assert_eq!(text(&call(&JsonType, &[x.clone(), t("$.a[0]")])), "integer");
        assert_eq!(text(&call(&JsonType, &[x.clone(), t("$.a[1]")])), "real");
        assert_eq!(text(&call(&JsonType, &[x.clone(), t("$.a[2]")])), "true");
        assert_eq!(text(&call(&JsonType, &[x.clone(), t("$.a[3]")])), "false");
        assert_eq!(text(&call(&JsonType, &[x.clone(), t("$.a[4]")])), "null");
        assert_eq!(text(&call(&JsonType, &[x.clone(), t("$.a[5]")])), "text");
        // Path selecting nothing -> NULL.
        assert!(matches!(call(&JsonType, &[x, t("$.a[6]")]), Value::Null));
        // Malformed JSON -> error.
        assert!(matches!(call_err(&JsonType, &[t("{")]), Error::Sql(_)));
    }

    #[test]
    fn json_valid_flag_behavior() {
        fn v(res: Value) -> i64 {
            match res {
                Value::Integer(i) => i,
                other => panic!("expected Integer, got {other:?}"),
            }
        }
        assert_eq!(v(call(&JsonValid, &[t(r#"{"x":35}"#)])), 1);
        assert_eq!(v(call(&JsonValid, &[t("{x:35}")])), 0); // JSON5, default flag 1 rejects
        assert_eq!(v(call(&JsonValid, &[t("{x:35}"), Value::Integer(6)])), 1); // flag 6 accepts JSON5
        assert_eq!(v(call(&JsonValid, &[t("{\"x\":35")])), 0);
        // A numeric SQL value is a valid canonical JSON number under the default flag.
        assert_eq!(v(call(&JsonValid, &[Value::Integer(5)])), 1);
        assert_eq!(v(call(&JsonValid, &[Value::Real(2.5)])), 1);
        // NULL X / NULL flags -> NULL.
        assert!(matches!(call(&JsonValid, &[Value::Null]), Value::Null));
        assert!(matches!(call(&JsonValid, &[t("{}"), Value::Null]), Value::Null));
        // Out-of-range flags -> error.
        assert!(matches!(call_err(&JsonValid, &[t("{}"), Value::Integer(0)]), Error::Sql(_)));
        assert!(matches!(call_err(&JsonValid, &[t("{}"), Value::Integer(16)]), Error::Sql(_)));
    }

    #[test]
    fn json_quote_representations() {
        assert_eq!(text(&call(&JsonQuote, &[Value::Real(3.14159)])), "3.14159");
        assert_eq!(text(&call(&JsonQuote, &[t("verdant")])), r#""verdant""#);
        assert_eq!(text(&call(&JsonQuote, &[t("[1]")])), r#""[1]""#);
        assert_eq!(text(&call(&JsonQuote, &[t("[1,")])), r#""[1,""#);
        // A BLOB is read as text (the crate's legacy BLOB-as-text convention) and
        // quoted as a JSON string. PROVISIONAL: real SQLite errors ("JSON cannot hold
        // BLOB values") for a BLOB here; this pins current behavior, revisit if
        // real sqlite's json_quote BLOB handling turns out to differ.
        assert_eq!(text(&call(&JsonQuote, &[Value::Blob(b"ab".to_vec())])), r#""ab""#);
        // NULL is quoted as the JSON text 'null' (the special case).
        assert_eq!(text(&call(&JsonQuote, &[Value::Null])), "null");
    }

    #[test]
    fn json_array_length_rules() {
        assert!(matches!(call(&JsonArrayLength, &[t("[1,2,3,4]")]), Value::Integer(4)));
        assert!(matches!(call(&JsonArrayLength, &[t("[1,2,3,4]"), t("$")]), Value::Integer(4)));
        assert!(matches!(call(&JsonArrayLength, &[t("[1,2,3,4]"), t("$[2]")]), Value::Integer(0)));
        assert!(matches!(call(&JsonArrayLength, &[t(r#"{"one":[1,2,3]}"#)]), Value::Integer(0)));
        assert!(matches!(call(&JsonArrayLength, &[t(r#"{"one":[1,2,3]}"#), t("$.one")]), Value::Integer(3)));
        // Path selecting nothing -> NULL.
        assert!(matches!(call(&JsonArrayLength, &[t(r#"{"one":[1,2,3]}"#), t("$.two")]), Value::Null));
    }

    #[test]
    fn json_pretty_default_and_custom_indent() {
        let out = text(&call(&JsonPretty, &[t(r#"{"a":1}"#)])).to_string();
        assert_eq!(out, "{\n    \"a\": 1\n}");
        let out2 = text(&call(&JsonPretty, &[t(r#"{"a":1}"#), t("\t")])).to_string();
        assert_eq!(out2, "{\n\t\"a\": 1\n}");
        // NULL indent falls back to four spaces.
        let out3 = text(&call(&JsonPretty, &[t(r#"{"a":1}"#), Value::Null])).to_string();
        assert_eq!(out3, "{\n    \"a\": 1\n}");
    }

    #[test]
    fn json_error_position_reports_first_error() {
        fn v(res: Value) -> i64 {
            match res {
                Value::Integer(i) => i,
                other => panic!("expected Integer, got {other:?}"),
            }
        }
        // Well-formed (JSON and JSON5) -> 0.
        assert_eq!(v(call(&JsonErrorPosition, &[t(r#"{"x":35}"#)])), 0);
        assert_eq!(v(call(&JsonErrorPosition, &[t("{x:35}")])), 0); // JSON5 is well-formed here
        // A number / NULL has no error.
        assert_eq!(v(call(&JsonErrorPosition, &[Value::Integer(1)])), 0);
        assert_eq!(v(call(&JsonErrorPosition, &[Value::Null])), 0);
        // Malformed -> the exact 1-based character position of the first error.
        // In `{"x":}` a value is expected at the `}`, which is character 6.
        assert_eq!(v(call(&JsonErrorPosition, &[t(r#"{"x":}"#)])), 6);
    }

    #[test]
    fn registered_names_resolve() {
        let reg = FunctionRegistry::builtins();
        for (name, argc) in [
            ("json", 1usize),
            ("json_array", 3),
            ("json_object", 2),
            ("json_extract", 2),
            ("json_type", 1),
            ("json_type", 2),
            ("json_valid", 1),
            ("json_quote", 1),
            ("json_array_length", 1),
            ("json_pretty", 1),
            ("json_error_position", 1),
        ] {
            assert!(reg.resolve_scalar(name, argc).is_ok(), "{name}/{argc} should resolve");
        }
    }
}
