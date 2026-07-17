//! The mutating JSON scalar functions (`spec/sqlite-doc/json1.html` §4.11, §4.15,
//! §4.18): `json_insert`, `json_replace`, `json_set`, `json_remove`, and
//! `json_patch`.
//!
//! `json_insert`/`json_replace`/`json_set` share one driver over the input JSON and
//! a list of `(path, value)` pairs, differing only by the [`SetMode`] passed to the
//! per-step edit; edits apply left to right so a later path sees earlier changes
//! (json1.html §4.11). `json_remove` applies a list of paths, and removing the root
//! path `$` yields SQL NULL (json1.html §4.18). `json_patch` runs the RFC-7396
//! MergePatch algorithm (json1.html §4.15).
//!
//! All raise `"malformed JSON"` for a bad JSON argument and a path error for a bad
//! path; a NULL JSON argument yields NULL, and a BLOB *value* argument is an error.

use std::sync::Arc;

use minisqlite_expr::{FnContext, ScalarFunction};
use minisqlite_types::{Error, Result, Value};

use super::parse::parse_json_arg;
use super::path::{apply_edit, parse_path, remove_at, value_text, SetMode};
use super::value::{value_to_json_with_subtype, Json, JSON_SUBTYPE};
use crate::registry::{Arity, FunctionRegistry};

/// Register the mutating JSON scalar functions.
pub(crate) fn register(reg: &mut FunctionRegistry) {
    reg.add_scalar("json_insert", Arity::AtLeast(3), Arc::new(JsonInsert));
    reg.add_scalar("json_replace", Arity::AtLeast(3), Arc::new(JsonReplace));
    reg.add_scalar("json_set", Arity::AtLeast(3), Arc::new(JsonSet));
    reg.add_scalar("json_remove", Arity::AtLeast(1), Arc::new(JsonRemove));
    reg.add_scalar("json_patch", Arity::Exact(2), Arc::new(JsonPatch));
}

/// Shared driver for `json_insert`/`json_replace`/`json_set`: parse the JSON
/// argument, then apply each `(path, value)` pair under `mode`. These take an odd
/// number of arguments (the JSON plus path/value pairs); an even count is an error.
/// A NULL JSON argument yields NULL; a NULL path skips that pair.
///
/// A `value` operand produced by another JSON function is EMBEDDED as its JSON
/// structure rather than quoted as a string (json1.html §4.11 note on the subtype):
/// `json_set('{"a":1}','$.b',json('[2]'))` sets `$.b` to the array `[2]`, not the
/// string `"[2]"`. Each value's subtype is read from `ctx` by its absolute argument
/// index; the result is itself JSON, so it carries the subtype outward.
fn run_edit(args: &[Value], mode: SetMode, name: &str, ctx: &mut dyn FnContext) -> Result<Value> {
    if args.len() % 2 != 1 {
        return Err(Error::sql(format!("{name}() requires an odd number of arguments")));
    }
    if args[0].is_null() {
        return Ok(Value::Null);
    }
    let mut root = parse_json_arg(&args[0])?;
    let mut i = 1;
    while i + 1 < args.len() {
        let path_v = &args[i];
        let value_v = &args[i + 1];
        // Capture the value operand's subtype before advancing `i`.
        let value_subtype = ctx.arg_subtype(i + 1);
        i += 2;
        if path_v.is_null() {
            continue; // a NULL path is a no-op for this pair
        }
        let steps = parse_path(&value_text(path_v))?;
        let value = value_to_json_with_subtype(value_v, value_subtype)?; // BLOB value -> error
        apply_edit(&mut root, &steps, value, mode);
    }
    let text = root.to_text();
    ctx.set_result_subtype(JSON_SUBTYPE);
    Ok(Value::Text(text))
}

/// `json_insert(X, path, value, …)` — add new elements without overwriting existing
/// ones (json1.html §4.11).
#[derive(Debug)]
struct JsonInsert;
impl ScalarFunction for JsonInsert {
    fn call(&self, args: &[Value], ctx: &mut dyn FnContext) -> Result<Value> {
        run_edit(args, SetMode::Insert, "json_insert", ctx)
    }
}

/// `json_replace(X, path, value, …)` — overwrite existing elements without creating
/// new ones (json1.html §4.11).
#[derive(Debug)]
struct JsonReplace;
impl ScalarFunction for JsonReplace {
    fn call(&self, args: &[Value], ctx: &mut dyn FnContext) -> Result<Value> {
        run_edit(args, SetMode::Replace, "json_replace", ctx)
    }
}

/// `json_set(X, path, value, …)` — overwrite existing and create missing elements
/// (json1.html §4.11).
#[derive(Debug)]
struct JsonSet;
impl ScalarFunction for JsonSet {
    fn call(&self, args: &[Value], ctx: &mut dyn FnContext) -> Result<Value> {
        run_edit(args, SetMode::Set, "json_set", ctx)
    }
}

/// `json_remove(X, path, …)` — return `X` with the elements at each path removed
/// (json1.html §4.18). With no paths, `X` is just reformatted (minified). Removing
/// the root path `$` yields SQL NULL. A NULL path is a no-op.
#[derive(Debug)]
struct JsonRemove;
impl ScalarFunction for JsonRemove {
    fn call(&self, args: &[Value], ctx: &mut dyn FnContext) -> Result<Value> {
        debug_assert!(!args.is_empty(), "json_remove/AtLeast(1) arity precondition");
        if args[0].is_null() {
            return Ok(Value::Null);
        }
        // `root` becomes None once a `$` removal drops the whole document; later
        // paths then have nothing to act on.
        let mut root = Some(parse_json_arg(&args[0])?);
        for p in &args[1..] {
            if p.is_null() {
                continue;
            }
            let steps = parse_path(&value_text(p))?;
            if let Some(j) = root.as_mut() {
                if steps.is_empty() {
                    // `$` removes the whole document.
                    root = None;
                } else {
                    remove_at(j, &steps);
                }
            }
        }
        Ok(match root {
            Some(j) => {
                let text = j.to_text();
                // The surviving document is JSON: tag it so a wrapping call embeds it.
                ctx.set_result_subtype(JSON_SUBTYPE);
                Value::Text(text)
            }
            None => Value::Null,
        })
    }
}

/// `json_patch(T, P)` — apply the RFC-7396 MergePatch `P` to target `T` (json1.html
/// §4.15). A NULL argument yields NULL.
#[derive(Debug)]
struct JsonPatch;
impl ScalarFunction for JsonPatch {
    fn call(&self, args: &[Value], ctx: &mut dyn FnContext) -> Result<Value> {
        debug_assert!(args.len() == 2, "json_patch/2 arity precondition");
        if args[0].is_null() || args[1].is_null() {
            return Ok(Value::Null);
        }
        let target = parse_json_arg(&args[0])?;
        let patch = parse_json_arg(&args[1])?;
        let text = merge_patch(target, patch).to_text();
        ctx.set_result_subtype(JSON_SUBTYPE);
        Ok(Value::Text(text))
    }
}

/// The RFC-7396 MergePatch algorithm. When the patch is an object it merges into the
/// target (coercing a non-object target to an empty object first): a `null` member
/// deletes that key, any other member recursively patches it (creating the key if
/// absent). A non-object patch replaces the target wholesale. Existing keys keep
/// their position; new keys append in patch order (matching SQLite, json1.html
/// §4.15). Recursion is bounded by the patch's nesting depth (parser-capped).
fn merge_patch(target: Json, patch: Json) -> Json {
    let Json::Object(patch_members) = patch else {
        return patch; // array/scalar patch replaces the target entirely
    };
    let mut base = match target {
        Json::Object(members) => members,
        _ => Vec::new(), // non-object target is treated as an empty object
    };
    for (key, pval) in patch_members {
        if matches!(pval, Json::Null) {
            base.retain(|(bk, _)| *bk != key);
        } else if let Some(pos) = base.iter().position(|(bk, _)| *bk == key) {
            let old = std::mem::replace(&mut base[pos].1, Json::Null);
            base[pos].1 = merge_patch(old, pval);
        } else {
            base.push((key, merge_patch(Json::Null, pval)));
        }
    }
    Json::Object(base)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::testutil::NullCtx;

    fn call(f: &dyn ScalarFunction, args: &[Value]) -> Value {
        f.call(args, &mut NullCtx).expect("json edit should succeed")
    }
    fn call_err(f: &dyn ScalarFunction, args: &[Value]) -> Error {
        f.call(args, &mut NullCtx).expect_err("json edit should error")
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
    fn insert_adds_only() {
        // Existing key untouched; missing key created.
        assert_eq!(text(&call(&JsonInsert, &[t(r#"{"a":2,"c":4}"#), t("$.a"), Value::Integer(99)])), r#"{"a":2,"c":4}"#);
        assert_eq!(text(&call(&JsonInsert, &[t(r#"{"a":2,"c":4}"#), t("$.e"), Value::Integer(99)])), r#"{"a":2,"c":4,"e":99}"#);
        // Append to an array with `$[#]`.
        assert_eq!(text(&call(&JsonInsert, &[t("[1,2,3,4]"), t("$[#]"), Value::Integer(99)])), "[1,2,3,4,99]");
        assert_eq!(text(&call(&JsonInsert, &[t("[1,[2,3],4]"), t("$[1][#]"), Value::Integer(99)])), "[1,[2,3,99],4]");
    }

    #[test]
    fn replace_overwrites_only() {
        assert_eq!(text(&call(&JsonReplace, &[t(r#"{"a":2,"c":4}"#), t("$.a"), Value::Integer(99)])), r#"{"a":99,"c":4}"#);
        assert_eq!(text(&call(&JsonReplace, &[t(r#"{"a":2,"c":4}"#), t("$.e"), Value::Integer(99)])), r#"{"a":2,"c":4}"#);
    }

    #[test]
    fn set_adds_and_overwrites() {
        assert_eq!(text(&call(&JsonSet, &[t(r#"{"a":2,"c":4}"#), t("$.a"), Value::Integer(99)])), r#"{"a":99,"c":4}"#);
        assert_eq!(text(&call(&JsonSet, &[t(r#"{"a":2,"c":4}"#), t("$.e"), Value::Integer(99)])), r#"{"a":2,"c":4,"e":99}"#);
        // A TEXT value is stored as a quoted JSON string, even when it looks like JSON.
        assert_eq!(text(&call(&JsonSet, &[t(r#"{"a":2,"c":4}"#), t("$.c"), t("[97,96]")])), r#"{"a":2,"c":"[97,96]"}"#);
        // The `$[#]` append example from json1.html §3.3.
        assert_eq!(text(&call(&JsonSet, &[t("[0,1,2]"), t("$[#]"), t("new")])), r#"[0,1,2,"new"]"#);
    }

    #[test]
    fn set_applies_pairs_left_to_right() {
        // Two pairs: the second sees the first's change.
        let out = call(&JsonSet, &[t("{}"), t("$.a"), Value::Integer(1), t("$.b"), Value::Integer(2)]);
        assert_eq!(text(&out), r#"{"a":1,"b":2}"#);
    }

    #[test]
    fn edit_null_and_arity_rules() {
        // NULL JSON argument -> NULL.
        assert!(matches!(call(&JsonSet, &[Value::Null, t("$.a"), Value::Integer(1)]), Value::Null));
        // Even argument count -> error.
        assert!(matches!(call_err(&JsonSet, &[t("{}"), t("$.a"), Value::Integer(1), t("$.b")]), Error::Sql(_)));
        // A BLOB value argument -> error.
        assert!(matches!(call_err(&JsonSet, &[t("{}"), t("$.a"), Value::Blob(vec![1])]), Error::Sql(_)));
        // Malformed JSON -> error.
        assert!(matches!(call_err(&JsonSet, &[t("{"), t("$.a"), Value::Integer(1)]), Error::Sql(_)));
    }

    #[test]
    fn remove_paths_and_root() {
        assert_eq!(text(&call(&JsonRemove, &[t("[0,1,2,3,4]"), t("$[2]")])), "[0,1,3,4]");
        assert_eq!(text(&call(&JsonRemove, &[t("[0,1,2,3,4]"), t("$[2]"), t("$[0]")])), "[1,3,4]");
        assert_eq!(text(&call(&JsonRemove, &[t("[0,1,2,3,4]"), t("$[0]"), t("$[2]")])), "[1,2,4]");
        assert_eq!(text(&call(&JsonRemove, &[t("[0,1,2,3,4]"), t("$[#-1]"), t("$[0]")])), "[1,2,3]");
        assert_eq!(text(&call(&JsonRemove, &[t(r#"{"x":25,"y":42}"#), t("$.y")])), r#"{"x":25}"#);
        // Path not found -> unchanged; no paths -> reformat.
        assert_eq!(text(&call(&JsonRemove, &[t(r#"{"x":25,"y":42}"#), t("$.z")])), r#"{"x":25,"y":42}"#);
        assert_eq!(text(&call(&JsonRemove, &[t(r#"{"x":25,"y":42}"#)])), r#"{"x":25,"y":42}"#);
        // Removing the root yields NULL.
        assert!(matches!(call(&JsonRemove, &[t(r#"{"x":25,"y":42}"#), t("$")]), Value::Null));
        // NULL JSON -> NULL.
        assert!(matches!(call(&JsonRemove, &[Value::Null, t("$.a")]), Value::Null));
    }

    #[test]
    fn patch_merges_per_rfc7396() {
        // The five worked examples from json1.html §4.15.
        assert_eq!(text(&call(&JsonPatch, &[t(r#"{"a":1,"b":2}"#), t(r#"{"c":3,"d":4}"#)])), r#"{"a":1,"b":2,"c":3,"d":4}"#);
        assert_eq!(text(&call(&JsonPatch, &[t(r#"{"a":[1,2],"b":2}"#), t(r#"{"a":9}"#)])), r#"{"a":9,"b":2}"#);
        assert_eq!(text(&call(&JsonPatch, &[t(r#"{"a":[1,2],"b":2}"#), t(r#"{"a":null}"#)])), r#"{"b":2}"#);
        assert_eq!(text(&call(&JsonPatch, &[t(r#"{"a":1,"b":2}"#), t(r#"{"a":9,"b":null,"c":8}"#)])), r#"{"a":9,"c":8}"#);
        assert_eq!(text(&call(&JsonPatch, &[t(r#"{"a":{"x":1,"y":2},"b":3}"#), t(r#"{"a":{"y":9},"c":8}"#)])), r#"{"a":{"x":1,"y":9},"b":3,"c":8}"#);
        // A non-object patch replaces the target entirely.
        assert_eq!(text(&call(&JsonPatch, &[t(r#"{"a":1}"#), t("42")])), "42");
        // NULL argument -> NULL.
        assert!(matches!(call(&JsonPatch, &[Value::Null, t("{}")]), Value::Null));
        assert!(matches!(call(&JsonPatch, &[t("{}"), Value::Null]), Value::Null));
    }

    #[test]
    fn registered_names_resolve() {
        let reg = FunctionRegistry::builtins();
        for (name, argc) in [
            ("json_insert", 3usize),
            ("json_replace", 3),
            ("json_set", 3),
            ("json_remove", 1),
            ("json_remove", 2),
            ("json_patch", 2),
        ] {
            assert!(reg.resolve_scalar(name, argc).is_ok(), "{name}/{argc} should resolve");
        }
    }
}
