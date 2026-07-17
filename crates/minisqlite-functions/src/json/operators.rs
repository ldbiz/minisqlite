//! The JSON `->` and `->>` operators (`spec/sqlite-doc/json1.html` §4.10).
//!
//! These are the operator forms of a single-path extraction. They are registered as
//! ordinary scalar functions under the **sentinel names** `"->"` and `"->>"`. Those
//! names are not valid SQL identifiers, so a `SELECT "->"(x, y)` can never reach them
//! and a user function can never collide with them; the only caller is the binder,
//! which resolves the `JsonArrow` / `JsonArrow2` binary operator to these handles and
//! lowers `X -> P` / `X ->> P` to a two-argument call. Routing the operators through
//! the registry keeps them on the same evaluation path as every other scalar.
//!
//! Both operators select the SAME subcomponent of the left JSON operand; they differ
//! only in how the selected node is represented (§4.10):
//!
//! * `->`  returns the JSON text representation — a string comes back QUOTED, and a
//!   number / boolean / JSON-null / array / object as its canonical JSON text. That is
//!   exactly [`Json::to_text`] of the selected node.
//! * `->>` returns the SQL representation — the same thing single-path
//!   [`json_extract`](super::scalar) yields: a string DEQUOTED, a number as INTEGER or
//!   REAL, a JSON boolean as SQL INTEGER 1/0, a JSON null as SQL NULL, and a container
//!   as its JSON text. That is [`Json::to_sql_scalar`] of the selected node.
//!
//! A path that selects nothing, or a SQL NULL left operand, yields SQL NULL. A
//! malformed-JSON left operand is an error (via [`parse_json_arg`], as with the other
//! JSON functions). Chaining (`a -> b -> c`) works because `->` returns canonical JSON
//! text that re-parses, and only containers are ever chained through.

use std::borrow::Cow;
use std::sync::Arc;

use minisqlite_expr::{FnContext, ScalarFunction};
use minisqlite_types::{Result, Value};

use super::parse::parse_json_arg;
use super::path::{navigate, parse_path, value_text};
use super::value::{Json, JSON_SUBTYPE};
use crate::registry::{Arity, FunctionRegistry};

/// Register the `->` and `->>` operators under their sentinel names.
pub(crate) fn register(reg: &mut FunctionRegistry) {
    reg.add_scalar("->", Arity::Exact(2), Arc::new(JsonArrow));
    reg.add_scalar("->>", Arity::Exact(2), Arc::new(JsonArrow2));
}

/// `X -> P` — the JSON text representation of the selected subcomponent, or NULL if
/// it does not exist (json1.html §4.10). Because a selected node is ALWAYS rendered as
/// JSON text (`Json::to_text`), a non-NULL result is itself JSON, so it carries the
/// ephemeral JSON subtype (json1.html §3.4): a directly wrapping JSON function embeds it
/// rather than re-quoting it (`json_array('{"c":[4,5]}' -> '$.c')` is `[[4,5]]`, not
/// `["[4,5]"]`). This is the operator twin of single-path `json_extract` — but where
/// `json_extract` returns the SQL representation and so tags the subtype only for a
/// container, `->` returns the JSON representation for every node and so tags whenever a
/// node is selected. Its sibling `->>` (the SQL representation, `JsonArrow2`) never does.
#[derive(Debug)]
struct JsonArrow;
impl ScalarFunction for JsonArrow {
    fn call(&self, args: &[Value], ctx: &mut dyn FnContext) -> Result<Value> {
        debug_assert!(args.len() == 2, "-> /2 arity precondition");
        let out = with_selected(&args[0], &args[1], |node| Value::Text(node.to_text()))?;
        // `to_text` never yields NULL, so a non-NULL result means a node was selected and
        // is JSON text — tag it. Only a missing path / NULL operand returns NULL here.
        if !out.is_null() {
            ctx.set_result_subtype(JSON_SUBTYPE);
        }
        Ok(out)
    }
}

/// `X ->> P` — the SQL representation of the selected subcomponent, or NULL if it
/// does not exist (json1.html §4.10). Identical selection to `->`; only the leaf
/// representation differs.
#[derive(Debug)]
struct JsonArrow2;
impl ScalarFunction for JsonArrow2 {
    fn call(&self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<Value> {
        debug_assert!(args.len() == 2, "->> /2 arity precondition");
        with_selected(&args[0], &args[1], |node| node.to_sql_scalar())
    }
}

/// Shared selection for both operators: parse `doc`, normalize `spec` to a JSON path,
/// navigate, and hand the selected node (if any) to `f`. A NULL `doc` or `spec`, or a
/// path that selects nothing, is SQL NULL; a malformed `doc` is an error. Borrowing
/// the node into `f` avoids cloning the selected subtree — each operator allocates
/// only the one representation it returns.
fn with_selected<F>(doc: &Value, spec: &Value, f: F) -> Result<Value>
where
    F: FnOnce(&Json) -> Value,
{
    if doc.is_null() || spec.is_null() {
        return Ok(Value::Null);
    }
    let root = parse_json_arg(doc)?;
    let steps = parse_path(&normalize_path(spec))?;
    Ok(match navigate(&root, &steps) {
        Some(node) => f(node),
        None => Value::Null,
    })
}

/// Normalize the right operand of `->` / `->>` into a JSON path per json1.html §4.10's
/// abbreviation rules, applied to the evaluated value (so a column or expression right
/// operand works, not only a literal). The dispatch is by value TYPE, matching §4.10's
/// wording ("if the right operand is an integer value N … an alphanumeric text label X"):
///
/// * an INTEGER `N` is an array index: `$[N]` when non-negative, or `$[#-K]` when it is
///   a negative value `-K` (indexing from the end of the array, SQLite 3.47.0+);
/// * a text value beginning with `$` is already a full JSON path, used verbatim;
/// * any other text value is an alphanumeric object label `X`, meaning `$.X` — emitted
///   verbatim, exactly the spec's `'$.X'` notation (so a text label that is all digits,
///   e.g. `"3"`, is the object key `$.3`, NOT the array index `$[3]`).
///
/// A non-integer, non-text value falls back to its text rendering as a label — kept total
/// rather than panicking (§4.10 defines only integer and text operands).
///
/// EDGE (spec-undefined, flagged — not just an omission): §4.10 restricts the label form
/// to *alphanumeric* labels, and this follows the spec literally by emitting `$.X`
/// unescaped. So a label that itself contains a path metacharacter (`.` / `[` / `"`), such
/// as `'a.b'`, becomes the path `$.a.b` and is split by the path parser (into `a`→`b`)
/// rather than selecting one member literally named `a.b`. That input is outside §4.10's
/// defined (alphanumeric) surface, so there is no spec-correct value to pin. If a real-
/// `sqlite3` ever shows the whole label is treated as a single key, switch
/// THIS branch to emit a quoted, escaped `$."<X>"` (via `super::value::escape_string_into`,
/// which round-trips through the quoted-label path decoder) so a dotted/spaced label
/// resolves as one key. Left as `$.X` because that is what the spec text says.
fn normalize_path(spec: &Value) -> Cow<'_, str> {
    if let Value::Integer(n) = spec {
        // A negative `n` renders as "-K", so "$[#" + "-K" + "]" is the "$[#-K]" form.
        return Cow::Owned(if *n >= 0 { format!("$[{n}]") } else { format!("$[#{n}]") });
    }
    let text = value_text(spec);
    if text.starts_with('$') {
        text
    } else {
        Cow::Owned(format!("$.{text}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::testutil::NullCtx;

    /// Call an operator scalar against a no-op context, expecting `Ok`.
    fn call(f: &dyn ScalarFunction, doc: &Value, spec: &Value) -> Value {
        f.call(&[doc.clone(), spec.clone()], &mut NullCtx).expect("operator should succeed")
    }
    fn call_err(f: &dyn ScalarFunction, doc: &Value, spec: &Value) -> minisqlite_types::Error {
        f.call(&[doc.clone(), spec.clone()], &mut NullCtx).expect_err("operator should error")
    }
    fn t(s: &str) -> Value {
        Value::Text(s.into())
    }
    fn as_text(v: &Value) -> &str {
        match v {
            Value::Text(s) => s,
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn normalize_path_covers_abbreviation_rules() {
        // Non-negative / negative integer indices (§4.10, 3.47.0+ for negative).
        assert_eq!(normalize_path(&Value::Integer(3)), "$[3]");
        assert_eq!(normalize_path(&Value::Integer(0)), "$[0]");
        assert_eq!(normalize_path(&Value::Integer(-1)), "$[#-1]");
        assert_eq!(normalize_path(&Value::Integer(-3)), "$[#-3]");
        // A text value that is already a path is used verbatim.
        assert_eq!(normalize_path(&t("$.c[2].f")), "$.c[2].f");
        assert_eq!(normalize_path(&t("$")), "$");
        // An alphanumeric label X becomes '$.X'.
        assert_eq!(normalize_path(&t("a")), "$.a");
        assert_eq!(normalize_path(&t("c")), "$.c");
        // Dispatch is by TYPE, so a text label that is all digits is still a label
        // ('$.3' object key), distinct from the integer index '$[3]' above.
        assert_eq!(normalize_path(&t("3")), "$.3");
    }

    #[test]
    fn abbreviation_dispatch_is_by_value_type() {
        // §4.10 dispatches on the operand's TYPE (not its text): an INTEGER is an array
        // index, a TEXT value is an object label — even when that text is all digits.
        // So text "3" selects object member "3" (`$.3`); integer 3 selects array element
        // 3 (`$[3]`). This is the one branch that distinguishes label from index, and it
        // is otherwise unexercised by the doc examples (which use non-numeric labels).
        assert!(matches!(call(&JsonArrow2, &t(r#"{"3":9}"#), &t("3")), Value::Integer(9)));
        assert!(matches!(call(&JsonArrow2, &t("[11,22,33,44]"), &Value::Integer(3)), Value::Integer(44)));
        // The text label "3" looks for object key "3"; on an ARRAY there is no such key,
        // so it selects nothing (it does NOT fall through to integer indexing).
        assert!(matches!(call(&JsonArrow2, &t("[11,22,33,44]"), &t("3")), Value::Null));
    }

    #[test]
    fn arrow_returns_json_representation() {
        // A string subcomponent comes back QUOTED; a number/bool/null as JSON text.
        assert_eq!(as_text(&call(&JsonArrow, &t(r#"{"a":"xyz"}"#), &t("$.a"))), r#""xyz""#);
        assert_eq!(as_text(&call(&JsonArrow, &t(r#"{"a":2}"#), &t("a"))), "2");
        assert_eq!(as_text(&call(&JsonArrow, &t(r#"{"a":2.5}"#), &t("a"))), "2.5");
        assert_eq!(as_text(&call(&JsonArrow, &t(r#"{"a":true}"#), &t("a"))), "true");
        assert_eq!(as_text(&call(&JsonArrow, &t(r#"{"a":null}"#), &t("$.a"))), "null");
        // A container comes back as its canonical JSON text.
        assert_eq!(as_text(&call(&JsonArrow, &t(r#"{"c":[4,5]}"#), &t("c"))), "[4,5]");
        // An integer index into an array (verbatim doc example).
        assert_eq!(as_text(&call(&JsonArrow, &t("[11,22,33,44]"), &Value::Integer(3))), "44");
    }

    #[test]
    fn arrow2_returns_sql_representation() {
        // The same string DEQUOTED; numbers as INTEGER/REAL; bool as 1/0.
        assert_eq!(as_text(&call(&JsonArrow2, &t(r#"{"a":"xyz"}"#), &t("$.a"))), "xyz");
        assert!(matches!(call(&JsonArrow2, &t(r#"{"a":2}"#), &t("a")), Value::Integer(2)));
        assert!(matches!(call(&JsonArrow2, &t(r#"{"a":2.5}"#), &t("a")), Value::Real(r) if r == 2.5));
        assert!(matches!(call(&JsonArrow2, &t(r#"{"a":true}"#), &t("a")), Value::Integer(1)));
        assert!(matches!(call(&JsonArrow2, &t(r#"{"a":false}"#), &t("a")), Value::Integer(0)));
        // A JSON null subcomponent is SQL NULL under ->> (but text 'null' under ->).
        assert!(matches!(call(&JsonArrow2, &t(r#"{"a":null}"#), &t("$.a")), Value::Null));
        // A container is its JSON text as a TEXT value.
        assert_eq!(as_text(&call(&JsonArrow2, &t(r#"{"c":[4,5]}"#), &t("c"))), "[4,5]");
        // An integer index returns the element's SQL value.
        assert!(matches!(call(&JsonArrow2, &t("[11,22,33,44]"), &Value::Integer(3)), Value::Integer(44)));
        assert!(matches!(call(&JsonArrow2, &t("[3,2,1]"), &Value::Integer(0)), Value::Integer(3)));
    }

    #[test]
    fn missing_and_null_yield_null() {
        // Missing key / out-of-range index -> NULL.
        assert!(matches!(call(&JsonArrow, &t(r#"{"a":2}"#), &t("$.x")), Value::Null));
        assert!(matches!(call(&JsonArrow, &t("[1]"), &Value::Integer(5)), Value::Null));
        // NULL left operand -> NULL (both operators).
        assert!(matches!(call(&JsonArrow, &Value::Null, &t("a")), Value::Null));
        assert!(matches!(call(&JsonArrow2, &Value::Null, &t("a")), Value::Null));
        // NULL right operand -> NULL.
        assert!(matches!(call(&JsonArrow, &t(r#"{"a":2}"#), &Value::Null), Value::Null));
    }

    #[test]
    fn negative_index_from_end_selects_last() {
        // '$.c[#-1]' selects the last element; passed here as an integer -1 index into
        // the array directly, and as the full path against the object.
        assert_eq!(as_text(&call(&JsonArrow, &t("[4,5]"), &Value::Integer(-1))), "5");
        assert_eq!(as_text(&call(&JsonArrow, &t(r#"{"a":2,"c":[4,5],"f":7}"#), &t("$.c[#-1]"))), "5");
    }

    #[test]
    fn chaining_reparses_intermediate_json() {
        // `-> 'c'` yields the array's JSON text; feeding it back to `-> 2` (index) then
        // `->> 'f'` mirrors the doc chain `-> 'c' -> 2 ->> 'f'` -> 7.
        let step1 = call(&JsonArrow, &t(r#"{"a":2,"c":[4,5,{"f":7}]}"#), &t("c"));
        let step2 = call(&JsonArrow, &step1, &Value::Integer(2));
        assert_eq!(as_text(&step2), r#"{"f":7}"#);
        assert!(matches!(call(&JsonArrow2, &step2, &t("f")), Value::Integer(7)));
    }

    #[test]
    fn malformed_document_is_an_error() {
        // A non-JSON left operand errors, just like the other JSON functions.
        assert!(matches!(call_err(&JsonArrow, &t("{"), &t("a")), minisqlite_types::Error::Sql(m) if m == "malformed JSON"));
        assert!(matches!(call_err(&JsonArrow2, &t("not json"), &t("a")), minisqlite_types::Error::Sql(_)));
    }

    #[test]
    fn registered_under_sentinel_names() {
        let reg = FunctionRegistry::builtins();
        assert!(reg.resolve_scalar("->", 2).is_ok());
        assert!(reg.resolve_scalar("->>", 2).is_ok());
    }
}
