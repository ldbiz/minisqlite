//! The JSON aggregate functions (`spec/sqlite-doc/json1.html` §4.23):
//! `json_group_array(X)` and `json_group_object(name, value)`.
//!
//! Both build canonical JSON text incrementally in the accumulator (the doc notes
//! the aggregates work in text, not JSONB). A group with no rows yields the empty
//! aggregate — `'[]'` / `'{}'`, never NULL — and a NULL `X` in `json_group_array`
//! is included as JSON `null` (unlike `group_concat`, which skips NULLs). A value
//! that is a BLOB is an error (`"JSON cannot hold BLOB values"`), and a NULL name
//! in `json_group_object` is an error (an object key cannot be NULL).
//!
//! `finalize` takes `&mut self` and must be re-callable for window frames, so it
//! only wraps the running buffer in brackets and never consumes the state.
//!
//! # Value subtype (json1.html §3.4)
//!
//! Like the scalar constructors, an aggregate embeds a `value` operand that came
//! directly from another JSON function rather than re-quoting it:
//! `json_group_array(json('[1]'))` is `[[1]]`, not `["[1]"]`. Each `step` reads the
//! ephemeral subtype the driver publishes for the VALUE operand
//! ([`FnContext::arg_subtype`] — argument 0 for `json_group_array`, argument 1 for
//! `json_group_object`) and renders through [`render_value_with_subtype_into`], which
//! embeds only a genuinely-subtyped TEXT value and quotes everything else (so a plain
//! TEXT value that merely looks like JSON is still quoted). The result of both
//! aggregates is itself JSON text, so `finalize` marks its own result subtype
//! ([`FnContext::set_result_subtype`]) — the same tag the scalars set — so a directly
//! enclosing JSON function would embed it. (The subtype is ephemeral, so it only
//! reaches such an enclosing function within one expression evaluation; carrying it
//! across the aggregate-output row into a separate projection is a further hop that
//! does not affect the direct embedding done here.)

use std::sync::Arc;

use minisqlite_expr::{AggregateAccumulator, AggregateFunction, FnContext};
use minisqlite_types::{Collation, Error, Result, Value};

use super::path::value_text;
use super::value::{escape_string_into, render_value_with_subtype_into, JSON_SUBTYPE};
use crate::registry::{Arity, FunctionRegistry};

/// Register the JSON aggregate functions.
pub(crate) fn register(reg: &mut FunctionRegistry) {
    reg.add_aggregate("json_group_array", Arity::Exact(1), Arc::new(JsonGroupArray));
    reg.add_aggregate("json_group_object", Arity::Exact(2), Arc::new(JsonGroupObject));
}

/// `json_group_array(X)` — a JSON array of every `X` in the group (NULLs included as
/// JSON null); an empty group is `'[]'`.
#[derive(Debug)]
struct JsonGroupArray;
impl AggregateFunction for JsonGroupArray {
    fn new_accumulator(&self, _collation: Collation) -> Box<dyn AggregateAccumulator> {
        Box::new(GroupArrayAcc { body: String::new(), first: true })
    }
}

/// Running array body for one group: the comma-joined element renderings, wrapped
/// in `[]` at finalize. `first` suppresses the leading comma.
struct GroupArrayAcc {
    body: String,
    first: bool,
}

impl AggregateAccumulator for GroupArrayAcc {
    fn step(&mut self, args: &[Value], ctx: &mut dyn FnContext) -> Result<()> {
        let mark = self.body.len();
        if !self.first {
            self.body.push(',');
        }
        // Render the element straight into the buffer (no per-row node/clone unless the
        // value is a subtyped JSON result to embed). If the value can't be JSON (a BLOB),
        // roll the separator back so the errored step leaves the buffer exactly as it was.
        // `X` is argument 0, so its subtype is `arg_subtype(0)`.
        if let Err(e) = render_value_with_subtype_into(&args[0], ctx.arg_subtype(0), &mut self.body) {
            self.body.truncate(mark);
            return Err(e);
        }
        self.first = false;
        Ok(())
    }

    fn finalize(&mut self, ctx: &mut dyn FnContext) -> Result<Value> {
        // The result is JSON text, so tag it with the JSON subtype (json1.html §3.4) —
        // the same mark the scalar constructors set — so a directly enclosing JSON
        // function embeds it rather than re-quoting.
        ctx.set_result_subtype(JSON_SUBTYPE);
        Ok(Value::Text(format!("[{}]", self.body)))
    }
}

/// `json_group_object(name, value)` — a JSON object of every name/value pair in the
/// group; an empty group is `'{}'`.
#[derive(Debug)]
struct JsonGroupObject;
impl AggregateFunction for JsonGroupObject {
    fn new_accumulator(&self, _collation: Collation) -> Box<dyn AggregateAccumulator> {
        Box::new(GroupObjectAcc { body: String::new(), first: true })
    }
}

/// Running object body for one group: the comma-joined `"name":value` members,
/// wrapped in `{}` at finalize.
struct GroupObjectAcc {
    body: String,
    first: bool,
}

impl AggregateAccumulator for GroupObjectAcc {
    fn step(&mut self, args: &[Value], ctx: &mut dyn FnContext) -> Result<()> {
        // A NULL name has no text form — an object key cannot be NULL. Any other
        // type is coerced to its text form for the key.
        if args[0].is_null() {
            return Err(Error::sql("json_group_object() labels must be TEXT"));
        }
        let name = value_text(&args[0]);
        let mark = self.body.len();
        if !self.first {
            self.body.push(',');
        }
        escape_string_into(&name, &mut self.body);
        self.body.push(':');
        // Render the value straight into the buffer (no per-row node/clone unless the
        // value is a subtyped JSON result to embed). A BLOB value is an error; roll the
        // whole member back so the buffer is unchanged. The VALUE is argument 1 (the
        // name is argument 0), so its subtype is `arg_subtype(1)`.
        if let Err(e) = render_value_with_subtype_into(&args[1], ctx.arg_subtype(1), &mut self.body) {
            self.body.truncate(mark);
            return Err(e);
        }
        self.first = false;
        Ok(())
    }

    fn finalize(&mut self, ctx: &mut dyn FnContext) -> Result<Value> {
        // The result is JSON text — tag it with the JSON subtype (json1.html §3.4), as
        // `json_group_array` and the scalar constructors do.
        ctx.set_result_subtype(JSON_SUBTYPE);
        Ok(Value::Text(format!("{{{}}}", self.body)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::testutil::{drive1, drive_rows};

    fn text(v: Result<Value>) -> String {
        match v.expect("json aggregate should succeed") {
            Value::Text(s) => s,
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn group_array_includes_nulls_and_empty_is_bracket() {
        let vals = [Value::Integer(1), Value::Null, Value::Text("x".into())];
        assert_eq!(text(drive1(&JsonGroupArray, &vals)), r#"[1,null,"x"]"#);
        // A group with no rows is '[]', not NULL.
        assert_eq!(text(drive1(&JsonGroupArray, &[])), "[]");
    }

    #[test]
    fn group_array_quotes_text_values() {
        let vals = [Value::Text("[1,2]".into()), Value::Real(2.5)];
        assert_eq!(text(drive1(&JsonGroupArray, &vals)), r#"["[1,2]",2.5]"#);
    }

    #[test]
    fn group_array_blob_is_error() {
        assert!(drive1(&JsonGroupArray, &[Value::Blob(vec![1])]).is_err());
    }

    #[test]
    fn group_object_builds_object_and_empty_is_brace() {
        let rows = [
            vec![Value::Text("a".into()), Value::Integer(1)],
            vec![Value::Text("b".into()), Value::Text("x".into())],
        ];
        assert_eq!(text(drive_rows(&JsonGroupObject, &rows)), r#"{"a":1,"b":"x"}"#);
        // A group with no rows is '{}', not NULL.
        assert_eq!(text(drive_rows(&JsonGroupObject, &[])), "{}");
    }

    #[test]
    fn group_object_null_name_is_error() {
        let rows = [vec![Value::Null, Value::Integer(1)]];
        assert!(drive_rows(&JsonGroupObject, &rows).is_err());
    }

    #[test]
    fn group_object_coerces_non_text_name_to_its_text_form() {
        // A non-text name is coerced to text for the key (only NULL is an error).
        let rows = [
            vec![Value::Integer(7), Value::Integer(1)],
            vec![Value::Real(2.5), Value::Text("x".into())],
        ];
        assert_eq!(text(drive_rows(&JsonGroupObject, &rows)), r#"{"7":1,"2.5":"x"}"#);
    }

    /// A [`FnContext`] that carries the ephemeral subtype channel the real executor
    /// backs on the `Runtime`, so these unit tests can drive an accumulator with the
    /// exact publish-arg-subtypes / read-result-subtype protocol the exec driver uses.
    struct SubtypeCtx {
        arg_subtypes: Vec<u8>,
        result_subtype: u8,
    }
    impl FnContext for SubtypeCtx {
        fn now_unix_millis(&self) -> i64 {
            0
        }
        fn random_i64(&mut self) -> i64 {
            0
        }
        fn fill_random(&mut self, _buf: &mut [u8]) {}
        fn last_insert_rowid(&self) -> i64 {
            0
        }
        fn changes(&self) -> i64 {
            0
        }
        fn total_changes(&self) -> i64 {
            0
        }
        fn set_arg_subtypes(&mut self, s: &[u8]) {
            self.arg_subtypes.clear();
            self.arg_subtypes.extend_from_slice(s);
        }
        fn arg_subtype(&self, i: usize) -> u8 {
            self.arg_subtypes.get(i).copied().unwrap_or(0)
        }
        fn set_result_subtype(&mut self, st: u8) {
            self.result_subtype = st;
        }
        fn take_result_subtype(&mut self) -> u8 {
            std::mem::take(&mut self.result_subtype)
        }
    }

    /// Drive an aggregate over `rows` (each an `(args, per-arg subtypes)` pair),
    /// publishing the row's subtypes before each `step` exactly as the exec driver does,
    /// then finalizing and reading back the finalize result subtype. Returns the
    /// finalized value and that result subtype.
    fn drive_subtyped(
        func: &dyn AggregateFunction,
        rows: &[(Vec<Value>, Vec<u8>)],
    ) -> (Result<Value>, u8) {
        let mut ctx = SubtypeCtx { arg_subtypes: Vec::new(), result_subtype: 0 };
        let mut acc = func.new_accumulator(Collation::Binary);
        for (args, subs) in rows {
            ctx.set_arg_subtypes(subs);
            if let Err(e) = acc.step(args, &mut ctx) {
                return (Err(e), 0);
            }
        }
        // Mirror the driver: clear any stale result subtype, finalize, read it back.
        ctx.set_result_subtype(0);
        let out = acc.finalize(&mut ctx);
        (out, ctx.take_result_subtype())
    }

    #[test]
    fn group_array_embeds_subtyped_value_and_marks_result() {
        // A VALUE that carries the JSON subtype (index 0) is embedded as its structure,
        // and finalize marks the whole result as JSON — the aggregate counterpart of
        // json1.html §4.3/§3.4.
        let rows = [
            (vec![Value::Text("[1]".into())], vec![JSON_SUBTYPE]),
            (vec![Value::Text("[2]".into())], vec![JSON_SUBTYPE]),
        ];
        let (v, st) = drive_subtyped(&JsonGroupArray, &rows);
        assert_eq!(text(v), r#"[[1],[2]]"#);
        assert_eq!(st, JSON_SUBTYPE, "json_group_array result is itself JSON");
    }

    #[test]
    fn group_array_quotes_unsubtyped_lookalike() {
        // The SAME text with NO subtype is quoted as a JSON string — the guard that a
        // plain TEXT value that merely looks like JSON is never inlined.
        let rows = [(vec![Value::Text("[1]".into())], vec![0])];
        let (v, st) = drive_subtyped(&JsonGroupArray, &rows);
        assert_eq!(text(v), r#"["[1]"]"#);
        assert_eq!(st, JSON_SUBTYPE);
    }

    #[test]
    fn group_object_embeds_subtyped_value_and_marks_result() {
        // The VALUE operand (index 1) carrying the JSON subtype is embedded; the name is
        // still an ordinary quoted key.
        let rows = [(vec![Value::Text("a".into()), Value::Text("[1]".into())], vec![0, JSON_SUBTYPE])];
        let (v, st) = drive_subtyped(&JsonGroupObject, &rows);
        assert_eq!(text(v), r#"{"a":[1]}"#);
        assert_eq!(st, JSON_SUBTYPE, "json_group_object result is itself JSON");
    }

    #[test]
    fn group_object_quotes_unsubtyped_value() {
        // The value quote-guard: no subtype -> the JSON-looking text is quoted.
        let rows = [(vec![Value::Text("a".into()), Value::Text("[1]".into())], vec![0, 0])];
        let (v, _st) = drive_subtyped(&JsonGroupObject, &rows);
        assert_eq!(text(v), r#"{"a":"[1]"}"#);
    }

    #[test]
    fn aggregates_register_and_classify() {
        let reg = FunctionRegistry::builtins();
        assert!(reg.resolve_aggregate("json_group_array", 1).is_ok());
        assert!(reg.resolve_aggregate("json_group_object", 2).is_ok());
        assert!(reg.is_aggregate("json_group_array"));
        assert!(reg.is_aggregate("json_group_object"));
        // They are aggregates, not scalars.
        assert!(reg.resolve_scalar("json_group_array", 1).is_err());
    }
}
