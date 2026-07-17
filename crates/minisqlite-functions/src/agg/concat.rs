//! `group_concat(X)`, `group_concat(X, SEP)`, and `string_agg(X, SEP)`
//! (`lang_aggfunc.html`).
//!
//! All concatenate the text renderings of the non-NULL `X` values of a group,
//! joined by a separator that appears *between* values only (never before the first
//! or after the last). `string_agg(X, Y)` is an exact alias of the two-argument
//! `group_concat(X, Y)` (PostgreSQL/SQL-Server spelling), so one implementation
//! backs both; they differ only in the registered arity. The separator defaults to
//! `","` for the one-argument `group_concat(X)`.
//!
//! The separator is read from each row's second argument, so it is effectively
//! per-row: the separator placed before the k-th emitted value is taken from the
//! row that produced that k-th value (matching sqlite, which appends the current
//! row's separator before the current row's value). A NULL separator contributes an
//! empty string. Rows whose `X` is NULL are skipped entirely (no value and no
//! separator), and a group with no non-NULL `X` yields NULL.

use std::borrow::Cow;
use std::sync::Arc;

use minisqlite_expr::{AggregateAccumulator, AggregateFunction, FnContext};
use minisqlite_types::{integer_to_text, real_to_text, Collation, Result, Value};

use crate::registry::{Arity, FunctionRegistry};

/// Register `group_concat` (1 or 2 args) and its alias `string_agg` (exactly 2).
pub(crate) fn register(reg: &mut FunctionRegistry) {
    reg.add_aggregate("group_concat", Arity::Range(1, 2), Arc::new(GroupConcat));
    // string_agg(X, Y) is group_concat(X, Y) under a different name; the required
    // 2-argument form is the only difference, enforced by the arity.
    reg.add_aggregate("string_agg", Arity::Exact(2), Arc::new(GroupConcat));
}

/// Shared factory for `group_concat`/`string_agg`.
#[derive(Debug)]
struct GroupConcat;
impl AggregateFunction for GroupConcat {
    fn new_accumulator(&self, _collation: Collation) -> Box<dyn AggregateAccumulator> {
        Box::new(ConcatAcc { out: String::new(), seen: false })
    }
}

/// Running concatenation for one group. `seen` records whether any non-NULL value
/// has been appended, which both suppresses the leading separator and distinguishes
/// an empty result ("" from an appended empty string) from NULL (no values at all).
struct ConcatAcc {
    /// Built in place across steps — never re-concatenated from scratch.
    out: String,
    seen: bool,
}

impl AggregateAccumulator for ConcatAcc {
    fn step(&mut self, args: &[Value], _ctx: &mut dyn FnContext) -> Result<()> {
        let x = &args[0];
        if x.is_null() {
            return Ok(()); // Skip the whole row: no value and no separator.
        }
        if self.seen {
            // Separator before every value after the first, taken from THIS row's
            // second argument (or "," for the one-argument form; empty if NULL).
            match args.get(1) {
                None => self.out.push(','),
                Some(Value::Null) => {}
                Some(sep) => self.out.push_str(&render(sep)),
            }
        }
        self.out.push_str(&render(x));
        self.seen = true;
        Ok(())
    }

    fn finalize(&mut self, _ctx: &mut dyn FnContext) -> Result<Value> {
        if self.seen {
            // Clone rather than take: `finalize` must be re-callable for window frames.
            Ok(Value::Text(self.out.clone()))
        } else {
            Ok(Value::Null)
        }
    }
}

/// The text rendering of a value, as `sqlite3_value_text` would yield it: TEXT
/// as-is, numbers via their canonical text form, a BLOB as its bytes read as UTF-8
/// (lossily, since this engine's TEXT is UTF-8; the common valid-UTF-8 blob is
/// exact). NULL is never rendered — callers skip NULL `X` and treat a NULL
/// separator as empty before reaching here.
fn render(v: &Value) -> Cow<'_, str> {
    match v {
        Value::Text(s) => Cow::Borrowed(s.as_str()),
        Value::Integer(i) => Cow::Owned(integer_to_text(*i)),
        Value::Real(r) => Cow::Owned(real_to_text(*r)),
        Value::Blob(b) => String::from_utf8_lossy(b),
        Value::Null => Cow::Borrowed(""),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agg::testutil::{drive1, drive_rows};

    fn text(v: Result<Value>) -> String {
        match v.expect("group_concat should succeed") {
            Value::Text(s) => s,
            other => panic!("expected Text, got {other:?}"),
        }
    }
    fn is_null(v: Result<Value>) -> bool {
        matches!(v.expect("group_concat should succeed"), Value::Null)
    }

    #[test]
    fn default_separator_is_comma() {
        let vals = [Value::Text("a".into()), Value::Text("b".into()), Value::Text("c".into())];
        assert_eq!(text(drive1(&GroupConcat, &vals)), "a,b,c");
    }

    #[test]
    fn custom_separator_between_values_only() {
        let rows = vec![
            vec![Value::Text("a".into()), Value::Text("-".into())],
            vec![Value::Text("b".into()), Value::Text("-".into())],
            vec![Value::Text("c".into()), Value::Text("-".into())],
        ];
        // The separator appears between values, never leading or trailing.
        assert_eq!(text(drive_rows(&GroupConcat, &rows)), "a-b-c");
    }

    #[test]
    fn separator_is_taken_from_the_current_row() {
        // The separator before the k-th value comes from the k-th row, so the very
        // first row's separator is never used (there is nothing before the first).
        let rows = vec![
            vec![Value::Text("a".into()), Value::Text("X".into())],
            vec![Value::Text("b".into()), Value::Text("+".into())],
            vec![Value::Text("c".into()), Value::Text("*".into())],
        ];
        assert_eq!(text(drive_rows(&GroupConcat, &rows)), "a+b*c");
    }

    #[test]
    fn null_values_are_skipped_without_separators() {
        // A NULL X emits neither a value nor a separator, so no doubled/leading sep.
        let vals = [
            Value::Null,
            Value::Text("a".into()),
            Value::Null,
            Value::Text("b".into()),
            Value::Null,
        ];
        assert_eq!(text(drive1(&GroupConcat, &vals)), "a,b");
    }

    #[test]
    fn no_non_null_values_is_null() {
        assert!(is_null(drive1(&GroupConcat, &[])));
        assert!(is_null(drive1(&GroupConcat, &[Value::Null, Value::Null])));
    }

    #[test]
    fn numbers_and_blobs_render_to_text() {
        let vals = [Value::Integer(1), Value::Real(2.5), Value::Integer(3)];
        assert_eq!(text(drive1(&GroupConcat, &vals)), "1,2.5,3");
        // A (valid-UTF-8) blob is concatenated as its bytes.
        let with_blob = vec![
            vec![Value::Text("x".into()), Value::Text(",".into())],
            vec![Value::Blob(b"yz".to_vec()), Value::Text(",".into())],
        ];
        assert_eq!(text(drive_rows(&GroupConcat, &with_blob)), "x,yz");
    }

    #[test]
    fn null_separator_joins_with_empty_string() {
        let rows = vec![
            vec![Value::Text("a".into()), Value::Null],
            vec![Value::Text("b".into()), Value::Null],
        ];
        assert_eq!(text(drive_rows(&GroupConcat, &rows)), "ab");
    }

    #[test]
    fn string_agg_behaves_like_two_arg_group_concat() {
        let rows = vec![
            vec![Value::Text("a".into()), Value::Text("; ".into())],
            vec![Value::Text("b".into()), Value::Text("; ".into())],
        ];
        assert_eq!(text(drive_rows(&GroupConcat, &rows)), "a; b");
    }

    #[test]
    fn single_value_has_no_separator() {
        assert_eq!(text(drive1(&GroupConcat, &[Value::Text("solo".into())])), "solo");
    }
}
