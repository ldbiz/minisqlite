//! SQLite's JSON (JSON1) functions, text-JSON forms (`spec/sqlite-doc/json1.html`).
//!
//! SQLite stores JSON as ordinary TEXT (there is no JSON storage class), so every
//! function here reads and writes JSON text. The subsystem is split by concern so a
//! feature lands in its own file:
//!
//! * [`value`] — the parsed [`Json`](value::Json) tree, canonical/pretty rendering,
//!   and the `Value`<->`Json` bridge.
//! * [`parse`] — a total recursive-descent parser for RFC-8259 JSON plus the JSON5
//!   extensions SQLite accepts, tracking canonicity and first-error position.
//! * [`path`] — JSONPath parsing, read navigation, and the in-place edit primitives.
//! * [`scalar`] — the read/construct scalar functions (`json`, `json_array`,
//!   `json_extract`, `json_type`, `json_valid`, …).
//! * [`edit`] — the mutating scalar functions (`json_insert`/`replace`/`set`,
//!   `json_remove`, `json_patch`).
//! * [`agg`] — the aggregates (`json_group_array`, `json_group_object`).
//! * [`operators`] — the JSON operator forms (`->`, `->>`).
//! * [`table`] — the row generation for the `json_each()`/`json_tree()` table-valued
//!   functions (a FROM-clause row source, not a registry scalar/aggregate).
//!
//! # Value subtype (json1.html §3.4)
//!
//! A JSON function tags its result with the ephemeral JSON *subtype* so that when the
//! result is passed *directly* to another JSON function it is embedded as JSON rather
//! than re-quoted as a string — `json_array(json('[1,2]'))` is `[[1,2]]` (not
//! `["[1,2]"]`) and `json_set('{}','$.a',json('[1]'))` stores the array `[1]` (not the
//! string `"[1]"`). The subtype rides a value only within a single expression
//! evaluation — it never touches [`Value`](minisqlite_types::Value) or a stored row,
//! and is lost across storage/rows/subqueries — exactly as in SQLite. The evaluator
//! (`minisqlite-expr`) publishes each argument's subtype and reads back the result's
//! through the `FnContext` subtype channel; a JSON function whose result is JSON marks
//! it via [`set_result_subtype`](minisqlite_expr::FnContext::set_result_subtype), and
//! embeds a subtyped `value` argument via
//! [`value_to_json_with_subtype`](value::value_to_json_with_subtype) instead of quoting
//! it.
//!
//! # Known limitations (deliberate, flagged)
//!
//! * **JSONB deferred.** The `jsonb_*` binary variants are not implemented (they need
//!   the JSONB on-disk codec). A BLOB JSON argument is read as UTF-8 *text* JSON (the
//!   documented legacy behavior), never as JSONB. The *text* table-valued functions
//!   `json_each()`/`json_tree()` ARE implemented (see [`table`]), including their hidden
//!   `json`/`root` input columns (excluded from `SELECT *`, selectable by name).
//! * **Number tokens are preserved.** A number parsed from JSON *text* renders
//!   verbatim on a minify (`json('1e3')` -> `'1e3'`, `json('[1.50]')` -> `'[1.50]'`);
//!   only JSON5 spellings (hex, leading `+`, leading/trailing `.`, `Infinity`) are
//!   canonicalized. A JSON integer literal that overflows `i64` still renders
//!   verbatim and reports `json_type` 'integer', but a single-path `json_extract` of
//!   it yields a REAL (there is no `i64`-typed value to return).
//!
//! (String escapes are *not* a limitation: an input `\uXXXX` is decoded on parse and
//! re-emitted in canonical form on render — e.g. `json('"\u0041"')` -> `'"A"'`, a
//! control char re-escapes to `\uXXXX`. This matches SQLite, which canonicalizes
//! string escapes on a minify rather than preserving the input spelling.)

mod agg;
mod edit;
mod operators;
mod parse;
mod path;
mod scalar;
mod table;
mod value;

pub use table::{
    json_table_rows, JsonTableKind, JsonTableRows, JSON_TABLE_COLUMN_COUNT,
    JSON_TABLE_HIDDEN_COLUMN_COUNT,
};

/// Register every JSON scalar and aggregate function into `reg`. This is the single
/// entry point the crate root wires in (`crate::json::register`), mirroring the
/// other function families.
pub(crate) fn register(reg: &mut crate::FunctionRegistry) {
    scalar::register(reg);
    edit::register(reg);
    agg::register(reg);
    operators::register(reg);
}

#[cfg(test)]
pub(crate) mod testutil {
    //! Shared test scaffolding for the JSON submodules: a do-nothing [`FnContext`]
    //! (the JSON functions never touch the clock/RNG/counters) and two aggregate
    //! drivers mirroring how the executor drives one group.

    use minisqlite_expr::{AggregateFunction, FnContext};
    use minisqlite_types::{Collation, Result, Value};

    /// A [`FnContext`] whose every capability is a fixed constant. The JSON
    /// functions never call these, so the values are arbitrary; the stub only
    /// exists to satisfy the `&mut dyn FnContext` parameter.
    pub(crate) struct NullCtx;
    impl FnContext for NullCtx {
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
    }

    /// Drive `func` over a group of single-argument rows and finalize.
    pub(crate) fn drive1(func: &dyn AggregateFunction, vals: &[Value]) -> Result<Value> {
        let mut ctx = NullCtx;
        let mut acc = func.new_accumulator(Collation::Binary);
        for v in vals {
            acc.step(std::slice::from_ref(v), &mut ctx)?;
        }
        acc.finalize(&mut ctx)
    }

    /// Drive `func` over a group of multi-argument rows and finalize.
    pub(crate) fn drive_rows(func: &dyn AggregateFunction, rows: &[Vec<Value>]) -> Result<Value> {
        let mut ctx = NullCtx;
        let mut acc = func.new_accumulator(Collation::Binary);
        for row in rows {
            acc.step(row, &mut ctx)?;
        }
        acc.finalize(&mut ctx)
    }
}
