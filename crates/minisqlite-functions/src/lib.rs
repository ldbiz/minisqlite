//! The built-in SQL function library: the registry that resolves a function name
//! (and argument count) to a concrete [`ScalarFunction`](minisqlite_expr::ScalarFunction)
//! or [`AggregateFunction`](minisqlite_expr::AggregateFunction) handle, plus the
//! implementations of the built-in functions themselves.
//!
//! The binder (in `minisqlite-plan`) is the only caller of the registry: it looks
//! a name up once at bind time and stores the resolved `Arc<dyn …>` handle in the
//! IR, so per-row dispatch is a vtable call and never a name lookup. This crate
//! therefore owns two concerns kept in separate files: name -> handle *resolution*
//! ([`registry`]) and the per-family *implementations* ([`scalar`], [`agg`],
//! [`datetime`]). Each family lives in its own module so a later family lands
//! in its own file and never has to edit the registry or this hub.
//!
//! This root is a thin re-export hub — the real code lives in the submodules.

mod agg;
mod datetime;
mod json;
mod registry;
mod scalar;

pub use json::{
    json_table_rows, JsonTableKind, JsonTableRows, JSON_TABLE_COLUMN_COUNT,
    JSON_TABLE_HIDDEN_COLUMN_COUNT,
};
pub use registry::{Arity, FunctionRegistry};
