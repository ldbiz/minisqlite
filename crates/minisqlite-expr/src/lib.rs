//! Compiled expression IR, three-valued-logic evaluator, and the function-calling
//! contract for the query engine.
//!
//! This crate is deliberately **AST-free**: it depends only on `minisqlite-types`
//! and knows nothing about SQL syntax. A separate *binder* (in `minisqlite-plan`)
//! lowers the SQL AST into [`EvalExpr`], resolving column names to integer register
//! slots ([`EvalExpr::Column`]), function names to resolved handles
//! ([`ScalarFunction`]/[`AggregateFunction`]), and comparison affinity + collation
//! to precomputed [`CompareMeta`]. Because the binder resolves everything up front,
//! an unresolved name is *unrepresentable* here and the per-row evaluator does no
//! string lookups, hashmap probes, or affinity recomputation.
//!
//! It is the highest fan-in seam of the query layer: the binder, the executor
//! (`minisqlite-exec`), and the built-in function library (`minisqlite-functions`)
//! all build on the types and traits re-exported below. Real code lives in the
//! submodule files; this root is only a re-export hub.

mod context;
mod coerce;
mod datetime;
mod eval;
mod function;
mod ir;
mod pattern;

pub use ir::{
    ArithOp, BitOp, CaseWhen, CmpOp, CompareMeta, EvalExpr, LikeKind, NowKind, RaiseKind,
    SubqueryId, UnaryOp,
};

pub use context::{EvalContext, FnContext};

pub use function::{
    AggregateAccumulator, AggregateCall, AggregateFunction, ScalarFunction, SortKey,
};

pub use coerce::{to_integer, to_number_for_arith, truth};

pub use pattern::{glob_matches, like_matches};

pub use datetime::civil_from_unix;

pub use eval::{eval, eval_with_subtype};
