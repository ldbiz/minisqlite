//! Statement compilation: turns a bound statement into a plan tree, split by concern
//! so each feature lands in its own file.
//!
//! - [`select`] — the SELECT orchestrator (a single query core), over [`from`],
//!   [`values`], and the output / [`aggregate`] layer.
//! - [`compound`] — a compound SELECT (`UNION` / `INTERSECT` / `EXCEPT`).
//! - [`ctas`] — the column schema (names + declared types) a `CREATE TABLE ... AS
//!   SELECT` builds from its SELECT.
//! - [`cte`] — a leading `WITH` (common table expressions).
//! - [`view`] — expanding a `CREATE VIEW`-defined view referenced in a FROM.
//! - [`insert`] / [`update`] / [`delete`] — the DML compilers.
//! - [`instead_of`] — redirecting DML on a VIEW through its `INSTEAD OF` triggers.
//! - [`minmax_index`] — the MIN/MAX index/rowid seek fast path (optoverview.html §13):
//!   satisfy a single bare-column `MIN`/`MAX` over a whole table with one b-tree seek
//!   instead of a full-scan aggregate.
//! - [`order_scan`] — the opportunistic `ORDER BY` -> scan-order rewrite that lets the
//!   output layer omit the `Sort` when a rowid/index walk already yields the order.
//! - [`covering`] — the covering-index (index-only scan) post-pass: mark an `IndexScan`
//!   leaf `covering` when its index already carries every column the query reads, so the
//!   executor skips the by-rowid table fetch (optoverview.html §"Covering Indexes").
//! - [`trigger`] — compiling a table's triggers onto a DML node (NEW/OLD scope).
//! - [`check`] — binding a table's CHECK predicates onto an INSERT/UPDATE node.
//! - [`generated`] — binding a table's GENERATED-column expressions onto the plan-level
//!   [`crate::plan::Plan::generated`] map the read/write paths consume.
//! - [`index_expr`] — binding an index's stored key EXPRESSIONS (an index on an
//!   expression, `lang_createindex.html` §1.2) into executable [`minisqlite_expr::EvalExpr`]s.

pub mod aggregate;
pub mod check;
pub mod compound;
pub mod covering;
pub mod ctas;
pub mod cte;
pub mod delete;
pub mod from;
pub mod generated;
pub mod index_expr;
pub mod insert;
pub mod instead_of;
pub mod minmax_index;
pub mod order_scan;
pub mod pragma_tvf;
pub mod select;
pub mod trigger;
pub mod update;
pub mod values;
pub mod view;
pub mod window;

pub use aggregate::{compile_result, Compiled};
pub use delete::compile_delete;
pub use insert::compile_insert;
pub use select::{compile_select, compile_subquery};
pub use update::compile_update;

// The `*_with_parent` DML/SELECT entrypoints and `compile_triggers` are reached by their
// module paths (`compile::trigger::…`, `compile::insert::…`) from the trigger compiler
// and the DML compilers, so they are intentionally NOT re-exported here.
