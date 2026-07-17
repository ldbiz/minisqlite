//! Planner — the planning layer of the engine. It compiles a parsed statement into
//! an executable [`Plan`] (an operator tree of [`PlanNode`]s over the access-path
//! leaves), choosing an index seek over a full scan when an index covers a lookup or
//! range. The executor runs the plan it is handed; it never re-plans.
//!
//! This root is a thin re-export hub. Real code lives in the submodule files so a
//! feature lands in its own file rather than contending on `lib.rs`:
//! - [`plan`] — the [`Plan`] and the [`PlanNode`] operator vocabulary, plus the
//!   shared ROW/REGISTER convention (its module doc) that the compiler and executor
//!   both bind against.
//! - [`window`] — the window-function KIND ([`WindowFuncKind`]) and FRAME
//!   ([`WindowFrame`]) vocabulary each [`WindowFunc`] carries.
//! - [`access`] — the access-path leaves ([`SeqScan`], [`RowidScan`], [`IndexScan`]
//!   and their ops) where the index-vs-scan choice is recorded.
//! - [`planner`] — the [`Planner`] seam trait (the one planning route).
//! - [`query_planner`] — [`QueryPlanner`], the concrete [`Planner`] implementation.
//! - [`plan_ctx`] — the per-statement compile context ([`PlanCtx`]).
//! - [`bind`] — the binder: SQL AST expression → [`minisqlite_expr::EvalExpr`].
//! - [`access_path`] — base-table access-path selection ([`plan_table_access`]).
//! - [`compile`] — the statement compilers (FROM / VALUES / aggregate / SELECT).
//! - [`colname`] — result-column name derivation.

mod access;
mod access_path;
mod bind;
mod colname;
mod compile;
mod plan;
mod plan_ctx;
mod planner;
mod query_planner;
mod window;

#[cfg(test)]
mod tests;

pub use access::{IndexOp, IndexScan, RangeBound, RowidOp, RowidScan, ScanDirection, SeqScan};

pub use plan::{
    Aggregate, CheckConstraint, ColumnSpec, ConflictTarget, CreateIndexPlan, CreateTablePlan,
    CtePlan, Delete, GeneratedProgram, Insert, InsteadOf, InsteadOfEvent, Join, JoinStrategy,
    JoinType, MinMaxBare, MinMaxSeek, MinMaxSource, OnConflict, Plan, PlanNode, SetOp, SetOpKind,
    SortLimit, SubPlan, TableGenerated, TriggerProgram, Update, UpsertActionPlan, UpsertClause,
    UpsertPlan, Window, WindowFunc,
};

// The trigger-timing vocabulary a `TriggerProgram` carries. Re-exported from the SQL
// AST crate so the executor (which fires triggers) can name it through the plan layer
// without taking a direct dependency on `minisqlite-sql`.
pub use minisqlite_sql::TriggerTiming;

pub use window::{FrameBound, FrameExclude, FrameUnits, WindowFrame, WindowFuncKind};

pub use planner::Planner;

pub use query_planner::QueryPlanner;

pub use access_path::{col_reg, plan_table_access, TableAccess};
pub use bind::{
    best_effort_collation, bind_expr, compare_meta, compare_meta_subject, function_is_aggregate,
    operand_collation, parse_collation, Grouping, ResolvedColumn, Scope, Source,
};
pub use compile::ctas::{ctas_columns, CtasColumn};
pub use compile::{compile_result, compile_select, compile_subquery, Compiled};
pub use colname::result_column_name;
pub use plan_ctx::PlanCtx;

// The trigger-compile entrypoint and its DML-event descriptor. Exposed for the firing
// executor to drive RUNTIME trigger recursion: the compile pass expands only ONE level
// (a trigger action's own DML carries an empty `triggers` vec, so a self-referential
// trigger cannot loop at compile time), so when an action performs DML the executor
// recompiles the action target's own triggers through this entrypoint and fires them.
pub use compile::trigger::{compile_triggers, TriggerDmlEvent};

// The table-index-key-expression compiler. Exposed for the executor's FK-cascade path,
// which discovers a child table at RUNTIME and must compile its expression-index key
// programs there (the plan-time DML compilers use it crate-internally for INSERT/UPDATE/
// DELETE nodes).
pub use compile::index_expr::compile_table_index_key_exprs;

// The table-index PARTIAL-predicate compiler, exposed for the SAME FK-cascade runtime path:
// a cascade that deletes/updates a child row must gate a PARTIAL child index's maintenance on
// its WHERE predicate exactly as a plan-time DML node does, so it compiles the child's partial
// predicates at RUNTIME through this entrypoint. The plan-time DML compilers use it
// crate-internally for INSERT/UPDATE/DELETE nodes.
pub use compile::index_expr::compile_table_index_partial_predicates;

// The generated-column program binder. Exposed for the executor's FK enforcement path,
// which scans a parent/child table (discovered at RUNTIME) whose VIRTUAL generated columns
// must be computed to read/compare the key — the same runtime-compile pattern as the index
// key compiler above. `populate_generated_for_triggers` is exposed for the executor's
// `recursive_triggers` recompile path, which compiles a trigger set at RUNTIME (outside the
// top-level `populate_generated` pass) and must fill each recompiled action plan's generated
// map before running it — otherwise a nested action writing a generated-column table stores
// VIRTUAL columns and computes NULLs.
pub use compile::generated::{bind_generated_programs, populate_generated_for_triggers};
