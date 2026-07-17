//! The binder: name resolution ([`scope`]) and expression lowering ([`expr`]) that
//! turn the SQL AST into compiled `minisqlite_expr::EvalExpr`. Split into small
//! files so a feature lands in its own edit cell.

pub mod expr;
pub mod scope;

pub use expr::{
    arg_list_collations, best_effort_collation, bind_expr, compare_meta, compare_meta_subject,
    defined_collation, function_is_aggregate, operand_collation,
};
pub use scope::{
    new_old_sources, parse_collation, BareCapture, Coalesced, Grouping, ResolvedColumn, Scope,
    Source, Windowing,
};
