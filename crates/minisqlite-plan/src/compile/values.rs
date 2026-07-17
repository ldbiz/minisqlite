//! `VALUES (...), (...)` compilation: bind each row's expressions into a
//! [`PlanNode::Values`], and name the columns `column1`, `column2`, … as SQLite does.
//!
//! VALUES rows have no local columns in scope, so they normally bind against an empty
//! scope. The optional `parent` supports a trigger action's `INSERT INTO t VALUES
//! (NEW.x)`: with the trigger's OLD/NEW scope as parent, a `NEW.x` / `OLD.x` reference
//! resolves through it (the correlated-subquery machinery), exactly as it would in a
//! correlated subquery. `None` = today's behavior (an empty scope, no columns visible).

use minisqlite_sql::Expr;
use minisqlite_types::{Error, Result};

use crate::bind::{bind_expr, Scope};
use crate::plan::PlanNode;
use crate::plan_ctx::PlanCtx;

/// Compile a `VALUES` row list into a [`PlanNode::Values`] plus its column names.
/// Every row must have the same arity; the width is taken from the first row.
/// `parent` is the enclosing scope for a trigger action (so `VALUES (NEW.x)` resolves
/// `NEW`/`OLD` through it); `None` for an ordinary VALUES with no columns in scope.
pub fn compile_values(
    ctx: &mut PlanCtx,
    rows: &[Vec<Expr>],
    parent: Option<&Scope>,
) -> Result<(PlanNode, Vec<String>)> {
    if rows.is_empty() {
        return Err(Error::sql("VALUES must have at least one row"));
    }
    let width = rows[0].len();
    // No local sources — only the (optional) parent's OLD/NEW columns are visible.
    // `with_parent(&[], None)` is exactly `Scope::empty()`, so the non-trigger path is
    // unchanged.
    let scope = Scope::with_parent(&[], parent);
    let mut out_rows = Vec::with_capacity(rows.len());
    for row in rows {
        if row.len() != width {
            return Err(Error::sql(format!(
                "all VALUES rows must have the same number of terms ({width})"
            )));
        }
        let mut bound = Vec::with_capacity(width);
        for e in row {
            bound.push(bind_expr(&scope, ctx, e)?);
        }
        out_rows.push(bound);
    }
    let names = (1..=width).map(|i| format!("column{i}")).collect();
    Ok((PlanNode::Values { rows: out_rows }, names))
}
