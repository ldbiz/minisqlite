//! Shared `RETURNING`-clause evaluation for the DML write operators — INSERT (rowid,
//! WITHOUT ROWID, and the UPSERT `DO UPDATE` path), UPDATE, and DELETE.
//!
//! ONE home for the RETURNING eval loop so its namespace behavior is defined once and a
//! future copy cannot silently reintroduce the single-namespace assumption. A RETURNING
//! expression may contain a scalar subquery over ANY namespace: `lang_returning.html` §3
//! says "If there are subqueries in the RETURNING clause, those subqueries may contain
//! aggregates and window functions" (subqueries are explicitly allowed), and §2.2 warns
//! only about subqueries that reference the table BEING MODIFIED — a subquery that reads a
//! DIFFERENT table (an `ATTACH`-ed db, `temp`) is well-defined. So RETURNING MUST evaluate
//! under the whole-slice shared read view (`PagerSet::source()`), the SAME view INSERT's
//! `VALUES` and UPDATE's `SET` exprs evaluate under; a single-namespace
//! `Pagers::One { db }` view fails closed on any read naming another namespace
//! (`env::Pagers::One::get` → "single-namespace context (db N) cannot reach namespace M").
//!
//! Safe because RETURNING eval is READ-ONLY and runs AFTER the row's own write: each caller
//! releases its exclusive `&mut` target-store borrow (or, for a path that streamed under a
//! held write borrow, closes it) BEFORE handing the shared `source()` view here, and eval
//! fills an OWNED `Row` of `Value`s — so nothing derived from the shared borrow outlives
//! this call, and the next per-row iteration is free to reborrow the target mutably.

use minisqlite_catalog::Catalog;
use minisqlite_expr::{eval, EvalExpr};
use minisqlite_plan::Plan;
use minisqlite_types::{Result, Row, Value};

use crate::context::EvalCtx;
use crate::env::{Env, Pagers};
use crate::runtime::Runtime;

/// Evaluate a DML statement's `RETURNING` expressions over one written row `regs` and
/// return the produced output row.
///
/// `regs` is the row the RETURNING exprs bind against — `[c0..c_{N-1}, rowid]` for a rowid
/// table (rowid in register `N`) or `[c0..c_{N-1}]` for a WITHOUT ROWID table — and doubles
/// as the `outer` a correlated RETURNING subquery reads. `pagers` MUST be the whole-slice
/// shared read view (`PagerSet::source()`) so a RETURNING subquery may reach any namespace
/// (see the module doc); passing a single-namespace `Pagers::One` would reintroduce the
/// cross-namespace fail-closed bug this helper exists to prevent.
pub(crate) fn eval_returning(
    returning: &[EvalExpr],
    regs: &[Value],
    catalog: &dyn Catalog,
    pagers: Pagers<'_>,
    plan: &Plan,
    rt: &mut Runtime,
) -> Result<Row> {
    let env = Env { catalog, pagers, plan };
    let mut ctx = EvalCtx { rt, env, outer: regs };
    let mut out = Vec::with_capacity(returning.len());
    for expr in returning {
        out.push(eval(expr, regs, &mut ctx)?);
    }
    Ok(out)
}
