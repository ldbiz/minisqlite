//! CHECK-constraint compilation: bind a table's stored CHECK predicates into the
//! [`CheckConstraint`]s an [`Insert`](crate::plan::Insert) / [`Update`](crate::plan::Update)
//! node carries for the executor to evaluate.
//!
//! Per `spec/sqlite-doc/lang_createtable.html` §3.7 a CHECK predicate is evaluated on
//! INSERT and UPDATE against the NEW row; a result of integer 0 / real 0.0 (after a
//! cast-to-NUMERIC) is a violation, while NULL or any nonzero value passes, and the
//! conflict algorithm is always ABORT. Those EVALUATION semantics are the executor's job
//! — this only carries the raw BOUND predicate, mirroring how a column DEFAULT is bound
//! by the planner but evaluated by the executor.
//!
//! Binding: each predicate binds against a single base-table [`Source`] over `def` at
//! register base 0, so a column reference resolves exactly as every other DML expression
//! does — column `i` → `Column(i)`, and an INTEGER PRIMARY KEY alias column → the trailing
//! rowid register `N` (where its logical value lives in the executor's `[c0..c_{N-1},
//! rowid]` row). The executor evaluates INSERT checks against the inserted row and UPDATE
//! checks against the POST-assignment new row — the SAME layout — so no remapping is
//! needed here.

use minisqlite_catalog::TableDef;
use minisqlite_types::{DbIndex, Result};

use crate::bind::{bind_expr, Scope, Source};
use crate::plan::CheckConstraint;
use crate::plan_ctx::PlanCtx;

/// Bind every CHECK predicate stored on `def` into a [`CheckConstraint`] for its
/// INSERT/UPDATE node. `detail` is the table name — the conformance tests assert only the
/// error KIND (`Error::Constraint`), not the exact `CHECK constraint failed: <detail>`
/// text, so a richer per-constraint detail (a constraint name, or the predicate's source
/// text) is a later refinement, and the table name is a correct, stable choice now.
///
/// A predicate that names a column the table does not have fails to BIND here (an
/// `Error::Sql("no such column: …")`), matching real SQLite rejecting such a CREATE TABLE
/// — the gap surfaces loudly at plan time rather than being silently dropped. The one
/// exception is a bare *double-quoted* unknown name: SQLite's DQS legacy makes
/// `CHECK(x = "lit")` fall back to the text literal 'lit' rather than erroring
/// (quirks.html §8), applied uniformly here because CHECK binds through `bind_expr`.
pub(crate) fn compile_checks(ctx: &mut PlanCtx, def: &TableDef) -> Result<Vec<CheckConstraint>> {
    if def.checks.is_empty() {
        return Ok(Vec::new());
    }
    // A single base-table scope over `def` at base 0: column `i` → `Column(i)`, and the
    // INTEGER PRIMARY KEY alias → the rowid register `N` — the same scope shape UPDATE's
    // assignment RHS binds against (see `compile::update`).
    let sources =
        [Source::BaseTable { exposed_name: def.name.clone(), table: def, db: DbIndex::MAIN, base: 0 }];
    let scope = Scope::new(&sources);

    // ASSUMPTION: a CHECK predicate references only the new row's own columns — SQLite
    // rejects a subquery or bind parameter in a CHECK at CREATE TABLE time, so `def.checks`
    // never carries one. That is what keeps this bind INERT w.r.t. the planner's numbering
    // state: `bind_expr` here advances none of `ctx`'s param/subquery/CTE counters, so
    // compile_checks needs no trial-bind savepoint and its placement relative to one does
    // not matter. If a future extension (or a non-validating catalog-load path) ever lets a
    // subquery/param into `def.checks`, this inertness breaks — re-check the savepoint
    // interaction in `plan_ctx` before relying on it.

    let mut out = Vec::with_capacity(def.checks.len());
    for expr in &def.checks {
        out.push(CheckConstraint { expr: bind_expr(&scope, ctx, expr)?, detail: def.name.clone() });
    }
    Ok(out)
}
