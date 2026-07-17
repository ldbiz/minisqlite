//! Trigger FIRING: for one DML row, build the `OLD ++ NEW` register frame and run the
//! pre-compiled trigger programs the DML node carries (their `WHEN` gate + action
//! `Plan`s). Shared by the INSERT / UPDATE / DELETE write operators so the frame layout
//! and the fire loop have ONE home and cannot drift between the three.
//!
//! ## The register frame (the binding contract, mirrored from the plan side)
//! For a target of `C` columns, `W = C + 1` (the `[c0..c_{N-1}, rowid]` base-row width).
//! The frame is `OLD ++ NEW`, width `2W`: OLD in `[0, W)`, NEW in `[W, 2W)`. The compile
//! side (`minisqlite_plan::compile::trigger`) bound each `OLD.col_k`→`Column(k)`,
//! `NEW.col_k`→`Column(W+k)`, and the rowids to `Column(C)` / `Column(W+C)`, and placed
//! each action's own operators at base `2W`. So running an action (or evaluating `WHEN`)
//! with the frame as the correlated OUTER row — exactly the mechanism a correlated
//! subplan uses (`context::open_subquery`) — makes every `NEW`/`OLD` reference resolve to
//! the frame and every action-table column resolve to its own scan above it. By event one
//! half is absent and all-NULL: an INSERT has no OLD, a DELETE has no NEW.
//!
//! ## Firing order, atomicity, recursion
//! The caller fires BEFORE triggers before the base row write and AFTER triggers after
//! it, once per affected row (FOR EACH ROW). Any error a trigger raises (a `RAISE`, a
//! failing action) propagates out of the whole statement; autocommit DML runs inside an
//! implicit transaction that rolls back on that error, so propagation IS the statement
//! atomicity — there is deliberately no per-statement savepoint here. An action that
//! fires the same event again recurses, bounded by [`Runtime::enter_trigger`] and
//! normally terminated sooner by a `WHEN` condition.

use std::sync::{Arc, OnceLock};

use minisqlite_catalog::{Catalog, TableDef};
use minisqlite_expr::{eval, truth, EvalExpr};
use minisqlite_functions::FunctionRegistry;
use minisqlite_plan::{
    compile_triggers, populate_generated_for_triggers, Plan, PlanCtx, PlanNode, TriggerDmlEvent,
    TriggerProgram, TriggerTiming,
};
use minisqlite_types::{DbIndex, Result, Value};

use crate::context::EvalCtx;
use crate::env::{PagerSet, Env};
use crate::runner::{build_cursor, build_dml};
use crate::runtime::{Runtime, TriggerEventKey};

/// When a set of triggers fires relative to the base-table row operation.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Phase {
    /// BEFORE the base row is written.
    Before,
    /// AFTER the base row is written.
    After,
}

/// What a trigger fire tells the calling DML operator to do with the current row.
/// [`fire_triggers`] returns this so the operator can honour a `RAISE(IGNORE)` control
/// signal (lang_createtrigger.html §RAISE) without it being an error.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum TriggerFlow {
    /// No `RAISE(IGNORE)` fired — process the row normally.
    Proceed,
    /// A fired trigger evaluated `RAISE(IGNORE)`: abandon the current row's operation and
    /// the rest of its trigger programs, WITHOUT an error and WITHOUT rolling back changes
    /// already made — the statement continues at the next row. A BEFORE fire returning this
    /// means the row is not written; an AFTER fire means the write already happened and
    /// stays (only the row's remaining post-write work is skipped).
    IgnoreRow,
}

/// Build the `OLD ++ NEW` frame (width `2W`, `W = col_count + 1`) for one row. Each of
/// `old` / `new` is the OWNED `[c0..c_{N-1}, rowid]` half (width `W`), moved into the frame;
/// a `None` half is filled with `W` NULLs (INSERT has no OLD, DELETE has no NEW). A short
/// half is padded with NULL and an over-long one truncated, so the frame is ALWAYS exactly
/// `2W` wide and a `Column(i)` read can never shift into the wrong half. Taking the halves
/// BY VALUE lets the caller (which already cloned the row's live columns into the half) hand
/// them off without this builder re-cloning every column again — one clone per fired row,
/// not two.
pub(crate) fn build_frame(
    col_count: usize,
    old: Option<Vec<Value>>,
    new: Option<Vec<Value>>,
) -> Vec<Value> {
    build_frame_w(col_count + 1, old, new)
}

/// Build the `OLD ++ NEW` frame for a WITHOUT ROWID target: identical to [`build_frame`]
/// except each half is width `W = col_count` (a WR row has NO rowid register — the plan
/// side's [`new_old_sources`](minisqlite_plan) lays `NEW` at `old.width()`, which is `N`
/// for a WR table and `N+1` for a rowid table, so the executor must match that width or a
/// `Column(i)` read shifts into the wrong half). Each of `old`/`new` is the width-`N`
/// `[c0..c_{N-1}]` half; an absent half is `N` NULLs.
pub(crate) fn build_frame_wr(
    col_count: usize,
    old: Option<Vec<Value>>,
    new: Option<Vec<Value>>,
) -> Vec<Value> {
    build_frame_w(col_count, old, new)
}

/// Build a `2W`-wide `OLD ++ NEW` frame from the two owned halves (each reshaped to `w`).
/// `w` is the caller's base-row width — `col_count + 1` for a rowid target (trailing rowid
/// register), `col_count` for a WITHOUT ROWID target — so the OLD/NEW split lands exactly
/// where the compile side bound `Column(k)` / `Column(w+k)`.
fn build_frame_w(w: usize, old: Option<Vec<Value>>, new: Option<Vec<Value>>) -> Vec<Value> {
    let mut frame = Vec::with_capacity(2 * w);
    push_half(&mut frame, old, w);
    push_half(&mut frame, new, w);
    frame
}

/// Move exactly `w` values for one frame half into `frame`: the provided values (reshaped in
/// place to width `w` — padded with NULL if short, truncated if long, no per-value clone),
/// or `w` NULLs when the half is absent.
fn push_half(frame: &mut Vec<Value>, half: Option<Vec<Value>>, w: usize) {
    match half {
        Some(mut vals) => {
            debug_assert_eq!(vals.len(), w, "a trigger frame half must equal the base-row width W");
            vals.resize(w, Value::Null);
            frame.append(&mut vals);
        }
        None => frame.resize(frame.len() + w, Value::Null),
    }
}

/// Fire the triggers matching `phase` for one DML row. `frame` is the already-built
/// `OLD ++ NEW` frame. `enclosing_plan` is the DML statement's own plan — the home for
/// any subquery a `WHEN` references (the compile side registers a WHEN's subqueries into
/// the enclosing statement, not a plan of its own). For each trigger, in firing order:
/// skip unless its timing matches `phase`; if it has a `WHEN`, evaluate it over the frame
/// and skip unless true (false / NULL do not fire); else run every action in order.
///
/// The caller gates this on a non-empty `triggers` vec, so a table with no triggers pays
/// nothing (no frame build, no call) and behaves exactly as before firing existed.
///
/// Returns [`TriggerFlow`]: normally `Proceed`, but `IgnoreRow` when a fired trigger
/// evaluated `RAISE(IGNORE)` — detected via the runtime flag the evaluator set on the
/// (sentinel-error) unwind. On `IgnoreRow` the remaining triggers in this phase are NOT
/// fired ("any subsequent trigger programs … are abandoned", lang_createtrigger.html); a
/// genuine error propagates as before.
pub(crate) fn fire_triggers(
    triggers: &[TriggerProgram],
    phase: Phase,
    frame: &[Value],
    catalog: &dyn Catalog,
    pagers: &mut PagerSet,
    rt: &mut Runtime,
    enclosing_plan: &Plan,
) -> Result<TriggerFlow> {
    for trig in triggers {
        if !timing_matches(trig.timing, phase) {
            continue;
        }
        match fire_program(trig, frame, catalog, &mut *pagers, rt, enclosing_plan) {
            Ok(()) => {}
            // A `RAISE(IGNORE)` in the fired body set the runtime flag and unwound with a
            // sentinel error: consume the flag and report the row-skip (not the error), and
            // stop firing further triggers for this row. Any other error is genuine.
            Err(e) => {
                if rt.take_raise_ignore() {
                    return Ok(TriggerFlow::IgnoreRow);
                }
                return Err(e);
            }
        }
    }
    Ok(TriggerFlow::Proceed)
}

/// Fire ONE trigger program against `frame`: evaluate its `WHEN` gate (a false / NULL
/// result skips the body, per lang_createtrigger.html §2), then run its actions bounded
/// by the recursion guard. Shared by base-table [`fire_triggers`] (after its per-phase
/// timing filter) and the VIEW `INSTEAD OF` operator ([`crate::ops::instead_of`]), which
/// pre-selected its matching programs at compile time and fires each here.
///
/// `enter_trigger` bounds the recursion around this body (an action firing the same event
/// enters one level deeper) AND snapshots the change counters so the actions' own DML is
/// excluded from the outer `changes()` / `last_insert_rowid()` (lang_corefunc.html);
/// `exit_trigger` restores both counter and depth even when an action errors, so they are
/// correct before the error unwinds the statement.
///
/// `pagers` is the connection's WHOLE store set (reborrowed), not a single namespace's
/// pager: each fired action resolves its OWN stamped `node.db` against it, so a
/// cross-namespace trigger action (a TEMP trigger writing `main`, an `INSTEAD OF` view
/// action writing another schema) writes the CORRECT store. A `WHEN` subquery reads the
/// same set. When the set is a single-namespace [`PagerSet::One`] (a test / single-namespace
/// caller), an action naming a different namespace still fails closed loudly in
/// [`PagerSet::target`].
pub(crate) fn fire_program(
    trig: &TriggerProgram,
    frame: &[Value],
    catalog: &dyn Catalog,
    pagers: &mut PagerSet,
    rt: &mut Runtime,
    enclosing_plan: &Plan,
) -> Result<()> {
    if let Some(when) = &trig.when {
        if !when_is_true(when, frame, catalog, pagers, rt, enclosing_plan)? {
            return Ok(());
        }
    }
    let saved = rt.enter_trigger()?;
    let result = run_actions(&trig.actions, frame, catalog, pagers, rt);
    rt.exit_trigger(saved);
    result
}

/// The trigger set a DML write must fire, resolving the two sources WITHOUT cloning the
/// hot-path programs: a TOP-LEVEL write borrows its pre-compiled `node.triggers`; a nested
/// trigger-ACTION write (whose `node.triggers` is empty by the one-level compile bound)
/// gets a memoized recompiled set shared via `Arc`. Keeping the two behind one type lets
/// each DML operator write a single `effective_triggers(...)?.as_slice()` and hold the
/// result across its write loop while it mutates the runtime.
pub(crate) enum EffectiveTriggers<'a> {
    /// The DML node's own pre-compiled set (top-level), borrowed from the plan.
    Borrowed(&'a [TriggerProgram]),
    /// A recompiled set memoized on the runtime (nested recursion), a cheap `Arc` handle.
    Shared(Arc<Vec<TriggerProgram>>),
}

impl EffectiveTriggers<'_> {
    /// The triggers to iterate, from whichever source.
    pub(crate) fn as_slice(&self) -> &[TriggerProgram] {
        match self {
            EffectiveTriggers::Borrowed(s) => s,
            EffectiveTriggers::Shared(a) => &a[..],
        }
    }
}

/// Resolve the [`EffectiveTriggers`] for a DML write on `target` firing `event`. ONE home
/// for the selection so INSERT / UPDATE / DELETE (and the WITHOUT ROWID insert path) cannot
/// drift — including the `recursive_triggers` gate:
///
/// * a top-level write returns its non-empty `node_triggers` (borrowed);
/// * a nested trigger-action write (empty `node_triggers`) recompiles the target's own
///   triggers ONLY when `PRAGMA recursive_triggers` is ON, which SQLite defaults OFF
///   (pragma.html). DOCUMENTED SIMPLIFICATION: with the flag off we suppress ALL nested
///   firing (depth > 0 with the flag off, or depth 0 with no compiled triggers → empty set,
///   nothing fires), whereas real SQLite suppresses only re-entry of a trigger already on the
///   stack (same-trigger / cyclic recursion) and STILL fires a chain of DISTINCT triggers
///   (A's action inserts into a table with its own trigger B). This reduced handling is a
///   blessed scope cut; a faithful version would track the active-trigger stack by name
///   rather than gate all-or-nothing on depth.
pub(crate) fn effective_triggers<'a>(
    node_triggers: &'a [TriggerProgram],
    rt: &mut Runtime,
    catalog: &dyn Catalog,
    db: DbIndex,
    target: &TableDef,
    event: TriggerDmlEvent,
) -> Result<EffectiveTriggers<'a>> {
    if !node_triggers.is_empty() {
        return Ok(EffectiveTriggers::Borrowed(node_triggers));
    }
    if rt.trigger_depth() > 0 && rt.recursive_triggers() {
        return Ok(EffectiveTriggers::Shared(recompile_target_triggers(
            rt, catalog, db, target, event,
        )?));
    }
    Ok(EffectiveTriggers::Borrowed(&[]))
}

/// Re-compile the triggers that fire for `event` on `target`, for RUNTIME trigger
/// recursion, MEMOIZED per statement. The plan compile pass expands only ONE level — a
/// trigger action's own DML carries an EMPTY `triggers` vec (otherwise a self-referential
/// trigger would loop at compile time; see `minisqlite_plan::compile::trigger` and its
/// guard tests) — so when a trigger ACTION performs DML, the executor recompiles the action
/// target's own triggers here to fire them (bounded by the depth guard and, normally, each
/// trigger's `WHEN`).
///
/// The result is target-and-event-stable within a statement, so it is cached on the
/// [`Runtime`] keyed by (table, event) and reused across the firings (once per outer row):
/// the parse+bind+plan happens at most once per (table, event) per statement, not once per
/// row. The builtin function registry it plans against is likewise built once process-wide
/// ([`builtin_registry`]) rather than rebuilt per call. A trigger body that called a custom
/// function would still need the connection's registry threaded in (a documented follow-up);
/// a `WHEN` subquery would not resolve here (its id registers into a throwaway context) — but
/// the common `NEW`/`OLD` comparison `WHEN` recompiles cleanly and no in-scope case uses one.
fn recompile_target_triggers(
    rt: &mut Runtime,
    catalog: &dyn Catalog,
    db: DbIndex,
    target: &TableDef,
    event: TriggerDmlEvent,
) -> Result<Arc<Vec<TriggerProgram>>> {
    // `db` is the action target's namespace; it is BOTH part of the memo key (so a temp/attached
    // table shadowing a same-named one does not share its recompiled set) and passed into
    // `compile_triggers` so the trigger lookup reads that namespace's store (`triggers_on_in`).
    let key = (db, target.name.to_ascii_lowercase(), event_key(event));
    if let Some(hit) = rt.cached_recompiled_triggers(&key) {
        return Ok(hit);
    }
    let mut ctx = PlanCtx::new(builtin_registry(), catalog);
    let mut compiled = compile_triggers(&mut ctx, catalog, db, target, event)?;
    // `compile_triggers` leaves each action plan's `generated` map empty (the compile pass
    // expands one level and defers the generated-column binding to the top-level
    // `populate_generated` pass). A runtime-recompiled set never reaches that pass, so run it
    // here before caching — otherwise a nested action that writes a generated-column table
    // stores its VIRTUAL columns and computes NULLs (record + on-disk-format corruption).
    populate_generated_for_triggers(&mut compiled, builtin_registry(), catalog)?;
    let programs = Arc::new(compiled);
    rt.cache_recompiled_triggers(key, Arc::clone(&programs));
    Ok(programs)
}

/// The owned memo key for `event` (the borrow-carrying [`TriggerDmlEvent`] cannot outlive
/// the call). An `UPDATE`'s assigned columns are part of the key: `UPDATE OF <cols>`
/// filtering makes two updates of the same table with different assigned columns fire
/// different sets, so they must not share a memo entry.
fn event_key(event: TriggerDmlEvent) -> TriggerEventKey {
    match event {
        TriggerDmlEvent::Insert => TriggerEventKey::Insert,
        TriggerDmlEvent::Delete => TriggerEventKey::Delete,
        TriggerDmlEvent::Update { changed_cols } => TriggerEventKey::Update(changed_cols.to_vec()),
    }
}

/// The process-wide builtin function registry, built exactly ONCE. `FunctionRegistry::builtins`
/// does ~100+ registrations (a `String` + `Arc` alloc each) and is target-INDEPENDENT, so it
/// must not be rebuilt per fired row on the nested-DML path; a `OnceLock` shares one immutable
/// copy (it is `Send + Sync`) across every recompile.
///
/// `pub(crate)` so the other executor path that compiles a plan fragment at RUNTIME — the FK
/// cascade in [`crate::ops::foreign_key`], which compiles a child table's expression-index key
/// programs when it discovers the child dynamically — shares this ONE registry rather than
/// standing up a second copy. Both are runtime plan-compilation, both key only builtin
/// functions; a connection-registered custom function is the same documented follow-up for both.
pub(crate) fn builtin_registry() -> &'static FunctionRegistry {
    static BUILTINS: OnceLock<FunctionRegistry> = OnceLock::new();
    BUILTINS.get_or_init(FunctionRegistry::builtins)
}

/// Whether a trigger's compiled timing fires in `phase`. `INSTEAD OF` is view-only and
/// never reaches a base-table DML operator (the compile side rejects it on a table), so
/// it matches no base-table phase here.
fn timing_matches(timing: TriggerTiming, phase: Phase) -> bool {
    matches!(
        (timing, phase),
        (TriggerTiming::Before, Phase::Before) | (TriggerTiming::After, Phase::After)
    )
}

/// Evaluate a trigger `WHEN` over the frame, returning whether it is TRUE (false / NULL
/// → do not fire, per lang_createtrigger.html §2). A subquery in WHEN reads storage
/// immutably through a shared view of the whole store set, so each base leaf resolves its
/// own namespace; the subquery ids resolve against `enclosing_plan`.
fn when_is_true(
    when: &EvalExpr,
    frame: &[Value],
    catalog: &dyn Catalog,
    pagers: &PagerSet,
    rt: &mut Runtime,
    enclosing_plan: &Plan,
) -> Result<bool> {
    let env = Env { catalog, pagers: pagers.source(), plan: enclosing_plan };
    let mut ctx = EvalCtx { rt, env, outer: frame };
    let v = eval(when, frame, &mut ctx)?;
    Ok(truth(&v) == Some(true))
}

/// Run each of a fired trigger's action `Plan`s in order, with `frame` as the correlated
/// outer registers. A DML action goes through the write path ([`build_dml`]); a SELECT
/// action (e.g. a `SELECT RAISE(...)`) through the read path. Each cursor is fully
/// drained — for a DML action that runs its mutation (deferred to the first pull); for a
/// SELECT it forces every row's expression evaluation, which is how a `SELECT
/// RAISE(ABORT,…)` surfaces its error. The action's OWN plan (`action`) supplies its
/// subqueries / CTEs and is passed as `outer` so its `NEW`/`OLD` references read `frame`.
///
/// The enclosing statement's uncorrelated-subquery cache is SAVED and restored around the
/// run: an action `Plan` has its own `SubqueryId` namespace starting at 0 (a FRESH PlanCtx),
/// but runs nested inside the enclosing statement's drain sharing this `Runtime`. Without
/// the save the action's id 0 would read the enclosing statement's cached id 0 — a variant
/// mismatch error, or the action silently computing over the enclosing statement's data.
/// This mirrors `StatementRoot`'s per-statement reset, reopened at the action boundary.
fn run_actions(
    actions: &[Plan],
    frame: &[Value],
    catalog: &dyn Catalog,
    pagers: &mut PagerSet,
    rt: &mut Runtime,
) -> Result<()> {
    let saved = rt.take_subquery_cache();
    let result = run_actions_inner(actions, frame, catalog, pagers, rt);
    // Restore the enclosing cache even if an action errored, discarding the nested entries.
    rt.restore_subquery_cache(saved);
    result
}

/// Run the actions with the subquery cache already emptied (see [`run_actions`]), clearing
/// it again before EACH action so a sibling action's id 0 cannot read the previous action's
/// cached id 0 — each action `Plan` restarts its ids at 0, the same reason `StatementRoot`
/// resets per statement.
fn run_actions_inner(
    actions: &[Plan],
    frame: &[Value],
    catalog: &dyn Catalog,
    pagers: &mut PagerSet,
    rt: &mut Runtime,
) -> Result<()> {
    for action in actions {
        rt.clear_subquery_cache();
        match &action.root {
            PlanNode::Insert(_) | PlanNode::Update(_) | PlanNode::Delete(_) => {
                // Route THIS action to ITS OWN stamped `node.db`: a reborrow of the whole
                // store set lets the action's DML operator target ANY namespace's pager
                // (a cross-namespace trigger action — a TEMP trigger writing `main`, an
                // `INSTEAD OF` view action writing another schema — writes the CORRECT
                // store), while a single-namespace [`PagerSet::One`] keeps its fail-closed
                // guard (an action naming another namespace still errors loudly).
                //
                // Atomicity of that cross-namespace write is the engine's: an autocommit
                // statement's implicit transaction spans EVERY live namespace (see
                // `minisqlite-engine` `run_mutating_collect`), so this action's write to
                // another store is STAGED in that store's open transaction, not
                // auto-committed on the spot — a statement ABORT rolls it back with the rest
                // of the statement, the same atomicity real sqlite gives it.
                let mut cur =
                    build_dml(&action.root, catalog, pagers.reborrow(), action, frame)?;
                while cur.next_row(rt)?.is_some() {}
            }
            _ => {
                // A SELECT action (e.g. `SELECT RAISE(...)`) reads through a shared view of
                // the whole set, so each base leaf resolves its own namespace.
                let env = Env { catalog, pagers: pagers.source(), plan: action };
                let mut cur = build_cursor(&action.root, env, frame)?;
                while cur.next_row(rt)?.is_some() {}
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `build_frame` lays OLD then NEW, each padded to `W = C + 1`, and fills an absent
    /// half with NULLs — so `Column(k)` / `Column(W+k)` / the rowid registers land where
    /// the compile side bound them, independent of which half is present.
    #[test]
    fn frame_places_old_then_new_each_width_w() {
        // C = 2 => W = 3, frame width 6. OLD=[10,11,rowid=1], NEW=[20,21,rowid=2].
        let old = vec![Value::Integer(10), Value::Integer(11), Value::Integer(1)];
        let new = vec![Value::Integer(20), Value::Integer(21), Value::Integer(2)];
        let frame = build_frame(2, Some(old), Some(new));
        assert_eq!(frame.len(), 6, "2W = 6");
        assert!(matches!(frame[0], Value::Integer(10)), "OLD.col0 -> Column(0)");
        assert!(matches!(frame[2], Value::Integer(1)), "OLD.rowid -> Column(C)=2");
        assert!(matches!(frame[3], Value::Integer(20)), "NEW.col0 -> Column(W+0)=3");
        assert!(matches!(frame[5], Value::Integer(2)), "NEW.rowid -> Column(W+C)=5");
    }

    #[test]
    fn insert_frame_has_all_null_old_half() {
        // INSERT: OLD absent -> [NULL; W]; NEW present.
        let new = vec![Value::Integer(7), Value::Integer(1)]; // C=1 => W=2
        let frame = build_frame(1, None, Some(new));
        assert_eq!(frame.len(), 4);
        assert!(frame[..2].iter().all(Value::is_null), "OLD half all NULL for INSERT");
        assert!(matches!(frame[2], Value::Integer(7)), "NEW.col0 -> Column(W+0)=2");
        assert!(matches!(frame[3], Value::Integer(1)), "NEW.rowid -> Column(W+C)=3");
    }

    #[test]
    fn delete_frame_has_all_null_new_half() {
        // DELETE: NEW absent -> [NULL; W]; OLD present.
        let old = vec![Value::Integer(9), Value::Integer(1)]; // C=1 => W=2
        let frame = build_frame(1, Some(old), None);
        assert_eq!(frame.len(), 4);
        assert!(matches!(frame[0], Value::Integer(9)), "OLD.col0 -> Column(0)");
        assert!(matches!(frame[1], Value::Integer(1)), "OLD.rowid -> Column(C)=1");
        assert!(frame[2..].iter().all(Value::is_null), "NEW half all NULL for DELETE");
    }

    #[test]
    fn timing_matches_splits_before_and_after() {
        assert!(timing_matches(TriggerTiming::Before, Phase::Before));
        assert!(timing_matches(TriggerTiming::After, Phase::After));
        assert!(!timing_matches(TriggerTiming::Before, Phase::After));
        assert!(!timing_matches(TriggerTiming::After, Phase::Before));
        // INSTEAD OF fires in no base-table phase.
        assert!(!timing_matches(TriggerTiming::InsteadOf, Phase::Before));
        assert!(!timing_matches(TriggerTiming::InsteadOf, Phase::After));
    }
}
