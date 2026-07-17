//! The function-calling contract: the traits a built-in function implements, plus
//! the aggregate-call vocabulary the planner/executor build on.
//!
//! Scalar and aggregate functions are resolved to `Arc<dyn …>` handles at bind
//! time and stored in the IR, so per-row dispatch is a vtable call, never a name
//! lookup. The library that *implements* these traits lives in a later crate
//! (`minisqlite-functions`); this crate only defines the contract and proves it
//! with tiny stub functions in the tests.

use std::sync::Arc;

use minisqlite_types::{Collation, Result, Value};

use crate::context::FnContext;
use crate::ir::EvalExpr;

/// A scalar (row-at-a-time) SQL function.
///
/// `call` receives the already-evaluated arguments and decides its own NULL
/// behavior — the evaluator does **not** pre-check for NULL arguments, because
/// functions like `coalesce`/`ifnull`/`quote` are meaningful on NULL. `Send + Sync`
/// so a resolved handle can be shared across threads (WAL readers); `Debug` so the
/// IR can derive `Debug`.
pub trait ScalarFunction: std::fmt::Debug + Send + Sync {
    fn call(&self, args: &[Value], ctx: &mut dyn FnContext) -> Result<Value>;

    /// Whether this function is DETERMINISTIC: two calls with identical arguments within
    /// one statement always yield the same result. `true` by default (the vast majority of
    /// built-ins are pure functions of their arguments); override to `false` for a function
    /// whose result can vary across calls with the same arguments — `random()` /
    /// `randomblob()` (a fresh draw each call), the date/time functions (which read the
    /// wall clock for `'now'`), and the connection counters `last_insert_rowid()` /
    /// `changes()` / `total_changes()` (which advance per row during DML).
    ///
    /// Used to decide whether a correlated subquery containing the function is safe to
    /// memoize (see `SubPlan::deterministic` in `minisqlite-plan`): a non-deterministic
    /// function makes the subquery non-deterministic, so the evaluator re-runs it rather
    /// than reusing a cached result. Reporting `false` only forgoes an optimization;
    /// reporting a non-deterministic function as `true` would be a wrong answer.
    ///
    /// SAFETY-CRITICAL CONVENTION (hand-maintained): the default is `true`, so a NEW
    /// non-pure built-in that forgets to override this, or a new impurity source that is
    /// neither a `ScalarFunction` nor an `EvalExpr::Now`, silently reports deterministic and
    /// becomes a wrong answer once such a subquery is memoized. It is safe TODAY only because
    /// every current built-in that is NOT pure overrides it (a standing enumeration test in
    /// `minisqlite-functions` `scalar/misc.rs` guards the known non-pure set). Keep that
    /// invariant as the function surface grows.
    fn deterministic(&self) -> bool {
        true
    }

    /// A COLLATION-SENSITIVE scalar function (the multi-argument `min`/`max`,
    /// lang_corefunc.html) returns a handle specialized to `collation` — the collation the
    /// binder resolved by the function's own rule (min/max: the first argument, left→right,
    /// that defines a collating function, else BINARY). Baking the choice into the handle
    /// here (rather than a per-row lookup) mirrors the aggregate
    /// [`AggregateFunction::new_accumulator`] seam, so the per-row comparison is a plain
    /// `compare_values(_, _, self.collation)`.
    ///
    /// The default is collation-INSENSITIVE and returns `None`, keeping the shared registry
    /// handle unchanged — every function whose result does not depend on a text collating
    /// sequence (all of them except min/max) keeps it. The genuine guarantee here is that the
    /// chosen collation is baked into an IMMUTABLE handle at bind time (no per-row lookup),
    /// mirroring [`AggregateFunction::new_accumulator`].
    ///
    /// It is NOT un-forgettable: like `deterministic()`'s default `true`, a NEW
    /// collation-sensitive scalar that forgot to override this would silently compare under
    /// BINARY (the same "forgot to opt in" hazard as a post-construction setter, not a
    /// stronger guarantee). It is a hand-maintained convention — safe today because the
    /// collation-sensitive set is exactly min/max, guarded by a standing enumeration test in
    /// `minisqlite-functions` `scalar/math.rs`. Keep that guard as the set grows.
    fn specialize_collation(&self, _collation: Collation) -> Option<Arc<dyn ScalarFunction>> {
        None
    }
}

/// An aggregate SQL function: a factory for per-group accumulators.
///
/// One accumulator is created per group; `DISTINCT`, `FILTER`, and `ORDER BY` are
/// applied by the aggregation *operator* (see [`AggregateCall`]) before the values
/// reach [`AggregateAccumulator::step`], so an accumulator only sees the values it
/// must fold.
///
/// `collation` is the argument's collating sequence (the caller passes
/// [`AggregateCall::arg_collations`]`[0]`, or BINARY when there is none). It matters only
/// to ORDER-SENSITIVE aggregates — `min`/`max`, whose result is the value sorting
/// first/last under the argument's §7.1 collation (datatype3.html §7.1,
/// lang_aggfunc.html) — so most accumulators ignore it. It is a REQUIRED parameter (not a
/// post-construction setter) so a caller cannot forget to supply it and silently fold a
/// NOCASE/RTRIM column under BINARY.
pub trait AggregateFunction: std::fmt::Debug + Send + Sync {
    fn new_accumulator(&self, collation: Collation) -> Box<dyn AggregateAccumulator>;

    /// Whether this aggregate's result is a pure function of how many rows are stepped,
    /// independent of the argument values AND of any per-step context — i.e. `count`.
    ///
    /// The aggregate operator's `count(*)` row-count fast path (in `minisqlite-exec`,
    /// `AggregateCursor::build_row_count_only`) uses this to replay a bare row COUNT
    /// instead of folding every input row. It defaults to `false`, so the gate is
    /// FAIL-CLOSED: a newly-added aggregate — even a zero-argument one — is never
    /// mis-classified into that fast path until it explicitly opts in here. Return `true`
    /// only if `finalize` after `n` steps depends on `n` alone (the caller still requires
    /// the *bare* `count(*)` form — no argument, `FILTER`, `DISTINCT`, or `ORDER BY`).
    fn is_row_count(&self) -> bool {
        false
    }
}

/// The mutable per-group state of an aggregate: fold each row's arguments in with
/// [`step`](AggregateAccumulator::step), then produce the group result with
/// [`finalize`](AggregateAccumulator::finalize).
///
/// `finalize` takes `&mut self` (not `self`) so window functions can read a running
/// aggregate without consuming it.
pub trait AggregateAccumulator {
    /// Fold one row's (post-FILTER, post-DISTINCT) arguments into the group state.
    fn step(&mut self, args: &[Value], ctx: &mut dyn FnContext) -> Result<()>;
    /// Fold the SAME argument row `n` times. The default replays `step` `n` times, so
    /// every accumulator is correct unchanged; an accumulator whose repeated fold has a
    /// closed form (e.g. `count`, which adds `n`) MAY override this to do it in O(1).
    ///
    /// This is an OPTIMIZATION HOOK, not a new fold rule: it MUST leave the accumulator in
    /// exactly the state `n` `step` calls with the same `args`/`ctx` would. The `count(*)`
    /// row-count fast path uses it to apply one identical (empty-argument) step a large
    /// number of times without a per-row virtual call.
    fn step_many(&mut self, n: usize, args: &[Value], ctx: &mut dyn FnContext) -> Result<()> {
        for _ in 0..n {
            self.step(args, ctx)?;
        }
        Ok(())
    }
    /// Produce the current group result.
    fn finalize(&mut self, ctx: &mut dyn FnContext) -> Result<Value>;
}

/// One ORDER BY term, bound to a ready-to-evaluate key expression plus its sort
/// modifiers. Shared vocabulary for `ORDER BY`, aggregate `ORDER BY`, and window
/// framing.
///
/// `nulls_first` is `None` when the query did not specify `NULLS FIRST/LAST`, so
/// the planner can apply SQLite's default (NULLs sort with the low end: first for
/// `ASC`, last for `DESC`).
#[derive(Debug, Clone)]
pub struct SortKey {
    pub expr: EvalExpr,
    pub desc: bool,
    pub nulls_first: Option<bool>,
    pub collation: Collation,
}

/// A bound aggregate invocation as it appears in a query: the resolved function,
/// its argument expressions, and the modifiers the aggregation operator applies
/// around the accumulator (`DISTINCT`, `FILTER (WHERE …)`, and aggregate
/// `ORDER BY`).
#[derive(Debug, Clone)]
pub struct AggregateCall {
    pub func: Arc<dyn AggregateFunction>,
    pub distinct: bool,
    pub args: Vec<EvalExpr>,
    pub filter: Option<EvalExpr>,
    pub order_by: Vec<SortKey>,
    /// The collating sequence for each argument (aligned with `args`), each its §7.1
    /// collation (an explicit postfix `COLLATE`, else the argument column's declared
    /// collation, else BINARY — datatype3.html §7.1, lang_select.html §2.6), NOT
    /// unconditionally BINARY. Resolved once at bind time from each argument's source
    /// expression; empty for a `count(*)` / zero-argument call. Two consumers read it:
    /// (1) `DISTINCT` argument dedup compares the argument tuple with SQLite's `=`
    /// operator, so each argument folds under `arg_collations[i]`; (2) order-sensitive
    /// aggregates (`min`/`max`) compare under `arg_collations[0]` (passed to
    /// [`AggregateFunction::new_accumulator`]). A non-`DISTINCT`, non-min/max call carries
    /// but never reads it.
    pub arg_collations: Vec<Collation>,
}

// Compile-time contract: the aggregate/sort vocabulary shares the IR's `Send + Sync`
// requirement (see the matching guard in `ir.rs`) so a prepared plan holding these can
// cross WAL reader threads. `AggregateFunction: Send + Sync` makes the `Arc` shareable;
// the rest is `EvalExpr`/`Collation`/scalars. A non-`Sync` field added later breaks
// this arm at compile time rather than silently downstream.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<AggregateCall>();
    assert_send_sync::<SortKey>();
};
