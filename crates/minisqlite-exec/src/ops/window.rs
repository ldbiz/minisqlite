//! `Window` — window functions over the input, emitting `input_row ++ [v_0 … v_{k-1}]`
//! (width = input width + `functions.len()`), one appended column per [`WindowFunc`].
//!
//! This operator materializes: a window value cannot be known until its whole partition
//! has been seen, so on the first pull it drains the input into a buffer, computes every
//! appended value, then streams the extended rows in INPUT order. The buffer is the
//! input rows once (the required window materialization, bounded by input cardinality);
//! partitioning and framing operate on ROW INDICES ([`partition::Partition`]), so no row
//! is copied — the only extra allocation is the computed values (`n × k`) plus, per
//! partition, an index/order-key vector.
//!
//! # Module layout
//!
//! * [`partition`] — the partition / order / peer-group machinery ([`Partition`]).
//! * [`frame`] — the frame engine ([`frame::frame_positions`] → [`frame::Frame`]) that
//!   turns a `ROWS`/`RANGE`/`GROUPS` frame plus `EXCLUDE` into the ordered positions a
//!   window value reads.
//! * [`ranking`] / [`navigation`] — the ranking (`row_number`/`rank`/…) and navigation
//!   (`lag`/`first_value`/…) kinds, each computed over exactly the [`partition`]/[`frame`]
//!   API that the aggregate kind in this file uses.
//!
//! # Scope of THIS node (loud gaps, never silent wrong answers)
//!
//! * The AGGREGATE window kind ([`WindowFuncKind::Aggregate`]) is implemented FULLY over
//!   every frame the plan can express — `ROWS`/`RANGE`/`GROUPS`, each boundary form
//!   (`UNBOUNDED`, `n PRECEDING/FOLLOWING`, `CURRENT ROW`), and all four `EXCLUDE` modes.
//!   The DEFAULT frame (`RANGE UNBOUNDED PRECEDING … CURRENT ROW EXCLUDE NO OTHERS`) is
//!   routed to an O(p) running-aggregate path (one accumulator stepped across the sorted
//!   partition, a non-consuming `finalize` snapshot per peer group); any other frame
//!   rebuilds a fresh accumulator over that row's frame (O(frame) per row — correct and
//!   bounded, since `AggregateAccumulator` has no `remove`).
//! * RANKING (`row_number`, `rank`, `dense_rank`, `percent_rank`, `cume_dist`, `ntile`)
//!   and NAVIGATION (`lag`, `lead`, `first_value`, `last_value`, `nth_value`) kinds are
//!   implemented in [`ranking`] / [`navigation`], each over the same [`partition`]/[`frame`]
//!   API — [`compute_function`] dispatches one explicit arm per kind (no `_` catch-all).
//!
//! # Collation
//!
//! * `PARTITION BY` — partitions are keyed under each key's §7.1 collation, carried on
//!   `WindowFunc.partition_collations` (`row_key(&keys, &wf.partition_collations)`), so
//!   `PARTITION BY x COLLATE NOCASE` (or a NOCASE `x` column) folds text when forming
//!   partitions, matching real SQLite. The window `ORDER BY` likewise honors each
//!   `SortKey.collation`. (Both mirror `GROUP BY` in [`crate::ops::aggregate`].)
//!
//! # Aggregate-internal `ORDER BY` (ordered-set window aggregate)
//!
//! * An aggregate's OWN `ORDER BY` (`AggregateCall.order_by`, e.g.
//!   `group_concat(x ORDER BY y) OVER (…)`, lang_aggfunc.html) sets the order in which the
//!   aggregate folds its inputs. For a window aggregate the "inputs" at each row are that
//!   row's window FRAME, so `eval_ordered_frame` gathers each row's frame positions, sorts
//!   them by the aggregate's `ORDER BY` keys (via the shared `ops::sortkey` comparator, so
//!   the fold order matches the non-window `ops::aggregate` path), then folds. It applies to
//!   EVERY frame shape — the O(p) running default-frame path cannot both keep its running
//!   snapshot AND reorder, so a non-empty `call.order_by` always routes to the per-frame
//!   rebuild (O(frame · log frame) per row). The common no-`ORDER BY` case is untouched and
//!   keeps its fast running / general paths.

mod frame;
mod navigation;
mod partition;
mod ranking;

use std::collections::{HashMap, HashSet};

use minisqlite_expr::{
    eval, eval_with_subtype, truth, AggregateAccumulator, AggregateCall, EvalExpr, FnContext,
};
use minisqlite_plan::{
    FrameBound as PlanBound, FrameExclude as PlanExclude, FrameUnits as PlanUnits, Window,
    WindowFrame, WindowFunc, WindowFuncKind,
};
use minisqlite_types::{Collation, Error, Result, Row, Value};

use crate::context::EvalCtx;
use crate::env::Env;
use crate::keys::{cell_key, row_key, CellKey};
use crate::ops::sortkey;
use crate::runner::build_cursor;
use crate::runtime::Runtime;
use crate::RowCursor;

use frame::{Frame, FrameExclude, FrameUnits, ResolvedBound, ResolvedFrame};
use partition::Partition;

/// The `outer` a `finalize`/bound eval site reads: none. Finalization folds only the
/// values already `step`ped, and a frame bound is a constant expression, so an empty
/// outer is the correct one — mirrors `ops::aggregate`.
const NO_OUTER: &[Value] = &[];

/// Build a window operator over `w.input`. The input cursor is built eagerly (it needs
/// no [`Runtime`]); the per-partition computation happens lazily on the first
/// [`RowCursor::next_row`] pull, where the `rt` needed to evaluate expressions is
/// available.
pub(crate) fn window<'e>(
    w: &'e Window,
    env: Env<'e>,
    outer: &'e [Value],
) -> Result<Box<dyn RowCursor + 'e>> {
    let input = build_cursor(&w.input, env, outer)?;
    Ok(Box::new(WindowCursor { w, env, input, output: None }))
}

struct WindowCursor<'e> {
    w: &'e Window,
    env: Env<'e>,
    input: Box<dyn RowCursor + 'e>,
    /// The extended output rows, produced on the first pull. `None` until then.
    output: Option<std::vec::IntoIter<Row>>,
}

impl RowCursor for WindowCursor<'_> {
    fn next_row(&mut self, rt: &mut Runtime) -> Result<Option<Row>> {
        if self.output.is_none() {
            self.output = Some(self.build(rt)?);
        }
        Ok(self.output.as_mut().expect("output set on first pull").next())
    }
}

impl WindowCursor<'_> {
    /// Drain the input once, compute each function's appended column, and return the
    /// input rows extended with those values, in input order. The only materialization
    /// the operator performs.
    fn build(&mut self, rt: &mut Runtime) -> Result<std::vec::IntoIter<Row>> {
        let mut all_rows: Vec<Row> = Vec::new();
        while let Some(row) = self.input.next_row(rt)? {
            all_rows.push(row);
        }
        let n = all_rows.len();
        let k = self.w.functions.len();

        // `results[i]` holds the k appended values for input row i (row-major), filled by
        // index because each function visits rows grouped by partition, not in row order.
        // `Null` placeholders are overwritten as each function is computed.
        let mut results: Vec<Vec<Value>> = (0..n).map(|_| vec![Value::Null; k]).collect();
        for (j, wf) in self.w.functions.iter().enumerate() {
            compute_function(self.env, rt, wf, &all_rows, j, &mut results)?;
        }

        // Emit `row ++ its values`, in input order. `extend` moves the values out of
        // `results[i]` (no clone), and rows were never copied — only buffered.
        let out: Vec<Row> = all_rows
            .into_iter()
            .zip(results)
            .map(|(mut row, vals)| {
                row.extend(vals);
                row
            })
            .collect();
        Ok(out.into_iter())
    }
}

/// Compute one window function's appended value for every row (writing `results[i][j]`).
///
/// The dispatch has one arm per [`WindowFuncKind`], each routed to its implementation: the
/// AGGREGATE kind to `aggregate_over_frames`, the RANKING kinds to [`ranking::compute`], and
/// the NAVIGATION kinds to [`navigation`]. It is exhaustive (no `_` catch-all), so adding a
/// kind is a compile error here rather than a silently-absorbed variant.
fn compute_function(
    env: Env<'_>,
    rt: &mut Runtime,
    wf: &WindowFunc,
    all_rows: &[Row],
    j: usize,
    results: &mut [Vec<Value>],
) -> Result<()> {
    match &wf.kind {
        WindowFuncKind::Aggregate(call) => {
            aggregate_over_frames(env, rt, call, wf, all_rows, j, results)
        }
        WindowFuncKind::RowNumber => {
            ranking::compute(env, rt, wf, ranking::RankingKind::RowNumber, all_rows, j, results)
        }
        WindowFuncKind::Rank => {
            ranking::compute(env, rt, wf, ranking::RankingKind::Rank, all_rows, j, results)
        }
        WindowFuncKind::DenseRank => {
            ranking::compute(env, rt, wf, ranking::RankingKind::DenseRank, all_rows, j, results)
        }
        WindowFuncKind::PercentRank => {
            ranking::compute(env, rt, wf, ranking::RankingKind::PercentRank, all_rows, j, results)
        }
        WindowFuncKind::CumeDist => {
            ranking::compute(env, rt, wf, ranking::RankingKind::CumeDist, all_rows, j, results)
        }
        WindowFuncKind::Ntile(e) => {
            ranking::compute(env, rt, wf, ranking::RankingKind::Ntile(e), all_rows, j, results)
        }
        WindowFuncKind::Lag { expr, offset, default } => {
            navigation::lag_lead(env, rt, wf, expr, offset, default, false, all_rows, j, results)
        }
        WindowFuncKind::Lead { expr, offset, default } => {
            navigation::lag_lead(env, rt, wf, expr, offset, default, true, all_rows, j, results)
        }
        WindowFuncKind::FirstValue(expr) => {
            navigation::value_fn(env, rt, wf, expr, navigation::ValueKind::First, all_rows, j, results)
        }
        WindowFuncKind::LastValue(expr) => {
            navigation::value_fn(env, rt, wf, expr, navigation::ValueKind::Last, all_rows, j, results)
        }
        WindowFuncKind::NthValue { expr, n } => {
            navigation::value_fn(env, rt, wf, expr, navigation::ValueKind::Nth(n), all_rows, j, results)
        }
    }
}

/// The full AGGREGATE-over-frame kind: resolve the frame once, partition the input, then
/// per partition route the DEFAULT frame to the O(p) running path and any other frame to
/// the general per-row rebuild.
fn aggregate_over_frames(
    env: Env<'_>,
    rt: &mut Runtime,
    call: &AggregateCall,
    wf: &WindowFunc,
    all_rows: &[Row],
    j: usize,
    results: &mut [Vec<Value>],
) -> Result<()> {
    // Frame bounds are constant expressions, so resolve (evaluate + validate) them once
    // for the whole function, not per partition or per row.
    let resolved = resolve_frame(&wf.frame, env, rt)?;
    let partitions = partition_rows(env, rt, wf, all_rows)?;

    // Every input row lands in exactly one partition, so the partitions cover
    // `0..all_rows.len()` with no gaps — which is what guarantees every `results[i][j]`
    // placeholder is overwritten and no `Value::Null` can masquerade as a real value.
    debug_assert_eq!(
        partitions.iter().map(Partition::len).sum::<usize>(),
        all_rows.len(),
        "window partitions must cover every input row exactly once",
    );

    // An ordered-set aggregate window fn (`group_concat(x ORDER BY y) OVER (…)`) folds each
    // frame in the aggregate's OWN ORDER BY order, which the O(p) running default-frame path
    // cannot do (its snapshot is committed in window order), so a non-empty `call.order_by`
    // ALWAYS routes to the per-frame rebuild — for every frame, default or explicit.
    let ordered = !call.order_by.is_empty();
    for part in &partitions {
        if ordered {
            eval_ordered_frame(env, rt, call, wf, part, all_rows, &resolved, j, results)?;
        } else if resolved.is_default() {
            eval_default_frame(env, rt, call, part, all_rows, j, results)?;
        } else {
            eval_general_frame(env, rt, call, wf, part, all_rows, &resolved, j, results)?;
        }
    }
    Ok(())
}

/// Group input row INDICES into partitions by `wf.partition_by` (first-appearance order,
/// NULL keys sharing one partition per `row_key`'s contract, keyed under each key's
/// `wf.partition_collations` — §7.1, so a NOCASE key folds text), then build a
/// [`Partition`] per group — evaluating each row's `ORDER BY` keys, which the
/// [`Partition`] stable-sorts and groups into peers.
fn partition_rows(
    env: Env<'_>,
    rt: &mut Runtime,
    wf: &WindowFunc,
    all_rows: &[Row],
) -> Result<Vec<Partition>> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut index: HashMap<Vec<CellKey>, usize> = HashMap::new();
    for (i, row) in all_rows.iter().enumerate() {
        let mut key_vals = Vec::with_capacity(wf.partition_by.len());
        {
            let mut ctx = EvalCtx { rt, env, outer: row };
            for p in &wf.partition_by {
                key_vals.push(eval(p, row, &mut ctx)?);
            }
        }
        let key = row_key(&key_vals, &wf.partition_collations);
        match index.get(&key) {
            Some(&gi) => groups[gi].push(i),
            None => {
                let gi = groups.len();
                groups.push(vec![i]);
                index.insert(key, gi);
            }
        }
    }

    let mut parts = Vec::with_capacity(groups.len());
    for group in groups {
        let mut rows = Vec::with_capacity(group.len());
        for i in group {
            let row = &all_rows[i];
            let mut key_vals = Vec::with_capacity(wf.order_by.len());
            {
                let mut ctx = EvalCtx { rt, env, outer: row };
                for sk in &wf.order_by {
                    key_vals.push(eval(&sk.expr, row, &mut ctx)?);
                }
            }
            rows.push((i, key_vals));
        }
        parts.push(Partition::new(rows, &wf.order_by));
    }
    Ok(parts)
}

/// The DEFAULT frame (`RANGE UNBOUNDED PRECEDING … CURRENT ROW EXCLUDE NO OTHERS`): one
/// running accumulator stepped across the sorted partition, taking a non-consuming
/// `finalize` snapshot at each peer-group end that every row of the group shares. O(p)
/// steps + O(groups) finalizes — not an O(p²) rebuild. Handles the no-`ORDER BY` case
/// too (the whole partition is one peer group, so all rows get the whole-partition
/// aggregate).
fn eval_default_frame(
    env: Env<'_>,
    rt: &mut Runtime,
    call: &AggregateCall,
    part: &Partition,
    all_rows: &[Row],
    j: usize,
    results: &mut [Vec<Value>],
) -> Result<()> {
    let mut acc = new_window_accumulator(call);
    let mut seen = distinct_set(call);
    for g in 0..part.num_groups() {
        let (lo, hi) = part.group_bounds(g);
        for pos in lo..hi {
            step_row(env, rt, call, &all_rows[part.input_index(pos)], acc.as_mut(), &mut seen)?;
        }
        let v = finalize_snapshot(env, rt, acc.as_mut())?;
        for pos in lo..hi {
            results[part.input_index(pos)][j] = v.clone();
        }
    }
    Ok(())
}

/// A GENERAL (non-default) frame: for each row, rebuild a fresh accumulator over exactly
/// that row's frame positions (in window order) and finalize. O(frame) per row — correct
/// and bounded; `AggregateAccumulator` has no `remove`, so a sliding frame rebuilds
/// rather than incrementally sliding.
fn eval_general_frame(
    env: Env<'_>,
    rt: &mut Runtime,
    call: &AggregateCall,
    wf: &WindowFunc,
    part: &Partition,
    all_rows: &[Row],
    resolved: &ResolvedFrame,
    j: usize,
    results: &mut [Vec<Value>],
) -> Result<()> {
    for c in 0..part.len() {
        let fr: Frame = frame::frame_positions(part, c, resolved, &wf.order_by);
        let mut acc = new_window_accumulator(call);
        let mut seen = distinct_set(call);
        for pos in fr.positions() {
            step_row(env, rt, call, &all_rows[part.input_index(pos)], acc.as_mut(), &mut seen)?;
        }
        let v = finalize_snapshot(env, rt, acc.as_mut())?;
        results[part.input_index(c)][j] = v;
    }
    Ok(())
}

/// An ORDERED-SET aggregate window fn (`group_concat(x ORDER BY y) OVER (…)`, and — per
/// `bind_window_kind` — every `sum/count/avg/… (x ORDER BY y) OVER (…)`): each row's frame is
/// folded in the aggregate's OWN `ORDER BY` order, which the running default-frame path cannot
/// do (it commits its snapshot in window order), so a non-empty `call.order_by` routes here for
/// EVERY frame shape. `step_row` re-applies `FILTER`, so a filtered-out frame row is sorted but
/// skipped at fold time — the surviving order is unchanged, as in the non-window path.
///
/// Two costs the naive per-(frame, position) rebuild pays needlessly are avoided:
///
///   1. The aggregate's `ORDER BY` keys for a row are POSITION-INTRINSIC — they do not depend on
///      the frame — so they are evaluated ONCE per partition row (O(p)) into `order_keys`, not
///      re-evaluated for every (frame, position) pair (which was O(p²) evals). Mirrors how
///      [`Partition`] precomputes the WINDOW `ORDER BY` keys.
///   2. On the DEFAULT frame (`RANGE UNBOUNDED PRECEDING … CURRENT ROW EXCLUDE NO OTHERS`) every
///      row's frame is `[0, end-of-its-peer-group)`, so all rows in a peer group share one frame
///      and one result: fold ONCE per peer group and BROADCAST (the shape of `eval_default_frame`)
///      rather than re-sort+fold per row. The ordered fold still re-folds the growing `[0, hi)` per
///      peer group (the accumulator can neither reorder nor remove), but collapsing per-row to
///      per-peer-group turns the whole-partition case `sum(x ORDER BY y) OVER (PARTITION BY k)`
///      (one peer group) from O(p²·log p) to O(p·log p). Non-default frames (ROWS/GROUPS, general
///      RANGE, any EXCLUDE) keep the per-row fold — but still over the once-computed `order_keys`.
#[allow(clippy::too_many_arguments)]
fn eval_ordered_frame(
    env: Env<'_>,
    rt: &mut Runtime,
    call: &AggregateCall,
    wf: &WindowFunc,
    part: &Partition,
    all_rows: &[Row],
    resolved: &ResolvedFrame,
    j: usize,
    results: &mut [Vec<Value>],
) -> Result<()> {
    // (1) Evaluate every partition row's aggregate `ORDER BY` keys ONCE, indexed by ordered
    // position (the same indexing `Partition::order_key` uses for the window ORDER BY keys).
    let mut order_keys: Vec<Vec<Value>> = Vec::with_capacity(part.len());
    for pos in 0..part.len() {
        let row = &all_rows[part.input_index(pos)];
        let mut vals = Vec::with_capacity(call.order_by.len());
        {
            let mut ctx = EvalCtx { rt, env, outer: row };
            for sk in &call.order_by {
                vals.push(eval(&sk.expr, row, &mut ctx)?);
            }
        }
        order_keys.push(vals);
    }

    // (2) Default frame: one sorted fold per peer group, broadcast to every row of the group.
    if resolved.is_default() {
        for g in 0..part.num_groups() {
            let (lo, hi) = part.group_bounds(g);
            let v = fold_ordered(env, rt, call, part, all_rows, &order_keys, 0..hi)?;
            for pos in lo..hi {
                results[part.input_index(pos)][j] = v.clone();
            }
        }
        return Ok(());
    }

    // General frames: each row's frame may differ, so fold per row (over the once-computed keys).
    for c in 0..part.len() {
        let fr: Frame = frame::frame_positions(part, c, resolved, &wf.order_by);
        let v = fold_ordered(env, rt, call, part, all_rows, &order_keys, fr.positions())?;
        results[part.input_index(c)][j] = v;
    }
    Ok(())
}

/// Fold one frame of an ordered-set aggregate: STABLE-sort the frame's ordered `positions` by
/// the aggregate's OWN `ORDER BY` (via the precomputed `order_keys`, so equal keys keep window
/// order), then step the corresponding input rows in that order and finalize. Shared by the
/// default-frame peer-group broadcast and the general per-row path so the sort/fold lives once.
fn fold_ordered(
    env: Env<'_>,
    rt: &mut Runtime,
    call: &AggregateCall,
    part: &Partition,
    all_rows: &[Row],
    order_keys: &[Vec<Value>],
    positions: impl Iterator<Item = usize>,
) -> Result<Value> {
    let mut buf: Vec<usize> = positions.collect();
    buf.sort_by(|&a, &b| sortkey::compare_sort_keys(&order_keys[a], &order_keys[b], &call.order_by));
    let mut acc = new_window_accumulator(call);
    let mut seen = distinct_set(call);
    for &pos in &buf {
        step_row(env, rt, call, &all_rows[part.input_index(pos)], acc.as_mut(), &mut seen)?;
    }
    finalize_snapshot(env, rt, acc.as_mut())
}

/// Resolve a plan [`WindowFrame`] into a [`ResolvedFrame`]: map the unit / exclude enums
/// and evaluate + validate the `PRECEDING`/`FOLLOWING` bound expressions (constant, so
/// evaluated against an empty row) to non-negative numeric offsets.
fn resolve_frame(spec: &WindowFrame, env: Env<'_>, rt: &mut Runtime) -> Result<ResolvedFrame> {
    let units = match spec.units {
        PlanUnits::Rows => FrameUnits::Rows,
        PlanUnits::Range => FrameUnits::Range,
        PlanUnits::Groups => FrameUnits::Groups,
    };
    let exclude = match spec.exclude {
        PlanExclude::NoOthers => FrameExclude::NoOthers,
        PlanExclude::CurrentRow => FrameExclude::CurrentRow,
        PlanExclude::Group => FrameExclude::Group,
        PlanExclude::Ties => FrameExclude::Ties,
    };
    let start = resolve_bound(&spec.start, units, env, rt, true)?;
    let end = resolve_bound(&spec.end, units, env, rt, false)?;
    Ok(ResolvedFrame { units, start, end, exclude })
}

/// Resolve one plan [`FrameBound`](PlanBound) to a [`ResolvedBound`].
fn resolve_bound(
    b: &PlanBound,
    units: FrameUnits,
    env: Env<'_>,
    rt: &mut Runtime,
    is_start: bool,
) -> Result<ResolvedBound> {
    Ok(match b {
        PlanBound::UnboundedPreceding => ResolvedBound::UnboundedPreceding,
        PlanBound::UnboundedFollowing => ResolvedBound::UnboundedFollowing,
        PlanBound::CurrentRow => ResolvedBound::CurrentRow,
        PlanBound::Preceding(e) => ResolvedBound::Preceding(eval_offset(e, units, env, rt, is_start)?),
        PlanBound::Following(e) => ResolvedBound::Following(eval_offset(e, units, env, rt, is_start)?),
    })
}

/// Evaluate a frame bound expression (constant) and validate it as a non-negative offset:
/// an integer for `ROWS`/`GROUPS`, any non-negative number for `RANGE`.
fn eval_offset(
    e: &EvalExpr,
    units: FrameUnits,
    env: Env<'_>,
    rt: &mut Runtime,
    is_start: bool,
) -> Result<Value> {
    let v = {
        let mut ctx = EvalCtx { rt, env, outer: NO_OUTER };
        eval(e, NO_OUTER, &mut ctx)?
    };
    let which = if is_start { "starting" } else { "ending" };
    match units {
        FrameUnits::Rows | FrameUnits::Groups => match &v {
            Value::Integer(n) if *n >= 0 => Ok(v),
            // A whole-valued real is accepted as its integer (e.g. `1.0` ⇒ `1`).
            Value::Real(r) if r.is_finite() && *r >= 0.0 && r.fract() == 0.0 => {
                Ok(Value::Integer(*r as i64))
            }
            _ => Err(Error::sql(format!("frame {which} offset must be a non-negative integer"))),
        },
        FrameUnits::Range => match &v {
            Value::Integer(n) if *n >= 0 => Ok(v),
            Value::Real(r) if r.is_finite() && *r >= 0.0 => Ok(v),
            _ => Err(Error::sql(format!("frame {which} offset must be a non-negative number"))),
        },
    }
}

/// The per-window-function DISTINCT dedup set, present iff the call is `DISTINCT`. (SQLite
/// disallows `DISTINCT` window functions, so a well-formed plan never sets it; handled
/// here for consistency with `ops::aggregate` rather than dropped.)
fn distinct_set(call: &AggregateCall) -> Option<HashSet<Vec<CellKey>>> {
    if call.distinct {
        Some(HashSet::new())
    } else {
        None
    }
}

/// Build a fresh accumulator for a window aggregate, keyed by the FIRST argument's §7.1
/// collation (BINARY if absent) — the same rule the plain-aggregate operator uses in
/// `ops::aggregate`. Only order-sensitive aggregates (`min`/`max`, which are valid as
/// window functions: `max(b) OVER (…)`) read it; the rest ignore it. Resolved from
/// `AggregateCall.arg_collations`, which the binder fills for window aggregates too.
fn new_window_accumulator(call: &AggregateCall) -> Box<dyn AggregateAccumulator> {
    call.func.new_accumulator(call.arg_collations.first().copied().unwrap_or(Collation::Binary))
}

/// Fold one input `row` into `acc`, applying the call's modifiers the way `ops::aggregate`
/// does: skip on `FILTER` (unless TRUE), evaluate the arguments, dedup on `DISTINCT` (a
/// defensive, currently-UNREACHABLE branch — window DISTINCT is rejected at bind time; see
/// below), then `step`. Each `EvalCtx` is scoped so its `rt` reborrow ends before the next
/// one (the reborrow rule the streaming operators share).
///
/// The argument evaluation captures each argument's ephemeral JSON value-subtype and
/// PUBLISHES it right before `step` (json1.html §3.4), so a windowed JSON aggregate
/// embeds a subtyped `value` operand — `json_group_array(json(x)) OVER (…)` — exactly as
/// the plain (`ops::aggregate`) path does. `arg_subtypes` stays empty (no allocation)
/// until an argument actually carries a subtype; an empty publish clears the channel so
/// every `arg_subtype(i)` reads 0.
fn step_row(
    env: Env<'_>,
    rt: &mut Runtime,
    call: &AggregateCall,
    row: &Row,
    acc: &mut dyn AggregateAccumulator,
    seen: &mut Option<HashSet<Vec<CellKey>>>,
) -> Result<()> {
    // FILTER (WHERE …): three-valued — NULL/FALSE both skip.
    if let Some(filter) = &call.filter {
        let mut ctx = EvalCtx { rt, env, outer: row };
        if truth(&eval(filter, row, &mut ctx)?) != Some(true) {
            return Ok(());
        }
    }

    let mut args = Vec::with_capacity(call.args.len());
    let mut arg_subtypes: Vec<u8> = Vec::new();
    let mut any_subtype = false;
    {
        let mut ctx = EvalCtx { rt, env, outer: row };
        for (i, arg) in call.args.iter().enumerate() {
            let (v, st) = eval_with_subtype(arg, row, &mut ctx)?;
            if st != 0 && !any_subtype {
                any_subtype = true;
                arg_subtypes = vec![0u8; i];
            }
            if any_subtype {
                arg_subtypes.push(st);
            }
            args.push(v);
        }
    }

    if let Some(seen) = seen.as_mut() {
        // Effectively unreachable dead defensive code: the binder rejects DISTINCT on every
        // window function (`minisqlite-plan/src/compile/window.rs`, pinned by
        // `distinct_window_aggregate_is_rejected`), and a window `AggregateCall` is always
        // built with `distinct: false`, so `distinct_set` returns `None` and this `seen`
        // branch never runs. The `cell_key` folds under BINARY here — a dead default this path
        // cannot exercise, not a collation choice the code can ever make.
        let dkey: Vec<CellKey> = args.iter().map(|v| cell_key(v, Collation::Binary)).collect();
        if !seen.insert(dkey) {
            return Ok(());
        }
    }

    let mut ctx = EvalCtx { rt, env, outer: row };
    ctx.set_arg_subtypes(&arg_subtypes);
    acc.step(&args, &mut ctx)
}

/// Read the accumulator's current result without consuming it (a running snapshot).
/// Finalization reads only folded state, so the outer is empty.
///
/// The result subtype is cleared before finalize and read back after, matching the
/// subtype protocol (json1.html §3.4). A windowed value is written into the output row,
/// which carries no subtype, so the mark is discarded here — the same deferred
/// outer-nesting hop as the plain aggregate path.
fn finalize_snapshot(
    env: Env<'_>,
    rt: &mut Runtime,
    acc: &mut dyn AggregateAccumulator,
) -> Result<Value> {
    let mut ctx = EvalCtx { rt, env, outer: NO_OUTER };
    ctx.set_result_subtype(0);
    let v = acc.finalize(&mut ctx)?;
    let _result_subtype = ctx.take_result_subtype();
    Ok(v)
}
