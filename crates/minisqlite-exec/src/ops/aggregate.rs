//! `Aggregate` — GROUP BY / aggregation. Emits one row per group,
//! `[group_key_0..group_key_{G-1}, agg_result_0..agg_result_{A-1}]` (width `G+A`).
//!
//! This is an allowed materialization: on the first pull it drains the whole input
//! and builds the group table, then streams the finalized group rows. The buffer is
//! bounded by the real group cardinality (one [`GroupState`] per group) plus, for
//! aggregates that carry `DISTINCT` or aggregate `ORDER BY`, a per-(group,aggregate)
//! side buffer bounded by that group's rows — never a copy of the whole input.
//!
//! Groups are streamed in FIRST-APPEARANCE order (a `Vec` of states plus a
//! `HashMap` from the row key to its index) — a deterministic scan order, NOT the
//! sorted order SQLite returns for an unordered `GROUP BY`. Real SQLite groups via a
//! sort, so an unordered `GROUP BY` comes back ASCENDING by group key; this operator
//! does NOT reorder for that. Instead the planner inserts a plan-side `Sort` on the
//! group-key columns above this operator on the non-window aggregate path (see
//! `minisqlite_plan::compile::aggregate` — the implicit group-order rule, and its
//! documented window-path residual where that Sort is intentionally not added), which
//! presents the groups sorted by key. Fixing the order there — not here — keeps the
//! ordering out of the hot per-row aggregation loop and mirrors the compound-SELECT
//! implicit-sort fix (`compile::compound`).
//!
//! `DISTINCT`, `FILTER (WHERE …)`, and aggregate `ORDER BY` are the OPERATOR's job,
//! applied around each accumulator; the accumulator itself only folds the values it
//! is handed via `step`/`finalize`. NULL group keys group together (that is
//! [`row_key`]'s contract). An empty input with an empty `GROUP BY` still yields
//! exactly one row (each aggregate finalized over zero steps, e.g. `count → 0`,
//! `sum → NULL`); an empty input with a non-empty `GROUP BY` yields zero rows.
//!
//! # Ephemeral JSON value-subtype (json1.html §3.4)
//!
//! An aggregate argument is evaluated here (not through the evaluator's `Func` arm), so
//! this driver is what threads the subtype channel for aggregates: it captures each
//! argument's value-subtype with [`eval_with_subtype`], PUBLISHES the per-argument
//! subtypes ([`FnContext::set_arg_subtypes`]) immediately before every `step` (including
//! the buffered replay for aggregate `ORDER BY`), and reads back the subtype `finalize`
//! marked. That makes `json_group_array(json('[1]'))` embed `[[1]]` (a JSON aggregate's
//! `step` sees the operand's subtype), the direct case this fixes.
//!
//! DEFERRED — outer nesting: the finalized value is pushed into the result [`Row`] here,
//! and the subtype is dropped at that `Row` boundary (the subtype never touches a
//! `Value`/`Row`, by design). So a SEPARATE enclosing projection —
//! `json_array(json_group_array(json('[1]')))` — still re-quotes, because it reads the
//! aggregate output back as a plain `Column`. Carrying the subtype across the
//! aggregate-output-row → projection seam needs a new channel touching the shared
//! evaluator `Column` path and the plan; it is a documented follow-up, not done here.

use std::collections::{HashMap, HashSet};

use minisqlite_expr::{
    eval, eval_with_subtype, truth, AggregateAccumulator, AggregateCall, FnContext,
};
use minisqlite_plan::Aggregate;
use minisqlite_types::{extremum_wins, Collation, Result, Row, Value};

use crate::context::EvalCtx;
use crate::env::Env;
use crate::keys::{cell_key, row_key, CellKey};
use crate::ops::sortkey;
use crate::runner::build_cursor;
use crate::runtime::Runtime;
use crate::RowCursor;

/// The `outer` a `FINALIZE`/`step` eval site reads: none. Aggregate accumulators
/// receive an `&mut dyn FnContext`, which cannot read outer columns, so an empty
/// slice is the correct (and only meaningful) outer there.
const NO_OUTER: &[Value] = &[];

/// Build an aggregation of `a.input`. The input cursor is built eagerly (it needs no
/// [`Runtime`]); grouping happens lazily on the first [`RowCursor::next_row`] pull.
pub(crate) fn aggregate<'e>(
    a: &'e Aggregate,
    env: Env<'e>,
    outer: &'e [Value],
) -> Result<Box<dyn RowCursor + 'e>> {
    let input = build_cursor(&a.input, env, outer)?;
    Ok(Box::new(AggregateCursor { a, env, input, output: None }))
}

struct AggregateCursor<'e> {
    a: &'e Aggregate,
    env: Env<'e>,
    input: Box<dyn RowCursor + 'e>,
    /// The finalized group rows, produced on the first pull. `None` until then.
    output: Option<std::vec::IntoIter<Row>>,
}

impl RowCursor for AggregateCursor<'_> {
    fn next_row(&mut self, rt: &mut Runtime) -> Result<Option<Row>> {
        if self.output.is_none() {
            self.output = Some(self.build(rt)?);
        }
        Ok(self.output.as_mut().expect("output set on first pull").next())
    }
}

impl AggregateCursor<'_> {
    /// Drain the input into groups, then finalize each group (applying `HAVING`) into
    /// the ordered output rows. The only materialization the operator performs.
    fn build(&mut self, rt: &mut Runtime) -> Result<std::vec::IntoIter<Row>> {
        // Fast path: when the result depends ONLY on the input row count — a `count(*)`
        // with no GROUP BY / HAVING / bare columns and no per-row modifiers — ask the
        // input for its row count instead of pulling and discarding a row each time. A
        // join answers by summing its bucket sizes (see `RowCursor::count_rows`), so
        // `count(*)` over a high-cardinality join no longer builds a combined row per
        // pair. The emitted row is byte-identical to the row-by-row drain.
        if is_row_count_only(self.a) {
            return self.build_row_count_only(rt);
        }
        let mut groups: Vec<GroupState> = Vec::new();
        // Maps a group's canonical row key to its index in `groups`. First appearance
        // decides the index, so iterating `groups` later replays that order.
        let mut index: HashMap<Vec<CellKey>, usize> = HashMap::new();

        while let Some(row) = self.input.next_row(rt)? {
            ingest_row(self.a, self.env, rt, &row, &mut groups, &mut index)?;
        }

        // An empty input with an implicit (empty) GROUP BY still emits exactly one
        // row: the aggregates finalized over zero rows. A non-empty GROUP BY over an
        // empty input emits nothing (no group ever appeared).
        if self.a.group_by.is_empty() && groups.is_empty() {
            groups.push(GroupState::new(Vec::new(), &self.a.aggregates));
        }

        let mut out: Vec<Row> = Vec::new();
        for mut group in groups {
            // The group key values are emitted verbatim, first; then one finalized
            // result per aggregate.
            let mut result = std::mem::take(&mut group.key_vals);
            for (j, ac) in self.a.aggregates.iter().enumerate() {
                let agg = &mut group.aggs[j];
                // Aggregate ORDER BY: rows were buffered during ingest; sort them now
                // (stable, so equal keys keep first-appearance order) and step in that
                // order before finalizing.
                if let Some(mut buf) = agg.ordered.take() {
                    buf.sort_by(|x, y| sortkey::compare_sort_keys(&x.1, &y.1, &ac.order_by));
                    for (args, _, subs) in &buf {
                        let mut ctx = EvalCtx { rt, env: self.env, outer: NO_OUTER };
                        // Republish this buffered row's per-argument subtypes before the
                        // replayed step, so a JSON aggregate embeds a subtyped operand in
                        // the ORDER BY path exactly as in the immediate-fold path.
                        ctx.set_arg_subtypes(subs);
                        agg.acc.step(args, &mut ctx)?;
                    }
                }
                let v = {
                    let mut ctx = EvalCtx { rt, env: self.env, outer: NO_OUTER };
                    // Clear any stale result subtype first, so a non-JSON aggregate (which
                    // never sets one) reads back 0 rather than a previous group's JSON
                    // finalize mark; then finalize and read the subtype this aggregate
                    // marked on its result (JSON_SUBTYPE for json_group_*).
                    ctx.set_result_subtype(0);
                    let v = agg.acc.finalize(&mut ctx)?;
                    // The finalize subtype is read back per the subtype protocol. It is
                    // discarded for now: the result is stored into the output row below,
                    // which carries no subtype, so an enclosing projection cannot yet see
                    // it (the deferred outer-nesting hop — see the module header).
                    let _result_subtype = ctx.take_result_subtype();
                    v
                };
                result.push(v);
            }

            // Engineering-rigor guard (debug-only): the bare-column capture re-derives the
            // extremum in ingest `(b')` independently of the accumulator that produced the
            // reported min/max result. They MUST agree — the bare columns are meant to come
            // from the row that produced the REPORTED extremum — so assert the captured
            // extremum equals the finalized min/max here, where both are in hand. This
            // turns a future silent divergence (e.g. the accumulator becoming
            // collation-aware while the capture stays BINARY) into a failing test rather
            // than a wrong row travelling far. A group whose marked column was all-NULL
            // captured nothing and finalizes to NULL; both read as NULL, hence equal.
            #[cfg(debug_assertions)]
            if let Some(m) = &self.a.minmax_bare {
                let finalized = &result[self.a.group_by.len() + m.agg_index];
                let captured = group.capture_best.as_ref().unwrap_or(&Value::Null);
                debug_assert_eq!(
                    minisqlite_types::compare_values(captured, finalized, Collation::Binary),
                    std::cmp::Ordering::Equal,
                    "single-min/max capture diverged from the finalized extremum: \
                     captured {captured:?} vs finalized {finalized:?}",
                );
            }

            // Single-min/max special case (lang_select.html §2.5): append the bare
            // columns captured from this group's extremum row, extending the emitted row
            // to `[keys.., results.., captured..]` — the layout the projection (and any
            // bare column in HAVING) binds against. A group whose marked column was
            // entirely NULL captured no row (min/max is NULL with no "row for which it
            // was computed"): its bare columns emit as NULL. SQLite leaves this all-NULL
            // case unspecified; NULL is this engine's defined, deterministic choice.
            if let Some(m) = &self.a.minmax_bare {
                match group.captured.take() {
                    Some(vals) => result.extend(vals),
                    None => result.extend(std::iter::repeat(Value::Null).take(m.captured_regs.len())),
                }
            } else if let Some(regs) = &self.a.bare_arbitrary {
                // General §2.5 bare columns: append the values captured from this group's
                // first row (ingest). The `None` case is only the synthetic empty-input
                // implicit group (created post-drain with no capture), whose bare columns —
                // like SQLite's — emit as NULL.
                match group.captured.take() {
                    Some(vals) => result.extend(vals),
                    None => result.extend(std::iter::repeat(Value::Null).take(regs.len())),
                }
            }

            // HAVING is evaluated against the emitted `[keys.., results.., captured..]` row.
            if let Some(having) = &self.a.having {
                let keep = {
                    let mut ctx = EvalCtx { rt, env: self.env, outer: &result };
                    truth(&eval(having, &result, &mut ctx)?) == Some(true)
                };
                if !keep {
                    continue;
                }
            }
            out.push(result);
        }
        Ok(out.into_iter())
    }

    /// The [`is_row_count_only`] fast path: the whole aggregation is a single implicit
    /// group whose result depends only on how many rows the input has. Ask the input for
    /// that count ([`RowCursor::count_rows`] — a join sums bucket sizes; the default
    /// drains) and replay exactly that many empty `step`s into each accumulator (the same
    /// `&[]` fold `ingest_row` makes for a zero-argument call), then finalize. Emits
    /// exactly one row, identical to the row-by-row drain but without materializing an
    /// input row per count.
    fn build_row_count_only(&mut self, rt: &mut Runtime) -> Result<std::vec::IntoIter<Row>> {
        let n = self.input.count_rows(rt)?;
        let mut result: Row = Vec::with_capacity(self.a.aggregates.len());
        for ac in &self.a.aggregates {
            let coll = ac.arg_collations.first().copied().unwrap_or(Collation::Binary);
            let mut acc = ac.func.new_accumulator(coll);
            {
                let mut ctx = EvalCtx { rt, env: self.env, outer: NO_OUTER };
                // A zero-argument call folds an empty argument list once per row. Clear the
                // subtype channel once (it never changes across these steps), then apply
                // all `n` identical empty-argument steps at once: `count` overrides
                // `step_many` to add `n` in O(1) (the default replays `step` `n` times, so
                // any accumulator stays correct). The finalized value is identical to the
                // per-row fold — this collapses a per-pair virtual call on a
                // high-cardinality join to a single closed-form step.
                ctx.set_arg_subtypes(&[]);
                acc.step_many(n, &[], &mut ctx)?;
            }
            let v = {
                let mut ctx = EvalCtx { rt, env: self.env, outer: NO_OUTER };
                ctx.set_result_subtype(0);
                let v = acc.finalize(&mut ctx)?;
                let _ = ctx.take_result_subtype();
                v
            };
            result.push(v);
        }
        Ok(vec![result].into_iter())
    }
}

/// True when this aggregation's result depends ONLY on the number of input rows, never
/// on any row's values — the precondition for [`AggregateCursor::build_row_count_only`].
/// It requires: no GROUP BY (a single implicit group), no HAVING, no §2.5 bare-column
/// capture (`minmax_bare` / `bare_arbitrary`), and every aggregate call is a bare
/// ZERO-ARGUMENT call with no FILTER / DISTINCT / aggregate ORDER BY.
///
/// `count` is the only zero-argument aggregate (`count(*)` lowers to a zero-arg `count`),
/// and its finalized value is exactly the step count. More generally a zero-argument
/// accumulator only ever sees an empty argument list, so whatever it computes is a
/// function of the step COUNT alone — hence replaying `n` empty steps reproduces the
/// exact result. FILTER, DISTINCT, aggregate ORDER BY, bare columns, and HAVING all read
/// row values, so any of them disqualifies the fast path (and it falls back to the
/// row-by-row drain).
fn is_row_count_only(a: &Aggregate) -> bool {
    // Destructured with no `..` so a NEW field on `Aggregate` or `AggregateCall` is a
    // COMPILE error here — this gate is a load-bearing correctness gate (a
    // mis-classification returns a wrong `count(*)`), so a new field must be considered,
    // not silently absorbed. `input` is what we count and `*_collations` are inert with an
    // empty GROUP BY / bare-count call, so both are intentionally ignored.
    let Aggregate {
        input: _,
        group_by,
        group_collations: _,
        aggregates,
        having,
        minmax_bare,
        bare_arbitrary,
    } = a;
    group_by.is_empty()
        && having.is_none()
        && minmax_bare.is_none()
        && bare_arbitrary.is_none()
        && !aggregates.is_empty()
        && aggregates.iter().all(|ac| {
            let AggregateCall { func, distinct, args, filter, order_by, arg_collations: _ } = ac;
            // FAIL-CLOSED on the aggregate identity: only a function explicitly marked as a
            // pure row count (`count`) may take this path, so a future zero-argument
            // aggregate whose result is NOT a function of the row count alone defaults to
            // ineligible rather than silently returning a wrong answer. It must also be the
            // bare `count(*)` form — an argument (`count(X)`) skips NULLs, and FILTER /
            // DISTINCT / aggregate ORDER BY all read row values.
            func.is_row_count()
                && args.is_empty()
                && filter.is_none()
                && !*distinct
                && order_by.is_empty()
        })
}

/// Fold one input `row` into its group (creating the group on first appearance), then
/// feed each aggregate: skip on `FILTER`, dedup on `DISTINCT`, buffer for aggregate
/// `ORDER BY`, else step immediately.
///
/// Each `EvalCtx` is created in its own scope so the `rt` reborrow it holds ends
/// before the next eval or the next input pull (the reborrow rule the streaming
/// operators all follow).
fn ingest_row(
    a: &Aggregate,
    env: Env<'_>,
    rt: &mut Runtime,
    row: &Row,
    groups: &mut Vec<GroupState>,
    index: &mut HashMap<Vec<CellKey>, usize>,
) -> Result<()> {
    // Group key: evaluate each GROUP BY expr against the input row, then canonicalize
    // under the group collations so NULLs and numeric-equal values group together.
    let mut key_vals = Vec::with_capacity(a.group_by.len());
    {
        let mut ctx = EvalCtx { rt, env, outer: row };
        for g in &a.group_by {
            key_vals.push(eval(g, row, &mut ctx)?);
        }
    }
    let key = row_key(&key_vals, &a.group_collations);
    let idx = match index.get(&key) {
        Some(&i) => i,
        None => {
            let i = groups.len();
            let mut gs = GroupState::new(key_vals, &a.aggregates);
            // General §2.5 bare-column capture (see the plan's `bare_arbitrary` doc): a bare
            // column takes an ARBITRARY input row of the group; capture it from the FIRST row
            // (this branch runs exactly once per group, on first appearance). Deterministic,
            // and identical to any row for a functionally-dependent bare column. Mutually
            // exclusive with `minmax_bare`, which captures the extremum row in step (b')
            // instead — so at most one of the two ever writes `captured`.
            if let Some(regs) = &a.bare_arbitrary {
                gs.captured = Some(regs.iter().map(|&r| row[r].clone()).collect());
            }
            groups.push(gs);
            index.insert(key, i);
            i
        }
    };

    for (j, ac) in a.aggregates.iter().enumerate() {
        // (a) FILTER (WHERE …): skip this row for this aggregate unless the predicate
        // is TRUE (three-valued: NULL/FALSE both skip).
        if let Some(filter) = &ac.filter {
            let pass = {
                let mut ctx = EvalCtx { rt, env, outer: row };
                truth(&eval(filter, row, &mut ctx)?) == Some(true)
            };
            if !pass {
                continue;
            }
        }

        // (b) Evaluate the argument expressions against the input row, capturing each
        // argument's ephemeral JSON value-subtype (json1.html §3.4) so a JSON aggregate
        // embeds a `value` operand produced by another JSON function instead of quoting
        // it. `arg_subtypes` stays empty (no allocation) until some argument actually
        // carries a subtype — the common non-JSON case — mirroring the lazy capture in
        // `minisqlite_expr::eval_with_subtype`; an empty vec published before `step`
        // clears the channel so every `arg_subtype(i)` reads 0.
        let mut args = Vec::with_capacity(ac.args.len());
        let mut arg_subtypes: Vec<u8> = Vec::new();
        let mut any_subtype = false;
        {
            let mut ctx = EvalCtx { rt, env, outer: row };
            for (i, arg) in ac.args.iter().enumerate() {
                let (v, st) = eval_with_subtype(arg, row, &mut ctx)?;
                if st != 0 && !any_subtype {
                    // First subtype seen: back-fill zeros so index `i` lines up with arg `i`.
                    any_subtype = true;
                    arg_subtypes = vec![0u8; i];
                }
                if any_subtype {
                    arg_subtypes.push(st);
                }
                args.push(v);
            }
        }

        // (b') Single-min/max bare-column capture (lang_select.html §2.5): when this is
        // the marked min/max aggregate, remember the bare columns of the row that
        // produced the running extremum. Done here — after FILTER and arg evaluation,
        // before the `ag` borrow below — so a FILTERed row never captures and the
        // `&mut groups[idx]` it needs does not overlap the `ag` borrow. Extremum
        // selection goes through the SHARED `extremum_wins` predicate — the exact same one
        // the built-in min/max accumulator (`minisqlite-functions` `agg/minmax.rs`) uses —
        // so the captured row is always the row that produced the reported min/max: a
        // strict win, ties keeping the first-seen row, NULLs skipped. It keys by the SAME
        // collation the accumulator uses (`ac.arg_collations[0]`, i.e. the min/max
        // argument's §7.1 collation, BINARY if absent) — the two MUST move in lockstep or
        // a NOCASE/RTRIM column would capture a row chosen under a different order than the
        // reported min/max (see `extremum_wins`). `ac` here is the marked min/max call
        // (`j == m.agg_index`). A DISTINCT duplicate is never a strict win, so capturing
        // before the DISTINCT skip below is safe. The marked aggregate is a one-argument
        // min/max, so `args[0]` is its value; the row is cloned only when the extremum
        // actually advances.
        if let Some(m) = &a.minmax_bare {
            if j == m.agg_index && !args[0].is_null() {
                let coll = ac.arg_collations.first().copied().unwrap_or(Collation::Binary);
                let group = &mut groups[idx];
                let win = match &group.capture_best {
                    None => true,
                    Some(cur) => extremum_wins(cur, &args[0], m.is_max, coll),
                };
                if win {
                    group.capture_best = Some(args[0].clone());
                    // Reuse the existing capture buffer (clear + re-push) rather than a
                    // fresh `.collect()` per win: on monotonic input (e.g. `max` over
                    // ascending values) every row is a strict win, so a fresh Vec per row
                    // would churn the allocator; the length is always `captured_regs.len()`.
                    let vals = m.captured_regs.iter().map(|&r| row[r].clone());
                    match &mut group.captured {
                        Some(buf) => {
                            buf.clear();
                            buf.extend(vals);
                        }
                        slot @ None => *slot = Some(vals.collect()),
                    }
                }
            }
        }

        // Bind this (group, aggregate) slot once. The DISTINCT/step/buffer steps below
        // otherwise re-index `groups[idx].aggs[j]` up to three times per aggregate per
        // input row — a repeated double bounds-check on the hottest loop. The `&mut`
        // into `groups` is disjoint from the `rt` reborrow each `EvalCtx` holds, so the
        // eval sites still compile alongside it.
        let ag = &mut groups[idx].aggs[j];

        // (c) DISTINCT: dedup the argument tuple within this (group, aggregate). The
        // dedup key folds numeric-equal values (via `cell_key`) AND folds text under each
        // argument's collation — SQLite compares aggregate DISTINCT arguments with `=`, so
        // an explicit `COLLATE` / column collation applies (lang_select.html §2.6,
        // datatype3.html §7.1), not unconditionally BINARY. `arg_collations` is aligned
        // with the arguments; a missing entry defaults to BINARY (the core default).
        if let Some(seen) = ag.seen.as_mut() {
            let dkey: Vec<CellKey> = args
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    cell_key(v, ac.arg_collations.get(i).copied().unwrap_or(Collation::Binary))
                })
                .collect();
            if !seen.insert(dkey) {
                continue;
            }
        }

        // (d) Aggregate ORDER BY: buffer `(args, order_key_vals, arg_subtypes)` to sort at
        // finalize; otherwise fold immediately. Either way the per-argument subtypes are
        // PUBLISHED right before the `step` that reads them — for the immediate fold, here;
        // for the buffered path, on replay in `build` (the buffer carries them along).
        if ac.order_by.is_empty() {
            let mut ctx = EvalCtx { rt, env, outer: row };
            ctx.set_arg_subtypes(&arg_subtypes);
            ag.acc.step(&args, &mut ctx)?;
        } else {
            let mut order_vals = Vec::with_capacity(ac.order_by.len());
            {
                let mut ctx = EvalCtx { rt, env, outer: row };
                for sk in &ac.order_by {
                    order_vals.push(eval(&sk.expr, row, &mut ctx)?);
                }
            }
            ag.ordered
                .as_mut()
                .expect("ordered buffer allocated when order_by is non-empty")
                .push((args, order_vals, arg_subtypes));
        }
    }
    Ok(())
}

/// One group's state: the key values (emitted verbatim as the row prefix), one
/// [`AggState`] per aggregate call, and — for either §2.5 bare-column case — the
/// captured bare-column values.
struct GroupState {
    key_vals: Vec<Value>,
    aggs: Vec<AggState>,
    /// Running best value of the marked min/max aggregate for the single-min/max
    /// bare-column special case (lang_select.html §2.5); `None` until its first
    /// non-NULL value. Stays `None` when the plan carries no `minmax_bare` marker
    /// (the general `bare_arbitrary` case does not use it).
    capture_best: Option<Value>,
    /// The captured bare-column values, in the marker's register order. For
    /// `minmax_bare` these come from the extremum row (set/overwritten as the extremum
    /// advances, in step (b')); for `bare_arbitrary` from the group's FIRST row (set
    /// once, at group creation). `None` until captured — and forever for a min/max group
    /// whose marked column is entirely NULL, or the synthetic empty-input group, whose
    /// bare columns then emit as NULL.
    captured: Option<Vec<Value>>,
}

impl GroupState {
    /// A fresh group: capture its key values and build a zero-step accumulator (plus
    /// the DISTINCT/ORDER BY side buffers each call needs) for every aggregate.
    fn new(key_vals: Vec<Value>, calls: &[AggregateCall]) -> GroupState {
        GroupState {
            key_vals,
            aggs: calls.iter().map(AggState::new).collect(),
            capture_best: None,
            captured: None,
        }
    }
}

/// Per-(group, aggregate) state: the accumulator plus the optional side buffers the
/// operator uses to apply DISTINCT and aggregate ORDER BY before the accumulator sees
/// the values.
struct AggState {
    acc: Box<dyn AggregateAccumulator>,
    /// Seen argument tuples, present iff the call is `DISTINCT`.
    seen: Option<HashSet<Vec<CellKey>>>,
    /// Buffered `(args, order_key_vals, arg_subtypes)`, present iff the call has aggregate
    /// `ORDER BY`. Sorted by the order keys and replayed through `step` at finalize; the
    /// captured per-argument JSON value-subtypes ride along so they can be republished
    /// before each replayed `step` (json1.html §3.4).
    ordered: Option<Vec<(Vec<Value>, Vec<Value>, Vec<u8>)>>,
}

impl AggState {
    fn new(call: &AggregateCall) -> AggState {
        AggState {
            // The accumulator compares under the FIRST argument's §7.1 collation (BINARY if
            // absent). Only order-sensitive aggregates (min/max) read it; the rest ignore
            // it. This is the SAME collation the bare-column capture keys by (see
            // `capture_best` / `extremum_wins` below), so a captured `SELECT b, max(c)`
            // row always agrees with the reported `max(c)`.
            acc: call
                .func
                .new_accumulator(call.arg_collations.first().copied().unwrap_or(Collation::Binary)),
            seen: if call.distinct { Some(HashSet::new()) } else { None },
            ordered: if call.order_by.is_empty() { None } else { Some(Vec::new()) },
        }
    }
}
