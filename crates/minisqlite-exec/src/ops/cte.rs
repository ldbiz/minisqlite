//! CTE operators: [`cte_scan`] scans a materialized CTE / derived table from
//! [`Plan::ctes`], and [`recursive_scan`] reads the current working table inside a
//! recursive-CTE step. Rows here are the CTE's OUTPUT width (`column_count`) with NO
//! trailing rowid register — unlike a base-table scan.
//!
//! ## The `outer` prefix (correlated subqueries / trigger actions)
//! `CteScan` is a FROM leaf, so it obeys the same contract as [`scan`](crate::ops::scan):
//! it emits `outer ++ local`, contributing the enclosing correlated/trigger prefix EXACTLY
//! ONCE. The body (and a recursive seed/step) is compiled STANDALONE — a CTE / derived
//! table / view has no LATERAL, so its `Column(i)` are base-0 local — and is therefore
//! drained with [`NO_OUTER`]; the recursive working-table frame stays `column_count`-wide
//! for the same reason (its `RecursiveScan` reads it at base 0). The `outer` prefix is
//! prepended once at the [`CteScan`] OUTPUT boundary ([`CteScanCursor::next_row`]), never
//! threaded into the base-0 body. At the top level `outer` is empty, so all of this is a
//! no-op — a derived/CTE scan is byte-for-byte unchanged at `base_offset == 0`.
//!
//! ## Materialized CTE: buffered once, bounded
//! A materialized CTE runs its `body` ONCE and buffers the rows, because a CTE can be
//! scanned more than once. The buffer is bounded by the CTE's own cardinality — never a
//! copy of a base table. Everything downstream still streams one row per pull.
//!
//! ## Recursive fixpoint (semi-naive), streamed lazily
//! `run seed -> frontier`, then each round runs `step` with the previous round's rows
//! visible through [`recursive_scan`], keeping only the NEW rows as the next frontier,
//! until a round adds nothing (spec `lang_with.html` §3). `UNION ALL` keeps every row;
//! `UNION` admits a row only if its whole-row key is new — the `seen` set persists
//! across ALL rounds (a row admitted once is never re-admitted). A hard row cap turns a
//! non-terminating `UNION ALL` recursion into a loud error rather than an infinite loop
//! or an out-of-memory kill (an unbounded result is an incorrect result).
//!
//! The generator ([`RecursiveGen`]) is LAZY: it buffers one round and computes the next
//! only when that buffer empties and the consumer keeps pulling. So a downstream
//! `LIMIT n` that stops after `n` rows never drives the recursion further — the idiom
//! that bounds an otherwise-infinite recursive CTE with an OUTER limit. A body
//! `LIMIT`/`OFFSET` (spec §3) is handled the same way: the compiler lowers it to a
//! [`CtePlan::Materialized`] wrapping a [`PlanNode::Limit`] over this fixpoint's scan, and
//! that wrapper's `LIMIT` — pulling lazily — bounds the recursion just by stopping.
//!
//! DOCUMENTED REFINEMENTS (follow-ups, out of scope for now):
//! * `UNION` dedup is under BINARY collation for every column (numeric-folding, NULLs
//!   equal), because [`CtePlan::Recursive`] carries no per-column collation — mirrors
//!   the `Distinct` operator's refinement.
//! * A CTE referenced by several `CteScan` sites is re-materialized per site — a constant
//!   factor (O(k) scans), not a complexity blowup. A shared per-statement cache would
//!   avoid the repeat, and the mechanism it needs now exists: the uncorrelated-subquery
//!   cache keeps a by-`id` map in [`Runtime`] that `StatementRoot` clears at the start of
//!   each statement, so a prior statement's rows cannot leak into the next `ctes[id]`; a
//!   shared-CTE cache can follow that same shape. This is not only a perf follow-up — a
//!   volatile CTE body (`WITH c AS (SELECT random()) ... c JOIN c`) re-materialized per
//!   reference would yield DIFFERENT rows per site, the same evaluate-once correctness
//!   the subquery cache fixes, so a multi-referenced CTE must materialize once per
//!   statement.
//!
//! [`Plan::ctes`]: minisqlite_plan::Plan::ctes
//! [`CtePlan::Recursive`]: minisqlite_plan::CtePlan::Recursive
//! [`Runtime`]: crate::runtime::Runtime

use std::collections::HashSet;

use minisqlite_plan::{CtePlan, PlanNode};
use minisqlite_types::{Error, Result, Row, Value};

use super::with_outer;
use crate::env::Env;
use crate::keys::{row_key, CellKey};
use crate::runner::build_cursor;
use crate::runtime::Runtime;
use crate::RowCursor;

/// The EMPTY outer a CTE body / recursive seed+step is drained with. A CTE (derived
/// table, view, or CTE reference) is compiled STANDALONE (base 0, non-correlated — SQLite
/// has no LATERAL), so its body's `Column(i)` are all local and must be read against a row
/// with NO outer prefix. The enclosing correlated/trigger `outer` is instead prepended
/// ONCE at the `CteScan` OUTPUT boundary (see [`CteScanCursor::next_row`]), so this leaf
/// obeys the same `outer ++ local` contract as every other leaf. `'static` so it coerces
/// to any cursor lifetime.
const NO_OUTER: &[Value] = &[];

/// Hard upper bound on rows a single recursive CTE may produce. A non-terminating
/// `UNION ALL` recursion (`... UNION ALL SELECT x+1 FROM c` with no stopping `WHERE`)
/// would otherwise loop forever and exhaust memory; hitting this cap is a loud error,
/// not a raised limit — the recursion, not the number, is the bug.
const MAX_RECURSIVE_ROWS: usize = 1_000_000;

/// Scan `Plan.ctes[id]`. Setup is LAZY (deferred to the first pull) because both the
/// materialized drain and the recursive fixpoint need the [`Runtime`], which is only
/// available in [`RowCursor::next_row`]. `column_count` is the CTE's output width. `outer`
/// is the correlated-subquery / trigger-action prefix this leaf prepends to each output
/// row (empty and free at the top level); see [`CteScanCursor::next_row`].
pub(crate) fn cte_scan<'e>(
    id: usize,
    column_count: usize,
    env: Env<'e>,
    outer: &'e [Value],
) -> Result<Box<dyn RowCursor + 'e>> {
    Ok(Box::new(CteScanCursor { id, column_count, env, outer, output: None }))
}

struct CteScanCursor<'e> {
    id: usize,
    column_count: usize,
    env: Env<'e>,
    /// The correlated/trigger outer row prepended to every emitted row so this leaf obeys
    /// the `outer ++ local` contract (empty at the top level). The CTE body itself is
    /// drained with [`NO_OUTER`], not this — the prefix is added at the output boundary.
    outer: &'e [Value],
    /// The output source, built on the first pull. `None` until then.
    output: Option<CteOutput<'e>>,
}

/// The two shapes a `CteScan` streams: a materialized CTE / derived table buffers its
/// whole body once; a recursive CTE runs a LAZY semi-naive fixpoint so a downstream
/// `LIMIT` can bound an otherwise-infinite recursion just by stopping its pulls.
enum CteOutput<'e> {
    Buffered(std::vec::IntoIter<Row>),
    Recursive(Box<RecursiveGen<'e>>),
}

impl RowCursor for CteScanCursor<'_> {
    fn next_row(&mut self, rt: &mut Runtime) -> Result<Option<Row>> {
        if self.output.is_none() {
            self.output = Some(self.build(rt)?);
        }
        let local = match self.output.as_mut().expect("output built on first pull") {
            CteOutput::Buffered(iter) => iter.next(),
            CteOutput::Recursive(generator) => generator.next(rt)?,
        };
        // A `CteScan` is a FROM leaf, so — like every other leaf (see `ops::scan`) — it
        // emits `outer ++ local` and thus contributes a correlated subquery's / trigger
        // action's outer prefix EXACTLY ONCE. The body was drained STANDALONE with an empty
        // outer (`build` / `NO_OUTER`), so `local` is the CTE's own `column_count`-wide row
        // (recursive frames stay that width too); the prefix is prepended HERE, at the
        // output boundary. `with_outer` is a no-op when `outer` is empty (the top level), so
        // this is byte-for-byte unchanged at `base_offset == 0`.
        Ok(local.map(|row| with_outer(self.outer, row)))
    }
}

impl<'e> CteScanCursor<'e> {
    /// Resolve the CTE and set up its output on the first pull. A materialized CTE
    /// drains its `body` once (bounded by its cardinality); a recursive CTE builds the
    /// lazy fixpoint generator, running only the seed up front. An out-of-range `id` is
    /// a malformed plan and errors rather than panicking.
    fn build(&mut self, rt: &mut Runtime) -> Result<CteOutput<'e>> {
        // `env.plan` is `&'e Plan` (an `Env` field, `Copy`), so the borrow of the CTE and
        // its child plans has lifetime `'e`, not the `&mut self` borrow — the returned
        // output can hold them.
        let plan = self.env.plan;
        let cte = plan
            .ctes
            .get(self.id)
            .ok_or_else(|| Error::sql("CteScan references an out-of-range CTE id"))?;
        match cte {
            CtePlan::Materialized { body, .. } => {
                // Drain the body STANDALONE with an empty outer: it is a base-0,
                // non-correlated plan, so its `Column(i)` must read its own rows, never the
                // outer prefix. The correlated/trigger prefix is prepended once at the
                // output boundary (`next_row`).
                let rows = drain_all(body, self.env, NO_OUTER, rt)?;
                // The child plans already emit the CTE's output width; this guards a
                // planner producing a body whose width disagrees with the carried count.
                // Checked on the PRE-prefix body rows (the outer prefix is added in
                // `next_row`).
                debug_assert!(
                    rows.iter().all(|r| r.len() == self.column_count),
                    "CteScan body rows must be column_count wide (no trailing rowid, no prefix)"
                );
                Ok(CteOutput::Buffered(rows.into_iter()))
            }
            CtePlan::Recursive { seed, step, union_all, .. } => {
                let generator = RecursiveGen::new(
                    seed,
                    step,
                    *union_all,
                    self.env,
                    self.column_count,
                    rt,
                )?;
                Ok(CteOutput::Recursive(Box::new(generator)))
            }
        }
    }
}

/// Read the current recursive-CTE working table (the previous round's rows). Reads no
/// plan node; it snapshots the frontier from the [`Runtime`] frame stack. `column_count`
/// is the recursive CTE's output width (used only to guard the row shape in debug).
pub(crate) fn recursive_scan<'e>(column_count: usize) -> Box<dyn RowCursor + 'e> {
    Box::new(RecursiveScanCursor { column_count, rows: None })
}

struct RecursiveScanCursor {
    column_count: usize,
    /// The frontier snapshot, taken on the FIRST pull (when `rt` is available). Taking
    /// it then — not at build — gives a stable view even if this cursor is rebuilt
    /// several times within one step round (e.g. as a nested-loop join's inner side,
    /// rebuilt per outer row): each rebuild re-snapshots the same frozen frontier.
    rows: Option<std::vec::IntoIter<Row>>,
}

impl RowCursor for RecursiveScanCursor {
    fn next_row(&mut self, rt: &mut Runtime) -> Result<Option<Row>> {
        if self.rows.is_none() {
            self.rows = Some(rt.current_recursive_frame()?.into_iter());
        }
        let row = self.rows.as_mut().expect("frame snapshotted on first pull").next();
        if let Some(r) = &row {
            debug_assert_eq!(
                r.len(),
                self.column_count,
                "RecursiveScan row width must equal column_count"
            );
        }
        Ok(row)
    }
}

/// A lazy semi-naive recursive-CTE generator. Emits the CTE's rows on demand — the seed
/// rows first, then round by round — so a consumer that stops pulling (a downstream or
/// wrapper `LIMIT`) never drives the recursion to completion. That laziness is what bounds
/// an otherwise-infinite recursion under an outer/wrapper `LIMIT`.
///
/// FRAME-STACK INVARIANT: a round pushes the frontier as a recursive frame, drains `step`
/// fully into a buffer, then pops — all within one [`Self::refill`], i.e. within a single
/// [`Self::next`] call. No frame is ever held across `next` calls, so an interleaved outer
/// operator (or an enclosing recursion) never observes an unbalanced stack.
struct RecursiveGen<'e> {
    step: &'e PlanNode,
    union_all: bool,
    env: Env<'e>,
    column_count: usize,
    /// UNION whole-row dedup set; persists across ALL rounds (a row admitted once is never
    /// re-admitted, even after it left the frontier). Left empty (unused) for `UNION ALL`.
    seen: HashSet<Vec<CellKey>>,
    /// The frontier for the NEXT round: rows admitted last round whose `step` has not run
    /// yet. `mem::take`n when a round runs; an empty frontier ends the fixpoint.
    working: Vec<Row>,
    /// The current round's admitted rows, waiting to be emitted.
    buffer: std::vec::IntoIter<Row>,
    /// Rows admitted (added to the recursive table) so far, for the [`MAX_RECURSIVE_ROWS`]
    /// safety cap.
    admitted: usize,
}

impl<'e> RecursiveGen<'e> {
    /// Build the generator and run the seed round, staging its admitted rows as both the
    /// first frontier and the first emit buffer.
    fn new(
        seed: &'e PlanNode,
        step: &'e PlanNode,
        union_all: bool,
        env: Env<'e>,
        column_count: usize,
        rt: &mut Runtime,
    ) -> Result<Self> {
        let mut generator = RecursiveGen {
            step,
            union_all,
            env,
            column_count,
            seen: HashSet::new(),
            working: Vec::new(),
            buffer: Vec::new().into_iter(),
            admitted: 0,
        };
        // The seed is a base-0, non-correlated plan (a CTE has no LATERAL); drain it
        // STANDALONE. The correlated/trigger prefix is added at the `CteScan` output.
        let seed_rows = drain_all(seed, env, NO_OUTER, rt)?;
        let admitted = generator.admit_round(seed_rows)?;
        generator.buffer = admitted.clone().into_iter();
        generator.working = admitted;
        Ok(generator)
    }

    /// Emit the next CTE row, running further rounds lazily as the buffer empties.
    fn next(&mut self, rt: &mut Runtime) -> Result<Option<Row>> {
        loop {
            match self.buffer.next() {
                Some(row) => {
                    debug_assert_eq!(
                        row.len(),
                        self.column_count,
                        "recursive CTE row width must equal column_count"
                    );
                    return Ok(Some(row));
                }
                // Current round drained; run the next one. `false` = fixpoint reached.
                None => {
                    if !self.refill(rt)? {
                        return Ok(None);
                    }
                }
            }
        }
    }

    /// Run one round of the fixpoint: hand the current frontier to the frame stack, drain
    /// `step` over it (the `RecursiveScan` inside `step` reads that frontier), admit the
    /// new rows, and stage them as both the emit buffer and the next frontier. Returns
    /// `false` when the recursion is finished (an empty frontier, or a round that admitted
    /// no new rows). The frame is ALWAYS popped before an error propagates, so the stack
    /// stays balanced for any later statement on this connection.
    fn refill(&mut self, rt: &mut Runtime) -> Result<bool> {
        let frontier = std::mem::take(&mut self.working);
        if frontier.is_empty() {
            return Ok(false);
        }
        rt.push_recursive_frame(frontier);
        // The step is a base-0 plan whose `RecursiveScan` reads the frontier frame (also
        // `column_count`-wide); drain it STANDALONE with an empty outer.
        let step_result = drain_all(self.step, self.env, NO_OUTER, rt);
        rt.pop_recursive_frame();
        let admitted = self.admit_round(step_result?)?;
        if admitted.is_empty() {
            return Ok(false);
        }
        self.buffer = admitted.clone().into_iter();
        self.working = admitted;
        Ok(true)
    }

    /// Admit every row a round (seed or step) produced, returning the newly-admitted rows
    /// (the next frontier). `UNION ALL` keeps every row; `UNION` keeps a row only if its
    /// whole-row key is new (numeric-folding, NULLs equal — lang_with.html: "NULL values
    /// compare equal to one another"). Enforces [`MAX_RECURSIVE_ROWS`] so a non-terminating
    /// recursion fails loudly, not forever.
    ///
    /// SCOPED FOLLOW-UP (same bug class as FIX 1/2/3, deliberately NOT fixed here): the
    /// `UNION` dedup keys TEXT under BINARY (`row_key(&row, &[])`), not the CTE column's
    /// §7.1 collation. So a recursive CTE whose seed column is `COLLATE NOCASE` could
    /// over-emit case-variant rows real SQLite folds. It is recorded rather than fixed
    /// because the collation is UNVERIFIABLE from the docs here: lang_with.html specifies
    /// only NULL handling for "no identical row" and is silent on text collation, and real
    /// SQLite implements this dedup via a transient index whose key collation is not
    /// documented — so threading a guessed collation risks REGRESSING the common,
    /// currently-correct BINARY case. If confirmed collation-sensitive against real
    /// sqlite3, the fix mirrors FIX 3: add `column_collations` to `CtePlan::Recursive`
    /// (resolved from the seed select's per-column collation, left-arm precedence) and pass
    /// it here as `row_key(&row, &collations)`.
    fn admit_round(&mut self, rows: Vec<Row>) -> Result<Vec<Row>> {
        let mut admitted = Vec::new();
        for row in rows {
            let is_new = self.union_all || self.seen.insert(row_key(&row, &[]));
            if is_new {
                self.admitted += 1;
                if self.admitted > MAX_RECURSIVE_ROWS {
                    return Err(Error::sql("recursive CTE exceeded the maximum row limit"));
                }
                admitted.push(row);
            }
        }
        Ok(admitted)
    }
}

/// Build a cursor for `node` and drain it fully into a buffer. The buffer is bounded by
/// the node's cardinality (a CTE body or one recursion round), the allowed
/// materialization here.
fn drain_all<'e>(
    node: &'e PlanNode,
    env: Env<'e>,
    outer: &'e [Value],
    rt: &mut Runtime,
) -> Result<Vec<Row>> {
    let mut cur = build_cursor(node, env, outer)?;
    let mut rows = Vec::new();
    while let Some(row) = cur.next_row(rt)? {
        rows.push(row);
    }
    Ok(rows)
}
