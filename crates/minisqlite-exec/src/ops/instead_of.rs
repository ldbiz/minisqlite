//! The `INSTEAD OF` view-DML operator: fire a view's triggers instead of writing a base
//! table.
//!
//! A view owns no b-tree, so `INSERT`/`UPDATE`/`DELETE` on a view with a matching
//! `INSTEAD OF` trigger makes NO storage write of its own — it fires the trigger body FOR
//! EACH affected view row (`lang_createtrigger.html` §3). Like the base-table DML
//! operators this is eager: all work runs on the FIRST `next_row` pull (where the
//! [`Runtime`] is in hand for the recursion/change bookkeeping). A view DML with a
//! `RETURNING` clause then streams one buffered output row per affected view row on the
//! later pulls (computed in the read phase, before firing — see [`InsteadOf::returning`]);
//! without `RETURNING` the cursor yields no rows.
//!
//! Two phases, and why they cannot overlap (the same reason as [`crate::ops::delete`]):
//! the `frame_source` reads pages through a shared view of the store set, while a fired
//! action mutates a base table through an exclusive `&mut` reborrow of that set. So phase
//! one drains the ENTIRE `frame_source` into a buffer under the shared view, and only once
//! that view is released does phase two take `&mut` and fire. Buffering the whole frame set
//! also makes a redirect whose view reads the very base table an action mutates safe: the
//! scan is fully consumed before any write. The buffer is bounded by the affected view
//! rows (the rows the redirected DML touches); nothing larger is materialized.
//!
//! Row shape consumed here: `frame_source` emits `outer ++ (OLD ++ NEW)`, the `OLD ++
//! NEW` frame exactly `2W = 2*(column_count+1)` wide (the planner shapes it — see
//! [`InsteadOf`]). The frame is fired through each program with `trigger::fire_program`,
//! the SAME routine base-table triggers use, so a view redirect and a base-table trigger
//! cannot drift.

use minisqlite_catalog::Catalog;
use minisqlite_plan::{InsteadOf, InsteadOfEvent, Plan};
use minisqlite_types::{Error, Result, Row, Value};

use crate::env::{PagerSet, Env};
use crate::ops::returning::eval_returning;
use crate::ops::trigger;
use crate::runner::build_cursor;
use crate::runtime::Runtime;
use crate::RowCursor;

/// Build the `INSTEAD OF` cursor. It holds the write-side pager set ([`PagerSet`]) a
/// fired action needs and defers all work to the first pull (see the module doc).
pub(crate) fn instead_of<'e>(
    node: &'e InsteadOf,
    catalog: &'e dyn Catalog,
    pagers: PagerSet<'e>,
    plan: &'e Plan,
    outer: &'e [Value],
) -> Result<Box<dyn RowCursor + 'e>> {
    Ok(Box::new(InsteadOfCursor {
        node,
        catalog,
        pagers,
        plan,
        outer,
        fired: false,
        returning: Vec::new(),
        ret_idx: 0,
    }))
}

struct InsteadOfCursor<'e> {
    node: &'e InsteadOf,
    catalog: &'e dyn Catalog,
    pagers: PagerSet<'e>,
    plan: &'e Plan,
    /// The correlated frame this redirect runs under: empty at top level (a view DML is
    /// always compiled top-level — a view DML nested in a trigger action stays the "cannot
    /// modify" error), so `frame_source` rows have no prefix to strip. Kept for symmetry
    /// with the other DML operators and to stay correct if that cut is ever lifted.
    outer: &'e [Value],
    /// Whether the firing has run (exactly once, on the first pull).
    fired: bool,
    /// Buffered `RETURNING` rows, computed during the read phase (one per affected view
    /// row) and streamed one per `next_row` after firing. Empty when the view DML has no
    /// `RETURNING`, in which case the cursor yields no rows.
    returning: Vec<Row>,
    /// Index of the next buffered `RETURNING` row to emit.
    ret_idx: usize,
}

impl RowCursor for InsteadOfCursor<'_> {
    fn next_row(&mut self, rt: &mut Runtime) -> Result<Option<Row>> {
        if !self.fired {
            self.fire_all(rt)?;
            self.fired = true;
        }
        // Stream the buffered RETURNING rows (empty when the statement had no RETURNING).
        if self.ret_idx < self.returning.len() {
            let row = std::mem::take(&mut self.returning[self.ret_idx]);
            self.ret_idx += 1;
            return Ok(Some(row));
        }
        Ok(None)
    }
}

impl InsteadOfCursor<'_> {
    /// Drive `frame_source` and fire the view's `INSTEAD OF` programs per affected row.
    fn fire_all(&mut self, rt: &mut Runtime) -> Result<()> {
        let node = self.node;
        let catalog = self.catalog;
        let plan = self.plan;
        // `W = C + 1`; each frame is `OLD ++ NEW`, width `2W`.
        let frame_width = 2 * (node.column_count + 1);

        // (1) READ PHASE (shared borrow of the whole store set). Drain the entire frame
        // source into `frames`, then release the shared view before firing takes the
        // exclusive target pager. Buffering gives a stable set even when a fired action
        // mutates a base table the view reads.
        let mut frames: Vec<Row> = Vec::new();
        {
            let env = Env { catalog, pagers: self.pagers.source(), plan };
            let mut cur = build_cursor(&node.frame_source, env, self.outer)?;
            while let Some(r) = cur.next_row(rt)? {
                frames.push(r);
            }
        }

        // Validate every frame's width ONCE, up front — so both the RETURNING eval below and
        // the fire loop can index the frame halves without re-checking. Each `frame_source`
        // row is `outer ++ frame`, the `2W` frame prefixed by the correlated `outer`.
        let base = self.outer.len();
        for row in &frames {
            if row.len() != base + frame_width {
                return Err(Error::sql(format!(
                    "INSTEAD OF frame width {} != expected {} (2*(C+1)={frame_width}, outer={base})",
                    row.len(),
                    base + frame_width
                )));
            }
        }

        // (1b) RETURNING (before firing). `lang_returning.html`: the emitted values are those
        // "as seen by the top-level … statement", i.e. the frame the planner built — NOT any
        // change a fired body makes — so evaluate them here, while the shared source view is
        // still available (a RETURNING subquery may read any namespace) and BEFORE phase 2
        // takes the store mutably. Each affected view row yields one output row, evaluated
        // over that row's C view columns: the NEW half for INSERT/UPDATE, the OLD half for
        // DELETE. The frame layout is `OLD(0..W) ++ NEW(W..2W)` with `W = C + 1`, so the
        // view columns of a half start at that half's base (the trailing rowid slot is
        // skipped — a view has none). `regs` is the C-wide half, doubling as a correlated
        // RETURNING subquery's outer (see `eval_returning`).
        if !node.returning.is_empty() {
            let c = node.column_count;
            let half_base = base
                + match node.event {
                    InsteadOfEvent::Insert | InsteadOfEvent::Update => c + 1, // NEW half
                    InsteadOfEvent::Delete => 0,                              // OLD half
                };
            for row in &frames {
                let regs = &row[half_base..half_base + c];
                let out =
                    eval_returning(&node.returning, regs, catalog, self.pagers.source(), plan, rt)?;
                self.returning.push(out);
            }
        }

        // (2) FIRE PHASE (exclusive per-action reborrow of the WHOLE store set, one frame at
        // a time). The read phase above released its shared view, so firing can take the set
        // mutably; each fired action then resolves its OWN stamped `node.db` against it (see
        // `trigger::fire_program`). A view redirect whose action writes a temp / attached /
        // main table each reaches the CORRECT store — there is no fixed-namespace pin.
        //
        // Each `frame_source` row is `outer ++ frame`; strip the correlated prefix to recover
        // the `2W` frame the programs bind against, then fire every matching program (each
        // gated by its own WHEN inside `fire_program`), in catalog order.
        for row in &frames {
            let frame = &row[base..];
            for program in &node.programs {
                match trigger::fire_program(program, frame, catalog, &mut self.pagers, rt, plan) {
                    Ok(()) => {}
                    // `RAISE(IGNORE)` in an INSTEAD OF body abandons this view row's
                    // remaining programs without an error (lang_createtrigger.html §RAISE);
                    // the outer loop moves on to the next frame row. A real error propagates.
                    Err(e) => {
                        if rt.take_raise_ignore() {
                            break;
                        }
                        return Err(e);
                    }
                }
            }
        }
        Ok(())
    }
}
