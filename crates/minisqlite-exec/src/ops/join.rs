//! `Join` — pair rows of two subtrees, emitting `left ++ right` (width
//! `left_width + right_width`). Runs the physical [`JoinStrategy`] the planner
//! chose; it never re-decides and never falls back to a full nested scan when a
//! keyed strategy is given.
//!
//! ## Semantics (shared by all three strategies)
//! * A pair is kept only when `on` (when `Some`) evaluates to TRUE under
//!   three-valued logic (`truth == Some(true)`); a NULL/FALSE `on` drops it. A
//!   `Cross` join has `on == None` and keeps every pair.
//! * Outer fills use [`Value::Null`] for every register of the missing side, sized
//!   by the plan's declared `left_width` / `right_width` (there is no real row to
//!   take the width from). `need_left_fill` = `Left`/`Full` (emit `L ++ NULLs` for a
//!   left row that matched nothing); `need_right_fill` = `Right`/`Full` (emit
//!   `NULLs ++ R` for a right row that matched nothing).
//! * A NULL equijoin key never equals anything (SQL `=` with NULL is never true), so
//!   hash join excludes such rows from the table and from probe matches, but still
//!   emits them through the outer-join path.
//!
//! ## The `outer` prefix (correlated subqueries / trigger actions)
//! When this join runs inside a correlated subquery or a trigger action, the executor
//! prepends an `outer` row (the enclosing / `OLD ++ NEW` frame, width `base_offset`) to
//! the combined join row, which must appear EXACTLY ONCE: `outer ++ left ++ right`. It
//! is contributed by the LEFT spine only — the join builds its LEFT child with `outer`
//! (so the leftmost leaf prepends it once) and its RIGHT child with an EMPTY outer (so
//! the right leaf emits its own columns with no prefix). `concat(left, right)` then
//! yields `outer ++ left_local ++ right_local`, exactly what the planner bound `on` /
//! WHERE / the projection against (sources laid out at `base_offset`, `base_offset +
//! WL`, …). At the top level `outer` is empty, so building the right with an empty outer
//! is byte-for-byte the same as building it with the (already empty) `outer` — this
//! whole prefix path is a no-op there. The null-fills preserve the single prefix: a
//! `Left`/`Full` fill keeps the left's prefix (`left ++ NULLs`), a `Right`/`Full` fill
//! re-prepends it (`outer ++ NULLs ++ right`). See [`fill_right`] / [`fill_left`].
//!
//! ## Streaming vs. materialization (the performance contract)
//! * NestedLoop streams the left once and materializes the right ONCE (a `Vec<Row>`
//!   bounded by the right size), then iterates that buffer by index for every left row.
//!   The right subtree is built with an EMPTY outer (it does not depend on the current
//!   left row — that is `IndexNestedLoop`'s job), so its rows are identical every pass;
//!   draining it once and replaying the buffer is therefore result-identical to
//!   re-scanning it per left row, and it turns an O(left_rows * right_rows) RE-SCAN into
//!   a single right scan (critical when the left side is large — a per-left rebuild would
//!   re-read the whole right table millions of times). `Right`/`Full` additionally keep a
//!   matched bitmap over that same buffer so unmatched right rows can be null-filled after
//!   the left loop. (This mirrors SQLite materializing an uncorrelated FROM subquery once.)
//! * Hash builds a table on the RIGHT (the build side the planner sized smaller) by
//!   draining it once — the one bounded materialization — then streams the left as
//!   the probe side.
//! * IndexNestedLoop drives a per-left seek of the right; it buffers only one left
//!   row's worth of right matches at a time (bounded by a single seek), never the
//!   whole right.

use std::collections::HashMap;

use minisqlite_expr::{eval, truth, EvalExpr};
use minisqlite_plan::{Join, JoinStrategy, JoinType};
use minisqlite_types::{Collation, Error, Result, Row, Value};

use crate::context::EvalCtx;
use crate::env::Env;
use crate::keys::{cell_key, CellKey};
use crate::runner::build_cursor;
use crate::runtime::Runtime;
use crate::RowCursor;

/// The empty `outer` a join builds its RIGHT child with: the outer prefix is
/// contributed ONCE by the left spine, so the right side emits only its own columns
/// (see the module-level "outer prefix" note). `'static` so it coerces to any cursor
/// lifetime. At the top level the incoming `outer` is already empty, so using this for
/// the right build is byte-for-byte identical there.
const NO_OUTER: &[Value] = &[];

/// Build a cursor for a `Join` node, dispatching on the planner-chosen strategy.
pub(crate) fn join<'e>(
    j: &'e Join,
    env: Env<'e>,
    outer: &'e [Value],
) -> Result<Box<dyn RowCursor + 'e>> {
    match &j.strategy {
        JoinStrategy::NestedLoop => nested_loop(j, env, outer),
        JoinStrategy::Hash { left_keys, right_keys, key_collations } => {
            hash_join(j, env, outer, left_keys, right_keys, key_collations)
        }
        JoinStrategy::IndexNestedLoop => index_nested_loop(j, env, outer),
    }
}

// ---- shared row helpers ---------------------------------------------------

/// `left ++ right` — the combined join row. Width is `left.len() + right.len()`
/// (which is `WL + WR` when each side emits its declared width).
fn concat(left: &[Value], right: &[Value]) -> Row {
    let mut row = Vec::with_capacity(left.len() + right.len());
    row.extend_from_slice(left);
    row.extend_from_slice(right);
    row
}

/// `left ++ NULLs` — a left row whose right side matched nothing (`Left`/`Full`).
/// The left operand ALREADY carries the correlated `outer` prefix (the left spine
/// prepended it once), so this appends only the declared right-width NULLs and the
/// single prefix is preserved: `outer ++ left_local ++ NULLs`. The NULL count is the
/// plan's declared right width, not any real right row; the present side keeps its
/// actual bytes, asserted (debug) to equal `outer_width + left_width` so a subtree
/// that emits a mis-sized row is caught at its cause rather than shipping a ragged
/// output row downstream. `outer_width` is 0 at the top level, so the assertion and
/// the shape are unchanged there.
fn fill_right(left: &[Value], outer_width: usize, left_width: usize, right_width: usize) -> Row {
    debug_assert_eq!(
        left.len(),
        outer_width + left_width,
        "left row width matches the outer prefix plus the plan's declared WL"
    );
    let mut row = Vec::with_capacity(left.len() + right_width);
    row.extend_from_slice(left);
    row.extend(std::iter::repeat(Value::Null).take(right_width));
    row
}

/// `outer ++ NULLs ++ right` — a right row that matched no left (`Right`/`Full`). The
/// right operand is the STANDALONE right row (built with an empty outer, so no prefix),
/// so the correlated `outer` prefix is re-prepended here and the left side null-filled:
/// the combined row keeps the single prefix `outer ++ NULLs(left_width) ++ right_local`.
/// The NULL count is the plan's declared left width; the present right side keeps its
/// actual bytes, asserted (debug) to equal `right_width`. `outer` is empty at the top
/// level, so the shape (`NULLs ++ right`) is unchanged there.
fn fill_left(outer: &[Value], right: &[Value], left_width: usize, right_width: usize) -> Row {
    debug_assert_eq!(right.len(), right_width, "right row width matches the plan's declared WR");
    let mut row = Vec::with_capacity(outer.len() + left_width + right.len());
    row.extend_from_slice(outer);
    row.extend(std::iter::repeat(Value::Null).take(left_width));
    row.extend_from_slice(right);
    row
}

/// Keep a candidate pair iff `on` (when `Some`) evaluates TRUE under three-valued
/// logic — a NULL or FALSE `on` rejects it; `on == None` (a cross product) keeps
/// every pair. `combined` is the already-assembled join row the predicate binds
/// against. The single home of the join-`on` 3VL rule, so NestedLoop, Hash, and
/// IndexNestedLoop cannot drift on it.
fn on_keeps(on: Option<&EvalExpr>, env: Env, rt: &mut Runtime, combined: &[Value]) -> Result<bool> {
    match on {
        None => Ok(true),
        Some(pred) => {
            let mut ctx = EvalCtx { rt, env, outer: combined };
            Ok(truth(&eval(pred, combined, &mut ctx)?) == Some(true))
        }
    }
}

/// Assemble the combined `left ++ right` row and keep it only if [`on_keeps`]. Returns
/// `Some(combined)` for a kept pair, `None` for a dropped one. Used by NestedLoop and
/// Hash, which assemble the pair here; IndexNestedLoop's combined row is pre-built by
/// its right leaf, so it calls [`on_keeps`] directly.
fn try_pair(
    on: Option<&EvalExpr>,
    env: Env,
    rt: &mut Runtime,
    left: &[Value],
    right: &[Value],
) -> Result<Option<Row>> {
    let combined = concat(left, right);
    Ok(if on_keeps(on, env, rt, &combined)? { Some(combined) } else { None })
}

/// Evaluate the equijoin key expressions against `row` into a fresh OWNED canonical key
/// vector, or `None` if ANY key value is NULL (an unmatchable row — SQL `=` with NULL is
/// never true). Used by the BUILD side, where the key is moved into the table and so must
/// be owned; the probe side reuses a buffer via [`eval_cell_keys_into`]. This is a thin
/// wrapper so the actual key fold lives in ONE place (`eval_cell_keys_into`).
fn eval_cell_keys(
    rt: &mut Runtime,
    env: Env,
    keys: &[EvalExpr],
    collations: &[Collation],
    row: &[Value],
) -> Result<Option<Vec<CellKey>>> {
    let mut out = Vec::with_capacity(keys.len());
    Ok(eval_cell_keys_into(rt, env, keys, collations, row, &mut out)?.then_some(out))
}

/// The SINGLE home of the equijoin key fold: evaluate each key expression against `row`
/// and canonicalize it into `out` (cleared first), returning `false` when ANY key value is
/// NULL (an unmatchable row — SQL `=` with NULL is never true; on a `false` return `out` is
/// left partially filled and the caller must NOT use it). The PROBE path calls this
/// directly with a REUSED buffer, so a probe row makes no per-row key allocation (the
/// capacity is retained across rows); [`eval_cell_keys`] wraps it with a fresh `Vec` for
/// the BUILD side, whose key is moved into the table. A missing collation defaults to
/// `Binary`, matching [`row_key`](crate::keys::row_key).
fn eval_cell_keys_into(
    rt: &mut Runtime,
    env: Env,
    keys: &[EvalExpr],
    collations: &[Collation],
    row: &[Value],
    out: &mut Vec<CellKey>,
) -> Result<bool> {
    out.clear();
    let mut ctx = EvalCtx { rt, env, outer: row };
    for (i, k) in keys.iter().enumerate() {
        let v = eval(k, row, &mut ctx)?;
        if v.is_null() {
            return Ok(false);
        }
        out.push(cell_key(&v, collations.get(i).copied().unwrap_or(Collation::Binary)));
    }
    Ok(true)
}

/// `(need_left_fill, need_right_fill)` for a join flavor — an EXHAUSTIVE match so a
/// new [`JoinType`] variant is a compile error here rather than silently classifying
/// as no-fill (an inner join).
fn outer_fills(t: &JoinType) -> (bool, bool) {
    match t {
        JoinType::Inner => (false, false),
        JoinType::Left => (true, false),
        JoinType::Right => (false, true),
        JoinType::Full => (true, true),
        JoinType::Cross => (false, false),
    }
}

/// Phase-B step shared by NestedLoop and Hash: advance `emit_pos` to the next right
/// row that matched no left and return it NULL-filled on the left (with the `outer`
/// prefix re-prepended, via [`fill_left`]), or `None` once the unmatched-right rows are
/// exhausted. Keeps the `Right`/`Full` emission identical across the two strategies.
fn next_unmatched_right(
    emit_pos: &mut usize,
    outer: &[Value],
    right_rows: &[Row],
    right_matched: &[bool],
    left_width: usize,
    right_width: usize,
) -> Option<Row> {
    while *emit_pos < right_rows.len() {
        let idx = *emit_pos;
        *emit_pos += 1;
        if !right_matched[idx] {
            return Some(fill_left(outer, &right_rows[idx], left_width, right_width));
        }
    }
    None
}

// ---- NestedLoop -----------------------------------------------------------

fn nested_loop<'e>(
    j: &'e Join,
    env: Env<'e>,
    outer: &'e [Value],
) -> Result<Box<dyn RowCursor + 'e>> {
    let left = build_cursor(&j.left, env, outer)?;
    let (need_left_fill, need_right_fill) = outer_fills(&j.join_type);
    Ok(Box::new(NestedLoopCursor {
        left,
        right_node: &j.right,
        on: j.on.as_ref(),
        env,
        outer,
        left_width: j.left_width,
        right_width: j.right_width,
        need_left_fill,
        need_right_fill,
        cur_left: None,
        cur_left_matched: false,
        right_rows: Vec::new(),
        right_matched: Vec::new(),
        right_pos: 0,
        right_materialized: false,
        left_done: false,
        emit_right_pos: 0,
    }))
}

/// Nested-loop join. The left is streamed once; the right is materialized once into
/// `right_rows` and iterated by index for every left row. `Right`/`Full` additionally use
/// `right_matched` to null-fill the right rows that matched no left after the left loop.
struct NestedLoopCursor<'e> {
    left: Box<dyn RowCursor + 'e>,
    right_node: &'e minisqlite_plan::PlanNode,
    on: Option<&'e EvalExpr>,
    env: Env<'e>,
    /// The correlated `outer` prefix (empty at the top level). It is NOT used to build
    /// the right (that gets [`NO_OUTER`]) — the left spine already prepended it once —
    /// it is used only to re-prepend the prefix on a `Right`/`Full` null-fill and to
    /// size the `Left`/`Full` fill's debug assertion.
    outer: &'e [Value],
    left_width: usize,
    right_width: usize,
    need_left_fill: bool,
    need_right_fill: bool,

    /// The left row currently being paired; `None` between rows.
    cur_left: Option<Row>,
    /// Whether `cur_left` has produced at least one kept pair (for `Left`/`Full`).
    cur_left_matched: bool,

    // --- the right side, materialized once and replayed per left row ---
    /// The right side buffered once; iterated by index for every left row.
    right_rows: Vec<Row>,
    /// Which `right_rows[i]` have been emitted in a kept pair (for `Right`/`Full`).
    right_matched: Vec<bool>,
    /// Position within `right_rows` for the current left row.
    right_pos: usize,
    right_materialized: bool,

    left_done: bool,
    /// Phase-B cursor over `right_rows` for the unmatched-right emission.
    emit_right_pos: usize,
}

impl NestedLoopCursor<'_> {
    /// Drain the right subtree ONCE into `right_rows`, bounded by the right side's size.
    /// The right is built with an EMPTY outer ([`NO_OUTER`]) so it does not depend on the
    /// current left row — every left row would re-scan an identical right — hence draining
    /// it once and replaying the buffer is result-identical to a per-left rebuild while
    /// scanning the right only once (the difference between O(left*right) re-reads and one
    /// right scan). It emits its own columns with no prefix; the single `outer` prefix is
    /// re-prepended per unmatched row by [`fill_left`] and rides along with the left in a
    /// matched pair's `concat`. The matched bitmap is allocated only for `Right`/`Full`,
    /// which are the only flavors that read it (to null-fill unmatched right rows).
    fn materialize_right(&mut self, rt: &mut Runtime) -> Result<()> {
        let mut right = build_cursor(self.right_node, self.env, NO_OUTER)?;
        let mut rows: Vec<Row> = Vec::new();
        while let Some(r) = right.next_row(rt)? {
            rows.push(r);
        }
        self.right_matched = if self.need_right_fill {
            vec![false; rows.len()]
        } else {
            Vec::new()
        };
        self.right_rows = rows;
        self.right_materialized = true;
        Ok(())
    }
}

impl RowCursor for NestedLoopCursor<'_> {
    fn next_row(&mut self, rt: &mut Runtime) -> Result<Option<Row>> {
        // Materialize the right once; every left row then iterates the same buffer.
        if !self.right_materialized {
            self.materialize_right(rt)?;
        }

        loop {
            // Ensure a current left row, positioned at the start of the right buffer.
            if self.cur_left.is_none() {
                if self.left_done {
                    break;
                }
                match self.left.next_row(rt)? {
                    None => {
                        self.left_done = true;
                        break;
                    }
                    Some(l) => {
                        self.cur_left = Some(l);
                        self.cur_left_matched = false;
                        self.right_pos = 0;
                    }
                }
            }

            // Emit the next kept pair for the current left row over the buffered right.
            while self.right_pos < self.right_rows.len() {
                let idx = self.right_pos;
                self.right_pos += 1;
                let pair = {
                    let l = self.cur_left.as_ref().expect("cur_left present");
                    try_pair(self.on, self.env, rt, l, &self.right_rows[idx])?
                };
                if let Some(row) = pair {
                    self.cur_left_matched = true;
                    if self.need_right_fill {
                        self.right_matched[idx] = true;
                    }
                    return Ok(Some(row));
                }
            }

            // Right exhausted for this left row.
            let l = self.cur_left.take().expect("cur_left present at exhaustion");
            if self.need_left_fill && !self.cur_left_matched {
                return Ok(Some(fill_right(&l, self.outer.len(), self.left_width, self.right_width)));
            }
            // Otherwise advance to the next left row.
        }

        // Phase B: right rows that matched no left (Right/Full).
        if self.need_right_fill {
            return Ok(next_unmatched_right(
                &mut self.emit_right_pos,
                self.outer,
                &self.right_rows,
                &self.right_matched,
                self.left_width,
                self.right_width,
            ));
        }
        Ok(None)
    }
}

// ---- Hash -----------------------------------------------------------------

fn hash_join<'e>(
    j: &'e Join,
    env: Env<'e>,
    outer: &'e [Value],
    left_keys: &'e [EvalExpr],
    right_keys: &'e [EvalExpr],
    key_collations: &'e [Collation],
) -> Result<Box<dyn RowCursor + 'e>> {
    let left = build_cursor(&j.left, env, outer)?;
    let (need_left_fill, need_right_fill) = outer_fills(&j.join_type);
    Ok(Box::new(HashJoinCursor {
        left,
        right_node: &j.right,
        on: j.on.as_ref(),
        env,
        outer,
        left_keys,
        right_keys,
        key_collations,
        left_width: j.left_width,
        right_width: j.right_width,
        need_left_fill,
        need_right_fill,
        built: false,
        right_rows: Vec::new(),
        table: HashMap::new(),
        flat_buckets: Vec::new(),
        right_matched: Vec::new(),
        cur_left: None,
        cur_left_matched: false,
        cur_start: 0,
        cur_len: 0,
        cur_pos: 0,
        probe_key: Vec::new(),
        emit_right_pos: 0,
    }))
}

/// Hash join. The RIGHT is the build side: drained once into `right_rows`, with
/// non-NULL-key rows indexed in `table` (key → its bucket range in `flat_buckets`).
/// The LEFT is the probe side, streamed. NULL-key rows are excluded from the table
/// (they can never match on `=`) but stay in `right_rows` so `Right`/`Full` can emit
/// them through the outer path.
struct HashJoinCursor<'e> {
    left: Box<dyn RowCursor + 'e>,
    right_node: &'e minisqlite_plan::PlanNode,
    on: Option<&'e EvalExpr>,
    env: Env<'e>,
    /// The correlated `outer` prefix (empty at the top level). The right BUILD side is
    /// drained with [`NO_OUTER`], so this is not used to build it; it re-prepends the
    /// prefix on a `Right`/`Full` null-fill and sizes the `Left`/`Full` fill assertion.
    /// The prefix on a matched pair rides along with the probing left row.
    outer: &'e [Value],
    left_keys: &'e [EvalExpr],
    right_keys: &'e [EvalExpr],
    key_collations: &'e [Collation],
    left_width: usize,
    right_width: usize,
    need_left_fill: bool,
    need_right_fill: bool,

    built: bool,
    /// Every right row, in build order; the index space `flat_buckets`/`right_matched` use.
    right_rows: Vec<Row>,
    /// Equijoin key → the `[start, start+len)` slice of `flat_buckets` holding that
    /// key's right-row indices (non-NULL-key rows only). A probe stores the matched
    /// key's `(start, len)` — two `usize`s, `Copy` — instead of cloning the bucket, so
    /// there is no per-probe allocation on the hottest path (see `build`).
    table: HashMap<Vec<CellKey>, (usize, usize)>,
    /// All buckets' `right_rows` indices concatenated, grouped by key: the range
    /// `table[key]` names is contiguous here. One flat allocation shared by every
    /// probe (a probe iterates its range by index, holding no borrow of it across the
    /// `&mut rt` pulls — the fields it advances are plain `usize`s).
    flat_buckets: Vec<usize>,
    /// Which `right_rows[i]` have been emitted (for `Right`/`Full`).
    right_matched: Vec<bool>,

    /// The left row currently probing; `None` between rows.
    cur_left: Option<Row>,
    cur_left_matched: bool,
    /// The current left row's matching-bucket range in `flat_buckets`
    /// (`cur_start .. cur_start + cur_len`) and the position within it. Plain `usize`s
    /// (no owned bucket), so advancing them never conflicts with borrowing
    /// `flat_buckets` / `right_rows` / `rt` in the emit loop.
    cur_start: usize,
    cur_len: usize,
    cur_pos: usize,
    /// Reused probe-key scratch: each left row's equijoin key is folded into this buffer
    /// (cleared first) and the table is looked up by its SLICE, so a probe makes no
    /// per-row `Vec<CellKey>` spine allocation. `HashMap<Vec<CellKey>, _>::get` accepts a
    /// `&[CellKey]` and hashes it identically to the stored `Vec`, so the lookup is exact.
    /// The build side still owns its keys (they are moved into the table), so only the
    /// probe path reuses this — the hottest loop (once per probe row).
    probe_key: Vec<CellKey>,

    emit_right_pos: usize,
}

impl HashJoinCursor<'_> {
    /// Drain the right build side once, indexing non-NULL-key rows. Built with an EMPTY
    /// outer ([`NO_OUTER`]) so each stored right row is its own columns with no prefix:
    /// the planner rebased every right key by `base_offset + left_width`, so the keys
    /// index that standalone row, and the single `outer` prefix rides on the probing
    /// left row (matched pairs) or is re-prepended by [`fill_left`] (unmatched right).
    fn build(&mut self, rt: &mut Runtime) -> Result<()> {
        let mut right = build_cursor(self.right_node, self.env, NO_OUTER)?;
        let mut rows: Vec<Row> = Vec::new();
        // Group right-row indices by key first, then flatten into one contiguous array
        // so a probe names its bucket by a `(start, len)` range instead of cloning a
        // per-key `Vec`. Grouping preserves each bucket's build order (ascending index),
        // so a probe emits its matches in the same order as before — the result is
        // byte-identical, only the storage changed.
        let mut groups: HashMap<Vec<CellKey>, Vec<usize>> = HashMap::new();
        while let Some(r) = right.next_row(rt)? {
            let idx = rows.len();
            if let Some(key) = eval_cell_keys(rt, self.env, self.right_keys, self.key_collations, &r)? {
                groups.entry(key).or_default().push(idx);
            }
            // A NULL-key row is not indexed (unmatchable) but is still stored so a
            // Right/Full join can null-fill it in phase B.
            rows.push(r);
        }
        // Flatten: concatenate each key's indices into `flat_buckets` and record its
        // `(start, len)` range in `table`. `append` moves each bucket's elements once
        // (emptying and freeing the small per-key `Vec`), so the whole index set ends
        // up in one allocation the probe side borrows by range.
        let mut flat_buckets: Vec<usize> = Vec::with_capacity(rows.len());
        let mut table: HashMap<Vec<CellKey>, (usize, usize)> =
            HashMap::with_capacity(groups.len());
        for (key, mut idxs) in groups {
            let start = flat_buckets.len();
            let len = idxs.len();
            flat_buckets.append(&mut idxs);
            table.insert(key, (start, len));
        }
        // Only Right/Full read `right_matched`; skip the allocation for Inner/Left.
        self.right_matched = if self.need_right_fill {
            vec![false; rows.len()]
        } else {
            Vec::new()
        };
        self.right_rows = rows;
        self.table = table;
        self.flat_buckets = flat_buckets;
        self.built = true;
        Ok(())
    }

    /// Pull the next probe (left) row and set up its bucket range: fold the probe key into
    /// the reused `probe_key` buffer, look the table up by SLICE (no per-probe key
    /// allocation), and set `cur_start`/`cur_len`/`cur_pos`/`cur_left`. Returns `false`
    /// when the left side is exhausted. Shared by `next_row` and `count_rows` so the two
    /// cannot drift on the probe-key lookup. Does NOT touch `cur_left_matched` — only the
    /// outer-fill path in `next_row` reads it, and it resets it itself.
    fn load_next_probe(&mut self, rt: &mut Runtime) -> Result<bool> {
        match self.left.next_row(rt)? {
            Some(l) => {
                // A NULL left key, or a key absent from the table, matches nothing (an
                // empty range). The probe key is canonicalized the same way the build side
                // was, so a present key finds the exact bucket.
                let (start, len) = if eval_cell_keys_into(
                    rt,
                    self.env,
                    self.left_keys,
                    self.key_collations,
                    &l,
                    &mut self.probe_key,
                )? {
                    self.table.get(self.probe_key.as_slice()).copied().unwrap_or((0, 0))
                } else {
                    (0, 0)
                };
                self.cur_start = start;
                self.cur_len = len;
                self.cur_pos = 0;
                self.cur_left = Some(l);
                Ok(true)
            }
            None => Ok(false),
        }
    }
}

impl RowCursor for HashJoinCursor<'_> {
    fn next_row(&mut self, rt: &mut Runtime) -> Result<Option<Row>> {
        if !self.built {
            self.build(rt)?;
        }

        loop {
            // Emit remaining matches for the current left row. `cur_start`/`cur_len`
            // name a range of `flat_buckets`; indexing it fresh each step holds no
            // borrow across `try_pair`'s `&mut rt` pulls (the range fields are `usize`),
            // so there is no per-probe bucket clone.
            while self.cur_pos < self.cur_len {
                let idx = self.flat_buckets[self.cur_start + self.cur_pos];
                self.cur_pos += 1;
                let pair = {
                    let l = self.cur_left.as_ref().expect("cur_left present during probe");
                    try_pair(self.on, self.env, rt, l, &self.right_rows[idx])?
                };
                if let Some(row) = pair {
                    self.cur_left_matched = true;
                    if self.need_right_fill {
                        self.right_matched[idx] = true;
                    }
                    return Ok(Some(row));
                }
            }

            // Bucket exhausted: null-fill an unmatched left row (Left/Full), then
            // advance. `take` clears cur_left so the next pull loads a new one.
            if let Some(l) = self.cur_left.take() {
                if self.need_left_fill && !self.cur_left_matched {
                    return Ok(Some(fill_right(
                        &l,
                        self.outer.len(),
                        self.left_width,
                        self.right_width,
                    )));
                }
            }

            if self.load_next_probe(rt)? {
                self.cur_left_matched = false;
            } else {
                break;
            }
        }

        // Phase B: right rows that matched no left (Right/Full).
        if self.need_right_fill {
            return Ok(next_unmatched_right(
                &mut self.emit_right_pos,
                self.outer,
                &self.right_rows,
                &self.right_matched,
                self.left_width,
                self.right_width,
            ));
        }
        Ok(None)
    }

    /// Count the join's rows without materializing a combined `Row` per pair — the
    /// alloc-free path a `count(*)`-over-join consumer takes (see
    /// [`RowCursor::count_rows`]). For an INNER equijoin (no outer fills) the count is
    /// exactly the number of matched `(probe, bucket-entry)` pairs, so it is accumulated
    /// by walking each probe's bucket range and never building the pair row. A residual
    /// `on` (a predicate the hash keys did not fully capture) is evaluated against a
    /// REUSED scratch row, so even that case allocates no per-pair row. `Left`/`Right`/
    /// `Full` additionally emit null-filled rows whose count the buckets do not capture,
    /// so they defer to the default drain (correct, just not alloc-free). The pair count
    /// and the `on` evaluation match [`RowCursor::next_row`]'s inner path exactly, so the
    /// returned count is identical — this changes cost, not results.
    fn count_rows(&mut self, rt: &mut Runtime) -> Result<usize> {
        if self.need_left_fill || self.need_right_fill {
            let mut n = 0usize;
            while self.next_row(rt)?.is_some() {
                n += 1;
            }
            return Ok(n);
        }
        if !self.built {
            self.build(rt)?;
        }
        let mut count = 0usize;
        // Reused only for the residual-`on` case; stays empty (no allocation) otherwise.
        let mut scratch: Row = Vec::new();
        loop {
            // Count the in-flight probe's remaining matches. A freshly built cursor has
            // `cur_left == None`, so this is skipped until the first probe is loaded; it
            // also correctly resumes if some rows were already pulled via `next_row`.
            if self.cur_left.is_some() {
                match self.on {
                    // ON fully captured by the hash keys: EVERY entry in the bucket
                    // matches (this mirrors `try_pair(None, ..)`, which never evaluates a
                    // predicate). So count the whole remaining range at once — O(1) per
                    // probe instead of one bounds-checked read + increment per pair — and
                    // skip to the range end.
                    None => {
                        count += self.cur_len - self.cur_pos;
                        self.cur_pos = self.cur_len;
                    }
                    // Residual ON the hash keys did not fully capture: evaluate it per pair
                    // against the combined row (reused `scratch` buffer), routed through
                    // `on_keeps` — the single home of the join-`on` 3VL rule — so the count
                    // path cannot drift from `next_row`/`try_pair`.
                    Some(_) => {
                        let l = self.cur_left.as_ref().expect("cur_left present during probe");
                        while self.cur_pos < self.cur_len {
                            let idx = self.flat_buckets[self.cur_start + self.cur_pos];
                            self.cur_pos += 1;
                            scratch.clear();
                            scratch.extend_from_slice(l);
                            scratch.extend_from_slice(&self.right_rows[idx]);
                            if on_keeps(self.on, self.env, rt, &scratch)? {
                                count += 1;
                            }
                        }
                    }
                }
            }
            // Advance to the next probe row — shared with `next_row` so the probe-key
            // lookup cannot drift.
            if !self.load_next_probe(rt)? {
                break;
            }
        }
        self.cur_left = None;
        Ok(count)
    }
}

// ---- IndexNestedLoop ------------------------------------------------------

fn index_nested_loop<'e>(
    j: &'e Join,
    env: Env<'e>,
    outer: &'e [Value],
) -> Result<Box<dyn RowCursor + 'e>> {
    // The right subtree is a per-left seek keyed off the current left row, so it is
    // not a fixed set that could be materialized once for match-tracking. Right/Full
    // (which must emit unmatched right rows) therefore has no correct shape here;
    // fail loud rather than silently drop rows.
    let (need_left_fill, need_right_fill) = outer_fills(&j.join_type);
    if need_right_fill {
        return Err(Error::sql(
            "IndexNestedLoop join does not support RIGHT/FULL (the right side is rebuilt per \
             left row; the planner should choose a materializing strategy for these)",
        ));
    }
    let left = build_cursor(&j.left, env, outer)?;
    Ok(Box::new(IndexNestedLoopCursor {
        left,
        right_node: &j.right,
        on: j.on.as_ref(),
        env,
        outer,
        left_width: j.left_width,
        right_width: j.right_width,
        need_left_fill,
        pending: Vec::new().into_iter(),
    }))
}

/// Index nested-loop join. For each left row `L`, the right subtree is built with
/// `outer = &L`, so its leaf prepends `L` and each right row it yields is ALREADY
/// `L ++ right_local` (width `WL + WR`). One left row's right matches are buffered in
/// `pending` (bounded by a single seek), filtered by `on`, and streamed out; the
/// whole right is never materialized.
struct IndexNestedLoopCursor<'e> {
    left: Box<dyn RowCursor + 'e>,
    right_node: &'e minisqlite_plan::PlanNode,
    on: Option<&'e EvalExpr>,
    env: Env<'e>,
    /// The correlated `outer` prefix (empty at the top level — the planner only chooses
    /// IndexNestedLoop at `base_offset == 0`). The per-left seek uses the WHOLE left row
    /// `l` (which already carries this prefix) as its outer, so each right row it yields
    /// is `outer ++ left_local ++ right_local` and the single prefix is preserved; this
    /// field only sizes the `Left`/`Full` fill's debug assertion.
    outer: &'e [Value],
    left_width: usize,
    right_width: usize,
    need_left_fill: bool,
    /// Combined rows for the current left row, awaiting emission.
    pending: std::vec::IntoIter<Row>,
}

impl RowCursor for IndexNestedLoopCursor<'_> {
    fn next_row(&mut self, rt: &mut Runtime) -> Result<Option<Row>> {
        loop {
            if let Some(row) = self.pending.next() {
                return Ok(Some(row));
            }
            let Some(l) = self.left.next_row(rt)? else { return Ok(None) };

            // Build the right seek with the current left row as `outer`. `build_cursor`
            // is generic and `Env` is covariant, so it instantiates at `l`'s (short)
            // lifetime; we fully drain the cursor within this scope, so nothing
            // borrows `l` after it (no self-referential state, no unsafe).
            let mut matches: Vec<Row> = Vec::new();
            {
                let mut right = build_cursor(self.right_node, self.env, &l)?;
                while let Some(r) = right.next_row(rt)? {
                    // `r` is already `l ++ right_local`; keep it iff the residual `on`
                    // holds under 3VL (same rule NestedLoop/Hash apply via `try_pair`).
                    if on_keeps(self.on, self.env, rt, &r)? {
                        matches.push(r);
                    }
                }
            }
            if matches.is_empty() && self.need_left_fill {
                matches.push(fill_right(&l, self.outer.len(), self.left_width, self.right_width));
            }
            self.pending = matches.into_iter();
        }
    }
}
