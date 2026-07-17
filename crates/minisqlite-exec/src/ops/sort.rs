//! `Sort` — order the input by a list of [`SortKey`]s.
//!
//! Two paths, one visible behavior:
//!
//! - **Full sort** (no usable `LIMIT`): on the first pull, drain the whole input,
//!   compute each row's sort keys once, stable-sort, then stream the ordered rows.
//!   Bounded by the input size — this is one of the two operators allowed to
//!   materialize (the other is `distinct`).
//! - **Bounded top-k** (a `LIMIT` constrains it): keep only the `retain = offset + limit`
//!   rows a full sort would place first, in a capacity-`retain` max-heap
//!   ([`TopK`]), so peak retained memory is `O(retain)` — NOT `O(N)` — by construction.
//!   The emitted rows are BYTE-IDENTICAL to the full sort then the `LIMIT`: the retained
//!   set is the `retain` smallest rows under the total order "`ORDER BY` keys, then input
//!   order for ties", which is exactly the first-`retain` prefix of the stable full sort
//!   (see [`HeapItem`]). The [`Limit`](minisqlite_plan::PlanNode::Limit) node above still
//!   applies the real skip/take; this operator only avoids buffering rows it can prove the
//!   answer discards.
//!
//! The retention bound is a pure OPTIMIZATION and never changes results: when it cannot be
//! used safely (a non-deterministic or negative/NULL/non-integral bound) the operator
//! falls back to the full sort. See [`SortCursor::retain_bound`] for why that fallback is
//! always correct.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::vec::IntoIter;

use minisqlite_expr::{eval, EvalExpr, SortKey};
use minisqlite_plan::SortLimit;
use minisqlite_types::{Result, Row, Value};

use crate::context::EvalCtx;
use crate::env::Env;
use crate::ops::must_be_int::{must_be_int, MustBeInt};
use crate::ops::sortkey;
use crate::runtime::Runtime;
use crate::RowCursor;

/// The initial heap capacity is capped here so a huge but valid `LIMIT` (e.g. `LIMIT
/// 9223372036854775807`) does not try to pre-allocate a gigantic heap up front. The heap
/// still GROWS to hold `min(retain, N)` rows as they arrive, so peak memory stays
/// `O(min(retain, N))` = `O(retain)`; this only bounds the *eager* allocation.
const RETAIN_PREALLOC_CAP: usize = 1024;

/// Build a sort of `input` by `keys`. `limit`, when `Some`, is the retention bound the
/// enclosing `LIMIT`/`OFFSET` places on the sort (a bounded top-k); `None` is a full
/// stable sort.
pub(crate) fn sort<'a>(
    env: Env<'a>,
    keys: &'a [SortKey],
    limit: Option<&'a SortLimit>,
    input: Box<dyn RowCursor + 'a>,
) -> Box<dyn RowCursor + 'a> {
    Box::new(SortCursor { env, keys, limit, input, ordered: None })
}

struct SortCursor<'a> {
    env: Env<'a>,
    keys: &'a [SortKey],
    /// The retention bound (`offset + limit`) when a `LIMIT` constrains this sort, or
    /// `None` for a full sort. Borrowed from the plan, so its `EvalExpr`s live as long as
    /// the cursor.
    limit: Option<&'a SortLimit>,
    input: Box<dyn RowCursor + 'a>,
    /// The materialized, ordered `(key_values, row)` pairs, produced on the first pull.
    /// `None` until then; the keys ride along and are dropped as rows stream out.
    ordered: Option<IntoIter<(Vec<Value>, Row)>>,
}

impl RowCursor for SortCursor<'_> {
    fn next_row(&mut self, rt: &mut Runtime) -> Result<Option<Row>> {
        if self.ordered.is_none() {
            self.ordered = Some(self.materialize(rt)?);
        }
        // `ordered` is Some by construction above; stream the next buffered row and
        // drop its now-spent sort keys.
        Ok(self.ordered.as_mut().expect("ordered set on first pull").next().map(|(_, row)| row))
    }
}

impl<'a> SortCursor<'a> {
    /// Produce the ordered `(keys, row)` pairs on the first pull, choosing the bounded
    /// top-k when a usable `LIMIT` is present and the full sort otherwise. Both paths
    /// return the SAME ordered stream shape (the caller drops the keys as it yields each
    /// row), and — over any input — emit byte-identical rows.
    fn materialize(&mut self, rt: &mut Runtime) -> Result<IntoIter<(Vec<Value>, Row)>> {
        match self.retain_bound(rt)? {
            Some(retain) => self.materialize_bounded(rt, retain),
            None => self.materialize_full(rt),
        }
    }

    /// The retention count `retain = offset + limit` to keep, or `None` to fall back to a
    /// full sort.
    ///
    /// Evaluated ONCE, up front, with an empty row — the bound expressions were bound with
    /// no columns in scope, exactly like the [`Limit`](minisqlite_plan::PlanNode::Limit)
    /// node — using [`must_be_int`] (SQLite's `OP_MustBeInt`) so the sort agrees with the
    /// limit on the count. Absent/negative `OFFSET` counts as 0; a `LIMIT` that is not a
    /// clean NON-NEGATIVE integer yields `None` (full sort).
    ///
    /// # Why the fallback is always correct
    /// The `Limit` node sits directly above this sort (the compiler attaches a bound ONLY
    /// when it also builds that `Limit`), and `Limit::init` evaluates and VALIDATES the
    /// same bound expressions BEFORE it pulls its input — i.e. before this sort ever
    /// materializes. So:
    /// - A present NULL / fractional / non-numeric `LIMIT` or `OFFSET` has already errored
    ///   at the `Limit`; this sort is never pulled, and the defensive `None` here would be
    ///   harmless anyway.
    /// - A negative `LIMIT` is unbounded at the `Limit`, so a full sort (all rows, ordered)
    ///   is exactly right.
    /// - The compiler only attaches the bound when the expressions are DETERMINISTIC, so
    ///   this sort's own single evaluation returns the identical value the `Limit` used,
    ///   making `retain == offset + limit` precisely and the top-k a byte-identical prefix
    ///   of the full sort. (A non-deterministic bound is never attached — the sort stays a
    ///   full sort — so it cannot retain fewer rows than the `Limit` later takes.)
    ///
    /// This never duplicates the `Limit`'s error/skip/take semantics: the `Limit` remains
    /// the single source of truth; this is only a memory hint.
    fn retain_bound(&self, rt: &mut Runtime) -> Result<Option<usize>> {
        let bound = match self.limit {
            Some(b) => b,
            None => return Ok(None),
        };
        // Absent OFFSET => 0; a clean integer offset contributes max(0, n) (a negative
        // offset is 0, mirroring `Limit`); anything else => not a clean bound => full sort.
        let offset = match &bound.offset {
            None => 0i64,
            Some(e) => match must_be_int(&self.eval_bound(rt, e)?) {
                MustBeInt::Int(n) => n.max(0),
                MustBeInt::Null | MustBeInt::NotInt => return Ok(None),
            },
        };
        // A clean non-negative LIMIT bounds the sort; a negative (unbounded), NULL, or
        // non-integral LIMIT falls back to the full sort (see the doc comment for why
        // that is safe in every case). The sign/coercion rules here MIRROR
        // `LimitCursor::init` (ops/limit.rs): negative OFFSET => 0, negative LIMIT =>
        // unbounded. If that node's sign semantics ever change, re-check this fallback.
        let limit = match must_be_int(&self.eval_bound(rt, &bound.limit)?) {
            MustBeInt::Int(n) if n >= 0 => n,
            // Negative LIMIT (unbounded), plus NULL / non-integral (which already errored at
            // the Limit node before this sort was pulled): all fall back to the full sort.
            // Matched as explicitly as the offset arm above (no `_`) so a future `MustBeInt`
            // variant forces a decision here instead of silently taking this fallback.
            MustBeInt::Int(_) | MustBeInt::Null | MustBeInt::NotInt => return Ok(None),
        };
        let retain = offset.saturating_add(limit);
        // On a 64-bit target every non-negative i64 fits in usize; the `.ok()` fallback to
        // a full sort covers the (practically impossible) narrow-usize truncation.
        Ok(usize::try_from(retain).ok())
    }

    /// Evaluate a single `LIMIT`/`OFFSET` bound expression against an empty row (the
    /// bounds reference no columns), mirroring `Limit`'s own evaluation.
    fn eval_bound(&self, rt: &mut Runtime, e: &EvalExpr) -> Result<Value> {
        let mut ctx = EvalCtx { rt, env: self.env, outer: &[] };
        eval(e, &[], &mut ctx)
    }

    /// Bounded top-k: pull every input row (top-k must see them all), keep only the
    /// `retain` smallest in a capacity-`retain` heap, then stream them in order. Peak
    /// retained memory is `O(retain)`, not `O(N)`.
    fn materialize_bounded(
        &mut self,
        rt: &mut Runtime,
        retain: usize,
    ) -> Result<IntoIter<(Vec<Value>, Row)>> {
        let mut topk = TopK::new(retain);
        // The input sequence index is the stability tiebreak: lower = earlier in the
        // input, so ties resolve to input order exactly as the stable full sort does.
        let mut seq: usize = 0;
        while let Some(row) = self.input.next_row(rt)? {
            let keys = self.eval_keys(rt, &row)?;
            topk.offer(HeapItem { keys, seq, row, sort_keys: self.keys });
            seq += 1;
        }
        Ok(topk.into_sorted_pairs().into_iter())
    }

    /// Full stable sort: drain the whole input, compute each row's key values once, and
    /// stable-sort by the multi-key comparator.
    fn materialize_full(&mut self, rt: &mut Runtime) -> Result<IntoIter<(Vec<Value>, Row)>> {
        let mut buf: Vec<(Vec<Value>, Row)> = Vec::new();
        while let Some(row) = self.input.next_row(rt)? {
            let keys = self.eval_keys(rt, &row)?;
            buf.push((keys, row));
        }
        // Stable sort so rows equal on all keys keep input order (matches SQLite's
        // otherwise-unspecified but stable-in-practice ordering for ties). The comparator
        // lives in `sortkey` so `Sort` and aggregate `ORDER BY` share one set of
        // NULL/collation/direction rules.
        buf.sort_by(|a, b| sortkey::compare_sort_keys(&a.0, &b.0, self.keys));
        Ok(buf.into_iter())
    }

    /// Compute one row's sort-key values (one per key). Shared by both materialize paths
    /// so the two agree on key evaluation exactly.
    fn eval_keys(&self, rt: &mut Runtime, row: &Row) -> Result<Vec<Value>> {
        let mut key_vals = Vec::with_capacity(self.keys.len());
        let mut ctx = EvalCtx { rt, env: self.env, outer: row };
        for k in self.keys {
            key_vals.push(eval(&k.expr, row, &mut ctx)?);
        }
        Ok(key_vals)
    }
}

/// One buffered candidate in the bounded top-k heap: its precomputed sort-key values, the
/// input sequence index (`seq` — the stability tiebreak, lower = earlier), the row, and a
/// borrow of the sort modifiers so [`Ord`] can rank two items.
///
/// `Ord` IS the desired OUTPUT ORDER — `compare_sort_keys` on the keys, then ascending
/// `seq` for ties — NOT its reverse. `std::collections::BinaryHeap` is a MAX-heap, so its
/// root is the greatest item, i.e. the one that sorts LAST: exactly the row to EVICT to
/// keep the `retain` SMALLEST (first-k) rows. Two items compare `Equal` only when both
/// their keys AND `seq` match, which never happens (`seq` is unique per row), so the order
/// is total and the retained set — and its final order — is deterministic and equals the
/// first-`retain` prefix of the stable full sort.
struct HeapItem<'k> {
    keys: Vec<Value>,
    seq: usize,
    row: Row,
    sort_keys: &'k [SortKey],
}

impl HeapItem<'_> {
    fn output_cmp(&self, other: &Self) -> Ordering {
        sortkey::compare_sort_keys(&self.keys, &other.keys, self.sort_keys)
            .then(self.seq.cmp(&other.seq))
    }
}

impl Ord for HeapItem<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.output_cmp(other)
    }
}

impl PartialOrd for HeapItem<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for HeapItem<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.output_cmp(other) == Ordering::Equal
    }
}

impl Eq for HeapItem<'_> {}

/// A bounded top-k accumulator: a max-heap that retains only the `retain` smallest items
/// offered to it (smallest in [`HeapItem`] output order). The whole memory win of the
/// operator lives here — the heap never holds more than `retain` rows, so peak retained
/// memory is `O(retain)`, independent of how many rows are offered.
struct TopK<'k> {
    retain: usize,
    heap: BinaryHeap<HeapItem<'k>>,
    /// The greatest retained size observed — a test-only witness the non-materialization
    /// unit tests assert on. Gated out of non-test builds so it costs the hot `offer` path
    /// nothing.
    #[cfg(test)]
    peak: usize,
}

impl<'k> TopK<'k> {
    fn new(retain: usize) -> Self {
        // Cap the EAGER allocation so a huge-but-valid LIMIT does not pre-allocate a giant
        // heap; it still grows to hold `min(retain, N)` rows, keeping peak memory O(retain).
        let cap = retain.saturating_add(1).min(RETAIN_PREALLOC_CAP);
        TopK {
            retain,
            heap: BinaryHeap::with_capacity(cap),
            #[cfg(test)]
            peak: 0,
        }
    }

    /// Offer one candidate, keeping the heap at the `retain` smallest items seen so far.
    ///
    /// Below capacity, push. Once full, the COMMON case — a row no better than the current
    /// worst kept (the max-heap root) — is a single O(1) comparison with NO heap mutation;
    /// only a row that displaces the worst pays one O(log retain) sift-down (via `PeekMut`,
    /// which re-sifts on drop ONLY when the root was actually mutated). That is strictly less
    /// work than an unconditional push-then-pop (two heap operations) on every row, and it
    /// never transiently exceeds `retain`.
    ///
    /// Stability is preserved: `seq` strictly increases, so a later row whose keys equal the
    /// root compares GREATER (never `< *worst`) and is rejected — ties keep the earliest
    /// rows, exactly as the stable full sort does. `retain == 0` keeps nothing (the fill
    /// branch is skipped and `peek_mut` on the empty heap is a no-op).
    fn offer(&mut self, item: HeapItem<'k>) {
        if self.heap.len() < self.retain {
            self.heap.push(item);
        } else if let Some(mut worst) = self.heap.peek_mut() {
            // At capacity: replace the worst kept row iff the new one is strictly smaller in
            // output order, else drop it in O(1) (no heap mutation).
            if item < *worst {
                *worst = item;
            }
        }
        #[cfg(test)]
        {
            self.peak = self.peak.max(self.heap.len());
        }
        // The O(retain) invariant, checked on the REAL path in every debug (test) build: a
        // materializing regression (buffering all N) trips this immediately.
        debug_assert!(
            self.heap.len() <= self.retain,
            "TopK retained {} rows, exceeding retain {}",
            self.heap.len(),
            self.retain
        );
    }

    /// Drain the heap into `(keys, row)` pairs in ascending output order (the stream
    /// order). `BinaryHeap::into_sorted_vec` sorts ascending by the item `Ord`, which is
    /// the output order, so this is exactly the first-`retain` prefix of the stable full
    /// sort, in order.
    fn into_sorted_pairs(self) -> Vec<(Vec<Value>, Row)> {
        self.heap.into_sorted_vec().into_iter().map(|it| (it.keys, it.row)).collect()
    }
}

#[cfg(test)]
mod tests {
    use minisqlite_catalog::SchemaCatalog;
    use minisqlite_pager::MemPager;
    use minisqlite_plan::{Plan, PlanNode};
    use minisqlite_types::{Collation, DbIndex, Value};

    use super::*;

    fn key(desc: bool) -> SortKey {
        SortKey { expr: EvalExpr::Column(0), desc, nulls_first: None, collation: Collation::Binary }
    }

    // ---- TopK: the memory-bound + correctness witness on the real structure ----

    /// A `HeapItem` over a single integer key column, for the structure tests.
    fn item<'k>(k: i64, seq: usize, keys: &'k [SortKey]) -> HeapItem<'k> {
        HeapItem { keys: vec![Value::Integer(k)], seq, row: vec![Value::Integer(k)], sort_keys: keys }
    }

    #[test]
    fn topk_never_retains_more_than_k_and_keeps_the_k_smallest() {
        // The non-materialization proof: offer 5000 shuffled keys with retain=3 and assert
        // the heap NEVER holds more than 3 rows (a full-materialize regression would hold
        // ~5000 and fail here), and that the survivors are the 3 smallest, in order.
        let keys = [key(false)];
        let retain = 3;
        let mut topk = TopK::new(retain);
        // A deterministic shuffle of 0..5000 (a full-period LCG step, coprime to 5000).
        let n: i64 = 5000;
        let mut x: i64 = 1234;
        for seq in 0..n as usize {
            x = (x + 2_237) % n; // step by a value coprime to 5000 => visits every residue
            topk.offer(item(x, seq, &keys));
            assert!(topk.len() <= retain, "retained {} > {}", topk.len(), retain);
        }
        assert_eq!(topk.peak(), retain, "steady retained size is exactly retain once filled");
        let got: Vec<i64> = topk
            .into_sorted_pairs()
            .into_iter()
            .map(|(_, row)| match row[0] {
                Value::Integer(i) => i,
                ref other => panic!("expected Integer, got {other:?}"),
            })
            .collect();
        assert_eq!(got, vec![0, 1, 2], "the 3 smallest keys, ascending");
    }

    #[test]
    fn topk_breaks_ties_by_input_order_lowest_seq_wins() {
        // All keys equal => the retained set is decided purely by the stability tiebreak:
        // keep the EARLIEST rows (lowest seq), in seq order. This is what makes the bounded
        // result byte-identical to the stable full sort truncated to k. The row payload
        // carries `seq` so the kept rows' original positions are directly observable.
        let keys = [key(false)];
        let retain = 3;
        let mut topk = TopK::new(retain);
        for seq in 0..1000usize {
            let it = HeapItem {
                keys: vec![Value::Integer(42)], // identical key for every row
                seq,
                row: vec![Value::Integer(seq as i64)],
                sort_keys: &keys,
            };
            topk.offer(it);
            assert!(topk.len() <= retain);
        }
        let kept_seqs: Vec<i64> = topk
            .into_sorted_pairs()
            .into_iter()
            .map(|(_, row)| match row[0] {
                Value::Integer(i) => i,
                ref other => panic!("expected Integer, got {other:?}"),
            })
            .collect();
        assert_eq!(kept_seqs, vec![0, 1, 2], "ties keep the earliest rows, in input order");
    }

    #[test]
    fn topk_retain_zero_keeps_nothing() {
        let keys = [key(false)];
        let mut topk = TopK::new(0);
        for seq in 0..10usize {
            topk.offer(item(seq as i64, seq, &keys));
            assert_eq!(topk.len(), 0, "retain 0 holds nothing");
        }
        assert!(topk.into_sorted_pairs().is_empty());
    }

    impl TopK<'_> {
        fn len(&self) -> usize {
            self.heap.len()
        }
        fn peak(&self) -> usize {
            self.peak
        }
    }

    // ---- SortCursor end-to-end: the bounded cursor emits the correct first-k ----

    /// An `n`-row single-column source yielding descending keys `n-1, n-2, …, 0`, so the
    /// correct ascending top-k is the LAST rows pulled — a materialize-then-truncate that
    /// ignored ordering would get this wrong.
    struct DescSource {
        next: i64,
    }

    impl RowCursor for DescSource {
        fn next_row(&mut self, _rt: &mut Runtime) -> Result<Option<Row>> {
            if self.next < 0 {
                return Ok(None);
            }
            let v = self.next;
            self.next -= 1;
            Ok(Some(vec![Value::Integer(v)]))
        }
    }

    #[test]
    fn sort_cursor_bounded_emits_first_k_over_a_large_source() {
        // Drive the REAL cursor over 5000 rows with LIMIT 3. Correctness: the emitted rows
        // are the 3 smallest, ascending. Memory: the `debug_assert!` in `TopK::offer` runs
        // on this real path in the (debug) test build and trips if the heap ever exceeds
        // retain — the end-to-end non-materialization guard.
        let pager = MemPager::new(4096);
        let cat = SchemaCatalog::new();
        let plan = Plan {
            root: PlanNode::SingleRow,
            result_columns: Vec::new(),
            ctes: Vec::new(),
            subqueries: Vec::new(),
            mutates: false,
            generated: Vec::new(),
        };
        let env = Env {
            catalog: &cat,
            pagers: crate::env::Pagers::One { db: DbIndex::MAIN, pager: &pager },
            plan: &plan,
        };
        let keys = [key(false)];
        let sl = SortLimit { limit: EvalExpr::Literal(Value::Integer(3)), offset: None };
        let source = DescSource { next: 4999 };

        let mut cur = sort(env, &keys, Some(&sl), Box::new(source));
        let mut rt = Runtime::new();
        let mut got = Vec::new();
        while let Some(row) = cur.next_row(&mut rt).unwrap() {
            got.push(match row[0] {
                Value::Integer(i) => i,
                ref other => panic!("expected Integer, got {other:?}"),
            });
        }
        assert_eq!(got, vec![0, 1, 2], "bounded sort emits the 3 smallest, ascending");
    }

    #[test]
    fn sort_cursor_bounded_matches_full_sort_with_offset() {
        // retain = offset + limit: `LIMIT 3 OFFSET 2` keeps the first 5 ordered rows; the
        // cursor here emits those 5 (the Limit node, absent in this unit, would then skip
        // 2 and take 3). Compare against the full ascending order truncated to 5.
        let pager = MemPager::new(4096);
        let cat = SchemaCatalog::new();
        let plan = Plan {
            root: PlanNode::SingleRow,
            result_columns: Vec::new(),
            ctes: Vec::new(),
            subqueries: Vec::new(),
            mutates: false,
            generated: Vec::new(),
        };
        let env = Env {
            catalog: &cat,
            pagers: crate::env::Pagers::One { db: DbIndex::MAIN, pager: &pager },
            plan: &plan,
        };
        let keys = [key(false)];
        let sl = SortLimit {
            limit: EvalExpr::Literal(Value::Integer(3)),
            offset: Some(EvalExpr::Literal(Value::Integer(2))),
        };
        let source = DescSource { next: 99 };

        let mut cur = sort(env, &keys, Some(&sl), Box::new(source));
        let mut rt = Runtime::new();
        let mut got = Vec::new();
        while let Some(row) = cur.next_row(&mut rt).unwrap() {
            got.push(match row[0] {
                Value::Integer(i) => i,
                ref other => panic!("expected Integer, got {other:?}"),
            });
        }
        assert_eq!(got, vec![0, 1, 2, 3, 4], "retain = offset + limit = 5 smallest, ascending");
    }
}
