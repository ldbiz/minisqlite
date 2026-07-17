//! `SetOp` — the compound set operators `UNION ALL`, `UNION`, `INTERSECT`, and
//! `EXCEPT`, as pull-based streaming [`RowCursor`](crate::RowCursor)s over two
//! same-width inputs (per the plan convention the two children have equal width and
//! the output is that width; result column *names* come from the left and are the
//! planner's job, so this operator only produces rows).
//!
//! ## Row equality (the dedup contract)
//! Every operator except `UNION ALL` collapses duplicate rows. Two rows are the
//! "same row" when their [`row_key`] under `column_collations` is equal — which folds
//! numeric-equal values across storage classes (`2` and `2.0`), folds text under each
//! column's collation, and treats `NULL` as a normal self-equal value. That last
//! point is why compound-select dedup unions two `NULL` rows into one, *unlike* SQL
//! `=` where `NULL = NULL` is not true; the shared key encodes exactly SQLite's
//! compound-select comparison, so this operator never special-cases `NULL`.
//!
//! ## Streaming vs. materialization (the performance contract)
//! * `UNION ALL` concatenates and buffers **nothing** — it streams every left row,
//!   then every right row.
//! * `UNION` streams the same concatenation but keeps a set of the row keys it has
//!   already emitted (bounded by the result cardinality), so each distinct row is
//!   emitted once, left-first.
//! * `INTERSECT` / `EXCEPT` are asymmetric: they must know the whole RIGHT input
//!   before deciding any left row, so on the first pull they drain the right child
//!   once into a key set (the one required materialization, bounded by the right
//!   cardinality) and then stream the LEFT. Both are `DISTINCT`, so a second set
//!   dedups the rows they emit. Neither ever materializes the left, and no operator
//!   here materializes both inputs.

use std::collections::HashSet;

use minisqlite_plan::{SetOp, SetOpKind};
use minisqlite_types::{Collation, Result, Row, Value};

use crate::env::Env;
use crate::keys::{row_key, CellKey};
use crate::runner::build_cursor;
use crate::runtime::Runtime;
use crate::RowCursor;

/// Build the streaming cursor for a `SetOp` node. Both children are built eagerly
/// (building needs no [`Runtime`]); the right-side drain that `INTERSECT`/`EXCEPT`
/// require happens lazily on the first pull, where the `rt` needed to drive a child
/// is available.
pub(crate) fn set_op<'e>(
    s: &'e SetOp,
    env: Env<'e>,
    outer: &'e [Value],
) -> Result<Box<dyn RowCursor + 'e>> {
    let left = build_cursor(&s.left, env, outer)?;
    let right = build_cursor(&s.right, env, outer)?;
    let collations: &[Collation] = &s.column_collations;

    // Each arm's `Box<Concrete>` unsizes to the function's `Box<dyn RowCursor>` return
    // type at the `Ok(..)` coercion site.
    match s.op {
        // UNION ALL: concatenate, keep every duplicate (`seen == None`).
        SetOpKind::UnionAll => {
            Ok(Box::new(ConcatCursor { left, right, left_done: false, seen: None, collations }))
        }
        // UNION: concatenate, emit each distinct row once (dedup on the emitted set).
        SetOpKind::Union => Ok(Box::new(ConcatCursor {
            left,
            right,
            left_done: false,
            seen: Some(HashSet::new()),
            collations,
        })),
        // INTERSECT: distinct left rows whose key is ALSO in the right (`keep_present`).
        SetOpKind::Intersect => Ok(Box::new(ExistsCursor {
            left,
            right: Some(right),
            collations,
            right_keys: HashSet::new(),
            emitted: HashSet::new(),
            keep_present: true,
        })),
        // EXCEPT: distinct left rows whose key is NOT in the right (`!keep_present`).
        SetOpKind::Except => Ok(Box::new(ExistsCursor {
            left,
            right: Some(right),
            collations,
            right_keys: HashSet::new(),
            emitted: HashSet::new(),
            keep_present: false,
        })),
    }
}

/// `UNION ALL` / `UNION`: emit every left row, then every right row. When `seen` is
/// `Some`, a row is emitted only the first time its key appears (`UNION`); when
/// `None`, every row is emitted (`UNION ALL`, zero buffering).
struct ConcatCursor<'e> {
    left: Box<dyn RowCursor + 'e>,
    right: Box<dyn RowCursor + 'e>,
    /// Once the left child is exhausted we switch to pulling the right. Latched so we
    /// never re-pull the (already exhausted) left.
    left_done: bool,
    /// The emitted-row keys for `UNION`; `None` for `UNION ALL` (no dedup, no buffer).
    seen: Option<HashSet<Vec<CellKey>>>,
    collations: &'e [Collation],
}

impl ConcatCursor<'_> {
    /// The next raw row of the concatenation `left ++ right`: pull the left until it
    /// is exhausted (latching `left_done`), then the right. `None` only when both are
    /// exhausted.
    fn next_concat(&mut self, rt: &mut Runtime) -> Result<Option<Row>> {
        if !self.left_done {
            if let Some(row) = self.left.next_row(rt)? {
                return Ok(Some(row));
            }
            self.left_done = true;
        }
        self.right.next_row(rt)
    }
}

impl RowCursor for ConcatCursor<'_> {
    fn next_row(&mut self, rt: &mut Runtime) -> Result<Option<Row>> {
        while let Some(row) = self.next_concat(rt)? {
            match &mut self.seen {
                // UNION ALL: pass everything straight through.
                None => return Ok(Some(row)),
                // UNION: emit only the first occurrence of each row key.
                Some(seen) => {
                    if seen.insert(row_key(&row, self.collations)) {
                        return Ok(Some(row));
                    }
                }
            }
        }
        Ok(None)
    }
}

/// `INTERSECT` / `EXCEPT`: distinct left rows filtered by membership in the right
/// input's key set. `keep_present` selects the operator — `true` keeps rows present
/// in the right (`INTERSECT`), `false` keeps rows absent from it (`EXCEPT`).
struct ExistsCursor<'e> {
    left: Box<dyn RowCursor + 'e>,
    /// The right child, drained into `right_keys` on the first pull and then dropped
    /// (freeing any buffers it held). `None` once drained; that also serves as the
    /// "already initialized" flag.
    right: Option<Box<dyn RowCursor + 'e>>,
    collations: &'e [Collation],
    /// Keys of every right row — the one required materialization, bounded by the
    /// right cardinality. Built on the first pull.
    right_keys: HashSet<Vec<CellKey>>,
    /// Keys already emitted, so each surviving row is emitted once (both operators are
    /// `DISTINCT`). Bounded by the result cardinality.
    emitted: HashSet<Vec<CellKey>>,
    keep_present: bool,
}

impl RowCursor for ExistsCursor<'_> {
    fn next_row(&mut self, rt: &mut Runtime) -> Result<Option<Row>> {
        // First pull: drain the whole right child into the key set, then drop it.
        if let Some(mut right) = self.right.take() {
            while let Some(row) = right.next_row(rt)? {
                self.right_keys.insert(row_key(&row, self.collations));
            }
        }
        // Stream the left, emitting each not-yet-emitted row whose right-membership
        // matches `keep_present`. `contains(&key)` releases its borrow before
        // `insert(key)` consumes the key.
        while let Some(row) = self.left.next_row(rt)? {
            let key = row_key(&row, self.collations);
            if self.right_keys.contains(&key) == self.keep_present && self.emitted.insert(key) {
                return Ok(Some(row));
            }
        }
        Ok(None)
    }
}
