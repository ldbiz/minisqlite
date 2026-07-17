//! [`StatementRoot`] — the thin cursor the executor wraps around every statement's
//! root, so the connection [`Runtime`]'s per-statement scratch is reset once at the
//! start of each statement's drain.
//!
//! Why it exists: a [`Runtime`] is created once per connection and reused for every
//! statement, but the uncorrelated-subquery cache it carries is per-*statement* — a
//! `SubqueryId` restarts at 0 for each plan, so statement B's `subqueries[0]` is a
//! different subquery from statement A's. Without a reset, B would read A's cached
//! value for id 0 and silently return the wrong answer. `execute()` is the single
//! place a root cursor is produced, and the engine drives each statement as exactly
//! one `execute()` + one full drain, so wrapping that one cursor and clearing the
//! cache on its FIRST pull resets the cache exactly once per statement, right before
//! any operator (hence any subquery evaluation) runs.
//!
//! Clearing on the first pull rather than at build time ties the reset to the drain
//! that actually reads the cache (and to the point where the `&mut Runtime` is in
//! hand); every DML operator likewise defers its work to the first pull, so a DML
//! statement's subqueries are also evaluated only after this clear. Later pulls skip
//! the clear, so values this statement caches persist across its own rows — which is
//! the whole point (evaluate once, reuse across the outer rows).

use minisqlite_types::{Result, Row};

use crate::executor::RowCursor;
use crate::runtime::Runtime;

/// Wraps a statement's root cursor: clears the per-statement subquery cache on the
/// first [`next_row`](RowCursor::next_row), then delegates every pull to `inner`.
pub(crate) struct StatementRoot<'a> {
    inner: Box<dyn RowCursor + 'a>,
    /// `false` until the first pull; set on the pull that performs the one-time clear.
    started: bool,
}

impl<'a> StatementRoot<'a> {
    /// Wrap `inner` (a statement's root cursor) so the subquery cache is cleared at
    /// the start of its drain.
    pub(crate) fn new(inner: Box<dyn RowCursor + 'a>) -> StatementRoot<'a> {
        StatementRoot { inner, started: false }
    }
}

impl RowCursor for StatementRoot<'_> {
    fn next_row(&mut self, rt: &mut Runtime) -> Result<Option<Row>> {
        if !self.started {
            rt.clear_subquery_cache();
            // The re-compiled-trigger memo shares the subquery cache's per-statement
            // lifecycle: cleared here so a `CREATE`/`DROP TRIGGER` between statements is
            // never masked by a set memoized under a prior statement's schema.
            rt.clear_recompiled_triggers();
            self.started = true;
        }
        self.inner.next_row(rt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::CachedSubquery;
    use minisqlite_types::Value;
    use std::cell::Cell;
    use std::rc::Rc;

    /// A leaf cursor whose first pull records, into a shared cell we retain a handle
    /// to, whether the subquery cache was already empty at `id 0` — the witness that
    /// `StatementRoot` cleared before delegating inward.
    struct SawClearProbe {
        saw_clear: Rc<Cell<Option<bool>>>,
        pulled: bool,
    }

    impl RowCursor for SawClearProbe {
        fn next_row(&mut self, rt: &mut Runtime) -> Result<Option<Row>> {
            if !self.pulled {
                self.pulled = true;
                self.saw_clear.set(Some(rt.cached_subquery(0).is_none()));
            }
            Ok(None)
        }
    }

    #[test]
    fn clears_subquery_cache_before_first_inner_pull() {
        let mut rt = Runtime::new();
        // A stale entry, as if left by a previous statement on the same connection.
        rt.cache_subquery(0, CachedSubquery::Exists(false));

        let saw_clear = Rc::new(Cell::new(None));
        let probe = Box::new(SawClearProbe { saw_clear: saw_clear.clone(), pulled: false });
        let mut root = StatementRoot::new(probe);
        let _ = root.next_row(&mut rt).unwrap();

        assert_eq!(
            saw_clear.get(),
            Some(true),
            "StatementRoot cleared the cache before delegating to the inner cursor"
        );
    }

    #[test]
    fn clear_preserves_change_counters() {
        // The per-statement clear must leave the change counters and last_insert_rowid
        // alone — a DML statement followed by `SELECT changes()` relies on it.
        let mut rt = Runtime::new();
        rt.record_insert(5);
        let probe = Box::new(SawClearProbe { saw_clear: Rc::new(Cell::new(None)), pulled: false });
        let mut root = StatementRoot::new(probe);
        let _ = root.next_row(&mut rt).unwrap();
        assert_eq!(rt.changes(), 1, "changes() untouched by the cache clear");
        assert_eq!(rt.total_changes(), 1, "total_changes() untouched");
        assert_eq!(rt.last_insert_rowid(), 5, "last_insert_rowid() untouched");
    }

    /// A cursor that caches a value on its first pull (after the wrapper's clear) and
    /// on its second pull asserts the value is still there — proving the clear happens
    /// ONLY on the first pull, so values cached mid-statement survive across rows.
    struct CacheThenCheckProbe {
        pulls: usize,
    }

    impl RowCursor for CacheThenCheckProbe {
        fn next_row(&mut self, rt: &mut Runtime) -> Result<Option<Row>> {
            self.pulls += 1;
            if self.pulls == 1 {
                rt.cache_subquery(0, CachedSubquery::Scalar(Value::Integer(42)));
                return Ok(Some(vec![Value::Integer(1)]));
            }
            assert!(
                matches!(rt.cached_subquery(0), Some(CachedSubquery::Scalar(Value::Integer(42)))),
                "a value cached after the first-pull clear must survive later pulls"
            );
            Ok(None)
        }
    }

    #[test]
    fn does_not_reclear_on_later_pulls() {
        let mut rt = Runtime::new();
        let mut root = StatementRoot::new(Box::new(CacheThenCheckProbe { pulls: 0 }));
        while root.next_row(&mut rt).unwrap().is_some() {}
    }
}
