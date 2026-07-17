//! The pager resolvers the executor reads and writes storage through, and [`Env`], the
//! shared read context every operator cursor carries.
//!
//! A connection has SEVERAL stores — one [`Pager`] per database namespace (`main` = 0,
//! `temp` = 1, attached = 2..). A cursor is bound to ONE b-tree in ONE store, so a plan
//! node names its namespace ([`DbIndex`]) and the executor resolves it to the right pager:
//!
//! * [`PagerSet`] is what the executor is HANDED (`&mut`): the connection's live stores, or
//!   a single store for a trigger action. It is the executor's storage input — the read
//!   path derives a shared [`Pagers`] view from it ([`PagerSet::into_source`]) and the write
//!   path reborrows one target store mutably ([`PagerSet::target`]). One type serves BOTH
//!   the top level (over the whole store slice) and a trigger action (over the single target
//!   store the firing DML already holds), without a second executor route.
//! * [`Pagers`] is the READ resolver inside the `Copy` [`Env`] (shared `&dyn Pager`), so
//!   many read cursors — possibly across namespaces — borrow storage at once.

use minisqlite_catalog::Catalog;
use minisqlite_pager::Pager;
use minisqlite_plan::Plan;
use minisqlite_types::{DbIndex, Error, Result};

/// Resolves a [`DbIndex`] (the namespace a plan node names) to the `&dyn Pager` backing
/// it, for the READ path. `Copy`, so it rides inside the `Copy` [`Env`] with no allocation.
///
/// Two forms, matching the two read contexts:
/// * [`Pagers::Set`] wraps the connection's WHOLE live store slice. A read that may span
///   namespaces uses this — a top-level query; a DML statement's buffered SOURCE scan
///   (`INSERT INTO main.t SELECT … FROM temp.s`); and EVERY DML expression eval that may
///   hold a scalar subquery over another namespace: an INSERT `VALUES`, an `UPDATE … SET`
///   assignment, an UPSERT `DO UPDATE` `SET`/`WHERE`, and a `RETURNING` clause
///   (`lang_returning.html` §3 allows subqueries, and a subquery reading a DIFFERENT table —
///   an attached db / `temp` — is well-defined). Each base leaf resolves its own `db`
///   against it.
/// * [`Pagers::One`] wraps a SINGLE namespace's store. A context that PROVABLY touches one
///   namespace — a DML statement's CONSTRAINT-class expression eval (generated columns /
///   CHECK / DEFAULT / index-key expressions, where the spec FORBIDS subqueries so no
///   cross-namespace table scan is reachable), the DML helper ops (FK probe, index
///   backfill), and a same-namespace trigger action — uses this. It carries the namespace it
///   stands for and `get` FAILS CLOSED on any other `db`, so a read that unexpectedly
///   reaches across namespaces surfaces loudly instead of silently reading the wrong store.
#[derive(Clone, Copy)]
pub(crate) enum Pagers<'a> {
    /// The whole live store slice, indexed by `DbIndex` (`main` = 0, `temp` = 1, …).
    Set(&'a [Box<dyn Pager>]),
    /// One namespace's store, serving only reads that name that same namespace.
    One { db: DbIndex, pager: &'a dyn Pager },
}

impl<'a> Pagers<'a> {
    /// The pager backing namespace `db`.
    ///
    /// [`Pagers::Set`] indexes the slice and errors LOUDLY if `db` names a store that is
    /// not live (a plan referencing a missing namespace is a bug, not an empty result).
    /// [`Pagers::One`] returns its store only when `db` matches the namespace it stands
    /// for, and errors otherwise — the fail-closed guard for a single-namespace context.
    pub(crate) fn get(self, db: DbIndex) -> Result<&'a dyn Pager> {
        match self {
            Pagers::Set(set) => set.get(db.index()).map(|b| &**b).ok_or_else(|| {
                Error::sql(format!("no database attached at namespace index {}", db.index()))
            }),
            Pagers::One { db: mine, pager } if mine == db => Ok(pager),
            Pagers::One { db: mine, .. } => Err(Error::sql(format!(
                "single-namespace context (db {}) cannot reach namespace {}",
                mine.index(),
                db.index()
            ))),
        }
    }
}

/// The pager set the executor is handed — the WRITE-capable holder of the connection's
/// live stores (or a single store for a trigger action). The read path derives a shared
/// [`Pagers`] view from it ([`PagerSet::into_source`]); a DML write exposes a shared
/// [`Pagers`] view for the phase-1 SOURCE scan ([`PagerSet::source`]) and a `&mut dyn Pager`
/// for the phase-2 write ([`PagerSet::target`]) — never both at once (the shared view is
/// dropped before the mutable reborrow), which is what lets the operator compile with
/// ordinary borrows.
///
/// It is a `pub` input to the [`Executor::execute`](crate::Executor::execute) seam so the
/// engine can hand over its whole store slice and a test can hand over one store; the
/// executor never stores it, only reads/reborrows through it.
///
/// Its two forms both drive [`build_dml`](crate::runner::build_dml):
/// * [`PagerSet::Set`] — the connection's whole store slice, from the top-level executor
///   AND from a firing trigger / INSTEAD-OF action, which receives a
///   [`reborrow`](PagerSet::reborrow) of the set so it routes to its OWN stamped `node.db`
///   (a cross-namespace action reaches the right store). The write targets `pagers[node.db]`;
///   the source may scan any namespace.
/// * [`PagerSet::One`] — a single namespace's store, from a single-namespace caller (the
///   `-exec` unit tests). It carries that namespace, so a plan (or a nested action under a
///   `reborrow`ed `One`) naming a DIFFERENT namespace fails closed rather than reading or
///   writing the wrong store.
pub enum PagerSet<'e> {
    /// The whole live store slice; the write targets one element by `DbIndex`.
    Set(&'e mut [Box<dyn Pager>]),
    /// One namespace's store, for a single-namespace caller (fails closed on any other db).
    One { db: DbIndex, pager: &'e mut dyn Pager },
}

impl<'e> PagerSet<'e> {
    /// Consume the set into a shared [`Pagers`] read view for the whole statement lifetime.
    /// Used by the executor's READ path, where the returned cursor must borrow the stores
    /// for `'e` and no later write occurs — so the exclusive borrow is downgraded to a
    /// shared one for its full lifetime (a `&'e mut` reborrowed as `&'e` by consuming it).
    pub(crate) fn into_source(self) -> Pagers<'e> {
        match self {
            PagerSet::Set(set) => Pagers::Set(&*set),
            PagerSet::One { db, pager } => Pagers::One { db, pager: &*pager },
        }
    }

    /// A shared [`Pagers`] view for the phase-1 source scan. Borrows `self` immutably, so
    /// the DML operator can still touch its OTHER fields (the RETURNING buffer) while the
    /// source drains — the shared borrow is only of the store(s).
    pub(crate) fn source(&self) -> Pagers<'_> {
        match self {
            PagerSet::Set(set) => Pagers::Set(set),
            PagerSet::One { db, pager } => Pagers::One { db: *db, pager: &**pager },
        }
    }

    /// The exclusive `&mut dyn Pager` for the phase-2 write to namespace `db`.
    ///
    /// [`PagerSet::Set`] reborrows `pagers[db]`; [`PagerSet::One`] returns its store only
    /// when `db` matches the namespace it stands for (a cross-namespace trigger action
    /// fails closed here). Errors LOUDLY on a missing/other namespace rather than writing
    /// the wrong store.
    pub(crate) fn target(&mut self, db: DbIndex) -> Result<&mut (dyn Pager + '_)> {
        // Each arm returns `Ok(&mut **_)` DIRECTLY (not via `.map`) so the `Ok(...)`
        // constructor propagates the function's return type inward and the `&mut dyn Pager`
        // trait-object lifetime coerces (shortens) to the returned reference's — a `.map`
        // closure would instead infer the source `'static`/`'e` bound and then fail the
        // invariant `&mut` match against the shorter return lifetime.
        match self {
            PagerSet::Set(set) => {
                let i = db.index();
                match set.get_mut(i) {
                    Some(b) => Ok(&mut **b),
                    None => {
                        Err(Error::sql(format!("no database attached at namespace index {i}")))
                    }
                }
            }
            PagerSet::One { db: mine, pager } if *mine == db => Ok(&mut **pager),
            PagerSet::One { db: mine, .. } => Err(Error::sql(format!(
                "single-namespace trigger action (db {}) cannot write namespace {}",
                mine.index(),
                db.index()
            ))),
        }
    }

    /// A shorter-lived reborrow of this whole set, for handing to a nested write (a firing
    /// trigger / `INSTEAD OF` action) WITHOUT giving up ownership of the outer borrow.
    ///
    /// A trigger fire is SEQUENTIAL (one action at a time) and runs AFTER the read/frame
    /// phase has been buffered, so the outer store borrow is free while an action fires. The
    /// action then resolves its OWN stamped `node.db` against this reborrowed set — a
    /// [`PagerSet::Set`] routes each action to any namespace's pager (a cross-namespace
    /// trigger action writes the correct store), while a [`PagerSet::One`] keeps its
    /// single-namespace fail-closed guarantee (an action naming another namespace still
    /// errors loudly in [`PagerSet::target`]). Reborrowing (not moving) `self` is what lets
    /// the caller loop over many frames/actions, taking a fresh exclusive handle each time.
    pub(crate) fn reborrow(&mut self) -> PagerSet<'_> {
        match self {
            PagerSet::Set(set) => PagerSet::Set(&mut **set),
            PagerSet::One { db, pager } => PagerSet::One { db: *db, pager: &mut **pager },
        }
    }
}

/// The shared read context threaded through every read operator. All fields are shared
/// references with the same lifetime, so `Env` is `Copy` and covariant.
#[derive(Clone, Copy)]
pub(crate) struct Env<'a> {
    /// Schema lookups (`table`/`table_in`, `index`, `indexes_on`), read-only.
    pub catalog: &'a dyn Catalog,
    /// Namespace-aware storage resolver (b-tree cursors borrow through it), read-only on
    /// this path. A base leaf opens its cursor on `pagers.get(node.db)`.
    pub pagers: Pagers<'a>,
    /// The whole plan, so subquery/CTE operators can resolve `subqueries`/`ctes`. The
    /// subquery callbacks in `context.rs` read `plan.subqueries[id]` through it; a CTE
    /// operator reads `plan.ctes[id]` the same way.
    pub plan: &'a Plan,
}
