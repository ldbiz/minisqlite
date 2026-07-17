//! The COMMIT-time DEFERRED foreign-key recheck the engine runs before making a
//! transaction durable (`foreignkeys.html` §4.2). A `DEFERRABLE INITIALLY DEFERRED`
//! constraint — or, under `PRAGMA defer_foreign_keys`, every constraint — is allowed to sit
//! in violation for the life of an open transaction and is only verified here, at COMMIT.
//!
//! This is a THIN engine seam over the executor's [`minisqlite_exec::check_deferred_foreign_keys`]:
//! it early-returns when FK enforcement is off, reads the two connection flags, and drives
//! the per-namespace child-side scan. The real row-walking / parent-existence logic lives in
//! the executor next to the immediate FK checks (so the deferred and immediate checks cannot
//! drift on what "the parent exists" means). It is invoked from exactly two commit points —
//! [`SqlEngine::exec_commit`] and the outermost-savepoint [`SqlEngine::exec_release`] — each
//! time BEFORE the pager commit/release and before any `in_explicit_txn`/`savepoints` state is
//! cleared, so a violation fails the operation and LEAVES THE TRANSACTION OPEN (§4.2), and so
//! the concurrent auto-vacuum pre-commit hook reconciles against one added line rather than an
//! inlined block.
//!
//! LOAD-BEARING CONCURRENCY INVARIANT (do not break in a future WAL/locking rework): this
//! recheck is a plain read over the connection's OWN in-transaction pager view, and its
//! correctness rests on the write transaction being held CONTINUOUSLY from the first deferring
//! write through this check and on to the durable commit/release — with NO write-lock
//! release/downgrade or read-bracket in between. Because the connection holds the write lock
//! across the whole window, no concurrent writer/checkpoint can mutate the committed view
//! between "the parent still exists" here and the pager commit that follows, so this is NOT a
//! TOCTOU (the same discipline the immediate FK checks and the auto-vacuum pre-commit hook
//! rely on). If a future change ever lazily (re)acquires or downgrades the write lock around
//! commit, this read-then-commit ordering must be re-audited: a lock gap here reintroduces the
//! stale-snapshot race (the exact WAL-bug shape) where a concurrent commit invalidates the
//! parent this check just verified.

use minisqlite_catalog::MultiCatalog;
use minisqlite_types::{DbIndex, Result};

use crate::engine::SqlEngine;

impl SqlEngine {
    /// Re-verify every deferred foreign key across all live namespaces at COMMIT. Returns
    /// `Err(FOREIGN KEY constraint failed)` on the first unresolved deferred violation, which
    /// [`exec_commit`](SqlEngine::exec_commit) propagates WITHOUT committing or clearing the
    /// transaction (the transaction stays open, per `foreignkeys.html` §4.2).
    ///
    /// Off-cost by construction: a no-op unless `PRAGMA foreign_keys` is ON, and within that,
    /// the executor scans only tables that actually declare a deferred FK (or every FK when
    /// `PRAGMA defer_foreign_keys` is ON) — an all-immediate schema walks its catalog and
    /// touches no data page. A cross-database FK is illegal, so each namespace's children
    /// reference parents in that SAME namespace: the scan reads `catalogs[i]`/`pagers[i]`
    /// together per namespace, mirroring the immediate FK path's single-namespace scoping.
    pub(crate) fn check_deferred_foreign_keys(&self) -> Result<()> {
        if !self.rt.foreign_keys() {
            return Ok(());
        }
        // Use the STICKY armed flag, not the live `defer_foreign_keys()`: if the defer pragma
        // was turned ON at any point this transaction (deferring an otherwise-immediate FK) and
        // then toggled OFF before COMMIT, we must STILL rescan every FK, or the row deferred
        // under the pragma would commit unchecked. `defer_foreign_keys_armed()` stays set across
        // a mid-txn toggle-off (it clears only at txn end), so coverage never shrinks.
        let defer_all = self.rt.defer_foreign_keys_armed();
        // A pure read view over every live schema store; the executor resolves each FK's
        // parent WITHIN its own namespace (`table_in(db, …)`), so the composite catalog is
        // only ever asked for same-namespace lookups here.
        let catalog = MultiCatalog::new(&self.catalogs, &self.namespaces);
        for i in 0..self.pagers.len() {
            let db = DbIndex(i as u16);
            minisqlite_exec::check_deferred_foreign_keys(
                &catalog,
                db,
                &*self.pagers[i],
                defer_all,
            )?;
        }
        Ok(())
    }
}
