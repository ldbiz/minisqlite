//! `Cow<S>` — the copy-on-write transaction layer, written ONCE and shared by
//! every `Pager` backing (the in-memory `MemPager` today, a disk-backed pager
//! later). It is generic over a [`PageStore`](crate::store::PageStore): the store
//! owns the committed image and the single durable-apply step; this layer owns all
//! of begin / read-through / copy-on-write write / allocate / free / commit /
//! rollback. Keeping this logic in one generic place is the whole point of the
//! split — a second backing must NOT re-implement any of it.
//!
//! ## Model
//! A transaction is a dirty `overlay` (the pages written or allocated so far) plus
//! the transaction's page count. The committed store is never mutated until
//! `commit`, so:
//! - a read consults the overlay first, then falls through to the committed store
//!   (borrowing, never copying the database);
//! - `allocate_page` grows only the transaction's count and stages a zero page in
//!   the overlay — it does NOT grow the committed store — so `rollback` is just
//!   "drop the overlay and forget the grown count", with no truncation to undo;
//! - `commit` gathers the overlay into id order and hands it to the store's single
//!   `apply_commit`.
//!
//! ## Auto-commit
//! A write / allocate / free issued with no active transaction runs as an implicit
//! single-statement transaction (begin → op → commit), exactly like SQLite's
//! auto-commit. That keeps one code path for both explicit and implicit
//! transactions and works identically for a durable disk backing.

use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap};

use minisqlite_types::{Error, Result};

use crate::PageId;
use crate::checkpoint::{CheckpointMode, CheckpointReport};
use crate::reclaim::VacuumOutcome;
use crate::store::PageStore;

/// One open savepoint: a BOUNDED pre-image delta over the transaction overlay, not
/// a snapshot of it. `pre_images[id]` is captured lazily the FIRST time page `id` is
/// modified after this savepoint was established, so the stored bytes are exactly
/// the overlay contents "as of this savepoint" for every page that has since
/// changed — pages untouched since the mark are already correct in the live overlay
/// and need no entry. This keeps a savepoint's cost proportional to the writes made
/// under it, never to the whole overlay (a full-overlay clone per savepoint would be
/// unbounded and is forbidden).
struct Savepoint {
    /// Pre-images captured since this savepoint was established, keyed by page id.
    /// `Some(bytes)` = the overlay held `bytes` for this page at capture time;
    /// `None` = the overlay had NO entry for this page (a "was-absent" marker), so
    /// restoring it means removing the overlay entry (falling back to the committed
    /// image), not writing bytes.
    pre_images: HashMap<PageId, Option<Box<[u8]>>>,
    /// The transaction `page_count` as of this savepoint's establishment. `rollback
    /// to` restores it, undoing any pages allocated (grown) under the savepoint.
    page_count: PageId,
}

/// The in-flight write transaction: the dirty overlay plus the live page count.
struct Txn {
    /// Pages written or freshly allocated during the transaction, keyed by id. A
    /// read consults this before the committed store, so a committed page is only
    /// mutated at `commit`, never before.
    overlay: HashMap<PageId, Box<[u8]>>,
    /// The transaction's page count. Starts at the committed count captured at
    /// `begin` and grows as pages are allocated; `commit` passes it to the store
    /// as the new committed count, and `rollback` discards it.
    page_count: PageId,
    /// The open savepoints, innermost last. Empty when no savepoint is set. Each is
    /// a bounded pre-image delta (see [`Savepoint`]); together they form the undo
    /// stack for `rollback to` / `release`.
    savepoints: Vec<Savepoint>,
    /// `true` when this transaction was started implicitly by its first savepoint
    /// (a `SAVEPOINT` issued in autocommit), rather than by an explicit `begin`.
    /// Releasing the outermost savepoint of such a transaction commits it, matching
    /// SQLite ("If a RELEASE command releases the outermost savepoint ... then
    /// RELEASE is the same as COMMIT"). A `begin`-started transaction leaves this
    /// `false`, so releasing all its savepoints keeps the transaction open for its
    /// enclosing `COMMIT`/`ROLLBACK`.
    started_by_savepoint: bool,
    /// `true` when this transaction pinned a read snapshot at begin (a DEFERRED
    /// begin, via [`PageStore::begin_read`]) that must be released at commit/rollback.
    /// An eager `begin` takes the write lock instead and leaves this `false`.
    read_pinned: bool,
    /// `true` while this transaction currently holds the store's single write lock.
    /// An eager `begin` sets it at BEGIN; a DEFERRED begin starts `false` and flips it
    /// on the first write's lazy upgrade (see [`Cow::ensure_write_lock`]). It governs
    /// whether commit/rollback release the write lock, so a read-only DEFERRED
    /// transaction (which never took the lock) does not spuriously release one.
    ///
    /// This is the COW layer's per-transaction view. A coordinating store separately
    /// tracks its own per-connection lock ownership (e.g. `WalStore::holds_write_lock`,
    /// which makes `end_write` idempotent and drives its `Drop`). This flag decides
    /// *whether to release*; the store's flag makes the release itself safe to repeat.
    lock_held: bool,
}

impl Txn {
    /// Record the pre-image of page `id` in the innermost open savepoint, once, the
    /// FIRST time `id` is modified after that savepoint was established. Call it
    /// BEFORE mutating `overlay[id]` so the captured bytes are the pre-mutation
    /// state. A no-op when no savepoint is open, or when the innermost savepoint has
    /// already captured this page (its earliest pre-image is the one to keep).
    ///
    /// Only the innermost savepoint records here; `release` merges an inner
    /// savepoint's deltas down into its enclosing one, and `rollback to` walks the
    /// deltas from the top down, so a page's undo information reaches every level
    /// that needs it without eagerly cloning the pre-image into each open savepoint.
    /// Recording a "was-absent" marker (the common case — a page first staged under
    /// the savepoint) allocates nothing.
    fn capture_pre_image(&mut self, id: PageId) {
        let Some(last) = self.savepoints.len().checked_sub(1) else {
            return;
        };
        // Separate statements keep the `savepoints` and `overlay` borrows disjoint.
        if self.savepoints[last].pre_images.contains_key(&id) {
            return;
        }
        let pre = self.overlay.get(&id).cloned();
        self.savepoints[last].pre_images.insert(id, pre);
    }
}

/// The shared copy-on-write layer over a committed-image [`PageStore`]. See the
/// module docs for the transaction model.
pub(crate) struct Cow<S: PageStore> {
    store: S,
    txn: Option<Txn>,
    /// Set by [`commit`](Cow::commit) when its `finalize` RELOCATED a b-tree root to
    /// satisfy §1.8 roots-first (rewriting a `sqlite_schema.rootpage`). Read-and-cleared
    /// by [`take_root_moved`](Cow::take_root_moved) so the engine can reload a cached
    /// catalog whose root pages just moved. It is a committed-state fact (a durable
    /// rootpage change), so — unlike the per-transaction `txn` state — it persists across
    /// transactions until the engine consumes it, and `rollback` never touches it.
    root_moved: bool,
}

impl<S: PageStore> Cow<S> {
    /// Wrap a committed-image store in the copy-on-write layer, with no active
    /// transaction.
    pub(crate) fn new(store: S) -> Cow<S> {
        Cow { store, txn: None, root_moved: false }
    }

    /// Read and clear the "a commit relocated a b-tree root" flag (see
    /// [`root_moved`](Cow::root_moved)). The engine calls this after a statement settles
    /// to autocommit and reloads the namespace's catalog when it returns `true`, so a
    /// `CREATE INDEX`/`CREATE TABLE` that moved a root to the front (§1.8) does not leave
    /// the cached `sqlite_schema.rootpage` pointing at the pre-move slot. `false` for
    /// every commit that moved no root (the common path).
    pub(crate) fn take_root_moved(&mut self) -> bool {
        std::mem::take(&mut self.root_moved)
    }

    /// Byte size of every page (delegated to the store).
    pub(crate) fn page_size(&self) -> u32 {
        self.store.page_size()
    }

    /// Current page count: the transaction's live count inside a transaction,
    /// otherwise the committed count.
    pub(crate) fn page_count(&self) -> Result<PageId> {
        match self.txn.as_ref() {
            Some(txn) => Ok(txn.page_count),
            None => self.store.committed_count(),
        }
    }

    /// Borrow a page's bytes: the transaction's staged copy if it has one, else the
    /// committed image. The borrow ties to `&self`, so it never overlaps a `&mut`
    /// write and the bytes never move under it. An out-of-range or zero id fails
    /// closed rather than panicking.
    pub(crate) fn read_page(&self, id: PageId) -> Result<&[u8]> {
        let count = self.page_count()?;
        if id == 0 || id > count {
            return Err(Error::Io(format!("page {id} is out of range (page_count = {count})")));
        }
        if let Some(txn) = self.txn.as_ref()
            && let Some(buf) = txn.overlay.get(&id)
        {
            return Ok(&buf[..]);
        }
        self.store.read_committed(id)
    }

    /// Begin a transaction. Fails closed if one is already active (nesting is a
    /// caller bug), rather than silently discarding the overlay.
    ///
    /// A coordinating store's [`begin_write`](PageStore::begin_write) hook runs
    /// first — it acquires the single write lock (returning a BUSY error if another
    /// connection holds it) and refreshes the committed snapshot — so the base page
    /// count captured below already reflects the latest committed state. The hook is
    /// a no-op for the local backings, leaving their begin behavior unchanged.
    ///
    /// This is the EAGER begin: it takes the write lock up front, so it backs
    /// autocommit writes, `BEGIN IMMEDIATE`/`EXCLUSIVE`, DDL, CTAS, and VACUUM. A
    /// `BEGIN DEFERRED` uses [`Cow::begin_deferred`] instead (write lock deferred to
    /// the first write).
    pub(crate) fn begin(&mut self) -> Result<()> {
        if self.txn.is_some() {
            return Err(Error::Io("begin: a transaction is already active".into()));
        }
        self.store.begin_write()?;
        let base = match self.store.committed_count() {
            Ok(b) => b,
            Err(e) => {
                // A failed begin must leave no write lock held.
                let _ = self.store.end_write();
                return Err(e);
            }
        };
        self.txn = Some(Txn {
            overlay: HashMap::new(),
            page_count: base,
            savepoints: Vec::new(),
            started_by_savepoint: false,
            read_pinned: false,
            lock_held: true,
        });
        Ok(())
    }

    /// Begin a DEFERRED transaction: pin a read snapshot only, deferring the single
    /// write lock until the first write (lang_transaction §2.2). A coordinating store's
    /// [`begin_read`](PageStore::begin_read) hook refreshes to and pins the latest
    /// committed snapshot (and registers a reader mark) WITHOUT taking the write lock,
    /// so a read-only DEFERRED transaction never blocks a concurrent connection's commit
    /// and keeps its historic snapshot while other connections commit. The first write
    /// upgrades the transaction via [`Cow::ensure_write_lock`] (acquire-or-BUSY). For a
    /// non-coordinating backing `begin_read` is a no-op and the eager write lock is taken
    /// lazily by the same upgrade path (which forwards to `begin_write`), so behavior is
    /// unchanged there.
    pub(crate) fn begin_deferred(&mut self) -> Result<()> {
        if self.txn.is_some() {
            return Err(Error::Io("begin: a transaction is already active".into()));
        }
        self.store.begin_read()?;
        let base = match self.store.committed_count() {
            Ok(b) => b,
            Err(e) => {
                // A failed begin must release the reader mark it just pinned, mirroring
                // the eager begin's write-lock cleanup.
                let _ = self.store.end_read();
                return Err(e);
            }
        };
        self.txn = Some(Txn {
            overlay: HashMap::new(),
            page_count: base,
            savepoints: Vec::new(),
            started_by_savepoint: false,
            read_pinned: true,
            lock_held: false,
        });
        Ok(())
    }

    /// Ensure the current transaction holds the store's write lock, acquiring it lazily
    /// on the first write of a DEFERRED transaction (the read-only-to-write upgrade,
    /// lang_transaction §2.1). A no-op when the lock is already held (the eager `begin`
    /// path, and every write after the first) or when no transaction is active (the
    /// public write entry points then fail closed on their own "no active transaction"
    /// check). A coordinating store's [`begin_write_upgrade`](PageStore::begin_write_upgrade)
    /// returns BUSY if the lock is held by another connection or the pinned snapshot is
    /// stale; the flag is set only AFTER a successful acquire, so a BUSY leaves the
    /// transaction read-only with its overlay and snapshot intact.
    fn ensure_write_lock(&mut self) -> Result<()> {
        let needs_lock = matches!(self.txn.as_ref(), Some(txn) if !txn.lock_held);
        if needs_lock {
            self.store.begin_write_upgrade()?;
            self.txn.as_mut().expect("txn present when a lock upgrade is needed").lock_held = true;
        }
        Ok(())
    }

    /// Release exactly the cross-connection resources a transaction acquired: the single
    /// write lock iff `lock_held`, and the pinned read snapshot / reader mark iff
    /// `read_pinned`. The ONE release point shared by `commit` and `rollback`, so the
    /// "release what you acquired" rule is spelled once rather than copied — a missed
    /// release on either path would strand the write lock (permanent BUSY for every other
    /// connection) or a reader mark (a checkpoint that can never reset). Each store hook
    /// is idempotent, so this is safe alongside the `Drop` that also releases an orphaned
    /// mid-transaction txn.
    ///
    /// A release failure is deliberately swallowed: a coordinating hook cannot fail here,
    /// and the caller needs the primary commit/rollback outcome, not a release error.
    fn release_coordination(&mut self, lock_held: bool, read_pinned: bool) {
        // Every live transaction owns at least one resource — an eager/upgraded txn holds
        // the write lock, a DEFERRED txn holds a read pin, an upgraded DEFERRED txn holds
        // both. `(false, false)` is never constructed, so releasing nothing would mean a
        // resource was silently lost track of.
        debug_assert!(
            lock_held || read_pinned,
            "a live transaction always owns the write lock, a pinned read snapshot, or both"
        );
        if lock_held {
            let _ = self.store.end_write();
        }
        if read_pinned {
            let _ = self.store.end_read();
        }
    }

    /// Commit the transaction: gather the overlay into ascending id order and hand
    /// it to the store's single durable-apply step, then clear the transaction and
    /// release exactly the coordination resources this transaction acquired — the
    /// write lock iff it holds one (never for a read-only DEFERRED transaction), and
    /// the pinned read snapshot iff a DEFERRED begin pinned one.
    ///
    /// A read-only DEFERRED transaction (`lock_held == false`) applies an EMPTY overlay
    /// — [`PageStore::apply_commit`] returns early on an empty dirty set — and releases
    /// only its reader mark, so `BEGIN; SELECT ...; COMMIT` never touches the write lock.
    pub(crate) fn commit(&mut self) -> Result<()> {
        // Auto_vacuum WRITE side, run as PART of this one durable transaction (see
        // [`crate::av_commit`]): before the overlay is gathered, rebuild the pointer-map
        // pages and page-1 offset 52 so the committed image is a valid auto_vacuum file,
        // and — for a FULL auto_vacuum database with reclaimable free pages — COMPACT it
        // (relocate the trailing live pages into the freed slots and truncate) in the same
        // overlay, so the user's data write and the compaction land in ONE `apply_commit`.
        // That is real sqlite's durability contract: a committed FULL database always has
        // `freelist_count == 0`, so a crash never exposes a committed-but-un-compacted file
        // and a free-page-releasing FULL commit costs one fsync, not two. It stages extra
        // pages into THIS transaction's overlay, so it must run while the transaction is
        // still active and holds the write lock (a read-only DEFERRED transaction stages
        // nothing). It is a one-header-read no-op for a non-vacuum database, keeping that
        // path byte-identical. A failure abandons the transaction (releasing the write lock
        // / read pin) rather than stranding coordination state, then surfaces the error —
        // the same fail-closed shape as an `apply_commit` error below.
        let moved_roots = if matches!(self.txn.as_ref(), Some(t) if t.lock_held) {
            match crate::av_commit::finalize_and_compact(self) {
                Ok(moved) => moved,
                Err(e) => {
                    let _ = self.rollback();
                    return Err(e);
                }
            }
        } else {
            false
        };
        let txn =
            self.txn.take().ok_or_else(|| Error::Io("commit: no active transaction".into()))?;
        // Read the flags (Copy) before the overlay is moved out of `txn`.
        let new_count = txn.page_count;
        let lock_held = txn.lock_held;
        let read_pinned = txn.read_pinned;
        let dirty: BTreeMap<PageId, Box<[u8]>> = txn.overlay.into_iter().collect();
        let result = self.store.apply_commit(dirty, new_count);
        // Release the coordination resources whether or not the durable apply
        // succeeded; the apply's error is what the caller needs, so a release failure
        // (which a no-op hook cannot produce) must not mask it.
        self.release_coordination(lock_held, read_pinned);
        // A relocated root is only part of the committed image once the durable apply
        // SUCCEEDS, so only then is the cached catalog stale; record it for the engine to
        // reload (see [`Cow::take_root_moved`]). A failed apply left the relocation
        // uncommitted, so the catalog still matches the on-disk roots and needs no reload.
        if result.is_ok() && moved_roots {
            self.root_moved = true;
        }
        result
    }

    /// Roll back the transaction. The committed store was never touched, so undoing
    /// the transaction is exactly dropping the overlay (done by taking `txn`) and
    /// forgetting the grown page count, then releasing exactly the coordination
    /// resources this transaction acquired — the write lock iff held, the pinned read
    /// snapshot iff a DEFERRED begin pinned one.
    pub(crate) fn rollback(&mut self) -> Result<()> {
        let txn =
            self.txn.take().ok_or_else(|| Error::Io("rollback: no active transaction".into()))?;
        self.release_coordination(txn.lock_held, txn.read_pinned);
        Ok(())
    }

    /// Begin a read transaction: a coordinating store pins its read snapshot so
    /// concurrent commits do not shift it mid-transaction. A no-op for the local
    /// backings. Callers must not have a write transaction active.
    pub(crate) fn begin_read(&mut self) -> Result<()> {
        self.store.begin_read()
    }

    /// End a read transaction, releasing the pinned snapshot.
    pub(crate) fn end_read(&mut self) -> Result<()> {
        self.store.end_read()
    }

    /// Run a checkpoint in `mode` on a store that keeps a WAL, returning its report.
    /// A no-op for the local backings (their store returns [`CheckpointReport::not_wal`]).
    pub(crate) fn checkpoint(&mut self, mode: CheckpointMode) -> Result<CheckpointReport> {
        self.store.checkpoint(mode)
    }

    /// Auto_vacuum RECLAMATION as its OWN transaction — the on-demand `PRAGMA
    /// incremental_vacuum(N)` path. It shrinks the database by removing up to `max` free
    /// pages (all when `None`), running in its own eager transaction so a reclamation fault
    /// or a declined (unsafe) relocation rolls back cleanly to the pre-reclaim, fully-valid
    /// state — never a corrupt file and never a failed CALLER statement. Must be called in
    /// autocommit (no transaction active); the commit re-derives the ptrmap via `finalize`
    /// for the smaller file.
    ///
    /// FULL auto_vacuum no longer routes through here: it compacts ATOMICALLY inside the
    /// user's own [`Cow::commit`] (see [`crate::av_commit`]), so a committed FULL database
    /// is never left un-compacted across a crash. This entry point remains for the explicit,
    /// budgeted INCREMENTAL-mode PRAGMA, which reclaims on demand rather than at every commit.
    ///
    /// A [`crate::reclaim`] error is treated as a DECLINE, not a failure: it rolls back
    /// and reports "reclaimed nothing" rather than propagating, so a relocation bug can
    /// only decline to compact (leaving free pages) instead of erroring the commit that
    /// triggered it — matching the module's "never corrupt, decline instead" stance. A
    /// genuine durable-apply fault at commit/rollback still surfaces.
    pub(crate) fn incremental_vacuum(&mut self, max: Option<PageId>) -> Result<VacuumOutcome> {
        if self.txn.is_some() {
            return Err(Error::Io(
                "incremental_vacuum: must run in autocommit, not inside an active transaction".into(),
            ));
        }
        self.begin()?;
        match crate::reclaim::reclaim(self, max) {
            Ok(outcome) => {
                self.commit()?;
                Ok(outcome)
            }
            Err(_decline) => {
                self.rollback()?;
                Ok(VacuumOutcome::default())
            }
        }
    }

    /// Copy-on-write write, auto-committing when no transaction is active. See
    /// [`Cow::stage_write`] for the in-transaction staging itself.
    pub(crate) fn write_page(&mut self, id: PageId, bytes: &[u8]) -> Result<()> {
        self.autocommit(|c| c.stage_write(id, bytes))
    }

    /// Allocate a page (reusing the freelist when possible, else growing),
    /// auto-committing when no transaction is active.
    pub(crate) fn allocate_page(&mut self) -> Result<PageId> {
        self.autocommit(crate::alloc::alloc_in_txn)
    }

    /// Release a page to the freelist, auto-committing when no transaction is
    /// active.
    pub(crate) fn free_page(&mut self, id: PageId) -> Result<()> {
        self.autocommit(|c| crate::alloc::free_in_txn(c, id))
    }

    /// Run `op` inside a transaction: directly if one is already active, otherwise
    /// as an implicit begin → op → commit (auto-commit), rolling back if `op`
    /// fails so a rejected op leaves no trace. This is the single place the
    /// implicit-transaction rule lives.
    fn autocommit<T>(&mut self, op: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        if self.txn.is_some() {
            return op(self);
        }
        self.begin()?;
        match op(self) {
            Ok(v) => {
                self.commit()?;
                Ok(v)
            }
            Err(e) => {
                // Roll back the implicit transaction; the op's error is what the
                // caller needs, so a rollback failure (impossible here — a txn is
                // active) must not mask it.
                let _ = self.rollback();
                Err(e)
            }
        }
    }

    /// Stage one page into the current transaction's overlay (copy-on-write),
    /// reusing this page's dirty buffer on a repeated write to avoid per-write
    /// allocation churn. Validates the id against the transaction page count and
    /// the length against the page size, so a bad write is rejected rather than
    /// landing silently. Requires an active transaction (the public entry points
    /// guarantee one via [`Cow::autocommit`]); it is also the write primitive the
    /// freelist policy uses for trunk and header pages.
    pub(crate) fn stage_write(&mut self, id: PageId, bytes: &[u8]) -> Result<()> {
        // Acquire the write lock lazily on a DEFERRED transaction's first write, BEFORE
        // touching the overlay or any savepoint, so a BUSY leaves the transaction's
        // state untouched (clean, rollback-safe). A no-op once the lock is held.
        self.ensure_write_lock()?;
        let count = self.page_count()?;
        if id == 0 || id > count {
            return Err(Error::Io(format!("page {id} is out of range (page_count = {count})")));
        }
        let page_size = self.store.page_size() as usize;
        if bytes.len() != page_size {
            return Err(Error::Format(format!(
                "stage_write: a page must be exactly {page_size} bytes, got {}",
                bytes.len()
            )));
        }
        let txn = self
            .txn
            .as_mut()
            .ok_or_else(|| Error::Io("stage_write: no active transaction".into()))?;
        // Capture the page's pre-image for any open savepoint BEFORE overwriting it,
        // so `rollback to` can restore exactly the bytes present at the mark.
        txn.capture_pre_image(id);
        match txn.overlay.entry(id) {
            Entry::Occupied(mut slot) => slot.get_mut().copy_from_slice(bytes),
            Entry::Vacant(slot) => {
                slot.insert(Box::from(bytes));
            }
        }
        Ok(())
    }

    /// Borrow a page MUTABLY within the current transaction (copy-on-write), the
    /// in-place counterpart of [`Cow::stage_write`]. Semantics match `stage_write`
    /// EXACTLY except that the caller edits the dirty buffer in place instead of
    /// handing over whole-page bytes: the first `page_mut`-or-`write_page` of a page in
    /// the transaction stages it (a single copy-on-write clone of the committed image)
    /// and captures its savepoint pre-image; the returned `&mut` points into that same
    /// dirty overlay buffer, so repeated edits to one page in one transaction never
    /// re-copy or re-journal it. Durability is `stage_write`'s: the dirty overlay is
    /// journalled once at `commit` from the committed image, so rollback, savepoints,
    /// and crash recovery treat a `page_mut` page identically to a `write_page` page.
    ///
    /// Requires an active transaction: a returned `&mut` borrow cannot auto-commit, so
    /// (unlike `write_page`) this never begins an implicit transaction — it fails
    /// closed, and the b-tree falls back to `write_page` when no transaction is open.
    pub(crate) fn page_mut(&mut self, id: PageId) -> Result<&mut [u8]> {
        // Acquire the write lock lazily on a DEFERRED transaction's first write, BEFORE
        // staging or capturing any savepoint pre-image, so a BUSY leaves the
        // transaction's state untouched (clean, rollback-safe). A no-op once held.
        self.ensure_write_lock()?;
        let count = self.page_count()?;
        if id == 0 || id > count {
            return Err(Error::Io(format!("page {id} is out of range (page_count = {count})")));
        }
        if self.txn.is_none() {
            return Err(Error::Io(
                "page_mut: no active transaction (a mutable page borrow cannot auto-commit)".into(),
            ));
        }
        // Copy-on-write: clone the committed image into the overlay only if this page
        // is not already staged. Read the committed bytes (an immutable borrow of the
        // store) BEFORE the `&mut` borrow of the transaction, so the two disjoint field
        // borrows never overlap.
        let staged = self.txn.as_ref().expect("txn present").overlay.contains_key(&id);
        let committed: Option<Box<[u8]>> =
            if staged { None } else { Some(Box::from(self.store.read_committed(id)?)) };
        let txn = self.txn.as_mut().expect("txn present");
        // Capture the savepoint pre-image BEFORE the page is populated, so a first-touch
        // page records the "was-absent" marker (rollback-to removes the overlay entry,
        // falling back to the committed image) — the same ordering as `stage_write`.
        txn.capture_pre_image(id);
        if let Some(bytes) = committed {
            txn.overlay.insert(id, bytes);
        }
        Ok(&mut txn.overlay.get_mut(&id).expect("overlay entry present after staging")[..])
    }

    /// Grow the database by one zero-filled page within the current transaction and
    /// return its id. The new page is staged in the overlay (so a read of it before
    /// any write sees zeros) and the committed store is left untouched until
    /// `commit`. Used by the freelist allocator's grow path. Requires an active
    /// transaction. Refuses to grow past `PageId::MAX` so the count never wraps.
    pub(crate) fn grow_one(&mut self) -> Result<PageId> {
        // Acquire the write lock lazily on a DEFERRED transaction's first write, BEFORE
        // growing the count or staging the new page, so a BUSY leaves the transaction's
        // state untouched (clean, rollback-safe). A no-op once the lock is held.
        self.ensure_write_lock()?;
        let page_size = self.store.page_size() as usize;
        let txn = self
            .txn
            .as_mut()
            .ok_or_else(|| Error::Io("allocate_page: no active transaction".into()))?;
        if txn.page_count == PageId::MAX {
            return Err(Error::Io("page count would overflow PageId (u32)".into()));
        }
        let id = txn.page_count + 1;
        // A freshly grown page has no prior overlay entry, so its pre-image is the
        // "was-absent" marker: `rollback to` a mark below `id` both resets the
        // page_count (making `id` out of range) and removes this overlay entry.
        txn.capture_pre_image(id);
        txn.page_count = id;
        txn.overlay.insert(id, vec![0u8; page_size].into_boxed_slice());
        Ok(id)
    }

    /// Shrink the transaction's page count to `new_count`, dropping any overlay
    /// entries past it. This is the transactional counterpart of [`Cow::grow_one`]:
    /// it only moves the transaction's live count (and forgets staged bytes for the
    /// removed ids); the committed store is not touched until `commit` hands it the
    /// smaller `new_count`, at which point the backing truncates (a `MemStore` shrinks
    /// its vector, a `DiskStore` journals the removed pages and `set_len`s the file, a
    /// `WalStore` records the smaller size in the commit frame). `rollback` restores the
    /// committed count exactly as for a grow, so a declined compaction leaves no trace.
    ///
    /// Used by auto_vacuum reclamation ([`crate::reclaim`]) both standalone (the explicit
    /// `PRAGMA incremental_vacuum`, no open savepoint) and inside the atomic FULL-commit
    /// reclaim scope ([`Cow::with_reclaim_scope`]). SAVEPOINT-AWARE: when any undo scope is
    /// open it captures the pre-image of each OVERLAY page it is about to drop BEFORE
    /// dropping it, so a `rollback to` / declined reclaim restores those bytes AND the
    /// higher page count exactly. The cost stays bounded by the overlay — only staged pages
    /// need a pre-image; a page absent from the overlay is restored by the page count alone
    /// (its committed bytes are untouched until `apply_commit`) — never by
    /// `page_count - new_count`. Refuses to drop page 1 (`new_count >= 1`) or to "shrink"
    /// upward.
    pub(crate) fn truncate_to(&mut self, new_count: PageId) -> Result<()> {
        self.ensure_write_lock()?;
        let txn = self
            .txn
            .as_mut()
            .ok_or_else(|| Error::Io("truncate_to: no active transaction".into()))?;
        if new_count < 1 {
            return Err(Error::Io("truncate_to: cannot shrink below page 1".into()));
        }
        if new_count > txn.page_count {
            return Err(Error::Io(format!(
                "truncate_to: new_count {new_count} exceeds current page_count {}",
                txn.page_count
            )));
        }
        // Under an open undo scope, record the pre-image of every overlay page about to be
        // dropped so the scope can restore it. Collect the ids first (owned) to release the
        // `overlay` borrow before the `capture_pre_image` mutable borrow. A no-op when no
        // savepoint is open (the standalone reclaim path), leaving that path unchanged.
        if !txn.savepoints.is_empty() {
            let dropped: Vec<PageId> =
                txn.overlay.keys().copied().filter(|&id| id > new_count).collect();
            for id in dropped {
                txn.capture_pre_image(id);
            }
        }
        txn.overlay.retain(|&id, _| id <= new_count);
        txn.page_count = new_count;
        Ok(())
    }

    /// Run `f` inside a bounded overlay UNDO scope for an in-commit reclamation attempt,
    /// returning `Some(value)` when `f` succeeds (its overlay changes are KEPT) or `None`
    /// when `f` returns an error (its overlay changes are ROLLED BACK, leaving the
    /// transaction exactly as it was on entry). The scope is internally a [`Savepoint`], so
    /// every `stage_write` / `page_mut` / `grow_one` / `truncate_to` under it captures a
    /// pre-image automatically — but it carries NONE of a user `SAVEPOINT`'s naming or
    /// outermost-release-commits semantics. It exists only so the auto_vacuum commit
    /// orchestrator ([`crate::av_commit`]) can DECLINE a FULL compaction (its verify-or-abort
    /// step, or any relocation fault) and still commit the user's data UN-compacted, rather
    /// than ever writing a torn or incorrect file.
    ///
    /// `f`'s error is deliberately mapped to `None`, not propagated: for the compaction
    /// caller a reclaim decline/fault is a safe "keep the data, skip the compaction" outcome
    /// — the same "never corrupt, decline instead" stance the standalone
    /// [`Cow::incremental_vacuum`] takes — not a reason to fail the user's commit. A genuine
    /// durable-apply fault still surfaces later, in [`Cow::commit`] itself.
    pub(crate) fn with_reclaim_scope<T>(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<Option<T>> {
        self.begin_reclaim_scope()?;
        match f(self) {
            Ok(value) => {
                self.commit_reclaim_scope()?;
                Ok(Some(value))
            }
            Err(_decline) => {
                self.rollback_reclaim_scope()?;
                Ok(None)
            }
        }
    }

    /// Push a fresh reclaim undo scope: a [`Savepoint`] marking the current overlay (an
    /// empty pre-image delta) and page count.
    ///
    /// INVARIANT — why reusing the user `savepoints` stack for an anonymous scope is safe: a
    /// reclaim scope is pushed ONLY by [`Cow::with_reclaim_scope`] and popped by its matching
    /// commit/rollback within the SAME synchronous [`Cow::commit`] call, which then takes
    /// (drops) the whole `txn`. So the scope is always the INNERMOST savepoint, is always
    /// balanced (push-use-pop) before any other code runs, and no user `SAVEPOINT` / `RELEASE`
    /// / `ROLLBACK TO` can interleave to address it by depth. Correctness does NOT rely on the
    /// stack being empty here: if an enclosing user savepoint happens to exist, KEEP merges
    /// this scope's pre-images down into it (harmless — `commit` discards it) and DECLINE
    /// restores to this scope's own mark. If the commit path ever gains re-entrancy, or a
    /// reclaim scope's lifetime widens beyond one `commit`, revisit this coupling.
    fn begin_reclaim_scope(&mut self) -> Result<()> {
        let txn = self
            .txn
            .as_mut()
            .ok_or_else(|| Error::Io("reclaim scope: no active transaction".into()))?;
        txn.savepoints.push(Savepoint { pre_images: HashMap::new(), page_count: txn.page_count });
        Ok(())
    }

    /// Pop the innermost reclaim scope KEEPING its changes, merging its pre-images down into
    /// any enclosing scope (so that scope still restores to its own mark) — mirroring
    /// [`Cow::release_savepoint`]'s merge-down. In the commit orchestrator there is no
    /// enclosing scope, so this simply drops the delta.
    fn commit_reclaim_scope(&mut self) -> Result<()> {
        let txn = self
            .txn
            .as_mut()
            .ok_or_else(|| Error::Io("reclaim scope: no active transaction".into()))?;
        let scope = txn
            .savepoints
            .pop()
            .ok_or_else(|| Error::Io("reclaim scope: release with no scope open".into()))?;
        if let Some(enclosing) = txn.savepoints.last_mut() {
            for (page, pre) in scope.pre_images {
                enclosing.pre_images.entry(page).or_insert(pre);
            }
        }
        Ok(())
    }

    /// Pop the innermost reclaim scope RESTORING the overlay and page count to its mark —
    /// undo every relocation, freelist rewrite, and truncation the compaction staged,
    /// leaving the user's data exactly as `finalize` produced it (the decline path).
    /// Mirrors [`Cow::rollback_to_savepoint`]'s restore for a single anonymous scope.
    fn rollback_reclaim_scope(&mut self) -> Result<()> {
        let txn = self
            .txn
            .as_mut()
            .ok_or_else(|| Error::Io("reclaim scope: no active transaction".into()))?;
        let scope = txn
            .savepoints
            .pop()
            .ok_or_else(|| Error::Io("reclaim scope: rollback with no scope open".into()))?;
        for (page, pre) in scope.pre_images {
            match pre {
                Some(bytes) => {
                    txn.overlay.insert(page, bytes);
                }
                None => {
                    txn.overlay.remove(&page);
                }
            }
        }
        txn.page_count = scope.page_count;
        Ok(())
    }

    /// Open a savepoint over the current transaction, starting one implicitly when
    /// none is active (a `SAVEPOINT` issued in autocommit begins a transaction that
    /// persists until the matching `RELEASE`, or a full `commit`/`rollback`). Returns
    /// the savepoint's depth (its index in the stack), which the caller pairs with a
    /// name to later address it in [`Cow::release_savepoint`] /
    /// [`Cow::rollback_to_savepoint`].
    ///
    /// The implicitly-started transaction is DEFERRED: a bare `SAVEPOINT` outside a
    /// `BEGIN...COMMIT` behaves "the same as BEGIN DEFERRED TRANSACTION"
    /// (lang_savepoint.html §2), so it must pin a read snapshot only and take NO write
    /// lock — the first actual page write under the savepoint upgrades via
    /// [`Cow::ensure_write_lock`], exactly like a bare `BEGIN`. Using the eager
    /// [`Cow::begin`] here would wrongly grab the single WAL write lock and `SQLITE_BUSY`
    /// a concurrent writer that real SQLite lets commit. `begin_deferred` is observably
    /// identical to `begin` on the single-connection backings (MemStore and the rollback
    /// DiskStore, where the first write upgrades exactly as before), so this changes only
    /// the WAL two-connection case — correctly.
    pub(crate) fn savepoint(&mut self) -> Result<usize> {
        if self.txn.is_none() {
            self.begin_deferred()?;
            // Mark the implicitly-started transaction so releasing this savepoint (if
            // it is the outermost) commits, per SQLite's "outermost RELEASE == COMMIT".
            self.txn
                .as_mut()
                .expect("begin_deferred just set the transaction")
                .started_by_savepoint = true;
        }
        let txn =
            self.txn.as_mut().ok_or_else(|| Error::Io("savepoint: no active transaction".into()))?;
        let depth = txn.savepoints.len();
        txn.savepoints.push(Savepoint { pre_images: HashMap::new(), page_count: txn.page_count });
        Ok(depth)
    }

    /// Release the savepoint at `depth` and every savepoint above it, keeping their
    /// changes (a savepoint's `RELEASE` is "like a COMMIT for a SAVEPOINT").
    ///
    /// - Inner release (`depth > 0`): merge the released savepoints' pre-images DOWN
    ///   into the enclosing savepoint `depth - 1`, keeping only pages it does not yet
    ///   record — so its delta still restores the overlay to ITS mark, and a later
    ///   `rollback to`/`release` of an outer savepoint remains correct. Draining in
    ///   ascending order makes the oldest pre-image win for a page recorded at
    ///   several levels.
    /// - Outermost release (`depth == 0`): drop the savepoints. If the transaction
    ///   was started by this savepoint, commit it (outermost RELEASE == COMMIT);
    ///   otherwise an explicit `BEGIN` encloses it, so the transaction stays open.
    pub(crate) fn release_savepoint(&mut self, depth: usize) -> Result<()> {
        {
            let txn = self
                .txn
                .as_mut()
                .ok_or_else(|| Error::Io("release_savepoint: no active transaction".into()))?;
            if depth >= txn.savepoints.len() {
                return Err(Error::Io(format!(
                    "release_savepoint: depth {depth} is out of range (open savepoints = {})",
                    txn.savepoints.len()
                )));
            }
            if depth > 0 {
                let released: Vec<Savepoint> = txn.savepoints.drain(depth..).collect();
                let enclosing = &mut txn.savepoints[depth - 1];
                for sp in released {
                    for (page, pre) in sp.pre_images {
                        enclosing.pre_images.entry(page).or_insert(pre);
                    }
                }
                return Ok(());
            }
            // depth == 0: releasing the outermost savepoint.
            txn.savepoints.clear();
            if !txn.started_by_savepoint {
                // A BEGIN transaction encloses it: keep the transaction open, just
                // discard the (now empty) savepoint stack.
                return Ok(());
            }
            // Fall through to commit; drop the `txn` borrow first.
        }
        self.commit()
    }

    /// Roll the transaction back to the savepoint at `depth`, restoring the overlay
    /// and page count to exactly their state when that savepoint was established
    /// (lang_savepoint §2: "reverts the state of the database back to what it was
    /// just after the corresponding SAVEPOINT"). The savepoint is KEPT and reusable —
    /// `rollback to` does not cancel the transaction — with its own delta cleared so
    /// later writes re-record. Savepoints ABOVE `depth` are discarded.
    ///
    /// Applying the deltas from the top down (innermost first, `depth` last) makes
    /// `depth`'s pre-image win for any page recorded at multiple levels, so the final
    /// overlay is `depth`'s state, not an inner one's.
    pub(crate) fn rollback_to_savepoint(&mut self, depth: usize) -> Result<()> {
        let txn = self
            .txn
            .as_mut()
            .ok_or_else(|| Error::Io("rollback_to_savepoint: no active transaction".into()))?;
        if depth >= txn.savepoints.len() {
            return Err(Error::Io(format!(
                "rollback_to_savepoint: depth {depth} is out of range (open savepoints = {})",
                txn.savepoints.len()
            )));
        }
        let mark = txn.savepoints[depth].page_count;
        let mut released: Vec<Savepoint> = txn.savepoints.drain(depth..).collect();
        // `released[0]` is `depth`; popping from the end applies innermost first and
        // `depth` last, so `depth`'s bytes are the ones that survive.
        while let Some(sp) = released.pop() {
            for (page, pre) in sp.pre_images {
                match pre {
                    Some(bytes) => {
                        txn.overlay.insert(page, bytes);
                    }
                    None => {
                        txn.overlay.remove(&page);
                    }
                }
            }
        }
        txn.page_count = mark;
        // Re-establish `depth` (kept, with an empty delta) so it can be rolled back
        // to or released again.
        txn.savepoints.push(Savepoint { pre_images: HashMap::new(), page_count: mark });
        Ok(())
    }
}
