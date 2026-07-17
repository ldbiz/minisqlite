//! The storage seam definition. `Pager` is the one owned trait the whole engine
//! builds on for page access and copy-on-write transactions; it stays alone in this
//! module so the concrete backings (in-memory today, disk-backed later) live in
//! their own files without churning the crate root. It must remain declared exactly
//! once in this crate — do not fork a second or competing storage trait.

use minisqlite_types::Result;

use crate::PageId;
use crate::reclaim::VacuumOutcome;
use crate::checkpoint::{CheckpointMode, CheckpointReport};

/// The single storage seam: page access plus a copy-on-write transaction.
/// Object-safe, so the engine can hold a `Box<dyn Pager>` and swap an in-memory
/// backing for an on-disk one behind this seam without touching the layers above.
pub trait Pager {
    /// Borrow a page's bytes for reading. The slice is owned by the pager's cache,
    /// so a read never copies the database. Outside a transaction this is the
    /// committed image; inside one it is the transaction's view.
    fn read_page(&self, id: PageId) -> Result<&[u8]>;

    /// Number of pages currently in the database.
    fn page_count(&self) -> Result<PageId>;

    /// Byte size of every page. Fixed for the life of the pager and a power of two
    /// in `[512, 65536]`. The b-tree and overflow layers need it to size cells and
    /// overflow chains, so it belongs on the seam rather than hidden inside one
    /// backing's constructor.
    fn page_size(&self) -> u32;

    /// Begin a transaction. Later writes are copy-on-write and journalled; the
    /// database is not copied up front.
    fn begin(&mut self) -> Result<()>;

    /// Begin a DEFERRED transaction: pin a read snapshot only, deferring the single
    /// write lock until the first write (which acquires it, or returns BUSY). The
    /// default forwards to `begin()` (eager), so a non-coordinating backing is
    /// unchanged; a WAL backing overrides this to defer the write lock.
    ///
    /// This is the storage side of `BEGIN` / `BEGIN DEFERRED` (lang_transaction §2.2):
    /// a DEFERRED transaction takes no write lock until it first writes, so a read-only
    /// DEFERRED transaction never blocks a concurrent connection's commit and keeps its
    /// own historic snapshot while other connections commit. `IMMEDIATE`/`EXCLUSIVE` map
    /// to the eager `begin()` (a write lock at BEGIN).
    fn begin_deferred(&mut self) -> Result<()> {
        self.begin()
    }

    /// Stage a modified page within the current transaction (copy-on-write). The
    /// page becomes visible to reads in this transaction, and its pre-image is
    /// journalled so a crash before commit can be rolled back.
    fn write_page(&mut self, id: PageId, bytes: &[u8]) -> Result<()>;

    /// Borrow a page MUTABLY within the current transaction, copy-on-write, so the
    /// caller can edit its bytes in place instead of rebuilding a whole page and
    /// writing it back. This is the O(edit) counterpart of the O(page)
    /// read-modify-`write_page` cycle: it lets a b-tree splice one cell into a leaf
    /// without re-encoding every other cell or copying the page twice.
    ///
    /// # Semantics — identical to `write_page` except for the copy
    /// The FIRST `page_mut` or `write_page` of a page in the current transaction
    /// stages it into the transaction-private dirty buffer (copy-on-write from the
    /// committed image) and captures its pre-image for any open savepoint. The
    /// returned `&mut [u8]` points into that dirty buffer, so repeated `page_mut`
    /// calls (or edits through the borrow) for the same page in the same transaction
    /// neither re-copy nor re-journal it. Reads in this transaction observe the edits
    /// immediately (they resolve the same dirty buffer). Durability is exactly
    /// `write_page`'s: the dirty page is journalled / WAL-captured ONCE at commit from
    /// the committed image, however many times it was edited, so commit / rollback /
    /// crash recovery / savepoints treat a `page_mut` page and a `write_page` page
    /// identically.
    ///
    /// # Borrow contract
    /// - The returned slice is exactly `page_size()` bytes — the WHOLE page, including
    ///   page 1's first 100 database-header bytes and any reserved tail. The caller
    ///   must preserve bytes it does not own (the b-tree edits only the b-tree region).
    /// - The borrow ties to `&mut self`, so no other pager call overlaps it; edits are
    ///   confined to the current transaction and become durable only at `commit`.
    /// - Requires an ACTIVE transaction. A returned borrow cannot auto-commit, so
    ///   calling it in autocommit is an error; a caller with no transaction open must
    ///   use `write_page` (which auto-commits) instead.
    ///
    /// # Default
    /// The default fails closed, so a backing that has not implemented in-place
    /// mutation stays correct via the caller's `write_page` fallback
    /// (expand-then-contract: the seam grows without breaking existing backings).
    fn page_mut(&mut self, _id: PageId) -> Result<&mut [u8]> {
        Err(minisqlite_types::Error::io("this pager does not support in-place page mutation"))
    }

    /// Allocate a fresh page (from the freelist when possible) and return its id.
    fn allocate_page(&mut self) -> Result<PageId>;

    /// Release a page to the freelist so a later `allocate_page` reuses it before
    /// the file grows (fileformat2 §1.5). Like the other writes it is copy-on-write
    /// and transactional: the freelist trunk pages and the page-1 header fields it
    /// touches are staged in the current transaction and undone by `rollback`.
    /// Freeing page 1 (it holds the database header), page 0, or an id past the
    /// current page count is an error — it fails closed rather than panicking.
    ///
    /// Caller contract: the page must be *live* (reachable, not already on the
    /// freelist). Double-freeing a page is corruption — as in real SQLite — and is
    /// NOT detected here, because catching it would require scanning the freelist
    /// and break the O(1) bound; a double-freed id is listed twice and can be
    /// handed back twice by `allocate_page`. Likewise, do not write to a page
    /// between freeing it and its reuse: a freed page may be repurposed as a
    /// freelist trunk (its bytes then hold the trunk structure), so a stray write
    /// can corrupt the trunk chain.
    fn free_page(&mut self, id: PageId) -> Result<()>;

    /// Commit the transaction durably: flush the journal/WAL, then the page image,
    /// so a crash after this call returns still leaves the committed state intact.
    fn commit(&mut self) -> Result<()>;

    /// Roll back the transaction, restoring pre-image pages so no partial write
    /// remains.
    fn rollback(&mut self) -> Result<()>;

    // ---- Savepoints (default: unsupported) -------------------------------------
    //
    // Savepoints are nested, named undo points WITHIN a transaction. They are pure
    // pre-commit overlay state (no WAL/journal interaction), so a backing built on
    // the shared copy-on-write layer implements all three by forwarding to it; the
    // seam grows non-breakingly (expand-then-contract) because each method has a
    // default. A backing with no savepoint machinery fails closed rather than
    // silently pretending success. Savepoints are addressed by DEPTH (stack index);
    // the engine owns the NAME -> depth mapping.

    /// Open a savepoint over the current transaction, starting one implicitly when
    /// none is active (a `SAVEPOINT` in autocommit begins a transaction that lasts
    /// until the matching `RELEASE` or a full `commit`/`rollback`). Returns the
    /// savepoint's depth (its stack index), which the caller pairs with a name.
    fn savepoint(&mut self) -> Result<usize> {
        Err(minisqlite_types::Error::io("this pager does not support savepoints"))
    }

    /// Release the savepoint at `depth` and every savepoint above it, keeping their
    /// changes (a savepoint `RELEASE` is "like a COMMIT for a SAVEPOINT"). Releasing
    /// the outermost savepoint of a transaction that a savepoint itself started
    /// commits the transaction (outermost RELEASE == COMMIT).
    fn release_savepoint(&mut self, _depth: usize) -> Result<()> {
        Err(minisqlite_types::Error::io("this pager does not support savepoints"))
    }

    /// Roll the transaction back to the savepoint at `depth`, restoring the state to
    /// what it was when that savepoint was established. The savepoint is kept and
    /// reusable (`rollback to` does not cancel the transaction); savepoints above
    /// `depth` are discarded.
    fn rollback_to_savepoint(&mut self, _depth: usize) -> Result<()> {
        Err(minisqlite_types::Error::io("this pager does not support savepoints"))
    }

    // ---- WAL-mode additions (default no-ops) -----------------------------------
    //
    // These grow the ONE storage seam (expand-then-contract, non-breaking): every
    // method has a default so a backing that needs no coordination is unaffected,
    // while the WAL-backed disk pager overrides them (and the rollback disk pager
    // overrides `begin_read` to refresh its committed view for cross-connection
    // visibility). The engine brackets a read statement in `begin_read`/`end_read`
    // (see `minisqlite-engine`'s dispatch) and calls `checkpoint` for
    // `PRAGMA wal_checkpoint`, passing the requested [`CheckpointMode`].

    /// Begin a read transaction, refreshing to and pinning a stable committed
    /// snapshot for its duration. In WAL mode a reader records the current WAL commit
    /// ceiling so a concurrent writer's commits do not shift its view; a checkpoint
    /// will not drain the WAL past this reader's mark. The rollback disk backing
    /// re-reads the page-1 header and drops its page cache when another connection has
    /// committed (cross-connection visibility), then holds that image for the read.
    /// A no-op for the in-memory backing (single connection, no shared file).
    fn begin_read(&mut self) -> Result<()> {
        Ok(())
    }

    /// End the read transaction begun by [`Pager::begin_read`], releasing the pinned
    /// snapshot so a later checkpoint may drain past it. A no-op outside WAL mode.
    fn end_read(&mut self) -> Result<()> {
        Ok(())
    }

    /// Checkpoint in the requested [`CheckpointMode`]: fold WAL content back into the
    /// database file (WAL mode) and, when `mode` wants a reset and the whole log has
    /// drained with no reader still behind it, reset (RESTART) or truncate (TRUNCATE)
    /// the WAL. Returns the [`CheckpointReport`] the engine maps to `PRAGMA
    /// wal_checkpoint`'s `(busy, log, checkpointed)` row.
    ///
    /// A no-op outside WAL mode, where every commit already writes the database file
    /// directly; the default returns [`CheckpointReport::not_wal`] (the pragma then
    /// reports `(0, -1, -1)`), so a non-WAL backing is unaffected.
    fn checkpoint(&mut self, _mode: CheckpointMode) -> Result<CheckpointReport> {
        Ok(CheckpointReport::not_wal())
    }

    /// Auto_vacuum RECLAMATION: shrink an auto_vacuum database by removing up to `max`
    /// free pages (all of them when `None`) — the `PRAGMA incremental_vacuum(N)` reclaim
    /// and the per-commit truncation FULL mode performs (pragma.html "auto_vacuum" /
    /// "incremental_vacuum"). Relocates live tail pages into freed slots, rewrites the one
    /// pointer each moved page is named by, re-derives the ptrmap, and truncates the file;
    /// [`VacuumOutcome::root_moved`] tells the caller a `sqlite_schema.rootpage` changed so
    /// it can reload a cached catalog. Runs as its own autocommit transaction and DECLINES
    /// (reclaims nothing) rather than risk corruption on any inconsistency.
    ///
    /// Must be called in autocommit (no transaction open). The default is a no-op returning
    /// [`VacuumOutcome::default`], so a backing that does not maintain a freelist/ptrmap
    /// (or has nothing to reclaim) simply does not shrink — the file stays valid
    /// (expand-then-contract: the seam grows without breaking a backing).
    fn incremental_vacuum(&mut self, _max: Option<PageId>) -> Result<VacuumOutcome> {
        Ok(VacuumOutcome::default())
    }

    /// Read and clear "a commit since the last check RELOCATED a b-tree root". An
    /// auto_vacuum database keeps b-tree roots at the front of the file (§1.8), so a
    /// `CREATE INDEX`/`CREATE TABLE` that created a root above existing data has its root
    /// moved down at commit — which rewrites a `sqlite_schema.rootpage`. The engine polls
    /// this after a statement settles to autocommit and RELOADS the namespace's cached
    /// catalog when it returns `true`, so a cached root page never points at the pre-move
    /// slot. This is the commit-path analogue of [`VacuumOutcome::root_moved`] (which
    /// reports the reclamation path's root moves).
    ///
    /// The default returns `false` (a backing that never relocates roots never moves one),
    /// so a non-ptrmap backing pays nothing and needs no override. Any real Cow-based backing
    /// that runs `finalize` (i.e. can host an auto_vacuum database) MUST override this to
    /// delegate to `Cow::take_root_moved` — forgetting to would silently drop the root-move
    /// signal and leave the catalog pointing at a pre-relocation root (`index_insert reached a
    /// non-index b-tree page`). `MemPager` and `DiskPager` both delegate.
    fn take_root_moved(&mut self) -> bool {
        false
    }
}
