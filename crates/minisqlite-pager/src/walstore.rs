//! `WalStore` — the WAL-mode committed-image backing behind the shared
//! copy-on-write layer ([`Cow`](crate::cow::Cow)), the WAL counterpart of the
//! rollback-journal [`DiskStore`](crate::diskstore::DiskStore).
//!
//! It implements the same crate-private [`PageStore`](crate::store::PageStore) seam,
//! so it reuses ALL of begin/read/write/allocate/free/commit/rollback from `Cow`
//! (there is ONE transaction layer, never a forked WAL pager). What it owns that the
//! rollback store does not:
//!
//! - **Reads resolve through the WAL** at a pinned snapshot (see
//!   [`crate::walindex::FrameIndex::resolve`]), falling back to the database file for
//!   pages the WAL does not carry.
//! - **A commit appends frames to the `<db>-wal` file** (never writing the database
//!   file); the fsync of the commit frame is the durability point (walformat §4).
//! - **A checkpoint** folds the WAL back into the database file and, when it fully
//!   drains with no reader behind, resets the WAL.
//! - **Cross-connection coordination** via [`SharedWal`]: the write lock, reader
//!   marks, and the generation fence that closes the WAL reset-vs-writer bug.
//!
//! ## Local snapshot vs shared truth
//! Each connection keeps a private [`LocalSnapshot`]: a copy of the WAL bytes, the
//! index (a cheap `Arc` clone of the shared one), and a pinned `snapshot_mx`, so its
//! reads borrow page bytes tied to `&self` with no lock on the hot path (see
//! [`crate::walshared`]). It is refreshed from [`SharedWal`] as ONE unit — via
//! [`LocalSnapshot::adopt`] — at the begin-write / begin-read / commit / checkpoint
//! boundaries and nowhere else, so between boundaries the snapshot is stable: that is
//! exactly WAL snapshot isolation.
//!
//! ## Cost discipline (no per-operation re-scan)
//! Refreshing a snapshot and re-indexing after a commit both adopt the shared index by
//! `Arc` clone / in-place extend, NEVER by re-`scan`ning the WAL — `scan` is O(total
//! WAL bytes) and would make every read/write/commit cost proportional to the whole
//! log (O(N^2) as the WAL grows). Only `open`, a reset, and `checkpoint` (each
//! inherently O(WAL) or once-per-connection) scan. See [`crate::walindex`].
//!
//! ## Stable-address reads (soundness, no `unsafe`)
//! `read_committed(&self) -> &[u8]` must hand out a borrow whose address survives
//! later reads under the same `&self`. A WAL-resolved read borrows into
//! `snap.wal_bytes` (a `Vec` never mutated under `&self` — every refresh takes
//! `&mut self`); a database-file read borrows into `snap.db_cache`, the same
//! append-only stable-address [`elsa::FrozenMap`] the rollback store uses.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use elsa::FrozenMap;
use minisqlite_fileformat::{DatabaseHeader, HEADER_SIZE};
use minisqlite_types::{Error, Result};
use minisqlite_wal::index::FrameMeta;
use minisqlite_wal::{
    checkpoint as wal_checkpoint, frame_offset, frame_stride, reset_header, scan, write_frame,
    CheckpointOptions, DbSink, WalHeader, WAL_HEADER_SIZE,
};

use crate::busy::busy_error;
use crate::checkpoint::{CheckpointMode, CheckpointReport};
use crate::store::{preflight_write_set, PageStore};
use crate::walindex::FrameIndex;
use crate::walshared::{adopt_bytes, fresh_u32, remove_one, SharedInner, SharedWal};

/// A connection's private, lock-free-readable snapshot of the committed WAL: a copy
/// of the WAL bytes plus the index and pinned ceiling consistent with them, the
/// generation / db-generation those belong to, and the base-image page cache. All of
/// it is refreshed together, from the shared truth, by [`LocalSnapshot::adopt`].
///
/// Grouping these fields (rather than scattering them on `WalStore`) is what lets the
/// four snapshot boundaries share ONE refresh routine without fighting the borrow
/// checker: `self.snap.adopt(&g)` borrows `self.snap` mutably and `self.shared`
/// (through the guard `g`) immutably — disjoint fields — so the refresh cannot drift
/// per-site (a missed step used to be a latent snapshot/cache bug).
struct LocalSnapshot {
    /// This connection's copy of the WAL image, consistent with `snapshot_mx`. Reads
    /// borrow page bytes out of this (tied to `&self`); only a `&mut self` refresh
    /// mutates it, so an outstanding read borrow never sees it move.
    wal_bytes: Vec<u8>,
    /// The frame index at this snapshot — a cheap `Arc` clone of the shared index (no
    /// per-snapshot re-scan / re-checksum).
    index: Arc<FrameIndex>,
    /// The pinned snapshot ceiling: reads resolve against `1..=snapshot_mx`, so
    /// commits by other connections past this point are invisible until the next
    /// snapshot boundary (WAL snapshot isolation).
    snapshot_mx: u32,
    /// The WAL generation the local copy belongs to; a mismatch with the shared
    /// generation (a reset happened) forces a wholesale re-copy on refresh.
    local_generation: u64,
    /// The database file's committed page count (used only when the snapshot has no
    /// WAL commit — otherwise the logical size comes from the WAL).
    db_page_count: u32,
    /// Load-on-miss, stable-address cache of DATABASE-FILE pages (the base image).
    /// Cleared when a checkpoint (this connection's or another's) rewrites the file.
    db_cache: FrozenMap<u32, Box<[u8]>>,
    /// The database-file generation this connection's `db_cache` reflects. A mismatch
    /// with the shared `db_generation` (some connection checkpointed the file) means
    /// the base-page cache may be stale and is dropped on the next refresh.
    local_db_generation: u64,
}

impl LocalSnapshot {
    /// Refresh this snapshot from the shared truth `g` (called under the shared lock
    /// at EVERY snapshot boundary — the single point so the boundaries cannot drift).
    /// Copies the appended WAL tail, mirrors the db page count, drops the base-image
    /// cache iff a checkpoint advanced the db file (`db_generation` changed), adopts
    /// the shared index by cheap `Arc` clone (never a re-scan), and re-pins the
    /// snapshot ceiling.
    fn adopt(&mut self, g: &SharedInner) {
        adopt_bytes(&mut self.wal_bytes, &mut self.local_generation, &g.wal_bytes, g.generation);
        self.db_page_count = g.db_page_count;
        if g.db_generation != self.local_db_generation {
            self.db_cache = FrozenMap::new();
            self.local_db_generation = g.db_generation;
        }
        self.index = Arc::clone(&g.index);
        self.snapshot_mx = self.index.mx_frame();
    }
}

/// The WAL-mode committed image for one connection: the database file, this
/// connection's handle to the shared `-wal` file, its private [`LocalSnapshot`], and
/// a handle to the cross-connection [`SharedWal`] coordinator plus the reader/writer
/// tokens it currently holds there.
pub(crate) struct WalStore {
    /// The database file, opened read+write. Read via positioned I/O (`&self`), so
    /// no cursor state is shared. In WAL mode a commit never writes it — only a
    /// checkpoint does.
    db_file: File,
    /// This connection's handle to the shared `<db>-wal` file. Every connection to
    /// the same database opens the same file, so an fsync on any handle flushes the
    /// shared log.
    wal_file: File,
    /// Byte size of every page, fixed for the life of the store (from the db header).
    page_size: u32,

    /// The cross-connection coordinator (shared WAL truth + write lock + reader
    /// marks + generation). Shared by every connection to the same canonical path.
    shared: Arc<SharedWal>,

    /// This connection's private snapshot of the committed WAL (see [`LocalSnapshot`]).
    snap: LocalSnapshot,

    /// The reader mark this connection currently holds in `SharedWal::readers`
    /// (`Some` between `begin_read` and `end_read`), so `end_read`/`Drop` remove the
    /// right one.
    reader_mark: Option<u32>,
    /// True while this connection holds the shared write lock (between `begin_write`
    /// and `end_write`), so `end_write`/`Drop` only release a lock we actually took.
    holds_write_lock: bool,
}

impl WalStore {
    /// Open a WAL-mode database at `path` (the caller — [`crate::diskpager`] — has
    /// already detected WAL mode from the page-1 header). Reads the database header
    /// for the page size and base page count (mirroring [`DiskStore::open`]), opens
    /// (creating) the `<db>-wal` sidecar, and joins (or creates) the process-global
    /// [`SharedWal`] for this canonical path.
    pub(crate) fn open(path: &Path) -> Result<WalStore> {
        // The database must already exist to be in WAL mode; do not create it here.
        let db_file = OpenOptions::new().read(true).write(true).create(false).open(path)?;
        let file_len = db_file.metadata()?.len();
        if file_len < HEADER_SIZE as u64 {
            return Err(Error::Format(format!(
                "WAL-mode database file is {file_len} bytes, too small to hold a 100-byte header"
            )));
        }
        let mut head = [0u8; HEADER_SIZE];
        db_file.read_exact_at(&mut head, 0)?;
        let header = DatabaseHeader::read(&head)?;
        let page_size = header.page_size;
        let ps = page_size as u64;
        if file_len % ps != 0 {
            return Err(Error::Format(format!(
                "WAL-mode database file length {file_len} is not a whole multiple of page size {ps}"
            )));
        }
        let db_page_count = if header.in_header_size_valid() {
            header.database_size_pages
        } else {
            (file_len / ps) as u32
        };

        let wal_path = wal_path_for(path);
        let wal_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&wal_path)?;

        // The db exists, so canonicalize succeeds; it is the coordination key so two
        // handles to the same file (even via different relative paths) share one
        // coordinator.
        let key = std::fs::canonicalize(path)?;
        let shared = SharedWal::acquire(key, || {
            // Only the FIRST connection reads the -wal file and builds the shared
            // state; later connections reuse the live coordinator.
            let disk_wal = read_all(&wal_file)?;
            Ok(SharedInner::new(page_size, db_page_count, disk_wal))
        })?;

        // Snapshot the shared state into the local view under the lock. The WAL bytes
        // are copied once here (O(WAL) at open); the index is a cheap `Arc` clone.
        let snap = {
            let g = shared.lock();
            if g.page_size != page_size {
                return Err(Error::Format(format!(
                    "WAL coordinator page size {} disagrees with database page size {page_size}",
                    g.page_size
                )));
            }
            LocalSnapshot {
                wal_bytes: g.wal_bytes.clone(),
                index: Arc::clone(&g.index),
                snapshot_mx: g.index.mx_frame(),
                local_generation: g.generation,
                db_page_count: g.db_page_count,
                db_cache: FrozenMap::new(),
                local_db_generation: g.db_generation,
            }
        };

        Ok(WalStore {
            db_file,
            wal_file,
            page_size,
            shared,
            snap,
            reader_mark: None,
            holds_write_lock: false,
        })
    }

    /// Read page `id` from the database file (the base image), caching it in the
    /// append-only stable-address cache. Used when the WAL snapshot does not carry
    /// the page.
    fn read_db_page(&self, id: u32) -> Result<&[u8]> {
        if let Some(bytes) = self.snap.db_cache.get(&id) {
            return Ok(bytes);
        }
        let page_size = self.page_size as usize;
        let mut buf = vec![0u8; page_size].into_boxed_slice();
        self.db_file.read_exact_at(&mut buf, db_page_offset(id, self.page_size))?;
        Ok(self.snap.db_cache.insert(id, buf))
    }
}

impl PageStore for WalStore {
    fn page_size(&self) -> u32 {
        self.page_size
    }

    fn committed_count(&self) -> Result<u32> {
        // When the snapshot carries a WAL commit, the logical size is the WAL's
        // db-size at that snapshot (the WAL is authoritative for a WAL-mode
        // database's size). Otherwise the database file's own count applies.
        if self.snap.snapshot_mx > 0 {
            let n = self.snap.index.db_size_at(self.snap.snapshot_mx);
            if n > 0 {
                return Ok(n);
            }
        }
        Ok(self.snap.db_page_count)
    }

    fn read_committed(&self, id: u32) -> Result<&[u8]> {
        let count = self.committed_count()?;
        if id == 0 || id > count {
            return Err(Error::Io(format!("page {id} is out of range (committed_count = {count})")));
        }
        // Resolve through the WAL at the pinned snapshot; a hit borrows the frame's
        // page data out of the local WAL bytes. A miss means the page's newest
        // committed image is in the database file.
        if self.snap.snapshot_mx > 0 {
            if let Some(frame) = self.snap.index.resolve(id, self.snap.snapshot_mx) {
                return self.snap.index.frame_page_data(&self.snap.wal_bytes, frame);
            }
        }
        self.read_db_page(id)
    }

    fn apply_commit(&mut self, dirty: BTreeMap<u32, Box<[u8]>>, new_count: u32) -> Result<()> {
        // A no-op commit (nothing staged) performs no I/O and does not bump the
        // change counter — same as the rollback path and real SQLite.
        if dirty.is_empty() {
            return Ok(());
        }
        let page_size = self.page_size as usize;

        // Build the write set with page-1 header maintenance from the LOCAL committed
        // view (refreshed at begin_write). Only read the committed page 1 when this
        // transaction did not already stage it (so freelist edits survive).
        let mut write_set = dirty;
        let current = if !write_set.contains_key(&1) && self.committed_count()? >= 1 {
            Some(self.read_committed(1)?.to_vec())
        } else {
            None
        };
        crate::page1::maintain_header(&mut write_set, new_count, current.as_deref())?;

        // Preflight the whole write set before any durable write: every id in range
        // and exactly page_size bytes (shared fail-closed guard at every durable
        // boundary — see [`preflight_write_set`]).
        preflight_write_set(&write_set, new_count, page_size)?;

        // ---- Append under the shared lock (we hold the write lock) ----------------
        let mut g = self.shared.lock();
        // FENCE: the write lock we hold blocks any reset, so no generation change can
        // have happened since begin_write. Re-read the append parameters (salts,
        // running checksum) from the shared state ANYWAY — never trust a copied
        // header — which is precisely how the 16-year SQLite WAL reset-vs-writer bug
        // is avoided.
        debug_assert_eq!(
            g.generation, self.snap.local_generation,
            "WAL reset raced a held write lock — the fence was violated"
        );

        // Initialize the WAL header on the first append (fresh or post-reset WAL).
        if g.header.is_none() {
            let h = WalHeader::new(self.page_size, fresh_u32(), fresh_u32(), 0, false);
            let hbytes = h.serialize();
            // Write the header to the FILE before publishing it into shared state: if
            // this I/O faults, `?` returns with `g.header` still `None` and `g.wal_bytes`
            // untouched, so the shared state stays consistent with the file. Otherwise a
            // sibling could adopt "header initialized" and append frames to a headerless
            // file, which reopen's `scan` would then discard. (The header is not itself
            // the durability point — the frame fsync below is — but this ordering keeps
            // the error path clean, mirroring the append's "publish only after fsync".)
            self.wal_file.write_all_at(&hbytes, 0)?;
            self.wal_file.set_len(WAL_HEADER_SIZE as u64)?;
            g.wal_bytes.clear();
            g.wal_bytes.extend_from_slice(&hbytes);
            g.header = Some(h);
            // A one-frame-less scan of a 32-byte header is trivial; seeds the index's
            // header checksum so the first frame's running checksum continues it.
            g.index = FrameIndex::rebuilt(&g.wal_bytes, self.page_size);
        }
        let header = g.header.expect("WAL header is initialized above");
        let salts = (header.salt1, header.salt2);
        let big_endian = header.big_endian;

        // Append starting right after the committed ceiling (dropping any leftover).
        let start_frame = g.index.mx_frame() + 1;
        let ceiling = frame_offset(start_frame, self.page_size);
        let mut running = g.index.running_checksum_for_append();

        // Build the frames once, capturing each frame's metadata (page, db-size,
        // cumulative checksum) so the index can be extended WITHOUT re-scanning. The
        // last frame is the commit frame carrying the new page count (walformat §4.1).
        let n = write_set.len();
        let mut tail: Vec<u8> = Vec::with_capacity(n * frame_stride(self.page_size));
        let mut new_frames: Vec<FrameMeta> = Vec::with_capacity(n);
        for (i, (&id, buf)) in write_set.iter().enumerate() {
            let db_size = if i + 1 == n { new_count } else { 0 };
            running = write_frame(&mut tail, running, id, db_size, salts, buf, big_endian);
            new_frames.push(FrameMeta { page_no: id, db_size, checksum: running });
        }

        // Persist: trim the shared image to the ceiling, position-write the frames,
        // set the file length, and fsync — the fsync of the commit frame is the WAL
        // commit point (atomiccommit / walformat).
        g.wal_bytes.truncate(ceiling);
        self.wal_file.write_all_at(&tail, ceiling as u64)?;
        self.wal_file.set_len((ceiling + tail.len()) as u64)?;
        self.wal_file.sync_all()?;

        // Publish into the shared image and extend the shared index IN PLACE with the
        // frames we just wrote — O(pages touched), never a whole-WAL re-scan. Release
        // this writer's own snapshot handle to the index first, so `Arc::make_mut` can
        // extend without cloning whenever no *reader* is concurrently holding the
        // shared index (a concurrent reader forces one O(frames) copy — still far
        // cheaper than re-scanning + re-checksumming the entire log).
        g.wal_bytes.extend_from_slice(&tail);
        self.snap.index = Arc::new(FrameIndex::default());
        Arc::make_mut(&mut g.index).extend_commit(&new_frames);
        debug_assert_eq!(
            g.index.mx_frame(),
            start_frame - 1 + n as u32,
            "the shared index advanced by exactly the appended frame count"
        );

        // Mirror the shared committed image into this connection's snapshot so the
        // writer's own subsequent reads observe its commit. `adopt` appends just the
        // new tail in the common case, but re-copies wholesale when the local buffer
        // is not a prefix of the shared image — exactly the FIRST commit into a fresh
        // WAL, whose 32-byte header was just created in the SHARED buffer while this
        // connection's local copy was taken (at begin_write) when the WAL was empty.
        self.snap.adopt(&g);
        drop(g);
        Ok(())
    }

    fn begin_write(&mut self) -> Result<()> {
        let mut g = self.shared.lock();
        // TRY-lock: a second writer gets BUSY immediately (never blocks forever — an
        // unbounded wait would be a liveness bug) and can retry.
        if g.write_locked {
            return Err(busy_error());
        }
        g.write_locked = true;
        // Refresh the local snapshot to the latest committed state under the lock, so
        // the writer builds on the newest data.
        self.snap.adopt(&g);
        drop(g);
        self.holds_write_lock = true;
        Ok(())
    }

    fn begin_write_upgrade(&mut self) -> Result<()> {
        let mut g = self.shared.lock();
        // TRY-lock: another connection holds the write lock -> BUSY (never block).
        if g.write_locked {
            return Err(busy_error());
        }
        // Staleness: another connection committed (mx advanced) or a reset bumped the
        // generation since this connection pinned its read snapshot at begin. Upgrading
        // a read transaction over an intervening modification is not possible
        // (lang_transaction §2.1) -> BUSY. (A live reader mark blocks a reset while this
        // transaction is open, so a generation change here is defensive.)
        if g.index.mx_frame() > self.snap.snapshot_mx || g.generation != self.snap.local_generation
        {
            return Err(busy_error());
        }
        // Take the lock, KEEPING the pinned snapshot (do NOT adopt) — the writer reads
        // its own snapshot plus its own uncommitted overlay. The snapshot is current
        // (we just checked it is not stale), so the commit's frame append starts at the
        // correct mx and the `apply_commit` generation FENCE still holds.
        g.write_locked = true;
        drop(g);
        self.holds_write_lock = true;
        Ok(())
    }

    fn end_write(&mut self) -> Result<()> {
        // Idempotent: only release a lock this connection actually acquired, so a
        // begin_write that failed with BUSY (never took the lock) is unaffected.
        if self.holds_write_lock {
            let mut g = self.shared.lock();
            g.write_locked = false;
            drop(g);
            self.holds_write_lock = false;
        }
        Ok(())
    }

    fn begin_read(&mut self) -> Result<()> {
        let mut g = self.shared.lock();
        // Refresh to the latest committed state, then pin this snapshot so a
        // concurrent checkpoint will not drain the WAL past it (and, crucially, will
        // not RESET it — a reset is deferred until no reader is active).
        self.snap.adopt(&g);
        let mark = self.snap.snapshot_mx;
        // Replace any prior mark from this connection (defensive against a missing
        // end_read), then register the new one.
        if let Some(old) = self.reader_mark.take() {
            remove_one(&mut g.readers, old);
        }
        g.readers.push(mark);
        self.reader_mark = Some(mark);
        Ok(())
    }

    fn end_read(&mut self) -> Result<()> {
        if let Some(mark) = self.reader_mark.take() {
            let mut g = self.shared.lock();
            remove_one(&mut g.readers, mark);
        }
        Ok(())
    }

    fn checkpoint(&mut self, mode: CheckpointMode) -> Result<CheckpointReport> {
        let mut g = self.shared.lock();

        // The committed frames currently in the log — the pragma's `log` column
        // (pragma.html: "modified pages written to the write-ahead log"). Read BEFORE
        // any drain or reset mutates the index.
        let log_frames = g.index.mx_frame();

        // NOOP obtains the returned values only: it copies no frame and never resets
        // (pragma.html #pragma_wal_checkpoint). `checkpointed` is the frames already
        // backfilled (`nBackfill`); the `.min` is defensive (nBackfill ≤ mxFrame).
        if mode == CheckpointMode::Noop {
            let checkpointed = g.nbackfill.min(log_frames);
            return Ok(CheckpointReport {
                busy: false,
                log: Some(log_frames),
                checkpointed: Some(checkpointed),
            });
        }

        // Bound the drain by the oldest active reader mark (a reader still needs
        // everything up to its mark in the WAL). `None` ⇒ no reader ⇒ full drain.
        let reader_limit = g.readers.iter().copied().min();
        let opts = CheckpointOptions { nbackfill: g.nbackfill, reader_limit };

        // Drive the pure checkpoint over the shared WAL into the database file. A
        // checkpoint is inherently O(WAL), so building a `WalIndex` by `scan` here (a
        // rare, non-hot path) — rather than on every read/commit — is acceptable.
        let wi = scan(&g.wal_bytes);
        let res = {
            let mut sink = DbFileSink { file: &self.db_file, page_size: self.page_size };
            let wal_file = &self.wal_file;
            wal_checkpoint(&g.wal_bytes, &wi, &mut sink, opts, || {
                wal_file.sync_all().map_err(Error::from)
            })?
        };

        // A drain that advanced nBackfill copied frames into the database file, so
        // every connection's base-page cache for those pages is now stale; bump the
        // shared db generation so they drop it on their next refresh.
        let progressed = res.frames_backfilled > opts.nbackfill;
        g.nbackfill = res.frames_backfilled;
        if res.db_size_pages > 0 {
            g.db_page_count = res.db_size_pages;
        }
        if progressed {
            g.db_generation = g.db_generation.wrapping_add(1);
        }

        // The frames now durably in the database file — the pragma's `checkpointed`
        // column (the cumulative nBackfill). Captured BEFORE a reset zeroes it below.
        let checkpointed = res.frames_backfilled;

        // The reset gate, UNCHANGED from the single behavior this store has always
        // had: a COMPLETE drain, no writer holding the lock (the fence half — a reset
        // must never race a mid-txn append), and no active reader. A reader whose mark
        // EQUALS mxFrame still makes the drain report `complete`, but resetting then
        // would strand that reader's snapshot (a later reset+recommit advances the db
        // file past frames it must still resolve from the WAL). Every reset-eligible
        // mode resets here — so a bare/PASSIVE checkpoint still shrinks the log, which
        // the durability suite requires — and TRUNCATE additionally truncates the file.
        let did_reset = res.complete && !g.write_locked && g.readers.is_empty();

        // The `busy` column (pragma.html #pragma_wal_checkpoint): 1 iff a RESTART/FULL/
        // TRUNCATE checkpoint was blocked from completing its guarantee.
        // - PASSIVE never blocks ("the busy-handler callback is never invoked").
        // - FULL blocks until there is no writer and every reader is on the most recent
        //   snapshot — i.e. it must drain the whole log (`res.complete`) with no writer.
        // - RESTART/TRUNCATE add "block until all readers are finished with the log",
        //   which is exactly the reset condition, so busy ⇔ the reset did not happen.
        // An empty log (nothing to checkpoint) is never busy.
        let busy = log_frames > 0
            && match mode {
                CheckpointMode::Passive | CheckpointMode::Noop => false,
                CheckpointMode::Full => g.write_locked || !res.complete,
                CheckpointMode::Restart | CheckpointMode::Truncate => !did_reset,
            };

        if did_reset {
            // Bump the generation so every other connection re-copies on its next
            // refresh (its old frames are gone), and reset nBackfill.
            g.generation = g.generation.wrapping_add(1);
            g.nbackfill = 0;
            if mode == CheckpointMode::Truncate {
                // TRUNCATE: the WAL file is truncated to zero bytes on success
                // (pragma.html). Drop the in-memory header/index too, so the next
                // writer re-initializes a fresh header (apply_commit's `header == None`
                // branch) exactly as for a brand-new WAL.
                g.wal_bytes.clear();
                g.header = None;
                g.index = FrameIndex::rebuilt(&g.wal_bytes, self.page_size);
                self.wal_file.set_len(0)?;
                self.wal_file.sync_all()?;
            } else {
                // RESTART (and PASSIVE/FULL, which reset here too): rewrite a fresh
                // 32-byte header whose bumped salts invalidate every leftover frame,
                // and truncate the file back to that header so stale frames cannot be
                // mistaken for live ones on the next scan.
                let prev = g.header.expect("a complete checkpoint implies an initialized WAL");
                let new_h = reset_header(&prev, fresh_u32());
                let hbytes = new_h.serialize();
                g.wal_bytes.clear();
                g.wal_bytes.extend_from_slice(&hbytes);
                g.header = Some(new_h);
                g.index = FrameIndex::rebuilt(&g.wal_bytes, self.page_size);
                self.wal_file.write_all_at(&hbytes, 0)?;
                self.wal_file.set_len(WAL_HEADER_SIZE as u64)?;
                self.wal_file.sync_all()?;
            }
        }

        // Refresh the local snapshot from the (possibly reset) shared state. `adopt`
        // drops this connection's base-page cache exactly when the db file advanced
        // (db_generation changed) — i.e. when a drain backfilled pages — so a stale
        // cached base page is never served after our own or a peer's checkpoint.
        self.snap.adopt(&g);
        drop(g);
        Ok(CheckpointReport { busy, log: Some(log_frames), checkpointed: Some(checkpointed) })
    }
}

impl Drop for WalStore {
    fn drop(&mut self) {
        // Release any shared resource this connection still holds, so a drop
        // mid-transaction (no explicit commit/rollback/end_read — e.g. the engine
        // discarding a connection on error, matching SQLite's implicit rollback on
        // close) cannot strand the single write lock (which would permanently BUSY
        // every sibling and every fresh open on this path) or a reader mark (which
        // would pin every future checkpoint's drain bound and forbid the reset
        // forever). The shared `SharedInner` outlives any one connection, so the store
        // must defend this invariant itself rather than trust the begin/end pairing.
        if self.holds_write_lock || self.reader_mark.is_some() {
            let mut g = self.shared.lock();
            if self.holds_write_lock {
                g.write_locked = false;
                self.holds_write_lock = false;
            }
            if let Some(mark) = self.reader_mark.take() {
                remove_one(&mut g.readers, mark);
            }
        }
    }
}

/// A [`DbSink`] over the database file: checkpointed pages are position-written, the
/// size is a truncate/extend, and the barrier is an `fsync`.
struct DbFileSink<'a> {
    file: &'a File,
    page_size: u32,
}

impl DbSink for DbFileSink<'_> {
    fn write_page(&mut self, page_no: u32, data: &[u8]) -> Result<()> {
        self.file.write_all_at(data, db_page_offset(page_no, self.page_size))?;
        Ok(())
    }

    fn set_size_pages(&mut self, n_pages: u32) -> Result<()> {
        self.file.set_len(n_pages as u64 * self.page_size as u64)?;
        Ok(())
    }

    fn sync(&mut self) -> Result<()> {
        self.file.sync_all()?;
        Ok(())
    }
}

/// Byte offset of page `id` (1-based) in a page-`page_size` database file:
/// `(id - 1) * page_size`. Kept in one place so the offset math is never re-derived
/// (an off-by-one here would misplace a read or a checkpoint write).
fn db_page_offset(id: u32, page_size: u32) -> u64 {
    (id as u64 - 1) * page_size as u64
}

/// Read an entire file into a `Vec`. Used once (per database) to seed the shared WAL
/// image from the `-wal` file; the shared in-memory image is authoritative thereafter.
fn read_all(file: &File) -> Result<Vec<u8>> {
    let len = file.metadata()?.len() as usize;
    let mut buf = vec![0u8; len];
    if len > 0 {
        file.read_exact_at(&mut buf, 0)?;
    }
    Ok(buf)
}

/// The `-wal` sidecar path for a database: the database path with `-wal` appended
/// (SQLite's convention, `spec/sqlite-doc/tempfiles.html`), mirroring
/// `diskstore::journal_path_for`. Appends to the full path, so `foo.db` →
/// `foo.db-wal`.
fn wal_path_for(db: &Path) -> PathBuf {
    let mut s = db.as_os_str().to_os_string();
    s.push("-wal");
    PathBuf::from(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wal_path_appends_suffix_to_full_path() {
        assert_eq!(wal_path_for(Path::new("/tmp/foo.db")), PathBuf::from("/tmp/foo.db-wal"));
        assert_eq!(wal_path_for(Path::new("bar")), PathBuf::from("bar-wal"));
    }
}
