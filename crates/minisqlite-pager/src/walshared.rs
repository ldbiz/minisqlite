//! Cross-connection WAL coordination: the authoritative shared WAL state and the
//! process-global registry that lets two connections to the same file share it.
//!
//! ## Same-process assumption (documented, and the coordination layer is swappable)
//! The supported "two connections" are two `DiskPager`/`Connection` objects in the
//! SAME process on the same database file, coordinated by an in-process global
//! write lock over a fenced WAL. Connections
//! coordinate through a process-global [`REGISTRY`] keyed by the CANONICAL database
//! path (`std::fs::canonicalize`), so two [`crate::walstore::WalStore`] opened on the
//! same file share one [`SharedWal`]. If a future change needs cross-process WAL, only
//! this module changes (it becomes shared memory + file locks); the store above it is
//! written against `SharedWal`'s interface.
//!
//! ## What is shared vs local
//! [`SharedInner`] is the single source of truth for the committed WAL: the header
//! (salts / checkpoint sequence), the in-memory WAL image, the frame index, a
//! `generation` bumped on every reset, `nbackfill`, the single write lock, and the
//! active reader marks. Each connection ALSO keeps its own copy of the WAL bytes and
//! a pinned snapshot so its reads can borrow page bytes tied to `&self` without
//! holding the shared lock (the hot read path is lock-free). The local copy is
//! refreshed from the shared image at snapshot boundaries via [`adopt_bytes`].
//!
//! ## The fence (the 16-year SQLite WAL bug)
//! A writer that appends using a salt / running-checksum copied *before* a concurrent
//! checkpoint reset the WAL corrupts the log — the exact class of the long-lived
//! SQLite WAL bug. Two mechanisms close it here: a reset only runs while the write
//! lock is free (so it never races a mid-transaction writer), and every append reads
//! its salts / running-checksum from [`SharedInner`] under the lock (never a copied
//! header), re-checking `generation`. See `crate::walstore` for the append path.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError, Weak};
use std::time::{SystemTime, UNIX_EPOCH};

use minisqlite_types::Result;
use minisqlite_wal::codec::{frame_offset, WalHeader, WAL_FILE_FORMAT, WAL_HEADER_SIZE};
use minisqlite_wal::index::{scan, WalIndex};

use crate::walindex::FrameIndex;

/// The authoritative committed WAL state shared by every connection to one file.
/// All fields are `pub(crate)`: the store above manipulates them under the lock
/// together with its own local snapshot and file I/O, so the coordination logic
/// stays in one place (`crate::walstore`) rather than being split across setters.
pub(crate) struct SharedInner {
    /// Database page size in bytes (must match every connection's db header).
    pub(crate) page_size: u32,
    /// The in-memory WAL image (`[32-byte header] ++ frames`), always ending exactly
    /// at a commit frame (uncommitted leftovers are trimmed) so a same-generation
    /// local refresh is a pure append.
    pub(crate) wal_bytes: Vec<u8>,
    /// The current WAL header (salts / checkpoint sequence / byte order). `None`
    /// until the first append initializes it.
    pub(crate) header: Option<WalHeader>,
    /// Index over `wal_bytes` (committed frame ceiling and per-page frame lists),
    /// maintained INCREMENTALLY: extended in place on commit and shared with each
    /// connection's snapshot by cheap `Arc` clone, never re-`scan`ned on a hot path.
    /// See [`crate::walindex::FrameIndex`].
    pub(crate) index: Arc<FrameIndex>,
    /// Bumped on every WAL reset; a connection whose local copy carries a different
    /// generation must re-copy the shared image wholesale (its old frames are gone).
    pub(crate) generation: u64,
    /// Frames already copied into the database file by checkpoints (nBackfill).
    pub(crate) nbackfill: u32,
    /// The single write lock: `true` while one connection holds a write transaction.
    /// A second writer's `begin` sees this set and gets BUSY (it never interleaves).
    pub(crate) write_locked: bool,
    /// Active reader snapshot marks (a multiset). The oldest bounds a checkpoint so
    /// it never drains the WAL past a snapshot a reader still needs.
    pub(crate) readers: Vec<u32>,
    /// The database file's committed page count (the base image). Advanced by a
    /// checkpoint that grows/shrinks the file.
    pub(crate) db_page_count: u32,
    /// Bumped whenever a checkpoint copies frames into the database file. A
    /// connection compares this against its own copy on each snapshot refresh and,
    /// on a change, drops its base-image page cache — otherwise it could serve a
    /// stale database-file page a *different* connection's checkpoint overwrote.
    pub(crate) db_generation: u64,
}

impl SharedInner {
    /// Build the shared state from a database's page size, its file page count, and
    /// the raw `-wal` file bytes (empty if none). A WAL whose header is unusable, or
    /// whose page size disagrees with the database, is ignored (treated as empty); a
    /// mismatched sidecar must not be interpreted against the wrong page size.
    pub(crate) fn new(page_size: u32, db_page_count: u32, mut wal_bytes: Vec<u8>) -> SharedInner {
        let header = WalHeader::decode(&wal_bytes).ok().filter(|h| {
            h.verify_checksum()
                && h.file_format == WAL_FILE_FORMAT
                && h.page_size_is_valid()
                && h.page_size == page_size
        });
        if header.is_none() {
            wal_bytes.clear();
        }
        // Trim uncommitted leftovers (a partial transaction from a crash) so the
        // image ends at a commit frame — the invariant that keeps refresh a pure
        // append and keeps `scan` self-consistent with the bytes.
        let index0 = scan(&wal_bytes);
        let ceiling = committed_ceiling(&index0, page_size, header.is_some());
        wal_bytes.truncate(ceiling);
        // Build the incremental index once from a scan of the trimmed image (cheap
        // here — the WAL is empty or small at open); it is extended in place after.
        let index = FrameIndex::rebuilt(&wal_bytes, page_size);

        SharedInner {
            page_size,
            wal_bytes,
            header,
            index,
            generation: 0,
            nbackfill: 0,
            write_locked: false,
            readers: Vec::new(),
            db_page_count,
            db_generation: 0,
        }
    }
}

/// The byte length of the WAL image up to and including its last commit frame — the
/// committed ceiling. Bytes past this are an uncommitted (interrupted) transaction.
pub(crate) fn committed_ceiling(index: &WalIndex, page_size: u32, header_valid: bool) -> usize {
    let mx = index.mx_frame();
    if mx > 0 {
        // Start of frame (mx + 1) == end of frame mx == WAL_HEADER_SIZE + mx*stride.
        frame_offset(mx + 1, page_size)
    } else if header_valid {
        WAL_HEADER_SIZE
    } else {
        0
    }
}

/// A shared WAL coordinator, handed out as `Arc<SharedWal>` and shared by all
/// connections to one canonical database path.
pub(crate) struct SharedWal {
    inner: Mutex<SharedInner>,
}

impl SharedWal {
    /// Lock the shared state. A poisoned lock (a prior panic while holding it) is
    /// recovered rather than cascading the panic; the WAL state is plain data and the
    /// next operation re-establishes its invariants.
    pub(crate) fn lock(&self) -> MutexGuard<'_, SharedInner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Get (or create) the shared coordinator for `key` (a canonical db path). The
    /// first connection runs `init` to build the state from the file; later
    /// connections to the same path reuse the live `Arc`. Dead entries (all
    /// connections closed) are pruned lazily so the registry does not grow without
    /// bound.
    pub(crate) fn acquire(
        key: PathBuf,
        init: impl FnOnce() -> Result<SharedInner>,
    ) -> Result<Arc<SharedWal>> {
        let mut reg = registry().lock().unwrap_or_else(PoisonError::into_inner);
        reg.retain(|_, weak| weak.strong_count() > 0);
        if let Some(existing) = reg.get(&key).and_then(Weak::upgrade) {
            return Ok(existing);
        }
        let arc = Arc::new(SharedWal { inner: Mutex::new(init()?) });
        reg.insert(key, Arc::downgrade(&arc));
        Ok(arc)
    }
}

/// Process-global registry mapping a canonical database path to its live shared WAL
/// coordinator. `Weak` so a fully-closed database's state is dropped; the entry is
/// pruned on the next `acquire`.
static REGISTRY: OnceLock<Mutex<HashMap<PathBuf, Weak<SharedWal>>>> = OnceLock::new();

fn registry() -> &'static Mutex<HashMap<PathBuf, Weak<SharedWal>>> {
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// A best-effort 32-bit value for a WAL salt. Salts only need to differ across
/// resets (so leftover frames from before a reset fail validation), so a mix of the
/// clock, a process-global counter, and the pid suffices without a crypto RNG
/// dependency. Not used for anything security-sensitive.
pub(crate) fn fresh_u32() -> u32 {
    use std::sync::atomic::{AtomicU32, Ordering};
    static CTR: AtomicU32 = AtomicU32::new(0x9E37_79B9);
    let c = CTR.fetch_add(0x0100_01F3, Ordering::Relaxed);
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.subsec_nanos()).unwrap_or(0);
    nanos.wrapping_mul(2_654_435_761).wrapping_add(c).wrapping_add(std::process::id())
}

/// Refresh a connection's local WAL byte copy to match the shared image.
///
/// When the generation is unchanged the shared WAL only grew by pure appends past
/// the committed ceiling, so the local copy is a prefix of it and we copy just the
/// appended tail (O(new frames)). A generation change means a reset happened (the old
/// frames are gone, salts bumped), so we re-copy wholesale and adopt the new
/// generation. An unexpected shrink at the same generation also forces a wholesale
/// re-copy, so the local copy can never keep stale trailing bytes.
pub(crate) fn adopt_bytes(
    local: &mut Vec<u8>,
    local_gen: &mut u64,
    shared_bytes: &[u8],
    shared_gen: u64,
) {
    if shared_gen == *local_gen && shared_bytes.len() >= local.len() {
        let old = local.len();
        local.extend_from_slice(&shared_bytes[old..]);
    } else {
        local.clear();
        local.extend_from_slice(shared_bytes);
        *local_gen = shared_gen;
    }
}

/// Remove one occurrence of `value` from a reader-mark multiset (order irrelevant).
pub(crate) fn remove_one(marks: &mut Vec<u32>, value: u32) {
    if let Some(pos) = marks.iter().position(|&m| m == value) {
        marks.swap_remove(pos);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adopt_bytes_appends_tail_same_generation() {
        let mut local = vec![1, 2, 3];
        let mut generation = 5;
        adopt_bytes(&mut local, &mut generation, &[1, 2, 3, 4, 5], 5);
        assert_eq!(local, vec![1, 2, 3, 4, 5], "only the tail is appended");
        assert_eq!(generation, 5);
    }

    #[test]
    fn adopt_bytes_recopies_on_generation_change() {
        let mut local = vec![1, 2, 3, 4];
        let mut generation = 5;
        // A reset: shorter, different generation → wholesale re-copy.
        adopt_bytes(&mut local, &mut generation, &[9, 9], 6);
        assert_eq!(local, vec![9, 9]);
        assert_eq!(generation, 6, "adopts the new generation");
    }

    #[test]
    fn adopt_bytes_recopies_on_same_generation_shrink() {
        let mut local = vec![1, 2, 3, 4];
        let mut generation = 5;
        adopt_bytes(&mut local, &mut generation, &[7, 8], 5);
        assert_eq!(local, vec![7, 8], "a shrink forces a wholesale re-copy, no stale tail");
    }

    #[test]
    fn remove_one_removes_a_single_occurrence() {
        let mut marks = vec![3, 5, 3, 7];
        remove_one(&mut marks, 3);
        // Exactly one 3 removed; the multiset still holds the other.
        assert_eq!(marks.iter().filter(|&&m| m == 3).count(), 1);
        assert_eq!(marks.len(), 3);
        remove_one(&mut marks, 42); // absent: no-op
        assert_eq!(marks.len(), 3);
    }

    #[test]
    fn fresh_u32_varies() {
        // Not a randomness test; just guard against a constant (which would make two
        // resets reuse a salt and fail to invalidate stale frames).
        let a = fresh_u32();
        let b = fresh_u32();
        assert_ne!(a, b, "successive salts differ (counter advances)");
    }

    #[test]
    fn shared_inner_ignores_wal_with_mismatched_page_size() {
        // A -wal whose header claims a different page size than the db is ignored.
        let h = WalHeader::new(1024, 1, 2, 0, false);
        let inner = SharedInner::new(512, 3, h.serialize().to_vec());
        assert!(inner.header.is_none(), "mismatched page size => WAL ignored");
        assert!(inner.wal_bytes.is_empty());
        assert_eq!(inner.index.mx_frame(), 0);
    }

    #[test]
    fn shared_inner_trims_uncommitted_leftovers() {
        use minisqlite_wal::codec::WalBuilder;
        let mut b = WalBuilder::new(WalHeader::new(512, 1, 2, 0, false));
        b.append_frame(1, 1, &[0xAA; 512]).unwrap(); // commit at frame 1
        b.append_frame(2, 0, &[0xBB; 512]).unwrap(); // uncommitted leftover
        let bytes = b.into_bytes();
        let full_len = bytes.len();
        let inner = SharedInner::new(512, 1, bytes);
        assert_eq!(inner.index.mx_frame(), 1);
        assert!(inner.wal_bytes.len() < full_len, "leftover frame trimmed");
        // The image ends exactly at the commit frame ceiling.
        assert_eq!(inner.wal_bytes.len(), frame_offset(2, 512));
    }
}
