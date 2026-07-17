//! `PageStore` — the committed-image backing behind the copy-on-write layer.
//!
//! This is the seam that separates *where committed pages live* (RAM today, an
//! on-disk file with a journal/WAL later) from *how transactions work* (the shared
//! [`Cow`](crate::cow::Cow) layer). The COW layer owns begin/write/read-through/
//! commit/rollback/allocate/free ONCE, generically over any `PageStore`, so a
//! `DiskStore` reuses all of it and only implements the four operations below.
//!
//! A store models only the *durable, committed* image. It never sees an in-flight
//! transaction: `apply_commit` is the single mutation point, called with the whole
//! set of pages a committed transaction changed. Reads borrow committed pages by a
//! stable address (see [`PageStore::read_committed`]) so the engine reads a row
//! without copying a page, let alone the database.

use std::collections::BTreeMap;

use minisqlite_types::{Error, Result};

use crate::checkpoint::{CheckpointMode, CheckpointReport};

/// The committed-image backing. Crate-private: it is an internal seam of the pager,
/// not part of the public `Pager` surface. Object safety is unnecessary here (the
/// COW layer is generic over the concrete store), which keeps `read_committed` free
/// to return a borrow tied to `&self`.
pub(crate) trait PageStore {
    /// Byte size of every page (a power of two in `[512, 65536]`), fixed for the
    /// life of the store.
    fn page_size(&self) -> u32;

    /// Number of pages in the durable/base image. This is the page count seen
    /// outside a transaction.
    fn committed_count(&self) -> Result<u32>;

    /// Borrow a committed page's bytes. The returned slice must keep a STABLE
    /// address for the life of the borrow and be tied to `&self`, so the COW layer
    /// can hand it out for reads with no copy and no `unsafe`. A disk-backed store
    /// loads the page on a cache miss and returns a borrow into its cache. An
    /// out-of-range or zero `id` fails closed rather than panicking.
    fn read_committed(&self, id: u32) -> Result<&[u8]>;

    /// Durably apply a committed transaction. `dirty` holds every page whose bytes
    /// changed this transaction, keyed by 1-based id (a `BTreeMap` so pages apply
    /// in ascending id order — deterministic, and sequential for a file backing).
    /// `new_count` is the page count after the transaction: allocations grow it, and
    /// auto_vacuum reclamation ([`crate::reclaim`]) SHRINKS it (the file loses its
    /// trailing pages). A `MemStore` resizes its vector to `new_count` and moves each
    /// buffer into place; `DiskStore` runs its journaled write protocol here (journaling
    /// the dropped pages on a shrink so an interrupted commit rolls back) and `WalStore`
    /// appends WAL frames whose commit marker carries the new size. Every buffer in
    /// `dirty` is exactly
    /// `page_size` bytes and every id is `<= new_count` (guaranteed by the COW layer);
    /// a violation is a programming error and fails closed.
    fn apply_commit(&mut self, dirty: BTreeMap<u32, Box<[u8]>>, new_count: u32) -> Result<()>;

    // ---- Transaction/coordination lifecycle hooks (default no-ops) -------------
    //
    // These let a store that coordinates with other connections (the WAL store)
    // hook the write-lock acquisition, the read-snapshot pin, and the checkpoint
    // into the shared [`Cow`](crate::cow::Cow) transaction flow without the COW
    // layer knowing anything about WAL. The purely-local backings (`MemStore`,
    // rollback `DiskStore`) need none of this, so every hook defaults to a no-op
    // and the existing single-image paths are completely unaffected.

    /// Called at the start of a write transaction ([`Cow::begin`](crate::cow::Cow),
    /// including the implicit auto-commit begin), BEFORE the transaction's base page
    /// count is captured. A coordinating store acquires the single write lock here
    /// (failing with a BUSY error if another connection holds it) and refreshes its
    /// committed snapshot to the latest so the writer builds on the newest state.
    fn begin_write(&mut self) -> Result<()> {
        Ok(())
    }

    /// Acquire the write lock for a transaction that was begun read-only (a DEFERRED
    /// begin's lazy upgrade on the first write). Unlike [`PageStore::begin_write`], it
    /// must NOT move the pinned read snapshot: the writer reads its own snapshot plus
    /// its own uncommitted overlay. A coordinating store returns BUSY if the write lock
    /// is held OR if the pinned snapshot is stale (another connection committed since it
    /// was pinned), because lang_transaction §2.1 forbids upgrading over another
    /// connection's intervening modification. The default forwards to `begin_write`
    /// (a no-op for the purely-local backings, which never take this path).
    fn begin_write_upgrade(&mut self) -> Result<()> {
        self.begin_write()
    }

    /// Called when a write transaction ends (both commit and rollback), AFTER any
    /// `apply_commit`. A coordinating store releases the write lock here. It must be
    /// safe to call once per completed `begin_write`.
    fn end_write(&mut self) -> Result<()> {
        Ok(())
    }

    /// Called at the start of a read transaction. A coordinating store pins its read
    /// snapshot (records the current committed ceiling and registers a reader mark so
    /// a concurrent checkpoint does not drain past it), refreshing the snapshot to the
    /// latest committed state first.
    fn begin_read(&mut self) -> Result<()> {
        Ok(())
    }

    /// Called at the end of a read transaction, releasing the reader mark pinned by
    /// [`PageStore::begin_read`].
    fn end_read(&mut self) -> Result<()> {
        Ok(())
    }

    /// Copy WAL content back into the durable image (a checkpoint) in the requested
    /// [`CheckpointMode`], where the store keeps a WAL. A no-op for a store that
    /// already writes the image directly; the default returns
    /// [`CheckpointReport::not_wal`].
    fn checkpoint(&mut self, _mode: CheckpointMode) -> Result<CheckpointReport> {
        Ok(CheckpointReport::not_wal())
    }
}

/// Validate a committed write set at the one durable boundary, BEFORE any store
/// mutates memory or touches a file: every page id must be in range (`1..=new_count`)
/// and every buffer exactly `page_size` bytes. The COW layer already guarantees both
/// (its `stage_write` hard-checks id range and length), so a violation is a broken
/// invariant, not normal input — but this is the single place committed bytes become
/// durable, so all three `apply_commit` impls (`MemStore`, `DiskStore`, `WalStore`)
/// share this fail-closed, all-or-nothing guard rather than each open-coding it (a
/// missed copy would be a silent-corruption latent defect at the worst place). Fails
/// closed with an `Error::Io`; the messages are the contract, so keep them identical.
pub(crate) fn preflight_write_set(
    write_set: &BTreeMap<u32, Box<[u8]>>,
    new_count: u32,
    page_size: usize,
) -> Result<()> {
    for (&id, buf) in write_set {
        if id == 0 || id > new_count {
            return Err(Error::Io(format!(
                "commit: dirty page {id} is outside the new page count {new_count}"
            )));
        }
        if buf.len() != page_size {
            return Err(Error::Io(format!(
                "commit: dirty page {id} buffer is {} bytes, expected {page_size}",
                buf.len()
            )));
        }
    }
    Ok(())
}

/// A fully-resident committed image: one heap allocation per page, addressed
/// 1-based by page id (index = id - 1). Each page is its own `Box<[u8]>`, so a
/// page keeps a stable address when the outer vector grows — that stability is
/// what lets `read_committed` return `&[u8]` borrowed from `&self` with no
/// `unsafe`.
pub(crate) struct MemStore {
    page_size: u32,
    pages: Vec<Box<[u8]>>,
}

impl MemStore {
    /// A new, empty store (0 pages). `page_size` is validated by the public
    /// `MemPager::new` constructor before it reaches here.
    pub(crate) fn new(page_size: u32) -> MemStore {
        MemStore { page_size, pages: Vec::new() }
    }
}

impl PageStore for MemStore {
    fn page_size(&self) -> u32 {
        self.page_size
    }

    fn committed_count(&self) -> Result<u32> {
        // `apply_commit` refuses to grow past `u32::MAX`, so the length always
        // fits a `u32`.
        Ok(self.pages.len() as u32)
    }

    fn read_committed(&self, id: u32) -> Result<&[u8]> {
        if id == 0 || id as usize > self.pages.len() {
            return Err(Error::Io(format!(
                "page {id} is out of range (page_count = {})",
                self.pages.len()
            )));
        }
        Ok(&self.pages[(id - 1) as usize][..])
    }

    fn apply_commit(&mut self, mut dirty: BTreeMap<u32, Box<[u8]>>, new_count: u32) -> Result<()> {
        let new_len = new_count as usize;
        let page_size = self.page_size as usize;

        // Validate every dirty id AND buffer length BEFORE mutating anything, so a
        // bad commit fails closed and is all-or-nothing (no half-applied store). See
        // [`preflight_write_set`] for why this lives at every durable boundary.
        preflight_write_set(&dirty, new_count, page_size)?;

        let old_len = self.pages.len();
        if new_len < old_len {
            // Shrink. LIVE for an in-memory auto_vacuum database: reclamation commits a
            // `new_count` below the committed count — a FULL auto_vacuum commit or
            // `PRAGMA incremental_vacuum` runs `crate::reclaim`, whose `Cow::truncate_to`
            // lowers the transaction's page count, so this drops the reclaimed trailing
            // pages. Keeps the store's `pages.len() == committed_count` invariant exact;
            // do not remove it as dead code.
            self.pages.truncate(new_len);
        } else if new_len > old_len {
            // Grow by appending ids `old_len+1..=new_len` in order. A grown page the
            // transaction wrote is MOVED straight out of `dirty` (the common case —
            // the COW layer stages a zero page for every id it grows, so a grown id
            // is always present), so it is allocated exactly once here instead of
            // being zero-filled by a resize and then immediately overwritten. Only a
            // grown id the transaction never touched (not currently reachable) costs
            // a fresh zero page.
            self.pages.reserve(new_len - old_len);
            for id in (old_len as u32 + 1)..=(new_len as u32) {
                // Buffers were length-validated up front, so a grown+dirty page is
                // moved in exactly-sized; a true gap gets a fresh zero page.
                match dirty.remove(&id) {
                    Some(buf) => self.pages.push(buf),
                    None => self.pages.push(vec![0u8; page_size].into_boxed_slice()),
                }
            }
        }

        // Apply the remaining dirty writes — overwrites of pages that already
        // existed before this transaction (id <= old_len). Grown ids were consumed
        // by the loop above, so every id here indexes an existing slot.
        for (id, buf) in dirty {
            self.pages[(id - 1) as usize] = buf;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PS: u32 = 512;

    fn boxed(byte: u8) -> Box<[u8]> {
        vec![byte; PS as usize].into_boxed_slice()
    }

    #[test]
    fn empty_store_has_no_pages_and_reads_fail_closed() {
        let s = MemStore::new(PS);
        assert_eq!(s.committed_count().unwrap(), 0);
        assert!(s.read_committed(1).is_err());
        assert!(s.read_committed(0).is_err());
        assert_eq!(s.page_size(), PS);
    }

    #[test]
    fn apply_commit_grows_and_writes_dirty_pages() {
        let mut s = MemStore::new(PS);
        let mut dirty = BTreeMap::new();
        dirty.insert(1u32, boxed(0xAA));
        dirty.insert(3u32, boxed(0xCC));
        // new_count 3 but only pages 1 and 3 dirty: page 2 must appear zero-filled.
        s.apply_commit(dirty, 3).unwrap();
        assert_eq!(s.committed_count().unwrap(), 3);
        assert_eq!(s.read_committed(1).unwrap(), &boxed(0xAA)[..]);
        assert_eq!(s.read_committed(2).unwrap(), &vec![0u8; PS as usize][..]);
        assert_eq!(s.read_committed(3).unwrap(), &boxed(0xCC)[..]);
    }

    #[test]
    fn read_returns_stable_addresses_across_growth() {
        // The load-bearing invariant: a page's bytes do not move when the store
        // grows, so a borrow handed out earlier stays valid. We check addresses
        // directly rather than trusting the type system alone.
        let mut s = MemStore::new(PS);
        let mut d1 = BTreeMap::new();
        d1.insert(1u32, boxed(0x11));
        s.apply_commit(d1, 1).unwrap();
        let addr_before = s.read_committed(1).unwrap().as_ptr();
        let mut d2 = BTreeMap::new();
        d2.insert(2u32, boxed(0x22));
        s.apply_commit(d2, 2).unwrap();
        let addr_after = s.read_committed(1).unwrap().as_ptr();
        assert_eq!(addr_before, addr_after, "page 1 must not move when the store grows");
    }

    #[test]
    fn apply_commit_rejects_dirty_id_past_new_count() {
        let mut s = MemStore::new(PS);
        let mut dirty = BTreeMap::new();
        dirty.insert(5u32, boxed(0x55));
        assert!(s.apply_commit(dirty, 2).is_err());
    }

    #[test]
    fn apply_commit_over_existing_base_grows_and_overwrites_in_place() {
        // Pins the grow-over-a-non-empty-base path: an existing page is overwritten
        // in place, an untouched base page is preserved, a grown+dirty page is
        // filled from `dirty`, and a grown id absent from `dirty` comes back
        // zero-filled (the gap branch).
        let mut s = MemStore::new(PS);
        let mut base = BTreeMap::new();
        base.insert(1u32, boxed(0x11));
        base.insert(2u32, boxed(0x22));
        s.apply_commit(base, 2).unwrap();

        let mut d2 = BTreeMap::new();
        d2.insert(1u32, boxed(0xEE)); // overwrite an existing page in place
        d2.insert(3u32, boxed(0x33)); // grown AND dirty -> moved in from `dirty`
        s.apply_commit(d2, 4).unwrap(); // page 4 grown but absent -> zero gap

        assert_eq!(s.committed_count().unwrap(), 4);
        assert_eq!(s.read_committed(1).unwrap(), &boxed(0xEE)[..], "existing page overwritten");
        assert_eq!(s.read_committed(2).unwrap(), &boxed(0x22)[..], "untouched base page preserved");
        assert_eq!(s.read_committed(3).unwrap(), &boxed(0x33)[..], "grown+dirty page filled");
        assert_eq!(
            s.read_committed(4).unwrap(),
            &vec![0u8; PS as usize][..],
            "a grown id absent from dirty comes back zero-filled"
        );
    }

    #[test]
    fn apply_commit_rejects_bad_id_before_mutating() {
        // The validate-before-mutate guarantee: a commit carrying an out-of-range id
        // leaves the store completely unchanged (all-or-nothing), not half-grown.
        let mut s = MemStore::new(PS);
        let mut base = BTreeMap::new();
        base.insert(1u32, boxed(0x11));
        s.apply_commit(base, 1).unwrap();

        let mut bad = BTreeMap::new();
        bad.insert(2u32, boxed(0x22)); // would be a valid grow...
        bad.insert(9u32, boxed(0x99)); // ...but this id is past new_count -> reject all
        assert!(s.apply_commit(bad, 3).is_err());
        assert_eq!(s.committed_count().unwrap(), 1, "a rejected commit does not grow the store");
        assert_eq!(s.read_committed(1).unwrap(), &boxed(0x11)[..], "base is untouched after reject");
    }

    #[test]
    fn apply_commit_rejects_wrong_length_buffer_fail_closed() {
        // The trait contract says a buffer whose length != page_size "fails closed".
        // The COW layer guarantees exact-size buffers, so this is defense-in-depth at
        // the durable mutation point — but it must be a hard error (not a debug-only
        // check that silently stores a wrong-length page in release), and atomic.
        let mut s = MemStore::new(PS);
        let mut base = BTreeMap::new();
        base.insert(1u32, boxed(0x11));
        s.apply_commit(base, 1).unwrap();

        // A too-short buffer for a grown id is rejected before any mutation.
        let mut short_grow = BTreeMap::new();
        short_grow.insert(2u32, vec![0x22u8; (PS - 1) as usize].into_boxed_slice());
        assert!(s.apply_commit(short_grow, 2).is_err(), "short grown buffer rejected");
        assert_eq!(s.committed_count().unwrap(), 1, "rejected commit did not grow");

        // A too-long buffer overwriting an existing id is likewise rejected, atomically.
        let mut long_overwrite = BTreeMap::new();
        long_overwrite.insert(1u32, vec![0xAAu8; (PS + 1) as usize].into_boxed_slice());
        assert!(s.apply_commit(long_overwrite, 1).is_err(), "long overwrite buffer rejected");
        assert_eq!(s.read_committed(1).unwrap(), &boxed(0x11)[..], "page 1 untouched after reject");
    }

    #[test]
    fn apply_commit_shrinks_via_truncate() {
        // The shrink branch is LIVE: an in-memory auto_vacuum database reaches it when
        // reclamation commits a `new_count` below the committed count (a FULL auto_vacuum
        // commit or `PRAGMA incremental_vacuum` -> `reclaim` -> `Cow::truncate_to`). This
        // pins its unit contract directly — a `new_count` below the committed count
        // truncates, keeping `pages.len() == committed_count` exact.
        let mut s = MemStore::new(PS);
        let mut base = BTreeMap::new();
        base.insert(1u32, boxed(0x11));
        base.insert(2u32, boxed(0x22));
        base.insert(3u32, boxed(0x33));
        s.apply_commit(base, 3).unwrap();

        // Shrink to 2 pages while overwriting the surviving page 1.
        let mut shrink = BTreeMap::new();
        shrink.insert(1u32, boxed(0xEE));
        s.apply_commit(shrink, 2).unwrap();

        assert_eq!(s.committed_count().unwrap(), 2, "store truncated to new_count");
        assert_eq!(s.read_committed(1).unwrap(), &boxed(0xEE)[..], "surviving page overwritten");
        assert_eq!(s.read_committed(2).unwrap(), &boxed(0x22)[..], "surviving page preserved");
        assert!(s.read_committed(3).is_err(), "the truncated page is gone");
    }
}
