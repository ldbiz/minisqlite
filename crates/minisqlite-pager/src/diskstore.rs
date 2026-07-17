//! `DiskStore` — the on-disk committed-image backing behind the shared
//! copy-on-write layer ([`Cow`](crate::cow::Cow)). It implements the crate-private
//! [`PageStore`](crate::store::PageStore) seam over a real file, so it reuses ALL of
//! begin/read/write/allocate/free/commit/rollback from `Cow` and only owns two
//! things the in-memory store does not: a durable, journaled commit, and loading
//! pages from the file on demand.
//!
//! ## Page cache (soundness)
//! `read_committed(&self, id) -> &[u8]` must return a borrow tied to `&self` whose
//! address stays valid for the life of the borrow — even though later reads insert
//! more pages into the cache under the same `&self`. The cache is therefore
//! **append-only and stable-address**: [`elsa::FrozenMap`] inserts behind `&self`
//! and returns a reference into the boxed slice's OWN heap allocation, whose address
//! is independent of the map's internal storage, so an earlier borrow survives a
//! later rehash. `apply_commit` takes `&mut self` (exclusive — no read borrow can be
//! outstanding), so it overwrites the bytes of dirty pages in place via
//! [`FrozenMap::as_mut`]. The cache never evicts; it grows with the set of distinct
//! pages actually read (load-on-miss), never an eager whole-file load.
//!
//! ## Durable commit
//! [`DiskStore::apply_commit`] follows the rollback-journal ordering of
//! `spec/sqlite-doc/atomiccommit.html` §3.5–3.11: journal the pre-images and fsync
//! the journal, then write and fsync the database, then invalidate the journal (the
//! commit point). A crash before the commit point leaves a hot journal that
//! [`crate::diskpager::DiskPager::open`] rolls back via `minisqlite_journal::recover`.
//! This store implements DELETE (rollback-journal) mode; WAL mode is a sibling backing
//! ([`crate::walstore::WalStore`]) selected by [`crate::diskpager`] from the header.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use elsa::FrozenMap;
use minisqlite_fileformat::{DatabaseHeader, DEFAULT_PAGE_SIZE, HEADER_SIZE};
use minisqlite_journal::{Journal, JournalMode};
use minisqlite_types::{Error, Result};

use crate::store::{preflight_write_set, PageStore};

/// A disk-backed committed image: a real file plus a load-on-miss, stable-address
/// page cache and the rollback journal wired into commit and recovery.
pub(crate) struct DiskStore {
    /// The database file, opened read+write. All I/O is positioned
    /// ([`FileExt::read_exact_at`] / [`FileExt::write_all_at`], both `&self`), so no
    /// cursor state is shared and reads need no `&mut`.
    file: File,
    /// The `<db>-journal` sibling path, derived once at `open` — the single place the
    /// journal filename is decided, shared by commit and recovery.
    journal_path: PathBuf,
    /// Byte size of every page. Fixed for the life of the store (a store never
    /// changes page size — a differently-sized database is a different file).
    page_size: u32,
    /// Number of pages in the durable image. Advanced only by a successful
    /// `apply_commit`; the source of truth for `read_committed`'s range check.
    committed_count: u32,
    /// The page-1 `file_change_counter` this store's cache + `committed_count`
    /// reflect. A commit by ANOTHER connection bumps the on-disk counter past this;
    /// [`DiskStore::refresh_committed_view`] detects that at a transaction boundary
    /// and drops the stale cache so the reader picks up the other connection's
    /// commit (cross-connection visibility, matching real sqlite's change-counter
    /// cache-invalidation). This connection's OWN commits update it in step, so the
    /// common single-connection case never invalidates.
    change_counter: u32,
    /// How a commit invalidates its journal. DELETE (remove the file) is the default
    /// and the only mode implemented; the field exists so a later
    /// `journal_mode` pragma can select TRUNCATE/PERSIST without reshaping commit.
    mode: JournalMode,
    /// Append-only, stable-address page cache. See the module docs for why this
    /// shape is what makes `read_committed`'s `&self`-tied borrow sound.
    cache: FrozenMap<u32, Box<[u8]>>,
}

impl DiskStore {
    /// Open (creating if absent) the database at `path`, first rolling back any hot
    /// rollback journal so the header and pages are read from a consistent image.
    ///
    /// - New/empty file: page size defaults to [`DEFAULT_PAGE_SIZE`] and the page
    ///   count is 0. A higher layer formats page 1 and the first commit persists it.
    /// - Existing file: the page size comes from the page-1 header; the page count is
    ///   the in-header size when it is valid ([`DatabaseHeader::in_header_size_valid`]),
    ///   otherwise the file length divided by the page size (the documented fallback).
    ///
    /// A file that is non-empty but cannot be parsed as a SQLite header, or whose
    /// length is not a whole number of pages, fails closed with [`Error::Format`]
    /// rather than panicking.
    pub(crate) fn open(path: &Path) -> Result<DiskStore> {
        let journal_path = journal_path_for(path);

        // Recover BEFORE reading the header: a hot journal means a prior process
        // crashed mid-commit, so the on-disk header/pages may be inconsistent until
        // rolled back (atomiccommit §4). `recover` is a no-op when no hot journal
        // exists. Gated on the db existing so an orphan journal for a missing db does
        // not turn recovery into an error.
        if path.exists() {
            minisqlite_journal::recover(path, &journal_path)?;
        }

        // `truncate(false)` is deliberate and load-bearing: opening an existing
        // database must NEVER discard its contents (only `create` a missing one).
        let file =
            OpenOptions::new().read(true).write(true).create(true).truncate(false).open(path)?;
        let file_len = file.metadata()?.len();

        let (page_size, committed_count, change_counter) = if file_len == 0 {
            // A brand-new/empty file has nothing committed yet; the first commit
            // (formatting page 1) sets the counter to 1 and updates our field then.
            (DEFAULT_PAGE_SIZE, 0u32, 0u32)
        } else {
            if file_len < HEADER_SIZE as u64 {
                return Err(Error::Format(format!(
                    "database file is {file_len} bytes, too small to hold a 100-byte header"
                )));
            }
            let mut head = [0u8; HEADER_SIZE];
            file.read_exact_at(&mut head, 0)?;
            // `DatabaseHeader::read` returns `Error::Format` on a bad magic or an
            // out-of-spec page size, which is exactly the fail-closed behaviour we
            // want for a non-SQLite / corrupt file.
            let header = DatabaseHeader::read(&head)?;
            let ps = header.page_size as u64;
            if file_len % ps != 0 {
                return Err(Error::Format(format!(
                    "database file length {file_len} is not a whole multiple of page size {ps}"
                )));
            }
            let count = if header.in_header_size_valid() {
                header.database_size_pages
            } else {
                // Fallback (§1.3.7): the in-header size is only trustworthy when the
                // change counter matches version-valid-for; otherwise the file length
                // is authoritative.
                (file_len / ps) as u32
            };
            (header.page_size, count, header.file_change_counter)
        };

        Ok(DiskStore {
            file,
            journal_path,
            page_size,
            committed_count,
            change_counter,
            mode: JournalMode::Delete,
            cache: FrozenMap::new(),
        })
    }

    /// Maintain the page-1 header for a commit that touches a formatted database:
    /// bump the file change counter, keep the in-header size valid, and record the
    /// new page count. Adds the (patched) page 1 to the write set so it is journaled
    /// and written like any other dirty page.
    ///
    /// The page-1 bytes to patch are the version being committed (`write_set[1]`) if
    /// this transaction already changed page 1 — so freelist edits made by
    /// `alloc.rs` are preserved — otherwise the current committed page 1. Only the
    /// first 100 bytes are rewritten; the b-tree region beyond byte 100 and every
    /// header field other than the change counter / version-valid-for / size are left
    /// exactly as they were. If page 1 is not a formatted SQLite header (e.g. a store
    /// still being formatted, or no page 1 at all) there is nothing to maintain and
    /// the write set is left unchanged.
    fn maintain_page1_header(
        &self,
        write_set: &mut BTreeMap<u32, Box<[u8]>>,
        new_count: u32,
        initial: u32,
    ) -> Result<()> {
        // Only read the committed page 1 when the write set does not already carry it
        // (so freelist edits made by `alloc.rs` are preserved) and a page 1 exists;
        // the shared helper prefers `write_set[1]` when present. This keeps the read
        // pattern identical to the original inline logic.
        let current = if !write_set.contains_key(&1) && initial >= 1 {
            Some(self.read_committed(1)?.to_vec())
        } else {
            None
        };
        crate::page1::maintain_header(write_set, new_count, current.as_deref())
    }

    /// Byte offset in the file of a 1-based page id. Pages are laid out contiguously,
    /// page 1 at offset 0, so page `id` starts at `(id - 1) * page_size`.
    fn offset_of(&self, id: u32) -> u64 {
        (id as u64 - 1) * self.page_size as u64
    }

    /// Re-read the page-1 header from the file to pick up a commit made by ANOTHER
    /// connection to the same database since this store last refreshed. Called at a
    /// transaction boundary (the start of a read OR a write), so a reader sees the
    /// latest committed state and a writer builds on it (never over a stale base).
    ///
    /// When the on-disk `file_change_counter` has advanced past the one this store
    /// reflects, the cached pages and `committed_count` are stale, so the page cache
    /// is dropped and the committed size re-adopted from the header; the next
    /// `read_committed` reloads fresh bytes from the file. When the counter is
    /// unchanged — the common single-connection case, since this connection's own
    /// commits advance the field in step (see `apply_commit`) — it is a cheap
    /// 100-byte read with NO invalidation, so a hot cache survives across statements
    /// (the page cache is not thrown away on every query).
    ///
    /// Taking `&mut self` proves no read borrow is outstanding, so replacing the
    /// cache here is sound (same reasoning as `apply_commit`). A brand-new, too-short,
    /// or unformatted file has nothing committed yet and is treated as "no change" so
    /// formatting the very first page-1 header is unaffected.
    ///
    /// NOTE: this gives cross-connection visibility for connections in ONE process
    /// with sequenced access (the supported two-connection interleaving). It does not add
    /// OS file locking, so it is not a substitute for WAL under genuinely simultaneous
    /// multi-process writers — WAL mode ([`crate::walstore`]) owns that coordination.
    fn refresh_committed_view(&mut self) -> Result<()> {
        let mut head = [0u8; HEADER_SIZE];
        match self.file.read_exact_at(&mut head, 0) {
            Ok(()) => {}
            // Empty/short file: nothing committed yet, nothing to refresh.
            Err(ref e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e.into()),
        }
        // An unformatted page 1 (no valid header yet) has no counter to compare;
        // leave the view untouched rather than treating garbage as a change.
        let header = match DatabaseHeader::read(&head) {
            Ok(h) => h,
            Err(_) => return Ok(()),
        };
        if header.file_change_counter == self.change_counter {
            return Ok(());
        }
        // Another connection committed since our snapshot: drop the stale cache and
        // re-adopt the committed size so the next read reloads current bytes.
        self.cache = FrozenMap::new();
        self.committed_count = if header.in_header_size_valid() {
            header.database_size_pages
        } else {
            let file_len = self.file.metadata()?.len();
            (file_len / self.page_size as u64) as u32
        };
        self.change_counter = header.file_change_counter;
        Ok(())
    }
}

#[cfg(test)]
impl DiskStore {
    /// Test-only: `open`, then override the journal mode. Production always uses
    /// DELETE; a test uses PERSIST so a clean commit LEAVES its journal on disk (only
    /// the magic is zeroed), letting a test drive `apply_commit`'s own journal through
    /// recovery instead of hand-building one.
    pub(crate) fn open_with_mode(path: &Path, mode: JournalMode) -> Result<DiskStore> {
        let mut store = DiskStore::open(path)?;
        store.mode = mode;
        Ok(store)
    }
}

impl PageStore for DiskStore {
    fn page_size(&self) -> u32 {
        self.page_size
    }

    fn committed_count(&self) -> Result<u32> {
        Ok(self.committed_count)
    }

    fn read_committed(&self, id: u32) -> Result<&[u8]> {
        if id == 0 || id > self.committed_count {
            return Err(Error::Io(format!(
                "page {id} is out of range (committed_count = {})",
                self.committed_count
            )));
        }
        if let Some(bytes) = self.cache.get(&id) {
            return Ok(bytes);
        }
        // Load-on-miss: read exactly one page (the full page, including any reserved
        // trailing bytes) and cache it. The returned borrow is tied to `&self` and
        // points into the boxed slice's stable heap allocation.
        let page_size = self.page_size as usize;
        let mut buf = vec![0u8; page_size].into_boxed_slice();
        self.file.read_exact_at(&mut buf, self.offset_of(id))?;
        Ok(self.cache.insert(id, buf))
    }

    fn apply_commit(&mut self, dirty: BTreeMap<u32, Box<[u8]>>, new_count: u32) -> Result<()> {
        // An empty overlay means nothing changed this transaction (the page count can
        // only grow by staging a page, so an empty overlay also implies the count is
        // unchanged). Match SQLite: a no-op commit performs no I/O and does not bump
        // the change counter.
        if dirty.is_empty() {
            return Ok(());
        }

        let page_size = self.page_size as usize;
        let initial = self.committed_count;

        // The write set is the dirty overlay plus, for a formatted database, page 1
        // (its header must record the new size and a bumped change counter).
        let mut write_set = dirty;
        self.maintain_page1_header(&mut write_set, new_count, initial)?;

        // 0. Preflight the ENTIRE write set before touching the journal or the file:
        //    every page must be in-range (1..=new_count) and exactly page_size bytes.
        //    Unlike MemStore's in-memory move, a bad id/length here would `write_all_at`
        //    a mis-sized or misplaced buffer to a PERSISTENT file, so this all-or-nothing
        //    guard is shared across every durable boundary — see [`preflight_write_set`].
        preflight_write_set(&write_set, new_count, page_size)?;

        // 1. Journal the pre-image of every write-set page that already exists on
        //    disk, reading the ORIGINAL committed bytes — nothing has been written
        //    yet, so the cache/file still hold them (page 1's pre-image is the OLD
        //    committed page 1, not the patched buffer). Newly-allocated pages
        //    (id > initial) need no pre-image: recovery truncates them away.
        let mut journal = Journal::create(&self.journal_path, self.page_size, initial)?;
        for &id in write_set.keys() {
            if id <= initial {
                let pre_image = self.read_committed(id)?;
                journal.append_page(id, pre_image)?;
            }
        }
        // Auto_vacuum reclamation can SHRINK the file (`new_count < initial`). The pages
        // being dropped (`new_count < id <= initial`) are not in the write set, but their
        // pre-images MUST be journaled: recovery of an interrupted commit `set_len`s the
        // file back to `initial` (which zero-fills the re-extended tail) and replays the
        // journal, so without these pre-images a crash mid-shrink would resurrect the
        // freed pages as zeros — corrupting the freelist trunks / b-tree pages they held.
        // Journaling them makes rollback restore the pre-commit image byte-for-byte.
        for id in (new_count + 1)..=initial {
            let pre_image = self.read_committed(id)?;
            journal.append_page(id, pre_image)?;
        }
        // 2. Make the journal durable. ONLY after this returns is it safe to modify
        //    the database file (rollback depends on the pre-images being on disk).
        journal.sync()?;

        // 3. Write every write-set page to the database file. Writing the highest
        //    allocated id extends the file to cover any growth; `set_len` afterwards
        //    fixes the exact final length — EXTENDING for a grow, or TRUNCATING for an
        //    auto_vacuum shrink. Truncating is safe here because every dropped page's
        //    pre-image was journaled above, so an interrupted shrink still rolls back.
        for (&id, buf) in write_set.iter() {
            self.file.write_all_at(buf, self.offset_of(id))?;
        }
        let target_len = new_count as u64 * self.page_size as u64;
        if self.file.metadata()?.len() != target_len {
            self.file.set_len(target_len)?;
        }
        // 4. fsync the database so the page writes are durable before the commit point.
        self.file.sync_all()?;

        // 5. COMMIT POINT: invalidate the journal. After this returns the transaction
        //    is durable; a crash before here rolls back via the hot journal on reopen,
        //    which is correct because `commit()` has not yet returned to the caller.
        journal.finish(self.mode)?;

        // 6. Reflect the new committed state in the cache. `&mut self` proves no read
        //    borrow is outstanding, so overwriting a page's bytes here is sound.
        //    Record the change counter this store now reflects (from the maintained
        //    page-1 header) BEFORE the buffers are moved into the cache, so a later
        //    `refresh_committed_view` does not mistake our own commit for another
        //    connection's and needlessly drop the cache we just populated.
        if let Some(p1) = write_set.get(&1) {
            if let Ok(head) = <&[u8; HEADER_SIZE]>::try_from(&p1[..HEADER_SIZE]) {
                if let Ok(header) = DatabaseHeader::read(head) {
                    self.change_counter = header.file_change_counter;
                }
            }
        }
        let cache = self.cache.as_mut();
        for (id, buf) in write_set {
            cache.insert(id, buf);
        }
        self.committed_count = new_count;
        Ok(())
    }

    // A rollback-journal store still refreshes its committed view at every
    // transaction boundary so it observes commits made by OTHER connections to the
    // same file (real sqlite invalidates its page cache when the file change counter
    // moves). `begin_write` refreshes before the COW layer captures the base page
    // count, so a writer never builds on a stale base; `begin_read` refreshes so a
    // reader sees the latest committed state. Both are cheap (a 100-byte header read)
    // and only drop the cache when the counter actually advanced — see
    // [`DiskStore::refresh_committed_view`]. `checkpoint` stays a no-op: a
    // rollback-journal commit already writes the database file directly.
    fn begin_write(&mut self) -> Result<()> {
        self.refresh_committed_view()
    }

    fn begin_read(&mut self) -> Result<()> {
        self.refresh_committed_view()
    }
}

/// The rollback-journal path for a database: the database path with `-journal`
/// appended (SQLite's convention, `spec/sqlite-doc/tempfiles.html`). Appends to the
/// full path (including any extension), so `foo.db` → `foo.db-journal`.
fn journal_path_for(db: &Path) -> PathBuf {
    let mut s = db.as_os_str().to_os_string();
    s.push("-journal");
    PathBuf::from(s)
}
