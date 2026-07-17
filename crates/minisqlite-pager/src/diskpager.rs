//! `DiskPager` — the disk-backed implementation of the [`Pager`](crate::Pager) seam.
//!
//! Like [`MemPager`](crate::MemPager) it is a thin wrapper over the shared
//! copy-on-write layer ([`Cow`]); the difference is the committed-image backing.
//! `DiskPager` selects ONE of two backings from the page-1 header at open time and
//! delegates every `Pager` method to whichever it built — there is still exactly one
//! transaction layer (`Cow`), never a forked pager:
//!
//! - [`DiskStore`] — the rollback-journal backing (write/read version 1, the default
//!   for a new or legacy file).
//! - [`WalStore`](crate::walstore::WalStore) — the write-ahead-log backing (write and
//!   read version both 2), which appends commits to a `<db>-wal` file and checkpoints
//!   them back into the database.
//!
//! The mode is fixed for the life of an open handle: a live rollback→WAL switch
//! within one handle is not supported; `PRAGMA journal_mode=WAL` writes the version-2
//! header (a rollback-mode commit) and the next `open` picks up WAL mode.

use std::fs::OpenOptions;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use minisqlite_fileformat::{DatabaseHeader, HEADER_SIZE};
use minisqlite_types::Result;

use crate::checkpoint::{CheckpointMode, CheckpointReport};
use crate::cow::Cow;
use crate::diskstore::DiskStore;
use crate::walstore::WalStore;
use crate::{PageId, Pager, VacuumOutcome};

/// A durable, disk-backed pager. The concrete backing (rollback journal or WAL) is
/// chosen from the page-1 header at open time; both share the one `Cow` transaction
/// layer.
pub struct DiskPager(DiskInner);

/// The two disk backings behind `DiskPager`, each the same `Cow` transaction layer
/// over a different committed store. An enum (not `Box<dyn>`) so the hot read/write
/// path stays a static dispatch with no per-call vtable indirection.
enum DiskInner {
    Rollback(Cow<DiskStore>),
    Wal(Cow<WalStore>),
}

impl DiskPager {
    /// Open (creating if absent) the official-format database at `path`. A hot
    /// rollback journal is recovered first, then the mode is chosen from the page-1
    /// header: write and read version both 2 selects WAL, anything else (including a
    /// new/empty file, which defaults to version 1) selects the rollback journal. A
    /// non-SQLite or corrupt file fails closed with [`minisqlite_types::Error::Format`].
    pub fn open(path: &Path) -> Result<DiskPager> {
        // A hot rollback journal is ALWAYS rollback-mode state (WAL mode uses `-wal`,
        // never `-journal`). Recover it before reading the header so a crashed
        // `journal_mode=WAL` transition — whose version-2 header is on disk but whose
        // commit never finished — is rolled back to version 1 rather than routing to
        // the WAL path on a half-flipped header. Recovery is a no-op when no hot
        // journal exists, and `DiskStore::open` runs it again idempotently.
        if path.exists() {
            minisqlite_journal::recover(path, &journal_path_for(path))?;
        }
        if is_wal_mode(path)? {
            Ok(DiskPager(DiskInner::Wal(Cow::new(WalStore::open(path)?))))
        } else {
            Ok(DiskPager(DiskInner::Rollback(Cow::new(DiskStore::open(path)?))))
        }
    }

    /// Test-only: like [`open`](Self::open) but forcing the rollback backing with an
    /// explicit journal mode (see [`DiskStore::open_with_mode`]). Production always
    /// uses DELETE; a test uses PERSIST so a clean commit LEAVES its journal on disk,
    /// letting a test drive `apply_commit`'s own commit-time journaling through
    /// recovery.
    #[cfg(test)]
    pub(crate) fn open_with_mode(
        path: &Path,
        mode: minisqlite_journal::JournalMode,
    ) -> Result<DiskPager> {
        Ok(DiskPager(DiskInner::Rollback(Cow::new(DiskStore::open_with_mode(path, mode)?))))
    }

    /// Test-only: shrink the active transaction's page count to `new_count` — the
    /// transactional shrink primitive [`Cow::truncate_to`] that auto_vacuum reclamation
    /// drives ([`crate::reclaim`]). It is not part of the `Pager` seam because production
    /// only ever reaches a shrink through reclaim; exposing it here lets a crash-recovery
    /// test drive a genuine SHRINKING commit (new count < committed count) straight
    /// through the SAME `apply_commit` path production uses, without hand-building a whole
    /// valid auto_vacuum database (ptrmap + freelist + forest) just to reach it via
    /// `incremental_vacuum`. Same test-only pattern as [`open_with_mode`](Self::open_with_mode).
    #[cfg(test)]
    pub(crate) fn truncate_to(&mut self, new_count: PageId) -> Result<()> {
        match &mut self.0 {
            DiskInner::Rollback(c) => c.truncate_to(new_count),
            DiskInner::Wal(c) => c.truncate_to(new_count),
        }
    }
}

/// True if the database at `path` is in WAL mode: its page-1 header has both the
/// write and read file-format version set to 2 (fileformat2 §1.3). A missing, empty,
/// too-short, or non-SQLite file is not WAL mode (the rollback path then opens it,
/// creating or failing closed as it does today). The version bytes are set once at
/// database creation and are stable across a rollback commit, so reading them here is
/// safe even before the rollback path's own header parse.
fn is_wal_mode(path: &Path) -> Result<bool> {
    let file = match OpenOptions::new().read(true).open(path) {
        Ok(f) => f,
        Err(ref e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(e) => return Err(e.into()),
    };
    if file.metadata()?.len() < HEADER_SIZE as u64 {
        return Ok(false);
    }
    let mut head = [0u8; HEADER_SIZE];
    file.read_exact_at(&mut head, 0)?;
    // A header that does not parse is not WAL mode; the rollback path opens the file
    // and fails closed with a Format error exactly as before (no behavior change for
    // a corrupt file).
    match DatabaseHeader::read(&head) {
        Ok(h) => Ok(h.is_wal()),
        Err(_) => Ok(false),
    }
}

/// The `-journal` sibling path (SQLite's convention). Mirrors
/// `diskstore::journal_path_for` — a filesystem naming rule, duplicated here only so
/// `open` can recover before deciding the mode without reaching into the rollback
/// store's private helper.
fn journal_path_for(db: &Path) -> PathBuf {
    let mut s = db.as_os_str().to_os_string();
    s.push("-journal");
    PathBuf::from(s)
}

impl Pager for DiskPager {
    fn read_page(&self, id: PageId) -> Result<&[u8]> {
        match &self.0 {
            DiskInner::Rollback(c) => c.read_page(id),
            DiskInner::Wal(c) => c.read_page(id),
        }
    }

    fn page_count(&self) -> Result<PageId> {
        match &self.0 {
            DiskInner::Rollback(c) => c.page_count(),
            DiskInner::Wal(c) => c.page_count(),
        }
    }

    fn page_size(&self) -> u32 {
        match &self.0 {
            DiskInner::Rollback(c) => c.page_size(),
            DiskInner::Wal(c) => c.page_size(),
        }
    }

    fn begin(&mut self) -> Result<()> {
        match &mut self.0 {
            DiskInner::Rollback(c) => c.begin(),
            DiskInner::Wal(c) => c.begin(),
        }
    }

    fn begin_deferred(&mut self) -> Result<()> {
        match &mut self.0 {
            // Rollback journal: the eager begin already defers nothing that matters
            // for concurrency (single-image, no shared write lock), so DEFERRED == the
            // eager begin — ZERO behavior change on the non-WAL path.
            DiskInner::Rollback(c) => c.begin(),
            // WAL: pin the read snapshot and defer the single write lock to the first
            // write (acquire-or-BUSY), the two-connection concurrency path.
            DiskInner::Wal(c) => c.begin_deferred(),
        }
    }

    fn write_page(&mut self, id: PageId, bytes: &[u8]) -> Result<()> {
        match &mut self.0 {
            DiskInner::Rollback(c) => c.write_page(id, bytes),
            DiskInner::Wal(c) => c.write_page(id, bytes),
        }
    }

    fn page_mut(&mut self, id: PageId) -> Result<&mut [u8]> {
        match &mut self.0 {
            DiskInner::Rollback(c) => c.page_mut(id),
            DiskInner::Wal(c) => c.page_mut(id),
        }
    }

    fn allocate_page(&mut self) -> Result<PageId> {
        match &mut self.0 {
            DiskInner::Rollback(c) => c.allocate_page(),
            DiskInner::Wal(c) => c.allocate_page(),
        }
    }

    fn free_page(&mut self, id: PageId) -> Result<()> {
        match &mut self.0 {
            DiskInner::Rollback(c) => c.free_page(id),
            DiskInner::Wal(c) => c.free_page(id),
        }
    }

    fn commit(&mut self) -> Result<()> {
        match &mut self.0 {
            DiskInner::Rollback(c) => c.commit(),
            DiskInner::Wal(c) => c.commit(),
        }
    }

    fn rollback(&mut self) -> Result<()> {
        match &mut self.0 {
            DiskInner::Rollback(c) => c.rollback(),
            DiskInner::Wal(c) => c.rollback(),
        }
    }

    fn savepoint(&mut self) -> Result<usize> {
        match &mut self.0 {
            DiskInner::Rollback(c) => c.savepoint(),
            DiskInner::Wal(c) => c.savepoint(),
        }
    }

    fn release_savepoint(&mut self, depth: usize) -> Result<()> {
        match &mut self.0 {
            DiskInner::Rollback(c) => c.release_savepoint(depth),
            DiskInner::Wal(c) => c.release_savepoint(depth),
        }
    }

    fn rollback_to_savepoint(&mut self, depth: usize) -> Result<()> {
        match &mut self.0 {
            DiskInner::Rollback(c) => c.rollback_to_savepoint(depth),
            DiskInner::Wal(c) => c.rollback_to_savepoint(depth),
        }
    }

    fn begin_read(&mut self) -> Result<()> {
        match &mut self.0 {
            DiskInner::Rollback(c) => c.begin_read(),
            DiskInner::Wal(c) => c.begin_read(),
        }
    }

    fn end_read(&mut self) -> Result<()> {
        match &mut self.0 {
            DiskInner::Rollback(c) => c.end_read(),
            DiskInner::Wal(c) => c.end_read(),
        }
    }

    fn checkpoint(&mut self, mode: CheckpointMode) -> Result<CheckpointReport> {
        match &mut self.0 {
            // Rollback backing: no WAL, so the Cow forwards to DiskStore's default
            // (a not-WAL report — the pragma reports (0, -1, -1)).
            DiskInner::Rollback(c) => c.checkpoint(mode),
            DiskInner::Wal(c) => c.checkpoint(mode),
        }
    }

    fn incremental_vacuum(&mut self, max: Option<PageId>) -> Result<VacuumOutcome> {
        match &mut self.0 {
            DiskInner::Rollback(c) => c.incremental_vacuum(max),
            DiskInner::Wal(c) => c.incremental_vacuum(max),
        }
    }

    fn take_root_moved(&mut self) -> bool {
        match &mut self.0 {
            DiskInner::Rollback(c) => c.take_root_moved(),
            DiskInner::Wal(c) => c.take_root_moved(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs::{File, OpenOptions};
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use minisqlite_fileformat::{DatabaseHeader, HEADER_SIZE};
    use minisqlite_journal::{Journal, JournalMode, MAGIC};
    use minisqlite_types::Error;

    use crate::{DiskPager, Pager};

    /// A database plus its `-journal` sibling under the OS temp dir, both removed on
    /// drop. Unique per test so parallel runs never collide, and nothing is left
    /// behind (mirrors the journal crate's `Case` helper).
    struct Case {
        db: PathBuf,
        journal: PathBuf,
    }

    impl Case {
        fn new(tag: &str) -> Case {
            static N: AtomicU32 = AtomicU32::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            let pid = std::process::id();
            let nanos =
                SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
            let mut db = std::env::temp_dir();
            db.push(format!("mspd-{tag}-{pid}-{n}-{nanos}.db"));
            let journal = {
                let mut s = db.as_os_str().to_os_string();
                s.push("-journal");
                PathBuf::from(s)
            };
            Case { db, journal }
        }
    }

    impl Drop for Case {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.db);
            let _ = std::fs::remove_file(&self.journal);
        }
    }

    /// One page filled with a repeating byte.
    fn marker(page_size: u32, byte: u8) -> Vec<u8> {
        vec![byte; page_size as usize]
    }

    fn read_file(path: &Path) -> Vec<u8> {
        let mut v = Vec::new();
        File::open(path).unwrap().read_to_end(&mut v).unwrap();
        v
    }

    /// Write pages concatenated as a database file (byte-exact crafting).
    fn write_db(path: &Path, pages: &[Vec<u8>]) {
        let mut f = File::create(path).unwrap();
        for p in pages {
            f.write_all(p).unwrap();
        }
        f.sync_all().unwrap();
    }

    /// Overwrite (or extend) a database file at an absolute offset, without
    /// truncating the rest — used to scribble garbage over pages / grow the file
    /// when simulating an aborted transaction by hand.
    fn write_at(path: &Path, offset: u64, bytes: &[u8]) {
        let mut f = OpenOptions::new().write(true).open(path).unwrap();
        f.seek(SeekFrom::Start(offset)).unwrap();
        f.write_all(bytes).unwrap();
        f.sync_all().unwrap();
    }

    /// A page-1 buffer holding a valid SQLite header for `page_size`, with the rest
    /// of the page filled with `tail` so tests can check bytes past the header
    /// survive a commit.
    fn header_page(page_size: u32, tail: u8) -> Vec<u8> {
        let hdr = DatabaseHeader { page_size, ..DatabaseHeader::default() };
        let mut page1 = vec![tail; page_size as usize];
        page1[..HEADER_SIZE].copy_from_slice(&hdr.to_bytes());
        page1
    }

    /// Format page 1 (matching the pager's page size) and allocate `extra` more
    /// pages, committing pages `1..=extra+1`. Stands in for the btree's
    /// `init_database`, which we cannot call from this crate.
    fn format_and_fill(p: &mut DiskPager, extra: u32) {
        let page_size = p.page_size();
        p.begin().unwrap();
        let one = p.allocate_page().unwrap();
        assert_eq!(one, 1);
        p.write_page(1, &header_page(page_size, 0)).unwrap();
        for _ in 0..extra {
            p.allocate_page().unwrap();
        }
        p.commit().unwrap();
    }

    fn page1_header(p: &DiskPager) -> DatabaseHeader {
        let page1 = p.read_page(1).unwrap();
        let mut buf = [0u8; HEADER_SIZE];
        buf.copy_from_slice(&page1[..HEADER_SIZE]);
        DatabaseHeader::read(&buf).unwrap()
    }

    #[test]
    fn persistence_roundtrip_survives_reopen() {
        let case = Case::new("persist");
        let ps;
        {
            let mut p = DiskPager::open(&case.db).unwrap();
            ps = p.page_size();
            p.begin().unwrap();
            p.allocate_page().unwrap(); // page 1
            // A real header plus recognizable marker bytes AFTER the 100-byte header.
            p.write_page(1, &header_page(ps, 0x91)).unwrap();
            let two = p.allocate_page().unwrap();
            let three = p.allocate_page().unwrap();
            p.write_page(two, &marker(ps, 0x22)).unwrap();
            p.write_page(three, &marker(ps, 0x33)).unwrap();
            assert_eq!(p.page_count().unwrap(), 3);
            p.commit().unwrap();
        }

        // Reopen from disk: page count, page-1 tail bytes, and every page survive.
        let p = DiskPager::open(&case.db).unwrap();
        assert_eq!(p.page_size(), ps);
        assert_eq!(p.page_count().unwrap(), 3);
        let page1 = p.read_page(1).unwrap();
        assert_eq!(
            &page1[HEADER_SIZE..],
            &vec![0x91u8; ps as usize - HEADER_SIZE][..],
            "b-tree region past the header is preserved across commit + reopen"
        );
        assert_eq!(p.read_page(2).unwrap(), &marker(ps, 0x22)[..]);
        assert_eq!(p.read_page(3).unwrap(), &marker(ps, 0x33)[..]);
        assert_eq!(page1_header(&p).database_size_pages, 3);
    }

    #[test]
    fn page_mut_edit_survives_commit_and_reopen() {
        // In-place edits through page_mut are as durable as write_page: they land in
        // the dirty overlay and commit writes them to disk.
        let case = Case::new("pagemut-persist");
        let ps;
        {
            let mut p = DiskPager::open(&case.db).unwrap();
            ps = p.page_size();
            p.begin().unwrap();
            p.allocate_page().unwrap();
            p.write_page(1, &header_page(ps, 0x00)).unwrap();
            let two = p.allocate_page().unwrap();
            p.write_page(two, &marker(ps, 0x00)).unwrap();
            p.commit().unwrap();

            // A second transaction edits page 1's b-tree region and page 2 IN PLACE.
            p.begin().unwrap();
            for b in &mut p.page_mut(1).unwrap()[HEADER_SIZE..] {
                *b = 0x91;
            }
            {
                let two_buf = p.page_mut(2).unwrap();
                two_buf[0] = 0x22;
                two_buf[ps as usize - 1] = 0x2F;
            }
            p.commit().unwrap();
        }

        let p = DiskPager::open(&case.db).unwrap();
        assert_eq!(p.page_size(), ps);
        let page1 = p.read_page(1).unwrap();
        assert_eq!(
            &page1[HEADER_SIZE..],
            &vec![0x91u8; ps as usize - HEADER_SIZE][..],
            "page_mut edit past the header is durable across reopen"
        );
        assert_eq!(p.read_page(2).unwrap()[0], 0x22, "page_mut edit at cell region durable");
        assert_eq!(p.read_page(2).unwrap()[ps as usize - 1], 0x2F, "page_mut edit at page tail durable");
    }

    #[test]
    fn page_mut_commit_journals_pre_image_for_recovery() {
        // Durability equivalence: a transaction whose ONLY writes are page_mut edits
        // journals the pre-images at commit exactly like write_page, so crash recovery
        // rolls it back to the pre-commit image. Mirrors
        // `apply_commit_journal_drives_recovery_to_pre_commit_image` but drives every
        // edit through page_mut.
        let case = Case::new("pagemut-commitjournal");
        let ps;
        let committed;
        {
            let mut p = DiskPager::open(&case.db).unwrap();
            ps = p.page_size();
            format_and_fill(&mut p, 2); // pages 1..=3
            p.begin().unwrap();
            p.write_page(2, &marker(ps, 0x22)).unwrap();
            p.write_page(3, &marker(ps, 0x33)).unwrap();
            p.commit().unwrap();
            committed = read_file(&case.db);
        }
        assert!(!case.journal.exists());

        // PERSIST transaction that edits pages 2 and 3 ONLY through page_mut.
        {
            let mut p = DiskPager::open_with_mode(&case.db, JournalMode::Persist).unwrap();
            p.begin().unwrap();
            for b in p.page_mut(2).unwrap().iter_mut() {
                *b = 0x99;
            }
            for b in p.page_mut(3).unwrap().iter_mut() {
                *b = 0xEE;
            }
            p.commit().unwrap();
        }
        let after_commit = read_file(&case.db);
        assert_ne!(after_commit, committed, "the page_mut commit changed the database on disk");
        assert_eq!(
            &after_commit[ps as usize..2 * ps as usize],
            &marker(ps, 0x99)[..],
            "page 2 on disk holds the page_mut bytes after commit"
        );
        assert!(case.journal.exists(), "PERSIST leaves apply_commit's journal in place");

        // Restore the magic PERSIST zeroed → the journal is hot → recovery runs.
        write_at(&case.journal, 0, &MAGIC);
        {
            let p = DiskPager::open(&case.db).unwrap();
            assert_eq!(p.page_count().unwrap(), 3, "recovered to the pre-commit page count");
            assert_eq!(
                p.read_page(2).unwrap(),
                &marker(ps, 0x22)[..],
                "page 2 rolled back to the pre-image the page_mut commit journaled"
            );
            assert_eq!(
                p.read_page(3).unwrap(),
                &marker(ps, 0x33)[..],
                "page 3 rolled back to the pre-image the page_mut commit journaled"
            );
        }
        assert_eq!(
            read_file(&case.db),
            committed,
            "page_mut commit's own journal rolls the file back to the exact pre-commit image"
        );
        assert!(!case.journal.exists(), "journal removed after recovery");
    }

    #[test]
    fn header_maintenance_bumps_counter_and_keeps_size_valid() {
        let case = Case::new("hdrmaint");
        let mut p = DiskPager::open(&case.db).unwrap();
        let ps = p.page_size();
        format_and_fill(&mut p, 2); // commit #1 → pages 1..=3

        // commit #2 grows the database by one page without touching page 1 directly;
        // header maintenance must still update off28 and bump the change counter.
        p.begin().unwrap();
        let four = p.allocate_page().unwrap();
        p.write_page(four, &marker(ps, 0x44)).unwrap();
        p.commit().unwrap();
        drop(p);

        let p = DiskPager::open(&case.db).unwrap();
        let hdr = page1_header(&p);
        assert_eq!(hdr.database_size_pages, p.page_count().unwrap(), "off28 tracks the page count");
        assert_eq!(hdr.database_size_pages, 4);
        assert!(hdr.in_header_size_valid(), "in-header size stays valid after a write");
        assert_eq!(hdr.file_change_counter, hdr.version_valid_for, "counter == version_valid_for");
        assert_eq!(hdr.file_change_counter, 2, "two committing transactions bump the counter twice");
    }

    #[test]
    fn reads_a_crafted_existing_file_byte_for_byte() {
        let case = Case::new("craft");
        let ps = 512u32;
        // A valid header whose in-header size is trustworthy (default counters are
        // equal), plus two content pages.
        let header = DatabaseHeader { page_size: ps, database_size_pages: 3, ..DatabaseHeader::default() };
        let mut page1 = vec![0u8; ps as usize];
        page1[..HEADER_SIZE].copy_from_slice(&header.to_bytes());
        let page2 = marker(ps, 0x22);
        let page3 = marker(ps, 0x33);
        write_db(&case.db, &[page1.clone(), page2.clone(), page3.clone()]);

        let p = DiskPager::open(&case.db).unwrap();
        assert_eq!(p.page_size(), ps, "page size comes from the header, not the default");
        assert_eq!(p.page_count().unwrap(), 3, "trusts the valid in-header size");
        assert_eq!(p.read_page(1).unwrap(), &page1[..]);
        assert_eq!(p.read_page(2).unwrap(), &page2[..]);
        assert_eq!(p.read_page(3).unwrap(), &page3[..]);
    }

    #[test]
    fn falls_back_to_file_length_when_in_header_size_invalid() {
        let case = Case::new("fallback");
        let ps = 512u32;
        // change counter != version_valid_for ⇒ in-header size is NOT trustworthy;
        // database_size_pages is bogus and must be ignored in favour of file length.
        let header = DatabaseHeader {
            page_size: ps,
            file_change_counter: 7,
            version_valid_for: 4,
            database_size_pages: 99,
            ..DatabaseHeader::default()
        };
        let mut page1 = vec![0u8; ps as usize];
        page1[..HEADER_SIZE].copy_from_slice(&header.to_bytes());
        // Four whole pages on disk.
        write_db(&case.db, &[page1, marker(ps, 0xAB), marker(ps, 0xCD), marker(ps, 0xEF)]);

        let p = DiskPager::open(&case.db).unwrap();
        assert_eq!(p.page_size(), ps, "page size still comes from the header on the fallback path");
        assert_eq!(p.page_count().unwrap(), 4, "count falls back to file_len / page_size");
    }

    #[test]
    fn reserved_space_reads_full_pages() {
        let case = Case::new("reserved");
        let ps = 512u32;
        // reserved_space 32 ⇒ usable 480 (the minimum); read() accepts it.
        let header = DatabaseHeader {
            page_size: ps,
            reserved_space: 32,
            database_size_pages: 2,
            ..DatabaseHeader::default()
        };
        let mut page1 = vec![0u8; ps as usize];
        page1[..HEADER_SIZE].copy_from_slice(&header.to_bytes());
        let page2 = marker(ps, 0x5A);
        write_db(&case.db, &[page1, page2.clone()]);

        let p = DiskPager::open(&case.db).unwrap();
        assert_eq!(p.page_size(), 512);
        assert_eq!(p.page_count().unwrap(), 2);
        let read2 = p.read_page(2).unwrap();
        assert_eq!(read2.len(), 512, "a read returns the FULL page, reserved bytes included");
        assert_eq!(read2, &page2[..]);
    }

    #[test]
    fn rollback_writes_nothing_to_disk_and_leaves_no_journal() {
        let case = Case::new("rollback");
        let mut p = DiskPager::open(&case.db).unwrap();
        let ps = p.page_size();
        format_and_fill(&mut p, 2); // pages 1..=3 committed
        let before = read_file(&case.db);

        p.begin().unwrap();
        p.write_page(2, &marker(ps, 0xEE)).unwrap();
        let four = p.allocate_page().unwrap();
        p.write_page(four, &marker(ps, 0xDD)).unwrap();
        assert_eq!(p.page_count().unwrap(), 4, "inside the txn the count reflects the allocation");
        p.rollback().unwrap();

        assert_eq!(p.page_count().unwrap(), 3, "rollback forgets the grown page count");
        assert!(p.read_page(4).is_err(), "the allocated page is gone");
        assert_eq!(read_file(&case.db), before, "rollback never touched the database file");
        assert!(!case.journal.exists(), "a rolled-back transaction leaves no journal");
    }

    #[test]
    fn clean_commit_leaves_no_journal() {
        let case = Case::new("nojournal");
        let mut p = DiskPager::open(&case.db).unwrap();
        let ps = p.page_size();
        format_and_fill(&mut p, 3);
        assert!(!case.journal.exists(), "DELETE-mode commit removes the journal");
        p.begin().unwrap();
        p.write_page(2, &marker(ps, 0xAB)).unwrap();
        p.commit().unwrap();
        assert!(!case.journal.exists(), "still no journal after a second commit");
    }

    #[test]
    fn crash_recovery_rolls_back_an_aborted_commit() {
        let case = Case::new("recover");
        let ps;
        let committed;
        {
            let mut p = DiskPager::open(&case.db).unwrap();
            ps = p.page_size();
            format_and_fill(&mut p, 2); // pages 1..=3
            p.begin().unwrap();
            p.write_page(2, &marker(ps, 0x22)).unwrap();
            p.write_page(3, &marker(ps, 0x33)).unwrap();
            p.commit().unwrap();
            committed = read_file(&case.db);
        }
        assert_eq!(committed.len(), 3 * ps as usize);
        assert!(!case.journal.exists());

        // Simulate a crash mid-commit BY HAND: journal the pre-images of pages 1..=3,
        // sync, then scribble garbage over them and grow the file to 5 pages — but
        // NEVER finish the journal, so it stays hot (a crash before the commit point).
        {
            let mut j = Journal::create(&case.journal, ps, 3).unwrap();
            for id in 1..=3u32 {
                let start = (id as usize - 1) * ps as usize;
                j.append_page(id, &committed[start..start + ps as usize]).unwrap();
            }
            j.sync().unwrap();
            drop(j); // hot journal survives — the "crash"
            for id in 1..=5u32 {
                write_at(&case.db, (id as u64 - 1) * ps as u64, &marker(ps, 0xFF));
            }
        }
        assert_eq!(read_file(&case.db).len(), 5 * ps as usize, "aborted txn grew the file");
        assert!(case.journal.exists(), "hot journal present before recovery");

        // Reopen: recovery rolls the database back to the committed image and
        // truncates it to the pre-crash page count.
        {
            let p = DiskPager::open(&case.db).unwrap();
            assert_eq!(p.page_count().unwrap(), 3, "recovered to the pre-crash page count");
            assert_eq!(p.read_page(2).unwrap(), &marker(ps, 0x22)[..], "page 2 rolled back");
            assert_eq!(p.read_page(3).unwrap(), &marker(ps, 0x33)[..], "page 3 rolled back");
        }
        let after = read_file(&case.db);
        assert_eq!(after, committed, "database rolled back to the committed image");
        assert_eq!(after.len(), 3 * ps as usize, "database truncated to the pre-crash page count");
        assert!(!case.journal.exists(), "journal removed after recovery");
    }

    #[test]
    fn freelist_persists_across_reopen() {
        let case = Case::new("freelist");
        let mut p = DiskPager::open(&case.db).unwrap();
        format_and_fill(&mut p, 4); // pages 1..=5
        assert_eq!(p.page_count().unwrap(), 5);

        // Free page 3 and commit; the freelist head lives in the page-1 header.
        p.begin().unwrap();
        p.free_page(3).unwrap();
        p.commit().unwrap();
        assert_eq!(page1_header(&p).first_freelist_trunk, 3, "freed page recorded in the header");
        drop(p);

        // Reopen: the freelist survived on disk, so the next allocation REUSES page 3
        // (before growing the file).
        let mut p = DiskPager::open(&case.db).unwrap();
        assert_eq!(p.page_count().unwrap(), 5);
        let reused = p.allocate_page().unwrap();
        assert_eq!(reused, 3, "the persisted freelist is reused after reopen");
        assert_eq!(p.page_count().unwrap(), 5, "reuse did not grow the file");
        assert_eq!(page1_header(&p).first_freelist_trunk, 0, "freelist is empty again");
    }

    #[test]
    fn corrupt_header_fails_closed() {
        let case = Case::new("corrupt");
        // A non-empty file whose first bytes are not the SQLite magic is not a
        // database; opening it must fail closed as a format error, never panic.
        write_db(&case.db, &[marker(512, 0x00)]);
        let opened = DiskPager::open(&case.db);
        assert!(matches!(opened, Err(Error::Format(_))), "non-SQLite file must fail closed");
    }

    #[test]
    fn reads_out_of_range_fail_closed() {
        let case = Case::new("range");
        let mut p = DiskPager::open(&case.db).unwrap();
        format_and_fill(&mut p, 1); // pages 1..=2
        assert!(p.read_page(0).is_err(), "page 0 is never valid (1-based)");
        assert!(p.read_page(3).is_err(), "a page past the count is out of range");
        assert!(p.read_page(1).is_ok());
        assert!(p.read_page(2).is_ok());
    }

    #[test]
    fn cached_page_keeps_a_stable_address_across_later_reads() {
        // Pins the load-bearing soundness invariant: a borrow handed out by
        // `read_page` must stay valid as later reads insert more pages into the same
        // append-only cache. We compare the raw address directly.
        let case = Case::new("stable");
        let mut p = DiskPager::open(&case.db).unwrap();
        format_and_fill(&mut p, 4); // pages 1..=5
        drop(p);

        let p = DiskPager::open(&case.db).unwrap(); // fresh, empty cache
        let addr_before = p.read_page(2).unwrap().as_ptr();
        for id in [3u32, 4, 5, 1] {
            let _ = p.read_page(id).unwrap(); // each inserts into the cache
        }
        let addr_after = p.read_page(2).unwrap().as_ptr();
        assert_eq!(addr_before, addr_after, "a cached page must not move when the cache grows");
    }

    #[test]
    fn empty_commit_is_a_noop() {
        // The `dirty.is_empty()` short-circuit in apply_commit: a transaction that
        // stages no write must create no journal, write nothing, and NOT bump the
        // change counter (matching SQLite's no-op commit). Without the early return,
        // header maintenance would re-read page 1 and bump the counter on every
        // begin();commit() — this test would then go red.
        let case = Case::new("emptycommit");
        let mut p = DiskPager::open(&case.db).unwrap();
        format_and_fill(&mut p, 2); // pages 1..=3, counter → 1
        let before = read_file(&case.db);
        let counter_before = page1_header(&p).file_change_counter;
        assert_eq!(counter_before, 1, "sanity: exactly one real commit happened in setup");

        p.begin().unwrap();
        p.commit().unwrap();
        assert!(!case.journal.exists(), "an empty commit creates no journal");
        assert_eq!(read_file(&case.db), before, "an empty commit writes nothing to disk");
        drop(p);

        let p = DiskPager::open(&case.db).unwrap();
        assert_eq!(
            page1_header(&p).file_change_counter,
            counter_before,
            "an empty commit does not bump the change counter"
        );
    }

    #[test]
    fn overwritten_page_is_visible_same_instance_after_commit() {
        // Pins step 6 (the post-commit cache refresh) for a NON-header page directly:
        // after committing an overwrite, the SAME open instance must read the NEW
        // bytes, not a stale cached pre-image. Also checked after reopen for good
        // measure. (freelist_persists_across_reopen only covers this via page 1.)
        let case = Case::new("sameinstance");
        let mut p = DiskPager::open(&case.db).unwrap();
        let ps = p.page_size();
        format_and_fill(&mut p, 2); // pages 1..=3
        assert_eq!(p.read_page(2).unwrap(), &marker(ps, 0)[..], "page 2 starts zero-filled");

        p.begin().unwrap();
        p.write_page(2, &marker(ps, 0x7E)).unwrap();
        p.commit().unwrap();
        assert_eq!(
            p.read_page(2).unwrap(),
            &marker(ps, 0x7E)[..],
            "same instance sees the committed bytes, not a stale cache entry"
        );
        drop(p);

        let p = DiskPager::open(&case.db).unwrap();
        assert_eq!(
            p.read_page(2).unwrap(),
            &marker(ps, 0x7E)[..],
            "reopened instance sees the committed bytes"
        );
    }

    #[test]
    fn apply_commit_journal_drives_recovery_to_pre_commit_image() {
        // Directly regression-tests apply_commit's OWN commit-time journaling — the
        // PRE-image bytes and the initial page count it writes — which the hand-built
        // crash test does not exercise (that one fabricates the journal itself). A
        // PERSIST-mode commit leaves apply_commit's journal on disk with only the
        // 8-byte magic zeroed; restoring the magic makes it hot again, so a fresh
        // open()->recover() rolls the database back USING the journal apply_commit
        // produced. If apply_commit journaled the NEW bytes instead of the pre-images,
        // or passed the wrong initial page count to Journal::create, the rollback
        // would not reproduce the pre-commit image and the asserts below fail.
        let case = Case::new("commitjournal");
        let ps;
        let committed;
        {
            let mut p = DiskPager::open(&case.db).unwrap();
            ps = p.page_size();
            format_and_fill(&mut p, 2); // pages 1..=3 (counter → 1)
            p.begin().unwrap();
            p.write_page(2, &marker(ps, 0x22)).unwrap();
            p.write_page(3, &marker(ps, 0x33)).unwrap();
            p.commit().unwrap(); // counter → 2
            committed = read_file(&case.db);
        }
        assert!(!case.journal.exists(), "DELETE-mode setup left no journal");

        // A PERSIST-mode transaction overwrites pages 2 and 3 (and, via header
        // maintenance, page 1). On PERSIST, apply_commit leaves its journal in place.
        {
            let mut p = DiskPager::open_with_mode(&case.db, JournalMode::Persist).unwrap();
            p.begin().unwrap();
            p.write_page(2, &marker(ps, 0x99)).unwrap();
            p.write_page(3, &marker(ps, 0xEE)).unwrap();
            p.commit().unwrap(); // counter → 3, journal persists (magic zeroed)
        }
        // The commit really changed the file, so the rollback below is a real undo,
        // not a vacuous no-op.
        let after_commit = read_file(&case.db);
        assert_ne!(after_commit, committed, "the PERSIST commit changed the database on disk");
        assert_eq!(
            &after_commit[ps as usize..2 * ps as usize],
            &marker(ps, 0x99)[..],
            "page 2 on disk holds the new bytes after the commit"
        );
        assert!(case.journal.exists(), "PERSIST leaves the journal file in place");

        // Restore the 8 magic bytes PERSIST zeroed → apply_commit's journal is hot.
        write_at(&case.journal, 0, &MAGIC);

        // A fresh open runs recovery over apply_commit's OWN journal.
        {
            let p = DiskPager::open(&case.db).unwrap();
            assert_eq!(p.page_count().unwrap(), 3, "recovered to the pre-commit page count");
            assert_eq!(
                p.read_page(2).unwrap(),
                &marker(ps, 0x22)[..],
                "page 2 rolled back to the pre-image apply_commit journaled"
            );
            assert_eq!(
                p.read_page(3).unwrap(),
                &marker(ps, 0x33)[..],
                "page 3 rolled back to the pre-image apply_commit journaled"
            );
        }
        assert_eq!(
            read_file(&case.db),
            committed,
            "apply_commit's own journal rolls the file back to the exact pre-commit image"
        );
        assert!(!case.journal.exists(), "journal removed after recovery");
    }

    /// Build a formatted database of `pages` pages with a DISTINCT marker per page, so a
    /// dropped/restored page is verifiable byte-exact (not merely "some bytes came back").
    /// Page 1 carries a real header (tail 0x11); page `id` (2..=pages) is filled with
    /// `marker(0x10 + id)`. Commits in whatever journal mode `p` was opened with.
    ///
    /// Precondition: `pages <= 0xEF`. The per-page marker `0x10 + id as u8` only stays
    /// distinct and non-overflowing while `id <= 0xEF`; beyond that the `u8` add would
    /// panic in debug (or wrap into a marker collision), silently defeating the "distinct"
    /// guarantee. Both current callers pass <= 8; the assert makes the bound explicit for
    /// any future caller rather than leaving it a latent trap.
    fn fill_distinct_pages(p: &mut DiskPager, ps: u32, pages: u32) {
        debug_assert!(pages <= 0xEF, "fill_distinct_pages: `pages` must be <= 0xEF for distinct markers");
        p.begin().unwrap();
        assert_eq!(p.allocate_page().unwrap(), 1);
        p.write_page(1, &header_page(ps, 0x11)).unwrap();
        for id in 2..=pages {
            assert_eq!(p.allocate_page().unwrap(), id, "sequential allocation with an empty freelist");
            p.write_page(id, &marker(ps, 0x10 + id as u8)).unwrap();
        }
        p.commit().unwrap();
        assert_eq!(p.page_count().unwrap(), pages);
    }

    #[test]
    fn shrink_commit_journal_drives_recovery_to_old_larger_image() {
        // The auto_vacuum SHRINK path's crash safety: apply_commit's OWN journal must
        // carry the pre-images of the DROPPED tail pages (`new_count+1 ..= initial`) —
        // pages NOT in the write set — so a crash BEFORE the commit point rolls the file
        // back to the OLD, LARGER image byte-for-byte instead of leaving it truncated
        // (with the freed pages resurrected as zeros on the recovery `set_len`).
        //
        // The two existing crash-recovery tests cover a GROWING abort
        // (`crash_recovery_rolls_back_an_aborted_commit`) and a NON-shrinking commit
        // (`apply_commit_journal_drives_recovery_to_pre_commit_image`); NEITHER combines
        // apply_commit's own journaling WITH a real shrink, which is the branch pinned
        // here. The shrink is driven through `Cow::truncate_to` (the exact primitive
        // reclaim uses) so it exercises the SAME `apply_commit` path production takes,
        // and the PERSIST 'leave the journal hot' trick replays apply_commit's OWN
        // journal on reopen (a hand-built journal could not prove apply_commit journaled
        // the dropped tail pages).
        let case = Case::new("shrinkrecover");
        let ps;
        let committed;
        let initial;
        {
            let mut p = DiskPager::open(&case.db).unwrap();
            ps = p.page_size();
            fill_distinct_pages(&mut p, ps, 8); // pages 1..=8, markers 0x11 / 0x12..0x18
            committed = read_file(&case.db);
            initial = p.page_count().unwrap();
        }
        assert_eq!(initial, 8);
        assert_eq!(committed.len(), 8 * ps as usize);
        assert!(!case.journal.exists(), "DELETE-mode setup left no journal");

        // A PERSIST-mode transaction shrinks 8 -> 4. It also overwrites an in-range page
        // (page 2) because a PURE truncate stages nothing and apply_commit no-ops an
        // empty overlay (so no shrink would happen) — and this also proves a MODIFIED
        // surviving page is rolled back. new_count = 4 drops pages 5..=8, whose
        // pre-images only the shrink branch journals.
        let new_count = 4u32;
        {
            let mut p = DiskPager::open_with_mode(&case.db, JournalMode::Persist).unwrap();
            p.begin().unwrap();
            p.write_page(2, &marker(ps, 0xE2)).unwrap();
            p.truncate_to(new_count).unwrap();
            assert_eq!(p.page_count().unwrap(), new_count, "the txn count reflects the shrink");
            p.commit().unwrap();
        }
        // The commit REALLY shrank the file, so the rollback below undoes a genuine
        // truncation rather than a vacuous no-op.
        let after_shrink = read_file(&case.db);
        assert_eq!(
            after_shrink.len(),
            new_count as usize * ps as usize,
            "the shrink commit truncated the file to the smaller page count"
        );
        assert!(after_shrink.len() < committed.len(), "the file genuinely shrank on commit");
        assert!(case.journal.exists(), "PERSIST leaves apply_commit's shrink journal in place");

        // Restore the 8 magic bytes PERSIST zeroed → apply_commit's OWN journal is hot.
        write_at(&case.journal, 0, &MAGIC);

        // A fresh open runs recovery over apply_commit's own shrink journal.
        {
            let p = DiskPager::open(&case.db).unwrap();
            assert_eq!(
                p.page_count().unwrap(),
                initial,
                "recovery restored the OLD, LARGER page count (the file re-grew to 8)"
            );
            // Every DROPPED tail page comes back BYTE-EXACT from the journaled pre-image;
            // without the shrink branch's journaling these would be zeros (or gone).
            for id in (new_count + 1)..=initial {
                let start = (id as usize - 1) * ps as usize;
                assert_eq!(
                    p.read_page(id).unwrap(),
                    &committed[start..start + ps as usize],
                    "dropped tail page {id} rolled back to its pre-shrink committed image"
                );
            }
            // The modified in-range page 2 is rolled back to its pre-image too.
            assert_eq!(
                p.read_page(2).unwrap(),
                &marker(ps, 0x12)[..],
                "the modified surviving page 2 rolled back to its pre-image"
            );
        }
        assert_eq!(
            read_file(&case.db),
            committed,
            "the shrink commit's own journal rolls the WHOLE file back to the exact pre-commit image"
        );
        assert!(!case.journal.exists(), "journal removed after recovery");
    }

    #[test]
    fn clean_shrink_commit_persists_the_smaller_image() {
        // The committed shrink OUTCOME: a shrink commit that FINISHES (DELETE mode, the
        // normal path) leaves the file at EXACTLY `new_count` pages on reopen — never a
        // torn mix of the old and new sizes, and never a resurrected tail. Paired with
        // `shrink_commit_journal_drives_recovery_to_old_larger_image` (crash BEFORE the
        // commit point → old-larger) this pins BOTH shrink crash outcomes: old-larger on
        // abort, new-smaller on success.
        let case = Case::new("shrinkclean");
        let ps;
        {
            let mut p = DiskPager::open(&case.db).unwrap();
            ps = p.page_size();
            fill_distinct_pages(&mut p, ps, 6); // pages 1..=6

            // Shrink 6 -> 3 and COMMIT cleanly (DELETE mode removes the journal). Writing
            // page 2 keeps the overlay non-empty (a pure truncate is a no-op commit) and
            // proves the surviving edit is durable across the shrink.
            p.begin().unwrap();
            p.write_page(2, &marker(ps, 0xE2)).unwrap();
            p.truncate_to(3).unwrap();
            p.commit().unwrap();
            assert!(!case.journal.exists(), "a clean DELETE-mode shrink removes the journal");
        }

        // Reopen: the committed state is the SMALLER image, exact to the byte length.
        let p = DiskPager::open(&case.db).unwrap();
        assert_eq!(p.page_count().unwrap(), 3, "reopen sees the committed smaller page count");
        assert_eq!(
            read_file(&case.db).len(),
            3 * ps as usize,
            "the file is exactly new_count pages, never a torn old/new mix"
        );
        assert_eq!(page1_header(&p).database_size_pages, 3, "the in-header size records the shrink");
        assert_eq!(p.read_page(2).unwrap(), &marker(ps, 0xE2)[..], "the surviving edit is durable");
        assert!(p.read_page(4).is_err(), "a dropped page is out of range after the shrink");
    }

    #[test]
    fn full_auto_vacuum_compaction_is_atomic_and_crash_safe() {
        // The atomic-in-commit FULL auto_vacuum contract, exercised through the pager's OWN
        // commit + `apply_commit` (NOT a fabricated journal): a commit that frees pages in a
        // FULL auto_vacuum database COMPACTS the file inside that SAME durable transaction, so
        // a committed FULL database always has `freelist_count == 0`, and a crash at the
        // durable-write boundary recovers to EITHER the exact pre-commit image (rolled back)
        // OR the exact compacted post-commit image — NEVER a committed-but-un-compacted FULL
        // db (`freelist_count > 0` after a committed data change), a state real sqlite never
        // presents.
        //
        // This pins the compacting commit's WRITE-side journaling directly — the gap the
        // facade's fabricated-journal test (`conformance_auto_vacuum_reclaim.rs`) explicitly
        // cannot reach: the shrink here is produced by `av_commit::finalize_and_compact`
        // reclaiming a DROP-leaked page (not a hand-built `truncate_to`), and the PERSIST
        // 'leave the journal hot' trick replays that very commit's own journal on reopen.
        //
        // RED against the OLD two-phase design: there `Cow::commit` ran finalize ONLY (it
        // swept the freed page onto the freelist but did NOT truncate), so a single pager
        // commit left 3 pages with `freelist_count == 1`; the compacted-length and
        // freelist_count assertions below would both fail.
        use minisqlite_fileformat::{encode_record, encode_table_leaf_cell, PageBuilder, PageType};
        use minisqlite_types::Value;

        // A page-1 `sqlite_schema` leaf carrying a FULL auto_vacuum header (off52 = largest
        // root != 0, off64 = 0). `Some(rp)` names one table `t` rooted at `rp`; `None` is an
        // EMPTY schema (the "DROP" that leaks the table's page). finalize derives the ptrmap
        // and re-sets off52, so an empty schema still keeps off52 = 1 (stays auto_vacuum).
        fn av_schema_page1(root: Option<i64>, ps: u32) -> Vec<u8> {
            let mut b = PageBuilder::new(ps as usize, ps as usize, 1, PageType::LeafTable);
            if let Some(rp) = root {
                let rec = encode_record(&[
                    Value::Text("table".into()),
                    Value::Text("t".into()),
                    Value::Text("t".into()),
                    Value::Integer(rp),
                    Value::Text("CREATE TABLE t(a)".into()),
                ]);
                let mut cell = Vec::new();
                encode_table_leaf_cell(rec.len() as u64, 1, &rec, None, &mut cell);
                assert!(b.add_cell(&cell), "schema row fits on page 1");
            }
            let mut page1 = b.finish();
            let hdr = DatabaseHeader {
                page_size: ps,
                largest_root_btree: root.map(|r| r as u32).unwrap_or(1),
                incremental_vacuum: 0,
                ..DatabaseHeader::default()
            };
            let head: &mut [u8; HEADER_SIZE] = (&mut page1[..HEADER_SIZE]).try_into().unwrap();
            hdr.write(head);
            page1
        }

        let empty_leaf =
            |page_no: u32, ps: u32| PageBuilder::new(ps as usize, ps as usize, page_no, PageType::LeafTable).finish();

        let case = Case::new("avatomic");
        let ps;
        let committed; // the pre-commit image C: a 3-page FULL av db holding table t.

        // Build C through the pager (DELETE mode leaves no journal): page 1 names t@3, page 2
        // is the ptrmap finalize derives, page 3 is t's (empty) leaf. No free pages, so this
        // commit does NOT compact — C is the larger, pre-DROP baseline.
        {
            let mut p = DiskPager::open(&case.db).unwrap();
            ps = p.page_size();
            p.begin().unwrap();
            assert_eq!(p.allocate_page().unwrap(), 1);
            p.write_page(1, &av_schema_page1(Some(3), ps)).unwrap();
            // In a FULL av db the allocator reserves page 2 for the ptrmap and hands back
            // page 3 — matching the rootpage named in the schema above.
            assert_eq!(p.allocate_page().unwrap(), 3, "av alloc skips the ptrmap page 2");
            p.write_page(3, &empty_leaf(3, ps)).unwrap();
            p.commit().unwrap();
            assert_eq!(p.page_count().unwrap(), 3, "3-page FULL av db; no compaction (no free pages)");
            assert_eq!(page1_header(&p).freelist_count, 0, "insert-only FULL: freelist empty");
            assert!(page1_header(&p).largest_root_btree != 0, "auto_vacuum header set (off52 != 0)");
        }
        committed = read_file(&case.db);
        assert_eq!(committed.len(), 3 * ps as usize, "C is exactly 3 pages");
        assert!(!case.journal.exists(), "clean DELETE-mode setup left no journal");

        // The compacting commit (PERSIST so apply_commit's own journal stays on disk):
        // rewrite page 1 to an EMPTY schema. finalize sweeps the now-unreferenced page 3 onto
        // the freelist, then `finalize_and_compact` reclaims it and truncates — all in ONE
        // durable transaction.
        {
            let mut p = DiskPager::open_with_mode(&case.db, JournalMode::Persist).unwrap();
            p.begin().unwrap();
            p.write_page(1, &av_schema_page1(None, ps)).unwrap();
            p.commit().unwrap();
            assert_eq!(p.page_count().unwrap(), 1, "atomic FULL commit compacted 3 -> 1 IN ONE commit");
        }

        // The committed file is the COMPACTED image: 1 page, freelist_count == 0, still av.
        // Read the raw bytes (no intermediate open, which would touch the hot-able journal).
        // These two assertions are what go RED against the old two-phase behavior (which left
        // 3 pages with freelist_count == 1 after a single commit).
        let after_compact = read_file(&case.db);
        assert_eq!(
            after_compact.len(),
            ps as usize,
            "a committed FULL commit is COMPACTED to 1 page in the same transaction (old two-phase left 3)"
        );
        let compacted_freelist = u32::from_be_bytes(after_compact[36..40].try_into().unwrap());
        assert_eq!(compacted_freelist, 0, "a committed FULL db has freelist_count == 0 (old two-phase left 1)");
        let compacted_off52 = u32::from_be_bytes(after_compact[52..56].try_into().unwrap());
        assert!(compacted_off52 != 0, "the compacted file is still an auto_vacuum db");
        assert!(case.journal.exists(), "PERSIST leaves the compacting commit's own journal in place");

        // Crash injection: restore the 8 magic bytes PERSIST zeroed so apply_commit's OWN
        // journal is HOT, then reopen. Recovery replays it and rolls the file all the way back
        // to the exact pre-commit image C (the 'rolled back' crash outcome).
        write_at(&case.journal, 0, &MAGIC);
        {
            let p = DiskPager::open(&case.db).unwrap();
            assert_eq!(
                p.page_count().unwrap(),
                3,
                "recovery restored the OLD, larger pre-commit image (the file re-grew to 3)"
            );
            // Table t's leaf (page 3) comes back BYTE-EXACT from the journaled pre-image —
            // without the compacting commit journaling the dropped tail it would be gone.
            let start = 2 * ps as usize;
            assert_eq!(
                p.read_page(3).unwrap(),
                &committed[start..start + ps as usize],
                "the dropped table leaf rolled back to its pre-commit bytes"
            );
            // The recovered committed FULL db is consistent: still auto_vacuum with
            // freelist_count == 0 — i.e. NOT a committed-but-un-compacted FULL db.
            let h = page1_header(&p);
            assert!(h.largest_root_btree != 0, "recovered file is still an auto_vacuum db");
            assert_eq!(h.freelist_count, 0, "the recovered pre-commit FULL db has freelist_count == 0");
        }
        assert_eq!(
            read_file(&case.db),
            committed,
            "recovery restored the EXACT pre-commit image, never a torn or un-compacted mix"
        );
        assert!(!case.journal.exists(), "the hot journal is removed once recovery completes");
    }
}
