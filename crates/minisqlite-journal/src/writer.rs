//! The [`Journal`] a pager writes during a commit. It records each page's original
//! content *before* the database is modified, so a crash mid-commit can be rolled
//! back from the journal on the next open (see [`crate::recover`]).
//!
//! Commit ordering the pager must follow for durability (atomiccommit §3):
//!
//! 1. [`Journal::create`] — write the header.
//! 2. [`Journal::append_page`] for every page about to change (its pre-image).
//! 3. [`Journal::sync`] — make the journal durable. **Only after this returns is it
//!    safe to modify the database file**, because rollback depends on the pre-images
//!    already being on disk.
//! 4. (pager writes and fsyncs the database file)
//! 5. [`Journal::commit_delete`] / [`Journal::commit_truncate`] /
//!    [`Journal::commit_persist`] — invalidate the journal. This is the instant the
//!    transaction commits: with the journal gone (or its header cleared) there is
//!    nothing to roll back.

use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use minisqlite_types::{Error, Result};

use crate::codec::{
    is_valid_page_size, is_valid_sector_size, journal_checksum, page_record_len, JournalHeader,
    DEFAULT_SECTOR_SIZE, OFF_PAGE_COUNT,
};
use crate::util::{fsync_parent_dir, remove_file_durable, write_be32};

/// How a transaction invalidates its journal to commit — the three settings of the
/// `journal_mode` pragma this crate implements (fileformat2 §3, atomiccommit §3.11).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JournalMode {
    /// Delete the journal file. The default; the file's absence signals "committed".
    Delete,
    /// Truncate the journal to zero length. An empty file is not a valid journal.
    Truncate,
    /// Overwrite the header magic with zeros. The file persists but is no longer a
    /// valid (hot) journal, so recovery skips it.
    Persist,
}

/// A rollback journal open for writing. Holds the file handle plus the fixed header
/// fields (nonce, page size, sector size, initial database size) and the running
/// record count backfilled into the header on [`sync`](Journal::sync).
#[derive(Debug)]
pub struct Journal {
    file: File,
    path: PathBuf,
    page_size: u32,
    sector_size: u32,
    nonce: u32,
    initial_db_pages: u32,
    record_count: u32,
    /// Reused per-record scratch buffer, sized once to a full record, so appending a
    /// page does not allocate on the hot commit path.
    record_buf: Vec<u8>,
}

impl Journal {
    /// Create a fresh journal at `journal_path` for a database with `page_size`-byte
    /// pages that currently holds `initial_db_pages` pages. Uses the default sector
    /// size and a fresh random nonce, writing the zero-padded header immediately.
    pub fn create(
        journal_path: impl AsRef<Path>,
        page_size: u32,
        initial_db_pages: u32,
    ) -> Result<Journal> {
        Self::create_with_sector(journal_path, page_size, initial_db_pages, DEFAULT_SECTOR_SIZE)
    }

    /// Like [`create`](Journal::create) but with an explicit sector size (the header
    /// is padded to it). A caller that knows the underlying device's sector size can
    /// pass it here so the header occupies its own physical sector (matching real
    /// SQLite); [`create`](Journal::create) uses [`DEFAULT_SECTOR_SIZE`].
    pub fn create_with_sector(
        journal_path: impl AsRef<Path>,
        page_size: u32,
        initial_db_pages: u32,
        sector_size: u32,
    ) -> Result<Journal> {
        if !is_valid_page_size(page_size) {
            return Err(Error::format(format!("journal create: invalid page size {page_size}")));
        }
        if !is_valid_sector_size(sector_size) {
            return Err(Error::format(format!(
                "journal create: invalid sector size {sector_size}"
            )));
        }
        let path = journal_path.as_ref().to_path_buf();
        // A journal is created fresh each transaction; truncate any stale file so a
        // leftover longer body cannot masquerade as extra records under the new nonce.
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;
        let nonce = fresh_nonce();
        // Page count starts at 0 and is backfilled with the true count on `sync`.
        let header = JournalHeader {
            page_count: 0,
            nonce,
            initial_db_pages,
            sector_size,
            page_size,
        };
        file.write_all(&header.to_padded_header())?;
        Ok(Journal {
            file,
            path,
            page_size,
            sector_size,
            nonce,
            initial_db_pages,
            record_count: 0,
            // Sized once to exactly one record; `page_size` is fixed for the journal's
            // life, so `append_page` overwrites this in place without re-allocating or
            // re-zeroing on the hot commit path.
            record_buf: vec![0u8; page_record_len(page_size as usize)],
        })
    }

    /// Append one page record: the original `original_bytes` of database page
    /// `page_no`, plus its checksum. `original_bytes` must be exactly one page and
    /// `page_no` must be 1-based; both are validated (fail closed) since a wrong size
    /// or a page 0 would corrupt the journal.
    pub fn append_page(&mut self, page_no: u32, original_bytes: &[u8]) -> Result<()> {
        if page_no == 0 {
            return Err(Error::format("journal append: page number must be >= 1".to_string()));
        }
        if original_bytes.len() != self.page_size as usize {
            return Err(Error::format(format!(
                "journal append: page {page_no} is {} bytes, expected page size {}",
                original_bytes.len(),
                self.page_size
            )));
        }
        let n = self.page_size as usize;
        // `record_buf` was sized to exactly one record in `create`; the three writes
        // below cover every byte (page number, full page content, checksum), so there
        // is nothing stale to clear and no per-record zero-fill to pay for.
        debug_assert_eq!(self.record_buf.len(), page_record_len(n));
        write_be32(&mut self.record_buf, 0, page_no);
        self.record_buf[4..4 + n].copy_from_slice(original_bytes);
        let cksum = journal_checksum(self.nonce, original_bytes);
        write_be32(&mut self.record_buf, 4 + n, cksum);
        self.file.write_all(&self.record_buf)?;
        self.record_count = self.record_count.checked_add(1).ok_or_else(|| {
            Error::format("journal append: too many page records (u32 overflow)".to_string())
        })?;
        Ok(())
    }

    /// Backfill the true record count into the header and flush everything to disk.
    ///
    /// After this returns the journal is durable and the database may be modified.
    /// A single `fsync` is sufficient rather than the classic write-records-then-
    /// write-count double sync: recovery stops at the first record whose checksum
    /// fails, and because the nonce is fresh per transaction, any record the header
    /// claims but that never reached disk (stale/zeroed bytes) cannot reproduce this
    /// nonce's checksum — so an over-large count is self-correcting on replay.
    pub fn sync(&mut self) -> Result<()> {
        // Rewrite the page-count field in place, then restore the cursor to the end
        // so a later append (not expected in the normal protocol, but harmless)
        // continues after the last record rather than overwriting the header.
        self.file.seek(SeekFrom::Start(OFF_PAGE_COUNT as u64))?;
        self.file.write_all(&self.record_count.to_be_bytes())?;
        self.file.seek(SeekFrom::End(0))?;
        self.file.sync_all()?;
        // The file's own fsync above does NOT make the journal's directory entry
        // durable. Since this is the single barrier before the pager modifies the
        // database, the journal's *name* must also be on stable storage here:
        // otherwise a fresh journal (created this transaction) could vanish across a
        // power loss even though its pre-images were flushed, and recovery would then
        // find no journal and skip rollback — leaving the aborted transaction's
        // partial database writes with nothing to undo them. This is the create-side
        // twin of the dir fsync on unlink (`remove_file_durable`).
        //
        // Unlike the unlink side, the directory entry here is load-bearing, so a
        // genuine dir-sync failure is PROPAGATED (fail closed): the pager sees `sync()`
        // fail and must not modify the db, rather than proceeding as if the journal
        // name were durable. (A platform that cannot open a directory for fsync at all
        // is tolerated inside `fsync_parent_dir`.) The success path can only be
        // distinguished from the swallowed-error path with crash / no-dir-flush
        // injection, so it is not covered by the in-process suite.
        fsync_parent_dir(&self.path)?;
        Ok(())
    }

    /// Commit in DELETE mode: drop the handle and remove the journal file (the
    /// standard commit; the file's absence is the durable "committed" signal).
    pub fn commit_delete(self) -> Result<()> {
        let Journal { file, path, .. } = self;
        // Drop the handle before unlinking so no descriptor lingers on the removed
        // inode; the durable-unlink protocol (tolerate already-gone, then fsync the
        // parent dir so the removal survives a crash) lives in `remove_file_durable`.
        drop(file);
        remove_file_durable(&path)
    }

    /// Commit in TRUNCATE mode: shrink the journal to zero length and fsync. An empty
    /// file is not a valid journal, so recovery treats it as nothing to roll back.
    pub fn commit_truncate(self) -> Result<()> {
        self.file.set_len(0)?;
        self.file.sync_all()?;
        Ok(())
    }

    /// Commit in PERSIST mode: overwrite the header magic with zeros and fsync. The
    /// file remains on disk (avoiding an unlink) but no longer has a valid header, so
    /// recovery skips it.
    pub fn commit_persist(mut self) -> Result<()> {
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&[0u8; MAGIC_LEN])?;
        self.file.sync_all()?;
        Ok(())
    }

    /// Commit using the given [`JournalMode`].
    pub fn finish(self, mode: JournalMode) -> Result<()> {
        match mode {
            JournalMode::Delete => self.commit_delete(),
            JournalMode::Truncate => self.commit_truncate(),
            JournalMode::Persist => self.commit_persist(),
        }
    }

    /// The per-transaction checksum nonce stored in the header.
    pub fn nonce(&self) -> u32 {
        self.nonce
    }

    /// The journal's page size in bytes.
    pub fn page_size(&self) -> u32 {
        self.page_size
    }

    /// The sector size the header is padded to.
    pub fn sector_size(&self) -> u32 {
        self.sector_size
    }

    /// The database size (in pages) recorded at the start of the transaction.
    pub fn initial_db_pages(&self) -> u32 {
        self.initial_db_pages
    }

    /// Number of page records appended so far.
    pub fn record_count(&self) -> u32 {
        self.record_count
    }

    /// The journal file path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// The magic is 8 bytes; PERSIST only needs to clear those to invalidate the header.
const MAGIC_LEN: usize = 8;

/// A fresh, non-crypto checksum nonce that varies per call.
///
/// The spec only needs the nonce to *differ* between transactions so a stale sector
/// from an earlier journal is not mistaken for a valid record (fileformat2 §3), so a
/// cryptographic source is unnecessary. This mixes the wall clock, a process-global
/// counter (guarantees distinct inputs within a process even inside one clock tick),
/// and a stack address, then avalanches the bits. The result is forced non-zero so
/// an all-zero stale sector (checksum 0) never accidentally matches.
fn fresh_nonce() -> u32 {
    static COUNTER: AtomicU32 = AtomicU32::new(0);
    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let nanos = now.subsec_nanos();
    let secs = now.as_secs() as u32;
    let addr = &counter as *const u32 as usize as u32;

    let mut h = nanos
        ^ secs.rotate_left(11)
        ^ counter.rotate_left(19)
        ^ addr.rotate_left(7);
    // splitmix32-style avalanche so nearby inputs produce well-spread nonces.
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^= h >> 16;
    if h == 0 { 1 } else { h }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{decode_page_record, has_valid_magic};
    use std::io::Read;

    /// A unique temp path under the OS temp dir; the file is created by the code
    /// under test. Cleaned up by [`TempPath`]'s `Drop`.
    struct TempPath(PathBuf);
    impl TempPath {
        fn new(tag: &str) -> TempPath {
            static N: AtomicU32 = AtomicU32::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            let pid = std::process::id();
            let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
            let mut p = std::env::temp_dir();
            p.push(format!("msj-{tag}-{pid}-{n}-{nanos}.tmp"));
            TempPath(p)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempPath {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn read_all(path: &Path) -> Vec<u8> {
        let mut f = File::open(path).unwrap();
        let mut v = Vec::new();
        f.read_to_end(&mut v).unwrap();
        v
    }

    #[test]
    fn create_writes_padded_header() {
        let tp = TempPath::new("hdr");
        let j = Journal::create(tp.path(), 1024, 5).unwrap();
        drop(j);
        let bytes = read_all(tp.path());
        assert_eq!(bytes.len(), DEFAULT_SECTOR_SIZE as usize);
        assert!(has_valid_magic(&bytes));
        let hdr = JournalHeader::decode(&bytes).unwrap();
        assert_eq!(hdr.page_size, 1024);
        assert_eq!(hdr.initial_db_pages, 5);
        assert_eq!(hdr.sector_size, DEFAULT_SECTOR_SIZE);
        assert_eq!(hdr.page_count, 0, "count not yet backfilled");
        assert_ne!(hdr.nonce, 0, "nonce is non-zero");
    }

    #[test]
    fn append_and_sync_backfills_count_and_records() {
        let tp = TempPath::new("rec");
        let ps = 512u32;
        let mut j = Journal::create(tp.path(), ps, 3).unwrap();
        let p1 = vec![0x11u8; ps as usize];
        let p2 = vec![0x22u8; ps as usize];
        j.append_page(1, &p1).unwrap();
        j.append_page(2, &p2).unwrap();
        let nonce = j.nonce();
        j.sync().unwrap();
        drop(j);

        let bytes = read_all(tp.path());
        let hdr = JournalHeader::decode(&bytes).unwrap();
        assert_eq!(hdr.page_count, 2, "sync backfills the true record count");

        let rec_len = page_record_len(ps as usize);
        let base = DEFAULT_SECTOR_SIZE as usize;
        let r1 = decode_page_record(&bytes[base..base + rec_len], ps as usize, nonce).unwrap();
        assert_eq!(r1.page_no, 1);
        assert_eq!(r1.content, &p1[..]);
        assert!(r1.checksum_ok);
        let r2 =
            decode_page_record(&bytes[base + rec_len..base + 2 * rec_len], ps as usize, nonce)
                .unwrap();
        assert_eq!(r2.page_no, 2);
        assert_eq!(r2.content, &p2[..]);
        assert!(r2.checksum_ok);
    }

    #[test]
    fn append_rejects_wrong_page_size() {
        let tp = TempPath::new("badsize");
        let mut j = Journal::create(tp.path(), 512, 1).unwrap();
        assert!(j.append_page(1, &vec![0u8; 500]).is_err());
    }

    #[test]
    fn append_rejects_page_zero() {
        let tp = TempPath::new("page0");
        let mut j = Journal::create(tp.path(), 512, 1).unwrap();
        assert!(j.append_page(0, &vec![0u8; 512]).is_err());
    }

    #[test]
    fn create_rejects_bad_page_size() {
        let tp = TempPath::new("badcreate");
        assert!(Journal::create(tp.path(), 1000, 1).is_err());
    }

    #[test]
    fn commit_delete_removes_file() {
        let tp = TempPath::new("del");
        let mut j = Journal::create(tp.path(), 512, 1).unwrap();
        j.append_page(1, &vec![7u8; 512]).unwrap();
        j.sync().unwrap();
        assert!(tp.path().exists());
        j.commit_delete().unwrap();
        assert!(!tp.path().exists(), "DELETE removes the journal file");
    }

    #[test]
    fn commit_truncate_zeroes_length() {
        let tp = TempPath::new("trunc");
        let mut j = Journal::create(tp.path(), 512, 1).unwrap();
        j.append_page(1, &vec![7u8; 512]).unwrap();
        j.sync().unwrap();
        j.commit_truncate().unwrap();
        assert!(tp.path().exists(), "TRUNCATE leaves the file in place");
        assert_eq!(read_all(tp.path()).len(), 0, "TRUNCATE zeroes the length");
    }

    #[test]
    fn commit_persist_zeroes_magic() {
        let tp = TempPath::new("persist");
        let mut j = Journal::create(tp.path(), 512, 1).unwrap();
        j.append_page(1, &vec![7u8; 512]).unwrap();
        j.sync().unwrap();
        j.commit_persist().unwrap();
        let bytes = read_all(tp.path());
        assert!(!bytes.is_empty(), "PERSIST keeps the file body");
        assert!(!has_valid_magic(&bytes), "PERSIST clears the header magic");
        assert_eq!(&bytes[0..8], &[0u8; 8]);
    }

    #[test]
    fn nonce_varies_between_journals() {
        let tp1 = TempPath::new("n1");
        let tp2 = TempPath::new("n2");
        let a = Journal::create(tp1.path(), 512, 1).unwrap();
        let b = Journal::create(tp2.path(), 512, 1).unwrap();
        assert_ne!(a.nonce(), b.nonce(), "each transaction gets a distinct nonce");
    }
}
